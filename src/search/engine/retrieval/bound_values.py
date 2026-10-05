"""Shared bound-value and measure-family logic for Search resolvers.

Every election reads a `code_index.members.FamilyQuery`: `identity_query` builds
it from an identity, as the Datasets mint does, and `shape_query` from a shape and
its bound values. `shape_member_facts` resolves a query through
`code_index.members.index_family`, the function the mint resolves through, and
reads recall arcs only for a population the index does not decide.
`five_tuple_member_facts` reads every index row of a shape's five-tuple, for slot
keys and taxonomy values.
`bound_values_from_bindings` expands hierarchical values for both resolvers
and bind repair. The standard resolver refuses bound values no member holds
together, which the region unbinds, while bind repair's filter fails open. The
role-bearing resolver uses a distinct filter. Measure helpers elect and observe partitions, while
slot helpers stamp expansions, summability, and declaration versions onto
specs. Summability and the declaration version read the dataset
`identity_dataset` elects for a spec's identity. A binding with no `property_dcid`, no `value_dcids`, or a kind other
than `value` or `set` names nothing.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import Counter, defaultdict
from collections.abc import Collection, Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from code_index.labels import PRICE_BASIS_AXIS, is_price_basis
from code_index.members import (
    DatasetFamily,
    FamilyQuery,
    SourceNotInFamily,
    absent_member_reads_zero,
    absent_readings,
    deciding_query,
    elect_dataset,
    identity_family_query,
    index_covers_slot_keys,
    index_family,
    unbound_pins,
)
from code_index.shared import get_index
from code_index.types import (
    Bound,
    BoundSet,
    MeasureAxis,
    MeasurePartitionLabel,
    PartitionLabelMatch,
    PropertyBinding,
    label_for_pairs,
    summable_axes,
)
from dc_graph import GraphReadError
from materialise.roles import DegenerateObservationProperties, axis_for_role, declared_role_for
from search.engine.config import settings
from search.engine.errors import (
    FamilyNotResolvableError,
    GraphInfraError,
    RoleSignatureError,
)
from search.engine.reading_coverage import covers
from search.engine.retrieval.facts import FiveTuple, _dcids_for, read_name, read_slot_taxonomy
from search.engine.retrieval.member_facts import (
    ArcsMemberFacts,
    IndexMemberFacts,
    MemberFacts,
    asserted_constraint_properties,
)
from search.engine.shape import (
    ShapeDraft,
    _declared_role_names,
    declares_shape_roles,
    identity_shape_for,
    is_role_bearing,
)
from search.engine.spec_id import mint_identity
from search.models import (
    BaseIdentity,
    Expansion,
    GraphRef,
    IdentityShapeDenominatorRef,
    IdentityShapeDenominatorUnresolved,
    IdentityShapeSchema,
)
from statvar_members import (
    DegenerateObservationProperties as MemberDegenerateObservationProperties,
)
from statvar_members import FiveTuple as MemberFiveTuple
from statvar_members import (
    MemberEnumerationTooLarge,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    SlotBinding,
    elect_role_class,
    enumerate_members,
    five_tuple_from_identity,
    member_digest,
)

if TYPE_CHECKING:
    from code_index import CodeIndex
    from code_index.types import MeasureIndex, StatVarRow
    from materialise.graph import GraphReader
    from search.engine.bind import SlotBindingDraft
    from search.engine.clients.graph import EngineGraphClient
    from search.engine.qualifier_match import QualifierMatch
    from search.models import Slot, Spec, TimeWindow
    from statvar_members import MemberGroup, SubjectPage

logger = logging.getLogger(__name__)

# Cache index five-tuples by object id. A weakref finalizer removes each entry
# before Python can reuse its id. This O(1) check avoids a full index scan for
# shapes the index does not carry.
_index_five_tuples_cache: dict[int, frozenset[FiveTuple]] = {}


def _index_five_tuples(index: CodeIndex) -> frozenset[FiveTuple]:
    key = id(index)
    cached = _index_five_tuples_cache.get(key)
    if cached is None:
        cached = frozenset(IndexMemberFacts(row=row).read_five_tuple() for row in index.statvars)
        _index_five_tuples_cache[key] = cached
        weakref.finalize(index, _index_five_tuples_cache.pop, key, None)
    return cached


# Cache each family the index resolves, by index object id, with the same finalizer, and
# by the slots that decide it (`code_index.members.deciding_query`), so unbound slots no
# row declares share one entry. One answer resolves its own family as it is materialised,
# counted, and finalised, and each resolution scans every index row. An election that
# refuses raises each time, as its outcome is not kept. Oldest entries go first once the
# memo holds `_FAMILY_CACHE_SIZE` entries or `_FAMILY_CACHE_ROWS` row references in all.
# Each entry holds a tuple of references to rows the index owns, 8 bytes a row, so the row
# bound caps the memo near 8 MiB: about 80 copies of the packaged index's largest family
# (12,771 of its 22,208 rows), where the entry bound alone allowed 4,096 copies (400 MiB).
_FAMILY_CACHE_SIZE = 4096
_FAMILY_CACHE_ROWS = 1 << 20


@dataclass(slots=True)
class _FamilyMemo:
    """One index's resolved families, in insertion order, and the rows they hold."""

    families: dict[Hashable, DatasetFamily | None] = field(default_factory=dict)
    rows: int = 0


_index_family_cache: dict[int, _FamilyMemo] = {}
_family_memo_lock = threading.Lock()


def _family_rows(family: DatasetFamily | None) -> int:
    return len(family.rows) if family is not None else 0


def _resolved_family(index: CodeIndex, query: FamilyQuery) -> DatasetFamily | None:
    """`code_index.members.index_family` for `query`, read once per index."""
    key = id(index)
    deciding = deciding_query(index, query)
    query_key = (
        deciding.five_tuple,
        tuple(sorted(deciding.slots.items())),
        deciding.source,
        deciding.role_keys,
    )
    with _family_memo_lock:
        memo = _index_family_cache.get(key)
        if memo is None:
            memo = _index_family_cache[key] = _FamilyMemo()
            weakref.finalize(index, _index_family_cache.pop, key, None)
        if query_key in memo.families:
            return memo.families[query_key]
    # Resolve outside the lock: two threads may both resolve one family, and the later
    # one replaces the earlier's entry.
    family = index_family(index, deciding)
    rows = _family_rows(family)
    if rows > _FAMILY_CACHE_ROWS:
        return family
    with _family_memo_lock:
        memo.rows -= _family_rows(memo.families.pop(query_key, None))
        while len(memo.families) >= _FAMILY_CACHE_SIZE or memo.rows + rows > _FAMILY_CACHE_ROWS:
            memo.rows -= _family_rows(memo.families.pop(next(iter(memo.families))))
        memo.families[query_key] = family
        memo.rows += rows
    return family


def clear_family_memo() -> None:
    """Forget every family `_resolved_family` keeps."""
    _index_family_cache.clear()


# Cache each five-tuple's index rows, in index order, by index object id, with the same
# finalizer.
_index_rows_cache: dict[int, dict[FiveTuple, tuple[StatVarRow, ...]]] = {}


def _index_rows_by_five_tuple(index: CodeIndex) -> dict[FiveTuple, tuple[StatVarRow, ...]]:
    key = id(index)
    cached = _index_rows_cache.get(key)
    if cached is None:
        rows: dict[FiveTuple, list[StatVarRow]] = defaultdict(list)
        for row in index.statvars:
            rows[IndexMemberFacts(row=row).read_five_tuple()].append(row)
        cached = {five_tuple: tuple(found) for five_tuple, found in rows.items()}
        _index_rows_cache[key] = cached
        weakref.finalize(index, _index_rows_cache.pop, key, None)
    return cached


# Cache the constraint properties each five-tuple's index rows carry, by index object id,
# with the same finalizer.
_index_vocabulary_cache: dict[int, dict[FiveTuple, frozenset[str]]] = {}


def _index_vocabulary(index: CodeIndex) -> dict[FiveTuple, frozenset[str]]:
    key = id(index)
    cached = _index_vocabulary_cache.get(key)
    if cached is None:
        vocabulary: dict[FiveTuple, set[str]] = defaultdict(set)
        for row in index.statvars:
            vocabulary[IndexMemberFacts(row=row).read_five_tuple()].update(
                row.constraint_properties
            )
        cached = {five_tuple: frozenset(props) for five_tuple, props in vocabulary.items()}
        _index_vocabulary_cache[key] = cached
        weakref.finalize(index, _index_vocabulary_cache.pop, key, None)
    return cached


# Which StatVarRow field a `kind="shape"` measure axis reads -- the only
# translation `_measure_axis_value` needs between a `MeasureAxis.axis`
# string and the row itself, since a `kind="constraint"` axis is already
# its own key into `StatVarRow.constraint_properties`.
_MEASURE_AXIS_SHAPE_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}


@dataclass(frozen=True, slots=True)
class MeasurePartition:
    """One populated partition, with axis values and its member dcids.

    `None` means members declare no value on that axis; `values` follows axis order.
    """

    values: tuple[tuple[str, str | None], ...]
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeasureElection:
    """The preferred partition and populated siblings matching its basis axes."""

    elected: MeasurePartition
    siblings: tuple[MeasurePartition, ...]


def label_for_partition(
    partition: MeasurePartition, labels: Sequence[MeasurePartitionLabel]
) -> PartitionLabelMatch | None:
    """Match authored labels to `partition.values` by subset containment.

    The largest match wins; a tie raises `ValueError`.
    """
    return label_for_pairs(partition.values, labels)


def _measure_axis_value(row: StatVarRow, axis: MeasureAxis) -> str | None:
    if axis.kind == "shape":
        return getattr(row, _MEASURE_AXIS_SHAPE_FIELDS[axis.axis])
    return row.constraint_properties.get(axis.axis)


def sibling_partitions(
    elected: MeasurePartition,
    partitions: Sequence[MeasurePartition],
    axes: Sequence[MeasureAxis],
) -> tuple[MeasurePartition, ...]:
    """Return other partitions that match `elected` on every basis axis."""
    elected_values = dict(elected.values)
    basis_axes = tuple(axis.axis for axis in axes if axis.role == "basis")
    return tuple(
        partition
        for partition in partitions
        if partition is not elected
        and all(dict(partition.values).get(axis) == elected_values.get(axis) for axis in basis_axes)
    )


def elect_default_partition(
    partitions: Sequence[MeasurePartition], axes: Sequence[MeasureAxis]
) -> MeasureElection | None:
    """Elect by each axis's `prefer` order, in declaration order, and return siblings.

    Unlisted values, including `None`, sort after preferred values. Return
    `None` when `partitions` is empty.
    """
    if not partitions:
        return None

    def sort_key(partition: MeasurePartition) -> tuple[int, ...]:
        values = dict(partition.values)
        ranks = []
        for axis in axes:
            value = values.get(axis.axis)
            ranks.append(axis.prefer.index(value) if value in axis.prefer else len(axis.prefer))
        return tuple(ranks)

    elected = min(partitions, key=sort_key)
    siblings = sibling_partitions(elected, partitions, axes)
    return MeasureElection(elected=elected, siblings=siblings)


# ShapeDraft's counterpart to _MEASURE_AXIS_SHAPE_FIELDS above: the same
# three axis names, read off ShapeDraft's own five-tuple fields instead of
# StatVarRow's.
_MEASURE_AXIS_DRAFT_FIELDS = {
    "populationType": "pop_type_dcid",
    "measuredProperty": "meas_prop_dcid",
    "measurementQualifier": "meas_qual_dcid",
}

# IdentityShapeSchema's counterpart to _MEASURE_AXIS_SHAPE_FIELDS and
# _MEASURE_AXIS_DRAFT_FIELDS above: the same three axis names, read off
# `mint_sibling_identity`'s own `elected.shape` five-tuple fields instead of
# StatVarRow's or ShapeDraft's.
_MEASURE_AXIS_ELECTED_SHAPE_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}


def _family_key(shape: ShapeDraft, basis_shape_axes: Sequence[MeasureAxis]) -> tuple[object, ...]:
    """Return shape identity excluding free shape axes, but including basis axes."""
    return (
        shape.stat_type_dcid,
        shape.meas_denom_dcid,
        shape.meas_denom_unresolved,
        shape.declared_roles,
        shape.optional_roles,
        tuple(getattr(shape, _MEASURE_AXIS_DRAFT_FIELDS[axis.axis]) for axis in basis_shape_axes),
    )


def _member_five_tuple(five_tuple: FiveTuple) -> MemberFiveTuple:
    """The `statvar_members` spelling of a Search five-tuple, which the code index reads."""
    return MemberFiveTuple(
        population_type=five_tuple.pop_type_dcid or None,
        measured_property=five_tuple.meas_prop_dcid or None,
        stat_type=five_tuple.stat_type_dcid or None,
        measurement_qualifier=five_tuple.meas_qual_dcid,
        measurement_denominator=(
            None if five_tuple.meas_denom_unresolved else five_tuple.meas_denom_dcid
        ),
        denominator_unresolved=five_tuple.meas_denom_unresolved,
    )


def shape_five_tuple(shape: ShapeDraft) -> FiveTuple:
    """The five-tuple `shape` reads onto."""
    return FiveTuple(
        pop_type_dcid=shape.pop_type_dcid,
        meas_prop_dcid=shape.meas_prop_dcid,
        stat_type_dcid=shape.stat_type_dcid,
        meas_qual_dcid=shape.meas_qual_dcid,
        meas_denom_dcid=shape.meas_denom_dcid,
        meas_denom_unresolved=shape.meas_denom_unresolved,
    )


def _named_free_values(
    named: Sequence[QualifierMatch], axes: Sequence[MeasureAxis]
) -> tuple[tuple[MeasureAxis, str], ...]:
    """Return one named value per free axis.

    Ignore basis axes, unknown code systems, and axes with conflicting values.
    """
    free_axes = {axis.axis: axis for axis in axes if axis.role == "free"}
    values_by_axis: dict[str, set[str]] = defaultdict(set)
    for match in named:
        if match.code_system in free_axes:
            values_by_axis[match.code_system].add(match.code_dcid)
    return tuple(
        (free_axes[axis_name], next(iter(values)))
        for axis_name, values in values_by_axis.items()
        if len(values) == 1
    )


def _five_tuple_satisfies_all(
    index: CodeIndex, target: FiveTuple, checks: Sequence[tuple[MeasureAxis, str]]
) -> bool:
    """Return whether an indexed row with `target`'s tuple satisfies all checks.

    The index supplies constraint-axis values absent from the five-tuple; an
    empty check set passes.
    """
    return any(
        all(_measure_axis_value(row, axis) == value for axis, value in checks)
        for row in _index_rows_by_five_tuple(index).get(target, ())
    )


@dataclass(frozen=True, slots=True)
class DefaultReading:
    """The declared price-basis preference the collapse applies to shapes the index does
    not carry.

    `order` ranks measurement-qualifier values, best first (`price_basis_order`); a value
    outside it is unranked. `window` is the years the query requests, `"latest"`, or `None`
    for an undated query.
    """

    order: tuple[str, ...]
    window: TimeWindow | Literal["latest"] | None = None


def price_basis_order(measure_index: MeasureIndex, named: Collection[str]) -> tuple[str, ...]:
    """The measurement-qualifier axis's reading order with the query's named readings first.

    Each named value counts with every spelling of its reading (`MeasureIndex.reading_set`),
    and both parts keep the declared order, so "real" ranks RealValue, InflationAdjusted,
    Nominal, PurchasingPowerParity.
    """
    order = measure_index.reading_order(PRICE_BASIS_AXIS)
    readings = frozenset().union(*(measure_index.reading_set(value) for value in named))
    return (
        *(value for value in order if value in readings),
        *(value for value in order if value not in readings),
    )


def _reading_group_key(shape: ShapeDraft) -> tuple[object, ...]:
    """Every five-tuple and role field but the qualifier: shapes sharing it differ only in
    price basis."""
    return (
        shape.stat_type_dcid,
        shape.meas_denom_dcid,
        shape.meas_denom_unresolved,
        _declared_role_names(shape),
        shape.optional_roles,
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
    )


def _member_reading(arcs: dict) -> tuple[object, ...]:
    """A member's constraint map, with the properties whose values are unresolved."""
    facts = ArcsMemberFacts(arcs=arcs)
    return tuple(sorted(facts.read_constraints().items())), facts.read_unresolved_constraints()


def _narrowed(shape: ShapeDraft, removed: Collection[str]) -> ShapeDraft:
    """`shape` without the `removed` members, its taxonomy read over the rest as
    `derive_shapes` reads it: over the members with data at a probed place, else all."""
    arcs = {sv: a for sv, a in (shape.sv_arc_facts or {}).items() if sv not in removed}
    coverage = shape.member_coverage
    if coverage is not None:
        coverage = {sv: places for sv, places in coverage.items() if sv not in removed}
    available = {sv for sv in arcs if coverage is not None and coverage.get(sv)}
    taxonomy = shape.slot_taxonomy
    if taxonomy is not None:
        taxonomy = read_slot_taxonomy(
            slot_keys=shape.slot_keys,
            member_facts={
                sv: ArcsMemberFacts(arcs=a)
                for sv, a in arcs.items()
                if sv in available or not available
            },
        )
    return replace(
        shape,
        sv_arc_facts=arcs,
        member_coverage=coverage,
        slot_taxonomy=taxonomy,
        place_available=bool(available) if coverage is not None else shape.place_available,
    )


def _covered_members(
    shape: ShapeDraft, better: ShapeDraft, window: TimeWindow | Literal["latest"] | None
) -> set[str]:
    """The members of `shape` that a member of `better` with the same constraint map
    covers (`reading_coverage.covers`). Coverage unknown for a whole shape reads as unknown
    per member."""
    coverage, better_coverage = shape.member_coverage, better.member_coverage
    if coverage is None:
        return set()
    counterparts = [
        (_member_reading(arcs), None if better_coverage is None else better_coverage.get(sv))
        for sv, arcs in (better.sv_arc_facts or {}).items()
    ]
    covered: set[str] = set()
    for sv, arcs in (shape.sv_arc_facts or {}).items():
        own = _member_reading(arcs)
        if any(
            reading == own and covers(a, coverage.get(sv), window) for reading, a in counterparts
        ):
            covered.add(sv)
    return covered


def _collapse_default_readings(
    shapes: Sequence[ShapeDraft], reading: DefaultReading
) -> tuple[ShapeDraft, ...]:
    """Remove, from shapes the index does not carry, the members a better-ranked price basis
    answers.

    Shapes sharing `_reading_group_key` differ only in their qualifier. For each pair with
    ranks strictly ordered, a member `b` of the lower-ranked shape leaves when the
    higher-ranked shape holds a member `a` with the same constraint map that covers it
    (`reading_coverage.covers`): at every probed place `b` has data, `a` is shown to hold
    every requested year `b` may hold there. A shape left with no member leaves the menu,
    and the shape that covered it takes the higher of the two scores and the earlier
    position. A shape left with some members keeps them and its own score. Unranked
    qualifiers, including none, take no part.
    """
    rank = {value: position for position, value in enumerate(reading.order)}
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for i, shape in enumerate(shapes):
        if shape.sv_arc_facts and shape.meas_qual_dcid in rank and not index_carries_shape(shape):
            groups[_reading_group_key(shape)].append(i)

    # Members each shape loses, and the best-ranked shape that answers one of them.
    removed: dict[int, set[str]] = defaultdict(set)
    covered_by: dict[int, int] = {}
    for indices in groups.values():
        for b_i in indices:
            b_rank = rank[cast("str", shapes[b_i].meas_qual_dcid)]
            better = sorted(
                (i for i in indices if rank[cast("str", shapes[i].meas_qual_dcid)] < b_rank),
                key=lambda i: rank[cast("str", shapes[i].meas_qual_dcid)],
            )
            for a_i in better:
                covered = _covered_members(shapes[b_i], shapes[a_i], reading.window)
                if covered:
                    removed[b_i] |= covered
                    covered_by.setdefault(b_i, a_i)

    # An emptied shape merges into the shape that answered it, or into that shape's own
    # survivor when it was emptied too.
    emptied = {
        i for i, members in removed.items() if members >= (shapes[i].sv_arc_facts or {}).keys()
    }
    merged: dict[int, list[int]] = defaultdict(list)
    for b_i in emptied:
        a_i = covered_by[b_i]
        while a_i in emptied:
            a_i = covered_by[a_i]
        merged[a_i].append(b_i)

    survivors: list[ShapeDraft | None] = [None] * len(shapes)
    for i, shape in enumerate(shapes):
        if i in emptied:
            continue
        if removed.get(i):
            shape = _narrowed(shape, removed[i])
        family = [i, *merged.get(i, ())]
        scores = [
            s.representative_score
            for s in (shapes[j] for j in family)
            if s.representative_score is not None
        ]
        recalled = any(shapes[j].recalled for j in family)
        if scores and max(scores) != shape.representative_score:
            shape = replace(shape, representative_score=max(scores))
        if recalled != shape.recalled:
            shape = replace(shape, recalled=recalled)
        survivors[min(family)] = shape
    return tuple(shape for shape in survivors if shape is not None)


def collapse_measure_families(
    shapes: Sequence[ShapeDraft],
    index: CodeIndex,
    axes: Sequence[MeasureAxis],
    named: Sequence[QualifierMatch] = (),
    *,
    reading: DefaultReading | None = None,
) -> tuple[ShapeDraft, ...]:
    """Collapse indexed shape families to the partition best matching the query, and apply
    the default reading to shapes the index does not carry.

    With `reading`, members of a shape the index does not carry leave first where a
    better-ranked price basis answers them (`_collapse_default_readings`). `None` skips
    that step, as for a leg that left a price-basis phrase unread.

    Only shapes the index carries compete in the partition election
    (`index_carries_shape`); other shapes pass through, a debt family on a five-tuple CRS
    flows share among them. `_family_key` keeps distinct non-measure and basis identities
    separate. Named free-axis values filter each family using index rows,
    including constraint axes. If no member matches every value, elect from
    that family's full set. With no named values, the declared default wins.

    Skip the partition election when there is no shape axis or a constraint
    basis axis; a constraint value never splits a shape, so grouping could merge two
    different measures. The elected partition keeps its preferred measure, but takes
    the family's strongest measured score and that member's input position
    (earliest on a tie). Topic-only families remain unscored; survivors retain input order.
    """
    if reading is not None:
        shapes = _collapse_default_readings(shapes, reading)
    shape_axes = [axis for axis in axes if axis.kind == "shape"]
    if not shape_axes:
        return tuple(shapes)
    if any(axis.kind == "constraint" and axis.role == "basis" for axis in axes):
        return tuple(shapes)

    basis_shape_axes = [axis for axis in shape_axes if axis.role == "basis"]
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    targets: dict[int, FiveTuple] = {}
    survivors: list[ShapeDraft | None] = [None] * len(shapes)

    for i, shape in enumerate(shapes):
        if not index_carries_shape(shape, index):
            survivors[i] = shape
            continue
        targets[i] = shape_five_tuple(shape)
        groups[_family_key(shape, basis_shape_axes)].append(i)

    checks = _named_free_values(named, axes)

    for indices in groups.values():
        partitions = tuple(
            MeasurePartition(
                values=tuple(
                    (axis.axis, getattr(shapes[i], _MEASURE_AXIS_DRAFT_FIELDS[axis.axis]))
                    for axis in shape_axes
                ),
                members=(str(i),),
            )
            for i in indices
        )
        eligible = (
            tuple(
                partition
                for partition in partitions
                if _five_tuple_satisfies_all(index, targets[int(partition.members[0])], checks)
            )
            if checks
            else partitions
        )
        election = elect_default_partition(eligible or partitions, shape_axes)
        assert election is not None  # indices, and so partitions, is never empty here
        survivor_index = int(election.elected.members[0])
        # The family's own best evidence, not the axis election's: every
        # candidate in this group is retrieval evidence for the same
        # underlying concept, so the group's highest representative_score is
        # the family's true retrieval strength, and that member's own list
        # position is where the survivor belongs -- ties keep the earliest
        # index, `shapes`' own tie-break (breadth, then shape_id) already
        # having settled which one that is. A member with no score of its own
        # is never the best evidence; when no member of the family carries a
        # score, the survivor keeps its own None and there is nothing to
        # raise it to.
        scored_indices = [i for i in indices if shapes[i].representative_score is not None]
        best_index = (
            max(
                scored_indices,
                key=lambda i: cast("float", shapes[i].representative_score),
            )
            if scored_indices
            else survivor_index
        )
        best_score = shapes[best_index].representative_score
        survivor = shapes[survivor_index]
        # A collapsed sibling carries its own recall provenance: the
        # survivor is recalled if any sibling in its group was, even one
        # that lost the election.
        group_recalled = any(shapes[i].recalled for i in indices)
        score_changed = survivor.representative_score != best_score
        recalled_changed = survivor.recalled != group_recalled
        if score_changed and recalled_changed:
            survivor = replace(survivor, representative_score=best_score, recalled=group_recalled)
        elif score_changed:
            survivor = replace(survivor, representative_score=best_score)
        elif recalled_changed:
            survivor = replace(survivor, recalled=group_recalled)
        survivors[best_index] = survivor

    return tuple(shape for shape in survivors if shape is not None)


def shape_carries_measure_value(
    shape: ShapeDraft, index: CodeIndex, axis: MeasureAxis, value: str
) -> bool:
    """Return whether an indexed member of this shape asserts `value` on `axis`."""
    return _five_tuple_satisfies_all(index, shape_five_tuple(shape), ((axis, value),))


@dataclass(frozen=True, slots=True)
class MeasureFamilyElection:
    """Election, siblings, forced open-constraint values, and dropped axes.

    `forced` and `dropped` partition the same open constraint axes.
    """

    elected: MeasurePartition
    siblings: tuple[MeasurePartition, ...]
    forced: tuple[tuple[str, str], ...]
    dropped: tuple[str, ...]


def _measure_family_free_fields(axes: Sequence[MeasureAxis]) -> frozenset[str]:
    """Return fixed five-tuple fields; only declared shape axes may vary.

    Current declarations leave stat type and denominator fields fixed.
    """
    named = {_MEASURE_AXIS_DRAFT_FIELDS[axis.axis] for axis in axes if axis.kind == "shape"}
    return frozenset(FiveTuple._fields) - named


def _measure_family_rows(
    shape: ShapeDraft,
    index: CodeIndex,
    axes: Sequence[MeasureAxis],
    query: FamilyQuery,
) -> tuple[StatVarRow, ...]:
    """Return index rows matching this shape's non-measure identity and `query`'s bounds.

    Ignore bound values on declared measure axes so named values cannot hide siblings.

    Keep rows whose role set a member of `shape` could declare
    (`shape.declares_shape_roles`); an empty row declaration means
    `observationAbout`. Shapes without declared roles skip this check.

    The rows are one dataset's: `code_index.members.elect_dataset` applies the election
    the mint applies, over every property `query` binds, measure axes included, and the
    dataset it names.

    A shape the index does not carry (`index_carries_shape`) has no measure family.
    """
    target = shape_five_tuple(shape)
    if not index_carries_shape(shape, index):
        return ()
    free_fields = _measure_family_free_fields(axes)
    axis_names = {axis.axis for axis in axes}
    constraints: dict[str, PropertyBinding] = {
        prop: (Bound(values[0]) if len(values) == 1 else BoundSet(frozenset(values)))
        for prop, values in query_bound_values(query).items()
        if prop not in axis_names
    }
    matched = tuple(
        row
        for row in index.filter(constraints)
        if all(
            getattr(IndexMemberFacts(row=row).read_five_tuple(), field) == getattr(target, field)
            for field in free_fields
        )
        and declares_shape_roles(shape, IndexMemberFacts(row=row).read_declared_roles())
    )
    try:
        return elect_dataset(index, matched, query).rows
    except SourceNotInFamily as exc:
        _log_refused_election(exc)
        return ()


def _log_refused_election(exc: SourceNotInFamily) -> None:
    logger.warning("dataset election refused: %s (candidates %r)", exc, exc.candidates)


def query_bound_values(query: FamilyQuery) -> dict[str, list[str]]:
    """Return the {property_dcid: [value_dcids]} map of the properties `query` binds."""
    values: dict[str, list[str]] = {}
    for prop, binding in query.slots.items():
        match binding:
            case ("value", ref):
                values[prop] = [ref]
            case ("set", refs):
                values[prop] = list(refs)
    return values


MeasureFamilyCacheKey = tuple[
    FiveTuple,
    tuple[str, ...] | None,
    frozenset[str],
    tuple[tuple[str, tuple[str, ...]], ...],
    str | None,
    frozenset[str],
]


def measure_family_cache_key(
    shape: ShapeDraft, axes: Sequence[MeasureAxis], query: FamilyQuery
) -> MeasureFamilyCacheKey:
    """Key the exact inputs `_measure_family_rows` reads.

    A measure-axis bound stays out of the row filter, and keys and values are sorted, since
    their input order is irrelevant. The election reads the dataset the query names and its
    role keys, so both enter the key.
    """
    axis_names = {axis.axis for axis in axes}
    filtered = tuple(
        sorted(
            (prop, tuple(sorted(values)))
            for prop, values in query_bound_values(query).items()
            if prop not in axis_names
        )
    )
    return (
        shape_five_tuple(shape),
        shape.declared_roles,
        shape.optional_roles,
        filtered,
        query.source,
        query.role_keys,
    )


def measure_family_partitions(
    shape: ShapeDraft,
    index: CodeIndex,
    axes: Sequence[MeasureAxis],
    query: FamilyQuery,
    *,
    cache: MutableMapping[MeasureFamilyCacheKey, tuple[MeasurePartition, ...]] | None = None,
) -> tuple[MeasurePartition, ...]:
    """Group matching rows into partitions across declared measure axes.

    Return empty when no rows match. Served rows determine their observed
    partition; do not substitute a query's preferred election. Optional
    `cache` reuses results with the same key.
    """
    if cache is None:
        return _measure_family_partitions_uncached(shape, index, axes, query)
    key = measure_family_cache_key(shape, axes, query)
    if key not in cache:
        cache[key] = _measure_family_partitions_uncached(shape, index, axes, query)
    return cache[key]


def _measure_family_partitions_uncached(
    shape: ShapeDraft,
    index: CodeIndex,
    axes: Sequence[MeasureAxis],
    query: FamilyQuery,
) -> tuple[MeasurePartition, ...]:
    rows = _measure_family_rows(shape, index, axes, query)
    if not rows:
        return ()

    axis_names = tuple(axis.axis for axis in axes)
    grouped: dict[tuple[str | None, ...], list[str]] = defaultdict(list)
    for row in rows:
        grouped[tuple(_measure_axis_value(row, axis) for axis in axes)].append(row.dcid)
    return tuple(
        MeasurePartition(values=tuple(zip(axis_names, key, strict=True)), members=tuple(dcids))
        for key, dcids in grouped.items()
    )


def partition_for_values(
    partitions: Sequence[MeasurePartition], values: Mapping[str, str | None]
) -> MeasurePartition | None:
    """Find an exact axis-value match; `None` matches only `None`.

    Pure: `values` must contain observed axis values, not query bounds, which
    cannot distinguish an observed `None` from an open axis. Return `None` if
    no partition matches.
    """
    for partition in partitions:
        if dict(partition.values) == dict(values):
            return partition
    return None


def observed_measure_values(
    axes: Sequence[MeasureAxis], members: Sequence[MemberFacts]
) -> MeasurePartition | None:
    """Return the measure values shared by all served members.

    Return `None` for an empty set or any disagreement; `None` on an axis is
    itself a value. The result has no member ids, only the shared values.
    """
    if not members:
        return None

    def _values(member: MemberFacts) -> dict[str, str | None]:
        five_tuple = member.read_five_tuple()
        constraints = member.read_constraints()
        return {
            axis.axis: (
                getattr(five_tuple, _MEASURE_AXIS_DRAFT_FIELDS[axis.axis])
                if axis.kind == "shape"
                else constraints.get(axis.axis)
            )
            for axis in axes
        }

    first, *rest = (_values(member) for member in members)
    if any(candidate != first for candidate in rest):
        return None
    return MeasurePartition(
        values=tuple((axis.axis, first[axis.axis]) for axis in axes), members=()
    )


def observed_measure_family(
    shape: ShapeDraft,
    index: CodeIndex,
    axes: Sequence[MeasureAxis],
    query: FamilyQuery,
    members: Sequence[MemberFacts],
    *,
    cache: MutableMapping[MeasureFamilyCacheKey, tuple[MeasurePartition, ...]] | None = None,
) -> tuple[MeasurePartition | None, tuple[MeasurePartition, ...]]:
    """Return the observed partition and its basis-compatible siblings.

    Look up the observed values in the family because sibling exclusion uses
    object identity. Return `(None, ())` if members disagree or no partition
    matches. `cache` is forwarded to `measure_family_partitions`.
    """
    observed = observed_measure_values(axes, members)
    if observed is None:
        return None, ()
    partitions = measure_family_partitions(shape, index, axes, query, cache=cache)
    partition = partition_for_values(partitions, dict(observed.values))
    if partition is None:
        return None, ()
    return partition, sibling_partitions(partition, partitions, axes)


def synthetic_shape_for_identity(shape: IdentityShapeSchema) -> ShapeDraft:
    """Build an election-only ShapeDraft from a posted identity.

    Missing string fields become `""`; qualifiers and denominator stay `None`.
    Slots are empty because measure-family election does not read them.
    """
    denominator = shape.measurement_denominator
    return ShapeDraft(
        shape_id="",
        label="",
        pop_type_dcid=shape.population_type or "",
        meas_prop_dcid=shape.measured_property or "",
        stat_type_dcid=shape.stat_type or "",
        meas_qual_dcid=shape.measurement_qualifier,
        meas_denom_dcid=(
            denominator.ref if isinstance(denominator, IdentityShapeDenominatorRef) else None
        ),
        meas_denom_unresolved=isinstance(denominator, IdentityShapeDenominatorUnresolved),
        slot_keys=(),
        # Rebuilt from a posted identity, not from a live recall pool. This
        # shape's only use is a measure-family election, which ignores the
        # recalled field. See ShapeDraft.retrieved: the identity came from a
        # caller, so nothing about its match to a query was measured.
        recalled=False,
        retrieved=False,
    )


def identity_names_measure_family(identity: BaseIdentity, axes: Sequence[MeasureAxis]) -> bool:
    """Return whether a described identity's shape and measure bindings select a family.

    This lets resubmit recover indexed StatVars without a posted
    `resolved_stat_var_dcids` seed. Shape axes are fixed by the identity;
    constraint axes are fixed only when bound, with their role affecting only
    open-axis preference. Other slots are excluded here,
    so an unrelated bound-value mismatch is reported later as no data, not an
    invalid identity. Return false when the index does not decide the identity's
    membership (`index_decides_membership`, the mint's test), so an IDS identity on a
    five-tuple CRS rows share names no index family, and when no family partition is
    consistent.
    """
    if not index_decides_membership(identity):
        return False
    axis_names = {axis.axis for axis in axes}
    query = identity_query(identity)
    measure_query = replace(
        query,
        slots={
            prop: binding if prop in axis_names else ("unbound",)
            for prop, binding in query.slots.items()
        },
    )
    election = elect_measure_family(
        synthetic_shape_for_identity(cast(IdentityShapeSchema, identity.shape)),
        get_index(),
        axes,
        measure_query,
    )
    return election is not None


@dataclass(frozen=True, slots=True)
class OpenAxisFact:
    """Each served member's value on one open constraint axis.

    `member_values` maps a member dcid to the value dcid it declares, or `None` when it
    declares none. Keying by member makes a union over overlapping member sets count
    each member once. `value_names` maps a value dcid to the graph `name` a served
    member's arc carries, for values the code index does not label.
    """

    property: str
    member_values: Mapping[str, str | None]
    value_names: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def values(self) -> frozenset[str | None]:
        return frozenset(self.member_values.values())

    @property
    def undeclared_count(self) -> int:
        return sum(1 for value in self.member_values.values() if value is None)

    def value_counts(self) -> Counter[str]:
        """Members per declared value. The counts and `undeclared_count` sum to the
        number of members."""
        return Counter(value for value in self.member_values.values() if value is not None)


def open_constraint_axes(
    bound_values: Mapping[str, list[str]], members: Mapping[str, MemberFacts]
) -> tuple[OpenAxisFact, ...]:
    """Return unbound constraint axes in first-observed member order.

    `members` maps each served member's dcid to its facts. Include unresolved
    properties. A member with no resolved value contributes `None`. Each fact carries
    the members' graph names for its values. Return `()` when no open axis exists.
    """
    member_constraints = {dcid: m.read_constraints() for dcid, m in members.items()}
    value_names = {
        value: name
        for m in members.values()
        for value, name in m.read_constraint_value_names().items()
    }
    order: list[str] = []
    seen: set[str] = set()
    for dcid, m in members.items():
        for prop in (*member_constraints[dcid], *m.read_unresolved_constraints()):
            if prop not in seen:
                seen.add(prop)
                order.append(prop)
    facts = []
    for prop in order:
        if prop in bound_values:
            continue
        member_values = {
            dcid: constraints.get(prop) for dcid, constraints in member_constraints.items()
        }
        facts.append(
            OpenAxisFact(
                property=prop,
                member_values=member_values,
                value_names={v: value_names[v] for v in member_values.values() if v in value_names},
            )
        )
    return tuple(facts)


def elect_measure_family(
    shape: ShapeDraft,
    index: CodeIndex,
    axes: Sequence[MeasureAxis],
    query: FamilyQuery,
) -> MeasureFamilyElection | None:
    """Elect a populated partition consistent with the shape and `query`'s bounds.

    Candidates must match the shape axes and every bound measure value.
    Elect by declared preference; return `None` if no candidate is consistent.
    Siblings come from the full family, so a bound-inconsistent partition can
    remain an alternative when it matches the elected partition's basis axes.

    `forced` contains non-`None` values for open constraint axes. `dropped`
    lists open constraint axes with no value on the elected partition. Existing
    bound axes remain unchanged.
    """
    partitions = measure_family_partitions(shape, index, axes, query)
    if not partitions:
        return None
    bound_values = query_bound_values(query)

    def _consistent(partition: MeasurePartition) -> bool:
        values = dict(partition.values)
        for axis in axes:
            if axis.kind == "shape" and values.get(axis.axis) != getattr(
                shape, _MEASURE_AXIS_DRAFT_FIELDS[axis.axis]
            ):
                return False
        return all(
            values.get(axis.axis) in bound_values[axis.axis]
            for axis in axes
            if axis.axis in bound_values
        )

    consistent = tuple(partition for partition in partitions if _consistent(partition))
    if not consistent:
        return None

    election = elect_default_partition(consistent, axes)
    assert election is not None  # `consistent` is non-empty here
    elected = election.elected
    siblings = sibling_partitions(elected, partitions, axes)

    elected_values = dict(elected.values)
    open_constraint_axes = tuple(
        axis for axis in axes if axis.kind == "constraint" and axis.axis not in bound_values
    )
    forced = tuple(
        (axis.axis, value)
        for axis in open_constraint_axes
        if (value := elected_values.get(axis.axis)) is not None
    )
    dropped = tuple(
        axis.axis for axis in open_constraint_axes if elected_values.get(axis.axis) is None
    )
    return MeasureFamilyElection(elected=elected, siblings=siblings, forced=forced, dropped=dropped)


@dataclass(frozen=True, slots=True)
class SiblingIdentity:
    """One sibling measure partition's minted identity: its own `spec_id`,
    the `BaseIdentity` document that hashes to it, and the authored label
    match for the partition it was minted from (`label_for_partition`'s own
    result). `label` is `None` when no authored label covers the partition --
    an unlabelled sibling is never published, and `None` is how a caller
    knows to drop it rather than compose a substitute.
    """

    spec_id: str
    identity: BaseIdentity
    label: PartitionLabelMatch | None


def family_asserted_measure_axes(
    axes: Sequence[MeasureAxis], partition: MeasurePartition
) -> frozenset[str]:
    """The `kind="constraint"` axis names `partition`'s own members actually
    carry a non-`None` value for -- a per-partition fact, read straight off
    `partition.values` (grouping's own per-axis fold over exactly that
    partition's members; see `MeasurePartition`), never off a sibling's.
    `mint_sibling_identity` uses this set to decide which axes belong on
    `partition`'s own minted identity at all.

    An axis `partition` itself leaves `None` is omitted here regardless of
    whether some OTHER partition in the same measure family carries a value
    for it: a `financialObligationType` value some `FinancialObligation`
    sibling asserts says nothing about whether a `FinancialFlow` partition's
    own members can ever assert it too, and they cannot -- the two
    populations are mutually exclusive on that axis by the graph's own data
    model, so a slot naming it on the `FinancialFlow` partition would
    describe a spec no StatVar in that partition's own five-tuple family
    could ever confirm (`region.resubmit_promote`'s own axis-admission
    check refuses exactly such a slot).
    """
    return frozenset(
        axis.axis
        for axis in axes
        if axis.kind == "constraint" and dict(partition.values).get(axis.axis) is not None
    )


def mint_sibling_identity(
    elected: BaseIdentity,
    partition: MeasurePartition,
    axes: Sequence[MeasureAxis],
    labels: Sequence[MeasurePartitionLabel],
) -> SiblingIdentity:
    """Mint one sibling measure partition's identity by editing the elected
    spec's own already-minted `BaseIdentity`, rather than assembling a fresh
    projection from `Slot`/`Entity` objects: a sibling has no grounded
    `GraphRef` to build a `Slot` from, and identity never hashes labels
    anyway, so fabricating one to satisfy a type would be inventing data
    (see `engine.spec_id.mint_identity`'s own docstring, the seam this mints
    through).

    `elected`'s entities, entity_groups, requested_window and facets describe a fact the
    sibling agrees with -- being a sibling means sharing every axis but the
    declared measure ones -- so they carry over byte-identical, read
    straight off `elected`'s own validated document.

    The measure axes do not carry over from `elected.slots` unmodified.
    Every slot in `elected.slots` whose property is a declared measure axis
    is dropped, and a fresh slot is added only for a `kind="constraint"`
    axis named in `partition`'s own `family_asserted_measure_axes(axes,
    partition)` -- computed here, from `partition` alone, never from a
    sibling's: bound to `partition`'s own value on that axis where it has
    one, unbound where it does not. An axis `axes` declares but `partition`
    itself never asserts gets no slot at all, bound or unbound: no member of
    `partition`'s own five-tuple family carries a value for it, so a slot
    naming it would only be consistent with `partition` by construction,
    without support from anything the sibling's own members actually
    assert. This holds whether `partition` merely leaves the axis open or
    the axis belongs to a different `populationType` entirely, since
    neither reading is available from `partition`'s own values alone, and
    both read the same way here.

    `partition`'s own slot key set can still differ from `elected`'s -- an
    axis `elected` left bound may be open (or absent) on `partition`, or the
    reverse -- and that difference is a fact about the sibling's own
    identity, not an inconsistency to smooth over: `spec_id.py`'s
    `_slots_member` hashes every what/how slot including unbound ones, so the
    key set itself is part of `spec_id`.

    Carrying `elected`'s non-measure slots over byte-identical only covers
    an axis `elected` itself already named a slot for. `partition`'s own
    real members can declare a non-measure "how" axis `elected`'s members
    never did at all; that axis is never padded onto the minted sibling
    here. A resubmit's own declared slot set is the POSTED identity's own
    slot properties (`identity_edit.check_slots_declared` only requires the
    posted identity to cover its BASE's declared properties, not the
    reverse; see that function's docstring), so a resubmit of this
    sibling's own `resolved_stat_var_dcids` can freely post a wider slot set
    naming such an axis: it is admitted as an edit (an open axis the sibling
    itself left undeclared, now bound), never refused for omitting a slot
    this identity never declared in the first place.

    The `shape` member is rebuilt through `identity_shape_for`, from
    `partition`'s own value on each declared `kind="shape"` axis and
    `elected.shape`'s own value on the two five-tuple fields no declared
    axis ever names (`stat_type`, `measurement_denominator`). Those two are
    shared across the whole measure family rather than elected per-partition
    (see `_measure_family_free_fields`), so `elected`'s value is the
    sibling's value too.

    The label is `label_for_partition(partition, labels)`, computed once
    here rather than left for the caller to re-derive.

    Raises `ValueError` when `elected.shape` is not the described arm
    (`kind="statvar_schema"`): a pinned shape (`kind="stat_var"`, a single
    StatVar dcid with no slots) has no measure family to have a sibling of.
    """
    elected_shape = elected.shape
    if not isinstance(elected_shape, IdentityShapeSchema):
        raise ValueError(f"cannot mint a sibling identity from a {elected_shape.kind!r}-kind shape")

    partition_values = dict(partition.values)

    def carried(axis_name: str, elected_value: str | None) -> str | None:
        return partition_values.get(axis_name, elected_value)

    denominator = elected_shape.measurement_denominator
    meas_denom_unresolved = isinstance(denominator, IdentityShapeDenominatorUnresolved)
    meas_denom_dcid = (
        denominator.ref if isinstance(denominator, IdentityShapeDenominatorRef) else None
    )

    # Read through _MEASURE_AXIS_ELECTED_SHAPE_FIELDS for every declared
    # `kind="shape"` axis, the same way `_measure_axis_value` and `_family_key`
    # read through their own two tables -- an axis the table does not name
    # raises KeyError here too, rather than being silently dropped. An axis
    # `axes` never declares at all keeps `elected_shape`'s own value, same as
    # before.
    carried_shape_values = {
        axis.axis: carried(
            axis.axis, getattr(elected_shape, _MEASURE_AXIS_ELECTED_SHAPE_FIELDS[axis.axis])
        )
        for axis in axes
        if axis.kind == "shape"
    }

    sibling_shape = identity_shape_for(
        pop_type_dcid=carried_shape_values.get("populationType", elected_shape.population_type),
        meas_prop_dcid=carried_shape_values.get(
            "measuredProperty", elected_shape.measured_property
        ),
        stat_type_dcid=elected_shape.stat_type,
        meas_qual_dcid=carried_shape_values.get(
            "measurementQualifier", elected_shape.measurement_qualifier
        ),
        meas_denom_dcid=meas_denom_dcid,
        meas_denom_unresolved=meas_denom_unresolved,
    )

    measure_axis_names = {axis.axis for axis in axes}
    base = elected.core_document()
    sibling_slots = [slot for slot in base["slots"] if slot["property"] not in measure_axis_names]
    # asserted_axes names exactly the constraint axes `partition_values` is
    # non-None on (family_asserted_measure_axes reads it off nothing else),
    # so every axis reached here binds to a real value -- there is no
    # "asserted but unbound" case left to represent.
    asserted_axes = family_asserted_measure_axes(axes, partition)
    for axis in axes:
        if axis.kind != "constraint" or axis.axis not in asserted_axes:
            continue
        binding = {"state": "value", "ref": partition_values[axis.axis]}
        sibling_slots.append({"property": axis.axis, "binding": binding})

    projection = {**base, "shape": sibling_shape, "slots": sibling_slots}
    spec_id, identity = mint_identity(projection, entity_groups=elected.entity_groups)
    return SiblingIdentity(
        spec_id=spec_id, identity=identity, label=label_for_partition(partition, labels)
    )


def price_basis_sibling(elected: MeasurePartition, sibling: MeasurePartition) -> str | None:
    """The price-basis value `sibling` switches to when it differs from `elected` on the
    price-basis axis alone and both state a price basis, else `None`. Such a sibling is an
    option of the unit token and absent from the measure menu."""
    elected_values = dict(elected.values)
    sibling_values = dict(sibling.values)
    if elected_values.keys() != sibling_values.keys():
        return None
    differing = [axis for axis in elected_values if elected_values[axis] != sibling_values[axis]]
    if differing != [PRICE_BASIS_AXIS]:
        return None
    elected_basis, sibling_basis = (
        elected_values[PRICE_BASIS_AXIS],
        sibling_values[PRICE_BASIS_AXIS],
    )
    if is_price_basis(PRICE_BASIS_AXIS, elected_basis) and is_price_basis(
        PRICE_BASIS_AXIS, sibling_basis
    ):
        return sibling_basis
    return None


def measure_without_price_basis(partition: MeasurePartition) -> tuple[tuple[str, str | None], ...]:
    """`partition.values` with a price-basis value blanked, which is the measure a menu entry names."""
    return tuple(
        (axis, None if is_price_basis(axis, value) else value) for axis, value in partition.values
    )


def _collapse_bindings(bindings: Sequence[SlotBindingDraft]) -> dict[str, list[str]]:
    """The pre-expansion {property_dcid: [value_dcids]} map: what `bindings`
    itself asks for (kind=value/set bindings that actually carry a value),
    before the code index's descendant walk widens any of it. Two bindings
    naming the same property_dcid collide on the dict key; the later one in
    `bindings` order wins, silently. Shared by bound_values_from_bindings and
    slot_expansions so the two can never disagree about the query's own
    baseline.
    """
    return {
        b.property_dcid: list(b.value_dcids)
        for b in bindings
        if b.property_dcid and b.kind in ("value", "set") and b.value_dcids
    }


def _expand_values(index: CodeIndex, values: list[str]) -> list[str]:
    """`values` widened by each value's own code-index descendants
    (``CodeIndex.descendants``), deduped, first-seen order (each value's own
    dcid first -- ``descendants`` returns it first -- then its descendants,
    across every value in `values` in turn).

    A genuine no-op -- returns `values` unchanged, same order -- for a value
    naming a flat code system (e.g. crsTypeOfFlow, per A4c) or one this index
    carries no code row for at all (see ``descendants``' own docstring); the
    latter covers every non-CRS property, since this deployment's index
    carries CRS code systems alone.
    """
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for descendant in index.descendants(value):
            if descendant not in seen:
                seen.add(descendant)
                result.append(descendant)
    return result


type ServedSeries = tuple[str, str]
"""One series an answer serves: its StatVar and the dcids of the entities it is read at,
joined by "|" (`read_plan.PlanSeries.key`). An answer's download counts these
(`DownloadFacts.series`)."""


@dataclass(frozen=True, slots=True)
class SeriesRows:
    """The rows one served series writes to a pick's file, one per year it holds a point
    in, and whether that count is exact or an upper bound (`DownloadFacts.rows_exact`).

    `years` holds the series' year mask from the coverage index
    (`CoverageIndex.series_years`), bit `i` for the horizon's epoch year plus `i`, or None
    for a series counted from a read plan's facet metadata. `bulk_facets` holds, for an
    answer read in bulk, the facet ids its StatVar's rows sit in: a bulk read elects one
    facet for every member, so a pick whose series sit in more than one is an upper bound
    (`served_rows`). Empty for a per-series read.
    """

    rows: int
    exact: bool
    years: int | None = None
    bulk_facets: frozenset[str] = frozenset()

    def union(self, other: SeriesRows) -> SeriesRows:
        """This series as two reads of it together. With year masks on both, the masks are
        ORed. Without, the count is the larger one, exact only when both are exact and
        equal."""
        exact = self.exact and other.exact
        bulk_facets = self.bulk_facets | other.bulk_facets
        if self.years is not None and other.years is not None:
            years = self.years | other.years
            return SeriesRows(
                rows=years.bit_count(), exact=exact, years=years, bulk_facets=bulk_facets
            )
        return SeriesRows(
            rows=max(self.rows, other.rows),
            exact=exact and self.rows == other.rows,
            bulk_facets=bulk_facets,
        )


type ServedSeriesRows = Mapping[ServedSeries, SeriesRows]
"""The series a pick serves, each with the rows it writes."""


def union_served(left: ServedSeriesRows, right: ServedSeriesRows) -> dict[ServedSeries, SeriesRows]:
    """The series of `left` and `right` read together, a series both serve counted once
    (`SeriesRows.union`)."""
    merged = dict(left)
    for key, rows in right.items():
        merged[key] = merged[key].union(rows) if key in merged else rows
    return merged


class ServedRows(NamedTuple):
    """The rows a pick's file holds and whether the count is exact
    (`SlotOption.served_rows`, `SlotOption.served_rows_exact`)."""

    rows: int
    exact: bool


def served_rows(served: ServedSeriesRows) -> ServedRows:
    """The rows `served` writes: each series' rows, summed. Exact when every series' count
    is and, read in bulk, the series sit in one facet (`SeriesRows.bulk_facets`)."""
    bulk_facets = frozenset[str]().union(*(s.bulk_facets for s in served.values()))
    return ServedRows(
        rows=sum(s.rows for s in served.values()),
        exact=all(s.exact for s in served.values()) and len(bulk_facets) <= 1,
    )


@dataclass(frozen=True)
class OptionServedFact:
    """The series a pick of each option of one slot menu would serve.

    `served` maps an option's value dcid to those series and their rows. Keying by series
    makes a union over overlapping answers count each series once.
    """

    property: str
    served: Mapping[str, ServedSeriesRows]

    def rows(self, value: str) -> ServedRows:
        """The rows a pick of `value` would serve; zero for a value not counted."""
        return served_rows(self.served.get(value, {}))


def members_by_value(
    member_values: Mapping[str, str | None], values: Iterable[str]
) -> dict[str, list[str]]:
    """The members of `member_values` carrying each of `values` or one of its descendants,
    in member order: the members a binding of that value would admit, widened as
    `bound_values_from_bindings` widens a bound value (`_expand_values`)."""
    index = get_index()
    admitted: dict[str, list[str]] = {}
    for value in values:
        codes = frozenset(_expand_values(index, [value]))
        admitted[value] = [sv for sv, code in member_values.items() if code in codes]
    return admitted


def bound_values_from_bindings(bindings: Sequence[SlotBindingDraft]) -> dict[str, list[str]]:
    """Collapse a bindings list to the {property_dcid: [value_dcids]} map both
    resolve paths, and engine.bind_repair, treat as "asked for" (kind=value/set
    bindings that actually carry a value). Two bindings naming the same
    property_dcid collide on the dict key; the later one in `bindings` order
    wins, silently. Shared by standard_resolve's own member filter,
    graph_confirm_resolve's own member filter, and engine.bind_repair's
    realizability check, so all three compute "what was bound" the same way.

    Each property's value list is then widened to every bound value's own
    descendants, via the packaged code index's descendant walk (see
    _expand_values). This is the resolver's whole expansion step: a value
    bound on a dimension the index declares ``hierarchical`` resolves to that
    node and its descendants that carry members, and every caller of this
    function sees the widened set by construction -- no dispatch here on
    which resolve path, or which code system, is in play. The widening is a
    genuine no-op for a flat code system or an unindexed property (see
    _expand_values), which is what keeps this safe to call unconditionally
    for every shape, CRS or not. Adds values under an existing key only, and
    never adds a key -- _collapse_bindings above already decided which
    properties are bound.
    """
    index = get_index()
    collapsed = _collapse_bindings(bindings)
    return {prop: _expand_values(index, values) for prop, values in collapsed.items()}


def without_values(
    bindings: Sequence[SlotBindingDraft], properties: Collection[str]
) -> list[SlotBindingDraft]:
    """`bindings` with the slot of each property in `properties` left unbound, in order."""
    from search.engine.bind import SlotBindingDraft

    return [
        SlotBindingDraft(axis=b.axis, property_dcid=b.property_dcid, kind="unbound")
        if b.property_dcid in properties
        else b
        for b in bindings
    ]


def slot_expansions(bindings: Sequence[SlotBindingDraft]) -> dict[str, Expansion]:
    """Per-property ``Expansion`` for a caller building the wire
    ``Slot.expansion`` field: what ``bound_values_from_bindings``' own
    descendant walk added beyond what `bindings` itself asked for, keyed by
    property_dcid.

    Reads the same process-wide code index ``bound_values_from_bindings``
    itself reads (``code_index.shared.get_index``), so the two can never
    disagree about what was added. A property_dcid is absent from the
    returned mapping entirely -- never present with an empty
    ``Expansion.added`` -- when its expansion added nothing: a flat code
    system (crsTypeOfFlow), a property this index carries no code row for at
    all, or a bound value already at a leaf.
    """
    index = get_index()
    collapsed = _collapse_bindings(bindings)
    expansions: dict[str, Expansion] = {}
    for prop, values in collapsed.items():
        original = set(values)
        added_dcids = [d for d in _expand_values(index, values) if d not in original]
        if not added_dcids:
            continue
        expansions[prop] = Expansion(
            added=[
                GraphRef(
                    dcid=dcid,
                    label=row.label if (row := index.code_by_dcid(dcid)) else dcid,
                )
                for dcid in added_dcids
            ]
        )
    return expansions


def _identity_document(identity: BaseIdentity) -> dict[str, object]:
    """Return `identity` as the document the Datasets mint reads."""
    return identity.core_document(mode="json")


def identity_query(identity: BaseIdentity) -> FamilyQuery:
    """Return the `FamilyQuery` of a described `identity`.

    The reader is the Datasets mint's, `code_index.members.identity_family_query`, over the
    document the mint reads: the identity's five-tuple, its slots widened to their
    descendants, and the dataset its provenance facet names.
    """
    return identity_family_query(get_index(), _identity_document(identity))


def source_binding(source: str) -> SlotBindingDraft:
    """The binding that names the dataset `source`: a source-axis value binding. It is the
    one carrier of a named dataset in a bindings list, and `shape_query` reads it."""
    from search.engine.bind import SlotBindingDraft

    return SlotBindingDraft(axis="source", property_dcid=None, kind="value", value_dcids=[source])


def bound_source(bindings: Sequence[SlotBindingDraft]) -> str | None:
    """The dataset `bindings` name: the value of their one source-axis value binding
    (`source_binding`), else `None`. A set or an unbound source binding names none, and
    neither do two value bindings."""
    sources = [
        binding.value_dcids[0]
        for binding in bindings
        if binding.axis == "source" and binding.kind == "value" and len(binding.value_dcids) == 1
    ]
    return sources[0] if len(sources) == 1 else None


def shape_query(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: Collection[str] = ()
) -> FamilyQuery:
    """Return the `FamilyQuery` of `shape` under `bindings`, for a caller that resolves
    members before an identity exists.

    The five-tuple is the shape's, with no denominator ref when the shape's denominator is
    unresolved, as an identity states it (`statvar_members.five_tuple_from_identity`). The
    slots are its what and how slot keys, and each property `bound_values_from_bindings`
    reads from `bindings` binds its widened values.
    `role_keys` holds the role keys the caller binds entities under, the keys the minted
    identity's `entities` will carry (`entity_role_keys`).
    The query names the dataset a source-axis binding binds to one value (`source_binding`),
    as an identity names the dataset its provenance facet binds to one ref. The identity
    minted from the same bindings must bind its source slot from that binding, so that
    `identity_query` returns the same query.
    """
    slots: dict[str, SlotBinding] = {
        slot.property_dcid: ("unbound",)
        for slot in shape.slot_keys
        if slot.property_dcid is not None and slot.axis in ("what", "how")
    }
    for prop, values in bound_values_from_bindings(bindings).items():
        slots[prop] = ("value", values[0]) if len(values) == 1 else ("set", tuple(values))
    return FamilyQuery(
        five_tuple=_member_five_tuple(shape_five_tuple(shape)),
        slots=slots,
        source=bound_source(bindings),
        role_keys=frozenset(role_keys),
    )


def entity_role_keys(shape: ShapeDraft, entities: Mapping[str, Sequence[str]]) -> frozenset[str]:
    """Return the role keys an identity minted for `shape` binds `entities` under.

    `entities` is keyed as `shape.entities_by_role` keys it: `observationAbout`, or a
    role name the shape declares. A declared role maps to its flow axis, the key
    `spec_id.entity_role_key` mints, and keeps its own name when the axis vocabulary does
    not place it. A role with no entity binds nothing."""
    return frozenset(axis_for_role(role) or role for role, dcids in entities.items() if dcids)


def index_decides_membership(identity: BaseIdentity) -> bool:
    """Whether the packaged code index decides which StatVars `identity` denotes.

    The test is the Datasets mint's: a described identity whose slot keys hold every
    constraint property of at least one index row of its five-tuple
    (`code_index.members.index_covers_slot_keys`). A pinned identity names its one
    StatVar and reads no index row.
    """
    document = _identity_document(identity)
    shape = cast("Mapping[str, object]", document["shape"])
    if shape.get("kind") == "stat_var":
        return False
    slots = cast("Sequence[Mapping[str, object]]", document.get("slots", ()))
    return index_covers_slot_keys(
        get_index(),
        five_tuple_from_identity(shape),
        {cast(str, slot["property"]) for slot in slots},
    )


def _identity_family(identity: BaseIdentity) -> DatasetFamily | None:
    """The family the code index resolves for `identity`, or None.

    The resolution is the Datasets mint's, `code_index.members.index_family`, over the
    identity's `identity_query`. None for a pinned identity, when the index does not decide
    the identity's membership, when the election refuses, and when the identity leaves a
    pinned slot unbound (`code_index.members.unbound_pins`), which the mint refuses.
    """
    if not isinstance(identity.shape, IdentityShapeSchema):
        return None
    query = identity_query(identity)
    if unbound_pins(get_index(), query):
        return None
    try:
        return _resolved_family(get_index(), query)
    except SourceNotInFamily:
        return None


def identity_dataset(identity: BaseIdentity) -> str | None:
    """The provenance of the dataset elected for `identity`'s family, or None.

    None wherever `_identity_family` is, and when no index row matches.
    """
    family = _identity_family(identity)
    return family.provenance if family is not None else None


def shape_dataset(shape: ShapeDraft, index: CodeIndex | None = None) -> str | None:
    """The provenance of the dataset that publishes `shape`'s family, the one dataset whose
    index rows hold its five-tuple (`CodeIndex.dataset_by_five_tuple`). None for a shape
    the index does not carry (`index_carries_shape`), a debt series beside CRS flows,
    whose family no declared dataset publishes."""
    index = index or get_index()
    if not index_carries_shape(shape, index):
        return None
    return index.dataset_by_five_tuple.get(_member_five_tuple(shape_five_tuple(shape)))


def identity_family_digest(identity: BaseIdentity) -> str | None:
    """The member digest of the family the code index resolves for `identity`, or None.

    The Datasets mint states the same digest, over the same rows, as
    `trace.family_member_digest` for an identity it resolves from the index. None wherever
    `_identity_family` is, and when no index row matches: the mint then walks the graph or
    refuses, and Search cannot state the family it would read.
    """
    family = _identity_family(identity)
    if family is None or not family.rows:
        return None
    return member_digest(row.dcid for row in family.rows)


def declaration_version(bindings: Sequence[SlotBindingDraft], identity: BaseIdentity) -> str | None:
    """The packaged code index's ``content_hash``, for a caller installing
    ``ResolutionTrace.declaration_version`` -- the same model_copy-after-the-
    fact pattern ``engine.unit.bind.install_unit_facet`` uses for
    ``unit_lexicon_version``.

    The hash covers the code-system and dataset declarations, so the version
    moves when either changes. It is stamped when the index decides
    `identity`'s membership (``index_decides_membership``), which the dataset
    declarations govern even for an identity that binds no code-system
    property, and when a bound property in `bindings` names a code system
    this index declares. None otherwise, mirroring that field's own "None
    when nothing of this kind happened" contract. Non-None does not imply
    expansion added anything (see ``slot_expansions``): a flat system's own
    declaration still governed the (no-op) expansion decision, and is still
    the fact this version identifies.
    """
    index = get_index()
    bound_props = _collapse_bindings(bindings)
    if not index_decides_membership(identity) and not any(
        prop in index.declarations for prop in bound_props
    ):
        return None
    return index.stamp.content_hash


def apply_slot_expansions(slots: list[Slot], expansions: dict[str, Expansion]) -> list[Slot]:
    """Patch each slot whose property_dcid is a key in `expansions` (see
    slot_expansions) with its own Expansion. Order-preserving and a no-op,
    slot by slot, wherever expansions has no entry for that property --
    mirrors spec_build.bind_when_slot's own model_copy-per-match shape for a
    slots list.

    The one shared home for this patch: every region/*.py module that calls
    build_spec applies it through this one function, so a fix lands in one
    place instead of drifting across separate copies. Call this before
    build_spec, on the slots list that build_spec will consume.
    """
    if not expansions:
        return slots
    result = []
    for slot in slots:
        prop = slot.key.property
        if prop is not None and (expansion := expansions.get(prop.dcid)) is not None:
            slot = slot.model_copy(update={"expansion": expansion})
        result.append(slot)
    return result


def apply_slot_summability(slots: list[Slot], identity: BaseIdentity) -> list[Slot]:
    """Set each slot's ``summable``: whether a Datasets ``groupby`` sum over the slot
    yields one complete series for the family `identity` denotes.

    True for a slot whose property is a declared-summable code system
    (``code_index.types.summable_axes``, the axes the datasets door's ``groupby``
    collapses), when the dataset elected for the family (``identity_dataset``) reads an
    absent observation as zero (``code_index.members.absent_member_reads_zero``, the
    reading the mint's sum applies to a member with no rows). False for every other
    slot: one with no property, one on a code system not declared summable, and every
    slot of a family with no elected dataset or with a dataset that declares no absence
    reading, where a member with no rows refuses the sum.
    The binding does not enter. Order-preserving. ``build_spec`` and the bind-time
    sentence event (``_build_sentence_event``) call this on every slots list they
    assemble, so every served slot carries the flag.
    """
    index = get_index()
    summable: frozenset[str] = (
        summable_axes(index.declarations)
        if absent_member_reads_zero(index, identity_dataset(identity))
        else frozenset()
    )
    return [
        slot.model_copy(
            update={
                "summable": slot.key.property is not None and slot.key.property.dcid in summable
            }
        )
        for slot in slots
    ]


def apply_declaration_version(spec: Spec, bindings: Sequence[SlotBindingDraft]) -> Spec:
    """Stamp spec.resolved.trace.declaration_version from `bindings` and
    `spec`'s own identity (see declaration_version) once build_spec has
    assembled `spec` -- the same
    model_copy-after-the-fact style engine.unit.bind.install_unit_facet uses
    for unit_lexicon_version. A no-op, leaving spec unchanged, when
    declaration_version returns None: stamping a version is a claim that the
    index decided something, and None means it never got the chance to.

    The one shared home for this patch; see apply_slot_expansions's docstring
    for why a per-call-site copy is the bug this consolidates away. Call this
    after build_spec, on the Spec it returned.
    """
    version = declaration_version(bindings, spec.definition.identity)
    if version is None:
        return spec
    return spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={
                    "trace": spec.resolved.trace.model_copy(update={"declaration_version": version})
                }
            )
        }
    )


def index_member_rows(shape: ShapeDraft, index: CodeIndex) -> tuple[StatVarRow, ...]:
    """Return every index row of `shape`'s five-tuple, whatever dataset publishes it, the
    rows `five_tuple_member_facts` reads. An exact posted set (`shape.sv_arc_facts_complete`)
    and a shape the index does not carry (`index_carries_shape`) both read no row."""
    if shape.sv_arc_facts_complete or not index_carries_shape(shape, index):
        return ()
    return _index_rows_by_five_tuple(index).get(shape_five_tuple(shape), ())


def five_tuple_member_facts(shape: ShapeDraft) -> tuple[dict[str, MemberFacts], bool]:
    """Every index row of `shape`'s five-tuple, across datasets, else the shape's own
    recall-sourced ``sv_arc_facts``. Returns (member_facts_by_sv, index_sourced).

    The pre-bind readers use this: a shape's slot keys and its taxonomy values describe
    the whole five-tuple, so a property only one dataset's rows carry stays a slot of
    the identity and its values stay bindable. `shape_member_facts` is the post-bind
    reader and keeps one dataset's members.
    """
    rows = index_member_rows(shape, get_index())
    if rows:
        return {row.dcid: IndexMemberFacts(row=row) for row in rows}, True
    return {sv: ArcsMemberFacts(arcs=a) for sv, a in (shape.sv_arc_facts or {}).items()}, False


def _posted_member_facts(shape: ShapeDraft, query: FamilyQuery) -> dict[str, MemberFacts]:
    """A posted member set as posted, each member that is a row of the family the index
    elects for `query` carrying the elected dataset (`materialise.facets.ElectedDataset`, as
    `IndexMemberFacts.read_dataset` builds it). A query the index does not decide, or whose
    election refuses, leaves every member without one."""
    posted: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=a) for sv, a in (shape.sv_arc_facts or {}).items()
    }
    try:
        family = _resolved_family(get_index(), query)
    except SourceNotInFamily:
        return posted
    if family is None or family.provenance is None:
        return posted
    for row in family.rows:
        if row.dcid in posted:
            posted[row.dcid] = ArcsMemberFacts(
                arcs=(shape.sv_arc_facts or {})[row.dcid],
                dataset=IndexMemberFacts(row=row, dataset=family.provenance).read_dataset(),
            )
    return posted


def shape_member_facts(
    shape: ShapeDraft, query: FamilyQuery
) -> tuple[dict[str, MemberFacts], bool]:
    """The member population of `shape` that `query` asks for: the family the code index
    resolves, else the shape's own recall-sourced ``sv_arc_facts``. Returns
    (member_facts_by_sv, index_sourced).

    `query` is the shape's `shape_query` or an identity's `identity_query`. The family is
    `code_index.members.index_family`'s, the function the Datasets mint resolves an
    identity through, so both read the same rows for the same query: one dataset's rows,
    the dataset the query's five-tuple belongs to.

    An exact posted set (`shape.sv_arc_facts_complete`, stamped by the resubmit's shape
    constructor) is returned as posted, with `index_sourced=False`. A posted member that is
    a row of the family the index elects for `query` carries that family's dataset
    (`_posted_member_facts`), as the same member carries it on a first answer.

    For a recall sample the index answers whenever it decides the query's membership
    (`code_index.members.index_covers_slot_keys`). A query whose values match no row, and
    one whose election refuses (the named dataset publishes no matching row), return an
    empty population with `index_sourced=True`: the mint refuses the same identity, so no
    recall member stands in for the family.

    A recall sample the index does not decide is returned narrowed to the role class the
    query's role keys elect (`_recall_role_class`), with `index_sourced=False`. This
    covers a five-tuple the index does not carry and an unresolved denominator, which no
    exported row has.
    """
    recall: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=a) for sv, a in (shape.sv_arc_facts or {}).items()
    }
    if shape.sv_arc_facts_complete:
        return _posted_member_facts(shape, query), False
    index = get_index()
    try:
        family = _resolved_family(index, query)
    except SourceNotInFamily as exc:
        _log_refused_election(exc)
        return {}, True
    if family is not None:
        return {
            row.dcid: IndexMemberFacts(row=row, dataset=family.provenance) for row in family.rows
        }, True
    return _recall_role_class(recall, query), False


def query_members(shape: ShapeDraft, query: FamilyQuery) -> tuple[dict[str, MemberFacts], bool]:
    """The members an answer over `shape` to `query` reads from, before antichain reduction,
    and whether the code index decided them.

    The index's family when it decides `query` (a posted member set is read past, as a
    resubmit of `query` reads the index's family). Otherwise the shape's own members that
    the mint's graph walk confirms for `query`'s slots (`_matches_query_slots`): the family
    a walk settled for the answer, or its recall when no walk settled one.
    """
    members, index_sourced = shape_member_facts(replace(shape, sv_arc_facts_complete=False), query)
    if index_sourced:
        return members, True
    return {sv: m for sv, m in members.items() if _matches_query_slots(m, query.slots)}, False


def _matches_query_slots(member: MemberFacts, slots: Mapping[str, SlotBinding]) -> bool:
    """Whether `member` is one the graph walk confirms for `slots`: every constraint it
    asserts is a slot key holding an allowed value, and it asserts every bound slot
    (`statvar_members.enumerate._confirm_member`)."""
    asserted = asserted_constraint_properties(member)
    if asserted is None or not asserted <= slots.keys():
        return False
    constraints = member.read_constraints()
    for prop, binding in slots.items():
        match binding:
            case ("value", ref):
                if constraints.get(prop) != ref:
                    return False
            case ("set", refs):
                if constraints.get(prop) not in refs:
                    return False
            case _:
                if prop in asserted and prop not in constraints:
                    return False
    return True


def _recall_role_election(
    recall: Mapping[str, MemberFacts], query: FamilyQuery
) -> tuple[frozenset[str], tuple[str, ...]] | None:
    """Return the role set `query` elects among `recall` and its least-constrained
    members, or `None` when the election decides nothing.

    The election is `statvar_members.elect_role_class`, the one the mint's graph walk
    runs, over the recalled members that match the query's slots. It decides nothing when
    the members declare one role set, when none matches the slots, when it ties, and when
    a member's role arc is degenerate."""
    try:
        roles = {sv: member.read_declared_roles() for sv, member in recall.items()}
    except DegenerateObservationProperties:
        return None
    if len({frozenset(declared) for declared in roles.values()}) <= 1:
        return None
    matching = [sv for sv, member in recall.items() if _matches_query_slots(member, query.slots)]
    if not matching:
        return None
    election = elect_role_class(
        {sv: recall[sv].read_constraints() for sv in matching},
        roles,
        matching,
        role_keys=query.role_keys,
        readings=absent_readings(get_index()),
    )
    if election.tied:
        return None
    return frozenset(roles[election.members[0]]), election.members


def _recall_role_class(
    recall: dict[str, MemberFacts], query: FamilyQuery
) -> dict[str, MemberFacts]:
    """Return the members of `recall` that declare the role set `query` elects
    (`_recall_role_election`).

    Every recalled member of the elected role set is returned, matching the query's slots
    or not, so each caller's own bound-value filter still reads the whole class. `recall`
    is returned whole when the election decides nothing."""
    elected = _recall_role_election(recall, query)
    if elected is None:
        return recall
    role_set, _members = elected
    return {
        sv: member
        for sv, member in recall.items()
        if frozenset(member.read_declared_roles()) == role_set
    }


# The longest a member walk may run before Search answers from the members it holds.
_ROLE_CLASS_WALK_BUDGET_S = 8.0


class MembershipAuthority(StrEnum):
    """How Search decided the members an answer reads.

    A settled authority (`AUTHORITATIVE`, `PROVEN`) means the members are the ones the
    Datasets mint resolves for the same identity, seeded with the members Search holds. An
    unsettled one (`SAMPLED`, `FAILED`) means Search answers from the members it recalled
    or was posted, which the mint may not serve.
    """

    AUTHORITATIVE = "authoritative"
    """The code index decides the members, or the mint's walk read every candidate."""
    PROVEN = "proven"
    """The mint's walk hit its candidate window and proved that no unread candidate
    changes its members (`MemberGroup.proven_complete`)."""
    SAMPLED = "sampled"
    """No walk ran, so the recalled or posted members answer."""
    FAILED = "failed"
    """The walk or the read of its members' arcs settled nothing: it ran past its bound,
    a graph read failed, it truncated without proof, it left a candidate unread, or it
    found no member. The recalled or posted members answer."""

    @property
    def settled(self) -> bool:
        return self in (MembershipAuthority.AUTHORITATIVE, MembershipAuthority.PROVEN)


@dataclass(frozen=True)
class MembershipResolution[T]:
    """A membership answer and how Search decided it."""

    value: T
    authority: MembershipAuthority


@dataclass(frozen=True)
class WalkedFamily:
    """The group the mint's walk resolves, and the `time.monotonic()` instant by which every
    read of that walk, and of its members' arcs after it, finishes."""

    group: MemberGroup
    bound: float


_WALK_FAILED: MembershipResolution[WalkedFamily | None] = MembershipResolution(
    None, MembershipAuthority.FAILED
)


class _BoundedArcReader:
    """Read arcs through `graph` until `bound`, a `time.monotonic()` instant. A read
    started past it raises `TimeoutError`, and the bound rides on each read, so every
    page of a paginated read and every request's timeout end by it."""

    def __init__(self, graph: GraphReader, bound: float) -> None:
        self._graph = graph
        self._bound = bound

    def _check(self) -> None:
        if time.monotonic() >= self._bound:
            raise TimeoutError("the role-class walk ran past its bound")

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
    ) -> SubjectPage:
        self._check()
        return self._graph.node_subjects(
            dcid,
            property=property,
            filters=filters,
            max_results=max_results,
            deadline=self._bound,
        )

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self._check()
        return self._graph.node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=self._bound,
        )


class RoleClassWalks:
    """One request's walks of the mint's member enumeration, kept by query and seeds.

    Search runs the walk the mint runs (`statvar_members.enumerate_members`) to learn the
    members an identity the code index does not decide resolves to, which recalled or
    posted members cannot settle. A request asks the same query for its elected shape, on
    each pass, and for alternates, so each distinct query walks once.
    """

    def __init__(self, graph: GraphReader) -> None:
        self._graph = graph
        self._walked: dict[
            Hashable,
            MembershipResolution[WalkedFamily | None]
            | RoleSignatureError
            | FamilyNotResolvableError,
        ] = {}

    def family(
        self, query: FamilyQuery, candidates: Sequence[str], *, deadline: float | None
    ) -> MembershipResolution[WalkedFamily | None]:
        """Return the group the mint's walk resolves for `query`, seeded with `candidates`,
        as `AUTHORITATIVE` when the walk read every candidate and `PROVEN` when it
        truncated with proof (`MemberGroup.proven_complete`).

        The walk is bound by `deadline` and by `_ROLE_CLASS_WALK_BUDGET_S`, whichever comes
        first, and every graph read it makes carries that bound. It settles nothing, a
        `FAILED` resolution with no group, when it runs past that bound, when a graph read
        fails, when it truncates without proof or leaves a candidate unread, and when it
        finds no member.

        Raises `RoleSignatureError` when the walk's members declare role sets the election
        cannot order, which the mint refuses `RoleSignatureIncompatible`, and
        `FamilyNotResolvableError` when a member is at fault, which the mint refuses
        `MemberNotResolvable`."""
        key = self._key(query, candidates)
        if key not in self._walked:
            walked = self._walked[key] = self._walk(query, candidates, deadline=deadline)
            # Every member of a settled family is a candidate the walk confirmed, so the
            # walk seeded with that family resolves it again, so a resubmit that elects a
            # family and then holds it as its shape walks once.
            if isinstance(walked, MembershipResolution) and isinstance(
                settled := walked.value, WalkedFamily
            ):
                self._walked.setdefault(self._key(query, settled.group.family), walked)
        walked = self._walked[key]
        if isinstance(walked, (RoleSignatureError, FamilyNotResolvableError)):
            raise walked
        return walked

    @staticmethod
    def _key(query: FamilyQuery, candidates: Sequence[str]) -> Hashable:
        return (
            query.five_tuple,
            tuple(sorted(query.slots.items())),
            query.role_keys,
            tuple(candidates),
        )

    def _walk(
        self, query: FamilyQuery, candidates: Sequence[str], *, deadline: float | None
    ) -> MembershipResolution[WalkedFamily | None] | RoleSignatureError | FamilyNotResolvableError:
        bound = time.monotonic() + _ROLE_CLASS_WALK_BUDGET_S
        if deadline is not None:
            bound = min(bound, deadline)
        index = get_index()
        try:
            group = enumerate_members(
                five_tuple=query.five_tuple,
                slots=query.slots,
                graph=_BoundedArcReader(self._graph, bound),
                readings=absent_readings(index),
                member_cap=settings().options_member_cap,
                extra_candidates=tuple(candidates),
                role_keys=query.role_keys,
            )
        except MemberRoleDisagreement as exc:
            return RoleSignatureError(str(exc))
        except (
            MemberDegenerateObservationProperties,
            MultiValuedConstraint,
            NotAStatisticalVariable,
        ) as exc:
            return FamilyNotResolvableError(str(exc))
        except (
            MemberQueryUnanchored,
            MemberEnumerationTooLarge,
            GraphReadError,
            GraphInfraError,
            TimeoutError,
        ):
            return _WALK_FAILED
        if (
            not group.members
            or (group.truncated and not group.proven_complete)
            or group.undismissed_candidates
        ):
            return _WALK_FAILED
        authority = (
            MembershipAuthority.PROVEN if group.truncated else MembershipAuthority.AUTHORITATIVE
        )
        return MembershipResolution(WalkedFamily(group=group, bound=bound), authority)


def elected_role_class_shape(
    shape: ShapeDraft,
    query: FamilyQuery,
    *,
    walks: RoleClassWalks | None = None,
    arcs_graph: EngineGraphClient | None = None,
    deadline: float | None = None,
) -> MembershipResolution[ShapeDraft]:
    """Return `shape` holding the family the mint resolves for `query`, and how its members
    were decided.

    The role-class election first runs among the shape's recalled members
    (`_recalled_role_class_shape`). Recalled or posted members can miss the members the
    mint resolves, whichever roles the query binds: a by-creditor sample with only the
    recipient bound, where the mint elects the recipient-only total; a recipient-only
    sample whose role is bound, where the mint elects a by-creditor series that asserts
    fewer constraints; a guaranteed part of a recipient-only total with the guarantee left
    open, where the mint elects the total; and, with no flow role, a female population
    with gender left open, where the mint serves the total population. So for any shape
    the index does not decide, the mint's own walk, seeded with the shape's members as the
    mint is seeded with a resubmit's posted members, settles the members when `walks` and
    `arcs_graph` are given (`RoleClassWalks.family`). A settled shape holds the walked
    family exactly: when the family differs from the shape's members, it replaces them,
    the arcs of a member the shape lacks are read through `arcs_graph` under the walk's
    bound, and the shape takes the name of the family's least-constrained member. A member
    the walk does not confirm is dropped, so no later reader of the shape's members
    answers it. A posted shape (`ShapeDraft.sv_arc_facts_complete`) stays exact, holding
    the family in place of the posted members.

    The authority is `AUTHORITATIVE` when the index decides the shape, and the walk's own
    authority when the walk settles the members. Otherwise the shape's members answer:
    `SAMPLED` when no walk runs and `FAILED` when the walk, or the read of its members'
    arcs, settles nothing.

    Raises `RoleSignatureError` when the walk meets role sets the election cannot order,
    and `FamilyNotResolvableError` when it meets a member at fault: the mint refuses that
    identity, so no sample answers it."""
    narrowed = _recalled_role_class_shape(shape, query)
    if index_covers_slot_keys(get_index(), query.five_tuple, query.slots.keys()):
        return MembershipResolution(narrowed, MembershipAuthority.AUTHORITATIVE)
    if walks is None or arcs_graph is None or not narrowed.sv_arc_facts:
        return MembershipResolution(narrowed, MembershipAuthority.SAMPLED)
    walked = walks.family(query, tuple(narrowed.sv_arc_facts), deadline=deadline)
    if walked.value is None:
        return MembershipResolution(narrowed, walked.authority)
    group = walked.value.group
    recalled = narrowed.sv_arc_facts or {}
    if set(recalled) == set(group.family):
        return MembershipResolution(narrowed, walked.authority)
    unread = [sv for sv in group.family if sv not in recalled]
    arcs_by_dcid: Mapping[str, dict | None] = {}
    if unread:
        try:
            arcs_by_dcid = arcs_graph.node_arcs_batch(unread, deadline=walked.value.bound)
        except GraphInfraError:
            return MembershipResolution(narrowed, MembershipAuthority.FAILED)
    # The shape's own members keep their order, which later readers rank by, and the
    # members the walk adds follow.
    family = set(group.family)
    members = {
        sv: arcs
        for sv in (*(sv for sv in recalled if sv in family), *unread)
        if (arcs := recalled.get(sv) or arcs_by_dcid.get(sv)) is not None
    }
    if len(members) != len(group.family):
        return MembershipResolution(narrowed, MembershipAuthority.FAILED)
    label = next(
        (
            name
            for member in group.members
            if (name := read_name(members.get(member.stat_var, {}))) is not None
        ),
        narrowed.label,
    )
    return MembershipResolution(
        replace(
            narrowed,
            declared_roles=group.members[0].declared_roles,
            optional_roles=frozenset(),
            sv_arc_facts=members,
            label=label,
        ),
        walked.authority,
    )


def posted_members_outside_role_class(
    posted: Mapping[str, dict],
    query: FamilyQuery,
    *,
    walks: RoleClassWalks | None = None,
    deadline: float | None = None,
) -> MembershipResolution[tuple[str, ...]]:
    """Return the posted members that are outside the family `query` resolves to, in order,
    and how that was decided.

    `posted` maps each posted StatVar to its arcs. The mint seeds its walk with the posted
    members, elects one role class for the identity, and refuses a posted member outside
    its family (`MemberSelectionMismatch`), so a resubmit that holds one cannot be answered
    as posted. When `walks` is given, Search runs that walk (`RoleClassWalks.family`) and
    a posted member outside the walked family is outside. Otherwise, and when the walk
    settles nothing, the election runs among the posted members: a member of a role set
    it does not elect is outside.

    The authority is `AUTHORITATIVE` for an identity the index decides, which is left to
    the index's own checks; the walk's own authority when it settles the family; `SAMPLED`
    when no walk runs; and `FAILED` when the walk settles nothing.

    Raises `RoleSignatureError` when the posted members, or the walk's, declare role sets
    the election cannot order, and `FamilyNotResolvableError` when the walk meets a member
    at fault."""
    if not posted or index_covers_slot_keys(get_index(), query.five_tuple, query.slots.keys()):
        return MembershipResolution((), MembershipAuthority.AUTHORITATIVE)
    members: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in posted.items()
    }
    role_sets = {sv: frozenset(member.read_declared_roles()) for sv, member in members.items()}
    authority = MembershipAuthority.SAMPLED
    if walks is not None:
        walked = walks.family(query, tuple(posted), deadline=deadline)
        if walked.value is not None:
            family = frozenset(walked.value.group.family)
            return MembershipResolution(
                tuple(sv for sv in posted if sv not in family), walked.authority
            )
        authority = walked.authority
    if len(set(role_sets.values())) == 1:
        return MembershipResolution((), authority)
    elected = _recall_role_election(members, query)
    if elected is None:
        raise RoleSignatureError("the posted members declare role sets the election cannot order")
    return MembershipResolution(
        tuple(sv for sv in posted if role_sets[sv] != elected[0]), authority
    )


def _recalled_role_class_shape(shape: ShapeDraft, query: FamilyQuery) -> ShapeDraft:
    """Return `shape` narrowed to the role class `query` elects among its recalled members.

    A role-bearing shape can hold members of several role sets and declares their union
    (`retrieval.facts.group_roles`). Once the bound roles are known, the members that answer are
    one role set (`shape_member_facts`). The narrowed shape holds those members, declares
    their roles, and takes the name of the least-constrained one, the member the mint
    serves, so every later reader sees one role signature. `shape` is returned unchanged
    when the index decides the query's membership and when the election decides nothing
    (`_recall_role_election`)."""
    members = shape.sv_arc_facts
    if (
        shape.sv_arc_facts_complete
        or not members
        or index_covers_slot_keys(get_index(), query.five_tuple, query.slots.keys())
    ):
        return shape
    recall: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in members.items()
    }
    elected = _recall_role_election(recall, query)
    if elected is None:
        return shape
    role_set, least_constrained = elected
    kept = {
        sv: arcs
        for sv, arcs in members.items()
        if frozenset(recall[sv].read_declared_roles()) == role_set
    }
    return replace(
        shape,
        declared_roles=recall[least_constrained[0]].read_declared_roles(),
        optional_roles=frozenset(),
        sv_arc_facts=kept,
        label=next(
            (name for sv in least_constrained if (name := read_name(members[sv])) is not None),
            shape.label,
        ),
    )


def shape_pins(shape: ShapeDraft, index: CodeIndex | None = None) -> Mapping[str, str]:
    """The code-system values `shape`'s five-tuple pins: each property every index row of
    the five-tuple carries with one value (`CodeIndex.five_tuple_pins`), mapped to it."""
    pins = (index or get_index()).five_tuple_pins
    return pins.get(_member_five_tuple(shape_five_tuple(shape)), {})


def bind_identity_pins(identity: BaseIdentity) -> BaseIdentity:
    """`identity` with each slot it leaves unbound that its five-tuple pins
    (`code_index.members.unbound_pins`) bound to the pinned value, as `bind_shared_values`
    binds it on a first answer, so a resubmit mints the spec_id a first answer mints for
    the same family. A slot bound to a value stays as posted, and a slot the identity
    omits stays omitted, as a first answer leaves a slot its shape does not declare.
    Returns `identity` itself when nothing changes."""
    if not isinstance(identity.shape, IdentityShapeSchema):
        return identity
    unbound = unbound_pins(get_index(), identity_query(identity))
    if not unbound:
        return identity
    document = _identity_document(identity)
    for slot in cast("list[dict[str, object]]", document["slots"]):
        if (value := unbound.get(cast(str, slot["property"]))) is not None:
            slot["binding"] = {"state": "value", "ref": value}
    return mint_identity(document)[1]


def _unexported_shared_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> dict[str, str]:
    """Each open slot the exported family shares, mapped to the one value every member of
    `shape`'s unexported family holds on it (`bind_shared_values`)."""
    members = shape.sv_arc_facts
    if (
        not members
        or shape.sv_arc_facts_complete
        or shape_five_tuple(shape) not in _index_five_tuples(get_index())
        or index_carries_shape(shape)
    ):
        return {}
    constraints = [ArcsMemberFacts(arcs=arcs).read_constraints() for arcs in members.values()]
    exported = _index_vocabulary(get_index())[shape_five_tuple(shape)]
    closed = {
        binding.property_dcid
        for binding in bindings
        if binding.kind in ("value", "set") or binding.evidence == "requested_all"
    }
    shared: dict[str, str] = {}
    for slot in shape.slot_keys:
        prop = slot.property_dcid
        if prop is None or prop in closed or prop not in exported:
            continue
        values = {member.get(prop) for member in constraints}
        if len(values) == 1 and (value := next(iter(values))) is not None:
            shared[prop] = value
    return shared


def bind_shared_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Bind each what or how slot that can hold one value only.

    Two rules decide such a slot.

    - A slot `shape`'s five-tuple pins (`shape_pins`) binds the pinned value, whatever
      the draft held: unbound, absent, `requested_all` or a value. The pin is measured
      over every index row of the five-tuple, whatever family the other bindings leave,
      so every identity minted from a question on the five-tuple binds it alike (the
      imputed multilateral table's `dac2aAidType` and `crsTypeOfFlow`).
    - An open slot the exported family shares binds the one value every member of an
      unexported family holds on it. The rule applies to a shape whose five-tuple the
      index carries and which the index does not (`index_carries_shape`): debt service
      beside CRS flows. The mint walks the graph for its identity, and with every slot
      open the walk reads the exported family's rows too and runs past its candidate
      window. Only a property the index's rows of the five-tuple also carry
      (`financialFlowType`) is bound: on it the two families hold different values, so
      the shared value tells this family apart and filters the walk. A property of this
      family alone (a guarantee status) stays open, since the recalled members can share
      a value there that the family's total does not hold. A slot the query asked to see
      in full (`requested_all`) stays open.

    Returns the bindings with each such slot bound as a `sole_option` value, and the
    bindings added. Both follow slot-key order."""
    from search.engine.bind import SlotBindingDraft

    values = {**_unexported_shared_values(shape, bindings), **shape_pins(shape)}
    added = [
        SlotBindingDraft(
            axis=slot.axis,
            property_dcid=slot.property_dcid,
            kind="value",
            value_dcids=[values[slot.property_dcid]],
            evidence="sole_option",
        )
        for slot in shape.slot_keys
        if slot.property_dcid in values and slot.axis in ("what", "how")
    ]
    if not added:
        return list(bindings), []
    bound = {binding.property_dcid for binding in added}
    return [b for b in bindings if b.property_dcid not in bound] + added, added


def index_decides_member(
    five_tuple: FiveTuple, constraint_properties: Collection[str], index: CodeIndex | None = None
) -> bool:
    """Whether the packaged code index is the authority for a StatVar's family membership.

    It is when a StatVar keyed on `constraint_properties` resolves through the index at the
    Datasets mint: some index row of `five_tuple` has its every constraint property among
    them (`code_index.members.index_covers_slot_keys`, the test the mint and
    `index_decides_membership` apply to an identity's slot keys). Search and the mint so
    resolve a member through the index under the same condition, whether or not the index
    holds the member's own row. A debt series beside CRS flows lists none of the `crs*`
    properties every CRS row carries, so the graph walk decides its family.
    """
    return index_covers_slot_keys(
        index or get_index(), _member_five_tuple(five_tuple), frozenset(constraint_properties)
    )


def index_carries_five_tuple(five_tuple: FiveTuple, index: CodeIndex | None = None) -> bool:
    """Whether the packaged code index holds a row of `five_tuple`."""
    return five_tuple in _index_five_tuples(index or get_index())


def index_carries_shape(shape: ShapeDraft, index: CodeIndex | None = None) -> bool:
    """Whether the packaged code index carries `shape`: rows of its five-tuple, and, for
    a shape with recalled members, a member whose family the index decides
    (`index_decides_member`). `five_tuple_member_facts` gates on it. It says the index
    describes the shape's slot values.

    `derive_shapes` builds separate shapes for the members the index decides and for
    those it does not, so a shape's members agree on this. A member the index decides is one
    whose constraint keys cover an index row of its five-tuple, the condition under which
    the Datasets mint resolves an identity through the index, so a shape of members the
    index does not decide has an identity whose slot keys cover no index row.
    """
    index = index or get_index()
    five_tuple = shape_five_tuple(shape)
    if five_tuple not in _index_five_tuples(index):
        return False
    members = shape.sv_arc_facts
    return not members or any(
        index_decides_member(five_tuple, _dcids_for(arcs, "constraintProperties"), index)
        for arcs in members.values()
    )


def elect_five_tuple_family(
    shapes: Sequence[ShapeDraft], named: Sequence[QualifierMatch]
) -> list[ShapeDraft]:
    """Keep one family where the index's family and an unindexed family share a
    five-tuple and role class, and return the shapes that remain, in order.

    CRS flows and debt series share a five-tuple, and each is its own shape
    (`derive_shapes`). They are rival readings of one structural question, so one of them
    reaches the election:

    1. The family whose members hold more of the values the query names. `named` holds
       the code rows the query's words name. A shape holds a named value when some member
       asserts that property at the code or one of its descendants: every index row of
       the five-tuple for a shape the index carries, the recalled members for the other.
       "Net disbursements" names a netting basis only the CRS rows hold. A family that
       holds a proper subset of the other's named values is dropped.
    2. Otherwise the family with the stronger recall score. An unscored family loses to a
       scored one.

    Families that neither rule separates both stay.
    """
    index = get_index()
    wanted = {
        (match.code_system, match.code_dcid): frozenset(index.descendants(match.code_dcid))
        | {match.code_dcid}
        for match in named
    }

    def held(shape: ShapeDraft) -> frozenset[tuple[str, str]]:
        if not wanted:
            return frozenset()
        member_facts, _index_sourced = five_tuple_member_facts(shape)
        constraints = [member.read_constraints() for member in member_facts.values()]
        return frozenset(
            key
            for key, values in wanted.items()
            if any(member.get(key[0]) in values for member in constraints)
        )

    def score(shape: ShapeDraft) -> float:
        return -1.0 if shape.representative_score is None else shape.representative_score

    held_by_shape: dict[int, frozenset[tuple[str, str]]] = {}
    dropped: set[int] = set()
    for i, shape in enumerate(shapes):
        for j, other in enumerate(shapes):
            if (
                i == j
                or shape_five_tuple(shape) != shape_five_tuple(other)
                or is_role_bearing(shape) != is_role_bearing(other)
                or index_carries_shape(shape, index) == index_carries_shape(other, index)
            ):
                continue
            for k, candidate in ((i, shape), (j, other)):
                if k not in held_by_shape:
                    held_by_shape[k] = held(candidate)
            mine, theirs = held_by_shape[i], held_by_shape[j]
            if mine < theirs or (not theirs < mine and score(shape) < score(other)):
                dropped.add(i)
    return [shape for i, shape in enumerate(shapes) if i not in dropped]


def declared_role_signature(member_facts: Mapping[str, MemberFacts]) -> tuple[str, ...]:
    """The declared-role signature every member of `member_facts` agrees on.

    Each member's `read_declared_roles()` is compared as a set, so index-sourced and
    recall-sourced populations share one agreement rule. The result follows the first
    member's order, and an empty mapping gives `("observationAbout",)`.

    Raises:
        RoleSignatureError: `member_facts` disagree on declared roles.
    """
    if not member_facts:
        return ("observationAbout",)
    roles_by_sv = {sv: mf.read_declared_roles() for sv, mf in member_facts.items()}
    if len({frozenset(roles) for roles in roles_by_sv.values()}) > 1:
        raise RoleSignatureError(
            "index-sourced population members disagree on declared entity-role "
            f"signatures: {dict(sorted(roles_by_sv.items()))!r}"
        )
    return next(iter(roles_by_sv.values()))


def undeclared_bound_roles(
    role_keys: Sequence[str], member_facts: Mapping[str, MemberFacts]
) -> dict[str, tuple[str, ...]]:
    """Every member of `member_facts` that does not declare some role in
    `role_keys`, mapped to its own declared roles.

    The comparison is `materialise.roles.declared_role_for` alone, the same
    rule `datasets_api.adapter.check_reproducible` applies at the mint door:
    a member absent from the result declares every one of `role_keys` -- a
    key spelled as a declared property, or as a flow-axis alias placing a
    single declared role -- and a member present in it fails at least one,
    either because the key names nothing this member declares or because the
    key names an axis two of its declared roles share.

    Search resolves role-bearing shapes against the roles a member declares
    at read time, while the mint door resolves the identity's own arm
    against the same rule. Running this before a role-keyed probe is what
    keeps the two surfaces agreeing over the roles they both answer under,
    so a candidate this door would refuse for an undeclared role is absent
    from the answer rather than served and refused later.

    An empty `role_keys` returns an empty mapping: no role is bound, so no
    member can fail one.
    """
    undeclared: dict[str, tuple[str, ...]] = {}
    for sv_dcid, member in member_facts.items():
        declared = member.read_declared_roles()
        if any(declared_role_for(role_key, declared) is None for role_key in role_keys):
            undeclared[sv_dcid] = declared
    return undeclared


def _bound_value_filter_facts(
    member_facts: Mapping[str, MemberFacts],
    bound_values: Mapping[str, list[str]],
) -> dict[str, MemberFacts]:
    """The members of `member_facts` no bound value contradicts, read through
    ``MemberFacts.read_constraints`` so a recall-sourced and an index-sourced member take
    the same match. Silence on a bound property passes, and only an explicit mismatch
    excludes. The result is empty when every member contradicts some bound value.
    ``standard.standard_resolve`` then refuses (`retrieve.BoundValuesUnrealisable`), and
    ``bound_value_filter`` falls back to the unfiltered group.
    """
    if not bound_values:
        return dict(member_facts)
    return {
        sv: member
        for sv, member in member_facts.items()
        if not _bound_value_conflicts(member, bound_values)
    }


def _bound_value_conflicts(
    member: MemberFacts, bound_values: Mapping[str, list[str]]
) -> frozenset[str]:
    """The bound properties whose value `member` asserts outside the bound values, the
    filter's own predicate (``_bound_value_filter_facts``). Silence on a bound property
    is no conflict. Empty when the member passes the filter, and unbinding exactly these
    properties lets it pass."""
    constraints = member.read_constraints()
    return frozenset(
        prop
        for prop, vals in bound_values.items()
        if prop in constraints and constraints[prop] not in vals
    )


def _fully_bound_members_facts(
    svs: Sequence[str],
    member_facts: Mapping[str, MemberFacts],
    bound_values: Mapping[str, list[str]],
) -> list[str]:
    """The protocol-generic implementation ``fully_bound_members`` (arcs) and
    ``standard.standard_resolve`` (either source) both reduce to. See
    ``fully_bound_members``'s own docstring for the filter's semantics.
    """
    if not bound_values:
        return list(svs)
    bound_props = set(bound_values)
    return [
        sv for sv in svs if all(prop in member_facts[sv].read_constraints() for prop in bound_props)
    ]


def bound_value_filter(
    sv_arc_facts: dict[str, dict],
    bound_values: dict[str, list[str]],
) -> tuple[dict[str, dict], bool]:
    """Narrow sv_arc_facts to members whose asserted bound properties all match.

    standard_resolve's own filter, over a raw arcs dict. Silence on a bound
    property counts as a partial record (see derive_shapes). Only an EXPLICIT
    contradiction excludes a member here. Fails OPEN to the original, unfiltered
    sv_arc_facts when the filter would empty the set entirely, and the second
    return value says so. No-ops (returns sv_arc_facts unchanged, False) when
    bound_values is empty. engine.bind_repair's realizability check reads it, so
    a repair keeps a binding the filter empties on. standard_resolve refuses such
    a binding (`retrieve.BoundValuesUnrealisable`) and the region unbinds it and
    resolves again (`region.unrealisable`).

    The role-bearing path (retrieval.role_resolve.graph_confirm_resolve) does
    NOT call this. Its own inline filter differs on purpose in three ways: it
    has no fail-open (an excluded set stays excluded, surfacing
    NoDataDraft(reason="binding_mismatch"), which the region does not unbind),
    it drops a member silent on a bound property at admission (standard_resolve
    drops those later, through ``fully_bound_members``, before it probes), and
    it drops a member whose bound property is present but unresolved
    (read_unresolved_constraints), a case this filter does not consider at
    all. All three differences are load-bearing; do not unify the two filters
    on the strength of the seam they share here.

    A thin arcs-dict wrapper over ``_bound_value_filter_facts``, the same
    protocol-generic filter ``standard_resolve`` itself reads member
    facts through (see ``shape_member_facts``) -- kept as its own function,
    with this exact signature, because ``engine.bind_repair`` calls it
    directly with a raw ``sv_arc_facts`` dict and must see byte-identical
    behaviour, never a MemberFacts object.
    """
    member_facts = {sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in sv_arc_facts.items()}
    filtered = _bound_value_filter_facts(member_facts, bound_values)
    if bound_values and not filtered:
        return sv_arc_facts, True
    return {sv: sv_arc_facts[sv] for sv in filtered}, False


def fully_bound_members(
    svs: Sequence[str],
    sv_arc_facts: dict[str, dict],
    bound_values: dict[str, list[str]],
) -> list[str]:
    """Restrict svs to members that explicitly assert every bound property.

    A member's presence in an answer is itself a claim that it satisfies every
    bound axis. A member merely silent on one cannot stand in for one that
    actually asserts it, whatever its own data availability. Presence-only. An
    explicit value mismatch is a separate, earlier concern (the caller's own
    value filter). This only asks whether the property is asserted at all.
    Returns svs unchanged, in order, when bound_values is empty (no axis to
    satisfy). Shared by standard_resolve's entity-probe member filter and its
    no-entity representative selection, so the two agree on what "backs a
    bound claim" means -- and by engine.bind_repair's realizability check.

    A thin arcs-dict wrapper over ``_fully_bound_members_facts``; see
    ``bound_value_filter``'s own docstring for why this stays a separate,
    signature-stable function rather than folding into the generic one.
    """
    member_facts = {sv: ArcsMemberFacts(arcs=sv_arc_facts[sv]) for sv in svs}
    return _fully_bound_members_facts(svs, member_facts, bound_values)


def bound_realizable_members(
    members: Sequence[str],
    sv_arc_facts: dict[str, dict],
    bound_values: dict[str, list[str]],
    *,
    strict: bool = False,
) -> list[str]:
    """standard_resolve's realizability composite, in the order standard_resolve
    applies it. The fail-open value filter (bound_value_filter) runs first, then
    fully_bound_members. Non-empty means some member of
    `members` explicitly backs every bound property. Restricted to `members`
    first (a repair caller may pass a subset of a larger sv_arc_facts dict).
    standard_resolve itself always passes its own full member set, so this is a
    strict generalisation, not a behaviour change. Shared with
    engine.bind_repair.repair_draft_bindings so a repair and the standard
    resolve path resolve these cases the same way. The role-bearing path
    (retrieval.role_resolve.graph_confirm_resolve) does not call this; see
    bound_value_filter's docstring for how its own filter differs.

    strict=False (the default) keeps the fail-open branch. Bound values that
    together match zero members fall back to the unfiltered set, and a member
    merely PRESENT on the property (whether or not it matches) can satisfy the
    composite. standard_resolve() refuses that case
    (`retrieve.BoundValuesUnrealisable`), and the region then unbinds values
    and resolves again (`region.unrealisable`), so a repair keeps such a
    binding and leaves the choice of which values to drop to the region.

    strict=True additionally requires an EXPLICIT match. The value filter must
    not have fallen back (some member actually satisfied every bound value it
    carries, beyond simply not being contradicted), and that same non-fallback
    filtered set must pass fully_bound_members. Returns [] whenever the
    composite would only realize via fail-open.
    """
    scoped = {sv: sv_arc_facts[sv] for sv in members if sv in sv_arc_facts}
    filtered, fell_back = bound_value_filter(scoped, bound_values)
    if strict and fell_back:
        return []
    return fully_bound_members(list(filtered), filtered, bound_values)
