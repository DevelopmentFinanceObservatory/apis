"""Shared bound-value and measure-family logic for Search resolvers.

Every election reads a `code_index.members.FamilyQuery`: `identity_query` builds
it from an identity, as the Datasets mint does, and `shape_query` from a shape and
its bound values. `shape_member_facts` resolves a query through
`code_index.members.index_family`, the function the mint resolves through, and
reads recall arcs only for a population the index does not decide.
`five_tuple_member_facts` reads the index rows of a shape's five-tuple that the datasets
its part may answer from publish (`shape_for_part`), for slot keys and taxonomy values.
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

import asyncio
import logging
import re
import threading
import time
import weakref
from collections import Counter, defaultdict
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from collections.abc import Set as AbstractSet
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from code_index.labels import PRICE_BASIS_AXIS, is_price_basis
from code_index.members import (
    DatasetFamily,
    DatasetScope,
    FamilyQuery,
    ProvenanceScope,
    SourceNotInFamily,
    absent_member_reads_zero,
    absent_readings,
    antichain_minimal_rows,
    canonical_identity,
    carried_properties,
    deciding_query,
    elect_dataset,
    elected_invariants,
    eligible_among,
    eligible_datasets,
    family_among,
    family_query_digest,
    identity_family_query,
    index_decides,
    index_family,
    invariant_bindings,
    is_opt_in,
    listed_group,
    prepare_member_views,
    query_family,
    shared_pins,
)
from code_index.query import IndexFamily, row_five_tuple
from code_index.shared import get_index, get_measure_index
from code_index.types import (
    Bound,
    BoundSet,
    DatasetDeclaration,
    MeasureAxis,
    MeasurePartitionLabel,
    PartitionLabelMatch,
    PropertyBinding,
    label_for_pairs,
    summable_axes,
)
from dc_graph import GraphDeadlineExceededError, GraphReadError, parse_arcs
from dfo_platform.named import cut, named_capped
from graph_release.context import served_origin_key
from materialise.roles import DegenerateObservationProperties, axis_for_role, declared_role_for
from origin_catalogue import current_catalogue
from origin_catalogue.families import FamilyKey
from origin_catalogue.graph import mirrored_arcs
from origin_catalogue.origins import UN_ID_PREFIX
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import (
    EngineInputError,
    FamilyNotResolvableError,
    GraphInfraError,
    MembershipTooBroadError,
    RoleSignatureError,
)
from search.engine.reading_coverage import covers
from search.engine.retrieval.facts import (
    FiveTuple,
    _dcids_for,
    read_constraints,
    read_declared_roles,
    read_name,
    read_slot_taxonomy,
)
from search.engine.retrieval.member_facts import (
    ArcsMemberFacts,
    IndexMemberFacts,
    MemberFacts,
    asserted_constraint_properties,
)
from search.engine.shape import (
    ShapeDraft,
    declares_shape_roles,
    family_label,
    identity_shape_for,
    is_role_bearing,
    role_class,
)
from search.engine.spec_id import mint_identity
from search.models import (
    BaseIdentity,
    Expansion,
    GraphRef,
    IdentityShapeDenominatorRef,
    IdentityShapeDenominatorUnresolved,
    IdentityShapeSchema,
    RoleAxis,
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
    group_of_arcs,
    member_digest,
    shared_enumerator,
)

if TYPE_CHECKING:
    from code_index import CodeIndex
    from code_index.types import MeasureIndex, StatVarRow
    from materialise.graph import GraphReader
    from origin_catalogue.catalogue import Catalogue
    from origin_catalogue.families import FamilySummary
    from search.engine.bind import SlotBindingDraft
    from search.engine.clients.graph import EngineGraphClient
    from search.engine.qualifier_match import QualifierMatch, QualifierTie
    from search.models import Slot, Spec, TimeWindow
    from statvar_members import MemberEnumerator, MemberGroup

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
# A key is a 32-byte digest of the deciding query (`family_query_digest`), so an entry costs its
# rows plus a constant, whatever the query binds.
# Each entry holds a tuple of references to rows the index owns, 8 bytes a row, so the row
# bound caps the memo near 8 MiB: about 80 copies of the dev release's code index's largest family
# (12,771 of its 22,208 rows), where the entry bound alone allowed 4,096 copies (400 MiB).
_FAMILY_CACHE_SIZE = 4096
_FAMILY_CACHE_ROWS = 1 << 20


@dataclass(slots=True)
class _FamilyMemo:
    """One index's resolved families, in insertion order, and the rows they hold."""

    families: dict[bytes, DatasetFamily | None] = field(default_factory=dict)
    rows: int = 0


_index_family_cache: dict[int, _FamilyMemo] = {}
_family_memo_lock = threading.Lock()


def _family_rows(family: DatasetFamily | None) -> int:
    return len(family.rows) if family is not None else 0


def _resolved_family(index: CodeIndex, query: FamilyQuery) -> DatasetFamily | None:
    """`code_index.members.index_family` for `query`, read once per index."""
    key = id(index)
    deciding = deciding_query(index, query)
    query_key = family_query_digest(deciding)
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


def _family_key(
    shape: ShapeDraft, basis_shape_axes: Sequence[MeasureAxis], index: CodeIndex
) -> tuple[object, ...]:
    """Return shape identity excluding free shape axes, but including basis axes and the
    datasets publishing the shape's family (`shape_datasets`), so a CRS family and a DAC1
    family of one five-tuple never collapse into each other."""
    return (
        shape_datasets(shape, index),
        shape.stat_type_dcid,
        shape.meas_denom_dcid,
        shape.meas_denom_unresolved,
        role_class(shape),
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


def _rows_satisfy_all(
    rows: Iterable[StatVarRow], checks: Sequence[tuple[MeasureAxis, str]]
) -> bool:
    """Return whether one of `rows`, a shape's family rows, satisfies all checks.

    The rows supply constraint-axis values absent from the five-tuple; an empty check set
    passes when there is a row.
    """
    return any(
        all(_measure_axis_value(row, axis) == value for axis, value in checks) for row in rows
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
        role_class(shape),
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
    `derive_shapes` reads it: over the members with data at a probed place, else all. It
    stands for those members alone, so it holds no family (`ShapeDraft.held_family`)."""
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
        held_family=None,
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
    targets: dict[int, tuple[StatVarRow, ...]] = {}
    survivors: list[ShapeDraft | None] = [None] * len(shapes)

    for i, shape in enumerate(shapes):
        if not index_carries_shape(shape, index):
            survivors[i] = shape
            continue
        family = shape_family(shape, index)
        targets[i] = family.rows if family is not None else ()
        groups[_family_key(shape, basis_shape_axes, index)].append(i)

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
                if _rows_satisfy_all(targets[int(partition.members[0])], checks)
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
    """Return whether an indexed member of this shape's family (`shape_family`) asserts
    `value` on `axis`."""
    family = shape_family(shape, index)
    return family is not None and _rows_satisfy_all(family.rows, ((axis, value),))


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

    The rows are those the datasets of the family `query`'s slot keys select on its own
    five-tuple (`query_family`) publish, on every five-tuple that differs from it only on a
    free measure axis, so a CRS query reads CRS flows and commitments and no DAC1 row. A
    query naming one of those datasets (`FamilyQuery.source`) reads its rows alone, so a
    sibling five-tuple the named dataset does not publish adds no partition.
    Ignore bound values on declared measure axes so named values cannot hide siblings.

    Keep rows whose role set a member of `shape` could declare
    (`shape.declares_shape_roles`); an empty row declaration means
    `observationAbout`. Shapes without declared roles skip this check.

    The rows are one dataset's: `code_index.members.elect_dataset` applies the election
    the mint applies, over every property `query` binds, measure axes included, and the
    dataset it names, on each five-tuple apart, as the mint elects the family of an
    identity on that five-tuple: the role class one five-tuple's rows elect does not narrow
    another's.

    A shape the index does not carry (`index_carries_shape`) has no measure family.
    """
    target = shape_five_tuple(shape)
    if not index_carries_shape(shape, index):
        return ()
    family = query_family(index, query)
    if family is None:
        return ()
    free_fields = _measure_family_free_fields(axes)
    axis_names = {axis.axis for axis in axes}
    constraints: dict[str, PropertyBinding] = {
        prop: (Bound(values[0]) if len(values) == 1 else BoundSet(frozenset(values)))
        for prop, values in query_bound_values(query).items()
        if prop not in axis_names
    }
    datasets = frozenset(family.datasets)
    if query.source in datasets:
        datasets = frozenset({query.source})
    matched = tuple(
        row
        for row in index.filter(constraints)
        if not datasets.isdisjoint(row.provenances)
        and all(
            getattr(IndexMemberFacts(row=row).read_five_tuple(), field) == getattr(target, field)
            for field in free_fields
        )
        and declares_shape_roles(shape, IndexMemberFacts(row=row).read_declared_roles())
    )
    by_five_tuple: dict[MemberFiveTuple, list[StatVarRow]] = defaultdict(list)
    for row in matched:
        by_five_tuple[row_five_tuple(row)].append(row)
    try:
        elected = {
            row.dcid
            for five_tuple, rows in by_five_tuple.items()
            for row in elect_dataset(index, rows, replace(query, five_tuple=five_tuple)).rows
        }
    except SourceNotInFamily as exc:
        _log_refused_election(exc)
        return ()
    return tuple(row for row in matched if row.dcid in elected)


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
    frozenset[str],
    DatasetScope | None,
]


def measure_family_cache_key(
    shape: ShapeDraft, axes: Sequence[MeasureAxis], query: FamilyQuery
) -> MeasureFamilyCacheKey:
    """Key the exact inputs `_measure_family_rows` reads.

    A measure-axis bound stays out of the row filter, and keys and values are sorted, since
    their input order is irrelevant. The election reads the dataset the query names, its
    bound and open role keys and its scope, so all four enter the key.
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
        query.open_role_keys,
        query.scope,
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

    A resubmit elects such an identity's StatVars from the index. Shape axes are fixed by
    the identity;
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
    each member once. `value_names` maps a value dcid to its label from the shape's value
    labels (`ShapeDraft.value_labels`), for values the code index does not label.
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
    bound_values: Mapping[str, list[str]],
    members: Mapping[str, MemberFacts],
    *,
    value_labels: Mapping[str, str],
) -> tuple[OpenAxisFact, ...]:
    """Return unbound constraint axes in first-observed member order.

    `members` maps each served member's dcid to its facts. Include unresolved
    properties. A member with no resolved value contributes `None`. Each fact carries
    the labels `value_labels` holds for its values: the shape's value labels
    (`ShapeDraft.value_labels`), which one batched label read gave every constraint value
    of the shape's members and family. Return `()` when no open axis exists.
    """
    member_constraints = {dcid: m.read_constraints() for dcid, m in members.items()}
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
                value_names={
                    v: label
                    for v in member_values.values()
                    if v is not None and (label := value_labels.get(v)) is not None
                },
            )
        )
    return tuple(facts)


def served_open_axes(
    bindings: Sequence[SlotBindingDraft],
    served: Sequence[str],
    member_facts: Mapping[str, MemberFacts],
    *,
    value_labels: Mapping[str, str],
) -> tuple[OpenAxisFact, ...]:
    """The open constraint axes an answer states (`open_constraint_axes`): the axes
    `bindings` leave unbound, over the members its read serves, `served`, in that order,
    each read from `member_facts`. A first answer, its resubmit and a nearest-real
    suggestion state the same facts because each reads them here."""
    return open_constraint_axes(
        bound_values_from_bindings(bindings),
        {dcid: member_facts[dcid] for dcid in served},
        value_labels=value_labels,
    )


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
    *,
    shape: ShapeDraft | None = None,
) -> SiblingIdentity:
    """Mint one sibling measure partition's identity by editing the elected
    spec's own already-minted `BaseIdentity`, rather than assembling a fresh
    projection from `Slot`/`Entity` objects: a sibling has no grounded
    `GraphRef` to build a `Slot` from, and identity never hashes labels
    anyway, so fabricating one to satisfy a type would be inventing data
    (see `engine.spec_id.mint_identity`'s own docstring, the seam this mints
    through).

    `elected`'s entities, requested_window and facets describe a fact the
    sibling agrees with -- being a sibling means sharing every axis but the
    declared measure ones -- so they carry over byte-identical, read
    straight off `elected`'s own validated document.
    Its open roles are dropped, since they name roles of the elected family that the
    sibling's family need not declare.

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
    here. A resubmit's own declared slot set is the posted identity's own
    slot properties, so a resubmit of this sibling can freely post a wider slot set
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

    `shape`, when given, is the sibling's own candidate shape, and the identity is the one
    a question electing that shape under `elected`'s bindings mints: its non-measure slots
    are the ones `shape` declares, each carrying `elected`'s binding where `elected`
    declares the property and unbound where it does not, its source is the dataset `shape`
    names (`shape_named_source`), as `named_source.with_named_source` writes it for an
    elected shape, and the identity takes its canonical form (`canonicalise_identity`), so
    a named dataset's invariants bind. Without `shape` the sibling keeps `elected`'s
    non-measure slots and facets, which holds for a sibling of `elected`'s own dataset.

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
    base = elected.model_dump(exclude_none=True)
    base.pop("open_roles", None)
    sibling_slots = [slot for slot in base["slots"] if slot["property"] not in measure_axis_names]
    if shape is not None:
        carried_slots = {slot["property"]: slot for slot in sibling_slots}
        declared = dict.fromkeys(
            key.property_dcid
            for key in shape.slot_keys
            if key.axis in ("what", "how")
            and key.property_dcid is not None
            and key.property_dcid not in measure_axis_names
        )
        sibling_slots = [
            carried_slots.get(prop, {"property": prop, "binding": {"state": "unbound"}})
            for prop in declared
        ]
        facets = {
            key: value for key, value in base.get("facets", {}).items() if key != "provenance"
        }
        if (source := shape_named_source(shape)) is not None:
            facets["provenance"] = {"state": "value", "ref": source}
        base = {key: value for key, value in base.items() if key != "facets"}
        if facets:
            base["facets"] = facets
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
    if shape is not None:
        projection = dict(canonical_identity(get_index(), projection))
    spec_id, identity = mint_identity(projection)
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
    """`values` widened by each value's own code-index expansion
    (``CodeIndex.expansion``: its descendants in a system whose parent-coded records are
    apart from its children's), deduped, first-seen order (each value's own dcid first,
    then its descendants, across every value in `values` in turn).

    A genuine no-op -- returns `values` unchanged, same order -- for a value
    naming a flat code system (e.g. crsTypeOfFlow, per A4c) or one this index
    carries no code row for at all (see ``descendants``' own docstring); the
    latter covers every non-CRS property, since this deployment's index
    carries CRS code systems alone.
    """
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for descendant in index.expansion(value):
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


def applies_value(requested: str, held: str | None) -> bool:
    """Whether a member or binding holding `held` applies the value `requested` the
    request names: `held` is `requested` or, in a system whose parent-coded records are
    apart from its children's, one of its descendants, as a binding of `requested` reads
    (`CodeIndex.expansion`). The one rule for every value the request names, a code term
    (`requirements.judge` on a `CodeRule`), a value named by its graph label or drafted on a
    slot the elected reading lacks (`phase_rank_bind.judge_value_requirements`), and the
    stray values the alternate guard pins (`alternate_guard.AlternateJudge`)."""
    return held is not None and held in get_index().expansion(requested)


def applies_one_of(requested: Iterable[str], held: str | None) -> bool:
    """Whether a member or binding holding `held` applies a set of values the request names
    together, as a set bound on a slot reads: by membership, `held` applying one of them
    (`applies_value`). The one rule for a value or set the bind drafted on a slot the
    elected reading lacks, in the alternate guard (`AlternateJudge._stray_refusal`) and in
    the ledger (`phase_rank_bind.judge_value_requirements`)."""
    return any(applies_value(value, held) for value in requested)


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
    descendants, via the release's code index's descendant walk (see
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
                    ref=dcid,
                    label=row.label if (row := index.code_by_dcid(dcid)) else dcid,
                )
                for dcid in added_dcids
            ]
        )
    return expansions


def _identity_document(identity: BaseIdentity) -> dict[str, object]:
    """Return `identity` as the document the Datasets mint reads."""
    return identity.model_dump(mode="json", exclude_none=True)


def identity_query(identity: BaseIdentity) -> FamilyQuery:
    """Return the `FamilyQuery` of a described `identity`.

    The reader is the Datasets mint's, `code_index.members.identity_family_query`, over the
    document the mint reads: the identity's five-tuple, its slots widened to their
    descendants, and the dataset its provenance facet names.
    """
    return identity_family_query(get_index(), _identity_document(identity))


class PartNotServed(Exception):
    """No dataset a part may answer its shape from holds a row matching its bindings."""


def part_source(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: Callable[[], RoleKeys],
    *,
    scope: ProvenanceScope | None = None,
) -> str | None:
    """The dataset the identity of a part under the source scope `scope` names for `shape`
    under `bindings` and the role keys `role_keys` returns, or `None` when it names none.

    The part is served from one of its own tables: the dataset the shape names
    (`ShapeDraft.named_dataset`), else the one elected among the tables a part naming none
    may answer the shape from (`part_datasets`), by the election every door runs
    (`code_index.members.family_among`). That table's naming identity must resolve rows in
    the index (`code_index.members.index_family`), which is what naming it does at the
    mint. The identity Search mints names only the open roles the elected members serve
    (`DatasetFamily.role_keys` of that election), so the naming identity is read under
    those: a table whose rows tie under the open keys the part asks, and serve under the
    ones that joined, is read as the mint reads it.

    A shape that names a table keeps it. Otherwise the identity names the elected table
    unless the unnamed query elects it twice over: under the role keys the part asks, as
    Search reads the part's members, and under the keys that election ran under, as the
    mint reads the identity. An identity cannot state an exclusion, and its slot keys may
    select another family than the one the shape is built from, so naming the table is
    what makes every door elect it. An identity that names none is the plain question's,
    and every door elects the same table of the part's own for it.

    A shape the index does not carry (`shape_datasets`) keeps the name it has, and so does
    a part whose membership the index does not decide: the mint walks the graph there and
    elects no table. A part that excludes no source and has at most one table of its own,
    and whose slot keys select no other table can be served from no other table, so
    its shape keeps the name it has whatever rows that table holds, and no row is read:
    it names none, or the one opt-in table that alone publishes the shape
    (`shape_for_part`). `role_keys` is
    called once, and only when the shape names no table and its tables or those its slot
    keys select are several, since the role keys decide nothing among fewer.

    Raises `PartNotServed` when another table could serve the part and no table of the
    part's own holds a row matching the bindings together, whatever a table outside its
    scope or a table of another family holds. The part then has no identity that resolves its own
    rows at every door, and the caller refuses it."""
    index = get_index()
    if not shape_datasets(shape, index):
        return shape.named_dataset
    query = replace(shape_query(shape, bindings, ()), source=None)
    if not index_decides(index, query):
        return shape.named_dataset
    dataset = shape.named_dataset
    own = part_datasets(shape, scope, index)
    selected = eligible_datasets(index, query)
    if (
        (scope is None or scope.named)
        and len(own) <= 1
        and set(selected) <= set(own)
        and (dataset is None or (dataset in own and is_opt_in(index, dataset)))
    ):
        return dataset
    if dataset is not None:
        family = _named_family(index, replace(query, source=dataset))
        if family is None or not family.rows:
            raise PartNotServed
        return dataset
    if len(own) > 1 or len(selected) > 1:
        keys = role_keys()
        query = replace(query, role_keys=keys.bound, open_role_keys=keys.open)
    elected = family_among(index, query, own)
    if elected is None or elected.provenance is None:
        raise PartNotServed
    dataset = elected.provenance
    settled = replace(query, open_role_keys=elected.role_keys - query.role_keys)
    family = _named_family(index, replace(settled, source=dataset))
    if family is None or not family.rows:
        raise PartNotServed
    if all(
        (plain := _resolved_family(index, unnamed)) is not None and plain.provenance == dataset
        for unnamed in (query, settled)
    ):
        return None
    return dataset


def _named_family(index: CodeIndex, query: FamilyQuery) -> DatasetFamily | None:
    """The family the index resolves for `query`, which names a dataset
    (`code_index.members.index_family`), or `None` where the election refuses the dataset
    as outside the family the query's slot keys select."""
    try:
        return _resolved_family(index, query)
    except SourceNotInFamily:
        return None


def shape_named_source(shape: ShapeDraft) -> str | None:
    """The dataset a part electing `shape` binds as its source: the one the shape names
    (`ShapeDraft.named_dataset`) when it publishes the shape's family
    (`shape_datasets`), else `None`."""
    source = shape.named_dataset
    if source is None or source not in shape_datasets(shape):
        return None
    return source


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
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: Collection[str],
    *,
    open_role_keys: Collection[str] = (),
) -> FamilyQuery:
    """Return the `FamilyQuery` of `shape` under `bindings`, for a caller that resolves
    members before an identity exists.

    The five-tuple is the shape's, with no denominator ref when the shape's denominator is
    unresolved, as an identity states it (`statvar_members.five_tuple_from_identity`). The
    slots are its what and how slot keys, and each property `bound_values_from_bindings`
    reads from `bindings` binds its widened values.
    `role_keys` holds the role keys the caller binds entities under, the keys the minted
    identity's `entities` will carry, and `open_role_keys` the open axes
    (`entity_role_keys` over the shape's place binding and open axes). Before election
    these are the axes the request asks for. After it, they are the axes the elected members
    serve (`PlaceContext.settled`), which the minted identity's `open_roles` names.
    Every caller states them, since they elect the dataset on a five-tuple several datasets
    publish. A caller reading only the five-tuple, or a shape bound to no place, passes
    none.
    The query names the dataset a source-axis binding binds to one value (`source_binding`),
    as an identity names the dataset its provenance facet binds to one ref. A query built
    after election elects the same family and members as the minted identity's
    `identity_query`, and equals it when the part has no source mention. A part with one
    binds the identity's source to the provenances that served
    (`spec_build.with_served_provenance`), so under an exclusion the plain election already
    honours, the shape names no dataset and this query names none, while the identity's
    query names the dataset that served.
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
        open_role_keys=frozenset(open_role_keys),
    )


@dataclass(frozen=True)
class RoleKeys:
    """The role keys an identity minted for a shape elects under (`entity_role_keys`)."""

    bound: frozenset[str] = frozenset()
    """The keys it binds entities under, as its `entities` keys them."""
    open: frozenset[str] = frozenset()
    """The flow axes it asks every member of, as its `open_roles` names them."""


def entity_role_keys(
    entities: Mapping[str, Sequence[str]], *, open_axes: Collection[RoleAxis]
) -> RoleKeys:
    """Return the role keys an identity binds `entities` under and the flow axes of
    `open_axes` it leaves open.

    `entities` is keyed as `ShapeDraft.entities_by_role` keys it: `observationAbout`, or
    a role name the shape declares. A declared role maps to its flow axis, the key
    `spec_id.entity_role_key` mints, and keeps its own name when the axis vocabulary does
    not place it. A role with no entity binds nothing. The open keys are the axes of
    `open_axes` that no bound key binds, whatever roles the shape declares. Before election
    the caller passes the axes the request asks for, and the members' election decides
    which of them it serves. After election it passes the served axes
    (`PlaceContext.settled`)."""
    bound = frozenset(axis_for_role(role) or role for role, dcids in entities.items() if dcids)
    return RoleKeys(bound=bound, open=frozenset(open_axes) - bound)


def index_decides_membership(identity: BaseIdentity) -> bool:
    """Whether the release's code index decides which StatVars `identity` denotes.

    The test is the Datasets mint's: a described identity whose deciding slot keys hold
    every constraint property of at least one index row of its five-tuple
    (`code_index.members.index_decides`). A pinned identity names its one StatVar and
    reads no index row.
    """
    if identity.shape.kind == "stat_var":
        return False
    return index_decides(get_index(), identity_query(identity))


def _identity_family(identity: BaseIdentity) -> DatasetFamily | None:
    """The family the code index resolves for `identity`, or None.

    The resolution is the Datasets mint's, `code_index.members.index_family`, over the
    identity's `identity_query`. None for a pinned identity, when the index does not decide
    the identity's membership, and when the election refuses.
    """
    if not isinstance(identity.shape, IdentityShapeSchema):
        return None
    query = identity_query(identity)
    try:
        return _resolved_family(get_index(), query)
    except SourceNotInFamily:
        return None


def _index_served_rows(family: DatasetFamily) -> tuple[StatVarRow, ...] | None:
    """The rows the Datasets mint serves from `family`, the family the code index resolves
    for an identity it decides: its antichain-minimal rows. `None` when the family holds no
    row or its role election ties (`DatasetFamily.tied`), which the mint refuses."""
    if family.tied:
        return None
    return antichain_minimal_rows(get_index(), family.rows) or None


def served_members(identity: BaseIdentity) -> frozenset[str] | None:
    """The StatVars a Datasets mint of `identity` serves, when the code index resolves them
    in memory as the mint does (`datasets_api.adapter.resolve_definition_members`).

    An identity the index decides serves the antichain-minimal rows of the family the index
    resolves (`_index_served_rows`). Any other serves the members the index's listed rows
    elect (`code_index.members.listed_group`), the antichain-minimal members the graph walk
    would elect. `None` for a pinned identity, for one the graph walk decides, and wherever the mint refuses:
    a provenance outside the five-tuple's publishers (`SourceNotInFamily`), a tied or empty
    family, an `observationAbout` member, an empty or truncated group, or candidates the
    group leaves undismissed. Raises the listed election's own refusal
    (`MemberEnumerationTooLarge`, `MemberRoleDisagreement`,
    `MemberDegenerateObservationProperties`) when it refuses."""
    if not isinstance(identity.shape, IdentityShapeSchema):
        return None
    index = get_index()
    query = identity_query(identity)
    try:
        family = _resolved_family(index, query)
    except SourceNotInFamily:
        return None
    if family is not None:
        rows = _index_served_rows(family)
        if rows is None or any(
            "observationAbout" in (row.observation_properties or ()) for row in rows
        ):
            return None
        return frozenset(row.dcid for row in rows)
    group = listed_group(index, query, member_cap=shared_enumerator().member_cap)
    if (
        group is None
        or not group.members
        or group.undismissed_candidates
        or (group.truncated and not group.proven_complete)
    ):
        return None
    return frozenset(member.stat_var for member in group.members)


def elects_within(identity: BaseIdentity, dcids: Collection[str]) -> bool:
    """Whether the StatVars a Datasets mint of `identity` serves (`served_members`) are
    known and all among `dcids`. False for members the graph walk decides and for a mint
    that refuses, so a mint of `identity` may serve a StatVar outside `dcids`."""
    try:
        served = served_members(identity)
    except (
        MemberEnumerationTooLarge,
        MemberRoleDisagreement,
        MemberDegenerateObservationProperties,
    ):
        return False
    return served is not None and served <= set(dcids)


def refuse_source_outside_family(identity: BaseIdentity, *, pointer: str) -> None:
    """Refuse `facet_not_available` when `identity` names a dataset that does not publish
    its five-tuple while another dataset does (`code_index.members.index_family`), whatever
    slots it declares, as the Datasets mint refuses it. The resubmit and the options door
    read a posted identity through it. `pointer` is the door's own path to the identity's
    provenance facet."""
    try:
        index_family(get_index(), identity_query(identity))
    except SourceNotInFamily as exc:
        detail = (
            f"the identity binds provenance {cut(exc.source)!r}, which does not publish this "
            f"five-tuple; it is published by {named_capped(exc.candidates)}"
        )
        raise EngineInputError(
            detail,
            code="facet_not_available",
            status=422,
            pointer=pointer,
        ) from exc


def identity_dataset(identity: BaseIdentity) -> str | None:
    """The provenance of the dataset elected for `identity`'s family, or None.

    None wherever `_identity_family` is, and when no index row matches.
    """
    family = _identity_family(identity)
    return family.provenance if family is not None else None


def shape_slot_properties(shape: ShapeDraft) -> frozenset[str]:
    """The constraint properties of `shape`'s what and how slots, the keys that select its
    family (`shape_family`), as `shape_query` declares them."""
    return frozenset(
        slot.property_dcid
        for slot in shape.slot_keys
        if slot.property_dcid is not None and slot.axis in ("what", "how")
    )


def shape_family(shape: ShapeDraft, index: CodeIndex | None = None) -> IndexFamily | None:
    """The index family `shape` is built from, or None for a shape the index does not
    carry (`index_carries_shape`), a debt series beside CRS flows.

    A shape with recalled members reads the one family they belong to (`member_family`):
    the family the index records for a member that is an index row of the shape's
    five-tuple (`CodeIndex.family_by_dcid`), whatever the member's or the shape's keys
    select, and the family a member the index does not hold selects by its constraint
    properties. A shape built from one dataset's rows so reads that dataset's family when
    another's holds fewer properties, or holds properties that overlap its own. A CRS shape
    and a DAC1 shape of one five-tuple each read their own family. Members the index
    decides nothing for are passed over. A shape whose members belong to two families
    reads none: `derive_shapes` builds one shape per family, and the rows of neither family
    stand in for the members of the other.

    A shape with no recalled member reads the family its slot properties select
    (`CodeIndex.family_of`)."""
    index = index or get_index()
    five_tuple = shape_five_tuple(shape)
    if not shape.sv_arc_facts:
        return index.family_of(_member_five_tuple(five_tuple), shape_slot_properties(shape))
    families = {
        id(family): family
        for dcid, arcs in shape.sv_arc_facts.items()
        if (
            family := member_family(
                dcid, five_tuple, _dcids_for(arcs, "constraintProperties"), index
            )
        )
        is not None
    }
    return next(iter(families.values())) if len(families) == 1 else None


def other_families_properties(shape: ShapeDraft, index: CodeIndex | None = None) -> frozenset[str]:
    """The constraint properties only the other families of `shape`'s five-tuple carry:
    those of every family but the shape's own (`shape_family`) that the shape declares no
    slot for. On a CRS shape, `dac1AidType`; none on a five-tuple with one family."""
    index = index or get_index()
    own = shape_family(shape, index)
    families = index.families_by_five_tuple.get(_member_five_tuple(shape_five_tuple(shape)), ())
    others = frozenset().union(
        *(family.properties for family in families if own is None or family != own)
    )
    return others - shape_slot_properties(shape) if own is not None else frozenset()


def shape_datasets(shape: ShapeDraft, index: CodeIndex | None = None) -> tuple[str, ...]:
    """The provenances of the datasets publishing `shape`'s family (`shape_family`), in
    serving order. Empty for a shape the index does not carry (`index_carries_shape`), a
    debt series beside CRS flows, whose family no declared dataset publishes."""
    family = shape_family(shape, index)
    return family.datasets if family is not None else ()


def declared_datasets(shape: ShapeDraft, index: CodeIndex) -> list[DatasetDeclaration]:
    """The declared datasets `shape` answers from: the dataset it names
    (`ShapeDraft.named_dataset`), else those candidacy stamped on it (`ShapeDraft.datasets`),
    and before candidacy, as in the family election, those publishing its family
    (`shape_datasets`), none for a shape the index does not carry."""
    if shape.named_dataset is not None:
        provenances: Sequence[str] = (shape.named_dataset,)
    elif shape.datasets is not None:
        provenances = shape.datasets
    else:
        provenances = shape_datasets(shape, index)
    return [index.datasets[dataset] for dataset in provenances if dataset in index.datasets]


def part_datasets(
    shape: ShapeDraft, scope: ProvenanceScope | None, index: CodeIndex | None = None
) -> tuple[str, ...]:
    """The datasets a part under the source scope `scope`, `None` for none, may answer
    `shape` from, in serving order: the datasets publishing the shape
    (`_publishing_datasets`) that `code_index.members.eligible_among` keeps, so the family
    is not selected again by the shape's slot keys. Shape candidacy stamps them on the
    shape (`ShapeDraft.datasets`)."""
    index = index or get_index()
    return eligible_among(index, _publishing_datasets(shape, index), scope=scope)


def named_scope(dataset: str) -> DatasetScope:
    """The scope of a part that names `dataset` alone."""
    return DatasetScope(allowed=frozenset({dataset}), named=True)


def shape_for_part(
    shape: ShapeDraft, scope: ProvenanceScope | None, index: CodeIndex | None = None
) -> ShapeDraft:
    """`shape` as a part under the source scope `scope` answers from it: stamped with the
    datasets the part may answer from (`part_datasets`) and the dataset the shape names
    (`ShapeDraft.named_dataset`), its index-sourced taxonomy narrowed to those datasets'
    rows (`five_tuple_member_facts`). The slot keys stay the whole family's. The taxonomy
    is kept as it is when the part may answer from every dataset publishing the family.

    The shape names the one dataset the part may answer it from when the scope names a
    source (`ProvenanceScope.named`) or that dataset is opt-in. An unnamed part answers
    from an opt-in dataset only on a family no other dataset publishes
    (`eligible_datasets`), so electing the shape names the dataset, and its identity is
    the one a part naming it mints."""
    index = index or get_index()
    datasets = part_datasets(shape, scope, index)
    named = None
    if len(datasets) == 1 and (
        (scope is not None and scope.named) or is_opt_in(index, datasets[0])
    ):
        (named,) = datasets
    stamped = replace(shape, datasets=datasets, named_dataset=named)
    if not shape.slot_taxonomy_index_sourced or datasets == shape_datasets(shape, index):
        return stamped
    member_facts, _index_sourced = five_tuple_member_facts(stamped)
    return replace(
        stamped,
        slot_taxonomy=read_slot_taxonomy(slot_keys=shape.slot_keys, member_facts=member_facts),
    )


def shape_listed_rows(shape: ShapeDraft, index: CodeIndex | None = None) -> tuple[StatVarRow, ...]:
    """The rows the index lists for `shape`'s five-tuple without a declared dataset
    (`CodeIndex.listed_by_five_tuple`), each with the provenances holding it."""
    index = index or get_index()
    return index.listed_by_five_tuple.get(_member_five_tuple(shape_five_tuple(shape)), ())


def shape_partly_stated(
    shape: ShapeDraft, index: CodeIndex | None = None
) -> Mapping[str, frozenset[tuple[str, str]]]:
    """Each dataset mapped to the (property, value) pairs it states partly on `shape`'s
    five-tuple (`CodeIndex.partly_stated_on`)."""
    index = index or get_index()
    return index.partly_stated_on(_member_five_tuple(shape_five_tuple(shape)))


def shape_part_datasets(shape: ShapeDraft, index: CodeIndex) -> tuple[str, ...]:
    """The datasets stamped on `shape` (`ShapeDraft.datasets`), else every dataset
    publishing it (`_publishing_datasets`)."""
    if shape.datasets is not None:
        return shape.datasets
    return _publishing_datasets(shape, index)


def ratio_base_axis(shape: ShapeDraft, index: CodeIndex) -> RoleAxis | None:
    """The flow axis whose party's own total ratio `shape` divides by, as the datasets it
    answers from (`shape_part_datasets`) declare it (`DatasetDeclaration.ratio_base_axis`):
    `origin` for DAC1's ODA as a share of the donor's GNI, `destination` for DAC2a's share
    of the recipient's. `None` for a shape that divides by nothing, and where the datasets
    declare no axis or disagree."""
    if shape.meas_denom_dcid is None and not shape.meas_denom_unresolved:
        return None
    axes = {
        dataset.ratio_base_axis
        for provenance in shape_part_datasets(shape, index)
        if (dataset := index.datasets.get(provenance)) is not None
    }
    return next(iter(axes)) if len(axes) == 1 else None


def _publishing_datasets(shape: ShapeDraft, index: CodeIndex) -> tuple[str, ...]:
    """Every dataset publishing `shape`'s family (`shape_family`), else, for a shape the
    index does not carry, every dataset publishing its five-tuple, in serving order."""
    family = shape_family(shape, index)
    if family is not None:
        return family.datasets
    return index.datasets_by_five_tuple.get(_member_five_tuple(shape_five_tuple(shape)), ())


@dataclass(frozen=True)
class StatedMembers:
    """What a Datasets mint of an identity states of its members: how many it serves (the
    length of its `trace.resolved_stat_vars`) and the digest of the family they come from
    (`trace.family_member_digest`), `None` when Search cannot name every member of that
    family."""

    served_count: int
    digest: str | None


type WalkKey = tuple[
    MemberFiveTuple, tuple[tuple[str, SlotBinding], ...], frozenset[str], frozenset[str]
]


def walk_key(query: FamilyQuery) -> WalkKey:
    """What a member walk reads of `query`: its five-tuple, slots, and bound and open role
    keys. A walk admits a member stating any ref of a ``set`` binding, so the key holds
    each set as the identity states it, sorted and deduplicated (`IdentityBindingSet`),
    and a query built from the binder's order keys the walk of the identity minted from
    it."""
    return (
        query.five_tuple,
        tuple(sorted((prop, _walked_binding(binding)) for prop, binding in query.slots.items())),
        query.role_keys,
        query.open_role_keys,
    )


def _walked_binding(binding: SlotBinding) -> SlotBinding:
    """`binding` as a walk reads it: a ``set`` binding's refs sorted and deduplicated, and
    a set of one ref the ``value`` binding of it, as `slot_bindings_from_identity` maps
    one."""
    match binding:
        case ("set", refs):
            ordered = tuple(sorted(set(refs)))
            return ("set", ordered) if len(ordered) > 1 else ("value", ordered[0])
        case _:
            return binding


@dataclass(frozen=True)
class WalkedMembers:
    """The members Search's walk settled for the query `key` names (`walk_key`), as the
    mint states them for an identity it resolves by the same walk: `MemberGroup.members`
    are the members it serves and `MemberGroup.stated_family` the family it digests."""

    key: WalkKey
    stated: StatedMembers
    asked: WalkKey | None = None
    """The key of the query the walk ran for, when its open role keys are not the ones
    the walk elected under."""

    @classmethod
    def of(cls, query: FamilyQuery, group: MemberGroup) -> WalkedMembers:
        """`group` as the walk of `query` settled it, keyed by `query` with the open role
        keys the walk elected under (`MemberGroup.elected_open_role_keys`). A part can ask
        an open role its members do not serve, and the identity minted from the answer
        names only the served ones, so its own query carries the elected keys. The mint's
        walk of that identity asks exactly those keys and elects the same members.

        For a walked family the identity names every asked axis the elected members
        declare (`role_resolve.graph_confirm_resolve`), and a member elected under one
        open key can declare a role on another the election set aside. Where those axes
        are every axis asked, the identity's query is `query` itself (`asked`), which the
        mint walks as Search did. Where they sit strictly between the elected keys and
        the asked ones, the identity's query is neither, and the walk settles nothing for
        it (`settles`)."""
        family = group.stated_family
        asked = walk_key(query)
        key = walk_key(replace(query, open_role_keys=group.elected_open_role_keys))
        return cls(
            key=key,
            stated=StatedMembers(
                served_count=len(group.members),
                digest=member_digest(family) if family is not None else None,
            ),
            asked=asked if asked != key else None,
        )

    def settles(self, query: FamilyQuery) -> bool:
        """Whether the walk settled the members `query` asks for: `query` reads as the
        walked query does under its elected open role keys (`key`) or under the ones it
        asked (`asked`)."""
        return walk_key(query) in (self.key, self.asked)

    def binding(self, prop: str, slot: SlotBinding) -> WalkedMembers:
        """These members keyed with `prop` bound by `slot`, for a walk that read `prop`
        open and whose every member asserts the value `slot` binds. The walk of the query
        binding it admits exactly those members, so it settles the same members and
        family."""

        def bound(key: WalkKey) -> WalkKey:
            five_tuple, slots, role_keys, open_role_keys = key
            rebound = {**dict(slots), prop: _walked_binding(slot)}
            return five_tuple, tuple(sorted(rebound.items())), role_keys, open_role_keys

        return replace(
            self, key=bound(self.key), asked=None if self.asked is None else bound(self.asked)
        )


def stated_members(identity: BaseIdentity, walked: WalkedMembers | None) -> StatedMembers | None:
    """What a Datasets mint of `identity`, a described identity, states of its members, when
    Search knows it.

    An identity the code index decides serves the antichain-minimal rows of the family the
    index resolves (`_identity_family`), and the mint digests every row of that family. Any
    other identity serves what the mint's walk settles, which Search knows when `walked` is
    the walk of the identity's own query. `None` otherwise: the walk ran for another query,
    settled nothing, or did not run, or the index resolves no family or one whose role
    election ties (`DatasetFamily.tied`), so the mint walks or refuses. Search answers no
    pinned identity (`check_described_arm`)."""
    if index_decides_membership(identity):
        family = _identity_family(identity)
        if family is None or (served := _index_served_rows(family)) is None:
            return None
        return StatedMembers(
            served_count=len(served), digest=member_digest(row.dcid for row in family.rows)
        )
    if (
        walked is None
        or not isinstance(identity.shape, IdentityShapeSchema)
        or not walked.settles(identity_query(identity))
    ):
        return None
    return walked.stated


def declaration_version(bindings: Sequence[SlotBindingDraft], identity: BaseIdentity) -> str | None:
    """The release's code index's ``content_hash``, for a caller stating
    ``ResolvedSpec.declaration_version`` (``apply_declaration_version``).

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
        if prop is not None and (expansion := expansions.get(prop.ref)) is not None:
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
    The binding does not enter. Order-preserving. ``build_spec`` calls this on every
    slots list it assembles, so every served slot carries the flag.
    """
    index = get_index()
    summable: frozenset[str] = (
        summable_axes(index.declarations)
        if absent_member_reads_zero(index, identity_dataset(identity))
        else frozenset()
    )
    return [
        slot.model_copy(
            update={"summable": slot.key.property is not None and slot.key.property.ref in summable}
        )
        for slot in slots
    ]


def apply_declaration_version(spec: Spec, bindings: Sequence[SlotBindingDraft]) -> Spec:
    """State spec.resolved.declaration_version from `bindings` and `spec`'s own
    identity (see declaration_version) once build_spec has assembled `spec`, as
    engine.unit.bind.with_unit_lexicon_version states the unit lexicon version.
    A no-op, leaving spec unchanged, when
    declaration_version returns None: stamping a version is a claim that the
    index decided something, and None means it never got the chance to.

    The one shared home for this patch; see apply_slot_expansions's docstring
    for why a per-call-site copy is the bug this consolidates away. Call this
    after build_spec, on the Spec it returned.
    """
    version = declaration_version(bindings, spec.definition.identity)
    if version is None:
        return spec
    return spec.model_copy(update={"resolved": spec.resolved.stating_declaration_version(version)})


def index_member_rows(shape: ShapeDraft, index: CodeIndex) -> tuple[StatVarRow, ...]:
    """Return every row of `shape`'s family (`shape_family`), whatever dataset of it
    publishes the row. An exact elected set (`shape.sv_arc_facts_complete`) and a shape the
    index does not carry (`index_carries_shape`) both read no row."""
    if shape.sv_arc_facts_complete:
        return ()
    family = shape_family(shape, index)
    return family.rows if family is not None else ()


def part_member_rows(shape: ShapeDraft, index: CodeIndex) -> tuple[StatVarRow, ...]:
    """The rows of `index_member_rows` that `shape`'s stamped datasets publish
    (`ShapeDraft.datasets`), or all of them for an unstamped shape: the members of the
    datasets its part may answer from, which `five_tuple_member_facts` reads."""
    return _published_by(index_member_rows(shape, index), shape.datasets)


def serving_member_rows(
    shape: ShapeDraft, index: CodeIndex, query: FamilyQuery
) -> tuple[tuple[str, ...] | None, tuple[StatVarRow, ...]]:
    """The datasets that serve `shape` under `query` and their rows of its five-tuple:
    the dataset of the family the index elects for `query` (`code_index.members
    .index_family`, as `shape_member_facts` reads the answer's members), when it is one
    stamped on the shape. A shape stamped with one dataset is served by it, and an
    unstamped shape, or one whose query elects no stamped dataset, by every dataset its
    rows belong to (`None` for an unstamped shape). The alternate guard judges a shape by
    these rows."""
    rows = part_member_rows(shape, index)
    datasets = shape.datasets
    if datasets is None or len(datasets) <= 1:
        return datasets, rows
    try:
        family = index_family(index, query)
    except SourceNotInFamily:
        family = None
    if family is None or family.provenance not in datasets:
        return datasets, rows
    return (family.provenance,), _published_by(rows, (family.provenance,))


def _published_by(
    rows: tuple[StatVarRow, ...], datasets: tuple[str, ...] | None
) -> tuple[StatVarRow, ...]:
    """The rows of `rows` one of `datasets` publishes, or all of them for `None`."""
    if datasets is None:
        return rows
    return tuple(row for row in rows if any(dataset in row.provenances for dataset in datasets))


def five_tuple_member_facts(shape: ShapeDraft) -> tuple[dict[str, MemberFacts], bool]:
    """The rows of `shape`'s family its part may answer from, across those datasets, else
    the shape's own recall-sourced ``sv_arc_facts``. Returns (member_facts_by_sv,
    index_sourced).

    The pre-bind readers use this. A shape that shape candidacy keeps carries the datasets its
    part may answer from (`ShapeDraft.datasets`): the dataset the part names, else every
    eligible publisher of its family (`part_datasets`), so the bind model sees the codes
    of every dataset the election may serve. `derive_shapes` reads the shape before
    candidacy, with no datasets stamped, so the slot keys describe the whole family
    (`shape_family`): a property only one of its datasets' rows carry stays a slot of the
    identity whichever of them serves it, and a property only another family of the
    five-tuple carries is no slot. `shape_member_facts` is the post-bind reader and keeps one
    dataset's members. A shape the index does not carry reads the members it stands for
    (`shape_population_arcs`): the family the catalogue holds for it beside its recalled
    members, else those alone.
    """
    rows = part_member_rows(shape, get_index())
    if rows:
        return {row.dcid: IndexMemberFacts(row=row) for row in rows}, True
    return {sv: ArcsMemberFacts(arcs=a) for sv, a in shape_population_arcs(shape).items()}, False


def identity_member_facts(shape: ShapeDraft) -> tuple[dict[str, MemberFacts], bool]:
    """The members of the family `shape`'s identity denotes, in the form
    `five_tuple_member_facts` returns. A part naming a dataset (`ShapeDraft.named_dataset`)
    denotes that dataset's rows (`part_member_rows`). A part naming none carries no source
    in its identity, so it denotes every row of its family whatever dataset publishes it
    (`index_member_rows`): a source exclusion acts on election, not on the identity. A
    shape the index does not carry reads the members it stands for
    (`shape_population_arcs`).

    A claim that a slot forces one value is a claim about this family, so the readers that
    judge one read it. The menu the binder sees stays narrowed to the part's datasets
    (`shape_for_part`), so a slot the menu offers one code for stays open when a row of an
    excluded dataset holds another."""
    index = get_index()
    rows = (
        part_member_rows(shape, index)
        if shape.named_dataset is not None
        else index_member_rows(shape, index)
    )
    if rows:
        return {row.dcid: IndexMemberFacts(row=row) for row in rows}, True
    return {sv: ArcsMemberFacts(arcs=a) for sv, a in shape_population_arcs(shape).items()}, False


def _exact_member_facts(shape: ShapeDraft, query: FamilyQuery) -> dict[str, MemberFacts]:
    """An exact member set as it stands, each member that is a row of the family the index
    elects for `query` carrying the elected dataset (`materialise.facets.ElectedDataset`, as
    `IndexMemberFacts.read_dataset` builds it). A query the index does not decide, or whose
    election refuses, leaves every member without one."""
    exact: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=a) for sv, a in (shape.sv_arc_facts or {}).items()
    }
    try:
        family = _resolved_family(get_index(), query)
    except SourceNotInFamily:
        return exact
    if family is None or family.provenance is None:
        return exact
    for row in family.rows:
        if row.dcid in exact:
            exact[row.dcid] = ArcsMemberFacts(
                arcs=(shape.sv_arc_facts or {})[row.dcid],
                dataset=IndexMemberFacts(row=row, dataset=family.provenance).read_dataset(),
            )
    return exact


def shape_member_facts(
    shape: ShapeDraft, query: FamilyQuery
) -> tuple[dict[str, MemberFacts], bool]:
    """The member population of `shape` that `query` asks for: the family the code index
    resolves, else the family the catalogue's tables hold for the shape
    (`ShapeDraft.held_family`) beside the shape's own recall-sourced ``sv_arc_facts``, else
    those alone. Returns (member_facts_by_sv, index_sourced).

    `query` is the shape's `shape_query` or an identity's `identity_query`. The family is
    `code_index.members.index_family`'s, the function the Datasets mint resolves an
    identity through, so both read the same rows for the same query: one dataset's rows,
    the dataset the query's five-tuple belongs to.

    An exact elected set (`shape.sv_arc_facts_complete`, stamped by the resubmit's shape
    constructor) is returned as it stands, with `index_sourced=False`. A member that is a
    row of the family the index elects for `query` carries that family's dataset
    (`_exact_member_facts`), as the same member carries it on a first answer.

    For a recall sample the index answers whenever it decides the query's membership
    (`code_index.members.index_decides`). A query whose values match no row, and
    one whose election refuses (the named dataset publishes no matching row), return an
    empty population with `index_sourced=True`: the mint refuses the same identity, so no
    recall member stands in for the family.

    A population the index does not decide is returned narrowed to the role class the
    query's role keys elect (`_recall_role_class`), with `index_sourced=False`. This
    covers a five-tuple the index does not carry and an unresolved denominator, which no
    exported row has. The population is the members the shape stands for
    (`shape_population_arcs`): for a shape stamped with a held family, the recalled members
    and then every other member of that family, so its members are the population its slot
    taxonomy describes, exact when the tables hold the family whole (`held_family_exact`)
    and a floor otherwise. Any other is the recall sample.
    """
    if shape.sv_arc_facts_complete:
        return _exact_member_facts(shape, query), False
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
    recall: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=a) for sv, a in shape_population_arcs(shape).items()
    }
    return _recall_role_class(recall, query), False


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _HeldMembers:
    """The members of a family the catalogue holds, as `_held_members` reads them. Weakly
    referable, so `_held_decodes` shares one decode while a request holds it."""

    arcs: dict[str, dict]
    """Each member's arcs, in table order."""
    roles: dict[str, frozenset[str] | None]
    """Each member's declared roles (`facts.read_declared_roles`), None for a degenerate
    declaration. Empty for a subject-only family, which no role-class election narrows."""
    groups: dict[WalkKey, Future[MemberGroup | _HeldRefusal]] = field(default_factory=dict)
    """The group each query elects over this family when the tables hold it whole, or the
    refusal it meets (`_held_family_group`), by walk key, held as long as the decode is.
    Guarded by `_held_lock`."""


type _HeldKey = tuple[str, str, FamilyKey]
"""A held family's key: the catalogue build, the release whose DFO members the catalogue
joins (`served_origin_key`), and the family."""

# One request's held-family members (`_held_members`), keyed by `_HeldKey`, bound by
# `held_family_memo` for the request's life. A plain answer reads its population three times,
# and option coverage and repair read each held shape of the menu again. On the recorded dev
# build the largest held families are three BLS families of 8,248 members, whose arcs take 196
# to 236 ms and 45 to 46 MB to read (5.5 KB a member), and the UNHCR family of 8,182. A process
# memo holding four such families would pin about 185 MB in a container whose loaded build
# leaves about 1.7 GiB for everything else, and a smaller one evicts a family a menu reads
# beside another. A request holds each family it reads until it ends, and requests share one
# decode through `_held_decodes`, so six concurrent BLS questions hold one copy of each family
# where they would otherwise hold six, about 830 MB. A family no request holds is freed.
_held_family_memo: ContextVar[dict[_HeldKey, _HeldMembers] | None] = ContextVar(
    "search_held_family_memo", default=None
)

_held_lock = threading.Lock()
_held_decodes: weakref.WeakValueDictionary[_HeldKey, _HeldMembers] = weakref.WeakValueDictionary()
"""Each held family some request still holds, by `_HeldKey`. Guarded by `_held_lock`."""
_held_flights: dict[_HeldKey, Future[_HeldMembers]] = {}
"""Each held family being decoded, so a second reader waits for the first decode. Guarded by
`_held_lock`."""


@contextmanager
def held_family_memo() -> Iterator[None]:
    """Bind a fresh held-family member memo (`_held_members`) for one request, or keep the
    one bound: `core.resolve_events` binds one for the request and `resolve_variable` for a
    part, which shares the request's."""
    if _held_family_memo.get() is not None:
        yield
        return
    token = _held_family_memo.set({})
    try:
        yield
    finally:
        _held_family_memo.reset(token)


def shape_population_arcs(shape: ShapeDraft) -> dict[str, dict]:
    """The arcs of the members `shape` stands for: its own members (`sv_arc_facts`), then,
    for a shape stamped with a family the catalogue holds (`ShapeDraft.held_family`), every
    other member of that family, of the role set the role-class election narrowed it to
    when one did (`ShapeDraft.held_roles`). Every reader of a shape's members reads them
    here, so the members agree with the slot taxonomy derive read from the same family."""
    own = shape.sv_arc_facts or {}
    if shape.held_family is None:
        return dict(own)
    held = _held_members(shape.held_family)
    return {
        **own,
        **{
            sv: arcs
            for sv, arcs in held.arcs.items()
            if sv not in own and (shape.held_roles is None or held.roles[sv] == shape.held_roles)
        },
    }


def held_family_arcs(shape: ShapeDraft) -> dict[str, dict]:
    """The arcs of every member of the family the catalogue holds for `shape`
    (`ShapeDraft.held_family`) beside the shape's own members, of every role class: unlike
    `shape_population_arcs`, no role-class election narrows them (`ShapeDraft.held_roles`).
    The shape's own members alone for a shape holding no family. Reads no graph."""
    own = shape.sv_arc_facts or {}
    if shape.held_family is None:
        return dict(own)
    return {
        **own,
        **{sv: arcs for sv, arcs in _held_members(shape.held_family).arcs.items() if sv not in own},
    }


def narrowed_to(shape: ShapeDraft, members: dict[str, dict]) -> ShapeDraft:
    """`shape` standing for `members` alone, some of the members it stands for
    (`shape_population_arcs`). It holds no catalogue family (`ShapeDraft.held_family`). A
    shape narrowed from a held family's members is seeded, when walked, with the members
    among them it held before (`ShapeDraft.walk_seeds`), so the family widens what it
    stands for and never what a walk reads. A shape that held no family stood for its own
    members alone, and keeps its seeds."""
    if shape.held_family is None:
        return replace(shape, sv_arc_facts=members)
    own = shape.sv_arc_facts or {}
    return replace(
        shape,
        sv_arc_facts=members,
        held_family=None,
        walk_seeds=frozenset(
            sv
            for sv in members
            if sv in own and (shape.walk_seeds is None or sv in shape.walk_seeds)
        ),
    )


def shape_constraint_values(shape: ShapeDraft) -> list[tuple[str, str]]:
    """Each (property, value) pair some member `shape` stands for asserts
    (`shape_population_arcs`), its own members' first in member order, then, for a held
    shape, its family's (`FamilySummary.values`), which the catalogue job read with the same
    reader. Reads no member's arcs beyond the shape's own."""
    pairs = dict.fromkeys(
        (prop, value)
        for arcs in (shape.sv_arc_facts or {}).values()
        for prop, value in read_constraints(arcs).items()
    )
    if shape.held_family is not None:
        family = current_catalogue().held_family(shape.held_family)
        if family is not None:
            pairs.update(
                dict.fromkeys(
                    (prop, value) for prop, found in family.values.items() for value in found
                )
            )
    return list(pairs)


def held_family_exact(shape: ShapeDraft) -> bool:
    """Whether the family the catalogue holds for `shape` (`ShapeDraft.held_family`) is every
    member a walk of it can elect, so its population is exact rather than a floor.

    True for a family of UN variables alone (`FamilySummary.un_only`): the UN tables hold
    every admitted variable of each series, and the origin graph client reads a variable of an
    excluded series as absent, so a walk confirms none outside them. False for a family with
    a DFO or base member. The arc tables hold every DFO variable but only the base variables
    a resolver index or a topic names, and the mint's walk enumerates the five-tuple over the
    whole graph, where base Data Commons can hold further members: on the recorded dev
    build 5 of the 31 families holding DFO members alone share a population type with a base
    family, and one family holds both. Such a family is a floor a walk may extend."""
    if shape.held_family is None:
        return False
    family = current_catalogue().held_family(shape.held_family)
    return family is not None and family.un_only


async def read_held_families(shapes: Iterable[ShapeDraft]) -> None:
    """Read the family each of `shapes` stands for (`ShapeDraft.held_family`) into the
    request's memo (`held_family_memo`) off the event loop, so a reader that runs on it
    finds the family read. The phases call it before their first on-loop reader: the bind
    reads the menu's, the elected shape's among them, before the words it nominates read
    their members' names, and settling the election reads any the bind did not, which
    candidate admission, the alternates' walks and refusals read after it. Reads nothing with no
    memo bound or every family already read."""
    memo = _held_family_memo.get()
    if memo is None:
        return
    build_id = current_catalogue().build_id
    origin = served_origin_key()
    keys = list(
        dict.fromkeys(
            shape.held_family
            for shape in shapes
            if shape.held_family is not None and (build_id, origin, shape.held_family) not in memo
        )
    )
    if keys:
        await asyncio.to_thread(lambda: [_held_members(key) for key in keys])


def _held_members(key: FamilyKey) -> _HeldMembers:
    """Every member of family `key` the bound catalogue's tables hold, in table order, with
    the arcs the tables hold for it (`origin_catalogue.graph.mirrored_arcs`), read once a
    request (`held_family_memo`) and shared by every request holding it (`_held_decodes`),
    with one decode at a time per family. Reads no graph. Empty when the tables hold no
    member."""
    catalogue = current_catalogue()
    memo = _held_family_memo.get()
    memo_key = (catalogue.build_id, served_origin_key(), key)
    if memo is not None and (cached := memo.get(memo_key)) is not None:
        return cached
    held = _shared_held_members(memo_key, catalogue)
    if memo is not None:
        held = memo.setdefault(memo_key, held)
    return held


def _shared_held_members(memo_key: _HeldKey, catalogue: Catalogue) -> _HeldMembers:
    """The decode of `memo_key`'s family some request holds, else one decode: the first
    reader decodes and every reader arriving meanwhile waits for it."""
    with _held_lock:
        if (held := _held_decodes.get(memo_key)) is not None:
            return held
        flight = _held_flights.get(memo_key)
        if flight is None:
            flight = _held_flights[memo_key] = Future()
            decoding = True
        else:
            decoding = False
    if not decoding:
        return flight.result()
    try:
        held = _decode_held_members(memo_key[2], catalogue)
    except BaseException as exc:
        with _held_lock:
            del _held_flights[memo_key]
        flight.set_exception(exc)
        raise
    with _held_lock:
        _held_decodes[memo_key] = held
        del _held_flights[memo_key]
    flight.set_result(held)
    return held


def _decode_held_members(key: FamilyKey, catalogue: Catalogue) -> _HeldMembers:
    family = catalogue.held_family(key)
    arcs = (
        {}
        if family is None
        else {
            sv: found
            for sv in family.members
            if (found := mirrored_arcs(sv, catalogue=catalogue)) is not None
        }
    )
    return _HeldMembers(
        arcs=arcs,
        roles={sv: _declared_or_none(found) for sv, found in arcs.items()}
        if key.role_bearing
        else {},
    )


def _declared_or_none(arcs: dict) -> frozenset[str] | None:
    try:
        return frozenset(read_declared_roles(arcs))
    except DegenerateObservationProperties:
        return None


def index_open_role_keys(query: FamilyQuery) -> frozenset[str] | None:
    """The open role keys that joined the election of the family the code index resolves
    for `query` (`DatasetFamily.role_keys` less the bound keys), or `None` when it
    resolves none.

    Asked again with these open keys alone, the index elects the same dataset, rows, keys
    and tie (`code_index.members.elect_dataset`), which no other subset of the open keys
    asked guarantees: among three datasets, a subset between the two can elect another."""
    family = _resolved_family(get_index(), query)
    if family is None:
        return None
    return family.role_keys - query.role_keys


def query_members(shape: ShapeDraft, query: FamilyQuery) -> tuple[dict[str, MemberFacts], bool]:
    """The members an answer over `shape` to `query` reads from, before antichain reduction,
    and whether the code index decided them.

    The index's family when it decides `query` (an exact member set is read past, as a
    resubmit of `query` reads the index's family). Otherwise the members of the population
    `shape_member_facts` reads that the mint's graph walk confirms for `query`'s slots
    (`_matches_query_slots`): the family a walk settled for the answer, the catalogue's
    held family beside the shape's recall when no walk settled one, or its recall alone.
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
        open_role_keys=query.open_role_keys,
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
    Datasets mint resolves for the same identity, seeded on a first answer with the members
    Search recalled. An unsettled one (`SAMPLED`, `FAILED`) means Search answers from the
    members it holds, which the mint may not serve.
    """

    AUTHORITATIVE = "authoritative"
    """The code index decides the members, the catalogue's tables hold the five-tuple whole
    (`_tables_hold_five_tuple`), or the mint's walk read every candidate."""
    PROVEN = "proven"
    """The mint's walk hit its candidate window and proved that no unread candidate
    changes its members (`MemberGroup.proven_complete`)."""
    SAMPLED = "sampled"
    """No walk ran, so the members Search holds answer."""
    FAILED = "failed"
    """The walk or the read of its members' arcs settled nothing: it ran past its bound,
    a graph read failed, it truncated without proof, it left a candidate unread, or it
    found no member. The members Search holds answer."""

    @property
    def settled(self) -> bool:
        return self in (MembershipAuthority.AUTHORITATIVE, MembershipAuthority.PROVEN)


@dataclass(frozen=True)
class MembershipResolution[T]:
    """A membership answer, how Search decided it, and, when Search's walk settled it, what
    the mint states of the walked members (`WalkedMembers`)."""

    value: T
    authority: MembershipAuthority
    walked: WalkedMembers | None = None
    no_member: bool = False
    """Whether the walk, or the catalogue's tables in its place, read the family whole and
    found no member under the query: the mint serves none for the same identity. Such a
    resolution is `FAILED`, and false on every other one, including a walk that could not
    read the family."""


@dataclass(frozen=True)
class WalkedFamily:
    """The group the mint's walk resolves, and the `time.monotonic()` instant by which every
    read of that walk, and of its members' arcs after it, finishes, or `None` for no bound."""

    group: MemberGroup
    bound: float | None


_WALK_FAILED: MembershipResolution[WalkedFamily | None] = MembershipResolution(
    None, MembershipAuthority.FAILED
)
_WALK_FOUND_NONE: MembershipResolution[WalkedFamily | None] = MembershipResolution(
    None, MembershipAuthority.FAILED, no_member=True
)


@dataclass(frozen=True)
class ResubmitWalk:
    """How a resubmit walks. It takes no seed and reads under the request deadline. When the
    first page truncates without proof, it reads the whole filtered set up to
    `subject_ceiling` subjects under `full_read_deadline`, as the Datasets mint does. A
    resubmit holds no recalled members to answer from, so membership is a function of the
    identity alone.
    `spec_id`, the posted identity's, names the walk in its log line."""

    subject_ceiling: int
    full_read_deadline: float | None
    spec_id: str | None = None


class RoleClassWalks:
    """One request's walks of the mint's member enumeration, kept by query and seeds.

    Search runs the walk the mint runs, through the enumerator the Datasets doors share
    (`statvar_members.shared_enumerator` unless `enumerator` is given), to learn the
    members an identity the code index does not decide resolves to, which recalled
    members cannot settle. A family holding a row the index lists outside its code systems
    (`code_index.members.listed_group`) is elected from the index, as the mint elects it,
    and reads no graph. A request asks the same query for its elected shape, on each
    pass, and for alternates, so each distinct query walks once.

    A first answer's walk is seeded with its recalled members and bound by
    `_ROLE_CLASS_WALK_BUDGET_S`. A read failure or a crossed bound settles nothing and
    notes the lost read (`degraded_reads.note_degraded_read`), as does a failed read of the
    arcs of members it adds (`elected_role_class_shape`): the answer then stands on less
    than the walk would have read, so no cache stores it. A resubmit's walk (`resubmit`)
    takes no seed and raises those errors, since no recalled member can answer for it.
    """

    def __init__(
        self,
        graph: GraphReader,
        *,
        enumerator: MemberEnumerator | None = None,
        resubmit: ResubmitWalk | None = None,
    ) -> None:
        self._graph = graph
        self._enumerator = enumerator if enumerator is not None else shared_enumerator()
        self._resubmit = resubmit
        self._walked: dict[
            Hashable,
            MembershipResolution[WalkedFamily | None]
            | RoleSignatureError
            | FamilyNotResolvableError
            | MembershipTooBroadError,
        ] = {}

    def family(
        self, query: FamilyQuery, candidates: Sequence[str], *, deadline: float | None
    ) -> MembershipResolution[WalkedFamily | None]:
        """Return the group the mint's walk resolves for `query`, seeded with `candidates`
        on a first answer, as `AUTHORITATIVE` when the walk read every candidate and
        `PROVEN` when it truncated with proof (`MemberGroup.proven_complete`).

        A first answer's walk is bound by `deadline` and by `_ROLE_CLASS_WALK_BUDGET_S`,
        whichever comes first, and every graph read it makes carries that bound. It settles
        nothing, a `FAILED` resolution with no group, when it runs past that bound, when a
        graph read fails, when it truncates without proof or leaves a candidate unread, and
        when it finds no member. The first two also note a degraded read
        (`degraded_reads.note_degraded_read`), and the last, a family read whole that holds
        no member under the query, says so (`MembershipResolution.no_member`). A reverse read
        holding no candidate is such a read whatever the seeds left unsettled, since the
        mint reads no seed.

        A resubmit's walk ignores `candidates`, reads its first page under `deadline` and
        its full read under `ResubmitWalk.full_read_deadline`. It raises a failed read as
        `GraphInfraError`, a read past a deadline as `TimeoutError`, and
        `MembershipTooBroadError` for a family the mint refuses as too broad: a set past
        the subject ceiling, or more members than the member cap. A query with no measured
        property and a walk whose reads leave a candidate undismissed raise
        `FamilyNotResolvableError`, as the mint refuses them. A walk that settles and finds
        no member is the one `FAILED` resolution a resubmit returns.

        Raises `RoleSignatureError` when the walk's members declare role sets the election
        cannot order, which the mint refuses `RoleSignatureIncompatible`, and
        `FamilyNotResolvableError` when a member is at fault, which the mint refuses
        `MemberNotResolvable`."""
        if self._resubmit is not None:
            candidates = ()
        key = self._key(query, candidates)
        if key not in self._walked:
            walked = self._walked[key] = self._walk(query, candidates, deadline=deadline)
            # Every member of a settled family is a candidate the walk confirmed, so the
            # walk seeded with that family resolves it again, so a first answer that elects
            # a family and then holds it as its shape walks once.
            if isinstance(walked, MembershipResolution) and isinstance(
                settled := walked.value, WalkedFamily
            ):
                self._walked.setdefault(self._key(query, settled.group.family), walked)
        walked = self._walked[key]
        if isinstance(walked, Exception):
            raise walked
        return walked

    @staticmethod
    def _key(query: FamilyQuery, candidates: Sequence[str]) -> Hashable:
        return walk_key(query), tuple(candidates)

    def _walk(
        self, query: FamilyQuery, candidates: Sequence[str], *, deadline: float | None
    ) -> (
        MembershipResolution[WalkedFamily | None]
        | RoleSignatureError
        | FamilyNotResolvableError
        | MembershipTooBroadError
    ):
        resubmit = self._resubmit
        if resubmit is None:
            bound = time.monotonic() + _ROLE_CLASS_WALK_BUDGET_S
            if deadline is not None:
                bound = min(bound, deadline)
        else:
            bound = deadline
        try:
            group = listed_group(get_index(), query, member_cap=self._enumerator.member_cap)
            if group is None:
                group = self._enumerator.enumerate(
                    graph=self._graph,
                    stamp_key=self._graph.generation_stamp().key,
                    five_tuple=query.five_tuple,
                    slots=query.slots,
                    readings=absent_readings(get_index()),
                    extra_candidates=tuple(candidates),
                    role_keys=query.role_keys,
                    open_role_keys=query.open_role_keys,
                    subject_ceiling=None if resubmit is None else resubmit.subject_ceiling,
                    deadline=bound,
                    full_read_deadline=None if resubmit is None else resubmit.full_read_deadline,
                    spec_id=None if resubmit is None else resubmit.spec_id,
                )
        except MemberRoleDisagreement as exc:
            return RoleSignatureError(str(exc))
        except (
            MemberDegenerateObservationProperties,
            MultiValuedConstraint,
            NotAStatisticalVariable,
        ) as exc:
            return FamilyNotResolvableError(str(exc))
        except MemberEnumerationTooLarge as exc:
            if resubmit is not None:
                return MembershipTooBroadError(
                    f"{exc.count} StatVars are consistent with this identity, exceeding the "
                    f"{exc.member_cap}-member cap"
                )
            return _WALK_FAILED
        except MemberQueryUnanchored as exc:
            if resubmit is not None:
                return FamilyNotResolvableError(str(exc))
            return _WALK_FAILED
        except (GraphReadError, GraphInfraError, TimeoutError) as exc:
            if resubmit is None:
                note_degraded_read("member walk")
                return _WALK_FAILED
            if isinstance(exc, GraphDeadlineExceededError):
                raise TimeoutError(f"the member walk's graph read timed out: {exc}") from exc
            if isinstance(exc, GraphReadError):
                raise GraphInfraError(f"the member walk's graph read failed: {exc}") from exc
            raise
        if group.truncated and not group.proven_complete:
            if resubmit is not None:
                return MembershipTooBroadError(
                    f"more than {group.candidate_cap} StatVars match this identity's "
                    "five-tuple and bound values"
                )
            return _WALK_FAILED
        if not group.members and not group.candidates_considered and not group.truncated:
            # The reverse read under the query holds no candidate, and the mint walks that
            # read alone, with no seed, so it serves no member for the identity whatever a
            # seed the walk could not settle reads.
            return _WALK_FOUND_NONE
        if group.undismissed_candidates and resubmit is not None:
            return FamilyNotResolvableError(
                f"the member walk of {query.five_tuple!r} left candidates undismissed: "
                f"{list(group.undismissed_candidates)}"
            )
        if group.undismissed_candidates:
            return _WALK_FAILED
        if not group.members:
            return _WALK_FOUND_NONE
        authority = (
            MembershipAuthority.PROVEN if group.truncated else MembershipAuthority.AUTHORITATIVE
        )
        # A resubmit's later reads run to the bound its walk ran to, which its full read
        # extends.
        if resubmit is not None and group.full_read and resubmit.full_read_deadline is not None:
            bound = resubmit.full_read_deadline
        return MembershipResolution(
            WalkedFamily(group=group, bound=bound),
            authority,
        )


class _HeldRefusal(NamedTuple):
    """A refusal `_elect_held_family` returns, which follows from the tables and the query
    alone. Kept as a record and raised afresh by each reader (`elected_role_class_shape`),
    since an exception shared across requests would pin each reader's frames in its
    traceback."""

    error: type[RoleSignatureError | FamilyNotResolvableError] | None
    """The error each reader raises, or None past the mint's member cap, which settles
    nothing."""
    message: str


def _held_family_group(key: FamilyKey, query: FamilyQuery) -> MemberGroup | _HeldRefusal:
    """The group the mint's walk resolves for `query` over family `key`, whose five-tuple
    the catalogue's tables hold whole (`_tables_hold_five_tuple`), elected from the tables'
    arcs with no graph read (`_elect_held_family`). Elected once per held family (`_HeldKey`)
    and walk key, and shared with every request holding the family's decode
    (`_HeldMembers.groups`): the first reader elects and every reader arriving meanwhile
    waits for it. On the recorded eval build the largest family, UNHCR's 8,182 members,
    elects in about 0.23 s, so the phases call it off the event loop.

    A refusal the walk would make (`_HeldRefusal`) is the entry's result, so every later
    reader reads it; any other failure clears the entry, so the next reader elects again."""
    groups = _held_members(key).groups
    walk = walk_key(query)
    with _held_lock:
        flight = groups.get(walk)
        electing = flight is None
        if flight is None:
            flight = groups[walk] = Future()
    if not electing:
        return flight.result()
    try:
        group = _elect_held_family(key, query)
    except BaseException as exc:
        with _held_lock:
            del groups[walk]
        flight.set_exception(exc)
        raise
    flight.set_result(group)
    return group


def _tables_hold_five_tuple(key: FamilyKey) -> bool:
    """Whether the catalogue's tables hold every variable of family `key`'s five-tuple a
    walk can elect: `key`'s family is UN variables alone (`held_family_exact`), and so is
    the family of the other role class when the tables hold one. A walk reads both
    classes, and a class with a DFO or base member is a floor it may extend."""
    catalogue = current_catalogue()
    own = catalogue.held_family(key)
    other = catalogue.held_family(key._replace(role_bearing=not key.role_bearing))
    return own is not None and own.un_only and (other is None or other.un_only)


def _elect_held_family(key: FamilyKey, query: FamilyQuery) -> MemberGroup | _HeldRefusal:
    """One election of `_held_family_group` through the walk's admission and election
    (`statvar_members.group_of_arcs`). The walk reads every variable of the five-tuple, so
    the family of the other role class joins when the tables hold one. Returns the refusal
    the walk would make, as `RoleClassWalks._walk` maps it: past the member cap, role sets
    the election cannot order (`RoleSignatureError`), and a member at fault
    (`FamilyNotResolvableError`)."""
    arcs = {
        sv: parse_arcs(found, tuple(found), mark_unresolved=True)
        for role_bearing in (False, True)
        for sv, found in _held_members(key._replace(role_bearing=role_bearing)).arcs.items()
    }
    try:
        return group_of_arcs(
            query.five_tuple,
            arcs,
            slots=query.slots,
            readings=absent_readings(get_index()),
            member_cap=shared_enumerator().member_cap,
            role_keys=query.role_keys,
            open_role_keys=query.open_role_keys,
        )
    except MemberEnumerationTooLarge as exc:
        return _HeldRefusal(None, str(exc))
    except MemberRoleDisagreement as exc:
        return _HeldRefusal(RoleSignatureError, str(exc))
    except (
        MemberDegenerateObservationProperties,
        MultiValuedConstraint,
        NotAStatisticalVariable,
    ) as exc:
        return _HeldRefusal(FamilyNotResolvableError, str(exc))


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
    (`_recalled_role_class_shape`). Recalled members can miss the members the
    mint resolves, whichever roles the query binds: a by-creditor sample with only the
    recipient bound, where the mint elects the recipient-only total; a recipient-only
    sample whose role is bound, where the mint elects a by-creditor series that asserts
    fewer constraints; a guaranteed part of a recipient-only total with the guarantee left
    open, where the mint elects the total; and, with no flow role, a female population
    with gender left open, where the mint serves the total population. So for any shape
    the index does not decide, the mint's own walk, seeded on a first answer with the
    shape's members, settles the members when `walks` and
    `arcs_graph` are given (`RoleClassWalks.family`). A settled shape holds the walked
    family exactly: when the family differs from the shape's members, it replaces them,
    the arcs of a member the shape lacks are read through `arcs_graph` under the walk's
    bound, and a settled shape is named by `_walked_label`. A member the walk does not
    confirm is dropped, so no later reader of the shape's members answers it, and a settled
    shape holds no catalogue family (`ShapeDraft.held_family`). An exact shape
    (`ShapeDraft.sv_arc_facts_complete`) stays exact, holding the family in place of its
    members. A failed read of the members' arcs notes a degraded read
    (`degraded_reads.note_degraded_read`).

    The authority is `AUTHORITATIVE` when the index decides the shape, and when the
    catalogue's tables hold its five-tuple whole (`_tables_hold_five_tuple`), which no walk
    could extend: no walk runs, and the tables elect the family as the walk would
    (`_held_family_group`). That election settles nothing (`FAILED`) where the walk's would
    not: past the member cap or with no member. It also settles nothing when it elects a
    member the shape does not stand for (`shape_population_arcs`), since the answer could
    then not serve what it states. Otherwise it is the walk's own authority when the walk
    settles the members, and the shape's members answer: `SAMPLED` when no walk runs and
    `FAILED` when the walk, or the read of its members' arcs, settles nothing. A resolution
    the walk or the tables settled states what the mint states of the members
    (`MembershipResolution.walked`). The walk is seeded with the shape's recalled members
    (`ShapeDraft.walk_seeds`).

    Raises `RoleSignatureError` when the walk meets role sets the election cannot order,
    and `FamilyNotResolvableError` when it meets a member at fault: the mint refuses that
    identity, so no sample answers it."""
    narrowed = _recalled_role_class_shape(shape, query)
    if index_decides(get_index(), query):
        return MembershipResolution(narrowed, MembershipAuthority.AUTHORITATIVE)
    if narrowed.held_family is not None and _tables_hold_five_tuple(narrowed.held_family):
        held_group = _held_family_group(narrowed.held_family, query)
        if isinstance(held_group, _HeldRefusal):
            if held_group.error is None:
                return MembershipResolution(narrowed, MembershipAuthority.FAILED)
            raise held_group.error(held_group.message)
        if not held_group.members:
            return MembershipResolution(narrowed, MembershipAuthority.FAILED, no_member=True)
        if not set(held_group.family) <= shape_population_arcs(narrowed).keys():
            return MembershipResolution(narrowed, MembershipAuthority.FAILED)
        return MembershipResolution(
            narrowed, MembershipAuthority.AUTHORITATIVE, WalkedMembers.of(query, held_group)
        )
    if walks is None or arcs_graph is None or not narrowed.sv_arc_facts:
        return MembershipResolution(narrowed, MembershipAuthority.SAMPLED)
    seeds = tuple(
        sv
        for sv in narrowed.sv_arc_facts
        if narrowed.walk_seeds is None or sv in narrowed.walk_seeds
    )
    walked = walks.family(query, seeds, deadline=deadline)
    if walked.value is None:
        return MembershipResolution(narrowed, walked.authority, no_member=walked.no_member)
    group = walked.value.group
    recalled = narrowed.sv_arc_facts or {}
    settled = WalkedMembers.of(query, group)
    if set(recalled) == set(group.family):
        label = _walked_label(narrowed, group, recalled)
        if (
            narrowed.held_family is not None
            or narrowed.walk_seeds is not None
            or label != narrowed.label
        ):
            narrowed = replace(narrowed, held_family=None, walk_seeds=None, label=label)
        return MembershipResolution(narrowed, walked.authority, settled)
    unread = [sv for sv in group.family if sv not in recalled]
    arcs_by_dcid: Mapping[str, dict | None] = {}
    if unread:
        try:
            arcs_by_dcid = arcs_graph.node_arcs_batch(unread, deadline=walked.value.bound)
        except GraphInfraError:
            note_degraded_read("walked family arcs")
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
    label = _walked_label(narrowed, group, members)
    return MembershipResolution(
        replace(
            narrowed,
            declared_roles=group.members[0].declared_roles,
            optional_roles=frozenset(),
            sv_arc_facts=members,
            held_family=None,
            walk_seeds=None,
            label=label,
        ),
        walked.authority,
        settled,
    )


def walked_family_label(
    members: Mapping[str, dict],
    *,
    held: FamilySummary | None,
    population_type_label: str | None,
    prop_labels: Mapping[str, str],
    value_labels: Mapping[str, str],
) -> str | None:
    """The label of a shape whose membership the graph walk decides, the one rule derive,
    the mint's walk (`elected_role_class_shape`) and a resubmit
    (`promote_shape._phase_build_shape`) name such a shape by.

    `held` is the shape's family as the catalogue's tables hold it (`Catalogue.held_family`),
    and `members` maps the shape's members to their arcs in the caller's order. A member's
    name spells out its own constraint values, which an unbound slot spans, so a family reads
    as its total's name (`FamilySummary.total_name`), the member asserting no constraint, and
    a multi-member UN family with no total as its population type and the values every member
    shares (`shape.family_label`), given `population_type_label`. Without a held family,
    several UN members read that way over their own values. Otherwise the shape reads as the
    first of `members` with a name, so a held family with no total and no UN membership takes
    a label that follows the caller's order, and one shape id can carry the name of any of
    its members. None when no member has a name."""
    if held is not None and held.total_name is not None:
        return held.total_name
    un_family = (
        len(held.members) > 1 and held.un_only
        if held is not None
        else len(members) > 1 and all(sv.startswith(UN_ID_PREFIX) for sv in members)
    )
    if un_family and population_type_label is not None:
        return family_label(
            population_type_label=population_type_label,
            member_constraints=(
                [held.shared]
                if held is not None
                else [read_constraints(arcs) for arcs in members.values()]
            ),
            prop_labels=prop_labels,
            value_labels=value_labels,
        )
    return next(
        (name for dcid, arcs in members.items() if (name := read_name(arcs, dcid)) is not None),
        None,
    )


def _shape_held_family(shape: ShapeDraft) -> FamilySummary | None:
    """The family of `shape`'s five-tuple and role class as the bound catalogue's tables
    hold it (`Catalogue.held_family`), or None when they hold none of its members."""
    return current_catalogue().held_family(
        FamilyKey(*shape_five_tuple(shape), is_role_bearing(shape))
    )


def _walked_label(shape: ShapeDraft, group: MemberGroup, members: Mapping[str, dict]) -> str:
    """The label of `shape` once the walk settles `group`, whose arcs `members` holds. A
    family the catalogue's tables hold keeps the label derive or a resubmit gave it by
    `walked_family_label`, which for a family with no total and no UN membership is the
    name of the first member in the caller's order; any other reads as the first member
    of `group`, least constrained first, with a name, else `shape`'s label."""
    if _shape_held_family(shape) is not None:
        return shape.label
    return (
        walked_family_label(
            {m.stat_var: members[m.stat_var] for m in group.members if m.stat_var in members},
            held=None,
            population_type_label=None,
            prop_labels={},
            value_labels={},
        )
        or shape.label
    )


def _recalled_role_class_shape(shape: ShapeDraft, query: FamilyQuery) -> ShapeDraft:
    """Return `shape` narrowed to the role class `query` elects among the members it stands
    for (`shape_population_arcs`): its recalled members, beside the family the catalogue
    holds for it when it holds one.

    A role-bearing shape can hold members of several role sets and declares their union
    (`retrieval.facts.group_roles`). Once the bound roles are known, the members that answer are
    one role set (`shape_member_facts`). The narrowed shape holds its own members of that set,
    declares their roles, and takes the name of the least-constrained one, the member the
    mint serves, so every later reader sees one role signature. A shape standing for a held
    family keeps the family's label (`walked_family_label`) and its stamp, and records the
    elected role set (`ShapeDraft.held_roles`), so its population is the family's members of
    that set; when none of its own members is of that set it
    holds the set's least-constrained member. `shape` is returned unchanged when the index
    decides the query's membership and when the election decides nothing
    (`_recall_role_election`)."""
    members = shape.sv_arc_facts
    if shape.sv_arc_facts_complete or not members or index_decides(get_index(), query):
        return shape
    population = shape_population_arcs(shape)
    recall: dict[str, MemberFacts] = {
        sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in population.items()
    }
    elected = _recall_role_election(recall, query)
    if elected is None:
        return shape
    role_set, least_constrained = elected
    kept = {
        sv: arcs
        for sv, arcs in members.items()
        if frozenset(recall[sv].read_declared_roles()) == role_set
    } or {least_constrained[0]: population[least_constrained[0]]}
    label = (
        shape.label
        if shape.held_family is not None
        else next(
            (
                name
                for sv in least_constrained
                if (name := read_name(population[sv], sv)) is not None
            ),
            shape.label,
        )
    )
    return replace(
        shape,
        declared_roles=recall[least_constrained[0]].read_declared_roles(),
        optional_roles=frozenset(),
        sv_arc_facts=kept,
        label=label,
        held_roles=role_set if shape.held_family is not None else None,
    )


def shape_pins(shape: ShapeDraft, index: CodeIndex | None = None) -> Mapping[str, str]:
    """The code-system values every dataset `shape`'s part may answer from pins alike
    (`shape_part_datasets`): each property every such dataset's index rows of the
    five-tuple carry with the same one value (`CodeIndex.dataset_pins`), mapped to it. A
    part that names a dataset reads that dataset's pins."""
    index = index or get_index()
    return shared_pins(
        index, shape_part_datasets(shape, index), _member_five_tuple(shape_five_tuple(shape))
    )


def shape_invariants(shape: ShapeDraft, index: CodeIndex | None = None) -> Mapping[str, str]:
    """The invariants the dataset `shape`'s part names declares (`ShapeDraft.named_dataset`),
    mapped to their values, as `code_index.members.elected_invariants` reads them. A part
    that names no dataset implies none. The query declares the shape's what and how slot
    keys unbound, as `shape_query` does."""
    query = replace(shape_query(shape, (), ()), source=shape.named_dataset)
    return elected_invariants(index or get_index(), query)


def canonicalise_identity(identity: BaseIdentity) -> BaseIdentity:
    """`identity` in its canonical form (`code_index.members.canonical_identity`): each slot
    an invariant of the dataset the identity names fixes binds the invariant's value, and an
    identity that names no dataset keeps its bindings. Every Search door that reads a posted
    identity reads it through this function, so an identity naming a dataset posted with an
    invariant bound and one posted without mint one spec_id and elect one family. Returns
    `identity` itself when nothing changes."""
    document = _identity_document(identity)
    canonical = canonical_identity(get_index(), document)
    if canonical is document:
        return identity
    return mint_identity(canonical)[1]


def _unexported_shared_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> dict[str, str]:
    """Each open slot the exported family shares, mapped to the one value every member of
    `shape`'s unexported family holds on it (`bind_shared_values`)."""
    members = shape.sv_arc_facts
    index = get_index()
    five_tuple = shape_five_tuple(shape)
    if (
        not members
        or shape.sv_arc_facts_complete
        or five_tuple not in _index_five_tuples(index)
        or index_carries_shape(shape)
    ):
        return {}
    constraints = [ArcsMemberFacts(arcs=arcs).read_constraints() for arcs in members.values()]
    exported = carried_properties(
        index, shape_part_datasets(shape, index), _member_five_tuple(five_tuple)
    )
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
        if (value := sole_shared_value(constraints, prop)) is not None:
            shared[prop] = value
    return shared


def sole_shared_value(constraints: Sequence[Mapping[str, str]], prop: str) -> str | None:
    """The one value every member's constraints (`read_constraints`) assert on `prop`, else
    None. A member asserting nothing on `prop`, as a family's total does, or two members
    asserting different values leave the property without a forced value."""
    values = {member.get(prop) for member in constraints}
    return next(iter(values)) if len(values) == 1 else None


def bind_shared_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Bind each open slot the exported family shares to the one value every member of an
    unexported family holds on it.

    The rule applies to a shape whose five-tuple the index carries and which the index does
    not (`index_carries_shape`): debt service beside CRS flows. The mint walks the graph for
    its identity, and with every slot open the walk reads the exported family's rows too and
    runs past its candidate window. Only a property the index's rows of the five-tuple also
    carry (`financialFlowType`) is bound: on it the two families hold different values, so
    the shared value tells this family apart and filters the walk. A property of this
    family alone (a guarantee status) stays open, since the recalled members can share a
    value there that the family's total does not hold. A slot the query asked to see in full
    (`requested_all`) stays open.

    Returns the bindings with each such slot bound as a `sole_option` value, and the
    bindings added. Both follow slot-key order."""
    return _bind_sole_options(shape, bindings, _unexported_shared_values(shape, bindings))


def bind_member_values(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], members: Collection[str]
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """`bindings` with each what or how slot of `shape` on which every StatVar of `members`
    holds one value, by its index row, bound to that value as a `sole_option` draft, in
    place of what `bindings` drafted there, and the drafts of `bindings` that ask to see
    such a slot in full (`requested_all`).

    A part's series code names its StatVars (`ResolveState.source_code`), so the identity
    minted from the result elects those StatVars and no broader family: DT.TDS.DEGG.CD binds
    `idsDebtorType` to `IDSDebtorTypeGG`, where an open slot would elect total debt service
    over every debtor. The code's value replaces a value drafted from the part's words, since
    the code is exact. A member the index holds no row for binds nothing.

    A slot asked in full that the members hold one value on is a request they cannot
    answer: "across all debtor types" beside DT.TDS.DEGG.CD. `bindings` then return as
    drafted beside those drafts, for the caller to refuse the part."""
    index = get_index()
    rows = [index.statvar_by_dcid(dcid) for dcid in members]
    if not rows or any(row is None for row in rows):
        return list(bindings), []
    first, *rest = (row.constraint_properties for row in rows if row is not None)
    shared = {
        prop: value for prop, value in first.items() if all(c.get(prop) == value for c in rest)
    }
    bound, added = _bind_sole_options(shape, bindings, shared)
    fixed = {binding.property_dcid for binding in added}
    asked_in_full = [
        binding
        for binding in bindings
        if binding.evidence == "requested_all" and binding.property_dcid in fixed
    ]
    if asked_in_full:
        return list(bindings), asked_in_full
    return bound, []


def canonical_pin_bindings(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: RoleKeys
) -> list[SlotBindingDraft]:
    """`bindings` in the canonical form of the identity minted from them
    (`code_index.members.invariant_bindings`): each invariant slot of the dataset the part
    names that `bindings` leave unbound, or draw no draft for, bound to its value as a
    `sole_option` draft. The query is the `shape_query` of `shape`, `bindings` and
    `role_keys`, which names the dataset the source binding names, as the identity minted
    from them names it in its provenance facet. The identity minted from the result is
    canonical, and every reader of the bindings after it (option coverage, qualifier
    dispositions, a refusal's flagged identity) reads the values it binds."""
    query = shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open)
    return _bind_sole_options(shape, bindings, invariant_bindings(get_index(), query))[0]


def _bind_sole_options(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], values: Mapping[str, str]
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """`bindings` with each what or how slot of `shape` that `values` names bound to its
    value as a `sole_option` draft, and the drafts added, in slot-key order."""
    from search.engine.bind import SlotBindingDraft

    added = [
        SlotBindingDraft(
            axis=slot.axis,
            property_dcid=slot.property_dcid,
            kind="value",
            value_dcids=[values[slot.property_dcid]],
            evidence="sole_option",
        )
        for slot in shape.slot_keys
        if slot.property_dcid is not None
        and slot.property_dcid in values
        and slot.axis in ("what", "how")
    ]
    if not added:
        return list(bindings), []
    bound = {binding.property_dcid for binding in added}
    return [b for b in bindings if b.property_dcid not in bound] + added, added


def member_family(
    dcid: str,
    five_tuple: FiveTuple,
    constraint_properties: Collection[str],
    index: CodeIndex | None = None,
) -> IndexFamily | None:
    """The index family StatVar `dcid` belongs to, or None when the index does not decide
    its membership. Derive groups members by it, so a CRS member and a DAC1 member of one
    five-tuple land in different shapes, and a shape reads its family from its members
    through it (`shape_family`).

    An index row of `five_tuple` belongs to the family its provenances assign it
    (`CodeIndex.family_by_dcid`), so a dataset's rows stay together when another dataset's
    family holds fewer properties, or holds properties that overlap its own so that the
    row's keys select neither. Another StatVar belongs to the family its
    `constraint_properties` select (`CodeIndex.family_of`) when they hold every constraint
    property of one of its rows (`IndexFamily.covers`). That is the condition under which
    the Datasets mint resolves an identity with those slot keys through the index
    (`code_index.members.index_covers_slot_keys`). A debt series beside CRS flows lists
    none of the `crs*` properties every CRS row carries, so the graph walk decides its
    family."""
    index = index or get_index()
    member_five_tuple = _member_five_tuple(five_tuple)
    family = index.family_by_dcid.get(dcid)
    if family is not None and family.five_tuple == member_five_tuple:
        return family
    keys = frozenset(constraint_properties)
    family = index.family_of(member_five_tuple, keys)
    return family if family is not None and family.covers(keys) else None


def five_tuple_families(
    five_tuple: FiveTuple, index: CodeIndex | None = None
) -> tuple[IndexFamily, ...]:
    """The index families of `five_tuple` (`CodeIndex.families_by_five_tuple`), in serving
    order. Empty for a five-tuple the index does not carry."""
    return (index or get_index()).families_by_five_tuple.get(_member_five_tuple(five_tuple), ())


def prepare_index_views() -> None:
    """Loads the release's code index and measure index and builds the views of the index
    that search reads on every resolve: its five-tuples (`index_carries_five_tuple`) and
    the member views (`code_index.members.prepare_member_views`). Each is otherwise built
    by the first request that reads it, about 135 ms on the dev release's code index."""
    index = get_index()
    get_measure_index()
    _index_five_tuples(index)
    prepare_member_views(index)


def index_carries_five_tuple(five_tuple: FiveTuple, index: CodeIndex | None = None) -> bool:
    """Whether the release's code index holds a row of `five_tuple`."""
    return five_tuple in _index_five_tuples(index or get_index())


def index_carries_shape(shape: ShapeDraft, index: CodeIndex | None = None) -> bool:
    """Whether the release's code index carries `shape`: rows of its five-tuple, and, for
    a shape with recalled members, the one family those members belong to
    (`shape_family`). `five_tuple_member_facts` gates on it. It says the index describes the
    shape's slot values.

    `derive_shapes` builds separate shapes for the members of each index family and for
    those the index does not decide, so a shape's members agree on this. A member the index
    decides is an index row of the five-tuple, or one whose constraint keys cover an index
    row of it (`member_family`), the condition under which the Datasets mint resolves an
    identity through the index. A shape of members the index does not decide has an
    identity whose slot keys cover no index row.
    """
    index = index or get_index()
    if shape_five_tuple(shape) not in _index_five_tuples(index):
        return False
    return not shape.sv_arc_facts or shape_family(shape, index) is not None


def held_codes(shape: ShapeDraft) -> Callable[[QualifierMatch], bool]:
    """Whether `shape`'s family holds a match's value, read once and kept per code.

    A code's family holds it when some member asserts the code's property at the code or
    one of its descendants. The members are the rows of its family that its part may answer
    from for a shape the index carries, its recalled members for any other
    (`five_tuple_member_facts`), read on the first call. A measure value on a five-tuple
    axis ("grant equivalent" names `measuredProperty` `grantEquivalent`) is held when the
    shape's five-tuple carries it, and one on a constraint axis ("net" names
    `nettingBasis` `Net`) as a code is.
    """
    index = get_index()
    constraints: list[dict[str, str]] | None = None
    held: dict[tuple[str, str], bool] = {}

    def holds(match: QualifierMatch) -> bool:
        nonlocal constraints
        key = (match.code_system, match.code_dcid)
        if key in held:
            return held[key]
        field_name = _MEASURE_AXIS_DRAFT_FIELDS.get(match.code_system)
        if field_name is not None:
            held[key] = getattr(shape, field_name) == match.code_dcid
            return held[key]
        if constraints is None:
            member_facts, _index_sourced = five_tuple_member_facts(shape)
            constraints = [member.read_constraints() for member in member_facts.values()]
        values = frozenset(index.descendants(match.code_dcid)) | {match.code_dcid}
        held[key] = any(member.get(match.code_system) in values for member in constraints)
        return held[key]

    return holds


def query_phrases_read(
    holds: Callable[[QualifierMatch], bool],
    matches: Iterable[QualifierMatch],
    ties: Iterable[QualifierTie],
) -> frozenset[str]:
    """The query phrases a shape's family holds a value for (`holds`, the shape's
    `held_codes`), each as its normalised text (`QualifierMatch.phrase`).

    `matches` and `ties` are the query as the shape's own vocabulary reads it
    (`qualifier_match.read_by_vocabulary`), measure values included. A tie's phrase is
    read when the shape holds any of its codes: "tertiary education" ties two CRS purposes,
    and a table holding one of them reads the words. A phrase reads the same in every
    vocabulary that reads it, so two tables of different code systems compare by the words
    their codes cover. "ODA for basic education" reads {"oda", "basic education"} on CRS
    and {"oda"} on DAC2a. A measure word reads only on the five-tuple it names. "Grant
    equivalent ODA" reads {"grant equivalent", "oda"} on CRS grant equivalents and {"oda"}
    on DAC1 disbursements.
    """
    readings: list[tuple[str, tuple[QualifierMatch, ...]]] = [
        (match.phrase, (match,)) for match in matches
    ]
    readings += [(tie.phrase, tie.rows) for tie in ties]
    return frozenset(phrase for phrase, rows in readings if any(holds(row) for row in rows))


def phrases_within(phrases: AbstractSet[str], reader: AbstractSet[str]) -> bool:
    """Whether every phrase of `phrases` lies, as whole words, within some phrase of
    `reader`. Two vocabularies can claim spans of different lengths over one text. DAC1
    reads "multilateral ODA" as one code where CRS reads "ODA", so {"oda"} lies within
    {"multilateral oda"}."""
    return all(
        any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", read) for read in reader)
        for phrase in phrases
    )


def reads_fewer(mine: AbstractSet[str], theirs: AbstractSet[str]) -> bool:
    """Whether `mine` lies within `theirs` (`phrases_within`) and `theirs` does not lie
    within `mine`, so the query words `theirs` reads strictly cover those `mine` reads."""
    return phrases_within(mine, theirs) and not phrases_within(theirs, mine)


def elect_five_tuple_family(
    shapes: Sequence[ShapeDraft],
    phrases_read: Callable[[ShapeDraft], AbstractSet[str]],
    *,
    series_at_places: Callable[[ShapeDraft], bool | None] | None = None,
    stated_words: Callable[[ShapeDraft], AbstractSet[str]] | None = None,
) -> list[ShapeDraft]:
    """Keep the family of a five-tuple and role class that reads the most of the query, and
    return the shapes that remain, in order.

    CRS flows and debt series share a five-tuple, and so do DAC2a, DAC2b, ONE's imputed
    sectors and CRS. Each family is its own shape (`derive_shapes`), and they are rival
    readings of one structural question:

    1. A family reading fewer of the query's phrases than another (`reads_fewer`) leaves.
       `phrases_read` gives the phrases each family's own codes read
       (`query_phrases_read`). "Imputed multilateral ODA to basic education" reads the
       sector only on the sectors table, so DAC2a's imputed multilateral total leaves, and
       "net disbursements" reads a netting basis only the CRS rows hold, so the debt
       series leaves. The rule holds for every pair, indexed or not. `stated_words` gives
       the query words a family's datasets state of every row it would read. The family
       reads such a word as a phrase beside a rival the index carries whose phrases hold
       the word nowhere. "Bilateral ODA from Germany by recipient" reads "ODA" on DAC2a
       and on the CRS, and the CRS, whose every row is bilateral, reads "bilateral" too,
       so DAC2a, which holds core contributions to multilateral organisations, leaves. A
       stated word adds nothing beside a rival that codes it, as DAC1 reads "bilateral
       ODA" as one aid type, and nothing beside a rival the index does not carry, whose
       series name what they hold in words no phrase records: a debt series by bilateral
       creditors compares with the CRS as it does without the word.
    2. Between families reading the same phrases, as rule 1 reads them, one holding no
       series at the places the
       part names leaves when another holds one. `series_at_places` says whether a family
       holds a series there on the roles the places bind, or `None` when it cannot tell:
       the part names no place, or the coverage index holds none of the family's members.
       "Other official flows from Germany to Brazil" reads alike on the CRS and DAC2b, and
       DAC2b, whose bilateral donors report only against aggregate recipients, leaves for
       the CRS. Without `series_at_places` no family leaves on this rule.
    3. Between a family the index carries and one it does not, neither reading fewer, the
       family with the stronger recall score stays. An unscored family loses to a scored
       one.

    Each rule runs among the families the rules before it keep. Reading fewer is a strict
    partial order, with a stated word too, since it counts only beside a rival that reads
    it in no phrase. Rule 2 drops a family only beside one that holds a series and stays, and
    recall score is a total order, so each leaves a family standing, and families no rule
    separates all stay, for bind to choose among.
    """
    index = get_index()

    def rivals(shape: ShapeDraft, other: ShapeDraft) -> bool:
        return (
            other is not shape
            and shape_five_tuple(shape) == shape_five_tuple(other)
            and is_role_bearing(shape) == is_role_bearing(other)
        )

    def score(shape: ShapeDraft) -> float:
        return -1.0 if shape.representative_score is None else shape.representative_score

    def read(shape: ShapeDraft, rival: ShapeDraft) -> AbstractSet[str]:
        """The phrases `shape` reads beside `rival`: those its codes read, and, beside a
        rival the index carries, each word its datasets state that lies in no phrase
        `rival` reads."""
        phrases = phrases_read(shape)
        stated = stated_words(shape) if stated_words is not None else ()
        if not stated or not index_carries_shape(rival, index):
            return phrases
        theirs = phrases_read(rival)
        return phrases | {word for word in stated if not phrases_within({word}, theirs)}

    read_most = [
        shape
        for shape in shapes
        if not any(
            rivals(shape, other) and reads_fewer(read(shape, other), read(other, shape))
            for other in shapes
        )
    ]
    served: dict[int, bool | None] = {}

    def holds_series(shape: ShapeDraft) -> bool | None:
        if series_at_places is None:
            return None
        if id(shape) not in served:
            served[id(shape)] = series_at_places(shape)
        return served[id(shape)]

    def unserved(shape: ShapeDraft) -> bool:
        alike = [
            other
            for other in read_most
            if rivals(shape, other)
            and phrases_within(read(shape, other), read(other, shape))
            and phrases_within(read(other, shape), read(shape, other))
        ]
        return (
            bool(alike)
            and holds_series(shape) is False
            and any(holds_series(other) is True for other in alike)
        )

    with_series = [shape for shape in read_most if not unserved(shape)]
    return [
        shape
        for shape in with_series
        if not any(
            rivals(shape, other)
            and index_carries_shape(shape, index) != index_carries_shape(other, index)
            and score(shape) < score(other)
            for other in with_series
        )
    ]


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
    region.phase_rank_bind._named_value_bindings so a stand-in's bindings and the standard
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
