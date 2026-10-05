"""Queries and reduces `StatVarRow` members by five-tuple and slot bindings.

A `FamilyQuery` holds what one identity asks of the index: its five-tuple, its
slot bindings, the role keys it binds entities under and the dataset it names.
`identity_family_query` reads it from an identity document and is the one reader of
the named dataset. `index_family` matches all five scalar fields, filters constraint
properties, and keeps the rows `elect_dataset` elects: one role class
(`statvar_members.elect_role_class`, the election graph enumeration runs) of the one
dataset the five-tuple belongs to. `antichain_minimal_rows` then applies the same
antichain reduction used by graph enumeration."""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from code_index.query import CodeIndex, row_five_tuple
from code_index.types import Bound, BoundSet, Open, PropertyBinding
from statvar_members import (
    AbsentReading,
    FiveTuple,
    RoleClassElection,
    SlotBinding,
    antichain_minimal_ids,
    elect_role_class,
    five_tuple_from_identity,
    slot_bindings_from_identity,
)

if TYPE_CHECKING:
    from code_index.types import StatVarRow

# The facet axis that names an identity's dataset (`named_dataset`).
_SOURCE_FACET = "provenance"

# The role a row with no `observation_properties` declares.
_SUBJECT_ROLES = ("observationAbout",)

# Per-index cache of the distinct constraint-key sets under each `FiveTuple` the
# index carries, keyed by the `CodeIndex` instance's identity. A weakref
# finalizer removes each entry before Python can reuse its id.
_index_key_sets_cache: dict[int, Mapping[FiveTuple, frozenset[frozenset[str]]]] = {}


class SourceNotInFamily(Exception):
    """The bound source names a dataset that publishes none of the surviving rows.

    `candidates` holds the provenance of the dataset that does."""

    def __init__(self, *, source: str, candidates: tuple[str, ...]) -> None:
        self.source = source
        self.candidates = candidates
        super().__init__(f"no matching row is published by {source!r}")


@dataclass(frozen=True)
class FamilyQuery:
    """What one identity asks of the index: the inputs every election reads."""

    five_tuple: FiveTuple
    slots: Mapping[str, SlotBinding]
    """The identity's slot bindings by constraint property, descendant-widened."""
    source: str | None
    """The provenance of the dataset the identity names, or `None` when it names none."""
    role_keys: frozenset[str] = frozenset()
    """The role keys the identity binds entities under: `observationAbout`, `origin`,
    `destination`, or a declared role name. The entities themselves stay out."""

    @property
    def bound(self) -> frozenset[str]:
        """The constraint properties the query binds to a value or a set."""
        return frozenset(prop for prop, binding in self.slots.items() if binding != ("unbound",))


def named_dataset(facets: Mapping[str, Mapping[str, object]]) -> str | None:
    """Returns the provenance of the dataset an identity's `facets` name, or `None`.

    The provenance facet names a dataset when it binds one distinct ref, as a `value` or
    as a `set` of one. The spec kit hashes the two spellings to one `spec_id`, so both
    name the same dataset. A set of two or more distinct refs names no dataset: the
    family is then its five-tuple's dataset's, and the facet restricts the observations
    served, as it does for any identity."""
    binding = facets.get(_SOURCE_FACET, {})
    ref = binding.get("ref")
    refs = {ref} if isinstance(ref, str) else set(cast("Sequence[str]", binding.get("refs", ())))
    return next(iter(refs)) if len(refs) == 1 else None


def identity_family_query(index: CodeIndex, identity: Mapping[str, object]) -> FamilyQuery:
    """Returns the `FamilyQuery` of a described identity document.

    The five-tuple comes from `shape`, the bindings from `slots` widened to their
    descendants in `index`, the role keys from the keys of `entities`, and the dataset from
    `facets` (`named_dataset`). The Datasets mint and Search both read an identity through
    this function, so they elect over the same inputs."""
    return FamilyQuery(
        five_tuple=five_tuple_from_identity(cast("Mapping[str, object]", identity["shape"])),
        slots=slot_bindings_from_identity(
            cast("Sequence[Mapping[str, object]]", identity.get("slots", ())),
            widen=index.descendants,
        ),
        source=named_dataset(
            cast("Mapping[str, Mapping[str, object]]", identity.get("facets", {}))
        ),
        role_keys=frozenset(cast("Mapping[str, object]", identity.get("entities", {}))),
    )


@dataclass(frozen=True)
class DatasetFamily:
    """The dataset elected for a query and the rows that form its family."""

    provenance: str | None
    """The elected dataset's provenance, or `None` when no row survives."""
    rows: tuple[StatVarRow, ...]
    """The elected dataset's rows of the elected role class before antichain reduction, in
    the order given. They declare one role set unless the election tied
    (`statvar_members.RoleClassElection.tied`), which the caller refuses."""


def _index_key_sets(index: CodeIndex) -> Mapping[FiveTuple, frozenset[frozenset[str]]]:
    key = id(index)
    cached = _index_key_sets_cache.get(key)
    if cached is None:
        key_sets: dict[FiveTuple, set[frozenset[str]]] = defaultdict(set)
        for row in index.statvars:
            key_sets[row_five_tuple(row)].add(frozenset(row.constraint_properties))
        cached = {five_tuple: frozenset(keys) for five_tuple, keys in key_sets.items()}
        _index_key_sets_cache[key] = cached
        weakref.finalize(index, _index_key_sets_cache.pop, key, None)
    return cached


def index_covers_slot_keys(
    index: CodeIndex, five_tuple: FiveTuple, slot_keys: AbstractSet[str]
) -> bool:
    """Returns whether `slot_keys` holds every constraint property of at least one index
    row of `five_tuple`.

    This is the key test `family_for_five_tuple` applies to each row, asked before any
    binding. An identity that passes resolves from the index, and one whose values match no
    row resolves to nothing there. An identity that fails names a population the index rows
    of its five-tuple do not describe, and the graph walk resolves it. Exported rows carry
    resolved denominators, so an unresolved-denominator five-tuple is never covered."""
    return key_sets_cover(_index_key_sets(index), five_tuple, slot_keys)


def key_sets_cover[K](
    key_sets: Mapping[K, Iterable[AbstractSet[str]]], five_tuple: K, slot_keys: AbstractSet[str]
) -> bool:
    """Returns whether `slot_keys` holds every key of at least one set under `five_tuple`.

    `key_sets` maps each five-tuple to the constraint-property sets of its rows. This is the
    predicate behind `index_covers_slot_keys`. The code-index export applies it to rows it has
    not yet built into a `CodeIndex`."""
    return any(keys <= slot_keys for keys in key_sets.get(five_tuple, ()))


def _constraint_query(slots: Mapping[str, SlotBinding]) -> dict[str, PropertyBinding]:
    """Translates `value`, `set`, and `unbound` slots to `Bound`, `BoundSet`, and `Open`.

    Callers must descendant-widen slots first."""
    query: dict[str, PropertyBinding] = {}
    for prop, binding in slots.items():
        match binding:
            case ("value", ref):
                query[prop] = Bound(ref)
            case ("set", refs):
                query[prop] = BoundSet(frozenset(refs))
            case ("unbound",):
                query[prop] = Open()
    return query


def _matches_five_tuple(row: StatVarRow, five_tuple: FiveTuple) -> bool:
    """Compares all five scalar fields. `CodeIndex.filter` checks constraints only.

    An unresolved-denominator request returns `False` because exported rows are resolved."""
    if five_tuple.denominator_unresolved:
        return False
    return (
        row.population_type == five_tuple.population_type
        and row.measured_property == five_tuple.measured_property
        and row.stat_type == five_tuple.stat_type
        and row.measurement_qualifier == five_tuple.measurement_qualifier
        and row.measurement_denominator == five_tuple.measurement_denominator
    )


def absent_readings(index: CodeIndex) -> Mapping[str, AbsentReading]:
    """Returns each code system's declared absent-reading rule."""
    return {
        code_system: declaration.absent_reading
        for code_system, declaration in index.declarations.items()
    }


def elect_dataset(
    index: CodeIndex, rows: Sequence[StatVarRow], query: FamilyQuery
) -> DatasetFamily:
    """Returns the rows of the dataset `query`'s five-tuple belongs to that form the family.

    `rows` already match `query`. The dataset is the one the index maps the query's
    five-tuple to (`CodeIndex.dataset_by_five_tuple`), and `rows` it does not publish are
    dropped. The role class is elected among the rest (`elect_role_class`), reading the
    role keys the query binds entities under (`FamilyQuery.role_keys`). A five-tuple no
    dataset publishes, and rows of which the dataset publishes none, give a family with no
    provenance and no rows.

    Raises `SourceNotInFamily` when the query names another dataset
    (`FamilyQuery.source`)."""
    dataset = index.dataset_by_five_tuple.get(query.five_tuple)
    published = tuple(row for row in rows if dataset in row.provenances)
    if dataset is None or not published:
        return DatasetFamily(provenance=None, rows=())
    if query.source is not None and query.source != dataset:
        raise SourceNotInFamily(source=query.source, candidates=(dataset,))
    _election, family = _elect_rows(index, published, query.role_keys)
    return DatasetFamily(provenance=dataset, rows=family)


def _row_roles(row: StatVarRow) -> tuple[str, ...]:
    """Returns the roles `row` declares, `observationAbout` when it names none."""
    return row.observation_properties or _SUBJECT_ROLES


def _elect_rows(
    index: CodeIndex, rows: Sequence[StatVarRow], role_keys: frozenset[str]
) -> tuple[RoleClassElection, tuple[StatVarRow, ...]]:
    """Runs `elect_role_class` over `rows` under the index's declared absent readings.

    Returns the election beside the rows of its family, in the order given."""
    election = elect_role_class(
        {row.dcid: row.constraint_properties for row in rows},
        {row.dcid: _row_roles(row) for row in rows},
        [row.dcid for row in rows],
        role_keys=role_keys,
        readings=absent_readings(index),
    )
    family = frozenset(election.family)
    return election, tuple(row for row in rows if row.dcid in family)


def family_for_five_tuple(index: CodeIndex, query: FamilyQuery) -> DatasetFamily:
    """Returns the dataset elected for `query`, with its rows before reduction.

    Rows match all five fields, satisfy every binding, and declare only keys in
    `query.slots`. `elect_dataset` then keeps the rows of the five-tuple's dataset. Rows
    keep `CodeIndex.filter` order. No match returns a family with no provenance and no
    rows, including when the index carries no such five-tuple.

    Raises `SourceNotInFamily` as `elect_dataset` does."""
    matched = tuple(
        row
        for row in index.filter(_constraint_query(query.slots))
        if _matches_five_tuple(row, query.five_tuple)
        and set(row.constraint_properties) <= query.slots.keys()
    )
    return elect_dataset(index, matched, query)


def index_family(index: CodeIndex, query: FamilyQuery) -> DatasetFamily | None:
    """Returns the family the index resolves for `query`, or `None` when the index does
    not decide its membership.

    The index decides membership when `index_covers_slot_keys` holds for the query's slot
    keys; the family is then `family_for_five_tuple`'s. The Datasets mint and Search both
    resolve a family through this function.

    Raises `SourceNotInFamily` as `elect_dataset` does."""
    if not index_covers_slot_keys(index, query.five_tuple, query.slots.keys()):
        return None
    return family_for_five_tuple(index, query)


def deciding_query(index: CodeIndex, query: FamilyQuery) -> FamilyQuery:
    """Returns `query` without the unbound slots no index row of its five-tuple declares.

    `index_family` resolves both queries to the same family. An unbound slot matches every
    row (`Open`), and the key tests (`index_covers_slot_keys` and the row filter of
    `family_for_five_tuple`) compare slot keys against the keys rows declare, so a key no
    row of the five-tuple declares changes neither. A bound slot stays, as it requires the
    rows to declare its value."""
    declared = frozenset().union(*_index_key_sets(index).get(query.five_tuple, ()))
    slots = {
        prop: binding
        for prop, binding in query.slots.items()
        if binding != ("unbound",) or prop in declared
    }
    return query if len(slots) == len(query.slots) else replace(query, slots=slots)


def unbound_pins(index: CodeIndex, query: FamilyQuery) -> dict[str, str]:
    """Returns each slot `query` declares unbound that its five-tuple pins
    (`CodeIndex.five_tuple_pins`), mapped to the pinned value, in property order.

    An identity with such a slot is not canonical. Search binds every pin on the
    identities it mints, so one family has one spec_id. A door that serves a posted
    identity as is refuses one with a slot here, and the Search resubmit binds the pins
    and answers with the pinned identity. A slot the query omits is not listed, because
    an identity that omits a pinned slot covers no index row of the five-tuple."""
    pins = index.five_tuple_pins.get(query.five_tuple, {})
    return {
        prop: value for prop, value in sorted(pins.items()) if query.slots.get(prop) == ("unbound",)
    }


def absent_member_reads_zero(index: CodeIndex, dataset: str | None) -> bool:
    """Returns whether a family member with no observation adds zero to a sum over the
    family of `dataset`.

    `dataset` is the provenance of the elected dataset, and the reading is its
    `DatasetDeclaration.absent_observation`. No dataset, and a provenance no dataset
    declares, read unknown. The mint's sum and Search's `Slot.summable` both read absence
    through this function."""
    return dataset is not None and index.absent_observation(dataset) == "zero"


def antichain_minimal_rows(index: CodeIndex, rows: Sequence[StatVarRow]) -> tuple[StatVarRow, ...]:
    """Returns the antichain-minimal subset of `rows`, in their order.

    Dominance compares each row's constraint properties under the index's declared
    absent readings."""
    order = [row.dcid for row in rows]
    keys_by_dcid = {row.dcid: frozenset(row.constraint_properties) for row in rows}
    kept_dcids = antichain_minimal_ids(keys_by_dcid, order, readings=absent_readings(index))
    rows_by_dcid = {row.dcid: row for row in rows}
    return tuple(rows_by_dcid[dcid] for dcid in kept_dcids)


__all__ = [
    "DatasetFamily",
    "FamilyQuery",
    "SourceNotInFamily",
    "absent_member_reads_zero",
    "absent_readings",
    "antichain_minimal_rows",
    "deciding_query",
    "elect_dataset",
    "family_for_five_tuple",
    "identity_family_query",
    "index_covers_slot_keys",
    "index_family",
    "key_sets_cover",
    "named_dataset",
    "unbound_pins",
]
