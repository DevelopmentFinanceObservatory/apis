"""Queries and reduces `StatVarRow` members by five-tuple and slot bindings.

A `FamilyQuery` holds what one identity asks of the index: its five-tuple, its
slot bindings, the role keys it binds entities under, the dataset it names and the
datasets it excludes. `identity_family_query` reads it from an identity document and is
the one reader of the named dataset. `index_family` matches all five scalar fields,
filters constraint properties, and keeps the rows `elect_dataset` elects: one role class
(`statvar_members.elect_role_class`, the election graph enumeration runs) of the one
dataset elected among those publishing the five-tuple. `antichain_minimal_rows` then
applies the same antichain reduction used by graph enumeration."""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, cast

from code_index.query import CodeIndex, row_five_tuple
from dfo_platform.roles import declared_role_for
from statvar_members import (
    AbsentReading,
    FiveTuple,
    RoleClass,
    RoleClassFacts,
    SlotBinding,
    antichain_minimal_ids,
    elect_role_class_of,
    five_tuple_from_identity,
    role_class_covers,
    role_class_facts,
    slot_bindings_from_identity,
)

if TYPE_CHECKING:
    from code_index.types import StatVarRow

# The facet axis that names an identity's dataset (`named_dataset`).
_SOURCE_FACET = "provenance"

# The role a row with no `observation_properties` declares.
_SUBJECT_ROLES = ("observationAbout",)

# Per-index memos (`_per_index`) of `elected_pins` by `FamilyQuery` and of
# `option_families`' rows and families by query less the slot and the slot. Each is a
# `_LruMemo` keyed by a 32-byte digest (`family_query_digest`), so a key costs the same
# whatever the query binds. The pins memo holds at most `_MEMO_SIZE` entries. The
# slot-options memo holds at most `_SLOT_OPTIONS_REFS` references in all, counted as they
# are added: each entry's rows and bucket positions, and the rows of each family it elects.
# On the packaged index one menu of `crsPurpose` over the largest five-tuple holds about
# 50,000, so the bound keeps about 20 such menus, near 16 MiB, where the entry bound alone
# kept 4,096 of them, about 0.8 MiB each.
_elected_pins_cache: dict[int, _LruMemo[Mapping[str, str]]] = {}
_slot_options_cache: dict[int, _LruMemo[_SlotOptions]] = {}
_MEMO_SIZE = 4096
_SLOT_OPTIONS_REFS = 1 << 20
_memo_lock = threading.Lock()
# Per-index memo of each five-tuple's role-class facts (`_five_tuple_facts`), at most one
# entry per five-tuple the index carries, so it holds at most one copy of the index's rows.
# A five-tuple it does not carry is never stored.
_five_tuple_facts_cache: dict[
    int, dict[Hashable, tuple[RoleClassFacts, Mapping[str, StatVarRow]]]
] = {}


@dataclass(slots=True)
class _LruMemo[V]:
    """One index's entries of a memo, least recently read first, and their weight.

    Each entry weighs at least 1, and the entries weigh at most `budget` in all."""

    budget: int
    entries: dict[bytes, V] = field(default_factory=dict)
    weights: dict[bytes, int] = field(default_factory=dict)
    weight: int = 0

    def get(self, key: bytes) -> V | None:
        """Returns the entry under `key`, now the most recently read, or `None`."""
        with _memo_lock:
            found = self.entries.pop(key, None)
            if found is not None:
                self.entries[key] = found
            return found

    def put(self, key: bytes, value: V, weight: int) -> None:
        """Keeps `value` under `key`, evicting the least recently read entries until it
        fits. A value heavier than `budget` is not kept."""
        with _memo_lock:
            self._drop(key)
            if weight > self.budget:
                return
            self.weight += weight
            self._evict()
            self.entries[key] = value
            self.weights[key] = weight

    def grow(self, key: bytes, by: int) -> None:
        """Adds `by` to the weight of the entry under `key`, if it is still kept, and evicts
        the least recently read entries until the memo fits."""
        with _memo_lock:
            if key in self.weights:
                self.weights[key] += by
                self.weight += by
                self._evict()

    def _evict(self) -> None:
        while self.entries and self.weight > self.budget:
            self._drop(next(iter(self.entries)))

    def _drop(self, key: bytes) -> None:
        if self.entries.pop(key, None) is not None:
            self.weight -= self.weights.pop(key)


@dataclass(frozen=True)
class _FiveTupleRows:
    rows: Mapping[FiveTuple, tuple[StatVarRow, ...]]
    """Each five-tuple's rows, in index order."""
    carried: Mapping[tuple[str, FiveTuple], frozenset[str]]
    """The constraint properties each dataset's rows of each five-tuple carry."""


# Per-index cache of `_FiveTupleRows`, keyed and finalised as `_index_key_sets_cache` is.
_five_tuple_rows_cache: dict[int, _FiveTupleRows] = {}

# Per-index cache of the distinct constraint-key sets under each `FiveTuple` the
# index carries, keyed by the `CodeIndex` instance's identity. A weakref
# finalizer removes each entry before Python can reuse its id.
_index_key_sets_cache: dict[int, Mapping[FiveTuple, frozenset[frozenset[str]]]] = {}


class SourceNotInFamily(Exception):
    """The bound source names a dataset that does not publish the query's five-tuple.

    `candidates` holds the provenances of the datasets that do, in serving order."""

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
    excluded: frozenset[str] = frozenset()
    """The provenances of the datasets the query excludes. An unnamed query elects among
    the others."""

    @property
    def bound(self) -> frozenset[str]:
        """The constraint properties the query binds to a value or a set."""
        return frozenset(prop for prop, binding in self.slots.items() if binding != ("unbound",))


def named_dataset(facets: Mapping[str, Mapping[str, object]]) -> str | None:
    """Returns the provenance of the dataset an identity's `facets` name, or `None`.

    The provenance facet names a dataset when it binds one distinct ref, as a `value` or
    as a `set` of one. The spec kit hashes the two spellings to one `spec_id`, so both
    name the same dataset. A set of two or more distinct refs names no dataset: the
    family is then the one the unnamed election serves (`eligible_datasets`), and the
    facet restricts the observations served, as it does for any identity."""
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
    """Returns the dataset that serves `query`'s family and its rows of the elected role
    class.

    `rows` already match `query`. A query that names a dataset (`FamilyQuery.source`)
    elects it. Otherwise the candidates are the datasets publishing the five-tuple
    (`CodeIndex.datasets_by_five_tuple`) that are not `opt_in`, or its `opt_in` publishers
    when it has no other, less those in `FamilyQuery.excluded`. `opt_in` decides which
    shapes a question may answer from, so an identity on a five-tuple only `opt_in`
    datasets publish still elects one. Each candidate's role class is elected among the rows it
    publishes (`elect_role_class`), reading the query's role keys. A candidate none of
    whose rows covers the role keys drops out when another candidate's rows do, since
    `elect_role_class` keeps every row when none covers. Of the rest, a candidate whose
    family declares exactly the roles the keys place serves before one whose family
    declares more, and then the lowest rank serves. No candidate with a published row
    gives a family with no provenance and no rows, as does a named dataset that publishes
    none of `rows`.

    Raises `SourceNotInFamily` when the named dataset does not publish the five-tuple and
    another dataset does (`_refuse_non_publisher`)."""
    _refuse_non_publisher(index, query)
    if not rows:
        return DatasetFamily(provenance=None, rows=())
    facts = _rows_facts(index, query.five_tuple, rows)
    published = {
        dataset: dataset_rows
        for dataset in eligible_datasets(index, query)
        if (dataset_rows := tuple(row for row in rows if dataset in row.provenances))
    }
    covering = {
        dataset
        for dataset, dataset_rows in published.items()
        if any(
            role_class_covers(role_class.roles, query.role_keys)
            for role_class in _classes(facts, dataset_rows)
        )
    }
    families = [
        (dataset, _elect_rows(index, facts, dataset_rows, query.role_keys))
        for dataset, dataset_rows in published.items()
        if not covering or dataset in covering
    ]
    # `sorted` is stable, so candidates whose roles match alike keep their rank order.
    for dataset, family in sorted(
        families, key=lambda elected: not _places_exactly(facts, elected[1], query.role_keys)
    ):
        if family:
            return DatasetFamily(provenance=dataset, rows=family)
    return DatasetFamily(provenance=None, rows=())


def _refuse_non_publisher(index: CodeIndex, query: FamilyQuery) -> None:
    """Raises `SourceNotInFamily` when `query` names a dataset that does not publish its
    five-tuple and another dataset does. The test reads no row."""
    publishers = index.datasets_by_five_tuple.get(query.five_tuple, ())
    if query.source is not None and publishers and query.source not in publishers:
        raise SourceNotInFamily(source=query.source, candidates=publishers)


def bind_slot_value(index: CodeIndex, query: FamilyQuery, prop: str, value: str) -> FamilyQuery:
    """Returns `query` with `prop` bound to `value` widened to its descendants, as an identity
    that binds the value states it (`identity_family_query`)."""
    refs = index.descendants(value)
    binding: SlotBinding = ("set", refs) if len(refs) > 1 else ("value", value)
    return replace(query, slots={**query.slots, prop: binding})


def _five_tuple_rows(index: CodeIndex) -> _FiveTupleRows:
    key = id(index)
    cached = _five_tuple_rows_cache.get(key)
    if cached is None:
        rows: dict[FiveTuple, list[StatVarRow]] = defaultdict(list)
        carried: dict[tuple[str, FiveTuple], set[str]] = defaultdict(set)
        for row in index.statvars:
            five_tuple = row_five_tuple(row)
            rows[five_tuple].append(row)
            for dataset in row.provenances:
                carried[dataset, five_tuple].update(row.constraint_properties)
        cached = _FiveTupleRows(
            rows={five_tuple: tuple(found) for five_tuple, found in rows.items()},
            carried={pair: frozenset(props) for pair, props in carried.items()},
        )
        _five_tuple_rows_cache[key] = cached
        weakref.finalize(index, _five_tuple_rows_cache.pop, key, None)
    return cached


def _allowed_values(slots: Mapping[str, SlotBinding]) -> dict[str, frozenset[str]]:
    """The values each value or set binding of `slots` allows, by property. A scan builds
    them once, so testing a row against a set of hundreds of refs is one hash lookup."""
    allowed: dict[str, frozenset[str]] = {}
    for prop, binding in slots.items():
        match binding:
            case ("value", ref):
                allowed[prop] = frozenset((ref,))
            case ("set", refs):
                allowed[prop] = frozenset(refs)
    return allowed


def _satisfies(row: StatVarRow, allowed: Mapping[str, AbstractSet[str]]) -> bool:
    """Whether `row` holds a value `allowed` allows for each of its properties, as
    `CodeIndex.filter` reads `Bound` and `BoundSet` bindings."""
    values = row.constraint_properties
    # A loop: `all` over a generator runs about 1.7 times slower per row here.
    for prop, refs in allowed.items():  # noqa: SIM110
        if values.get(prop) not in refs:
            return False
    return True


def _slot_open_rows(index: CodeIndex, query: FamilyQuery, prop: str) -> tuple[StatVarRow, ...]:
    """The rows of `query`'s five-tuple that satisfy every binding but `prop`'s and declare
    only `query`'s slot keys and `prop`, in index order: the rows `family_for_five_tuple`
    reads for `query` with any value bound to `prop`."""
    others = _allowed_values({other: b for other, b in query.slots.items() if other != prop})
    keys = query.slots.keys() | {prop}
    class_of = _five_tuple_facts(index, query.five_tuple)[0].class_of
    return tuple(
        row
        for row in _five_tuple_rows(index).rows.get(query.five_tuple, ())
        if class_of[row.dcid].keys <= keys and (not others or _satisfies(row, others))
    )


def offers_slot(index: CodeIndex, query: FamilyQuery, prop: str) -> bool:
    """Returns whether a dataset that can serve `query` publishes a row of its five-tuple
    that carries `prop`, so that a menu of `prop`'s values offers a value some election
    serves.

    The datasets that can serve are those of `eligible_datasets` (the named one alone for a
    query that names one) that publish a row matching every other binding of `query`, and of
    those the ones with such a row covering `query.role_keys` when any has one, as
    `elect_dataset` reads coverage. When none publishes a matching row, every eligible
    dataset can. On a five-tuple one dataset publishes, this holds exactly when that
    dataset's rows carry `prop`, and no row is read. A five-tuple the index does not carry
    offers every slot (`datasets_carry`)."""
    serving = eligible_datasets(index, query)
    if len(serving) > 1:
        rows = _slot_open_rows(index, query, prop)
        matching = [d for d in serving if any(d in row.provenances for row in rows)]
        covering = [
            d
            for d in matching
            if any(
                d in row.provenances and role_class_covers(_row_roles(row), query.role_keys)
                for row in rows
            )
        ]
        serving = tuple(covering or matching or serving)
    return datasets_carry(index, serving, query.five_tuple, prop)


def datasets_carry(
    index: CodeIndex, datasets: Iterable[str], five_tuple: FiveTuple, prop: str
) -> bool:
    """Returns whether a row of `five_tuple` that one of `datasets` publishes carries `prop`.
    A five-tuple the index does not carry carries every property, since the index states
    nothing about its rows."""
    if five_tuple not in index.datasets_by_five_tuple:
        return True
    return prop in carried_properties(index, datasets, five_tuple)


def carried_properties(
    index: CodeIndex, datasets: Iterable[str], five_tuple: FiveTuple
) -> frozenset[str]:
    """Returns the constraint properties the rows of `five_tuple` that `datasets` publish
    carry, none for a five-tuple the index does not carry."""
    carried = _five_tuple_rows(index).carried
    return frozenset().union(
        *(carried.get((dataset, five_tuple), frozenset()) for dataset in datasets)
    )


def option_families(
    index: CodeIndex, query: FamilyQuery, prop: str, values: Iterable[str]
) -> dict[str, DatasetFamily] | None:
    """Returns the family `family_for_five_tuple` elects for `query` with `prop` bound to
    each of `values` (`bind_slot_value`), or `None` when the index does not decide those
    queries (`index_covers_slot_keys`).

    The rows are read once per query and `prop` and bucketed by their value of `prop`; each
    value elects over the buckets of its descendants. Both the buckets and each value's
    family are memoised per index, query less `prop`'s binding, and `prop`, since every
    option menu of one answer reads them.

    Raises `SourceNotInFamily` as `elect_dataset` does."""
    if not index_covers_slot_keys(index, query.five_tuple, query.slots.keys() | {prop}):
        return None
    memo = _per_index(_slot_options_cache, index, lambda: _LruMemo(budget=_SLOT_OPTIONS_REFS))
    key, options = _slot_options(memo, index, query, prop)
    families: dict[str, DatasetFamily] = {}
    for value in values:
        family = options.families.get(value)
        if family is None:
            refs = dict.fromkeys(index.descendants(value))
            positions = sorted(p for ref in refs for p in options.buckets.get(ref, ()))
            family = options.families[value] = elect_dataset(
                index,
                [options.rows[position] for position in positions],
                bind_slot_value(index, query, prop, value),
            )
            memo.grow(key, 1 + len(family.rows))
        families[value] = family
    return families


@dataclass(frozen=True)
class _SlotOptions:
    rows: tuple[StatVarRow, ...]
    """The rows any value of the slot reads (`_slot_open_rows`)."""
    buckets: Mapping[str, tuple[int, ...]]
    """The positions in `rows` of the rows holding each value of the slot."""
    families: dict[str, DatasetFamily]
    """The family elected for each value read so far."""


def _slot_options(
    memo: _LruMemo[_SlotOptions], index: CodeIndex, query: FamilyQuery, prop: str
) -> tuple[bytes, _SlotOptions]:
    """The `_SlotOptions` of `query` and `prop` in `memo`, beside their key there. A new
    entry weighs its rows and bucket positions; `option_families` adds its families."""
    others = replace(query, slots={key: bound for key, bound in query.slots.items() if key != prop})
    key = hashlib.sha256(family_query_digest(others) + prop.encode()).digest()
    options = memo.get(key)
    if options is None:
        rows = _slot_open_rows(index, query, prop)
        buckets: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            if (value := row.constraint_properties.get(prop)) is not None:
                buckets[value].append(position)
        options = _SlotOptions(
            rows=rows,
            buckets={value: tuple(found) for value, found in buckets.items()},
            families={},
        )
        positions = sum(len(found) for found in buckets.values())
        memo.put(key, options, 1 + len(rows) + positions)
    return key, options


def eligible_datasets(index: CodeIndex, query: FamilyQuery) -> tuple[str, ...]:
    """Returns the datasets `elect_dataset` elects among, in rank order: the named one,
    else the non-`opt_in` publishers of the five-tuple, or its `opt_in` ones when it has no
    other, less the excluded. Search's shape candidacy and its pre-bind readers read a
    part's datasets through this function, so they and every door's election agree."""
    if query.source is not None:
        return (query.source,)
    publishers = index.datasets_by_five_tuple.get(query.five_tuple, ())
    default = tuple(dataset for dataset in publishers if not is_opt_in(index, dataset))
    return tuple(dataset for dataset in default or publishers if dataset not in query.excluded)


def is_opt_in(index: CodeIndex, dataset: str) -> bool:
    """Returns whether `dataset` serves only a query that names it. An undeclared dataset does not."""
    declaration = index.datasets.get(dataset)
    return declaration is not None and declaration.opt_in


def _places_exactly(
    facts: RoleClassFacts, rows: Sequence[StatVarRow], role_keys: frozenset[str]
) -> bool:
    """Whether every row of `rows` declares exactly the roles `role_keys` place: a donor-only
    row for a donor key, where a donor-and-recipient row declares one role more."""
    for role_class in _classes(facts, rows):
        roles = role_class.roles
        placed = (
            {declared_role_for(key, roles) for key in role_keys}
            if role_keys
            else set(_SUBJECT_ROLES)
        )
        if placed != set(roles):
            return False
    return True


def _row_roles(row: StatVarRow) -> tuple[str, ...]:
    """Returns the roles `row` declares, `observationAbout` when it names none."""
    return row.observation_properties or _SUBJECT_ROLES


def _elect_rows(
    index: CodeIndex, facts: RoleClassFacts, rows: Sequence[StatVarRow], role_keys: frozenset[str]
) -> tuple[StatVarRow, ...]:
    """Runs `elect_role_class` over `rows`, whose facts `facts` holds, under the index's
    declared absent readings, and returns the rows of its family in the order given."""
    election = elect_role_class_of(
        facts, [row.dcid for row in rows], role_keys=role_keys, readings=absent_readings(index)
    )
    if len(election.family) == len(rows):
        return tuple(rows)
    family = frozenset(election.family)
    return tuple(row for row in rows if row.dcid in family)


def _classes(facts: RoleClassFacts, rows: Iterable[StatVarRow]) -> set[RoleClass]:
    """The role classes of `rows`."""
    class_of = facts.class_of
    return {class_of[row.dcid] for row in rows}


def _rows_facts(
    index: CodeIndex, five_tuple: FiveTuple, rows: Sequence[StatVarRow]
) -> RoleClassFacts:
    """The role-class facts of `rows`: those of `five_tuple`'s rows in `index`
    (`_five_tuple_facts`) when `rows` are its rows, else built for `rows`."""
    facts, own = _five_tuple_facts(index, five_tuple)
    if all(own.get(row.dcid) is row for row in rows):
        return facts
    return role_class_facts(
        {row.dcid: row.constraint_properties for row in rows},
        {row.dcid: _row_roles(row) for row in rows},
    )


def _five_tuple_facts(
    index: CodeIndex, five_tuple: FiveTuple
) -> tuple[RoleClassFacts, Mapping[str, StatVarRow]]:
    """The role-class facts of `five_tuple`'s rows in `index`, built once per index and
    five-tuple the index carries, beside those rows by dcid. A five-tuple the index does
    not carry reads empty facts, built each time and never stored, so a client posting
    arbitrary five-tuples cannot grow the memo."""
    rows = _five_tuple_rows(index).rows.get(five_tuple)
    if rows is None:
        return role_class_facts({}, {}), {}
    by_five_tuple = _per_index(_five_tuple_facts_cache, index, dict)
    cached = by_five_tuple.get(five_tuple)
    if cached is None:
        facts = role_class_facts(
            {row.dcid: row.constraint_properties for row in rows},
            {row.dcid: _row_roles(row) for row in rows},
        )
        cached = by_five_tuple[five_tuple] = (facts, {row.dcid: row for row in rows})
    return cached


def family_for_five_tuple(index: CodeIndex, query: FamilyQuery) -> DatasetFamily:
    """Returns the dataset elected for `query`, with its rows before reduction.

    Rows match all five fields, satisfy every binding, and declare only keys in
    `query.slots`. `elect_dataset` then keeps the rows of the elected dataset. Rows
    keep index order, as `CodeIndex.filter` does. No match returns a family with no
    provenance and no rows, including when the index carries no such five-tuple.

    Raises `SourceNotInFamily` as `elect_dataset` does."""
    allowed = _allowed_values(query.slots)
    matched = tuple(
        row
        for row in _five_tuple_rows(index).rows.get(query.five_tuple, ())
        if _satisfies(row, allowed) and row.constraint_properties.keys() <= query.slots.keys()
    )
    return elect_dataset(index, matched, query)


def index_family(index: CodeIndex, query: FamilyQuery) -> DatasetFamily | None:
    """Returns the family the index resolves for `query`, or `None` when the index does
    not decide its membership.

    The index decides membership when `index_covers_slot_keys` holds for the query's slot
    keys; the family is then `family_for_five_tuple`'s. The Datasets mint and Search both
    resolve a family through this function.

    Raises `SourceNotInFamily` as `elect_dataset` does, whether or not the index decides
    membership, so a named dataset that does not publish the five-tuple refuses whatever
    slots the identity declares."""
    _refuse_non_publisher(index, query)
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


def elected_pins(index: CodeIndex, query: FamilyQuery) -> dict[str, str]:
    """Returns the pins of the dataset serving `query`'s family on its five-tuple
    (`CodeIndex.dataset_pins`), the one `family_for_five_tuple` elects.

    When no dataset elects a family, as for slot keys no index row is covered by, the pins
    are those every eligible dataset holds with one value (`eligible_datasets`). A named
    dataset that does not publish the five-tuple pins nothing; electing its family refuses.

    A pin is implied by the election: every row of the elected dataset holds it, so no
    sentence or family label states it. A pin observed today may stop holding in a later
    export, so what decides whether a slot enters the identity reads the declared invariants
    instead (`elected_invariants`). The result is memoised per index and query, since every
    door and several steps of one answer read it."""
    return dict(_memoised(_elected_pins_cache, index, query, _elected_pins))


def _elected_pins(index: CodeIndex, query: FamilyQuery) -> Mapping[str, str]:
    publishers = index.datasets_by_five_tuple.get(query.five_tuple, ())
    if query.source is not None and query.source not in publishers:
        return {}
    elected = family_for_five_tuple(index, query).provenance
    eligible = (elected,) if elected is not None else eligible_datasets(index, query)
    return shared_pins(index, eligible, query.five_tuple)


def elected_invariants(index: CodeIndex, query: FamilyQuery) -> dict[str, str]:
    """Returns the invariants the dataset `query` names declares
    (`DatasetDeclaration.invariants`), or none when `query` names no dataset, names one that
    does not publish its five-tuple, or declares slot keys that leave membership to the
    graph walk (`index_covers_slot_keys`). The loader's guarantee covers index rows alone,
    so where the graph walk decides membership nothing is implied.

    An invariant holds on every row the dataset publishes in every export, so an identity
    that names the dataset implies it: the canonical identity binds it
    (`invariant_bindings`), no declared default binds it, and Search withholds it from the
    bind menu. An identity that names no dataset may elect another in a later export, so
    for it nothing is implied. The result reads the declaration alone and runs no
    election."""
    source = query.source
    if source is None or source not in index.datasets_by_five_tuple.get(query.five_tuple, ()):
        return {}
    if not index_covers_slot_keys(index, query.five_tuple, query.slots.keys()):
        return {}
    declaration = index.datasets.get(source)
    return dict(declaration.invariants) if declaration is not None else {}


def invariant_bindings(index: CodeIndex, query: FamilyQuery) -> dict[str, str]:
    """Returns the slots `query` leaves unbound that its canonical identity binds, each
    mapped to the value it binds, in property order.

    An invariant of a dataset (`DatasetDeclaration.invariants`) is a value every row the
    dataset publishes carries in every export. An identity that names the dataset
    (`elected_invariants`) reads only that dataset's rows, so binding the invariant changes
    no member, and the bound slot states what the data holds: a reader of the identity's
    slots sees the value without consulting the declaration. So each invariant slot `query`
    declares unbound is bound to the invariant's value. A slot bound to that value stays as
    it is, and one bound to another value or a set stays as posted, so its family is
    narrower or empty. An identity that names no dataset may elect another publisher in a
    later export, whose rows need not hold the value, so it keeps the bindings it was posted
    with.

    The rule reads the declaration alone and runs no election. It is idempotent, and every
    identity naming one dataset that differs only in whether it binds an invariant or
    leaves it unbound has one canonical form, and so one `spec_id`. Every door
    canonicalises an identity through it (`canonical_identity`)."""
    return {
        prop: value
        for prop, value in sorted(elected_invariants(index, query).items())
        if query.slots.get(prop) == ("unbound",)
    }


def canonical_identity(index: CodeIndex, identity: Mapping[str, object]) -> Mapping[str, object]:
    """Returns the canonical form of a described identity document: each slot
    `invariant_bindings` names bound to its value, as Search spells a value binding, and
    each slot bound `exact` to an invariant's own value spelt as that value binding. Every
    row of the dataset holds exactly that code, so the two spellings select the same rows;
    the family query cannot tell `exact` from a value with no descendants, so the document
    is read here. Returns `identity` itself when nothing changes, as for a pinned arm
    (`shape.kind` `stat_var`), which binds no slot."""
    shape = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape.get("kind") == "stat_var":
        return identity
    query = identity_family_query(index, identity)
    invariants = elected_invariants(index, query)
    slots = cast("Sequence[Mapping[str, object]]", identity.get("slots", ()))
    bound = invariant_bindings(index, query) | {
        prop: value
        for slot in slots
        if (prop := cast(str, slot["property"])) in invariants
        and slot["binding"] == {"state": "exact", "ref": (value := invariants[prop])}
    }
    if not bound:
        return identity
    return {
        **identity,
        "slots": [
            {**slot, "binding": {"state": "value", "ref": bound[prop]}}
            if (prop := cast(str, slot["property"])) in bound
            else slot
            for slot in slots
        ],
    }


def _memoised[V](
    cache: dict[int, _LruMemo[V]],
    index: CodeIndex,
    query: FamilyQuery,
    compute: Callable[[CodeIndex, FamilyQuery], V],
) -> V:
    """`compute(index, query)`, memoised in `index`'s memo of `cache` (`_per_index`), at
    most `_MEMO_SIZE` entries of weight 1."""
    memo = _per_index(cache, index, lambda: _LruMemo[V](budget=_MEMO_SIZE))
    key = family_query_digest(query)
    found = memo.get(key)
    if found is None:
        found = compute(index, query)
        memo.put(key, found, 1)
    return found


def _per_index[M](cache: dict[int, M], index: CodeIndex, new: Callable[[], M]) -> M:
    """Returns `index`'s memo in `cache`, made by `new` and keyed and finalised as
    `_index_key_sets_cache` is."""
    memo = cache.get(id(index))
    if memo is None:
        memo = cache[id(index)] = new()
        weakref.finalize(index, cache.pop, id(index), None)
    return memo


def family_query_digest(query: FamilyQuery) -> bytes:
    """Returns the memo key of `query`: the sha256 digest of its canonical JSON.

    A binding set can hold thousands of refs, so the query itself as a key would cost
    memory a memo's budget does not count. The digest is 32 bytes whatever the query holds.
    The JSON spells every field with its name and sorts the slots and role keys, so two
    queries share a key only when they are equal. Two different queries sharing a digest
    would need a sha256 collision, which is out of scope: finding one takes about 2**128
    evaluations."""
    canonical = json.dumps(
        {
            "five_tuple": asdict(query.five_tuple),
            "slots": sorted(query.slots.items()),
            "source": query.source,
            "role_keys": sorted(query.role_keys),
            "excluded": sorted(query.excluded),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).digest()


def shared_pins(index: CodeIndex, datasets: Iterable[str], five_tuple: FiveTuple) -> dict[str, str]:
    """Returns the pins every dataset of `datasets` holds alike on `five_tuple`: each
    property every one's `CodeIndex.dataset_pins` maps to the same value. An empty
    `datasets` gives no pins."""
    return _held_alike([index.dataset_pins.get((dataset, five_tuple), {}) for dataset in datasets])


def _held_alike(values: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """The properties every mapping of `values` maps to one value. None when empty."""
    if not values:
        return {}
    return {
        prop: value
        for prop, value in values[0].items()
        if all(other.get(prop) == value for other in values[1:])
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
    "bind_slot_value",
    "canonical_identity",
    "carried_properties",
    "datasets_carry",
    "deciding_query",
    "elect_dataset",
    "elected_invariants",
    "elected_pins",
    "eligible_datasets",
    "family_for_five_tuple",
    "family_query_digest",
    "identity_family_query",
    "index_covers_slot_keys",
    "index_family",
    "invariant_bindings",
    "is_opt_in",
    "key_sets_cover",
    "named_dataset",
    "offers_slot",
    "option_families",
    "shared_pins",
]
