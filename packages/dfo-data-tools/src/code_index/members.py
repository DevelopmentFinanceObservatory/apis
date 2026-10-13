"""Queries and reduces `StatVarRow` members by five-tuple and slot bindings.

A `FamilyQuery` holds what one identity asks of the index: its five-tuple, its
slot bindings, the role keys it binds entities under, the dataset it names and the
provenances it may answer from. `identity_family_query` reads it from an identity document and is
the one reader of the named dataset. `index_family` matches all five scalar fields,
filters constraint properties, and keeps the rows `elect_dataset` elects: one role class
(`statvar_members.elect_role_class`, the election graph enumeration runs) of the one
dataset elected among those publishing the five-tuple. `antichain_minimal_rows` then
applies the same antichain reduction used by graph enumeration. `listed_group` elects the
graph walk's members of a family holding a listed row (`CodeIndex.listed_statvars`) from the
index alone."""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, cast

from code_index.query import CodeIndex, IndexFamily, row_five_tuple
from dfo_platform.roles import declared_role_for
from statvar_members import (
    AbsentReading,
    FiveTuple,
    MemberGroup,
    RoleClass,
    RoleClassElection,
    RoleClassFacts,
    SlotBinding,
    admitting_roles,
    antichain_minimal_ids,
    elect_role_class_of,
    electing_role_keys,
    five_tuple_from_identity,
    group_of_arcs,
    role_class_covers,
    role_class_facts,
    slot_bindings_from_identity,
)

if TYPE_CHECKING:
    from code_index.types import StatVarRow

# The facet axis that names an identity's dataset (`named_dataset`).
_SOURCE_FACET = "provenance"

# The type every exported row's StatVar carries.
_STATISTICAL_VARIABLE = "StatisticalVariable"

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
# Per-index memo of the roles each dataset declares in each family the index carries
# (`_declared_roles`), at most one entry per family and dataset publishing it.
_declared_roles_cache: dict[int, dict[Hashable, frozenset[tuple[str, ...]]]] = {}


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
    datasets: frozenset[str]
    """The provenances any row records (`_held_datasets`)."""


# Per-index cache of `_FiveTupleRows`, keyed by the `CodeIndex` instance's identity. A
# weakref finalizer removes each entry before Python can reuse its id.
_five_tuple_rows_cache: dict[int, _FiveTupleRows] = {}


class SourceNotInFamily(Exception):
    """The bound source names a dataset that does not publish the query's five-tuple.

    `candidates` holds the provenances of the datasets that do, in serving order."""

    def __init__(self, *, source: str, candidates: tuple[str, ...]) -> None:
        self.source = source
        self.candidates = candidates
        super().__init__(f"no matching row is published by {source!r}")


class ProvenanceScope(Protocol):
    """The provenances a question may answer from (`allowed`), and whether it named any
    (`named`), which lets election serve an opt-in dataset the scope allows."""

    @property
    def allowed(self) -> frozenset[str]: ...

    @property
    def named(self) -> bool: ...


@dataclass(frozen=True)
class DatasetScope:
    """A `ProvenanceScope` a `FamilyQuery` carries."""

    allowed: frozenset[str]
    named: bool


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
    open_role_keys: frozenset[str] = frozenset()
    """The flow axes the identity asks every member of (its `open_roles`). Every election
    reads them softly beside `role_keys` (`statvar_members.electing_role_keys`), and
    `DatasetFamily.role_keys` holds the ones that joined."""
    scope: DatasetScope | None = None
    """The provenances the query may answer from, or `None` for every one. An unnamed query
    elects among its publishers in scope (`eligible_among`)."""

    @property
    def bound(self) -> frozenset[str]:
        """The constraint properties the query binds to a value or a set."""
        return frozenset(prop for prop, binding in self.slots.items() if binding != ("unbound",))


def named_dataset(index: CodeIndex, facets: Mapping[str, Mapping[str, object]]) -> str | None:
    """Returns the provenance of the dataset an identity's `facets` name, or `None`.

    The provenance facet names a dataset when it binds one distinct ref, as a `value` or
    as a `set` of one, that `index` holds rows of (`_held_datasets`). The spec kit hashes
    the two spellings to one `spec_id`, so both name the same dataset. A set of two or
    more distinct refs names no dataset: the family is then the one the unnamed election
    serves among those refs (`identity_family_query`, `eligible_datasets`). A ref only
    listed rows record (`CodeIndex.listed_statvars`), as IDS is, or one the index does not
    hold at all, names no dataset either: the pin scopes the reads alone, and members
    resolve as for an identity that pins nothing."""
    refs = frozenset(pinned_refs(facets))
    return next(iter(refs)) if len(refs) == 1 and refs <= _held_datasets(index) else None


def _held_datasets(index: CodeIndex) -> frozenset[str]:
    """Returns the provenances `index`'s rows (`CodeIndex.statvars`) record: the datasets
    whose families it holds, declared (`CodeIndex.datasets`) or not."""
    return _five_tuple_rows(index).datasets


def identity_family_query(index: CodeIndex, identity: Mapping[str, object]) -> FamilyQuery:
    """Returns the `FamilyQuery` of a described identity document.

    The five-tuple comes from `shape`, the bindings from `slots` widened to their
    expansion in `index` (`CodeIndex.expansion`), the role keys from the keys of `entities`,
    the open role keys from `open_roles`, and the dataset from
    `facets` (`named_dataset`). A provenance facet binding two or more refs scopes the
    election to those of them `index` holds rows of (`_held_datasets`), as a named scope,
    so the mint elects among the datasets a set pin names. A set naming none of them
    scopes nothing. The Datasets mint and Search both read an identity through this function, so they
    elect over the same inputs."""
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets", {}))
    refs = frozenset(pinned_refs(facets))
    datasets = refs & _held_datasets(index)
    return FamilyQuery(
        five_tuple=five_tuple_from_identity(cast("Mapping[str, object]", identity["shape"])),
        slots=slot_bindings_from_identity(
            cast("Sequence[Mapping[str, object]]", identity.get("slots", ())),
            widen=index.expansion,
        ),
        source=named_dataset(index, facets),
        role_keys=frozenset(cast("Mapping[str, object]", identity.get("entities", {}))),
        open_role_keys=frozenset(cast("Sequence[str]", identity.get("open_roles") or ())),
        scope=DatasetScope(allowed=datasets, named=True) if len(refs) > 1 and datasets else None,
    )


def pinned_refs(facets: Mapping[str, Mapping[str, object]]) -> tuple[str, ...]:
    """The distinct refs an identity's provenance facet binds, as a value or a set, in
    posted order. Every reader of a provenance pin (the Datasets mint, Search resubmit and
    the options door) reads it here."""
    binding = facets.get(_SOURCE_FACET, {})
    ref = binding.get("ref")
    if isinstance(ref, str):
        return (ref,)
    return tuple(dict.fromkeys(cast("Sequence[str]", binding.get("refs", ()))))


@dataclass(frozen=True)
class DatasetFamily:
    """The dataset elected for a query and the rows that form its family."""

    provenance: str | None
    """The elected dataset's provenance, or `None` when no row survives."""
    rows: tuple[StatVarRow, ...]
    """The elected dataset's rows of the elected role class before antichain reduction, in
    the order given. They declare one role set unless the election tied (`tied`)."""
    role_keys: frozenset[str] = frozenset()
    """The role keys the election ran under: the query's role keys and those of its open
    ones that joined them (`statvar_members.RoleClassElection.role_keys`). The query's
    role keys when no row survives."""
    tied: bool = False
    """Whether the role-class election tied (`statvar_members.RoleClassElection.tied`),
    which the caller refuses. `rows` then hold every row that places `role_keys`."""


def index_covers_slot_keys(
    index: CodeIndex, five_tuple: FiveTuple, slot_keys: AbstractSet[str]
) -> bool:
    """Returns whether `slot_keys` select a family of `five_tuple` (`CodeIndex.family_of`)
    and hold every constraint property of one of its rows (`IndexFamily.covers`).

    This is the key test `family_for_five_tuple` applies to each row, asked before any
    binding. `index_decides` asks it of a query, on the slot keys that decide its family.
    An identity that passes resolves from the index, and one whose values match no
    row resolves to nothing there. An identity that fails names a population the index rows
    of its five-tuple do not describe, and the graph walk resolves it. Exported rows carry
    resolved denominators, so an unresolved-denominator five-tuple is never covered."""
    family = index.family_of(five_tuple, slot_keys)
    return family is not None and family.covers(slot_keys)


def index_decides(index: CodeIndex, query: FamilyQuery) -> bool:
    """Returns whether the index decides `query`'s membership: `index_covers_slot_keys`
    for the slot keys that decide its family (`_deciding_keys`), whatever dataset it names.
    Every door asks this of an identity or a shape's query, so each resolves from the
    index, or leaves it to the graph walk, for the same queries."""
    return _decides(index, query.five_tuple, query.slots.keys(), query.bound)


def _decides(
    index: CodeIndex, five_tuple: FiveTuple, keys: AbstractSet[str], bound: AbstractSet[str]
) -> bool:
    return index_covers_slot_keys(index, five_tuple, _deciding_keys(index, five_tuple, keys, bound))


def _deciding_keys(
    index: CodeIndex, five_tuple: FiveTuple, keys: AbstractSet[str], bound: AbstractSet[str]
) -> AbstractSet[str]:
    """The slot keys `keys`, of which `bound` are bound, that decide a query's family: the
    bound ones, and the unbound ones some row of `five_tuple` declares.

    An unbound slot matches every row, so one no row of the five-tuple declares names no
    population: it covers no row the other keys leave uncovered and tells no family from
    another. A bound slot requires its rows to declare a value, so it always decides."""
    if len(bound) == len(keys):
        return keys
    families = index.families_by_five_tuple.get(five_tuple, ())
    declared = frozenset().union(*(family.properties for family in families))
    return {key for key in keys if key in bound or key in declared}


def query_family(index: CodeIndex, query: FamilyQuery) -> IndexFamily | None:
    """The family `query` asks of the index. Every election, publisher list and row read of
    an identity resolves its family here.

    A query naming a dataset takes the family of its five-tuple that dataset publishes when
    that family carries every key the query binds and either has a row declaring only the
    query's deciding keys (`_deciding_keys`, `IndexFamily.covers`), or no family of the
    five-tuple has such a row and it carries every deciding key. The family then holds the
    rows the query describes, whichever family the keys alone select, and a slot the query
    leaves unbound that only another dataset's rows carry does not change it. `None`
    otherwise: when the dataset publishes no family of the five-tuple, when the query binds
    a property the family's rows do not carry, and when the keys describe rows of another
    family and none of this one, as CRS slot keys do for a query naming DAC1. `publishers`
    then lists the datasets the keys select and the named one refuses, or, when the keys
    select no family, lists every dataset of the five-tuple and the query goes to the graph
    walk.

    A query naming none asks for the family its deciding slot keys select
    (`CodeIndex.family_of`)."""
    return _family(index, query.five_tuple, query.slots.keys(), query.bound, query.source)


def _family(
    index: CodeIndex,
    five_tuple: FiveTuple,
    keys: AbstractSet[str],
    bound: AbstractSet[str],
    source: str | None,
) -> IndexFamily | None:
    """`query_family` for a query on `five_tuple` naming `source` with slot keys `keys`, of
    which `bound` are bound."""
    deciding = _deciding_keys(index, five_tuple, keys, bound)
    if source is None:
        return index.family_of(five_tuple, deciding)
    published = _dataset_family(index, five_tuple, source)
    if published is None or not bound <= published.properties:
        return None
    if published.covers(deciding):
        return published
    families = index.families_by_five_tuple[five_tuple]
    if deciding <= published.properties and not any(f.covers(deciding) for f in families):
        return published
    return None


def publishers(index: CodeIndex, query: FamilyQuery) -> tuple[str, ...]:
    """The datasets publishing `query`'s family (`query_family`), in serving order. When the
    named dataset publishes no family holding the query's rows, the datasets of the family
    the slot keys select, so a named dataset outside it refuses (`_refuse_non_publisher`).
    Else every dataset publishing the five-tuple, for slot keys that select no family."""
    family = query_family(index, query)
    if family is None and query.source is not None:
        family = query_family(index, replace(query, source=None))
    if family is not None:
        return family.datasets
    return index.datasets_by_five_tuple.get(query.five_tuple, ())


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


def dataset_declares_roles(
    index: CodeIndex, family: IndexFamily, dataset: str | None, role_keys: AbstractSet[str]
) -> bool:
    """Whether `dataset` declares `role_keys` in `family`: a row the rows it publishes
    there read with no role key bound, or a row it publishes there with the constraint map
    of one, its role twin, places every key (`statvar_members.role_class_covers`). With no
    key, a row places them when it declares the subject alone. `None` names the family's
    rows no dataset publishes, which form a group of their own. False when the group holds
    no row of `family`.

    The rows read with no key bound are those the shared election elects among the
    dataset's rows (`statvar_members.admitting_roles`), under the index's absent readings.
    A role only rows outside that reading hold marks those series, not the dataset: DAC1's
    provider totals stand beside two flows to the West Bank and Gaza Strip at other
    constraint values, so DAC1 declares no recipient there, and "aid to Kenya" is not one
    of those flows. A total published beside its by-creditor twin is one series read two
    ways, so a dataset holding such a pair declares the creditor."""
    return any(
        role_class_covers(roles, role_keys) for roles in _declared_roles(index, family, dataset)
    )


def _declared_roles(
    index: CodeIndex, family: IndexFamily, dataset: str | None
) -> frozenset[tuple[str, ...]]:
    """The role tuples `dataset` declares in `family` (`statvar_members.admitting_roles`
    of the rows it publishes there, with no role key), memoised for a family `index`
    carries."""
    carried = index.families_by_five_tuple.get(family.five_tuple, ())
    position = next((at for at, found in enumerate(carried) if found is family), None)
    memo = _per_index(_declared_roles_cache, index, dict)
    key = (family.five_tuple, position, dataset)
    if position is not None and (cached := memo.get(key)) is not None:
        return cached
    rows = tuple(
        row
        for row in family.rows
        if (dataset in row.provenances if dataset is not None else not row.provenances)
    )
    declared = frozenset(
        admitting_roles(
            _rows_facts(index, family.five_tuple, rows),
            [row.dcid for row in rows],
            role_keys=(),
            readings=absent_readings(index),
        )
    )
    if position is not None:
        memo[key] = declared
    return declared


def family_declares_roles(
    index: CodeIndex,
    family: IndexFamily,
    role_keys: AbstractSet[str],
    *,
    datasets: Iterable[str],
) -> bool:
    """Whether some dataset of `datasets` declares `role_keys` in `family`
    (`dataset_declares_roles`). A family no dataset publishes is read from the rows no
    dataset publishes, whatever `datasets` holds, since it has no other row. Beside a
    publisher those rows serve no query (`elect_dataset`) and declare nothing here."""
    groups = datasets if family.datasets else (None,)
    return any(dataset_declares_roles(index, family, dataset, role_keys) for dataset in groups)


def _dataset_family(index: CodeIndex, five_tuple: FiveTuple, dataset: str) -> IndexFamily | None:
    """The family of `five_tuple` that `dataset` publishes, if any: a dataset publishes one
    family of each five-tuple."""
    families = index.families_by_five_tuple.get(five_tuple, ())
    return next((family for family in families if dataset in family.datasets), None)


def _declaring_datasets(
    index: CodeIndex,
    five_tuple: FiveTuple,
    datasets: Iterable[str],
    role_keys: AbstractSet[str],
) -> tuple[str, ...]:
    """The datasets of `datasets` that declare `role_keys` in the family of `five_tuple`
    they publish (`dataset_declares_roles`), in the order given."""
    return tuple(
        dataset
        for dataset in datasets
        if (family := _dataset_family(index, five_tuple, dataset)) is not None
        and dataset_declares_roles(index, family, dataset, role_keys)
    )


def elect_dataset(
    index: CodeIndex,
    rows: Sequence[StatVarRow],
    query: FamilyQuery,
    *,
    among: Sequence[str] | None = None,
) -> DatasetFamily:
    """Returns the dataset that serves `query`'s family and its rows of the elected role
    class.

    `rows` already match `query`. A query that names a dataset (`FamilyQuery.source`)
    elects it. Otherwise the candidates are the datasets publishing the query's family
    (`publishers`) that `eligible_among` keeps under `FamilyQuery.scope`. `opt_in` decides which
    shapes a question may answer from, so an identity on a five-tuple only `opt_in`
    datasets publish still elects one. A caller that holds the candidates passes them in
    rank order as `among` (`family_among`), and the election runs among those alone.

    Each candidate's role class is elected among the rows it publishes
    (`elect_role_class`) under the query's role keys and its open ones, and the election
    returns the keys it ran under: the open keys join where the candidate holds a series by
    them. A candidate whose election ties serves only when every candidate's does, so an
    unnamed question refuses only where no publisher answers it. The family is read under
    the largest set of keys the remaining candidates ran under (`_serving_role_keys`), by
    those whose own election ran under it: a table of provider totals does not serve "aid
    from France to every recipient" beside one that holds each recipient.

    The open keys that joined are the only ones the election may depend on, so while they
    differ from the open keys asked, every candidate is elected again with them alone. A
    question and the same question asked with only the open keys of
    `DatasetFamily.role_keys` therefore elect the same dataset, rows, keys and tie.

    A serving candidate that does not declare the keys in the family it publishes
    (`dataset_declares_roles`) drops out when another does, since `elect_role_class` keeps
    every row when none covers: a dataset only some of whose rows hold a recipient does not
    serve "aid to Kenya" beside one that declares it. Of the rest, a candidate whose family
    declares exactly the roles the keys place serves before one whose family declares
    more, and then the lowest rank serves. `DatasetFamily.role_keys` and
    `DatasetFamily.tied` state the election of the family returned. No candidate with a
    published row gives a family with no provenance and no rows, as does a named dataset
    that publishes none of `rows`.

    Raises `SourceNotInFamily` when the named dataset does not publish the five-tuple and
    another dataset does (`_refuse_non_publisher`)."""
    _refuse_non_publisher(index, query)
    facts = _rows_facts(index, query.five_tuple, rows)
    published = {
        dataset: dataset_rows
        for dataset in (eligible_datasets(index, query) if among is None else among)
        if (dataset_rows := tuple(row for row in rows if dataset in row.provenances))
    }
    if not published:
        return DatasetFamily(provenance=None, rows=(), role_keys=query.role_keys)
    open_role_keys = query.open_role_keys - query.role_keys
    while True:
        elections = {
            dataset: _elect(
                index,
                facts,
                dataset_rows,
                role_keys=query.role_keys,
                open_role_keys=open_role_keys,
            )
            for dataset, dataset_rows in published.items()
        }
        answering = {
            dataset: election for dataset, election in elections.items() if not election.tied
        } or elections
        role_keys = _serving_role_keys(
            query.role_keys, (election.role_keys for election in answering.values())
        )
        if role_keys - query.role_keys == open_role_keys:
            break
        open_role_keys = role_keys - query.role_keys
    serving = [
        dataset for dataset, election in answering.items() if election.role_keys == role_keys
    ]
    covering = _declaring_datasets(index, query.five_tuple, serving, role_keys)
    families = {
        dataset: _family_rows(published[dataset], answering[dataset])
        for dataset in covering or serving
    }
    # `min` returns the first of equals, so candidates whose roles match alike keep their
    # rank order.
    elected = min(
        families, key=lambda dataset: not _places_exactly(facts, families[dataset], role_keys)
    )
    return DatasetFamily(
        provenance=elected,
        rows=families[elected],
        role_keys=role_keys,
        tied=answering[elected].tied,
    )


def _serving_role_keys(
    role_keys: frozenset[str], elected: Iterable[frozenset[str]]
) -> frozenset[str]:
    """The role keys a query's family is read under, given the keys each candidate
    dataset's own election ran under (`statvar_members.RoleClassElection.role_keys`): the
    largest of them. When candidates ran under two or more different sets of that size, no
    open key outranks another, and the keys are the query's `role_keys` alone, as they are
    with no candidate."""
    distinct = set(elected)
    size = max(map(len, distinct), default=0)
    largest = [keys for keys in distinct if len(keys) == size]
    return largest[0] if len(largest) == 1 else role_keys


def _refuse_non_publisher(index: CodeIndex, query: FamilyQuery) -> None:
    """Raises `SourceNotInFamily` when `query` names a dataset that does not publish its
    five-tuple and another dataset does. The test reads no row."""
    serving = publishers(index, query)
    if query.source is not None and serving and query.source not in serving:
        raise SourceNotInFamily(source=query.source, candidates=serving)


def bind_slot_value(index: CodeIndex, query: FamilyQuery, prop: str, value: str) -> FamilyQuery:
    """Returns `query` with `prop` bound to `value` widened to its expansion
    (`CodeIndex.expansion`), as an identity that binds the value states it
    (`identity_family_query`)."""
    refs = index.expansion(value)
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
            datasets=frozenset(dataset for dataset, _ in carried),
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
    """The rows of the family `query`'s slot keys and `prop` select that satisfy every
    binding but `prop`'s and declare only those keys, in index order: the rows
    `family_for_five_tuple` reads for `query` with any value bound to `prop`."""
    others = _allowed_values({other: b for other, b in query.slots.items() if other != prop})
    keys = query.slots.keys() | {prop}
    family = _family(index, query.five_tuple, keys, query.bound | {prop}, query.source)
    if family is None:
        return ()
    class_of = _five_tuple_facts(index, query.five_tuple)[0].class_of
    return tuple(
        row
        for row in family.rows
        if class_of[row.dcid].keys <= keys and (not others or _satisfies(row, others))
    )


def offers_slot(index: CodeIndex, query: FamilyQuery, prop: str) -> bool:
    """Returns whether a dataset that can serve `query` publishes a row of its five-tuple
    that carries `prop`, so that a menu of `prop`'s values offers a value some election
    serves.

    The datasets that can serve are those of `eligible_datasets` (the named one alone for a
    query that names one) that publish a row matching every other binding of `query`,
    whatever its value of `prop`. The role keys are the largest set those datasets' rows
    elect under (`_serving_role_keys` of `statvar_members.electing_role_keys`), read once
    over the rows of every value, and of those datasets the ones with a matching row that
    places the keys can serve when any has one. This is a wider test than
    `elect_dataset`'s, which elects per value, drops a tied candidate and reads coverage
    from the roles each dataset declares in its family. The datasets publishing one family
    carry the same properties (`IndexFamily`), so the two tests differ only for slot keys
    that select no family. When none publishes a matching row, every eligible dataset can. On a five-tuple one dataset publishes, this
    holds exactly when that dataset's rows carry `prop`, and no row is read. A five-tuple
    the index does not carry offers every slot (`datasets_carry`)."""
    serving = eligible_datasets(index, query)
    if len(serving) > 1:
        rows = _slot_open_rows(index, query, prop)
        matching = [d for d in serving if any(d in row.provenances for row in rows)]
        facts = _rows_facts(index, query.five_tuple, rows)
        readings = absent_readings(index)
        role_keys = _serving_role_keys(
            query.role_keys,
            (
                electing_role_keys(
                    facts,
                    [row.dcid for row in rows if d in row.provenances],
                    role_keys=query.role_keys,
                    open_role_keys=query.open_role_keys,
                    readings=readings,
                )
                for d in matching
            ),
        )
        covering = [
            d
            for d in matching
            if any(
                d in row.provenances and role_class_covers(_row_roles(row), role_keys)
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
    queries (`index_decides`).

    The rows are read once per query and `prop` and bucketed by their value of `prop`; each
    value elects over the buckets of its expansion (`CodeIndex.expansion`). Both the buckets and each value's
    family are memoised per index, query less `prop`'s binding, and `prop`, since every
    option menu of one answer reads them.

    Raises `SourceNotInFamily` as `elect_dataset` does."""
    if not _decides(index, query.five_tuple, query.slots.keys() | {prop}, query.bound | {prop}):
        return None
    memo = _per_index(_slot_options_cache, index, lambda: _LruMemo(budget=_SLOT_OPTIONS_REFS))
    key, options = _slot_options(memo, index, query, prop)
    families: dict[str, DatasetFamily] = {}
    for value in values:
        family = options.families.get(value)
        if family is None:
            refs = dict.fromkeys(index.expansion(value))
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
    else the publishers of the query's family (`publishers`) that `eligible_among` keeps
    under the query's scope. Search's shape candidacy and its pre-bind readers hold a
    shape's own tables and apply the same filter to them (`eligible_among`), so they and
    every door's election keep one rule."""
    if query.source is not None:
        return (query.source,)
    return eligible_among(index, publishers(index, query), scope=query.scope)


def eligible_among(
    index: CodeIndex, datasets: Sequence[str], *, scope: ProvenanceScope | None
) -> tuple[str, ...]:
    """Returns the datasets of `datasets` a question under `scope` elects among, in the
    order given.

    With no scope: those that are not `opt_in`, or the `opt_in` ones when `datasets` holds
    no other. Under a scope, the pool is `datasets` in `scope.allowed`: its datasets that
    are not `opt_in`, else the whole pool when the scope names a source or when `datasets`
    holds no dataset that is not `opt_in`, else none. So excluding a table never opens an
    opt-in table on a family another table publishes, and an opt-in-only family stays
    answerable under an exclusion that leaves it a publisher. A caller that holds a
    family's datasets filters them here without selecting the family again."""
    if scope is None:
        default = tuple(dataset for dataset in datasets if not is_opt_in(index, dataset))
        return default or tuple(datasets)
    pool = tuple(dataset for dataset in datasets if dataset in scope.allowed)
    default = tuple(dataset for dataset in pool if not is_opt_in(index, dataset))
    if default:
        return default
    if scope.named or all(is_opt_in(index, dataset) for dataset in datasets):
        return pool
    return ()


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


def _elect(
    index: CodeIndex,
    facts: RoleClassFacts,
    rows: Sequence[StatVarRow],
    *,
    role_keys: frozenset[str],
    open_role_keys: frozenset[str] = frozenset(),
) -> RoleClassElection:
    """Runs `elect_role_class` over `rows`, whose facts `facts` holds, under the index's
    declared absent readings."""
    return elect_role_class_of(
        facts,
        [row.dcid for row in rows],
        role_keys=role_keys,
        open_role_keys=open_role_keys,
        readings=absent_readings(index),
    )


def _family_rows(rows: Sequence[StatVarRow], election: RoleClassElection) -> tuple[StatVarRow, ...]:
    """The rows of `rows` in `election`'s family, in the order given."""
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

    Rows are those of the family the query's slot keys select (`query_family`), satisfy
    every binding, and declare only keys in `query.slots`. `elect_dataset` then keeps the rows of the elected dataset. Rows
    keep index order, as `CodeIndex.filter` does. No match returns a family with no
    provenance and no rows, including when the index carries no such five-tuple.

    Raises `SourceNotInFamily` as `elect_dataset` does."""
    return elect_dataset(index, _matching_rows(index, query), query)


def _matching_rows(index: CodeIndex, query: FamilyQuery) -> tuple[StatVarRow, ...]:
    """The rows of `query`'s family (`query_family`) that satisfy every binding and declare
    only keys in `query.slots`, in index order."""
    allowed = _allowed_values(query.slots)
    family = query_family(index, query)
    return tuple(
        row
        for row in (family.rows if family is not None else ())
        if _satisfies(row, allowed) and row.constraint_properties.keys() <= query.slots.keys()
    )


def family_among(
    index: CodeIndex, query: FamilyQuery, datasets: Sequence[str]
) -> DatasetFamily | None:
    """Returns the family `elect_dataset` elects for `query`, which names no dataset, among
    `datasets` alone, or `None` when the index does not decide the query's membership
    (`index_decides`).

    `datasets` publish one family of the query's five-tuple and come in rank order. The
    rows are the ones a query naming one of them reads (`query_family`): that family's,
    whichever family the slot keys alone select. So the dataset elected is one whose own
    rows match the query, and none is elected when no dataset of `datasets` holds a
    matching row, whatever another dataset of the five-tuple holds. Search elects a part's
    dataset here among the tables the part may answer from."""
    if not datasets or not index_decides(index, query):
        return None
    rows = _matching_rows(index, replace(query, source=datasets[0]))
    return elect_dataset(index, rows, query, among=datasets)


def index_family(index: CodeIndex, query: FamilyQuery) -> DatasetFamily | None:
    """Returns the family the index resolves for `query`, or `None` when the index does
    not decide its membership.

    The index decides membership when `index_decides` holds; the family is then
    `family_for_five_tuple`'s. The Datasets mint and Search both resolve a family through
    this function.

    Raises `SourceNotInFamily` as `elect_dataset` does, whether or not the index decides
    membership, so a named dataset that does not publish the five-tuple refuses whatever
    slots the identity declares."""
    _refuse_non_publisher(index, query)
    if not index_decides(index, query):
        return None
    return family_for_five_tuple(index, query)


def deciding_query(index: CodeIndex, query: FamilyQuery) -> FamilyQuery:
    """Returns `query` with only the slots that decide its family (`_deciding_keys`): less
    the unbound slots no row of its five-tuple declares.

    `index_family` resolves both queries to the same family: `index_decides` and
    `query_family` read the deciding keys alone, and the row filter of
    `family_for_five_tuple` compares slot keys against the keys rows declare. Search keys
    its memo of resolved families by this query, so the queries one family answers share
    an entry."""
    keys = _deciding_keys(index, query.five_tuple, query.slots.keys(), query.bound)
    if len(keys) == len(query.slots):
        return query
    return replace(query, slots={prop: query.slots[prop] for prop in query.slots if prop in keys})


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
    if query.source is not None and query.source not in publishers(index, query):
        return {}
    elected = family_for_five_tuple(index, query).provenance
    eligible = (elected,) if elected is not None else eligible_datasets(index, query)
    return shared_pins(index, eligible, query.five_tuple)


def elected_invariants(index: CodeIndex, query: FamilyQuery) -> dict[str, str]:
    """Returns the invariants the dataset `query` names declares
    (`DatasetDeclaration.invariants`), or none when `query` names no dataset, names one that
    does not publish its five-tuple, or declares slot keys that leave membership to the
    graph walk (`index_decides`). The loader's guarantee covers index rows alone,
    so where the graph walk decides membership nothing is implied.

    An invariant holds on every row the dataset publishes in every export, so an identity
    that names the dataset implies it: the canonical identity binds it
    (`invariant_bindings`), no declared default binds it, and Search withholds it from the
    bind menu. An identity that names no dataset may elect another in a later export, so
    for it nothing is implied. The result reads the declaration alone and runs no
    election."""
    source = query.source
    if source is None or source not in publishers(index, query):
        return {}
    if not index_decides(index, query):
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


def prepare_member_views(index: CodeIndex) -> None:
    """Builds `index`'s per-index views the member queries read: each five-tuple's rows
    and carried properties (`_five_tuple_rows`) and role-class facts
    (`_five_tuple_facts`). The first query reading a view builds it otherwise, which on the
    packaged index costs that query about 60 ms."""
    for five_tuple in _five_tuple_rows(index).rows:
        _five_tuple_facts(index, five_tuple)


def _per_index[M](cache: dict[int, M], index: CodeIndex, new: Callable[[], M]) -> M:
    """Returns `index`'s memo in `cache`, made by `new` and keyed and finalised as
    `_five_tuple_rows_cache` is."""
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
            "open_role_keys": sorted(query.open_role_keys),
            "scope": (
                None
                if query.scope is None
                else {"allowed": sorted(query.scope.allowed), "named": query.scope.named}
            ),
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


def listed_group(index: CodeIndex, query: FamilyQuery, *, member_cap: int) -> MemberGroup | None:
    """Returns the group the graph walk resolves for `query`, elected from the index alone,
    when that group holds a listed row (`CodeIndex.listed_statvars`), else `None`, which
    leaves the query to the walk.

    The export reads every StatVar of each five-tuple a listed row has and refuses unless
    each is an index row or a listed row (`_verify_listed_five_tuples` in
    `scripts/export_code_index.py`), so those rows are the walk's candidates.
    `statvar_members.group_of_arcs` runs the walk's admission and election over them, so
    Search and the Datasets mint elect the walk's members without its reads, which take
    about 50 s for an IDS family on the dev graph. Raises what the walk raises for its
    members: `MemberEnumerationTooLarge`, `MemberRoleDisagreement` and
    `DegenerateObservationProperties`."""
    listed = index.listed_by_five_tuple.get(query.five_tuple)
    if not listed:
        return None
    keys = query.slots.keys()
    # The walk (`statvar_members`) rejects a StatVar declaring a property the query has no
    # slot for; dropping those here saves building their arcs (about 12,000 CRS rows).
    candidates = [
        row
        for row in (*_five_tuple_rows(index).rows.get(query.five_tuple, ()), *listed)
        if row.constraint_properties.keys() <= keys
    ]
    group = group_of_arcs(
        query.five_tuple,
        {row.dcid: _row_arcs(row) for row in candidates},
        slots=query.slots,
        readings=absent_readings(index),
        member_cap=member_cap,
        role_keys=query.role_keys,
        open_role_keys=query.open_role_keys,
    )
    if frozenset(row.dcid for row in listed).isdisjoint(group.family):
        return None
    return group


def _row_arcs(row: StatVarRow) -> dict[str, tuple[str, ...]]:
    """The out-arcs a forward read of `row`'s StatVar returns, by graph property."""
    return {
        "typeOf": (_STATISTICAL_VARIABLE,),
        "populationType": _arc(row.population_type),
        "measuredProperty": _arc(row.measured_property),
        "statType": _arc(row.stat_type),
        "measurementQualifier": _arc(row.measurement_qualifier),
        "measurementDenominator": _arc(row.measurement_denominator),
        "observationProperties": row.observation_properties,
        "constraintProperties": tuple(row.constraint_properties),
        **{prop: (value,) for prop, value in row.constraint_properties.items()},
    }


def _arc(value: str | None) -> tuple[str, ...]:
    return () if value is None else (value,)


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
    "dataset_declares_roles",
    "datasets_carry",
    "deciding_query",
    "elect_dataset",
    "elected_invariants",
    "elected_pins",
    "eligible_among",
    "eligible_datasets",
    "family_among",
    "family_declares_roles",
    "family_for_five_tuple",
    "family_query_digest",
    "identity_family_query",
    "index_covers_slot_keys",
    "index_decides",
    "index_family",
    "invariant_bindings",
    "is_opt_in",
    "key_sets_cover",
    "listed_group",
    "named_dataset",
    "offers_slot",
    "option_families",
    "pinned_refs",
    "prepare_member_views",
    "publishers",
    "query_family",
    "shared_pins",
]
