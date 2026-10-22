"""A grid of role-class elections and the outcome the election rule decides for each.

`elect` states the rule over rows held as plain sets, and reads none of the election code
in `statvar_members.enumerate` or `code_index.members`:

1. Among the rows that place every bound role key (every row when none does), after the
   constraint antichain and the twin collapse, the smallest surviving role set is
   elected when every other surviving role set contains it and the constraint keys of
   each surviving row declaring it are among those of each surviving row declaring
   another. Role sets that do not nest tie, and so do nested ones whose rows hold other
   constraint keys. The family is every placing row of the elected role set, and the
   query reads the antichain of the family alone: a row of a role set the query does not
   read dominates nothing.
2. An open key joins the bound keys only when a row of that reading, or a role twin of
   one (a row with the same constraint map), places it. Otherwise it is ignored and the
   election runs as if the role were unnamed.
3. A dataset declares a set of keys when its rows of a family, read with nothing bound,
   place them on that reading or on a twin of it (`declares`).

`elect_among` states the rule for a family several datasets publish:

4. Each dataset is read on its own rows by rules 1 and 2. A dataset whose reading ties
   serves only when every dataset's does. Of the datasets left, those whose reading ran
   under the largest set of keys serve, and when two different sets are the largest no
   open key joins. The question is then asked again with only the open keys that joined,
   until asking changes nothing. Of the serving datasets, those that declare the keys
   (rule 3) serve ahead of those that do not, then those whose family rows declare
   exactly the roles the keys place, then the first in rank order.

`elect_any_row` states the any-row rule: an open key joins when any row places it, and
every election whose surviving rows declare more than one role set ties. The two rules
agree on every cell whose label is outside `MOVED_LABELS`.

A cell is one population of rows, one slot binding, and one pair of bound and open role
keys. `fixture_cells` crosses every dataset's rows of every family of the fixture code
index, and `synthetic_cells` the populations of `SYNTHETIC`. `shared_cells` crosses the
families of `SHARED`, each published by two or three datasets.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import random
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

from code_index.query import CodeIndex
from code_index.types import DatasetDeclaration, IndexStamp, StatVarRow
from dfo_platform.roles import ROLE_AXES
from statvar_members import SlotBinding
from statvar_members.types import AbsentReading, FiveTuple
from tests.code_index.support import INDEX_FIXTURE, load_index

SUBJECT = "observationAbout"
ORIGIN = "origin"
DESTINATION = "destination"

ONE_ROLE_SET = "one role set"
TWIN = "twin"
NESTED = "nested without twin on shared constraint keys"
NESTED_APART = "nested without twin on other constraint keys"
NOT_NESTED = "not nested"
OPEN_HONOURED = "open honoured"
OPEN_UNPLACED = "open ignored with no row placing it"
OPEN_UNTWINNED = "open ignored with only untwinned rows placing it"

LABELS = (
    ONE_ROLE_SET,
    TWIN,
    NESTED,
    NESTED_APART,
    NOT_NESTED,
    OPEN_HONOURED,
    OPEN_UNPLACED,
    OPEN_UNTWINNED,
)
MOVED_LABELS = frozenset({NESTED, OPEN_UNTWINNED})
"""The labels of the cells `elect` and `elect_any_row` may decide differently."""

_NO_KEYS: frozenset[str] = frozenset()
_FLOW_AXES = (ORIGIN, DESTINATION)

BOUND_KEYS: tuple[frozenset[str], ...] = (
    _NO_KEYS,
    frozenset({ORIGIN}),
    frozenset({DESTINATION}),
    frozenset({ORIGIN, DESTINATION}),
    frozenset({SUBJECT}),
)
OPEN_KEYS: tuple[frozenset[str], ...] = (
    _NO_KEYS,
    frozenset({ORIGIN}),
    frozenset({DESTINATION}),
    frozenset({ORIGIN, DESTINATION}),
)
ROLE_KEY_CELLS: tuple[tuple[frozenset[str], frozenset[str]], ...] = tuple(
    dict.fromkeys(
        (bound, open_keys - bound) for bound, open_keys in itertools.product(BOUND_KEYS, OPEN_KEYS)
    )
)
"""Every pair of bound role keys and open role keys less the bound ones: 13 pairs."""


@dataclasses.dataclass(frozen=True)
class Row:
    """One StatVar as the rule reads it: a constraint map and a role set."""

    dcid: str
    constraints: Mapping[str, str]
    roles: tuple[str, ...]
    """The declared roles in declared order, `observationAbout` alone when it names none."""
    keys: frozenset[str]
    """The constraint properties."""
    held: frozenset[tuple[str, str]]
    """The constraint map, property and value. Rows holding equal maps are role twins."""
    role_set: frozenset[str]


def row(dcid: str, constraints: Mapping[str, str], roles: Sequence[str]) -> Row:
    """A `Row` of `constraints` declaring `roles`, the subject alone when it names none."""
    declared = tuple(roles) or (SUBJECT,)
    return Row(
        dcid=dcid,
        constraints=constraints,
        roles=declared,
        keys=frozenset(constraints),
        held=frozenset(constraints.items()),
        role_set=frozenset(declared),
    )


@functools.cache
def places(roles: tuple[str, ...], keys: frozenset[str]) -> bool:
    """Whether a row declaring `roles` places every key of `keys`.

    A key is placed by the role of its name, and a flow axis by the one declared role on
    that axis: two declared roles on one axis place neither. With no key, the row places
    them when it declares the subject alone."""
    if not keys:
        return roles == (SUBJECT,)
    return all(
        key in roles
        or (key in _FLOW_AXES and sum(ROLE_AXES.get(role) == key for role in roles) == 1)
        for key in keys
    )


@dataclasses.dataclass(frozen=True)
class _Reading:
    """What one set of role keys reads among rows."""

    candidates: tuple[Row, ...]
    """The rows that place the keys, every row when none does."""
    survivors: tuple[Row, ...]
    """The candidates left after the constraint antichain and the twin collapse."""
    collapsed: bool
    """Whether a twin displaced a row."""
    readings: Mapping[str, AbsentReading]
    role_sets: frozenset[frozenset[str]]
    """The role sets the survivors declare."""
    smallest: frozenset[str] | None
    """The surviving role set every other one contains, or `None` when the role sets do
    not nest."""
    elected: frozenset[str] | None
    """The role set the reading elects: `smallest` when the constraint keys of each
    survivor declaring it are among those of each survivor declaring another role set.
    `None` when the reading ties."""

    @property
    def structure(self) -> str:
        if len(self.role_sets) == 1:
            return TWIN if self.collapsed else ONE_ROLE_SET
        if self.smallest is None:
            return NOT_NESTED
        return NESTED_APART if self.elected is None else NESTED


def antichain(rows: Sequence[Row], readings: Mapping[str, AbsentReading]) -> tuple[Row, ...]:
    """The rows of `rows` no other dominates, in their order. A row is dominated when
    another's constraint properties are a proper subset of its own and every property
    the subset lacks reads `aggregate`, the reading of a property `readings` does not
    name."""
    key_sets = {found.keys for found in rows}
    dominated = {
        keys
        for keys in key_sets
        if any(
            other < keys
            and all(readings.get(prop, "aggregate") == "aggregate" for prop in keys - other)
            for other in key_sets
        )
    }
    return tuple(found for found in rows if found.keys not in dominated)


def _read(
    rows: Sequence[Row], keys: frozenset[str], readings: Mapping[str, AbsentReading]
) -> _Reading:
    """Read `rows` under `keys`.

    The antichain (`antichain`) runs over every candidate. The twin collapse then drops a
    row when another holds the same constraint map under a role set that is a proper
    subset of its own."""
    candidates = [found for found in rows if places(found.roles, keys)] or list(rows)
    minimal = antichain(candidates, readings)
    role_sets_held: dict[frozenset[tuple[str, str]], set[frozenset[str]]] = defaultdict(set)
    for found in minimal:
        role_sets_held[found.held].add(found.role_set)
    survivors = [
        found
        for found in minimal
        if not any(other < found.role_set for other in role_sets_held[found.held])
    ]
    role_sets = frozenset(found.role_set for found in survivors)
    smallest = next((s for s in role_sets if all(s <= other for other in role_sets)), None)
    own_keys = {found.keys for found in survivors if found.role_set == smallest}
    other_keys = {found.keys for found in survivors if found.role_set != smallest}
    inside = all(own <= other for own in own_keys for other in other_keys)
    return _Reading(
        candidates=tuple(candidates),
        survivors=tuple(survivors),
        collapsed=len(survivors) < len(minimal),
        readings=readings,
        role_sets=role_sets,
        smallest=smallest,
        elected=smallest if inside else None,
    )


def _admitted(
    role_tuples: Collection[tuple[str, ...]], bound: frozenset[str], open_keys: frozenset[str]
) -> frozenset[str]:
    """The open keys that join `bound` when rows declaring `role_tuples` may place them.

    They are the largest subset of `open_keys` one row places together with `bound`. Two
    subsets of that size placed by different rows outrank neither, and no key joins."""
    for size in range(len(open_keys), 0, -1):
        placed = [
            subset
            for subset in map(frozenset, itertools.combinations(sorted(open_keys), size))
            if any(places(roles, bound | subset) for roles in role_tuples)
        ]
        if placed:
            return placed[0] if len(placed) == 1 else _NO_KEYS
    return _NO_KEYS


def _twinned(rows: Iterable[Row], reading: _Reading) -> set[tuple[str, ...]]:
    """The role tuples of the rows `reading` reads and of their twins among `rows`."""
    held = {found.held for found in _members(reading, nested_tie=False)}
    return {found.roles for found in rows if found.held in held}


def _members(reading: _Reading, *, nested_tie: bool) -> tuple[Row, ...]:
    """The rows `reading` elects: the antichain of its family alone, or every survivor
    when it ties. Where one role set survives, those are its survivors."""
    if _tied(reading, nested_tie=nested_tie) or len(reading.role_sets) == 1:
        return reading.survivors
    return antichain(_family(reading, nested_tie=nested_tie), reading.readings)


def _family(reading: _Reading, *, nested_tie: bool) -> tuple[Row, ...]:
    """The rows of the elected role set before the antichain: the candidates declaring
    it, or every candidate when the reading ties."""
    if _tied(reading, nested_tie=nested_tie):
        return reading.candidates
    return tuple(found for found in reading.candidates if found.role_set == reading.elected)


def _tied(reading: _Reading, *, nested_tie: bool) -> bool:
    return len(reading.role_sets) > 1 and (nested_tie or reading.elected is None)


@dataclasses.dataclass(frozen=True)
class Election:
    """What a rule decides for one cell."""

    family: frozenset[str]
    """The dcids of the rows of the elected role set before the antichain. Every row that
    places the elected keys when the election ties."""
    members: frozenset[str]
    """The dcids the query reads: the antichain of the family. Every surviving row when
    the election ties."""
    tied: bool
    keys: frozenset[str]
    """The keys the election ran under: the bound ones and the open ones that joined."""
    label: str = ""
    """The case, one of `LABELS`. Empty on `elect_any_row`'s result."""

    @property
    def outcome(self) -> tuple[frozenset[str], frozenset[str], bool, frozenset[str]]:
        """Everything but the label."""
        return (self.family, self.members, self.tied, self.keys)


def _election(reading: _Reading, keys: frozenset[str], *, nested_tie: bool, label: str = ""):
    return Election(
        family=frozenset(found.dcid for found in _family(reading, nested_tie=nested_tie)),
        members=frozenset(found.dcid for found in _members(reading, nested_tie=nested_tie)),
        tied=_tied(reading, nested_tie=nested_tie),
        keys=keys,
        label=label,
    )


def elect(
    rows: Sequence[Row],
    *,
    bound: Collection[str] = (),
    open_keys: Collection[str] = (),
    readings: Mapping[str, AbsentReading],
) -> Election:
    """What the rule decides for `rows` under `bound` and `open_keys`, with its label.

    The label names the case in this order. `OPEN_UNTWINNED` when the rows the bound keys
    read and their twins admit other open keys than every row would. `NESTED` when the
    elected keys leave nested role sets no twin relates and the smallest is elected, and
    `NESTED_APART` when they tie on their constraint keys. With an open key asked,
    `OPEN_HONOURED` when one joined and `OPEN_UNPLACED` when no row places any. Else the
    structure the elected keys leave: `ONE_ROLE_SET`, `TWIN` when a twin was displaced, or
    `NOT_NESTED`, which also names open keys that different rows place and none outranks."""
    bound_keys = frozenset(bound)
    asked = frozenset(open_keys) - bound_keys
    reading = _read(rows, bound_keys, readings)
    admitted = _admitted(_twinned(rows, reading), bound_keys, asked)
    by_any_row = _admitted({found.roles for found in rows}, bound_keys, asked)
    keys = bound_keys | admitted
    elected = _read(rows, keys, readings) if admitted else reading
    if admitted != by_any_row:
        label = OPEN_UNTWINNED
    elif elected.structure in (NESTED, NESTED_APART):
        label = elected.structure
    elif admitted:
        label = OPEN_HONOURED
    elif asked and not any(
        places(found.roles, bound_keys | {key}) for found in rows for key in asked
    ):
        label = OPEN_UNPLACED
    else:
        label = elected.structure
    return _election(elected, keys, nested_tie=False, label=label)


def elect_any_row(
    rows: Sequence[Row],
    *,
    bound: Collection[str] = (),
    open_keys: Collection[str] = (),
    readings: Mapping[str, AbsentReading],
) -> Election:
    """What the any-row rule decides: an open key joins when any row of `rows` places it,
    and surviving rows of more than one role set tie, nested or not."""
    bound_keys = frozenset(bound)
    asked = frozenset(open_keys) - bound_keys
    keys = bound_keys | _admitted({found.roles for found in rows}, bound_keys, asked)
    return _election(_read(rows, keys, readings), keys, nested_tie=True)


def declares(
    rows: Sequence[Row], keys: Collection[str], *, readings: Mapping[str, AbsentReading]
) -> bool:
    """Whether a dataset publishing `rows` in one family declares `keys` there: a row its
    rows read with nothing bound, or a twin of one, places them. False for no row."""
    role_keys = frozenset(keys)
    return any(
        places(roles, role_keys) for roles in _twinned(rows, _read(rows, _NO_KEYS, readings))
    )


ONE_PUBLISHER = "one dataset publishes a row"
TIE_SET_ASIDE = "a tied dataset leaves for one that answers"
ALL_TIE = "every dataset ties"
LARGEST_KEYS = "the datasets under the largest keys serve"
DECLARING = "a dataset that declares the keys serves first"
EXACT = "a dataset placing the keys exactly serves first"
RANK = "rank decides"
ASKED_AGAIN = "asked again with the joined keys"

DATASET_CASES = (
    ONE_PUBLISHER,
    TIE_SET_ASIDE,
    ALL_TIE,
    LARGEST_KEYS,
    DECLARING,
    EXACT,
    RANK,
    ASKED_AGAIN,
)


@dataclasses.dataclass(frozen=True)
class DatasetElection:
    """What rule 4 decides for one cell of a family several datasets publish."""

    dataset: str
    election: Election
    """The elected dataset's own rows read under the bound keys and the open keys that
    joined. Its `keys` are the keys the family is read under."""
    case: str
    """The step that decided, one of `DATASET_CASES`: `ASKED_AGAIN` when the elected
    dataset reads differently under the open keys that joined than under those asked,
    else the first step that set a dataset aside."""


def _places_exactly(found: Row, keys: frozenset[str]) -> bool:
    """Whether `found` declares exactly the roles `keys` place: one role for each key, or
    the subject alone with no key."""
    return places(found.roles, keys) and len(found.roles) == max(len(keys), 1)


def elect_among(
    published: Mapping[str, Sequence[Row]],
    declared: Mapping[str, Sequence[Row]],
    *,
    bound: Collection[str] = (),
    open_keys: Collection[str] = (),
    readings: Mapping[str, AbsentReading],
) -> DatasetElection:
    """What rule 4 decides among datasets in rank order.

    `published` holds the rows of each dataset the question's slot binding admits, and
    `declared` every row each publishes in the family, which rule 3 reads. A dataset
    with no admitted row is no candidate, and at least one has a row."""
    bound_keys = frozenset(bound)

    def read_under(asked: frozenset[str]) -> dict[str, Election]:
        return {
            dataset: elect(rows, bound=bound_keys, open_keys=asked, readings=readings)
            for dataset, rows in published.items()
            if rows
        }

    asked = frozenset(open_keys) - bound_keys
    first = read = read_under(asked)
    while (keys := _serving_keys(read, bound_keys)) - bound_keys != asked:
        asked = keys - bound_keys
        read = read_under(asked)
    answering = _answering(read)
    serving = [dataset for dataset, found in answering.items() if found.keys == keys]
    declaring = [
        dataset for dataset in serving if declares(declared[dataset], keys, readings=readings)
    ] or serving
    by_dcid = {found.dcid: found for rows in published.values() for found in rows}
    exact = [
        dataset
        for dataset in declaring
        if all(_places_exactly(by_dcid[dcid], keys) for dcid in read[dataset].family)
    ] or declaring
    elected = exact[0]
    steps = (
        (ASKED_AGAIN, first[elected].outcome != read[elected].outcome),
        (ONE_PUBLISHER, len(read) == 1),
        (TIE_SET_ASIDE, len(answering) < len(read)),
        (ALL_TIE, all(found.tied for found in read.values())),
        (LARGEST_KEYS, len(serving) < len(answering)),
        (DECLARING, len(declaring) < len(serving)),
        (EXACT, len(exact) < len(declaring)),
    )
    case = next((name for name, decided in steps if decided), RANK)
    return DatasetElection(dataset=elected, election=read[elected], case=case)


def _answering(read: Mapping[str, Election]) -> Mapping[str, Election]:
    """The datasets of `read` whose reading does not tie, every one when all tie."""
    return {dataset: found for dataset, found in read.items() if not found.tied} or read


def _serving_keys(read: Mapping[str, Election], bound: frozenset[str]) -> frozenset[str]:
    """The largest set of keys an answering dataset's reading ran under, or `bound` when
    two different sets are the largest."""
    key_sets = {found.keys for found in _answering(read).values()}
    largest = [keys for keys in key_sets if len(keys) == max(map(len, key_sets))]
    return largest[0] if len(largest) == 1 else bound


@dataclasses.dataclass(frozen=True)
class Population:
    """The rows one dataset publishes in one family."""

    name: str
    five_tuple: FiveTuple
    dataset: str
    rows: tuple[Row, ...]
    readings: Mapping[str, AbsentReading]
    """How an absent constraint property reads. An unnamed property reads `aggregate`."""

    @property
    def properties(self) -> frozenset[str]:
        return frozenset().union(*(found.keys for found in self.rows))

    def slots(self) -> Iterator[tuple[str, str] | None]:
        """No binding, then each property bound to each value a row holds."""
        yield None
        yield from sorted({pair for found in self.rows for pair in found.held})


@dataclasses.dataclass(frozen=True)
class Cell:
    population: Population
    slot: tuple[str, str] | None
    """The one property bound to a value, or `None` when every slot is unbound."""
    bound: frozenset[str]
    open_keys: frozenset[str]
    rows: tuple[Row, ...]
    """The population's rows the slot binding admits."""
    expected: Election

    @property
    def slots(self) -> dict[str, SlotBinding]:
        """The slot bindings a query for this cell carries: every property of the
        population unbound, but for the one the cell binds."""
        slots: dict[str, SlotBinding] = dict.fromkeys(self.population.properties, ("unbound",))
        if self.slot is not None:
            slots[self.slot[0]] = ("value", self.slot[1])
        return slots

    def __str__(self) -> str:
        slot = "unbound" if self.slot is None else "=".join(self.slot)
        return (
            f"{self.population.name} [{slot}] bound={sorted(self.bound)} "
            f"open={sorted(self.open_keys)}: {self.expected.label}"
        )


def cells(population: Population) -> Iterator[Cell]:
    """Every cell of `population`: each slot binding under each pair of `ROLE_KEY_CELLS`."""
    for slot in population.slots():
        rows = tuple(found for found in population.rows if slot is None or slot in found.held)
        for bound, open_keys in ROLE_KEY_CELLS:
            expected = elect(rows, bound=bound, open_keys=open_keys, readings=population.readings)
            yield Cell(population, slot, bound, open_keys, rows, expected)


@dataclasses.dataclass(frozen=True)
class SharedFamily:
    """One family that several datasets publish."""

    name: str
    populations: tuple[Population, ...]
    """Each dataset's rows of the family, in rank order. A row two datasets publish is in
    both populations under one dcid."""

    @property
    def five_tuple(self) -> FiveTuple:
        return self.populations[0].five_tuple

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(population.dataset for population in self.populations)

    @property
    def readings(self) -> Mapping[str, AbsentReading]:
        return self.populations[0].readings

    @property
    def properties(self) -> frozenset[str]:
        return frozenset().union(*(population.properties for population in self.populations))

    def slots(self) -> Iterator[tuple[str, str] | None]:
        """No binding, then each property bound to each value a row of any dataset holds."""
        yield None
        yield from sorted({slot for p in self.populations for slot in p.slots() if slot})


@dataclasses.dataclass(frozen=True)
class SharedCell:
    """One slot binding and one pair of role keys asked of a `SharedFamily`."""

    family: SharedFamily
    slot: tuple[str, str] | None
    bound: frozenset[str]
    open_keys: frozenset[str]
    rows: Mapping[str, tuple[Row, ...]]
    """Each dataset's rows the slot binding admits, in rank order. Empty for a dataset
    with none."""
    expected: DatasetElection
    """What a question naming no dataset elects."""

    @property
    def slots(self) -> dict[str, SlotBinding]:
        """The slot bindings a query for this cell carries, as `Cell.slots`."""
        slots: dict[str, SlotBinding] = dict.fromkeys(self.family.properties, ("unbound",))
        if self.slot is not None:
            slots[self.slot[0]] = ("value", self.slot[1])
        return slots

    def named(self, dataset: str) -> Election | None:
        """What a question naming `dataset` elects: its own rows read by rules 1 and 2,
        or `None` when the slot binding admits none of them."""
        if not self.rows[dataset]:
            return None
        return elect(
            self.rows[dataset],
            bound=self.bound,
            open_keys=self.open_keys,
            readings=self.family.readings,
        )

    def __str__(self) -> str:
        slot = "unbound" if self.slot is None else "=".join(self.slot)
        return (
            f"{self.family.name} [{slot}] bound={sorted(self.bound)} "
            f"open={sorted(self.open_keys)}: {self.expected.case}"
        )


def cells_of_shared(family: SharedFamily) -> Iterator[SharedCell]:
    """Every cell of `family`: each slot binding under each pair of `ROLE_KEY_CELLS`."""
    declared = {population.dataset: population.rows for population in family.populations}
    for slot in family.slots():
        rows = {
            dataset: tuple(found for found in published if slot is None or slot in found.held)
            for dataset, published in declared.items()
        }
        for bound, open_keys in ROLE_KEY_CELLS:
            expected = elect_among(
                rows, declared, bound=bound, open_keys=open_keys, readings=family.readings
            )
            yield SharedCell(family, slot, bound, open_keys, rows, expected)


def count_by_case(found: Iterable[SharedCell]) -> dict[str, int]:
    """The number of cells under each of `DATASET_CASES`."""
    counts = dict.fromkeys(DATASET_CASES, 0)
    for cell in found:
        counts[cell.expected.case] += 1
    return counts


def count_by_label(found: Iterable[Cell]) -> dict[str, int]:
    """The number of cells under each of `LABELS`."""
    counts = dict.fromkeys(LABELS, 0)
    for cell in found:
        counts[cell.expected.label] += 1
    return counts


_PROVIDER = ("financingProvider",)
_RECIPIENT = ("financingRecipient",)
_PAIR = ("financingProvider", "financingRecipient")

SYNTHETIC_FIVE_TUPLE = FiveTuple(
    population_type="FinancialFlow", measured_property="amount", stat_type="measuredValue"
)
SYNTHETIC_DATASET = "provenance/RoleGrid"

_SERVICE = {"flowType": "Service"}
_SERVICE_PUBLIC = {"flowType": "Service", "guarantee": "Public"}
_DISBURSEMENT = {"flowType": "Disbursement"}
_ARREARS = {"flowType": "Arrears"}


def _population(name: str, *rows: Row, dataset: str = SYNTHETIC_DATASET) -> Population:
    return Population(
        name=name,
        five_tuple=SYNTHETIC_FIVE_TUPLE,
        dataset=dataset,
        rows=rows,
        readings={},
    )


SYNTHETIC: tuple[Population, ...] = (
    _population(
        "all rows two-role",
        row("flow_service", _SERVICE, _PAIR),
        row("flow_service_public", _SERVICE_PUBLIC, _PAIR),
        row("flow_disbursement", _DISBURSEMENT, _PAIR),
    ),
    _population(
        "totals with untwinned two-role rows",
        row("total_service", _SERVICE, _PROVIDER),
        row("total_service_public", _SERVICE_PUBLIC, _PROVIDER),
        row("total_disbursement", _DISBURSEMENT, _PROVIDER),
        row("flow_arrears", _ARREARS, _PAIR),
        row("flow_arrears_public", {"flowType": "Arrears", "guarantee": "Public"}, _PAIR),
    ),
    _population(
        "total and twin pairs",
        row("total_service", _SERVICE, _RECIPIENT),
        row("by_creditor_service", _SERVICE, _PAIR),
        row("total_service_public", _SERVICE_PUBLIC, _RECIPIENT),
        row("by_creditor_service_public", _SERVICE_PUBLIC, _PAIR),
        row("total_disbursement", _DISBURSEMENT, _RECIPIENT),
        row("by_creditor_disbursement", _DISBURSEMENT, _PAIR),
    ),
    _population(
        "pairs with an untwinned total and an untwinned two-role row",
        row("total_service", _SERVICE, _RECIPIENT),
        row("by_creditor_service", _SERVICE, _PAIR),
        row("total_service_public", _SERVICE_PUBLIC, _RECIPIENT),
        row("by_creditor_service_public", _SERVICE_PUBLIC, _PAIR),
        row("total_arrears", _ARREARS, _RECIPIENT),
        row("lender_disbursement", _DISBURSEMENT, _PAIR),
    ),
    _population(
        "recipient totals beside a subject-only row",
        row("total_service", _SERVICE, _RECIPIENT),
        row("total_disbursement", _DISBURSEMENT, _RECIPIENT),
        row("subject_service", _SERVICE, ()),
        row("subject_arrears_public", {"flowType": "Arrears", "guarantee": "Public"}, ()),
    ),
    _population(
        "provider-only beside recipient-only",
        row("line_service", _SERVICE, _PROVIDER),
        row("total_service", _SERVICE, _RECIPIENT),
        row("line_disbursement", _DISBURSEMENT, _PROVIDER),
        row("total_arrears", _ARREARS, _RECIPIENT),
    ),
    _population(
        "nested role sets on different constraint key sets",
        row("total_by_flow", _SERVICE, _RECIPIENT),
        row("by_creditor_by_guarantee", {"guarantee": "Public"}, _PAIR),
    ),
    _population(
        "a total only a two-role row dominates",
        row("total_service", _SERVICE, _RECIPIENT),
        row("total_public_by_term", {"guarantee": "Public", "term": "Long"}, _RECIPIENT),
        row("by_creditor_public", {"guarantee": "Public"}, _PAIR),
    ),
)
"""In the last, only a row of the larger role set dominates `total_public_by_term`, and
that row and `total_service` hold other constraint keys, so a bound recipient ties."""


FIRST_DATASET = "provenance/RoleGridFirst"
SECOND_DATASET = "provenance/RoleGridSecond"
THIRD_DATASET = "provenance/RoleGridThird"
"""The datasets of the families of `SHARED`, in rank order."""

_CREDITOR = ("creditor",)
_DEBTOR = ("debtor",)
_PUBLIC = {"guarantee": "Public"}
_SERVICE_PRIVATE = {"flowType": "Service", "guarantee": "Private"}
_DISBURSEMENT_PUBLIC = {"flowType": "Disbursement", "guarantee": "Public"}
_SERVICE_CREDITOR_BY_BOTH = row("service_public_by_creditor", _SERVICE_PUBLIC, _CREDITOR)


def _shared(name: str, *published: Sequence[Row]) -> SharedFamily:
    """A family whose datasets, in rank order, publish the rows of `published`."""
    datasets = (FIRST_DATASET, SECOND_DATASET, THIRD_DATASET)
    return SharedFamily(
        name=name,
        populations=tuple(
            _population(
                f"{name}: {dataset.removeprefix('provenance/RoleGrid').lower()}",
                *rows,
                dataset=dataset,
            )
            for dataset, rows in zip(datasets, published, strict=False)
        ),
    )


SHARED: tuple[SharedFamily, ...] = (
    _shared(
        "provider totals beside two-role flows",
        (
            row("total_service", _SERVICE, _PROVIDER),
            row("total_disbursement", _DISBURSEMENT, _PROVIDER),
        ),
        (
            row("flow_service", _SERVICE, _PAIR),
            row("flow_disbursement", _DISBURSEMENT, _PAIR),
            row("flow_arrears", _ARREARS, _PAIR),
        ),
    ),
    _shared(
        "two names on one axis beside one",
        (
            row("service_by_creditor", _SERVICE, _CREDITOR),
            row("public_by_provider", _PUBLIC, _PROVIDER),
        ),
        (row("service_public_by_provider", _SERVICE_PUBLIC, _PROVIDER),),
    ),
    _shared(
        "two names on one axis in both",
        (
            row("service_by_creditor", _SERVICE, _CREDITOR),
            row("public_by_provider", _PUBLIC, _PROVIDER),
        ),
        (
            row("service_by_provider", _SERVICE, _PROVIDER),
            row("public_by_creditor", _PUBLIC, _CREDITOR),
        ),
    ),
    _shared(
        "a subject total twinned on both axes beside one twinned on one",
        (
            row("subject_service", _SERVICE, ()),
            row("line_service", _SERVICE, _PROVIDER),
            row("total_service", _SERVICE, _RECIPIENT),
        ),
        (
            row("subject_disbursement", _DISBURSEMENT, ()),
            row("line_disbursement", _DISBURSEMENT, _PROVIDER),
        ),
    ),
    _shared(
        "a provider row and a recipient row beside a provider row",
        (
            row("line_service", _SERVICE, _PROVIDER),
            row("total_public", _PUBLIC, _RECIPIENT),
        ),
        (row("line_service_public", _SERVICE_PUBLIC, _PROVIDER),),
    ),
    _shared(
        "a creditor row both publish",
        (
            _SERVICE_CREDITOR_BY_BOTH,
            row("service_private_by_debtor", _SERVICE_PRIVATE, _DEBTOR),
        ),
        (
            row("service_private_by_creditor", _SERVICE_PRIVATE, _CREDITOR),
            _SERVICE_CREDITOR_BY_BOTH,
        ),
    ),
    _shared(
        "a creditor row both publish beside a provider row",
        (
            _SERVICE_CREDITOR_BY_BOTH,
            row("service_private_by_debtor", _SERVICE_PRIVATE, _DEBTOR),
            row("disbursement_public_by_provider", _DISBURSEMENT_PUBLIC, _PROVIDER),
        ),
        (
            row("service_private_by_creditor", _SERVICE_PRIVATE, _CREDITOR),
            _SERVICE_CREDITOR_BY_BOTH,
        ),
    ),
    _shared(
        "a provider row with an unplaced role, a provider row and a recipient row",
        (row("flow_by_provider_and_channel", {}, ("financingProvider", "channel")),),
        (row("flow_by_provider", {}, _PROVIDER),),
        (row("flow_by_recipient", {}, _RECIPIENT),),
    ),
)
"""Families two or three datasets publish. In the fourth, the first dataset's subject total
has a twin on each axis, so with both axes open neither joins for it, and with the origin
alone open it reads its provider twin. In the last, with nothing bound and both axes open,
two datasets read the origin and the third the destination, so no open key joins and rank
elects the first, whose row declares a provider. Asked with the origin alone open, the
second serves, since its row declares exactly the role the origin places."""


def fixture_populations(index: CodeIndex) -> Iterator[Population]:
    """The rows each dataset publishes in each family of `index`, under the absent
    readings its code systems declare."""
    readings = {system: declared.absent_reading for system, declared in index.declarations.items()}
    for five_tuple, families in index.families_by_five_tuple.items():
        for family, dataset in ((f, d) for f in families for d in f.datasets):
            measure = "/".join(
                str(part) for part in dataclasses.astuple(five_tuple)[:5] if part is not None
            )
            yield Population(
                name=f"{dataset} {measure}",
                five_tuple=five_tuple,
                dataset=dataset,
                rows=tuple(
                    row(found.dcid, found.constraint_properties, found.observation_properties)
                    for found in family.rows
                    if dataset in found.provenances
                ),
                readings=readings,
            )


@functools.cache
def synthetic_index(name: str) -> CodeIndex:
    """An index holding the population of `SYNTHETIC` named `name` alone, published by one
    dataset. It declares no code system, so every absent property reads `aggregate`."""
    return _index(name, [next(found for found in SYNTHETIC if found.name == name)])


@functools.cache
def shared_index(name: str) -> CodeIndex:
    """An index holding the family of `SHARED` named `name` alone, its datasets ranked in
    their order. It declares no code system, so every absent property reads `aggregate`."""
    return _index(name, next(found for found in SHARED if found.name == name).populations)


def _index(name: str, populations: Sequence[Population]) -> CodeIndex:
    """An index of the rows of `populations`, each published by the datasets whose
    population holds it, the datasets ranked in the order given."""
    five_tuple = populations[0].five_tuple
    rows = {found.dcid: found for population in populations for found in population.rows}
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash=f"role-grid/{name}",
        ),
        declarations=(),
        codes=(),
        statvars=tuple(
            StatVarRow(
                dcid=found.dcid,
                population_type=five_tuple.population_type,
                measured_property=five_tuple.measured_property or "",
                stat_type=five_tuple.stat_type,
                measurement_qualifier=five_tuple.measurement_qualifier,
                measurement_denominator=five_tuple.measurement_denominator,
                observation_properties=() if found.roles == (SUBJECT,) else found.roles,
                constraint_properties=dict(found.constraints),
                provenances=tuple(
                    population.dataset for population in populations if found in population.rows
                ),
            )
            for found in rows.values()
        ),
        datasets=tuple(
            DatasetDeclaration(
                provenance=population.dataset,
                label=population.dataset,
                description=population.dataset,
                opt_in=False,
                absent_observation="zero",
                account="provider",
                rank=rank,
                invariants={},
            )
            for rank, population in enumerate(populations)
        ),
    )


@functools.cache
def fixture_index() -> CodeIndex:
    return load_index(INDEX_FIXTURE)


@functools.cache
def fixture_cells() -> tuple[Cell, ...]:
    """Every cell of every population of the fixture code index."""
    return tuple(
        cell for population in fixture_populations(fixture_index()) for cell in cells(population)
    )


@functools.cache
def synthetic_cells() -> tuple[Cell, ...]:
    """Every cell of `SYNTHETIC`."""
    return tuple(cell for population in SYNTHETIC for cell in cells(population))


@functools.cache
def shared_cells() -> tuple[SharedCell, ...]:
    """Every cell of `SHARED`."""
    return tuple(cell for family in SHARED for cell in cells_of_shared(family))


@functools.cache
def sampled_fixture_cells(slots_per_population: int, seed: int = 0) -> tuple[Cell, ...]:
    """The fixture cells a door too slow for the whole grid reads: every population under
    every pair of role keys with no slot bound, every slot binding that holds a cell of a
    moved label, and `slots_per_population` single-value bindings per population drawn
    with `seed`."""
    by_slot: dict[tuple[str, tuple[str, str] | None], list[Cell]] = defaultdict(list)
    for cell in fixture_cells():
        by_slot[cell.population.name, cell.slot].append(cell)
    single: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, slot in by_slot:
        if slot is not None:
            single[name].append(slot)
    sampled = {
        (name, slot)
        for name, slots in single.items()
        for slot in random.Random(seed).sample(sorted(slots), min(slots_per_population, len(slots)))
    }
    return tuple(
        cell
        for key, found in by_slot.items()
        if key[1] is None
        or key in sampled
        or any(cell.expected.label in MOVED_LABELS for cell in found)
        for cell in found
    )
