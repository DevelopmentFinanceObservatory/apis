"""Search states the members of a role-bearing answer as the rule in
`tests.statvar_members.role_grid` elects them and as the members door serves them.

Three statements are read per cell of the role grid, each asked with the cell's bound axes
and the open axes the request asks for:

- `_admit_population`: the members the answer reads and the role names on each flow axis;
- `_served_members`: the members the answer lists as served;
- `stated_members`: the served count and family digest stated for the identity minted
  from the answer, which carries the open roles the rule admits.

Each is compared with the rule and with what the members door serves for the minted
identity (`tests.datasets_api.test_role_grid_members`). The fixture code index decides the
fixture cells, a synthetic index swapped in for the process index decides one population,
and the recall path reads a synthetic population from the shape's recalled members, with
the member walk Search runs for `stated_members`.

Search binds a place on a flow axis of a role-bearing shape, so the cells that bind
`observationAbout` are left out, as are the populations that hold a subject-only row. An
index population of more than 1,000 rows is asked naming its dataset only. A
bound value the fixture index widens to its descendants is left out too: the grid's cell
holds the rows of the value alone.

Each comparison is made twice: with the identity that carries the open roles the rule
admits, and with the one built from the open roles `_admit_population` itself records.
The families of `role_grid.SHARED`, each published by two or three datasets, are asked of
an index holding them, naming each dataset and none. One population is asked through the member
walk and `graph_confirm_resolve`: its elected member declares a role on an open axis the
election set aside.

The last tests go through `resolve_variable`. One asks OECD DAC1's net disbursements, with
the donor bound and with no place and the recipient open, and compares the counts Search
states with the mint's for the identity it mints. The other asks a family two datasets
publish with no place and both axes open, and compares the members.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, cast

import pytest

from code_index.types import StatVarRow
from datasets_api.adapter import resolve_definition_members
from dfo_platform.roles import ROLE_AXES
from search.engine.bind import SlotBindingDraft
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import _served_members
from search.engine.retrieval.bound_values import (
    RoleClassWalks,
    WalkedMembers,
    bound_values_from_bindings,
    elected_role_class_shape,
    shape_query,
    source_binding,
    stated_members,
)
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.retrieval.role_resolve import (
    _admit_population,
    _AdmittedPopulation,
    graph_confirm_resolve,
)
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, SlotKeyDraft, role_names_by_axis
from search.models import BaseIdentity, RoleAxis
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from statvar_members import (
    MemberEnumerator,
    MemberRoleDisagreement,
    enumerate_members,
    member_digest,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._member_bounds import mint_bounds
from tests.datasets_api.test_role_grid_members import (
    FIXTURE,
    NOT_REPRODUCIBLE,
    SYNTHETIC_INDEX,
    TIED,
    WALK,
    expected_served,
    minted_identity,
    population_graph,
    served,
    shared_identity,
    shared_served,
)
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.graph_release.support import serving_index
from tests.statvar_members.role_grid import (
    ASKED_AGAIN,
    DESTINATION,
    LABELS,
    NESTED,
    ONE_ROLE_SET,
    OPEN_HONOURED,
    OPEN_UNPLACED,
    OPEN_UNTWINNED,
    ORIGIN,
    SHARED,
    SUBJECT,
    SYNTHETIC_DATASET,
    SYNTHETIC_FIVE_TUPLE,
    TIE_SET_ASIDE,
    Cell,
    Population,
    Row,
    SharedCell,
    SharedFamily,
    cells,
    fixture_index,
    places,
    row,
    sampled_fixture_cells,
    shared_cells,
    shared_index,
    synthetic_cells,
    synthetic_index,
)

if TYPE_CHECKING:
    from materialise.graph import GraphReader
    from search.engine.clients.graph import EngineGraphClient

_SAMPLED_SLOTS = 2
# An index population of more rows than this is asked naming its dataset only.
_UNNAMED_ROWS = 1_000
_RECALL = WALK
_PATH_LABELS = {
    FIXTURE: (ONE_ROLE_SET, NESTED, OPEN_HONOURED, OPEN_UNPLACED, OPEN_UNTWINNED),
    SYNTHETIC_INDEX: LABELS,
    _RECALL: LABELS,
}

_DOOR: dict[tuple[str, int, bool], tuple] = {}
"""What the members door serves per path, cell of the cached grid, and dataset naming."""


def _role_bearing(population: Population) -> bool:
    return all(row.roles != (SUBJECT,) for row in population.rows)


@functools.cache
def _cells(path: str) -> tuple[Cell, ...]:
    index = fixture_index()
    found = sampled_fixture_cells(_SAMPLED_SLOTS) if path == FIXTURE else synthetic_cells()
    return tuple(
        cell
        for cell in found
        if SUBJECT not in cell.bound
        and _role_bearing(cell.population)
        and (cell.slot is None or index.expansion(cell.slot[1]) == (cell.slot[1],))
    )


@contextlib.contextmanager
def _process_index(path: str, population: Population) -> Iterator[None]:
    """The process code index Search reads: a synthetic one holding `population` on the
    synthetic index path, the fixture index otherwise."""
    if path != SYNTHETIC_INDEX:
        yield
        return
    with serving_index(synthetic_index(population.name)):
        yield


def _nodes(*dcids: str | None) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids if dcid is not None]}


@functools.cache
def _shape(path: str, name: str) -> ShapeDraft:
    """The shape of the population named `name`: a slot key per constraint property and
    the roles its rows declare. It recalls every row on the recall path and none on the
    index paths, where the index's family answers."""
    population = next(cell.population for cell in _cells(path) if cell.population.name == name)
    return _population_shape(population, population.rows if path == _RECALL else ())


def _population_shape(population: Population, recalled: Sequence[Row]) -> ShapeDraft:
    """The shape of `population` that recalls the role-bearing rows `recalled`."""
    five_tuple = population.five_tuple
    roles = tuple(dict.fromkeys(role for row in population.rows for role in row.roles))
    recall = {
        row.dcid: {
            "populationType": _nodes(five_tuple.population_type),
            "measuredProperty": _nodes(five_tuple.measured_property),
            "statType": _nodes(five_tuple.stat_type),
            "measurementQualifier": _nodes(five_tuple.measurement_qualifier),
            "constraintProperties": _nodes(*row.constraints),
            "observationProperties": _nodes(*row.roles),
            **{prop: _nodes(value) for prop, value in row.constraints.items()},
        }
        for row in recalled
    }
    return ShapeDraft(
        shape_id=f"role-grid/{population.name}",
        label=population.name,
        pop_type_dcid=five_tuple.population_type or "",
        meas_prop_dcid=five_tuple.measured_property or "",
        stat_type_dcid=five_tuple.stat_type or "",
        meas_qual_dcid=five_tuple.measurement_qualifier,
        meas_denom_dcid=five_tuple.measurement_denominator,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop)
            for prop in sorted(population.properties)
        ),
        sv_arc_facts=recall,
        sv_arc_facts_complete=False,
        declared_roles=roles,
        optional_roles=frozenset(
            role for role in roles if any(role not in row.roles for row in population.rows)
        ),
    )


def _bindings(cell: Cell, *, named: bool) -> list[SlotBindingDraft]:
    bindings = []
    if cell.slot is not None:
        prop, value = cell.slot
        bindings.append(
            SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
        )
    if named:
        bindings.append(source_binding(cell.population.dataset))
    return bindings


class _RoleGraph:
    """Serves the declared-role refresh of an index-sourced population from `rows`."""

    def __init__(self, rows: Iterable[Row]) -> None:
        self._roles = {row.dcid: () if row.roles == (SUBJECT,) else row.roles for row in rows}

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        return {dcid: self._roles[dcid] for dcid in dcids}


@dataclasses.dataclass(frozen=True)
class _Asked:
    """One cell asked of Search on one path, naming the dataset or not."""

    cell: Cell
    path: str
    named: bool

    def __str__(self) -> str:
        return f"{'named' if self.named else 'unnamed'}: {self.cell}"

    @property
    def shape(self) -> ShapeDraft:
        return _shape(self.path, self.cell.population.name)

    def query(self):
        cell = self.cell
        return shape_query(
            self.shape,
            _bindings(cell, named=self.named),
            cell.bound,
            open_role_keys=cell.open_keys,
        )

    def population(self) -> NoDataDraft | _AdmittedPopulation:
        """What `_admit_population` returns for the cell."""
        cell = self.cell
        bindings = _bindings(cell, named=self.named)
        with _process_index(self.path, cell.population):
            return _admit_population(
                self.shape,
                query=self.query(),
                bound_values=bound_values_from_bindings(bindings),
                names=role_names_by_axis(self.shape),
                subject_dcids=["country/KEN"] if DESTINATION in cell.bound else [],
                origin_dcids=["country/FRA"] if ORIGIN in cell.bound else [],
                open_role_keys=sorted(cell.open_keys),
                graph=cast("EngineGraphClient", _RoleGraph(cell.population.rows)),
                deadline=None,
            )

    def admitted(self):
        """`_admit_population`'s survivors and role names by axis, or its no-data
        reason."""
        outcome = self.population()
        if isinstance(outcome, NoDataDraft):
            return (outcome.reason,)
        return (frozenset(outcome.surviving_svs), tuple(sorted(outcome.names.items())))

    def served(self) -> frozenset[str]:
        with _process_index(self.path, self.cell.population):
            return frozenset(_served_members(self.shape, self.query()))

    def stated(self, open_roles: Collection[str] | None = None) -> tuple[int, str | None] | None:
        """`stated_members` for the identity minted from the answer (`minted_identity`):
        the served count and the family digest, or `None` when Search states nothing. On
        the recall path it reads the member walk Search runs for the asked query."""
        cell = self.cell
        identity = BaseIdentity.model_validate(
            minted_identity(cell, named=self.named, open_roles=open_roles)
        )
        with _process_index(self.path, cell.population):
            walked = None
            if self.path == _RECALL:
                query = self.query()
                try:
                    group = enumerate_members(
                        five_tuple=query.five_tuple,
                        slots=query.slots,
                        graph=population_graph(cell),
                        role_keys=query.role_keys,
                        open_role_keys=query.open_role_keys,
                    )
                except MemberRoleDisagreement:
                    group = None
                walked = WalkedMembers.of(query, group) if group is not None else None
            stated = stated_members(identity, walked)
        return None if stated is None else (stated.served_count, stated.digest)

    def door(self):
        """What the members door serves for the minted identity."""
        key = (self.path, id(self.cell), self.named)
        if key not in _DOOR:
            _DOOR[key] = served(self.cell, self.path, named=self.named)
        return _DOOR[key]


def _asked(path: str, label: str) -> list[_Asked]:
    found = [
        _Asked(cell, path, named)
        for cell in _cells(path)
        if cell.expected.label == label
        for named in (
            (True, False)
            if path != _RECALL and len(cell.population.rows) <= _UNNAMED_ROWS
            else (True,)
        )
    ]
    assert found, f"the {path} grid holds no cell labelled {label!r}"
    return found


_CASES = [
    pytest.param(path, label, id=f"{path}: {label}")
    for path, labels in _PATH_LABELS.items()
    for label in labels
]


def _report(differences: list[str], checked: int) -> str:
    return f"{len(differences)} of {checked} checks differ; first: {differences[:1]}"


def _elected_roles(cell: Cell) -> tuple[str, ...]:
    return next(row.roles for row in cell.rows if row.dcid in cell.expected.members)


def _expected_admitted(cell: Cell):
    """What the rule says `_admit_population` returns. Search refuses `role_incompatible`
    when no row places a bound axis, where the rule reads every row, and
    `role_signature_incompatible` on a tie. Else the rule's members and the roles of the
    elected role set by flow axis."""
    if cell.bound and not any(places(row.roles, cell.bound) for row in cell.rows):
        return ("role_incompatible",)
    if cell.expected.tied:
        return ("role_signature_incompatible",)
    names = sorted((ROLE_AXES[role], role) for role in _elected_roles(cell))
    return (cell.expected.members, tuple(names))


@pytest.mark.parametrize(("path", "label"), _CASES)
def test_search_admits_the_members_the_rule_elects_and_the_door_serves(path: str, label: str):
    """`_admit_population` reads the rule's members under the role names of the elected
    role set, and refuses where the rule ties or no row places a bound axis. Where the
    members door serves the minted identity, it serves those members."""
    found = _asked(path, label)
    differences = []

    for asked in found:
        admitted = asked.admitted()
        if admitted != _expected_admitted(asked.cell):
            differences.append(f"rule: {asked}")
        door = asked.door()
        if door not in (TIED, NOT_REPRODUCIBLE) and admitted[0] != door[0]:
            differences.append(f"members door: {asked}")

    assert not differences, _report(differences, 2 * len(found))


@pytest.mark.parametrize(("path", "label"), _CASES)
def test_search_lists_as_served_the_members_the_rule_elects_and_the_door_serves(
    path: str, label: str
):
    """Where the rule serves the minted identity, `_served_members` lists the rule's
    members, and where the members door serves it, the door's."""
    found = _asked(path, label)
    differences = []
    checked = 0

    for asked in found:
        listed = asked.served()
        expected = expected_served(asked.cell)
        if expected not in (TIED, NOT_REPRODUCIBLE):
            checked += 1
            if listed != expected[0]:
                differences.append(f"rule: {asked}")
        door = asked.door()
        if door not in (TIED, NOT_REPRODUCIBLE):
            checked += 1
            if listed != door[0]:
                differences.append(f"members door: {asked}")

    assert checked or all(expected_served(asked.cell) == TIED for asked in found)
    assert not differences, _report(differences, checked)


@pytest.mark.parametrize(("path", "label"), _CASES)
def test_search_states_the_members_the_rule_elects_and_the_door_serves(path: str, label: str):
    """`stated_members` states the count of the rule's members and the digest of its
    family, and nothing where the rule ties. It states the count and digest the members
    door serves, and nothing where the door refuses a tie. An identity that names a role
    no member places is left out: the door refuses it after members resolve."""
    found = _asked(path, label)
    differences = []
    checked = 0

    for asked in found:
        stated = asked.stated()
        expected = expected_served(asked.cell)
        if expected != NOT_REPRODUCIBLE:
            checked += 1
            ruled = (
                None
                if expected == TIED
                else (len(expected[0]), member_digest(asked.cell.expected.family))
            )
            if stated != ruled:
                differences.append(f"rule: {asked}")
        door = asked.door()
        if door != NOT_REPRODUCIBLE:
            checked += 1
            if stated != (None if door == TIED else (len(door[0]), door[2])):
                differences.append(f"members door: {asked}")

    assert checked
    assert not differences, _report(differences, checked)


@pytest.mark.parametrize(("path", "label"), _CASES)
def test_the_identity_search_records_resolves_at_the_mint_to_the_members_it_admits(
    path: str, label: str
):
    """The identity is built from the open roles `_admit_population` itself records, the
    asked axes its survivors declare, where the tests above take them from the rule. The
    members door serves that identity with the members Search admits, and `stated_members`
    states the door's count and digest for it."""
    found = _asked(path, label)
    differences = []
    checked = 0

    for asked in found:
        admitted = asked.population()
        if isinstance(admitted, NoDataDraft):
            continue
        checked += 1
        door = served(asked.cell, path, named=asked.named, open_roles=admitted.open_axes)
        if door in (TIED, NOT_REPRODUCIBLE):
            differences.append(f"members door refuses {door[0]}: {asked}")
        elif frozenset(admitted.surviving_svs) != door[0]:
            differences.append(f"members: {asked}")
        elif asked.stated(admitted.open_axes) != (len(door[0]), door[2]):
            differences.append(f"stated: {asked}")

    assert checked or all(asked.cell.expected.tied for asked in found)
    assert not differences, _report(differences, checked)


# ---------------------------------------------------------------------------
# A family several datasets publish
# ---------------------------------------------------------------------------


def _one_role_per_axis(family: SharedFamily) -> bool:
    roles = {
        role for population in family.populations for row in population.rows for role in row.roles
    }
    axes = [ROLE_AXES[role] for role in roles if role in ROLE_AXES]
    return len(axes) == len(set(axes))


@functools.cache
def _shared_shape(name: str) -> ShapeDraft:
    """The shape of the family of `SHARED` named `name`, recalling no row: the index
    holding the family answers."""
    family = next(found for found in SHARED if found.name == name)
    rows = {row.dcid: row for population in family.populations for row in population.rows}
    merged = Population(
        name=name,
        five_tuple=family.five_tuple,
        dataset="",
        rows=tuple(rows.values()),
        readings=family.readings,
    )
    return _population_shape(merged, ())


def _shared_asked() -> list[tuple[SharedCell, str | None]]:
    """Each cell of a shared family Search can ask, naming no dataset and each dataset
    that holds a row: the cells that bind no subject, of the families whose rows are all
    role-bearing and name one role on each flow axis, as one shape's members do."""
    return [
        (cell, dataset)
        for cell in shared_cells()
        if SUBJECT not in cell.bound
        and all(_role_bearing(population) for population in cell.family.populations)
        and _one_role_per_axis(cell.family)
        for dataset in (None, *cell.family.datasets)
        if dataset is None or cell.rows[dataset]
    ]


def _shared_admitted(cell: SharedCell, dataset: str | None) -> NoDataDraft | _AdmittedPopulation:
    shape = _shared_shape(cell.family.name)
    bindings = []
    if cell.slot is not None:
        prop, value = cell.slot
        bindings.append(
            SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
        )
    if dataset is not None:
        bindings.append(source_binding(dataset))
    return _admit_population(
        shape,
        query=shape_query(shape, bindings, cell.bound, open_role_keys=cell.open_keys),
        bound_values=bound_values_from_bindings(bindings),
        names=role_names_by_axis(shape),
        subject_dcids=["country/KEN"] if DESTINATION in cell.bound else [],
        origin_dcids=["country/FRA"] if ORIGIN in cell.bound else [],
        open_role_keys=sorted(cell.open_keys),
        graph=cast(
            "EngineGraphClient",
            _RoleGraph(row for population in cell.family.populations for row in population.rows),
        ),
        deadline=None,
    )


def test_search_admits_from_a_shared_family_what_the_mint_serves_for_its_identity():
    """Over a family two or three datasets publish, `_admit_population` reads the members
    the rule elects from the dataset it elects. The identity built from the open roles
    Search records is served by the members door with those members, from one dataset,
    and `stated_members` states the door's count and digest for it. Among three datasets
    with both axes open, the elected member declares a role on an axis the election set
    aside, and the identity names only the axes that joined."""
    found = _shared_asked()
    assert {cell.expected.case for cell, _dataset in found} >= {ASKED_AGAIN, TIE_SET_ASIDE}
    differences = []
    served_cells = 0

    for cell, dataset in found:
        elected = cell.expected.dataset if dataset is None else dataset
        election = cell.expected.election if dataset is None else cell.named(dataset)
        assert election is not None
        refusal = None
        if cell.bound and not any(places(row.roles, cell.bound) for row in cell.rows[elected]):
            refusal = "role_incompatible"
        elif election.tied:
            refusal = "role_signature_incompatible"
        with serving_index(shared_index(cell.family.name)):
            admitted = _shared_admitted(cell, dataset)
            if isinstance(admitted, NoDataDraft):
                if admitted.reason != refusal:
                    differences.append(f"{admitted.reason}, {dataset}: {cell}")
                continue
            served_cells += 1
            identity = shared_identity(cell, dataset=dataset, open_roles=admitted.open_axes)
            stated = stated_members(BaseIdentity.model_validate(identity), None)
        survivors = frozenset(admitted.surviving_svs)
        door = shared_served(cell, dataset=dataset, open_roles=admitted.open_axes)
        if refusal is not None or survivors != election.members:
            differences.append(f"rule, {dataset}: {cell}")
        elif door in (TIED, NOT_REPRODUCIBLE) or (survivors, elected) != (door[0], door[3]):
            differences.append(f"members door, {dataset}: {cell}")
        elif stated is None or (stated.served_count, stated.digest) != (len(door[0]), door[2]):
            differences.append(f"stated, {dataset}: {cell}")

    assert served_cells
    assert not differences, _report(differences, len(found))


# ---------------------------------------------------------------------------
# A member elected under one open key that declares a role on the other
# ---------------------------------------------------------------------------

_BOTH_AXES = frozenset({ORIGIN, DESTINATION})
_PROVIDER = ("financingProvider",)
_BY_BOTH = row("flow_by_creditor_and_recipient", {}, ("creditor", "financingRecipient"))
_BY_CREDITOR = row("service_by_creditor", {"flowType": "Service"}, ("creditor",))
_BY_RECIPIENT = row(
    "arrears_by_recipient",
    {"flowType": "Arrears", "guarantee": "Public"},
    ("financingRecipient",),
)
_OPEN_KEY_SET_ASIDE = Population(
    name="a two-role row elected under one open key",
    five_tuple=SYNTHETIC_FIVE_TUPLE,
    dataset=SYNTHETIC_DATASET,
    rows=(
        _BY_BOTH,
        _BY_CREDITOR,
        row("service_total", {"flowType": "Service"}, ()),
        _BY_RECIPIENT,
    ),
    readings={},
)
"""With nothing bound the subject-only total answers, and its by-creditor twin places the
origin, so the origin joins and the destination, which only rows that are twins of no
total place, is set aside. Under the origin the unconstrained two-role row dominates the
twin and is elected, and it declares a recipient too."""


def test_a_member_elected_under_one_open_key_states_the_mints_count_for_both_axes():
    """Both axes are asked with no place. The walk elects the two-role row under the
    origin alone, and Search records both axes open, since the row declares a role on
    each. The mint serves that identity with the same member, and `stated_members` states
    the mint's count and digest for it from Search's own walk. `derive_shapes` keeps the
    subject-only total in a shape of its own, so the role-bearing shape recalls the other
    rows."""
    population = _OPEN_KEY_SET_ASIDE
    cell = next(
        cell
        for cell in cells(population)
        if cell.slot is None and not cell.bound and cell.open_keys == _BOTH_AXES
    )
    assert cell.expected.keys == {ORIGIN}
    assert cell.expected.members == {_BY_BOTH.dcid}
    shape = _population_shape(population, (_BY_BOTH, _BY_CREDITOR, _BY_RECIPIENT))
    bindings = [source_binding(population.dataset)]
    query = shape_query(shape, bindings, (), open_role_keys=_BOTH_AXES)

    unread = FakeGraph(nodes={}, obs={}, detect={}, resolve={}, raise_on_call=True)

    resolution = elected_role_class_shape(
        shape,
        query,
        walks=RoleClassWalks(cast("GraphReader", population_graph(cell))),
        arcs_graph=unread,
    )
    answer = graph_confirm_resolve(
        shape=resolution.value,
        bindings=bindings,
        subject_dcids=[],
        graph=unread,
        sdmx=FakeSdmxClient(observations={}),
        discover_scope=True,
        open_axes=cast("frozenset[RoleAxis]", _BOTH_AXES),
        coverage_snapshot=FakeCoverageService(
            build_coverage_index(
                [(_BY_BOTH.dcid, "f1", "country/FRA", "country/KEN", [2020])],
                role_names={
                    _BY_BOTH.dcid: RoleNames(donor="creditor", recipient="financingRecipient")
                },
            )
        ).snapshot(),
    )

    assert resolution.authority.settled
    assert isinstance(answer, Materialised), answer
    assert answer.sv_dcids == [_BY_BOTH.dcid]
    assert answer.open_axes == _BOTH_AXES
    door = served(cell, WALK, named=True, open_roles=answer.open_axes)
    assert door[0] == {_BY_BOTH.dcid}
    identity = BaseIdentity.model_validate(
        minted_identity(cell, named=True, open_roles=answer.open_axes)
    )
    stated = stated_members(identity, resolution.walked)
    assert stated is not None
    assert (stated.served_count, stated.digest) == (len(door[0]), door[2])


# ---------------------------------------------------------------------------
# OECD DAC1's net disbursements, asked end to end
# ---------------------------------------------------------------------------

_DAC1 = "provenance/OECDDAC1"
_DAC1_NET_TOTAL = "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1010_Nominal"
# One of the two DAC1 rows that hold a recipient beside the donor.
_DAC1_NET_FLOW = "Amount_FinancialFlow_Net_Disbursement_DAC1AidType8863_Nominal"
_DAC1_NET_DONOR_ONLY = 128


class _ElectFirst:
    """Elects the first shape on the menu with no bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(0), None


def _index_row_node(row: StatVarRow) -> dict:
    arcs = {
        "typeOf": _nodes("StatisticalVariable"),
        "populationType": _nodes(row.population_type),
        "measuredProperty": _nodes(row.measured_property),
        "statType": _nodes(row.stat_type),
        "measurementQualifier": _nodes(row.measurement_qualifier),
        "constraintProperties": _nodes(*row.constraint_properties),
        "observationProperties": _nodes(*row.observation_properties or ()),
        **{prop: _nodes(value) for prop, value in row.constraint_properties.items()},
    }
    return {"label": row.dcid, "arcs": arcs}


@pytest.mark.parametrize(
    ("query", "entities", "open_axes"),
    [
        pytest.param("aid from France", {"France": "origin"}, frozenset(), id="donor-bound"),
        pytest.param("aid", {}, frozenset({"destination"}), id="no-place-recipient-open"),
    ],
)
def test_a_donor_only_dac1_question_answers_the_donor_totals_the_mint_serves(
    query: str, entities: dict[str, RoleAxis], open_axes: frozenset[RoleAxis]
):
    """Recall holds one of DAC1's two net flows to a named recipient. A question naming
    the donor alone, and one naming no place that asks for every recipient, both read the
    donor-only net disbursements of every aid type: the identity carries no open role,
    and the served count and family count Search states are the mint's for that
    identity."""
    graph = FakeGraph(
        nodes={
            "country/FRA": {"label": "France", "type": "Country"},
            **{
                row.dcid: _index_row_node(row)
                for row in fixture_index().statvars
                if _DAC1 in row.provenances
            },
        },
        detect={query: {"svs": [_DAC1_NET_FLOW], "cosine_scores": [0.9]}},
        resolve={"France": "country/FRA"},
        obs={},
    )
    donor_only = RoleNames(donor="financingProvider", recipient=None)
    coverage = FakeCoverageService(
        build_coverage_index(
            [(_DAC1_NET_TOTAL, "f1", "country/FRA", ABSENT_ENTITY, [2020, 2021])],
            role_names={_DAC1_NET_TOTAL: donor_only},
        )
    )

    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=list(entities),
            entity_roles={
                name: [DoorRole(axis=axis, span=name)] for name, axis in entities.items()
            },
            date_request=None,
            detect_query=query,
            role_query=query,
            open_axes=open_axes,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirst(),
            sdmx=FakeSdmxClient(observations={}),
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    identity = spec.definition.identity
    assert identity.entities == {axis: ["country/FRA"] for axis in entities.values()}
    assert identity.open_roles is None
    assert [item.ref.ref for item in spec.resolved.members] == [_DAC1_NET_TOTAL]
    minted = resolve_definition_members(
        identity.model_dump(mode="json", exclude_none=True),
        spec_id=spec.spec_id,
        graph=cast("GraphReader", FakeMintGraphReader(arcs={})),
        enumerator=MemberEnumerator(),
        code_index=fixture_index(),
        bounds=mint_bounds(),
    )
    family = minted.shapes[0].family
    assert family is not None
    assert {shape.declared_roles for shape in minted.shapes} == {("financingProvider",)}
    assert spec.resolved.stat_vars.served_count == len(minted.shapes) == _DAC1_NET_DONOR_ONLY
    assert spec.resolved.shape.member_count == family.count == _DAC1_NET_DONOR_ONLY


def test_a_question_opening_both_axes_over_two_datasets_mints_what_the_mint_serves():
    """A question that names no place and asks for every donor and every recipient reaches
    the role resolver with both axes open. The first dataset holds a provider row and a
    recipient row, so neither axis joins on its own rows, and the second holds a provider
    row. Search answers under the origin, and the identity it mints carries that one open
    role: the mint elects the same dataset and member for it, and Search states the
    mint's counts."""
    family = next(
        found
        for found in SHARED
        if found.name == "a provider row and a recipient row beside a provider row"
    )
    index = shared_index(family.name)
    recalled = family.populations[1].rows[0].dcid
    graph = FakeGraph(
        nodes={row.dcid: _index_row_node(row) for row in index.statvars},
        detect={"aid": {"svs": [recalled], "cosine_scores": [0.9]}},
        resolve={},
        obs={},
    )
    providers = [row for row in index.statvars if row.observation_properties == _PROVIDER]

    with serving_index(index):
        # Each provider row's facet takes the dataset the index records the row under.
        coverage = FakeCoverageService(
            build_coverage_index(
                [
                    (row.dcid, f"facet/{row.dcid}", "country/FRA", ABSENT_ENTITY, [2020, 2021])
                    for row in providers
                ],
                role_names={
                    row.dcid: RoleNames(donor="financingProvider", recipient=None)
                    for row in providers
                },
            )
        )
        result = asyncio.run(
            resolve_variable(
                "aid",
                entities=[],
                entity_roles={},
                date_request=None,
                detect_query="aid",
                role_query="aid",
                open_axes=cast("frozenset[RoleAxis]", _BOTH_AXES),
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_ElectFirst(),
                sdmx=FakeSdmxClient(observations={}),
                base_timing={},
                coverage=coverage.snapshot(),
            )
        )

    assert result.status == "definite", result
    spec = result.specs[0]
    identity = spec.definition.identity
    assert identity.entities == {}
    assert identity.open_roles == [ORIGIN]
    minted = resolve_definition_members(
        identity.model_dump(mode="json", exclude_none=True),
        spec_id=spec.spec_id,
        graph=cast("GraphReader", FakeMintGraphReader(arcs={})),
        enumerator=MemberEnumerator(),
        code_index=index,
        bounds=mint_bounds(),
    )
    minted_family = minted.shapes[0].family
    assert minted_family is not None
    assert [item.ref.ref for item in spec.resolved.members] == [
        shape.stat_var.removeprefix("dcid:") for shape in minted.shapes
    ]
    assert spec.resolved.stat_vars.served_count == len(minted.shapes)
    assert spec.resolved.shape.member_count == minted_family.count
