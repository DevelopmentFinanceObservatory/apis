"""Role declarations filter bound-role results and refresh index-sourced members.
Refresh failures retain frozen rows and set role_declaration_unverified."""

from __future__ import annotations

import pytest

from code_index import CodeIndex, IndexStamp, StatVarRow
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.errors import GraphInfraError
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.retrieve import Materialised, NoDataDraft, materialise
from search.engine.shape import ShapeDraft, SlotKeyDraft
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import (
    FIXTURE_PROVENANCE,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_SV_DECLARING = "TESTFIXTURE/RoleCurrentness/Declaring"
_SV_UNDECLARING = "TESTFIXTURE/RoleCurrentness/Undeclaring"
_SV_INDEX = "TESTFIXTURE/RoleCurrentness/IndexMember"

_ORIGIN = "country/FRA"
_DESTINATION = "country/KEN"

_POP_TYPE = "FinancialFlow"
_MEAS_PROP = "grantEquivalent"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"

_ROLE_TUPLE = (("financingProvider", _ORIGIN), ("financingRecipient", _DESTINATION))
_FINANCING_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_YEARS = [2018, 2019, 2020]


def _facet(facet_id: str = "f1") -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=3,
        dates=["2020"],
        values=[1.0],
        facet_id=facet_id,
    )


def _arcs(*, observation_properties: tuple[str, ...]) -> dict:
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
    }
    if observation_properties:
        arcs["observationProperties"] = {
            "nodes": [{"dcid": prop} for prop in observation_properties]
        }
    return arcs


def _shape(sv_arc_facts: dict[str, dict]) -> ShapeDraft:
    """Build a role-bearing shape from its members' observationProperties arcs."""
    return ShapeDraft(
        shape_id="test_role_currentness_shape",
        label="test flow",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts=sv_arc_facts,
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
    )


def _sdmx(*sv_dcids: str, role_tuple: tuple[tuple[str, str], ...] = _ROLE_TUPLE) -> FakeSdmxClient:
    return FakeSdmxClient(observations={dcid: {role_tuple: [_facet()]} for dcid in sv_dcids})


def _resolve(
    shape: ShapeDraft,
    *,
    graph: FakeGraph,
    sdmx: FakeSdmxClient,
    coverage: CoverageSnapshot | None = None,
):
    return materialise(
        shape,
        [],
        {
            "financingProvider": [_ORIGIN],
            "financingRecipient": [_DESTINATION],
        },
        graph=graph,
        sdmx=sdmx,
        discover_scope=False,
        coverage=coverage if coverage is not None else empty_coverage_snapshot(),
    )


# ---------------------------------------------------------------------------
# Recall-sourced population: roles in hand, no graph read
# ---------------------------------------------------------------------------


def test_a_declaring_member_survives_and_has_data(monkeypatch) -> None:
    """A recall-sourced member declaring both bound roles confirms without graph or SDMX reads."""
    shape = _shape(
        {_SV_DECLARING: _arcs(observation_properties=("financingProvider", "financingRecipient"))}
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DECLARING, "f1", _ORIGIN, _DESTINATION, _YEARS)],
        role_names={_SV_DECLARING: _FINANCING_ROLE_NAMES},
    )
    sdmx = _sdmx(_SV_DECLARING)

    result = _resolve(
        shape,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.sv_dcids == [_SV_DECLARING]
    assert result.role_declaration_unverified is False


def test_a_member_declaring_no_bound_role_is_dropped(monkeypatch) -> None:
    """A member declaring no bound role is excluded before coverage confirmation."""
    shape = _shape(
        {
            _SV_DECLARING: _arcs(
                observation_properties=("financingProvider", "financingRecipient")
            ),
            _SV_UNDECLARING: _arcs(observation_properties=()),
        }
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DECLARING, "f1", _ORIGIN, _DESTINATION, _YEARS),
            (_SV_UNDECLARING, "f2", _ORIGIN, _DESTINATION, _YEARS),
        ],
        role_names={
            _SV_DECLARING: _FINANCING_ROLE_NAMES,
            _SV_UNDECLARING: _FINANCING_ROLE_NAMES,
        },
    )
    sdmx = _sdmx(_SV_DECLARING, _SV_UNDECLARING)

    result = _resolve(
        shape,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DECLARING]


def test_every_member_undeclared_refuses_role_incompatible() -> None:
    """If no member declares a bound role, resolution returns role_incompatible."""
    shape = _shape({_SV_UNDECLARING: _arcs(observation_properties=())})
    sdmx = _sdmx(_SV_UNDECLARING)

    result = _resolve(
        shape, graph=FakeGraph(raise_on_call=True), sdmx=sdmx, coverage=empty_coverage_snapshot()
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_incompatible"
    assert sdmx.series_calls == []


def test_admitted_counts_the_bindings_admitted_population_including_a_dropped_member(
    monkeypatch,
) -> None:
    """The admitted count includes a member removed by the role verdict."""
    shape = _shape(
        {
            _SV_DECLARING: _arcs(
                observation_properties=("financingProvider", "financingRecipient")
            ),
            _SV_UNDECLARING: _arcs(observation_properties=()),
        }
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DECLARING, "f1", _ORIGIN, _DESTINATION, _YEARS)],
        role_names={_SV_DECLARING: _FINANCING_ROLE_NAMES},
    )
    sdmx = _sdmx(_SV_DECLARING, _SV_UNDECLARING)

    result = _resolve(
        shape,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DECLARING]
    assert result.population is not None
    assert result.population.admitted == 2
    assert result.population.confirmed == 1


def test_the_zero_observation_carrier_stamps_the_same_admitted_population() -> None:
    """The carrier's own admitted stamp reads the same expression as the
    data-bearing return: a member dropped for an undeclared role is counted,
    the drop happens before both stamps."""
    shape = _shape(
        {
            _SV_DECLARING: _arcs(
                observation_properties=("financingProvider", "financingRecipient")
            ),
            _SV_UNDECLARING: _arcs(observation_properties=()),
        }
    )
    sdmx = FakeSdmxClient(observations={})

    result = _resolve(
        shape, graph=FakeGraph(raise_on_call=True), sdmx=sdmx, coverage=empty_coverage_snapshot()
    )

    assert isinstance(result, Materialised)
    assert result.available_at_place is False
    assert result.has_data is False
    assert result.population is not None
    assert result.population.admitted == 2
    assert result.population.confirmed == 0


def test_a_member_declaring_the_origin_role_alone_is_dropped() -> None:
    """A member declaring only the origin role fails on the destination key:
    the verdict is per bound key, not per presence of any flow role."""
    shape = _shape(
        {
            _SV_UNDECLARING: _arcs(observation_properties=("financingProvider",)),
        }
    )
    sdmx = _sdmx(_SV_UNDECLARING)

    result = _resolve(
        shape, graph=FakeGraph(raise_on_call=True), sdmx=sdmx, coverage=empty_coverage_snapshot()
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_incompatible"


# ---------------------------------------------------------------------------
# Index-sourced population of any size: its declared roles are refreshed
# ---------------------------------------------------------------------------

# CRS creditor families exceed 5,000 members, and one refresh reads every member.
_WIDE_MEMBERS = 5001


def _wide_index(*, members: int) -> CodeIndex:
    """An index carrying `members` rows that all describe the same five-tuple,
    so shape_member_facts sources the whole set for one shape."""
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=f"{_SV_INDEX}/{i}",
                population_type=_POP_TYPE,
                measured_property=_MEAS_PROP,
                stat_type=_STAT_TYPE,
                measurement_qualifier=_MEAS_QUAL,
                measurement_denominator=None,
                observation_properties=("financingProvider", "financingRecipient"),
                constraint_properties={},
                provenances=(FIXTURE_PROVENANCE,),
            )
            for i in range(members)
        ],
    )


def _wide_shape(members: int) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_role_currentness_wide_shape",
        label="test flow",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={f"{_SV_INDEX}/{i}": _arcs(observation_properties=()) for i in range(members)},
        sv_arc_facts_complete=False,
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
    )


def _wide_nodes(members: int) -> dict:
    """The live graph state every one of `members` index rows reads back
    through, each declaring both bound roles."""
    return {
        f"{_SV_INDEX}/{i}": {
            "label": "index member",
            "type": "StatisticalVariable",
            "arcs": _arcs(observation_properties=("financingProvider", "financingRecipient")),
        }
        for i in range(members)
    }


class _RecordingWideGraph(FakeGraph):
    """A FakeGraph that records every declared-role refresh call."""

    def __init__(self, *, nodes: dict) -> None:
        super().__init__(nodes=nodes, obs={}, detect={}, resolve={})
        self.observation_properties_calls: list[list[str]] = []

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        self.observation_properties_calls.append(list(dcids))
        return super().node_observation_properties_batch(dcids)


def test_a_population_over_five_thousand_members_refreshes_its_roles(monkeypatch) -> None:
    """A family past 5,000 members refreshes in one read and stays verified."""
    members = _WIDE_MEMBERS
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _wide_index(members=members),
    )
    shape = _wide_shape(members)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(f"{_SV_INDEX}/0", "f1", _ORIGIN, _DESTINATION, _YEARS)],
        role_names={f"{_SV_INDEX}/0": _FINANCING_ROLE_NAMES},
    )
    sdmx = _sdmx(*[f"{_SV_INDEX}/{i}" for i in range(members)])
    graph = _RecordingWideGraph(nodes=_wide_nodes(members))

    result = _resolve(
        shape,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.role_declaration_unverified is False
    assert graph.observation_properties_calls == [[f"{_SV_INDEX}/{i}" for i in range(members)]]


# ---------------------------------------------------------------------------
# Index-sourced population: declared roles refreshed against the graph
# ---------------------------------------------------------------------------

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test-hash",
)


def _index(*, observation_properties: tuple[str, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=_SV_INDEX,
                population_type=_POP_TYPE,
                measured_property=_MEAS_PROP,
                stat_type=_STAT_TYPE,
                measurement_qualifier=_MEAS_QUAL,
                measurement_denominator=None,
                observation_properties=observation_properties,
                constraint_properties={},
                provenances=(FIXTURE_PROVENANCE,),
            )
        ],
    )


def _index_with_purpose() -> CodeIndex:
    """The same single-row index, with that row declaring a crsPurpose
    value: a bound crsPurpose then narrows the index branch's own
    index.filter() query to exactly the rows satisfying it."""
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=_SV_INDEX,
                population_type=_POP_TYPE,
                measured_property=_MEAS_PROP,
                stat_type=_STAT_TYPE,
                measurement_qualifier=_MEAS_QUAL,
                measurement_denominator=None,
                observation_properties=("financingProvider", "financingRecipient"),
                constraint_properties={"crsPurpose": "CRSPurposeCode51010"},
                provenances=(FIXTURE_PROVENANCE,),
            )
        ],
    )


def _index_shape() -> ShapeDraft:
    """A shape the patched index carries, whose recall pool is a single
    member: `sv_arc_facts_complete=False` so shape_member_facts takes its
    index branch and reads the frozen rows."""
    return ShapeDraft(
        shape_id="test_role_currentness_index_shape",
        label="test flow",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),),
        sv_arc_facts={_SV_INDEX: _arcs(observation_properties=())},
        sv_arc_facts_complete=False,
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
    )


def _index_graph(*, live_roles: tuple[str, ...]) -> FakeGraph:
    return FakeGraph(
        nodes={
            _SV_INDEX: {
                "label": "index member",
                "type": "StatisticalVariable",
                "arcs": _arcs(observation_properties=live_roles),
            }
        },
        obs={},
        detect={},
        resolve={},
    )


def test_an_index_row_whose_live_roles_moved_on_is_dropped(monkeypatch) -> None:
    """A live declaration with no bound roles drops the frozen index member."""
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _index(observation_properties=("financingProvider", "financingRecipient")),
    )
    shape = _index_shape()
    sdmx = _sdmx(_SV_INDEX)

    result = _resolve(
        shape, graph=_index_graph(live_roles=()), sdmx=sdmx, coverage=empty_coverage_snapshot()
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_incompatible"


def test_an_index_row_whose_live_roles_still_declare_the_bound_role_survives(
    monkeypatch,
) -> None:
    """A refreshed member declaring both bound roles confirms as verified."""
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _index(observation_properties=("financingProvider", "financingRecipient")),
    )
    shape = _index_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_INDEX, "f1", _ORIGIN, _DESTINATION, _YEARS)],
        role_names={_SV_INDEX: _FINANCING_ROLE_NAMES},
    )
    sdmx = _sdmx(_SV_INDEX)

    result = _resolve(
        shape,
        graph=_index_graph(live_roles=("financingProvider", "financingRecipient")),
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.sv_dcids == [_SV_INDEX]
    assert result.role_declaration_unverified is False


def test_a_failed_refresh_keeps_the_row_and_reports_it_unverified(monkeypatch) -> None:
    """A failed refresh keeps the frozen member and marks its roles unverified."""
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _index(observation_properties=("financingProvider", "financingRecipient")),
    )
    shape = _index_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_INDEX, "f1", _ORIGIN, _DESTINATION, _YEARS)],
        role_names={_SV_INDEX: _FINANCING_ROLE_NAMES},
    )
    sdmx = _sdmx(_SV_INDEX)

    class _RefreshRaises(FakeGraph):
        def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
            raise GraphInfraError("simulated refresh read failure")

    result = _resolve(
        shape,
        graph=_RefreshRaises(
            nodes={
                _SV_INDEX: {
                    "label": "index member",
                    "type": "StatisticalVariable",
                    "arcs": _arcs(
                        observation_properties=("financingProvider", "financingRecipient")
                    ),
                }
            },
            obs={},
            detect={},
            resolve={},
        ),
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.role_declaration_unverified is True
    assert result.sv_dcids == [_SV_INDEX]


def test_a_recall_sourced_population_is_never_reported_unverified() -> None:
    """Recall-sourced members require no refresh and remain verified."""
    shape = _shape(
        {_SV_DECLARING: _arcs(observation_properties=("financingProvider", "financingRecipient"))}
    )

    result = _resolve(
        shape,
        graph=FakeGraph(raise_on_call=True),
        sdmx=_sdmx(_SV_DECLARING),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.role_declaration_unverified is False


def test_the_refresh_reads_after_the_bound_value_filter_narrows_the_population(
    monkeypatch,
) -> None:
    """The refresh read names the surviving members, not the shape's whole
    index population: a bound constraint value that excludes a member takes
    that member out of the read too, so the read is bounded by what the
    answer can actually serve."""
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _index_with_purpose(),
    )
    shape = _index_shape()
    sdmx = _sdmx(_SV_INDEX)

    class _RecordingGraph(FakeGraph):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.observation_properties_calls: list[list[str]] = []

        def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
            self.observation_properties_calls.append(list(dcids))
            return super().node_observation_properties_batch(dcids)

    graph = _RecordingGraph(
        nodes={
            _SV_INDEX: {
                "label": "index member",
                "type": "StatisticalVariable",
                "arcs": _arcs(observation_properties=("financingProvider", "financingRecipient")),
            }
        },
        obs={},
        detect={},
        resolve={},
    )

    result = materialise(
        shape,
        [],
        {
            "financingProvider": [_ORIGIN],
            "financingRecipient": [_DESTINATION],
        },
        graph=graph,
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert graph.observation_properties_calls == [[_SV_INDEX]]


@pytest.mark.parametrize(
    "declared",
    [
        # A role this deployment's axis vocabulary does not place at all.
        ("beneficiaryOrganisation",),
        # Two roles sharing the origin axis: the destination key resolves the
        # axis to nothing, so this member fails the verdict even though it
        # names a role the axis table knows.
        ("creditor", "financingProvider"),
    ],
)
def test_the_verdict_reads_the_axis_alias_at_the_mint_doors_rule(declared) -> None:
    """The bound keys are flow axes (`origin`/`destination`), the spelling
    engine.spec_id.entity_role_key mints, so the verdict resolves them
    through the same alias rule the mint door applies: a declared role
    placed on the axis names that key, and two declared roles sharing the
    axis name neither. A member declaring one provider-family name alone
    satisfies the origin key and fails the destination one."""
    shape = _shape({_SV_DECLARING: _arcs(observation_properties=declared)})

    from search.engine.retrieval.bound_values import undeclared_bound_roles
    from search.engine.retrieval.member_facts import ArcsMemberFacts

    facts = {_SV_DECLARING: ArcsMemberFacts(arcs=shape.sv_arc_facts[_SV_DECLARING])}
    undeclared = undeclared_bound_roles(["origin", "destination"], facts)

    assert undeclared == {_SV_DECLARING: declared}


def test_the_verdict_resolves_a_bound_role_name_spelled_as_an_axis_alias() -> None:
    """The origin key resolves to `financingProvider` on a member declaring
    that name, so the member survives; the destination key resolves to
    `financingRecipient` and fails on the same member only when it is
    absent."""
    from search.engine.retrieval.bound_values import undeclared_bound_roles
    from search.engine.retrieval.member_facts import ArcsMemberFacts

    origin_only = _arcs(observation_properties=("creditor",))
    facts = {_SV_DECLARING: ArcsMemberFacts(arcs=origin_only)}

    assert undeclared_bound_roles(["origin"], facts) == {}
    assert undeclared_bound_roles(["destination"], facts) == {_SV_DECLARING: ("creditor",)}


def test_slot_bindings_do_not_change_the_verdict() -> None:
    """A bound constraint value narrows the population, and the role verdict
    applies to whatever survives it. A member that fails the value filter is
    absent before the verdict runs; a member that passes it and declares the
    roles is answered."""
    shape = _shape(
        {
            _SV_DECLARING: {
                **_arcs(observation_properties=("financingProvider", "financingRecipient")),
                "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
                "crsPurpose": {"nodes": [{"dcid": "CRSPurposeCode51010"}]},
            }
        }
    )
    bindings = [
        SlotBindingDraft(
            axis="what",
            property_dcid="crsPurpose",
            kind="value",
            value_dcids=["CRSPurposeCode51010"],
        )
    ]

    result = materialise(
        shape,
        bindings,
        {"financingProvider": [_ORIGIN], "financingRecipient": [_DESTINATION]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=_sdmx(_SV_DECLARING),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DECLARING]


# ---------------------------------------------------------------------------
# Index-sourced population: the probe queries under the refreshed names, not
# the shape's frozen ones
# ---------------------------------------------------------------------------


def test_a_refreshed_members_new_alias_is_queried_under_it_and_survives(monkeypatch) -> None:
    """The frozen shape names the two flow roles financingProvider/
    financingRecipient. This index member's live arc moved both to a
    different alias pair on the same two axes (creditor/debtor). The
    refresh updates its declared roles before the verdict, and the verdict
    passes it (declared_role_for resolves either spelling on its own axis).
    The coverage-backed confirmation binds the resolved origin/destination
    entity dcids directly, so a member whose live roles moved to a new
    alias pair confirms exactly like one whose alias never moved -- no role
    name of any kind ever reaches the confirmation itself, only the entity
    dcids already resolved by the caller."""
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _index(observation_properties=("financingProvider", "financingRecipient")),
    )
    shape = _index_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_INDEX, "f1", _ORIGIN, _DESTINATION, _YEARS)],
        role_names={_SV_INDEX: RoleNames(donor="creditor", recipient="debtor")},
    )
    sdmx = FakeSdmxClient(
        observations={_SV_INDEX: {(("creditor", _ORIGIN), ("debtor", _DESTINATION)): [_facet()]}}
    )

    result = _resolve(
        shape,
        graph=_index_graph(live_roles=("creditor", "debtor")),
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.sv_dcids == [_SV_INDEX]
    assert result.role_declaration_unverified is False


def _two_member_index(*, first: tuple[str, ...], second: tuple[str, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=_SV_INDEX,
                population_type=_POP_TYPE,
                measured_property=_MEAS_PROP,
                stat_type=_STAT_TYPE,
                measurement_qualifier=_MEAS_QUAL,
                measurement_denominator=None,
                observation_properties=first,
                constraint_properties={},
                provenances=(FIXTURE_PROVENANCE,),
            ),
            StatVarRow(
                dcid=_SV_DECLARING,
                population_type=_POP_TYPE,
                measured_property=_MEAS_PROP,
                stat_type=_STAT_TYPE,
                measurement_qualifier=_MEAS_QUAL,
                measurement_denominator=None,
                observation_properties=second,
                constraint_properties={},
                provenances=(FIXTURE_PROVENANCE,),
            ),
        ],
    )


def _two_member_index_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_role_currentness_two_member_shape",
        label="test flow",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SV_INDEX: _arcs(observation_properties=()),
            _SV_DECLARING: _arcs(observation_properties=()),
        },
        sv_arc_facts_complete=False,
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
    )


def test_refreshed_members_disagreeing_on_role_names_report_role_signature_incompatible(
    monkeypatch,
) -> None:
    """Both members individually survive the declared-role verdict (each
    declares a role on both the origin and destination axis), but they
    disagree with each other on the exact alias: one member's roles moved
    to creditor/debtor, the other's stayed on financingProvider/
    financingRecipient. No single signature names the whole group's roles,
    so this reports role_signature_incompatible -- the same outcome a
    wire-level signature mismatch already reports -- rather than querying
    the group under one survivor's names and silently dropping the other,
    or querying under the frozen shape names neither current declaration
    matches."""
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _two_member_index(
            first=("financingProvider", "financingRecipient"),
            second=("financingProvider", "financingRecipient"),
        ),
    )
    shape = _two_member_index_shape()
    sdmx = FakeSdmxClient(
        observations={
            _SV_INDEX: {(("creditor", _ORIGIN), ("debtor", _DESTINATION)): [_facet()]},
            _SV_DECLARING: {_ROLE_TUPLE: [_facet()]},
        }
    )
    graph = FakeGraph(
        nodes={
            _SV_INDEX: {
                "label": "moved member",
                "type": "StatisticalVariable",
                "arcs": _arcs(observation_properties=("creditor", "debtor")),
            },
            _SV_DECLARING: {
                "label": "unmoved member",
                "type": "StatisticalVariable",
                "arcs": _arcs(observation_properties=("financingProvider", "financingRecipient")),
            },
        },
        obs={},
        detect={},
        resolve={},
    )

    result = _resolve(shape, graph=graph, sdmx=sdmx, coverage=empty_coverage_snapshot())

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_signature_incompatible"
