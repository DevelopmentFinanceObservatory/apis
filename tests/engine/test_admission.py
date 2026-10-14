"""Test shape admission against entity types and declared graph roles."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import cast

import pytest

from entity_resolution import Admission
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.interpret import Recall
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult, resolve_variable
from search.engine.region.phase_intake import _prune_shapes_by_admission
from search.engine.region.place_set import PlaceNode, PlaceSet, PlaceSetKind
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY, QuantityReadOnlyLLM
from tests.fixtures import FakeGraph, make_resolver


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


@pytest.mark.parametrize(
    "entities",
    [["African Development Bank"], ["Kenya", "African Development Bank"]],
    ids=["alone", "beside-an-admitted-country"],
)
def test_population_with_a_bank_entity_refuses_entity_role_not_admitted(entities):
    """Subject-only population shapes reject the bank's Organization entity. The refusal
    names the bank alone, since a country named beside it is admitted."""
    graph = FakeGraph()
    result: RegionResult = asyncio.run(
        resolve_variable(
            "population",
            entities=entities,
            date_request=None,
            detect_query="population",
            role_query=f"population {' '.join(entities)}",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "entity_role_not_admitted"
    assert [e.text for e in result.role_not_admitted_entities] == ["African Development Bank"]
    assert [e.dcid for e in result.role_not_admitted_entities] == ["AfricanDevelopmentBank"]
    assert [e.graph_type for e in result.role_not_admitted_entities] == ["Organization"]


def test_ambiguous_entity_refuses_entity_ambiguous_before_admission_ever_runs():
    """Congo resolves to country/COD and country/COG.

    Ambiguity precedes admission.
    """
    graph = FakeGraph()
    result: RegionResult = asyncio.run(
        resolve_variable(
            "zzz nonexistent indicator zzz",
            entities=["Congo"],
            date_request=None,
            detect_query="zzz nonexistent indicator zzz",
            role_query="zzz nonexistent indicator zzz Congo",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "entity_ambiguous"
    assert len(result.ambiguous_entities) == 1
    assert result.ambiguous_entities[0].text == "Congo"
    assert sorted(c.dcid for c in result.ambiguous_entities[0].candidates) == [
        "country/COD",
        "country/COG",
    ]


def test_unresolved_entity_alongside_not_admitted_entity_refuses_entity_not_resolved():
    """An unresolved Atlantis entity takes precedence over the bank's role rejection."""
    graph = FakeGraph()
    result: RegionResult = asyncio.run(
        resolve_variable(
            "population",
            entities=["African Development Bank", "Republic of Atlantis"],
            date_request=None,
            detect_query="population",
            role_query="population African Development Bank Republic of Atlantis",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "entity_not_resolved"
    assert result.role_not_admitted_entities == ()


def test_ambiguous_entity_alongside_not_admitted_entity_refuses_entity_ambiguous():
    """Congo's ambiguity takes precedence over the bank's role rejection."""
    graph = FakeGraph()
    result: RegionResult = asyncio.run(
        resolve_variable(
            "population",
            entities=["African Development Bank", "Congo"],
            date_request=None,
            detect_query="population",
            role_query="population African Development Bank Congo",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "entity_ambiguous"
    assert len(result.ambiguous_entities) == 1
    assert result.ambiguous_entities[0].text == "Congo"
    assert sorted(c.dcid for c in result.ambiguous_entities[0].candidates) == [
        "country/COD",
        "country/COG",
    ]
    assert result.role_not_admitted_entities == ()


_CRS_SV = "TESTFIXTURE/Admission/FinancingFlow"
_CRS_ARCS = {
    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "observationProperties": {
        "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
    },
}


def test_crs_role_bearing_shape_admits_a_bank_on_the_provider_axis():
    """The Organization range on financingProvider admits the bank."""
    graph = FakeGraph(
        nodes={_CRS_SV: {"label": "Test financing flow", "arcs": _CRS_ARCS}},
        detect={"aid": {"svs": [_CRS_SV], "cosine_scores": [0.9]}},
    )
    result: RegionResult = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Asian Development Bank"],
            date_request=None,
            detect_query="aid",
            role_query="aid from Asian Development Bank",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.no_data_reason != "entity_role_not_admitted"


def test_income_in_california_reaches_materialise_not_a_resolution_time_refusal():
    """California's State-to-Place chain admits it.

    The empty graph yields no_observations.
    """
    graph = FakeGraph(resolve_any={"California": "geoId/06"})
    result: RegionResult = asyncio.run(
        resolve_variable(
            "income",
            entities=["California"],
            date_request=None,
            detect_query="income",
            role_query="income of California",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.no_data_reason not in (
        "entity_not_resolved",
        "entity_ambiguous",
        "entity_role_not_admitted",
    )


_SUBJECT_SHAPE = ShapeDraft(
    shape_id="subject-only",
    label="subject-only",
    pop_type_dcid="Person",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(),
    declared_roles=("observationAbout",),
)
_ROLE_BEARING_SHAPE = ShapeDraft(
    shape_id="role-bearing",
    label="role-bearing",
    pop_type_dcid="EconomicActivity",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(),
    declared_roles=("financingProvider", "financingRecipient"),
)


def _pruning_state(place_sets: dict[str, PlaceSet]) -> ResolveState:
    """A state naming `place_sets`, over the fixture graph's role ranges: the subject
    admits every place, and the flow roles countries and organisations."""
    graph = FakeGraph()
    return ResolveState(
        variable="aid",
        entities=list(place_sets),
        date_request=None,
        detect_query="aid",
        role_query="aid",
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        admission=Admission(graph=graph),
        llm=QuantityReadOnlyLLM(),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        rcl=Recall(candidates=[], resolutions={}),
        place_sets=place_sets,
        coverage=empty_coverage_snapshot(),
    )


def test_pruning_keeps_only_the_role_bearing_shape():
    """Prune the subject-only shape and retain the Organization-valued provider shape."""
    state = _pruning_state(
        {
            "Asian Development Bank": PlaceSet(
                name="Asian Development Bank",
                kind="place",
                node=PlaceNode(dcid="AsianDevelopmentBank", graph_type="Organization"),
            )
        }
    )

    survivors, never_admitted = asyncio.run(
        _prune_shapes_by_admission(state, [_SUBJECT_SHAPE, _ROLE_BEARING_SHAPE])
    )
    assert survivors == [_ROLE_BEARING_SHAPE]
    assert never_admitted == []


def _set(
    name: str, node: tuple[str, str] | None, member_type: str | None, kind: PlaceSetKind = "region"
) -> PlaceSet:
    """A set named `name` with the node `(dcid, type)` and two members of `member_type`."""
    return PlaceSet(
        name=name,
        kind=kind,
        node=None if node is None else PlaceNode(dcid=node[0], graph_type=node[1]),
        members=None if member_type is None else (f"{name}/m1", f"{name}/m2"),
        member_type=member_type,
    )


def test_a_flow_shape_admits_a_set_by_its_members_type_and_records_each_answer():
    """A continent's node is refused by both flow roles and its countries are admitted, so
    the flow shape survives. Both types are asked of every role, and the answers stand by
    role and type for settling to read."""
    state = _pruning_state({"Africa": _set("Africa", ("africa", "Continent"), "Country")})

    survivors, never_admitted = asyncio.run(
        _prune_shapes_by_admission(state, [_ROLE_BEARING_SHAPE])
    )

    assert survivors == [_ROLE_BEARING_SHAPE]
    assert never_admitted == []
    assert state.node_admissions == {
        ("financingProvider", "Continent"): False,
        ("financingProvider", "Country"): True,
        ("financingRecipient", "Continent"): False,
        ("financingRecipient", "Country"): True,
    }


@pytest.mark.parametrize(
    ("place_set", "reported"),
    [
        pytest.param(
            _set("Africa", ("africa", "Continent"), "State"),
            ("africa", "Continent"),
            id="region-names-its-node",
        ),
        pytest.param(
            _set("Union", ("TestUnion", "ContinentalUnion"), "State", "group"),
            ("TestUnion", "ContinentalUnion"),
            id="group-with-a-node-names-its-node",
        ),
        pytest.param(
            _set("States", None, "State", "group"),
            ("States/m1", "State"),
            id="group-with-no-node-names-its-first-member",
        ),
        pytest.param(
            dataclasses.replace(_set("Untyped", None, "State", "group"), member_type=None),
            ("Untyped/m1", None),
            id="group-with-no-node-and-untyped-members-names-its-first-member",
        ),
    ],
)
def test_a_set_neither_of_whose_types_a_role_admits_is_refused(
    place_set: PlaceSet, reported: tuple[str, str]
):
    """The flow roles admit neither the set's node nor its members, so no shape survives
    and the refusal names the set by its node, or by its first member when it has none."""
    state = _pruning_state({place_set.name: place_set})

    survivors, never_admitted = asyncio.run(
        _prune_shapes_by_admission(state, [_ROLE_BEARING_SHAPE])
    )

    assert survivors == []
    assert [(e.text, e.dcid, e.graph_type) for e in never_admitted] == [(place_set.name, *reported)]


@pytest.mark.parametrize(
    ("reading", "survives"),
    [("default", True), ("members", False)],
    ids=["node-or-members", "members-asked"],
)
def test_a_set_asked_for_its_members_is_admitted_by_their_type_alone(reading, survives):
    """The flow roles admit the set's node and not its members. Read by default the set
    can bind its node, so the shape survives; asked for its members it has no reading the
    roles admit, and the refusal names it."""
    place_set = dataclasses.replace(
        _set("Bank group", ("TestBank", "Organization"), "State", "group"), reading=reading
    )
    state = _pruning_state({place_set.name: place_set})

    survivors, never_admitted = asyncio.run(
        _prune_shapes_by_admission(state, [_ROLE_BEARING_SHAPE])
    )

    assert survivors == ([_ROLE_BEARING_SHAPE] if survives else [])
    assert [e.text for e in never_admitted] == ([] if survives else [place_set.name])


def test_a_group_with_no_node_is_admitted_by_its_registry_members_type():
    """The fixture resolver's euro area is a group with no graph node. The membership
    registry types its members `Country`, so the role admits them, and with no observations
    the answer is `no_observations`, not a role refusal. The refusal of a group whose members
    carry no type is `test_a_set_neither_of_whose_types_a_role_admits_is_refused`."""
    graph = FakeGraph()
    result: RegionResult = asyncio.run(
        resolve_variable(
            "GDP",
            entities=["euro area"],
            date_request=None,
            detect_query="GDP",
            role_query="GDP of the euro area",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.no_data_reason == "no_observations"
    assert result.role_not_admitted_entities == ()
