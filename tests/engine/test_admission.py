"""Test shape admission against entity types and declared graph roles."""

from __future__ import annotations

import asyncio
from typing import cast

from entity_resolution import Admission, Entity, Resolved
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.interpret import Recall
from search.engine.region import RegionResult, resolve_variable
from search.engine.region.phase_intake import _prune_shapes_by_admission
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def test_population_with_a_bank_entity_refuses_entity_role_not_admitted():
    """Subject-only population shapes reject the bank's Organization entity."""
    graph = FakeGraph()
    result: RegionResult = asyncio.run(
        resolve_variable(
            "population",
            entities=["African Development Bank"],
            date_request=None,
            detect_query="population",
            role_query="population African Development Bank",
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


def test_pruning_keeps_only_the_role_bearing_shape():
    """Prune the subject-only shape and retain the Organization-valued provider shape."""
    graph = FakeGraph()
    subject_shape = ShapeDraft(
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
    role_bearing_shape = ShapeDraft(
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
    bank = Entity(
        dcid="AsianDevelopmentBank",
        name="Asian Development Bank",
        kind="other",
        graph_type="Organization",
    )
    state = ResolveState(
        variable="aid",
        entities=["Asian Development Bank"],
        date_request=None,
        detect_query="aid",
        role_query="aid from Asian Development Bank",
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        admission=Admission(graph=graph),
        llm=cast(SupportsLLM, None),
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
        source_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        rcl=Recall(
            candidates=[],
            resolutions={"Asian Development Bank": Resolved(entity=bank)},
            resolved_entity_names={"Asian Development Bank": bank.dcid},
        ),
        coverage=empty_coverage_snapshot(),
    )

    survivors, never_admitted = asyncio.run(
        _prune_shapes_by_admission(state, [subject_shape, role_bearing_shape])
    )
    assert survivors == [role_bearing_shape]
    assert never_admitted == []
