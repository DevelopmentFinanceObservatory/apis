"""Pins entity resolution, place expansion, and one resolver call per request."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from search.engine.core import resolve_async
from search.engine.extract import ExtractedEntity, Extraction, ExtractionResult
from search.engine.region import RegionResult, resolve_variable
from search.models import CoverageExact, RawTextInput, ResolveRequest
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_NODE = {
    "label": "Nominal gross domestic product",
    "type": "StatisticalVariable",
    "arcs": {
        "constraintProperties": {
            "nodes": [{"name": "activitySource", "types": ["Property"], "dcid": "activitySource"}]
        },
        "activitySource": {
            "nodes": [{"name": "GrossDomesticProduction", "dcid": "GrossDomesticProduction"}]
        },
        "measuredProperty": {"nodes": [{"name": "amount", "dcid": "amount"}]},
        "measurementQualifier": {"nodes": [{"name": "Nominal", "dcid": "Nominal"}]},
        "statType": {"nodes": [{"name": "measuredValue", "dcid": "measuredValue"}]},
        "typeOf": {"nodes": [{"name": "StatisticalVariable", "dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Nominal gross domestic product"}]},
        "populationType": {"nodes": [{"name": "EconomicActivity", "dcid": "EconomicActivity"}]},
    },
}

# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_SOURCE = "dc/base/PlaceFixture"


def test_kenya_resolves_to_country_ken_and_the_response_carries_it():
    """The pipeline resolves Kenya to `country/KEN`."""
    req = ResolveRequest(input=RawTextInput(query="fertility rate in Kenya"))
    result = asyncio.run(
        resolve_async(req, graph=FakeGraph(), llm=FakeLLM(), sdmx=FakeSdmxClient())
    )
    inner = result.parts[0]
    assert inner.status == "definite"
    assert [(e.ref, e.label) for e in inner.spec.resolved.entities] == [("country/KEN", "Kenya")]


def test_africa_resolves_and_expands_to_its_member_countries():
    """Africa resolves to `africa`; its coverage counts observed member countries."""
    obs = {
        f"{_GDP_SV}|country/KEN": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE}
        ],
        f"{_GDP_SV}|country/UGA": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "provenanceId": _SOURCE}
        ],
    }
    graph = FakeGraph(
        nodes={_GDP_SV: _GDP_NODE, "africa": {"label": "Africa", "type": "Continent", "arcs": {}}},
        obs=obs,
        detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
        resolve={},
        places={},
        children={"africa": {"Country": ["country/KEN", "country/UGA", "country/NGA"]}},
    )
    result: RegionResult = asyncio.run(
        resolve_variable(
            "gdp",
            entities=["Africa"],
            date_request=None,
            detect_query="gdp",
            role_query="gdp Africa",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=FakeLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    spec = result.specs[0]
    assert [e.ref for e in spec.resolved.entities] == ["africa"]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.dimensions is not None
    countries_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "countries")
    assert countries_dim.count == 2


def test_dcid_absent_from_graph_node_table_yields_entity_not_resolved():
    """A resolvekit match absent from the graph fails as `entity_not_resolved`."""
    graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={}, places={})
    result: RegionResult = asyncio.run(
        resolve_variable(
            "gdp",
            entities=["Kenya"],
            date_request=None,
            detect_query="gdp",
            role_query="gdp Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=FakeLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "entity_not_resolved"


def test_lagos_resolves_to_a_place_other_kind_and_admission_lets_it_through():
    """The bare-arc City match is admitted on `observationAbout` and reaches materialisation."""
    graph = FakeGraph(
        nodes={"city/Lagos": {"label": "Lagos", "type": "City", "arcs": {}}, _GDP_SV: _GDP_NODE},
        obs={
            f"{_GDP_SV}|city/Lagos": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ]
        },
        detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
        resolve={},
        places={},
        resolve_any={"Lagos": "city/Lagos"},
    )
    result: RegionResult = asyncio.run(
        resolve_variable(
            "gdp",
            entities=["Lagos"],
            date_request=None,
            detect_query="gdp",
            role_query="gdp Lagos",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=FakeLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert [e.ref for e in result.specs[0].resolved.entities] == ["city/Lagos"]


def test_two_variable_legs_naming_the_same_entities_resolve_in_one_call():
    """Resolve one deduplicated entity list once for both variables."""
    fake_graph = FakeGraph()
    resolver = make_resolver(fake_graph)
    resolve_calls: list[list[str]] = []

    _original_resolve = resolver.resolve

    async def _tracked_resolve(texts):
        resolve_calls.append(list(texts))
        return await _original_resolve(texts)

    resolver.resolve = _tracked_resolve  # ty: ignore[invalid-assignment]

    async def _stub_resolve(variable, **_kwargs):
        return RegionResult(
            variable_text=variable,
            status="no_data",
            specs=(),
            no_data_reason="variable_not_resolved",
            warnings=(),
            timing_by_step={},
        )

    fake_extraction = AsyncMock(
        return_value=(
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution",
                    variables=["v1", "v2"],
                    entities=[ExtractedEntity(text=t) for t in ("Kenya", "Uganda", "Tanzania")],
                    dates=[],
                ),
                qualifiers={},
            ),
            None,
        )
    )
    req = ResolveRequest(input=RawTextInput(query="test"))
    with (
        patch("search.engine.entry.raw_text.extract", fake_extraction),
        patch("search.engine.fanout.resolve_variable", new=_stub_resolve),
    ):
        asyncio.run(
            resolve_async(
                req,
                graph=fake_graph,
                resolver=resolver,
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
            )
        )

    assert resolve_calls == [["Kenya", "Uganda", "Tanzania"]]
