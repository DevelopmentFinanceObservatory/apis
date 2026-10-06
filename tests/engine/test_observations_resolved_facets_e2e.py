"""End-to-end tests for resolved.facets in buffered, SSE, and resubmit responses."""

from __future__ import annotations

from fastapi.testclient import TestClient

from search.engine.core import resolve_async
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
)
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    make_identity,
    subject_entity,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    materialise_double_from_obs,
    parse_sse,
    run_pinned,
    with_member_arcs,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

# ---------------------------------------------------------------------------
# Preview-enabled definite responses publish resolved facets on every transport.
# ---------------------------------------------------------------------------


_PERSON_ETH_OBS: dict = {
    "Count_Person|country/ETH": [
        {
            "earliestDate": "1960",
            "latestDate": "2024",
            "obsCount": 65,
            "provenanceId": "dc/base/WorldDevelopmentIndicators",
        },
        {
            "earliestDate": "1960",
            "latestDate": "2023",
            "obsCount": 58,
            "provenanceId": "dc/base/WikidataPopulation",
            "measurementMethod": "WikidataPopulation",
        },
        {
            "earliestDate": "2002",
            "latestDate": "2022",
            "obsCount": 11,
            "provenanceId": "dc/base/EthiopiaStatistics",
        },
        {
            "earliestDate": "2000",
            "latestDate": "2016",
            "obsCount": 17,
            "provenanceId": "dc/base/Subnational_Demographics_Stats",
            "measurementMethod": "WorldBankSubnationalPopulationEstimate",
        },
    ],
}


def _person_eth_materialise_double():
    """Builds materialise facets from graph summaries without point rows."""
    return materialise_double_from_obs(_PERSON_ETH_OBS)


def _person_eth_resubmit_input(facets: dict | None = None) -> dict:
    """Builds an Ethiopia resubmit entry with four measurement-method facets."""
    entities = [subject_entity("country/ETH", "Ethiopia")]
    explicit_facets = None
    if facets is not None:
        explicit_facets = BaseIdentity.model_validate(
            {
                "shape": PERSON_COUNT_SHAPE,
                "slots": [],
                "entities": {},
                "requested_window": {"kind": "all"},
                "facets": facets,
            }
        ).facets
    result = {
        "entry_kind": "identity",
        "variable_text": "population of Ethiopia",
        "identity": make_identity(
            PERSON_COUNT_SHAPE,
            entities=entities,
            explicit_facets=explicit_facets,
        ).model_dump(mode="json"),
    }
    return result


def test_preview_enabled_buffered_definite_populates_resolved_facets_with_fallback_basis():
    """An unbound resubmit aggregates fallback facets from its series data."""
    materialise_graph, materialise_sdmx = _person_eth_materialise_double()
    result = run_pinned(
        resolve_async(
            ResolveRequest(
                input=DatasetResubmitInput(
                    entries=[ResubmitIdentityEntry(**_person_eth_resubmit_input())]
                ),
                response_options=ResolveOptions(include_observations=True),
            ),
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=with_member_arcs(materialise_graph, FakeGraph()),
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert result.status == "definite"
    facets = result.spec.resolved.facets
    assert facets["provenance"].basis == "fallback"
    assert facets["provenance"].ref == "dc/base/WorldDevelopmentIndicators"


def test_preview_enabled_sse_response_event_populates_resolved_facets():
    """The `response` event carries resolved facets before the trailing observations event."""
    materialise_graph, materialise_sdmx = _person_eth_materialise_double()
    app = search_app(
        graph=FakeGraph(),
        resolver=make_resolver(FakeGraph()),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=with_member_arcs(materialise_graph, FakeGraph()),
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "population of Ethiopia",
                        **_person_eth_resubmit_input(),
                    }
                ],
            },
            "response_options": {"include_observations": True},
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    response_idx = event_types.index("response")
    response_payload = frames[response_idx][1]["response"]
    facets = response_payload["parts"][0]["spec"]["resolved"]["facets"]
    assert facets["provenance"]["basis"] == "fallback"
    assert facets["provenance"]["ref"] == "dc/base/WorldDevelopmentIndicators"


def test_resubmit_sse_response_event_bound_axis_the_unconstrained_pick_disagrees_now_serves_its_data():
    """Binding `WikidataPopulation` selects it over the higher-count
    `WorldDevelopmentIndicators` facet.

    The response marks it bound and includes its series. The fixture has no point rows."""
    materialise_graph, materialise_sdmx = _person_eth_materialise_double()
    app = search_app(
        graph=FakeGraph(),
        resolver=make_resolver(FakeGraph()),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=with_member_arcs(materialise_graph, FakeGraph()),
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "population of Ethiopia",
                        **_person_eth_resubmit_input(
                            facets={
                                "measurementMethod": {
                                    "state": "value",
                                    "ref": "WikidataPopulation",
                                }
                            }
                        ),
                    }
                ],
            },
            "response_options": {"include_observations": True},
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert "response" in event_types, event_types
    response_idx = event_types.index("response")
    response_payload = frames[response_idx][1]["response"]
    assert response_payload["status"] == "definite"
    resolved = response_payload["parts"][0]["spec"]["resolved"]
    facets = resolved["facets"]
    assert facets["measurementMethod"]["basis"] == "bound"
    assert facets["measurementMethod"]["ref"] == "WikidataPopulation"

    obs_idx = event_types.index("observations")
    obs_payload = frames[obs_idx][1]["observations"]
    assert obs_payload["series"][0]["facet"]["measurementMethod"]["ref"] == "WikidataPopulation"
