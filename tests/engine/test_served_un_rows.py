"""UN System Data Commons rows read through the graph reader the served search app builds:
`LiveGraphClient.from_search_settings`, as `_open_search_deps` builds it. It returns the
origin client's UN reader, so a UN preview and an answer electing a UN series read rows
through it, embedded in a buffered answer and in the streamed answer's trailing observations
event."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Iterator

import httpx
import pytest

import search.engine.clients.graph_client as graph_client_module
from dc_sdmx import SdmxClient
from materialise.graph import GraphReader
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.materialise_bridge import build_materialise_sdmx_client
from search.engine.concurrency import scoped_concurrency
from search.engine.config import settings
from search.engine.region.codes import _ObsProbeResult
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import build_observations_planned
from tests.engine.test_pool_election import _ElectLLM
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.un_deployment import (
    OURS_API,
    UN_API,
    UN_FACET_ID,
    UN_PROVENANCE,
    UN_SERIES,
    UN_UNIT,
    UN_VARIABLE,
    UnDeployments,
)

_KEN = "country/KEN"
_OURS_VARIABLE = "Count_Death_Upto5Years_AsFractionOf_Count_BirthEvent_LiveBirth"
_QUERY = "under-five mortality"


@pytest.fixture
def deployments() -> UnDeployments:
    return UnDeployments()


@pytest.fixture
def served_graph(
    deployments: UnDeployments, search_settings, monkeypatch: pytest.MonkeyPatch
) -> Iterator[LiveGraphClient]:
    """The served app's graph client, with both deployments answered by `deployments`."""
    search_settings(
        graph_base=OURS_API.removesuffix("/core/api"),
        un_graph_base_url=UN_API,
    )
    monkeypatch.setattr(
        graph_client_module,
        "build_graph_client",
        functools.partial(build_graph_client, transport=httpx.MockTransport(deployments.handler)),
    )
    client = LiveGraphClient.from_search_settings(settings())
    yield client
    client.close()


@pytest.fixture
def served_sdmx() -> Iterator[SdmxClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a UN variable reached the SDMX service: {request.url}")

    client = build_materialise_sdmx_client(transport=httpx.MockTransport(handler))
    yield client
    client.close()


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda graph: graph, id="LiveGraphClient"),
    ],
)
def test_every_served_wrapper_returns_the_origin_clients_un_reader(
    served_graph: LiveGraphClient, wrap
) -> None:
    reader: GraphReader = wrap(served_graph)

    un_reader = reader.un_observation_reader()

    assert isinstance(un_reader, OriginGraphClient)
    assert un_reader is served_graph.un_observation_reader()


def test_a_un_preview_reads_its_rows_through_the_served_reader(
    served_graph: LiveGraphClient, served_sdmx: SdmxClient
) -> None:
    observations, warning = build_observations_planned(
        [UN_VARIABLE],
        {"observationAbout": [_KEN]},
        graph=FakeGraph(),
        materialise_graph=served_graph,
        materialise_sdmx=served_sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert warning is None
    assert observations is not None
    (series,) = observations.series
    assert series.stat_var.ref == UN_VARIABLE
    assert [(p.date, p.value) for p in series.points] == UN_SERIES[_KEN]


def _sv_node(dcid: str, *, population_type: str) -> dict:
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": dcid}]},
            "populationType": {"nodes": [{"dcid": population_type}]},
            "measuredProperty": {"nodes": [{"dcid": "mortalityRate"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": []},
        },
    }


def _un_answer_graph() -> FakeGraph:
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _OURS_VARIABLE: _sv_node(_OURS_VARIABLE, population_type="Person"),
            UN_VARIABLE: _sv_node(UN_VARIABLE, population_type=f"{UN_VARIABLE}/series"),
        },
        # Resolve reads the facet the UN deployment serves the preview.
        obs={
            f"{UN_VARIABLE}|{_KEN}": [
                {
                    "facetId": UN_FACET_ID,
                    "provenanceId": UN_PROVENANCE,
                    "unit": UN_UNIT,
                    "earliestDate": "1993",
                    "latestDate": "2014",
                    "obsCount": len(UN_SERIES[_KEN]),
                }
            ]
        },
        detect={_QUERY: {"svs": [_OURS_VARIABLE], "cosine_scores": [0.9]}},
        un_detect={_QUERY: {"svs": [UN_VARIABLE], "cosine_scores": [0.85]}},
        resolve={"Kenya": _KEN},
    )


async def _un_answer(options: EngineOptions) -> RegionResult:
    graph = _un_answer_graph()
    return await resolve_variable(
        _QUERY,
        entities=["Kenya"],
        date_request=None,
        detect_query=_QUERY,
        role_query=_QUERY,
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        llm=_ElectLLM(UN_VARIABLE),
        base_timing={},
        coverage=empty_coverage_snapshot(),
        options=options,
    )


def test_a_buffered_un_answer_embeds_un_rows_read_through_the_served_reader(
    served_graph: LiveGraphClient, served_sdmx: SdmxClient
) -> None:
    result = asyncio.run(
        _un_answer(
            EngineOptions(
                include_observations=True,
                materialise_graph=served_graph,
                materialise_sdmx=served_sdmx,
            )
        )
    )

    assert result.status == "definite"
    (spec,) = result.specs
    assert [sv.ref.ref for sv in spec.resolved.members] == [UN_VARIABLE]
    assert spec.resolved.observations is not None
    (series,) = spec.resolved.observations.series
    assert [p.value for p in series.points] == [value for _, value in UN_SERIES[_KEN]]


def test_a_streamed_un_answer_enriches_with_un_rows_read_through_the_served_reader(
    served_graph: LiveGraphClient, served_sdmx: SdmxClient
) -> None:
    async def resolve_and_enrich() -> tuple[RegionResult, _ObsProbeResult]:
        async with scoped_concurrency():
            result = await _un_answer(
                EngineOptions(
                    include_observations=True,
                    stream_observations=True,
                    materialise_graph=served_graph,
                    materialise_sdmx=served_sdmx,
                )
            )
            assert result.observations_task is not None
            return result, await result.observations_task

    result, (events, warnings) = asyncio.run(resolve_and_enrich())

    assert result.status == "definite"
    (spec,) = result.specs
    assert spec.resolved.observations is None
    assert warnings == ()
    (event,) = events
    assert event.spec_id == spec.spec_id
    assert event.observations is not None
    (series,) = event.observations.series
    assert series.stat_var.ref == UN_VARIABLE
    assert [(p.date, p.value) for p in series.points] == UN_SERIES[_KEN]
