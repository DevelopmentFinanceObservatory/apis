"""Test injected and lifespan-built rate caches across buffered and SSE requests. Each request
reads through its own `RequestRateTables` view over the one shared cache."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from materialise.rate_tables import RateTableCache, RequestRateTables
from search.engine.app import create_app
from search.engine.extract import Extraction, ExtractionResult
from search.engine.region.result import RegionResult
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def _stub_region(variable_text: str | None) -> RegionResult:
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


def _client(*, rate_cache: RateTableCache | None) -> TestClient:
    graph = FakeGraph()
    return TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            rate_cache=rate_cache,
        ),
        raise_server_exceptions=True,
    )


async def _fake_extract(query, *, llm):
    return (
        ExtractionResult(
            extraction=Extraction(kind="resolution", variables=["gdp"], entities=[], dates=[]),
            qualifiers={},
        ),
        None,
    )


def test_buffered_resolve_forwards_injected_rate_cache_to_resolve_variable():
    sentinel = RateTableCache()
    captured: list[object] = []

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    client = _client(rate_cache=sentinel)
    with (
        patch("search.engine.entry.raw_text.extract", _fake_extract),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
    ):
        resp = client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "gdp"}}
        )
    assert resp.status_code == 200
    assert len(captured) == 1
    assert isinstance(captured[0], RequestRateTables)
    assert captured[0].shared is sentinel


def test_sse_resolve_forwards_injected_rate_cache_to_resolve_variable():
    sentinel = RateTableCache()
    captured: list[object] = []

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    client = _client(rate_cache=sentinel)
    with (
        patch("search.engine.entry.raw_text.extract", _fake_extract),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
    ):
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "gdp"}},
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    assert len(captured) == 1
    assert isinstance(captured[0], RequestRateTables)
    assert captured[0].shared is sentinel


def test_surface_with_no_rate_cache_still_serves():
    """An uninjected cache is None outside lifespan."""
    captured: list[object] = []

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    client = _client(rate_cache=None)
    with (
        patch("search.engine.entry.raw_text.extract", _fake_extract),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
    ):
        resp = client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "gdp"}}
        )
    assert resp.status_code == 200
    assert captured == [None]


def test_deployed_lifespan_builds_and_reuses_one_shared_rate_cache(search_settings):
    """Lifespan startup builds one shared cache. Each request reads through its own view of it.

    Disable response caching so each request reaches resolve_variable.
    Injected fakes keep startup offline.
    """
    search_settings(response_cache_enabled=False)
    captured: list[object] = []

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    graph = FakeGraph()
    app = create_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
    )
    with (
        patch("search.engine.entry.raw_text.extract", _fake_extract),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
        TestClient(app) as client,
    ):
        resp1 = client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "gdp"}}
        )
        resp2 = client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "gdp"}}
        )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert len(captured) == 2
    first, second = captured
    assert isinstance(first, RequestRateTables)
    assert isinstance(second, RequestRateTables)
    assert first is not second
    assert first.shared is second.shared
