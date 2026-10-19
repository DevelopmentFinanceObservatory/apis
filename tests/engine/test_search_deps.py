"""The lifespan publishes and clears `SearchDeps`, then releases acquired resources in order."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import search.engine.app as app_module
from graph_generation import FakeGeneration
from search.engine.deps import SEARCH_DEPS_SLOT
from search.engine.errors import LLMInfraError
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

# Registrations close first, then skills index, place-options pool, graph
# client, SDMX client, materialise SDMX client, resolvekit, generation.
_RELEASE_ORDER = [
    "registrations",
    "skills_index",
    "place_options_pool",
    "graph",
    "sdmx",
    "materialise_sdmx",
    "resolvekit",
    "generation",
]


def test_request_before_the_lifespan_refuses_503():
    """No lifespan entered and nothing injected: the same refusal a request
    reaching the graph client's absence gets."""
    client = TestClient(app_module.create_app(), raise_server_exceptions=False)

    resp = client.post("/api/search/resolve", json={"input": {"kind": "raw_text", "query": "gdp"}})

    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_two_sequential_lifespan_entries_each_publish_and_clear_the_bundle():
    graph = FakeGraph()
    app = app_module.create_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        skills_index=object(),  # type: ignore[arg-type]  # never read by /schema
        generation=FakeGeneration(),
    )
    for _ in range(2):
        with TestClient(app, raise_server_exceptions=False) as client:
            assert hasattr(app.state, SEARCH_DEPS_SLOT.name)
            assert client.get("/api/search/schema").status_code == 200
        assert not hasattr(app.state, SEARCH_DEPS_SLOT.name)


def _record_releases(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], FakeGeneration]:
    events: list[str] = []
    generation = FakeGeneration()

    real_register = generation.register

    def _register(name, *, stats, clear):
        handle = real_register(name, stats=stats, clear=clear)
        real_close = handle.close

        class _Handle:
            def close(self) -> None:
                if not events or events[-1] != "registrations":
                    events.append("registrations")
                real_close()

        return _Handle()

    monkeypatch.setattr(generation, "register", _register)
    monkeypatch.setattr(app_module, "acquire_shared_generation", lambda **_: generation)
    monkeypatch.setattr(
        app_module, "release_shared_generation", lambda _g: events.append("generation")
    )

    real_release_shared = app_module.release_shared

    def _release_shared(shared) -> None:
        events.append("resolvekit")
        real_release_shared(shared)

    monkeypatch.setattr(app_module, "release_shared", _release_shared)

    class _MaterialiseSdmx:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            events.append("materialise_sdmx")

        def stats(self) -> dict[str, int]:
            return {}

        def clear(self) -> None:
            pass

    monkeypatch.setattr(app_module, "build_materialise_sdmx_client", _MaterialiseSdmx)

    class _LiveSdmx(FakeSdmxClient):
        def close(self) -> None:
            events.append("sdmx")

    monkeypatch.setattr(app_module, "LiveSdmxClient", _LiveSdmx)

    class _LiveGraph(FakeGraph):
        @classmethod
        def from_search_settings(cls, *_args, **_kwargs):
            return cls()

        def close(self) -> None:
            events.append("graph")

        def stats(self) -> dict[str, int]:
            return {}

        def clear(self) -> None:
            pass

    monkeypatch.setattr(app_module, "LiveGraphClient", _LiveGraph)

    class _Pool:
        @classmethod
        def build(cls, *, max_workers: int, options_cache: object = None):
            return cls()

        def shutdown(self) -> None:
            events.append("place_options_pool")

    monkeypatch.setattr(app_module, "PlaceOptionsPool", _Pool)

    async def _acquire_index(*, graph, catalogue):
        return object()

    async def _release_index(_index) -> None:
        events.append("skills_index")

    monkeypatch.setattr(app_module, "acquire_shared_index", _acquire_index)
    monkeypatch.setattr(app_module, "release_shared_index", _release_index)
    return events, generation


def test_shutdown_releases_what_the_lifespan_built_in_order(monkeypatch):
    events, generation = _record_releases(monkeypatch)
    lifespan = app_module.search_surface(llm=FakeLLM()).lifespan
    assert lifespan is not None

    async def _body() -> None:
        async with lifespan(FastAPI()):
            assert events == []
            assert generation.registered_names()

    asyncio.run(_body())

    assert events == _RELEASE_ORDER
    assert generation.registered_names() == ()


def test_startup_failure_releases_what_was_built_in_order(monkeypatch):
    events, generation = _record_releases(monkeypatch)

    def _fail() -> None:
        raise LLMInfraError("no key")

    monkeypatch.setattr(app_module._llm_mod, "warm", _fail)
    lifespan = app_module.search_surface().lifespan
    assert lifespan is not None

    async def _body() -> None:
        async with lifespan(FastAPI()):
            pytest.fail("the lifespan must not yield after a startup failure")

    with pytest.raises(LLMInfraError):
        asyncio.run(_body())

    assert events == _RELEASE_ORDER
    assert generation.registered_names() == ()
