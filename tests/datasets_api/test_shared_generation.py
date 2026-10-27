"""Tests that the datasets graph clients key their caches on the shared graph generation.

The surface's `open_deps` and the worker acquire the process-wide provider in their lifespans,
pass it to `build_graph_client` beside the UN availability client, register the default-scope
memo on it, and release it after their clients close.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, cast

import pytest

from code_index.shared import get_index
from datasets_api import deps as deps_module
from datasets_api import generation as generation_module
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsOverrides, open_deps
from datasets_api.operation_store import OperationStore
from datasets_api.operations import worker_app as worker_module
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker_app import WORKER_PLACE_SCOPE_MEMO_NAME, build_worker_app
from dc_sdmx import DATA_HEDGE_AFTER_SECONDS
from graph_generation import GraphGeneration
from graph_generation.fake import FakeGeneration
from graph_release.service import get_release_service
from materialise.rate_tables import RateTableCache
from series_coverage.shared import get_coverage_service
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

from ._artifact_doubles import FakeArtifactStore

_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}


class _Events:
    """The order in which the shared provider and the clients were opened and closed."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.generation = FakeGeneration()
        self.acquire_kwargs: dict[str, Any] = {}
        self.graph_kwargs: dict[str, Any] = {}
        self.sdmx_clients: list[_RecordingClient] = []


class _RecordingClient:
    def __init__(self, events: _Events, name: str, **kwargs: Any) -> None:
        self.events = events
        self.name = name
        self.kwargs = kwargs

    def close(self) -> None:
        self.events.log.append(f"close {self.name}")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> _Events:
    recorded = _Events()

    def _acquire(**kwargs: Any) -> FakeGeneration:
        recorded.acquire_kwargs = kwargs
        recorded.log.append("acquire generation")
        return recorded.generation

    def _release(generation: object) -> None:
        assert generation is recorded.generation
        recorded.log.append("release generation")

    monkeypatch.setattr(generation_module, "acquire_shared", _acquire)
    monkeypatch.setattr(generation_module, "release_shared", _release)
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    return recorded


def _graph_builder(events: _Events) -> Callable[..., _RecordingClient]:
    def _build(**kwargs: Any) -> _RecordingClient:
        events.graph_kwargs = kwargs
        events.log.append("build graph")
        return _RecordingClient(events, "graph", **kwargs)

    return _build


def _sdmx_builder(events: _Events) -> Callable[..., _RecordingClient]:
    def _build(**kwargs: Any) -> _RecordingClient:
        client = _RecordingClient(events, f"sdmx {kwargs['base_url']}", **kwargs)
        events.sdmx_clients.append(client)
        return client

    return _build


def _hedging(events: _Events) -> dict[str, float | None]:
    """Each SDMX client's `hedge_after`, by base URL; only the mint's client on our mixer
    hedges."""
    return {
        client.kwargs["base_url"]: client.kwargs.get("hedge_after")
        for client in events.sdmx_clients
    }


_STORES = (
    "registry",
    "spec_reader",
    "snapshot_reader",
    "version_writer",
    "head_store",
    "version_lister",
    "operation_store",
    "recipe_store",
    "description_store",
    "reference_writer",
)


def _open_deps(**injected: Any) -> None:
    """Open and close the surface's `Deps` with every store injected."""
    overrides = DepsOverrides(
        coverage=get_coverage_service(),
        rate_cache=RateTableCache(),
        release=get_release_service(),
        code_index=get_index(),
        task_client=FakeTaskClient(),
        **{name: object() for name in _STORES},
        **({"artifact_store": FakeArtifactStore()} | injected),
    )

    async def _run() -> None:
        async with open_deps(
            settings=DatasetsBuildSettings(),
            overrides=overrides,
            catalogue=fake_catalogue_service(),
        ):
            pass

    asyncio.run(_run())


@pytest.fixture
def live_clients(monkeypatch: pytest.MonkeyPatch, events: _Events) -> None:
    monkeypatch.setattr(deps_module, "build_graph_client", _graph_builder(events))
    monkeypatch.setattr(deps_module, "SdmxClient", _sdmx_builder(events))


@pytest.mark.usefixtures("live_clients")
def test_the_graph_client_is_built_on_the_shared_generation(events: _Events) -> None:
    _open_deps()

    assert events.acquire_kwargs["expected_graph_base"] == _SETTINGS_ENV["DATASETS_GRAPH_BASE_URL"]
    assert events.graph_kwargs["generation"] is events.generation
    un_availability = events.graph_kwargs["un_availability"]
    assert un_availability.kwargs["base_url"] == events.graph_kwargs["un_base_url"]
    assert un_availability.kwargs["timeout"] == DatasetsBuildSettings().sdmx_timeout_seconds
    assert _hedging(events) == {
        _SETTINGS_ENV["DATASETS_SDMX_BASE_URL"]: DATA_HEDGE_AFTER_SECONDS,
        events.graph_kwargs["un_base_url"]: None,
    }


@pytest.mark.usefixtures("live_clients")
def test_the_shared_generation_is_released_after_the_graph_client_closes(events: _Events) -> None:
    _open_deps()

    assert events.log[0] == "acquire generation"
    assert events.log.index("build graph") > events.log.index("acquire generation")
    assert events.log.index("close graph") < events.log.index("release generation")
    assert events.log[-1] == "release generation"


@pytest.mark.usefixtures("live_clients")
def test_an_injected_graph_acquires_no_generation(events: _Events) -> None:
    _open_deps(graph=_RecordingClient(events, "injected graph"))

    assert events.log == [f"close sdmx {_SETTINGS_ENV['DATASETS_SDMX_BASE_URL']}"]


@pytest.mark.usefixtures("live_clients")
def test_the_shared_generation_is_released_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    def _explode(**_kwargs: object) -> object:
        raise RuntimeError("artifact store exploded")

    monkeypatch.setattr(deps_module, "shared_pool", object)
    monkeypatch.setattr(deps_module, "close_shared_pool", lambda: None)
    monkeypatch.setattr(deps_module, "LiveDatasetStore", lambda **_kwargs: object())
    monkeypatch.setattr(deps_module, "LiveArtifactStore", _explode)

    with pytest.raises(RuntimeError, match="artifact store exploded"):
        _open_deps(artifact_store=None)

    assert events.log.index("close graph") < events.log.index("release generation")
    assert events.log[-1] == "release generation"


def test_the_worker_builds_its_graph_client_on_the_shared_generation(
    monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    monkeypatch.setattr(worker_module, "build_graph_client", _graph_builder(events))
    monkeypatch.setattr(worker_module, "SdmxClient", _sdmx_builder(events))
    monkeypatch.setattr(worker_module, "shared_pool", lambda: object())
    monkeypatch.setattr(worker_module, "close_shared_pool", lambda: events.log.append("close pool"))
    monkeypatch.setattr(worker_module, "LiveDatasetStore", lambda **_: object())
    monkeypatch.setattr(worker_module, "LiveArtifactStore", lambda **_: object())
    monkeypatch.setattr(worker_module, "LiveArtifactInventory", lambda **_: object())
    index_reads: list[None] = []
    monkeypatch.setattr(worker_module, "get_index", lambda: index_reads.append(None))
    monkeypatch.setattr(
        worker_module,
        "get_coverage_service",
        lambda: FakeCoverageService(build_coverage_index([], role_names={})),
    )
    monkeypatch.setattr(worker_module, "_live_deps_factory", lambda **_: object())
    registered: list[tuple[str, ...]] = []

    app = build_worker_app(
        settings=DatasetsBuildSettings(),
        operation_store=cast(OperationStore, object()),  # the lifespan never reads it
        worker_url="https://worker.example.org",
        worker_invoker_service_account="worker-invoker@example.iam.gserviceaccount.com",
        catalogue=fake_catalogue_service(),
    )

    async def _run() -> None:
        async with app.router.lifespan_context(app):
            registered.append(events.generation.registered_names())

    asyncio.run(_run())

    assert events.graph_kwargs["generation"] is events.generation
    assert events.graph_kwargs["un_availability"] is not None
    assert _hedging(events) == {
        _SETTINGS_ENV["DATASETS_SDMX_BASE_URL"]: DATA_HEDGE_AFTER_SECONDS,
        events.graph_kwargs["un_base_url"]: None,
    }
    assert registered == [(WORKER_PLACE_SCOPE_MEMO_NAME,)]
    assert index_reads == [None]  # the startup read that fails a missing export
    assert events.generation.registered_names() == ()
    assert events.log.index("close graph") < events.log.index("release generation")


def test_the_memo_registration_is_removed_when_the_stack_closes() -> None:
    generation = FakeGeneration()

    async def _run() -> tuple[str, ...]:
        async with AsyncExitStack() as stack:
            generation_module.register_place_scope_memo(
                stack, generation, name=generation_module.PLACE_SCOPE_MEMO_NAME
            )
            return generation.registered_names()

    assert asyncio.run(_run()) == (generation_module.PLACE_SCOPE_MEMO_NAME,)
    assert generation.registered_names() == ()


def test_the_shared_provider_is_one_graph_generation_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two acquires in one process return the same instance, so two doors share a stamp key."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    catalogue = fake_catalogue_service()

    async def _run() -> tuple[object, object]:
        async with AsyncExitStack() as stack:
            first = await generation_module.enter_shared_generation(
                stack, catalogue=catalogue, graph_base_url=_SETTINGS_ENV["DATASETS_GRAPH_BASE_URL"]
            )
            second = await generation_module.enter_shared_generation(
                stack, catalogue=catalogue, graph_base_url=_SETTINGS_ENV["DATASETS_GRAPH_BASE_URL"]
            )
            return first, second

    first, second = asyncio.run(_run())

    assert isinstance(first, GraphGeneration)
    assert first is second
