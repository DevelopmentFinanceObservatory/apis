"""Tests that the datasets graph clients key their caches on the shared graph generation.

The mint door, the definition door and the worker acquire the process-wide provider in their
lifespans, pass it to `build_graph_client` beside the UN availability client, register the
default-scope memo on it, and release it after their clients close.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, cast

import pytest

from datasets_api import generation as generation_module
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import definition as definition_door
from datasets_api.doors import mint as mint_door
from datasets_api.doors.definition import build_definition_door
from datasets_api.doors.mint import build_mint_door
from datasets_api.operation_store import OperationStore
from datasets_api.operations import worker_app as worker_module
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker_app import WORKER_PLACE_SCOPE_MEMO_NAME, build_worker_app
from datasets_api.versions import NullVersionRecordWriter
from graph_generation import GraphGeneration
from graph_generation.fake import FakeGeneration
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

from ._artifact_doubles import FakeArtifactStore
from .conftest import FakeDatasetStore

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
        return _RecordingClient(events, f"sdmx {kwargs['base_url']}", **kwargs)

    return _build


def _enter_and_exit(door) -> None:
    async def _run() -> None:
        async with door.lifespan(None):
            pass

    asyncio.run(_run())


def _mint_door(**injected: Any):
    return build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        settings=DatasetsBuildSettings(),
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        task_client=FakeTaskClient(),
        catalogue=fake_catalogue_service(),
        **injected,
    )


def _definition_door(**injected: Any):
    return build_definition_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        settings=DatasetsBuildSettings(),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        task_client=FakeTaskClient(),
        catalogue=fake_catalogue_service(),
        **injected,
    )


@pytest.mark.parametrize(
    ("module", "build_door"),
    [(mint_door, _mint_door), (definition_door, _definition_door)],
    ids=["mint", "definition"],
)
def test_a_door_builds_its_graph_client_on_the_shared_generation(
    monkeypatch: pytest.MonkeyPatch, events: _Events, module: Any, build_door: Callable[..., Any]
) -> None:
    monkeypatch.setattr(module, "build_graph_client", _graph_builder(events))
    monkeypatch.setattr(module, "SdmxClient", _sdmx_builder(events))

    _enter_and_exit(build_door())

    assert events.acquire_kwargs["expected_graph_base"] == _SETTINGS_ENV["DATASETS_GRAPH_BASE_URL"]
    assert events.graph_kwargs["generation"] is events.generation
    un_availability = events.graph_kwargs["un_availability"]
    assert un_availability.kwargs["base_url"] == events.graph_kwargs["un_base_url"]
    assert un_availability.kwargs["timeout"] == DatasetsBuildSettings().sdmx_timeout_seconds


@pytest.mark.parametrize(
    ("module", "build_door"),
    [(mint_door, _mint_door), (definition_door, _definition_door)],
    ids=["mint", "definition"],
)
def test_a_door_releases_the_shared_generation_after_its_graph_client_closes(
    monkeypatch: pytest.MonkeyPatch, events: _Events, module: Any, build_door: Callable[..., Any]
) -> None:
    monkeypatch.setattr(module, "build_graph_client", _graph_builder(events))
    monkeypatch.setattr(module, "SdmxClient", _sdmx_builder(events))

    _enter_and_exit(build_door())

    assert events.log[0] == "acquire generation"
    assert events.log.index("build graph") > events.log.index("acquire generation")
    assert events.log.index("close graph") < events.log.index("release generation")
    assert events.log[-1] == "release generation"


@pytest.mark.parametrize(
    ("module", "build_door"),
    [(mint_door, _mint_door), (definition_door, _definition_door)],
    ids=["mint", "definition"],
)
def test_a_door_acquires_nothing_for_an_injected_graph(
    monkeypatch: pytest.MonkeyPatch, events: _Events, module: Any, build_door: Callable[..., Any]
) -> None:
    monkeypatch.setattr(module, "SdmxClient", _sdmx_builder(events))
    graph = _RecordingClient(events, "injected graph")

    _enter_and_exit(build_door(graph=graph))

    assert events.log == [f"close sdmx {_SETTINGS_ENV['DATASETS_SDMX_BASE_URL']}"]


def test_a_door_releases_the_shared_generation_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    monkeypatch.setattr(mint_door, "build_graph_client", _graph_builder(events))
    monkeypatch.setattr(mint_door, "SdmxClient", _sdmx_builder(events))

    def _raise_shared_pool() -> None:
        raise RuntimeError("shared_pool exploded")

    monkeypatch.setattr(mint_door, "shared_pool", _raise_shared_pool)
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        settings=DatasetsBuildSettings(),
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        version_writer=NullVersionRecordWriter(),
        task_client=FakeTaskClient(),
        catalogue=fake_catalogue_service(),
    )

    with pytest.raises(RuntimeError, match="shared_pool exploded"):
        _enter_and_exit(door)

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
    monkeypatch.setattr(worker_module, "get_index", lambda: object())
    monkeypatch.setattr(worker_module, "get_coverage_service", lambda: object())
    monkeypatch.setattr(worker_module, "_build_live_definition_deps", lambda **_: object())
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
    assert registered == [(WORKER_PLACE_SCOPE_MEMO_NAME,)]
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
