"""Tests live dependency construction, ownership, and shutdown draining for dataset doors.
Recording clients expose token wiring, close calls, and the order of shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import ClassVar, cast

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import definition as definition_door
from datasets_api.doors import meta as meta_door
from datasets_api.doors import mint as mint_door
from datasets_api.doors import observations as observations_door
from datasets_api.doors import pin as pin_door
from datasets_api.doors import publish as publish_door
from datasets_api.doors import register as register_door
from datasets_api.doors import snapshot as snapshot_door
from datasets_api.doors.definition import build_definition_door
from datasets_api.doors.mint import build_mint_door
from datasets_api.doors.register import build_register_door
from datasets_api.mint_models import MintRequest
from datasets_api.operations.task_client import FakeTaskClient, LiveTaskClient
from datasets_api.shutdown import ShutdownBudget
from datasets_api.versions import NullVersionRecordWriter
from dfo_platform.doors import DoorContribution
from origin_catalogue.graph import UN_BASE_URL
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_app
from ._artifact_doubles import FakeArtifactStore, RecordingArtifactStore
from ._graph_doubles import BlockingFakeMintGraphReader, FakeMintGraphReader
from .conftest import FakeDatasetStore
from .test_mint_concurrency import _BODY, _default_graph


class _RecordingClient:
    """Recording stand-in for both clients.
    Accepts `timeout` for SDMX construction.
    """

    created: ClassVar[list[_RecordingClient]] = []

    def __init__(
        self, *, base_url: str, token: str | None = None, timeout: float | None = None
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.closed = False
        type(self).created.append(self)

    def close(self) -> None:
        self.closed = True


class _RecordingGraphReader(_RecordingClient):
    """Stands in for `build_graph_client`, which also takes the UN backend's settings."""

    created: ClassVar[list[_RecordingGraphReader]] = []

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        generation: object | None = None,
        un_base_url: str | None = None,
        un_availability: _RecordingClient | None = None,
    ) -> None:
        super().__init__(base_url=base_url, token=token)
        self.generation = generation
        self.un_base_url = un_base_url
        self.un_availability = un_availability

    def close(self) -> None:
        super().close()
        if self.un_availability is not None:
            self.un_availability.close()


class _RecordingSdmxClient(_RecordingClient):
    created: ClassVar[list[_RecordingSdmxClient]] = []


class _RecordingTaskClient(LiveTaskClient):
    """`LiveTaskClient` stand-in that records `close()` without a Cloud Tasks client."""

    created: ClassVar[list[_RecordingTaskClient]] = []

    def __init__(self) -> None:
        super().__init__(
            client=object(),  # never read: close() is overridden below
            queue="projects/p/locations/l/queues/q",
            worker_url="https://worker.example.org",
            worker_invoker_service_account="worker-invoker@example.iam.gserviceaccount.com",
        )
        self.closed = False
        type(self).created.append(self)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_recordings():
    _RecordingGraphReader.created.clear()
    _RecordingSdmxClient.created.clear()
    _RecordingTaskClient.created.clear()
    yield


_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}

_TASK_CLIENT_ENV = {
    "DFO_WORKER_URL": "https://worker.example.org",
    "DFO_WORKER_INVOKER_SERVICE_ACCOUNT": "worker-invoker@example.iam.gserviceaccount.com",
    "DFO_TASKS_QUEUE": "projects/p/locations/l/queues/q",
}


def _build_uninjected_door(monkeypatch: pytest.MonkeyPatch, *, graph_token: str = ""):
    """Builds a door whose lifespan creates graph and SDMX clients.
    Other dependencies are fakes.
    """
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(mint_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(mint_door, "SdmxClient", _RecordingSdmxClient)
    return build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        settings=DatasetsBuildSettings(graph_token=graph_token),
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        # Inject a fake task client. Live construction requires unavailable Google credentials.
        task_client=FakeTaskClient(),
    )


def _ours_sdmx() -> _RecordingSdmxClient:
    """The SDMX client on our mixer, beside the one the graph client holds for the UN."""
    (ours,) = [
        client
        for client in _RecordingSdmxClient.created
        if client.base_url == _SETTINGS_ENV["DATASETS_SDMX_BASE_URL"]
    ]
    return ours


def _enter_and_exit(door) -> None:
    async def _run() -> None:
        async with door.lifespan(None):
            pass

    asyncio.run(_run())


def test_configured_graph_token_reaches_both_live_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    door = _build_uninjected_door(monkeypatch, graph_token="shh-secret")

    _enter_and_exit(door)

    assert _RecordingGraphReader.created[0].token == "shh-secret"
    assert _ours_sdmx().token == "shh-secret"
    # The UN deployment's availability client is unauthenticated, so our token never leaves.
    un_availability = _RecordingGraphReader.created[0].un_availability
    assert un_availability is not None
    assert (un_availability.base_url, un_availability.token) == (UN_BASE_URL, None)


def test_unset_graph_token_reaches_both_live_clients_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    door = _build_uninjected_door(monkeypatch, graph_token="")

    _enter_and_exit(door)

    assert _RecordingGraphReader.created[0].token == ""
    assert _ours_sdmx().token == ""


def test_lifespan_closes_the_live_clients_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    door = _build_uninjected_door(monkeypatch)

    _enter_and_exit(door)

    assert _RecordingGraphReader.created[0].closed
    assert _ours_sdmx().closed


def test_mint_door_lifespan_closes_the_live_task_client_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan closes a live task client that it creates."""
    for var, value in _TASK_CLIENT_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(mint_door, "build_live_task_client", lambda **_: _RecordingTaskClient())

    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        graph=_RecordingGraphReader(base_url="https://graph.example.org"),
        sdmx=_RecordingSdmxClient(base_url="https://sdmx.example.org"),
        # An uninjected task client makes this lifespan build a live client.
    )

    _enter_and_exit(door)

    assert _RecordingTaskClient.created[0].closed


def test_definition_door_lifespan_closes_the_live_task_client_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in _TASK_CLIENT_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(
        definition_door, "build_live_task_client", lambda **_: _RecordingTaskClient()
    )

    door = build_definition_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        graph=_RecordingGraphReader(base_url="https://graph.example.org"),
        sdmx=_RecordingSdmxClient(base_url="https://sdmx.example.org"),
        # An uninjected task client makes this lifespan build a live client.
    )

    _enter_and_exit(door)

    assert _RecordingTaskClient.created[0].closed


def test_lifespan_does_not_close_an_injected_task_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-injected `task_client` (`FakeTaskClient()`, as most tests
    in this module use) is never closed by the door's own shutdown --
    mirroring the injected-`graph`/`sdmx` discipline just below."""
    door = _build_uninjected_door(monkeypatch)

    _enter_and_exit(door)

    assert _RecordingTaskClient.created == []


def test_construction_failure_closes_door_built_clients_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`graph`/`sdmx` build live first. The artifact store's own
    `shared_pool()` call, further down the same construction block, then
    raises. Both already-built clients must still be closed before the
    exception reaches the caller.
    """
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(mint_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(mint_door, "SdmxClient", _RecordingSdmxClient)

    def _raise_shared_pool():
        raise RuntimeError("shared_pool exploded")

    monkeypatch.setattr(mint_door, "shared_pool", _raise_shared_pool)

    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        version_writer=NullVersionRecordWriter(),
        # graph, sdmx, artifact_store all left uninjected.
    )

    with pytest.raises(RuntimeError, match="shared_pool exploded"):
        _enter_and_exit(door)

    assert _RecordingGraphReader.created[0].closed
    assert _ours_sdmx().closed


def test_construction_failure_does_not_close_injected_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A construction failure leaves caller-injected graph and SDMX clients open."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)

    def _raise_shared_pool():
        raise RuntimeError("shared_pool exploded")

    monkeypatch.setattr(mint_door, "shared_pool", _raise_shared_pool)

    injected_graph = _RecordingGraphReader(base_url="https://graph.example.org")
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        version_writer=NullVersionRecordWriter(),
        graph=injected_graph,
        sdmx=injected_sdmx,
        # A live artifact store makes this branch call `shared_pool()`, which raises.
    )

    with pytest.raises(RuntimeError, match="shared_pool exploded"):
        _enter_and_exit(door)

    assert not injected_graph.closed
    assert not injected_sdmx.closed


def test_mint_door_construction_failure_in_registry_branch_closes_shared_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry-construction failure releases the pool borrowed by that branch."""
    recorder = _patch_shared_pool_interface(monkeypatch, mint_door)

    def _raise_live_spec_registry(*, store):
        raise RuntimeError("LiveSpecRegistry exploded")

    monkeypatch.setattr(mint_door, "LiveSpecRegistry", _raise_live_spec_registry)

    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        graph=_RecordingGraphReader(base_url="https://graph.example.org"),
        sdmx=_RecordingSdmxClient(base_url="https://sdmx.example.org"),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        # Missing registry and spec reader dependencies make this branch borrow `shared_pool()`.
    )

    with pytest.raises(RuntimeError, match="LiveSpecRegistry exploded"):
        _enter_and_exit(door)

    assert recorder.open_calls == 1
    assert recorder.close_calls == 1


def test_definition_door_construction_failure_in_spec_reader_branch_closes_shared_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec-reader construction failure releases its borrowed pool."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    recorder = _patch_shared_pool_interface(monkeypatch, definition_door)

    def _raise_live_spec_reader(*, store):
        raise RuntimeError("LiveSpecReader exploded")

    monkeypatch.setattr(definition_door, "LiveSpecReader", _raise_live_spec_reader)

    door = build_definition_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        graph=_RecordingGraphReader(base_url="https://graph.example.org"),
        sdmx=_RecordingSdmxClient(base_url="https://sdmx.example.org"),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        # Without a spec reader, this branch borrows `shared_pool()`.
    )

    with pytest.raises(RuntimeError, match="LiveSpecReader exploded"):
        _enter_and_exit(door)

    assert recorder.open_calls == 1
    assert recorder.close_calls == 1


def test_register_door_construction_failure_after_shared_pool_closes_it_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later constructor failure releases a successful pool borrow once."""
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    recorder = _patch_shared_pool_interface(monkeypatch, register_door)

    def _raise_live_spec_registry(*, store):
        raise RuntimeError("LiveSpecRegistry exploded")

    monkeypatch.setattr(register_door, "LiveSpecRegistry", _raise_live_spec_registry)

    door = build_register_door()

    with pytest.raises(RuntimeError, match="LiveSpecRegistry exploded"):
        _enter_and_exit(door)

    assert recorder.open_calls == 1
    assert recorder.close_calls == 1


def test_register_door_construction_failure_in_shared_pool_itself_does_not_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed pool acquisition must not close a borrow held by another door."""
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    recorder = _patch_shared_pool_interface(monkeypatch, register_door)

    def _raise_shared_pool():
        raise RuntimeError("shared_pool exploded")

    monkeypatch.setattr(register_door, "shared_pool", _raise_shared_pool)

    door = build_register_door()

    with pytest.raises(RuntimeError, match="shared_pool exploded"):
        _enter_and_exit(door)

    assert recorder.close_calls == 0


def test_lifespan_does_not_close_injected_clients() -> None:
    injected_graph = _RecordingGraphReader(base_url="https://graph.example.org")
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        graph=injected_graph,
        sdmx=injected_sdmx,
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert not injected_graph.closed
    assert not injected_sdmx.closed


class _RecordingSharedPool:
    """Counts pool open and close calls without a database connection."""

    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0

    def shared_pool(self) -> object:
        self.open_calls += 1
        return object()

    def close_shared_pool(self) -> None:
        self.close_calls += 1


def _patch_shared_pool_interface(monkeypatch: pytest.MonkeyPatch, *modules) -> _RecordingSharedPool:
    """Patches each door module's imported pool functions and returns a call recorder."""
    recorder = _RecordingSharedPool()
    for module in modules:
        monkeypatch.setattr(module, "shared_pool", recorder.shared_pool)
        monkeypatch.setattr(module, "close_shared_pool", recorder.close_shared_pool)
    return recorder


def test_surface_closes_shared_pool_when_registry_injected_but_mint_deps_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the registry injected, the other seven live doors each borrow and close the shared
    pool."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    recorder = _patch_shared_pool_interface(
        monkeypatch,
        mint_door,
        register_door,
        definition_door,
        snapshot_door,
        pin_door,
        publish_door,
        meta_door,
        observations_door,
    )
    monkeypatch.setattr(mint_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(definition_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(observations_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(observations_door, "SdmxClient", _RecordingSdmxClient)
    monkeypatch.setattr(snapshot_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(pin_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(publish_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(meta_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(observations_door, "LiveArtifactStore", RecordingArtifactStore)
    injected_graph = _RecordingGraphReader(base_url="https://graph.example.org")
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")

    app = make_app(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        graph=injected_graph,
        sdmx=injected_sdmx,
        # Mint, definition, snapshot, observations, pin, publish, and meta build live
        # dependencies over `shared_pool()`.
    )

    with TestClient(app):
        assert recorder.open_calls > 0

    # Mint, definition, snapshot, observations, pin, publish, and meta each open and close
    # one borrow. Register uses the injected registry.
    assert recorder.close_calls == 7


def test_surface_closes_shared_pool_from_every_door_without_error_when_nothing_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All eight live doors close the shared pool safely during shutdown."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    recorder = _patch_shared_pool_interface(
        monkeypatch,
        mint_door,
        register_door,
        definition_door,
        snapshot_door,
        pin_door,
        publish_door,
        meta_door,
        observations_door,
    )
    monkeypatch.setattr(mint_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(mint_door, "SdmxClient", _RecordingSdmxClient)
    monkeypatch.setattr(mint_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(definition_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(definition_door, "SdmxClient", _RecordingSdmxClient)
    monkeypatch.setattr(definition_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(observations_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(observations_door, "SdmxClient", _RecordingSdmxClient)
    monkeypatch.setattr(snapshot_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(pin_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(publish_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(meta_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(observations_door, "LiveArtifactStore", RecordingArtifactStore)

    app = make_app(registry=None)  # every dependency on every door is live

    with TestClient(app):
        assert recorder.open_calls > 0

    assert recorder.close_calls == 8  # one close from each of the eight live doors


def test_lifespan_does_not_touch_shared_pool_when_everything_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _patch_shared_pool_interface(monkeypatch, mint_door)
    injected_graph = _RecordingGraphReader(base_url="https://graph.example.org")
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        graph=injected_graph,
        sdmx=injected_sdmx,
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert recorder.open_calls == 0
    assert recorder.close_calls == 0


def test_definition_door_lifespan_does_not_touch_shared_pool_when_everything_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _patch_shared_pool_interface(monkeypatch, definition_door)
    injected_graph = _RecordingGraphReader(base_url="https://graph.example.org")
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")
    door = build_definition_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        graph=injected_graph,
        sdmx=injected_sdmx,
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert recorder.open_calls == 0
    assert recorder.close_calls == 0


class _RecordingLiveDatasetStore:
    """Records constructed stores and their connection objects."""

    created: ClassVar[list[_RecordingLiveDatasetStore]] = []

    def __init__(self, *, connections: object, borrow_timeout: float = 30.0) -> None:
        self.connections = connections
        type(self).created.append(self)


class _RecordingLiveVersionWriter:
    """Records the store used to construct the version writer."""

    created: ClassVar[list[_RecordingLiveVersionWriter]] = []

    def __init__(self, *, store: object) -> None:
        self.store = store
        type(self).created.append(self)

    def record_version(self, record: object, *, trace_id: str | None = None) -> None:
        pass

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: object,
        trace_id: str | None = None,
    ) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_version_writer_recordings():
    _RecordingLiveDatasetStore.created.clear()
    _RecordingLiveVersionWriter.created.clear()
    yield


def test_lifespan_wires_a_live_version_writer_sharing_the_store_it_already_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default version writer reuses the store built for the registry and spec reader."""
    recorder = _patch_shared_pool_interface(monkeypatch, mint_door)
    monkeypatch.setattr(mint_door, "LiveDatasetStore", _RecordingLiveDatasetStore)
    monkeypatch.setattr(mint_door, "LiveVersionRecordWriter", _RecordingLiveVersionWriter)
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        artifact_store=FakeArtifactStore(),
        graph=_RecordingGraphReader(base_url="https://graph.example.org"),
        sdmx=_RecordingSdmxClient(base_url="https://sdmx.example.org"),
        # registry, spec_reader, and version_writer all left uninjected.
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert recorder.open_calls == 1
    assert len(_RecordingLiveDatasetStore.created) == 1
    assert len(_RecordingLiveVersionWriter.created) == 1
    assert _RecordingLiveVersionWriter.created[0].store is _RecordingLiveDatasetStore.created[0]


def test_lifespan_leaves_the_version_writer_null_when_nothing_else_forces_a_live_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a live store, the default version writer stays a no-op."""
    recorder = _patch_shared_pool_interface(monkeypatch, mint_door)
    monkeypatch.setattr(mint_door, "LiveVersionRecordWriter", _RecordingLiveVersionWriter)
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        graph=_RecordingGraphReader(base_url="https://graph.example.org"),
        sdmx=_RecordingSdmxClient(base_url="https://sdmx.example.org"),
        # version_writer left uninjected.
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert recorder.open_calls == 0
    assert _RecordingLiveVersionWriter.created == []


def test_lifespan_wires_a_live_version_writer_sharing_the_artifact_inventory_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The version writer shares the store built for the artifact inventory."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    recorder = _patch_shared_pool_interface(monkeypatch, mint_door)
    monkeypatch.setattr(mint_door, "LiveDatasetStore", _RecordingLiveDatasetStore)
    monkeypatch.setattr(mint_door, "LiveVersionRecordWriter", _RecordingLiveVersionWriter)
    monkeypatch.setattr(mint_door, "LiveArtifactStore", RecordingArtifactStore)
    monkeypatch.setattr(mint_door, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(mint_door, "SdmxClient", _RecordingSdmxClient)
    door = build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        # A fake task client keeps live Cloud Tasks construction out of this test.
        task_client=FakeTaskClient(),
    )

    _enter_and_exit(door)

    assert recorder.open_calls == 1
    assert len(_RecordingLiveDatasetStore.created) == 1
    assert len(_RecordingLiveVersionWriter.created) == 1
    assert _RecordingLiveVersionWriter.created[0].store is _RecordingLiveDatasetStore.created[0]


class _RecordingBlockingGraphReader(BlockingFakeMintGraphReader):
    """Blocking graph fake that records when its read returns and when `close()` runs."""

    def __init__(self, delegate: FakeMintGraphReader) -> None:
        super().__init__(delegate)
        self.closed = False
        self.unblocked_at: float | None = None
        self.closed_at: float | None = None

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        result = super().node_arc_values(dcid, properties=properties)
        self.unblocked_at = time.monotonic()
        return result

    def close(self) -> None:
        self.closed = True
        self.closed_at = time.monotonic()


def _build_door_with_live_graph_and_sdmx_patched(
    monkeypatch: pytest.MonkeyPatch, *, graph: _RecordingBlockingGraphReader, grace_seconds: float
) -> DoorContribution:
    """Builds a door whose lifespan creates the supplied blocking graph and a recording SDMX client."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(mint_door, "build_graph_client", lambda **_options: graph)
    monkeypatch.setattr(mint_door, "SdmxClient", _RecordingSdmxClient)
    return build_mint_door(
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=LiveSpecReader(store=FakeDatasetStore()),
        artifact_store=FakeArtifactStore(),
        version_writer=NullVersionRecordWriter(),
        pool_size=1,
        budget=ShutdownBudget(total_seconds=grace_seconds),
        task_client=FakeTaskClient(),
    )


def _mint_endpoint_of(
    door: DoorContribution,
) -> Callable[[MintRequest], Coroutine[None, None, object]]:
    """Returns the mint endpoint for direct invocation during lifespan tests."""
    routes = cast("list[APIRoute]", door.router.routes)
    endpoint = next(route.endpoint for route in routes if route.path == "/api/datasets")
    return cast(Callable[[MintRequest], Coroutine[None, None, object]], endpoint)


def test_lifespan_shutdown_drains_an_in_flight_mint_before_closing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown waits for an in-flight mint before closing its graph client and keeps the event loop
    responsive. A 10 ms heartbeat must avoid gaps of 150 ms while the graph read blocks for 300 ms.
    """
    blocking_graph = _RecordingBlockingGraphReader(_default_graph())
    door = _build_door_with_live_graph_and_sdmx_patched(
        monkeypatch, graph=blocking_graph, grace_seconds=5.0
    )
    mint_endpoint = _mint_endpoint_of(door)
    body = MintRequest.model_validate(_BODY)
    mint_finished = threading.Event()
    tick_gaps: list[float] = []

    def _run_mint_call() -> None:
        try:
            asyncio.run(mint_endpoint(body))
        except BaseException:
            # The recording SDMX client stops the mint after the graph read.
            # The test observes that read.
            pass
        finally:
            mint_finished.set()

    async def _heartbeat(stop: asyncio.Event) -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            tick_gaps.append(now - last)
            last = now

    async def _drive() -> None:
        stop = asyncio.Event()
        heartbeat_task = asyncio.ensure_future(_heartbeat(stop))
        async with door.lifespan(None):
            mint_thread = threading.Thread(target=_run_mint_call)
            mint_thread.start()
            await asyncio.to_thread(blocking_graph.entered.wait, 5.0)
            # Release the graph read after shutdown begins draining.
            threading.Timer(0.3, blocking_graph.release.set).start()
        mint_thread.join(timeout=5.0)
        # Allow another heartbeat tick to record any gap across the drain.
        await asyncio.sleep(0.02)
        stop.set()
        await heartbeat_task

    asyncio.run(_drive())

    assert mint_finished.is_set(), "the in-flight mint call never finished"
    assert blocking_graph.unblocked_at is not None, "the blocking graph read never unblocked"
    assert blocking_graph.closed, "shutdown must still close the clients it built"
    assert blocking_graph.closed_at is not None
    assert blocking_graph.closed_at >= blocking_graph.unblocked_at, (
        "the graph client was closed before the in-flight mint call that was still using it "
        "had finished"
    )
    assert tick_gaps, "the heartbeat never ticked"
    assert max(tick_gaps) < 0.15, (
        f"heartbeat froze for {max(tick_gaps):.3f}s. The shutdown drain blocked the event "
        "loop and should have run off it on a thread"
    )


def test_lifespan_shutdown_abandons_a_mint_call_that_outlives_the_grace_period(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the grace period expires, shutdown logs the unfinished mint and closes its clients."""
    blocking_graph = _RecordingBlockingGraphReader(_default_graph())
    door = _build_door_with_live_graph_and_sdmx_patched(
        monkeypatch, graph=blocking_graph, grace_seconds=0.05
    )
    mint_endpoint = _mint_endpoint_of(door)
    body = MintRequest.model_validate(_BODY)

    def _run_mint_call() -> None:
        with contextlib.suppress(BaseException):
            asyncio.run(mint_endpoint(body))

    async def _drive() -> threading.Thread:
        async with door.lifespan(None):
            mint_thread = threading.Thread(target=_run_mint_call, daemon=True)
            mint_thread.start()
            await asyncio.to_thread(blocking_graph.entered.wait, 5.0)
            # Leave the graph read blocked until after the grace period.
        return mint_thread

    with caplog.at_level(logging.ERROR):
        mint_thread = asyncio.run(_drive())

    assert blocking_graph.closed, "the grace period expiring must not block shutdown forever"
    assert "shared shutdown budget" in caplog.text

    blocking_graph.release.set()
    mint_thread.join(timeout=5.0)
