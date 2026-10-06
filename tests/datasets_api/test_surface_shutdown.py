"""Pins shutdown order, shared-pool reference counting, and the shared time budget.
The app builds live doors with faked database and external clients.
Also checks that the definition door drains an in-flight call."""

from __future__ import annotations

import asyncio
import importlib
import pkgutil
import threading
import time
from types import ModuleType
from typing import cast

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import Request

from artifact_store import Tier
from dataset_store import provider
from dataset_store.pool import ConnectionPool
from datasets_api import doors as doors_package
from datasets_api import generation as generation_module
from datasets_api.app import datasets_surface
from datasets_api.doors import mint as mint_door
from datasets_api.doors.definition import build_definition_door
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.shutdown import SHUTDOWN_BUDGET_SECONDS, ShutdownBudget
from datasets_api.versions import NullVersionRecordWriter
from dfo_platform.app import create_app
from graph_generation.fake import FakeGeneration
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity

from ._artifact_doubles import FakeArtifactStore, RecordingArtifactStore
from ._identity_builders import pinned_identity, register_body
from ._snapshot_doubles import (
    BlockingFakeSpecReader,
    FakeSnapshotReader,
    FakeSpecReader,
    _store_and_locate,
    _version,
)
from ._store_fault_doubles import BlockingFakeDatasetStore
from .conftest import wire_definition_for
from .test_mint_lifecycle import _RecordingGraphReader, _RecordingSdmxClient
from .test_pin_door import BlockingFakeArtifactReferenceWriter

_PRINCIPAL = "first-party-key"

_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
    "DFO_WORKER_URL": "https://worker.example.org",
    "DFO_WORKER_INVOKER_SERVICE_ACCOUNT": "worker-invoker@example.iam.gserviceaccount.com",
    "DFO_TASKS_QUEUE": "projects/example/locations/us-central1/queues/datasets-worker",
}


class _FakeConnection:
    """Minimal DB-API connection for the real pool."""

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_provider_singleton(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(provider, "_pool", None)
    monkeypatch.setattr(provider, "_connector", None)
    monkeypatch.setattr(provider, "_borrowers", 0)
    yield
    monkeypatch.setattr(provider, "_pool", None)
    monkeypatch.setattr(provider, "_connector", None)
    monkeypatch.setattr(provider, "_borrowers", 0)


def _door_modules() -> list[ModuleType]:
    """Return every door module so live constructors can be patched across the surface."""
    return [
        importlib.import_module(f"{doors_package.__name__}.{module_info.name}")
        for module_info in pkgutil.iter_modules(doors_package.__path__)
    ]


def _build_fully_live_app(monkeypatch: pytest.MonkeyPatch):
    """Compose the surface with live dependencies, as `deployed_surfaces()` does."""
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
    monkeypatch.setattr("dataset_store.connection.connect", lambda **kwargs: _FakeConnection())
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    live_patches = (
        ("build_graph_client", _RecordingGraphReader),
        ("SdmxClient", _RecordingSdmxClient),
        ("LiveArtifactStore", RecordingArtifactStore),
        # The Cloud Tasks constructor requires credentials unavailable to this offline test.
        (
            "build_live_task_client",
            lambda **_kwargs: FakeTaskClient(),
        ),
    )
    installed: set[str] = set()
    for door_module in _door_modules():
        for live_name, recording_fake in live_patches:
            if hasattr(door_module, live_name):
                monkeypatch.setattr(door_module, live_name, recording_fake)
                installed.add(live_name)
    missing = [live_name for live_name, _ in live_patches if live_name not in installed]
    assert not missing, (
        f"no door module has any of these live constructor names, so their "
        f"recording fakes were never installed -- the name has drifted: {missing}"
    )
    return create_app([datasets_surface()])


def test_register_worker_outlives_mints_early_close_and_pool_shuts_down_after_full_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A register worker holds a pool connection while the composed lifespan shuts down.

    The pool stays open after mint releases its borrow and closes after register releases its own."""
    app = _build_fully_live_app(monkeypatch)

    close_calls: list[ConnectionPool] = []
    original_close_all = ConnectionPool.close_all

    def _spy_close_all(self: ConnectionPool) -> None:
        close_calls.append(self)
        original_close_all(self)

    monkeypatch.setattr(ConnectionPool, "close_all", _spy_close_all)

    worker_entered = threading.Event()
    worker_release = threading.Event()
    worker_errors: list[BaseException] = []

    def _register_worker(pool: ConnectionPool) -> None:
        try:
            with pool.borrow():
                worker_entered.set()
                worker_release.wait(timeout=5.0)
        except BaseException as exc:
            worker_errors.append(exc)

    mid_teardown_same_pool: list[bool] = []
    worker_pool: list[ConnectionPool] = []
    real_mint_close_shared_pool = mint_door.close_shared_pool

    def _mint_close_then_probe() -> None:
        real_mint_close_shared_pool()
        # Register unwinds after mint and still owns its pool borrow.
        # Record startup failure as False so the assertion reports it.
        mid_teardown_same_pool.append(bool(worker_pool) and provider._pool is worker_pool[0])
        worker_release.set()

    monkeypatch.setattr(mint_door, "close_shared_pool", _mint_close_then_probe)

    with TestClient(app):
        pool = provider._pool
        assert pool is not None
        worker_pool.append(pool)
        worker = threading.Thread(target=_register_worker, args=(pool,))
        worker.start()
        assert worker_entered.wait(timeout=5.0), "worker never borrowed its connection"
        # Exiting this block drives the surface's own lifespan shutdown,
        # which is where _mint_close_then_probe runs.

    worker.join(timeout=5.0)

    assert not worker_errors, f"the register worker's pool use failed: {worker_errors}"
    assert mid_teardown_same_pool == [True], (
        "the pool must still be the worker's own pool immediately after mint's "
        "own close, before register's own close has run"
    )

    assert close_calls == [pool], (
        "the pool that closed should be the very one the worker held a connection from"
    )


def test_the_surface_registers_the_default_scope_memo_on_the_shared_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memo is registered while the surface serves and unregistered when it stops."""
    shared = FakeGeneration()
    monkeypatch.setattr(generation_module, "acquire_shared", lambda **_kwargs: shared)
    monkeypatch.setattr(generation_module, "release_shared", lambda _generation: None)
    app = _build_fully_live_app(monkeypatch)

    with TestClient(app):
        serving = shared.registered_names()

    assert serving == (generation_module.PLACE_SCOPE_MEMO_NAME,)
    assert shared.registered_names() == ()


def test_composed_surface_shutdown_fits_the_shared_budget_not_one_per_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two blocked doors expose per-door budgets. Teardown must stay under twice the shared budget.

    The 2x allowance tolerates slow CI while distinguishing a five-budget regression."""
    # The surface always validates `DatasetsSettings.from_env()`, even when door dependencies are injected.
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    blocking_store = BlockingFakeDatasetStore()
    blocking_writer = BlockingFakeArtifactReferenceWriter()
    artifact_store = FakeArtifactStore()
    # Pinned tier skips retention expiry, so the reference write can block in flight.
    location = _store_and_locate(artifact_store, b"shutdown-budget-data", tier=Tier.pinned)
    snapshot_reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    app = create_app(
        [
            datasets_surface(
                registry=LiveSpecRegistry(store=blocking_store),
                spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
                graph=object(),
                sdmx=object(),
                artifact_store=artifact_store,
                snapshot_reader=snapshot_reader,
                reference_writer=blocking_writer,
                version_writer=NullVersionRecordWriter(),
                pin_allowed_principals=frozenset({_PRINCIPAL}),
                # Inject `operation_store` to keep the meta door off the live shared pool.
                operation_store=NullOperationStore(),
                # Inject a task client because the live Cloud Tasks client requires credentials.
                task_client=FakeTaskClient(),
            )
        ]
    )

    identity = pinned_identity()
    spec_id = recompute_identity(identity).spec_id
    register_json = register_body(
        definition=wire_definition_for(spec_id=spec_id, identity=identity)
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        register_thread = threading.Thread(
            target=lambda: client.post("/api/datasets/register", json=register_json)
        )
        register_thread.start()
        assert blocking_store.entered.wait(timeout=5.0), (
            "the register call never reached the blocking store"
        )

        pin_thread = threading.Thread(
            target=lambda: client.post(
                "/api/datasets/spec_a/cv1/pin",
                json={"schema_version": 1, "artifact_digest": location.artifact_digest},
                headers={"x-dfo-principal": _PRINCIPAL},
            )
        )
        pin_thread.start()
        assert blocking_writer.entered.wait(timeout=5.0), (
            "the pin call never reached the blocking reference writer"
        )

        shutdown_start = time.monotonic()
    elapsed = time.monotonic() - shutdown_start

    assert elapsed < 2 * SHUTDOWN_BUDGET_SECONDS, (
        f"composed teardown took {elapsed:.1f}s -- expected it to fit inside "
        f"roughly one shared {SHUTDOWN_BUDGET_SECONDS:.1f}s budget, not one per door"
    )

    blocking_store.release.set()
    blocking_writer.release.set()
    register_thread.join(timeout=5.0)
    pin_thread.join(timeout=5.0)


def _definition_endpoint_of(door):
    routes = cast("list[APIRoute]", door.router.routes)
    # `path` includes the `dfo_spec_id` converter, so match on `path_format`.
    return next(route.endpoint for route in routes if route.path_format == "/api/datasets/{id}")


def _bare_get_request() -> Request:
    return Request(
        {"type": "http", "method": "GET", "path": "/", "query_string": b"", "headers": []}
    )


def test_definition_door_drain_waits_for_a_genuinely_in_flight_call_then_releases() -> None:
    """The drain waits for an in-flight retrieval and returns when the call releases, before budget expiry."""
    blocking_spec_reader = BlockingFakeSpecReader(spec_ids=["spec_a"])
    door = build_definition_door(
        spec_reader=blocking_spec_reader,
        graph=object(),
        sdmx=object(),
        artifact_store=FakeArtifactStore(),
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=50_000,
        member_concurrency=4,
        pool_size=1,
        budget=ShutdownBudget(total_seconds=5.0),
        task_client=FakeTaskClient(),
    )
    endpoint = _definition_endpoint_of(door)
    call_finished = threading.Event()

    def _run_call() -> None:
        try:
            asyncio.run(endpoint(id="spec_a", request=_bare_get_request()))
        except BaseException:
            pass
        finally:
            call_finished.set()

    drain_elapsed: list[float] = []

    async def _drive() -> None:
        async with door.lifespan(None):
            call_thread = threading.Thread(target=_run_call)
            call_thread.start()
            await asyncio.to_thread(blocking_spec_reader.entered.wait, 5.0)
            drain_start = time.monotonic()
            # Released only after this door's own shutdown has already
            # started draining -- exiting the `async with` below runs it.
            threading.Timer(0.2, blocking_spec_reader.release.set).start()
        drain_elapsed.append(time.monotonic() - drain_start)

    asyncio.run(_drive())

    assert call_finished.wait(timeout=5.0), "the in-flight retrieval never finished"
    assert drain_elapsed, "the drain's own timing was never recorded"
    assert 0.15 < drain_elapsed[0] < 2.0, (
        f"drain took {drain_elapsed[0]:.2f}s -- expected it to wait for the in-flight "
        "call's own ~0.2s release, not return instantly or wait out its own budget"
    )
