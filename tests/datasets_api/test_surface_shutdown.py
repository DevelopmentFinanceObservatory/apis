"""Pins the composed surface's shutdown: one pool release, the memo registration, and the shared
time budget. The app builds live dependencies over a faked database and external clients."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from artifact_store import Tier
from dataset_store import provider
from dataset_store.pool import ConnectionPool
from datasets_api import deps as deps_module
from datasets_api import generation as generation_module
from datasets_api.app import datasets_surface
from datasets_api.operation_store import NullOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.shutdown import SHUTDOWN_BUDGET_SECONDS
from datasets_api.versions import NullHeadWriter, NullVersionRecordWriter
from dfo_platform.app import create_app
from graph_generation.fake import FakeGeneration
from materialise import facet_profile, shared_profiles
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity

from ._app_factories import offline_recipe_stores
from ._artifact_doubles import FakeArtifactStore, RecordingArtifactStore
from ._identity_builders import pinned_identity, register_body
from ._snapshot_doubles import (
    FakeSnapshotReader,
    FakeSpecReader,
    _store_and_locate,
    _version,
)
from ._store_fault_doubles import BlockingFakeDatasetStore
from .conftest import wire_definition_for
from .test_deps_lifecycle import (
    _RecordingGraphReader,
    _RecordingSdmxClient,
    _RecordingTaskClient,
)
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


def _build_fully_live_app(monkeypatch: pytest.MonkeyPatch):
    """Compose the surface with live dependencies, as `deployed_surfaces()` does."""
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
    monkeypatch.setattr("dataset_store.connection.connect", lambda **kwargs: _FakeConnection())
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    # The process's profile cache builds its own graph and SDMX clients.
    for module in (deps_module, shared_profiles):
        monkeypatch.setattr(module, "build_graph_client", _RecordingGraphReader)
        monkeypatch.setattr(module, "SdmxClient", _RecordingSdmxClient)
    monkeypatch.setattr(deps_module, "LiveArtifactStore", RecordingArtifactStore)
    # The Cloud Tasks constructor requires credentials unavailable to this offline test.
    monkeypatch.setattr(
        deps_module, "build_live_task_client", lambda **_kwargs: _RecordingTaskClient()
    )
    return create_app([datasets_surface()])


def test_the_surface_releases_its_one_pool_borrow_after_a_register_in_flight_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A register worker holds a pool connection while the lifespan shuts down. The pool closes
    once, after the worker's connection returns."""
    app = _build_fully_live_app(monkeypatch)
    worker_done: list[float] = []
    close_calls: list[tuple[ConnectionPool, float]] = []
    original_close_all = ConnectionPool.close_all

    def _spy_close_all(self: ConnectionPool) -> None:
        close_calls.append((self, time.monotonic()))
        original_close_all(self)

    monkeypatch.setattr(ConnectionPool, "close_all", _spy_close_all)
    worker_entered = threading.Event()

    def _register_worker(pool: ConnectionPool) -> None:
        with pool.borrow():
            worker_entered.set()
            time.sleep(0.2)
        worker_done.append(time.monotonic())

    with TestClient(app):
        pool = provider._pool
        assert pool is not None
        worker = threading.Thread(target=_register_worker, args=(pool,))
        worker.start()
        assert worker_entered.wait(timeout=5.0), "worker never borrowed its connection"
        worker.join(timeout=5.0)

    assert [closed for closed, _at in close_calls] == [pool]
    assert worker_done and close_calls[0][1] >= worker_done[0]


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


def test_the_surface_acquires_one_facet_profile_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One Datasets process holds one profile cache, the process's shared one acquired with the
    surface's dependencies, so a mint and a definition rebuild of one spec share its reads and
    its memory bound."""
    built: list[facet_profile.FacetProfileCache] = []
    real_cache = facet_profile.FacetProfileCache

    def recording_cache(**kwargs) -> facet_profile.FacetProfileCache:
        built.append(real_cache(**kwargs))
        return built[-1]

    monkeypatch.setattr(shared_profiles, "FacetProfileCache", recording_cache)
    monkeypatch.setattr(generation_module, "acquire_shared", lambda **_kwargs: FakeGeneration())
    monkeypatch.setattr(generation_module, "release_shared", lambda _generation: None)
    app = _build_fully_live_app(monkeypatch)

    with TestClient(app):
        pass

    assert len(built) == 1


def test_composed_surface_shutdown_fits_the_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two blocked calls on the read executor. Teardown must stay under twice the shared budget.

    The 2x allowance tolerates slow CI while distinguishing a budget per drain."""
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
                head_writer=NullHeadWriter(),
                operation_store=NullOperationStore(),
                task_client=FakeTaskClient(),
                **offline_recipe_stores(),
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
