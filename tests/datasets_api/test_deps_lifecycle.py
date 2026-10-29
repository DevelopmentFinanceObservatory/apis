"""Tests `open_deps`: live construction, ownership of what it builds, one pool borrow, and the
shutdown drain. Recording clients expose token wiring, close calls, and shutdown order."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import ClassVar

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from datasets_api import deps as deps_module
from datasets_api import generation as generation_module
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import Deps, DepsOverrides, open_deps
from datasets_api.door_executor import DoorExecutor
from datasets_api.operations.task_client import FakeTaskClient, LiveTaskClient
from datasets_api.shutdown import ShutdownBudget
from dc_sdmx import DATA_HEDGE_AFTER_SECONDS
from graph_generation.fake import FakeGeneration
from graph_release.service import get_release_service
from materialise.rate_tables import RateTableCache
from origin_catalogue.graph import (
    ARCS_CACHE_SIZE,
    LABEL_TYPE_CACHE_SIZE,
    OBS_CACHE_SIZE,
    OBS_PROPERTIES_CACHE_SIZE,
    UN_BASE_URL,
)
from series_coverage.shared import get_coverage_service
from tests.code_index.conftest import FIXTURE_PATH
from tests.code_index.support import load_index
from tests.graph_release.support import serving_index

from ._artifact_doubles import FakeArtifactStore, RecordingArtifactStore

_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
    "DFO_WORKER_URL": "https://worker.example.org",
    "DFO_WORKER_INVOKER_SERVICE_ACCOUNT": "worker-invoker@example.iam.gserviceaccount.com",
    "DFO_TASKS_QUEUE": "projects/p/locations/l/queues/q",
}


class _RecordingClient:
    """Recording stand-in for the SDMX client. Accepts `timeout`, `hedge_after` and
    `total_concurrency` as `SdmxClient` does."""

    created: ClassVar[list[_RecordingClient]] = []

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout: float | None = None,
        hedge_after: float | None = None,
        total_concurrency: int | None = None,
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.hedge_after = hedge_after
        self.total_concurrency = total_concurrency
        self.closed = False
        self.closed_at: float | None = None
        type(self).created.append(self)

    def close(self) -> None:
        self.closed = True
        self.closed_at = time.monotonic()


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
        **cache_sizes: int,
    ) -> None:
        super().__init__(base_url=base_url, token=token)
        self.cache_sizes = cache_sizes
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


class _RecordingPool:
    """Counts pool borrows and releases and the stores built over them."""

    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self.stores: list[object] = []

    def shared_pool(self) -> object:
        self.open_calls += 1
        return object()

    def close_shared_pool(self) -> None:
        self.close_calls += 1

    def live_dataset_store(self, *, connections: object) -> object:
        store = object()
        self.stores.append(store)
        return store


@pytest.fixture(autouse=True)
def _live_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    for recorder in (_RecordingGraphReader, _RecordingSdmxClient, _RecordingTaskClient):
        recorder.created.clear()
    monkeypatch.setattr(deps_module, "build_graph_client", _RecordingGraphReader)
    monkeypatch.setattr(deps_module, "SdmxClient", _RecordingSdmxClient)
    monkeypatch.setattr(deps_module, "build_live_task_client", lambda **_: _RecordingTaskClient())
    monkeypatch.setattr(deps_module, "LiveArtifactStore", RecordingArtifactStore)
    shared = FakeGeneration()
    monkeypatch.setattr(generation_module, "acquire_shared", lambda **_kwargs: shared)
    monkeypatch.setattr(generation_module, "release_shared", lambda _generation: None)


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> _RecordingPool:
    recorder = _RecordingPool()
    monkeypatch.setattr(deps_module, "shared_pool", recorder.shared_pool)
    monkeypatch.setattr(deps_module, "close_shared_pool", recorder.close_shared_pool)
    monkeypatch.setattr(deps_module, "LiveDatasetStore", recorder.live_dataset_store)
    return recorder


def _overrides(*, code_index: CodeIndex | None = None, **injected: object) -> DepsOverrides:
    return DepsOverrides(
        coverage=get_coverage_service(),
        rate_cache=RateTableCache(),
        release=get_release_service(),
        code_index=code_index,
        **injected,
    )


def _every_store_injected() -> dict[str, object]:
    """Store doubles for every store field, so `open_deps` borrows no pool."""
    return {
        name: object()
        for name in (
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
    } | {"artifact_store": FakeArtifactStore()}


def _open_and_close(
    overrides: DepsOverrides, *, settings: DatasetsBuildSettings | None = None
) -> Deps:
    async def _run() -> Deps:
        async with open_deps(
            settings=settings or DatasetsBuildSettings.from_env(),
            overrides=overrides,
            catalogue=None,
        ) as deps:
            return deps

    return asyncio.run(_run())


def _ours_sdmx() -> _RecordingSdmxClient:
    """The SDMX client on our mixer, beside the one the graph client holds for the UN."""
    (ours,) = [
        client
        for client in _RecordingSdmxClient.created
        if client.base_url == _SETTINGS_ENV["DATASETS_SDMX_BASE_URL"]
    ]
    return ours


@pytest.mark.parametrize("token", ["shh-secret", ""])
def test_the_graph_token_reaches_both_live_clients_and_never_the_un(
    pool: _RecordingPool, token: str
) -> None:
    _open_and_close(
        _overrides(**_every_store_injected(), task_client=FakeTaskClient()),
        settings=DatasetsBuildSettings(graph_token=token),
    )

    (graph,) = _RecordingGraphReader.created
    assert graph.token == token
    assert _ours_sdmx().token == token
    # The UN deployment's availability client is unauthenticated, so our token never leaves.
    assert graph.un_availability is not None
    assert (graph.un_availability.base_url, graph.un_availability.token) == (UN_BASE_URL, None)


def test_live_clients_take_the_sdmx_timeout_and_searchs_graph_cache_sizes(
    pool: _RecordingPool,
) -> None:
    _open_and_close(
        _overrides(**_every_store_injected(), task_client=FakeTaskClient()),
        settings=DatasetsBuildSettings(sdmx_timeout_seconds=42, mint_member_max_concurrency=12),
    )

    assert [client.timeout for client in _RecordingSdmxClient.created] == [42, 42]
    # Only the mint's client on our mixer hedges its data reads, and it allows as many GETs
    # in flight as a mint has fetch workers.
    assert {
        client.base_url: (client.hedge_after, client.total_concurrency)
        for client in _RecordingSdmxClient.created
    } == {
        _SETTINGS_ENV["DATASETS_SDMX_BASE_URL"]: (DATA_HEDGE_AFTER_SECONDS, 12),
        UN_BASE_URL: (None, None),
    }
    (graph,) = _RecordingGraphReader.created
    assert graph.cache_sizes == {
        "label_cache_size": LABEL_TYPE_CACHE_SIZE,
        "type_cache_size": LABEL_TYPE_CACHE_SIZE,
        "arcs_cache_size": ARCS_CACHE_SIZE,
        "obs_properties_cache_size": OBS_PROPERTIES_CACHE_SIZE,
        "obs_cache_size": OBS_CACHE_SIZE,
    }


def test_the_live_clients_built_here_are_closed_and_injected_ones_are_not(
    pool: _RecordingPool,
) -> None:
    _open_and_close(_overrides(**_every_store_injected()))
    (graph,) = _RecordingGraphReader.created
    (task_client,) = _RecordingTaskClient.created
    assert graph.closed
    assert _ours_sdmx().closed
    assert task_client.closed

    injected_graph = _RecordingGraphReader(base_url="https://graph.example.org")
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")
    injected_tasks = _RecordingTaskClient()
    _open_and_close(
        _overrides(
            **_every_store_injected(),
            graph=injected_graph,
            sdmx=injected_sdmx,
            task_client=injected_tasks,
        )
    )
    assert not injected_graph.closed
    assert not injected_sdmx.closed
    assert not injected_tasks.closed


def test_a_build_failure_closes_what_was_built_and_leaves_injected_clients_open(
    monkeypatch: pytest.MonkeyPatch, pool: _RecordingPool
) -> None:
    """The artifact store is built after the pool, the graph, the SDMX client, and the task
    client. Its failure closes every one of those before the exception propagates."""

    def _explode(**_kwargs: object) -> object:
        raise RuntimeError("artifact store exploded")

    monkeypatch.setattr(deps_module, "LiveArtifactStore", _explode)
    injected_sdmx = _RecordingSdmxClient(base_url="https://sdmx.example.org")

    with pytest.raises(RuntimeError, match="artifact store exploded"):
        _open_and_close(_overrides(sdmx=injected_sdmx))

    (graph,) = _RecordingGraphReader.created
    (task_client,) = _RecordingTaskClient.created
    assert graph.closed
    assert task_client.closed
    assert not injected_sdmx.closed
    assert (pool.open_calls, pool.close_calls) == (1, 1)


def test_a_failed_pool_borrow_releases_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed borrow must not release one another surface holds."""
    releases: list[None] = []

    def _raise_shared_pool() -> object:
        raise RuntimeError("shared_pool exploded")

    monkeypatch.setattr(deps_module, "shared_pool", _raise_shared_pool)
    monkeypatch.setattr(deps_module, "close_shared_pool", lambda: releases.append(None))

    with pytest.raises(RuntimeError, match="shared_pool exploded"):
        _open_and_close(_overrides())

    assert releases == []


def test_every_omitted_store_shares_one_pool_borrow_and_one_store(pool: _RecordingPool) -> None:
    deps = _open_and_close(_overrides(task_client=FakeTaskClient()))

    assert (pool.open_calls, pool.close_calls) == (1, 1)
    (store,) = pool.stores
    for name in (
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
    ):
        assert getattr(deps, name)._store is store, name
    assert isinstance(deps.artifact_store, RecordingArtifactStore)
    assert deps.artifact_store.inventory._store is store


def test_a_surface_with_every_store_injected_borrows_no_pool(pool: _RecordingPool) -> None:
    _open_and_close(_overrides(**_every_store_injected(), task_client=FakeTaskClient()))

    assert (pool.open_calls, pool.close_calls) == (0, 0)


class _BlockingJob:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.released = False
        self.finished_at: float | None = None

    def __call__(self) -> None:
        self.entered.set()
        self.released = self.release.wait(timeout=5.0)
        self.finished_at = time.monotonic()


def _run_through_shutdown(
    job: _BlockingJob, monkeypatch: pytest.MonkeyPatch, *, release_when_draining: bool
) -> tuple[_RecordingGraphReader, list[int], int]:
    """Submit `job` to the work executor and shut down. With `release_when_draining`, the
    drain asks the event loop to release `job`, so `job` is released only if the loop runs
    while the drain waits. Returns the live graph, the threads each drain ran on, and the
    event loop's thread."""
    drain_threads: list[int] = []
    drain = DoorExecutor.drain

    async def _drive() -> int:
        loop = asyncio.get_running_loop()

        def _recording_drain(executor: DoorExecutor, budget: ShutdownBudget) -> None:
            drain_threads.append(threading.get_ident())
            if release_when_draining:
                loop.call_soon_threadsafe(job.release.set)
            drain(executor, budget)

        monkeypatch.setattr(DoorExecutor, "drain", _recording_drain)
        async with open_deps(
            settings=DatasetsBuildSettings.from_env(),
            overrides=_overrides(**_every_store_injected(), task_client=FakeTaskClient()),
            catalogue=None,
        ) as deps:
            call = asyncio.ensure_future(
                deps.work_executor.run(job, refuse=lambda: RuntimeError("full"))
            )
            await asyncio.to_thread(job.entered.wait, 5.0)
        job.release.set()
        await call
        return threading.get_ident()

    loop_thread = asyncio.run(_drive())
    (graph,) = _RecordingGraphReader.created
    return graph, drain_threads, loop_thread


def test_an_omitted_code_index_is_read_per_request_and_a_given_one_serves_every_request(
    pool: _RecordingPool,
) -> None:
    """The index `get_index()` returns can change between requests, as a graph release swap
    replaces it, so the deps read it on each access, party kinds included."""
    given, replacement = load_index(FIXTURE_PATH), load_index(FIXTURE_PATH)
    injected = {**_every_store_injected(), "task_client": FakeTaskClient()}
    live = _open_and_close(_overrides(**injected))
    fixed = _open_and_close(_overrides(code_index=given, **injected))

    with serving_index(replacement):
        assert live.code_index is replacement
        assert live.enrichment.parties is replacement.parties
        assert fixed.code_index is given
        assert fixed.enrichment.parties is given.parties
    assert live.code_index is get_index()


def test_shutdown_drains_running_work_before_closing_clients_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, pool: _RecordingPool
) -> None:
    """The drain waits on a worker thread for a job only the event loop can release, then
    closes the clients."""
    job = _BlockingJob()

    graph, drain_threads, loop_thread = _run_through_shutdown(
        job, monkeypatch, release_when_draining=True
    )

    assert drain_threads, "shutdown never drained the executors"
    assert loop_thread not in drain_threads, "the drain ran on the event-loop thread"
    assert job.released, "the event loop stalled while the drain waited for the job"
    assert job.finished_at is not None
    assert graph.closed_at is not None
    assert graph.closed_at >= job.finished_at, "the graph closed under running work"


def test_shutdown_closes_clients_once_the_budget_expires_and_logs_the_held_permits(
    monkeypatch: pytest.MonkeyPatch, pool: _RecordingPool, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(deps_module, "ShutdownBudget", lambda: ShutdownBudget(total_seconds=0.05))
    job = _BlockingJob()

    with caplog.at_level(logging.WARNING):
        graph, _drain_threads, _loop_thread = _run_through_shutdown(
            job, monkeypatch, release_when_draining=False
        )

    assert graph.closed
    assert "datasets-work executor's shutdown drain timed out" in caplog.text
