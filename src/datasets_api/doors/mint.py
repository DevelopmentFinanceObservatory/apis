"""The mint door: `POST /api/datasets`.

Mint calls run on a dedicated `ThreadPoolExecutor`, separate from the platform
and register-door pools. Its non-blocking semaphore uses
`DATASETS_MINT_MAX_CONCURRENCY`. A full pool returns retryable 503
`MintBackpressure`. `asyncio.wait_for` enforces `DATASETS_MINT_TIMEOUT_SECONDS`
and maps timeouts to retryable 504. Cancelling a queued future releases its
permit. A running thread continues after request cancellation and holds its
permit until completion. Graph and SDMX transport timeouts bound that work.

Shutdown cancels queued futures, then waits for running calls within the
shared `ShutdownBudget` before closing live clients and the pool. Calls that
outlast the deadline are logged and abandoned.

Callers can inject dependencies. Lifespan builds missing live clients from
`DatasetsSettings` and `shared_pool()`, then closes only what it owns. Graph and
SDMX clients use `DATASETS_GRAPH_TOKEN`. Injected clients belong to the caller.
Exception mapping and its inventory live in `datasets_api.doors.mint_errors`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import BrokenExecutor, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from typing import cast

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from code_index.query import CodeIndex
from code_index.shared import get_index
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.snapshot import LiveSnapshotReader, SnapshotReader
from datasets_api.config import DatasetsBuildSettings, DatasetsSettings
from datasets_api.door_executor import build_door_executor
from datasets_api.doors.mint_errors import (
    MINT_EXCEPTION_EXEMPLARS,
    MINT_EXCEPTION_TYPES,
    MintBackpressure,
    classify_mint_exception,
)
from datasets_api.generation import enter_shared_generation
from datasets_api.headers import no_store_headers
from datasets_api.kinds.spec_id import mint_spec_id
from datasets_api.kinds.statvar_set import mint_statvar_set
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintRequest, StatVarSetInput
from datasets_api.operation_store import LiveOperationStore, NullOperationStore, OperationStore
from datasets_api.operations.submit import (
    OperationSubmitted,
    operation_submitted_resource,
)
from datasets_api.operations.task_client import (
    LiveTaskClient,
    NullTaskClient,
    TaskClient,
    TaskClientSettings,
    build_live_task_client,
)
from datasets_api.render import check_options_supported
from datasets_api.revision import SentinelRevisionReader
from datasets_api.schema import OperationResource
from datasets_api.shutdown import ShutdownBudget
from datasets_api.timings import PhaseTimings, door_call, run_with_timings
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import (
    HeadStore,
    NullHeadWriter,
    NullVersionRecordWriter,
    VersionRecordWriter,
)
from dc_sdmx import SdmxClient
from dfo_platform.doors import DoorContribution
from dfo_platform.executor import submit_with_context
from dfo_platform.trace import get_trace_id
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise import EnrichmentLookup
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from materialise.revision import RevisionReader
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from origin_catalogue.service import CatalogueService
from series_coverage.shared import CoverageService, get_coverage_service
from spec_registry import LiveSpecRegistry, SpecRegistry
from spec_registry.reader import LiveSpecReader, SpecReader
from statvar_members import shared_enumerator

logger = logging.getLogger(__name__)


def _sdmx_deadline(timeout: float) -> float:
    """Reserve up to one second, or 10% of the timeout, for `wait_for` to unwind."""
    grace = min(1.0, timeout * 0.1)
    return time.monotonic() + max(0.0, timeout - grace)


@dataclass(frozen=True)
class _MintInjected:
    """Tracks which dependencies lifespan must build and close.

    Its fields mirror the factory arguments and record which dependencies were injected.
    """

    registry: bool
    spec_reader: bool
    graph: bool
    sdmx: bool
    artifact_store: bool
    task_client: bool
    version_writer: bool
    snapshot_reader: bool
    head_writer: bool
    operation_store: bool


@dataclass(frozen=True)
class _MintDoorState:
    """Shared mint-door state. Lifespan resolves `deps` and fixes other fields."""

    deps: MintDeps
    settings: DatasetsBuildSettings
    budget: ShutdownBudget
    executor: ThreadPoolExecutor
    semaphore: threading.BoundedSemaphore
    in_flight: set[Future]
    in_flight_lock: threading.Lock
    injected: _MintInjected
    catalogue: CatalogueService | None


@dataclass
class _MintStoreBorrow:
    """Tracks the lifespan's single `shared_pool()` borrow.

    Set `opened_here` as soon as the borrow succeeds, before store creation can
    fail and require cleanup.
    """

    store: LiveDatasetStore | None = None
    opened_here: bool = False


def _build_mint_deps(
    *,
    registry: SpecRegistry | None,
    spec_reader: SpecReader | None,
    graph: GraphReader | None,
    sdmx: SdmxClient | None,
    revision_reader: RevisionReader | None,
    artifact_store: ArtifactStore | None,
    materialiser_build: str,
    capability_version: str,
    ceiling: int,
    member_concurrency: int,
    code_index: CodeIndex | None,
    rate_cache: RateTableCache,
    version_writer: VersionRecordWriter | None,
    snapshot_reader: SnapshotReader | None,
    head_writer: HeadStore | None,
    materialise_semaphore: threading.BoundedSemaphore | None,
    coverage: CoverageService | None,
    operation_store: OperationStore | None,
    task_client: TaskClient | None,
    settings: DatasetsBuildSettings,
) -> MintDeps:
    """Build dependencies that do not need lifespan-time I/O.

    Lifespan resolves missing registry, graph, SDMX, and artifact-store clients.
    The in-memory code index and coverage service resolve here. A missing or
    unreadable coverage artifact fails door construction. `member_concurrency`
    bounds SDMX reads within described-arm groups.

    Lifespan wires missing version, snapshot, head, and operation stores to its existing
    `LiveDatasetStore`. The surface supplies a shared `materialise_semaphore`.
    Materialisation uses that gate when supplied and otherwise shares the
    executor's bound. Missing operation and task clients start as null implementations.
    """
    return MintDeps(
        registry=cast(SpecRegistry, registry),
        spec_reader=cast(SpecReader, spec_reader),
        sdmx=cast(SdmxClient, sdmx),
        graph=cast(GraphReader, graph),
        revision_reader=(
            revision_reader if revision_reader is not None else SentinelRevisionReader()
        ),
        artifact_store=cast(ArtifactStore, artifact_store),
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        enumerator=shared_enumerator(),
        enrichment=EnrichmentLookup(graph) if graph is not None else cast(EnrichmentLookup, None),
        code_index=code_index if code_index is not None else get_index(),
        rate_cache=rate_cache,
        version_writer=version_writer if version_writer is not None else NullVersionRecordWriter(),
        snapshot_reader=snapshot_reader,
        head_store=head_writer if head_writer is not None else NullHeadWriter(),
        materialise_semaphore=materialise_semaphore,
        coverage=coverage if coverage is not None else get_coverage_service(),
        operation_store=operation_store if operation_store is not None else NullOperationStore(),
        task_client=task_client if task_client is not None else NullTaskClient(),
        settings=settings,
    )


def _mint_close_live_clients_built_here(state: _MintDoorState, *, pool_opened_here: bool) -> None:
    """Close live clients and pool borrows opened here.

    Injected dependencies belong to the caller.
    """
    deps = state.deps
    # Lifespan-built graphs are `OriginGraphClient`. `MintDeps.graph` uses the
    # broader `GraphReader` protocol.
    if not state.injected.graph and deps.graph is not None:
        cast(OriginGraphClient, deps.graph).close()
    if not state.injected.sdmx and deps.sdmx is not None:
        deps.sdmx.close()
    # Only lifespan-created live task clients belong to this door.
    if not state.injected.task_client and isinstance(deps.task_client, LiveTaskClient):
        deps.task_client.close()
    if pool_opened_here:
        close_shared_pool()


def _mint_open_registry_and_reader(state: _MintDoorState, borrow: _MintStoreBorrow) -> None:
    """Build missing registry and reader on one store.

    If both are injected, this phase returns immediately.
    """
    if state.injected.registry and state.injected.spec_reader:
        return
    connections = shared_pool()
    # Mark the borrow first so cleanup closes it if store construction fails.
    borrow.opened_here = True
    borrow.store = LiveDatasetStore(connections=connections)
    if not state.injected.registry:
        state.deps.registry = LiveSpecRegistry(store=borrow.store)
    if not state.injected.spec_reader:
        state.deps.spec_reader = LiveSpecReader(store=borrow.store)


def _mint_open_graph_sdmx_and_artifact_store(
    state: _MintDoorState, borrow: _MintStoreBorrow, *, generation: GenerationProvider
) -> None:
    """Build missing clients and use the pool borrow for artifact inventory.

    A live graph client keys its caches on `generation`.
    """
    if state.injected.graph and state.injected.sdmx and state.injected.artifact_store:
        return
    live_settings = DatasetsSettings.from_env()
    if not state.injected.graph:
        state.deps.graph = build_graph_client(
            base_url=live_settings.graph_base_url,
            token=state.settings.graph_token,
            generation=generation,
            un_base_url=live_settings.un_graph_base_url,
            un_availability=SdmxClient(
                base_url=live_settings.un_graph_base_url,
                timeout=state.settings.sdmx_timeout_seconds,
            ),
            **live_settings.graph_cache_options(),
        )
        # `deps.enrichment` is `None` at this point, because `_build_mint_deps`
        # builds it only when `graph` is injected at door-build time. This
        # builds it here, once a live graph exists to wrap.
        state.deps.enrichment = EnrichmentLookup(state.deps.graph)
    if not state.injected.sdmx:
        state.deps.sdmx = SdmxClient(
            base_url=live_settings.sdmx_base_url,
            token=state.settings.graph_token,
            timeout=state.settings.sdmx_timeout_seconds,
        )
    if state.injected.artifact_store:
        return
    if borrow.store is None:
        connections = shared_pool()
        # Same ordering as `_mint_open_registry_and_reader`: flips before
        # the `LiveDatasetStore(...)` construction that could raise.
        borrow.opened_here = True
        borrow.store = LiveDatasetStore(connections=connections)
    inventory = LiveArtifactInventory(store=borrow.store)
    state.deps.artifact_store = LiveArtifactStore(
        buckets={
            Tier.working: live_settings.artifact_bucket_working,
            Tier.pinned: live_settings.artifact_bucket_pinned,
        },
        inventory=inventory,
    )


def _mint_open_task_client(state: _MintDoorState) -> None:
    """Build an uninjected task client from `TaskClientSettings` alone."""
    if state.injected.task_client:
        return
    task_settings = TaskClientSettings.from_env()
    state.deps.task_client = build_live_task_client(
        queue=task_settings.tasks_queue,
        worker_url=task_settings.worker_url,
        worker_invoker_service_account=task_settings.worker_invoker_service_account,
    )


def _mint_wire_shared_store_dependents(
    state: _MintDoorState, store: LiveDatasetStore | None
) -> None:
    """Wire missing version, snapshot, head, and operation stores to `store`.

    They share its pool borrow. A `None` store leaves null implementations intact.
    """
    if store is None:
        return
    if not state.injected.version_writer:
        state.deps.version_writer = LiveVersionRecordWriter(store=store)
    if not state.injected.snapshot_reader:
        state.deps.snapshot_reader = LiveSnapshotReader(store=store)
    if not state.injected.head_writer:
        state.deps.head_store = LiveHeadStore(store=store)
    if not state.injected.operation_store:
        state.deps.operation_store = LiveOperationStore(store=store, settings=state.settings)


async def _mint_drain_in_flight(state: _MintDoorState) -> None:
    """Wait off-loop for in-flight calls, bounded by the shared shutdown budget.

    Log calls that outlast the deadline. Client and pool cleanup proceeds.
    """
    with state.in_flight_lock:
        still_running = set(state.in_flight)
    if not still_running:
        return
    _done, not_done = await asyncio.to_thread(
        wait_futures, still_running, timeout=state.budget.remaining()
    )
    if not_done:
        logger.error(
            "Datasets mint door is shutting down with %d mint call(s) still "
            "running past the shared shutdown budget; closing "
            "dependencies out from under them and abandoning the work",
            len(not_done),
        )


@asynccontextmanager
async def _mint_lifespan(_app: FastAPI, *, state: _MintDoorState) -> AsyncIterator[None]:
    """Build missing dependencies, then drain in-flight calls before cleanup.

    Shutdown cancels queued futures. Running threads continue, so drain them
    before closing their graph, SDMX, or pool dependencies.
    """
    borrow = _MintStoreBorrow()
    async with AsyncExitStack() as stack:
        generation = UNKNOWN_GENERATION
        try:
            if not state.injected.graph:
                generation = await enter_shared_generation(
                    stack,
                    catalogue=state.catalogue,
                    graph_base_url=DatasetsSettings.from_env().graph_base_url,
                )
            _mint_open_registry_and_reader(state, borrow)
            _mint_open_graph_sdmx_and_artifact_store(state, borrow, generation=generation)
            _mint_open_task_client(state)
        except BaseException:
            # Close dependencies opened by earlier phases if later setup fails.
            _mint_close_live_clients_built_here(state, pool_opened_here=borrow.opened_here)
            raise
        _mint_wire_shared_store_dependents(state, borrow.store)
        try:
            yield
        finally:
            state.executor.shutdown(wait=False, cancel_futures=True)
            await _mint_drain_in_flight(state)
            _mint_close_live_clients_built_here(state, pool_opened_here=borrow.opened_here)


async def _run_mint(state: _MintDoorState, fn, /, *, timings: PhaseTimings, **kwargs) -> Manifest:
    """Run a mint on the shared executor and admission gate.

    Raises `MintBackpressure` when capacity is full or shutdown has begun.
    `BrokenExecutor` signals a permanently broken worker pool.

    Records `queue` from submission to the worker's start and `handoff` from the
    worker's finish to this coroutine resuming on the event loop.
    """
    if not state.semaphore.acquire(blocking=False):
        raise MintBackpressure()
    submitted_at = time.perf_counter()
    finished_at: list[float] = []

    def _timed_call(**call_kwargs):
        timings.record("queue", (time.perf_counter() - submitted_at) * 1000)
        try:
            return fn(**call_kwargs)
        finally:
            finished_at.append(time.perf_counter())

    try:
        future = submit_with_context(
            state.executor, functools.partial(run_with_timings, _timed_call, timings, **kwargs)
        )
    except BrokenExecutor:
        # This subclass must precede `RuntimeError` to preserve the 500 classification.
        state.semaphore.release()
        logger.error(
            "Datasets mint door's own executor pool is permanently broken; every "
            "future submitted to it will fail the same way",
            exc_info=True,
        )
        raise
    except RuntimeError:
        # Shutdown rejects new work as retryable backpressure.
        state.semaphore.release()
        raise MintBackpressure() from None
    except BaseException:
        # No future exists to release the acquired permit.
        state.semaphore.release()
        raise
    # No await here: shutdown reads this set on the same event-loop thread.
    with state.in_flight_lock:
        state.in_flight.add(future)

    def _on_done(done_future: Future, /) -> None:
        # Track the executor call's lifetime, including cancellation before start.
        # Remove it from `in_flight` before returning its permit.
        with state.in_flight_lock:
            state.in_flight.discard(done_future)
        state.semaphore.release()

    future.add_done_callback(_on_done)
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(future),
            timeout=state.settings.executor_wait_seconds(state.settings.mint_timeout_seconds),
        )
    finally:
        # Empty when the call timed out before the worker finished.
        if finished_at:
            timings.record("handoff", (time.perf_counter() - finished_at[0]) * 1000)


async def _mint_real_response(
    body: MintRequest, *, state: _MintDoorState, request_deps: MintDeps, trace_id: str
) -> JSONResponse:
    """Run a mint and return its manifest, or 202 when work is deferred.

    Reject unsupported render options before resolution, writes, or SDMX reads.
    """
    check_options_supported(body.options, code_index=request_deps.code_index)
    input_ = body.input
    submitted_resource: OperationResource | None = None
    with door_call("mint", logger=logger) as call:
        try:
            if isinstance(input_, StatVarSetInput):
                manifest = await _run_mint(
                    state,
                    mint_statvar_set,
                    timings=call.timings,
                    statvars=input_.statvars,
                    scope=input_.scope,
                    options=body.options,
                    deps=request_deps,
                    trace_id=trace_id,
                )
            else:
                manifest = await _run_mint(
                    state,
                    mint_spec_id,
                    timings=call.timings,
                    spec_id=input_.spec_id,
                    definition=input_.definition,
                    scope=input_.scope,
                    options=body.options,
                    deps=request_deps,
                    trace_id=trace_id,
                )
        except OperationSubmitted as submitted:
            # A deferred 202 is a successful outcome, not a refusal.
            submitted_resource = operation_submitted_resource(submitted)
    if submitted_resource is not None:
        assert submitted_resource.operation.poll is not None
        return JSONResponse(
            submitted_resource.model_dump(mode="json", exclude_none=True),
            status_code=202,
            headers={
                **no_store_headers(server_timing=call.server_timing),
                "Location": submitted_resource.operation.poll,
            },
        )
    return JSONResponse(
        manifest.model_dump(mode="json", exclude_none=True),
        headers=no_store_headers(server_timing=call.server_timing),
    )


async def _mint_endpoint_impl(body: MintRequest, *, state: _MintDoorState) -> JSONResponse:
    """Build the request's SDMX deadline, then dispatch the mint."""
    trace_id = get_trace_id()
    request_deps = replace(
        state.deps,
        deadline=_sdmx_deadline(state.settings.mint_timeout_seconds),
    )
    return await _mint_real_response(
        body, state=state, request_deps=request_deps, trace_id=trace_id
    )


def build_mint_door(
    *,
    materialiser_build: str,
    capability_version: str,
    ceiling: int,
    member_concurrency: int,
    settings: DatasetsBuildSettings | None = None,
    registry: SpecRegistry | None = None,
    spec_reader: SpecReader | None = None,
    graph: GraphReader | None = None,
    sdmx: SdmxClient | None = None,
    revision_reader: RevisionReader | None = None,
    artifact_store: ArtifactStore | None = None,
    version_writer: VersionRecordWriter | None = None,
    snapshot_reader: SnapshotReader | None = None,
    head_writer: HeadStore | None = None,
    materialise_semaphore: threading.BoundedSemaphore | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
    code_index: CodeIndex | None = None,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageService | None = None,
    operation_store: OperationStore | None = None,
    task_client: TaskClient | None = None,
    catalogue: CatalogueService | None = None,
) -> DoorContribution:
    """Build the mint door. Lifespan resolves missing live dependencies.

    `settings` defaults from the environment and supplies the concurrency,
    timeout, and graph-token settings used here. A live graph client keys its caches on the
    shared graph generation, acquired against `catalogue`, which defaults to the process-wide
    service.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    effective_pool_size = pool_size if pool_size is not None else _settings.mint_max_concurrency
    door_executor = build_door_executor(effective_pool_size, "mint")
    rate_cache = rate_cache if rate_cache is not None else RateTableCache()

    deps = _build_mint_deps(
        registry=registry,
        spec_reader=spec_reader,
        graph=graph,
        sdmx=sdmx,
        revision_reader=revision_reader,
        artifact_store=artifact_store,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        code_index=code_index,
        rate_cache=rate_cache,
        version_writer=version_writer,
        snapshot_reader=snapshot_reader,
        head_writer=head_writer,
        materialise_semaphore=materialise_semaphore,
        coverage=coverage,
        operation_store=operation_store,
        task_client=task_client,
        settings=_settings,
    )
    state = _MintDoorState(
        deps=deps,
        settings=_settings,
        budget=_budget,
        executor=door_executor.executor,
        semaphore=door_executor.semaphore,
        # Shutdown drains submitted futures until their completion callbacks remove them.
        in_flight=set(),
        in_flight_lock=threading.Lock(),
        injected=_MintInjected(
            registry=registry is not None,
            spec_reader=spec_reader is not None,
            graph=graph is not None,
            sdmx=sdmx is not None,
            artifact_store=artifact_store is not None,
            task_client=task_client is not None,
            version_writer=version_writer is not None,
            snapshot_reader=snapshot_reader is not None,
            head_writer=head_writer is not None,
            operation_store=operation_store is not None,
        ),
        catalogue=catalogue,
    )

    router = APIRouter()

    @router.post("/api/datasets", response_model=Manifest)
    async def mint_endpoint(body: MintRequest) -> JSONResponse:
        return await _mint_endpoint_impl(body, state=state)

    return DoorContribution(
        router=router,
        classify=classify_mint_exception,
        exception_types=MINT_EXCEPTION_TYPES,
        exception_exemplars=MINT_EXCEPTION_EXEMPLARS,
        lifespan=functools.partial(_mint_lifespan, state=state),
    )
