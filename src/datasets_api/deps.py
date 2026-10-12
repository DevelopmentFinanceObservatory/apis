"""The Datasets surface's process-level dependencies.

`open_deps` builds one `Deps` for the surface's lifespan, and every door reads it through the
surface's `DepsCell`. Doors build no client and no store. The bundle holds one graph client, one
SDMX client, one artifact store, the stores over one `dataset_store` pool borrow, the task client,
the coverage service, rate cache and code index, one materialise semaphore, and two executors.

The read executor runs snapshot, observations, operations polls, members, pin, register,
recipe, and head-hit reads. The work executor runs mints, definition rebuilds, frame refreshes,
and publishes, which parse and write frames of up to 250,000 rows, so a queue of builds cannot
delay a poll or a download. A
`GET /api/datasets/{id}` or `/options` call starts on the read executor and hands the call to
the work executor only when the head cannot answer it (`datasets_api.doors.definition`).

A dependency the caller supplies in `DepsOverrides` is used as given and never closed here.
Every other one is built live from `DatasetsSettings` and closed when the lifespan ends, after
both executors drain within one `ShutdownBudget`.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from code_index.query import CodeIndex
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import LiveSnapshotReader, SnapshotReader
from datasets_api.admission import MaterialisePermits, RebuildFlights
from datasets_api.config import DatasetsBuildSettings, DatasetsSettings
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.generation import (
    PLACE_SCOPE_MEMO_NAME,
    enter_shared_generation,
    register_place_scope_memo,
)
from datasets_api.operation_store import LiveOperationStore, OperationStore
from datasets_api.operations.task_client import (
    TaskClient,
    TaskClientSettings,
    build_live_task_client,
)
from datasets_api.publish_models import MAX_ROWS
from datasets_api.recipe_store import LiveRecipeStore, RecipeStore
from datasets_api.reference_store import LiveArtifactReferenceWriter, PinReferenceWriter
from datasets_api.shutdown import ShutdownBudget
from datasets_api.version_store import LiveHeadStore, LiveVersionLister, LiveVersionRecordWriter
from datasets_api.versions import HeadStore, VersionLister, VersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.build import build_commit
from materialise import EnrichmentLookup
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue.graph import build_graph_client
from origin_catalogue.service import CatalogueService
from series_coverage.shared import CoverageService
from spec_registry import LiveSpecRegistry, SpecRegistry
from spec_registry.reader import LiveSpecReader, SpecReader

# Raw-fetch floor for mint and definition materialisation. `raw_fetch_guard` scales it to each
# request's coverage estimate. Requests lacking a coverage row still need a bound. The floor
# reuses the `publish_models.MAX_ROWS` backstop.
MATERIALISE_RAW_FETCH_GUARD_ROWS = MAX_ROWS


@dataclass(frozen=True, kw_only=True)
class Deps:
    """Everything a Datasets door calls, built once per surface lifespan."""

    settings: DatasetsBuildSettings
    registry: SpecRegistry
    spec_reader: SpecReader
    snapshot_reader: SnapshotReader
    version_writer: VersionRecordWriter
    head_store: HeadStore
    version_lister: VersionLister
    operation_store: OperationStore
    recipe_store: RecipeStore
    reference_writer: PinReferenceWriter
    graph: GraphReader
    sdmx: SdmxClient
    artifact_store: ArtifactStore
    task_client: TaskClient
    coverage: CoverageService
    rate_cache: RateTableCache
    code_index: CodeIndex
    # Wraps `graph` and caches country-code and label lookups for the lifespan.
    enrichment: EnrichmentLookup
    materialiser_build: str
    now: Clock
    # Bounds the frames alive at once across mints, rebuilds, head renders, file renewals and
    # observation reads. Each holds one permit for as long as its frame's work runs
    # (`admission.bounded_materialise_slot`), and a coverage reload holds every permit. An
    # error's traceback can keep a frame until the door has answered the error. A publish
    # builds its frame outside the permits.
    materialise_semaphore: MaterialisePermits
    # One head-advancing rebuild per subject at a time, shared by definition and refresh.
    rebuilds: RebuildFlights
    read_executor: DoorExecutor
    work_executor: DoorExecutor


class DepsCell:
    """The surface's `Deps`, set when its lifespan starts.

    Doors are built before the lifespan opens their dependencies, so each route reads the bundle
    from the cell when it serves a request.
    """

    def __init__(self, deps: Deps | None = None) -> None:
        self._deps = deps

    def get(self) -> Deps:
        """The bundle. Raises `RuntimeError` before the surface's lifespan has started."""
        if self._deps is None:
            raise RuntimeError("the Datasets surface's dependencies are read before its lifespan")
        return self._deps

    def set(self, deps: Deps) -> None:
        self._deps = deps


@dataclass(frozen=True, kw_only=True)
class DepsOverrides:
    """Dependencies the caller supplies in place of live ones. `None` builds the live one.

    `coverage`, `rate_cache`, and `code_index` are process-wide services the surface resolves
    when it is built, so a missing coverage artifact fails construction.
    """

    coverage: CoverageService
    rate_cache: RateTableCache
    code_index: CodeIndex
    registry: SpecRegistry | None = None
    spec_reader: SpecReader | None = None
    snapshot_reader: SnapshotReader | None = None
    version_writer: VersionRecordWriter | None = None
    head_store: HeadStore | None = None
    version_lister: VersionLister | None = None
    operation_store: OperationStore | None = None
    recipe_store: RecipeStore | None = None
    reference_writer: PinReferenceWriter | None = None
    graph: GraphReader | None = None
    sdmx: SdmxClient | None = None
    artifact_store: ArtifactStore | None = None
    task_client: TaskClient | None = None
    now: Clock = utc_now

    def needs_store(self) -> bool:
        """Whether any omitted dependency reads or writes the dataset store."""
        return any(
            value is None
            for value in (
                self.registry,
                self.spec_reader,
                self.snapshot_reader,
                self.version_writer,
                self.head_store,
                self.version_lister,
                self.operation_store,
                self.recipe_store,
                self.reference_writer,
                self.artifact_store,
            )
        )


def _given_or[T](given: T | None, build: Callable[[], T]) -> T:
    return given if given is not None else build()


def build_executors(settings: DatasetsBuildSettings) -> tuple[DoorExecutor, DoorExecutor]:
    """The read and work executors, sized by `read_max_concurrency` and
    `work_max_concurrency`."""
    return (
        build_door_executor(settings.read_max_concurrency, "read"),
        build_door_executor(settings.work_max_concurrency, "work"),
    )


@asynccontextmanager
async def open_deps(
    *,
    settings: DatasetsBuildSettings,
    overrides: DepsOverrides,
    catalogue: CatalogueService | None,
) -> AsyncIterator[Deps]:
    """Build the surface's `Deps` and close what was built here when the context exits.

    `DatasetsSettings` is validated once a live graph, SDMX client, or artifact store is built.
    An omitted graph is a live origin client keyed on the process-wide graph generation,
    acquired against `catalogue`, with the default-scope memo registered on it. Shutdown
    cancels queued calls on both executors, waits for running ones within one budget, then
    closes the clients and the pool borrow. A failure while building closes whatever was
    already built.
    """
    live = functools.cache(DatasetsSettings.from_env)
    async with AsyncExitStack() as stack:
        store: LiveDatasetStore | None = None
        if overrides.needs_store():
            connections = shared_pool()
            stack.callback(close_shared_pool)
            store = LiveDatasetStore(connections=connections)

        graph = overrides.graph
        if graph is None:
            generation = await enter_shared_generation(
                stack, catalogue=catalogue, graph_base_url=live().graph_base_url
            )
            register_place_scope_memo(stack, generation, name=PLACE_SCOPE_MEMO_NAME)
            live_graph = build_graph_client(
                base_url=live().graph_base_url,
                token=settings.graph_token,
                generation=generation,
                un_base_url=live().un_graph_base_url,
                un_availability=SdmxClient(
                    base_url=live().un_graph_base_url, timeout=settings.sdmx_timeout_seconds
                ),
                **live().graph_cache_options(),
            )
            stack.callback(live_graph.close)
            graph = live_graph

        sdmx = overrides.sdmx
        if sdmx is None:
            sdmx = SdmxClient(
                base_url=live().sdmx_base_url,
                token=settings.graph_token,
                timeout=settings.sdmx_timeout_seconds,
            )
            stack.callback(sdmx.close)

        task_client = overrides.task_client
        if task_client is None:
            task_settings = TaskClientSettings.from_env()
            live_task_client = build_live_task_client(
                queue=task_settings.tasks_queue,
                worker_url=task_settings.worker_url,
                worker_invoker_service_account=task_settings.worker_invoker_service_account,
            )
            stack.callback(live_task_client.close)
            task_client = live_task_client

        def _store() -> LiveDatasetStore:
            assert store is not None  # `needs_store` opened it for every omitted store
            return store

        artifact_store = overrides.artifact_store
        if artifact_store is None:
            artifact_store = LiveArtifactStore(
                buckets={
                    Tier.working: live().artifact_bucket_working,
                    Tier.pinned: live().artifact_bucket_pinned,
                },
                inventory=LiveArtifactInventory(store=_store()),
            )

        materialise_semaphore = MaterialisePermits(settings.materialise_max_concurrency)
        # A coverage reload holds every permit, so it loads the new index beside no frame a
        # permit covers. A request that holds a permit confirmed at admission that its
        # coverage snapshot is the served one (`admission.under_current_coverage`), so it
        # builds its frame beside the one index the service serves. A request waiting for a
        # permit or a rebuild turn releases a replaced index within one wait slice of the
        # reload (`admission.STALE_CHECK_SECONDS`). Before it waits, a request can hold a
        # replaced index for as long as its head checks, its graph reads, member resolution,
        # the estimate and the budget decision take, beside the frames other requests build.
        # A publish holds no permit, so its frame can be resident during a reload.
        stack.callback(overrides.coverage.register_load_guard(materialise_semaphore.every_permit))

        read_executor, work_executor = build_executors(settings)
        budget = ShutdownBudget()

        async def _drain() -> None:
            for executor in (work_executor, read_executor):
                executor.executor.shutdown(wait=False, cancel_futures=True)
            for executor in (work_executor, read_executor):
                await asyncio.to_thread(executor.drain, budget)

        # Registered last, so the drain runs before any client or the pool closes.
        stack.push_async_callback(_drain)

        yield Deps(
            settings=settings,
            registry=_given_or(overrides.registry, lambda: LiveSpecRegistry(store=_store())),
            spec_reader=_given_or(overrides.spec_reader, lambda: LiveSpecReader(store=_store())),
            snapshot_reader=_given_or(
                overrides.snapshot_reader, lambda: LiveSnapshotReader(store=_store())
            ),
            version_writer=_given_or(
                overrides.version_writer, lambda: LiveVersionRecordWriter(store=_store())
            ),
            head_store=_given_or(overrides.head_store, lambda: LiveHeadStore(store=_store())),
            version_lister=_given_or(
                overrides.version_lister, lambda: LiveVersionLister(store=_store())
            ),
            operation_store=_given_or(
                overrides.operation_store,
                lambda: LiveOperationStore(store=_store(), settings=settings),
            ),
            recipe_store=_given_or(overrides.recipe_store, lambda: LiveRecipeStore(store=_store())),
            reference_writer=_given_or(
                overrides.reference_writer, lambda: LiveArtifactReferenceWriter(store=_store())
            ),
            graph=graph,
            sdmx=sdmx,
            artifact_store=artifact_store,
            task_client=task_client,
            coverage=overrides.coverage,
            rate_cache=overrides.rate_cache,
            code_index=overrides.code_index,
            enrichment=EnrichmentLookup(
                graph,
                type_kinds=overrides.code_index.party_types,
                parties=overrides.code_index.parties,
            ),
            materialiser_build=build_commit(),
            now=overrides.now,
            materialise_semaphore=materialise_semaphore,
            rebuilds=RebuildFlights(),
            read_executor=read_executor,
            work_executor=work_executor,
        )


__all__ = [
    "MATERIALISE_RAW_FETCH_GUARD_ROWS",
    "Deps",
    "DepsCell",
    "DepsOverrides",
    "build_executors",
    "open_deps",
]
