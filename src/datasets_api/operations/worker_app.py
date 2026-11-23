"""Standalone ASGI app for `POST /internal/datasets/operations/{operation_id}/run`.

The worker is a separate deployment: `dfo_apis.app` builds this app instead of the API surfaces
when `DFO_WORKER_SURFACE` is truthy, and the main app never mounts it. It bypasses `Surface`,
whose prefixes must start with `/api/`, so the route sits under `/internal/` for Cloud Run's
internal ingress and has no problem-details envelope or trace id. Cloud Tasks uses HTTP status
only. A 2xx ends a delivery. Other statuses retry. Cloud Run concurrency 1 allows one request per
instance.

The lifespan builds the dependencies every run shares once per process: one artifact store, one
graph client, one SDMX client, the whole-variable breadth reader over the process's facet profile
cache (`datasets_api.profile_breadth`), and one materialise semaphore with a single permit, which
every operation's materialisation takes. Each run's `DefinitionDeps` pairs them with the code index
read under the run's bindings. Materialisation runs in a worker thread.

The route verifies the OIDC token and claims the row (`OperationStore.claim`) for the task's
submission, with a static lease of `LEASE_SECONDS`, under Cloud Tasks' 1,800-second dispatch
deadline. A delivery that finds the row inside another attempt's lease answers 409. A delivery
that finds the row past the operation deadline fails it retryable (`OperationStore.abandon`) and
answers 200 without running it. The run's work deadline is `RUN_DEADLINE_SECONDS` after the stored
`claimed_at`, read against the same wall clock as the store, so the lease outlasts it by the time
the final writes need. The operation runs under one origin catalogue build. It is the loaded build
when it is the one the row names, otherwise the published build
(`CatalogueService.snapshot_current`), so an operation submitted under a build newer than this
instance's loaded one runs under the newer admission. A row with no build also takes the published
build. A failed catalogue read is a transient failure of the claimed attempt. With
`GRAPH_RELEASE_URI` set, the operation also runs under the release the process serves when the
delivery arrives, whichever release served its submission, and a release swap holds the run's
permit.

A run's place in the attempt cap (`DATASETS_OPERATION_MAX_ATTEMPTS`) is the larger of two counts:
Cloud Tasks' retry count header, and the claims the submission has made
(`Operation.submission_attempts`), which the row keeps across tasks. The claim itself stores the
run as the last (`Operation.final_delivery`) when either reaches the cap, so a run that crashes
on its last allowed attempt leaves a row the poll fails. A claim past the cap fails the row
retryable without running it.

A transient failure releases the row (`run_operation`, or a failed catalogue read here). The
route then schedules the operation's next run as a new task (`RetryTaskClient.enqueue_retry`),
named for the failed claim's place in the submission, and answers 200 `retrying`, so Cloud Tasks
does not also retry the delivery after its 1,680-second backoff. The retry after a submission's
first run is due `DATASETS_OPERATION_RETRY_DELAY_SECONDS` (60) ahead, and every later one
`DATASETS_OPERATION_RETRY_LATER_DELAY_SECONDS` (900) ahead, so three runs span about 16 minutes
of an outage. A release that finds the row moved on schedules nothing and answers 200
`superseded`. The retry task carries the submission's `submission_count`, so it claims like any
delivery: a row that completed, failed or moved to a new submission meanwhile answers it 200
without running. When the create fails, or the process has no retry client, the route answers
503 as before and the queue's own backoff retries the delivery. A run that crashes leaves its
lease live and answers 500, and the queue's backoff, longer than the lease, retries it. Every
kind of operation on the queue (mints, exports and head checks, renewals, render-aheads) takes
the same path, since each one's transient failure is a released row that any delivery can claim.

Tests can inject every dependency. With dependencies omitted, `build_worker_app` resolves live
services in its lifespan from shared and worker-specific settings.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from functools import partial

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from code_index.shared import get_index
from dataset_store import DatasetStore, LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.admission import MaterialisePermits, RebuildFlights
from datasets_api.config import DatasetsBuildSettings, DatasetsConfigError, DatasetsSettings
from datasets_api.current import DefinitionDeps
from datasets_api.generation import enter_shared_generation, register_place_scope_memo
from datasets_api.operation_store import (
    Clock,
    LiveOperationStore,
    Operation,
    OperationStore,
    utc_now,
)
from datasets_api.operations.lease import LEASE_SECONDS, RUN_DEADLINE_SECONDS
from datasets_api.operations.oidc import LiveOidcVerifier, OidcRejected, OidcVerifier
from datasets_api.operations.task_client import (
    LiveTaskClient,
    RetryTaskClient,
    build_live_task_client,
)
from datasets_api.operations.worker import (
    OperationSuperseded,
    OperationTransientFailure,
    logged_subject,
    operation_kind,
    run_operation,
)
from datasets_api.profile_breadth import open_profile_breadth
from datasets_api.publish_models import MAX_ROWS as _MATERIALISE_RAW_FETCH_GUARD_ROWS
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import (
    LiveHeadStore,
    LiveVersionLister,
    LiveVersionRecordWriter,
)
from dc_sdmx import DATA_HEDGE_AFTER_SECONDS, SdmxClient
from dfo_platform.build import build_commit
from graph_release.context import bound_served_release
from graph_release.service import ReleaseService, get_release_service
from materialise import EnrichmentLookup
from materialise.facets import VariableBreadthReader
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import bound_catalogue
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from origin_catalogue.service import CatalogueService, get_catalogue_service
from series_coverage.shared import ServedCoverage, get_coverage_service
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import shared_enumerator

logger = logging.getLogger(__name__)

WORKER_PLACE_SCOPE_MEMO_NAME = "datasets_worker.default_place_scope"
"""The default-scope memo's registration name on the shared graph generation."""

RUN_ROUTE_PATH = "/internal/datasets/operations/{operation_id}/run"


def _live_deps_factory(
    *,
    store: DatasetStore,
    graph: GraphReader,
    sdmx: SdmxClient,
    artifact_store: ArtifactStore,
    coverage: ServedCoverage,
    materialise_semaphore: MaterialisePermits,
    variable_breadth: VariableBreadthReader | None = None,
    settings: DatasetsBuildSettings | None = None,
) -> partial[DefinitionDeps]:
    """Assembles a run's dependencies without I/O, and returns the `DefinitionDeps` partial
    that a run completes with the code index it reads (`code_index=`).

    The ceiling, member concurrency, build and capability version match the mint and definition
    doors, keeping admission estimates and execution limits aligned and recording each run
    under the release that produced it. The snapshot reader reads the same store, so a worker
    render serves as the synchronous path does: a version on record keeps its stored citation and
    `prepared_at`, the response lists the artifact already linked under each rendering key, and
    `run_operation` checks that the request's rendering is servable before completing. A
    run breaks a single-place facet tie on `variable_breadth`, as a sync mint of the spec does,
    so both elect one facet.

    The version lister reads the renderings analysis roots read, which a run that materialised
    a version renders ahead (`render_ahead.render_ahead`).

    The factory holds no code index, so a run reads the one its bindings give
    (`build_worker_app`), party kinds included.
    It holds the process's `RebuildFlights`, so every run shares one.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    return partial(
        DefinitionDeps,
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        materialiser_build=build_commit(),
        capability_version=DATASETS_CAPABILITY_VERSION,
        ceiling=_MATERIALISE_RAW_FETCH_GUARD_ROWS,
        member_concurrency=_settings.mint_member_max_concurrency,
        enumerator=shared_enumerator(),
        enrichment=EnrichmentLookup(graph),
        rate_cache=RateTableCache(),
        coverage=coverage,
        version_writer=LiveVersionRecordWriter(store=store),
        head_store=LiveHeadStore(store=store),
        snapshot_reader=LiveSnapshotReader(store=store),
        version_lister=LiveVersionLister(store=store),
        materialise_semaphore=materialise_semaphore,
        rebuilds=RebuildFlights(),
        head_recheck_interval=timedelta(seconds=_settings.head_recheck_seconds),
        head_max_age=timedelta(seconds=_settings.head_max_age_seconds),
        variable_breadth=variable_breadth,
        settings=_settings,
    )


class WorkerSettings(BaseSettings):
    """No-safe-default env this module reads on its own, independent of
    `DatasetsSettings.from_env()`. `DFO_TASKS_QUEUE` is optional: with it the worker schedules
    a transient failure's retry as a new task on that queue, and without it the queue's own
    backoff retries the delivery.
    """

    model_config = SettingsConfigDict()

    worker_url: str = Field(min_length=1, validation_alias="DFO_WORKER_URL")
    worker_invoker_service_account: str = Field(
        min_length=1, validation_alias="DFO_WORKER_INVOKER_SERVICE_ACCOUNT"
    )
    tasks_queue: str | None = Field(default=None, min_length=1, validation_alias="DFO_TASKS_QUEUE")

    @classmethod
    def from_env(cls) -> WorkerSettings:
        try:
            return cls()
        except ValidationError as exc:
            raise DatasetsConfigError(str(exc)) from exc


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header.removeprefix("Bearer ").strip()


def _cloud_tasks_retry_count(request: Request) -> int:
    """Returns Cloud Tasks' retry count, starting at 0. Missing or invalid headers also return 0,
    so parsing errors cannot mark a task as final.
    """
    raw = request.headers.get("X-CloudTasks-TaskRetryCount")
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


async def _task_submission_count(request: Request) -> int | None:
    """Returns a non-negative integer `submission_count`, or `None` if missing or malformed.

    Booleans are invalid counts. The route rejects malformed bodies with 400. `superseded` is
    reserved for tasks from an older submission.
    """
    try:
        body = await request.json()
    except ValueError:
        return None
    value = body.get("submission_count") if isinstance(body, dict) else None
    # Python treats booleans as integers, but task counts require JSON integers.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def build_worker_app(
    *,
    settings: DatasetsBuildSettings | None = None,
    operation_store: OperationStore | None = None,
    deps: DefinitionDeps | None = None,
    oidc_verifier: OidcVerifier | None = None,
    worker_url: str | None = None,
    worker_invoker_service_account: str | None = None,
    max_attempts: int | None = None,
    retry_client: RetryTaskClient | None = None,
    retry_delay_seconds: int | None = None,
    retry_later_delay_seconds: int | None = None,
    run_in_thread: Callable[..., None] = run_operation,
    catalogue: CatalogueService | None = None,
    release: ReleaseService | None = None,
    now: Clock = utc_now,
) -> FastAPI:
    """Builds the worker app, resolving omitted dependencies in its lifespan.

    Tests can inject all dependencies and a `run_in_thread` stub. The runner receives the claimed
    operation, dependencies, the run's absolute monotonic work deadline, and the final-attempt flag.
    `now` is the wall clock the work deadline is counted against, the same clock the operation store
    stamps `claimed_at` with. After a normal return, the route reads the stored terminal state and
    answers 2xx so Cloud Tasks does not retry completed or failed work. `catalogue` defaults to the
    process-wide service, so a missing or unloadable `ORIGIN_CATALOGUE_URI` fails construction.
    `release` defaults to the process-wide graph release service, so an unset `GRAPH_RELEASE_URI`
    or an unreadable release fails construction. `retry_client` schedules a transient failure's
    next run; when omitted, a lifespan that builds live dependencies builds one on
    `DFO_TASKS_QUEUE` if it is set, and otherwise transient failures answer 503. Injected `deps` serve every run as given. Deps the lifespan builds
    take the code index each run reads under its bindings, since it can change between runs.
    The lifespan reads it once under the served release, so a missing or malformed index fails
    startup, and keeps none.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _max_attempts = max_attempts if max_attempts is not None else _settings.operation_max_attempts
    _retry_delay_seconds = (
        retry_delay_seconds
        if retry_delay_seconds is not None
        else _settings.operation_retry_delay_seconds
    )
    _retry_later_delay_seconds = (
        retry_later_delay_seconds
        if retry_later_delay_seconds is not None
        else _settings.operation_retry_later_delay_seconds
    )
    _retry_client: RetryTaskClient | None = retry_client
    _operation_store: OperationStore | None = operation_store
    _deps: DefinitionDeps | None = deps
    _deps_for_index: partial[DefinitionDeps] | None = None
    _oidc_verifier: OidcVerifier = (
        oidc_verifier if oidc_verifier is not None else LiveOidcVerifier()
    )
    _worker_url = worker_url
    _worker_invoker_service_account = worker_invoker_service_account
    _catalogue = catalogue if catalogue is not None else get_catalogue_service()
    _release = release if release is not None else get_release_service()
    _lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _operation_store, _deps, _deps_for_index, _worker_url
        nonlocal _worker_invoker_service_account, _retry_client
        async with AsyncExitStack() as stack:
            pool_opened_here = False
            store: LiveDatasetStore | None = None
            graph: OriginGraphClient | None = None
            sdmx: SdmxClient | None = None
            live_retry_client: LiveTaskClient | None = None

            def _close_live() -> None:
                if graph is not None:
                    graph.close()
                if sdmx is not None:
                    sdmx.close()
                if live_retry_client is not None:
                    live_retry_client.close()
                if pool_opened_here:
                    close_shared_pool()

            try:
                worker_settings: WorkerSettings | None = None
                if _worker_url is None or _worker_invoker_service_account is None:
                    worker_settings = WorkerSettings.from_env()
                    if _worker_url is None:
                        _worker_url = worker_settings.worker_url
                    if _worker_invoker_service_account is None:
                        _worker_invoker_service_account = (
                            worker_settings.worker_invoker_service_account
                        )

                if _deps is None and _deps_for_index is None:
                    if _retry_client is None:
                        if worker_settings is None:
                            worker_settings = WorkerSettings.from_env()
                        if worker_settings.tasks_queue is None:
                            logger.warning(
                                "Datasets worker: DFO_TASKS_QUEUE is unset, so a transient "
                                "failure waits for the queue's own retry backoff"
                            )
                        else:
                            live_retry_client = build_live_task_client(
                                queue=worker_settings.tasks_queue,
                                worker_url=_worker_url,
                                worker_invoker_service_account=_worker_invoker_service_account,
                            )
                            _retry_client = live_retry_client
                    live_settings = DatasetsSettings.from_env()
                    connections = shared_pool()
                    pool_opened_here = True
                    store = LiveDatasetStore(connections=connections)
                    generation = await enter_shared_generation(
                        stack, catalogue=_catalogue, graph_base_url=live_settings.graph_base_url
                    )
                    register_place_scope_memo(stack, generation, name=WORKER_PLACE_SCOPE_MEMO_NAME)
                    graph = build_graph_client(
                        base_url=live_settings.graph_base_url,
                        token=_settings.graph_token,
                        generation=generation,
                        un_base_url=live_settings.un_graph_base_url,
                        un_availability=SdmxClient(
                            base_url=live_settings.un_graph_base_url,
                            timeout=_settings.sdmx_timeout_seconds,
                        ),
                        **live_settings.graph_cache_options(),
                    )
                    sdmx = SdmxClient(
                        base_url=live_settings.sdmx_base_url,
                        token=_settings.graph_token,
                        timeout=_settings.sdmx_timeout_seconds,
                        hedge_after=DATA_HEDGE_AFTER_SECONDS,
                        total_concurrency=_settings.mint_member_max_concurrency,
                    )
                    artifact_store = LiveArtifactStore(
                        buckets={
                            Tier.working: live_settings.artifact_bucket_working,
                            Tier.pinned: live_settings.artifact_bucket_pinned,
                        },
                        inventory=LiveArtifactInventory(store=store),
                    )
                    coverage = get_coverage_service()
                    materialise_semaphore = MaterialisePermits(1)
                    # A release swap holds the run's permit, as on the API surface. The run
                    # confirms that the service still serves its bound release before and
                    # while it waits for the permit and once it holds it, so it builds its
                    # frame beside the one release the service serves and refuses within one
                    # wait slice of the swap (`admission.STALE_CHECK_SECONDS`), as it does
                    # while it waits its turn at a shared rebuild. Before it waits, a run can
                    # hold a replaced release for as long as its head checks, its graph reads,
                    # member resolution, the estimate and the budget decision take.
                    stack.callback(_release.register_swap_guard(materialise_semaphore.every_permit))
                    # Run state uses the top-level store. This worker only runs tasks.
                    # Dependency defaults stay inert.
                    # Parsed now, so a missing or malformed index fails startup.
                    with bound_served_release(_release):
                        get_index()
                    _deps_for_index = _live_deps_factory(
                        store=store,
                        graph=graph,
                        sdmx=sdmx,
                        artifact_store=artifact_store,
                        coverage=coverage,
                        materialise_semaphore=materialise_semaphore,
                        variable_breadth=open_profile_breadth(
                            stack,
                            settings=_settings,
                            live_settings=live_settings,
                            generation=generation,
                            coverage=coverage,
                        ),
                        settings=_settings,
                    )
                if _operation_store is None:
                    if store is None:
                        connections = shared_pool()
                        pool_opened_here = True
                        store = LiveDatasetStore(connections=connections)
                    _operation_store = LiveOperationStore(store=store, settings=_settings)
            except BaseException:
                _close_live()
                raise
            try:
                yield
            finally:
                _close_live()

    async def _schedule_retry(
        operation: Operation,
        *,
        subject_id: str | None,
        detail: str,
    ) -> JSONResponse:
        """Schedule the released operation's next run and answer 200 `retrying`, or answer 503
        so the queue's own backoff retries the delivery when there is no retry client or the
        create fails."""
        retry_client = _retry_client
        if retry_client is None:
            return JSONResponse({"detail": detail}, status_code=503)
        delay_seconds = (
            _retry_delay_seconds
            if operation.submission_attempts <= 1
            else _retry_later_delay_seconds
        )
        try:
            await asyncio.to_thread(
                retry_client.enqueue_retry,
                operation_id=operation.id,
                submission_count=operation.submission_count,
                generation=operation.submission_attempts,
                delay_seconds=delay_seconds,
            )
        except Exception:
            logger.warning(
                "Datasets worker: could not schedule the retry of operation %r "
                "(subject_id=%r); leaving it to the queue's own backoff",
                operation.id,
                subject_id,
                exc_info=True,
            )
            return JSONResponse({"detail": detail}, status_code=503)
        logger.info(
            "Datasets worker: operation %r (subject_id=%r) released after attempt %d; its "
            "retry runs in %d s",
            operation.id,
            subject_id,
            operation.submission_attempts,
            delay_seconds,
        )
        return JSONResponse({"status": "retrying"}, status_code=200)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.post(RUN_ROUTE_PATH)
    async def run(operation_id: str, request: Request) -> JSONResponse:
        token = _bearer_token(request)
        if token is None:
            return JSONResponse({"detail": "missing bearer token"}, status_code=401)
        assert _worker_url is not None and _worker_invoker_service_account is not None
        try:
            _oidc_verifier.verify(
                token, audience=_worker_url, expected_email=_worker_invoker_service_account
            )
        except OidcRejected as exc:
            return JSONResponse({"detail": exc.reason}, status_code=401)

        assert _operation_store is not None
        assert _deps is not None or _deps_for_index is not None
        submission_count = await _task_submission_count(request)
        if submission_count is None:
            # Reject malformed tasks before claiming.
            logger.error(
                "Datasets worker: task for operation %r has no valid submission_count",
                operation_id,
            )
            return JSONResponse({"detail": "missing or invalid submission_count"}, status_code=400)
        retry_count = _cloud_tasks_retry_count(request)
        final_delivery = retry_count >= _max_attempts - 1
        # Protects tests that issue concurrent claims through a thread pool.
        with _lock:
            # The claim records whether a later run is due, which decides whether a poll past
            # the lease fails the row or waits for that run: none is once this is the task's
            # last delivery or the submission's claims reach the cap.
            operation = _operation_store.claim(
                operation_id,
                submission_count=submission_count,
                lease_seconds=LEASE_SECONDS,
                final_delivery=final_delivery,
                max_attempts=_max_attempts,
            )
        if operation is None:
            current = _operation_store.get(operation_id)
            if current is None:
                return JSONResponse({"detail": "unknown operation"}, status_code=404)
            if current.submission_count > submission_count:
                # This task belongs to an older submission. A 2xx stops its retries.
                return JSONResponse({"status": "superseded"}, status_code=200)
            if current.submission_count < submission_count:
                # The task names a future submission, so reject it as malformed with 400.
                logger.error(
                    "Datasets worker: task for operation %r names submission_count %r"
                    " ahead of the row's %r",
                    operation_id,
                    submission_count,
                    current.submission_count,
                )
                return JSONResponse(
                    {"detail": "submission_count ahead of the operation's own"}, status_code=400
                )
            if current.state in ("done", "failed"):
                # The operation is terminal. A 2xx stops duplicate task retries.
                return JSONResponse({"status": current.state}, status_code=200)
            # A row past the operation deadline is unclaimable and abandoned. Failing it here
            # ends the submission without a poll, and the 2xx stops the task's deliveries.
            abandoned = _operation_store.abandon(operation_id, attempt=current.attempt_count)
            if abandoned is not None:
                logger.warning(
                    "Datasets worker: operation %r is abandoned; failed without running",
                    operation_id,
                )
                return JSONResponse({"status": abandoned.state}, status_code=200)
            return JSONResponse(
                {"detail": f"operation {operation_id!r} is claimed by a live attempt"},
                status_code=409,
            )
        subject_id = logged_subject(operation)
        logger.info(
            "Datasets worker: claimed operation %r (subject_id=%r, kind=%s), attempt %d of %d "
            "for its submission, Cloud Tasks retry count %d",
            operation_id,
            subject_id,
            operation_kind(operation),
            operation.submission_attempts,
            _max_attempts,
            retry_count,
        )
        if operation.submission_attempts > _max_attempts:
            # A retry task's own redelivery, after a crash, can claim past the cap.
            logger.warning(
                "Datasets worker: operation %r (subject_id=%r) has used its %d attempts; "
                "failing it without running",
                operation_id,
                subject_id,
                _max_attempts,
            )
            _operation_store.fail(
                operation_id,
                attempt=operation.attempt_count,
                refusal_code="internal_error",
                retryable=True,
            )
            return JSONResponse({"status": "failed"}, status_code=200)
        is_final_attempt = bool(operation.final_delivery)

        # Count the work deadline from the stored claim time, the instant the lease runs from.
        assert operation.claimed_at is not None
        work_ends = operation.claimed_at + timedelta(seconds=RUN_DEADLINE_SECONDS)
        deadline = time.monotonic() + (work_ends - now()).total_seconds()

        catalogue = _catalogue.snapshot()
        if catalogue.build_id != operation.origin_catalogue_build_id:
            try:
                catalogue = await asyncio.to_thread(_catalogue.snapshot_current)
            except Exception:
                logger.exception(
                    "Datasets worker: could not read the published origin catalogue"
                    " for operation %r",
                    operation_id,
                )
                if is_final_attempt:
                    _operation_store.fail(
                        operation_id,
                        attempt=operation.attempt_count,
                        refusal_code="internal_error",
                        retryable=True,
                    )
                    return JSONResponse({"status": "failed"}, status_code=200)
                if _operation_store.release(operation_id, attempt=operation.attempt_count) is None:
                    return JSONResponse({"status": "superseded"}, status_code=200)
                return await _schedule_retry(
                    operation, subject_id=subject_id, detail="origin catalogue unavailable"
                )
        try:
            # One catalogue version and one release serve the whole operation. `to_thread`
            # carries the bindings.
            with bound_catalogue(catalogue), bound_served_release(_release):
                if _deps is not None:
                    run_deps = _deps
                else:
                    assert _deps_for_index is not None
                    run_deps = _deps_for_index(code_index=get_index())
                await asyncio.to_thread(
                    run_in_thread,
                    operation=operation,
                    deps=run_deps,
                    operation_store=_operation_store,
                    deadline=deadline,
                    is_final_attempt=is_final_attempt,
                )
        except OperationSuperseded:
            # Another attempt owns the work. A 2xx stops this delivery.
            return JSONResponse({"status": "superseded"}, status_code=200)
        except OperationTransientFailure:
            return await _schedule_retry(
                operation, subject_id=subject_id, detail="transient worker failure"
            )
        except Exception:
            logger.exception("Datasets worker: operation %r crashed unexpectedly", operation_id)
            return JSONResponse({"detail": "unexpected worker failure"}, status_code=500)
        # Report the stored terminal state. Any terminal result gets 2xx to stop retries.
        final = _operation_store.get(operation_id)
        status = final.state if final is not None else "done"
        return JSONResponse({"status": status}, status_code=200)

    return app


__all__ = ["RUN_ROUTE_PATH", "WorkerSettings", "build_worker_app"]
