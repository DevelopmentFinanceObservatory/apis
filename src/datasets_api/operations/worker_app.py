"""Standalone ASGI app for `POST /internal/datasets/operations/{operation_id}/run`.

The main app mounts it when `DFO_WORKER_SURFACE` is truthy. It bypasses `Surface`, whose
prefixes must start with `/api/`, so the route sits under `/internal/` for Cloud Run's internal
ingress and has no problem-details envelope or trace id. Cloud Tasks uses HTTP status only. A 2xx ends
a delivery. Other statuses retry. Cloud Run concurrency 1 allows one request per instance. The
app builds one `DefinitionDeps` and runs materialisation in a worker thread, with one origin
catalogue build bound for that one operation. Before claiming, the route reads the operation row.
When the loaded build is the one the row names, the operation runs under it. Otherwise the route
reads the published build (`CatalogueService.snapshot_current`), so an operation submitted under a
build newer than this instance's loaded one runs under the newer admission. A row with no
build, written before migration 0015, also takes the published build. A failed read answers 503
without a claim, and Cloud Tasks retries.

Tests can inject every dependency. With dependencies omitted, `build_worker_app` resolves live
services in its lifespan from shared and worker-specific settings.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from code_index.query import CodeIndex
from code_index.shared import get_index
from dataset_store import DatasetStore, LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings, DatasetsConfigError, DatasetsSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.generation import enter_shared_generation, register_place_scope_memo
from datasets_api.operation_store import LiveOperationStore, OperationStore
from datasets_api.operations.oidc import LiveOidcVerifier, OidcRejected, OidcVerifier
from datasets_api.operations.worker import (
    OperationSuperseded,
    OperationTransientFailure,
    run_operation,
)
from datasets_api.publish_models import MAX_ROWS as _MATERIALISE_RAW_FETCH_GUARD_ROWS
from datasets_api.revision import SentinelRevisionReader
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import bound_catalogue
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from origin_catalogue.service import CatalogueService, get_catalogue_service
from series_coverage.shared import CoverageService, get_coverage_service
from spec_registry.reader import LiveSpecReader
from statvar_members import shared_enumerator

logger = logging.getLogger(__name__)

WORKER_PLACE_SCOPE_MEMO_NAME = "datasets_worker.default_place_scope"
"""The default-scope memo's registration name on the shared graph generation."""

RUN_ROUTE_PATH = "/internal/datasets/operations/{operation_id}/run"


def _build_live_definition_deps(
    *,
    store: DatasetStore,
    graph: GraphReader,
    sdmx: SdmxClient,
    artifact_store: ArtifactStore,
    code_index: CodeIndex,
    coverage: CoverageService,
    settings: DatasetsBuildSettings | None = None,
) -> DefinitionDeps:
    """Assembles dependencies without I/O.

    The ceiling and member concurrency match the mint and definition doors, keeping admission
    estimates and execution limits aligned. The snapshot reader reads the same store, so a worker
    render serves as the synchronous path does: a version on record keeps its stored citation and
    `prepared_at`, the response lists the artifact already linked under each rendering key, and
    `run_operation` checks that the request's rendering is servable before completing.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    return DefinitionDeps(
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        materialiser_build="",
        capability_version="",
        ceiling=_MATERIALISE_RAW_FETCH_GUARD_ROWS,
        member_concurrency=_settings.mint_member_max_concurrency,
        enumerator=shared_enumerator(),
        enrichment=EnrichmentLookup(graph),
        code_index=code_index,
        rate_cache=RateTableCache(),
        coverage=coverage,
        version_writer=LiveVersionRecordWriter(store=store),
        head_store=LiveHeadStore(store=store),
        snapshot_reader=LiveSnapshotReader(store=store),
        materialise_semaphore=None,
        head_recheck_interval=timedelta(seconds=_settings.head_recheck_seconds),
        head_max_age=timedelta(seconds=_settings.head_max_age_seconds),
        settings=_settings,
    )


class WorkerSettings(BaseSettings):
    """No-safe-default env this module reads on its own, independent of
    `DatasetsSettings.from_env()`. See this module's docstring for why
    `DFO_TASKS_QUEUE` (the one other var in that family) is not part of
    this same object: this app never enqueues anything.
    """

    model_config = SettingsConfigDict()

    worker_url: str = Field(min_length=1, validation_alias="DFO_WORKER_URL")
    worker_invoker_service_account: str = Field(
        min_length=1, validation_alias="DFO_WORKER_INVOKER_SERVICE_ACCOUNT"
    )

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
    lease_seconds: float | None = None,
    max_attempts: int | None = None,
    run_in_thread: Callable[..., None] = run_operation,
    catalogue: CatalogueService | None = None,
) -> FastAPI:
    """Builds the worker app, resolving omitted dependencies in its lifespan.

    Tests can inject all dependencies and a `run_in_thread` stub. The runner receives the
    claimed operation, lease duration, dependencies, and final-attempt flag. After a normal
    return, the route reads the stored terminal state and answers 2xx so Cloud Tasks does not
    retry completed or failed work. `catalogue` defaults to the process-wide service, so a
    missing or unloadable `ORIGIN_CATALOGUE_URI` fails construction.
    """
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _lease_seconds = (
        lease_seconds if lease_seconds is not None else _settings.operation_lease_seconds
    )
    _max_attempts = max_attempts if max_attempts is not None else _settings.operation_max_attempts
    _operation_store: OperationStore | None = operation_store
    _deps: DefinitionDeps | None = deps
    _oidc_verifier: OidcVerifier = (
        oidc_verifier if oidc_verifier is not None else LiveOidcVerifier()
    )
    _worker_url = worker_url
    _worker_invoker_service_account = worker_invoker_service_account
    _catalogue = catalogue if catalogue is not None else get_catalogue_service()
    _lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _operation_store, _deps, _worker_url, _worker_invoker_service_account
        async with AsyncExitStack() as stack:
            pool_opened_here = False
            store: LiveDatasetStore | None = None
            graph: OriginGraphClient | None = None
            sdmx: SdmxClient | None = None

            def _close_live() -> None:
                if graph is not None:
                    graph.close()
                if sdmx is not None:
                    sdmx.close()
                if pool_opened_here:
                    close_shared_pool()

            try:
                if _worker_url is None or _worker_invoker_service_account is None:
                    worker_settings = WorkerSettings.from_env()
                    if _worker_url is None:
                        _worker_url = worker_settings.worker_url
                    if _worker_invoker_service_account is None:
                        _worker_invoker_service_account = (
                            worker_settings.worker_invoker_service_account
                        )

                if _deps is None:
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
                    )
                    artifact_store = LiveArtifactStore(
                        buckets={
                            Tier.working: live_settings.artifact_bucket_working,
                            Tier.pinned: live_settings.artifact_bucket_pinned,
                        },
                        inventory=LiveArtifactInventory(store=store),
                    )
                    code_index = get_index()
                    # Run state uses the top-level store. This worker only runs tasks.
                    # Dependency defaults stay inert.
                    _deps = _build_live_definition_deps(
                        store=store,
                        graph=graph,
                        sdmx=sdmx,
                        artifact_store=artifact_store,
                        code_index=code_index,
                        coverage=get_coverage_service(),
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

        assert _operation_store is not None and _deps is not None
        submission_count = await _task_submission_count(request)
        if submission_count is None:
            # Reject malformed tasks before claiming a lease.
            # Cloud Tasks retries the 400 to its cap. Dead-row expiry then fails the operation.
            logger.error(
                "Datasets worker: task for operation %r has no valid submission_count",
                operation_id,
            )
            return JSONResponse({"detail": "missing or invalid submission_count"}, status_code=400)
        submitted = _operation_store.get(operation_id)
        if submitted is None:
            return JSONResponse({"detail": "unknown operation"}, status_code=404)
        catalogue = _catalogue.snapshot()
        if catalogue.build_id != submitted.origin_catalogue_build_id:
            try:
                catalogue = await asyncio.to_thread(_catalogue.snapshot_current)
            except Exception:
                logger.exception(
                    "Datasets worker: could not read the published origin catalogue"
                    " for operation %r",
                    operation_id,
                )
                return JSONResponse({"detail": "origin catalogue unavailable"}, status_code=503)
        # Protects tests that issue concurrent claims through a thread pool.
        with _lock:
            operation = _operation_store.claim_lease(
                operation_id, submission_count=submission_count, lease_seconds=_lease_seconds
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
            return JSONResponse(
                {"detail": f"operation {operation_id!r} already has a live lease"},
                status_code=409,
            )

        is_final_attempt = _cloud_tasks_retry_count(request) >= _max_attempts - 1
        try:
            # One catalogue version serves the whole operation. `to_thread` carries the binding.
            with bound_catalogue(catalogue):
                await asyncio.to_thread(
                    run_in_thread,
                    operation=operation,
                    deps=_deps,
                    operation_store=_operation_store,
                    lease_seconds=_lease_seconds,
                    is_final_attempt=is_final_attempt,
                )
        except OperationSuperseded:
            # Another attempt owns the work. A 2xx stops this delivery.
            return JSONResponse({"status": "superseded"}, status_code=200)
        except OperationTransientFailure as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception:
            logger.exception("Datasets worker: operation %r crashed unexpectedly", operation_id)
            return JSONResponse({"detail": "unexpected worker failure"}, status_code=500)
        # Report the stored terminal state. Any terminal result gets 2xx to stop retries.
        final = _operation_store.get(operation_id)
        status = final.state if final is not None else "done"
        return JSONResponse({"status": status}, status_code=200)

    return app


__all__ = ["RUN_ROUTE_PATH", "WorkerSettings", "build_worker_app"]
