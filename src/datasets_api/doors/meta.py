"""The meta door serves health, schema, capabilities, and operation polling.

Its static routes mount before other doors. Subject routes accept only `spec_`
ids, which occupy a separate route shape.

`GET /api/datasets/operations/{operation_id}` reads `datasets.operation`.
Unknown ids return `operation_not_found` (404). Pending and running operations
return progress (200), with rows written against the estimate. Failed operations
return `operation_failed` (500) as RFC 9457 problem details, with the stored
refusal code in `cause.code`. A deterministic failure also stores the refusal the
worker classified, and the body carries its detail, pointer, alternatives, issues,
ceiling fields and cause entities, as `GET /api/datasets/{id}` renders them for
the same failure.
A done operation returns the mint manifest narrowed to the rendering its request
posted, with signed URLs refreshed at read time. The rendering key is
`artifact_rendering_key` of the row's stored `options` under this service's site
settings, the key the synchronous definition door narrows by, so the poll lists
that rendering's artifact plus statistics artifacts, as the synchronous mint does.
When the snapshot lists no artifact under that key (a bundle the worker could not
link, a bundle of a superseded rendition, an expired working-tier artifact), the
poll calls `OperationStore.fail_unservable_done` and answers retryable
`operation_failed` with `cause.code` `internal_error`. A resubmit resets the row and
re-runs it. Other snapshot errors keep their `datasets.snapshot` wire codes, so this
route maps a revoked version, an unknown version and a signing failure to the
platform's `internal_error` catch-all, as it does stored options that fail
`MintOptions` validation.

The fronting proxy authenticates operations-poll callers, as for public mint
and snapshot reads.

Each poll calls `OperationStore.expire_if_dead` before reading. One conditional
update fails work whose lease outlasts Cloud Tasks' retry window plus a margin,
or whose first dispatch is overdue by the worst-case backlog wait. Heartbeats,
completions, terminal rows, and unknown ids stay unchanged. `create_or_get` uses
the same thresholds to sweep rows. `OperationStore` defines
`lease_stale_after_seconds` and `never_claimed_stale_after_seconds`.
Expiration records retryable `internal_error`, which the poll returns as
`operation_failed`.

A pending poll also calls `dispatch_if_due`. Polls and submissions compete for
a conditional claim. The winner enqueues and confirms the task. An enqueue
failure is logged. The caller's normal poll loop retries the task.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from code_index.shared import get_index
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import LiveSnapshotReader, SnapshotReader, SnapshotRevoked
from datasets_api.capabilities import Capabilities, build_capabilities
from datasets_api.codes_snapshot import (
    SnapshotNotFound,
    SnapshotSigningFailed,
    SpecNotFound,
)
from datasets_api.config import (
    DatasetsBuildSettings,
    DatasetsConfigError,
    DatasetsSettings,
)
from datasets_api.door_executor import build_door_executor
from datasets_api.doors.snapshot import read_snapshot_manifest
from datasets_api.headers import no_store_headers
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    LiveOperationStore,
    Operation,
    OperationBacklogFull,
    OperationStore,
)
from datasets_api.operations.dispatch import dispatch_if_due
from datasets_api.operations.task_client import (
    LiveTaskClient,
    TaskClient,
    TaskClientSettings,
    build_live_task_client,
)
from datasets_api.render import artifact_rendering_key, is_statistics_rendering, rendition_token
from datasets_api.schema import (
    OperationEnvelope,
    OperationProgress,
    OperationResource,
    build_schema_document,
    schema_document_response_schema,
)
from datasets_api.shutdown import ShutdownBudget
from datasets_api.timings import door_call, run_with_timings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform import HealthResponse, ProblemSpec, health_response
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution
from dfo_platform.problem import PROBLEM_JSON, Cause
from dfo_platform.static_response import freeze_public_json
from series_coverage.shared import get_coverage_service
from spec_registry.reader import LiveSpecReader, SpecReader

logger = logging.getLogger(__name__)


class OperationNotFound(Exception):
    """An unknown operation id. Maps to non-retryable `operation_not_found` (404)."""

    def __init__(self, *, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__(f"no operation exists with id {operation_id!r}")


class OperationFailed(Exception):
    """A failed operation. Maps to `operation_failed` (500) with the stored refusal code as
    `cause.code` and the stored `retryable` flag. `refusal` is the classified refusal a
    deterministic failure stored, whose detail, pointer, alternatives, issues, ceiling fields
    and cause entities the problem body carries. Without one the detail names the refusal
    code alone."""

    def __init__(
        self,
        *,
        operation_id: str,
        refusal_code: str | None,
        retryable: bool | None = None,
        refusal: ProblemSpec | None = None,
    ) -> None:
        self.operation_id = operation_id
        self.refusal_code = refusal_code
        self.retryable = bool(retryable)
        self.refusal = refusal
        detail = f"operation {operation_id!r} did not complete successfully"
        if refusal_code is not None:
            detail = f"{detail}: {refusal_code}"
        super().__init__(detail)


class OperationBackpressure(Exception):
    """A full operations-read semaphore maps to retryable `service_unavailable` (503)."""


def classify_meta_exception(exc: BaseException) -> ProblemSpec | None:
    """Map this door's exceptions to problems. Return `None` for unclassified types."""
    if isinstance(exc, OperationNotFound):
        return ProblemSpec(status=404, code="operation_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, OperationFailed):
        problem = ProblemSpec(
            status=500,
            code="operation_failed",
            detail=str(exc),
            retryable=exc.retryable,
            cause=None if exc.refusal_code is None else Cause(code=exc.refusal_code),
        )
        stored = exc.refusal
        if stored is None:
            return problem
        # The status, code and retryable flag stay the operation's. The stored refusal's code
        # is `cause.code`, beside the entities its own cause names. Its status and its own
        # cause code are not rendered.
        cause = problem.cause
        if cause is not None and stored.cause is not None:
            cause = cause.model_copy(
                update={"entities": stored.cause.entities, "truncated": stored.cause.truncated}
            )
        return problem._replace(
            cause=cause,
            detail=stored.detail if stored.detail is not None else problem.detail,
            pointer=stored.pointer,
            alternatives=stored.alternatives,
            truncated=stored.truncated,
            issues=stored.issues,
            constraint_id=stored.constraint_id,
            ceiling_kind=stored.ceiling_kind,
            count=stored.count,
            cap=stored.cap,
        )
    if isinstance(exc, OperationBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the operations door is at its concurrent-read capacity right now; retry",
            retryable=True,
        )
    return None


SCHEMA_ROUTE_LIVE = True
"""Whether `GET /api/datasets/schema` is mounted on this deployment. The
schemas it serves carry `$id`s that are immutable once published."""


def _load_done_manifest(
    *,
    operation: Operation,
    spec_reader: SpecReader,
    snapshot_reader: SnapshotReader,
    artifact_store: ArtifactStore,
    now: Clock,
    settings: DatasetsBuildSettings,
) -> Manifest | None:
    """The done operation's manifest, narrowed to the rendering its request posted.

    The rendering key is `artifact_rendering_key` of the row's `options` under this service's
    `settings.site`, the key the synchronous definition door narrows by. The read lists that
    rendering's artifact plus statistics artifacts. Returns `None` when it lists no artifact
    other than statistics, as for a bundle the worker could not link, a bundle of a superseded
    rendition or an expired working-tier artifact.

    Stored options that fail `MintOptions` validation raise `ValidationError`, and snapshot
    failures raise `RuntimeError`. This route maps both to `internal_error`.
    """
    assert operation.subject_id is not None and operation.content_version is not None
    options = MintOptions.model_validate(operation.options)
    key = artifact_rendering_key(options, site=settings.site)
    try:
        manifest = read_snapshot_manifest(
            id=operation.subject_id,
            content_version=operation.content_version,
            spec_reader=spec_reader,
            snapshot_reader=snapshot_reader,
            artifact_store=artifact_store,
            now=now,
            settings=settings,
            rendering_key=key,
        )
    except (
        SpecNotFound,
        SnapshotNotFound,
        SnapshotRevoked,
        SnapshotSigningFailed,
    ) as exc:
        # Snapshot failures map to this route's `internal_error` catch-all.
        logger.error(
            "Datasets operations door: operation %r is done (subject_id=%r, "
            "content_version=%r) but its own manifest could not be re-read: %r",
            operation.id,
            operation.subject_id,
            operation.content_version,
            exc,
        )
        raise RuntimeError(
            f"operation {operation.id!r} completed but its own manifest is no longer servable"
        ) from exc
    if not any(not is_statistics_rendering(artifact.rendering) for artifact in manifest.artifacts):
        logger.warning(
            "Datasets operations door: operation %r is done (subject_id=%r, "
            "content_version=%r) but the snapshot lists no artifact under its rendering "
            "key %r at rendition %r",
            operation.id,
            operation.subject_id,
            operation.content_version,
            key,
            rendition_token(settings.site),
        )
        return None
    return manifest


def _failed(operation: Operation) -> OperationFailed:
    return OperationFailed(
        operation_id=operation.id,
        refusal_code=operation.refusal_code,
        retryable=operation.retryable,
        refusal=operation.refusal,
    )


def _progress(operation: Operation) -> OperationResource:
    return OperationResource(
        operation=OperationEnvelope(
            operation_id=operation.id,
            status=operation.state,
            poll=f"/api/datasets/operations/{operation.id}",
            progress=OperationProgress(
                rows_written=operation.rows_written,
                estimate_rows=operation.estimate.rows,
            ),
        )
    )


def _resolve_operation(
    *,
    operation_id: str,
    operation_store: OperationStore | None,
    task_client: TaskClient | None,
    spec_reader: SpecReader | None,
    snapshot_reader: SnapshotReader | None,
    artifact_store: ArtifactStore | None,
    now: Clock,
    settings: DatasetsBuildSettings,
) -> OperationResource:
    if operation_store is None:
        # Requests before lifespan setup follow the unknown-id path.
        raise OperationNotFound(operation_id=operation_id)
    # Expire stale work before reading its latest state.
    operation = operation_store.expire_if_dead(operation_id) or operation_store.get(operation_id)
    if operation is None:
        raise OperationNotFound(operation_id=operation_id)
    if task_client is not None:
        # Reconcile overdue or unconfirmed task dispatches before replying.
        operation = dispatch_if_due(
            operation, operation_store=operation_store, task_client=task_client
        )

    if operation.state in ("pending", "running"):
        return _progress(operation)
    if operation.state == "failed":
        raise _failed(operation)

    if spec_reader is None or snapshot_reader is None or artifact_store is None:
        # Lifespan supplies the readers and artifact store used to re-sign manifests.
        raise RuntimeError(
            f"operation {operation.id!r} is done but this door holds no live "
            "spec_reader/snapshot_reader/artifact_store to re-sign its manifest"
        )
    manifest = _load_done_manifest(
        operation=operation,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
        artifact_store=artifact_store,
        now=now,
        settings=settings,
    )
    if manifest is not None:
        return OperationResource(
            operation=OperationEnvelope(operation_id=operation.id, status="done"),
            manifest=manifest.model_dump(mode="json", exclude_none=True),
        )

    # The rendering is gone: fail the row retryable so a resubmit re-runs it.
    failed = operation_store.fail_unservable_done(operation.id, attempt=operation.attempt_count)
    if failed is not None:
        raise _failed(failed)
    # A concurrent poll or resubmit moved the row first. Answer from its current state.
    current = operation_store.get(operation.id)
    if current is None:
        raise OperationNotFound(operation_id=operation.id)
    if current.state in ("pending", "running"):
        return _progress(current)
    if current.state == "failed":
        raise _failed(current)
    # Done again after a resubmit re-ran it: the next poll reads the new completion.
    raise OperationFailed(operation_id=operation.id, refusal_code="internal_error", retryable=True)


def build_meta_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    operation_store: OperationStore | None = None,
    task_client: TaskClient | None = None,
    spec_reader: SpecReader | None = None,
    snapshot_reader: SnapshotReader | None = None,
    artifact_store: ArtifactStore | None = None,
    now: Clock = utc_now,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the meta door and its operations-route executor.

    `settings` defaults from the environment and sets concurrency and published
    capability limits. Inject `spec_reader`, `snapshot_reader`, and
    `artifact_store` together. If all three are unset, lifespan builds them with
    one pool borrow. A partial set raises `DatasetsConfigError`.
    `operation_store` is independent and shares that borrow when built live.
    Lifespan builds an omitted `task_client` from `TaskClientSettings` and closes
    it at shutdown.

    `pool_size` defaults to `settings.operations_max_concurrency`. `budget` is
    shared across mounted doors. An omitted budget creates a private one.
    """
    missing = [
        name
        for name, value in (
            ("spec_reader", spec_reader),
            ("snapshot_reader", snapshot_reader),
            ("artifact_store", artifact_store),
        )
        if value is None
    ]
    if missing and len(missing) != 3:
        raise DatasetsConfigError(
            "datasets meta door built with "
            f"{', '.join(missing)} left uninjected while another dependency is injected. A "
            "caller injecting any one of spec_reader/snapshot_reader/artifact_store must "
            "inject every one of them. This door builds a live dependency for all three "
            "only when every one of them is left None."
        )

    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    _operation_store: OperationStore | None = operation_store
    _task_client: TaskClient | None = task_client
    _spec_reader: SpecReader | None = spec_reader
    _snapshot_reader: SnapshotReader | None = snapshot_reader
    _artifact_store: ArtifactStore | None = artifact_store

    effective_pool_size = (
        pool_size if pool_size is not None else _settings.operations_max_concurrency
    )
    door_executor = build_door_executor(effective_pool_size, "operations")
    executor = door_executor.executor

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _operation_store, _task_client, _spec_reader, _snapshot_reader, _artifact_store
        pool_opened_here = False
        store: LiveDatasetStore | None = None

        def _open_store() -> LiveDatasetStore:
            nonlocal store, pool_opened_here
            if store is None:
                store = LiveDatasetStore(connections=shared_pool())
                pool_opened_here = True
            return store

        def _close_what_this_lifespan_opened() -> None:
            if pool_opened_here:
                close_shared_pool()
            if task_client is None and isinstance(_task_client, LiveTaskClient):
                _task_client.close()

        if operation_store is None:
            _operation_store = LiveOperationStore(store=_open_store(), settings=_settings)
        if spec_reader is None:
            _spec_reader = LiveSpecReader(store=_open_store())
            _snapshot_reader = LiveSnapshotReader(store=_open_store())
        if task_client is None:
            try:
                task_settings = TaskClientSettings.from_env()
                _task_client = build_live_task_client(
                    queue=task_settings.tasks_queue,
                    worker_url=task_settings.worker_url,
                    worker_invoker_service_account=task_settings.worker_invoker_service_account,
                )
            except BaseException:
                _close_what_this_lifespan_opened()
                raise
        if artifact_store is None:
            try:
                live_settings = DatasetsSettings.from_env()
                _artifact_store = LiveArtifactStore(
                    buckets={
                        Tier.working: live_settings.artifact_bucket_working,
                        Tier.pinned: live_settings.artifact_bucket_pinned,
                    },
                    inventory=LiveArtifactInventory(store=_open_store()),
                )
            except BaseException:
                _close_what_this_lifespan_opened()
                raise
        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(door_executor.drain, _budget)
            _close_what_this_lifespan_opened()

    router = APIRouter()

    # Load the process-wide coverage artifact at startup. A missing URI or
    # unreadable artifact fails surface construction.
    get_coverage_service()

    @router.get("/api/datasets/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return health_response(capability_version=DATASETS_CAPABILITY_VERSION)

    if SCHEMA_ROUTE_LIVE:
        schema_document = freeze_public_json(build_schema_document(), max_age=300)

        @router.get(
            "/api/datasets/schema",
            responses={
                200: {
                    "content": {"application/json": {"schema": schema_document_response_schema()}}
                }
            },
        )
        async def schema(request: Request) -> Response:
            return schema_document.respond(request)

    capabilities_payload = build_capabilities(
        capability_version=DATASETS_CAPABILITY_VERSION,
        build=build_commit(),
        code_index_stamp=get_index().stamp,
        declarations=get_index().declarations,
        settings=_settings,
    )
    capabilities_document = freeze_public_json(
        capabilities_payload.model_dump(mode="json"), max_age=300
    )

    @router.get("/api/datasets/capabilities", response_model=Capabilities)
    async def capabilities(request: Request) -> Response:
        return capabilities_document.respond(request)

    @router.get(
        "/api/datasets/operations/{operation_id}",
        response_model=OperationResource,
        responses={
            404: {
                "description": "Operation not found",
                "content": {
                    PROBLEM_JSON: {"schema": {"$ref": "#/components/schemas/ProblemDetails"}}
                },
            },
            500: {
                "description": "The operation ran and did not complete successfully",
                "content": {
                    PROBLEM_JSON: {"schema": {"$ref": "#/components/schemas/ProblemDetails"}}
                },
            },
        },
    )
    async def get_operation(operation_id: str) -> JSONResponse:
        with door_call("operations", logger=logger) as call:
            resource = await door_executor.run(
                functools.partial(
                    run_with_timings,
                    _resolve_operation,
                    call.timings,
                    operation_id=operation_id,
                    operation_store=_operation_store,
                    task_client=_task_client,
                    spec_reader=_spec_reader,
                    snapshot_reader=_snapshot_reader,
                    artifact_store=_artifact_store,
                    now=now,
                    settings=_settings,
                ),
                refuse=OperationBackpressure,
            )
        return JSONResponse(
            resource.model_dump(mode="json", exclude_none=True),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        classify=classify_meta_exception,
        # Surface-level store classification handles failures this door can raise.
        exception_types=(
            OperationNotFound,
            OperationFailed,
            OperationBackpressure,
            OperationBacklogFull,
        ),
        exception_exemplars=(
            OperationNotFound(operation_id="op_exemplar"),
            OperationFailed(operation_id="op_exemplar", refusal_code="excel_limit"),
        ),
        lifespan=lifespan,
    )
