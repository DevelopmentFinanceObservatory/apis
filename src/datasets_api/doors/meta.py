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
posted, each artifact under its stable download link. The rendering key is
`artifact_rendering_key` of the row's stored `options` under this service's site
settings, the key the synchronous definition door narrows by, so the poll lists
that rendering's artifact plus statistics artifacts, as the synchronous mint does.
Its `trace` carries `served_from: materialise` and the `confirmed_at` the row
recorded at completion, the currency pair a synchronous mint that materialised
reports. A row an image recording no `confirmed_at` completed carries neither.
When the snapshot lists no artifact under that key (a bundle the worker could not
link, a bundle of a superseded rendition, an expired working-tier artifact), the
poll calls `OperationStore.fail_unservable_done` and answers retryable
`operation_failed` with `cause.code` `internal_error`. A resubmit resets the row and
re-runs it. Other snapshot errors keep their `datasets.snapshot` wire codes, so this
route maps a revoked version and an unknown version to the platform's `internal_error`
catch-all, as it does stored options that fail
`MintOptions` validation.

The fronting proxy authenticates operations-poll callers, as for public mint
and snapshot reads.

A poll of an abandoned row calls `OperationStore.abandon`, fenced on the attempt it read, and
answers the retryable `operation_failed` with `cause.code` `internal_error`; a resubmit resets
the row under a new task. A lease that has ended (`claimed_at + lease_seconds` before now) means
the attempt has stopped owning the row. The row is abandoned (`Operation.abandoned`) only when
nothing will finish it:

- `running` with its lease ended, claimed by the task's final delivery (`final_delivery` true),
  so no delivery is left;
- `running` with no `claimed_at` and an `updated_at` `LEASE_SECONDS` old, so the earlier image
  that claimed it and renewed its lease through `updated_at` has stopped, and the row records
  no delivery to wait for;
- `running` with its lease ended, claimed by an earlier delivery (`final_delivery` false), or
  `pending`, once `operation_deadline_seconds` have passed since its dispatch, or its submission
  when no dispatch is recorded.

A `running` row whose lease has ended while a later delivery is due polls as `running`, and the
poll writes nothing: the task's next delivery claims the row.

A poll of a `pending` row whose submission records no dispatch and is older than
`ENQUEUE_RECOVERY_SECONDS` recovers the enqueue: `dispatch_if_due` creates the task again
under the same name, `AlreadyExists` counting as success, and records `dispatched_at`. A
failed create is logged, and the caller's next poll tries again.
"""

from __future__ import annotations

import functools
import logging
from datetime import timedelta

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from code_index.query import CodeIndex
from code_index.shared import get_index, per_index
from dataset_store.clock import Clock
from dataset_store.snapshot import SnapshotNotFound, SnapshotReader, SnapshotRevoked
from datasets_api.backpressure import Backpressure
from datasets_api.capabilities import Capabilities, build_capabilities
from datasets_api.codes_snapshot import (
    SpecNotFound,
)
from datasets_api.config import (
    DatasetsBuildSettings,
)
from datasets_api.current import served_trace
from datasets_api.deps import DepsCell
from datasets_api.doors.snapshot import read_snapshot_manifest
from datasets_api.headers import no_store_headers
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    Operation,
    OperationBacklogFull,
    OperationStore,
)
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS, dispatch_if_due
from datasets_api.operations.task_client import (
    TaskClient,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import artifact_rendering_key, is_statistics_rendering, rendition_token
from datasets_api.retention import with_retention
from datasets_api.schema import (
    OperationEnvelope,
    OperationProgress,
    OperationResource,
    build_schema_document,
    schema_document_response_schema,
)
from datasets_api.timings import door_call, run_with_timings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform import HealthResponse, ProblemSpec, health_response
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution
from dfo_platform.problem import PROBLEM_JSON, Cause, Refusal
from dfo_platform.static_response import FrozenJSON, freeze_public_json
from spec_registry import IdentityQuarantined
from spec_registry.reader import SpecReader

logger = logging.getLogger(__name__)


class OperationNotFound(Refusal):
    """An unknown operation id. Maps to non-retryable `operation_not_found` (404)."""

    code = "operation_not_found"

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


class OperationBackpressure(Backpressure):
    """A full operations-read semaphore maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the operations door is at its concurrent-read capacity right now; retry"


def classify_meta_exception(exc: BaseException) -> ProblemSpec | None:
    """Map a failed operation, replaying its stored refusal, then any refusal. Return `None` for
    every other type."""
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
    return classify_datasets_refusal(exc)


SCHEMA_ROUTE_LIVE = True
"""Whether `GET /api/datasets/schema` is mounted on this deployment. The
schemas it serves carry `$id`s that are immutable once published."""


def _load_done_manifest(
    *,
    operation: Operation,
    spec_reader: SpecReader,
    snapshot_reader: SnapshotReader,
    now: Clock,
    settings: DatasetsBuildSettings,
) -> Manifest | None:
    """The done operation's manifest, narrowed to the rendering its request posted.

    Its `trace` carries the currency pair a synchronous mint that materialised reports:
    `served_from: materialise` and the `confirmed_at` the row recorded. It carries the
    version's `retention` (`datasets_api.retention`).

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
            now=now,
            settings=settings,
            rendering_key=key,
        )
    except (
        SpecNotFound,
        SnapshotNotFound,
        SnapshotRevoked,
        IdentityQuarantined,
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
    manifest = with_retention(
        manifest, subject_id=operation.subject_id, reader=snapshot_reader, now=now()
    )
    if operation.confirmed_at is None:
        return manifest
    trace = served_trace(
        manifest.trace, served_from="materialise", confirmed_at=operation.confirmed_at
    )
    return manifest.model_copy(update={"trace": trace})


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
    now: Clock,
    settings: DatasetsBuildSettings,
) -> OperationResource:
    if operation_store is None:
        # Requests before lifespan setup follow the unknown-id path.
        raise OperationNotFound(operation_id=operation_id)
    operation = operation_store.get(operation_id)
    if operation is None:
        raise OperationNotFound(operation_id=operation_id)
    polled_at = now()
    if operation.abandoned(
        polled_at, deadline=timedelta(seconds=settings.operation_deadline_seconds)
    ):
        operation = operation_store.abandon(
            operation.id, attempt=operation.attempt_count
        ) or operation_store.get(operation_id)
        if operation is None:
            raise OperationNotFound(operation_id=operation_id)
    if task_client is not None:
        operation = dispatch_if_due(
            operation,
            operation_store=operation_store,
            task_client=task_client,
            older_than=polled_at - timedelta(seconds=ENQUEUE_RECOVERY_SECONDS),
        )

    if operation.state in ("pending", "running"):
        return _progress(operation)
    if operation.state == "failed":
        raise _failed(operation)

    if spec_reader is None or snapshot_reader is None:
        # Lifespan supplies the readers the done manifest is read through.
        raise RuntimeError(
            f"operation {operation.id!r} is done but this door holds no live "
            "spec_reader/snapshot_reader to read its manifest"
        )
    manifest = _load_done_manifest(
        operation=operation,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
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


def build_meta_door(*, deps: DepsCell, settings: DatasetsBuildSettings) -> DoorContribution:
    """Build the meta door over the surface's dependency cell. `settings` sets the published
    capability limits. Operation polls run on the read executor.
    """
    router = APIRouter()

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

    @per_index
    def capabilities_document(index: CodeIndex) -> FrozenJSON:
        payload = build_capabilities(
            capability_version=DATASETS_CAPABILITY_VERSION,
            build=build_commit(),
            code_index_stamp=index.stamp,
            declarations=index.declarations,
            settings=settings,
        )
        return freeze_public_json(payload.model_dump(mode="json"), max_age=300)

    # Built at construction, so a build failure stops startup.
    capabilities_document(get_index())

    @router.get("/api/datasets/capabilities", response_model=Capabilities)
    async def capabilities(request: Request) -> Response:
        # The document carries the code index's stamp, so it is frozen per index.
        return capabilities_document(get_index()).respond(request)

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
        current = deps.get()
        with door_call("operations", logger=logger) as call:
            resource = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    _resolve_operation,
                    call.timings,
                    operation_id=operation_id,
                    operation_store=current.operation_store,
                    task_client=current.task_client,
                    spec_reader=current.spec_reader,
                    snapshot_reader=current.snapshot_reader,
                    now=current.now,
                    settings=current.settings,
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
            OperationBackpressure(),
            OperationBacklogFull(in_flight=1, limit=1),
        ),
        registry_endpoints={"/api/datasets/operations/{operation_id}": "datasets.operations"},
    )
