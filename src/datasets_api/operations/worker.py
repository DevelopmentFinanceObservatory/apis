"""Runs claimed operations through the shared definition materialisation path.

The worker calls `serve_definition` after admission. The door's `_serve_spec` closures would
re-check the sync budget and raise `OperationSubmitted` again, so the worker bypasses them. It
reuses the request resolvers from `datasets_api.doors.definition`. A definition resolves its
members as the door does, from the identity alone. When they differ from the `member_dcids` its
operation stored, the graph changed after submission: the row fails retryable with
`upstream_version_changed`, and a resubmission resolves the new members under a new operation id.

A heartbeat renews the lease with `RowProgress.value` every third of its duration, with a
one-second minimum. `run_operation` starts the heartbeat with materialisation and joins it
before returning or raising. A lost fencing token cancels work at the next chunk boundary.
Deterministic failures are terminal. Transient failures release the row to `pending` with
`lease_until` set to now, so the next Cloud Tasks retry can claim it at once. The final attempt
records a retryable failure instead, so an identical resubmit can reset the row. Errors from
parsing the stored request or `raw_fetch_guard` take the same failure path.

A row completes only when a snapshot read of its version narrowed to the request's rendering
key (`artifact_rendering_key` under `settings.site`) lists an artifact other than statistics
(`rendering_servable`), the read the operations poll answers with. A render that leaves its
bundle unlinked, after a failed graph label or documentation read, takes the transient path.
"""

from __future__ import annotations

import logging
import threading

from datasets_api.admission import refuse_async_over_ceiling, sync_estimate
from datasets_api.doors.definition import (
    DefinitionDeps,
    _resolve_spec_requests,
    _serve_deps,
    classify_definition_exception,
)
from datasets_api.doors.snapshot import rendering_servable
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import Operation, OperationStore
from datasets_api.operations.submit import (
    member_dcids,
    request_identity_member_dcids,
    request_identity_subject_id,
)
from datasets_api.render import artifact_rendering_key
from datasets_api.serve import raw_fetch_guard, serve_definition
from materialise import MaterialisationCancelled, RowProgress
from materialise.errors import UpstreamVersionChangedError

logger = logging.getLogger(__name__)

# A third-lease interval leaves two missed ticks before expiry on longer leases.
_HEARTBEAT_TICKS_PER_LEASE = 3
_HEARTBEAT_MIN_INTERVAL_SECONDS = 1.0


class _MembersChanged(UpstreamVersionChangedError):
    """The definition resolves to members other than the ones its operation stored. It is a
    retryable failure the worker records at once, since another attempt reads the same graph."""

    def __init__(self, *, subject_id: str, stored: tuple[str, ...], resolved: tuple[str, ...]):
        super().__init__(
            f"{subject_id!r} resolves {len(resolved)} member(s) now, not the {len(stored)} its "
            "operation was submitted with; the graph changed after submission, so submit the "
            "request again"
        )


class OperationSuperseded(Exception):
    """Raised when this attempt loses its fence before finishing.

    A newer attempt owns the row, or the row is terminal. The worker app answers 2xx and writes
    nothing further for this attempt.
    """


class _RenderingNotServable(RuntimeError):
    """The completed render left no servable artifact under the request's rendering key.
    It is unclassified, so it takes the transient path."""


class OperationTransientFailure(Exception):
    """Raised for unclassified or retryable failures before the final attempt, prompting a
    Cloud Tasks retry. The original exception is `__cause__`.
    """


def _start_heartbeat(
    *,
    operation: Operation,
    operation_store: OperationStore,
    lease_seconds: float,
    row_progress: RowProgress,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    interval = max(_HEARTBEAT_MIN_INTERVAL_SECONDS, lease_seconds / _HEARTBEAT_TICKS_PER_LEASE)

    def _beat() -> None:
        while not stop.wait(interval):
            try:
                renewed = operation_store.progress(
                    operation.id,
                    attempt=operation.attempt_count,
                    rows_written=row_progress.value,
                    lease_seconds=lease_seconds,
                )
            except Exception:
                # Later heartbeat ticks can renew after a missed store write.
                logger.warning(
                    "Datasets operation %r: heartbeat renewal failed; will retry next tick",
                    operation.id,
                    exc_info=True,
                )
                continue
            if renewed is None:
                # Stop at the next chunk boundary after losing the fence.
                logger.warning(
                    "Datasets operation %r: attempt %d no longer owns the row; cancelling the run",
                    operation.id,
                    operation.attempt_count,
                )
                row_progress.cancel()
                return

    thread = threading.Thread(
        target=_beat, name=f"datasets-operation-heartbeat-{operation.id}", daemon=True
    )
    thread.start()
    return stop, thread


def run_operation(
    *,
    operation: Operation,
    deps: DefinitionDeps,
    operation_store: OperationStore,
    lease_seconds: float,
    is_final_attempt: bool = False,
) -> None:
    """Runs a caller-claimed operation in `running` state.

    Completes or fails the row, or releases it and raises `OperationTransientFailure` for a
    transient error before the final attempt. Each run reads `current.json` and loads its build
    before resolving requests. It re-estimates against that snapshot and records
    the execution stamp and estimate. The fresh estimate sizes `raw_fetch_guard` and checks async
    row and request limits before fetching. Pointer or build failures are transient.
    `over_ceiling` is terminal. The operation id includes the admission stamp.

    The row completes only when the request's rendering is servable (`rendering_servable`).
    Otherwise the run is a transient failure, so a final attempt fails it with `internal_error`.

    Deterministic failures set `retryable=False`. A definition whose members differ from the
    operation's `member_dcids` fails at once with `upstream_version_changed` and
    `retryable=True`. A final transient or unclassified failure sets
    `retryable=True`, using the classified code or `internal_error`. Raises `OperationSuperseded`
    if the execution record or heartbeat loses its fencing token. Cancellation stops work at the
    next chunk boundary.
    """
    row_progress = RowProgress()
    stop_heartbeat, heartbeat_thread = _start_heartbeat(
        operation=operation,
        operation_store=operation_store,
        lease_seconds=lease_seconds,
        row_progress=row_progress,
    )
    try:
        try:
            # Keep request parsing and guard checks inside the failure-classification path.
            subject_id = request_identity_subject_id(operation.request_identity)
            options = MintOptions.model_validate(operation.options)
            serve_deps = _serve_deps(deps)
            # Use one index/stamp snapshot for estimation, recording, and materialisation.
            coverage_snapshot = deps.coverage.snapshot_current()
            if not subject_id.startswith("spec_"):
                raise ValueError(
                    f"operation {operation.id!r} names subject_id {subject_id!r}, which is "
                    "not spec_-prefixed"
                )
            stored, requests = _resolve_spec_requests(spec_id=subject_id, deps=deps)
            _check_members_unchanged(
                subject_id,
                stored=request_identity_member_dcids(operation.request_identity),
                resolved=member_dcids([request.stat_var for request in requests]),
            )
            estimate = sync_estimate(
                requests,
                index=coverage_snapshot.index,
                member_concurrency=deps.member_concurrency,
            )
            recorded = operation_store.record_execution(
                operation.id,
                attempt=operation.attempt_count,
                coverage_stamp=coverage_snapshot.stamp,
                estimate=estimate,
            )
            if recorded is None:
                row_progress.cancel()
            row_progress.raise_if_cancelled()
            refuse_async_over_ceiling(
                estimate,
                async_max_rows=deps.settings.async_max_rows,
                async_max_requests=deps.settings.async_max_requests,
            )
            manifest = serve_definition(
                stored=stored,
                requests=requests,
                options=options,
                deps=serve_deps,
                ceiling=raw_fetch_guard(estimate, floor=deps.ceiling),
                coverage_stamp=coverage_snapshot.stamp,
                coverage_index=coverage_snapshot.index,
                progress=row_progress,
            )
            completed_subject_id = manifest.definition.spec_id
            key = artifact_rendering_key(options, site=deps.settings.site)
            if not rendering_servable(
                subject_id=completed_subject_id,
                content_version=manifest.content_version,
                rendering_key=key,
                snapshot_reader=deps.snapshot_reader,
                now=deps.now,
                settings=deps.settings,
            ):
                raise _RenderingNotServable(
                    f"subject_id={completed_subject_id!r} content_version="
                    f"{manifest.content_version!r} has no servable artifact under rendering "
                    f"key {key!r}"
                )
        except MaterialisationCancelled as exc:
            logger.warning(
                "Datasets operation %r: attempt %d stopped without writing; the row "
                "belongs to another attempt or is no longer running",
                operation.id,
                operation.attempt_count,
            )
            raise OperationSuperseded(str(exc)) from exc
        except Exception as exc:
            spec = classify_definition_exception(exc)
            if spec is not None and (not spec.retryable or isinstance(exc, _MembersChanged)):
                logger.info(
                    "Datasets operation %r failed without a retry (%s): %s",
                    operation.id,
                    spec.code,
                    exc,
                )
                operation_store.fail(
                    operation.id,
                    attempt=operation.attempt_count,
                    refusal_code=spec.code,
                    retryable=spec.retryable,
                    refusal=spec,
                )
                return
            if is_final_attempt:
                code = spec.code if spec is not None else "internal_error"
                logger.warning(
                    "Datasets operation %r failed transiently on its own last Cloud Tasks "
                    "attempt; marking it failed (%s) rather than leaving it for a retry that "
                    "will never come: %r",
                    operation.id,
                    code,
                    exc,
                )
                operation_store.fail(
                    operation.id,
                    attempt=operation.attempt_count,
                    refusal_code=code,
                    retryable=True,
                )
                return
            logger.warning(
                "Datasets operation %r failed transiently, releasing its lease for the next "
                "retry to reclaim immediately: %r",
                operation.id,
                exc,
            )
            operation_store.release(operation.id, attempt=operation.attempt_count)
            raise OperationTransientFailure(str(exc)) from exc
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=lease_seconds)

    completed = operation_store.complete(
        operation.id,
        attempt=operation.attempt_count,
        subject_id=completed_subject_id,
        content_version=manifest.content_version,
    )
    if completed is None:
        # The operation row has moved on. Its durable artifact has no operation reference.
        logger.error(
            "Datasets operation %r: materialisation for subject_id=%r "
            "content_version=%r completed successfully, but the operation row was no "
            "longer 'running' by the time this call tried to record it; the artifact "
            "needs no cleanup (content-addressed), but no operation row names it now",
            operation.id,
            completed_subject_id,
            manifest.content_version,
        )


def _check_members_unchanged(
    subject_id: str, *, stored: tuple[str, ...] | None, resolved: tuple[str, ...]
) -> None:
    """Raise `_MembersChanged` when the operation stored members and `resolved` differs.
    An operation recorded before definitions stored members has none to compare."""
    if stored is not None and tuple(sorted(stored)) != resolved:
        raise _MembersChanged(subject_id=subject_id, stored=stored, resolved=resolved)


__all__ = ["OperationSuperseded", "OperationTransientFailure", "run_operation"]
