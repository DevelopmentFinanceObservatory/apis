"""Runs claimed operations through the shared definition materialisation path.

The worker admits the operation itself, then calls `current.serve_current` with that estimate,
so the sync budget is not checked again and no `OperationSubmitted` is raised. The head rule is
the doors': a current head that holds the request's rendering answers with no fetch, and a
rebuild records the version and advances the head, so the next GET of the dataset is a head
hit. It reuses the request resolver from `datasets_api.doors.definition`, which resolves the
operation's subject to its served definition. A definition resolves its members as the door
does, from the identity alone. When they differ from the `member_dcids` its
operation stored, the graph changed after submission: the row fails retryable with
`upstream_version_changed`, and a resubmission resolves the new members under a new operation id.

Every write is fenced on the claim's attempt token. Progress is written from the materialiser's own
progress calls (`_FencedProgress`), with no thread: each completed chunk writes `rows_written`, and
a write the fence refuses cancels the run at the next chunk boundary. The head advance carries the
claim as a `HeadFence` (`DefinitionDeps.head_fence`), checked under a lock on the operation row in
the head's own transaction, so a run that lost the row to another attempt raises `HeadFenceLost`,
leaves the head and the materialisation table unchanged, and ends as `OperationSuperseded`. The
run's work deadline, which the worker route sets `RUN_DEADLINE_SECONDS` after the claim time,
replaces `deps.deadline` and bounds the whole run: the write half of `serve_definition` checks it
before the version record and before each artifact render and upload, and a run past it takes the
transient path (`WorkDeadlinePassed`). The claim's lease, `LEASE_SECONDS`, outlasts the work
deadline by the time the final writes need, and a head advance after the deadline and inside the
lease is safe because the fence refuses it once another attempt claims the row. Deterministic
failures are terminal. Transient failures release the row to `pending`, so the next Cloud Tasks
retry can claim it at once. The final attempt records a retryable failure instead, so an identical
resubmit can reset the row. Errors from parsing the stored request or `raw_fetch_guard` take the
same failure path.

A row completes only when a snapshot read of its version narrowed to the request's rendering
keys (`artifact_rendering_keys` under `settings.site`) lists an artifact other than statistics
(`rendering_servable`), the read the operations poll answers with. A render that links its
bundle as incomplete, after a failed graph label or documentation read, takes the transient
path. The completed row records the `trace.confirmed_at` the run served, the instant it
confirmed the subject head, which the poll stamps on the done manifest.
"""

from __future__ import annotations

import dataclasses
import logging
import threading

from datasets_api.adapter import family_basis, identity_composite
from datasets_api.admission import (
    refuse_async_over_ceiling,
    sync_estimate,
)
from datasets_api.current import DefinitionDeps, serve_current, trace_confirmed_at
from datasets_api.default_enrichment import resolve_enrichment
from datasets_api.doors.definition import _resolve_spec_requests, classify_definition_exception
from datasets_api.doors.snapshot import rendering_servable
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import Operation, OperationStore
from datasets_api.operations.lease import LEASE_SECONDS, RUN_DEADLINE_SECONDS
from datasets_api.operations.submit import (
    member_dcids,
    request_identity_member_dcids,
    request_identity_subject_id,
)
from datasets_api.render import artifact_rendering_keys
from datasets_api.timings import collecting, door_call
from datasets_api.versions import HeadFence, HeadFenceLost
from dfo_platform.codes import bound_registry_endpoint
from materialise import MaterialisationCancelled, RowProgress
from materialise.errors import UpstreamVersionChangedError
from materialise.trace import trace_family
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


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


class _FencedProgress(RowProgress):
    """Writes the running total on every `add`, fenced on the claim's attempt token. A write
    the fence refuses cancels the run; a store failure is logged and the next chunk writes
    again. One lock orders the writes, so the stored total never moves backwards."""

    def __init__(self, *, operation: Operation, operation_store: OperationStore) -> None:
        super().__init__()
        self._operation = operation
        self._operation_store = operation_store
        self._write_lock = threading.Lock()

    def add(self, rows: int) -> None:
        super().add(rows)
        with self._write_lock:
            try:
                written = self._operation_store.progress(
                    self._operation.id,
                    attempt=self._operation.attempt_count,
                    rows_written=self.value,
                )
            except Exception:
                logger.warning(
                    "Datasets operation %r: progress write failed; the next chunk writes again",
                    self._operation.id,
                    exc_info=True,
                )
                return
        if written is None:
            logger.warning(
                "Datasets operation %r: attempt %d no longer owns the row; cancelling the run",
                self._operation.id,
                self._operation.attempt_count,
            )
            self.cancel()


def run_operation(
    *,
    operation: Operation,
    deps: DefinitionDeps,
    operation_store: OperationStore,
    deadline: float | None = None,
    is_final_attempt: bool = False,
) -> None:
    """Runs a caller-claimed operation in `running` state.

    `deadline` is the run's absolute monotonic work deadline. It replaces `deps.deadline`, which
    bounds SDMX and member reads and the materialise permit wait, and becomes the serve
    dependencies' `work_deadline`, checked before every write.

    Completes or fails the row, or releases it and raises `OperationTransientFailure` for a
    transient error before the final attempt. Each run takes its bound release's coverage
    snapshot before resolving requests. It re-estimates against that snapshot and records the
    estimate. The fresh estimate sizes `raw_fetch_guard` and checks async row and request limits
    before fetching. `over_ceiling` is terminal.

    The row completes only when the request's rendering is servable (`rendering_servable`).
    Otherwise the run is a transient failure, so a final attempt fails it with `internal_error`.

    Deterministic failures set `retryable=False`. A definition whose members differ from the
    operation's `member_dcids` fails at once with `upstream_version_changed` and
    `retryable=True`. A final transient or unclassified failure sets
    `retryable=True`, using the classified code or `internal_error`. A failure whose
    classification raises fails the row at once with `internal_error` and `retryable=True`, so
    an identical resubmit can reset it. Raises `OperationSuperseded`
    if the estimate record, a progress write, or the head advance loses its fencing token. Cancellation stops work
    at the next chunk boundary.
    """
    deps = dataclasses.replace(
        deps, head_fence=HeadFence(operation_id=operation.id, attempt=operation.attempt_count)
    )
    if deadline is not None:
        deps = dataclasses.replace(deps, deadline=deadline, work_deadline=deadline)
    row_progress = _FencedProgress(operation=operation, operation_store=operation_store)
    try:
        # Keep request parsing and guard checks inside the failure-classification path.
        subject_id = request_identity_subject_id(operation.request_identity)
        options = MintOptions.model_validate(operation.options)

        def _run(coverage_snapshot: CoverageSnapshot) -> Manifest:
            # Use one index/stamp snapshot for estimation, recording, and materialisation.
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
                composite=identity_composite(stored.identity, code_index=deps.code_index),
            )
            recorded = operation_store.record_estimate(
                operation.id, attempt=operation.attempt_count, estimate=estimate
            )
            if recorded is None:
                row_progress.cancel()
            row_progress.raise_if_cancelled()
            refuse_async_over_ceiling(
                estimate,
                async_max_rows=deps.settings.async_max_rows,
                async_max_requests=deps.settings.async_max_requests,
            )
            with door_call("worker", logger=logger) as call, collecting(call.timings):
                return serve_current(
                    stored,
                    resolve_requests=lambda: requests,
                    options=options,
                    trigger="worker",
                    deps=deps,
                    trace_id=None,
                    admitted=estimate,
                    coverage_snapshot=coverage_snapshot,
                    progress=row_progress,
                )

        manifest = _run(deps.coverage.snapshot())
        completed_subject_id = manifest.definition.spec_id
        if options.enrichment is not None:
            # The serve resolved the default enrichment against the version it built.
            options = resolve_enrichment(options, coverage=manifest.coverage)
        keys = artifact_rendering_keys(
            options, site=deps.settings.site, enrichment_digest=deps.code_index.enrichment_digest
        )
        if not rendering_servable(
            subject_id=completed_subject_id,
            content_version=manifest.content_version,
            rendering_keys=keys,
            snapshot_reader=deps.snapshot_reader,
            now=deps.now,
            settings=deps.settings,
        ):
            raise _RenderingNotServable(
                f"subject_id={completed_subject_id!r} content_version="
                f"{manifest.content_version!r} has no servable artifact under rendering "
                f"keys {keys!r}"
            )
        # The member-resolution inputs the run's family was resolved under, the ones its head
        # advance recorded, read here so the completion below has nothing left to raise.
        basis = family_basis(deps)
    except (MaterialisationCancelled, HeadFenceLost) as exc:
        logger.warning(
            "Datasets operation %r: attempt %d stopped writing; the row "
            "belongs to another attempt or is no longer running",
            operation.id,
            operation.attempt_count,
        )
        raise OperationSuperseded(str(exc)) from exc
    except Exception as exc:
        # The run classifies outside any request, so it binds the endpoint its refusals map under.
        try:
            with bound_registry_endpoint("datasets.mint"):
                spec = classify_definition_exception(exc)
        except Exception:
            # A failure the classifier cannot render, as one whose code the registry maps on no
            # status under `datasets.mint`, would otherwise leave the row running.
            logger.exception(
                "Datasets operation %r: classifying its failure %r raised; marking it failed "
                "(internal_error)",
                operation.id,
                exc,
            )
            operation_store.fail(
                operation.id,
                attempt=operation.attempt_count,
                refusal_code="internal_error",
                retryable=True,
            )
            return
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
            "Datasets operation %r failed transiently, releasing it for the next "
            "retry to claim immediately: %r",
            operation.id,
            exc,
        )
        operation_store.release(operation.id, attempt=operation.attempt_count)
        raise OperationTransientFailure(str(exc)) from exc

    completed = operation_store.complete(
        operation.id,
        attempt=operation.attempt_count,
        subject_id=completed_subject_id,
        content_version=manifest.content_version,
        confirmed_at=trace_confirmed_at(manifest.trace),
        family=trace_family(manifest.trace),
        basis=basis,
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


__all__ = [
    "LEASE_SECONDS",
    "RUN_DEADLINE_SECONDS",
    "OperationSuperseded",
    "OperationTransientFailure",
    "run_operation",
]
