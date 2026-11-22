"""Queues the render ahead of a version a request thread stored, for the worker to run.

A rebuild that runs inline on a request thread stores the new version and advances the head
there, and the request answers with it. Rendering ahead from that version on the same thread
would hold a request thread and a materialise permit for other roots' renders, so the serve
queues it instead (`current._check_currency`): after the head's transaction commits, a rebuild
whose trigger is not `worker` and that won the head, of a dataset whose links an analysis root read within
`DATASETS_RENDER_AHEAD_READ_WINDOW_SECONDS`, records a render-ahead operation and creates its
task. A worker run renders ahead from the version it materialised itself
(`operations.worker.run_operation`).

The operation is an ordinary row, one per `(subject_id, head content_version)`: its
`request_identity` is `{subject_id, render_ahead: <head content_version>}`, with no members,
`HEAD_CHECK_OPTIONS` as its rendering and a zero estimate, as a head check's is
(`operations.head_check`), so no request can name it and nobody polls it. The worker runs it
through `render_ahead.render_ahead` and completes it naming that head.

A row already on record for the pair queues nothing more: concurrent inline rebuilds of one head
share one row, and a second store of the same head queues nothing new. Two rows are retried by a
later store of the same head: a retryable failed row, which `create_or_get` resets under a new
submission, and a `pending` row whose task create was lost, whose retry the store claims once the
row has waited `ENQUEUE_RECOVERY_SECONDS` (`claim_dispatch_retry`). Nothing else retries a lost
create, since nobody polls the row. A new row leaves half of `DATASETS_OPERATION_BACKLOG_MAX`
free, as a head check does, and the task create waits at most
`HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS`. A full backlog, a store error or a failed create is logged
and fails nothing, and that head is rendered ahead by nothing until a later store of it queues
again or `datasets-maintenance render-ahead` runs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import timedelta

from datasets_api.operation_store import (
    NullOperationStore,
    OperationBacklogFull,
    OperationStore,
    compute_operation_id,
)
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS, dispatch_if_due
from datasets_api.operations.head_check import (
    HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
    HEAD_CHECK_HEADROOM_SHARE,
    HEAD_CHECK_OPTIONS,
)
from datasets_api.operations.task_client import TaskClient
from datasets_api.origin import catalogue_build_id
from series_coverage.index import Estimate

logger = logging.getLogger(__name__)

# The `request_identity` key that marks an operation as a render ahead, holding the head
# content version it renders from.
RENDER_AHEAD_MARKER = "render_ahead"

_ZERO_ESTIMATE = Estimate(rows=0, requests=0)
_RECOVERY = timedelta(seconds=ENQUEUE_RECOVERY_SECONDS)


def render_ahead_identity(subject_id: str, *, head_content_version: str) -> dict[str, object]:
    """The `request_identity` of the render ahead of `subject_id` from `head_content_version`."""
    return {"subject_id": subject_id, RENDER_AHEAD_MARKER: head_content_version}


def render_ahead_head(request_identity: object) -> str | None:
    """The head content version a render-ahead operation's `request_identity` names, or `None`
    for any other operation."""
    if not isinstance(request_identity, Mapping):
        return None
    head = request_identity.get(RENDER_AHEAD_MARKER)
    return head if isinstance(head, str) else None


def queue_render_ahead(
    *,
    subject_id: str,
    head_content_version: str,
    coverage_stamp: str,
    operation_store: OperationStore,
    task_client: TaskClient,
    backlog_max: int,
) -> None:
    """Record the render ahead of `subject_id` from `head_content_version` unless a row for the
    pair is on record, and create its task, or retry a lost create. See the module docstring.
    Every failure is logged and swallowed."""
    if isinstance(operation_store, NullOperationStore):
        return
    identity = render_ahead_identity(subject_id, head_content_version=head_content_version)
    options = HEAD_CHECK_OPTIONS.model_dump(mode="json")
    build_id = catalogue_build_id()
    try:
        before = operation_store.get(
            compute_operation_id(
                request_identity=identity, origin_catalogue_build_id=build_id, options=options
            )
        )
        if before is None or (before.state == "failed" and before.retryable):
            # A retryable failed row resets under a new submission.
            operation = operation_store.create_or_get(
                request_identity=identity,
                options=options,
                coverage_stamp=coverage_stamp,
                origin_catalogue_build_id=build_id,
                estimate=_ZERO_ESTIMATE,
                headroom=backlog_max // HEAD_CHECK_HEADROOM_SHARE,
            )
        elif before.state == "pending" and before.dispatched_at is None:
            claimed = operation_store.claim_dispatch_retry(
                before.id, submission_count=before.submission_count, retry_after=_RECOVERY
            )
            if claimed is None:
                return
            operation = claimed
        else:
            return
        dispatch_if_due(
            operation,
            operation_store=operation_store,
            task_client=task_client,
            timeout=HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
        )
    except OperationBacklogFull as exc:
        logger.warning(
            "Datasets render ahead of %r from %r not queued: %s",
            subject_id,
            head_content_version,
            exc,
        )
    except Exception:
        logger.warning(
            "Datasets render ahead of %r from %r failed to queue",
            subject_id,
            head_content_version,
            exc_info=True,
        )


__all__ = [
    "RENDER_AHEAD_MARKER",
    "queue_render_ahead",
    "render_ahead_head",
    "render_ahead_identity",
]
