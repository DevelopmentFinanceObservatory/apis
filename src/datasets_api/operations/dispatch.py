"""Creates the Cloud Tasks task for a pending operation whose submission records no dispatch.

`OperationStore.create_or_get` commits the row before the create, so no database lock spans the
Cloud Tasks call. A create that returns, or reports the task already exists, is recorded with
`mark_dispatched`. A create that fails or never returns leaves `dispatched_at` NULL. The
operations poll then calls `dispatch_if_due` with `older_than`, so a row past the enqueue
recovery window gets its task created again under the same name. Cloud Tasks deduplicates by
name, so a repeated create never runs the operation twice.
"""

from __future__ import annotations

import logging
from datetime import datetime

from datasets_api.operation_store import Operation, OperationStore
from datasets_api.operations.task_client import ENQUEUE_TIMEOUT_SECONDS, TaskClient
from datasets_api.store_errors import DatasetsStoreUnavailable

logger = logging.getLogger(__name__)

# A poll recovers an enqueue only after a create started at submission has had time to return.
ENQUEUE_RECOVERY_SECONDS = 3 * ENQUEUE_TIMEOUT_SECONDS


def dispatch_if_due(
    operation: Operation,
    *,
    operation_store: OperationStore,
    task_client: TaskClient,
    older_than: datetime | None = None,
) -> Operation:
    """Creates the task for a `pending` row with no recorded dispatch and records it, returning
    the updated row. With `older_than`, only a submission made before it is due. Any other row
    is returned unchanged. Create and record failures are logged and leave the row for the next
    poll.
    """
    if operation.state != "pending" or operation.dispatched_at is not None:
        return operation
    if older_than is not None and operation.submitted_at >= older_than:
        return operation
    try:
        task_client.enqueue(operation_id=operation.id, submission_count=operation.submission_count)
    except Exception:
        logger.warning(
            "Datasets operation %r: task create for submission %d had an unknown outcome; "
            "left undispatched for the next poll",
            operation.id,
            operation.submission_count,
            exc_info=True,
        )
        return operation
    try:
        marked = operation_store.mark_dispatched(
            operation.id, submission_count=operation.submission_count
        )
    except DatasetsStoreUnavailable:
        # The task exists. A later poll creates it again, and Cloud Tasks deduplicates the name.
        logger.warning(
            "Datasets operation %r: task for submission %d was created but not recorded",
            operation.id,
            operation.submission_count,
            exc_info=True,
        )
        return operation
    return marked if marked is not None else operation


__all__ = ["ENQUEUE_RECOVERY_SECONDS", "dispatch_if_due"]
