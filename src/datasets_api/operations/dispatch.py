"""Creates Cloud Tasks entries after a successful `OperationStore.claim_dispatch`.

The claim commits before enqueue, so no database lock spans the Cloud Tasks call. Failed or
unconfirmed enqueues stay pending and can be retried after the grace period under the same task
name, up to `MAX_REDISPATCHES`. `AlreadyExists` confirms an existing task. Confirmed dispatches
leave redispatch. Dead-row expiry waits the full unclaimed window after the last dispatch.
"""

from __future__ import annotations

import logging

from datasets_api.operation_store import Operation, OperationStore
from datasets_api.operations.task_client import TaskClient
from datasets_api.store_errors import DatasetsStoreUnavailable

logger = logging.getLogger(__name__)


def dispatch_if_due(
    operation: Operation, *, operation_store: OperationStore, task_client: TaskClient
) -> Operation:
    """Claims and enqueues a due dispatch, returning the claimed row. Otherwise returns the
    original operation. Enqueue and confirmation failures are logged.
    """
    if operation.state != "pending":
        return operation
    claimed = operation_store.claim_dispatch(operation.id)
    if claimed is None:
        return operation
    try:
        task_client.enqueue(operation_id=claimed.id, submission_count=claimed.submission_count)
    except Exception:
        logger.warning(
            "Datasets operation %r: enqueue of dispatch %d had an unknown outcome; left "
            "pending and unconfirmed for a later dispatch claim",
            claimed.id,
            claimed.dispatch_count,
            exc_info=True,
        )
        return claimed
    try:
        operation_store.confirm_dispatch(
            claimed.id,
            submission_count=claimed.submission_count,
            dispatch_count=claimed.dispatch_count,
        )
    except DatasetsStoreUnavailable:
        # The task exists, but the row is unconfirmed.
        # Lease and terminal-state checks fence redelivery.
        logger.warning(
            "Datasets operation %r: dispatch %d was enqueued but could not be confirmed",
            claimed.id,
            claimed.dispatch_count,
            exc_info=True,
        )
    return claimed


__all__ = ["dispatch_if_due"]
