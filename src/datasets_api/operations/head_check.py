"""Submits the freshness check of a stored head that `GET /api/datasets/{id}` served past its
recheck window (`current.serve_stored_head`).

The check is an ordinary operation, one per spec. Its `request_identity` is the subject with the
marker `HEAD_CHECK_MARKER` and no `member_dcids`, so its id differs from an over-budget
submission, and the worker resolves the members afresh with nothing to compare them with
(`worker._check_members_unchanged`). Every check carries `HEAD_CHECK_OPTIONS`, whatever
rendering the read asked for, so the readers of every rendering share one row. Those options
name a bare parquet file in the frame layout, which the worker writes from the frame as
reshaped, with no name read. The row carries a zero estimate, which the worker replaces with
its own. `run_operation` serves it through `current.serve_current`, which judges the head,
confirms it or rebuilds it, and advances it by compare-and-set. The head is the spec's, so a
confirmation makes it current for every rendering it holds. A rebuild advances it to a version
that holds the check's rendering, and a reader of a rendering that version lacks takes the
in-request path, which renders it from the stored frame.

`create_or_get` shares the row across readers and instances. A `done` row resets under a new
`submission_count`, so its task takes a new name and a head the last check confirmed is checked
again once it passes the window once more. A new row or a reset leaves
`1 / HEAD_CHECK_HEADROOM_SHARE` of `operation_backlog_max` free, so checks never fill the
backlog that callers waiting on an export need.

The read that inserted or reset the row records it in the request (the `head_check` phase) and
hands its task create to the surface's `TaskCreatePool` (`operations.task_create_pool`), which
creates it on a pool thread waiting at most `HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS`, so the reader
waits on no Cloud Tasks call and states `pending`. A pool already full refuses the create, and
the read states `stale`. A create that fails, times out, or never runs leaves the row
undispatched. Nobody polls a check, so the poll's enqueue recovery never runs for it. A later
read retries the create once the row has waited `ENQUEUE_RECOVERY_SECONDS`, after claiming
that retry in the store (`claim_dispatch_retry`), so one read per recovery interval retries it
across instances, and during an outage each spec costs one short create per interval.

A check that failed for good (a deterministic refusal, `over_ceiling` for one) stays `failed`
until retention deletes it, so the read sends the call to the work executor, which judges the
head in the request and answers the refusal.
A full backlog, a store error, or a full pool logs and answers `stale`, and a failed task create
logs on the pool thread; none of them fails the read.

An analysis root under `latest: "stored"` judges the head by the same rules
(`current.stored_head_version`), and a dataset with no servable head queues this same operation
as its rebuild (`head_check_operation`), whose id the 202 names. That rebuild creates its task
in the request, so the 202 names the row as the create left it.
"""

from __future__ import annotations

import functools
import logging
from datetime import timedelta
from typing import Literal

from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    Operation,
    OperationBacklogFull,
    OperationStore,
    compute_operation_id,
)
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS, dispatch_if_due
from datasets_api.operations.task_client import TaskClient
from datasets_api.operations.task_create_pool import TaskCreates
from datasets_api.origin import catalogue_build_id
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import phase
from series_coverage.index import Estimate

logger = logging.getLogger(__name__)

# What `x-dfo-freshness` states about a served head. `current`: confirmed inside the recheck
# window. `pending`: past it, and its check is pending or running. `stale`: past it, and no
# check is running.
type HeadFreshness = Literal["current", "pending", "stale"]

# What a read learns of its head's check: the freshness it states, or `failed` for a check that
# failed for good, whose read is judged in the request.
type HeadCheckState = Literal["pending", "stale", "failed"]

# The `request_identity` key that marks an operation as a head check.
HEAD_CHECK_MARKER = "head_check"

# The rendering every head check carries, a bare parquet file in the frame layout written from
# the frame as reshaped with no name read. It is the cheapest file a rebuild can leave on the head.
HEAD_CHECK_OPTIONS = MintOptions(format="parquet", bundle=False, layout="frame")

# A new head check leaves `1 / HEAD_CHECK_HEADROOM_SHARE` of `operation_backlog_max` free, which
# is half of it.
HEAD_CHECK_HEADROOM_SHARE = 2

# The longest a check's task create waits, on a pool thread for a read past the window and in
# the request for a queued rebuild. A Cloud Tasks create answers in well under a second when
# the service is healthy, and a create that times out is retried after
# `ENQUEUE_RECOVERY_SECONDS` under the same task name, so a longer wait buys nothing.
HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS = 1.0

_ZERO_ESTIMATE = Estimate(rows=0, requests=0)
_RECOVERY = timedelta(seconds=ENQUEUE_RECOVERY_SECONDS)


def head_check_identity(subject_id: str) -> dict[str, object]:
    """The `request_identity` of `subject_id`'s head check."""
    return {"subject_id": subject_id, HEAD_CHECK_MARKER: True}


def submit_head_check(
    *,
    subject_id: str,
    coverage_stamp: str,
    operation_store: OperationStore,
    task_client: TaskClient,
    backlog_max: int,
    task_creates: TaskCreates,
) -> HeadCheckState:
    """Record or join the head check of `subject_id`, hand its task create to `task_creates`
    when this read wrote the row or claimed the retry of a lost create, and return what the
    served head states: `pending` while the check is pending or running with its task created
    or handed off, `stale` when it could not be submitted or its task is not created, and
    `failed` when it failed."""
    try:
        state, _operation = request_head_check(
            subject_id=subject_id,
            coverage_stamp=coverage_stamp,
            operation_store=operation_store,
            task_client=task_client,
            backlog_max=backlog_max,
            task_creates=task_creates,
        )
    except OperationBacklogFull as exc:
        logger.warning(
            "Datasets head check for %r not submitted: %s; serving the head as stale",
            subject_id,
            exc,
        )
        return "stale"
    except DatasetsStoreUnavailable:
        logger.warning(
            "Datasets head check for %r failed to submit; serving the head as stale",
            subject_id,
            exc_info=True,
        )
        return "stale"
    return state


def request_head_check(
    *,
    subject_id: str,
    coverage_stamp: str,
    operation_store: OperationStore,
    task_client: TaskClient,
    backlog_max: int,
    task_creates: TaskCreates,
) -> tuple[HeadCheckState, Operation]:
    """The state a read past the recheck window states for `subject_id`'s head and the check's
    row as the read left it. The row is recorded or joined in the request (the `head_check`
    phase), and its task create, when this read is due to make it, is handed to
    `task_creates`, so the read waits on no Cloud Tasks call. A handed-off create states
    `pending`, and one `task_creates` refused leaves the row undispatched and states `stale`.

    Raises:
        OperationBacklogFull: The backlog leaves no room for a new row.
        DatasetsStoreUnavailable: The store failed.
    """
    with phase("head_check"):
        operation, due = _record(
            subject_id=subject_id,
            coverage_stamp=coverage_stamp,
            operation_store=operation_store,
            backlog_max=backlog_max,
        )
    if not due:
        return head_check_state(operation), operation
    create = functools.partial(
        dispatch_if_due,
        operation,
        operation_store=operation_store,
        task_client=task_client,
        timeout=HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
    )
    if not task_creates.offer(create):
        logger.warning(
            "Datasets head check for %r recorded but its task create was not queued; a later "
            "read retries it; serving the head as stale",
            subject_id,
        )
        return "stale", operation
    return "pending", operation


def head_check_operation(
    *,
    subject_id: str,
    coverage_stamp: str,
    operation_store: OperationStore,
    task_client: TaskClient,
    backlog_max: int,
) -> Operation:
    """The head check row of `subject_id`, recorded or joined as `request_head_check` records
    it, with its task created in the request when this read wrote the row or claimed the retry
    of a lost create. An analysis root under `latest: "stored"` whose dataset has no servable
    head queues its rebuild here, since the worker's `serve_current` rebuilds a head it cannot
    confirm, and the 202 names the row as it stands after the create.

    Raises:
        OperationBacklogFull: The backlog leaves no room for a new row.
        DatasetsStoreUnavailable: The store failed.
    """
    with phase("head_check"):
        operation, due = _record(
            subject_id=subject_id,
            coverage_stamp=coverage_stamp,
            operation_store=operation_store,
            backlog_max=backlog_max,
        )
        if not due:
            return operation
        return dispatch_if_due(
            operation,
            operation_store=operation_store,
            task_client=task_client,
            timeout=HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
        )


def _record(
    *,
    subject_id: str,
    coverage_stamp: str,
    operation_store: OperationStore,
    backlog_max: int,
) -> tuple[Operation, bool]:
    """The head check row of `subject_id`, recorded or joined, and whether this read is due to
    create its task: the row is `pending` with no recorded dispatch, and this read inserted or
    reset it or claimed the retry of its lost create."""
    identity = head_check_identity(subject_id)
    options = HEAD_CHECK_OPTIONS.model_dump(mode="json")
    build_id = catalogue_build_id()
    before = operation_store.get(
        compute_operation_id(
            request_identity=identity, origin_catalogue_build_id=build_id, options=options
        )
    )
    operation = operation_store.create_or_get(
        request_identity=identity,
        options=options,
        coverage_stamp=coverage_stamp,
        origin_catalogue_build_id=build_id,
        estimate=_ZERO_ESTIMATE,
        headroom=backlog_max // HEAD_CHECK_HEADROOM_SHARE,
    )
    if operation.state != "pending" or operation.dispatched_at is not None:
        return operation, False
    # A new row, or a reset under a new submission, is this read's to create a task for.
    if before is None or before.submission_count != operation.submission_count:
        return operation, True
    claimed = operation_store.claim_dispatch_retry(
        operation.id, submission_count=operation.submission_count, retry_after=_RECOVERY
    )
    if claimed is None:
        return operation, False
    return claimed, True


def head_check_state(operation: Operation) -> HeadCheckState:
    """`failed` for a failed check; `stale` for one pending whose task create has not returned;
    `pending` for one running, pending with its task created, or done since the head was
    read."""
    if operation.state == "failed":
        return "failed"
    if operation.state == "pending" and operation.dispatched_at is None:
        return "stale"
    return "pending"


__all__ = [
    "HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS",
    "HEAD_CHECK_HEADROOM_SHARE",
    "HEAD_CHECK_MARKER",
    "HEAD_CHECK_OPTIONS",
    "HeadCheckState",
    "HeadFreshness",
    "head_check_identity",
    "head_check_operation",
    "head_check_state",
    "request_head_check",
    "submit_head_check",
]
