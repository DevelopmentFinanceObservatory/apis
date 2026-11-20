"""Submits the freshness check of a stored head that `GET /api/datasets/{id}` served past its
recheck window (`current.serve_stored_head`).

The check is an ordinary operation. Its `request_identity` is the subject with the marker
`HEAD_CHECK_MARKER` and no `member_dcids`, so its id differs from an over-budget submission of
the same rendering, and the worker resolves the members afresh with nothing to compare them
with (`worker._check_members_unchanged`). It carries the request's options and a zero estimate,
which the worker replaces with its own. `run_operation` serves it through
`current.serve_current`, which judges the head, confirms it or rebuilds it, advances it by
compare-and-set, and renders the request's rendering.

`create_or_get` shares one row across readers and instances. A `done` row resets under a new
`submission_count`, so its task takes a new name and a head the last check confirmed is checked
again once it passes the window once more. A new row or a reset leaves `HEAD_CHECK_HEADROOM`
of `operation_backlog_max` free, so checks never fill the backlog that callers waiting on an
export need. Every read that finds the row `pending` with no recorded dispatch creates its task
again (`dispatch_if_due`), since nobody polls a check and the poll's enqueue recovery would
never run for it.

The check never fails the read. A full backlog, a store error, or a failed task create logs
and answers `stale`.
"""

from __future__ import annotations

import logging
from typing import Literal

from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import Operation, OperationBacklogFull, OperationStore
from datasets_api.operations.dispatch import dispatch_if_due
from datasets_api.operations.task_client import TaskClient
from datasets_api.origin import catalogue_build_id
from datasets_api.timings import phase
from series_coverage.index import Estimate

logger = logging.getLogger(__name__)

# What `x-dfo-freshness` states about a served head. `current`: confirmed inside the recheck
# window. `pending`: past it, and its check is pending or running. `stale`: past it, and no
# check is running.
type HeadFreshness = Literal["current", "pending", "stale"]

# The `request_identity` key that marks an operation as a head check.
HEAD_CHECK_MARKER = "head_check"

# The share of `operation_backlog_max` a new head check leaves free: half of it.
HEAD_CHECK_HEADROOM_SHARE = 2

_ZERO_ESTIMATE = Estimate(rows=0, requests=0)


def head_check_identity(subject_id: str) -> dict[str, object]:
    """The `request_identity` of `subject_id`'s head check."""
    return {"subject_id": subject_id, HEAD_CHECK_MARKER: True}


def submit_head_check(
    *,
    subject_id: str,
    options: MintOptions,
    coverage_stamp: str,
    operation_store: OperationStore,
    task_client: TaskClient,
    backlog_max: int,
) -> HeadFreshness:
    """Record or join the head check of `subject_id` for the rendering `options` names, create
    its task when no dispatch is recorded, and return the freshness the served head states:
    `pending` while the check is pending or running with its task created, `stale` otherwise."""
    try:
        with phase("head_check"):
            operation = operation_store.create_or_get(
                request_identity=head_check_identity(subject_id),
                options=options.model_dump(mode="json"),
                coverage_stamp=coverage_stamp,
                origin_catalogue_build_id=catalogue_build_id(),
                estimate=_ZERO_ESTIMATE,
                headroom=backlog_max // HEAD_CHECK_HEADROOM_SHARE,
            )
            operation = dispatch_if_due(
                operation, operation_store=operation_store, task_client=task_client
            )
    except OperationBacklogFull as exc:
        logger.warning(
            "Datasets head check for %r not submitted: %s; serving the head as stale",
            subject_id,
            exc,
        )
        return "stale"
    except Exception:
        # The read answers whatever the check's store or queue did.
        logger.warning(
            "Datasets head check for %r failed to submit; serving the head as stale",
            subject_id,
            exc_info=True,
        )
        return "stale"
    return _freshness(operation)


def _freshness(operation: Operation) -> HeadFreshness:
    """`pending` for a check that is running, done since the head was read, or pending with its
    task created; `stale` for a failed check or one whose task create failed (`dispatch_if_due`
    logged it)."""
    if operation.state == "failed":
        return "stale"
    if operation.state == "pending" and operation.dispatched_at is None:
        return "stale"
    return "pending"


__all__ = [
    "HEAD_CHECK_HEADROOM_SHARE",
    "HEAD_CHECK_MARKER",
    "HeadFreshness",
    "head_check_identity",
    "submit_head_check",
]
