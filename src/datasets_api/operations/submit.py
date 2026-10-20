"""Submits over-budget mint and definition requests as async operations.

`request_identity` contains the resolved `subject_id`, so either door reuses the same operation
for identical identity, origin catalogue build, and rendering. The build is the one bound to the
submitting request. Options stay on the row. Their rendering key sets the id. Different
renderings get separate operations and isolate rendering-specific failures such as
`excel_limit`. The coverage stamp the estimate ran against is stored for lineage and stays out
of the id, so a coverage build never orphans a done operation.

`request_identity` also holds `member_dcids`, the StatVars the submitting door resolved. The
members are part of the operation id: two submissions share an operation only when their doors
resolved the same members, and a failure recorded for one member set is never returned to a
submission that resolved another. The worker resolves the members again and fails the operation
when they differ from `member_dcids`, so a row never records one member set and mints another.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import cast

from datasets_api.admission import refuse_async_over_ceiling
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import Operation, OperationStore
from datasets_api.operations.dispatch import dispatch_if_due
from datasets_api.operations.task_client import TaskClient
from datasets_api.origin import catalogue_build_id
from datasets_api.schema import OperationEnvelope, OperationResource
from datasets_api.timings import phase
from series_coverage.index import Estimate

# The shortest poll interval an operation advertises, and the refresh door's fixed retry interval.
OPERATION_SUBMIT_RETRY_AFTER_MS = 1_500

# The longest poll interval an operation advertises.
OPERATION_POLL_MAX_MS = 8_000


class OperationSubmitted(Exception):
    """Signals a successful 202 submission after recording the operation and creating its task
    when the submission records no dispatch.

    Mint and definition doors render the envelope outside RFC 9457 error handling.
    """

    def __init__(self, *, operation: Operation) -> None:
        self.operation = operation
        super().__init__(f"operation {operation.id!r} submitted for async execution")


def submit_over_budget(
    *,
    subject_id: str,
    options: MintOptions,
    estimate: Estimate,
    coverage_stamp: str,
    operation_store: OperationStore,
    task_client: TaskClient,
    within_budget: bool,
    async_max_rows: int,
    async_max_requests: int,
    members: Sequence[str],
) -> None:
    """Returns when the caller's sync-budget decision allows inline work.

    `members` are the StatVar refs a door resolved, `dcid:`-prefixed or bare. The operation
    stores them as `member_dcids`.

    Otherwise, async row or request limits raise `OverCeilingError` before a row or task exists.
    `OperationBacklogFull` also propagates. The function records or reuses the operation,
    creates its task when the submission records no dispatch, and raises `OperationSubmitted`
    for a 202 response.

    The create follows the committed store transaction. A create that fails or times out leaves
    the dispatch unrecorded, and the operations poll creates the task again under the same name
    once the enqueue recovery window has passed.
    """
    if within_budget:
        return
    refuse_async_over_ceiling(
        estimate, async_max_rows=async_max_rows, async_max_requests=async_max_requests
    )
    with phase("operation"):
        operation = operation_store.create_or_get(
            request_identity=_request_identity(subject_id, members),
            options=options.model_dump(mode="json"),
            coverage_stamp=coverage_stamp,
            origin_catalogue_build_id=catalogue_build_id(),
            estimate=estimate,
        )
    with phase("dispatch"):
        operation = dispatch_if_due(
            operation, operation_store=operation_store, task_client=task_client
        )
    raise OperationSubmitted(operation=operation)


_STAT_VAR_REF_PREFIX = "dcid:"


def _request_identity(subject_id: str, members: Sequence[str]) -> dict[str, object]:
    """The operation's identity: the subject and its resolved members as sorted bare dcids, so
    the id is independent of resolution order."""
    return {"subject_id": subject_id, "member_dcids": list(member_dcids(members))}


def member_dcids(members: Sequence[str]) -> tuple[str, ...]:
    """`members`, StatVar refs `dcid:`-prefixed or bare, as sorted distinct bare dcids, the form
    `member_dcids` stores them in."""
    return tuple(sorted({ref.removeprefix(_STAT_VAR_REF_PREFIX) for ref in members}))


def request_identity_subject_id(request_identity: Mapping[str, object]) -> str:
    """Returns the stored `subject_id`.

    Raises `KeyError` if the identity has another shape.
    """
    return str(request_identity["subject_id"])


def request_identity_member_dcids(request_identity: Mapping[str, object]) -> tuple[str, ...] | None:
    """Returns the stored `member_dcids`, or `None` for an operation recorded without them."""
    member_dcids = request_identity.get("member_dcids")
    if member_dcids is None:
        return None
    return tuple(str(dcid) for dcid in cast("Sequence[object]", member_dcids))


def poll_interval_ms(operation: Operation, *, now: datetime) -> int:
    """The milliseconds a caller waits before polling `operation` again: a quarter of the time
    since its current submission (`submitted_at`), held between
    `OPERATION_SUBMIT_RETRY_AFTER_MS` and `OPERATION_POLL_MAX_MS`. A repeat submission that
    joins a long build gets the slow value, a resubmission that restarts a failed build starts
    at the floor, and a waiting browser backs off as the build runs. A done or failed
    operation answers its next poll at once, so it gets the shortest interval.
    """
    if operation.state in ("done", "failed"):
        return OPERATION_SUBMIT_RETRY_AFTER_MS
    age_ms = (now - operation.submitted_at) // timedelta(milliseconds=1)
    return min(OPERATION_POLL_MAX_MS, max(OPERATION_SUBMIT_RETRY_AFTER_MS, age_ms // 4))


def retry_after_headers(envelope: OperationEnvelope) -> dict[str, str]:
    """The `Retry-After` header for `envelope`: its `retry_after_ms` in whole seconds, rounded
    up. A done envelope carries no `retry_after_ms` and gets no header."""
    if envelope.retry_after_ms is None:
        return {}
    return {"Retry-After": str(math.ceil(envelope.retry_after_ms / 1000))}


def operation_submitted_resource(
    submitted: OperationSubmitted, *, now: datetime
) -> OperationResource:
    """Builds the 202 resource from the returned operation state, which may already be running or
    terminal. Progress stays unset in the submit response and appears on polls.
    """
    operation = submitted.operation
    return OperationResource(
        operation=OperationEnvelope(
            operation_id=operation.id,
            status=operation.state,
            poll=f"/api/datasets/operations/{operation.id}",
            retry_after_ms=poll_interval_ms(operation, now=now),
        )
    )


__all__ = [
    "OPERATION_POLL_MAX_MS",
    "OPERATION_SUBMIT_RETRY_AFTER_MS",
    "OperationSubmitted",
    "member_dcids",
    "operation_submitted_resource",
    "poll_interval_ms",
    "request_identity_member_dcids",
    "request_identity_subject_id",
    "retry_after_headers",
    "submit_over_budget",
]
