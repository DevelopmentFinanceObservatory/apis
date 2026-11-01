"""Pins async operation fencing on the attempt token.

A superseded attempt's late writes leave the newer submission's row unchanged. A delivery that
arrives while another attempt is inside `claimed_at + lease_seconds` answers 409, and one
after it claims. A crash between the create commit and task creation leaves `dispatched_at` null,
and the next poll past the enqueue recovery window creates the task under the same name. A
transient failure on the final Cloud Tasks attempt records a retryable failure.

The delivery tests drive the worker route and the poll door over one store. A crash on a
non-final delivery leaves the row to the next delivery across a poll past the lease, and a crash
on the final one fails at that poll. A delivery past the operation deadline fails the row without
running it, so a row the backlog stopped counting never runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.meta import OperationFailed, _resolve_operation
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import LiveOperationStore, OperationBacklogFull
from datasets_api.operations import worker as worker_module
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS
from datasets_api.operations.lease import LEASE_SECONDS
from datasets_api.operations.submit import OperationSubmitted, submit_over_budget
from datasets_api.operations.worker import RUN_DEADLINE_SECONDS, run_operation
from materialise.trace import UNKNOWN_FAMILY
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import TEST_BASIS
from .test_worker import _deps
from .test_worker_app import _client, _run_path, _submit, _task_body

_CONFIRMED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_T0 = datetime(2026, 6, 1, tzinfo=UTC)
_SETTINGS = DatasetsBuildSettings()


class _Clock:
    def __init__(self) -> None:
        self.t = _T0

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


def _store(
    clock: _Clock, *, backlog_max: int | None = None, deadline_seconds: int | None = None
) -> LiveOperationStore:
    return LiveOperationStore(
        store=FakeOperationDatasetStore(),
        now=clock,
        backlog_max=backlog_max,
        deadline_seconds=deadline_seconds,
    )


def test_a_late_fail_from_a_superseded_attempt_leaves_the_resubmission_unchanged() -> None:
    clock = _Clock()
    store = _store(clock)
    operation_id = _submit(store)
    attempt_a = store.claim(
        operation_id, submission_count=0, lease_seconds=1_500, final_delivery=False
    )
    assert attempt_a is not None and attempt_a.attempt_count == 1

    # A outlives its claim; the resubmission resets the row and B claims it.
    clock.advance(1_501)
    assert _submit(store) == operation_id
    attempt_b = store.claim(
        operation_id, submission_count=1, lease_seconds=1_500, final_delivery=False
    )
    assert attempt_b is not None and attempt_b.attempt_count == 2

    assert (
        store.fail(operation_id, attempt=1, refusal_code="internal_error", retryable=True) is None
    )
    assert store.progress(operation_id, attempt=1, rows_written=99) is None
    assert (
        store.complete(
            operation_id,
            attempt=1,
            subject_id="spec_x",
            content_version="cv_x",
            confirmed_at=_CONFIRMED_AT,
            family=UNKNOWN_FAMILY,
            basis=TEST_BASIS,
        )
        is None
    )

    assert store.get(operation_id) == attempt_b


def test_a_second_delivery_inside_the_claim_answers_409_and_one_after_it_claims() -> None:
    clock = _Clock()
    store = _store(clock)
    operation_id = _submit(store)
    attempt_a = store.claim(
        operation_id, submission_count=0, lease_seconds=RUN_DEADLINE_SECONDS, final_delivery=False
    )
    assert attempt_a is not None
    ran: list[int] = []

    def run(*, operation, operation_store, **_) -> None:
        ran.append(operation.attempt_count)
        operation_store.complete(
            operation.id,
            attempt=operation.attempt_count,
            subject_id="spec_x",
            content_version="cv",
            confirmed_at=_CONFIRMED_AT,
            family=UNKNOWN_FAMILY,
            basis=TEST_BASIS,
        )

    client = _client(operation_store=store, run_in_thread=run)
    headers = {"Authorization": "Bearer t"}

    clock.advance(RUN_DEADLINE_SECONDS - 1)
    inside = client.post(_run_path(operation_id), json=_task_body(), headers=headers)
    assert inside.status_code == 409
    assert ran == []
    assert store.get(operation_id) == attempt_a

    clock.advance(2)
    after = client.post(_run_path(operation_id), json=_task_body(), headers=headers)
    assert after.status_code == 200
    assert ran == [2]


class _CrashingTaskClient:
    """Fails the first create the way a process dying after the commit does, then records."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        self.calls.append((operation_id, submission_count))
        if len(self.calls) == 1:
            raise TimeoutError("process died before create_task returned")


def _poll(store: LiveOperationStore, operation_id: str, task_client, clock: _Clock):
    return _resolve_operation(
        operation_id=operation_id,
        operation_store=store,
        task_client=task_client,
        spec_reader=None,
        snapshot_reader=None,
        now=clock,
        settings=_SETTINGS,
        enrichment_digest="digest",
        serving_basis=lambda: TEST_BASIS,
    )


def test_a_crash_between_commit_and_enqueue_is_recovered_by_the_next_poll() -> None:
    clock = _Clock()
    store = _store(clock)
    task_client = _CrashingTaskClient()

    with pytest.raises(OperationSubmitted) as excinfo:
        submit_over_budget(
            subject_id="spec_abc",
            options=MintOptions(),
            estimate=Estimate(rows=10, requests=1),
            coverage_stamp="cov_1",
            operation_store=store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
            members=["dcid:sv_a"],
        )
    operation_id = excinfo.value.operation.id
    crashed = store.get(operation_id)
    assert crashed is not None
    assert crashed.state == "pending"
    assert crashed.dispatched_at is None

    clock.advance(ENQUEUE_RECOVERY_SECONDS - 1)
    assert _poll(store, operation_id, task_client, clock).operation.status == "pending"
    assert task_client.calls == [(operation_id, 0)]

    clock.advance(2)
    assert _poll(store, operation_id, task_client, clock).operation.status == "pending"
    assert task_client.calls == [(operation_id, 0), (operation_id, 0)]
    recovered = store.get(operation_id)
    assert recovered is not None and recovered.dispatched_at == clock.t

    clock.advance(3_600)
    _poll(store, operation_id, task_client, clock)
    assert len(task_client.calls) == 2


def test_a_transient_failure_on_the_final_attempt_records_retryable() -> None:
    clock = _Clock()
    store = _store(clock)
    operation_id = _submit(store)

    def _raise_transient(*, spec_id, deps):
        raise ConnectionError("upstream SDMX call failed")

    def run(**kwargs) -> None:
        run_operation(**{**kwargs, "deps": _deps()})

    client = _client(operation_store=store, run_in_thread=run, max_attempts=5)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_module, "_resolve_spec_requests", _raise_transient)
        resp = client.post(
            _run_path(operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t", "X-CloudTasks-TaskRetryCount": "4"},
        )

    assert resp.status_code == 200
    stored = store.get(operation_id)
    assert stored is not None
    assert (stored.state, stored.refusal_code, stored.retryable) == (
        "failed",
        "internal_error",
        True,
    )


_HEADERS = {"Authorization": "Bearer t"}
_MAX_ATTEMPTS = 3
_DEADLINE_SECONDS = 600


def _deliver(client, operation_id: str, *, retry_count: int = 0):
    return client.post(
        _run_path(operation_id),
        json=_task_body(),
        headers={**_HEADERS, "X-CloudTasks-TaskRetryCount": str(retry_count)},
    )


def _crash_once_then_complete(ran: list[int]):
    """A runner that crashes on its first call and completes on every later one, recording
    each attempt token it ran under."""

    def run(*, operation, operation_store, **_) -> None:
        ran.append(operation.attempt_count)
        if len(ran) == 1:
            raise RuntimeError("worker died mid-run")
        operation_store.complete(
            operation.id,
            attempt=operation.attempt_count,
            subject_id="spec_x",
            content_version="cv",
            confirmed_at=_CONFIRMED_AT,
            family=UNKNOWN_FAMILY,
            basis=TEST_BASIS,
        )

    return run


def test_a_poll_past_the_lease_of_a_crashed_delivery_leaves_the_row_to_the_next_one() -> None:
    clock = _Clock()
    store = _store(clock)
    operation_id = _submit(store)
    ran: list[int] = []
    client = _client(
        operation_store=store, run_in_thread=_crash_once_then_complete(ran), max_attempts=3
    )

    assert _deliver(client, operation_id, retry_count=0).status_code == 500
    claimed = store.get(operation_id)

    clock.advance(LEASE_SECONDS + 1)
    assert _poll(store, operation_id, None, clock).operation.status == "running"
    assert store.get(operation_id) == claimed

    retried = _deliver(client, operation_id, retry_count=1)
    assert (retried.status_code, retried.json()) == (200, {"status": "done"})
    assert ran == [1, 2]


def test_a_poll_past_the_lease_of_a_crashed_final_delivery_fails_retryable() -> None:
    clock = _Clock()
    store = _store(clock)
    operation_id = _submit(store)
    client = _client(
        operation_store=store,
        run_in_thread=_crash_once_then_complete([]),
        max_attempts=_MAX_ATTEMPTS,
    )

    assert _deliver(client, operation_id, retry_count=_MAX_ATTEMPTS - 1).status_code == 500

    clock.advance(LEASE_SECONDS)
    assert _poll(store, operation_id, None, clock).operation.status == "running"
    clock.advance(1)
    with pytest.raises(OperationFailed) as excinfo:
        _poll(store, operation_id, None, clock)
    assert (excinfo.value.refusal_code, excinfo.value.retryable) == ("internal_error", True)


@pytest.mark.parametrize(
    "claimed_by_an_earlier_delivery", [False, True], ids=["pending", "claimed"]
)
def test_a_delivery_past_the_deadline_fails_an_unpolled_row_without_running_it(
    claimed_by_an_earlier_delivery: bool,
) -> None:
    clock = _Clock()
    store = _store(clock, deadline_seconds=_DEADLINE_SECONDS)
    operation_id = _submit(store)
    if claimed_by_an_earlier_delivery:
        assert store.claim(operation_id, submission_count=0, lease_seconds=60, final_delivery=False)
    ran: list[int] = []
    client = _client(operation_store=store, run_in_thread=_crash_once_then_complete(ran))

    clock.advance(_DEADLINE_SECONDS + 1)
    late = _deliver(client, operation_id, retry_count=1)

    assert (late.status_code, late.json()) == (200, {"status": "failed"})
    assert ran == []
    stored = store.get(operation_id)
    assert stored is not None
    assert (stored.state, stored.refusal_code, stored.retryable) == (
        "failed",
        "internal_error",
        True,
    )


def test_a_delivery_at_the_deadline_still_claims_and_runs() -> None:
    clock = _Clock()
    store = _store(clock, deadline_seconds=_DEADLINE_SECONDS)
    operation_id = _submit(store)
    ran: list[int] = []

    def run(*, operation, operation_store, **_) -> None:
        ran.append(operation.attempt_count)
        operation_store.complete(
            operation.id,
            attempt=operation.attempt_count,
            subject_id="spec_x",
            content_version="cv",
            confirmed_at=_CONFIRMED_AT,
            family=UNKNOWN_FAMILY,
            basis=TEST_BASIS,
        )

    client = _client(operation_store=store, run_in_thread=run)

    clock.advance(_DEADLINE_SECONDS)
    on_time = _deliver(client, operation_id)

    assert (on_time.status_code, on_time.json()) == (200, {"status": "done"})
    assert ran == [1]


def test_a_late_delivery_cannot_take_the_backlog_over_its_limit() -> None:
    """Row A leaves the backlog at the deadline and row B takes its place. A's delivery then
    arrives: it runs nothing, so B stays the one live row and a third submission is refused."""
    clock = _Clock()
    store = _store(clock, backlog_max=1, deadline_seconds=_DEADLINE_SECONDS)
    a = _submit(store, subject_id="spec_a")
    ran: list[str] = []

    def run(*, operation, **_) -> None:
        ran.append(operation.id)

    client = _client(operation_store=store, run_in_thread=run)

    clock.advance(_DEADLINE_SECONDS + 1)
    b = _submit(store, subject_id="spec_b")
    late = _deliver(client, a)

    assert (late.status_code, late.json()) == (200, {"status": "failed"})
    assert ran == []
    states = {id: operation.state for id in (a, b) if (operation := store.get(id)) is not None}
    assert states == {a: "failed", b: "pending"}
    with pytest.raises(OperationBacklogFull):
        _submit(store, subject_id="spec_c")
