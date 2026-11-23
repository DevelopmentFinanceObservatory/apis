"""Tests `submit_head_check`, the freshness check a GET past the recheck window submits as an
ordinary operation, and the `x-dfo-freshness` it answers, and the `TaskCreatePool` that creates
its task off the request path. The operation store is the real `LiveOperationStore` over SQLite,
on a clock the test moves.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import override

import pytest

from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    LiveOperationStore,
    NullOperationStore,
    Operation,
    OperationStore,
)
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS
from datasets_api.operations.head_check import (
    HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
    HEAD_CHECK_OPTIONS,
    HeadCheckState,
    submit_head_check,
)
from datasets_api.operations.task_client import (
    ENQUEUE_TIMEOUT_SECONDS,
    FakeTaskClient,
    TaskClient,
)
from datasets_api.operations.task_create_pool import (
    InlineTaskCreates,
    TaskCreatePool,
    TaskCreates,
)
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from materialise.trace import UNKNOWN_FAMILY
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import TEST_BASIS

_SPEC = "spec_head_check"
_BACKLOG_MAX = 20
_START = datetime(2026, 11, 20, 9, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.t = _START

    def __call__(self) -> datetime:
        return self.t


class _RaisingTaskClient:
    """Fails every create once its timeout runs out, recording each timeout it was given."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def enqueue(
        self,
        *,
        operation_id: str,
        submission_count: int,
        timeout: float = ENQUEUE_TIMEOUT_SECONDS,
    ) -> None:
        self.timeouts.append(timeout)
        raise TimeoutError("create_task deadline exceeded")


class _UnavailableOperationStore(NullOperationStore):
    @override
    def create_or_get(
        self,
        *,
        request_identity: Mapping[str, object],
        options: Mapping[str, object],
        coverage_stamp: str,
        origin_catalogue_build_id: str,
        estimate: Estimate,
        headroom: int = 0,
    ) -> Operation:
        raise DatasetsStoreUnavailable("creating operation failed: connection refused")


class _BrokenOperationStore(NullOperationStore):
    @override
    def get(self, id: str) -> Operation | None:
        raise TypeError("get() got an unexpected keyword argument")


class _LostMarkOperationStore(LiveOperationStore):
    """Records the dispatch of every create as another instance would, just before this
    instance's own record, which then finds it marked."""

    @override
    def mark_dispatched(self, id: str, *, submission_count: int) -> Operation | None:
        super().mark_dispatched(id, submission_count=submission_count)
        return super().mark_dispatched(id, submission_count=submission_count)


class _HeldCreates:
    """Accepts every create and runs none until `run` is called, as a pool whose threads have
    not reached them yet."""

    def __init__(self) -> None:
        self.held: list[Callable[[], object]] = []

    def offer(self, create: Callable[[], object]) -> bool:
        self.held.append(create)
        return True

    def run(self) -> None:
        held, self.held = self.held, []
        for create in held:
            create()


def _check(
    operation_store: OperationStore,
    task_client: TaskClient,
    *,
    backlog_max: int = _BACKLOG_MAX,
    task_creates: TaskCreates | None = None,
) -> HeadCheckState:
    return submit_head_check(
        subject_id=_SPEC,
        coverage_stamp="cov_stamp_1",
        operation_store=operation_store,
        task_client=task_client,
        backlog_max=backlog_max,
        task_creates=task_creates if task_creates is not None else InlineTaskCreates(),
    )


def _store(*, backlog_max: int = _BACKLOG_MAX, clock: _Clock | None = None) -> LiveOperationStore:
    return LiveOperationStore(
        store=FakeOperationDatasetStore(),
        backlog_max=backlog_max,
        now=clock if clock is not None else _Clock(),
    )


def _other_operation(operation_store: LiveOperationStore, n: int) -> None:
    operation_store.create_or_get(
        request_identity={"subject_id": f"spec_other_{n}", "member_dcids": ["sv_a"]},
        options=MintOptions().model_dump(mode="json"),
        coverage_stamp="cov_stamp_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=10, requests=1),
    )


def test_stale_reads_share_one_check_with_one_task() -> None:
    operation_store = _store()
    task_client = FakeTaskClient()

    answers = [_check(operation_store, task_client), _check(operation_store, task_client)]

    assert answers == ["pending", "pending"]
    ((operation_id, submission),) = task_client.enqueued
    assert submission == 0
    assert task_client.timeouts == [HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS]
    stored = operation_store.get(operation_id)
    assert stored is not None
    # The marker keeps the check apart from an over-budget submission, and with no members the
    # worker has none to compare. Every check carries one rendering, whatever the read asked.
    assert stored.request_identity == {"subject_id": _SPEC, "head_check": True}
    assert stored.options == HEAD_CHECK_OPTIONS.model_dump(mode="json")
    assert (stored.estimate.rows, stored.estimate.requests) == (0, 0)
    assert stored.dispatched_at is not None


def test_a_check_done_runs_again_under_a_new_submission() -> None:
    """A check that confirmed the head leaves its row done, and the next stale read resets it
    under a new submission count, so its task takes a new name."""
    operation_store = _store()
    task_client = FakeTaskClient()
    _check(operation_store, task_client)
    ((operation_id, _),) = task_client.enqueued
    operation_store.claim(operation_id, submission_count=0, lease_seconds=60, final_delivery=False)
    operation_store.complete(
        operation_id,
        attempt=1,
        subject_id=_SPEC,
        content_version="cv_1",
        confirmed_at=_START,
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
    )

    assert _check(operation_store, task_client) == "pending"
    assert task_client.enqueued == [(operation_id, 0), (operation_id, 1)]


def test_reads_against_a_failing_task_queue_retry_the_create_once_per_recovery_interval() -> None:
    """The read that wrote the row hands off the first create, under the short timeout, and
    answers `pending`. Every other read inside the recovery interval answers `stale` with no
    create, and past it one read claims the retry."""
    clock = _Clock()
    operation_store = _store(clock=clock)
    raising = _RaisingTaskClient()

    first = [_check(operation_store, raising) for _ in range(10)]
    clock.t += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS - 1)
    inside = [_check(operation_store, raising) for _ in range(10)]
    assert raising.timeouts == [HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS]
    clock.t += timedelta(seconds=2)
    past = [_check(operation_store, raising) for _ in range(10)]

    # Each read that handed off a create answers before it fails.
    assert first == ["pending"] + ["stale"] * 9
    assert inside == ["stale"] * 10
    assert past == ["pending"] + ["stale"] * 9
    assert raising.timeouts == [HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS] * 2
    # The claim restarts the interval, so the next retry waits a full one again.
    clock.t += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS - 1)
    _check(operation_store, raising)
    assert len(raising.timeouts) == 2

    task_client = FakeTaskClient()
    clock.t += timedelta(seconds=2)
    assert _check(operation_store, task_client) == "pending"
    ((operation_id, submission),) = task_client.enqueued
    assert submission == 0
    stored = operation_store.get(operation_id)
    assert stored is not None and stored.dispatched_at is not None


def test_a_create_another_instance_recorded_first_answers_pending() -> None:
    """The create returned but this read's record of it found the row already marked, so the
    read judges the row as it now stands."""
    operation_store = _LostMarkOperationStore(
        store=FakeOperationDatasetStore(), backlog_max=_BACKLOG_MAX, now=_Clock()
    )
    task_client = FakeTaskClient()

    assert _check(operation_store, task_client) == "pending"
    assert len(task_client.enqueued) == 1


def test_a_check_leaves_half_the_backlog_to_callers_who_wait() -> None:
    """With half the backlog live a check is refused and the read answers `stale`, while an
    over-budget submission still finds a place."""
    operation_store = _store(backlog_max=4)
    for n in range(2):
        _other_operation(operation_store, n)
    task_client = FakeTaskClient()

    assert _check(operation_store, task_client, backlog_max=4) == "stale"
    assert task_client.enqueued == []
    _other_operation(operation_store, 2)


def test_a_store_that_cannot_record_the_check_answers_stale() -> None:
    task_client = FakeTaskClient()

    assert _check(_UnavailableOperationStore(), task_client) == "stale"
    assert task_client.enqueued == []


def test_a_check_that_failed_for_good_answers_failed() -> None:
    """The read hands such a check's call to the work executor (`serve_stored_head`)."""
    operation_store = _store()
    task_client = FakeTaskClient()
    _check(operation_store, task_client)
    ((operation_id, _),) = task_client.enqueued
    operation_store.claim(operation_id, submission_count=0, lease_seconds=60, final_delivery=False)
    operation_store.fail(operation_id, attempt=1, refusal_code="over_ceiling", retryable=False)

    assert _check(operation_store, task_client) == "failed"
    assert len(task_client.enqueued) == 1


def test_a_programming_error_in_the_check_fails_the_read() -> None:
    """Only the store's and the queue's own errors answer `stale`; anything else surfaces."""
    with pytest.raises(TypeError):
        _check(_BrokenOperationStore(), FakeTaskClient())


def test_a_read_answers_pending_before_its_checks_task_is_created() -> None:
    """The read records the row and hands the create off: it answers `pending` with no call to
    the task client, and the create runs once afterwards."""
    operation_store = _store()
    task_client = FakeTaskClient()
    creates = _HeldCreates()

    assert _check(operation_store, task_client, task_creates=creates) == "pending"
    assert task_client.enqueued == []
    assert len(creates.held) == 1

    creates.run()

    ((operation_id, submission),) = task_client.enqueued
    assert submission == 0
    assert task_client.timeouts == [HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS]
    stored = operation_store.get(operation_id)
    assert stored is not None and stored.dispatched_at is not None


def test_reads_racing_the_create_hand_off_one() -> None:
    """A second read while the first read's create is still queued joins the row and offers
    nothing, so the check is submitted once. It states `stale`, as a read during an inline
    create did."""
    operation_store = _store()
    task_client = FakeTaskClient()
    creates = _HeldCreates()

    answers = [
        _check(operation_store, task_client, task_creates=creates),
        _check(operation_store, task_client, task_creates=creates),
    ]
    creates.run()

    assert answers == ["pending", "stale"]
    assert len(task_client.enqueued) == 1
    assert _check(operation_store, task_client, task_creates=creates) == "pending"
    assert creates.held == []


def test_a_failed_create_leaves_the_check_due_for_a_later_read() -> None:
    """The handed-off create fails after the read answered. The row stays undispatched, so a
    read past the recovery interval offers the create again and it is submitted."""
    clock = _Clock()
    operation_store = _store(clock=clock)
    creates = _HeldCreates()
    raising = _RaisingTaskClient()
    assert _check(operation_store, raising, task_creates=creates) == "pending"
    creates.run()
    assert len(raising.timeouts) == 1

    task_client = FakeTaskClient()
    assert _check(operation_store, task_client, task_creates=creates) == "stale"
    assert creates.held == []
    clock.t += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS + 1)
    assert _check(operation_store, task_client, task_creates=creates) == "pending"
    creates.run()

    ((operation_id, submission),) = task_client.enqueued
    assert submission == 0
    stored = operation_store.get(operation_id)
    assert stored is not None and stored.dispatched_at is not None


def test_a_full_pool_skips_the_create_and_a_later_read_submits_it() -> None:
    """With the pool's one place taken, the read records the row, offers the create, is refused
    and answers `stale` with no call to the task client. A read past the recovery interval,
    with the pool free, submits it."""
    clock = _Clock()
    operation_store = _store(clock=clock)
    task_client = FakeTaskClient()
    pool = TaskCreatePool(workers=1, capacity=1)
    release = threading.Event()
    assert pool.offer(lambda: release.wait(5))

    assert _check(operation_store, task_client, task_creates=pool) == "stale"
    release.set()
    assert pool.wait_idle(5)
    assert task_client.enqueued == []

    clock.t += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS + 1)
    assert _check(operation_store, task_client, task_creates=pool) == "pending"
    assert pool.wait_idle(5)
    assert len(task_client.enqueued) == 1


def test_the_pool_holds_at_most_its_capacity_and_frees_a_place_per_create() -> None:
    """Creates queued or running count against the capacity, a create that raises frees its
    place, and a shut-down pool refuses."""
    pool = TaskCreatePool(workers=1, capacity=2)
    release = threading.Event()
    ran: list[str] = []

    def failing() -> None:
        release.wait(5)
        ran.append("failing")
        raise RuntimeError("create failed")

    assert pool.offer(failing)
    assert pool.offer(lambda: ran.append("queued"))
    assert not pool.offer(lambda: ran.append("refused"))
    release.set()
    assert pool.wait_idle(5)
    assert ran == ["failing", "queued"]

    assert pool.offer(lambda: ran.append("after"))
    assert pool.wait_idle(5)
    pool.shutdown(ShutdownBudget(total_seconds=1))
    assert not pool.offer(lambda: ran.append("after shutdown"))
    assert ran == ["failing", "queued", "after"]


def test_shutdown_drops_queued_creates_and_frees_their_places() -> None:
    """A create still queued at shutdown never runs, so its row waits for the recovery, and the
    drain does not wait on it."""
    pool = TaskCreatePool(workers=1, capacity=2)
    started, release = threading.Event(), threading.Event()
    ran: list[str] = []

    def running() -> None:
        started.set()
        release.wait(5)
        ran.append("running")

    assert pool.offer(running)
    assert pool.offer(lambda: ran.append("queued"))
    assert started.wait(5)

    threading.Timer(0.05, release.set).start()
    pool.shutdown(ShutdownBudget(total_seconds=5))

    assert ran == ["running"]
    assert pool.wait_idle(0)
