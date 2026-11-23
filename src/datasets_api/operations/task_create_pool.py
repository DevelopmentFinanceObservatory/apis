"""Creates the Cloud Tasks task of a head check on a pool thread, beside the request.

A stored read past its head's recheck window records the check's row in the request
(`head_check.request_head_check`) and hands the task create to the surface's `TaskCreatePool`,
so the create runs while the read resolves, reads and builds the rest of its answer. The pool
holds `TASK_CREATE_WORKERS` threads and at most `TASK_CREATE_CAPACITY` creates queued or
running. A create offered to a full pool, or to one shut down, is refused, and the read answers
`stale`. The pool starts no thread until its first create.

Cloud Run throttles an instance's CPU once no request is in flight, so a create still running
after its response left could freeze until the next request and outlive its deadline. A door
that serves such reads therefore opens `joining_creates` around its work: each create a read
offers inside it is recorded (`OfferedCreates`), and the door waits for them on the event loop
before its response leaves, at most until the latest create's own deadline, under the
`head_check_join` phase. A read with no door around it (`offered_creates()` is `None`) creates
inline, as `InlineTaskCreates` does.

A create that fails, times out, never runs, or outlives the join leaves the row `pending` with
no recorded dispatch, which is what a failed inline create left. A later read past
`ENQUEUE_RECOVERY_SECONDS` claims the retry (`claim_dispatch_retry`) and offers the create
again, under the same task name, so Cloud Tasks runs the check once. Shutdown refuses new
creates, drops queued ones, and waits for running ones for what remains of the surface's drain
budget, and at least `TASK_CREATE_DRAIN_RESERVE_SECONDS`, so the task client and the store are
not closed under a create. The recovery retries a dropped create.

Each create runs in an empty context, so it adds nothing to the timings of the request that
offered it, whose join is timed in its place, and carries no request deadline.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Protocol

from datasets_api.shutdown import ShutdownBudget
from dfo_platform.timings import PhaseTimings

logger = logging.getLogger(__name__)

# Creates run at once. Each waits at most `HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS` (one second), so
# two threads clear a full pool in at most `TASK_CREATE_CAPACITY / 2` seconds.
TASK_CREATE_WORKERS = 2

# Creates queued or running at once. One read per due head offers a create, so a burst of reads
# across this many distinct heads fills it.
TASK_CREATE_CAPACITY = 32

# The least a shutdown waits for running creates, even when the executors' drains spent the
# surface's budget: one create's one-second deadline with room to record its dispatch.
TASK_CREATE_DRAIN_RESERVE_SECONDS = 1.5

# The phase a door's wait for its offered creates is timed under.
JOIN_PHASE = "head_check_join"


class TaskCreates(Protocol):
    """Where a read hands the task create of a head check it recorded."""

    def offer[T](self, create: Callable[[], T]) -> Future[T] | None:
        """Run `create`, now or later, and return its future, or `None` when it is refused."""
        ...


class InlineTaskCreates:
    """Runs each create in the caller's thread and returns its settled future. The default of
    a `DefinitionDeps` built outside the surface's lifespan, such as a worker run's or a
    test's. What `create` raises propagates to the caller."""

    def offer[T](self, create: Callable[[], T]) -> Future[T]:
        future: Future[T] = Future()
        future.set_result(create())
        return future


class OfferedCreates:
    """The creates one request offered, each with the monotonic deadline its join waits to."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offered: list[tuple[Future[Any], float]] = []

    def add(self, future: Future[Any], *, timeout: float) -> None:
        """Record `future`, which the join waits for at most `timeout` seconds from now."""
        with self._lock:
            self._offered.append((future, time.monotonic() + timeout))

    async def join(self, timings: PhaseTimings) -> None:
        """Wait on the event loop for every recorded create still running, until the latest of
        their deadlines, timing the wait under `JOIN_PHASE`. A request that offered none
        records nothing. Creates still running at the deadline are logged once and left to
        the recovery."""
        with self._lock:
            offered = list(self._offered)
        if not offered:
            return
        started = time.perf_counter()
        running = [future for future, _deadline in offered if not future.done()]
        if running:
            timeout = max(0.0, max(deadline for _future, deadline in offered) - time.monotonic())
            waiting = [asyncio.wrap_future(future) for future in running]
            for wrapped in waiting:
                wrapped.add_done_callback(_retrieve)
            _done, pending = await asyncio.wait(waiting, timeout=timeout)
            if pending:
                logger.warning(
                    "%d head check task create(s) still running when the request's join "
                    "ended; a later read retries any that fail",
                    len(pending),
                )
        timings.record(JOIN_PHASE, (time.perf_counter() - started) * 1000)


_offered: contextvars.ContextVar[OfferedCreates | None] = contextvars.ContextVar(
    "datasets_offered_task_creates", default=None
)


def offered_creates() -> OfferedCreates | None:
    """The creates of the request this runs in, or `None` outside `offering_creates`."""
    return _offered.get()


@contextmanager
def offering_creates() -> Iterator[OfferedCreates]:
    """Record the creates offered inside the block, and in the executor jobs it submits, which
    run under a copy of its context."""
    offered = OfferedCreates()
    token = _offered.set(offered)
    try:
        yield offered
    finally:
        _offered.reset(token)


@asynccontextmanager
async def joining_creates(timings: PhaseTimings) -> AsyncIterator[None]:
    """Record the creates offered inside the block (`offering_creates`) and join them when it
    ends (`OfferedCreates.join`), so a door's response leaves after them."""
    with offering_creates() as offered:
        try:
            yield
        finally:
            await offered.join(timings)


class TaskCreatePool:
    """A bounded pool that runs each create on its own thread."""

    def __init__(
        self,
        *,
        workers: int = TASK_CREATE_WORKERS,
        capacity: int = TASK_CREATE_CAPACITY,
        name: str = "datasets-task-create",
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self._capacity = capacity
        self._name = name
        self._held = 0
        self._closed = False
        self._idle = threading.Condition()

    def offer[T](self, create: Callable[[], T]) -> Future[T] | None:
        """Queue `create` and return its future, or return `None` when the pool already holds
        `capacity` creates or has shut down."""
        with self._idle:
            if self._closed or self._held >= self._capacity:
                return None
            self._held += 1

        def run_empty() -> T:
            return contextvars.Context().run(create)

        try:
            future = self._executor.submit(run_empty)
        except RuntimeError:
            # Shut down between the check and the submit.
            self._release()
            return None
        # Fires when the create returns and when shutdown drops it unrun.
        future.add_done_callback(self._settled)
        return future

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no create is queued or running, at most `timeout` seconds, and return
        whether the pool is idle."""
        with self._idle:
            return self._idle.wait_for(lambda: self._held == 0, timeout=timeout)

    def shutdown(
        self, budget: ShutdownBudget, *, reserve: float = TASK_CREATE_DRAIN_RESERVE_SECONDS
    ) -> None:
        """Refuse new creates, drop queued ones and wait for running ones for what remains of
        `budget`, and at least `reserve` seconds. Run off the event loop."""
        with self._idle:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        if not self.wait_idle(max(budget.remaining(), reserve)):
            logger.warning(
                "%s pool's shutdown drain timed out with %d task create(s) running; their "
                "rows wait for a later read to retry them",
                self._name,
                self._held,
            )

    def _settled(self, future: Future[Any]) -> None:
        """Free the create's place and log what it raised."""
        self._release()
        if future.cancelled():
            return
        if (exc := future.exception()) is not None:
            logger.warning("%s: a task create raised", self._name, exc_info=exc)

    def _release(self) -> None:
        with self._idle:
            self._held -= 1
            self._idle.notify_all()


def _retrieve(future: asyncio.Future[Any]) -> None:
    """Mark the outcome of a joined create as read, so asyncio logs nothing for it; the pool
    logs what a create raised."""
    if not future.cancelled():
        future.exception()


__all__ = [
    "JOIN_PHASE",
    "TASK_CREATE_CAPACITY",
    "TASK_CREATE_DRAIN_RESERVE_SECONDS",
    "TASK_CREATE_WORKERS",
    "InlineTaskCreates",
    "OfferedCreates",
    "TaskCreatePool",
    "TaskCreates",
    "joining_creates",
    "offered_creates",
    "offering_creates",
]
