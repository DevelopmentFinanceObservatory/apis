"""Creates the Cloud Tasks task of a head check off the request path.

A stored read past its head's recheck window records the check's row in the request
(`head_check.submit_head_check`) and hands the task create to the surface's `TaskCreatePool`,
so the reader waits on no Cloud Tasks call. The pool holds `TASK_CREATE_WORKERS` threads and
at most `TASK_CREATE_CAPACITY` creates queued or running. A create offered to a full pool, or
to one shut down, is refused, and the read answers `stale`. The pool starts no thread until its
first create.

A create that fails, times out or never runs leaves the row `pending` with no recorded
dispatch, which is what a failed inline create left. A later read past
`ENQUEUE_RECOVERY_SECONDS` claims the retry (`claim_dispatch_retry`) and offers the create
again, under the same task name, so Cloud Tasks runs the check once.

Cloud Run throttles an instance's CPU once no request is in flight. The create starts on a pool
thread while the read that offered it is still building its answer, so it usually returns
inside that request. One still running when the instance goes idle resumes with the next
request, or times out after `HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS` of wall-clock time, and the
recovery above retries it. Shutdown refuses new creates and waits for running ones within the
surface's drain budget. A create still queued then is dropped, and the recovery retries it.

Each create runs in an empty context, so it adds nothing to the timings of the request that
offered it, whose answer has usually left by then, and carries no request deadline.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from datasets_api.shutdown import ShutdownBudget

logger = logging.getLogger(__name__)

# Creates run at once. Each waits at most `HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS` (one second), so
# two threads clear a full pool in at most `TASK_CREATE_CAPACITY / 2` seconds.
TASK_CREATE_WORKERS = 2

# Creates queued or running at once. One read per due head offers a create, so a burst of reads
# across this many distinct heads fills it.
TASK_CREATE_CAPACITY = 32


class TaskCreates(Protocol):
    """Where a read hands the task create of a head check it recorded."""

    def offer(self, create: Callable[[], object]) -> bool:
        """Run `create`, now or later, and return whether it was accepted."""
        ...


class InlineTaskCreates:
    """Runs each create in the caller's thread. The default of a `DefinitionDeps` built
    outside the surface's lifespan, such as a worker run's or a test's."""

    def offer(self, create: Callable[[], object]) -> bool:
        create()
        return True


class TaskCreatePool:
    """A bounded pool that runs each create on its own thread, after the caller returns."""

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

    def offer(self, create: Callable[[], object]) -> bool:
        """Queue `create` and return `True`, or return `False` when the pool already holds
        `capacity` creates or has shut down."""
        with self._idle:
            if self._closed or self._held >= self._capacity:
                return False
            self._held += 1
        try:
            future = self._executor.submit(self._run, create)
        except RuntimeError:
            # Shut down between the check and the submit.
            self._release()
            return False
        # Fires when the create returns and when shutdown drops it unrun.
        future.add_done_callback(lambda _future: self._release())
        return True

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no create is queued or running, at most `timeout` seconds, and return
        whether the pool is idle."""
        with self._idle:
            return self._idle.wait_for(lambda: self._held == 0, timeout=timeout)

    def shutdown(self, budget: ShutdownBudget) -> None:
        """Refuse new creates, drop queued ones and wait for running ones within `budget`.
        Run off the event loop."""
        with self._idle:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        if not self.wait_idle(budget.remaining()):
            logger.warning(
                "%s pool's shutdown drain timed out with %d task create(s) running; their "
                "rows wait for a later read to retry them",
                self._name,
                self._held,
            )

    def _run(self, create: Callable[[], object]) -> None:
        """Run `create` in an empty context, logging what it raises."""
        try:
            contextvars.Context().run(create)
        except Exception:
            logger.warning("%s: a task create raised", self._name, exc_info=True)

    def _release(self) -> None:
        with self._idle:
            self._held -= 1
            self._idle.notify_all()


__all__ = [
    "TASK_CREATE_CAPACITY",
    "TASK_CREATE_WORKERS",
    "InlineTaskCreates",
    "TaskCreatePool",
    "TaskCreates",
]
