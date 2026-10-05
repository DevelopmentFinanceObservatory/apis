"""A door's executor, admission semaphore, and shutdown drain.

Each door owns one `DoorExecutor`. The module imports only the standard library,
`dfo_platform.executor`, and `datasets_api.shutdown`, so any surface's doors can use it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor
from dataclasses import dataclass

from datasets_api.shutdown import ShutdownBudget
from dfo_platform.executor import submit_with_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DoorExecutor:
    """A door's executor and same-sized admission semaphore.

    `name` is the surface-qualified door name (`datasets-pin`). It prefixes the
    worker thread names and appears in the shutdown drain log line.
    """

    executor: ThreadPoolExecutor
    semaphore: threading.BoundedSemaphore
    pool_size: int
    name: str

    async def run[T](self, job: Callable[[], T], /, *, refuse: Callable[[], Exception]) -> T:
        """Run `job` on the executor under the caller's context behind non-blocking admission.

        `refuse` builds the door's backpressure exception, raised when no permit is free
        and when the executor has shut down. A done callback on the executor future
        releases the permit, so cancelling a queued future frees it and a running job
        keeps it until the job returns. On `BrokenExecutor`, which signals a permanently failed
        pool, the permit is released, the failure is logged, and the error propagates.
        """
        if not self.semaphore.acquire(blocking=False):
            raise refuse()
        try:
            future = submit_with_context(self.executor, job)
        except BrokenExecutor:
            self.semaphore.release()
            logger.error(
                "%s door's own executor pool is permanently broken; every future "
                "submitted to it will fail the same way",
                self.name,
                exc_info=True,
            )
            raise
        except RuntimeError:
            self.semaphore.release()
            raise refuse() from None
        except BaseException:
            self.semaphore.release()
            raise
        future.add_done_callback(lambda _future: self.semaphore.release())
        return await asyncio.wrap_future(future)

    def drain(self, budget: ShutdownBudget) -> None:
        """Wait until every permit is free or the shutdown budget expires.

        Run off the event loop. Acquired permits are released immediately, so
        the drain leaves the semaphore's capacity unchanged. Log the permits
        still held when the budget expires. The caller closes its dependencies
        anyway.
        """
        acquired = 0
        for _ in range(self.pool_size):
            remaining = budget.remaining()
            if remaining <= 0 or not self.semaphore.acquire(timeout=remaining):
                break
            acquired += 1
        for _ in range(acquired):
            self.semaphore.release()
        if acquired < self.pool_size:
            logger.warning(
                "%s door's shutdown drain timed out with %d/%d backpressure permit(s) "
                "still held; closing dependencies anyway",
                self.name,
                self.pool_size - acquired,
                self.pool_size,
            )


def build_door_executor(pool_size: int, name: str, *, surface: str = "datasets") -> DoorExecutor:
    """Build a door-specific executor and same-sized bounded semaphore.

    Separate executors keep blocked datastore work from starving other
    surfaces. The bounded semaphore raises on double release.
    """
    door_name = f"{surface}-{name}"
    return DoorExecutor(
        executor=ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix=door_name),
        semaphore=threading.BoundedSemaphore(pool_size),
        pool_size=pool_size,
        name=door_name,
    )


__all__ = ["DoorExecutor", "build_door_executor"]
