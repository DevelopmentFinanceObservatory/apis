"""The process-wide thread pool installed as the event loop's default executor, and
helpers that carry the caller's context into any pool."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# Each worker's initializer records the executor that started it. The app compares
# that owner by identity, since a replacement pool can reuse the `dfo-io` thread name
# and every SaturationTrackingExecutor stamps its workers the same way.
_worker_owner = threading.local()


def current_thread_is_owned_by(executor: Executor) -> bool:
    """True on a worker thread that `executor` itself started, regardless of the
    thread's own name."""
    return getattr(_worker_owner, "executor", None) is executor


class SaturationTrackingExecutor(ThreadPoolExecutor):
    """Thread pool that logs a WARNING when its workers are all busy, so further work queues.

    The platform installs one as the loop's default executor (`dfo-io`), which every
    `asyncio.to_thread()` call across mounted surfaces submits to. The Graph surface
    runs its DCP reads on another (`graph-dcp`). A queued cancellation cancels the
    future and releases its slot. A running thread cannot be interrupted and holds its
    slot until the function returns. Each log line names the pool by `thread_name_prefix`.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str, **kwargs: Any) -> None:
        super().__init__(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            initializer=self._mark_worker,
            **kwargs,
        )
        self._name = thread_name_prefix
        self._pool_size = max_workers
        self._lock = threading.Lock()
        self._in_flight = 0
        self._saturated = False

    def submit(self, fn, /, *args: Any, **kwargs: Any):  # type: ignore[override]
        with self._lock:
            self._in_flight += 1
            newly_saturated = self._in_flight >= self._pool_size and not self._saturated
            self._saturated = self._in_flight >= self._pool_size
        if newly_saturated:
            logger.warning(
                "thread pool %s saturated (%d/%d workers busy); new work will queue",
                self._name,
                self._in_flight,
                self._pool_size,
            )

        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            # A refused submit never runs, so it must not hold a slot.
            self._release_slot(None)
            raise
        # The raw future's callback releases slots on completion and when queued cancellation
        # skips `fn`.
        future.add_done_callback(self._release_slot)
        return future

    def _mark_worker(self) -> None:
        _worker_owner.executor = self

    def _release_slot(self, _future: Future | None) -> None:
        with self._lock:
            self._in_flight -= 1
            cleared = self._saturated and self._in_flight < self._pool_size
            if cleared:
                self._saturated = False
        if cleared:
            logger.info(
                "thread pool %s no longer saturated (%d/%d workers busy)",
                self._name,
                self._in_flight,
                self._pool_size,
            )


def submit_with_context[**P, R](
    executor: Executor, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> Future[R]:
    """Submit `fn` to run under a copy of the caller's context.

    Context variables do not cross `Executor.submit` or `loop.run_in_executor`, so
    request-scoped state such as the bound origin catalogue is lost on a pool thread.
    """
    context = contextvars.copy_context()

    def run() -> R:
        return context.run(fn, *args, **kwargs)

    return executor.submit(run)


async def run_in_executor_with_context[*Ts, R](
    executor: Executor | None, fn: Callable[[*Ts], R], /, *args: *Ts
) -> R:
    """Await `fn(*args)` on `executor`, or on the loop's default executor when `None`,
    under a copy of the caller's context, as `asyncio.to_thread` does for the default."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, contextvars.copy_context().run, fn, *args)
