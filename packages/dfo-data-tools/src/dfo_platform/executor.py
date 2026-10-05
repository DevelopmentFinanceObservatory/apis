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

# The initializer tags owned workers. The app checks this sentinel by identity,
# since a replacement pool can reuse the `dfo-io` thread name.
_pool_identity = threading.local()
_POOL_SENTINEL = object()


def _mark_worker_as_ours() -> None:
    _pool_identity.value = _POOL_SENTINEL


def current_thread_is_pool_owned() -> bool:
    """True on a thread started by SaturationTrackingExecutor itself,
    regardless of the thread's own name.
    """
    return getattr(_pool_identity, "value", None) is _POOL_SENTINEL


class SaturationTrackingExecutor(ThreadPoolExecutor):
    """Shared default executor that logs at capacity.

    Every `asyncio.to_thread()` call across mounted surfaces submits here. A
    queued cancellation cancels the future and releases its slot. A running
    thread cannot be interrupted and holds its slot until the function returns.

    The gauge covers the shared pool. Ad hoc executors fall outside it. A slot can span
    sequential HTTP calls. Each runs once, bounded by `SEARCH_GRAPH_TIMEOUT_S` or
    `SEARCH_LLM_TIMEOUT_S`. Observation fetches add at most one
    `SEARCH_GRAPH_TIMEOUT_S` permit wait per fetch.
    """

    def __init__(self, *, max_workers: int, **kwargs: Any) -> None:
        super().__init__(max_workers=max_workers, initializer=_mark_worker_as_ours, **kwargs)
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
                "shared thread pool saturated (%d/%d workers busy); new to_thread work will queue",
                self._in_flight,
                self._pool_size,
            )

        future = super().submit(fn, *args, **kwargs)
        # The raw future's callback releases slots on completion and when queued cancellation
        # skips `fn`.
        future.add_done_callback(self._release_slot)
        return future

    def _release_slot(self, _future: Future) -> None:
        with self._lock:
            self._in_flight -= 1
            cleared = self._saturated and self._in_flight < self._pool_size
            if cleared:
                self._saturated = False
        if cleared:
            logger.info(
                "shared thread pool no longer saturated (%d/%d workers busy)",
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
