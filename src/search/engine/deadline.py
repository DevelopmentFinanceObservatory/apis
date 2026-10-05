"""Bound pipeline awaits to one monotonic deadline and finish task cancellation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Iterable
from typing import Any, Protocol


async def _await_bounded[T](coro: Coroutine[Any, Any, T], *, deadline: float) -> T:
    """Await `coro` until the shared monotonic deadline.

    If the deadline has passed, close the coroutine and raise `asyncio.TimeoutError`
    before awaiting it. Cancellation propagates to native async work. Cancelling a
    `to_thread` or executor await only detaches it, leaving worker work running.
    The pipeline transports map this timeout to HTTP 504.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        coro.close()  # never awaited -- close it to avoid a "coroutine was never awaited" warning
        raise TimeoutError("pipeline deadline already elapsed")
    return await asyncio.wait_for(coro, timeout=remaining)


async def _cancel_and_await(task: asyncio.Task | None) -> None:
    """Cancel and await a live task. Do nothing for `None` or a finished task."""
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _CarriesTrailingTasks(Protocol):
    """Structural protocol for a region's trailing task handles."""

    @property
    def slot_options_task(self) -> asyncio.Task | None: ...

    @property
    def observations_task(self) -> asyncio.Task | None: ...

    @property
    def place_options_task(self) -> asyncio.Task | None: ...


async def _cancel_trailing_tasks(regions: Iterable[_CarriesTrailingTasks]) -> None:
    """Cancel all trailing tasks before awaiting any, so caller cancellation
    cannot leave later tasks running during unwind.
    """
    owned = list(regions)
    for region in owned:
        for task in (region.slot_options_task, region.observations_task, region.place_options_task):
            if task is not None and not task.done():
                task.cancel()
    for region in owned:
        await _cancel_and_await(region.slot_options_task)
        await _cancel_and_await(region.observations_task)
        await _cancel_and_await(region.place_options_task)
