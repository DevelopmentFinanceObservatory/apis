"""Process-wide skills index with a reference-counted background refresh task.

Acquire and release run on the application event loop. The singleton lock guards
only the refcount. Task cancellation and its await happen outside that lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

from dfo_platform.singleton import RefCountedSingleton
from origin_catalogue import Catalogue, bound_catalogue
from skills_index.build import build_index
from skills_index.config import (
    SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS,
    SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS,
    SKILLS_INDEX_REFRESH_SECONDS,
)
from skills_index.graph import SkillsGraph
from skills_index.types import SkillsIndex

logger = logging.getLogger(__name__)

# A 1-second floor keeps shared caches from revalidating every request after the
# current generation's refresh deadline.
_REFRESH_DEADLINE_FLOOR_SECONDS = 1.0


class SharedSkillsIndex:
    """Process-wide index and refresh task. Use `acquire_shared_index` and
    `release_shared_index` to manage its lifecycle.

    `index` stays `None` until the first build succeeds, and every door answers 503
    until then. Pre-bind failures retry
    with exponential backoff. Later failures keep the last generation serving and
    wait `refresh_seconds`. Its deadline follows the generation currently served.

    Each build runs under a catalogue from `catalogue`, bound for that build alone,
    since the refresh task serves no request. `None` binds nothing.

    `wait_for_build_attempt` observes one additional attempt, successful or not.
    `_close` cancels and awaits the refresh task. A running worker thread may finish
    after cancellation. The cancelled awaiter discards the result. The application shuts down
    the executor with `wait=False` and `cancel_futures=True`.
    """

    def __init__(
        self,
        *,
        graph: SkillsGraph,
        refresh_seconds: float,
        catalogue: Callable[[], Catalogue] | None = None,
    ) -> None:
        self._graph = graph
        self._catalogue = catalogue
        self._refresh_seconds = refresh_seconds
        self.index: SkillsIndex | None = None
        self.build_attempts = 0
        # Monotonic bind time for the generation in `index`. `None` before the first build.
        self._bound_at: float | None = None
        # The event belongs to the running loop that constructs this instance.
        self._build_attempted = asyncio.Event()
        self._task = asyncio.create_task(self._refresh_loop(), name="skills-index-refresh")

    async def wait_for_build_attempt(self) -> None:
        """Await one more build attempt than the count at call start. Repeated calls
        observe successive attempts."""
        target = self.build_attempts + 1
        while self.build_attempts < target:
            await self._build_attempted.wait()
            self._build_attempted.clear()

    def seconds_until_next_rebuild(self) -> float:
        """Return seconds until the current generation's next scheduled build, floored
        at `_REFRESH_DEADLINE_FLOOR_SECONDS` after its deadline. Raises
        `RuntimeError` before the first generation binds."""
        if self._bound_at is None:
            raise RuntimeError(
                "seconds_until_next_rebuild() called before any build has bound a generation"
            )
        deadline = self._bound_at + self._refresh_seconds
        return max(deadline - time.monotonic(), _REFRESH_DEADLINE_FLOOR_SECONDS)

    async def _refresh_loop(self) -> None:
        """Build immediately, then retry forever. Failures log and keep the last index.
        Cancellation exits the loop. After the first successful bind, retries use
        `refresh_seconds`. Earlier failures use exponential backoff from
        `SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS` to
        `SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS`."""
        retry_seconds = SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS
        while True:
            try:
                with (
                    bound_catalogue(self._catalogue())
                    if self._catalogue is not None
                    else contextlib.nullcontext()
                ):
                    self.index = await asyncio.to_thread(build_index, graph=self._graph)
                # Tie the deadline to the generation published in `index`.
                # Failures leave it unchanged.
                self._bound_at = time.monotonic()
            except Exception:
                logger.exception("skills index build failed; the previous index keeps serving")
            self.build_attempts += 1
            self._build_attempted.set()
            if self._bound_at is None:
                await asyncio.sleep(retry_seconds)
                retry_seconds = min(
                    retry_seconds * 2, SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS
                )
            else:
                await asyncio.sleep(self._refresh_seconds)

    async def _close(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


def _build(
    graph: SkillsGraph | None,
    refresh_seconds: float,
    catalogue: Callable[[], Catalogue] | None,
) -> SharedSkillsIndex:
    if graph is None:
        raise RuntimeError(
            "acquire_shared_index() needs a graph on the first call: no skills index "
            "has been built yet, and this acquirer passed graph=None"
        )
    return SharedSkillsIndex(graph=graph, refresh_seconds=refresh_seconds, catalogue=catalogue)


_singleton: RefCountedSingleton[SharedSkillsIndex] = RefCountedSingleton(
    build=_build, label="skills index"
)


async def acquire_shared_index(
    *,
    graph: SkillsGraph | None = None,
    refresh_seconds: float = SKILLS_INDEX_REFRESH_SECONDS,
    catalogue: Callable[[], Catalogue] | None = None,
) -> SharedSkillsIndex:
    """Acquire the process-wide index, building it on the first call and releasing it
    after the last caller. Only the first acquirer supplies `graph` and `catalogue`.
    Later arguments are ignored. `graph=None` on the first call raises `RuntimeError`.
    Call from a running event loop, which owns the refresh task."""
    shared = _singleton.acquire(graph, refresh_seconds, catalogue)
    if graph is not None and graph is not shared._graph:
        logger.info(
            "skills index already built; the existing index's graph is in use, this "
            "acquirer's graph is discarded"
        )
    return shared


async def release_shared_index(shared: SharedSkillsIndex) -> None:
    """Release one reference. The last release cancels the refresh task and drops the
    shared instance. Raises `RuntimeError` for a stale instance or excess release."""
    # `begin_release` clears the instance under lock. Await task teardown after unlocking.
    closing = _singleton.begin_release(shared)
    if closing is not None:
        await closing._close()
