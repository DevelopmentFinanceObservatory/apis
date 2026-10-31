"""Process-wide skills index with a reference-counted background refresh task.

Acquire and release run on the application event loop. The singleton lock guards
only the refcount. Task cancellation and its await happen outside that lock.

The index rebuilds when the release service swaps releases, and after a build with a failed
graph read until one has none. Acquirers wait for the first build attempt until one deadline,
`SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS` after the index is created, so startup completes with
full CPU behind the build and every acquirer together holds it no longer than one wait.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from dfo_platform.singleton import RefCountedSingleton
from graph_release.context import bound_release
from origin_catalogue import Catalogue, bound_catalogue
from skills_index.build import build_index
from skills_index.config import (
    SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS,
    SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS,
    SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS,
)
from skills_index.graph import SkillsGraph
from skills_index.types import SkillsIndex

if TYPE_CHECKING:
    from graph_release.service import Release, ReleaseService

logger = logging.getLogger(__name__)


class SharedSkillsIndex:
    """Process-wide index and refresh task. Use `acquire_shared_index` and
    `release_shared_index` to manage its lifecycle.

    `index` stays `None` until the first build succeeds, and every door answers 503
    until then. A failed build keeps the last generation serving.

    The task builds once and then again after each swap of `release`, and runs no timer. Its
    `on_swap` callback only wakes the task, so the build never runs in the service's recheck
    thread. A build that fails, or that serves an index with failed graph reads
    (`SkillsIndex.failed_reads`), retries with exponential backoff until one completes with
    none, which resets the backoff.

    Each build runs under a catalogue from `catalogue` and `release`'s served release,
    each bound for that build alone, since the refresh task serves no request. `None`
    binds nothing.

    `wait_for_build_attempt` observes one additional attempt, successful or not.
    `first_build_deadline` is the event-loop time at which acquirers stop waiting for the
    first build, `SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS` after construction.
    `_close` cancels and awaits the refresh task. A running worker thread may finish
    after cancellation. The cancelled awaiter discards the result. The application shuts down
    the executor with `wait=False` and `cancel_futures=True`.
    """

    def __init__(
        self,
        *,
        graph: SkillsGraph,
        catalogue: Callable[[], Catalogue] | None = None,
        release: ReleaseService | None = None,
    ) -> None:
        self._graph = graph
        self._catalogue = catalogue
        self._release = release
        self.index: SkillsIndex | None = None
        self.build_attempts = 0
        self.first_build_deadline = (
            asyncio.get_running_loop().time() + SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS
        )
        # The event belongs to the running loop that constructs this instance.
        self._build_attempted = asyncio.Event()
        self._first_build_attempted = asyncio.Event()
        self._swapped = asyncio.Event()
        self._unregister_swap: Callable[[], None] | None = None
        if release is not None:
            loop = asyncio.get_running_loop()

            def wake(_release: Release) -> None:
                loop.call_soon_threadsafe(self._swapped.set)

            self._unregister_swap = release.on_swap(wake)
        self._task = asyncio.create_task(self._refresh_loop(), name="skills-index-refresh")

    async def wait_for_build_attempt(self) -> None:
        """Await one more build attempt than the count at call start. Repeated calls
        observe successive attempts."""
        target = self.build_attempts + 1
        while self.build_attempts < target:
            await self._build_attempted.wait()
            self._build_attempted.clear()

    async def wait_for_first_build_attempt(self) -> None:
        """Await the first build attempt, successful or not. Later calls return at once."""
        await self._first_build_attempted.wait()

    async def _refresh_loop(self) -> None:
        """Build immediately, then again after each swap. Failures log and keep the last
        index. Cancellation exits the loop. After a build that failed or had a read fail, the
        next waits a backoff from `SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS` doubling to
        `SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS`."""
        retry_seconds = SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS
        while True:
            # Cleared before the build reads the served release, so a swap that lands
            # during the build triggers another.
            self._swapped.clear()
            clean = await self._build_once()
            self.build_attempts += 1
            self._build_attempted.set()
            self._first_build_attempted.set()
            if clean:
                retry_seconds = SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS
                await self._swapped.wait()
            else:
                await asyncio.sleep(retry_seconds)
                retry_seconds = min(
                    retry_seconds * 2, SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS
                )

    async def _build_once(self) -> bool:
        """Builds and publishes one generation. Returns whether it completed with no
        failed graph read."""
        started = time.monotonic()
        try:
            with (
                bound_catalogue(self._catalogue())
                if self._catalogue is not None
                else contextlib.nullcontext(),
                bound_release(self._release.snapshot())
                if self._release is not None
                else contextlib.nullcontext(),
            ):
                index = await asyncio.to_thread(build_index, graph=self._graph)
        except Exception:
            logger.exception("skills index build failed; the previous index keeps serving")
            return False
        self.index = index
        logger.info(
            "skills index built: %d skills in %.1f s, %d failed graph reads",
            len(index.skills),
            time.monotonic() - started,
            index.failed_reads,
        )
        if index.failed_reads:
            logger.warning(
                "skills index build had %d failed graph reads; it serves and is rebuilt",
                index.failed_reads,
            )
        return not index.failed_reads

    async def _close(self) -> None:
        if self._unregister_swap is not None:
            self._unregister_swap()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


def _build(
    graph: SkillsGraph | None,
    catalogue: Callable[[], Catalogue] | None,
    release: ReleaseService | None,
) -> SharedSkillsIndex:
    if graph is None:
        raise RuntimeError(
            "acquire_shared_index() needs a graph on the first call: no skills index "
            "has been built yet, and this acquirer passed graph=None"
        )
    return SharedSkillsIndex(graph=graph, catalogue=catalogue, release=release)


_singleton: RefCountedSingleton[SharedSkillsIndex] = RefCountedSingleton(
    build=_build, label="skills index"
)


async def acquire_shared_index(
    *,
    graph: SkillsGraph | None = None,
    catalogue: Callable[[], Catalogue] | None = None,
    release: ReleaseService | None = None,
) -> SharedSkillsIndex:
    """Acquire the process-wide index, building it on the first call and releasing it
    after the last caller. Only the first acquirer supplies `graph`, `catalogue` and
    `release`.
    Later arguments are ignored. `graph=None` on the first call raises `RuntimeError`.
    Call from a running event loop, which owns the refresh task.

    Returns only after the first build attempt has finished or the index's
    `first_build_deadline` has passed, so a later acquirer waits only for what remains of the
    first one's wait. Either shortfall logs a warning, and the refresh task keeps retrying."""
    shared = _singleton.acquire(graph, catalogue, release)
    if graph is not None and graph is not shared._graph:
        logger.info(
            "skills index already built; the existing index's graph is in use, this "
            "acquirer's graph is discarded"
        )
    await _wait_for_first_build(shared)
    return shared


async def _wait_for_first_build(shared: SharedSkillsIndex) -> None:
    try:
        async with asyncio.timeout_at(shared.first_build_deadline):
            await shared.wait_for_first_build_attempt()
    except TimeoutError:
        logger.warning(
            "skills index first build did not finish within %.0f s; serving starts and the "
            "doors answer 503 until it does",
            SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS,
        )
        return
    if shared.index is None:
        logger.warning(
            "skills index first build failed; serving starts and the doors answer 503 "
            "until a retry succeeds"
        )


async def release_shared_index(shared: SharedSkillsIndex) -> None:
    """Release one reference. The last release cancels the refresh task and drops the
    shared instance. Raises `RuntimeError` for a stale instance or excess release."""
    # `begin_release` clears the instance under lock. Await task teardown after unlocking.
    closing = _singleton.begin_release(shared)
    if closing is not None:
        await closing._close()
