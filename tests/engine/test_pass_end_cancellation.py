"""A pass cancels the member walks and the listing of Earth's countries it queued, however
it ends (`pipeline._cancel_queued_reads`). Each test drives `_run_phases` over fake phases
on a one-worker pool that a blocker holds, so what the pass queued is still queued when it
ends. The blocker is released only once the test has checked the futures.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast, override

import pytest

from dfo_platform.executor import submit_with_context
from entity_resolution import Admission, EntityResolver
from materialise.place_scope import DefaultPresence
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.region import phase_place_materialise, pipeline
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import QuantityReadOnlyLLM

type _Phase = Callable[[ResolveState], Awaitable[RegionResult | None]]


class _Boom(Exception):
    pass


class _RecordingExecutor(ThreadPoolExecutor):
    """A thread pool that keeps every future submitted to it."""

    def __init__(self) -> None:
        super().__init__(max_workers=1, thread_name_prefix="test-walks")
        self.submitted: list[Future[Any]] = []

    @override
    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        future = super().submit(fn, *args, **kwargs)
        self.submitted.append(future)
        return future


class _Pool:
    """A one-worker pool held by a blocker, and the record of what was queued behind the
    blocker and what ran on it."""

    def __init__(self) -> None:
        self.executor = _RecordingExecutor()
        self.release = threading.Event()
        self.ran: list[str] = []
        self.executor.submit(self.release.wait)
        self.executor.submitted.clear()

    def walk(self) -> None:
        self.ran.append("walk")

    def listing(self) -> DefaultPresence:
        self.ran.append("listing")
        return cast(DefaultPresence, None)

    def drain(self) -> list[str]:
        """Release the blocker, let every queued item run, and return what ran."""
        self.release.set()
        self.executor.shutdown(wait=True)
        return self.ran


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Pool]:
    held = _Pool()
    monkeypatch.setattr(phase_place_materialise, "_WALK_POOL", held.executor)
    try:
        yield held
    finally:
        held.drain()


def _state() -> ResolveState:
    """Build a pass's state. The fake phases read none of its fields."""
    return ResolveState(
        variable="aid",
        entities=[],
        date_request=None,
        detect_query="aid",
        role_query="aid",
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=QuantityReadOnlyLLM(),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )


def _start_walk(pool: _Pool) -> _Phase:
    """A phase that queues a walk on `_WALK_POOL` into `state.alternate_walks`, as
    `_start_alternate_walks` does, and leaves it unawaited."""

    async def phase(state: ResolveState) -> None:
        state.alternate_walks[("walk",)] = submit_with_context(
            phase_place_materialise._WALK_POOL, pool.walk
        )

    return phase


async def _terminal(state: ResolveState) -> RegionResult:
    return RegionResult(
        variable_text="aid",
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


async def _raises(state: ResolveState) -> None:
    raise _Boom


def _waits(reached: asyncio.Event) -> _Phase:
    """A phase that signals `reached`, then waits on an event nothing sets."""

    async def phase(state: ResolveState) -> None:
        reached.set()
        await asyncio.Event().wait()

    return phase


def _walk_of(state: ResolveState) -> Future[object]:
    return cast(Future[object], state.alternate_walks[("walk",)])


def test_a_terminal_result_cancels_the_queued_walk(
    pool: _Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "_PHASES", (_start_walk(pool), _terminal))
    state = _state()

    result = asyncio.run(pipeline._run_phases(state))

    assert result.status == "no_data"
    assert _walk_of(state).cancelled()
    assert pool.drain() == []


def test_a_raising_phase_cancels_the_queued_walk(
    pool: _Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "_PHASES", (_start_walk(pool), _raises, _terminal))
    state = _state()

    with pytest.raises(_Boom):
        asyncio.run(pipeline._run_phases(state))

    assert _walk_of(state).cancelled()
    assert pool.drain() == []


def test_cancelling_the_pass_cancels_the_queued_walk(
    pool: _Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state()

    async def cancel_mid_pass() -> None:
        reached = asyncio.Event()
        monkeypatch.setattr(pipeline, "_PHASES", (_start_walk(pool), _waits(reached), _terminal))
        task = asyncio.create_task(pipeline._run_phases(state))
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_pass())

    assert _walk_of(state).cancelled()
    assert pool.drain() == []


def test_a_timeout_around_the_pass_cancels_the_queued_walk(
    pool: _Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline that cancels `_run_phases` inside `asyncio.timeout`."""
    state = _state()
    monkeypatch.setattr(
        pipeline, "_PHASES", (_start_walk(pool), _waits(asyncio.Event()), _terminal)
    )

    async def time_out_mid_pass() -> None:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0):
                await pipeline._run_phases(state)

    asyncio.run(time_out_mid_pass())

    assert _walk_of(state).cancelled()
    assert pool.drain() == []


def test_the_pass_end_cancels_an_untaken_countries_listing(
    pool: _Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing runs on the default executor (`_start_countries_listing`), here the
    held pool, so it is still queued when the pass ends."""
    state = _state()

    async def start_listing(state: ResolveState) -> None:
        state.countries_listing = asyncio.ensure_future(asyncio.to_thread(pool.listing))
        # One step of the listing's task submits the read to the executor's queue.
        await asyncio.sleep(0)
        assert len(pool.executor.submitted) == 1

    monkeypatch.setattr(pipeline, "_PHASES", (start_listing, _terminal))

    async def run_pass() -> bool:
        asyncio.get_running_loop().set_default_executor(pool.executor)
        await pipeline._run_phases(state)
        listing = state.countries_listing
        assert listing is not None
        # Five seconds only bounds a regression, which would otherwise hang on the blocker.
        await asyncio.wait({listing}, timeout=5)
        # `asyncio.run` shuts the default executor down on exit, which waits on the blocker.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        # Read before `asyncio.run` cancels every task left at its exit.
        return listing.cancelled() and pool.executor.submitted[0].cancelled()

    assert asyncio.run(run_pass())
    assert pool.drain() == []
