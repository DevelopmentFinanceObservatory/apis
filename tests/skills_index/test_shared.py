"""Tests shared-index lifecycle, rebuilds on swap, task ownership, and acquire/release behavior.

Local graph doubles count or block roster reads. Each test isolates the process-wide singleton to
keep refresh tasks on its own event loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from unittest import mock

import pytest
from fastapi import FastAPI

import skills_index.build as build_module
import skills_index.shared as shared_module
from graph_release.context import bound_release_or_none
from skills_index.config import HIERARCHY_PROPERTIES
from skills_index.graph import SubjectsCoverage, SubjectsPage
from skills_index.shared import acquire_shared_index, release_shared_index

from .conftest import FakeSkillsGraph, skill_node


@pytest.fixture(autouse=True)
def _isolated_singleton():
    with shared_module._singleton.isolated():
        yield


_real_sleep = asyncio.sleep


class _CountingRosterGraph:
    """Return an empty roster and raise on the 1-based attempts in `fail_on_attempts`.

    Each build attempt makes one roster call.
    """

    def __init__(self, *, fail_on_attempts: frozenset[int] = frozenset()) -> None:
        self._fail_on_attempts = fail_on_attempts
        self.attempts = 0

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        self.attempts += 1
        if self.attempts in self._fail_on_attempts:
            raise RuntimeError(f"boom on build attempt {self.attempts}")
        return {dcid: SubjectsPage(dcids=(), coverage=SubjectsCoverage.complete) for dcid in dcids}

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return dict.fromkeys(dcids)

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return dict.fromkeys(dcids, ())


class _BlockingRosterGraph:
    """Block the roster read on an event so tests can release the holder mid-build."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release_build = threading.Event()
        self.completed = threading.Event()

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        self.entered.set()
        self.release_build.wait()
        result = {
            dcid: SubjectsPage(dcids=(), coverage=SubjectsCoverage.complete) for dcid in dcids
        }
        self.completed.set()
        return result

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return dict.fromkeys(dcids)

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return dict.fromkeys(dcids, ())


async def _instant_sleep(_seconds: float) -> None:
    """Skip real refresh waits while tests advance the separately mocked clock."""


def _recording_sleep() -> tuple[object, list[float]]:
    """Record requested sleep durations in order without waiting."""
    calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    return fake_sleep, calls


def _pending_refresh_tasks() -> list[asyncio.Task]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "skills-index-refresh" and not task.done()
    ]


def test_acquire_returns_with_the_first_build_serving():
    async def scenario() -> None:
        graph = _CountingRosterGraph()
        shared = await acquire_shared_index(graph=graph)
        try:
            assert shared.build_attempts == 1
            assert shared.index is not None
            assert shared.index.skills == {}
        finally:
            await release_shared_index(shared)

    asyncio.run(scenario())


def test_release_abandons_a_build_genuinely_in_flight(monkeypatch):
    monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS", 0.05)

    async def scenario() -> None:
        graph = _BlockingRosterGraph()
        shared = await acquire_shared_index(graph=graph)

        # Wait for the worker to enter the roster read before releasing mid-build.
        await asyncio.to_thread(graph.entered.wait)
        assert shared.index is None

        await release_shared_index(shared)
        # Release returns while the worker blocks on `release_build`.
        assert not graph.completed.is_set()
        assert not _pending_refresh_tasks()

        # Let the cancelled worker finish and confirm its index stays unpublished.
        graph.release_build.set()
        await asyncio.to_thread(graph.completed.wait)
        assert shared.index is None

    asyncio.run(scenario())


class _ReachOutageGraph(FakeSkillsGraph):
    """One attached skill whose reach reads raise on the 1-based build attempts in
    `outage_on_attempts`. Each build attempt makes one roster read."""

    def __init__(self, *, outage_on_attempts: frozenset[int]) -> None:
        super().__init__(
            subjects={
                ("AgentSkill", "typeOf"): ["skill/skill-a"],
                ("Person", "populationType"): ["sv-1"],
            },
            arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        )
        self._outage_on_attempts = outage_on_attempts
        self.attempts = 0

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        if property == "typeOf" and "AgentSkill" in dcids:
            self.attempts += 1
            self.failing_subject_reads = (
                {("Person", prop) for prop in HIERARCHY_PROPERTIES}
                if self.attempts in self._outage_on_attempts
                else set()
            )
        return super().node_subjects_many(dcids, property=property, max_results=max_results)


def test_a_reach_outage_on_refresh_keeps_the_previous_generation_serving():
    async def scenario() -> None:
        graph = _ReachOutageGraph(outage_on_attempts=frozenset({2}))
        service = _FakeReleaseService(object())
        shared = await acquire_shared_index(
            graph=graph,
            release=service,  # ty: ignore[invalid-argument-type]
        )
        try:
            first_index = shared.index  # Attempt 1 reads every reach before acquire returns.
            assert first_index is not None
            assert first_index.skills["skill-a"].attached is True

            service.swap(object())
            await shared.wait_for_build_attempt()  # Attempt 2's reach reads all raise.
            assert shared.build_attempts == 2
            assert shared.index is first_index
        finally:
            await release_shared_index(shared)

    asyncio.run(scenario())


def test_a_reach_outage_on_the_first_build_leaves_the_holder_unbuilt():
    async def scenario() -> None:
        graph = _ReachOutageGraph(outage_on_attempts=frozenset({1}))
        with mock.patch.object(shared_module.asyncio, "sleep", _instant_sleep):
            shared = await acquire_shared_index(graph=graph)
            try:
                assert shared.build_attempts == 1
                assert shared.index is None
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_a_failed_first_build_retries_after_the_initial_backoff():
    async def scenario() -> None:
        fake_sleep, calls = _recording_sleep()
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            graph = _CountingRosterGraph(fail_on_attempts=frozenset({1}))
            shared = await acquire_shared_index(graph=graph)
            try:
                assert shared.index is None  # Attempt 1 fails before acquire returns.

                await shared.wait_for_build_attempt()  # Attempt 2 succeeds and binds.
                assert shared.index is not None
            finally:
                await release_shared_index(shared)

        assert calls == [shared_module.SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS]

    asyncio.run(scenario())


def test_the_backoff_doubles_up_to_a_ceiling_and_a_clean_build_waits_for_a_swap(monkeypatch):
    async def scenario() -> None:
        monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS", 1.0)
        monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS", 4.0)
        fake_sleep, calls = _recording_sleep()
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            graph = _CountingRosterGraph(fail_on_attempts=frozenset({1, 2, 3}))
            shared = await acquire_shared_index(graph=graph)
            try:
                for _ in range(3):
                    await shared.wait_for_build_attempt()
                assert shared.build_attempts == 4
                assert shared.index is not None
            finally:
                await release_shared_index(shared)

        # The delays double to the 4.0 ceiling, and none follows the clean fourth build.
        assert calls == [1.0, 2.0, 4.0]

    asyncio.run(scenario())


def test_two_acquires_share_one_instance_and_start_one_task():
    async def scenario() -> None:
        first_graph = _CountingRosterGraph()
        second_graph = _CountingRosterGraph()

        first = await acquire_shared_index(graph=first_graph)
        second = await acquire_shared_index(graph=second_graph)
        try:
            assert first is second
            assert len(_pending_refresh_tasks()) == 1

            # The first acquirer's graph supplies the shared index.
            assert first_graph.attempts == 1
            assert second_graph.attempts == 0
        finally:
            await release_shared_index(second)
            assert _pending_refresh_tasks()  # One reference keeps the refresh task active.

            await release_shared_index(first)
            assert not _pending_refresh_tasks()  # The final release ends the refresh task.

    asyncio.run(scenario())


def test_no_task_pending_after_the_last_release():
    async def scenario() -> None:
        graph = _CountingRosterGraph()
        shared = await acquire_shared_index(graph=graph)
        assert _pending_refresh_tasks()

        await release_shared_index(shared)
        assert not _pending_refresh_tasks()

    asyncio.run(scenario())


def test_lifespan_releases_when_the_body_raises_after_acquiring():
    async def scenario() -> None:
        graph = _CountingRosterGraph()

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            shared = await acquire_shared_index(graph=graph)
            try:
                raise RuntimeError("boom after acquiring")
                yield  # pragma: no cover -- the line above always raises first
            finally:
                await release_shared_index(shared)

        app = FastAPI(lifespan=lifespan)

        with pytest.raises(RuntimeError, match="boom after acquiring"):
            async with app.router.lifespan_context(app):
                pass  # pragma: no cover -- startup raises before reaching here

        assert not _pending_refresh_tasks()

    asyncio.run(scenario())


def test_release_refuses_a_foreign_instance_and_a_release_past_zero():
    async def scenario() -> None:
        graph = _CountingRosterGraph()
        shared = await acquire_shared_index(graph=graph)
        try:
            with pytest.raises(RuntimeError):
                await release_shared_index(object())  # type: ignore[arg-type]
        finally:
            await release_shared_index(shared)

        with pytest.raises(RuntimeError):
            await release_shared_index(shared)

    asyncio.run(scenario())


def test_acquire_with_graph_none_raises_on_the_0_to_1_transition():
    async def scenario() -> None:
        with pytest.raises(RuntimeError):
            await acquire_shared_index(graph=None)

    asyncio.run(scenario())


def test_a_later_acquirer_passing_graph_none_is_fine():
    async def scenario() -> None:
        first_graph = _CountingRosterGraph()
        first = await acquire_shared_index(graph=first_graph)
        try:
            second = await acquire_shared_index(graph=None)
            assert second is first
        finally:
            await release_shared_index(first)
            await release_shared_index(first)

    asyncio.run(scenario())


class _ReleaseRecordingGraph(_CountingRosterGraph):
    """Records the release bound to each build attempt's roster read."""

    def __init__(self, *, fail_on_attempts: frozenset[int] = frozenset()) -> None:
        super().__init__(fail_on_attempts=fail_on_attempts)
        self.bound: list[object] = []

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        self.bound.append(bound_release_or_none())
        return super().node_subjects_many(dcids, property=property, max_results=max_results)


class _FakeReleaseService:
    """Serves one release at a time and runs swap callbacks in another thread, as the
    service's recheck thread does."""

    def __init__(self, served: object) -> None:
        self.served = served
        self.callbacks: list[Callable[[object], None]] = []

    def snapshot(self) -> object:
        return self.served

    def on_swap(self, callback: Callable[[object], None]) -> Callable[[], None]:
        self.callbacks.append(callback)
        return lambda: self.callbacks.remove(callback)

    def swap(self, release: object) -> None:
        self.served = release
        thread = threading.Thread(target=lambda: [cb(release) for cb in list(self.callbacks)])
        thread.start()
        thread.join()


async def _yield_to_the_loop() -> None:
    for _ in range(10):
        await _real_sleep(0)


def test_acquire_waits_for_the_first_build_attempt():
    graph = _BlockingRosterGraph()

    async def scenario() -> None:
        acquiring = asyncio.create_task(acquire_shared_index(graph=graph))
        try:
            await asyncio.to_thread(graph.entered.wait)
            await _yield_to_the_loop()
            returned_before_the_build = acquiring.done()
        finally:
            # A worker left blocked would hang `asyncio.run`'s executor shutdown.
            graph.release_build.set()
        shared = await acquiring
        assert not returned_before_the_build
        try:
            assert shared.build_attempts == 1
            assert shared.index is not None
            # A later acquirer returns at once. Waiting on one more attempt would wait for a
            # swap that never comes.
            second = await asyncio.wait_for(acquire_shared_index(), timeout=5)
            assert second is shared
            await release_shared_index(second)
        finally:
            await release_shared_index(shared)

    asyncio.run(scenario())


def _slow_first_build() -> tuple[_BlockingRosterGraph, Callable[[], None]]:
    graph = _BlockingRosterGraph()
    return graph, graph.release_build.set


def _failed_first_build() -> tuple[_CountingRosterGraph, Callable[[], None]]:
    return _CountingRosterGraph(fail_on_attempts=frozenset({1})), lambda: None


@pytest.mark.parametrize(
    ("first_build", "attempts_at_return", "returns_at_the_bound"),
    [(_slow_first_build, 0, True), (_failed_first_build, 1, False)],
    ids=["slower-than-the-bound", "failed"],
)
def test_startup_proceeds_past_a_first_build_shortfall_and_retries(
    monkeypatch, first_build, attempts_at_return, returns_at_the_bound
):
    bound = 0.05
    monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS", bound)
    graph, unblock = first_build()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        with mock.patch.object(shared_module.asyncio, "sleep", _instant_sleep):
            started = loop.time()
            shared = await acquire_shared_index(graph=graph)
            try:
                assert (loop.time() - started >= bound) is returns_at_the_bound
                assert shared.build_attempts == attempts_at_return
                assert shared.index is None

                unblock()
                while shared.index is None:
                    await shared.wait_for_build_attempt()
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_acquirers_share_one_first_build_wait(monkeypatch):
    """Search and Skills acquire in turn at startup; past a slow first build, the second
    returns at the first's deadline rather than waiting a bound of its own."""
    bound = 0.5
    monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_WAIT_SECONDS", bound)
    graph, unblock = _slow_first_build()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        shared = await acquire_shared_index(graph=graph)
        try:
            started = loop.time()
            second = await acquire_shared_index()
            assert loop.time() - started < bound / 2
            await release_shared_index(second)
        finally:
            unblock()
            await release_shared_index(shared)

    asyncio.run(scenario())


def test_a_swap_rebuilds_once_under_the_new_release():
    first, second = object(), object()

    async def scenario() -> None:
        fake_sleep, calls = _recording_sleep()
        service = _FakeReleaseService(first)
        graph = _ReleaseRecordingGraph()
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            shared = await acquire_shared_index(
                graph=graph,
                release=service,  # ty: ignore[invalid-argument-type]
            )
            try:
                first_index = shared.index
                assert first_index is not None
                await _yield_to_the_loop()
                assert graph.bound == [first]

                service.swap(second)
                await shared.wait_for_build_attempt()
                await _yield_to_the_loop()
                assert graph.bound == [first, second]
                assert shared.index is not first_index
            finally:
                await release_shared_index(shared)

        # No timer: the task waits on swaps alone.
        assert calls == []
        assert service.callbacks == []

    asyncio.run(scenario())


def test_a_failed_build_keeps_the_index_and_retries_with_backoff():
    first, second = object(), object()

    async def scenario() -> None:
        fake_sleep, calls = _recording_sleep()
        service = _FakeReleaseService(first)
        graph = _ReleaseRecordingGraph(fail_on_attempts=frozenset({2, 3}))
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            shared = await acquire_shared_index(
                graph=graph,
                release=service,  # ty: ignore[invalid-argument-type]
            )
            try:
                first_index = shared.index  # Attempt 1 succeeds before acquire returns.
                assert first_index is not None

                service.swap(second)
                await shared.wait_for_build_attempt()  # Attempt 2 fails.
                assert shared.index is first_index
                await shared.wait_for_build_attempt()  # Attempt 3 fails.
                assert shared.index is first_index
                await shared.wait_for_build_attempt()  # Attempt 4 succeeds.
                await _yield_to_the_loop()
                assert shared.index is not first_index
                assert graph.bound == [first, second, second, second]
            finally:
                await release_shared_index(shared)

        initial = shared_module.SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS
        assert calls == [initial, initial * 2]

    asyncio.run(scenario())


class _SkillReadFailingOnceGraph(FakeSkillsGraph):
    """Two published skills whose node reads each take their own chunk, the read naming
    `skill/b` raising on its first call alone."""

    def __init__(self) -> None:
        super().__init__(
            subjects={("AgentSkill", "typeOf"): ["skill/a", "skill/b"]},
            arcs={"skill/a": skill_node("a"), "skill/b": skill_node("b")},
        )
        self.failed = False

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if "skill/b" in dcids and not self.failed:
            self.failed = True
            raise RuntimeError("simulated read failure for skill/b")
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def test_a_build_with_failed_reads_retries_until_a_clean_build(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)

    async def scenario() -> None:
        fake_sleep, calls = _recording_sleep()
        service = _FakeReleaseService(object())
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            shared = await acquire_shared_index(
                graph=_SkillReadFailingOnceGraph(),
                release=service,  # ty: ignore[invalid-argument-type]
            )
            try:
                partial = shared.index
                assert partial is not None
                assert list(partial.skills) == ["a"]

                await asyncio.wait_for(shared.wait_for_build_attempt(), timeout=5)
                recovered = shared.index
                assert recovered is not None
                assert list(recovered.skills) == ["a", "b"]
            finally:
                await release_shared_index(shared)

        # One backoff after the build with a failed read, then a wait on swaps alone.
        assert calls == [shared_module.SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS]

    asyncio.run(scenario())
