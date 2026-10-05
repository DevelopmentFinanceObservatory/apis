"""Tests shared-index lifecycle, refresh, task ownership, and acquire/release behavior.

Local graph doubles count or block roster reads. Each test isolates the process-wide singleton to
keep refresh tasks on its own event loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from contextlib import asynccontextmanager
from unittest import mock

import pytest
from fastapi import FastAPI

import skills_index.shared as shared_module
from skills_index.config import HIERARCHY_PROPERTIES
from skills_index.graph import SubjectsCoverage, SubjectsPage
from skills_index.shared import acquire_shared_index, release_shared_index

from .conftest import FakeSkillsGraph, skill_node


@pytest.fixture(autouse=True)
def _isolated_singleton():
    with shared_module._singleton.isolated():
        yield


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


def test_holder_serves_none_then_the_built_index():
    async def scenario() -> None:
        graph = _CountingRosterGraph()
        shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)
        try:
            assert shared.index is None
            await shared.wait_for_build_attempt()
            assert shared.index is not None
            assert shared.index.skills == {}
        finally:
            await release_shared_index(shared)

    asyncio.run(scenario())


def test_release_abandons_a_build_genuinely_in_flight():
    async def scenario() -> None:
        graph = _BlockingRosterGraph()
        shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)

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


def test_a_failed_build_leaves_the_previous_index_serving_and_the_loop_alive():
    async def scenario() -> None:
        graph = _CountingRosterGraph(fail_on_attempts=frozenset({2}))
        # A zero-second interval stresses retries. Each `to_thread` await yields so the test can
        # observe attempts 1, 2, and 3 in order.
        shared = await acquire_shared_index(graph=graph, refresh_seconds=0.0)
        try:
            await shared.wait_for_build_attempt()  # Attempt 1 succeeds.
            assert shared.build_attempts == 1
            first_index = shared.index
            assert first_index is not None

            await shared.wait_for_build_attempt()  # Attempt 2 raises.
            assert shared.build_attempts == 2
            assert shared.index is first_index

            await shared.wait_for_build_attempt()  # Attempt 3 succeeds.
            assert shared.build_attempts == 3
            assert shared.index is not None
            assert shared.index is not first_index
        finally:
            await release_shared_index(shared)

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
        shared = await acquire_shared_index(graph=graph, refresh_seconds=0.0)
        try:
            await shared.wait_for_build_attempt()  # Attempt 1 reads every reach.
            first_index = shared.index
            assert first_index is not None
            assert first_index.skills["skill-a"].attached is True

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
            shared = await acquire_shared_index(graph=graph, refresh_seconds=0.0)
            try:
                await shared.wait_for_build_attempt()
                assert shared.index is None
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_a_failed_first_build_retries_quickly_rather_than_after_the_full_interval():
    async def scenario() -> None:
        fake_sleep, calls = _recording_sleep()
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            graph = _CountingRosterGraph(fail_on_attempts=frozenset({1}))
            shared = await acquire_shared_index(graph=graph, refresh_seconds=900.0)
            try:
                await shared.wait_for_build_attempt()  # Attempt 1 fails before binding.
                assert shared.index is None

                await shared.wait_for_build_attempt()  # Attempt 2 succeeds and binds.
                assert shared.index is not None
            finally:
                await release_shared_index(shared)

        # The pre-bind retry delay is shorter than `refresh_seconds`.
        # Attempt 2 may trigger a later sleep after it binds.
        assert calls[0] == shared_module.SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS
        assert calls[0] < 900.0

    asyncio.run(scenario())


def test_the_pre_bind_backoff_doubles_up_to_a_ceiling_then_the_normal_interval_resumes(
    monkeypatch,
):
    async def scenario() -> None:
        monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS", 1.0)
        monkeypatch.setattr(shared_module, "SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS", 4.0)
        fake_sleep, calls = _recording_sleep()
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            graph = _CountingRosterGraph(fail_on_attempts=frozenset({1, 2, 3}))
            shared = await acquire_shared_index(graph=graph, refresh_seconds=900.0)
            try:
                for _ in range(4):
                    await shared.wait_for_build_attempt()
                assert shared.index is not None
            finally:
                await release_shared_index(shared)

        # Pre-bind delays double to the 4.0 ceiling.
        # The loop uses `refresh_seconds` after binding.
        assert calls == [1.0, 2.0, 4.0, 900.0]

    asyncio.run(scenario())


def test_a_refresh_failure_after_binding_waits_the_normal_interval_not_the_backoff():
    async def scenario() -> None:
        fake_sleep, calls = _recording_sleep()
        with mock.patch.object(shared_module.asyncio, "sleep", fake_sleep):
            graph = _CountingRosterGraph(fail_on_attempts=frozenset({2}))
            shared = await acquire_shared_index(graph=graph, refresh_seconds=900.0)
            try:
                await shared.wait_for_build_attempt()  # Attempt 1 succeeds and binds.
                assert shared.index is not None

                await shared.wait_for_build_attempt()  # Attempt 2 fails after binding.
                assert shared.index is not None
            finally:
                await release_shared_index(shared)

        # The bound generation keeps serving through failure, so both waits use the normal interval.
        assert calls == [900.0, 900.0]

    asyncio.run(scenario())


def test_two_acquires_share_one_instance_and_start_one_task():
    async def scenario() -> None:
        first_graph = _CountingRosterGraph()
        second_graph = _CountingRosterGraph()

        first = await acquire_shared_index(graph=first_graph, refresh_seconds=1000.0)
        second = await acquire_shared_index(graph=second_graph, refresh_seconds=1000.0)
        try:
            assert first is second
            assert len(_pending_refresh_tasks()) == 1

            # The first acquirer's graph supplies the shared index.
            await first.wait_for_build_attempt()
            assert first_graph.attempts >= 1
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
        shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)
        assert _pending_refresh_tasks()

        await release_shared_index(shared)
        assert not _pending_refresh_tasks()

    asyncio.run(scenario())


def test_lifespan_releases_when_the_body_raises_after_acquiring():
    async def scenario() -> None:
        graph = _CountingRosterGraph()

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)
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


def test_deadline_shrinks_as_a_driven_clock_advances_within_one_generation():
    async def scenario() -> None:
        clock = [100.0]
        with mock.patch.object(shared_module.time, "monotonic", lambda: clock[0]):
            graph = _CountingRosterGraph()
            shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)
            try:
                await shared.wait_for_build_attempt()
                assert shared.seconds_until_next_rebuild() == 1000.0

                clock[0] += 400.0
                assert shared.seconds_until_next_rebuild() == 600.0

                clock[0] += 200.0
                assert shared.seconds_until_next_rebuild() == 400.0
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_a_new_generation_resets_the_deadline():
    async def scenario() -> None:
        clock = [0.0]
        with (
            mock.patch.object(shared_module.time, "monotonic", lambda: clock[0]),
            mock.patch.object(shared_module.asyncio, "sleep", _instant_sleep),
        ):
            graph = _CountingRosterGraph()
            shared = await acquire_shared_index(graph=graph, refresh_seconds=100.0)
            try:
                await shared.wait_for_build_attempt()
                assert shared.seconds_until_next_rebuild() == 100.0

                clock[0] = 90.0
                assert shared.seconds_until_next_rebuild() == 10.0

                await shared.wait_for_build_attempt()
                assert shared.seconds_until_next_rebuild() == 100.0
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_a_failed_rebuild_does_not_reset_the_deadline():
    async def scenario() -> None:
        clock = [0.0]
        with (
            mock.patch.object(shared_module.time, "monotonic", lambda: clock[0]),
            mock.patch.object(shared_module.asyncio, "sleep", _instant_sleep),
        ):
            graph = _CountingRosterGraph(fail_on_attempts=frozenset({2}))
            shared = await acquire_shared_index(graph=graph, refresh_seconds=100.0)
            try:
                await shared.wait_for_build_attempt()
                first_index = shared.index
                assert shared.seconds_until_next_rebuild() == 100.0

                clock[0] = 60.0
                await shared.wait_for_build_attempt()
                assert shared.index is first_index
                # The 100-second deadline leaves 40 seconds after 60 seconds elapse.
                assert shared.seconds_until_next_rebuild() == 40.0
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_the_floor_applies_past_the_deadline():
    async def scenario() -> None:
        clock = [0.0]
        with mock.patch.object(shared_module.time, "monotonic", lambda: clock[0]):
            graph = _CountingRosterGraph()
            shared = await acquire_shared_index(graph=graph, refresh_seconds=100.0)
            try:
                await shared.wait_for_build_attempt()
                assert shared.seconds_until_next_rebuild() == 100.0

                clock[0] = 500.0
                assert (
                    shared.seconds_until_next_rebuild()
                    == shared_module._REFRESH_DEADLINE_FLOOR_SECONDS
                )
            finally:
                await release_shared_index(shared)

    asyncio.run(scenario())


def test_seconds_until_next_rebuild_before_any_bind_raises():
    async def scenario() -> None:
        graph = _BlockingRosterGraph()
        shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)

        # Check after the worker enters the roster read, while the build is in flight.
        await asyncio.to_thread(graph.entered.wait)
        assert shared.index is None
        with pytest.raises(RuntimeError):
            shared.seconds_until_next_rebuild()

        await release_shared_index(shared)
        graph.release_build.set()
        await asyncio.to_thread(graph.completed.wait)

    asyncio.run(scenario())


def test_release_refuses_a_foreign_instance_and_a_release_past_zero():
    async def scenario() -> None:
        graph = _CountingRosterGraph()
        shared = await acquire_shared_index(graph=graph, refresh_seconds=1000.0)
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
            await acquire_shared_index(graph=None, refresh_seconds=1000.0)

    asyncio.run(scenario())


def test_a_later_acquirer_passing_graph_none_is_fine():
    async def scenario() -> None:
        first_graph = _CountingRosterGraph()
        first = await acquire_shared_index(graph=first_graph, refresh_seconds=1000.0)
        try:
            second = await acquire_shared_index(graph=None, refresh_seconds=1000.0)
            assert second is first
        finally:
            await release_shared_index(first)
            await release_shared_index(first)

    asyncio.run(scenario())
