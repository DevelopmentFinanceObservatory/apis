"""Tests pin the executor and bounded semaphore created for each door, and their shutdown drain."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor

import pytest

from datasets_api.door_executor import build_door_executor
from datasets_api.shutdown import ShutdownBudget


def test_build_door_executor_sizes_both_to_pool_size() -> None:
    door_executor = build_door_executor(3, "exemplar")

    assert isinstance(door_executor.executor, ThreadPoolExecutor)
    assert isinstance(door_executor.semaphore, threading.BoundedSemaphore)
    acquired = [door_executor.semaphore.acquire(blocking=False) for _ in range(3)]
    assert acquired == [True, True, True]
    assert door_executor.semaphore.acquire(blocking=False) is False


def test_build_door_executor_names_its_threads_from_name() -> None:
    door_executor = build_door_executor(1, "exemplar")

    assert door_executor.executor._thread_name_prefix == "datasets-exemplar"


def test_build_door_executor_surface_qualifies_the_name() -> None:
    door_executor = build_door_executor(1, "exemplar", surface="presentation")

    assert door_executor.name == "presentation-exemplar"
    assert door_executor.executor._thread_name_prefix == "presentation-exemplar"


def test_drain_waits_for_a_held_permit_and_leaves_capacity_unchanged() -> None:
    door_executor = build_door_executor(2, "exemplar")
    door_executor.semaphore.acquire()
    threading.Timer(0.1, door_executor.semaphore.release).start()

    door_executor.drain(ShutdownBudget(total_seconds=5.0))

    acquired = [door_executor.semaphore.acquire(blocking=False) for _ in range(2)]
    assert acquired == [True, True]
    assert door_executor.semaphore.acquire(blocking=False) is False


def test_drain_logs_the_permits_still_held_when_the_budget_expires(caplog) -> None:
    door_executor = build_door_executor(2, "exemplar")
    door_executor.semaphore.acquire()

    with caplog.at_level(logging.WARNING, logger="datasets_api.door_executor"):
        door_executor.drain(ShutdownBudget(total_seconds=0.05))

    assert "datasets-exemplar executor's shutdown drain timed out with 1/2" in caplog.text
    # The drain released what it took: one permit is still free.
    assert door_executor.semaphore.acquire(blocking=False) is True
    assert door_executor.semaphore.acquire(blocking=False) is False


def test_build_door_executor_semaphore_is_bounded() -> None:
    door_executor = build_door_executor(1, "exemplar")
    door_executor.semaphore.acquire()
    door_executor.semaphore.release()

    with pytest.raises(ValueError):
        door_executor.semaphore.release()


class _Refused(Exception):
    pass


def _free_permits(door_executor, pool_size: int) -> int:
    """Count the free permits, polling briefly since a done callback releases on a worker thread."""
    deadline = time.monotonic() + 2
    while True:
        free = 0
        while door_executor.semaphore.acquire(blocking=False):
            free += 1
        for _ in range(free):
            door_executor.semaphore.release()
        if free == pool_size or time.monotonic() > deadline:
            return free
        time.sleep(0.01)


def test_run_returns_the_job_result_under_the_callers_context_and_releases_the_permit() -> None:
    door_executor = build_door_executor(1, "exemplar")
    marker = contextvars.ContextVar[str]("marker")

    async def call() -> str:
        marker.set("request")
        return await door_executor.run(marker.get, refuse=_Refused)

    assert asyncio.run(call()) == "request"
    assert _free_permits(door_executor, 1) == 1
    door_executor.executor.shutdown()


def test_a_job_submitted_straight_to_the_executor_runs_under_the_submitters_context() -> None:
    """A door that submits to the executor itself, as publish does, keeps the request's bound
    state, so no call site has to remember to copy it."""
    door_executor = build_door_executor(1, "exemplar")
    marker = contextvars.ContextVar[str]("marker")

    def submit() -> str:
        marker.set("request")
        return door_executor.executor.submit(marker.get).result(timeout=5)

    assert contextvars.copy_context().run(submit) == "request"
    door_executor.executor.shutdown()


def test_run_refuses_at_capacity_and_frees_the_permit_when_the_running_job_ends() -> None:
    door_executor = build_door_executor(1, "exemplar")
    started = threading.Event()
    release = threading.Event()

    def blocked() -> str:
        started.set()
        release.wait(5)
        return "done"

    async def call() -> str:
        first = asyncio.ensure_future(door_executor.run(blocked, refuse=_Refused))
        await asyncio.to_thread(started.wait, 5)
        try:
            with pytest.raises(_Refused):
                await door_executor.run(lambda: "unreached", refuse=_Refused)
        finally:
            release.set()
        return await first

    assert asyncio.run(call()) == "done"
    assert _free_permits(door_executor, 1) == 1
    door_executor.executor.shutdown()


def test_run_releases_the_permit_when_the_job_fails() -> None:
    door_executor = build_door_executor(1, "exemplar")

    def failing() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(door_executor.run(failing, refuse=_Refused))

    assert _free_permits(door_executor, 1) == 1
    door_executor.executor.shutdown()


def test_run_refuses_after_shutdown_and_releases_the_permit() -> None:
    door_executor = build_door_executor(1, "exemplar")
    door_executor.executor.shutdown()

    with pytest.raises(_Refused):
        asyncio.run(door_executor.run(lambda: "unreached", refuse=_Refused))

    assert _free_permits(door_executor, 1) == 1


def test_run_logs_a_broken_executor_and_reraises(monkeypatch, caplog) -> None:
    door_executor = build_door_executor(1, "exemplar")

    def broken(*_args, **_kwargs):
        raise BrokenExecutor("pool is broken")

    monkeypatch.setattr(door_executor.executor, "submit", broken)

    with (
        caplog.at_level(logging.ERROR, logger="datasets_api.door_executor"),
        pytest.raises(BrokenExecutor),
    ):
        asyncio.run(door_executor.run(lambda: "unreached", refuse=_Refused))

    assert "datasets-exemplar executor pool is permanently broken" in caplog.text
    assert _free_permits(door_executor, 1) == 1
