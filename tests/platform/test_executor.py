"""Tests thread-pool saturation logging and slot release, the Graph surface's bounded
admission, context-carrying submission, and `map_in_threads`."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dfo_platform.executor import (
    SaturationTrackingExecutor,
    map_in_threads,
    run_in_executor_with_context,
    submit_with_context,
)
from graph_api.deps import DcpExecutorSaturated, DcpReadExecutor

_REQUEST_VALUE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_value", default=None
)


class TestSaturationTrackingExecutor:
    @pytest.mark.parametrize(
        ("executor_type", "consequence"),
        [
            (SaturationTrackingExecutor, "new work will queue"),
            (DcpReadExecutor, "new reads are refused"),
        ],
        ids=["queueing", "refusing"],
    )
    def test_saturation_logged_on_full_and_cleared_on_drain(
        self, caplog, executor_type: type[SaturationTrackingExecutor], consequence: str
    ):
        """The warning names what happens to the next submit: the shared pool queues it, the
        Graph surface's DCP-read pool refuses it."""
        release = threading.Event()

        def slow() -> None:
            release.wait(timeout=5)

        executor = executor_type(max_workers=2, thread_name_prefix="test-satur")
        try:
            with caplog.at_level(logging.INFO, logger="dfo_platform.executor"):
                futures = [executor.submit(slow) for _ in range(2)]
                release.set()
                for fut in futures:
                    fut.result(timeout=5)
        finally:
            executor.shutdown(wait=True)

        saturated = [
            r for r in caplog.records if r.levelno == logging.WARNING and "saturated" in r.message
        ]
        cleared = [r for r in caplog.records if "no longer saturated" in r.message]
        assert len(saturated) == 1
        assert saturated[0].message.endswith(consequence)
        assert len(cleared) == 1

    def test_cancelled_to_thread_call_frees_slot_without_wedging_pool(self):
        """Cancelling the coroutine leaves its slot occupied until the worker returns."""

        async def scenario() -> None:
            executor = SaturationTrackingExecutor(max_workers=1, thread_name_prefix="test-cancel")
            asyncio.get_running_loop().set_default_executor(executor)
            release = threading.Event()

            def slow() -> str:
                release.wait(timeout=5)
                return "done"

            task = asyncio.create_task(asyncio.to_thread(slow))
            for _ in range(50):  # let the worker thread actually pick up the call
                if executor._in_flight >= 1:
                    break
                await asyncio.sleep(0.01)
            assert executor._in_flight == 1

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # Cancellation ends the coroutine; the worker holds its slot until the call returns.
            assert executor._in_flight == 1

            release.set()  # let the underlying call return naturally
            for _ in range(50):
                if executor._in_flight == 0:
                    break
                await asyncio.sleep(0.01)
            assert executor._in_flight == 0  # slot freed once the call actually finished

            assert await asyncio.to_thread(lambda: "ok") == "ok"

            executor.shutdown(wait=True)

        asyncio.run(scenario())

    def test_cancelled_while_queued_releases_its_slot_and_never_runs(self):
        """A queued cancellation releases its counted slot even though the worker never starts."""

        async def scenario() -> None:
            executor = SaturationTrackingExecutor(max_workers=1, thread_name_prefix="test-queue")
            asyncio.get_running_loop().set_default_executor(executor)
            occupy_release = threading.Event()
            queued_ran = threading.Event()

            def occupy() -> None:
                occupy_release.wait(timeout=5)

            def queued() -> None:
                queued_ran.set()

            occupy_task = asyncio.create_task(asyncio.to_thread(occupy))
            for _ in range(50):  # let the sole worker actually pick up occupy()
                if executor._in_flight >= 1:
                    break
                await asyncio.sleep(0.01)
            assert executor._in_flight == 1

            queued_task = asyncio.create_task(asyncio.to_thread(queued))
            for _ in range(50):  # let queued() actually reach the (busy) pool
                if executor._in_flight >= 2:
                    break
                await asyncio.sleep(0.01)
            assert executor._in_flight == 2  # occupier running + queued call counted

            queued_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued_task

            for _ in range(50):  # slot release should be near-immediate; poll for safety
                if executor._in_flight == 1:
                    break
                await asyncio.sleep(0.01)
            assert executor._in_flight == 1  # queued call's slot released, not leaked
            assert not queued_ran.is_set()  # and it never actually ran

            occupy_release.set()
            await occupy_task
            for _ in range(50):
                if executor._in_flight == 0:
                    break
                await asyncio.sleep(0.01)
            assert executor._in_flight == 0

            executor.shutdown(wait=True)

        asyncio.run(scenario())

    def test_a_submit_refused_after_shutdown_leaves_the_in_flight_count_unchanged(self):
        executor = SaturationTrackingExecutor(max_workers=1, thread_name_prefix="test-refused")
        executor.shutdown(wait=True)

        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)

        assert executor._in_flight == 0


class TestDcpReadExecutorAdmission:
    def test_a_submit_with_every_worker_busy_is_refused_and_never_runs(self):
        """Admission is per worker: the refused call never runs and a returned read frees its
        slot for the next submit."""
        release = threading.Event()
        refused_ran = threading.Event()
        executor = DcpReadExecutor(max_workers=2, thread_name_prefix="test-admission")
        try:
            running = [executor.submit(release.wait, 5) for _ in range(2)]

            with pytest.raises(DcpExecutorSaturated):
                executor.submit(refused_ran.set)

            release.set()
            for future in running:
                future.result(timeout=5)
            assert executor.submit(lambda: "ok").result(timeout=5) == "ok"
        finally:
            executor.shutdown(wait=True)

        assert not refused_ran.is_set()


class TestContextCarryingSubmission:
    def test_plain_submit_drops_the_callers_context(self):
        """Control: without the helper a pool thread sees none of the caller's variables."""
        token = _REQUEST_VALUE.set("bound")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(_REQUEST_VALUE.get).result() is None
        finally:
            _REQUEST_VALUE.reset(token)

    def test_submit_with_context_carries_the_callers_context(self):
        token = _REQUEST_VALUE.set("bound")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = submit_with_context(
                    pool, lambda suffix: f"{_REQUEST_VALUE.get()}{suffix}", "!"
                )
                assert future.result() == "bound!"
        finally:
            _REQUEST_VALUE.reset(token)

    @pytest.mark.parametrize("own_pool", [True, False], ids=["own-pool", "default-executor"])
    def test_run_in_executor_with_context_carries_the_callers_context(self, own_pool: bool):
        async def _body() -> str | None:
            _REQUEST_VALUE.set("bound")
            if not own_pool:
                return await run_in_executor_with_context(None, _REQUEST_VALUE.get)
            with ThreadPoolExecutor(max_workers=1) as pool:
                return await run_in_executor_with_context(pool, _REQUEST_VALUE.get)

        assert asyncio.run(_body()) == "bound"


class TestMapInThreads:
    def test_results_keep_item_order_when_calls_finish_in_reverse(self):
        """Each call waits for the next item's call to finish, so the calls finish last item
        first, which also proves they run side by side. Each sees the caller's context, on a
        daemon thread that interpreter exit does not wait for."""
        finished = [threading.Event() for _ in range(4)]

        def call(index: int) -> str:
            if index + 1 < len(finished):
                assert finished[index + 1].wait(timeout=5)
            finished[index].set()
            daemon = "daemon" if threading.current_thread().daemon else "joined"
            return f"{_REQUEST_VALUE.get()}-{index}-{daemon}"

        token = _REQUEST_VALUE.set("bound")
        try:
            results = map_in_threads(call, range(4), max_workers=4, thread_name_prefix="t-map")
        finally:
            _REQUEST_VALUE.reset(token)

        assert results == [f"bound-{index}-daemon" for index in range(4)]

    def test_raises_the_earliest_failed_item_after_a_later_one_fails_first(self):
        """Item 0 starts before item 2, so item 2's failure cannot stop it."""
        later_failed = threading.Event()

        def call(index: int) -> int:
            if index == 2:
                later_failed.set()
                raise ValueError("item 2")
            if index == 0:
                assert later_failed.wait(timeout=5)
                raise KeyError("item 0")
            return index

        with pytest.raises(KeyError, match="item 0"):
            map_in_threads(call, range(3), max_workers=3, thread_name_prefix="t-map")

    def test_a_failure_starts_no_later_item(self):
        started: list[int] = []

        def call(index: int) -> int:
            started.append(index)
            if index == 0:
                raise ValueError("item 0")
            return index

        with pytest.raises(ValueError, match="item 0"):
            map_in_threads(call, range(50), max_workers=1, thread_name_prefix="t-map")

        assert started == [0]

    def test_no_items_start_no_threads(self):
        assert map_in_threads(str, [], max_workers=4, thread_name_prefix="t-map") == []
