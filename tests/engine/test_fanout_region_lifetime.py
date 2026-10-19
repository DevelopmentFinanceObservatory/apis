"""Checks that fan-out cancels landed region tasks on cancellation and infra errors."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from search.engine.binders import GeminiBinder
from search.engine.errors import GraphInfraError
from search.engine.fanout import _FanOutResult, _RegionDone, _run_variables_streaming
from search.engine.region import RegionResult
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


async def _drain(agen: AsyncIterator[object]) -> None:
    async for _ev in agen:
        pass


async def _run_cancellation_scenario() -> tuple[_FanOutResult, asyncio.Task]:
    """Cancel fan-out after leg 0 queues `_RegionDone` but before the generator consumes it.

    Return the result and leg 0's live task for cleanup assertions.
    """
    graph = FakeGraph()
    resolver = make_resolver(graph)
    admission = Admission(graph=graph)
    sdmx = FakeSdmxClient(observations={}, availability={})
    llm = FakeLLM(responses={})

    live_task = asyncio.create_task(asyncio.sleep(3600))
    marker_queued = asyncio.Event()

    class _SignalingQueue(asyncio.Queue):
        """Signals before `put` wakes `queue.get()`, letting the driver cancel before the generator resumes."""

        async def put(self, item: object) -> None:
            if isinstance(item, _RegionDone) and item.index == 0:
                marker_queued.set()
            await super().put(item)

    async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
        if kwargs["variable_index"] == 0:
            return RegionResult(
                variable_text=variable,
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
                slot_options_task=live_task,
            )
        # Leg 1 never finishes -- keeps n_pending > 0 so the generator stays
        # parked on `queue.get()` rather than reaching its terminal assembly.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    out = _FanOutResult()
    with (
        patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable),
        patch("search.engine.fanout.asyncio.Queue", new=_SignalingQueue),
    ):
        agen = _run_variables_streaming(
            ["v0", "v1"],
            entities=[],
            date_request=None,
            query="",
            pac=False,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            binder=GeminiBinder(),
            sdmx=sdmx,
            base_timing={},
            pre_resolved=None,
            deadline=time.monotonic() + 3600,
            degrade_on_error=True,
            use_semaphore=False,
            out=out,
            coverage=empty_coverage_snapshot(),
        )
        driver = asyncio.create_task(_drain(agen))
        await marker_queued.wait()
        driver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driver

    return out, live_task


def test_cancellation_mid_stream_cancels_a_landed_regions_live_task_unread():
    """Cancellation cleans up a queued leg's live task before the consumer reads the result."""

    async def _scenario() -> None:
        _out, live_task = await _run_cancellation_scenario()
        try:
            assert live_task.cancelled(), "leg 0's live task survived the cancelled request"
        finally:
            if not live_task.done():
                live_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await live_task

    asyncio.run(_scenario())


def test_infra_error_cancels_a_completed_siblings_live_task_unread():
    """Leg 1's infra error surfaces after leg 0 lands with a live task.

    Fan-out must cancel the sibling task before raising, even when the consumer only iterates.
    """

    async def _scenario() -> None:
        live_task = asyncio.create_task(asyncio.sleep(3600))

        async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
            if kwargs["variable_index"] == 0:
                return RegionResult(
                    variable_text=variable,
                    status="no_data",
                    specs=(),
                    no_data_reason="variable_not_resolved",
                    warnings=(),
                    timing_by_step={},
                    observations_task=live_task,
                )
            raise GraphInfraError("simulated graph transport failure")

        graph = FakeGraph()
        resolver = make_resolver(graph)
        admission = Admission(graph=graph)
        sdmx = FakeSdmxClient(observations={}, availability={})
        llm = FakeLLM(responses={})

        out = _FanOutResult()
        raised: BaseException | None = None
        try:
            with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
                async for _ev in _run_variables_streaming(
                    ["v0", "v1"],
                    entities=[],
                    date_request=None,
                    query="",
                    pac=False,
                    graph=graph,
                    resolver=resolver,
                    admission=admission,
                    llm=llm,
                    binder=GeminiBinder(),
                    sdmx=sdmx,
                    base_timing={},
                    pre_resolved=None,
                    deadline=time.monotonic() + 3600,
                    degrade_on_error=True,
                    use_semaphore=False,
                    out=out,
                    coverage=empty_coverage_snapshot(),
                ):
                    pass
        except GraphInfraError as exc:
            raised = exc

        try:
            assert raised is not None, "expected the injected graph infra error to surface"
            assert live_task.cancelled(), "sibling's live task survived the infra failure"
        finally:
            if not live_task.done():
                live_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await live_task

    asyncio.run(_scenario())
