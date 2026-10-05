"""Pins request deadline arithmetic and elapsed-deadline handling with an injected monotonic clock."""

from __future__ import annotations

import asyncio
import types

import pytest

from search.engine import core as core_mod
from search.engine import deadline as deadline_mod
from search.engine import envelope as envelope_mod
from search.engine.deadline import _await_bounded
from search.engine.entry import dataset_resubmit as dataset_resubmit_mod
from search.models import (
    DatasetResubmitInput,
    ParsedEntity,
    ParsedPart,
    ResolveRequest,
    ResubmitParsedEntry,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_TIMEOUT_S = 7.5


class _FakeClock:
    """A settable monotonic clock for deterministic deadline tests."""

    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _ReachedTheDoor(Exception):
    """Stops the request after the door records its deadline inputs."""


def test_request_deadline_is_start_plus_the_configured_timeout(monkeypatch, search_settings):
    """The door receives `start_ms / 1000 + SEARCH_REQUEST_TIMEOUT_S`."""
    clock = _FakeClock(4321.25)
    fake_time = types.SimpleNamespace(monotonic=clock)
    # `now_ms` stamps the start; `deadline` checks bounded awaits.
    monkeypatch.setattr(envelope_mod, "time", fake_time)
    monkeypatch.setattr(deadline_mod, "time", fake_time)
    search_settings(request_timeout_s=_TIMEOUT_S)

    captured: dict[str, float] = {}

    async def _door_stub(inp, **kwargs):
        captured["start_ms"] = kwargs["start_ms"]
        captured["deadline"] = kwargs["params"].deadline
        raise _ReachedTheDoor
        yield  # pragma: no cover -- the raise above is the only exit

    monkeypatch.setattr(dataset_resubmit_mod, "handle", _door_stub)

    request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitParsedEntry(
                    part=ParsedPart(
                        variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")]
                    )
                )
            ]
        )
    )

    async def _drive() -> None:
        graph = FakeGraph()
        with pytest.raises(_ReachedTheDoor):
            async for _ev in core_mod.resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
            ):
                pass

    asyncio.run(_drive())

    expected_start_ms = int(clock.now * 1000)
    assert captured["start_ms"] == expected_start_ms
    assert captured["deadline"] == expected_start_ms / 1000.0 + _TIMEOUT_S


def test_await_bounded_refuses_an_elapsed_deadline_without_starting_the_awaitable(monkeypatch):
    """An elapsed deadline raises before the awaitable runs or reaches the task factory."""
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(deadline_mod, "time", types.SimpleNamespace(monotonic=clock))

    started: list[str] = []
    scheduled: list[object] = []

    async def _never_run() -> None:
        started.append("started")

    async def _drive() -> None:
        loop = asyncio.get_running_loop()
        target_code = _never_run.__code__

        def _factory(loop, coro, **kwargs):
            # Exclude asyncio.run's shutdown tasks.
            if getattr(coro, "cr_code", None) is target_code:
                scheduled.append(coro)
            return asyncio.Task(coro, loop=loop, **kwargs)

        loop.set_task_factory(_factory)
        with pytest.raises(asyncio.TimeoutError):
            await _await_bounded(_never_run(), deadline=clock.now - 1.0)

    asyncio.run(_drive())

    assert started == [], "an awaitable past the deadline must never be given a first step"
    assert scheduled == [], "an awaitable past the deadline must never be handed to the loop"
