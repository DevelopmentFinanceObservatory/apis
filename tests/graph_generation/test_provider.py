"""Tests `GraphGeneration` staleness, single-flight refresh, cache clearing, and stats logging."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from graph_generation.provider import GraphGeneration
from graph_generation.source import GenerationReadResult, GenerationSourceError


class _FakeClock:
    def __init__(self, *, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedSource:
    """Return scripted results in order. Repeat the last result after the list is exhausted."""

    def __init__(self, results: list[GenerationReadResult | GenerationSourceError]) -> None:
        self._results = list(results)
        self._current = self._results[0]
        self.read_calls = 0

    def read(self) -> GenerationReadResult:
        self.read_calls += 1
        if self._results:
            self._current = self._results.pop(0)
        if isinstance(self._current, GenerationSourceError):
            raise self._current
        return self._current


class _RaisingSource:
    """Raise the supplied exception on every `read()` call."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.read_calls = 0

    def read(self) -> GenerationReadResult:
        self.read_calls += 1
        raise self._exc


class _BlockingSource:
    """Signal when `read()` starts, then block until the test sets `release`."""

    def __init__(self, result: GenerationReadResult) -> None:
        self._result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.read_calls = 0

    def read(self) -> GenerationReadResult:
        self.read_calls += 1
        self.started.set()
        assert self.release.wait(timeout=5.0), "test never released the blocked read()"
        return self._result


def _register_counting_cache(provider: GraphGeneration, *, name: str = "cache") -> list[int]:
    clear_calls: list[int] = [0]

    def clear() -> None:
        clear_calls[0] += 1

    provider.register(name, stats=lambda: {"entries": 0}, clear=clear)
    return clear_calls


def test_stale_after_turns_a_known_token_unknown() -> None:
    clock = _FakeClock()
    source = _ScriptedSource([GenerationReadResult(token="rev-a")])
    provider = GraphGeneration(
        source=source, clock=clock, stale_after=10.0, poll_interval=1_000_000.0
    )
    provider.refresh_now()
    assert provider.stamp().token == "rev-a"

    clock.advance(11.0)
    assert provider.stamp().token is None


def test_refresh_is_single_flight_and_stamp_never_blocks() -> None:
    clock = _FakeClock()
    source = _BlockingSource(GenerationReadResult(token="rev-a"))
    provider = GraphGeneration(source=source, clock=clock, poll_interval=1000.0, stale_after=1000.0)

    first = provider.stamp()  # Schedules the single refresh without blocking.
    assert first.token is None
    assert source.started.wait(timeout=5.0)
    assert source.read_calls == 1

    second = provider.stamp()  # Joins the in-flight refresh.
    assert second.token is None
    assert source.read_calls == 1

    source.release.set()
    deadline = time.monotonic() + 5.0
    while provider.stamp().token != "rev-a" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert provider.stamp().token == "rev-a"
    assert source.read_calls == 1


def test_token_change_clears_registered_caches_exactly_once() -> None:
    clock = _FakeClock()
    source = _ScriptedSource(
        [GenerationReadResult(token="rev-a"), GenerationReadResult(token="rev-b")]
    )
    provider = GraphGeneration(source=source, clock=clock, poll_interval=1_000_000.0)
    clear_calls = _register_counting_cache(provider)

    provider.refresh_now()  # The first known token establishes the cache generation.
    assert clear_calls[0] == 0

    provider.refresh_now()  # The token changes from rev-a to rev-b.
    assert clear_calls[0] == 1

    provider.refresh_now()
    assert clear_calls[0] == 1


def test_token_change_contains_one_raising_clear_and_clears_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    source = _ScriptedSource(
        [GenerationReadResult(token="rev-a"), GenerationReadResult(token="rev-b")]
    )
    provider = GraphGeneration(source=source, clock=clock, poll_interval=1_000_000.0)
    good_clear_calls = _register_counting_cache(provider, name="good")

    def raising_clear() -> None:
        raise RuntimeError("boom")

    provider.register("bad", stats=lambda: {"entries": 0}, clear=raising_clear)

    provider.refresh_now()  # The first known token establishes the cache generation.
    with caplog.at_level(logging.INFO):
        provider.refresh_now()  # A token change clears good even when bad raises.

    assert good_clear_calls[0] == 1
    stats_lines = [r.message for r in caplog.records if "registered cache stats" in r.message]
    assert len(stats_lines) == 1  # The stats log survives a failed clear.
    failure_lines = [r.message for r in caplog.records if "clearing cache" in r.message]
    assert len(failure_lines) == 1
    assert "'bad'" in failure_lines[0]


def test_same_token_recovery_clears_nothing() -> None:
    clock = _FakeClock()
    source = _ScriptedSource(
        [GenerationReadResult(token="rev-a"), GenerationReadResult(token="rev-a")]
    )
    provider = GraphGeneration(
        source=source, clock=clock, stale_after=10.0, poll_interval=1_000_000.0
    )
    clear_calls = _register_counting_cache(provider)

    provider.refresh_now()
    assert provider.stamp().token == "rev-a"

    clock.advance(11.0)  # The display expires without another source read.
    assert provider.stamp().token is None

    provider.refresh_now()  # the Admin API answers the same token it always has
    assert provider.stamp().token == "rev-a"
    assert clear_calls[0] == 0


def test_same_token_recovery_logs_unknown_to_known(caplog: pytest.LogCaptureFixture) -> None:
    clock = _FakeClock()
    source = _ScriptedSource(
        [GenerationReadResult(token="rev-a"), GenerationReadResult(token="rev-a")]
    )
    provider = GraphGeneration(
        source=source, clock=clock, stale_after=10.0, poll_interval=1_000_000.0
    )

    provider.refresh_now()
    assert provider.stamp().token == "rev-a"

    clock.advance(11.0)  # The display expires without another source read.
    assert provider.stamp().token is None

    with caplog.at_level(logging.INFO, logger="graph_generation.provider"):
        provider.refresh_now()

    assert provider.stamp().token == "rev-a"
    recovery_lines = [r.message for r in caplog.records if "unknown -> known" in r.message]
    assert len(recovery_lines) == 1


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(GenerationSourceError("boom"), id="generation_source_error"),
        pytest.param(RuntimeError("boom"), id="arbitrary_runtime_error"),
    ],
)
def test_refresh_now_contains_any_source_exception(
    caplog: pytest.LogCaptureFixture, exc: Exception
) -> None:
    clock = _FakeClock()
    source = _RaisingSource(exc)
    provider = GraphGeneration(source=source, clock=clock)

    with caplog.at_level(logging.WARNING, logger="graph_generation.provider"):
        provider.refresh_now()  # The provider contains every source exception type.

    assert provider.stamp().token is None
    assert any("read failed" in r.message for r in caplog.records)


def test_stats_summary_logged_on_token_change(caplog: pytest.LogCaptureFixture) -> None:
    clock = _FakeClock()
    source = _ScriptedSource(
        [GenerationReadResult(token="rev-a"), GenerationReadResult(token="rev-b")]
    )
    provider = GraphGeneration(source=source, clock=clock, poll_interval=1_000_000.0)
    provider.register("cache", stats=lambda: {"hits": 3}, clear=lambda: None)

    with caplog.at_level(logging.INFO, logger="graph_generation.provider"):
        provider.refresh_now()
        caplog.clear()
        provider.refresh_now()  # A token change logs a stats summary.

    stats_lines = [r.message for r in caplog.records if "registered cache stats" in r.message]
    assert len(stats_lines) == 1
    assert "'hits': 3" in stats_lines[0]


def test_register_logs_the_cache_name_and_current_token_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    source = _ScriptedSource([GenerationReadResult(token="rev-a")])
    provider = GraphGeneration(source=source, clock=clock, poll_interval=1_000_000.0)

    with caplog.at_level(logging.INFO, logger="graph_generation.provider"):
        provider.register("early.cache", stats=lambda: {"entries": 0}, clear=lambda: None)

    assert source.read_calls == 0, "register must never perform a source read"
    unknown_lines = [
        r.message for r in caplog.records if "registered cache 'early.cache'" in r.message
    ]
    assert len(unknown_lines) == 1
    assert "token unknown" in unknown_lines[0]

    provider.refresh_now()
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="graph_generation.provider"):
        provider.register("late.cache", stats=lambda: {"entries": 0}, clear=lambda: None)

    known_lines = [
        r.message for r in caplog.records if "registered cache 'late.cache'" in r.message
    ]
    assert len(known_lines) == 1
    assert "token='rev-a'" in known_lines[0]


def test_refresh_if_stale_is_a_no_op_when_fresh() -> None:
    clock = _FakeClock()
    source = _ScriptedSource([GenerationReadResult(token="rev-a")])
    provider = GraphGeneration(source=source, clock=clock, stale_after=300.0, poll_interval=60.0)
    provider.refresh_now()
    assert source.read_calls == 1

    asyncio.run(provider.refresh_if_stale(timeout_s=1.0))
    assert source.read_calls == 1


def test_refresh_if_stale_runs_and_awaits_a_bounded_single_flight_refresh() -> None:
    clock = _FakeClock()
    source = _BlockingSource(GenerationReadResult(token="rev-a"))
    provider = GraphGeneration(source=source, clock=clock, stale_after=1.0, poll_interval=1.0)

    async def scenario() -> None:
        source.release.set()  # Let this read return immediately.
        await provider.refresh_if_stale(timeout_s=1.0)

    asyncio.run(scenario())
    assert provider.stamp().token == "rev-a"
    assert source.read_calls == 1


def test_refresh_if_stale_times_out_without_blocking_forever() -> None:
    clock = _FakeClock()
    source = _BlockingSource(GenerationReadResult(token="rev-a"))
    provider = GraphGeneration(source=source, clock=clock, stale_after=1.0, poll_interval=1.0)

    started = time.monotonic()
    asyncio.run(provider.refresh_if_stale(timeout_s=0.1))
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert provider.stamp().token is None

    source.release.set()  # Finish the worker before the test ends.


def test_refresh_if_stale_concurrent_callers_join_the_same_refresh() -> None:
    clock = _FakeClock()
    source = _BlockingSource(GenerationReadResult(token="rev-a"))
    provider = GraphGeneration(source=source, clock=clock, stale_after=1.0, poll_interval=1.0)

    async def scenario() -> None:
        first = asyncio.create_task(provider.refresh_if_stale(timeout_s=5.0))
        # Wait until the first refresh enters `read()`.
        await asyncio.to_thread(source.started.wait, 5.0)
        second = asyncio.create_task(provider.refresh_if_stale(timeout_s=5.0))
        await asyncio.sleep(0.05)  # Let the second caller run before checking its state.
        assert not first.done()
        assert not second.done(), "second caller must join the in-flight refresh, not skip it"
        assert source.read_calls == 1

        source.release.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())
    assert source.read_calls == 1
    assert provider.stamp().token == "rev-a"


def test_refresh_if_stale_joins_a_refresh_started_by_stamp() -> None:
    clock = _FakeClock()
    source = _BlockingSource(GenerationReadResult(token="rev-a"))
    provider = GraphGeneration(source=source, clock=clock, stale_after=1.0, poll_interval=1.0)

    provider.stamp()  # Unknown token and due poll start the refresh.
    assert source.started.wait(timeout=5.0)
    assert source.read_calls == 1

    started = time.monotonic()
    asyncio.run(provider.refresh_if_stale(timeout_s=0.2))
    elapsed = time.monotonic() - started
    assert elapsed >= 0.2
    assert source.read_calls == 1  # The call joined stamp()'s refresh.

    source.release.set()  # Finish the worker before the test ends.


def test_refresh_if_stale_does_not_retry_a_too_recent_failed_attempt() -> None:
    clock = _FakeClock()
    source = _ScriptedSource([GenerationSourceError("boom")])
    provider = GraphGeneration(source=source, clock=clock, stale_after=1.0, poll_interval=60.0)

    provider.refresh_now()
    assert source.read_calls == 1
    assert provider.stamp().token is None

    asyncio.run(provider.refresh_if_stale(timeout_s=1.0))
    assert source.read_calls == 1
