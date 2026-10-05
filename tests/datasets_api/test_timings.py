"""Pins timing accumulation, collector cleanup, and `server_timing`/log rendering with a scripted clock."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from datasets_api import timings
from datasets_api.timings import PhaseTimings, collecting, phase, record_elapsed
from dfo_platform import timings as platform_timings
from dfo_platform.trace import reset_trace_id, set_trace_id


class _FakeClock:
    """Returns each tick in `ticks`, in order, one per call."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = iter(ticks)

    def __call__(self) -> float:
        return next(self._ticks)


def _install_clock(monkeypatch: pytest.MonkeyPatch, ticks: list[float]) -> None:
    monkeypatch.setattr(timings.time, "perf_counter", _FakeClock(ticks))


def test_record_accumulates_repeated_calls_with_the_same_name():
    t = PhaseTimings()

    t.record("render", 12.4)
    t.record("render", 3.6)

    assert t.snapshot() == {"render": 16.0}


def test_phase_records_duration_onto_the_bound_collector(monkeypatch: pytest.MonkeyPatch):
    _install_clock(monkeypatch, [0.0, 0.0104])
    t = PhaseTimings()

    with collecting(t), phase("resolve"):
        pass

    assert t.snapshot() == pytest.approx({"resolve": 10.4})


def test_phase_accumulates_across_repeated_calls(monkeypatch: pytest.MonkeyPatch):
    _install_clock(monkeypatch, [0.0, 0.010, 0.050, 0.075])
    t = PhaseTimings()

    with collecting(t):
        with phase("resolve"):
            pass
        with phase("resolve"):
            pass

    assert t.snapshot() == pytest.approx({"resolve": 35.0})


def test_phase_records_even_when_the_wrapped_block_raises(monkeypatch: pytest.MonkeyPatch):
    _install_clock(monkeypatch, [0.0, 0.020])
    t = PhaseTimings()

    with collecting(t), pytest.raises(ValueError, match="boom"), phase("materialise"):
        raise ValueError("boom")

    assert t.snapshot() == pytest.approx({"materialise": 20.0})


def test_phase_is_a_noop_when_no_collector_is_bound(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(timings.time, "perf_counter", lambda: calls.append(1) or 0.0)

    with phase("resolve"):
        pass

    assert calls == []


def test_record_elapsed_converts_seconds_to_milliseconds_onto_the_bound_collector():
    t = PhaseTimings()

    with collecting(t):
        record_elapsed("profile", 0.0104)

    assert t.snapshot() == pytest.approx({"profile": 10.4})


def test_record_elapsed_accumulates_across_repeated_calls():
    t = PhaseTimings()

    with collecting(t):
        record_elapsed("profile", 0.010)
        record_elapsed("profile", 0.025)

    assert t.snapshot() == pytest.approx({"profile": 35.0})


def test_record_elapsed_is_a_noop_when_no_collector_is_bound():
    # No `collecting()` block is bound and no `PhaseTimings` is passed in.
    # With no collector, `record_elapsed` is a no-op.
    record_elapsed("profile", 0.5)


def test_collecting_resets_the_binding_on_exit():
    t = PhaseTimings()

    with collecting(t):
        assert platform_timings._current.get() is t

    assert platform_timings._current.get() is None


def test_collecting_resets_the_binding_on_exit_even_when_the_block_raises():
    t = PhaseTimings()

    with pytest.raises(ValueError, match="boom"), collecting(t):
        raise ValueError("boom")

    assert platform_timings._current.get() is None


def test_snapshot_returns_an_independent_copy():
    t = PhaseTimings()
    t.record("resolve", 1.0)

    snap = t.snapshot()
    snap["resolve"] = 999.0
    t.record("resolve", 1.0)

    assert t.snapshot()["resolve"] == pytest.approx(2.0)


def test_server_timing_puts_total_first_then_phases_in_insertion_order():
    t = PhaseTimings()
    t.record("resolve", 12.44)
    t.record("materialise", 812.0)

    assert t.server_timing(total_ms=903.06) == (
        "total;dur=903.1, resolve;dur=12.4, materialise;dur=812.0"
    )


def test_server_timing_rounds_to_one_decimal():
    t = PhaseTimings()
    t.record("resolve", 12.449)

    assert t.server_timing(total_ms=0.0) == "total;dur=0.0, resolve;dur=12.4"


def test_as_log_field_format():
    t = PhaseTimings()
    t.record("resolve", 12.4)
    t.record("materialise", 812.0)

    assert t.as_log_field() == "resolve=12.4 materialise=812.0"


def test_as_log_field_empty_when_nothing_recorded():
    assert PhaseTimings().as_log_field() == ""


def test_phase_inside_a_worker_thread_is_a_noop_even_while_the_caller_thread_is_collecting():
    """A worker thread has its own context, so it cannot see the caller's bound collector."""
    t = PhaseTimings()

    def _in_worker() -> None:
        with phase("materialise"):
            pass

    with collecting(t), ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_in_worker).result()

    assert t.snapshot() == {}


def test_nested_and_interleaved_phases_stay_distinct(monkeypatch: pytest.MonkeyPatch):
    # The nested phases consume ticks on entry and exit. A second `resolve` follows the first.
    _install_clock(monkeypatch, [0.000, 0.001, 0.003, 0.006, 0.010, 0.015])
    t = PhaseTimings()

    with collecting(t):
        with phase("resolve"), phase("materialise"):
            pass
        with phase("resolve"):
            pass

    assert t.snapshot() == pytest.approx({"resolve": 11.0, "materialise": 2.0})


def test_run_with_timings_binds_timings_for_the_duration_of_the_call(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_clock(monkeypatch, [0.0, 0.010])
    t = PhaseTimings()

    def _inner(*, value: str) -> str:
        with phase("resolve"):
            pass
        return value

    result = timings.run_with_timings(_inner, t, value="ok")

    assert result == "ok"
    assert t.snapshot() == pytest.approx({"resolve": 10.0})


def test_run_with_timings_unbinds_the_collector_on_exit_even_when_fn_raises():
    t = PhaseTimings()

    def _inner() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        timings.run_with_timings(_inner, t)

    assert platform_timings._current.get() is None


def test_door_call_yields_timings_and_renders_server_timing_only_after_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_clock(monkeypatch, [0.0, 0.0125])
    token = set_trace_id("trace-abc")
    try:
        with timings.door_call("widget", logger=logging.getLogger("test.door_call")) as call:
            call.timings.record("resolve", 4.0)
            assert call.server_timing == ""
    finally:
        reset_trace_id(token)

    assert call.server_timing == "total;dur=12.5, resolve;dur=4.0"


def test_door_call_logs_one_datasets_timings_line_through_the_given_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _install_clock(monkeypatch, [0.0, 0.0125])
    token = set_trace_id("trace-abc")
    try:
        with (
            caplog.at_level(logging.INFO, logger="test.door_call"),
            timings.door_call("widget", logger=logging.getLogger("test.door_call")) as call,
        ):
            call.timings.record("resolve", 4.0)
    finally:
        reset_trace_id(token)

    [record] = [r for r in caplog.records if r.message.startswith("datasets.timings")]
    assert record.name == "test.door_call"
    assert record.message == (
        "datasets.timings door=widget trace_id=trace-abc outcome=ok total_ms=12.5 resolve=4.0"
    )


def test_door_call_records_the_raised_exceptions_class_name_as_outcome_and_reraises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _install_clock(monkeypatch, [0.0, 0.005])
    token = set_trace_id("trace-err")
    try:
        with (
            caplog.at_level(logging.INFO, logger="test.door_call"),
            pytest.raises(ValueError, match="boom"),
            timings.door_call("widget", logger=logging.getLogger("test.door_call")) as call,
        ):
            raise ValueError("boom")
    finally:
        reset_trace_id(token)

    assert call.server_timing == "total;dur=5.0"
    [record] = [r for r in caplog.records if r.message.startswith("datasets.timings")]
    assert "outcome=ValueError" in record.message
