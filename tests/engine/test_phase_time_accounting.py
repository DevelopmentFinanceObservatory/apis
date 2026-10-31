"""A part's booked steps account for the wall time its resolve took."""

from __future__ import annotations

import functools
import inspect
from types import SimpleNamespace

import pytest

from search.engine import envelope
from search.engine.conjoin import NESTED_SHAPE_STEPS
from search.engine.region import phase_timing, pipeline
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_candidates_unit_election_e2e import _run
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

# Every public call on a fake client takes this long on the fake clock.
_CALL_MS = 5


def _install_fixed_latency(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Run `envelope.now_ms` on a clock that moves `_CALL_MS` per fake-client call and
    return that clock, in ms, as a one-cell list."""
    clock = [0]
    monkeypatch.setattr(envelope, "time", SimpleNamespace(monotonic=lambda: clock[0] / 1000))

    def timed(method):
        @functools.wraps(method)
        def call(*args, **kwargs):
            clock[0] += _CALL_MS
            return method(*args, **kwargs)

        return call

    for owner in (FakeGraph, FakeGraphReader, FakeSdmxClient):
        for name, method in list(vars(owner).items()):
            if not name.startswith("_") and inspect.isfunction(method):
                monkeypatch.setattr(owner, name, timed(method))
    return clock


def test_booked_steps_sum_to_the_wall_time_of_the_resolve(monkeypatch):
    clock = _install_fixed_latency(monkeypatch)

    result = _run()

    by_step = result.timing_by_step
    booked = sum(ms for step, ms in by_step.items() if step not in NESTED_SHAPE_STEPS)
    assert abs(booked - clock[0]) <= 0.05 * clock[0], (booked, clock[0], by_step)
    assert by_step["phase_place_bindings"] > 0, "a phase that books nothing carries its time"
    assert {f"phase_{phase.__name__.removeprefix('_phase_')}" for phase in pipeline._PHASES} <= set(
        by_step
    )


def test_a_step_that_began_in_an_earlier_phase_takes_that_phase_s_time_back():
    """`shape` begins in the derive phase and ends in the rank phase, so the derive
    phase's unbooked time falls out of `phase_derive_shapes`."""
    state = SimpleNamespace(timing={"shape": 0})
    unbooked: dict[str, int] = {}

    phase_timing.book_unbooked_time(state, unbooked, "phase_derive_shapes", elapsed=30, booked=0)
    state.timing["shape"] = 50
    phase_timing.book_unbooked_time(state, unbooked, "phase_prune_and_rank", elapsed=40, booked=50)

    assert unbooked == {"phase_derive_shapes": 20, "phase_prune_and_rank": 0}
    assert state.timing["phase_derive_shapes"] + state.timing["shape"] == 70


def test_the_stages_inside_shape_book_no_phase_time():
    """`shape_arcs` and the other stages inside `shape` run within the time `shape` books,
    so a phase's booked time leaves them out and the phases before it keep theirs."""
    state = SimpleNamespace(timing={"phase_recall": 5, "shape": 30, "shape_arcs": 12})

    assert phase_timing.booked_ms(state) == 35
