"""Book a pass's phase time that its own steps left unbooked, so a part's `by_step`
accounts for the wall time its resolve took."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from search.engine.conjoin import NESTED_SHAPE_STEPS
from search.engine.envelope import now_ms


class Timed(Protocol):
    """A pass's state, whose `timing` maps each step it booked to its ms."""

    timing: dict[str, int]


def booked_ms(state: Timed) -> int:
    """The ms `state.timing` books, leaving out the stages inside `shape`
    (`conjoin.NESTED_SHAPE_STEPS`), since `shape` already spans them."""
    return sum(ms for step, ms in state.timing.items() if step not in NESTED_SHAPE_STEPS)


def book_unbooked_time(
    state: Timed,
    unbooked: dict[str, int],
    step: str,
    *,
    elapsed: int,
    booked: int,
) -> None:
    """Book under `step` the part of a phase's `elapsed` ms that its own steps (`booked`)
    left out, so no millisecond is counted twice. A step that books from before its phase
    began (`shape` starts in `_phase_derive_shapes` and ends in `_phase_prune_and_rank`)
    books more than the phase ran, and the excess comes off the latest earlier phases
    already booked in `unbooked`, whose entries `state.timing` mirrors."""
    excess = booked - elapsed
    for earlier in reversed(unbooked):
        if excess <= 0:
            break
        taken = min(unbooked[earlier], excess)
        unbooked[earlier] -= taken
        state.timing[earlier] -= taken
        excess -= taken
    unbooked[step] = state.timing[step] = max(0, elapsed - booked)


@contextmanager
def timed_step(state: Timed, step: str) -> Iterator[None]:
    """Book the block's wall time under `step` in `state.timing`, however it exits."""
    t0 = now_ms()
    try:
        yield
    finally:
        state.timing[step] = now_ms() - t0
