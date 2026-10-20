"""`admission.RebuildFlights`: how long a caller waits on a running rebuild, and what a rebuild
that raises, or a waiter that gives up, leaves for the others."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

from datasets_api.admission import RebuildFlights
from datasets_api.doors.mint_errors import MintBackpressure

_SUBJECT = "spec_" + "a" * 32
_SHORT = 0.05


class _Rebuild:
    """A caller holding its turn on another thread, under the test's bound release as a request
    runs, until `finish` is set. `raising` is what its rebuild raises on the way out."""

    def __init__(
        self,
        flights: RebuildFlights,
        *,
        deadline: float | None = None,
        raising: BaseException | None = None,
    ) -> None:
        self.entered = threading.Event()
        self.finish = threading.Event()
        self.error: BaseException | None = None
        self._flights = flights
        self._deadline = deadline
        self._raising = raising
        self._thread = threading.Thread(target=contextvars.copy_context().run, args=(self._run,))
        self._thread.start()

    def _run(self) -> None:
        try:
            with self._flights.turn(_SUBJECT, deadline=self._deadline, wait_seconds=_SHORT):
                self.entered.set()
                assert self.finish.wait(timeout=5.0)
                if self._raising is not None:
                    raise self._raising
        except BaseException as exc:
            self.error = exc

    def end(self) -> None:
        self.finish.set()
        self._thread.join(timeout=5.0)
        assert not self._thread.is_alive()


def test_a_waiter_waits_until_its_own_deadline_not_the_materialise_wait():
    flights = RebuildFlights()
    leader = _Rebuild(flights)
    assert leader.entered.wait(timeout=5.0)
    waiter = _Rebuild(flights, deadline=time.monotonic() + 5.0)

    # Longer than `wait_seconds`, which bounds only a caller with no deadline.
    assert not waiter.entered.wait(timeout=4 * _SHORT)
    leader.end()

    assert waiter.entered.wait(timeout=5.0)
    waiter.end()
    assert (leader.error, waiter.error) == (None, None)


@pytest.mark.parametrize("deadline_in", [None, _SHORT], ids=["no-deadline", "deadline-passed"])
def test_a_waiter_that_gives_up_leaves_the_rebuild_running_for_the_others(
    deadline_in: float | None,
):
    flights = RebuildFlights()
    leader = _Rebuild(flights)
    assert leader.entered.wait(timeout=5.0)
    deadline = None if deadline_in is None else time.monotonic() + deadline_in

    with (
        pytest.raises(MintBackpressure),
        flights.turn(_SUBJECT, deadline=deadline, wait_seconds=_SHORT),
    ):
        pytest.fail("entered while the rebuild was running")

    # The rebuild is still the one a later caller waits on, and it ends normally.
    later = _Rebuild(flights, deadline=time.monotonic() + 5.0)
    assert not later.entered.wait(timeout=2 * _SHORT)
    leader.end()
    assert later.entered.wait(timeout=5.0)
    later.end()
    assert (leader.error, later.error) == (None, None)


@pytest.mark.parametrize(
    "raised",
    [RuntimeError("upstream fell over"), asyncio.CancelledError()],
    ids=["failed", "cancelled"],
)
def test_a_rebuild_that_raises_strands_no_waiter_and_the_first_one_takes_its_place(
    raised: BaseException,
):
    flights = RebuildFlights()
    leader = _Rebuild(flights, raising=raised)
    assert leader.entered.wait(timeout=5.0)
    deadline = time.monotonic() + 5.0
    first, second = _Rebuild(flights, deadline=deadline), _Rebuild(flights, deadline=deadline)
    assert not first.entered.wait(timeout=_SHORT)

    leader.end()

    # Both waiters enter together: each rebuilds as it would have alone.
    assert first.entered.wait(timeout=5.0)
    assert second.entered.wait(timeout=5.0)
    assert leader.error is raised
    # One of them is the rebuild a later caller waits on.
    later = _Rebuild(flights, deadline=deadline)
    assert not later.entered.wait(timeout=_SHORT)
    first.end()
    second.end()
    assert later.entered.wait(timeout=5.0)
    later.end()
    assert (first.error, second.error, later.error) == (None, None, None)
