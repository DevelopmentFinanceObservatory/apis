"""The deadline a graph read that fails open receives (`degraded_reads.fail_open_deadline`).

A fake client records the deadline it is given, and `assert_fail_open_deadline` checks it
against the request's bound. The three requests span the cases: no bound, a bound far past
the budget, and a bound the budget outlasts.
"""

from __future__ import annotations

import time

import pytest

from search.engine.config import settings

FAIL_OPEN_REQUESTS = pytest.mark.parametrize(
    "request_in_s", [None, 60.0, 0.5], ids=["unbound", "far", "near"]
)


def request_deadline(request_in_s: float | None) -> float | None:
    """The request's monotonic deadline `request_in_s` from now, None for no bound."""
    return None if request_in_s is None else time.monotonic() + request_in_s


def assert_fail_open_deadline(
    deadline: float | None, *, requested: float | None, started: float
) -> None:
    """`deadline`, recorded at the read, is the request's `requested` bound when that ends
    within the budget of `started` (a `time.monotonic()` taken before the call), else a
    budget from the moment of the read, so no earlier than `started` plus the budget and no
    later than now plus it."""
    assert deadline is not None, "the read ran with no deadline"
    budget = settings().fail_open_read_budget_s
    if requested is not None and requested < started + budget:
        assert deadline == requested
    else:
        assert started + budget <= deadline <= time.monotonic() + budget
