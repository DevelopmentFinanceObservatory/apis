"""Render a door call's phase timings as `Server-Timing` and a log line.

The collector lives in `dfo_platform.timings`, so store layers below this
surface can record phases. `door_call` measures the call's total and renders
the `Server-Timing` header and log line. The header carries each phase as
`name;dur=ms` and each count, such as `db_transactions` or `sdmx_requests`, as
`name;desc=n`, so a client such as `scripts/bench_datasets.py` reads a
request's counts from its response. A door that serves data logs its
`datasets.delivery` line (`datasets_api.delivery`) through `DoorCall.delivered`
once it has built its response.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

from datasets_api.delivery import Delivery, delivering, log_delivery
from dfo_platform.timings import (
    PhaseTimings,
    collecting,
    count,
    label,
    phase,
    record_elapsed,
    run_with_timings,
)
from dfo_platform.trace import get_trace_id


@dataclass
class DoorCall:
    """Timing state for one endpoint call to `door`.

    `server_timing` is set on exit, once total time is known. `delivery` is what a door opened
    with `delivers=True` served, `None` on any other door.
    """

    door: str
    timings: PhaseTimings
    server_timing: str = ""
    delivery: Delivery | None = None

    def delivered[R](self, response: R) -> R:
        """Log the call's `datasets.delivery` line, when it delivers, and return `response`.
        A door passes the response it built, so a call that fails building one logs no line."""
        if self.delivery is not None:
            log_delivery(self.door, self.delivery)
        return response


@contextmanager
def door_call(door: str, *, logger: logging.Logger, delivers: bool = False) -> Iterator[DoorCall]:
    """Time a door call and log its outcome and phases.

    Read `call.timings` inside the block. Read `call.server_timing` after it
    exits, when this context manager has measured the total time. With
    `delivers`, the block runs with a `Delivery` bound (`datasets_api.delivery`),
    and the door logs its line with `call.delivered` once its response is built.
    """
    trace_id = get_trace_id()
    call = DoorCall(door=door, timings=PhaseTimings())
    start = time.perf_counter()
    outcome = "ok"
    try:
        with ExitStack() as stack:
            if delivers:
                call.delivery = stack.enter_context(delivering())
            yield call
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        total_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "datasets.timings door=%s trace_id=%s outcome=%s total_ms=%.1f %s",
            door,
            trace_id,
            outcome,
            total_ms,
            call.timings.as_log_field(),
        )
        call.server_timing = ", ".join(
            [
                call.timings.server_timing(total_ms=total_ms),
                *(f"{name};desc={n}" for name, n in call.timings.count_snapshot().items()),
            ]
        )


__all__ = [
    "DoorCall",
    "PhaseTimings",
    "collecting",
    "count",
    "door_call",
    "label",
    "phase",
    "record_elapsed",
    "run_with_timings",
]
