"""Collect named phase timings for one API call through a context variable.

The lock protects writes from worker threads while the event loop reads
snapshots. Callers bind the collector inside a worker with
`run_with_timings`. Overlapping phases accumulate by name and can exceed
the call's wall-clock total. Store layers such as `dataset_store` record
phases here without depending on the surface that renders them.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

_current: contextvars.ContextVar[PhaseTimings | None] = contextvars.ContextVar(
    "dfo_phase_timings", default=None
)


@dataclass
class PhaseTimings:
    """Per-phase wall-clock totals in milliseconds.
    Protected across worker and event-loop threads.
    """

    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    _totals: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, milliseconds: float) -> None:
        """Accumulate `milliseconds` onto `name`, so a collection's repeated
        per-member `render`/`store`/`sign` calls sum into one figure."""
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + milliseconds

    def snapshot(self) -> dict[str, float]:
        """An independent copy of the totals recorded so far."""
        with self._lock:
            return dict(self._totals)

    def server_timing(self, *, total_ms: float) -> str:
        """Render as a `Server-Timing` header value: `total` first, then
        each recorded phase in the order it was first recorded."""
        metrics = [f"total;dur={total_ms:.1f}"]
        for name, ms in self.snapshot().items():
            metrics.append(f"{name};dur={ms:.1f}")
        return ", ".join(metrics)

    def as_log_field(self) -> str:
        """Render as `name=ms name=ms ...` for the door's log line."""
        return " ".join(f"{name}={ms:.1f}" for name, ms in self.snapshot().items())


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Accumulate this block's duration under `name` on the current call's
    collector. If none is bound, yield without recording.
    """
    timings = _current.get()
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings.record(name, (time.perf_counter() - start) * 1000)


def record_elapsed(name: str, seconds: float) -> None:
    """Add a duration measured elsewhere, in seconds, to the current collector.
    Return without recording when none is bound. Use this for a span measured before
    this call, such as the profile time returned by `materialise`.
    """
    timings = _current.get()
    if timings is None:
        return
    timings.record(name, seconds * 1000)


@contextmanager
def collecting(timings: PhaseTimings) -> Iterator[None]:
    """Bind `timings` inside the executor worker. Reset it on exit before the pooled
    thread takes another task.
    """
    token = _current.set(timings)
    try:
        yield
    finally:
        _current.reset(token)


def run_with_timings[T](fn: Callable[..., T], timings: PhaseTimings, /, **kwargs: object) -> T:
    """Bind `timings` on this thread while calling `fn(**kwargs)`.

    Executor workers use their own context, so doors bind the collector here.
    """
    with collecting(timings):
        return fn(**kwargs)


__all__ = ["PhaseTimings", "collecting", "phase", "record_elapsed", "run_with_timings"]
