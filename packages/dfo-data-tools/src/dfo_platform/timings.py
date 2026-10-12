"""Collect named phase timings, labels and counts for one API call through a context variable.

The lock protects writes from worker threads while the event loop reads
snapshots. Callers bind the collector inside a worker with
`run_with_timings`, and a pool that does not carry the caller's context binds it
with `carrying_timings`. Overlapping phases accumulate by name and can exceed
the call's wall-clock total. Store layers such as `dataset_store` record
phases here, and the graph and SDMX clients count their requests (`count`), without
depending on the surface that renders them.
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
    """Per-phase wall-clock totals in milliseconds, named labels describing the call, and
    named counts. Protected across worker and event-loop threads.
    """

    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    _totals: dict[str, float] = field(default_factory=dict)
    _labels: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def record(self, name: str, milliseconds: float) -> None:
        """Accumulate `milliseconds` onto `name`, so repeated calls to one phase sum into one
        figure."""
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + milliseconds

    def label(self, name: str, value: str) -> None:
        """Set `name` to `value`. A later call for one name replaces the earlier value."""
        with self._lock:
            self._labels[name] = value

    def count(self, name: str, n: int = 1) -> None:
        """Add `n` to the count `name`, starting it at zero. `n` 0 makes the count appear."""
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + n

    def snapshot(self) -> dict[str, float]:
        """An independent copy of the totals recorded so far."""
        with self._lock:
            return dict(self._totals)

    def count_snapshot(self) -> dict[str, int]:
        """An independent copy of the counts recorded so far, in the order each started."""
        with self._lock:
            return dict(self._counts)

    def server_timing(self, *, total_ms: float) -> str:
        """Render as a `Server-Timing` header value: `total` first, then
        each recorded phase in the order it was first recorded."""
        metrics = [f"total;dur={total_ms:.1f}"]
        for name, ms in self.snapshot().items():
            metrics.append(f"{name};dur={ms:.1f}")
        return ", ".join(metrics)

    def as_log_field(self) -> str:
        """Render as `label=value ... count=n ... name=ms ...` for the door's log line. Labels
        and counts stay out of `Server-Timing`."""
        with self._lock:
            labels = dict(self._labels)
            counts = dict(self._counts)
        return " ".join(
            [
                *(f"{name}={value}" for name, value in labels.items()),
                *(f"{name}={n}" for name, n in counts.items()),
                *(f"{name}={ms:.1f}" for name, ms in self.snapshot().items()),
            ]
        )


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


def label(name: str, value: str) -> None:
    """Set a label on the current collector. Return without recording when none is bound."""
    timings = _current.get()
    if timings is None:
        return
    timings.label(name, value)


def count(name: str, n: int = 1) -> None:
    """Add `n` to a count on the current collector. Return without recording when none is
    bound."""
    timings = _current.get()
    if timings is None:
        return
    timings.count(name, n)


def carrying_timings[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """`fn`, run with the calling thread's collector bound, for a pool that does not carry the
    caller's context. `fn` itself when none is bound."""
    timings = _current.get()
    if timings is None:
        return fn

    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        with collecting(timings):
            return fn(*args, **kwargs)

    return run


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


__all__ = [
    "PhaseTimings",
    "carrying_timings",
    "collecting",
    "count",
    "label",
    "phase",
    "record_elapsed",
    "run_with_timings",
]
