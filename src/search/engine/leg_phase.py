"""The pipeline phase a fan-out leg is running, for the soft-deadline log.

`_run_variables_streaming` gives each leg a `LegPhase` and the leg binds it
with `track_leg_phase` inside its own task. The pipeline names each step with
`enter_phase`, and when the soft deadline fires the fan-out logs the name for
every leg that has not finished. Outside a tracked leg, `enter_phase` does
nothing.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(slots=True)
class LegPhase:
    """The step a leg is in. `queued` until the leg starts."""

    name: str = "queued"


_current: contextvars.ContextVar[LegPhase | None] = contextvars.ContextVar(
    "search_leg_phase", default=None
)


def track_leg_phase(phase: LegPhase) -> None:
    """Bind `phase` for the calling task and the tasks and threads it starts."""
    _current.set(phase)


def enter_phase(name: str) -> None:
    """Record that the current leg has entered step `name`."""
    phase = _current.get()
    if phase is not None:
        phase.name = name
