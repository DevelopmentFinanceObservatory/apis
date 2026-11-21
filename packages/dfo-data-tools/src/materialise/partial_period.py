"""Whether a frame row's period reaches the period its source's import holds only part of.

The pipeline marks a source's partial period at its own grain (`partialObservationDate`:
`2025`, or `2025-06` for a month). A row reaches it when the period the row's `time_period`
cell ends in holds any part of the mark or a later time: a year 2025 row reaches a `2025-06`
mark, a `2025-03` row does not, and a `2020/2025` row reaches a `2025` mark.
"""

from __future__ import annotations

from dfo_frame_grammar import FrameError
from dfo_frame_grammar.grammar import parse_time_period

__all__ = ["reaches", "start_of"]

# The last day of a period at each grain, appended to its end point, so a coarser period
# compares by its last day.
_LAST_DAY = {4: "-12-31", 7: "-31", 10: ""}


def reaches(period: str, mark: str) -> bool | None:
    """Whether the `time_period` cell `period` ends at or after the start of `mark`, a period
    at its own grain. `None` when the frame grammar reads either not at all."""
    try:
        end = parse_time_period(period).end
        start = parse_time_period(mark).start
    except FrameError:
        return None
    last_day = end + _LAST_DAY[len(end)]
    return last_day[: len(start)] >= start


def start_of(mark: str) -> str:
    """The first day `mark` holds at its own grain (`2025-06` for `2025-06`, `2025` for
    `2025`), so marks of different grains order by when they start; `mark` itself when the
    frame grammar cannot read it."""
    try:
        return parse_time_period(mark).start
    except FrameError:
        return mark
