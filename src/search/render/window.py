"""Period phrasing for a heading."""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from search.models import ResolvedStatVar, TimeWindow

# Supported facet date precisions, tried from least to most specific.
_RESOLVED_DATE_FORMATS = ("%Y", "%Y-%m", "%Y-%m-%d")


def _format_resolved_date(text: str) -> str:
    """Format ISO year, month, or day precision as prose (for example, `2025`, `June 2025`,
    or `2 April 2006`). Month names use locale-independent `calendar.month_name`. Unknown
    formats pass through unchanged.
    """
    for fmt in _RESOLVED_DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%Y":
            return text
        if fmt == "%Y-%m":
            return f"{calendar.month_name[parsed.month]} {parsed.year}"
        return f"{parsed.day} {calendar.month_name[parsed.month]} {parsed.year}"
    return text


def _served_bounds(stat_vars: Sequence[ResolvedStatVar]) -> tuple[str | None, str | None]:
    """The earliest start and latest end over the StatVars' served ranges, each `None` when
    no range states that bound. ISO dates of one precision order as strings."""
    starts = [sv.data_date_range.start for sv in stat_vars if sv.data_date_range is not None]
    ends = [sv.data_date_range.end for sv in stat_vars if sv.data_date_range is not None]
    return min(filter(None, starts), default=None), max(filter(None, ends), default=None)


PeriodKind = Literal["in", "span", "since", "until"]


@dataclass(frozen=True, slots=True)
class Period:
    """The span a heading states, its bounds as ISO dates of the precision served.

    ``kind`` "in" is one date (``start`` equals ``end``), "span" two, "since" a start
    alone and "until" an end alone.
    """

    kind: PeriodKind
    start: str | None
    end: str | None

    @property
    def text(self) -> str:
        """The period as a heading reads it: "in 2024", "1973 to 2024", "since 1973",
        "until 2024", each bound through `_format_resolved_date`."""
        start = _format_resolved_date(self.start) if self.start else ""
        end = _format_resolved_date(self.end) if self.end else ""
        match self.kind:
            case "in":
                return f"in {end}"
            case "span":
                return f"{start} to {end}"
            case "since":
                return f"since {start}"
            case "until":
                return f"until {end}"


def served_period(
    stat_vars: Sequence[ResolvedStatVar] | None, window: TimeWindow | None
) -> Period | None:
    """The span the answer serves as the period a heading states, or None when unknown.

    Each StatVar's range holds the dates the answer's read plan serves for it
    (`read_plan.served_dates`), inside the years a request named and all years when it
    named none, so the span runs from the earliest start to the latest end. With no
    served range the coverage ``window`` states the period in years.
    """
    start, end = _served_bounds(stat_vars or ())
    if start is None and end is None and window is not None:
        start = str(window.start_year) if window.start_year is not None else None
        end = str(window.end_year) if window.end_year is not None else None
    if start and end:
        return Period("in" if start == end else "span", start, end)
    if start:
        return Period("since", start, None)
    return Period("until", None, end) if end else None
