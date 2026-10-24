"""What a reading holds at each place or role tuple, and whether one reading covers another.

Two decisions remove or replace a reading on the strength of another: the default-reading
collapse drops a member a better-ranked price basis covers
(`retrieval.bound_values.collapse_measure_families`), and the twin diversion serves the
registry's preferred twin in the elected reading's place
(`region.candidate_collapse._preferred_twin`). Both describe each reading here, from the
facets their probes already hold, and both decide with `covers`.

Coverage is shown only from evidence. A facet's first and last years are observed years, and
the years between them are observed only when the facet carries every dated observation. A
comparison the evidence cannot settle answers that the reading is not covered, so the
alternative reading stays.

Pure module: no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from dc_graph import Facet
from search.models import TimeWindow

type CoverageYears = TimeWindow | Literal["latest"] | None
"""The years a question requests: a year range, `"latest"`, or None for an undated
question."""


def _year(date: str | None) -> int | None:
    """The year a date opens with, or None when it names none."""
    return int(date[:4]) if date and date[:4].isdigit() else None


@dataclass(frozen=True, slots=True)
class ObservedYears:
    """The years a reading holds observations in at one place or role tuple.

    `first` and `last` are the earliest and latest years its facets report, each None when
    no facet reports one. `years` is every year it holds a point in when each data-bearing
    facet carries all its dated observations, and None otherwise, when only `first` and
    `last` are known to hold one.
    """

    first: int | None
    last: int | None
    years: frozenset[int] | None

    def known(self) -> frozenset[int]:
        """The years it is shown to hold a point in."""
        if self.years is not None:
            return self.years
        return frozenset(year for year in (self.first, self.last) if year is not None)

    def requested(self, window: TimeWindow) -> frozenset[int] | None:
        """The years inside `window` it may hold a point in, or None when they are unbounded:
        its years there when complete, else every year its span shares with `window`."""
        if self.years is not None:
            return frozenset(year for year in self.years if _inside(year, window))
        if self.first is None or self.last is None:
            return None
        start = self.first if window.start_year is None else max(self.first, window.start_year)
        end = self.last if window.end_year is None else min(self.last, window.end_year)
        return frozenset(range(start, end + 1))


def _inside(year: int, window: TimeWindow) -> bool:
    return (window.start_year is None or year >= window.start_year) and (
        window.end_year is None or year <= window.end_year
    )


type ReadingCoverage = Mapping[str, ObservedYears]
"""Per place, or per role tuple's joined dcids, the years a reading holds there. A key where
it holds nothing is absent."""


def holds_in(coverage: ReadingCoverage, window: TimeWindow | None) -> bool:
    """Whether a reading holds a point at some key of `coverage`, inside `window` when one is
    given, or may: a span stated without its years counts when it overlaps the window."""
    return any(
        window is None or years.requested(window) != frozenset() for years in coverage.values()
    )


def observed_years(facets: Iterable[Facet]) -> ObservedYears | None:
    """What `facets` hold at one place or role tuple, or None when none holds an
    observation. A data-bearing facet without a complete dated read
    (`Facet.dates_complete`), or with a date naming no year, leaves `years` None."""
    held = [f for f in facets if f.obs_count > 0]
    if not held:
        return None
    dated = [_year(d) for f in held for d in f.dates]
    complete = all(f.dates and f.dates_complete for f in held) and None not in dated
    firsts = [y for f in held if (y := _year(f.earliest_date)) is not None]
    lasts = [y for f in held if (y := _year(f.latest_date)) is not None]
    years = frozenset(y for y in dated if y is not None)
    return ObservedYears(
        first=min([*firsts, *years], default=None),
        last=max([*lasts, *years], default=None),
        years=years if complete else None,
    )


def _covers_at(a: ObservedYears | None, b: ObservedYears, window: CoverageYears) -> bool:
    """Whether `a` (None: nothing there) is shown to hold every requested year `b` may hold
    at one key."""
    if window is None:
        return a is not None
    if window == "latest":
        return a is not None and a.last is not None and b.last is not None and a.last >= b.last
    needed = b.requested(window)
    if needed is None:
        return False
    if not needed:
        return True
    return a is not None and needed <= a.known()


def covers(a: ReadingCoverage | None, b: ReadingCoverage | None, window: CoverageYears) -> bool:
    """Whether reading `a` stands in for `b` under `window`: at every key `b` holds data,
    `a` is shown to hold every requested year `b` may hold there. Undated, `a` need only
    hold data at each such key; under `"latest"` its last year there must be no earlier
    than `b`'s. A `b` holding nothing at any key is covered. Otherwise unknown coverage on
    either side covers nothing."""
    if b is None:
        return False
    if not b:
        return True
    if a is None:
        return False
    return all(_covers_at(a.get(key), years, window) for key, years in b.items())
