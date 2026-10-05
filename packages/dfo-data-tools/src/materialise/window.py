"""Filters materialised rows by year and records the applied window.

`dc_sdmx` `/data` returns unwindowed rows, so `columnar` filters them
before frame construction. This module mirrors `search.models.in_window`
locally to keep `materialise` independent of Search. Unparseable periods
remain included.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class RequestedWindow:
    """The years a request elects its facets under and reads its rows in.

    `end_unstated` marks a range whose definition names no end year: `end_year` is then the
    year `election_window` ends it at. Election and the row filter read `end_year` alike.
    The applied window reads the mark, so data ending before `end_year` is the whole of
    what the definition asked for (`compute_applied_window`).
    """

    kind: Literal["all", "latest", "range"]
    start_year: int | None = None
    end_year: int | None = None
    end_unstated: bool = False


def current_year() -> int:
    """The calendar year an open-ended range ends at (`election_window`)."""
    return date.today().year


def election_window(
    window: RequestedWindow, *, clock_year: int | None = None, last_year: int | None = None
) -> RequestedWindow:
    """The window a request with `window` elects its facets under and reads its rows in.

    A range with an open end means recorded data: it ends at the current year
    (`clock_year`, or `current_year()` when None), so a facet holding only projection years
    does not lead a request that names no end, and no later year is read. A range that
    starts after the current year asks about the future, so it ends at `last_year`, the
    last year its caller's coverage holds, or stays open without one. A closed range,
    `latest` and `all` are kept, so a closed range past the current year elects and reads
    the facet holding those years.

    Each door applies it once, where it turns a requested window into the window it reads:
    the datasets adapter when it builds a request from an identity
    (`datasets_api.adapter.to_materialisation_request`), and Search when it builds a read
    plan's years (`search.engine.retrieval.read_plan.read_range`). An answer and a dataset
    minted from its Spec therefore elect and read under the same years. An identity keeps
    the open end it was given, and the returned window marks its end as unstated.
    """
    if window.kind != "range" or window.end_year is not None:
        return window
    now = clock_year if clock_year is not None else current_year()
    if window.start_year is not None and window.start_year > now:
        return replace(window, end_year=last_year, end_unstated=True)
    return replace(window, end_year=now, end_unstated=True)


@dataclass(frozen=True)
class AppliedWindow:
    kind: str
    start_year: int | None
    end_year: int | None
    origin: Literal["query", "default", "coverage_clamp"]

    def as_trace_entry(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "origin": self.origin,
        }


def _leading_year(time_period: str | None) -> int | None:
    if time_period is None:
        return None
    try:
        return int(time_period[:4])
    except ValueError:
        return None


def latest_period(periods: Iterable[str | None]) -> str | None:
    """The period a `latest` window serves for one StatVar. It is the newest of `periods`, the
    greatest leading year and, within that year, the greatest period string, which orders
    zero-padded `YYYY`, `YYYY-MM` and `YYYY-MM-DD` chronologically. Null periods and periods
    without a leading year are skipped. None when none is left.

    A StatVar serves this one period on every read, place-keyed and role-bearing alike, and
    only from the series holding it. A series whose own latest period is earlier serves
    nothing. A Datasets read reduces a StatVar's rows by it (`columnar.latest_rows`), and
    Search reduces its plan (`search.engine.retrieval.read_plan.PairPlan.served`) and the
    preview it reads (`latest_periods`) by it."""
    newest: tuple[int, str] | None = None
    for period in periods:
        if period is None or (year := _leading_year(period)) is None:
            continue
        if newest is None or (year, period) > newest:
            newest = (year, period)
    return newest[1] if newest is not None else None


def latest_periods[K](keyed: Iterable[tuple[K, str | None]]) -> dict[K, str]:
    """Each key's `latest_period` over the periods paired with it, for a key naming one
    StatVar. A key with no period left is absent."""
    by_key: dict[K, list[str | None]] = {}
    for key, period in keyed:
        by_key.setdefault(key, []).append(period)
    return {
        key: latest
        for key, periods in by_key.items()
        if (latest := latest_period(periods)) is not None
    }


def compute_applied_window(
    requested: RequestedWindow, served_years: tuple[int, int] | None
) -> AppliedWindow | None:
    """Returns the applied window for served rows.

    - `served_years=None` returns `None` and pairs with `empty_result`.
    - `all` uses served bounds with origin `default`.
    - `latest` uses the served maximum for both bounds with origin `query`.
    - `range` sets origin `coverage_clamp` when a stated bound is shortened.
      Open bounds resolve to served values. Otherwise origin is `query`. An end year the
      definition left unstated (`RequestedWindow.end_unstated`) is an open bound: data
      ending before the year the range was ended at falls short of nothing asked.
    """
    if served_years is None:
        return None
    served_start, served_end = served_years

    if requested.kind == "all":
        return AppliedWindow(
            kind="all", start_year=served_start, end_year=served_end, origin="default"
        )
    if requested.kind == "latest":
        return AppliedWindow(
            kind="latest", start_year=served_end, end_year=served_end, origin="query"
        )

    clamped = False
    if requested.start_year is not None:
        if served_start > requested.start_year:
            start_year = served_start
            clamped = True
        else:
            start_year = requested.start_year
    else:
        start_year = served_start

    if requested.end_year is not None and not requested.end_unstated:
        if served_end < requested.end_year:
            end_year = served_end
            clamped = True
        else:
            end_year = requested.end_year
    else:
        end_year = served_end

    origin: Literal["query", "coverage_clamp"] = "coverage_clamp" if clamped else "query"
    return AppliedWindow(kind="range", start_year=start_year, end_year=end_year, origin=origin)


@dataclass(frozen=True)
class Coverage:
    """Row and dimension counts, distinct refs, and applied window.

    `entity_refs` maps role labels to dcid sets and `years` to leading-year
    sets. It defaults empty. `materialise.columnar.coverage()` fills it.
    """

    rows: int
    dimensions: tuple[dict[str, object], ...]
    window: dict[str, object] | None
    entity_refs: Mapping[str, frozenset[str | int]] = field(default_factory=dict)


def unserved_entities(
    role_bindings: Mapping[str, Sequence[str]],
    served_entities_by_role: Mapping[str, set[str]],
) -> tuple[str, ...]:
    """Sorted, deduplicated bound dcids missing from served coverage.

    `role_bindings` includes plain dcid lists only. Callers omit
    `ContainedIn` bindings because their candidate sets are unknown here.
    """
    unserved: set[str] = set()
    for role, bound in role_bindings.items():
        served = served_entities_by_role.get(role, set())
        unserved.update(set(bound) - served)
    return tuple(sorted(unserved))


def coverage_below_scope(
    coverage: Coverage,
    *,
    applied_window: AppliedWindow | None,
    unserved: Collection[str],
) -> bool:
    """Whether `applied_window.origin` is `coverage_clamp` or some bound entity is missing:
    `unserved` lists the missing ones (`unserved_entities`) no other warning names.

    Returns `False` for zero rows, leaving that case to the `empty_result`
    warning.
    """
    if coverage.rows == 0:
        return False
    if applied_window is not None and applied_window.origin == "coverage_clamp":
        return True
    return bool(unserved)
