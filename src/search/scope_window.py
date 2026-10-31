"""The years a range read covers, and the window rule a role-bearing read admits.

A role-bearing read lists its years in one SDMX year filter. Search's role routes
(`search.engine.retrieval.read_plan.role_years`) and the Datasets mint
(`datasets_api.kinds.spec_id`) both take those years from `role_read_years`: the range floored
at `max(scope_min_year, horizon.epoch_year)`, capped at the coverage horizon's last year, and
refused as `WindowTooLong` past `scope_max_window_years`. A window the mint admits therefore
resolves on resubmit, and one Search would refuse is refused by the mint before it builds.

`ScopeWindowSettings` owns both settings, read from `SEARCH_*` environment variables. This
module imports nothing from `search.engine`, so a process without the engine extra loads it.
Consumers call `scope_window_settings()` at the point of use, so a test fixture can replace the
object and every reader sees the replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from materialise.window import RequestedWindow, election_window
from search.models import TimeWindow

if TYPE_CHECKING:
    from series_coverage.schema import YearHorizon


class ScopeWindowSettings(BaseSettings):
    """The year floor and window-length cap of a role-bearing read, read from `SEARCH_*`
    environment variables. Field names are the `SEARCH_` env var name lowercased with that
    prefix stripped."""

    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="forbid")

    # The earliest year a role-bearing read reaches: an open or earlier start floors here.
    # Coverage paths also floor at the artifact's epoch_year, the first year its bitmask holds.
    scope_min_year: int = 1960

    # The most years a role-bearing read lists before they become one SDMX year filter.
    scope_max_window_years: int = 80

    @model_validator(mode="after")
    def _validate_bounds(self) -> ScopeWindowSettings:
        """Raises ValueError when the settings are built."""
        if self.scope_min_year < 1:
            raise ValueError(f"SEARCH_SCOPE_MIN_YEAR must be >= 1; got {self.scope_min_year!r}")
        if self.scope_max_window_years < 1:
            raise ValueError(
                f"SEARCH_SCOPE_MAX_WINDOW_YEARS must be >= 1; got {self.scope_max_window_years!r}"
            )
        return self


# Built on first read, after `search.engine.config` has loaded `.env` into the environment. A
# test fixture installs a different instance with
# `monkeypatch.setattr(search.scope_window, "_SETTINGS", ScopeWindowSettings(...))`.
_SETTINGS: ScopeWindowSettings | None = None


def scope_window_settings() -> ScopeWindowSettings:
    """Return the process-wide `ScopeWindowSettings` object, building it on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = ScopeWindowSettings()
    return _SETTINGS


def read_range(
    window: TimeWindow,
    *,
    clock_year: int | None,
    first_year: int | None = None,
    last_year: int | None = None,
) -> TimeWindow | None:
    """The years a range read covers for `window`, or None when none fits.

    `first_year` floors the start (an open start reads from it) and `last_year`, the
    coverage horizon's last year, caps the end. An explicit end is kept, so a window past
    `clock_year` reads projection years. An open end takes the end a datasets read of the same
    identity takes (`materialise.window.election_window`): `clock_year`, or for a window
    starting after it `last_year`, and every later year without a horizon. A None
    `clock_year` reads the datasets clock (`materialise.window.current_year`).
    """
    start = window.start_year
    if first_year is not None:
        start = first_year if start is None else max(start, first_year)
    end = election_window(
        RequestedWindow(kind="range", start_year=start, end_year=window.end_year),
        clock_year=clock_year,
        last_year=last_year,
    ).end_year
    if last_year is not None and end is not None:
        end = min(end, last_year)
    if start is not None and end is not None and start > end:
        return None
    return TimeWindow(start_year=start, end_year=end)


class WindowTooLong(Exception):
    """A window whose role-bearing read lists more than `scope_max_window_years` years
    (`role_read_years`). `read` holds the years counted, and `first_year` and `last_year` the
    floor and horizon end they were counted within. Search's role routes refuse it as
    `scope_too_broad`, and the Datasets mint as `request_invalid`."""

    def __init__(
        self, *, read: TimeWindow, years: int, cap: int, first_year: int, last_year: int
    ) -> None:
        self.read = read
        self.years = years
        self.cap = cap
        self.first_year = first_year
        self.last_year = last_year
        super().__init__(f"the requested window spans {years} years, over the {cap}-year cap")


def role_read_years(
    window: TimeWindow, *, horizon: YearHorizon, clock_year: int | None
) -> TimeWindow | None:
    """The years a role-bearing read of `window` lists over `horizon`: `read_range` floored at
    `max(scope_min_year, horizon.epoch_year)` and capped at `horizon.max_year`, which carries
    projection years. None when no year is left. Raises `WindowTooLong` when the years number
    more than `scope_max_window_years`."""
    settings = scope_window_settings()
    first_year = max(settings.scope_min_year, horizon.epoch_year)
    years = read_range(
        window, clock_year=clock_year, first_year=first_year, last_year=horizon.max_year
    )
    if years is None:
        return None
    assert years.start_year is not None and years.end_year is not None  # both bounds set
    span = years.end_year - years.start_year + 1
    if span > settings.scope_max_window_years:
        raise WindowTooLong(
            read=years,
            years=span,
            cap=settings.scope_max_window_years,
            first_year=first_year,
            last_year=horizon.max_year,
        )
    return years
