"""Checks that materialise and Search range filters agree, including unparseable periods."""

from __future__ import annotations

import pytest

from dc_sdmx import SdmxObservation
from materialise import columnar
from materialise.window import RequestedWindow
from search.models import TimeWindow
from search.models import in_window as search_in_window

from ._table_support import table_from_observations

_ROLES = ("recipient",)


def _row(*, time_period: str | None) -> SdmxObservation:
    return SdmxObservation(
        variable="dcid:MalariaGrantDisbursements",
        role_values={"recipient": "country/TGO"},
        unit="USD",
        measurement_method="GrantEquivalent",
        observation_period=None,
        provenance="OECD_DAC2A",
        time_period=time_period,
        value="1000",
        scaling_factor=None,
        facet_id="facet1",
    )


def _table(rows):
    return table_from_observations(rows, roles=_ROLES)


# (time_period, start_year, end_year). Covers a normal in-range value, a
# value outside each bound, an open (None) bound on each side, and the
# fail-open cases: None, "", and a non-numeric leading segment.
# `TimeWindow` requires at least one of start_year/end_year to be non-None,
# so every row below satisfies that.
_MATRIX = [
    ("2018", 2015, 2020),
    ("2010", 2015, 2020),
    ("2025", 2015, 2020),
    ("2018", None, 2020),
    ("2018", 2015, None),
    (None, 2015, 2020),
    ("", 2015, 2020),
    ("abcd", 2015, 2020),
]


@pytest.mark.parametrize("time_period,start_year,end_year", _MATRIX)
def test_range_filtering_matches_search_in_window(time_period, start_year, end_year):
    window = RequestedWindow(kind="range", start_year=start_year, end_year=end_year)
    table = _table([_row(time_period=time_period)])
    materialise_kept = columnar.range_window_mask(table, window=window).to_pylist()[0]

    search_window = TimeWindow(start_year=start_year, end_year=end_year)
    search_kept = search_in_window(time_period, search_window)

    assert materialise_kept == search_kept, (
        f"materialise.columnar and search.models.in_window disagree for "
        f"time_period={time_period!r}, start_year={start_year!r}, end_year={end_year!r}"
    )


# This independently pins the fail-open behaviour on both sides, rather
# than only cross-checking that they currently agree -- the parametrized
# comparison above would keep passing even if both sides drifted to the
# same new convention (e.g. both starting to fail closed on an unparseable
# time_period).
def test_unparseable_time_period_is_kept_under_a_bounded_window_on_both_sides():
    window = RequestedWindow(kind="range", start_year=2015, end_year=2020)
    table = _table([_row(time_period="not-a-year")])
    assert columnar.range_window_mask(table, window=window).to_pylist() == [True]

    search_window = TimeWindow(start_year=2015, end_year=2020)
    assert search_in_window("not-a-year", search_window) is True
