"""Tests date conversion and the DateRequest invariant."""

from __future__ import annotations

import pytest

from search.engine.extract import DateRequest, ExtractedDate, dates_to_request
from search.models import TimeWindow


def _point(year: str) -> ExtractedDate:
    return ExtractedDate(kind="point", start=year, end=None)


def _range(start: str | None, end: str | None) -> ExtractedDate:
    return ExtractedDate(kind="range", start=start, end=end)


def _latest() -> ExtractedDate:
    return ExtractedDate(kind="latest", start=None, end=None)


def test_none_input_returns_none() -> None:
    assert dates_to_request([]) is None


def test_point_collapses_to_symmetric_window() -> None:
    req = dates_to_request([_point("2020")])
    assert req is not None
    assert req.window == TimeWindow(start_year=2020, end_year=2020)
    assert req.latest is False


def test_open_range_start_only() -> None:
    """A missing end bound stays open."""
    req = dates_to_request([_range("2010", None)])
    assert req is not None
    assert req.window == TimeWindow(start_year=2010, end_year=None)
    assert req.latest is False


def test_open_range_end_only() -> None:
    """A missing start bound stays open."""
    req = dates_to_request([_range(None, "2020")])
    assert req is not None
    assert req.window == TimeWindow(start_year=None, end_year=2020)
    assert req.latest is False


def test_multi_date_collapse_to_min_start_max_end() -> None:
    """Multiple dates span their minimum start and maximum end."""
    dates = [_point("2015"), _range("2010", "2018"), _point("2020")]
    req = dates_to_request(dates)
    assert req is not None
    assert req.window == TimeWindow(start_year=2010, end_year=2020)
    assert req.latest is False


def test_latest_only_returns_latest_request() -> None:
    req = dates_to_request([_latest()])
    assert req is not None
    assert req.window is None
    assert req.latest is True


def test_point_plus_latest_point_wins() -> None:
    """A concrete point takes priority over ``latest``."""
    req = dates_to_request([_point("2018"), _latest()])
    assert req is not None
    assert req.window == TimeWindow(start_year=2018, end_year=2018)
    assert req.latest is False


def test_unparseable_string_is_skipped() -> None:
    """Unparseable bounds produce no request."""
    dates = [ExtractedDate(kind="range", start="not-a-year", end="also-bad")]
    req = dates_to_request(dates)
    assert req is None


def test_unparseable_mixed_with_valid_uses_valid() -> None:
    """A valid bound survives beside an unparseable bound."""
    dates = [_range("not-a-year", "2022"), _point("2018")]
    req = dates_to_request(dates)
    assert req is not None
    assert req.window == TimeWindow(start_year=2018, end_year=2022)


def test_date_request_post_init_rejects_window_and_latest() -> None:
    """Both a window and ``latest=True`` raise ``ValueError``."""
    with pytest.raises(ValueError, match="window or latest"):
        DateRequest(window=TimeWindow(start_year=2010, end_year=2020), latest=True)
