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


@pytest.mark.parametrize(
    ("dates", "window", "trailing"),
    [
        # "the last five years" in 2026: its newest year may lie three years before the span,
        # and the window holds the five years ending at each such year.
        (
            [ExtractedDate(kind="range", start="2022", end="2026", trailing_years=5)],
            (2015, 2026),
            5,
        ),
        # "the past year" asks for one year.
        (
            [ExtractedDate(kind="range", start="2026", end="2026", trailing_years=1)],
            (2023, 2026),
            1,
        ),
        # The same years named outright trail nothing.
        ([_range("2022", "2026")], (2022, 2026), None),
        # A trailing span beside another date combines as named years.
        (
            [
                ExtractedDate(kind="range", start="2022", end="2026", trailing_years=5),
                _point("2010"),
            ],
            (2010, 2026),
            None,
        ),
    ],
)
def test_trailing_span_widens_its_window(
    dates: list[ExtractedDate], window: tuple[int, int], trailing: int | None
) -> None:
    """A lone span counted back from now asks for its newest years: its window holds every
    span ending in its judged years, `trailing_stale_years` before the calendar years its
    words cover (`asked_window`), and its judgement reads those judged years alone."""
    req = dates_to_request(dates)
    assert req is not None
    assert (req.window, req.trailing_years) == (
        TimeWindow(start_year=window[0], end_year=window[1]),
        trailing,
    )
    if trailing is not None:
        assert req.asked_window == TimeWindow(start_year=2026 - trailing + 1, end_year=2026)
        assert req.judged.window == TimeWindow(start_year=2026 - trailing - 2, end_year=2026)
        assert req.judged.trailing_years is None
