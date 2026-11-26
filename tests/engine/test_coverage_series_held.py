"""Whether a role-bearing shape holds a series at given places counts the years a read of
the window serves (`coverage_series.series_held`). A series holding only its source's
partial year is held by a window opting in to it, and by no other. A part judges its words'
window before its answer settles the opt-in, so it counts every held-back year its request
may serve (`coverage_series.requested_window`). The span a passed-over total states
(`coverage_series.series_span`) leaves out the same years."""

from __future__ import annotations

import pytest

from origin_catalogue import bound_catalogue
from search.engine.extract import DateRequest
from search.engine.region.coverage_series import requested_window, series_held, series_span
from search.models import TimeWindow
from series_coverage.schema import FacetMetadata
from tests.origin_catalogue.fake_catalogue import dev_provenance_nodes, documenting
from tests.series_coverage.support import KEN
from tests.series_coverage.test_index import (
    _CRS_PROVENANCE,
    _WINDOW_FACET,
    _WINDOW_SV,
    _donor,
    _donor_rows,
    _election_index,
)

_NAMED = DateRequest(
    window=TimeWindow(start_year=2020, end_year=2025),
    latest=False,
    reads_held=frozenset({"partial", "projected"}),
)


@pytest.mark.parametrize(
    ("window", "held"),
    [
        pytest.param(None, False, id="every-year"),
        pytest.param(TimeWindow(start_year=2020, end_year=2025), False, id="range"),
        pytest.param(
            TimeWindow(start_year=2020, end_year=2025, include=["partial"]), True, id="opted-in"
        ),
        pytest.param(requested_window(_NAMED), True, id="named-before-settling"),
    ],
)
def test_a_series_holding_only_its_partial_year_is_held_where_a_read_serves_it(
    window: TimeWindow | None, held: bool
) -> None:
    """One CRS donor's series into Kenya holds only 2025, which CRS marks partial."""
    index = _election_index(
        _donor_rows(_WINDOW_SV, {2025: 1}),
        {_WINDOW_FACET: FacetMetadata(provenance_id=_CRS_PROVENANCE, unit="USD")},
    )
    crs = {_CRS_PROVENANCE: dev_provenance_nodes()[_CRS_PROVENANCE]}

    with bound_catalogue(documenting(crs)):
        found = series_held(
            index,
            variables=[_WINDOW_SV],
            bound={"origin": [_donor(0)], "destination": [KEN]},
            window=window,
        )

    assert found is held


@pytest.mark.parametrize(
    ("window", "span"),
    [
        pytest.param(None, (2023, 2024), id="every-year"),
        pytest.param(TimeWindow(start_year=2026, end_year=2027), (2023, 2024), id="range"),
        pytest.param(
            TimeWindow(start_year=2026, end_year=2027, include=["partial"]),
            (2023, 2025),
            id="opted-in",
        ),
    ],
)
def test_a_series_span_ends_before_the_partial_year_a_read_holds_back(
    window: TimeWindow | None, span: tuple[int, int]
) -> None:
    """One CRS donor's series into Kenya holds 2023 to 2025, and CRS marks 2025 partial."""
    index = _election_index(
        _donor_rows(_WINDOW_SV, {2023: 1, 2024: 1, 2025: 1}),
        {_WINDOW_FACET: FacetMetadata(provenance_id=_CRS_PROVENANCE, unit="USD")},
    )
    crs = {_CRS_PROVENANCE: dev_provenance_nodes()[_CRS_PROVENANCE]}

    with bound_catalogue(documenting(crs)):
        found = series_span(
            index,
            variables=[_WINDOW_SV],
            bound={"origin": [_donor(0)], "destination": [KEN]},
            window=window,
        )

    assert found == span
