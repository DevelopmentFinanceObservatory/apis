"""A window that names a source's partial year as its end, without opting in, still holds the
year back and states the run (`datasets_api.manifest.held_back_coverage`). The user's naming
of 2025 reaches the mint only as an opt-in, so the default cut applies to it.
"""

from __future__ import annotations

import pytest

from datasets_api.manifest import held_back_coverage
from materialise.window import RequestedWindow

from . import _weo_recorded
from .test_held_back import (  # noqa: F401  (autouse fixtures: CRS marks 2025 partial)
    _crs,
    _marked_catalogue,
    _mint,
    _sdmx_serving_the_periods_asked,
    _years,
)


def _partial_runs(result: object, window: RequestedWindow, places: list[str]) -> list[dict]:
    request = _weo_recorded._request(places, window=window, facet_bindings=None)
    return [
        run
        for run in held_back_coverage(result, requests=[request])  # ty: ignore[invalid-argument-type]
        if run["kind"] == "partial"
    ]


@pytest.mark.parametrize(
    ("window", "served"),
    [
        pytest.param(
            RequestedWindow(kind="range", start_year=2020, end_year=2025),
            {"2020", "2021", "2022", "2023", "2024"},
            id="closed-range-ending-on-the-partial-year",
        ),
        pytest.param(
            RequestedWindow(kind="range", start_year=2025, end_year=2025),
            set(),
            id="single-partial-year",
        ),
        pytest.param(
            RequestedWindow(
                kind="range", start_year=2020, end_year=2025, include=frozenset({"partial"})
            ),
            {"2020", "2021", "2022", "2023", "2024", "2025"},
            id="control-closed-range-opted-in",
        ),
        pytest.param(
            RequestedWindow(
                kind="range", start_year=2025, end_year=2025, include=frozenset({"partial"})
            ),
            {"2025"},
            id="control-single-year-opted-in",
        ),
    ],
)
def test_a_named_partial_end_without_opt_in_serves_only_complete_years_and_states_the_run(
    window: RequestedWindow, served: set[str]
) -> None:
    source = _crs(ending_2024=80, reaching_2025=10)

    result = _mint(source, window)

    assert _years(result) == served
    runs = _partial_runs(result, window, source[0])
    assert [run["served"] for run in runs] == ["2025" in served]
