"""The last complete year of a group of series (`series_coverage.complete`).

The counts are families of the dev graph release read on 2026-11-18: the series of each family
under one provenance holding each year.
"""

from __future__ import annotations

import numpy as np
import pytest

from series_coverage.complete import group_year_counts, last_complete_year, last_complete_years
from series_coverage.schema import YearHorizon

_CRS_FLOWS = {2021: 223_518, 2022: 223_338, 2023: 228_543, 2024: 233_811, 2025: 8_812}
_DAC1_ODA_GNI_SHARE = {2021: 50, 2022: 51, 2023: 50, 2024: 51, 2025: 46}
_DAC1_MOBILISED_COMMITMENTS = {2021: 17, 2022: 8, 2023: 6, 2024: 6, 2025: 2}
_IDS_FLOWS = {
    2024: 435_280,
    2025: 435_354,
    2026: 146_511,
    2027: 143_948,
    2028: 138_502,
    2029: 131_354,
    2030: 126_190,
    2031: 119_666,
    2032: 114_339,
    2033: 107_463,
}


@pytest.mark.parametrize(
    ("series_by_year", "expected"),
    [
        pytest.param(_CRS_FLOWS, 2024, id="crs-preliminary-year-is-partial"),
        pytest.param(_DAC1_ODA_GNI_SHARE, 2025, id="control-nine-tenths-is-complete"),
        pytest.param(_DAC1_MOBILISED_COMMITMENTS, 2024, id="a-third-is-partial"),
        pytest.param(_IDS_FLOWS, 2033, id="a-fall-before-the-tail-leaves-it-complete"),
        pytest.param({2023: 1_000, 2024: 100, 2025: 5}, 2023, id="two-partial-years-step-back"),
        pytest.param({2023: 100, 2024: 50}, 2024, id="exactly-half-is-complete"),
        pytest.param({2020: 100, 2022: 3}, 2022, id="a-year-after-a-gap-is-complete"),
        pytest.param({2025: 0}, None, id="no-series-names-no-year"),
    ],
)
def test_last_complete_year_steps_back_while_a_year_holds_under_half_the_one_before(
    series_by_year: dict[int, int], expected: int | None
) -> None:
    assert last_complete_year(series_by_year) == expected


def test_group_year_counts_count_each_groups_rows_per_year() -> None:
    """2023 is the top bit of the first word from 1960, and 2024 the low bit of the second."""
    horizon = YearHorizon(epoch_year=1960, mask_words=2)
    rows = [
        (0, [2022, 2023, 2024]),
        (0, [2023, 2024]),
        (0, [2023]),
        (1, [2024]),
        (-1, [2023, 2024]),
    ]
    masks = np.array([horizon.masks_from_years(years) for _, years in rows], dtype=np.uint64)
    groups = np.array([group for group, _ in rows])

    counts = group_year_counts(masks, groups, group_count=3, horizon=horizon)

    first = horizon.epoch_year
    assert {first + int(i): int(counts[0, i]) for i in np.flatnonzero(counts[0])} == {
        2022: 1,
        2023: 3,
        2024: 2,
    }
    assert {first + int(i): int(counts[1, i]) for i in np.flatnonzero(counts[1])} == {2024: 1}
    assert not counts[2].any()
    assert last_complete_years(counts, horizon=horizon) == [2024, 2024, None]
