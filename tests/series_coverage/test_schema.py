"""`series_coverage.schema.YearHorizon`: sizing from the data, the per-word
year-mask arithmetic (`masks_from_years`, `window_masks`), and the on-disk
schema it names.
"""

from __future__ import annotations

import pytest

from series_coverage.schema import YearHorizon

_ONE_WORD = YearHorizon(epoch_year=1973, mask_words=1)
_TWO_WORDS = YearHorizon(epoch_year=1973, mask_words=2)
_ALL_BITS = 0xFFFFFFFFFFFFFFFF


@pytest.mark.parametrize(
    ("last_year", "words"),
    [(1973, 1), (2024, 1), (2036, 1), (2037, 2), (2040, 2), (2100, 2), (2101, 3)],
)
def test_covering_sizes_the_word_count_to_the_latest_year(last_year: int, words: int) -> None:
    horizon = YearHorizon.covering(epoch_year=1973, last_year=last_year)
    assert horizon.mask_words == words
    assert horizon.max_year >= last_year


def test_max_year_and_columns_follow_the_word_count() -> None:
    assert _ONE_WORD.max_year == 2036
    assert _TWO_WORDS.max_year == 2100
    assert _TWO_WORDS.mask_columns == ("year_mask_0", "year_mask_1")
    assert _TWO_WORDS.schema.names[-2:] == ["year_mask_0", "year_mask_1"]


def test_a_horizon_needs_at_least_one_word() -> None:
    with pytest.raises(ValueError, match="mask_words"):
        YearHorizon(epoch_year=1973, mask_words=0)


def test_masks_from_years_places_each_year_in_its_word() -> None:
    # 1973 is word 0 bit 0, 2036 word 0 bit 63, 2037 word 1 bit 0, 2040 bit 3.
    assert _TWO_WORDS.masks_from_years([1973, 2036, 2037, 2040]) == (
        1 | (1 << 63),
        1 | (1 << 3),
    )


@pytest.mark.parametrize("year", [1972, 2101])
def test_masks_from_years_refuses_a_year_outside_the_horizon(year: int) -> None:
    with pytest.raises(ValueError, match="outside the representable range"):
        _TWO_WORDS.masks_from_years([year])


def test_window_masks_no_bounds_is_every_bit_of_every_word() -> None:
    assert _TWO_WORDS.window_masks(first_year=None, last_year=None) == (_ALL_BITS, _ALL_BITS)


def test_window_masks_across_the_word_boundary() -> None:
    # 2035-2038: word 0 bits 62-63, word 1 bits 0-1.
    assert _TWO_WORDS.window_masks(first_year=2035, last_year=2038) == (
        (1 << 62) | (1 << 63),
        0b11,
    )


def test_window_masks_match_masks_from_years_over_the_same_span() -> None:
    lo, hi = 2000, 2050
    assert _TWO_WORDS.window_masks(first_year=lo, last_year=hi) == _TWO_WORDS.masks_from_years(
        range(lo, hi + 1)
    )


def test_window_masks_clamp_to_the_horizon() -> None:
    assert _ONE_WORD.window_masks(first_year=1900, last_year=1973) == (1,)
    assert _ONE_WORD.window_masks(first_year=2036, last_year=2200) == (1 << 63,)


@pytest.mark.parametrize(("first_year", "last_year"), [(1900, 1950), (2037, 2040), (2020, 2019)])
def test_window_masks_outside_or_inverted_are_empty(first_year: int, last_year: int) -> None:
    assert _ONE_WORD.window_masks(first_year=first_year, last_year=last_year) == (0,)
