"""Cross-checks `series_coverage._bits`'s vectorized popcount/bit-position
functions against Python's own `int.bit_count()`/`int.bit_length()`, over
values spanning the full uint64 range -- not just today's realistic
year_mask values, since the whole point of these functions is to be exact
at the top of the range where a float64 round trip (np.log2/np.frexp)
would not be (see the module's own docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from series_coverage._bits import bit_length64, lowest_set_bit_index64, popcount64

_EDGE_VALUES = [
    0,
    1,
    2,
    3,
    0b101010,
    2**52,
    2**52 - 1,
    2**53,
    2**53 + 1,
    2**63,
    2**63 - 1,
    2**63 + 1,
    0xFFFFFFFFFFFFFFFF,
]


def _random_values(count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, 2**64, size=count, dtype=np.uint64)]


@pytest.mark.parametrize("value", _EDGE_VALUES + _random_values(500, seed=0))
def test_popcount64_matches_int_bit_count(value: int) -> None:
    result = popcount64(np.array([value], dtype=np.uint64))
    assert int(result[0]) == value.bit_count()


@pytest.mark.parametrize("value", _EDGE_VALUES + _random_values(500, seed=1))
def test_bit_length64_matches_int_bit_length(value: int) -> None:
    result = bit_length64(np.array([value], dtype=np.uint64))
    assert int(result[0]) == value.bit_length()


@pytest.mark.parametrize("value", [v for v in _EDGE_VALUES if v != 0] + _random_values(500, seed=2))
def test_lowest_set_bit_index64_matches_manual_computation(value: int) -> None:
    if value == 0:
        pytest.skip("undefined for zero")
    expected = (value & -value).bit_length() - 1
    result = lowest_set_bit_index64(np.array([value], dtype=np.uint64))
    assert int(result[0]) == expected


def test_popcount64_is_vectorized_over_an_array() -> None:
    values = np.array([0, 1, 3, 7, 2**63], dtype=np.uint64)
    result = popcount64(values)
    assert result.tolist() == [0, 1, 2, 3, 1]
