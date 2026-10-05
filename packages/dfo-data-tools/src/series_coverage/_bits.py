"""Vectorised uint64 popcount and set-bit positions.

SWAR arithmetic avoids float64 rounding, which can misplace bit 63 beyond
float64's 53-bit exact-integer range. Results match Python's integer methods
across all 64 bits.
"""

from __future__ import annotations

import numpy as np

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)
_ONE = np.uint64(1)


def popcount64(values: np.ndarray) -> np.ndarray:
    """Bits set in each uint64 element (SWAR popcount), vectorized.
    Matches `int.bit_count()` element-wise.
    """
    x = values.astype(np.uint64, copy=True)
    x = x - ((x >> _ONE) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return (x * _H01) >> np.uint64(56)


def bit_length64(values: np.ndarray) -> np.ndarray:
    """Position of the highest set bit, plus one. Returns 0 for a zero element.
    Matches `int.bit_length()` element-wise. Vectorized "smear right"
    (OR each element with itself shifted right by 1, 2, 4, ... 32) fills
    every bit below the highest set one, turning the bit-length question
    into a popcount.
    """
    x = values.astype(np.uint64, copy=True)
    for shift in (1, 2, 4, 8, 16, 32):
        x = x | (x >> np.uint64(shift))
    return popcount64(x)


def lowest_set_bit_index64(values: np.ndarray) -> np.ndarray:
    """Position of the lowest set bit (0-indexed). Undefined for a zero
    element (returns 64, `popcount64` of an all-ones 64-bit value).
    Every caller in this package only calls this on values already
    filtered to nonzero.
    """
    x = values.astype(np.uint64, copy=True)
    lsb = x & ((~x) + _ONE)
    return popcount64(lsb - _ONE)


__all__ = ["bit_length64", "lowest_set_bit_index64", "popcount64"]
