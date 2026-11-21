"""The last complete year of a group of series, read from their year masks.

A source that publishes a year in stages (the OECD's preliminary DAC and CRS figures) holds
few series in its newest year until the full release. `last_complete_year` names the newest
year a group's series hold in full: the newest year with data, stepped back one year at a time
while that year holds fewer than `ANCHOR_SHARE` of the series the year before holds. The share
is the one the rolling anchor and `series_coverage.index.elected_year` decide a held year by.

Only the tail is read. An earlier fall, such as the step from reported years to projected
ones in a debt-service table, leaves the years after it complete.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from dfo_frame_grammar.rolling import ANCHOR_SHARE

from series_coverage.schema import MASK_WORD_BITS, YearHorizon

# Rows unpacked at once, so a large group costs at most this many rows times
# `MASK_WORD_BITS * mask_words` bytes of bits.
_CHUNK_ROWS = 1 << 16


def last_complete_year(series_by_year: Mapping[int, int]) -> int | None:
    """The newest year `series_by_year` holds in full, or `None` when no year holds a series.

    Start at the newest year holding a series and step back while the year holds fewer than
    `ANCHOR_SHARE` of the series the year before holds. A year the mapping omits holds none,
    so a year after a gap stays complete."""
    held = [year for year, series in series_by_year.items() if series > 0]
    if not held:
        return None
    year = max(held)
    while (before := series_by_year.get(year - 1, 0)) > 0 and series_by_year.get(
        year, 0
    ) < ANCHOR_SHARE * before:
        year -= 1
    return year


def group_year_counts(
    masks: np.ndarray, groups: np.ndarray, *, group_count: int, horizon: YearHorizon
) -> np.ndarray:
    """How many rows of each group hold each year of `horizon`.

    `masks` is a `(rows, mask_words)` uint64 array of year masks on `horizon`, and `groups`
    each row's group in `[0, group_count)`, or -1 for a row no group counts. Returns a
    `(group_count, mask_words * MASK_WORD_BITS)` array whose column `i` counts year
    `horizon.epoch_year + i`."""
    counts = np.zeros((group_count, horizon.mask_words * MASK_WORD_BITS), dtype=np.int64)
    kept = np.flatnonzero(groups >= 0)
    if kept.size == 0:
        return counts
    order = kept[np.argsort(groups[kept], kind="stable")]
    ordered = groups[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    for start, stop in zip(starts, [*starts[1:], ordered.size], strict=True):
        group = int(ordered[start])
        for low in range(start, stop, _CHUNK_ROWS):
            rows = order[low : min(low + _CHUNK_ROWS, stop)]
            words = np.ascontiguousarray(masks[rows], dtype="<u8")
            bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")
            counts[group] += bits.sum(axis=0, dtype=np.int64)
    return counts


def last_complete_years(counts: np.ndarray, *, horizon: YearHorizon) -> list[int | None]:
    """`last_complete_year` of each row of `counts`, a `group_year_counts` result."""
    years: list[int | None] = []
    for row in counts:
        held = np.flatnonzero(row)
        years.append(last_complete_year({horizon.epoch_year + int(i): int(row[i]) for i in held}))
    return years


__all__ = ["group_year_counts", "last_complete_year", "last_complete_years"]
