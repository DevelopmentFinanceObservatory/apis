"""Defines the coverage table schema and year-mask contract without I/O.

Each row represents `(variable, facet_id, donor, recipient)`. Word `w`, bit
`b` marks year `epoch_year + 64*w + b` of the table's `YearHorizon`.
`donor` and `recipient` map Spanner's entity columns, while `RoleNames` stores
graph property names per variable. A variable that declares one role stores
its entity in the column of that role's flow axis (`recipient` for a
destination-axis role, `donor` for an origin-axis one) and `ABSENT_ENTITY` in
the other, and its `RoleNames` holds `None` for the role it lacks. Each row's
`n_obs` equals its mask popcount, checked by `series_coverage.loader.checked_index`
and used for exact windowed counts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

from artifact_publication.jobs import Ingestion

MASK_WORD_BITS: Final[int] = 64

# Column order. The first four fields form the series key.
# `variable` has 22.5K distinct values across 1.78M rows, so dictionary encoding
# keeps the most repetitive key column compact.
VARIABLE_COLUMN: Final[str] = "variable"
FACET_COLUMN: Final[str] = "facet_id"
DONOR_COLUMN: Final[str] = "donor"
RECIPIENT_COLUMN: Final[str] = "recipient"
N_OBS_COLUMN: Final[str] = "n_obs"
FIRST_YEAR_COLUMN: Final[str] = "first_year"
LAST_YEAR_COLUMN: Final[str] = "last_year"

ABSENT_ENTITY: Final[str] = ""
"""The `donor` or `recipient` value of a row whose variable declares no role
for that column. No graph dcid is empty, so the value never collides with a
real entity."""


def widen_span(spans: dict[str, tuple[int, int]], provenance: str, first: int, last: int) -> None:
    """Widen `spans[provenance]` to cover the years `first` through `last`, inclusive."""
    low, high = spans.get(provenance, (first, last))
    spans[provenance] = (min(low, first), max(high, last))


def year_mask_column(word: int) -> str:
    """The column holding mask word `word` (years `epoch_year + 64 * word`
    onward)."""
    return f"year_mask_{word}"


@dataclass(frozen=True, slots=True)
class YearHorizon:
    """Representable years are `mask_words` 64-bit words from `epoch_year`
    through `max_year = epoch_year + 64 * mask_words - 1`."""

    epoch_year: int
    mask_words: int

    def __post_init__(self) -> None:
        if self.mask_words < 1:
            raise ValueError(f"mask_words must be >= 1; got {self.mask_words}")

    @classmethod
    def covering(cls, *, epoch_year: int, last_year: int) -> YearHorizon:
        """The narrowest horizon from `epoch_year` that represents
        `last_year` (at least one word)."""
        span = max(last_year - epoch_year + 1, 1)
        return cls(epoch_year=epoch_year, mask_words=-(-span // MASK_WORD_BITS))

    @property
    def max_year(self) -> int:
        return self.epoch_year + MASK_WORD_BITS * self.mask_words - 1

    @property
    def mask_columns(self) -> tuple[str, ...]:
        return tuple(year_mask_column(word) for word in range(self.mask_words))

    @property
    def schema(self) -> pa.Schema:
        """The schema of a coverage table with this horizon."""
        return pa.schema(
            [
                pa.field(VARIABLE_COLUMN, pa.string(), nullable=False),
                pa.field(FACET_COLUMN, pa.string(), nullable=False),
                pa.field(DONOR_COLUMN, pa.string(), nullable=False),
                pa.field(RECIPIENT_COLUMN, pa.string(), nullable=False),
                # Annual data has at most one observation per year. A future
                # Sub-annual series could exceed 16 bits. Uint32 costs the same.
                pa.field(N_OBS_COLUMN, pa.uint32(), nullable=False),
                pa.field(FIRST_YEAR_COLUMN, pa.uint16(), nullable=False),
                pa.field(LAST_YEAR_COLUMN, pa.uint16(), nullable=False),
                *(pa.field(name, pa.uint64(), nullable=False) for name in self.mask_columns),
            ]
        )

    def masks_from_years(self, years: Iterable[int]) -> tuple[int, ...]:
        """One mask value per word, with exactly the bits `years` names set.
        Raises `ValueError` for a year outside `[epoch_year, max_year]`."""
        words = [0] * self.mask_words
        for year in years:
            if not self.epoch_year <= year <= self.max_year:
                raise ValueError(
                    f"year {year} is outside the representable range "
                    f"[{self.epoch_year}, {self.max_year}]"
                )
            word, bit = divmod(year - self.epoch_year, MASK_WORD_BITS)
            words[word] |= 1 << bit
        return tuple(words)

    def window_masks(self, *, first_year: int | None, last_year: int | None) -> tuple[int, ...]:
        """One mask value per word with every bit in `[first_year,
        last_year]` set, each bound clamped into `[epoch_year, max_year]`
        (a missing bound reads as that range's own end). An empty overlap
        with the horizon returns all-zero masks."""
        lo = self.epoch_year if first_year is None else max(first_year, self.epoch_year)
        hi = self.max_year if last_year is None else min(last_year, self.max_year)
        words = [0] * self.mask_words
        if lo > hi:
            return tuple(words)
        for word in range(self.mask_words):
            word_lo = self.epoch_year + MASK_WORD_BITS * word
            start = max(lo, word_lo) - word_lo
            end = min(hi, word_lo + MASK_WORD_BITS - 1) - word_lo
            if start <= end:
                width = end - start + 1
                words[word] = ((1 << width) - 1) << start
        return tuple(words)


@dataclass(frozen=True, slots=True)
class RoleNames:
    """Graph property names bound to this variable's donor and recipient roles.

    A variable that declares one role holds `None` for the other: `donor` is
    `None` for a destination-axis role such as `financingRecipient`, and
    `recipient` is `None` for an origin-axis role such as `financingProvider`.
    Its rows carry `ABSENT_ENTITY` in that role's column. Raises `ValueError`
    when both are `None`.
    """

    donor: str | None
    recipient: str | None

    def __post_init__(self) -> None:
        if self.donor is None and self.recipient is None:
            raise ValueError("RoleNames needs a donor role, a recipient role, or both")

    @property
    def declared(self) -> tuple[str, ...]:
        """The roles this variable declares, donor first."""
        return tuple(role for role in (self.donor, self.recipient) if role is not None)


@dataclass(frozen=True, slots=True)
class FacetMetadata:
    """Per-facet provenance, unit, measurement method, observation period, and scaling factor.

    `checked_index` requires an entry for every coverage facet.
    `observation_period` is the facet's ISO 8601 duration (`P1Y`), or `None`
    when unknown. Counts and years live on the coverage rows.
    """

    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    scaling_factor: str | None = None
    observation_period: str | None = None


@dataclass(frozen=True, slots=True)
class CoverageManifest:
    """The identity of the coverage a surface serves, which
    `series_coverage.shared` builds from the bound graph release.

    `ingestion` is the successful `IngestionHistory` row the release's import
    names, which the origin catalogue's manifest records too, or `None` when
    unknown.
    """

    stamp: str
    build_id: str
    built_at: str
    horizon: YearHorizon
    ingestion: Ingestion | None = None


__all__ = [
    "ABSENT_ENTITY",
    "DONOR_COLUMN",
    "FACET_COLUMN",
    "FIRST_YEAR_COLUMN",
    "LAST_YEAR_COLUMN",
    "MASK_WORD_BITS",
    "N_OBS_COLUMN",
    "RECIPIENT_COLUMN",
    "VARIABLE_COLUMN",
    "CoverageManifest",
    "FacetMetadata",
    "RoleNames",
    "YearHorizon",
    "year_mask_column",
]
