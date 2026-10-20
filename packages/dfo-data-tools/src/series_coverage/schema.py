"""Defines the coverage artifact schema and year-mask contract without I/O.

Each row represents `(variable, facet_id, donor, recipient)`. Word `w`, bit
`b` marks year `epoch_year + 64*w + b`. The manifest records the horizon.
`donor` and `recipient` map Spanner's entity columns, while `RoleNames` stores
graph property names per variable. A variable that declares one role stores
its entity in the column of that role's flow axis (`recipient` for a
destination-axis role, `donor` for an origin-axis one) and `ABSENT_ENTITY` in
the other, and its `RoleNames` holds `None` for the role it lacks. Each row's `n_obs` equals its mask popcount, checked by the
loader and used for exact windowed counts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX
from artifact_publication.jobs import Ingestion

COVERAGE_FORMAT_VERSION: Final[int] = 3
"""The artifact's on-disk shape: the Parquet schema, the bit-packing
convention, and the manifest's required fields. The year mask is split into
`year_mask_<w>` words and the manifest records the `YearHorizon`. A variable
that declares one role has `ABSENT_ENTITY` in the column of the role it lacks
and `None` for that role in `RoleNames`: format 3 admits a one-role variable
on either axis, where format 2 admitted it on the recipient axis only, so a
format-2 reader refuses a format-3 build that holds an origin-axis variable.
Every artifact lives under its format's own root,
`<COVERAGE_ARTIFACT_URI>/format-<COVERAGE_FORMAT_VERSION>/` (`FORMAT_ROOT`),
so an image reads and writes only artifacts of its own format, and a change
to this number starts a new, empty root that the next job run fills.
`series_coverage.loader.load_coverage` refuses a manifest whose
`format_version` differs from this constant.
"""

FORMAT_ROOT: Final[str] = f"format-{COVERAGE_FORMAT_VERSION}"
"""This format's directory under `COVERAGE_ARTIFACT_URI`. It holds
`POINTER_FILE`, `VERSIONS_PREFIX/<build_id>/`, `COMPATIBLE_PREFIX/`,
`SERVING_PREFIX/`, and the job's `coverage_state.json`."""

COMPATIBLE_PREFIX: Final[str] = "compatible"
"""The directory under `FORMAT_ROOT` holding one pointer per code index,
`<content_hash>.json` in `POINTER_FILE`'s form, naming the newest build that
carries that index's provenances (`series_coverage.compatibility`)."""

SERVING_PREFIX: Final[str] = "serving"
"""The directory under `FORMAT_ROOT` holding `code_index.json`, the
`content_hash` of the code index every service ran when the last deploy
finished (`series_coverage.compatibility.record_serving_index`)."""

BUILD_EPOCH_YEAR: Final[int] = 1960
"""The epoch `series_coverage.job` builds new artifacts at, equal to search's
default `scope_min_year`, the earliest year a search window reaches. A reader
never uses this: it reads the loaded artifact's own `YearHorizon.epoch_year`."""

MASK_WORD_BITS: Final[int] = 64

# On-disk column order. The first four fields form the series key.
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
    through `max_year = epoch_year + 64 * mask_words - 1`, recorded in the manifest."""

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
        """The on-disk Parquet schema of an artifact with this horizon."""
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

    The job reads metadata from Spanner's `TimeSeries` table and writes
    `facets.json`. The loader requires an entry for every coverage facet.
    `observation_period` is the facet's ISO 8601 duration (`P1Y`). An artifact
    written without it loads with `None`. Counts and years live on the
    coverage rows.
    """

    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    scaling_factor: str | None = None
    observation_period: str | None = None


_BUILD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class CoveragePointer:
    """The `current.json` pointer: the published data-version `stamp` and
    the `build_id` of the directory, `versions/<build_id>/`, that holds it.
    Two builds of one stamp have different build ids, so a reader that
    keys on `build_id` notices a republish of an unchanged stamp."""

    stamp: str
    build_id: str

    def to_json(self) -> str:
        return json.dumps({"stamp": self.stamp, "build_id": self.build_id}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: bytes | str) -> CoveragePointer:
        """Raises `ValueError` for anything other than a JSON object with a
        non-empty string `stamp` and a `build_id` that is one path segment
        of letters, digits, `.`, `_` and `-` starting with a letter or
        digit."""
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"pointer is not a JSON object: {payload!r}")
        stamp = payload.get("stamp")
        build_id = payload.get("build_id")
        if not isinstance(stamp, str) or not stamp:
            raise ValueError(f"pointer has no string 'stamp': {payload!r}")
        if not isinstance(build_id, str) or not _BUILD_ID_PATTERN.fullmatch(build_id):
            raise ValueError(f"pointer has no valid 'build_id': {payload!r}")
        return cls(stamp=stamp, build_id=build_id)


@dataclass(frozen=True, slots=True)
class CoverageManifest:
    """Manifest for one artifact build.

    `files` maps `coverage`, `role_names`, and `facets` to paths relative to
    the build directory. `ingestion` is the successful `IngestionHistory`
    row the build's import key names, which the origin catalogue's manifest records
    too, and `None` for a build written before manifests recorded it.
    `format_version` defaults to the current format.
    """

    stamp: str
    build_id: str
    built_at: str
    source: str
    row_counts: dict[str, int]
    files: dict[str, str]
    horizon: YearHorizon
    ingestion: Ingestion | None = None
    format_version: int = COVERAGE_FORMAT_VERSION


__all__ = [
    "ABSENT_ENTITY",
    "BUILD_EPOCH_YEAR",
    "COMPATIBLE_PREFIX",
    "COVERAGE_FORMAT_VERSION",
    "DONOR_COLUMN",
    "FACET_COLUMN",
    "FIRST_YEAR_COLUMN",
    "FORMAT_ROOT",
    "LAST_YEAR_COLUMN",
    "MASK_WORD_BITS",
    "N_OBS_COLUMN",
    "POINTER_FILE",
    "RECIPIENT_COLUMN",
    "SERVING_PREFIX",
    "VARIABLE_COLUMN",
    "VERSIONS_PREFIX",
    "CoverageManifest",
    "CoveragePointer",
    "FacetMetadata",
    "RoleNames",
    "YearHorizon",
    "year_mask_column",
]
