"""In-memory lookups over a loaded coverage artifact.

Rows are sorted by `(variable, facet_id)`, so each variable's rows form one
contiguous block. Lookups find each requested variable's block by binary
search, filter bound roles with Arrow over those rows alone, then apply year
windows through uint64 masks. `PairRows` uses the same binary search to read
only the requested group. Load-time validation guarantees `n_obs` matches
each row's mask. `role_names` maps package role labels to graph properties.
A variable that declares one role carries `ABSENT_ENTITY` in the column of
the role it lacks, which no lookup returns as a dcid.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar.rolling import ANCHOR_SHARE

from dc_sdmx import DEFAULT_PLANNING_LANES, plan_request_chunks
from dfo_platform.roles import RoleAxis
from materialise.roles import axis_for_role
from origin_catalogue.context import current_catalogue, current_source_scope
from series_coverage._bits import bit_length64, lowest_set_bit_index64, popcount64
from series_coverage.coverage_types import (
    BoundRoles,
    Estimate,
    FacetCoverage,
    FacetSpan,
    Headline,
    Role,
    ServedExtent,
    VariableCoverage,
    YearCount,
)
from series_coverage.schema import (
    ABSENT_ENTITY,
    DONOR_COLUMN,
    FACET_COLUMN,
    FIRST_YEAR_COLUMN,
    LAST_YEAR_COLUMN,
    MASK_WORD_BITS,
    N_OBS_COLUMN,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)
from series_coverage.summary import CoverageSummary

_NO_BINDING: BoundRoles = MappingProxyType({})
_NO_FACET_METADATA: Mapping[str, FacetMetadata] = MappingProxyType({})
# CurrencyUnit dcids are `Currency` plus an upper-case ISO 4217 code (`CurrencyUSD`).
_CURRENCY_UNIT = re.compile(r"Currency([A-Z]{3})")
# The calendar years `Headline.recent_years` counts and `elected_year` elects among, the
# latest year with data included.
_HEADLINE_YEARS = 3


def elected_year(series_by_year: Mapping[int, int]) -> int | None:
    """The year a `latest` read of one variable serves, from how many of its series hold
    each year: of the three calendar years ending at the newest year any series holds, the
    newest one held by at least `ANCHOR_SHARE` of the series the fullest of the three
    holds. A year `series_by_year` omits holds none. None when no series holds a year. A
    caller leaves out the years it does not elect among, as `CoverageIndex.elected_years`
    leaves out those after the index's build year.

    The counts are of the series a read holds before any `latest` reduction. The series
    left by a reduction to one year elect another: of 44, 85 and 91 series the rule elects
    the middle year, and the 85 holding it, counted alone, elect the newest."""
    newest = max((year for year, series in series_by_year.items() if series > 0), default=None)
    if newest is None:
        return None
    recent = {
        year: series_by_year.get(year, 0) for year in range(newest, newest - _HEADLINE_YEARS, -1)
    }
    floor = ANCHOR_SHARE * max(recent.values())
    return next(year for year, series in recent.items() if series >= floor)


@dataclass(frozen=True, slots=True)
class ProvenanceShape:
    """The observation periods and flow axes of one provenance's coverage series.

    `observation_periods` holds each facet's ISO 8601 period, with `None`
    for a facet that has none. `role_axes` holds one set per variable: the
    flow axes of its declared roles (one axis for a one-role variable), with
    `None` for a role that `materialise.roles.axis_for_role` places on no
    axis.
    """

    observation_periods: frozenset[str | None]
    role_axes: frozenset[frozenset[RoleAxis | None]]


def _year_counts_from_masks(
    masks_by_word: Sequence[Sequence[int]], *, horizon: YearHorizon
) -> dict[int, int]:
    """Counts rows per represented year across each mask word.

    Zero-count years are omitted. No rows yield an empty mapping.
    """
    counts: dict[int, int] = {}
    offsets = np.arange(MASK_WORD_BITS, dtype=np.uint64)
    for word, masks in enumerate(masks_by_word):
        if not masks:
            continue
        arr = np.array(masks, dtype=np.uint64)
        per_bit = ((arr[:, None] >> offsets[None, :]) & np.uint64(1)).astype(bool).sum(axis=0)
        word_start = horizon.epoch_year + MASK_WORD_BITS * word
        counts.update(
            {word_start + offset: int(count) for offset, count in enumerate(per_bit) if count}
        )
    return counts


def _mask_matrix(table: pa.Table, horizon: YearHorizon) -> np.ndarray:
    """`table`'s year-mask words as an `(rows, mask_words)` uint64 array."""
    if table.num_rows == 0:
        return np.zeros((0, horizon.mask_words), dtype=np.uint64)
    return np.column_stack(
        [
            table[name].to_numpy(zero_copy_only=False).astype(np.uint64)
            for name in horizon.mask_columns
        ]
    )


def _window_words(
    horizon: YearHorizon, *, first_year: int | None, last_year: int | None
) -> np.ndarray:
    """`horizon.window_masks` as a uint64 array, one element per word."""
    return np.array(
        horizon.window_masks(first_year=first_year, last_year=last_year), dtype=np.uint64
    )


_ROLE_COLUMNS: dict[Role, str] = {"donor": DONOR_COLUMN, "recipient": RECIPIENT_COLUMN}
# The flow axis each party column holds (`series_coverage.schema`).
_ROLE_AXES: dict[Role, RoleAxis] = {"donor": "origin", "recipient": "destination"}

type SeriesKey = Literal["variable", "facet_id", "donor", "recipient"]
"""A column of a series' key (`_SERIES_KEY_COLUMNS`)."""

# Output dataclasses live in `coverage_types.py`, so type-only consumers
# avoid importing this module's numpy-dependent bit arithmetic.


# One series' key (`series_coverage.schema`'s own module docstring): the
# columns `CoverageIndex` holds dictionary-encoded, and the grain at which
# every row is unique.
_SERIES_KEY_COLUMNS: tuple[str, ...] = (
    VARIABLE_COLUMN,
    FACET_COLUMN,
    DONOR_COLUMN,
    RECIPIENT_COLUMN,
)


def _codes(column: pa.ChunkedArray) -> np.ndarray:
    """A one-chunk dictionary column's codes, as a numpy view over the
    chunk's own index buffer (no copy)."""
    return cast("pa.DictionaryArray", column.chunk(0)).indices.to_numpy()


def _decode(dictionary: pa.Array, codes: np.ndarray) -> list[str]:
    """Returns the strings selected by `codes`, converting only `codes.size` values."""
    return cast("list[str]", dictionary.take(codes).to_pylist())


def _sort_dictionary(column: pa.ChunkedArray) -> pa.DictionaryArray:
    """Returns one array with a sorted dictionary of distinct values, so codes
    follow string order and each value has one code. A dictionary that
    repeats a value maps every code holding it to that one code."""
    encoded = cast("pa.DictionaryArray", column.combine_chunks())
    distinct = pc.unique(encoded.dictionary)
    ordered = distinct.take(pc.array_sort_indices(distinct))
    rank = pc.index_in(encoded.dictionary, value_set=ordered).to_numpy()
    return pa.DictionaryArray.from_arrays(pa.array(rank[encoded.indices.to_numpy()]), ordered)


def _dictionary_encode_keys(table: pa.Table) -> pa.Table:
    """Dictionary-encodes key columns over sorted dictionaries of distinct
    values, unifying dictionaries first."""
    for name in _SERIES_KEY_COLUMNS:
        column = table[name]
        if not pa.types.is_dictionary(column.type):
            column = pc.dictionary_encode(column)
        index = table.schema.get_field_index(name)
        field = pa.field(name, column.type, nullable=table.schema.field(index).nullable)
        table = table.set_column(index, field, column)
    table = table.unify_dictionaries()
    for name in _SERIES_KEY_COLUMNS:
        index = table.schema.get_field_index(name)
        table = table.set_column(index, table.schema.field(index), _sort_dictionary(table[name]))
    return table


def _group_rows(table: pa.Table) -> pa.Table:
    """Combines chunks and stably sorts by `(variable, facet_id)` when
    needed, keeping each variable and pair group contiguous for binary search.
    """
    if table.num_rows == 0:
        return table.combine_chunks()
    facet_count = len(cast("pa.DictionaryArray", table[FACET_COLUMN].chunk(0)).dictionary)
    keys = _codes(table[VARIABLE_COLUMN]).astype(np.int64) * facet_count + _codes(
        table[FACET_COLUMN]
    )
    if bool(np.all(keys[:-1] <= keys[1:])):
        return table.combine_chunks()
    return table.take(np.argsort(keys, kind="stable"))


def _run(codes: np.ndarray, code: int) -> tuple[int, int]:
    """The `[start, stop)` positions `code` occupies in `codes`, a
    non-decreasing code array, by binary search. The needle takes `codes`'
    own dtype: a Python `int` needle makes `np.searchsorted` cast the whole
    int32 array to int64 first, a copy the size of the array per call."""
    needle = codes.dtype.type(code)
    return (
        int(np.searchsorted(codes, needle, side="left")),
        int(np.searchsorted(codes, needle, side="right")),
    )


def _code_lookup(column: pa.ChunkedArray) -> dict[str, int]:
    """Each value of a one-chunk dictionary column's dictionary -> its
    code. O(distinct values)."""
    dictionary = cast("pa.DictionaryArray", column.chunk(0)).dictionary
    return {name: code for code, name in enumerate(cast("list[str]", dictionary.to_pylist()))}


def _key_values(column: pa.ChunkedArray) -> list[str]:
    """Converts each dictionary value the rows use once per chunk, then
    indexes by row code.

    Over a 21,954-value dictionary, 1.6M rows take 0.02 s per key column
    versus 0.9 s for `to_pylist`, which creates a Python string per row, and
    10 rows take 0.03 ms.
    """
    values: list[str] = []
    for chunk in column.chunks:
        if not pa.types.is_dictionary(chunk.type):
            values.extend(cast("list[str]", chunk.to_pylist()))
            continue
        encoded = cast("pa.DictionaryArray", chunk)
        indices = encoded.indices.to_numpy(zero_copy_only=False)
        present = np.zeros(len(encoded.dictionary), dtype=bool)
        present[indices] = True
        used = np.flatnonzero(present)
        lookup = np.empty(len(encoded.dictionary), dtype=object)
        lookup[used] = _decode(encoded.dictionary, used)
        values.extend(lookup[indices].tolist())
    return values


def _key_is_in(column: pa.ChunkedArray, values: Iterable[str]) -> pa.ChunkedArray:
    """Returns a row mask for `values`, testing dictionary entries once per code.

    On 1.78M rows, this takes 3 ms without allocation. `pc.is_in` on the
    dictionary column takes 20 ms and 135 MB because it decodes every row.
    """
    value_set = pa.array(list(values), type=pa.string())
    chunks = []
    for chunk in column.chunks:
        if pa.types.is_dictionary(chunk.type):
            encoded = cast("pa.DictionaryArray", chunk)
            chunks.append(
                pc.take(pc.is_in(encoded.dictionary, value_set=value_set), encoded.indices)
            )
        else:
            chunks.append(pc.is_in(chunk, value_set=value_set))
    return pa.chunked_array(chunks, type=pa.bool_())


def _int_values(column: pa.ChunkedArray) -> list[int]:
    """`column`'s non-null integer values as `int`, one per row."""
    return cast("list[int]", column.to_numpy().tolist())


def _apply_window(
    table: pa.Table, *, horizon: YearHorizon, first_year: int | None, last_year: int | None
) -> pa.Table:
    """Recomputes masks and year counts within the window, dropping rows with
    no matching observations. Two open bounds return `table` unchanged.

    Bounds clamp to the horizon. A window with no overlap returns zero rows.
    """
    if first_year is None and last_year is None:
        return table
    if table.num_rows == 0:
        return table
    wmasks = _window_words(horizon, first_year=first_year, last_year=last_year)
    if not wmasks.any():
        return table.filter(pa.array([False] * table.num_rows))
    windowed = _mask_matrix(table, horizon) & wmasks
    nonzero = windowed != 0
    survives = nonzero.any(axis=1)
    windowed = windowed[survives]
    table = table.filter(pa.array(survives))
    if windowed.shape[0] == 0:
        return table
    n_obs = popcount64(windowed).sum(axis=1)
    first_years, last_years = _year_bounds(windowed, horizon)
    replacements = {
        N_OBS_COLUMN: pa.array(n_obs, type=pa.uint32()),
        FIRST_YEAR_COLUMN: pa.array(first_years, type=pa.uint16()),
        LAST_YEAR_COLUMN: pa.array(last_years, type=pa.uint16()),
        **{
            name: pa.array(windowed[:, word], type=pa.uint64())
            for word, name in enumerate(horizon.mask_columns)
        },
    }
    for name, column in replacements.items():
        table = table.set_column(table.schema.get_field_index(name), name, column)
    return table


def _year_bounds(masks: np.ndarray, horizon: YearHorizon) -> tuple[np.ndarray, np.ndarray]:
    """Each row's first and last year with data, from its `(rows,
    mask_words)` year masks. Every row must have a bit set."""
    nonzero = masks != 0
    rows = np.arange(masks.shape[0])
    first_word = nonzero.argmax(axis=1)
    last_word = masks.shape[1] - 1 - nonzero[:, ::-1].argmax(axis=1)
    first_offsets = lowest_set_bit_index64(masks[rows, first_word]).astype(np.int64)
    last_offsets = bit_length64(masks[rows, last_word]).astype(np.int64) - 1
    return (
        horizon.epoch_year + MASK_WORD_BITS * first_word + first_offsets,
        horizon.epoch_year + MASK_WORD_BITS * last_word + last_offsets,
    )


def _latest_year_only(masks: np.ndarray) -> np.ndarray:
    """Each row of `(rows, mask_words)` year masks keeping only its latest year's bit.
    Every row must have a bit set."""
    rows = np.arange(masks.shape[0])
    last_word = masks.shape[1] - 1 - (masks != 0)[:, ::-1].argmax(axis=1)
    top = bit_length64(masks[rows, last_word]).astype(np.uint64) - np.uint64(1)
    latest = np.zeros_like(masks)
    latest[rows, last_word] = np.uint64(1) << top
    return latest


@dataclass(frozen=True, slots=True)
class _ProvenanceViews:
    currencies: Mapping[str, frozenset[str]]
    shapes: Mapping[str, ProvenanceShape]


def _provenance_views(
    table: pa.Table,
    role_names: Mapping[str, RoleNames],
    facet_metadata: Mapping[str, FacetMetadata],
) -> _ProvenanceViews:
    """Each provenance's currency codes and shape, from one pass over the
    table's distinct `(variable, facet_id)` pairs.

    A pair counts when its facet has a provenance and its variable has role
    names. Both views read that one population, so every provenance with a
    currency entry also has a shape.
    """
    pairs = (
        table.select([VARIABLE_COLUMN, FACET_COLUMN])
        .group_by([VARIABLE_COLUMN, FACET_COLUMN], use_threads=False)
        .aggregate([])
    )
    codes: dict[str, set[str]] = {}
    periods: dict[str, set[str | None]] = {}
    axes: dict[str, set[frozenset[RoleAxis | None]]] = {}
    for variable, facet_id in zip(
        _key_values(pairs[VARIABLE_COLUMN]), _key_values(pairs[FACET_COLUMN]), strict=True
    ):
        metadata = facet_metadata.get(facet_id)
        names = role_names.get(variable)
        if metadata is None or metadata.provenance_id is None or names is None:
            continue
        provenance = metadata.provenance_id
        periods.setdefault(provenance, set()).add(metadata.observation_period)
        axes.setdefault(provenance, set()).add(
            frozenset(axis_for_role(role) for role in names.declared)
        )
        if (
            metadata.unit is not None
            and (match := _CURRENCY_UNIT.fullmatch(metadata.unit)) is not None
        ):
            codes.setdefault(provenance, set()).add(match.group(1))
    return _ProvenanceViews(
        currencies=MappingProxyType(
            {provenance: frozenset(found) for provenance, found in codes.items()}
        ),
        shapes=MappingProxyType(
            {
                provenance: ProvenanceShape(
                    observation_periods=frozenset(found), role_axes=frozenset(axes[provenance])
                )
                for provenance, found in periods.items()
            }
        ),
    )


_SERIES_COLUMNS = (VARIABLE_COLUMN, RECIPIENT_COLUMN, DONOR_COLUMN)


@dataclass(frozen=True, slots=True)
class KeptFacets:
    """The facets one variable's kept rows sit in (`SeriesYears.variables`).

    `groups` counts their collapse groups, the distinct `(provenance, method, unit)` of
    their metadata, the grain at which materialise collapses facets
    (`materialise.facet_collapse`) and a per-series read elects one group per member.
    `sub_annual` is whether one of them states an observation period shorter than a
    year, whose periods inside one year the index cannot order.
    """

    facet_ids: frozenset[str]
    groups: int
    sub_annual: bool


@dataclass(frozen=True, slots=True)
class SeriesYears:
    """The series a read serves and the years each serves (`CoverageIndex.series_years`).

    One entry per `(variable, recipient, donor)` series, in one order: `masks` holds its
    `(series, mask_words)` uint64 year masks on the index's horizon, and `key_rows` its
    key columns, which `keys` decodes. `variables` maps each variable to the facets its
    kept rows sit in, before `latest` reduces them.
    """

    masks: np.ndarray
    key_rows: pa.Table
    variables: Mapping[str, KeptFacets]

    @property
    def series(self) -> int:
        """How many series the read serves."""
        return int(self.masks.shape[0])

    @property
    def rows(self) -> int:
        """The points the read serves: each series' distinct years, summed."""
        return int(popcount64(self.masks).sum())

    def keys(self) -> list[tuple[str, str, str]]:
        """Each series' `(variable, recipient, donor)`, in order."""
        columns = (_key_values(self.key_rows[column]) for column in _SERIES_COLUMNS)
        return list(zip(*columns, strict=True))

    def year_bits(self) -> list[int]:
        """Each series' year mask as one integer, word `w` at bit `64 * w`, in order."""
        bits = [0] * self.series
        for word in range(self.masks.shape[1]):
            shift = MASK_WORD_BITS * word
            for i, value in enumerate(self.masks[:, word].tolist()):
                bits[i] |= value << shift
        return bits


type KeptYears = Callable[[str, str], Collection[int] | None]
"""The years a series keeps, given its recipient and donor dcids (`ABSENT_ENTITY` for a role
it does not bind), None for every year it serves (`CoverageIndex.series_years`)."""


# A whole number of years (`P1Y`, `P5Y`); any other stated period is shorter than a year.
_YEARLY_PERIOD = re.compile(r"P\d+Y")


def _sub_annual(metadata: FacetMetadata | None) -> bool:
    """Whether `metadata` states an observation period shorter than a year."""
    period = metadata.observation_period if metadata is not None else None
    return period is not None and _YEARLY_PERIOD.fullmatch(period) is None


def _at_variable_latest(masks: np.ndarray, variable_codes: np.ndarray) -> np.ndarray:
    """Whether each series' latest-year-only mask (`_latest_year_only`) holds its
    variable's latest year across `masks`. `variable_codes` gives each series' variable,
    the series grouped by it."""
    words = masks.shape[1]
    last_word = words - 1 - (masks != 0)[:, ::-1].argmax(axis=1)
    rows = np.arange(masks.shape[0])
    bit = last_word.astype(np.int64) * MASK_WORD_BITS + bit_length64(masks[rows, last_word]).astype(
        np.int64
    )
    starts = np.flatnonzero(np.r_[True, variable_codes[1:] != variable_codes[:-1]])
    latest = np.maximum.reduceat(bit, starts)
    return bit == np.repeat(latest, np.diff(np.r_[starts, masks.shape[0]]))


class CoverageIndex:
    def __init__(
        self,
        table: pa.Table,
        role_names: Mapping[str, RoleNames],
        facet_metadata: Mapping[str, FacetMetadata] = _NO_FACET_METADATA,
        *,
        horizon: YearHorizon,
        build_year: int | None = None,
    ) -> None:
        """Dictionary-encodes key columns and sorts rows by `(variable, facet_id)`.

        `build_year` is the calendar year the artifact was built in, the last year a
        `latest` read elects (`elected_years`). None sets no last year.

        Raises `ValueError` if mask columns differ from `horizon` or a
        `(variable, facet_id, donor, recipient)` key repeats.
        """
        mask_columns = tuple(name for name in table.column_names if name.startswith("year_mask"))
        if mask_columns != horizon.mask_columns:
            raise ValueError(
                f"coverage table carries year-mask columns {list(mask_columns)}, but its "
                f"horizon ({horizon.mask_words} word(s) from {horizon.epoch_year}) names "
                f"{list(horizon.mask_columns)}"
            )
        self._horizon = horizon
        self._build_year = build_year
        self._table = _group_rows(_dictionary_encode_keys(table))
        self._variable_codes = _code_lookup(self._table[VARIABLE_COLUMN])
        # Non-decreasing after `_group_rows`, so `_run` finds a variable's rows.
        self._variable_code_column = _codes(self._table[VARIABLE_COLUMN])
        self._facet_codes = _code_lookup(self._table[FACET_COLUMN])
        distinct = self._table.group_by(list(_SERIES_KEY_COLUMNS)).aggregate([]).num_rows
        if distinct != self._table.num_rows:
            raise ValueError(
                f"coverage table has {self._table.num_rows - distinct} row(s) repeating a "
                f"(variable, facet_id, donor, recipient) key already present; the building "
                "job writes one row per series"
            )
        self._role_names = dict(role_names)
        self._facet_metadata = dict(facet_metadata)
        views = _provenance_views(self._table, self._role_names, self._facet_metadata)
        self._provenance_currencies = views.currencies
        self._provenance_shapes = views.shapes
        self._facet_summary = CoverageSummary(
            self._table, variable_codes=self._variable_codes, facet_metadata=self._facet_metadata
        )

    @property
    def build_year(self) -> int | None:
        """The calendar year the artifact was built in, from its manifest, or None for an
        index built without one. A `latest` read elects no later year
        (`elected_years`), since a year after it holds projections."""
        return self._build_year

    def _to_build_year(self, masks: np.ndarray) -> np.ndarray:
        """`masks` without the years after `build_year`."""
        return masks & _window_words(self._horizon, first_year=None, last_year=self._build_year)

    @property
    def horizon(self) -> YearHorizon:
        """The years this artifact's year masks represent, from its
        manifest. A caller clamping a requested window to what coverage can
        confirm reads `horizon.epoch_year`/`horizon.max_year`."""
        return self._horizon

    @property
    def role_names(self) -> Mapping[str, RoleNames]:
        """Maps variables to their loaded donor and recipient role names.

        Every covered variable has an entry. The job may also record role
        names for variables without coverage rows.
        """
        return self._role_names

    @property
    def facet_metadata(self) -> Mapping[str, FacetMetadata]:
        """Loaded facet metadata, used to rank this index's coverage results.

        Real artifacts provide complete metadata. Direct construction
        defaults to an empty map.
        """
        return self._facet_metadata

    @property
    def provenance_currencies(self) -> Mapping[str, frozenset[str]]:
        """Maps each provenance id to the ISO 4217 codes of its covered facets' currency units.

        Built when the index loads, in the same pass and from the same
        distinct `(variable, facet_id)` pairs as `provenance_shapes`, so a
        facet counts only when coverage rows use it and its variable has role
        names. A unit counts when its dcid is `Currency` plus an upper-case
        three-letter code (`CurrencyUSD` gives `USD`). Other units and facets
        without a provenance or unit are skipped. A provenance with no covered
        currency facets has no entry, and every provenance with an entry also
        has one in `provenance_shapes`.
        """
        return self._provenance_currencies

    @property
    def provenance_shapes(self) -> Mapping[str, ProvenanceShape]:
        """Maps each provenance id to the periods and flow axes of its coverage series.

        Built when the index loads from the distinct `(variable, facet_id)`
        pairs of the coverage rows. A facet's provenance and period come from
        `facet_metadata`, and a variable's axes come from `role_names`. Pairs
        whose facet has no provenance, or whose variable has no role names,
        are skipped. A provenance with no coverage rows has no entry, which
        means unknown, because the building job keeps only role-bearing series.
        """
        return self._provenance_shapes

    @property
    def facet_summary(self) -> CoverageSummary:
        """Each variable's facets across the build, with their metadata, years, observation
        counts and parties by role, built when the index loads so a reload swaps it with
        the rows (`series_coverage.summary`)."""
        return self._facet_summary

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    def variable_provenances(self) -> dict[str, tuple[str, ...]]:
        """Maps each covered variable to the sorted provenances of its coverage facets.

        Read from the distinct `(variable, facet_id)` pairs of the coverage rows through
        `facet_metadata`. A facet with no metadata entry or no provenance adds none, so a
        variable whose facets all lack one maps to `()`.
        """
        pairs = (
            self._table.select([VARIABLE_COLUMN, FACET_COLUMN])
            .group_by([VARIABLE_COLUMN, FACET_COLUMN], use_threads=False)
            .aggregate([])
        )
        found: dict[str, set[str]] = {}
        for variable, facet_id in zip(
            _key_values(pairs[VARIABLE_COLUMN]), _key_values(pairs[FACET_COLUMN]), strict=True
        ):
            provenances = found.setdefault(variable, set())
            metadata = self._facet_metadata.get(facet_id)
            if metadata is not None and metadata.provenance_id is not None:
                provenances.add(metadata.provenance_id)
        return {variable: tuple(sorted(provenances)) for variable, provenances in found.items()}

    def dataset_parties(self, provenances: Collection[str]) -> dict[str, frozenset[RoleAxis]]:
        """Every donor and recipient on a row with an observation whose facet carries one of
        `provenances`, with the flow axes it is published on: `origin` as a donor,
        `destination` as a recipient.

        A one-role variable's absent role is no party. A provenance with no facet in the build
        adds none.
        """
        facets = {
            facet_id
            for facet_id, metadata in self._facet_metadata.items()
            if metadata.provenance_id in provenances
        }
        axes: dict[str, set[RoleAxis]] = {}
        for role, axis in _ROLE_AXES.items():
            by_facet = self.grouped(by=(role, "facet_id"))
            for party, facet_id in zip(
                _key_values(by_facet[role]), _key_values(by_facet[FACET_COLUMN]), strict=True
            ):
                if facet_id in facets and party != ABSENT_ENTITY:
                    axes.setdefault(party, set()).add(axis)
        return {party: frozenset(found) for party, found in axes.items()}

    def _member_rows(self, *, variables: Sequence[str], bound: BoundRoles) -> pa.Table:
        """The rows of `variables` surviving `bound`, in table order.

        Takes each known variable's row block in code order, then filters
        bound roles over those rows, so the cost follows the variables'
        rows rather than the table. Unknown variables contribute no rows.
        """
        codes = sorted({self._variable_codes[v] for v in variables if v in self._variable_codes})
        blocks = [np.arange(*_run(self._variable_code_column, code)) for code in codes]
        rows = self._table.take(np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64))
        for role, column in _ROLE_COLUMNS.items():
            if role in bound:
                rows = rows.filter(_key_is_in(rows[column], bound[role]))
        return self._admit_scoped_rows(rows)

    def _candidate_member_rows(self, *, variables: Sequence[str], bound: BoundRoles) -> pa.Table:
        """Rows that can validate a facet choice, independently of request scope.

        A pinned facet may replace the request's current provenance scope, so validating
        that pin must see every globally admitted provenance at the bound role tuples.
        This deliberately skips per-series source admission, which remains a
        served-read concern in :meth:`_member_rows`.
        """
        codes = sorted({self._variable_codes[v] for v in variables if v in self._variable_codes})
        blocks = [np.arange(*_run(self._variable_code_column, code)) for code in codes]
        rows = self._table.take(np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64))
        for role, column in _ROLE_COLUMNS.items():
            if role in bound:
                rows = rows.filter(_key_is_in(rows[column], bound[role]))
        catalogue = current_catalogue()
        admitted = [
            (metadata := self._facet_metadata.get(facet_id)) is not None
            and (metadata.provenance_id is None or catalogue.is_admitted(metadata.provenance_id))
            for facet_id in _key_values(rows[FACET_COLUMN])
        ]
        return rows.filter(pa.array(admitted, type=pa.bool_()))

    def _admit_scoped_rows(self, rows: pa.Table) -> pa.Table:
        """Keeps only facets the request's source-scoped catalogue admits per series.

        Coverage rows are role-bearing.  A source is admitted at the grain at
        which it is served: a complete `(variable, recipient, donor)` series.
        Applying that rule after role bindings preserves a named request's
        bound pairs.  Every provenance the scoped catalogue admits is kept,
        whichever origin it comes from, and the facet election chooses between
        DFO and base (`materialise.facets.dfo_precedence`).
        """
        if rows.num_rows == 0 or current_source_scope() is None:
            return rows

        rows = rows.combine_chunks()
        variables = _key_values(rows[VARIABLE_COLUMN])
        recipients = _key_values(rows[RECIPIENT_COLUMN])
        donors = _key_values(rows[DONOR_COLUMN])
        facet_ids = _key_values(rows[FACET_COLUMN])
        facets_by_series: dict[tuple[str, str, str], set[str]] = {}
        for variable, recipient, donor, facet_id in zip(
            variables, recipients, donors, facet_ids, strict=True
        ):
            facets_by_series.setdefault((variable, recipient, donor), set()).add(facet_id)

        catalogue = current_catalogue()
        admitted_by_series: dict[tuple[str, str, str], frozenset[str]] = {}
        for series, series_facet_ids in facets_by_series.items():
            provenances = [
                metadata.provenance_id
                for facet_id in series_facet_ids
                if (metadata := self._facet_metadata.get(facet_id)) is not None
                and metadata.provenance_id is not None
            ]
            admitted_by_series[series] = frozenset(catalogue.admit_series(provenances).admitted)
        admitted = [
            (
                (metadata := self._facet_metadata.get(facet_id)) is not None
                and metadata.provenance_id in admitted_by_series[(variable, recipient, donor)]
            )
            for variable, recipient, donor, facet_id in zip(
                variables, recipients, donors, facet_ids, strict=True
            )
        ]
        return rows.filter(pa.array(admitted, type=pa.bool_()))

    def admitted_pairs(
        self, *, variable: str, bound: BoundRoles = _NO_BINDING
    ) -> Mapping[str, frozenset[tuple[str, str]]] | None:
        """The source-admitted donor/recipient pairs per facet for one variable.

        ``None`` is the unscoped sentinel.  An empty mapping means a scoped
        request admitted no coverage rows, which must differ from serving an
        unscoped response.
        """
        if current_source_scope() is None:
            return None
        rows = self._member_rows(variables=[variable], bound=bound)
        admitted: dict[str, set[tuple[str, str]]] = {}
        for facet_id, donor, recipient in zip(
            _key_values(rows[FACET_COLUMN]),
            _key_values(rows[DONOR_COLUMN]),
            _key_values(rows[RECIPIENT_COLUMN]),
            strict=True,
        ):
            admitted.setdefault(facet_id, set()).add((donor, recipient))
        return {facet_id: frozenset(pairs) for facet_id, pairs in admitted.items()}

    def _served_row_mask(
        self, rows: pa.Table, served: Iterable[tuple[str, str, str | None, str | None]]
    ) -> np.ndarray:
        """Whether each of `rows` matches a `served` pattern. A pattern naming no role tests
        one combined (variable, facet) code per row; a pattern naming a recipient, or a
        recipient and a donor, also tests those rows' dcids."""
        width = len(self._facet_codes) + 1
        any_pair: list[int] = []
        by_pair: set[tuple[int, str | None, str | None]] = set()
        for variable, facet_id, recipient, donor in served:
            variable_code = self._variable_codes.get(variable)
            facet_code = self._facet_codes.get(facet_id)
            if variable_code is None or facet_code is None:
                continue
            key = variable_code * width + facet_code
            if recipient is None and donor is None:
                any_pair.append(key)
            else:
                by_pair.add((key, recipient, donor))
        rows = rows.combine_chunks()
        keys = _codes(rows[VARIABLE_COLUMN]).astype(np.int64) * width + _codes(
            rows[FACET_COLUMN]
        ).astype(np.int64)
        mask = np.isin(keys, np.array(any_pair, dtype=np.int64))
        if by_pair:
            recipients = _key_values(rows[RECIPIENT_COLUMN])
            donors = _key_values(rows[DONOR_COLUMN])
            for row, key in enumerate(keys.tolist()):
                if mask[row]:
                    continue
                mask[row] = any(
                    (key, recipient, donor) in by_pair
                    for recipient in (recipients[row], None)
                    for donor in (donors[row], None)
                )
        return mask

    def _excluded_row_mask(
        self, rows: pa.Table, excluded: Iterable[tuple[str, Role, str]]
    ) -> np.ndarray:
        """Whether each of `rows` binds, for its variable, an `excluded` dcid on that role.
        Each role's dictionary is tested once per variable naming it."""
        by_role: dict[Role, dict[int, set[str]]] = {}
        for variable, role, dcid in excluded:
            if (code := self._variable_codes.get(variable)) is not None:
                by_role.setdefault(role, {}).setdefault(code, set()).add(dcid)
        mask = np.zeros(rows.num_rows, dtype=bool)
        if not by_role:
            return mask
        rows = rows.combine_chunks()
        variables = _codes(rows[VARIABLE_COLUMN])
        for role, by_variable in by_role.items():
            column = rows[_ROLE_COLUMNS[role]]
            for code, dcids in by_variable.items():
                hit = _key_is_in(column, dcids).to_numpy(zero_copy_only=False)
                mask |= (variables == code) & hit
        return mask

    def _filtered(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles,
        first_year: int | None,
        last_year: int | None,
    ) -> pa.Table:
        filtered = self._member_rows(variables=variables, bound=bound)
        return _apply_window(
            filtered, horizon=self._horizon, first_year=first_year, last_year=last_year
        )

    def filter_candidates(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> list[VariableCoverage]:
        """Returns variables with surviving rows and their facet coverage.

        `year_counts` sums each facet's row masks after window clipping.
        Variables and facets with no rows in the window are omitted.
        """
        filtered = self._filtered(
            variables=variables, bound=bound, first_year=first_year, last_year=last_year
        )
        if filtered.num_rows == 0:
            return []
        by_variable: dict[str, list[FacetCoverage]] = {}
        for (variable,), facet in self._facet_coverages(filtered, role_columns=()):
            by_variable.setdefault(variable, []).append(facet)
        result: list[VariableCoverage] = []
        for variable, facets in by_variable.items():
            facets_sorted = tuple(sorted(facets, key=lambda f: f.facet_id))
            result.append(
                VariableCoverage(
                    variable=variable,
                    facets=facets_sorted,
                    n_obs=sum(f.n_obs for f in facets_sorted),
                    first_year=min(f.first_year for f in facets_sorted),
                    last_year=max(f.last_year for f in facets_sorted),
                )
            )
        return result

    def filter_facet_candidates(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> list[VariableCoverage]:
        """Return globally admitted facet candidates under role and year bounds.

        Unlike :meth:`filter_candidates`, this is for validating a prospective
        facet pin or listing replacements for one.  It ignores a request's active
        source scope and per-series DFO precedence, while still excluding unknown,
        globally unavailable, and disabled-UN provenances.
        """
        filtered = _apply_window(
            self._candidate_member_rows(variables=variables, bound=bound),
            horizon=self._horizon,
            first_year=first_year,
            last_year=last_year,
        )
        if filtered.num_rows == 0:
            return []
        by_variable: dict[str, list[FacetCoverage]] = {}
        for (variable,), facet in self._facet_coverages(filtered, role_columns=()):
            by_variable.setdefault(variable, []).append(facet)
        result: list[VariableCoverage] = []
        for variable, facets in by_variable.items():
            facets_sorted = tuple(sorted(facets, key=lambda f: f.facet_id))
            result.append(
                VariableCoverage(
                    variable=variable,
                    facets=facets_sorted,
                    n_obs=sum(f.n_obs for f in facets_sorted),
                    first_year=min(f.first_year for f in facets_sorted),
                    last_year=max(f.last_year for f in facets_sorted),
                )
            )
        return result

    def _pairs_with_data(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles,
        pair_roles: Sequence[Role],
        first_year: int | None,
        last_year: int | None,
        facet_candidates: bool = False,
    ) -> tuple[pa.Table, np.ndarray, np.ndarray]:
        """The rows of `variables` surviving `bound` over every year, each row's pair key (its
        variable and its dcid on each role in `pair_roles`, as one integer), and the distinct
        keys of the pairs with a row holding data inside `[first_year, last_year]`, every row
        when both bounds are open, since each row holds data. `pair_count` and
        `pair_facet_candidates` both select pairs here. `facet_candidates` reads the rows
        `filter_facet_candidates` reads (`_candidate_member_rows`) in place of the served
        rows (`_member_rows`)."""
        member_rows = self._candidate_member_rows if facet_candidates else self._member_rows
        rows = member_rows(variables=variables, bound=bound).combine_chunks()
        if rows.num_rows == 0:
            empty = np.empty(0, dtype=np.int64)
            return rows, empty, empty
        keys = np.zeros(rows.num_rows, dtype=np.int64)
        for column in (VARIABLE_COLUMN, *(_ROLE_COLUMNS[role] for role in pair_roles)):
            width = len(cast("pa.DictionaryArray", rows[column].chunk(0)).dictionary)
            keys = keys * width + _codes(rows[column]).astype(np.int64)
        held = keys
        if first_year is not None or last_year is not None:
            window = _window_words(self._horizon, first_year=first_year, last_year=last_year)
            held = keys[((_mask_matrix(rows, self._horizon) & window) != 0).any(axis=1)]
        # Arrow's hash-based distinct runs about twice as fast as `np.unique`'s sort here.
        return rows, keys, pc.unique(pa.array(held)).to_numpy()

    def pair_facet_candidates(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles,
        pair_roles: Sequence[Role],
        first_year: int | None = None,
        last_year: int | None = None,
        facet_candidates: bool = False,
    ) -> dict[tuple[str, tuple[str, ...]], tuple[FacetCoverage, ...]]:
        """Each pair's facet coverage over every year it holds, keyed by `(variable, dcids)`
        with one dcid per role in `pair_roles`, in that order, and facets sorted by
        `facet_id`, for the pairs holding data inside `[first_year, last_year]` (every pair
        with data when both bounds are open). A pair's facets are not clipped to the range.

        A pair is what a per-pair preview binds: its variable and one dcid on each role in
        `pair_roles`, with `bound` limiting those dcids. Rows on a role outside `pair_roles`
        pool into the pair, as `filter_candidates` pools them for a request binding that
        pair alone, so each value equals `filter_candidates(variables=[variable],
        bound=<the pair's dcids>)`'s facets from one grouped read. With `facet_candidates`,
        each value equals `filter_facet_candidates`' facets for the pair instead: every
        globally admitted facet, with no source scope or per-series precedence.
        """
        rows, keys, kept = self._pairs_with_data(
            variables=variables,
            bound=bound,
            pair_roles=pair_roles,
            first_year=first_year,
            last_year=last_year,
            facet_candidates=facet_candidates,
        )
        if kept.size == 0:
            return {}
        if not (in_kept := np.isin(keys, kept)).all():
            rows = rows.filter(pa.array(in_kept))
        by_pair: dict[tuple[str, tuple[str, ...]], list[FacetCoverage]] = {}
        role_columns = tuple(_ROLE_COLUMNS[role] for role in pair_roles)
        for (variable, *dcids), facet in self._facet_coverages(rows, role_columns=role_columns):
            by_pair.setdefault((variable, tuple(dcids)), []).append(facet)
        return {
            key: tuple(sorted(facets, key=lambda f: f.facet_id)) for key, facets in by_pair.items()
        }

    def pair_count(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles,
        pair_roles: Sequence[Role],
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> int:
        """How many pairs `pair_facet_candidates` returns for the same arguments: the pairs
        holding data inside `[first_year, last_year]`, counted without building any pair's
        facet coverage."""
        _rows, _keys, kept = self._pairs_with_data(
            variables=variables,
            bound=bound,
            pair_roles=pair_roles,
            first_year=first_year,
            last_year=last_year,
        )
        return int(kept.size)

    def series_counts(
        self,
        *,
        variable: str,
        bound: BoundRoles = _NO_BINDING,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> dict[str, int]:
        """How many series of `variable` each facet holds under `bound` with data inside
        `[first_year, last_year]`: one per row, a donor and recipient pair. Materialisation
        ranks a role-bearing member's facets by this count before their years and
        observations (`materialise.facets.rank_candidates`)."""
        rows = self._filtered(
            variables=[variable], bound=bound, first_year=first_year, last_year=last_year
        )
        counts: dict[str, int] = {}
        for facet_id in _key_values(rows[FACET_COLUMN]):
            counts[facet_id] = counts.get(facet_id, 0) + 1
        return counts

    def union_series_count(
        self,
        *,
        variable: str,
        facet_ids: Iterable[str],
        bound: BoundRoles = _NO_BINDING,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> int:
        """How many series of `variable` under `bound`, donor and recipient pairs, at least
        one of `facet_ids` holds with data inside `[first_year, last_year]`: the count a
        joined candidate ranks on (`materialise.facets.union_count`), as `series_counts`
        counts one facet's."""
        wanted = frozenset(facet_ids)
        rows = self._filtered(
            variables=[variable], bound=bound, first_year=first_year, last_year=last_year
        )
        return len(
            {
                (donor, recipient)
                for facet_id, donor, recipient in zip(
                    _key_values(rows[FACET_COLUMN]),
                    _key_values(rows[DONOR_COLUMN]),
                    _key_values(rows[RECIPIENT_COLUMN]),
                    strict=True,
                )
                if facet_id in wanted
            }
        )

    def facet_spans(
        self, *, variables: Sequence[str], bound: BoundRoles = _NO_BINDING
    ) -> dict[tuple[str, str], FacetSpan]:
        """Each `(variable, facet_id)`'s observation count, first and last years with data
        and first series' end under `bound`, over every year: the `n_obs`, `first_year`,
        `last_year` and `first_series_end` `filter_candidates` states for the same rows with
        no window, read from the row columns in one grouped pass that builds no year counts.
        A bulk election ranks and continues a facet across up to 1.6M rows by these, the
        values a mint reads (`materialise.facets.coverage_probe`)."""
        rows = self._member_rows(variables=variables, bound=bound)
        if rows.num_rows == 0:
            return {}
        grouped = rows.group_by([VARIABLE_COLUMN, FACET_COLUMN]).aggregate(
            (
                (N_OBS_COLUMN, "sum"),
                (FIRST_YEAR_COLUMN, "min"),
                (LAST_YEAR_COLUMN, "max"),
                (LAST_YEAR_COLUMN, "min"),
            )
        )
        return {
            (variable, facet_id): FacetSpan(
                n_obs=n_obs, first_year=first, last_year=last, first_series_end=series_end
            )
            for variable, facet_id, n_obs, first, last, series_end in zip(
                _key_values(grouped[VARIABLE_COLUMN]),
                _key_values(grouped[FACET_COLUMN]),
                _int_values(grouped[f"{N_OBS_COLUMN}_sum"]),
                _int_values(grouped[f"{FIRST_YEAR_COLUMN}_min"]),
                _int_values(grouped[f"{LAST_YEAR_COLUMN}_max"]),
                _int_values(grouped[f"{LAST_YEAR_COLUMN}_min"]),
                strict=True,
            )
        }

    def _facet_coverages(
        self, rows: pa.Table, *, role_columns: Sequence[str]
    ) -> list[tuple[tuple[str, ...], FacetCoverage]]:
        """`rows` grouped by variable, then `role_columns`, then facet, each group's masks
        summed into one `FacetCoverage`, keyed by its variable and role dcids, with the
        earliest of its rows' last years as the end of its first series to end."""
        # A tuple preserves each aggregate function's `Literal` type for
        # pyarrow-stubs.
        grouped = rows.group_by([VARIABLE_COLUMN, *role_columns, FACET_COLUMN]).aggregate(
            (
                *((name, "list") for name in self._horizon.mask_columns),
                (LAST_YEAR_COLUMN, "min"),
            )
        )
        # Arrow stubs mark `to_pylist()` values nullable. Writers populate
        # every non-nullable `YearHorizon.schema` field, as these casts record.
        key_columns = [_key_values(grouped[column]) for column in (VARIABLE_COLUMN, *role_columns)]
        facets_col = _key_values(grouped[FACET_COLUMN])
        words_by_column = [
            cast("list[list[int]]", grouped[f"{name}_list"].to_pylist())
            for name in self._horizon.mask_columns
        ]
        series_ends = _int_values(grouped[f"{LAST_YEAR_COLUMN}_min"])
        result: list[tuple[tuple[str, ...], FacetCoverage]] = []
        for row, facet_id in enumerate(facets_col):
            year_counts = _year_counts_from_masks(
                [words[row] for words in words_by_column], horizon=self._horizon
            )
            result.append(
                (
                    tuple(column[row] for column in key_columns),
                    FacetCoverage(
                        facet_id=facet_id,
                        n_obs=sum(year_counts.values()),
                        first_year=min(year_counts),
                        last_year=max(year_counts),
                        year_counts=year_counts,
                        first_series_end=series_ends[row],
                    ),
                )
            )
        return result

    def grouped(
        self, *, by: Sequence[SeriesKey], variables: Sequence[str] | None = None
    ) -> pa.Table:
        """One row per distinct combination of the `by` key columns among
        the rows with an observation, of `variables` or of every variable
        when `None`, in one pass over those rows.

        The `by` columns keep the index's dictionary encoding. `first_year`
        and `last_year` are the first and last year any row of the
        combination has data, read from the year masks.
        """
        rows = (
            self._table
            if variables is None
            else self._member_rows(variables=variables, bound=_NO_BINDING)
        )
        # A group's highest year bit is that of its largest mask, and its
        # lowest that of its smallest lowest-bit, so both come from Arrow
        # aggregates without a row-sized year column.
        columns: dict[str, pa.ChunkedArray] = {name: rows[name] for name in by}
        zero, one = pa.scalar(0, pa.uint64()), pa.scalar(1, pa.uint64())
        for name in self._horizon.mask_columns:
            mask = rows[name]
            lowest_bit = pc.bit_wise_and(mask, pc.add(pc.bit_wise_not(mask), one))
            columns[f"{name}_low"] = pc.if_else(pc.equal(mask, zero), None, lowest_bit)
            columns[f"{name}_high"] = mask
        spans = (
            pa.table(columns)
            .group_by(list(by), use_threads=False)
            .aggregate(
                tuple(
                    aggregate
                    for name in self._horizon.mask_columns
                    for aggregate in ((f"{name}_low", "min"), (f"{name}_high", "max"))
                )
            )
        )
        lows, highs = (
            np.column_stack(
                [
                    spans[f"{name}_{suffix}"].fill_null(zero).to_numpy().astype(np.uint64)
                    for name in self._horizon.mask_columns
                ]
            )
            for suffix in ("low_min", "high_max")
        )
        observed = (highs != 0).any(axis=1)
        first_years, _ = _year_bounds(lows[observed], self._horizon)
        _, last_years = _year_bounds(highs[observed], self._horizon)
        return pa.table(
            {
                **{name: spans[name].filter(pa.array(observed)) for name in by},
                FIRST_YEAR_COLUMN: pa.array(first_years, type=pa.uint16()),
                LAST_YEAR_COLUMN: pa.array(last_years, type=pa.uint16()),
            }
        )

    def pair_rows(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> PairRows:
        """Every row of `variables` surviving `bound` and the `[first_year,
        last_year]` window, held columnar as `PairRows`: one row per real
        (variable, facet_id, donor, recipient) series, with `n_obs`/
        `first_year`/`last_year`/`year_mask` window-clipped by
        `_apply_window`. `bound` filters rows the same way it does for
        `filter_candidates`, so a surviving row keeps its real dcid on the
        axis `bound` leaves open, and a role bound to an empty set matches
        nothing (`BoundRoles`).
        """
        return PairRows(
            self._filtered(
                variables=variables, bound=bound, first_year=first_year, last_year=last_year
            ),
            horizon=self._horizon,
            variable_codes=self._variable_codes,
            facet_codes=self._facet_codes,
        )

    def open_axis(
        self,
        *,
        variables: Sequence[str],
        axis: Role,
        bound: BoundRoles = _NO_BINDING,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> frozenset[str]:
        """Every distinct dcid on `axis` across the rows surviving `bound`
        and the window for `variables`. For example, every recipient with data,
        `axis="recipient"`, under `bound={"donor": frozenset({"country/FRA"})}`.
        Replaces one of `discover_role_scope`'s three `availability_values`
        reads (`role_scope.py`).
        """
        filtered = self._filtered(
            variables=variables, bound=bound, first_year=first_year, last_year=last_year
        )
        if filtered.num_rows == 0:
            return frozenset()
        found = cast("list[str]", pc.unique(filtered[_ROLE_COLUMNS[axis]]).to_pylist())
        return frozenset(found) - {ABSENT_ENTITY}

    def headline(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        served: Iterable[tuple[str, str, str | None, str | None]] | None = None,
        excluded: Iterable[tuple[str, Role, str]] = (),
        first_year: int | None = None,
        last_year: int | None = None,
        elected: Mapping[str, int] | None = None,
    ) -> Headline | None:
        """The latest year any row surviving `bound` has data inside `[first_year,
        last_year]` (unbounded by default, across a series' whole history), with the exact
        observation, series and per-role distinct-dcid counts in that year alone, and the
        same counts for it and the two calendar years before it (`recent_years`, newest
        first, a year with no data kept at zero). The window picks the latest year; the
        earlier years count whether or not the window reaches them. `recent_years` stops
        at the horizon's `epoch_year`, since the artifact holds no year before it. `None`
        when nothing survives `bound` and the window.

        `elected` is set for a `latest` read, which passes no window, and holds the year
        each variable elected (`elected_years`). Each variable then serves that year, or
        its own latest year when `elected` omits it. The headline's year is the newest any
        variable heads: the year it elected or, absent from `elected`, its latest year up
        to `build_year`, so a projection after `build_year` heads nothing. It is counted
        over the rows of the variables heading it, and a variable that holds the year and
        elected an earlier one counts for nothing there. With no variable holding a year
        that early it is the latest year any row holds. `recent_years`
        counts every row, and ends at the latest year any row holds up to `build_year`,
        the last year a variable elects, so with one variable the headline repeats the
        `recent_years` entry of its year. With no row holding a year that early it ends
        at the latest year any row holds.

        `served` keeps only the rows a preview serves, as `(variable,
        facet_id, recipient, donor)` patterns where `None` matches any dcid
        on that role: a per-pair preview serves each pair the facets of its
        variable's group that the pair holds, and a bulk preview one facet
        everywhere. A facet the preview passes over (a projection beyond the
        recorded years) sets no year. `None` keeps every row. `excluded` then drops
        the rows of each `(variable, role, dcid)`, rows a preview leaves out
        whatever facet serves them.

        A series is a `(variable, recipient, donor)` pair, however many of its
        facets are served: a preview collapses the rows of facets sharing a
        provenance, method and unit to one per date
        (`materialise.facet_collapse`), so each series serves one observation
        in the year and `observations` equals `series`. Both fields remain
        separate in the wire shape as "N observations across M series".

        `role_counts` omits a role no active row carries, such as `donor`
        when every active row belongs to a variable declaring a recipient
        role alone.
        """
        rows = self._member_rows(variables=variables, bound=bound)
        if served is not None and rows.num_rows:
            rows = rows.filter(pa.array(self._served_row_mask(rows, served)))
        if rows.num_rows and (dropped := self._excluded_row_mask(rows, excluded)).any():
            rows = rows.filter(pa.array(~dropped))
        if rows.num_rows == 0:
            return None
        masks = _mask_matrix(rows, self._horizon)
        window = _window_words(self._horizon, first_year=first_year, last_year=last_year)
        combined = np.bitwise_or.reduce(masks, axis=0) & window
        if not combined.any():
            return None
        if elected is not None and (built := self._to_build_year(combined)).any():
            combined = built
        latest_year = int(_year_bounds(combined[None, :], self._horizon)[1][0])
        earliest = max(latest_year - _HEADLINE_YEARS + 1, self._horizon.epoch_year)
        recent_years = tuple(
            self._year_count(rows, masks, year) for year in range(latest_year, earliest - 1, -1)
        )
        counted = recent_years[0] if elected is None else self._elected_count(rows, masks, elected)
        return Headline(
            year=counted.year,
            observations=counted.observations,
            series=counted.series,
            role_counts=counted.role_counts,
            recent_years=recent_years,
        )

    def _elected_count(
        self, rows: pa.Table, masks: np.ndarray, elected: Mapping[str, int]
    ) -> YearCount:
        """The counts of the newest year a variable of `rows` heads under `latest`, over
        the rows of the variables heading it. A variable heads its `elected` year or,
        absent from `elected`, its latest year up to `build_year`, and no year when it
        holds none that early. With no variable heading a year, each heads its own latest.
        `rows` are grouped by variable and `masks` is their `(rows, mask_words)` year-mask
        matrix."""
        rows = rows.combine_chunks()
        codes = _codes(rows[VARIABLE_COLUMN])
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        held = np.bitwise_or.reduceat(masks, starts)
        built = self._to_build_year(held)
        heads = built.any(axis=1)
        if not heads.any():
            built, heads = held, np.ones(len(starts), dtype=bool)
        years = np.full(len(starts), -1, dtype=np.int64)
        years[heads] = _year_bounds(built[heads], self._horizon)[1]
        for variable, year in elected.items():
            years[codes[starts] == self._variable_codes.get(variable, -1)] = year
        year = int(years.max())
        serving = np.isin(codes, codes[starts][years == year])
        return self._year_count(rows.filter(pa.array(serving)), masks[serving], year)

    def _year_count(self, rows: pa.Table, masks: np.ndarray, year: int) -> YearCount:
        """`rows`' series, and their distinct donor and recipient dcids, holding an
        observation in `year`, a year inside the horizon. `masks` is `rows`'
        `(rows, mask_words)` year-mask matrix."""
        active_rows = rows.filter(pa.array(self._year_only(masks, year).any(axis=1)))
        series = (
            active_rows.select([VARIABLE_COLUMN, RECIPIENT_COLUMN, DONOR_COLUMN])
            .group_by([VARIABLE_COLUMN, RECIPIENT_COLUMN, DONOR_COLUMN])
            .aggregate([])
            .num_rows
        )
        role_counts: dict[Role, int] = {}
        for role, column in _ROLE_COLUMNS.items():
            found = set(cast("list[str]", pc.unique(active_rows[column]).to_pylist()))
            if count := len(found - {ABSENT_ENTITY}):
                role_counts[role] = count
        return YearCount(year=year, observations=series, series=series, role_counts=role_counts)

    def _year_only(self, masks: np.ndarray, year: int) -> np.ndarray:
        """Each row of `(rows, mask_words)` year masks keeping only `year`'s bit: zero for
        a row holding no observation in `year`, and for a year outside the horizon."""
        only = np.zeros_like(masks)
        if self._horizon.epoch_year <= year <= self._horizon.max_year:
            word, bit = divmod(year - self._horizon.epoch_year, MASK_WORD_BITS)
            only[:, word] = masks[:, word] & (np.uint64(1) << np.uint64(bit))
        return only

    def elected_years(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        served: Iterable[tuple[str, str, str | None, str | None]] | None = None,
        excluded: Iterable[tuple[str, Role, str]] = (),
        admits: Callable[[str, str], bool] | None = None,
    ) -> dict[str, int]:
        """The year a `latest` read of `variables` under `bound` serves each variable
        (`elected_year`), counted over the series `served`, `excluded` and `admits` select
        as they do for `series_years`, across their whole history up to `build_year`.

        Those series are the ones the read holds before it reduces any to one year: every
        series of the facets it reads, less the rows it leaves out. Every count, reduction
        and headline of one read takes its years from one call.

        A variable elects no year, and is absent, when the index holds no selected row of
        it, when none of its selected series holds a year up to `build_year`, or when one
        of its selected facets states an observation period shorter than a year
        (`KeptFacets.sub_annual`), whose latest period the index cannot find within a
        year. Such a variable serves its own latest period."""
        found = self.series_years(
            variables=variables, bound=bound, served=served, excluded=excluded, admits=admits
        )
        if found is None:
            return {}
        names = _key_values(found.key_rows[VARIABLE_COLUMN])
        starts = [i for i, name in enumerate(names) if i == 0 or name != names[i - 1]]
        elected: dict[str, int] = {}
        for start, stop in zip(starts, [*starts[1:], len(names)], strict=True):
            variable = names[start]
            if found.variables[variable].sub_annual:
                continue
            masks = found.masks[start:stop]
            held = self._to_build_year(np.bitwise_or.reduce(masks, axis=0))
            if not held.any():
                continue
            newest = int(_year_bounds(held[None, :], self._horizon)[1][0])
            year = elected_year(
                {
                    year: int(self._year_only(masks, year).any(axis=1).sum())
                    for year in range(newest, newest - _HEADLINE_YEARS, -1)
                }
            )
            assert year is not None  # `newest` holds a series
            elected[variable] = year
        return elected

    def series_by_year(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        served: Iterable[tuple[str, str, str | None, str | None]] | None = None,
        excluded: Iterable[tuple[str, Role, str]] = (),
        first_year: int,
        last_year: int,
    ) -> dict[int, int]:
        """How many series of `variables` under `bound` hold each year of `[first_year,
        last_year]`, a year none holds kept at zero: the series `headline` counts in
        `recent_years` and `elected_years` elects over, selected by `served` and
        `excluded` as they select them."""
        found = self.series_years(
            variables=variables, bound=bound, served=served, excluded=excluded
        )
        return {
            year: 0 if found is None else int(self._year_only(found.masks, year).any(axis=1).sum())
            for year in range(first_year, last_year + 1)
        }

    def covers(self, variable: str) -> bool:
        """Whether the artifact holds any row of `variable`."""
        return variable in self._variable_codes

    def series_years(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        served: Iterable[tuple[str, str, str | None, str | None]] | None = None,
        excluded: Iterable[tuple[str, Role, str]] = (),
        admits: Callable[[str, str], bool] | None = None,
        first_year: int | None = None,
        last_year: int | None = None,
        latest: Mapping[str, int] | None = None,
        kept_years: KeptYears | None = None,
    ) -> SeriesYears | None:
        """The series a read of `variables` under `bound` serves inside `[first_year,
        last_year]` and the years each serves, the points `served_extent` counts. `served`
        and `excluded` select rows as they do for `headline`, and `admits`, given a row's
        variable and facet id, keeps the rows of the facets it admits. `kept_years` then
        keeps, of each series' years, those it returns for the series' recipient and donor
        (`ABSENT_ENTITY` for none), as a conversion keeps a series' rows in the years its
        tables hold, and a series left with no year serves nothing. `None` when nothing
        survives.

        A series is a `(variable, recipient, donor)` key. It serves one point per year any
        of its kept facets holds, as materialise collapses facets sharing a provenance,
        method and unit to one row per date (`materialise.facet_collapse`), so its mask
        ORs those facets' masks. `latest` is set for a `latest` read and holds the year
        each variable elected (`elected_years`): the variable serves that year alone, from
        the series holding it. A variable `latest` omits serves its own latest year, from
        the series whose latest year it is. A series serving neither serves nothing.
        """
        filtered = self._filtered(
            variables=variables, bound=bound, first_year=first_year, last_year=last_year
        )
        if admits is not None and filtered.num_rows:
            filtered = filtered.filter(pa.array(self._admitted_row_mask(filtered, admits)))
        if served is not None and filtered.num_rows:
            filtered = filtered.filter(pa.array(self._served_row_mask(filtered, served)))
        if filtered.num_rows and (dropped := self._excluded_row_mask(filtered, excluded)).any():
            filtered = filtered.filter(pa.array(~dropped))
        if filtered.num_rows == 0:
            return None
        filtered = filtered.combine_chunks()
        keys = np.zeros(filtered.num_rows, dtype=np.int64)
        for column in _SERIES_COLUMNS:
            width = len(cast("pa.DictionaryArray", filtered[column].chunk(0)).dictionary)
            keys = keys * width + _codes(filtered[column]).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        masks = np.bitwise_or.reduceat(_mask_matrix(filtered, self._horizon)[order], starts)
        firsts = order[starts]
        if latest is not None:
            variable_codes = _codes(filtered[VARIABLE_COLUMN])[firsts]
            held = masks
            masks = _latest_year_only(held)
            kept = _at_variable_latest(masks, variable_codes)
            for variable, year in latest.items():
                elects = variable_codes == self._variable_codes.get(variable, -1)
                masks[elects] = self._year_only(held[elects], year)
                kept[elects] = masks[elects].any(axis=1)
            masks, firsts = masks[kept], firsts[kept]
        found = SeriesYears(
            masks=masks,
            key_rows=filtered.select(list(_SERIES_COLUMNS)).take(firsts),
            variables=self._kept_facets(filtered),
        )
        return found if kept_years is None else self._keeping(found, kept_years)

    def _keeping(self, found: SeriesYears, kept_years: KeptYears) -> SeriesYears | None:
        """`found` with each series' years narrowed to `kept_years`, less the series left
        with none; None when none is left."""
        masks = found.masks.copy()
        by_pair: dict[tuple[str, str], np.ndarray | None] = {}
        for i, (_variable, recipient, donor) in enumerate(found.keys()):
            pair = (recipient, donor)
            if pair not in by_pair:
                years = kept_years(recipient, donor)
                by_pair[pair] = (
                    None
                    if years is None
                    else np.array(
                        self._horizon.masks_from_years(
                            y
                            for y in years
                            if self._horizon.epoch_year <= y <= self._horizon.max_year
                        ),
                        dtype=np.uint64,
                    )
                )
            if (mask := by_pair[pair]) is not None:
                masks[i] &= mask
        held = masks.any(axis=1)
        if not held.any():
            return None
        return SeriesYears(
            masks=masks[held],
            key_rows=found.key_rows.filter(pa.array(held)),
            variables=found.variables,
        )

    def _kept_facets(self, rows: pa.Table) -> dict[str, KeptFacets]:
        """The facets each variable of one-chunk `rows` sits in (`KeptFacets`)."""
        variable_column = cast("pa.DictionaryArray", rows[VARIABLE_COLUMN].chunk(0))
        facet_column = cast("pa.DictionaryArray", rows[FACET_COLUMN].chunk(0))
        width = len(facet_column.dictionary)
        pairs = np.unique(
            _codes(rows[VARIABLE_COLUMN]).astype(np.int64) * width + _codes(rows[FACET_COLUMN])
        )
        by_variable: dict[str, set[str]] = {}
        for variable, facet in zip(
            _decode(variable_column.dictionary, pairs // width),
            _decode(facet_column.dictionary, pairs % width),
            strict=True,
        ):
            by_variable.setdefault(variable, set()).add(facet)
        result: dict[str, KeptFacets] = {}
        for variable, facet_ids in by_variable.items():
            metadata = [self._facet_metadata.get(f) for f in facet_ids]
            groups = {
                (m.provenance_id, m.measurement_method, m.unit) if m is not None else None
                for m in metadata
            }
            result[variable] = KeptFacets(
                facet_ids=frozenset(facet_ids),
                groups=len(groups),
                sub_annual=any(_sub_annual(m) for m in metadata),
            )
        return result

    def _admitted_row_mask(self, rows: pa.Table, admits: Callable[[str, str], bool]) -> np.ndarray:
        """Whether `admits` admits each of `rows`' variable and facet, asked once per
        distinct pair."""
        rows = rows.combine_chunks()
        variable_column = cast("pa.DictionaryArray", rows[VARIABLE_COLUMN].chunk(0))
        facet_column = cast("pa.DictionaryArray", rows[FACET_COLUMN].chunk(0))
        width = len(facet_column.dictionary)
        keys = _codes(rows[VARIABLE_COLUMN]).astype(np.int64) * width + _codes(rows[FACET_COLUMN])
        pairs = np.unique(keys)
        variables = _decode(variable_column.dictionary, pairs // width)
        facets = _decode(facet_column.dictionary, pairs % width)
        admitted = [
            int(key)
            for key, variable, facet in zip(pairs.tolist(), variables, facets, strict=True)
            if admits(variable, facet)
        ]
        return np.isin(keys, np.array(admitted, dtype=np.int64))

    def served_extent(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles = _NO_BINDING,
        served: Iterable[tuple[str, str, str | None, str | None]] | None = None,
        excluded: Iterable[tuple[str, Role, str]] = (),
        first_year: int | None = None,
        last_year: int | None = None,
        latest: Mapping[str, int] | None = None,
        kept_years: KeptYears | None = None,
    ) -> ServedExtent | None:
        """The rows, series and years a read of `variables` under `bound` serves inside
        `[first_year, last_year]`, over every year of the window where `headline` counts
        its latest: `rows` sums each series' distinct years (`series_years`). `served`
        and `excluded` select rows as they do for `headline`. `None` when nothing
        survives. With `latest`, each variable serves one year alone, the one it elected
        (`series_years`), and the years span the earliest and latest of those. `kept_years`
        narrows each series' years as it does for `series_years`.
        """
        found = self.series_years(
            variables=variables,
            bound=bound,
            served=served,
            excluded=excluded,
            first_year=first_year,
            last_year=last_year,
            latest=latest,
            kept_years=kept_years,
        )
        if found is None:
            return None
        first, last = _year_bounds(
            np.bitwise_or.reduce(found.masks, axis=0)[None, :], self._horizon
        )
        return ServedExtent(
            rows=found.rows,
            series=found.series,
            first_year=int(first[0]),
            last_year=int(last[0]),
        )

    def estimate(
        self,
        *,
        variables: Sequence[str],
        bound: BoundRoles,
        first_year: int | None = None,
        last_year: int | None = None,
        lanes: int = DEFAULT_PLANNING_LANES,
        time_period: Literal["LATEST"] | Sequence[str] | None = None,
    ) -> Estimate:
        """Estimates covered observation rows and SDMX requests before I/O.

        `variables` is one already-chunked group where each StatVar appears
        once. Callers estimating a combined dispatch split it as the fetch does,
        call this once per chunk with `lanes=1`, and sum the results.

        `rows` sums `n_obs` over the rows surviving `bound` and the
        `[first_year, last_year]` window.

        `requests` ignores the window, as the fetch does: the facet probe
        (`materialise.facets.role_bearing_facets`) reads
        coverage across all years, and `materialise.generation._generate_chunk`
        names every StatVar of a chunk in which any member has a dispatch
        facet. So `variables` are grouped by their `RoleNames` in the order
        given, and each group with a coverage row under `bound` in any year
        counts `plan_request_chunks` over all of its variables. A group with
        no such row sends nothing and counts zero. A window outside coverage
        therefore gives zero rows and a non-zero request count. `lanes`
        defaults to 1. Already-chunked fetches also pass 1.

        `time_period` passes through unchanged and counts against the
        request byte budget. `None` models a fetch without a server-side
        time filter.

        Bound roles pass their dcid sets through. SDMX requires at least one
        entity filter, so a fully open request uses
        `exhaustive_single_axis_filter` over the group's variables, as the
        fetch does. A bound role matches no row of a variable that lacks it. A
        variable with a coverage row under `bound` and no `RoleNames` raises
        `KeyError`. Other variables without `RoleNames` add nothing to either
        count.
        """
        members = self._member_rows(variables=variables, bound=bound)
        if members.num_rows == 0:
            return Estimate(rows=0, requests=0)
        windowed = _apply_window(
            members, horizon=self._horizon, first_year=first_year, last_year=last_year
        )
        rows = int(pc.sum(windowed[N_OBS_COLUMN]).as_py() or 0)
        dispatching = set(cast("list[str]", pc.unique(members[VARIABLE_COLUMN]).to_pylist()))
        groups: dict[RoleNames, list[str]] = {}
        for variable in variables:
            names = self._role_names.get(variable)
            if names is None:
                if variable in dispatching:
                    raise KeyError(
                        f"variable {variable!r} has no RoleNames entry in this coverage artifact"
                    )
                continue
            groups.setdefault(names, []).append(variable)
        donor_bound = "donor" in bound
        recipient_bound = "recipient" in bound
        both_unbound = not donor_bound and not recipient_bound
        requests = 0
        for names, group_variables in groups.items():
            if dispatching.isdisjoint(group_variables):
                continue
            entities: dict[str, tuple[str, ...]] = {}
            # A bound role matches no row of a variable that lacks it, so a
            # surviving group has no entity to place on a role it lacks.
            if donor_bound and names.donor is not None:
                entities[names.donor] = tuple(sorted(bound["donor"]))
            if recipient_bound and names.recipient is not None:
                entities[names.recipient] = tuple(sorted(bound["recipient"]))
            if both_unbound:
                entities = self.exhaustive_single_axis_filter(
                    variables=group_variables,
                    donor_role=names.donor,
                    recipient_role=names.recipient,
                )
            chunks = plan_request_chunks(
                group_variables, entities=entities, lanes=lanes, time_period=time_period
            )
            requests += len(chunks)
        return Estimate(rows=rows, requests=requests)

    def exhaustive_single_axis_filter(
        self,
        *,
        variables: Sequence[str],
        donor_role: str | None,
        recipient_role: str | None,
    ) -> dict[str, tuple[str, ...]]:
        """Returns the exhaustive entities filter for an already-unbound
        fetch group. `variables` must share the supplied `RoleNames` pair.
        The filter lists every dcid `variables` cover on the smaller role
        axis, or on the one axis a one-role pair names, across all years,
        which satisfies SDMX's entity requirement while leaving the other axis
        open. `estimate` and the fetch both read it, so the
        request count priced is the one sent.

        The year window plays no part: a fetch chunk can hold members with
        different windows, and one list over all years serves each of them.
        Returns `{}` only when `variables` have no coverage row at all.
        """
        rows = self._member_rows(variables=variables, bound=_NO_BINDING)
        if rows.num_rows == 0:
            return {}
        axes = [
            (role, tuple(sorted(cast("list[str]", pc.unique(rows[column]).to_pylist()))))
            for role, column in ((donor_role, DONOR_COLUMN), (recipient_role, RECIPIENT_COLUMN))
            if role is not None
        ]
        role, dcids = min(axes, key=lambda axis: len(axis[1]))
        return {role: dcids}


@dataclass(frozen=True, slots=True)
class PairGroup:
    """One (variable, facet_id)'s rows in a `PairRows`: how many (donor,
    recipient) pairs it carries, the latest of their last years with data, and
    the distinct dcids on each role. A one-role variable's set for the role it
    lacks is empty."""

    variable: str
    facet_id: str
    pair_count: int
    last_year: int
    donors: frozenset[str]
    recipients: frozenset[str]


class PairRows:
    """Stores filtered rows as a dictionary-encoded Arrow table.

    Sorted `(variable, facet_id)` codes keep groups contiguous. `groups`
    summarises them. Other lookups use binary search to read one group.
    """

    def __init__(
        self,
        table: pa.Table,
        *,
        horizon: YearHorizon,
        variable_codes: Mapping[str, int],
        facet_codes: Mapping[str, int],
    ) -> None:
        """`variable_codes`/`facet_codes` map each value to its code in the
        `CoverageIndex` table's own `variable`/`facet_id` dictionaries,
        which `table` (a filter's output) keeps unchanged."""
        self._table = table
        self._horizon = horizon
        self._variable_codes = variable_codes
        self._facet_codes = facet_codes
        self._excluding: dict[tuple[Role, frozenset[str]], PairRows] = {}

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    def excluding(self, *, role: Role, dcids: frozenset[str]) -> PairRows:
        """These rows without those binding one of `dcids` on `role`, kept in the same
        sorted order, so every lookup reads only the remaining rows. Each `(role, dcids)`
        filters the table once; later calls reuse it."""
        if not dcids or self._table.num_rows == 0:
            return self
        key = (role, dcids)
        if (kept := self._excluding.get(key)) is None:
            mask = pc.invert(_key_is_in(self._table[_ROLE_COLUMNS[role]], dcids))
            kept = self._excluding[key] = PairRows(
                self._table.filter(mask).combine_chunks(),
                horizon=self._horizon,
                variable_codes=self._variable_codes,
                facet_codes=self._facet_codes,
            )
        return kept

    def groups(self) -> list[PairGroup]:
        """One `PairGroup` per (variable, facet_id) these rows carry, in
        (variable, facet_id) order. The distinct-dcid sets are computed on
        the dictionary indices and decoded once per distinct value."""
        if self._table.num_rows == 0:
            return []
        donor = cast("pa.DictionaryArray", self._table[DONOR_COLUMN].chunk(0))
        recipient = cast("pa.DictionaryArray", self._table[RECIPIENT_COLUMN].chunk(0))
        keyed = pa.table(
            {
                VARIABLE_COLUMN: self._table[VARIABLE_COLUMN],
                FACET_COLUMN: self._table[FACET_COLUMN],
                DONOR_COLUMN: donor.indices,
                RECIPIENT_COLUMN: recipient.indices,
                LAST_YEAR_COLUMN: self._table[LAST_YEAR_COLUMN],
            }
        )
        # use_threads=False keeps the output in first-appearance order,
        # and the row order makes that (variable, facet_id) order.
        grouped = keyed.group_by([VARIABLE_COLUMN, FACET_COLUMN], use_threads=False).aggregate(
            (
                (DONOR_COLUMN, "distinct"),
                (RECIPIENT_COLUMN, "distinct"),
                (LAST_YEAR_COLUMN, "max"),
                (LAST_YEAR_COLUMN, "count"),
            )
        )
        donor_names = cast("list[str]", donor.dictionary.to_pylist())
        recipient_names = cast("list[str]", recipient.dictionary.to_pylist())
        donor_ids = cast("list[list[int]]", grouped[f"{DONOR_COLUMN}_distinct"].to_pylist())
        recipient_ids = cast("list[list[int]]", grouped[f"{RECIPIENT_COLUMN}_distinct"].to_pylist())
        return [
            PairGroup(
                variable=variable,
                facet_id=facet_id,
                pair_count=count,
                last_year=last_year,
                donors=frozenset(donor_names[i] for i in donors) - {ABSENT_ENTITY},
                recipients=frozenset(recipient_names[i] for i in recipients) - {ABSENT_ENTITY},
            )
            for variable, facet_id, count, last_year, donors, recipients in zip(
                _key_values(grouped[VARIABLE_COLUMN]),
                _key_values(grouped[FACET_COLUMN]),
                _int_values(grouped[f"{LAST_YEAR_COLUMN}_count"]),
                _int_values(grouped[f"{LAST_YEAR_COLUMN}_max"]),
                donor_ids,
                recipient_ids,
                strict=True,
            )
        ]

    @cached_property
    def _views(self) -> _RowViews:
        """Zero-copy numpy views over Arrow buffers. A buffer that requires a
        copy raises, avoiding row-sized allocations.
        """
        return _RowViews(
            variables=_codes(self._table[VARIABLE_COLUMN]),
            facets=_codes(self._table[FACET_COLUMN]),
            donors=_codes(self._table[DONOR_COLUMN]),
            recipients=_codes(self._table[RECIPIENT_COLUMN]),
            n_obs=self._table[N_OBS_COLUMN].chunk(0).to_numpy(),
            year_masks=tuple(
                self._table[name].chunk(0).to_numpy() for name in self._horizon.mask_columns
            ),
            donor_dictionary=cast(
                "pa.DictionaryArray", self._table[DONOR_COLUMN].chunk(0)
            ).dictionary,
            recipient_dictionary=cast(
                "pa.DictionaryArray", self._table[RECIPIENT_COLUMN].chunk(0)
            ).dictionary,
        )

    def _group(self, variable: str, facet_id: str) -> slice | None:
        """Returns a row slice or `None`. Sorted codes bound each group with
        four binary searches in O(log rows) and constant extra storage.
        """
        variable_code = self._variable_codes.get(variable)
        facet_code = self._facet_codes.get(facet_id)
        if variable_code is None or facet_code is None or self._table.num_rows == 0:
            return None
        views = self._views
        low, high = _run(views.variables, variable_code)
        start, stop = _run(views.facets[low:high], facet_code)
        return slice(low + start, low + stop) if start < stop else None

    def _counts(self, rows: slice, *, first_year: int | None, last_year: int | None) -> np.ndarray:
        """Each of `rows`' `n_obs`, counted inside `[first_year,
        last_year]` as well as this lookup's own window."""
        views = self._views
        if first_year is None and last_year is None:
            return views.n_obs[rows]
        wmasks = _window_words(self._horizon, first_year=first_year, last_year=last_year)
        counts = popcount64(views.year_masks[0][rows] & wmasks[0])
        for word, wmask in zip(views.year_masks[1:], wmasks[1:], strict=True):
            counts += popcount64(word[rows] & wmask)
        return counts

    def pairs(
        self,
        *,
        variable: str,
        facet_id: str,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> list[tuple[str | None, str | None, int]]:
        """`(donor, recipient, n_obs)` for every row of one (variable,
        facet_id), in row order, with `None` for the role a one-role
        variable lacks. `n_obs` counts only the years inside `[first_year,
        last_year]` as well as this lookup's own window, and is 0 when a row
        has no observations in the requested range. Every pair row remains
        listed."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return []
        views = self._views
        counts = self._counts(rows, first_year=first_year, last_year=last_year)
        return [
            (donor or None, recipient or None, n_obs)
            for donor, recipient, n_obs in zip(
                _decode(views.donor_dictionary, views.donors[rows]),
                _decode(views.recipient_dictionary, views.recipients[rows]),
                cast("list[int]", counts.tolist()),
                strict=True,
            )
        ]

    def n_obs(
        self,
        *,
        variable: str,
        facet_id: str,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> int:
        """Summed `n_obs` over every row of one (variable, facet_id),
        counted inside `[first_year, last_year]` as well as this lookup's
        own window."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return 0
        return int(self._counts(rows, first_year=first_year, last_year=last_year).sum())

    def rows_with_data(
        self,
        *,
        variable: str,
        facet_id: str,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> int:
        """How many rows of one (variable, facet_id) have an observation inside
        `[first_year, last_year]`: one per full (donor, recipient) tuple, the series a read
        of those rows returns, whether or not its pairs name the donor. Counted without
        decoding any."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return 0
        return int(np.count_nonzero(self._counts(rows, first_year=first_year, last_year=last_year)))

    def max_n_obs(
        self,
        *,
        variable: str,
        facet_id: str,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> int:
        """The largest one row's `n_obs` among one (variable, facet_id)'s rows, counted
        inside `[first_year, last_year]` as well as this lookup's own window; 0 with no
        row."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return 0
        counts = self._counts(rows, first_year=first_year, last_year=last_year)
        return int(counts.max()) if counts.size else 0

    def year_span(
        self,
        *,
        variable: str,
        facet_id: str,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> tuple[int, int] | None:
        """The earliest and latest years any row of one (variable, facet_id) has data inside
        `[first_year, last_year]` as well as this lookup's own window; None with no data."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return None
        wmasks = _window_words(self._horizon, first_year=first_year, last_year=last_year)
        words = [
            int(np.bitwise_or.reduce(masks[rows] & wmask)) if rows.stop > rows.start else 0
            for masks, wmask in zip(self._views.year_masks, wmasks, strict=True)
        ]
        occupied = [index for index, word in enumerate(words) if word]
        if not occupied:
            return None
        first_word, last_word = occupied[0], occupied[-1]
        lowest = (words[first_word] & -words[first_word]).bit_length() - 1
        highest = words[last_word].bit_length() - 1
        epoch = self._horizon.epoch_year
        return (
            epoch + MASK_WORD_BITS * first_word + lowest,
            epoch + MASK_WORD_BITS * last_word + highest,
        )

    def members_with_data(
        self,
        *,
        variable: str,
        facet_id: str,
        member_is_donor: bool,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> frozenset[str]:
        """The distinct dcids on one role (the donor with `member_is_donor`, else the
        recipient) among one (variable, facet_id)'s rows with an observation inside
        `[first_year, last_year]`, decoding only those distinct codes."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return frozenset()
        views = self._views
        counts = self._counts(rows, first_year=first_year, last_year=last_year)
        codes, names = (
            (views.donors[rows], views.donor_dictionary)
            if member_is_donor
            else (views.recipients[rows], views.recipient_dictionary)
        )
        return frozenset(_decode(names, np.unique(codes[counts > 0]))) - {ABSENT_ENTITY}

    def head_pairs(
        self,
        *,
        variable: str,
        facet_id: str,
        limit: int,
        first_year: int | None = None,
        last_year: int | None = None,
        member_is_donor: bool,
        has_opposite: bool,
        member_sorts_first: bool = True,
        with_data: bool = False,
    ) -> list[tuple[str, str | None, int]]:
        """Returns the first `limit` rows ranked as caller-built role tuples,
        decoding strings only for the returned rows. `with_data` skips rows, or
        summed members, with no observations inside `[first_year, last_year]`
        before the cut.

        `member_is_donor` selects the returned role. With no opposite role,
        rows sharing a member are summed and sorted by member. With both
        roles, `member_sorts_first` selects the primary key, and the role a
        one-role variable lacks reads as a `None` opposite. Sorted dictionary
        codes preserve string order, so each group needs one numpy sort.
        """
        rows = self._group(variable, facet_id)
        if rows is None:
            return []
        views = self._views
        counts = self._counts(rows, first_year=first_year, last_year=last_year)
        if member_is_donor:
            member_codes, opposite_codes = views.donors[rows], views.recipients[rows]
            member_names, opposite_names = views.donor_dictionary, views.recipient_dictionary
        else:
            member_codes, opposite_codes = views.recipients[rows], views.donors[rows]
            member_names, opposite_names = views.recipient_dictionary, views.donor_dictionary

        if not has_opposite:
            head_codes, inverse = np.unique(member_codes, return_inverse=True)
            summed = np.zeros(head_codes.size, dtype=np.int64)
            np.add.at(summed, inverse, counts)
            if with_data:
                head_codes, summed = head_codes[summed > 0], summed[summed > 0]
            return [
                (name, None, n)
                for name, n in zip(
                    _decode(member_names, head_codes[:limit]),
                    cast("list[int]", summed[:limit].tolist()),
                    strict=True,
                )
            ]

        if with_data:
            kept = counts > 0
            member_codes, opposite_codes, counts = (
                member_codes[kept],
                opposite_codes[kept],
                counts[kept],
            )
        primary, secondary = (
            (member_codes, opposite_codes) if member_sorts_first else (opposite_codes, member_codes)
        )
        order = np.lexsort((secondary, primary))[:limit]
        return [
            (member_name, opposite_name or None, n)
            for member_name, opposite_name, n in zip(
                _decode(member_names, member_codes[order]),
                _decode(opposite_names, opposite_codes[order]),
                cast("list[int]", counts[order].tolist()),
                strict=True,
            )
        ]

    def pair_year_counts(
        self,
        *,
        variable: str,
        facet_id: str,
        first_year: int | None = None,
        last_year: int | None = None,
        member_is_donor: bool,
        has_opposite: bool,
    ) -> list[tuple[str, str | None, tuple[tuple[int, int], ...]]]:
        """Each row of one (variable, facet_id) with data inside `[first_year, last_year]`
        as well as this lookup's own window, keyed as `head_pairs` keys it, with each year
        it holds data in and the observations there, ascending: one a row, summed over the
        rows sharing a member when there is no opposite role, as a one-year `head_pairs`
        window counts them. Unordered, and read in one pass over the rows' year masks."""
        rows = self._group(variable, facet_id)
        if rows is None:
            return []
        views = self._views
        wmasks = _window_words(self._horizon, first_year=first_year, last_year=last_year)
        words = np.stack(
            [masks[rows] & wmask for masks, wmask in zip(views.year_masks, wmasks, strict=True)],
            axis=1,
        )
        # Bit `b` of word `w` marks year `epoch_year + 64 * w + b`: little-endian bytes
        # unpacked least significant bit first keep that order.
        bits = np.unpackbits(words.astype("<u8").view(np.uint8), axis=1, bitorder="little")
        held = np.flatnonzero(bits.any(axis=0))
        counts = bits[:, held].astype(np.int64)
        if member_is_donor:
            member_codes, opposite_codes = views.donors[rows], views.recipients[rows]
            member_names, opposite_names = views.donor_dictionary, views.recipient_dictionary
        else:
            member_codes, opposite_codes = views.recipients[rows], views.donors[rows]
            member_names, opposite_names = views.recipient_dictionary, views.donor_dictionary
        if not has_opposite:
            member_codes, inverse = np.unique(member_codes, return_inverse=True)
            summed = np.zeros((member_codes.size, held.size), dtype=np.int64)
            np.add.at(summed, inverse, counts)
            counts = summed
        kept = counts.any(axis=1)
        years = (self._horizon.epoch_year + held).tolist()
        by_row = [
            tuple((years[i], int(row[i])) for i in np.flatnonzero(row)) for row in counts[kept]
        ]
        opposites = (
            _decode(opposite_names, opposite_codes[kept]) if has_opposite else [""] * len(by_row)
        )
        return [
            (member, opposite or None, year_counts)
            for member, opposite, year_counts in zip(
                _decode(member_names, member_codes[kept]), opposites, by_row, strict=True
            )
        ]


@dataclass(frozen=True, slots=True)
class _RowViews:
    """`PairRows`' numpy views over its table's buffers, and the donor and
    recipient dictionaries (Arrow arrays, decoded per returned code)."""

    variables: np.ndarray
    facets: np.ndarray
    donors: np.ndarray
    recipients: np.ndarray
    n_obs: np.ndarray
    year_masks: tuple[np.ndarray, ...]
    donor_dictionary: pa.Array
    recipient_dictionary: pa.Array


__all__ = [
    "BoundRoles",
    "CoverageIndex",
    "Estimate",
    "FacetCoverage",
    "Headline",
    "PairGroup",
    "PairRows",
    "ProvenanceShape",
    "Role",
    "SeriesKey",
    "VariableCoverage",
    "YearCount",
]

# Not re-exported: `FacetMetadata` (imported for the `facet_metadata`
# property's type) lives in and is exported from `series_coverage.schema`.
