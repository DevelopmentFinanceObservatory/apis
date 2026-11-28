"""How much of a partial year a frame's source has reported, against the year before it.

A read records the share of its source's partial year reported so far on the frame's
`held_back` run and, when its window opts in to the year, on its `incomplete_after` entry
(`datasets_api.manifest_models.ReportedShare`). The mint folds the rows it read of that year
and the year before, unconverted and before its window and held-back cuts
(`fold_observations`), into `YearTotals` per `(provenance, stat_var)` pair and calendar year:

- `value` basis: the summed value at the year over the summed value at the year before, each
  over every series holding that year. It is 0 or more, and above 1 when the year already
  exceeds the year before. A year before that sums to zero or below, or a year that sums
  below zero (net flows can), states no share. Used when every folded row states one
  currency unit with no denominator, whose values add (`dfo_spec_kit.units.parse_unit` reads
  `Money` or `UnpricedMoney`), so no sum adds two currencies.
- `series` basis: the series holding the year over the series holding the year before, for
  any other rows.

A series is a row's variable, provenance, unit and role cells. Its `measurementMethod` is left
out, so a series an estimate continues stays one series, and rows under a projection method
(`member_folds.is_projection`) are left out, since they hold no reported value. Only annual
periods count, so a frame of a finer grain states no share. A pair whose rows the mint let go
past its row ceiling (`YearTotals.capped`) states no share.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitRequest,
    array_kernel,
    unit_cells,
    unit_codes,
)
from dfo_spec_kit.units import Money, UnpricedMoney, parse_unit

from dc_sdmx.results import VALUE_COLUMN as RAW_VALUE_COLUMN
from dfo_platform.columns import PROVENANCE_COLUMN
from materialise.columnar import distinct_combinations, observed_values
from materialise.held_back import source_of
from materialise.member_folds import is_projection

__all__ = [
    "Basis",
    "ReportedShareFigure",
    "YearTotals",
    "YearTotalsFold",
    "fold_observations",
    "reported_share",
]

type Basis = Literal["value", "series"]

type Pair = tuple[str | None, str]
"""A row's `provenance` cell and its `stat_var` cell."""

_METHOD_COLUMN: Final = "measurementMethod"
_RAW_PROVENANCE_COLUMN: Final = "provenance"
_RAW_UNIT_COLUMN: Final = "unit"
_RAW_TIME_PERIOD_COLUMN: Final = "TIME_PERIOD"
_STATUS_COLUMN: Final = "status"
_NOT_SERIES: Final = frozenset({TIME_PERIOD_COLUMN, VALUE_COLUMN, _METHOD_COLUMN, _STATUS_COLUMN})
_YEAR: Final = re.compile(r"^\d{4}$")


@dataclass(frozen=True, slots=True)
class ReportedShareFigure:
    """A partial year's reported share (`datasets_api.manifest_models.ReportedShare`)."""

    basis: Basis
    share: float
    against: int


@dataclass
class YearTotals:
    """One `(provenance, stat_var)` pair's sums per calendar year. `series[y]` counts the
    series holding `y`; `value[y]` is the summed value at `y` over those series. `additive`
    is false once a row states a unit other than a currency, and `units` holds the currency
    units the rows state. `capped` is true when the mint let some of the pair's rows go
    uncounted past its row ceiling, so the sums are short and state no share."""

    series: dict[int, int] = field(default_factory=dict)
    value: dict[int, float] = field(default_factory=dict)
    additive: bool = True
    units: set[str] = field(default_factory=set)
    capped: bool = False

    def add(self, other: YearTotals) -> None:
        """Adds `other`'s series and sums to these, as two pairs whose series are distinct."""
        for year, count in other.series.items():
            self.series[year] = self.series.get(year, 0) + count
        for year, total in other.value.items():
            self.value[year] = self.value.get(year, 0.0) + total
        self.additive = self.additive and other.additive
        self.units |= other.units
        self.capped = self.capped or other.capped


def reported_share(totals: Sequence[YearTotals], *, year: int) -> ReportedShareFigure | None:
    """The share of `year` reported against the year before, over `totals` together (module
    docstring). `None` when any of `totals` is `capped`, when no series holds the year before,
    or on the `value` basis when the year before sums to zero or below or the year sums below
    zero. A share above 1 stands."""
    held = YearTotals()
    for one in totals:
        held.add(one)
    against = year - 1
    if held.capped or not held.series.get(against):
        return None
    if held.additive and len(held.units) <= 1:
        before = held.value.get(against, 0.0)
        reported = held.value.get(year, 0.0)
        if before <= 0 or reported < 0:
            return None
        return ReportedShareFigure(basis="value", share=reported / before, against=against)
    return ReportedShareFigure(
        basis="series", share=held.series.get(year, 0) / held.series[against], against=against
    )


@dataclass
class YearTotalsFold:
    """Running `YearTotals` per pair over a read's canonical frame tables. A series never
    spans two tables. `component_of` maps a composite member's bare variable to its
    component's provenance, the source its rows' projection methods are judged by."""

    component_of: Mapping[str, str] = field(default_factory=dict)
    totals: dict[Pair, YearTotals] = field(default_factory=dict)

    def add(self, table: pa.Table) -> None:
        """Folds one canonical frame table (`frame_build.build_frame_from_member_tables`).
        Each distinct `(provenance, stat_var, measurementMethod)` and `(pair, unit)`
        combination is judged once."""
        if table.num_rows == 0:
            return
        table = table.filter(self._counted(table))
        if table.num_rows == 0:
            return
        series_columns = [name for name in table.column_names if name not in _NOT_SERIES]
        codes = _series_codes(table, series_columns)
        row_pairs, pairs = distinct_combinations([_provenances(table), table[STAT_VAR_COLUMN]])
        series_pairs = _series_pairs(codes, row_pairs)
        years = pc.cast(table[TIME_PERIOD_COLUMN], pa.int64())
        rows = pa.table(
            {
                "series": codes,
                "year": years,
                "value": pc.cast(table[VALUE_COLUMN], pa.float64()),
            }
        )
        rows = rows.group_by(["series", "year"], use_threads=False).aggregate((("value", "sum"),))
        rows = rows.append_column("pair", series_pairs.take(rows["series"]))
        for pair in _ints(pc.unique(rows["pair"])):
            self.totals.setdefault(cast(Pair, pairs[pair]), YearTotals())
        counts = rows.group_by(["pair", "year"], use_threads=False).aggregate(
            (("series", "count"),)
        )
        for pair, year, count in zip(
            _ints(counts["pair"]), _ints(counts["year"]), _ints(counts["series_count"]), strict=True
        ):
            held = self.totals[cast(Pair, pairs[pair])]
            held.series[year] = held.series.get(year, 0) + count
        sums = rows.group_by(["pair", "year"], use_threads=False).aggregate((("value_sum", "sum"),))
        for pair, year, total in zip(
            _ints(sums["pair"]),
            _ints(sums["year"]),
            cast("list[float]", sums["value_sum_sum"].to_pylist()),
            strict=True,
        ):
            held = self.totals[cast(Pair, pairs[pair])]
            held.value[year] = held.value.get(year, 0.0) + total
        for pair, (adds, units) in _additive_pairs(table, row_pairs).items():
            held = self.totals[cast(Pair, pairs[pair])]
            held.additive = held.additive and adds
            held.units |= units

    def _counted(self, table: pa.Table) -> pa.BooleanArray:
        """The rows the totals count: an annual period, a value, and no projection method,
        judged once per distinct `(provenance, stat_var, measurementMethod)`."""
        periods = table[TIME_PERIOD_COLUMN].combine_chunks()
        annual = array_kernel("fill_null", pc.match_substring_regex(periods, _YEAR.pattern), False)
        mask = pc.and_(annual, pc.is_valid(table[VALUE_COLUMN].combine_chunks()))
        if _METHOD_COLUMN not in table.column_names:
            return mask
        positions, combinations = distinct_combinations(
            [_provenances(table), table[STAT_VAR_COLUMN], table[_METHOD_COLUMN]]
        )
        projected = pa.array(
            [
                is_projection(
                    self._source(cast("str | None", provenance), cast(str, stat_var)),
                    cast("str | None", method),
                )
                for provenance, stat_var, method in combinations
            ],
            pa.bool_(),
        ).take(positions)
        return pc.and_(mask, pc.invert(projected))

    def _source(self, provenance: str | None, stat_var: str) -> str | None:
        if provenance is None:
            return None
        return source_of(provenance, stat_var, component_of=self.component_of)


def fold_observations(
    rows: pa.Table, *, stat_var: str, roles: Sequence[str]
) -> dict[Pair, YearTotals]:
    """The year sums of one member's `rows`, unconverted rows under
    `dc_sdmx.results.observation_schema(roles)`, each named by `stat_var`. A series is its
    role, provenance and unit cells; a row with no observed value counts nowhere."""
    if rows.num_rows == 0:
        return {}
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        STAT_VAR_COLUMN: pa.array([stat_var] * rows.num_rows, pa.string())
    }
    for role in roles:
        columns[role] = rows[role]
    columns[PROVENANCE_COLUMN] = rows[_RAW_PROVENANCE_COLUMN]
    columns[UNIT_COLUMN] = _unit_refs(rows[_RAW_UNIT_COLUMN])
    columns[_METHOD_COLUMN] = rows[_METHOD_COLUMN]
    columns[TIME_PERIOD_COLUMN] = rows[_RAW_TIME_PERIOD_COLUMN]
    columns[VALUE_COLUMN] = observed_values(rows[RAW_VALUE_COLUMN])
    fold = YearTotalsFold()
    fold.add(pa.table(columns))
    return fold.totals


def _unit_refs(units: pa.ChunkedArray) -> pa.StructArray:
    """A raw `unit` column as the canonical `unit` struct of ref cells, an empty cell null."""
    refs = units.combine_chunks()
    absent = array_kernel(
        "fill_null", array_kernel("equal", refs, pa.scalar("", pa.string())), True
    )
    refs = pc.if_else(absent, pa.scalar(None, pa.string()), refs)
    return pa.StructArray.from_arrays([refs], names=["ref"], mask=absent)


def _ints(column: pa.Array | pa.ChunkedArray) -> list[int]:
    return cast("list[int]", column.to_pylist())


def _provenances(table: pa.Table) -> pa.Array | pa.ChunkedArray:
    """`table`'s `provenance` column, all null when it has none."""
    if PROVENANCE_COLUMN not in table.column_names:
        return pa.nulls(table.num_rows, pa.string())
    return table[PROVENANCE_COLUMN]


def _series_codes(table: pa.Table, columns: Sequence[str]) -> pa.Array:
    """One code per row, equal for rows of one series (`columns` all equal)."""
    keys: list[pa.Array] = []
    for name in columns:
        column = table[name]
        if name == UNIT_COLUMN:
            keys.append(unit_codes(column).codes)
            continue
        encoded = pc.dictionary_encode(column.combine_chunks(), null_encoding="encode")
        keys.append(cast(pa.DictionaryArray, encoded).indices.cast(pa.int64()))
    if not keys:
        return pa.array([0] * table.num_rows, pa.int64())
    codes = pa.table({str(i): key for i, key in enumerate(keys)})
    distinct = codes.group_by(list(codes.column_names), use_threads=False).aggregate(())
    numbered = distinct.append_column("code", pa.array(range(distinct.num_rows), pa.int64()))
    joined = codes.append_column("row", pa.array(range(codes.num_rows), pa.int64())).join(
        numbered, keys=list(codes.column_names)
    )
    return joined.sort_by("row")["code"].combine_chunks()


def _series_pairs(codes: pa.Array, row_pairs: pa.Array) -> pa.Array:
    """Each series code's pair position (`distinct_combinations` over `provenance` and
    `stat_var`), indexed by series code. A series' rows share one pair, since both columns
    are among its key."""
    firsts = (
        pa.table({"series": codes, "pair": row_pairs})
        .group_by("series", use_threads=False)
        .aggregate((("pair", "min"),))
        .sort_by("series")
    )
    return firsts["pair_min"].combine_chunks()


def _additive_pairs(table: pa.Table, row_pairs: pa.Array) -> dict[int, tuple[bool, set[str]]]:
    """Whether every row of each pair position in `row_pairs` states a currency unit with no
    denominator, and the currency units its rows state, judged once per distinct
    `(pair, unit)`."""
    if UNIT_COLUMN not in table.column_names:
        return dict.fromkeys(_ints(pc.unique(row_pairs)), (False, set()))
    units = unit_codes(table[UNIT_COLUMN])
    currency = [_currency(cell) for cell in unit_cells(table[UNIT_COLUMN], units)]
    _, combinations = distinct_combinations([row_pairs, units.codes])
    additive: dict[int, tuple[bool, set[str]]] = {}
    for pair, code in combinations:
        adds, held = additive.setdefault(cast(int, pair), (True, set()))
        if (one := currency[cast(int, code)]) is not None:
            held.add(one)
        additive[cast(int, pair)] = (adds and one is not None, held)
    return additive


def _currency(cell: object) -> str | None:
    """The currency a `unit` cell states with no denominator, as a key equal for two cells
    whose values add: a converted unit, or a unit ref `parse_unit` reads as money, so
    `USD` and `USDollar` are one currency. `None` for any other cell."""
    if isinstance(cell, UnitRequest):
        return repr(cell)
    if not isinstance(cell, str):
        return None
    unit = parse_unit(cell.removeprefix("dcid:"))
    return repr(unit) if isinstance(unit, Money | UnpricedMoney) else None
