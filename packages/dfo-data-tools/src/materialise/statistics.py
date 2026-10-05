"""Builds the `series-statistics/1` Parquet artifact for a materialised frame.

Each row represents one series, keyed by every frame column except
`time_period`, `value`, and `status`, in deterministic first-appearance order.
Grouping and reductions use Arrow. Only the `entities` struct and canonical
JSON `facet` are built per group from its first row. The module reads `Frame`
structurally and projects converted-unit fields directly from Arrow.
"""

from __future__ import annotations

import decimal
import io
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from materialise.frame_build import Frame

_STAT_VAR_COLUMN = "stat_var"
_TIME_PERIOD_COLUMN = "time_period"
_VALUE_COLUMN = "value"
_UNIT_COLUMN = "unit"
_STATUS_COLUMN = "status"
_MEASUREMENT_METHOD_COLUMN = "measurementMethod"
_PROVENANCE_COLUMN = "provenance"

# `derive_columns` reserves these fixed columns. Remaining columns are
# alphabetically sorted role-property dcids.
_FIXED_COLUMNS: frozenset[str] = frozenset(
    {
        _STAT_VAR_COLUMN,
        _TIME_PERIOD_COLUMN,
        _VALUE_COLUMN,
        _UNIT_COLUMN,
        _STATUS_COLUMN,
        _MEASUREMENT_METHOD_COLUMN,
        _PROVENANCE_COLUMN,
    }
)

# 60-digit Decimal precision keeps realistic observation subtraction exact.
_CHANGE_DECIMAL_PRECISION = 60

# Arrow equi-joins drop null keys, so grouping uses a sentinel outside grain values.
# Output values come from the source frame.
_NULL_SENTINEL = "\x00"

SERIES_STATISTICS_SCHEMA_ID = "series-statistics/1"

_ENTITY_TYPE = pa.struct(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("ref", pa.string(), nullable=False),
    ]
)

# `dfo.schema` becomes Parquet key-value metadata for readers.
SERIES_STATISTICS_SCHEMA = pa.schema(
    [
        pa.field("stat_var", pa.string(), nullable=False),
        pa.field(
            "entities",
            pa.list_(pa.field("item", _ENTITY_TYPE, nullable=False)),
            nullable=False,
        ),
        pa.field("facet", pa.string(), nullable=False),
        pa.field("observations", pa.int64(), nullable=False),
        pa.field("first_value", pa.float64(), nullable=False),
        pa.field("latest_value", pa.float64(), nullable=False),
        pa.field("minimum_value", pa.float64(), nullable=False),
        pa.field("maximum_value", pa.float64(), nullable=False),
        pa.field("first_period", pa.string(), nullable=False),
        pa.field("latest_period", pa.string(), nullable=False),
        pa.field("minimum_period", pa.string(), nullable=False),
        pa.field("maximum_period", pa.string(), nullable=False),
        pa.field("change_absolute", pa.float64(), nullable=True),
        pa.field("change_percent", pa.float64(), nullable=True),
    ],
    metadata={"dfo.schema": SERIES_STATISTICS_SCHEMA_ID},
)


def _normalise_zero(value: float) -> float:
    """Folds `-0.0` to `0.0`. Other floats pass through unchanged."""
    return value + 0.0


def _project_unit(
    ref: str | None,
    currency: str | None,
    prices: str | None,
    base_year: int | None,
) -> str | dict[str, object] | None:
    """Projects a tagged `unit` cell to its wire form.

    Returns `None` when `ref` and `currency` are empty, the bound `ref`, or
    `{currency, prices, base_year?}` for an unbound request. The optional
    `base_year` follows `UnitRequest.as_json()` and is omitted when empty.
    """
    if ref is not None:
        return ref
    if currency is None:
        return None
    projected: dict[str, object] = {"currency": currency, "prices": prices}
    if base_year is not None:
        projected["base_year"] = base_year
    return projected


def _compute_change(
    *, first_value: float, latest_value: float
) -> tuple[float | None, float | None]:
    """Returns absolute and percentage change from first to latest value.

    Absolute change subtracts `Decimal(str(value))` operands at
    `_CHANGE_DECIMAL_PRECISION` precision to avoid binary-float error. An
    overflow returns `(None, None)`. Percentage change is unrounded and is
    `None` for a non-positive baseline or overflow.
    """
    with decimal.localcontext() as ctx:
        ctx.prec = _CHANGE_DECIMAL_PRECISION
        absolute = _normalise_zero(float(Decimal(str(latest_value)) - Decimal(str(first_value))))
    if not math.isfinite(absolute):
        return None, None
    percent = None if first_value <= 0 else absolute / abs(first_value) * 100
    if percent is not None and not math.isfinite(percent):
        percent = None
    return absolute, percent


def _sentinel_key(array: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Casts `array` to strings and fills nulls with `_NULL_SENTINEL`.

    Arrow columns and struct fields may return arrays or chunked arrays.
    Both support the compute operations used here.
    """
    if array.type != pa.string():
        array = pc.cast(array, pa.string())
    filled = pc.fill_null(array, pa.scalar(_NULL_SENTINEL, type=pa.string()))
    return cast("pa.Array | pa.ChunkedArray", filled)


def compute_series_statistics_table(frame: Frame) -> pa.Table:
    """Returns one `SERIES_STATISTICS_SCHEMA` row per distinct series.

    The grain excludes `time_period`, `value`, and `status`. Arrow performs
    grouping and reductions. Stable sorting by period and original row position makes
    first/latest ties follow frame order. Minimum and maximum ties use the
    earliest period. A fourth aggregate finds each series' first row, and
    the joined result is ordered by that position.

    Grouping keys, including decomposed `unit` fields, use `_sentinel_key`
    because Arrow joins drop null keys. Output values come from each
    series' first row in the original table. `entities` and canonical JSON
    `facet` are built in Python once per group.

    An empty frame returns the empty schema table.
    """
    columns = frame.columns
    table = frame.table
    row_count = table.num_rows

    if row_count == 0:
        return SERIES_STATISTICS_SCHEMA.empty_table()

    role_columns = tuple(c for c in columns if c not in _FIXED_COLUMNS)

    table = table.append_column("__orig_idx", pa.array(range(row_count), type=pa.int64()))

    unit_column = table.column(_UNIT_COLUMN)
    unit_ref = pc.struct_field(unit_column, "ref")
    unit_request = pc.struct_field(unit_column, "request")
    unit_currency = pc.struct_field(unit_request, "currency")
    unit_prices = pc.struct_field(unit_request, "prices")
    unit_base_year = pc.struct_field(unit_request, "base_year")
    table = table.append_column("__unit_ref", unit_ref)
    table = table.append_column("__unit_currency", unit_currency)
    table = table.append_column("__unit_prices", unit_prices)
    table = table.append_column("__unit_base_year", unit_base_year)

    key_sources = (
        (_STAT_VAR_COLUMN, table.column(_STAT_VAR_COLUMN)),
        *((role, table.column(role)) for role in role_columns),
        ("__unit_ref", unit_ref),
        ("__unit_currency", unit_currency),
        ("__unit_prices", unit_prices),
        ("__unit_base_year", unit_base_year),
        (_MEASUREMENT_METHOD_COLUMN, table.column(_MEASUREMENT_METHOD_COLUMN)),
        (_PROVENANCE_COLUMN, table.column(_PROVENANCE_COLUMN)),
    )
    grain_columns: list[str] = []
    for name, array in key_sources:
        key_name = f"__key_{name}"
        table = table.append_column(key_name, _sentinel_key(array))
        grain_columns.append(key_name)

    first_appearance = table.group_by(grain_columns, use_threads=False).aggregate(
        (("__orig_idx", "min"),)
    )

    by_period = table.sort_by([(_TIME_PERIOD_COLUMN, "ascending"), ("__orig_idx", "ascending")])
    first_latest = (
        by_period.group_by(grain_columns, use_threads=False)
        # A tuple preserves each aggregate's `Literal` function-name type for
        # pyarrow-stubs.
        .aggregate(
            (
                (_VALUE_COLUMN, "first"),
                (_TIME_PERIOD_COLUMN, "first"),
                (_VALUE_COLUMN, "last"),
                (_TIME_PERIOD_COLUMN, "last"),
                (_VALUE_COLUMN, "count"),
            )
        )
        .rename_columns(
            {
                "value_first": "first_value",
                "time_period_first": "first_period",
                "value_last": "latest_value",
                "time_period_last": "latest_period",
                "value_count": "observations",
                **{c: c for c in grain_columns},
            }
        )
    )

    by_value_ascending = table.sort_by(
        [
            (_VALUE_COLUMN, "ascending"),
            (_TIME_PERIOD_COLUMN, "ascending"),
            ("__orig_idx", "ascending"),
        ]
    )
    minimum = (
        by_value_ascending.group_by(grain_columns, use_threads=False)
        .aggregate(((_VALUE_COLUMN, "first"), (_TIME_PERIOD_COLUMN, "first")))
        .rename_columns(
            {
                "value_first": "minimum_value",
                "time_period_first": "minimum_period",
                **{c: c for c in grain_columns},
            }
        )
    )

    by_value_descending = table.sort_by(
        [
            (_VALUE_COLUMN, "descending"),
            (_TIME_PERIOD_COLUMN, "ascending"),
            ("__orig_idx", "ascending"),
        ]
    )
    maximum = (
        by_value_descending.group_by(grain_columns, use_threads=False)
        .aggregate(((_VALUE_COLUMN, "first"), (_TIME_PERIOD_COLUMN, "first")))
        .rename_columns(
            {
                "value_first": "maximum_value",
                "time_period_first": "maximum_period",
                **{c: c for c in grain_columns},
            }
        )
    )

    joined = first_latest.join(minimum, keys=grain_columns, join_type="inner")
    joined = joined.join(maximum, keys=grain_columns, join_type="inner")
    joined = joined.join(first_appearance, keys=grain_columns, join_type="inner")
    assert joined.num_rows == first_latest.num_rows, (
        "series-statistics grouping key lost or duplicated a series across "
        "the minimum/maximum/first-appearance joins"
    )

    joined = joined.take(pc.sort_indices(joined, sort_keys=[("__orig_idx_min", "ascending")]))

    representative_rows = table.take(joined.column("__orig_idx_min"))
    group_count = joined.num_rows

    # Arrow stubs mark `to_pylist()` values nullable. These casts reflect
    # `Frame.build`'s populated `stat_var`/role fields and sparse facet fields.
    rep_stat_var = cast("list[str]", representative_rows.column(_STAT_VAR_COLUMN).to_pylist())
    rep_roles = {
        role: cast("list[str]", representative_rows.column(role).to_pylist())
        for role in role_columns
    }
    rep_unit_ref = cast("list[str | None]", representative_rows.column("__unit_ref").to_pylist())
    rep_unit_currency = cast(
        "list[str | None]", representative_rows.column("__unit_currency").to_pylist()
    )
    rep_unit_prices = cast(
        "list[str | None]", representative_rows.column("__unit_prices").to_pylist()
    )
    rep_unit_base_year = cast(
        "list[int | None]", representative_rows.column("__unit_base_year").to_pylist()
    )
    rep_measurement_method = cast(
        "list[str | None]", representative_rows.column(_MEASUREMENT_METHOD_COLUMN).to_pylist()
    )
    rep_provenance = cast(
        "list[str | None]", representative_rows.column(_PROVENANCE_COLUMN).to_pylist()
    )

    entities_column: list[list[dict[str, str]]] = []
    facet_column: list[str] = []
    for i in range(group_count):
        entities_column.append([{"role": role, "ref": rep_roles[role][i]} for role in role_columns])
        facet: dict[str, object] = {}
        projected_unit = _project_unit(
            rep_unit_ref[i], rep_unit_currency[i], rep_unit_prices[i], rep_unit_base_year[i]
        )
        if projected_unit is not None:
            facet["unit"] = projected_unit
        if rep_measurement_method[i] is not None:
            facet["measurementMethod"] = rep_measurement_method[i]
        if rep_provenance[i] is not None:
            facet["provenance"] = rep_provenance[i]
        facet_column.append(json.dumps(facet, sort_keys=True, separators=(",", ":")))

    def _values(column: str) -> list[float]:
        return [_normalise_zero(v) for v in cast("list[float]", joined.column(column).to_pylist())]

    def _periods(column: str) -> list[str]:
        return cast("list[str]", joined.column(column).to_pylist())

    first_values = _values("first_value")
    latest_values = _values("latest_value")
    minimum_values = _values("minimum_value")
    maximum_values = _values("maximum_value")
    observations = cast("list[int]", joined.column("observations").to_pylist())

    change_absolute: list[float | None] = []
    change_percent: list[float | None] = []
    for i in range(group_count):
        if observations[i] == 1:
            change_absolute.append(None)
            change_percent.append(None)
            continue
        absolute, percent = _compute_change(
            first_value=first_values[i], latest_value=latest_values[i]
        )
        change_absolute.append(absolute)
        change_percent.append(percent)

    return pa.Table.from_arrays(
        [
            pa.array(rep_stat_var, type=pa.string()),
            pa.array(entities_column, type=SERIES_STATISTICS_SCHEMA.field("entities").type),
            pa.array(facet_column, type=pa.string()),
            pa.array(observations, type=pa.int64()),
            pa.array(first_values, type=pa.float64()),
            pa.array(latest_values, type=pa.float64()),
            pa.array(minimum_values, type=pa.float64()),
            pa.array(maximum_values, type=pa.float64()),
            pa.array(_periods("first_period"), type=pa.string()),
            pa.array(_periods("latest_period"), type=pa.string()),
            pa.array(_periods("minimum_period"), type=pa.string()),
            pa.array(_periods("maximum_period"), type=pa.string()),
            pa.array(change_absolute, type=pa.float64()),
            pa.array(change_percent, type=pa.float64()),
        ],
        schema=SERIES_STATISTICS_SCHEMA,
    )


@dataclass(frozen=True)
class SeriesStatisticsFile:
    """A closed `series-statistics/1` Parquet file and its row count.

    Empty results have zero rows in a valid file.
    """

    data: bytes
    series_count: int


class SeriesStatisticsSink:
    """Writes statistics tables as row groups in one in-memory Parquet file.

    A series includes `stat_var` in its grain and cannot span chunks, so
    appending row groups preserves first-appearance order. `finish()` creates
    a valid zero-row file with schema metadata when nothing was written.
    """

    def __init__(self) -> None:
        self._sink = io.BytesIO()
        self._writer: pq.ParquetWriter | None = None
        self._series_count = 0

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._sink, SERIES_STATISTICS_SCHEMA)
        if table.num_rows:
            self._writer.write_table(table)
        self._series_count += table.num_rows

    def finish(self) -> SeriesStatisticsFile:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._sink, SERIES_STATISTICS_SCHEMA)
        self._writer.close()
        return SeriesStatisticsFile(
            data=bytes(self._sink.getbuffer()), series_count=self._series_count
        )
