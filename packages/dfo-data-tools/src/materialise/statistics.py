"""Builds the `series-statistics/1` Parquet artifact for a materialised frame.

Each row represents one series, keyed by every frame column except
`time_period`, `value`, and `status`, in deterministic first-appearance order.
Grouping and reductions use Arrow. Only the `entities` struct and canonical
JSON `facet` are built per group from its first row. The module reads `Frame`
structurally and keys each `unit` cell by `dfo_frame_grammar.unit_codes`.

`SeriesStatisticsSink` also folds every table it writes into `SeriesFigures`,
the version-wide counts a description record stores: series reaching the latest
period, series per source and unit, the observed span and grain, the entities
filling each role with the variables holding data for each, and estimated and provisional value counts. Status counts class each
series' `measurementMethod`, and its provenance where it states no method, through the
declared classification (`materialise.method_status`). A series the mint's estimates continue
(`FacetSelection.continued_by`) is two statistics rows, one per method, and counts as one
series in the figures.
"""

from __future__ import annotations

import decimal
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dfo_frame_grammar import parse_time_period, unit_cell_to_json, unit_cells, unit_codes

from materialise.continued_series import ContinuedSeries, continuation_leads
from materialise.frame_build import Frame
from materialise.method_status import StatusClass, method_status_classes, reported_provenances

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

    Grouping keys use `_sentinel_key` because Arrow joins drop null keys. The
    `unit` key is the cell's `unit_codes` code, so a ref, a converted unit, a
    composed unit and an empty cell each key apart. Output values come from
    each series' first row in the original table. `entities` and canonical
    JSON `facet` are built in Python once per group, the facet's `unit` through
    `unit_cell_to_json`.

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
    codes = unit_codes(unit_column)
    units = unit_cells(unit_column, codes)
    table = table.append_column("__unit_code", codes.codes)

    key_sources = (
        (_STAT_VAR_COLUMN, table.column(_STAT_VAR_COLUMN)),
        *((role, table.column(role)) for role in role_columns),
        ("__unit_code", codes.codes),
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
    rep_unit_code = cast("list[int]", representative_rows.column("__unit_code").to_pylist())
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
        projected_unit = unit_cell_to_json(units[rep_unit_code[i]])
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


@dataclass(frozen=True, slots=True)
class StatusFigures:
    """How many values are estimated and how many provisional. Both are `None` unless
    every series' status is known (`count_statuses`)."""

    estimated: int | None
    provisional: int | None


def count_statuses(
    values_by_facet: Mapping[tuple[str | None, str | None], int],
    *,
    classes: Mapping[str, StatusClass],
    reported: frozenset[str],
) -> StatusFigures:
    """Counts estimated and provisional values from the values each `(measurementMethod,
    provenance)` pair produced.

    A method's status is its class in `classes`. A pair with no method is reported when
    its provenance is in `reported` (`reported_provenances`). A count is known only when
    every pair's status is: one unclassed method, or one method-less pair under another
    provenance, makes both counts `None`, so an unknown status never reads as 0.
    """
    totals: Counter[StatusClass] = Counter()
    for (method, provenance), values in values_by_facet.items():
        if method is None:
            status: StatusClass | None = "reported" if provenance in reported else None
        else:
            status = classes.get(method)
        if status is None:
            return StatusFigures(estimated=None, provisional=None)
        totals[status] += values
    return StatusFigures(estimated=totals["estimated"], provisional=totals["provisional"])


@dataclass(frozen=True, slots=True)
class EntityFigure:
    """One entity filling a role, and how many distinct variables hold data for it."""

    ref: str
    variables: int


@dataclass(frozen=True, slots=True)
class RoleFigure:
    """The distinct entities filling one role column, in first-appearance order."""

    role: str
    values: tuple[EntityFigure, ...]

    @property
    def entities(self) -> int:
        """How many distinct entities fill the role."""
        return len(self.values)


@dataclass(frozen=True, slots=True)
class SourceUnitFigure:
    """How many series one source holds in one unit.

    `source` is the series' `provenance` cell and `unit` its cell's JSON form
    (`unit_cell_to_json`); each is `None` where the frame holds none.
    """

    source: str | None
    unit: str | dict[str, object] | None
    series: int


@dataclass(frozen=True, slots=True)
class SeriesFigures:
    """A version's series figures, folded from its statistics tables.

    `span` runs from the earliest first period's start to the latest period's end, at
    `grain` ("year", "month" or "day"). `latest_period` is the latest end any series
    reaches and `series_reaching_latest` how many reach it. These four are `None` when
    the version holds no series, or when its series' first and latest periods mix grains
    and so do not order. `entities_by_role` and `by_source_unit` keep first-appearance
    order. Each entity counts the distinct variables holding data for it, not their
    series.
    """

    series: int
    span: tuple[str, str] | None
    grain: str | None
    latest_period: str | None
    series_reaching_latest: int | None
    entities_by_role: tuple[RoleFigure, ...]
    by_source_unit: tuple[SourceUnitFigure, ...]
    status: StatusFigures


def _series_identity(
    stat_var: str, entities: list[dict[str, str]], facet: Mapping[str, object]
) -> str:
    """A statistics row's series apart from its method: its variable, entities and facet
    without `measurementMethod`."""
    rest = {key: value for key, value in facet.items() if key != "measurementMethod"}
    return json.dumps([stat_var, entities, rest], sort_keys=True, separators=(",", ":"))


@dataclass
class _FiguresFold:
    """Running state for `SeriesFigures` across a version's statistics tables.

    A series never spans tables, so per-series counts add. Entities and variables can recur
    across tables, so each role keeps, per ref, the set of variables holding data for it.
    A row `continuing` admits extends the series of the row in its table holding the same
    variable, entities, unit and provenance under the method it continues
    (`continued_series.continuation_leads`): it adds its values and its latest period to that
    series and counts no series of its own. A continuing row with no such row is a series.
    """

    continuing: Sequence[ContinuedSeries]
    series: int = 0
    grains: set[str] = field(default_factory=set)
    start: str | None = None
    end: str | None = None
    reaching_end: int = 0
    entities: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    source_units: Counter[tuple[str | None, str]] = field(default_factory=Counter)
    units: dict[str, str | dict[str, object] | None] = field(default_factory=dict)
    values_by_facet: Counter[tuple[str | None, str | None]] = field(default_factory=Counter)

    def add(self, table: pa.Table) -> None:
        stat_vars = cast("list[str]", table.column("stat_var").to_pylist())
        facets = [
            cast("dict[str, object]", json.loads(text))
            for text in cast("list[str]", table.column("facet").to_pylist())
        ]
        entities = cast("list[list[dict[str, str]]]", table.column("entities").to_pylist())
        observations = cast("list[int]", table.column("observations").to_pylist())
        first_periods = [
            parse_time_period(period)
            for period in cast("list[str]", table.column("first_period").to_pylist())
        ]
        latest_periods = [
            parse_time_period(period)
            for period in cast("list[str]", table.column("latest_period").to_pylist())
        ]

        # Each series' latest end, keyed by its first row; a continued series' runs to the
        # later of its two rows'.
        series_ends = {i: latest.end for i, latest in enumerate(latest_periods)}
        if self.continuing:
            leads = continuation_leads(
                self.continuing,
                identities=[
                    _series_identity(stat_vars[i], entities[i], facet)
                    for i, facet in enumerate(facets)
                ],
                methods=[facet.get("measurementMethod") for facet in facets],
                provenances=[facet.get("provenance") for facet in facets],
            )
            for i, lead in enumerate(leads):
                if lead is not None:
                    series_ends[lead] = max(series_ends[lead], series_ends.pop(i))

        for i, (stat_var, facet, refs, values) in enumerate(
            zip(stat_vars, facets, entities, observations, strict=True)
        ):
            first, latest = first_periods[i], latest_periods[i]
            self.grains.update((first.grain, latest.grain))
            if self.start is None or first.start < self.start:
                self.start = first.start

            for entity in refs:
                held = self.entities.setdefault(entity["role"], {})
                held.setdefault(entity["ref"], set()).add(stat_var)

            provenance = cast("str | None", facet.get("provenance"))
            self.values_by_facet[
                (cast("str | None", facet.get("measurementMethod")), provenance)
            ] += values

            if (end := series_ends.get(i)) is None:
                continue
            if self.end is None or end > self.end:
                self.end, self.reaching_end = end, 0
            if end == self.end:
                self.reaching_end += 1
            unit = cast("str | dict[str, object] | None", facet.get("unit"))
            unit_key = json.dumps(unit, sort_keys=True, separators=(",", ":"))
            self.units.setdefault(unit_key, unit)
            self.source_units[(provenance, unit_key)] += 1
        self.series += len(series_ends)

    def figures(self) -> SeriesFigures:
        span = (
            (self.start, self.end)
            if len(self.grains) == 1 and self.start is not None and self.end is not None
            else None
        )
        return SeriesFigures(
            series=self.series,
            span=span,
            grain=next(iter(self.grains)) if span else None,
            latest_period=span[1] if span else None,
            series_reaching_latest=self.reaching_end if span else None,
            entities_by_role=tuple(
                RoleFigure(
                    role=role,
                    values=tuple(
                        EntityFigure(ref=ref, variables=len(stat_vars))
                        for ref, stat_vars in held.items()
                    ),
                )
                for role, held in self.entities.items()
            ),
            by_source_unit=tuple(
                SourceUnitFigure(source=source, unit=self.units[unit_key], series=series)
                for (source, unit_key), series in self.source_units.items()
            ),
            status=count_statuses(
                self.values_by_facet,
                classes=method_status_classes(),
                reported=reported_provenances(),
            ),
        )


@dataclass(frozen=True)
class SeriesStatisticsFile:
    """A closed `series-statistics/1` Parquet file, its series count, and the figures
    folded from its rows. `series_count` is `figures.series`, so a series and the estimates
    continuing it count once though the file holds a row for each method.

    Empty results have zero rows in a valid file.
    """

    data: bytes
    series_count: int
    figures: SeriesFigures


class SeriesStatisticsSink:
    """Writes statistics tables as row groups in one in-memory Parquet file and folds
    them into `SeriesFigures`.

    A series includes `stat_var` in its grain and cannot span chunks, so
    appending row groups preserves first-appearance order. `finish()` creates
    a valid zero-row file with schema metadata when nothing was written.

    `continuing` holds the continuations of the frame's series: a dataset version's from its
    selections or trace (`facets.selected_continued_series`,
    `continued_series.continued_series_of_trace`), a computed version's as it stores them, and
    empty when nothing continues them. The figures count a series and its continuation as one
    series; the file keeps a row for each.
    """

    def __init__(self, *, continuing: Sequence[ContinuedSeries]) -> None:
        self._sink = io.BytesIO()
        self._writer: pq.ParquetWriter | None = None
        self._figures = _FiguresFold(continuing=continuing)

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._sink, SERIES_STATISTICS_SCHEMA)
        if table.num_rows:
            self._writer.write_table(table)
        self._figures.add(table)

    def finish(self) -> SeriesStatisticsFile:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._sink, SERIES_STATISTICS_SCHEMA)
        self._writer.close()
        figures = self._figures.figures()
        return SeriesStatisticsFile(
            data=bytes(self._sink.getbuffer()), series_count=figures.series, figures=figures
        )
