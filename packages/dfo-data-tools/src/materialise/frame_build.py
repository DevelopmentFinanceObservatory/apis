"""Build grammar `Frame`s from SDMX rows or Arrow tables.

`build_frame` wraps `Frame.build`. The table functions call
`build_frame_from_arrow` under the observation-table contract. Content-version
minting lives in `content_version.py`, and FRAME1 serialisation in `formats.py`.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dfo_frame_grammar import (
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    Frame,
    canonical_frame_schema,
    derive_columns,
    parse_time_period,
    widen_frame_table,
)
from dfo_frame_grammar.arrow import build_frame_from_arrow
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest, unit_cell_to_json

from dc_sdmx import SdmxDataResult
from dc_sdmx.results import VALUE_COLUMN, VARIABLE_COLUMN
from materialise.columnar import _as_boolean_array, _as_dictionary_array, observed_values
from materialise.errors import MaterialisationUsageError, UpstreamResponseError

RECIPE_UNIT_TYPE = canonical_frame_schema(derive_columns(()), recipe=True).field(UNIT_COLUMN).type
"""The `unit` column's type when some cell carries a recipe."""

# Mirrors the grammar's internal StatVar prefix. `test_frame_build.py` checks it.
_STAT_VAR_REF_PREFIX = "dcid:"
_MEASUREMENT_METHOD_COLUMN = "measurementMethod"


def _canonicalize_stat_var_ref(value: str) -> str:
    """Adds `dcid:` to bare `variableMeasured` values. Prefixed values pass through."""
    if value.startswith(_STAT_VAR_REF_PREFIX):
        return value
    return f"{_STAT_VAR_REF_PREFIX}{value}"


def build_frame(*, role_keys: Iterable[str], rows: Iterable[Mapping[str, object]]) -> Frame:
    """Wraps `Frame.build`. An empty `rows` iterable yields a columns-only `Frame`."""
    return Frame.build(role_keys=role_keys, rows=rows)


@dataclass(frozen=True)
class ParquetFrameChunks:
    """A fresh pass over a canonical frame held as Parquet bytes, one `Frame` per row group, on
    each call. Each pass opens a new `ParquetFile` over a zero-copy `BufferReader`, so repeated
    passes copy nothing and hold at most one row group's rows. Every `Frame` takes the frame's
    own `unit` layout, the recipe layout when `recipe`, a cell of the frame carrying a recipe,
    whatever layout the bytes hold (`widen_frame_table`). `column_chunks` reads one column other
    than `unit` without decoding the others."""

    parquet_bytes: bytes | memoryview
    columns: tuple[str, ...]
    recipe: bool

    def __call__(self) -> Iterator[Frame]:
        reader = pq.ParquetFile(pa.BufferReader(self.parquet_bytes))
        for row_group_index in range(reader.num_row_groups):
            table = widen_frame_table(reader.read_row_group(row_group_index), recipe=self.recipe)
            yield Frame(columns=self.columns, table=table)

    def column_chunks(self, column: str) -> Iterator[pa.ChunkedArray]:
        """`column` of each row group in order."""
        reader = pq.ParquetFile(pa.BufferReader(self.parquet_bytes))
        for row_group_index in range(reader.num_row_groups):
            yield reader.read_row_group(row_group_index, columns=[column])[column]

    @property
    def num_rows(self) -> int:
        """The frame's row count, read from the Parquet footer alone."""
        return pq.ParquetFile(pa.BufferReader(self.parquet_bytes)).metadata.num_rows


class ParquetFrameSink:
    """Writes each frame chunk as one row group of an in-memory Parquet file, which `finish`
    returns as the `ParquetFrameChunks` that reads it back chunk for chunk.

    A chunk carries the recipe `unit` layout when one of its own cells carries a recipe, so a
    later chunk can need the layout an earlier one lacks. Every row group is written in the
    recipe layout, which keeps every cell, and the frame's own layout is applied as it is read
    back. The first chunk sets the shared columns. Durable storage happens only after a
    successful materialisation.
    """

    def __init__(self) -> None:
        self._sink = io.BytesIO()
        self._writer: pq.ParquetWriter | None = None
        self._recipe = False

    def write(self, table: pa.Table) -> None:
        own = widen_frame_table(table)
        self._recipe = self._recipe or own.schema.field(UNIT_COLUMN).type == RECIPE_UNIT_TYPE
        widened = widen_frame_table(own, recipe=True)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._sink, widened.schema)
        self._writer.write_table(widened)

    def finish(self, *, columns: tuple[str, ...]) -> ParquetFrameChunks:
        """The frame written, whose columns are `columns`, read back from a zero-copy view of
        the bytes. The view keeps the sink alive and blocks later writes, so this is the last
        call."""
        assert self._writer is not None, "ParquetFrameSink.finish() called before any chunk"
        self._writer.close()
        return ParquetFrameChunks(self._sink.getbuffer(), columns=columns, recipe=self._recipe)


def unit_cell_for_rebuild(value: object) -> object:
    """Converts a `UnitRequest` or `ComposedUnit` cell to JSON for rebuilding a `Frame` from
    `.rows`.

    `Frame.build` accepts a string, mapping, or `None` for `unit`, but rejects
    `UnitRequest` and `ComposedUnit` with `FrameError`. Other cell values pass through
    unchanged.
    """
    if isinstance(value, (UnitRequest, ComposedUnit)):
        return unit_cell_to_json(value)
    return value


def _canonical_variable_column(table: pa.Table) -> pa.Array:
    """`table[variableMeasured]`, canonicalised to the grammar's
    `dcid:`-prefixed spelling per distinct value: the column is
    dictionary-encoded, `_canonicalize_stat_var_ref` runs once per distinct
    value, and the result is mapped back through the dictionary indices.
    """
    dict_array = _as_dictionary_array(pc.dictionary_encode(table[VARIABLE_COLUMN]).combine_chunks())
    canonical_dictionary = pa.array(
        [
            _canonicalize_stat_var_ref(value) if value is not None else None
            for value in dict_array.dictionary.to_pylist()
        ],
        type=pa.string(),
    )
    return canonical_dictionary.take(dict_array.indices)


def _grammar_table(table: pa.Table, *, role_keys: Sequence[str], stat_var: str) -> pa.Table:
    """Validates `table` and derives the grammar table used by both table builders.

    `stat_var` must start with `dcid:` or raises `MaterialisationUsageError`.
    Each canonicalised `variableMeasured` value must match it, or the first
    mismatch in table order raises `UpstreamResponseError`. Null parsed values
    raise `MaterialisationUsageError` because callers must filter observed rows.

    The output contains `stat_var`, each role, `time_period`, `value`, `unit`,
    `measurementMethod`, and `provenance`. It remains a table so group members'
    tables can be concatenated before one `build_frame_from_arrow` call.
    """
    if not stat_var.startswith(_STAT_VAR_REF_PREFIX):
        raise MaterialisationUsageError(
            f"build_frame_from_table() requires a dcid:-prefixed stat_var, got {stat_var!r}"
        )
    canonical_stat_var = _canonicalize_stat_var_ref(stat_var)
    role_keys = tuple(role_keys)

    canonical_variables = _canonical_variable_column(table)
    mismatches = _as_boolean_array(
        pc.not_equal(canonical_variables, pa.scalar(canonical_stat_var, type=pa.string()))
    )
    if pc.any(mismatches).as_py():
        offending_index = pc.index(mismatches, True).as_py()
        assert offending_index is not None
        offending_wire_value = table[VARIABLE_COLUMN].combine_chunks()[offending_index].as_py()
        offending_canonical = canonical_variables[offending_index].as_py()
        raise UpstreamResponseError(
            f"SDMX response row's variableMeasured {offending_wire_value!r} "
            f"(normalized {offending_canonical!r}) does not match the requested "
            f"stat_var {stat_var!r}"
        )

    values = observed_values(table[VALUE_COLUMN])
    if pc.any(pc.is_null(values)).as_py():
        raise MaterialisationUsageError(
            "build_frame_from_table() requires table to already be filtered to "
            "observed rows; a null OBS_VALUE reached the grammar table"
        )

    grammar_columns: dict[str, object] = {"stat_var": canonical_variables}
    for role in role_keys:
        grammar_columns[role] = table[role]
    grammar_columns["time_period"] = table["TIME_PERIOD"]
    grammar_columns["value"] = values
    grammar_columns["unit"] = table["unit"]
    grammar_columns["measurementMethod"] = table["measurementMethod"]
    grammar_columns["provenance"] = table["provenance"]

    return pa.table(grammar_columns)


def build_frame_from_table(table: pa.Table, *, role_keys: Sequence[str], stat_var: str) -> Frame:
    """Builds from an observed-row table under `observation_schema`.

    Empty tables produce the same columns-only `Frame` as `build_frame`.
    """
    role_keys = tuple(role_keys)
    grammar_table = _grammar_table(table, role_keys=role_keys, stat_var=stat_var)
    return build_frame_from_arrow(grammar_table, role_keys=role_keys)


def build_frame_from_member_tables(
    tables_by_stat_var: Sequence[tuple[str, pa.Table]], *, role_keys: Sequence[str]
) -> Frame:
    """Builds one group `Frame` from each member's `(stat_var, table)` pair.

    Each table gets `_grammar_table`'s checks, then the tables are
    concatenated before one `build_frame_from_arrow` call. Empty input yields
    a columns-only `Frame` with no rows.
    """
    role_keys = tuple(role_keys)
    if not tables_by_stat_var:
        empty_columns: dict[str, object] = {"stat_var": pa.array([], type=pa.string())}
        for role in role_keys:
            empty_columns[role] = pa.array([], type=pa.string())
        empty_columns["time_period"] = pa.array([], type=pa.string())
        empty_columns["value"] = pa.array([], type=pa.float64())
        empty_columns["unit"] = pa.array([], type=pa.string())
        empty_columns["measurementMethod"] = pa.array([], type=pa.string())
        empty_columns["provenance"] = pa.array([], type=pa.string())
        return build_frame_from_arrow(pa.table(empty_columns), role_keys=role_keys)

    grammar_tables = [
        _grammar_table(table, role_keys=role_keys, stat_var=stat_var)
        for stat_var, table in tables_by_stat_var
    ]
    combined = pa.concat_tables(grammar_tables)
    return build_frame_from_arrow(combined, role_keys=role_keys)


def period_grains(frame: Frame) -> dict[str, str]:
    """Each grain of `frame`'s periods (`parse_time_period`) mapped to its earliest period.
    One entry for a frame that binds, none for an empty one."""
    grains: dict[str, str] = {}
    periods = pc.unique(frame.table[TIME_PERIOD_COLUMN]).to_pylist()
    for period in sorted(str(period) for period in periods if period is not None):
        grains.setdefault(parse_time_period(period).grain, period)
    return grains


def fold_method_periods(held: dict[str, tuple[str, str]], frame: Frame) -> None:
    """Widens `held`, each `measurementMethod` mapped to the earliest and latest period its
    rows hold, by `frame`'s rows. Rows with no method are left out. Periods compare as
    strings, which orders the periods of one grain, as `columnar.continuation_mask` does."""
    table = frame.table
    if table.num_rows == 0:
        return
    bounds = table.group_by(_MEASUREMENT_METHOD_COLUMN).aggregate(
        ((TIME_PERIOD_COLUMN, "min"), (TIME_PERIOD_COLUMN, "max"))
    )
    for method, first, last in zip(
        bounds[_MEASUREMENT_METHOD_COLUMN].to_pylist(),
        bounds[f"{TIME_PERIOD_COLUMN}_min"].to_pylist(),
        bounds[f"{TIME_PERIOD_COLUMN}_max"].to_pylist(),
        strict=True,
    ):
        if method is None or first is None or last is None:
            continue
        if (earlier := held.get(method)) is not None:
            first, last = min(earlier[0], first), max(earlier[1], last)
        held[method] = (first, last)


def declared_roles(result: SdmxDataResult) -> tuple[str, ...]:
    """Returns the role columns from the SDMX response header, unchanged.

    `role_bindings` can be empty when every entity is unbound. The SDMX
    response header supplies declared roles independently of graph defaults.
    """
    return result.roles
