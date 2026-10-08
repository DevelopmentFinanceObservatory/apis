"""Build grammar `Frame`s from SDMX rows or Arrow tables.

`build_frame` wraps `Frame.build`. The table functions call
`build_frame_from_arrow` under the observation-table contract. Content-version
minting lives in `content_version.py`, and FRAME1 serialisation in `formats.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import Frame
from dfo_frame_grammar.arrow import build_frame_from_arrow
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest, unit_cell_to_json

from dc_sdmx import SdmxDataResult
from dc_sdmx.results import VALUE_COLUMN, VARIABLE_COLUMN
from materialise.columnar import _as_boolean_array, _as_dictionary_array, observed_values
from materialise.errors import MaterialisationUsageError, UpstreamResponseError

# Mirrors the grammar's internal StatVar prefix. `test_frame_build.py` checks it.
_STAT_VAR_REF_PREFIX = "dcid:"


def _canonicalize_stat_var_ref(value: str) -> str:
    """Adds `dcid:` to bare `variableMeasured` values. Prefixed values pass through."""
    if value.startswith(_STAT_VAR_REF_PREFIX):
        return value
    return f"{_STAT_VAR_REF_PREFIX}{value}"


def build_frame(*, role_keys: Iterable[str], rows: Iterable[Mapping[str, object]]) -> Frame:
    """Wraps `Frame.build`. An empty `rows` iterable yields a columns-only `Frame`."""
    return Frame.build(role_keys=role_keys, rows=rows)


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


def declared_roles(result: SdmxDataResult) -> tuple[str, ...]:
    """Returns the role columns from the SDMX response header, unchanged.

    `role_bindings` can be empty when every entity is unbound. The SDMX
    response header supplies declared roles independently of graph defaults.
    """
    return result.roles
