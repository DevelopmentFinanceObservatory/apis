"""Vectorised filtering, matching, and aggregation over
`dc_sdmx.results.observation_schema` tables."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from dc_sdmx.errors import SdmxResponseError
from dc_sdmx.results import VALUE_COLUMN, VARIABLE_COLUMN
from materialise.facets import CoverageAdmission, FacetSelection, RowAdmission
from materialise.window import (
    AppliedWindow,
    Coverage,
    RequestedWindow,
    _leading_year,
    latest_period,
)

# `dc_sdmx.results.DIMENSION_COLUMNS`'s trailing entry, the wire's own SDMX
# component name. Kept as a literal here, the same way `test_observation_
# schema.py` spells every contract column name, since `dc_sdmx.results`
# exposes no separate `TIME_PERIOD_COLUMN` constant of its own.
_TIME_PERIOD_COLUMN = "TIME_PERIOD"

# Strip `dcid:` where coverage and member mappings use bare StatVar ids.
_STAT_VAR_REF_PREFIX = "dcid:"

# The columns `admission_mask` reads: a subject-only row's place and its facet id.
_PLACE_COLUMN = "observationAbout"
_FACET_ID_COLUMN = "facetId"

# `_NUMERIC_RE` covers `pc.cast`'s signed integers/decimals (including `.5` and `5.`),
# exponents, and case-insensitive nan/inf/infinity. `observed_values` handles Python `float()`'s
# extra whitespace and digit-grouping underscores.
_NUMERIC_RE = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|^[+-]?(nan|inf|infinity)$"

_EMPTY_STRING_SCALAR = pa.scalar("", type=pa.string())
_FALSE_SCALAR = pa.scalar(False, type=pa.bool_())
_TRUE_SCALAR = pa.scalar(True, type=pa.bool_())


def _as_dictionary_array(array: pa.Array) -> pa.DictionaryArray:
    """Narrows `pc.dictionary_encode`'s string-array result to `pa.DictionaryArray`,
    which its stubs type as the input's generic array type."""
    if not isinstance(array, pa.DictionaryArray):
        raise TypeError(
            f"pc.dictionary_encode() returned {type(array).__name__}, expected DictionaryArray"
        )
    return array


def _as_boolean_array(
    value: pa.Array | pa.ChunkedArray | pa.Scalar | pc.Expression,
) -> pa.BooleanArray:
    """Narrows array comparisons to `pa.BooleanArray`. PyArrow's stubs include scalar and
    expression results, so chunked table-column results are combined before the type check."""
    if not isinstance(value, pa.Array | pa.Scalar | pc.Expression):
        value = value.combine_chunks()
    if not isinstance(value, pa.BooleanArray):
        raise TypeError(f"expected a pa.BooleanArray, got {type(value).__name__}")
    return value


def _map_dictionary_values(
    array: pa.Array | pa.ChunkedArray,
    *,
    map_value: Callable[[str | None], object],
    result_type: pa.DataType,
) -> pa.Array:
    """Maps each distinct dictionary value once, then expands results by row index.
    Combines `pa.ChunkedArray` columns first. Record-batch arrays are already unchunked."""
    if not isinstance(array, pa.Array):
        array = array.combine_chunks()
    dict_array = _as_dictionary_array(pc.dictionary_encode(array))
    mapped_dictionary = pa.array(
        [map_value(value) for value in dict_array.dictionary.to_pylist()], type=result_type
    )
    return mapped_dictionary.take(dict_array.indices)


def observed_values(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Parses `OBS_VALUE` as `float64`. Nulls, empty strings, and invalid values become null.
    Matches `SdmxObservation.value_as_float`: NaN and infinities count as observations.
    `pc.cast` handles `_NUMERIC_RE` matches. Python `float()` handles residual whitespace
    and digit-grouping underscores, with `ValueError` mapped to null."""
    if not isinstance(values, pa.Array):
        values = values.combine_chunks()
    is_valid = pc.is_valid(values)
    is_blank = pc.equal(values, _EMPTY_STRING_SCALAR)
    non_empty = pc.and_(is_valid, pc.invert(is_blank))
    is_numeric_literal = _as_boolean_array(
        pc.match_substring_regex(values, _NUMERIC_RE, ignore_case=True)
    )
    is_numeric = pc.and_(
        non_empty, _as_boolean_array(pc.fill_null(is_numeric_literal, _FALSE_SCALAR))
    )
    null_string = pa.scalar(None, type=pa.string())
    castable = pc.if_else(is_numeric, values, null_string)
    cast_part = pc.cast(castable, pa.float64())

    residual = pc.and_(non_empty, pc.invert(is_numeric))
    residual_as_int = pc.cast(residual, pa.int64())
    residual_count = pc.sum(residual_as_int).as_py()
    if residual_count in (None, 0):
        return cast_part

    raw = values.to_pylist()
    cast_values = cast_part.to_pylist()
    is_residual = residual.to_pylist()
    merged: list[float | None] = []
    for raw_value, cast_value, residual_flag in zip(raw, cast_values, is_residual, strict=True):
        if not residual_flag:
            merged.append(cast_value)
            continue
        if raw_value is None:
            merged.append(None)
            continue
        try:
            merged.append(float(raw_value))
        except ValueError:
            merged.append(None)
    return pa.array(merged, type=pa.float64())


def observed_mask(table: pa.Table) -> pa.BooleanArray:
    """Returns a validity mask from `observed_values`. A true value marks a row whose
    `OBS_VALUE` cell parses as a float."""
    return pc.is_valid(observed_values(table[VALUE_COLUMN]))


def leading_years(table: pa.Table) -> pa.Array:
    """Returns each `TIME_PERIOD`'s leading year as `int64`. Parses with `int(period[:4])` and
    returns null for null cells or `ValueError`. Low-cardinality periods are parsed once each
    through `_map_dictionary_values`, then mapped back to their rows."""
    return _map_dictionary_values(
        table[_TIME_PERIOD_COLUMN], map_value=_leading_year, result_type=pa.int64()
    )


def range_window_mask(table: pa.Table, *, window: RequestedWindow) -> pa.BooleanArray:
    """Keeps rows within the requested year bounds. Null or unparseable years pass, as do
    unstated bounds. Comparisons that return null are filled with `True`."""
    years = leading_years(table)
    start_year = pa.scalar(window.start_year, type=pa.int64())
    end_year = pa.scalar(window.end_year, type=pa.int64())
    above_start = _as_boolean_array(pc.greater_equal(years, start_year))
    below_end = _as_boolean_array(pc.less_equal(years, end_year))
    lower_ok = _as_boolean_array(pc.fill_null(above_start, _TRUE_SCALAR))
    upper_ok = _as_boolean_array(pc.fill_null(below_end, _TRUE_SCALAR))
    return pc.and_(lower_ok, upper_ok)


def latest_rows(table: pa.Table) -> pa.Table:
    """Keeps the rows at the table's latest period (`window.latest_period` over its distinct
    `TIME_PERIOD`s), in input order. A table holding one StatVar's rows keeps the one period
    its `latest` window serves, and only the series holding it. Rows without a parseable
    period are dropped. Empty input returns empty."""
    periods = table[_TIME_PERIOD_COLUMN]
    latest = latest_period(pc.unique(periods).to_pylist())
    if latest is None:
        return table.slice(0, 0)
    return table.filter(pc.equal(periods, pa.scalar(latest, type=pa.string())))


def selection_mask(table: pa.Table, *, selections: Mapping[str, FacetSelection]) -> pa.BooleanArray:
    """For every axis in `selections`, the row's cell in that axis's own
    column (`provenance`, `measurementMethod`, or `unit`, the axis name is
    already the contract column name, see `materialise.facets.FACET_AXES`)
    equals `selection.ref`, a null cell equal to a `None` ref. All given axes
    must agree (an AND across axes). No axes given means every row matches.
    """
    mask: pa.BooleanArray | None = None
    for axis, selection in selections.items():
        column = table[axis]
        if selection.ref is None:
            axis_mask = pc.is_null(column)
        else:
            matches = _as_boolean_array(
                pc.equal(column, pa.scalar(selection.ref, type=pa.string()))
            )
            axis_mask = _as_boolean_array(pc.fill_null(matches, _FALSE_SCALAR))
        mask = axis_mask if mask is None else pc.and_(mask, axis_mask)
    if mask is None:
        return pa.array([True] * table.num_rows, type=pa.bool_())
    return mask


def admission_mask(table: pa.Table, *, admission: RowAdmission) -> pa.BooleanArray:
    """Keeps a row whose `facetId` was admitted at its `observationAbout`. `CoverageAdmission`
    keeps every row.

    Every masked read filters on `c[facetId]` and names the subject places, so a conforming
    response carries a facet and a place on every row. The parser fills an absent `facetId`
    column with nulls and maps an empty `facetId` cell or an empty or `NotApplicable` place to
    null, so under `PlaceAdmission` a null in either column raises `SdmxResponseError`: the
    row's admission cannot be checked."""
    if isinstance(admission, CoverageAdmission):
        return pa.array([True] * table.num_rows, type=pa.bool_())
    for column in (_FACET_ID_COLUMN, _PLACE_COLUMN):
        missing = table[column].null_count
        if missing:
            raise SdmxResponseError(
                f"{missing} of {table.num_rows} SDMX rows carry no {column}, so their "
                "admission cannot be checked; a masked read must name the facet and the "
                "place on every row"
            )
    facet_ids = table[_FACET_ID_COLUMN]
    places = table[_PLACE_COLUMN]
    mask: pa.BooleanArray = pa.array([False] * table.num_rows, type=pa.bool_())
    for facet_id in pc.unique(facet_ids).drop_null().to_pylist():
        admitted_places = admission.places_by_facet.get(facet_id)
        if not admitted_places:
            continue
        is_facet = pc.equal(facet_ids, pa.scalar(facet_id, type=pa.string()))
        at_admitted_place = pc.is_in(
            places, value_set=pa.array(sorted(admitted_places), type=pa.string())
        )
        admitted = _as_boolean_array(
            pc.fill_null(pc.and_(is_facet, at_admitted_place), _FALSE_SCALAR)
        )
        mask = _as_boolean_array(pc.or_(mask, admitted))
    return mask


def survivor_mask(
    table: pa.Table,
    *,
    selections: Mapping[str, FacetSelection],
    window: RequestedWindow,
    admission: RowAdmission,
) -> pa.BooleanArray:
    """Combines facet selection, admission and observed-value masks, adding a year-range mask
    only for `window.kind == "range"`."""
    mask = pc.and_(selection_mask(table, selections=selections), observed_mask(table))
    mask = pc.and_(mask, admission_mask(table, admission=admission))
    if window.kind == "range":
        mask = pc.and_(mask, range_window_mask(table, window=window))
    return mask


def _distinct_non_null_values(
    values: pa.Array | pa.ChunkedArray,
) -> frozenset[str | int]:
    """Returns the distinct, non-null scalar values from an Arrow column."""
    return frozenset(
        value for value in pc.unique(values).drop_null().to_pylist() if value is not None
    )


def coverage(
    table: pa.Table, *, declared_roles: Sequence[str], applied_window: AppliedWindow | None
) -> Coverage:
    """`rows` is the table row count. `dimensions` lists distinct non-null values per declared
    role, then `years`. `entity_refs` holds the counted values for each label. `window` is
    `applied_window.as_trace_entry()` or `None`."""
    dimensions: list[dict[str, object]] = []
    entity_refs: dict[str, frozenset[str | int]] = {}
    for role in declared_roles:
        distinct_entities = _distinct_non_null_values(table[role])
        dimensions.append({"label": role, "count": len(distinct_entities)})
        entity_refs[role] = distinct_entities
    distinct_years = _distinct_non_null_values(leading_years(table))
    dimensions.append({"label": "years", "count": len(distinct_years)})
    entity_refs["years"] = distinct_years
    window_entry = applied_window.as_trace_entry() if applied_window is not None else None
    return Coverage(
        rows=table.num_rows,
        dimensions=tuple(dimensions),
        window=window_entry,
        entity_refs=entity_refs,
    )


def served_entities_by_role(table: pa.Table, *, roles: Sequence[str]) -> dict[str, set[str]]:
    """The distinct non-null dcid values served in each role column. Only a
    role with at least one non-null cell gets a key.
    """
    served: dict[str, set[str]] = {}
    for role in roles:
        distinct = {value for value in table[role].drop_null().to_pylist() if value is not None}
        if distinct:
            served[role] = distinct
    return served


def served_year_bounds(table: pa.Table) -> tuple[int, int] | None:
    """The minimum and maximum of `leading_years(table)`, nulls skipped, the
    input `window.compute_applied_window` takes as `served_years`. `None`
    when `table` has no row with a parseable leading year, an empty table
    included.
    """
    bounds = pc.min_max(leading_years(table))
    minimum = bounds["min"].as_py()
    maximum = bounds["max"].as_py()
    if minimum is None or maximum is None:
        return None
    return minimum, maximum


def split_by_variable(
    batch: pa.RecordBatch, *, index_by_bare_stat_var: Mapping[str, int]
) -> dict[int, pa.RecordBatch]:
    """Routes rows by bare `variableMeasured`, stripping `dcid:` if present. Returns one
    batch per distinct member index, including empty batches. Rows with unknown variables
    are dropped."""

    def _member_index(value: str | None) -> int | None:
        if value is None:
            return None
        return index_by_bare_stat_var.get(value.removeprefix(_STAT_VAR_REF_PREFIX))

    target_index = _map_dictionary_values(
        batch[VARIABLE_COLUMN], map_value=_member_index, result_type=pa.int64()
    )
    return {
        member_index: batch.filter(pc.equal(target_index, pa.scalar(member_index, type=pa.int64())))
        for member_index in set(index_by_bare_stat_var.values())
    }
