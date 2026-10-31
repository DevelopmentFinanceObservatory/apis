"""A groupby collapse over a canonical frame held as Arrow: the decode (`reshape.decode_columns`),
the fold of its first axis (`reshape.fold_groupby_sum_chunks`) and the sum of each later axis
(`reshape.groupby_sum`), each with the row engine's result and refusals.

`reshape` reads Python rows and keeps no Arrow or grammar import. A materialised frame arrives as
Arrow, and folding it row by row costs one Python call per row and cell. This module groups,
checks and sums with Arrow kernels and leaves its results as Arrow, so a collapse builds Python
rows only for the groups its last axis keeps (`rows_table`). Where the input would draw a
refusal, or an output the plain sum does not give (an estimated total), it hands the rows to
the row engine, so every refusal is the row engine's own.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa
from dfo_frame_grammar import (
    UNIT_COLUMN,
    UnitCodes,
    array_kernel,
    scalar_kernel,
    unit_cells,
    unit_codes,
)

from materialise.codes import MaterialisationWarning
from materialise.columnar import strictly_increasing
from materialise.errors import MaterialisationError
from materialise.formats import canonical_frame_rows
from materialise.reshape import (
    _VALUE_COLUMN,
    ColumnClassification,
    ExpectedPopulation,
    RollupNotSummableError,
    Table,
    _cell_sort_key,
    _GroupbySumFoldPlan,
    _plan_groupby_sum_fold,
    _refuse_duplicate_requests,
    _refuse_unknown_entity_columns,
    classify_columns,
    enrichment_unavailable_warning,
    fold_groupby_sum_chunks,
    groupby_sum,
    lookup_cell,
)


def decode_table(
    table: pa.Table,
    *,
    requests: Sequence[tuple[str, str]],
    lookup: Mapping[str, Mapping[str, str]],
) -> tuple[pa.Table, tuple[MaterialisationWarning, ...]]:
    """`reshape.decode_columns` over an Arrow table: one `{entity_column}_{code_system}` string
    column per request, in request order, read with `reshape.lookup_cell` once per distinct
    entity value. A missing reading is an empty cell, and each request with any gets one
    `enrichment_unavailable` warning with its row count. Refuses as `decode_columns` does."""
    _refuse_unknown_entity_columns(
        Table(columns=tuple(table.column_names), rows=()),
        requests=requests,
        caller_name="decode_columns",
    )
    _refuse_duplicate_requests(requests, caller_name="decode_columns", field_noun="code_system")
    warnings: list[MaterialisationWarning] = []
    decoded = table
    for entity_column, code_system in requests:
        codes = unit_codes(table.column(entity_column))
        refs = table.column(entity_column).take(codes.first_rows).to_pylist()
        readings = [lookup_cell(lookup, ref, code_system) for ref in refs]
        cells = pa.array(["" if reading is None else reading for reading in readings], pa.string())
        name = f"{entity_column}_{code_system}"
        decoded = decoded.append_column(name, cells.take(codes.codes))
        missing = pa.array([reading is None for reading in readings]).take(codes.codes)
        if count := cast(int, scalar_kernel("sum", missing).as_py()):
            warnings.append(enrichment_unavailable_warning(column=name, count=count))
    return decoded, tuple(warnings)


type GroupedTable = pa.Table | Table
"""A groupby result: an Arrow table from this module's sums, or a `reshape.Table` where the row
engines gave it."""


@dataclass(frozen=True)
class ColumnarFoldOutcome:
    """`reshape.GroupbySumChunksOutcome` whose `table` may be Arrow. Exactly one of `table` and
    `violation` is set."""

    table: GroupedTable | None
    violation: RollupNotSummableError | ValueError | None


def fold_groupby_sum_table(
    table: pa.Table,
    *,
    axis: str,
    classification: ColumnClassification,
    summable_axes: frozenset[str],
    missing_declared_as_zero: bool = False,
    expected_population: ExpectedPopulation | None = None,
    value_column: str = _VALUE_COLUMN,
) -> ColumnarFoldOutcome:
    """`reshape.fold_groupby_sum_chunks` over the rows of `table`, in order, with no hierarchy.

    Each group's total is the sequential sum of its values in row order, the row fold's `+=`,
    which Arrow's grouped `sum` computes the same way. The rows go to `fold_groupby_sum_chunks`
    when they are not in strictly increasing full-grain order (out of order or a repeated
    grain), or `_sums_plainly` finds a refusal or an estimated total, one slice of rows at a
    time (`_row_slices`); its outcome then holds a `reshape.Table`.
    """
    _refuse_facet_axis(axis, classification)
    columns = tuple(table.column_names)
    plan = _plan_groupby_sum_fold(
        columns,
        axis=axis,
        classification=classification,
        summable_axes=summable_axes,
        hierarchy=None,
        missing_declared_as_zero=missing_declared_as_zero,
        value_column=value_column,
        expected_population=expected_population,
    )
    grouped = (
        _grouped_sums(table, plan, sequential=True)
        if table.num_rows and strictly_increasing(_full_grain_keys(table, plan))
        else None
    )
    if grouped is not None:
        return ColumnarFoldOutcome(table=grouped, violation=None)
    outcome = fold_groupby_sum_chunks(
        _row_slices(table),
        axis=axis,
        classification=classification,
        summable_axes=summable_axes,
        missing_declared_as_zero=missing_declared_as_zero,
        expected_population=expected_population,
        value_column=value_column,
    )
    return ColumnarFoldOutcome(table=outcome.table, violation=outcome.violation)


# Rows the row fold reads at a time, the writers' slice size. A row holds about 12 times the
# bytes of its Arrow cells.
_FALLBACK_SLICE_ROWS = 8_192


def _row_slices(table: pa.Table) -> Iterator[Table]:
    """`table`'s rows as `reshape.Table`s of up to `_FALLBACK_SLICE_ROWS` rows, in order, built
    one at a time. A table with no rows gives one empty table."""
    for start in range(0, max(table.num_rows, 1), _FALLBACK_SLICE_ROWS):
        yield rows_table(table.slice(start, _FALLBACK_SLICE_ROWS))


def groupby_sum_table(
    table: pa.Table,
    *,
    axis: str,
    summable_axes: frozenset[str],
    missing_declared_as_zero: bool = False,
    expected_population: ExpectedPopulation | None = None,
) -> GroupedTable:
    """`reshape.apply_reshapes` with a lone `GroupbyPlan` (no hierarchy) over `table`: the
    columns classified from `table`, then `reshape.groupby_sum`.

    Each group's total is Python's `sum` over its values in row order, as `groupby_sum` sums
    them. The rows go to `groupby_sum` when the grouping preconditions fail, a full grain
    repeats, or `_sums_plainly` finds a refusal or an estimated total; it then returns a
    `reshape.Table` or raises.
    """
    classification = classify_columns(tuple(table.column_names))
    try:
        _refuse_facet_axis(axis, classification)
        plan = _plan_groupby_sum_fold(
            tuple(table.column_names),
            axis=axis,
            classification=classification,
            summable_axes=summable_axes,
            hierarchy=None,
            missing_declared_as_zero=missing_declared_as_zero,
            value_column=_VALUE_COLUMN,
            expected_population=expected_population,
        )
    except (MaterialisationError, ValueError):
        plan = None
    grouped = (
        _grouped_sums(table, plan, sequential=False)
        if plan is not None
        and table.num_rows
        and _codes(table, [plan.columns[i] for i in plan.full_grain_positions]).count
        == table.num_rows
        else None
    )
    if grouped is not None:
        return grouped
    return groupby_sum(
        rows_table(table),
        axis=axis,
        classification=classification,
        summable_axes=summable_axes,
        missing_declared_as_zero=missing_declared_as_zero,
        expected_population=expected_population,
    )


def rows_table(table: GroupedTable) -> Table:
    """`table` as a `reshape.Table`, its cells as `Frame.rows` gives them
    (`formats.canonical_frame_rows`)."""
    if isinstance(table, Table):
        return table
    return Table(columns=tuple(table.column_names), rows=tuple(canonical_frame_rows(table)))


def _refuse_facet_axis(axis: str, classification: ColumnClassification) -> None:
    if axis in classification.facet_axis_columns:
        raise RollupNotSummableError(
            f"groupby axis {axis!r} is a facet axis and cannot be collapsed by summing",
            axis=axis,
            precondition="facet_axis",
        )


def _grouped_sums(
    table: pa.Table, plan: _GroupbySumFoldPlan, *, sequential: bool
) -> pa.Table | None:
    """One row per group of `plan`'s grouping columns: the group's first row with its value the
    group total, rows sorted stably by `reshape._cell_sort_key` of every cell, groups first in
    order of first appearance. `None` when `_sums_plainly` fails or a total is not finite.

    With `sequential`, a total is the left-to-right sum of the group's values from `0.0`.
    Without it, a total is Python's `sum` of them, which compensates rounding."""
    groups = _codes(table, [plan.columns[i] for i in plan.grouping_positions])
    if not _sums_plainly(table, plan, groups):
        return None
    values = table.column(plan.value_column)
    if sequential:
        totals = (
            pa.table({"group": groups.codes, "value": values})
            .group_by("group", use_threads=False)
            .aggregate((("value", "sum"),))
            .sort_by("group")
            .column("value_sum")
            .combine_chunks()
        )
    else:
        # A stable sort keeps each group's values in row order.
        ordered = cast(
            "list[float]", values.take(array_kernel("sort_indices", groups.codes)).to_pylist()
        )
        counts = (
            pa.table({"group": groups.codes})
            .group_by("group", use_threads=False)
            .aggregate((("group", "count"),))
            .sort_by("group")
            .column("group_count")
            .to_pylist()
        )
        sums: list[float] = []
        start = 0
        for count in cast("list[int]", counts):
            sums.append(sum(ordered[start : start + count]))
            start += count
        totals = pa.array(sums, pa.float64())
    if not scalar_kernel("all", array_kernel("is_finite", totals)).as_py():
        return None
    output = (
        table.select(list(plan.output_columns))
        .take(groups.first_rows)
        .set_column(plan.value_pos_in_remaining, pa.field(plan.value_column, pa.float64()), totals)
    )
    sort_keys = [
        array
        for column in plan.output_columns
        for array in _sort_key_arrays(output.column(column), unit=column == UNIT_COLUMN)
    ]
    order = array_kernel(
        "sort_indices",
        pa.table({str(i): array for i, array in enumerate(sort_keys)}),
        sort_keys=[(str(i), "ascending") for i in range(len(sort_keys))],
    )
    return output.take(order)


def _full_grain_keys(table: pa.Table, plan: _GroupbySumFoldPlan) -> list[pa.Array]:
    """The rank and text arrays of `reshape._cell_sort_key` over every column but the value."""
    return [
        array
        for position in plan.full_grain_positions
        for array in _sort_key_arrays(
            table.column(plan.columns[position]), unit=plan.columns[position] == UNIT_COLUMN
        )
    ]


def _codes(table: pa.Table, names: Sequence[str]) -> UnitCodes:
    """Dense codes over the rows' values in `names`, in order of first appearance. When `names`
    is empty, every row has the one code."""
    if not names:
        zeros = pa.array([0] * table.num_rows, pa.int64())
        return UnitCodes(codes=zeros, first_rows=pa.array([0][: table.num_rows], pa.int64()))
    columns = [table.column(name).combine_chunks() for name in names]
    return unit_codes(pa.StructArray.from_arrays(columns, names=list(names)))


def _sort_key_arrays(values: pa.ChunkedArray, *, unit: bool) -> tuple[pa.Array, pa.Array]:
    """The `reshape._cell_sort_key` of each cell as a rank array and a text array, computed once
    per distinct cell. Arrow orders text by UTF-8 bytes, the code-point order Python gives the
    key's string. A text cell is its own key string."""
    if pa.types.is_string(values.type):
        text = values.combine_chunks()
        ranks = array_kernel("cast", array_kernel("is_valid", text), pa.int8())
        return ranks, array_kernel("fill_null", text, "")
    codes = unit_codes(values)
    if unit:
        cells: list[object] = list(unit_cells(values, codes))
    else:
        cells = values.take(codes.first_rows).to_pylist()
    keys = [_cell_sort_key(cell) for cell in cells]
    ranks = pa.array([rank for rank, _text in keys], pa.int8()).take(codes.codes)
    texts = pa.array([text for _rank, text in keys], pa.string()).take(codes.codes)
    return ranks, texts


def _sums_plainly(table: pa.Table, plan: _GroupbySumFoldPlan, groups: UnitCodes) -> bool:
    """Whether the row engines would sum `table`, its full grains unique, with no refusal and no
    estimated total, short of the finite-total check: no null value, one value per group in
    each facet and status column, every group key described by the expected population, and
    no null or absent member unless the source reads missing members as zero."""
    columns = plan.columns
    if table.column(plan.value_column).null_count:
        return False
    for position in plan.passthrough_positions:
        cell_codes = unit_codes(table.column(columns[position])).codes
        pairs = unit_codes(
            pa.StructArray.from_arrays([groups.codes, cell_codes], names=["group", "cell"])
        )
        if pairs.count > groups.count:
            return False
    axis = table.column(plan.axis)
    if not plan.missing_declared_as_zero and axis.null_count:
        return False
    if not plan.has_expected_population:
        return True
    population_names = list(plan.other_population_columns)
    population_keys = _codes(table, population_names)
    distinct_keys = (
        [
            tuple(row[name] for name in population_names)
            for row in table.select(population_names).take(population_keys.first_rows).to_pylist()
        ]
        if population_names
        else [()]
    )
    if population_names and any(key not in plan.population_index for key in distinct_keys):
        return False
    if plan.missing_declared_as_zero:
        return True
    return _no_member_absent(table, plan, groups, population_keys, distinct_keys)


def _no_member_absent(
    table: pa.Table,
    plan: _GroupbySumFoldPlan,
    groups: UnitCodes,
    population_keys: UnitCodes,
    distinct_keys: list[tuple[object, ...]],
) -> bool:
    """Whether every group observes every axis member the expected population names for its
    key, which holds when each group's distinct expected members among its rows number as many
    as expected. `distinct_keys` holds each population key code's values."""
    expected = {key: frozenset(members) for key, members in plan.population_index.items()}
    names = [*plan.other_population_columns, plan.axis]
    members = _codes(table, names)
    is_expected = [
        row[plan.axis] in expected.get(tuple(row[name] for name in names[:-1]), frozenset())
        for row in table.select(names).take(members.first_rows).to_pylist()
    ]
    expected_member = array_kernel(
        "if_else",
        pa.array(is_expected, pa.bool_()).take(members.codes),
        members.codes,
        pa.scalar(None, pa.int64()),
    )
    observed = (
        pa.table({"group": groups.codes, "member": expected_member})
        .group_by("group", use_threads=False)
        .aggregate((("member", "count_distinct"),))
        .sort_by("group")
        .column("member_count_distinct")
        .combine_chunks()
    )
    expected_counts = pa.array(
        [len(expected.get(key, frozenset())) for key in distinct_keys], pa.int64()
    ).take(population_keys.codes.take(groups.first_rows))
    return bool(scalar_kernel("all", array_kernel("equal", observed, expected_counts)).as_py())
