"""Resolves duplicate `Frame.build` grain keys across facets before coverage and frame
construction. Contested keys keep the rows from the highest-ranked facet."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc

from materialise.frame_build import (
    # Reuse `frame_build`'s canonicaliser so grain keys match `Frame.build`.
    _canonicalize_stat_var_ref,
)
from materialise.window import (
    # Reuse `window`'s year parser so facet ranking matches the window rules.
    _leading_year,
)

_FACET_ID_COLUMN = "facetId"


class CollapseCell(NamedTuple):
    """Fields used to key and rank one row. `role_values` follows the caller's sorted role
    order."""

    variable: str
    role_values: tuple[str | None, ...]
    time_period: str | None
    unit: str | None
    measurement_method: str | None
    provenance: str | None
    observation_period: str | None
    facet_id: str | None


def collapse_duplicate_identity_facets_table(
    table: pa.Table,
    *,
    declared_roles: Sequence[str],
    rank_within_series: bool = False,
) -> pa.Table:
    """Ranks each facet over its full row set and keeps the best-ranked facet for every
    contested normalised `Frame.build` grain key. At most one distinct `facetId`, including
    null, returns the original table without conversion. Otherwise it converts only key/rank columns
    to Python, then filters in input order. Arrow joins are unsuitable because SQL nulls do
    not match, while grain-key fields may be null.

    `rank_within_series` ranks each facet over its rows in each series, one dcid on every
    declared role, and contests keys only within a series: the result of collapsing each
    series on its own."""
    facet_id_column = table[_FACET_ID_COLUMN]
    distinct = pc.count_distinct(facet_id_column, mode="all")
    distinct_facets = distinct.as_py()
    if distinct_facets <= 1:
        return table

    sorted_declared_roles = sorted(declared_roles)
    cells = _cells_from_table(table, sorted_declared_roles)
    keep = _duplicate_facet_keep_mask(cells, rank_within_series=rank_within_series)
    if keep is None:
        return table
    return table.filter(pa.array(keep, type=pa.bool_()))


def keep_mask(cells: Sequence[CollapseCell], *, rank_within_series: bool) -> tuple[bool, ...]:
    """The cells of `cells` that `collapse_duplicate_identity_facets_table` keeps of a table
    holding them, in input order, under the same `rank_within_series`: on a grain key two
    facets contest (one date at one set of role values under the same unit, method and
    provenance), those of the facet ranked best (`_facet_selection_rank`) over all its
    cells, or over its cells in each series under `rank_within_series`; every other cell.
    Cells under different methods hold different grain keys, so both stay. A caller holding
    a member's rows outside an Arrow table, such as Search's bulk role read, keeps by it."""
    keep = _duplicate_facet_keep_mask(cells, rank_within_series=rank_within_series)
    return keep if keep is not None else (True,) * len(cells)


def _non_null_variable(value: str | None) -> str:
    """Asserts `variableMeasured` is non-null, as guaranteed by `observation_schema`."""
    assert value is not None
    return value


def _cells_from_table(
    table: pa.Table,
    sorted_declared_roles: Sequence[str],
) -> tuple[CollapseCell, ...]:
    """Builds one cell per row, pulling only grain-key and ranking columns into Python lists."""
    variable_column = table["variableMeasured"].to_pylist()
    role_columns = [table[role].to_pylist() for role in sorted_declared_roles]
    time_period_column = table["TIME_PERIOD"].to_pylist()
    unit_column = table["unit"].to_pylist()
    measurement_method_column = table["measurementMethod"].to_pylist()
    provenance_column = table["provenance"].to_pylist()
    observation_period_column = table["observationPeriod"].to_pylist()
    facet_id_column = table[_FACET_ID_COLUMN].to_pylist()

    return tuple(
        CollapseCell(
            variable=_non_null_variable(variable_column[index]),
            role_values=tuple(column[index] for column in role_columns),
            time_period=time_period_column[index],
            unit=unit_column[index],
            measurement_method=measurement_method_column[index],
            provenance=provenance_column[index],
            observation_period=observation_period_column[index],
            facet_id=facet_id_column[index],
        )
        for index in range(table.num_rows)
    )


def _duplicate_facet_keep_mask(
    cells: Sequence[CollapseCell], *, rank_within_series: bool
) -> tuple[bool, ...] | None:
    """Ranks each facet over all its cells, or over its cells in each series under
    `rank_within_series`, then marks rows from the best facet for every contested grain key.
    Returns `None` for at most one facet. Otherwise returns a parallel boolean keep-mask."""

    def series(cell: CollapseCell) -> tuple[str | None, ...]:
        return cell.role_values if rank_within_series else ()

    cells_by_group: dict[tuple[tuple[str | None, ...], str | None], list[CollapseCell]] = {}
    for cell in cells:
        cells_by_group.setdefault((series(cell), cell.facet_id), []).append(cell)

    if len({facet_id for _series, facet_id in cells_by_group}) <= 1:
        return None

    rank_by_group = {
        group: _facet_selection_rank(group_cells) for group, group_cells in cells_by_group.items()
    }

    type SeriesGrainKey = tuple[tuple[str | None, ...], tuple[object, ...]]
    grain_key_by_cell: list[SeriesGrainKey] = []
    facet_ids_by_grain_key: dict[SeriesGrainKey, set[str | None]] = {}
    for cell in cells:
        key = (series(cell), _grain_key(cell))
        grain_key_by_cell.append(key)
        facet_ids_by_grain_key.setdefault(key, set()).add(cell.facet_id)

    winner_by_contested_key = {
        key: min(((key[0], fid) for fid in facet_ids), key=rank_by_group.__getitem__)[1]
        for key, facet_ids in facet_ids_by_grain_key.items()
        if len(facet_ids) > 1
    }

    return tuple(
        key not in winner_by_contested_key or cell.facet_id == winner_by_contested_key[key]
        for cell, key in zip(cells, grain_key_by_cell, strict=True)
    )


def _grain_key(cell: CollapseCell) -> tuple[object, ...]:
    """`Frame.build` grain key uses canonical variable, role values, period, unit, method, and
    provenance. Strings are NFC-normalised."""
    return (
        _nfc(_canonicalize_stat_var_ref(cell.variable)),
        tuple(_nfc(value) for value in cell.role_values),
        _nfc(cell.time_period),
        _nfc(cell.unit),
        _nfc(cell.measurement_method),
        _nfc(cell.provenance),
    )


def _nfc(value: str | None) -> str | None:
    """NFC-normalises non-ASCII strings. `None` and ASCII values pass through unchanged."""
    if value is None or value.isascii():
        return value
    return unicodedata.normalize("NFC", value)


def _facet_selection_rank(
    facet_cells: Sequence[CollapseCell],
) -> tuple[int, int, int, int, tuple[int, int, str] | tuple[int, str]]:
    """Sort key for competing facets. The smallest wins. Ranks each facet's full cell set."""
    facet_id = facet_cells[0].facet_id
    has_period = any(cell.observation_period is not None for cell in facet_cells)
    served_years = [
        year for cell in facet_cells if (year := _leading_year(cell.time_period)) is not None
    ]
    span_years = (max(served_years) - min(served_years)) if served_years else -1
    return (
        0 if facet_id is not None else 1,
        0 if has_period else 1,
        -len(facet_cells),
        -span_years,
        _facet_id_sort_key(facet_id),
    )


def _facet_id_sort_key(facet_id: str | None) -> tuple[int, int, str] | tuple[int, str]:
    """Sorts decimal ids numerically up to 32 digits (Data Commons ids are about 19 digits),
    then by raw string to distinguish `"1"` and `"01"`. Longer, non-decimal, and missing ids
    sort after decimals by text. The cap bounds `int()` conversion below Python's 4,300-digit
    limit."""
    if facet_id is not None and facet_id.isdecimal() and len(facet_id) <= 32:
        return (0, int(facet_id), facet_id)
    return (1, facet_id or "")
