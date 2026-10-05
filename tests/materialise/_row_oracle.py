"""Independent row oracles for latest-period reduction and facet collapse.

This module is a permanent copy of the row-path logic that the table functions replaced. It
imports only `dc_sdmx` types and the standard library, so it keeps running after the
production row functions are gone. A deliberate change to either rule deletes the differential
test that compares against it; this module stays as it is.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import NamedTuple

from dc_sdmx import SdmxObservation

_STAT_VAR_REF_PREFIX = "dcid:"


def _canonicalize_stat_var_ref(value: str) -> str:
    """Add the `dcid:` prefix when `value` is bare."""
    if value.startswith(_STAT_VAR_REF_PREFIX):
        return value
    return f"{_STAT_VAR_REF_PREFIX}{value}"


def _leading_year(time_period: str | None) -> int | None:
    """Parse the first four characters as a year; return `None` on failure."""
    if time_period is None:
        return None
    try:
        return int(time_period[:4])
    except ValueError:
        return None


def _latest_rows(rows: Sequence[SdmxObservation]) -> tuple[SdmxObservation, ...]:
    """Keep rows at the greatest parseable year and period, in input order."""
    parseable: list[tuple[SdmxObservation, int, str]] = []
    for row in rows:
        year = _leading_year(row.time_period)
        if year is None or row.time_period is None:
            continue
        parseable.append((row, year, row.time_period))
    if not parseable:
        return ()
    latest_year = max(year for _, year, _ in parseable)
    same_year = [(row, period) for row, year, period in parseable if year == latest_year]
    latest_period = max(period for _, period in same_year)
    return tuple(row for row, period in same_year if period == latest_period)


class _CollapseCell(NamedTuple):
    """Fields used to build the grain key and rank a facet."""

    variable: str
    role_values: tuple[str | None, ...]
    time_period: str | None
    unit: str | None
    measurement_method: str | None
    provenance: str | None
    observation_period: str | None
    facet_id: str | None


def _cells_from_observations(
    rows: Sequence[SdmxObservation],
    sorted_declared_roles: Sequence[str],
) -> tuple[_CollapseCell, ...]:
    """Build one `_CollapseCell` per observation in input order."""
    return tuple(
        _CollapseCell(
            variable=row.variable,
            role_values=tuple(row.role_values.get(role) for role in sorted_declared_roles),
            time_period=row.time_period,
            unit=row.unit,
            measurement_method=row.measurement_method,
            provenance=row.provenance,
            observation_period=row.observation_period,
            facet_id=row.facet_id,
        )
        for row in rows
    )


def _nfc(value: str | None) -> str | None:
    """Return NFC-normalised text; pass through `None` and ASCII values."""
    if value is None or value.isascii():
        return value
    return unicodedata.normalize("NFC", value)


def _grain_key(cell: _CollapseCell) -> tuple[object, ...]:
    """Build the Frame grain key with every string component NFC-normalised."""
    return (
        _nfc(_canonicalize_stat_var_ref(cell.variable)),
        tuple(_nfc(value) for value in cell.role_values),
        _nfc(cell.time_period),
        _nfc(cell.unit),
        _nfc(cell.measurement_method),
        _nfc(cell.provenance),
    )


def _facet_id_sort_key(facet_id: str | None) -> tuple[int, int, str] | tuple[int, str]:
    """Sort decimal ids numerically up to 32 digits, then other ids lexically."""
    if facet_id is not None and facet_id.isdecimal() and len(facet_id) <= 32:
        return (0, int(facet_id), facet_id)
    return (1, facet_id or "")


def _facet_selection_rank(
    facet_cells: Sequence[_CollapseCell],
) -> tuple[int, int, int, int, tuple[int, int, str] | tuple[int, str]]:
    """Return the facet rank; the smallest key wins."""
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


def _duplicate_facet_keep_mask(cells: Sequence[_CollapseCell]) -> tuple[bool, ...] | None:
    """Choose a facet per contested grain key and return a parallel keep mask.

    Return `None` when at most one facet id is present.
    """
    facet_ids_in_order: list[str | None] = []
    cells_by_facet_id: dict[str | None, list[_CollapseCell]] = {}
    for cell in cells:
        group = cells_by_facet_id.get(cell.facet_id)
        if group is None:
            facet_ids_in_order.append(cell.facet_id)
            group = cells_by_facet_id[cell.facet_id] = []
        group.append(cell)

    if len(facet_ids_in_order) <= 1:
        return None

    rank_by_facet_id = {
        facet_id: _facet_selection_rank(facet_cells)
        for facet_id, facet_cells in cells_by_facet_id.items()
    }

    grain_key_by_cell: list[tuple[object, ...]] = []
    facet_ids_by_grain_key: dict[tuple[object, ...], set[str | None]] = {}
    for cell in cells:
        key = _grain_key(cell)
        grain_key_by_cell.append(key)
        facet_ids_by_grain_key.setdefault(key, set()).add(cell.facet_id)

    winner_by_contested_key = {
        key: min(facet_ids, key=lambda fid: rank_by_facet_id[fid])
        for key, facet_ids in facet_ids_by_grain_key.items()
        if len(facet_ids) > 1
    }

    return tuple(
        key not in winner_by_contested_key or cell.facet_id == winner_by_contested_key[key]
        for cell, key in zip(cells, grain_key_by_cell, strict=True)
    )


def collapse_duplicate_identity_facets(
    rows: tuple[SdmxObservation, ...],
    *,
    declared_roles: Sequence[str],
) -> tuple[SdmxObservation, ...]:
    """Resolve duplicate facets per Frame grain key using each facet's full row set."""
    sorted_declared_roles = sorted(declared_roles)
    cells = _cells_from_observations(rows, sorted_declared_roles)
    keep = _duplicate_facet_keep_mask(cells)
    if keep is None:
        return rows
    return tuple(row for row, survives in zip(rows, keep, strict=True) if survives)
