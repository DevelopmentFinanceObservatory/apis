"""Builds the frame's structural view: dimensions, period grain, coverage, span, and series count.

Dimension names come from frame columns. Distinct values are capped, with
full counts retained for truncated listings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    parse_time_period,
)

from assembler.errors import BindingInvalid
from assembler.limits import PER_DIMENSION_LISTING_CAP
from assembler.ports import DimensionView, StructuralView

__all__ = [
    "compute_structural_view",
    "dimension_columns",
    "facet_dimension_names",
    "hashable_cell",
]

_NON_DIMENSION_COLUMNS = frozenset({TIME_PERIOD_COLUMN, VALUE_COLUMN, STATUS_COLUMN})


def dimension_columns(*, columns: Sequence[str]) -> tuple[str, ...]:
    """Dimension columns in frame order, including `unit`, `measurementMethod`, and `provenance`."""
    return tuple(column for column in columns if column not in _NON_DIMENSION_COLUMNS)


def facet_dimension_names() -> frozenset[str]:
    """Facet dimensions that require binding when they split a series: `unit` and
    frame-grammar identity columns."""
    return frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})


def compute_structural_view(*, frame: Mapping[str, object]) -> StructuralView:
    columns = tuple(cast("Sequence[str]", frame["columns"]))
    rows = tuple(cast("Sequence[Sequence[object]]", frame["rows"]))
    index_by_column = {column: position for position, column in enumerate(columns)}

    dimensions = tuple(
        _dimension_view(name=name, index=index_by_column[name], rows=rows)
        for name in dimension_columns(columns=columns)
    )

    period_index = index_by_column.get(TIME_PERIOD_COLUMN)
    periods = (
        sorted(cast("set[str]", {row[period_index] for row in rows}))
        if period_index is not None and rows
        else []
    )
    coverage = (periods[0], periods[-1]) if periods else None
    # Empty frames default to year. Window resolution refuses zero observations.
    grain, span = _period_grain_and_span(periods=periods) if periods else ("year", None)

    coordinate_columns = dimension_columns(columns=columns)
    coordinate_indices = tuple(index_by_column[column] for column in coordinate_columns)
    series_count = len({tuple(hashable_cell(row[i]) for i in coordinate_indices) for row in rows})

    return StructuralView(
        dimensions=dimensions,
        period_grain=grain,
        coverage=coverage,
        span=span,
        series_count=series_count,
    )


def _dimension_view(*, name: str, index: int, rows: Sequence[Sequence[object]]) -> DimensionView:
    """`values`, `total_count` and `truncated` cover the bindable, non-null values.
    `plural` also counts a null, so `{USDollar, None}` is plural."""
    distinct = sorted({_cell_text(row[index]) for row in rows if row[index] is not None})
    has_null = any(row[index] is None for row in rows)
    total_count = len(distinct)
    capped = tuple(distinct[:PER_DIMENSION_LISTING_CAP])
    return DimensionView(
        name=name,
        values=capped,
        total_count=total_count,
        truncated=total_count > len(capped),
        plural=total_count + has_null > 1,
    )


def _cell_text(value: object) -> str:
    """Returns dimension display text, rendering structured units through `hashable_cell`.

    Dimension value listings skip `None`. All values are strings before sorting."""
    return cast("str", hashable_cell(value))


def hashable_cell(value: object) -> object:
    """Makes frame cells hashable and gives structured units their display form.

    `None` and strings pass through. Unit objects render as
    `currency/prices[/base_year]`, keeping converted units distinct.
    This form is for grouping and display only. Binding preserves the raw
    unit object in the resolved payload."""
    if isinstance(value, Mapping):
        ref = cast("Mapping[str, object]", value)
        text = f"{ref['currency']}/{ref['prices']}"
        base_year = ref.get("base_year")
        return f"{text}/{base_year}" if base_year is not None else text
    return value


def _period_grain_and_span(*, periods: Sequence[str]) -> tuple[str, str | None]:
    """Returns the earliest period's grain and the frame's span, refusing a span mixed with
    any other period.

    A frame whose rows share one span is a single-period frame at its ends' grain. A span
    beside a point or a second span has no one period to bind, so it refuses
    `binding_invalid`. Points of different grains keep the earliest period's grain."""
    parsed = [parse_time_period(period) for period in periods]
    spans = [period for period, one in zip(periods, parsed, strict=True) if one.start != one.end]
    if spans and len(periods) > 1:
        others = [period for period in periods if period != spans[0]]
        raise BindingInvalid(
            pointer="/frame_ref",
            dimension="",
            legal_values=(),
            truncated=False,
            detail=(
                f"this frame mixes the span {spans[0]!r} with {len(others)} other "
                f"period(s), first {others[0]!r}; a frame binds one span or points only"
            ),
        )
    return parsed[0].grain, spans[0] if spans else None
