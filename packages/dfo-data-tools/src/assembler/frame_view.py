"""Builds the frame's structural view: dimensions, period grain, coverage, span, and series count.

Dimension names come from frame columns. Distinct values are capped, with
full counts retained for truncated listings. A row's series is its `series_keys` key, which
files the estimates continuing a series under that series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
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
from assembler.limits import LABEL_MAX, PER_DIMENSION_LISTING_CAP
from assembler.ports import Continuation, DimensionView, StructuralView
from assembler.series_id import Coordinates
from materialise.bundle_columns import MEASUREMENT_METHOD_COLUMN, PROVENANCE_COLUMN
from materialise.continued_series import (
    continuation_leads,
    continued_series_of_entries,
    continued_series_of_trace,
)
from materialise.unit_cells import shared_unit_key, unit_key_text

__all__ = [
    "compute_structural_view",
    "continuation_known",
    "dimension_columns",
    "facet_dimension_names",
    "frame_continuation",
    "hashable_cell",
    "series_keys",
]

_NON_DIMENSION_COLUMNS = frozenset({TIME_PERIOD_COLUMN, VALUE_COLUMN, STATUS_COLUMN})


def dimension_columns(*, columns: Sequence[str]) -> tuple[str, ...]:
    """Dimension columns in frame order, including `unit`, `measurementMethod`, and `provenance`."""
    return tuple(column for column in columns if column not in _NON_DIMENSION_COLUMNS)


def facet_dimension_names() -> frozenset[str]:
    """Facet dimensions that require binding when they split a series: `unit` and
    frame-grammar identity columns."""
    return frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})


def frame_continuation(*, manifest: Mapping[str, object]) -> Continuation | None:
    """The continuation of a frame's series, or `None` when nothing continues them.

    A dataset version's manifest records its own (`trace.continuations`, written by
    `materialise.finish`), scoped to the selections its `trace.facets` name
    (`materialise.continued_series.continued_series_of_trace`). A computed version's manifest
    holds the continuations it inherited from the versions it was computed from, stored when
    it was published (`continuations`, `datasets_api.lineage_continuations`). Nothing else is
    read, so a frame's series stay as they first loaded whatever description records arrive."""
    trace = manifest.get("trace")
    series = (
        continued_series_of_trace(cast("Mapping[str, object]", trace))
        if isinstance(trace, Mapping)
        else continued_series_of_entries(manifest.get("continuations"))
    )
    return Continuation(series=series) if series else None


def continuation_known(*, manifest: Mapping[str, object]) -> bool:
    """Whether `manifest` states its frame's continuation. A computed version recorded before
    versions stored continuations holds `continuations: null` until the backfill fills it;
    its `frame_continuation` reads as none meanwhile, and a reader that caches the frame reads
    the version again until it is known."""
    return "trace" in manifest or manifest.get("continuations", ()) is not None


def series_keys(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    continuation: Continuation | None,
) -> tuple[Coordinates, ...]:
    """Each row's series key: its dimension cells by column name, sorted, rendered with
    `hashable_cell`.

    A row that `continuation` admits takes the key of the row whose series it continues
    (`materialise.continued_series.continuation_leads`): the first row, in frame order, holding
    the same cells on every other dimension under the method it continues. So a series and the
    estimates continuing it are one series, as the manifest and a description count them. A
    continuing row with no such row keys a series of its own. With no continuation, or a frame
    without a `measurementMethod` column, the key is the row's own cells."""
    index = {column: position for position, column in enumerate(columns)}
    names = dimension_columns(columns=columns)
    keys = [
        cast(
            "Coordinates",
            tuple(sorted((name, hashable_cell(row[index[name]])) for name in names)),
        )
        for row in rows
    ]
    if continuation is None or MEASUREMENT_METHOD_COLUMN not in index:
        return tuple(keys)
    axis = index[MEASUREMENT_METHOD_COLUMN]
    source = index.get(PROVENANCE_COLUMN)
    leads = continuation_leads(
        continuation.series,
        # Each key without its method cell, the series identity a continuing row shares.
        identities=[
            tuple(item for item in key if item[0] != MEASUREMENT_METHOD_COLUMN) for key in keys
        ],
        methods=[row[axis] for row in rows],
        provenances=None if source is None else [row[source] for row in rows],
    )
    return tuple(key if lead is None else keys[lead] for key, lead in zip(keys, leads, strict=True))


def compute_structural_view(
    *, frame: Mapping[str, object], continuation: Continuation | None
) -> StructuralView:
    """`series_count` counts `series_keys` under `continuation`, the frame's
    `frame_continuation`, and a dimension is `plural` when those series hold more than one
    of its values. A continued series holds the method of the rows it continues, so its
    continuing method neither counts a series nor makes the method axis plural, though the
    method stays listed as a value a select may bind.

    Raises:
        FrameError: A `time_period` cell `parse_time_period` cannot read. The other cells are
            trusted as the grammar's writer wrote them.
        BindingInvalid: The frame's units do not read apart, or its periods bind no one grain.
    """
    columns = tuple(cast("Sequence[str]", frame["columns"]))
    rows = tuple(cast("Sequence[Sequence[object]]", frame["rows"]))
    index_by_column = {column: position for position, column in enumerate(columns)}
    if UNIT_COLUMN in index_by_column:
        unit_index = index_by_column[UNIT_COLUMN]
        shared = shared_unit_key(cells=tuple(row[unit_index] for row in rows))
        if shared is not None:
            # Grouping would put one cell's rows under the other's unit. Every reader takes its
            # frame with this view, so the refusal covers each of them.
            raise BindingInvalid(
                pointer="/frame_ref",
                dimension="",
                legal_values=(),
                truncated=False,
                detail=(
                    f"this frame holds the unit ref {shared[:LABEL_MAX]!r} beside a structured "
                    "unit that reads as the same text; a frame's units must read apart"
                ),
            )

    series = set(series_keys(columns=columns, rows=rows, continuation=continuation))
    dimensions = tuple(
        _dimension_view(name=name, index=index_by_column[name], rows=rows, series=series)
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

    series_count = len(series)

    return StructuralView(
        dimensions=dimensions,
        period_grain=grain,
        coverage=coverage,
        span=span,
        series_count=series_count,
    )


def _dimension_view(
    *, name: str, index: int, rows: Sequence[Sequence[object]], series: Set[Coordinates]
) -> DimensionView:
    """`values`, `total_count` and `truncated` cover the bindable, non-null values of every
    row. `plural` counts the values `series` hold, a null included, so `{USDollar, None}` is
    plural."""
    distinct = sorted({_cell_text(row[index]) for row in rows if row[index] is not None})
    total_count = len(distinct)
    capped = tuple(distinct[:PER_DIMENSION_LISTING_CAP])
    return DimensionView(
        name=name,
        values=capped,
        total_count=total_count,
        truncated=total_count > len(capped),
        plural=len({dict(key)[name] for key in series}) > 1,
    )


def _cell_text(value: object) -> str:
    """Returns dimension display text, rendering structured units through `hashable_cell`.

    Dimension value listings skip `None`. All values are strings before sorting."""
    return cast("str", hashable_cell(value))


def hashable_cell(value: object) -> object:
    """Makes frame cells hashable and gives structured units their display form.

    `None` and strings pass through. A converted or composed unit renders as
    `materialise.unit_cells.unit_key_text` gives it. `compute_structural_view`
    refuses a frame in which a string cell spells a structured cell's key.
    This form is for grouping and display only. Binding preserves the raw
    unit object in the resolved payload."""
    if isinstance(value, Mapping):
        return unit_key_text(cast("Mapping[str, object]", value))
    return value


def _period_grain_and_span(*, periods: Sequence[str]) -> tuple[str, str | None]:
    """Returns the frame's one grain and its span, refusing a span mixed with any other
    period and points of different grains.

    A frame whose rows share one span is a single-period frame at its ends' grain. A span
    beside a point or a second span has no one period to bind, so it refuses
    `binding_invalid`. So do points of different grains: a window binds periods at one grain,
    and a frame holding `2019` beside `2019-01` (an annual and a quarterly facet of one
    series) would bind its other grain's points to nothing."""
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
    grains: dict[str, str] = {}
    for period, one in zip(periods, parsed, strict=True):
        grains.setdefault(one.grain, period)
    if len(grains) > 1:
        raise BindingInvalid(
            pointer="/frame_ref",
            dimension="",
            legal_values=(),
            truncated=False,
            detail=(
                f"this frame mixes {' and '.join(grains)} periods, first "
                f"{', '.join(repr(period) for period in grains.values())}; a frame binds "
                "periods at one grain"
            ),
        )
    return parsed[0].grain, spans[0] if spans else None
