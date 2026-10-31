"""Binds each resolved series over the intent window at the frame's grain.

A frame whose rows share one span binds that span as its only period.
Missing rows become `status: "missing"`. Non-null statuses pass through,
and other rows contribute their observed value. `source_ref` comes from the
series coordinates. `unit` comes from the rows: a unit dcid becomes its label,
a converted unit passes through, and a composed unit binds side by side, a
StatVar side becoming `{stat_var, label}`. Labels use `ResolvedFrame.labels`
with the raw ref as fallback, and a unit label is cut to 200 characters.

A table over a frame that keeps a selection in each period lists only the points that
selection kept (`without_unselected_points`).
"""

from __future__ import annotations

import calendar
import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, cast

from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    parse_time_period,
)
from dfo_frame_grammar.grammar import STAT_VAR_REF_PREFIX
from pydantic.experimental.missing_sentinel import MISSING

from assembler.limits import LABEL_MAX, joined_label
from assembler.ports import ResolvedFrame
from assembler.resolve import ResolvedSeries
from description.steps import Step
from materialise.unit_cells import composed_unit_sides

__all__ = [
    "BoundSeries",
    "bind_all",
    "bind_series",
    "count_periods",
    "count_selected_points",
    "frame_periods",
    "role_columns",
    "selects_per_period",
    "without_unselected_points",
]

_PROVENANCE_COLUMN = "provenance"  # frame-grammar's FACET_IDENTITY_COLUMNS names this axis
_RESERVED_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN}
)
# The suffixes of the columns that describe the place or party another column names: its type,
# and a party's kind, class and whole cell (`dfo_analysis_engine.place_type.enrichment_role`).
# A derived frame keeps them beside the role they describe.
_ENRICHMENT_SUFFIXES = ("_type", "_party_kind", "_party_class", "_party_whole")


@dataclass(frozen=True)
class BoundSeries:
    """Bound series before catalogue-specific minting.

    `source_ref` is the raw graph ref or `None`. Minting resolves it to an
    envelope source id. `unit` is `None`, a unit dcid's label (the dcid when
    unlabelled) cut to 200 characters, the resolved `{currency, prices,
    base_year}` object, or the catalogue's composed unit `{numerator,
    denominator}`. No unit conversion occurs here.
    """

    id: str
    label: str
    unit: object | None
    source_ref: str | None
    points: tuple[Mapping[str, object], ...]


def bind_series(
    *, resolved_frame: ResolvedFrame, series: ResolvedSeries, window: Mapping[str, object]
) -> BoundSeries:
    coordinates = dict(series.coordinates)
    rows_by_period = {row[TIME_PERIOD_COLUMN]: row for row in series.rows}
    periods = _window_periods(resolved_frame=resolved_frame, window=window)

    points: list[Mapping[str, object]] = []
    for period in periods:
        row = rows_by_period.get(period)
        if row is None:
            points.append({"period": period, "status": "missing"})
            continue
        status = row.get(STATUS_COLUMN)
        if status is not None:
            points.append({"period": period, "status": status})
        else:
            points.append({"period": period, "value": row[VALUE_COLUMN]})

    return BoundSeries(
        id=series.id,
        label=_label(resolved_frame=resolved_frame, coordinates=coordinates),
        unit=_unit(resolved_frame=resolved_frame, rows=series.rows),
        source_ref=coordinates.get(_PROVENANCE_COLUMN),
        points=tuple(points),
    )


# The input roles a step reading several inputs takes its output's rows from: a share or
# quotient writes one row per numerator key, over the periods its denominator shares, and an
# inner join one row per key both sides hold. A sum or comparison writes series of its own,
# so it has none. A share or quotient whose `denominator_by` pairs each numerator key with a
# denominator group also keeps a key only in its group's periods (`_keyed_denominator`).
_ROW_SOURCE_ROLES: Final[Mapping[str, tuple[str, ...]]] = {
    "share": ("numerator",),
    "quotient": ("numerator",),
    "join": ("left", "right"),
}
_KEYED_DENOMINATOR_OPS: Final = frozenset({"share", "quotient"})


def selects_per_period(*, resolved_frame: ResolvedFrame) -> bool:
    """Whether the frame keeps a selection of entities in each period: walking back from its
    record's last step through the inputs its rows come from, a `rank` over each period on its
    own (`per_period`) with a `limit` is reached, which keeps each period's top or bottom
    entities. A step's one `input` supplies its rows, and of several `inputs` the roles
    `_ROW_SOURCE_ROLES` names, with a keyed denominator (`_keyed_denominator`). A `rank`
    without a `limit` keeps every entity, and one over the whole window writes a single
    period. A frame without a description record, or whose
    record holds no steps, keeps none."""
    record = resolved_frame.description_record
    known = record.known_steps if record is not None else []
    if not known:
        return False
    by_id = {step.id: step for step in known}
    pending = [known[-1].id]
    seen: set[str] = set()
    while pending:
        step = by_id.get(pending.pop())
        if step is None or step.id in seen:
            continue
        seen.add(step.id)
        if (
            step.op == "rank"
            and step.params is not MISSING
            and step.params.get("per_period") is True
            and "limit" in step.params
        ):
            return True
        pending.extend(_row_sources(step))
    return False


def _row_sources(step: Step) -> list[str]:
    """The ids of the earlier steps `step` takes its output's rows from."""
    if isinstance(step.input, str):
        return [step.input]
    if step.inputs is MISSING:
        return []
    roles = _ROW_SOURCE_ROLES.get(step.op, ())
    if _keyed_denominator(step):
        roles = (*roles, "denominator")
    return [one for role, one in step.inputs.items() if role in roles and isinstance(one, str)]


def _keyed_denominator(step: Step) -> bool:
    """Whether `step` is a share or quotient whose `denominator_by` pairs each numerator key
    with a denominator group, so a key holds a row only in the periods its group holds."""
    return (
        step.op in _KEYED_DENOMINATOR_OPS
        and step.params is not MISSING
        and bool(step.params.get("denominator_by"))
    )


def count_selected_points(
    *, resolved_frame: ResolvedFrame, series: Sequence[ResolvedSeries], window: Mapping[str, object]
) -> int:
    """Counts the points `without_unselected_points` keeps of `series` bound over `window`:
    each series' periods in the window that it holds a row in or that the frame holds no row
    in."""
    periods = frozenset(_window_periods(resolved_frame=resolved_frame, window=window))
    gaps = len(periods - frame_periods(resolved_frame=resolved_frame))
    return sum(
        len(periods.intersection(row[TIME_PERIOD_COLUMN] for row in one.rows)) + gaps
        for one in series
    )


def without_unselected_points(
    *,
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
) -> tuple[tuple[ResolvedSeries, ...], tuple[BoundSeries, ...]]:
    """`series` and `bound_series` without the missing points a per-period selection left
    out, and without each series that keeps no point.

    A frame that keeps a selection in each period (`selects_per_period`) holds no row for a
    series in a period where it holds a row for another, because the series was not chosen
    there, so that point is dropped. A period where the frame holds no row at all is a gap
    in the data, and its missing point stays."""
    held = frame_periods(resolved_frame=resolved_frame)
    kept: list[tuple[ResolvedSeries, BoundSeries]] = []
    for one, bound in zip(series, bound_series, strict=True):
        own = {row[TIME_PERIOD_COLUMN] for row in one.rows}
        points = tuple(
            point for point in bound.points if point["period"] in own or point["period"] not in held
        )
        if points:
            kept.append((one, replace(bound, points=points)))
    return tuple(one for one, _ in kept), tuple(bound for _, bound in kept)


def _window_periods(
    *, resolved_frame: ResolvedFrame, window: Mapping[str, object]
) -> tuple[str, ...]:
    """The periods a series binds over `window` at the frame's grain (`_period_sequence`)."""
    grain = resolved_frame.structural_view.period_grain
    return _period_sequence(
        start=str(window["start"]),
        end=str(window["end"]),
        grain=grain,
        observed=frame_periods(resolved_frame=resolved_frame) if grain == "day" else frozenset(),
        span=resolved_frame.structural_view.span,
    )


def _unit(*, resolved_frame: ResolvedFrame, rows: Sequence[Mapping[str, object]]) -> object | None:
    """Returns the first row's unit, a unit dcid replaced by its label when it has one.

    Coordinate groups are non-empty and units are uniform. Empty input returns `None`, and a
    converted unit passes through unchanged. A composed unit binds each side as a part."""
    unit = rows[0].get(UNIT_COLUMN) if rows else None
    labels = resolved_frame.labels
    if sides := composed_unit_sides(unit):
        numerator, denominator = sides
        return {
            "numerator": _unit_part(part=numerator, labels=labels),
            "denominator": _unit_part(part=denominator, labels=labels),
        }
    return labels.get(unit, unit)[:LABEL_MAX] if isinstance(unit, str) else unit


def _unit_part(*, part: object, labels: Mapping[str, str]) -> object:
    """Returns one side of a composed unit in the catalogue's `unit_part` shape.

    A unit dcid becomes its label, or stays the dcid when unlabelled. A converted unit passes
    through. A StatVar side keeps its ref and gains its label, the ref without `dcid:` when
    unlabelled. Each label is cut to 200 characters."""
    if isinstance(part, str):
        return labels.get(part, part)[:LABEL_MAX]
    if isinstance(part, Mapping) and "stat_var" in part:
        ref = cast("Mapping[str, str]", part)["stat_var"]
        label = labels.get(ref, ref.removeprefix(STAT_VAR_REF_PREFIX))
        return {"stat_var": ref, "label": label[:LABEL_MAX]}
    return part


def bind_all(
    *, resolved_frame: ResolvedFrame, series: Sequence[ResolvedSeries], window: Mapping[str, object]
) -> tuple[BoundSeries, ...]:
    return tuple(
        bind_series(resolved_frame=resolved_frame, series=one, window=window) for one in series
    )


def _label(*, resolved_frame: ResolvedFrame, coordinates: Mapping[str, str | None]) -> str:
    """A series' label on its own frame, from the frame's columns and labels."""
    return series_label(
        coordinates=coordinates,
        columns=cast("Sequence[str]", resolved_frame.frame["columns"]),
        labels=resolved_frame.labels,
    )


def series_label(
    *,
    coordinates: Mapping[str, str | None],
    columns: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    """Joins the series' label parts as `joined_label` fits them."""
    parts = series_label_parts(coordinates=coordinates, columns=columns, labels=labels)
    return joined_label(*parts, joiner=" / ")


def series_label_parts(
    *,
    coordinates: Mapping[str, str | None],
    columns: Sequence[str],
    labels: Mapping[str, str],
) -> tuple[str, ...]:
    """Returns the entity-role labels in column order, using raw refs for missing labels.

    Coordinates without an entity role use the stat-var ref or label."""
    roles = role_columns(columns=tuple(columns))
    refs = [coordinates[column] for column in roles if coordinates.get(column) is not None]
    if not refs:
        refs = [coordinates[STAT_VAR_COLUMN]]
    ref_strings = cast("list[str]", refs)
    return tuple(labels.get(ref, ref) for ref in ref_strings)


def role_columns(*, columns: Sequence[str]) -> tuple[str, ...]:
    """Entity-role columns in frame order, excluding reserved and facet-identity columns and
    the enrichment columns that describe another of `columns` (`_describes_a_role`).

    Used for series labels and table-row dimensions."""
    present = frozenset(columns)
    return tuple(
        column
        for column in columns
        if column not in _RESERVED_COLUMNS
        and column not in FACET_IDENTITY_COLUMNS
        and not _describes_a_role(column, columns=present)
    )


def _describes_a_role(column: str, *, columns: frozenset[str]) -> bool:
    """Whether `column` is the type, party kind, class or whole column of another of
    `columns`, which is a function of that column's entity, so it splits no series."""
    return any(
        column.endswith(suffix) and column.removesuffix(suffix) in columns
        for suffix in _ENRICHMENT_SUFFIXES
    )


def _point_fields(period: str, *, grain: str) -> tuple[int, ...]:
    """Returns a point period's year, month, and day fields, as far as `grain` reaches.

    Raises:
        ValueError: `period` is a span, a point at another grain, or not a period.
    """
    parsed = parse_time_period(period)
    if parsed.start != parsed.end or parsed.grain != grain:
        raise ValueError(f"{period!r} is not a {grain} period")
    return tuple(int(field) for field in period.split("-"))


def _clamped_day(period: str) -> datetime.date:
    """Parses a day-grain period, clamping invalid days for date arithmetic.

    Frame-grammar may admit dates such as `2024-02-30`. The source defines
    published dates. The clamp supports calendar arithmetic and the cap check."""
    year, month, day = _point_fields(period, grain="day")
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year, month, min(day, last_day))


def _real_day(period: str) -> datetime.date | None:
    """Returns the calendar date or `None` for an invalid day.

    A period that is not a day raises `ValueError`."""
    year, month, day = _point_fields(period, grain="day")
    if day > calendar.monthrange(year, month)[1]:
        return None
    return datetime.date(year, month, day)


def _day_grain_bounds(*, start: str, end: str) -> tuple[datetime.date, datetime.date] | None:
    """Returns the calendar-valid day span within `start`..`end`, or `None` when empty.

    Invalid starts advance past their clamped day. Invalid ends use the clamp."""
    start_valid = _real_day(start)
    lower = (
        start_valid if start_valid is not None else _clamped_day(start) + datetime.timedelta(days=1)
    )
    end_valid = _real_day(end)
    upper = end_valid if end_valid is not None else _clamped_day(end)
    return (lower, upper) if lower <= upper else None


def frame_periods(*, resolved_frame: ResolvedFrame) -> frozenset[str]:
    """Returns all frame period strings for day-grain binding."""
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    index = columns.index(TIME_PERIOD_COLUMN)
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    return frozenset(cast(str, row[index]) for row in rows)


def count_periods(
    *,
    start: str,
    end: str,
    grain: str,
    observed: frozenset[str] = frozenset(),
    span: str | None = None,
) -> int:
    """Counts the periods `_period_sequence` emits without enumerating the window.

    A span frame binds one period. Year and month counts are arithmetic. Day
    counts add observed periods outside the calendar span, work bounded by
    frame rows. The pipeline uses this count to enforce
    `MAX_POINTS_PER_SERIES` before binding.

    Raises:
        ValueError: a window bound is not a period at the frame's grain, or the
            window does not cover the frame's span.
    """
    if span is not None:
        return len(_span_sequence(start=start, end=end, span=span))
    if grain == "year":
        (start_year,) = _point_fields(start, grain="year")
        (end_year,) = _point_fields(end, grain="year")
        return end_year - start_year + 1
    if grain == "month":
        start_year, start_month = _point_fields(start, grain="month")
        end_year, end_month = _point_fields(end, grain="month")
        return (end_year - start_year) * 12 + (end_month - start_month) + 1

    bounds = _day_grain_bounds(start=start, end=end)
    backbone_count = 0 if bounds is None else (bounds[1] - bounds[0]).days + 1
    extras = sum(
        1
        for period in observed
        if start <= period <= end and not _in_day_grain_backbone(period=period, bounds=bounds)
    )
    return backbone_count + extras


def _in_day_grain_backbone(
    *, period: str, bounds: tuple[datetime.date, datetime.date] | None
) -> bool:
    if bounds is None:
        return False
    lower, upper = bounds
    real = _real_day(period)
    return real is not None and lower <= real <= upper


def _span_sequence(*, start: str, end: str, span: str) -> tuple[str, ...]:
    """Returns the span as a span frame's one period when the window covers it.

    The window's bounds may be the span itself or points at the span's grain."""
    lower, upper, held = (parse_time_period(one) for one in (start, end, span))
    if lower.grain != held.grain or upper.grain != held.grain:
        raise ValueError(f"window {start}..{end} is not at the {held.grain} grain of {span!r}")
    if not lower.start <= held.start <= held.end <= upper.end:
        raise ValueError(f"window {start}..{end} does not cover the frame's period {span!r}")
    return (span,)


def _period_sequence(
    *,
    start: str,
    end: str,
    grain: str,
    observed: frozenset[str] = frozenset(),
    span: str | None = None,
) -> tuple[str, ...]:
    if span is not None:
        return _span_sequence(start=start, end=end, span=span)
    if grain == "year":
        (start_year,) = _point_fields(start, grain="year")
        (end_year,) = _point_fields(end, grain="year")
        return tuple(str(year) for year in range(start_year, end_year + 1))
    if grain == "month":
        start_year, start_month = _point_fields(start, grain="month")
        end_year, end_month = _point_fields(end, grain="month")
        periods: list[str] = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            periods.append(f"{year:04d}-{month:02d}")
            month += 1
            if month > 12:
                month = 1
                year += 1
        return tuple(periods)

    # ISO strings sort chronologically, including invalid observed days. Filter both
    # sets to the window because clamping an invalid start rounds it down.
    start_date = _clamped_day(start)
    end_date = _clamped_day(end)
    one_day = datetime.timedelta(days=1)
    calendar_days: set[str] = set()
    current = start_date
    while current <= end_date:
        iso = current.isoformat()
        if start <= iso <= end:
            calendar_days.add(iso)
        current += one_day
    within_window = {period for period in observed if start <= period <= end}
    return tuple(sorted(calendar_days | within_window))
