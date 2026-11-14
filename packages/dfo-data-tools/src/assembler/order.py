"""Draws a chart's series in the order an intent's `order` member asks for.

`order` is `{"by": "value", "direction": "descending" | "ascending", "period"?: <label>}`.
The select stays a set; this module only reorders the series a draw has already resolved,
bound and trimmed (`_leave_out`), so every consumer of the series reads one order.

A series is ranked by its value at one period: the stated `period` when a drawn series holds
a value there, otherwise `ranking_period`. Series with no value at that period follow every
valued series in either direction, and ties keep their incoming order, which is the order the
select resolves to.

`order` moves whole series: a table's rows keep their period order within a series. A table
whose drawn series are in more than one unit has no common scale to rank on, so a write
refuses its `order` and a render draws it unranked, in select order. `assembler.reply` reads
the same rules over the payload to report what the draw did."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from dfo_frame_grammar import UNIT_COLUMN

from assembler.bind import BoundSeries
from assembler.errors import BindingInvalid
from assembler.limits import PER_DIMENSION_LISTING_CAP
from assembler.resolve import ResolvedSeries
from materialise.unit_cells import units_by_identity

__all__ = ["order_period", "order_series", "ranking_period", "units_differ", "value_order"]

_ORDER_POINTER = "/order"
_PERIOD_POINTER = "/order/period"


def ranking_period(*, points_by_series: Sequence[Sequence[Mapping[str, object]]]) -> str | None:
    """The latest period at which the most series hold a value, a tie in count going to the
    later period. `None` when no series holds a value. Each entry of `points_by_series` is
    one series' `{period, value?}` points, as a payload or a bound series holds them."""
    counts: dict[str, int] = {}
    for points in points_by_series:
        for period in _valued_periods(points):
            counts[period] = counts.get(period, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda period: (counts[period], period))


def order_period(
    *, stated: str | None, points_by_series: Sequence[Sequence[Mapping[str, object]]]
) -> str | None:
    """The period series are ranked at: `stated` when a series holds a value there,
    otherwise `ranking_period`. `assembler.reply` reports the same rule over the payload."""
    if stated is not None and any(stated in _valued_periods(points) for points in points_by_series):
        return stated
    return ranking_period(points_by_series=points_by_series)


def order_series(
    *,
    member: str,
    intent: Mapping[str, object],
    series: tuple[ResolvedSeries, ...],
    bound_series: tuple[BoundSeries, ...],
    refuses: bool,
    relative_period: bool,
) -> tuple[tuple[ResolvedSeries, ...], tuple[BoundSeries, ...]]:
    """`series` and `bound_series`, reordered together by `intent["order"]`, or unchanged
    when the intent has none. `refuses` is set for a write; a render never refuses on order.
    It ranks at `ranking_period` when no drawn series holds its stated period, and leaves a
    table whose series are in more than one unit in its incoming order.
    `relative_period` says the intent states a relative period.

    Raises:
        BindingInvalid: a write whose table draws series of more than one unit
            (`/order`), or whose stated `period` no drawn series holds or that sits beside
            a relative period (`/order/period`).
    """
    order = cast("Mapping[str, object] | None", intent.get("order"))
    if order is None:
        return series, bound_series
    stated = cast("str | None", order.get("period"))
    points_by_series = [one.points for one in bound_series]
    held = sorted({period for points in points_by_series for period in _valued_periods(points)})
    unranked = member == "table" and _mixed_units(series=series, bound_series=bound_series)
    if refuses:
        if unranked:
            raise BindingInvalid(
                pointer=_ORDER_POINTER,
                dimension="",
                legal_values=(),
                truncated=False,
                detail="a table of series in more than one unit cannot be ordered by value",
            )
        if stated is not None and (relative_period or stated not in held):
            raise BindingInvalid(
                pointer=_PERIOD_POINTER,
                dimension="",
                legal_values=() if relative_period else tuple(held[-PER_DIMENSION_LISTING_CAP:]),
                truncated=not relative_period and len(held) > PER_DIMENSION_LISTING_CAP,
                detail=(
                    "an order period beside a relative period would not follow the chart"
                    if relative_period
                    else f"no drawn series holds a value at {stated}"
                ),
            )
    if len(series) < 2 or unranked:
        return series, bound_series
    period = order_period(stated=stated, points_by_series=points_by_series)
    if period is None:
        return series, bound_series
    values = [_value_at(one, period) for one in bound_series]
    ranked = value_order(values=values, descending=order["direction"] == "descending")
    return (
        tuple(series[index] for index in ranked),
        tuple(bound_series[index] for index in ranked),
    )


def value_order(*, values: Sequence[float | None], descending: bool) -> list[int]:
    """Indexes of `values` in value order, largest first when `descending`. A `None` follows
    every number in either direction, and ties keep their incoming order. This is the one
    definition of value order: `order_series` and the unbound entity-mode bar chart use it."""
    sign = -1.0 if descending else 1.0
    keys = [(value is None, sign * (value or 0.0)) for value in values]
    return sorted(range(len(values)), key=keys.__getitem__)


def units_differ(
    *, entries: Iterable[tuple[object, object]], frame_unit_keys: Mapping[str, object] | None
) -> bool:
    """Whether `entries`, a `(series_id, payload_unit)` pair per series or table row, hold
    more than one unit, compared by `unit_identity`: by each series' frame `unit` coordinate
    where `frame_unit_keys` holds its id, so two units whose labels read alike are two units.
    A draw and `assembler.reply` both read this rule."""
    return len(units_by_identity(entries=entries, frame_unit_keys=frame_unit_keys)) > 1


def _valued_periods(points: Sequence[Mapping[str, object]]) -> list[str]:
    return [cast(str, point["period"]) for point in points if "value" in point]


def _value_at(one: BoundSeries, period: str) -> float | None:
    for point in one.points:
        if point["period"] == period and "value" in point:
            return cast(float, point["value"])
    return None


def _mixed_units(*, series: Sequence[ResolvedSeries], bound_series: Sequence[BoundSeries]) -> bool:
    """`units_differ` over the drawn series and their frame `unit` coordinates."""
    frame_unit_keys = {
        resolved.id: dict(resolved.coordinates).get(UNIT_COLUMN)
        for resolved in series
        if UNIT_COLUMN in dict(resolved.coordinates)
    }
    return units_differ(
        entries=((one.id, one.unit) for one in bound_series), frame_unit_keys=frame_unit_keys
    )
