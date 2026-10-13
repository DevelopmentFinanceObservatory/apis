"""Computes a card's selection facts: its window and each series it draws.

Dimension totals cannot give these. One health and one education variable crossed with one
grant and one loan can pair either way, and a series that reaches 2024 says nothing of its
last value inside a card that ends in 2020. So the facts come from the resolved and bound
series the card draws.

A line chart, bar chart, table and stat card draw series and have selection facts. A source
card and a note narration draw none and have none: the caller computes facts only for a card
with bound series, and `selection_facts` refuses an empty draw.

Each drawn series carries:

- its coordinates in frame column order, without `unit` and `provenance`, which `unit` and
  `source` carry;
- its source, the frame's `provenance` value;
- its unit, read from the frame's `unit` cell so a unit keeps its ref, with the frame's label
  where it has one. A converted unit names its currency, and a composed unit's StatVar side
  is a variable;
- the first and last period holding a value, and the counts of points with and without one,
  over the points bound inside the card's window;
- its estimated and provisional counts over those valued points, from each point's
  `measurementMethod` and `provenance` cells under the rule a version's record counts by
  (`materialise.statistics.count_statuses`), so a point whose status is unknown makes both
  counts `null`.

A variable is named by its bare dcid, as the description record and the code index name it,
so the frame's `dcid:` prefix comes off a `stat_var` coordinate and a StatVar unit side.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from dfo_frame_grammar import STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, UNIT_COLUMN
from dfo_frame_grammar.grammar import STAT_VAR_REF_PREFIX
from pydantic.experimental.missing_sentinel import MISSING

from assembler.bind import BoundSeries
from assembler.frame_view import dimension_columns
from assembler.limits import LABEL_MAX
from assembler.mint.builders import complete_window
from assembler.ports import ResolvedFrame
from assembler.resolve import ResolvedSeries
from description.facts import Coordinate, DrawnSeries, SelectionFacts, Span, StatusCounts
from description.steps import ComposedUnit, SimpleUnit, Unit, VariableSide
from materialise.method_status import method_status_classes, reported_provenances
from materialise.statistics import count_statuses
from materialise.unit_cells import composed_unit_sides

__all__ = ["selection_facts"]

_PROVENANCE_COLUMN = "provenance"
_METHOD_COLUMN = "measurementMethod"


def selection_facts(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
) -> SelectionFacts:
    """Returns the selection facts of a card that draws series.

    Args:
        merged_intent: the intent the card is minted from, its relative periods resolved.
            Its `window`, completed from the frame's coverage, is the card's window.
        resolved_frame: the frame the card draws from.
        series: the resolved series, in the order the card draws them.
        bound_series: the same series bound over the card's window, in the same order.

    Returns:
        The card's window and one `DrawnSeries` per drawn series, in drawing order.

    Raises:
        ValueError: the card draws no series, as a source card and a note narration do.
    """
    if not bound_series:
        raise ValueError("a card that draws no series has no selection facts")
    window = complete_window(
        window=cast("Mapping[str, object] | None", merged_intent.get("window")),
        resolved_frame=resolved_frame,
    )
    columns = [
        column
        for column in dimension_columns(
            columns=cast("Sequence[str]", resolved_frame.frame["columns"])
        )
        if column not in (UNIT_COLUMN, _PROVENANCE_COLUMN)
    ]
    return SelectionFacts(
        window=Span(start=str(window["start"]), end=str(window["end"])),
        series=[
            _drawn_series(
                resolved=resolved, bound=bound, columns=columns, labels=resolved_frame.labels
            )
            for resolved, bound in zip(series, bound_series, strict=True)
        ],
    )


def _drawn_series(
    *,
    resolved: ResolvedSeries,
    bound: BoundSeries,
    columns: Sequence[str],
    labels: Mapping[str, str],
) -> DrawnSeries:
    coordinates = dict(resolved.coordinates)
    valued = [
        cast("str", point["period"]) for point in bound.points if point.get("value") is not None
    ]
    unit_cell = resolved.rows[0].get(UNIT_COLUMN) if resolved.rows else None
    return DrawnSeries(
        id=bound.id,
        coordinates=[
            Coordinate(
                column=column, value=_coordinate(column=column, value=coordinates.get(column))
            )
            for column in columns
        ],
        source=bound.source_ref,
        unit=_unit(cell=unit_cell, labels=labels),
        first_period=min(valued, default=None),
        last_period=max(valued, default=None),
        values=len(valued),
        missing=len(bound.points) - len(valued),
        status=_status(rows=resolved.rows, periods=valued),
    )


def _status(*, rows: Sequence[Mapping[str, object]], periods: Sequence[str]) -> StatusCounts:
    """The estimated and provisional counts of the valued points at `periods`, each point
    counted under its row's `(measurementMethod, provenance)`."""
    rows_by_period = {row[TIME_PERIOD_COLUMN]: row for row in rows}
    values_by_facet: Counter[tuple[str | None, str | None]] = Counter(
        (
            cast("str | None", rows_by_period[period].get(_METHOD_COLUMN)),
            cast("str | None", rows_by_period[period].get(_PROVENANCE_COLUMN)),
        )
        for period in periods
    )
    figures = count_statuses(
        values_by_facet, classes=method_status_classes(), reported=reported_provenances()
    )
    return StatusCounts(estimated=figures.estimated, provisional=figures.provisional)


def _coordinate(*, column: str, value: str | None) -> str | None:
    """A coordinate's value, a `stat_var` cell's ref as its bare dcid."""
    if column == STAT_VAR_COLUMN and value is not None:
        return value.removeprefix(STAT_VAR_REF_PREFIX)
    return value


def _unit(*, cell: object, labels: Mapping[str, str]) -> Unit:
    """Maps a frame `unit` cell to the kit's unit: `None` stays `None`, a composed unit keeps
    its two sides, and any other cell is one `SimpleUnit`."""
    if cell is None:
        return None
    if sides := composed_unit_sides(cell):
        numerator, denominator = sides
        return ComposedUnit(
            numerator=_unit_side(part=numerator, labels=labels),
            denominator=_unit_side(part=denominator, labels=labels),
        )
    return _simple_unit(part=cell, labels=labels)


def _unit_side(*, part: object, labels: Mapping[str, str]) -> SimpleUnit | VariableSide:
    """Maps one side of a composed unit. A StatVar side `{stat_var}` is a `VariableSide`
    named by its bare dcid, with the frame's label for its prefixed ref."""
    if isinstance(part, Mapping) and "stat_var" in part:
        ref = cast("str", cast("Mapping[str, object]", part)["stat_var"])
        return VariableSide(
            stat_var=ref.removeprefix(STAT_VAR_REF_PREFIX), label=_label(ref=ref, labels=labels)
        )
    return _simple_unit(part=part, labels=labels)


def _simple_unit(*, part: object, labels: Mapping[str, str]) -> SimpleUnit:
    """Maps a unit ref or a converted unit `{currency, prices, base_year?}` to a `SimpleUnit`.
    A ref keeps the frame's label where it has one. A converted unit names its ISO 4217
    `currency`, and keeps `prices` and any `base_year`."""
    if isinstance(part, str):
        return SimpleUnit(ref=part, label=_label(ref=part, labels=labels))
    unit = cast("Mapping[str, object]", part)
    base_year = unit.get("base_year")
    return SimpleUnit(
        currency=cast("str", unit["currency"]),
        prices=cast("str", unit["prices"]),
        base_year=MISSING if base_year is None else cast("int", base_year),
    )


def _label(*, ref: str, labels: Mapping[str, str]) -> str | MISSING:
    label = labels.get(ref)
    return label[:LABEL_MAX] if label else MISSING
