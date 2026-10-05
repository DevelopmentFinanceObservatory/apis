"""Builds catalogue components from bound series and frame labels.

Authored axis scale and range pass through. Defaults are linear and
unbounded. Stat-card change periods must be observed. The default derivation
is `difference`.

`note_narration` is frameless and carries text. `source_card` resolves a
frame and lists every distinct provenance ref."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from dfo_frame_grammar import STAT_VAR_COLUMN

from assembler.bind import BoundSeries, role_columns
from assembler.errors import BindingInvalid, ConstraintViolation
from assembler.frame_view import dimension_columns
from assembler.ports import ResolvedFrame
from assembler.resolve import ResolvedSeries
from assembler.select import has_arrayed_dimension

_TITLE_MAX = 200
_LABEL_MAX = 200
_TEXT_MAX = 2000
# The label a table row carries for a dimension its series leaves null.
_ABSENT_COORDINATE_LABEL = "Not specified"


def default_window(*, resolved_frame: ResolvedFrame) -> Mapping[str, str]:
    """Returns the frame's full period coverage, matching `resolve_selectors`'s window check.

    This supplies a line chart's `x_axis.window` and the stat card's binding window."""
    coverage = resolved_frame.structural_view.coverage
    if coverage is None:
        # Resolution refuses empty frames; keep a fallback for totality.
        return {"start": "1900", "end": "1900"}
    return {"start": coverage[0], "end": coverage[1]}


def complete_window(
    *, window: Mapping[str, object] | None, resolved_frame: ResolvedFrame
) -> Mapping[str, object]:
    """The intent's `window` with any omitted side filled from
    `default_window`: the intent grammar lets a window leave out `start` or
    `end`, and an omitted side runs to the edge of the frame's coverage."""
    return {**default_window(resolved_frame=resolved_frame), **(window or {})}


def _build_line_chart(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    source_id_by_raw_ref: Mapping[str | None, str],
) -> tuple[dict[str, object], frozenset[str]]:
    y_label = _axis_label(resolved_frame=resolved_frame, series=series)
    window = complete_window(
        window=cast(Mapping[str, object] | None, merged_intent.get("window")),
        resolved_frame=resolved_frame,
    )
    title_text = cast(str | None, merged_intent.get("title")) or _default_title(
        bound_series=bound_series, y_label=y_label
    )

    payload_series = [
        _series_payload(bound=one, source_id_by_raw_ref=source_id_by_raw_ref)
        for one in bound_series
    ]
    observed_periods = frozenset(
        cast(str, point["period"])
        for one in bound_series
        for point in one.points
        if "value" in point
    )
    scale, range_ = _value_axis_scale_and_range(merged_intent=merged_intent)
    component: dict[str, object] = {
        "type": "line_chart",
        "title": {"text": title_text[:_TITLE_MAX], "authorship": "model"},
        "series": payload_series,
        "x_axis": {"window": {"start": str(window["start"]), "end": str(window["end"])}},
        "y_axis": {
            "label": y_label[:_LABEL_MAX],
            "scale": scale,
            "range": range_,
        },
    }
    return component, observed_periods


def _build_bar_chart(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    source_id_by_raw_ref: Mapping[str | None, str],
) -> tuple[dict[str, object], frozenset[str]]:
    y_label = _axis_label(resolved_frame=resolved_frame, series=series)
    title_text = cast(str | None, merged_intent.get("title")) or _default_title(
        bound_series=bound_series, y_label=y_label
    )
    select = cast(Mapping[str, object], merged_intent["select"])
    if has_arrayed_dimension(select=select):
        series_and_axis = _bar_chart_entity_mode(
            select=select, bound_series=bound_series, source_id_by_raw_ref=source_id_by_raw_ref
        )
    else:
        series_and_axis = _bar_chart_period_mode(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
            source_id_by_raw_ref=source_id_by_raw_ref,
        )
    payload_series, category_axis, observed_periods = series_and_axis
    scale, range_ = _value_axis_scale_and_range(merged_intent=merged_intent)
    component: dict[str, object] = {
        "type": "bar_chart",
        "title": {"text": title_text[:_TITLE_MAX], "authorship": "model"},
        "series": payload_series,
        "category_axis": category_axis,
        "value_axis": {
            "label": y_label[:_LABEL_MAX],
            "scale": scale,
            "range": range_,
        },
    }
    return component, observed_periods


def _bar_chart_entity_mode(
    *,
    select: Mapping[str, object],
    bound_series: Sequence[BoundSeries],
    source_id_by_raw_ref: Mapping[str | None, str],
) -> tuple[list[dict[str, object]], dict[str, object], frozenset[str]]:
    """Builds one bar per entity at the latest shared observed period."""
    category_dimension = next(name for name, value in select.items() if isinstance(value, list))
    shared_period = _shared_observed_period(bound_series=bound_series)
    payload_series: list[dict[str, object]] = [
        {
            "id": one.id,
            "label": one.label[:_LABEL_MAX],
            "unit": one.unit,
            "source_ref": source_id_by_raw_ref[one.source_ref],
            "points": [dict(_point_at(points=one.points, period=shared_period))],
        }
        for one in bound_series
    ]
    category_axis: dict[str, object] = {
        "mode": "entity",
        "category_dimension": category_dimension,
        "period": shared_period,
    }
    return payload_series, category_axis, frozenset({shared_period})


def _bar_chart_period_mode(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    source_id_by_raw_ref: Mapping[str | None, str],
) -> tuple[list[dict[str, object]], dict[str, object], frozenset[str]]:
    """Builds one series of bars across the selected window."""
    if len(bound_series) != 1:
        raise _period_mode_needs_one_series(
            merged_intent=merged_intent, resolved_frame=resolved_frame, series=series
        )
    primary = bound_series[0]
    window = complete_window(
        window=cast(Mapping[str, object] | None, merged_intent.get("window")),
        resolved_frame=resolved_frame,
    )
    observed_periods = frozenset(
        cast(str, point["period"]) for point in primary.points if "value" in point
    )
    payload_series = [_series_payload(bound=primary, source_id_by_raw_ref=source_id_by_raw_ref)]
    category_axis: dict[str, object] = {
        "mode": "period",
        "window": {"start": str(window["start"]), "end": str(window["end"])},
    }
    return payload_series, category_axis, observed_periods


def _period_mode_needs_one_series(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
) -> BindingInvalid:
    """The refusal for a period-mode bar chart whose select left a dimension unbound.

    It names every varying dimension and offers the first one's values, with their
    labels, as the list to compare. Retrying with that list draws an entity-mode chart;
    each other varying dimension still needs one value."""
    select = cast(Mapping[str, object], merged_intent["select"])
    varying = _varying_unselected_dimensions(
        resolved_frame=resolved_frame, series=series, selected_names=frozenset(select)
    )
    listed, *others = varying
    values = tuple(
        dict.fromkeys(
            value
            for one in series
            for name, value in one.coordinates
            if name == listed and value is not None
        )
    )
    detail = (
        f"a period-mode bar_chart names one series; this selection resolves to {len(series)} "
        f"because select leaves {', '.join(repr(name) for name in varying)} unbound. "
        f"To compare entities, list the values to compare on {listed!r} in select"
    )
    if others:
        detail += (
            "; bind each of "
            f"{', '.join(repr(name) for name in others)} to one value, or the chart draws "
            "one bar per combination"
        )
    return BindingInvalid(
        pointer="/select",
        dimension=listed,
        legal_values=values,
        truncated=False,
        detail=detail,
        labels=resolved_frame.labels,
    )


def _point_at(*, points: Sequence[Mapping[str, object]], period: str) -> Mapping[str, object]:
    return next(point for point in points if point["period"] == period)


def _shared_observed_period(*, bound_series: Sequence[BoundSeries]) -> str:
    """Latest period with an observed value in every selected series."""
    observed_by_series = [
        {cast(str, point["period"]) for point in one.points if "value" in point}
        for one in bound_series
    ]
    common = set.intersection(*observed_by_series) if observed_by_series else set()
    if not common:
        raise BindingInvalid(
            pointer="/select",
            dimension="",
            legal_values=(),
            truncated=False,
            detail="the entities share no observed period in the window",
        )
    return max(common)


def _varying_unselected_dimensions(
    *,
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    selected_names: frozenset[str],
) -> tuple[str, ...]:
    """Unselected dimensions that vary across table series, ordered by frame columns.

    They distinguish rows when unbound entity or stat-var values expand into
    separate series. A null value counts as a distinct value, so a series
    that lacks the dimension stays distinguishable from one that sets it.
    Omitted frame columns sort alphabetically."""
    values_by_name: dict[str, set[object]] = {}
    for one in series:
        for name, value in one.coordinates:
            values_by_name.setdefault(name, set()).add(value)
    varying = {
        name
        for name, values in values_by_name.items()
        if name not in selected_names and len(values) > 1
    }
    if not varying:
        return ()
    frame_columns = dimension_columns(
        columns=cast("Sequence[str]", resolved_frame.frame["columns"])
    )
    ordered = [name for name in frame_columns if name in varying]
    leftover = sorted(varying - set(ordered))
    return tuple(ordered) + tuple(leftover)


def _coordinate_label(*, labels: Mapping[str, str], value: str | None) -> str:
    """A table row's label for one coordinate value."""
    if value is None:
        return _ABSENT_COORDINATE_LABEL
    return labels.get(value, value)[:_LABEL_MAX]


def _build_table(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    source_id_by_raw_ref: Mapping[str | None, str],
) -> tuple[dict[str, object], frozenset[str]]:
    select = cast(Mapping[str, object], merged_intent["select"])
    selected_names = tuple(select)
    dimension_names = selected_names + _varying_unselected_dimensions(
        resolved_frame=resolved_frame, series=series, selected_names=frozenset(selected_names)
    )
    title_text = cast(str | None, merged_intent.get("title")) or ", ".join(selected_names)

    columns = [
        {"key": name, "label": resolved_frame.labels.get(name, name)[:_LABEL_MAX]}
        for name in dimension_names
    ] + [{"key": "period", "label": "Period"}, {"key": "value", "label": "Value"}]

    # Every entity-role dimension contributes to a row's label, including
    # constant unselected roles. Their values appear in `dimensions`.
    # Selected and varying roles also receive columns.
    frame_role_names = role_columns(
        columns=tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    )
    extra_role_names = tuple(name for name in frame_role_names if name not in dimension_names)

    rows: list[dict[str, object]] = []
    for resolved, bound in zip(series, bound_series, strict=True):
        coordinates = dict(resolved.coordinates)
        dimensions = {
            name: _coordinate_label(labels=resolved_frame.labels, value=coordinates.get(name))
            for name in dimension_names
        }
        for name in extra_role_names:
            value = coordinates.get(name)
            if value is not None:
                dimensions[name] = _coordinate_label(labels=resolved_frame.labels, value=value)
        source_ref = source_id_by_raw_ref[bound.source_ref]
        for point in bound.points:
            row: dict[str, object] = {
                "series_id": bound.id,
                "dimensions": dimensions,
                "period": point["period"],
                "source_ref": source_ref,
                # Tables can mix stat_vars with different units, so each row carries its
                # own unit, like a chart series.
                "unit": bound.unit,
            }
            if "value" in point:
                row["value"] = point["value"]
            else:
                row["status"] = point["status"]
            rows.append(row)

    observed_periods = frozenset(cast(str, row["period"]) for row in rows if "value" in row)
    component: dict[str, object] = {
        "type": "table",
        "title": {"text": title_text[:_TITLE_MAX], "authorship": "model"},
        "columns": columns,
        "rows": rows,
    }
    return component, observed_periods


def _build_stat_card(
    *,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
) -> tuple[dict[str, object], frozenset[str]]:
    primary = bound_series[0]
    headline = _headline_point(points=primary.points)
    y_label = _axis_label(resolved_frame=resolved_frame, series=series)
    title_text = cast(str | None, merged_intent.get("title")) or _default_title(
        bound_series=bound_series, y_label=y_label
    )
    component: dict[str, object] = {
        "type": "stat_card",
        "title": {"text": title_text[:_TITLE_MAX], "authorship": "model"},
        "value": cast(float, headline["value"]),
        "label": primary.label[:_LABEL_MAX],
        "period": cast(str, headline["period"]),
        "unit": primary.unit,
    }
    observed_values = _series_observed_values(points=primary.points)
    change = _stat_card_change(merged_intent=merged_intent, observed_values=observed_values)
    if change is not None:
        component["change"] = change
    # The same windowed set `_stat_card_change` validated `change.basis`
    # against, so the catalogue registry's `stat_change_basis_observed`
    # copy agrees with what this function already checked.
    observed_periods = frozenset(observed_values)
    return component, observed_periods


def _series_observed_values(*, points: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Observed primary-series values keyed by period for stat-card change calculations."""
    return {
        cast(str, point["period"]): cast(float, point["value"])
        for point in points
        if "value" in point
    }


def _stat_card_change(
    *, merged_intent: Mapping[str, object], observed_values: Mapping[str, float]
) -> dict[str, object] | None:
    """Builds `change` from observed values in the primary series' window, or returns
    `None` when absent. Unobserved basis periods raise
    `ConstraintViolation` at `/change/basis`. Percent change from a value at
    or below zero raises `BindingInvalid` at `/change`."""
    change_intent = cast(Mapping[str, object] | None, merged_intent.get("change"))
    if change_intent is None:
        return None
    from_period, to_period = cast(Sequence[str], change_intent["basis"])
    for period in (from_period, to_period):
        if period not in observed_values:
            raise ConstraintViolation(
                constraint_id="stat_change_basis_observed",
                pointer="/change/basis",
                detail=f"change basis period {period} has no observed point",
                fix="choose a basis period both sides of the change actually hold",
            )
    derivation = cast(str, change_intent.get("derivation") or "difference")
    from_value = observed_values[from_period]
    to_value = observed_values[to_period]
    if derivation == "percent_change":
        if from_value <= 0:
            raise BindingInvalid(
                pointer="/change",
                dimension="",
                legal_values=(),
                truncated=False,
                detail="percent_change is undefined for a non-positive basis 'from' value",
            )
        value = (to_value - from_value) / from_value * 100
    else:
        value = to_value - from_value
    return {"derivation": derivation, "basis": [from_period, to_period], "value": value}


def _build_note_narration(
    *, merged_intent: Mapping[str, object]
) -> tuple[dict[str, object], frozenset[str]]:
    text = cast(str, merged_intent["text"])
    component: dict[str, object] = {
        "type": "note_narration",
        "text": text[:_TEXT_MAX],
        "authorship": "model",
    }
    return component, frozenset()


def _build_source_card(
    *, sources: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], frozenset[str]]:
    component: dict[str, object] = {
        "type": "source_card",
        "source_refs": [cast(str, one["id"]) for one in sources],
    }
    return component, frozenset()


def _series_payload(
    *, bound: BoundSeries, source_id_by_raw_ref: Mapping[str | None, str]
) -> dict[str, object]:
    return {
        "id": bound.id,
        "label": bound.label[:_LABEL_MAX],
        "unit": bound.unit,
        "source_ref": source_id_by_raw_ref[bound.source_ref],
        "points": list(bound.points),
    }


def _value_axis_scale_and_range(
    *, merged_intent: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    """Returns authored scale and range, defaulting to linear and unbounded.

    A range with `min > max` raises `BindingInvalid` at `/range`."""
    scale = cast(str | None, merged_intent.get("scale")) or "linear"
    range_ = cast(Mapping[str, object] | None, merged_intent.get("range"))
    if range_ is None:
        return scale, {"min": None, "max": None}
    range_min = cast("float | None", range_.get("min"))
    range_max = cast("float | None", range_.get("max"))
    if range_min is not None and range_max is not None and range_min > range_max:
        raise BindingInvalid(
            pointer="/range",
            dimension="",
            legal_values=(),
            truncated=False,
            detail=f"range.min {range_min} exceeds range.max {range_max}",
        )
    return scale, {"min": range_min, "max": range_max}


def _axis_label(*, resolved_frame: ResolvedFrame, series: Sequence[ResolvedSeries]) -> str:
    """Returns the stat-var label for the value axis, distinct from the series label."""
    if not series:
        return "Value"
    stat_var = next(
        (value for name, value in series[0].coordinates if name == STAT_VAR_COLUMN), None
    )
    if stat_var is None:
        return "Value"
    return resolved_frame.labels.get(stat_var, stat_var)


def _default_title(*, bound_series: Sequence[BoundSeries], y_label: str) -> str:
    """Fallback title: measured value plus the sole series label, or selected table dimensions."""
    if len(bound_series) == 1:
        return f"{y_label} — {bound_series[0].label}"
    return y_label


def _headline_point(*, points: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Returns the latest observed point in the bound window.

    `period` names that point."""
    for point in reversed(points):
        if "value" in point:
            return point
    raise BindingInvalid(
        pointer="/select",
        dimension="",
        legal_values=(),
        truncated=False,
        detail="no observed point is available for the stat card's headline value",
    )
