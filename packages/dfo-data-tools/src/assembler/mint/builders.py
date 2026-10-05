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
from assembler.frame_view import dimension_columns, facet_dimension_names
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
    stat_var_labels = _stat_var_labels(resolved_frame=resolved_frame, series=series)
    y_label = _axis_label(stat_var_labels=stat_var_labels)
    window = complete_window(
        window=cast(Mapping[str, object] | None, merged_intent.get("window")),
        resolved_frame=resolved_frame,
    )
    title_text = cast(str | None, merged_intent.get("title")) or _default_title(
        bound_series=bound_series, y_label=y_label, stat_var_labels=stat_var_labels
    )

    payload_series = [
        _series_payload(bound=one, source_id_by_raw_ref=source_id_by_raw_ref)
        for one in bound_series
    ]
    if len(stat_var_labels) > 1:
        # `BoundSeries.label` names entity roles alone, so lines that differ only on stat_var
        # would share a label. Each line leads with its own stat-var label.
        for payload, resolved, bound in zip(payload_series, series, bound_series, strict=True):
            stat_var_label = _coordinate_label(
                labels=resolved_frame.labels,
                value=dict(resolved.coordinates).get(STAT_VAR_COLUMN),
            )
            if bound.label != stat_var_label:
                payload["label"] = f"{stat_var_label} / {bound.label}"[:_LABEL_MAX]
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
    """Builds a bar chart in entity or period mode.

    A select that lists values on one dimension draws one bar per value, in list order. A
    select that resolves to one series draws it across the window. Otherwise, when the
    series vary on exactly one dimension and each holds a value in the one period with data,
    the chart draws one bar per value of that dimension, largest first; `assembler.reply`
    states this default in the fact summary. Any other selection refuses `binding_invalid`."""
    stat_var_labels = _stat_var_labels(resolved_frame=resolved_frame, series=series)
    y_label = _axis_label(stat_var_labels=stat_var_labels)
    title_text = cast(str | None, merged_intent.get("title")) or _default_title(
        bound_series=bound_series, y_label=y_label, stat_var_labels=stat_var_labels
    )
    select = cast(Mapping[str, object], merged_intent["select"])
    if has_arrayed_dimension(select=select):
        category_dimension = next(name for name, value in select.items() if isinstance(value, list))
        _require_no_other_varying_dimension(
            category_dimension=category_dimension,
            select=select,
            resolved_frame=resolved_frame,
            series=series,
        )
        series_and_axis = _bar_chart_entity_mode(
            category_dimension=category_dimension,
            labels=resolved_frame.labels,
            series=series,
            bound_series=bound_series,
            source_id_by_raw_ref=source_id_by_raw_ref,
            largest_first=False,
        )
    elif len(bound_series) == 1:
        series_and_axis = _bar_chart_period_mode(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            primary=bound_series[0],
            source_id_by_raw_ref=source_id_by_raw_ref,
        )
    else:
        series_and_axis = _bar_chart_entity_mode(
            category_dimension=_default_category_dimension(
                select=select,
                resolved_frame=resolved_frame,
                series=series,
                bound_series=bound_series,
            ),
            labels=resolved_frame.labels,
            series=series,
            bound_series=bound_series,
            source_id_by_raw_ref=source_id_by_raw_ref,
            largest_first=True,
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
    category_dimension: str,
    labels: Mapping[str, str],
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    source_id_by_raw_ref: Mapping[str | None, str],
    largest_first: bool,
) -> tuple[list[dict[str, object]], dict[str, object], frozenset[str]]:
    """Builds one bar per value of `category_dimension` at the latest shared observed period,
    in series order or, with `largest_first`, by value from largest to smallest. Each bar
    carries its value's label on that dimension, which is what tells the bars apart."""
    shared_period = _shared_observed_period(bound_series=bound_series)
    payload_series: list[dict[str, object]] = [
        {
            "id": bound.id,
            "label": _coordinate_label(
                labels=labels, value=dict(resolved.coordinates).get(category_dimension)
            ),
            "unit": bound.unit,
            "source_ref": source_id_by_raw_ref[bound.source_ref],
            "points": [dict(_point_at(points=bound.points, period=shared_period))],
        }
        for resolved, bound in zip(series, bound_series, strict=True)
    ]
    if largest_first:
        # Every series is observed at the shared period, so each bar holds a value.
        payload_series.sort(
            key=lambda bar: cast(float, cast("list[dict[str, object]]", bar["points"])[0]["value"]),
            reverse=True,
        )
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
    primary: BoundSeries,
    source_id_by_raw_ref: Mapping[str | None, str],
) -> tuple[list[dict[str, object]], dict[str, object], frozenset[str]]:
    """Builds one series of bars across the selected window."""
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


def _default_category_dimension(
    *,
    select: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
) -> str:
    """The one dimension a bar chart can compare when select lists no values.

    The series must vary on exactly one dimension other than a facet and hold data in one
    period, and every series must hold a value in it. A facet that varies here pairs with
    another dimension, since resolution refuses a facet that splits a series. Any other
    selection is a choice the caller makes, so it refuses `binding_invalid` naming the values
    and the next call."""
    varying = tuple(
        name
        for name in _varying_unselected_dimensions(
            resolved_frame=resolved_frame, series=series, selected_names=frozenset(select)
        )
        if name not in facet_dimension_names()
    )
    observed = sorted(
        {
            cast(str, point["period"])
            for one in bound_series
            for point in one.points
            if "value" in point
        }
    )
    # With no observed period the entity-mode build refuses for want of a shared period.
    if len(varying) == 1 and len(observed) <= 1:
        if observed:
            _require_every_value_at(
                dimension=varying[0],
                period=observed[0],
                series=series,
                bound_series=bound_series,
                labels=resolved_frame.labels,
            )
        return varying[0]

    # Resolution refuses a facet that splits a series, so several series vary on at least
    # one dimension other than a facet.
    assert varying
    listed, *others = varying
    values_by_name = {
        name: tuple(
            dict.fromkeys(value for one in series for key, value in one.coordinates if key == name)
        )
        for name in varying
    }
    labels = resolved_frame.labels
    unbound = ", ".join(f"{name!r} ({len(values_by_name[name])} values)" for name in varying)
    if others:
        bindable = "; ".join(
            f"{name!r} holds "
            + ", ".join(
                _value_with_label(labels=labels, value=value) for value in values_by_name[name]
            )
            for name in others
        )
        detail = (
            f"a bar_chart compares one dimension; select leaves {unbound} unbound. List the "
            f"values to compare on {listed!r} in select and bind each of "
            f"{', '.join(repr(name) for name in others)} to one value. {bindable}"
        )
    else:
        detail = (
            f"a bar_chart draws one series across periods or one bar per value at one period; "
            f"select leaves {unbound} unbound and the window holds data in {len(observed)} "
            f"periods ({observed[0]}..{observed[-1]}). Bind {listed!r} to one value to draw its "
            f"periods, or list the {listed!r} values to compare in select to draw them at the "
            "latest period they share, or at the one period a window names"
        )
    raise BindingInvalid(
        pointer="/select",
        dimension=listed,
        legal_values=tuple(value for value in values_by_name[listed] if value is not None),
        truncated=False,
        detail=detail,
        labels=labels,
    )


def _require_no_other_varying_dimension(
    *,
    category_dimension: str,
    select: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
) -> None:
    """Refuses `binding_invalid` when two series share a value of `category_dimension`. A bar
    names only that value, so those bars would carry one label; they differ on a dimension
    select leaves unbound, which the refusal names."""
    category_values = [dict(one.coordinates).get(category_dimension) for one in series]
    if len(set(category_values)) == len(category_values):
        return
    others = tuple(
        name
        for name in _varying_unselected_dimensions(
            resolved_frame=resolved_frame, series=series, selected_names=frozenset(select)
        )
        if name not in facet_dimension_names()
    )
    # Resolution refuses a facet that splits a series, so series sharing a category value
    # differ on another unbound dimension.
    assert others
    first = others[0]
    values = tuple(
        dict.fromkeys(value for one in series for key, value in one.coordinates if key == first)
    )
    labels = resolved_frame.labels
    raise BindingInvalid(
        pointer="/select",
        dimension=first,
        legal_values=tuple(value for value in values if value is not None),
        truncated=False,
        detail=(
            f"a bar_chart compares one dimension; select lists {category_dimension!r} values "
            f"and leaves {', '.join(repr(name) for name in others)} unbound. Bind each to one "
            f"value. {first!r} holds "
            + ", ".join(_value_with_label(labels=labels, value=value) for value in values)
        ),
        labels=labels,
    )


def _require_every_value_at(
    *,
    dimension: str,
    period: str,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    labels: Mapping[str, str],
) -> None:
    """Refuses `binding_invalid` when a value of the compared dimension has no observed
    value at the one period the default draws, so the default never drops a value the
    caller did not leave out. `legal_values` holds the values observed there."""
    observed: list[str | None] = []
    unobserved: list[str | None] = []
    for resolved, bound in zip(series, bound_series, strict=True):
        value = dict(resolved.coordinates)[dimension]
        point = _point_at(points=bound.points, period=period)
        (observed if "value" in point else unobserved).append(value)
    if not unobserved:
        return
    lacking = ", ".join(_value_with_label(labels=labels, value=value) for value in unobserved)
    raise BindingInvalid(
        pointer="/select",
        dimension=dimension,
        legal_values=tuple(value for value in observed if value is not None),
        truncated=False,
        detail=(
            f"select leaves {dimension!r} unbound, and {period} holds no observed value for "
            f"{len(unobserved)} of its {len(series)} values: {lacking}. List the "
            f"{dimension!r} values to draw in select; alternatives holds the "
            f"{len(observed)} observed in {period}"
        ),
        labels=labels,
    )


def _value_with_label(*, labels: Mapping[str, str], value: str | None) -> str:
    """A coordinate value as `label (ref)`, or the ref alone when it has no label."""
    if value is None:
        return _ABSENT_COORDINATE_LABEL
    label = labels.get(value)
    return f"{label} ({value})" if label and label != value else value


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
    stat_var_labels = _stat_var_labels(resolved_frame=resolved_frame, series=series)
    y_label = _axis_label(stat_var_labels=stat_var_labels)
    title_text = cast(str | None, merged_intent.get("title")) or _default_title(
        bound_series=bound_series, y_label=y_label, stat_var_labels=stat_var_labels
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


def _stat_var_labels(
    *, resolved_frame: ResolvedFrame, series: Sequence[ResolvedSeries]
) -> tuple[str, ...]:
    """The distinct stat-var labels across `series`, in series order, raw refs when
    unlabelled."""
    refs = dict.fromkeys(
        value
        for one in series
        for name, value in one.coordinates
        if name == STAT_VAR_COLUMN and value is not None
    )
    return tuple(resolved_frame.labels.get(ref, ref) for ref in refs)


def _axis_label(*, stat_var_labels: Sequence[str]) -> str:
    """The value axis names the one stat-var the series share, distinct from the series
    labels. Series over several stat-vars share only their unit, which the series carry, so
    the axis reads `Value`."""
    return stat_var_labels[0] if len(stat_var_labels) == 1 else "Value"


def _default_title(
    *, bound_series: Sequence[BoundSeries], y_label: str, stat_var_labels: Sequence[str]
) -> str:
    """Fallback title: the measure plus the sole series label, or every stat-var the series
    show."""
    if len(bound_series) == 1:
        return f"{y_label} — {bound_series[0].label}"
    return ", ".join(stat_var_labels) or y_label


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
