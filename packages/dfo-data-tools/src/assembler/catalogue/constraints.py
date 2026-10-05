"""Versioned checks for resolved-payload constraints beyond JSON Schema.

Checks receive `component` and, when needed, binding-time facts such as
`observed_periods`. Their pointers address the component subtree. The
pipeline translates failures to intent-relative pointers. Bar baseline
checks bind by component type because only bars encode magnitude from zero.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class ConstraintFailure:
    """One constraint's refusal: enough to build a `ConstraintViolation`
    (`assembler.errors`) without this module knowing that type exists.
    """

    pointer: str
    detail: str
    fix: str


@dataclass(frozen=True)
class ConstraintSpec:
    """Registry entry bound by encoding class, component type, or both."""

    constraint_id: str
    encoding_classes: tuple[str, ...]
    component_types: tuple[str, ...]
    check: Callable[[Mapping[str, object]], ConstraintFailure | None]


def _mapping(obj: Mapping[str, object], key: str) -> Mapping[str, object]:
    """`obj[key]` narrowed to `Mapping[str, object]`, treating an absent key
    or an explicit `null` the same as an empty object -- every mapping-typed
    field these checks read is itself optional or nullable on the wire.
    """
    return cast("Mapping[str, object]", obj.get(key) or {})


def _sequence(obj: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    """`obj[key]` narrowed to a sequence of mappings, treating an absent key
    as empty.
    """
    return cast("Sequence[Mapping[str, object]]", obj.get(key) or ())


def _value_axis_key(component: Mapping[str, object]) -> str:
    """Returns `value_axis` for bars and `y_axis` for other value-axis components."""
    return "value_axis" if component.get("type") == "bar_chart" else "y_axis"


def _log_scale_requires_positive_values(context: Mapping[str, object]) -> ConstraintFailure | None:
    """Log scale requires every visible value and authored lower bound to be positive."""
    component = _mapping(context, "component")
    axis_key = _value_axis_key(component)
    value_axis = _mapping(component, axis_key)
    if value_axis.get("scale") != "log":
        return None
    range_ = _mapping(value_axis, "range")
    range_min = cast("float | None", range_.get("min"))
    if range_min is not None and range_min <= 0:
        return ConstraintFailure(
            pointer=f"/component/{axis_key}/range/min",
            detail=f"log scale invalid: authored range.min {range_min} is at or below zero",
            fix="raise range.min above zero or drop the authored bound",
        )
    for series in _sequence(component, "series"):
        for point in _sequence(series, "points"):
            value = cast("float | None", point.get("value"))
            if value is not None and value <= 0:
                return ConstraintFailure(
                    pointer=f"/component/{axis_key}/scale",
                    detail=f"log scale invalid, series {series.get('id')} contains value {value}",
                    fix="filter the series or keep the scale linear",
                )
    return None


def _range_contains_visible_data(context: Mapping[str, object]) -> ConstraintFailure | None:
    """Authored bounds must contain visible points. Use `select` or `window` to exclude data
    and disclose that exclusion."""
    component = _mapping(context, "component")
    axis_key = _value_axis_key(component)
    value_axis = _mapping(component, axis_key)
    range_ = _mapping(value_axis, "range")
    range_min = cast("float | None", range_.get("min"))
    range_max = cast("float | None", range_.get("max"))
    if range_min is None and range_max is None:
        return None
    for series in _sequence(component, "series"):
        for point in _sequence(series, "points"):
            value = cast("float | None", point.get("value"))
            if value is None:
                continue
            if range_min is not None and value < range_min:
                return ConstraintFailure(
                    pointer=f"/component/{axis_key}/range/min",
                    detail=(
                        f"authored range.min {range_min} excludes visible value {value} "
                        f"in series {series.get('id')}"
                    ),
                    fix="lower range.min to contain the value, or clear the bound",
                )
            if range_max is not None and value > range_max:
                return ConstraintFailure(
                    pointer=f"/component/{axis_key}/range/max",
                    detail=(
                        f"authored range.max {range_max} excludes visible value {value} "
                        f"in series {series.get('id')}"
                    ),
                    fix="raise range.max to contain the value, or clear the bound",
                )
    return None


def _comparable_units_required(context: Mapping[str, object]) -> ConstraintFailure | None:
    """Series sharing a value axis need identical units, including price basis and base year.

    Presentation performs no unit conversion, so units compare exactly.
    `null` and populated units differ. Currency-aware equivalence belongs to
    the structured-unit design."""
    component = _mapping(context, "component")
    series_list = _sequence(component, "series")
    if len(series_list) < 2:
        return None
    first = series_list[0]
    first_key = json.dumps(first.get("unit"), sort_keys=True)
    for series in series_list[1:]:
        key = json.dumps(series.get("unit"), sort_keys=True)
        if key != first_key:
            return ConstraintFailure(
                pointer="/component/series",
                detail=(
                    f"series {first.get('id')} and {series.get('id')} do not share a "
                    "commensurable unit"
                ),
                fix="convert upstream to a common unit, or split into separate charts",
            )
    return None


def _stat_change_basis_observed(context: Mapping[str, object]) -> ConstraintFailure | None:
    """Both change basis periods must have observed, non-missing, non-suppressed values.

    The binding-time observed-period set supplies this fact."""
    component = _mapping(context, "component")
    change = component.get("change")
    if change is None:
        return None
    change = cast("Mapping[str, object]", change)
    observed_periods = cast("Sequence[str]", context.get("observed_periods") or ())
    for period in cast("Sequence[str]", change.get("basis", ())):
        if period not in observed_periods:
            return ConstraintFailure(
                pointer="/component/change/basis",
                detail=f"change basis period {period} has no observed point",
                fix="choose a basis period both sides of the change actually hold",
            )
    return None


def _bar_zero_in_domain(context: Mapping[str, object]) -> ConstraintFailure | None:
    """Bars encode magnitude from zero. Log scales lack a zero value.

    Bounded linear ranges must include zero or bar lengths misstate ratios.
    Unbounded ranges render from zero, so single-sign data is valid."""
    component = _mapping(context, "component")
    value_axis = _mapping(component, "value_axis")
    if value_axis.get("scale") == "log":
        return ConstraintFailure(
            pointer="/component/value_axis/scale",
            detail="a bar renders its length from a zero baseline, and a log scale has no zero",
            fix="use a linear scale",
        )
    range_ = _mapping(value_axis, "range")
    range_min = cast("float | None", range_.get("min"))
    range_max = cast("float | None", range_.get("max"))
    if range_min is None or range_max is None:
        return None
    if range_min <= 0 <= range_max:
        return None
    return ConstraintFailure(
        pointer="/component/value_axis/range",
        detail=(
            f"bar_chart authored range [{range_min}, {range_max}] excludes zero; a bar drawn "
            "from a non-zero baseline misrepresents the magnitude it encodes"
        ),
        fix="widen the authored range to include zero, or clear the bound to render from zero",
    )


CONSTRAINT_REGISTRY: tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        constraint_id="log_scale_requires_positive_values",
        encoding_classes=("position",),
        component_types=(),
        check=_log_scale_requires_positive_values,
    ),
    ConstraintSpec(
        constraint_id="range_contains_visible_data",
        encoding_classes=("position",),
        component_types=(),
        check=_range_contains_visible_data,
    ),
    ConstraintSpec(
        constraint_id="comparable_units_required",
        encoding_classes=("position",),
        component_types=(),
        check=_comparable_units_required,
    ),
    ConstraintSpec(
        constraint_id="stat_change_basis_observed",
        encoding_classes=(),
        component_types=("stat_card",),
        check=_stat_change_basis_observed,
    ),
    ConstraintSpec(
        constraint_id="bar_zero_in_domain",
        encoding_classes=(),
        component_types=("bar_chart",),
        check=_bar_zero_in_domain,
    ),
)
"""Blocking set for the launch catalogue (line_chart, bar_chart, table,
stat_card). The three axis-shaped constraints key on the `"position"`
encoding class and auto-attach to any member that carries it.
`stat_change_basis_observed` and `bar_zero_in_domain` key on their one
component type, since both reason about a property specific to that
type."""
