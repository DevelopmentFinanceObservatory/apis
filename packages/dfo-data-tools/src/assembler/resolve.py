"""Resolves selectors to frame series, enforcing limits and bindings before row expansion.

The window must contain an observation for at least one resolved series. Series come
back in selector order, and within a selector in the order its binding lists name the
values. Series the lists leave unordered keep frame row order.

A series is the rows sharing a `frame_view.series_keys` key, so the estimates continuing a
series resolve with it, each row keeping its own method cell. The frame's continuation is
`ResolvedFrame.continuation`. A selector that binds `measurementMethod` resolves rows by their
own cells, so binding the continuing method draws
the estimates alone."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

from dfo_frame_grammar import TIME_PERIOD_COLUMN

from assembler.errors import BindingInvalid, ResourceExhausted, escape_pointer_segment
from assembler.frame_view import dimension_columns, facet_dimension_names, series_keys
from assembler.limits import MAX_SELECTORS, MAX_SERIES_PER_COMPONENT, PER_DIMENSION_LISTING_CAP
from assembler.ports import Continuation, DimensionView, ResolvedFrame, StructuralView
from assembler.series_id import Coordinates, mint_series_ids
from dfo_platform.codes import RegisteredLimitName
from materialise.bundle_columns import MEASUREMENT_METHOD_COLUMN

__all__ = [
    "COMPONENT_SERIES_LIMIT",
    "ResolvedSeries",
    "SeriesLimit",
    "check_resolved_series",
    "resolve_selectors",
    "series_limit_exhausted",
]

Row = Mapping[str, object]


def _bindings(selector: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    return cast("Sequence[Mapping[str, object]]", selector["bindings"])


def _binding_values(binding: Mapping[str, object]) -> Sequence[str]:
    return cast("Sequence[str]", binding["values"])


@dataclass(frozen=True)
class ResolvedSeries:
    """Resolved series id, full coordinates, and raw rows keyed by frame column."""

    id: str
    coordinates: Coordinates
    rows: tuple[Row, ...]


class SeriesLimit(NamedTuple):
    """The most series a component draws, and the `limit_name` a refusal of it names."""

    name: RegisteredLimitName
    cap: int


COMPONENT_SERIES_LIMIT = SeriesLimit(name="MAX_SERIES_PER_COMPONENT", cap=MAX_SERIES_PER_COMPONENT)


def series_limit_exhausted(
    *,
    limit: SeriesLimit,
    found: str,
    counts: Mapping[str, int],
    kept: str | None,
    offer: str = "",
    pointer: str = "/select",
) -> ResourceExhausted:
    """`resource_exhausted` at `pointer` naming `limit`. `found` states the series count the
    select reaches, and the advice that follows says what brings it within `limit.cap`
    (`multiplying_advice` over `counts` and `kept`). `offer` ends the detail."""
    return ResourceExhausted(
        pointer=pointer,
        limit_name=limit.name,
        detail=(
            f"{found}, over the maximum of {limit.cap}; "
            + multiplying_advice(counts, listed=limit.cap, kept=kept)
            + offer
        ),
    )


def resolve_selectors(
    *,
    resolved_frame: ResolvedFrame,
    selectors: Sequence[Mapping[str, object]],
    window: Mapping[str, object] | None,
    series_limit: SeriesLimit = COMPONENT_SERIES_LIMIT,
) -> tuple[ResolvedSeries, ...]:
    """An estimate over `MAX_SERIES_PER_COMPONENT` refuses naming `series_limit`, the limit a
    caller has to meet: a line chart held to its own cap passes the line chart's. The resolved
    series are then held to `COMPONENT_SERIES_LIMIT` (`check_resolved_series`)."""
    if len(selectors) > MAX_SELECTORS:
        raise ResourceExhausted(
            pointer="/select",
            limit_name="MAX_SELECTORS",
            detail=f"{len(selectors)} selectors exceeds the maximum of {MAX_SELECTORS}",
        )

    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    raw_rows = tuple(cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"]))
    coordinate_columns = dimension_columns(columns=columns)
    facet_names = facet_dimension_names()
    dimension_by_name = {
        dimension.name: dimension for dimension in resolved_frame.structural_view.dimensions
    }
    _check_series_estimate(
        selectors=selectors,
        coordinate_columns=coordinate_columns,
        facet_names=facet_names,
        dimension_by_name=dimension_by_name,
        series_limit=series_limit,
    )

    continuation = resolved_frame.continuation
    groups = _group_by_coordinate(columns=columns, rows=raw_rows, continuation=continuation)
    cell_groups = (
        groups
        if continuation is None
        else _group_by_coordinate(columns=columns, rows=raw_rows, continuation=None)
    )
    # Full legal-value sets support membership checks. Display listings expose
    # only the capped subset.
    legal_values = {
        column: frozenset(
            value
            for coordinates in cell_groups
            for name, value in coordinates
            if name == column and value is not None
        )
        for column in coordinate_columns
    }

    resolved: list[tuple[Coordinates, tuple[Row, ...]]] = []
    for selector_index, selector in enumerate(selectors):
        if "id" in selector:
            resolved.append(
                _resolve_by_id(
                    selector_index=selector_index,
                    target_id=str(selector["id"]),
                    groups={**cell_groups, **groups},
                )
            )
            continue

        resolved.extend(
            _resolve_by_bindings(
                selector_index=selector_index,
                bindings=_bindings(selector),
                coordinate_columns=coordinate_columns,
                facet_names=facet_names,
                dimension_by_name=dimension_by_name,
                legal_values=legal_values,
                groups=cell_groups
                if _binds_axis(selector=selector, continuation=continuation)
                else groups,
                structural_view=resolved_frame.structural_view,
                labels=resolved_frame.labels,
            )
        )

    # Overlapping selectors can resolve a coordinate more than once. Results
    # include each series once.
    unique: dict[Coordinates, tuple[Row, ...]] = {}
    order: list[Coordinates] = []
    for coordinates, rows in resolved:
        if coordinates not in unique:
            unique[coordinates] = rows
            order.append(coordinates)

    ids = mint_series_ids(coordinates_list=order)
    series = tuple(
        ResolvedSeries(id=series_id, coordinates=coordinates, rows=unique[coordinates])
        for series_id, coordinates in zip(ids, order, strict=True)
    )

    check_resolved_series(series=series, selectors=selectors, limit=COMPONENT_SERIES_LIMIT)
    _check_window(resolved_frame=resolved_frame, series=series, window=window)
    return series


def _no_offer() -> str:
    return ""


def check_resolved_series(
    *,
    series: Sequence[ResolvedSeries],
    selectors: Sequence[Mapping[str, object]],
    limit: SeriesLimit,
    offer: Callable[[], str] = _no_offer,
) -> None:
    """Refuses `series`, resolved from `selectors`, when they number more than `limit.cap`.

    The detail names the dimensions the series vary on with their counts, a facet only when
    a selector binds it, and keeps the one dimension the selectors list as the series
    (`series_limit_exhausted`). `offer`, called only on a refusal, ends the detail."""
    if len(series) <= limit.cap:
        return
    bindings = [
        binding for selector in selectors if "id" not in selector for binding in _bindings(selector)
    ]
    bound = {str(binding["dimension"]) for binding in bindings}
    listed = sorted(
        {str(binding["dimension"]) for binding in bindings if len(_binding_values(binding)) > 1}
    )
    facets = facet_dimension_names()
    counts: dict[str, set[object]] = {}
    for one in series:
        for name, value in one.coordinates:
            # An unbound facet varies only beside another dimension, so it never counts. A facet
            # a selector binds is a series dimension like any other.
            if name not in facets or name in bound:
                counts.setdefault(name, set()).add(value)
    raise series_limit_exhausted(
        limit=limit,
        found=f"the selection resolves to {len(series)} series",
        counts={name: len(values) for name, values in counts.items()},
        kept=listed[0] if len(listed) == 1 else None,
        offer=offer(),
    )


def _group_by_coordinate(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    continuation: Continuation | None,
) -> dict[Coordinates, tuple[Row, ...]]:
    """Groups rows by their `series_keys` key under `continuation`, rendering structured
    units for hashing.

    Returned rows retain raw cells so binding preserves the original unit."""
    groups: dict[Coordinates, list[Row]] = {}
    for coordinates, row in zip(
        series_keys(columns=columns, rows=rows, continuation=continuation), rows, strict=True
    ):
        groups.setdefault(coordinates, []).append(dict(zip(columns, row, strict=True)))
    return {coordinates: tuple(group_rows) for coordinates, group_rows in groups.items()}


def _binds_axis(*, selector: Mapping[str, object], continuation: Continuation | None) -> bool:
    """Whether `selector` binds `measurementMethod` on a frame with a continuation."""
    return continuation is not None and any(
        str(binding["dimension"]) == MEASUREMENT_METHOD_COLUMN for binding in _bindings(selector)
    )


def _check_series_estimate(
    *,
    selectors: Sequence[Mapping[str, object]],
    coordinate_columns: Sequence[str],
    facet_names: frozenset[str],
    dimension_by_name: Mapping[str, DimensionView],
    series_limit: SeriesLimit,
) -> None:
    """Estimates the series upper bound from selector shapes and dimension counts before
    grouping rows. Bound dimensions use their requested value count. Unbound facets count
    1, because a facet that splits a series is refused separately. Other unbound
    dimensions use their frame-wide count (minimum 1).

    A refusal names `series_limit` and the dimensions that multiply the largest selector's
    estimate past one, each with the count it contributes, and says what brings the select
    within the limit (`series_limit_exhausted`)."""
    estimate = 0
    widest: dict[str, int] = {}
    widest_listed: list[str] = []
    for selector in selectors:
        if "id" in selector:
            estimate += 1
            continue
        bound = {
            str(binding["dimension"]): _binding_values(binding) for binding in _bindings(selector)
        }
        factors: dict[str, int] = {}
        for column in coordinate_columns:
            if column in bound:
                factors[column] = len(bound[column])
            elif column not in facet_names:
                dimension = dimension_by_name.get(column)
                factors[column] = max(dimension.total_count, 1) if dimension is not None else 1
        factor = math.prod(factors.values())
        estimate += factor
        if factor > math.prod(widest.values()):
            widest = factors
            widest_listed = [name for name, values in bound.items() if len(values) > 1]
    if estimate <= MAX_SERIES_PER_COMPONENT:
        return
    raise series_limit_exhausted(
        limit=series_limit,
        found=f"selectors expand to an estimated {estimate} series",
        counts=widest,
        kept=widest_listed[0] if len(widest_listed) == 1 else None,
    )


def multiplying_advice(counts: Mapping[str, int], *, listed: int, kept: str | None) -> str:
    """What brings a select within `listed` series, given each dimension's count of series
    values (`counts`, those above one multiplying). `kept` is the dimension the select
    lists, which stays the series dimension. Every other multiplying dimension is bound to
    one value, and `kept` to at most `listed` values when it holds more. With no list, two or
    more multiplying dimensions are told to bind all but one, and a single one is the series
    dimension. With none, as for selectors naming series by id, fewer series are named.
    Dimensions are named with their counts, fewest first."""
    multiplying = sorted(
        ((name, count) for name, count in counts.items() if count > 1),
        key=lambda factor: (factor[1], factor[0]),
    )
    named = {name: f"{name} ({count} values)" for name, count in multiplying}
    if not named:
        return f"name at most {listed} series"
    if kept in named:
        others = [text for name, text in named.items() if name != kept]
        if not others:
            return f"{named[kept]} is the series dimension; bind it to at most {listed} values"
        trim = f", and list at most {listed} of its values" if counts[kept] > listed else ""
        return f"bind {_and(others)} to one value, keeping {named[kept]} as the series{trim}"
    if len(named) == 1:
        (only,) = named.values()
        return f"{only} is the series dimension; bind it to at most {listed} values"
    return (
        f"{_and(list(named.values()))} multiply; bind all but one of them to one value, and "
        f"the other to at most {listed} values"
    )


def _and(items: Sequence[str]) -> str:
    """`items` as an English list: "a", "a and b", "a, b, and c"."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _resolve_by_id(
    *, selector_index: int, target_id: str, groups: Mapping[Coordinates, tuple[Row, ...]]
) -> tuple[Coordinates, tuple[Row, ...]]:
    """Matches ids minted for the full candidate set so collision-extended ids resolve correctly."""
    order = list(groups)
    ids = mint_series_ids(coordinates_list=order)
    match = next(
        (
            coordinates
            for coordinates, series_id in zip(order, ids, strict=True)
            if series_id == target_id
        ),
        None,
    )
    if match is None:
        raise BindingInvalid(
            pointer="/select",
            dimension="",
            legal_values=(),
            truncated=False,
            detail=f"no series in this frame mints id {target_id!r}",
        )
    return match, groups[match]


def _resolve_by_bindings(
    *,
    selector_index: int,
    bindings: Sequence[Mapping[str, object]],
    coordinate_columns: Sequence[str],
    facet_names: frozenset[str],
    dimension_by_name: Mapping[str, DimensionView],
    legal_values: Mapping[str, frozenset[str]],
    groups: Mapping[Coordinates, tuple[Row, ...]],
    structural_view: StructuralView,
    labels: Mapping[str, str],
) -> list[tuple[Coordinates, tuple[Row, ...]]]:
    bound_values: dict[str, tuple[str, ...]] = {}
    for _binding_index, binding in enumerate(bindings):
        dimension_name = str(binding["dimension"])
        if dimension_name not in coordinate_columns:
            raise BindingInvalid(
                pointer=f"/select/{escape_pointer_segment(dimension_name)}",
                dimension=dimension_name,
                legal_values=tuple(sorted(coordinate_columns)),
                truncated=False,
                detail=f"{dimension_name!r} is not a dimension this frame declares",
            )
        values = tuple(_binding_values(binding))
        unknown = [value for value in values if value not in legal_values[dimension_name]]
        if unknown:
            view = dimension_by_name[dimension_name]
            raise BindingInvalid(
                pointer=f"/select/{escape_pointer_segment(dimension_name)}",
                dimension=dimension_name,
                legal_values=view.values,
                truncated=view.truncated,
                detail=f"{dimension_name!r} does not hold value(s) {unknown!r}",
                labels=labels,
            )
        bound_values[dimension_name] = values

    matches = [
        (coordinates, rows)
        for coordinates, rows in groups.items()
        if all(dict(coordinates).get(dim) in values for dim, values in bound_values.items())
    ]
    # Series follow each value's position in its binding's list, bindings in select
    # order. The sort is stable, so series that tie keep frame order.
    matches.sort(
        key=lambda match: tuple(
            values.index(cast("str", dict(match[0])[dim])) for dim, values in bound_values.items()
        )
    )
    if not matches:
        raise BindingInvalid(
            pointer="/select",
            dimension="",
            legal_values=(),
            truncated=False,
            detail="this selector's bindings match no series in the frame",
        )
    _check_facet_splits(
        matches=matches,
        unbound_facets=tuple(
            column
            for column in coordinate_columns
            if column in facet_names and column not in bound_values
        ),
        structural_view=structural_view,
        labels=labels,
    )
    return matches


def _check_facet_splits(
    *,
    matches: Sequence[tuple[Coordinates, tuple[Row, ...]]],
    unbound_facets: Sequence[str],
    structural_view: StructuralView,
    labels: Mapping[str, str],
) -> None:
    """Refuses an unbound facet that splits one series.

    Matched groups are keyed on every coordinate except the unbound facets, so bound
    facets stay in the key. A facet splits a series when one key holds more than one of
    its values, a null included. The first such facet in column order is refused, with
    the bindable values that key holds as its legal values."""
    keyed = [
        (tuple(item for item in coordinates if item[0] not in unbound_facets), dict(coordinates))
        for coordinates, _rows in matches
    ]
    for column in unbound_facets:
        held: dict[Coordinates, set[object]] = {}
        for key, values in keyed:
            held.setdefault(key, set()).add(values[column])
        split = next((values for values in held.values() if len(values) > 1), None)
        if split is None:
            continue
        legal = sorted(cast("str", value) for value in split if value is not None)
        capped = tuple(legal[:PER_DIMENSION_LISTING_CAP])
        raise BindingInvalid(
            pointer=f"/select/{escape_pointer_segment(column)}",
            dimension=column,
            legal_values=capped,
            truncated=len(legal) > len(capped),
            detail=(
                f"one series exists under {len(split)} {column!r} values, so select must bind "
                f"{column!r}: bind one value to draw that series once, or list the values to "
                "draw one series for each"
            ),
            structural_view=structural_view,
            labels=labels,
        )


def _check_window(
    *,
    resolved_frame: ResolvedFrame,
    series: Sequence[ResolvedSeries],
    window: Mapping[str, object] | None,
) -> None:
    coverage = resolved_frame.structural_view.coverage
    if window is not None:
        effective: Mapping[str, object] = window
    elif coverage is not None:
        effective = {"start": coverage[0], "end": coverage[1]}
    else:
        return
    start, end = str(effective["start"]), str(effective["end"])
    if any(any(start <= str(row[TIME_PERIOD_COLUMN]) <= end for row in one.rows) for one in series):
        return
    raise BindingInvalid(
        pointer="/window",
        dimension="",
        legal_values=(),
        truncated=False,
        detail=f"window {start}..{end} holds no observation for any selected series",
    )
