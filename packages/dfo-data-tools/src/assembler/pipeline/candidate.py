"""Validates intent, resolves frames and series, then builds candidates for writes.

`note_narration` skips frame resolution. `source_card` reads frame metadata
without selecting series. A frame whose rows share one span has one period, so
a line chart over it refuses `binding_invalid` at `/type` before its window is
read, and a bar chart whose select resolves to one series refuses the same way
after resolution. Both read the selected series and name the kinds that show a
span: a table, a stat card, and an entity-mode bar chart with a select that
resolves several series, which a frame with nothing to compare leaves out.

A candidate built with a `DescribedRevision` carries the card revision's description when its
frame's version holds a description record (`assembler.mint.with_card_description`).

An intent's relative periods are resolved on the frame by `assembler.rolling` before the
window is completed, and the component is built from the resolved periods. The candidate's
`stored_intent` keeps the relative form. A bar chart that compares entities up to a relative
`window.end` draws the series holding a value at that period and leaves the others out. The
series a rolling component leaves out are named in one `coverage_gap` warning, first in
`provenance.warnings`, or on a card with a description in its first caveat.

A render of a stored revision binds its select on the target frame by
`assembler.rolling.StoredSelect`'s rules, whatever its intent states. On a frame with an
anchor the warning names the anchor's left-out series the select admits, then the select
values the frame holds no row for. Without an anchor or a relative period it names those
values at the window's end. Any component names only the anchor's left-out series its
select admits (`assembler.rolling.admitted_left_out`)."""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import jsonschema
from dfo_frame_grammar import UNIT_COLUMN

from assembler.bind import BoundSeries, bind_all, count_periods, frame_periods
from assembler.catalogue import CatalogueRelease
from assembler.errors import (
    BindingInvalid,
    ConstraintViolation,
    ReleaseMismatch,
    ResourceExhausted,
    SchemaInvalid,
    pointer_from_path,
)
from assembler.frame_view import dimension_columns, facet_dimension_names, hashable_cell
from assembler.limits import (
    MAX_ENVELOPE_BYTES,
    MAX_POINTS_PER_SERIES,
    MAX_SERIES_PER_COMPONENT,
    MAX_TABLE_ROWS,
)
from assembler.merge import merge_intent
from assembler.mint import (
    DescribedRevision,
    MintedComponent,
    complete_window,
    default_window,
    mint_component,
    with_card_description,
)
from assembler.mint.provenance import LeftOut, _with_left_out_warning
from assembler.ports import FrameRef, FrameStore, NamedSeries, ResolvedFrame, RevisionRecord
from assembler.resolve import ResolvedSeries, resolve_selectors
from assembler.rolling import (
    ROLLING_ANCHOR_UNHELD,
    ResolvedRolling,
    RollingPeriodInvalid,
    StoredSelect,
    _without_absent_values,
    admitted_left_out,
    resolve_rolling,
)
from assembler.select import (
    check_select_nonempty,
    check_single_arrayed_dimension,
    has_arrayed_dimension,
    normalize_select,
)

# The component types that show a span frame's one period, with the label each offers.
_SPAN_COMPONENT_LABELS: Mapping[str, str] = {
    "bar_chart": "bar_chart listing the entities to compare in select",
    "table": "table",
    "stat_card": "stat_card",
}


@dataclass(frozen=True)
class _Candidate:
    """Validated write candidate, built before the transaction and rebuilt against its current
    revision if it moved."""

    stored_intent: Mapping[str, object]
    member: str
    resolved_frame: ResolvedFrame | None
    series: tuple[ResolvedSeries, ...]
    bound_series: tuple[BoundSeries, ...]
    minted: MintedComponent


def _build_candidate(
    *,
    patch: Mapping[str, object],
    current: RevisionRecord | None,
    is_create: bool,
    release: CatalogueRelease,
    frame_store: FrameStore,
    clock: Callable[[], datetime.datetime],
    resolved_frame: ResolvedFrame | None = None,
    load_stored_frame: Callable[[], ResolvedFrame | None] | None = None,
    described: DescribedRevision | None = None,
) -> _Candidate:
    """`load_stored_frame` is set for a render of `current` on `resolved_frame`: it returns
    the frame `current` was written on, or `None` when that frame cannot be loaded, and the
    select binds by `StoredSelect`'s rules. `described` names the revision the candidate's
    description describes; without it the candidate carries none."""
    if is_create:
        member = _resolve_type(intent=patch, release=release)
        _validate_intent(
            schema=release.intent_schemas_tolerant[member],
            intent=patch,
            release=release,
            member=member,
        )
        merged = dict(patch)
    else:
        assert current is not None  # guaranteed by write_component before this is called
        member = cast(str, current.stored_intent["type"])
        _validate_intent(
            schema=release.intent_schemas_tolerant[member],
            intent=patch,
            release=release,
            member=member,
        )
        merged = merge_intent(current=current.stored_intent, patch=patch)

    _validate_intent(
        schema=release.intent_schemas[member], intent=merged, release=release, member=member
    )

    if member == "note_narration":
        # The write-side grammar names only {type, text} for this member --
        # no frame_ref -- so no frame resolution, select/window gate, or
        # series binding ever runs for it.
        return _finish_candidate(
            member=member,
            merged_intent=merged,
            resolved_frame=None,
            series=(),
            bound_series=(),
            release=release,
            clock=clock,
            described=described,
        )

    if resolved_frame is None:
        frame_ref = FrameRef(**cast("Mapping[str, str]", merged["frame_ref"]))
        # An `ops_`/`ocv_` pair with no version row is minted but unpublished.
        # `LiveFrameStore` raises `FrameUnavailable(reason="ops_unsupported")`.
        resolved_frame = frame_store.load(frame_ref=frame_ref)

    if member == "source_card":
        # The write-side grammar names only {type, frame_ref} for this
        # member -- no select -- so it skips the select/window gates and
        # series resolution: a metadata-only read of the resolved frame's
        # own provenance.
        return _finish_candidate(
            member=member,
            merged_intent=merged,
            resolved_frame=resolved_frame,
            series=(),
            bound_series=(),
            release=release,
            clock=clock,
            described=described,
        )

    select = cast("Mapping[str, object]", merged["select"])
    check_select_nonempty(select=select)
    if member in ("line_chart", "bar_chart"):
        # Tables use the cross-product of every arrayed dimension.
        check_single_arrayed_dimension(select=select)
    selectors = normalize_select(select=select)
    if member == "line_chart" and resolved_frame.structural_view.span is not None:
        # The refusal reads the selected series, before the window, which a span frame's
        # line chart never reaches.
        _refuse_span_frame(
            resolved_frame=resolved_frame,
            drawn="line_chart",
            select=select,
            series=resolve_selectors(
                resolved_frame=resolved_frame, selectors=selectors, window=None
            ),
        )
    # The component is drawn from `drawn`, the merged intent with its relative periods
    # resolved on this frame. `merged` is stored as written.
    stored_select = (
        StoredSelect(load_frame=load_stored_frame) if load_stored_frame is not None else None
    )
    rolling = resolve_rolling(
        intent=merged,
        resolved_frame=resolved_frame,
        selectors=selectors,
        stored_select=stored_select,
    )
    drawn: Mapping[str, object] = merged
    # The series a render names when its intent states no relative period.
    rendered_names: tuple[NamedSeries, ...] = ()
    if rolling is not None:
        drawn, selectors = rolling.intent, list(rolling.selectors)
    elif stored_select is not None:
        selectors, absent = _without_absent_values(
            resolved_frame=resolved_frame,
            selectors=selectors,
            select=select,
            stored_select=stored_select,
        )
        anchor = resolved_frame.anchor
        left_out_names = (
            admitted_left_out(anchor=anchor, select=select) if anchor is not None else ()
        )
        rendered_names = _one_per_label((*left_out_names, *absent))
    window = cast("Mapping[str, object] | None", drawn.get("window"))
    if window is not None:
        window = complete_window(window=window, resolved_frame=resolved_frame)
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=selectors, window=window)
    if member == "bar_chart" and not has_arrayed_dimension(select=select) and len(series) == 1:
        # Several series draw one bar each by default (`assembler.mint.builders`); one
        # series would draw its periods.
        _refuse_span_frame(
            resolved_frame=resolved_frame,
            drawn="period-mode bar_chart",
            select=select,
            series=series,
        )

    if member == "stat_card" and len(series) != 1:
        raise BindingInvalid(
            pointer="/select",
            dimension="",
            legal_values=(),
            truncated=False,
            detail=f"a stat_card names exactly one series; this selection resolves to {len(series)}",
        )

    effective_window = window or default_window(resolved_frame=resolved_frame)
    grain = resolved_frame.structural_view.period_grain
    observed_periods = (
        frame_periods(resolved_frame=resolved_frame) if grain == "day" else frozenset()
    )
    try:
        period_count = count_periods(
            start=str(effective_window["start"]),
            end=str(effective_window["end"]),
            grain=grain,
            observed=observed_periods,
            span=resolved_frame.structural_view.span,
        )
    except ValueError as exc:
        # A schema-valid window at the wrong grain, or one that leaves out a span
        # frame's span, can pass lexical resolution and then fail `count_periods`.
        # Convert that `ValueError` to a binding refusal. Matching day-grain
        # windows clamp invalid dates.
        raise BindingInvalid(
            pointer="/window",
            dimension="",
            legal_values=(),
            truncated=False,
            detail=f"window start/end cannot bind this frame's {grain!r} periods: {exc}",
        ) from exc
    if period_count > MAX_POINTS_PER_SERIES:
        raise ResourceExhausted(
            pointer="/window",
            limit_name="MAX_POINTS_PER_SERIES",
            detail=(
                f"window spans {period_count} periods at this frame's grain, over the "
                f"maximum of {MAX_POINTS_PER_SERIES} points per series"
            ),
        )
    if member == "table":
        # `bind.py` emits exactly one row per (resolved coordinate, period
        # in the window) -- table row count is the exact product,
        # known before a single row is built.
        row_count = len(series) * period_count
        if row_count > MAX_TABLE_ROWS:
            raise ResourceExhausted(
                pointer="/select",
                limit_name="MAX_TABLE_ROWS",
                detail=(
                    f"selection resolves to {len(series)} coordinates across {period_count} "
                    f"periods, {row_count} rows, over the maximum of {MAX_TABLE_ROWS}"
                ),
            )
    bound = bind_all(resolved_frame=resolved_frame, series=series, window=effective_window)

    left_out: LeftOut | None = None
    if rolling is not None:
        series, bound, left_out = _leave_out(
            rolling=rolling,
            series=series,
            bound_series=bound,
            entity_period=(
                str(effective_window["end"])
                if member == "bar_chart"
                and rolling.window_end_relative
                and (has_arrayed_dimension(select=select) or len(series) > 1)
                else None
            ),
        )
    elif rendered_names:
        anchor = resolved_frame.anchor
        left_out = LeftOut(
            period=anchor.period if anchor is not None else str(effective_window["end"]),
            series=rendered_names,
        )

    return _finish_candidate(
        member=member,
        merged_intent=drawn,
        stored_intent=merged,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=clock,
        left_out=left_out,
        described=described,
    )


def _leave_out(
    *,
    rolling: ResolvedRolling,
    series: tuple[ResolvedSeries, ...],
    bound_series: tuple[BoundSeries, ...],
    entity_period: str | None,
) -> tuple[tuple[ResolvedSeries, ...], tuple[BoundSeries, ...], LeftOut | None]:
    """The series a rolling component draws, and the ones it leaves out.

    `entity_period` is the period an entity-mode bar chart with a relative `window.end`
    draws its bars at, and `None` for any other component. Such a chart keeps the series
    with a value at that period, so the latest period they share is that period, and the
    rest are named. Any other component keeps every series and names those with no row at
    the anchor. Select values the frame holds no row for are named in both cases."""
    period = entity_period or rolling.anchor
    names = list(rolling.left_out) if period == rolling.anchor else []
    if entity_period is not None:
        holds = [
            any(point["period"] == entity_period and "value" in point for point in one.points)
            for one in bound_series
        ]
        names.extend(
            NamedSeries(
                label=bound.label,
                coordinates={
                    name: value for name, value in resolved.coordinates if isinstance(value, str)
                },
            )
            for resolved, bound, held in zip(series, bound_series, holds, strict=True)
            if not held
        )
        series = tuple(one for one, held in zip(series, holds, strict=True) if held)
        bound_series = tuple(one for one, held in zip(bound_series, holds, strict=True) if held)
        if not bound_series:
            raise RollingPeriodInvalid(
                code=ROLLING_ANCHOR_UNHELD,
                pointer="/window",
                detail=f"no selected series holds a value at {entity_period}, the window's end",
            )
    names.extend(rolling.absent)
    unique = _one_per_label(names)
    return series, bound_series, LeftOut(period=period, series=unique) if unique else None


def _one_per_label(named: Sequence[NamedSeries]) -> tuple[NamedSeries, ...]:
    """`named` with the first series of each display label, in order."""
    first: dict[str, NamedSeries] = {}
    for one in named:
        first.setdefault(one.label, one)
    return tuple(first.values())


def _refuse_span_frame(
    *,
    resolved_frame: ResolvedFrame,
    drawn: str,
    select: Mapping[str, object],
    series: Sequence[ResolvedSeries],
) -> None:
    """Refuses `binding_invalid` when `drawn`, a kind that plots periods along an axis,
    binds a frame whose rows share one span.

    The text names a bar chart only as a call that draws bars. When the selected `series`
    vary on exactly one unselected dimension other than a facet, a bar chart with the same
    select draws one bar per value of it. When select resolves to one series, the text
    names the select that lists the values of one dimension the frame holds with the other
    bindings (`_comparable_values`). With nothing to compare, it offers a table and a stat
    card alone."""
    span = resolved_frame.structural_view.span
    if span is None:
        return
    if len(series) > 1:
        varying = _varying_dimensions(series=series, select=select)
        bar_chart = (
            f"a bar_chart, which with this select draws one bar per {varying[0]!r} value"
            if len(varying) == 1
            else "a bar_chart that lists the entities to compare in select"
        )
    elif (
        comparable := _comparable_values(
            resolved_frame=resolved_frame, select=select, one=series[0]
        )
    ) is not None:
        dimension, values = comparable
        listed = json.dumps({**select, dimension: list(values)}, ensure_ascii=False)
        bar_chart = f"a bar_chart that lists the {dimension!r} values to compare, select {listed}"
    else:
        bar_chart = None
    labels = (
        _SPAN_COMPONENT_LABELS
        if bar_chart is not None
        else {name: label for name, label in _SPAN_COMPONENT_LABELS.items() if name != "bar_chart"}
    )
    shown = f"{bar_chart}, a table, or a stat_card" if bar_chart else "a table or a stat_card"
    raise BindingInvalid(
        pointer="/type",
        dimension="type",
        legal_values=tuple(labels),
        truncated=False,
        detail=(
            f"every row of this frame covers the one period {span!r}, and a {drawn} needs "
            f"a run of periods to draw; show it as {shown}"
        ),
        labels=labels,
    )


def _varying_dimensions(
    *, series: Sequence[ResolvedSeries], select: Mapping[str, object]
) -> tuple[str, ...]:
    """Unselected dimensions other than a facet whose value differs across `series`, in
    coordinate order."""
    facets = facet_dimension_names()
    values_by_name: dict[str, set[object]] = {}
    for one in series:
        for name, value in one.coordinates:
            values_by_name.setdefault(name, set()).add(value)
    return tuple(
        name
        for name, values in values_by_name.items()
        if name not in select and name not in facets and len(values) > 1
    )


def _comparable_values(
    *, resolved_frame: ResolvedFrame, select: Mapping[str, object], one: ResolvedSeries
) -> tuple[str, tuple[str, ...]] | None:
    """The first dimension, in frame order and other than a facet, with two or more values
    that each hold exactly one series alongside the values `one` holds on the other
    dimensions select binds, with those values up to `MAX_SERIES_PER_COMPONENT`. Select
    with that dimension listing them resolves one series per value, so a bar chart draws one
    bar each. A dimension is a candidate only when select binds no other dimension to a
    list. `None` when no dimension qualifies."""
    coordinates = dict(one.coordinates)
    arrayed = [name for name, value in select.items() if isinstance(value, list)]
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    positions = [columns.index(name) for name in dimension_columns(columns=columns)]
    facets = facet_dimension_names()
    for dimension in dimension_columns(columns=columns):
        if dimension in facets or (arrayed and arrayed != [dimension]):
            continue
        held = {
            columns.index(name): coordinates[name]
            for name in select
            if name != dimension and name in coordinates
        }
        position = columns.index(dimension)
        series_by_value: dict[str, set[tuple[object, ...]]] = {}
        for row in rows:
            if row[position] is not None and all(
                hashable_cell(row[index]) == value for index, value in held.items()
            ):
                series_by_value.setdefault(cast(str, row[position]), set()).add(
                    tuple(hashable_cell(row[index]) for index in positions)
                )
        values = tuple(value for value, keys in series_by_value.items() if len(keys) == 1)
        if len(values) > 1:
            return dimension, values[:MAX_SERIES_PER_COMPONENT]
    return None


def _finish_candidate(
    *,
    member: str,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame | None,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    release: CatalogueRelease,
    clock: Callable[[], datetime.datetime],
    stored_intent: Mapping[str, object] | None = None,
    left_out: LeftOut | None = None,
    described: DescribedRevision | None = None,
) -> _Candidate:
    """Mints, checks constraints, and enforces `MAX_ENVELOPE_BYTES` before returning a
    candidate. The size check runs before a create pins its frame and repeats
    if the transaction rebuilds the candidate.

    The component is minted from `merged_intent`. The candidate stores `stored_intent`,
    which differs from it only where `merged_intent` holds a resolved relative period, and
    defaults to it. `left_out` adds the left-out series warning to the provenance.
    `described` composes the description of the revision it names, and the left-out series
    become one of its caveats."""
    minted = mint_component(
        member=member,
        merged_intent=merged_intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound_series,
        release=release,
        clock=clock,
    )
    if left_out is not None:
        minted = dataclasses.replace(
            minted,
            provenance=_with_left_out_warning(
                provenance=minted.provenance, period=left_out.period, names=left_out.names
            ),
        )
    if described is not None:
        minted = with_card_description(
            minted=minted,
            member=member,
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
            described=described,
            left_out=left_out,
        )
    _run_constraints(
        component=minted.component,
        member=member,
        release=release,
        observed_periods=minted.observed_periods,
        series=series,
    )
    _check_envelope_size(minted=minted)

    return _Candidate(
        stored_intent=merged_intent if stored_intent is None else stored_intent,
        member=member,
        resolved_frame=resolved_frame,
        series=tuple(series),
        bound_series=tuple(bound_series),
        minted=minted,
    )


def _check_envelope_size(*, minted: MintedComponent) -> None:
    """Measures `component`, `provenance` and any `description` in the encoding Starlette's
    `JSONResponse` sends them in (UTF-8, compact separators, `ensure_ascii`
    off)."""
    measured: dict[str, object] = {"component": minted.component, "provenance": minted.provenance}
    if minted.description is not None:
        measured["description"] = minted.description
    size = len(
        json.dumps(
            measured,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if size > MAX_ENVELOPE_BYTES:
        raise ResourceExhausted(
            pointer="/",
            limit_name="MAX_ENVELOPE_BYTES",
            detail=(
                f"the minted component encodes to {size} bytes, over the maximum of "
                f"{MAX_ENVELOPE_BYTES}; narrow the select or window, or shorten its text"
            ),
        )


def _resolve_type(*, intent: Mapping[str, object], release: CatalogueRelease) -> str:
    type_ = intent.get("type")
    if (
        isinstance(type_, str)
        and type_ in release.grammar_members
        and type_ not in release.intent_schemas_tolerant
    ):
        # A grammar-valid type absent from this release signals a catalogue mismatch.
        # A client can retry the same intent against a release that serves it.
        raise ReleaseMismatch(
            catalog_version=release.catalog_version,
            detail=(
                f"{type_!r} is a component type the presentation grammar defines, but this "
                "release's catalogue does not serve it"
            ),
        )
    if not isinstance(type_, str) or type_ not in release.intent_schemas_tolerant:
        # Missing or unknown types are correctable. With no member known, report
        # the catalogue's full type union as the minimal intent.
        raise SchemaInvalid(
            pointer="/type",
            detail=(
                f"{type_!r} is not a known component type"
                if type_ is not None
                else "a create must name a type from the catalogue's closed union"
            ),
            minimal_intent={"type": tuple(sorted(release.intent_schemas_tolerant))},
        )
    return type_


def _validate_intent(
    *,
    schema: Mapping[str, object],
    intent: Mapping[str, object],
    release: CatalogueRelease,
    member: str,
) -> None:
    """Validates an intent and raises `SchemaInvalid` for the first violation.

    Missing required fields carry the member's minimal intent. Other errors
    carry its legal field names."""
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(intent), key=jsonschema.exceptions.relevance)
    if not errors:
        return
    error = errors[0]
    legal_top_level = tuple(sorted(cast("Mapping[str, object]", schema.get("properties", {}))))
    if error.validator == "required":
        missing = sorted(set(cast("Sequence[str]", error.validator_value)) - set(intent.keys()))
        pointer = f"/{missing[0]}" if missing else pointer_from_path(error.absolute_path)
        raise SchemaInvalid(
            pointer=pointer, detail=error.message, minimal_intent=release.minimal_intents[member]
        )
    if error.validator == "additionalProperties" and not error.absolute_path:
        # jsonschema reports an `additionalProperties` violation against the
        # *parent* object's path (empty at the top level). Diff keys to identify the extra field.
        extra = sorted(set(intent.keys()) - set(legal_top_level))
        pointer = f"/{extra[0]}" if extra else pointer_from_path(error.absolute_path)
        raise SchemaInvalid(pointer=pointer, detail=error.message, legal_fields=legal_top_level)
    raise SchemaInvalid(
        pointer=pointer_from_path(error.absolute_path),
        detail=error.message,
        legal_fields=legal_top_level,
    )


def _translate_pointer(*, component_pointer: str) -> str:
    """Maps component-relative constraint pointers to intent fields.

    Axis fields become `/scale` or `/range`. `/component/change/basis` maps to
    `/change/basis`, and series-wide unit failures map to `/select`."""
    for prefix in ("/component/y_axis/", "/component/value_axis/", "/component/category_axis/"):
        if component_pointer.startswith(prefix):
            return "/" + component_pointer.removeprefix(prefix)
    if component_pointer.startswith("/component/change/"):
        return "/" + component_pointer.removeprefix("/component/")
    if component_pointer == "/component/series":
        return "/select"
    return component_pointer.removeprefix("/component") or "/"


def _run_constraints(
    *,
    component: Mapping[str, object],
    member: str,
    release: CatalogueRelease,
    observed_periods: frozenset[str],
    series: Sequence[ResolvedSeries],
) -> None:
    encoding_class = release.encoding_classes.get(member, "none")
    context = {
        "component": component,
        "observed_periods": tuple(sorted(observed_periods)),
        "unit_keys": {one.id: dict(one.coordinates).get(UNIT_COLUMN) for one in series},
    }
    for spec in release.constraints:
        if encoding_class not in spec.encoding_classes and member not in spec.component_types:
            continue
        failure = spec.check(context)
        if failure is None:
            continue
        raise ConstraintViolation(
            constraint_id=spec.constraint_id,
            pointer=_translate_pointer(component_pointer=failure.pointer),
            detail=failure.detail,
            fix=failure.fix,
        )
