"""Resolves an intent's relative periods to period labels on the frame it is drawn on.

A relative period is `{anchor: "latest", offset: n}` in `window.start`, `window.end`, or an
item of a stat card's `change.basis`. `resolve_rolling` finds the period "latest" names and
returns the intent with every relative period replaced by its label, so window completion,
binding, and the builders read labels alone. The stored intent keeps the relative form, and
each write or render resolves it again on the frame it draws.

"latest" is `ResolvedFrame.anchor.period` when the frame carries an anchor. Otherwise it is
`dfo_frame_grammar.latest_anchor` over one population, the component's selected series, each
holding the periods of its rows without a status: the newest period at least half of them
hold. On a frame that keeps a selection in each period (`assembler.bind.selects_per_period`),
where a series holds no row in the periods it was not chosen in, "latest" is the newest
period any selected series holds, and no series is named as left out at it. A select value
the frame holds no row for is dropped from a rolling selection and named, where a selection
of fixed periods refuses it.

A frame anchor's left-out series are named only when the select admits them
(`admitted_left_out`). A stored revision rendered on another frame binds its select by
`StoredSelect`'s rules, whether or not its intent states a relative period.

Every refusal is a `RollingPeriodInvalid`, a `BindingInvalid` whose `code` names the cause.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from dfo_frame_grammar import (
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    FrameError,
    RollingError,
    latest_anchor,
    shift_period,
)

from assembler.bind import _label, selects_per_period
from assembler.errors import BindingInvalid
from assembler.frame_view import dimension_columns, facet_dimension_names, hashable_cell
from assembler.ports import LeftOutSeries, NamedSeries, ResolvedFrame, RollingAnchor
from assembler.resolve import resolve_selectors

__all__ = [
    "ROLLING_ANCHOR_UNHELD",
    "ROLLING_SPAN_FRAME",
    "ROLLING_WINDOW_INVERTED",
    "ROLLING_WINDOW_SIDE_OMITTED",
    "ResolvedRolling",
    "RollingPeriodInvalid",
    "StoredSelect",
    "admitted_left_out",
    "resolve_rolling",
]

ROLLING_WINDOW_SIDE_OMITTED = "rolling_window_side_omitted"
ROLLING_WINDOW_INVERTED = "rolling_window_inverted"
ROLLING_SPAN_FRAME = "rolling_span_frame"
ROLLING_ANCHOR_UNHELD = "rolling_anchor_unheld"

_WINDOW_POINTER = "/window"
_BASIS_POINTER = "/change/basis"


class RollingPeriodInvalid(BindingInvalid):
    """A relative period this frame cannot resolve, refused as `binding_invalid` at
    `/window` or `/change/basis`.

    `code` is `rolling_anchor_unresolved` or `rolling_grain_unsupported`
    (`dfo_frame_grammar.RollingError.code`), `invalid_time_period`
    (`dfo_frame_grammar.FrameError.category`), or one of `ROLLING_WINDOW_SIDE_OMITTED`,
    `ROLLING_WINDOW_INVERTED`, `ROLLING_SPAN_FRAME`, and `ROLLING_ANCHOR_UNHELD`, which
    `assembler.pipeline.candidate` raises for a bar chart none of whose series holds a value
    at its relative `window.end`."""

    def __init__(self, *, code: str, pointer: str, detail: str) -> None:
        super().__init__(
            pointer=pointer, dimension="", legal_values=(), truncated=False, detail=detail
        )
        self.code = code


@dataclass(frozen=True)
class ResolvedRolling:
    """An intent's relative periods resolved on one frame.

    `intent` is the merged intent with each relative period replaced by its label.
    `selectors` are the intent's selectors without the values in `absent`. `anchor` is the
    period "latest" names. `left_out` names the series with no row at `anchor` that the
    select admits, in the component's series order, and `absent` the select values the frame
    holds no row for, each by its dimension and value. `window_end_relative` says the intent's
    `window.end` was relative."""

    intent: Mapping[str, object]
    selectors: Sequence[Mapping[str, object]]
    anchor: str
    left_out: tuple[NamedSeries, ...]
    absent: tuple[NamedSeries, ...]
    window_end_relative: bool


@dataclass(frozen=True)
class StoredSelect:
    """How a stored revision's `select` binds a frame other than the one it was written on.

    Each binding of a dimension `D` binds the frame by the first rule that applies:

    1. The frame holds every value: the binding is kept.
    2. A list binding on a dimension other than a facet whose values held every value the
       stored frame holds for `D` binds every value the frame holds for `D`. The stored
       frame's values the frame lacks are named.
    3. Any other list binding on a dimension other than a facet drops the values the frame
       holds no row for, which are named.
    4. A scalar binding, or a binding on a facet, binds the frame's one value for `D` when it
       holds none of the binding's values and exactly one value for `D`.

    A binding no rule rebinds is kept, so `resolve_selectors` refuses it as it refuses any
    selection. `load_frame` returns the frame the revision was written on, or `None` when it
    cannot be loaded, in which case rule 2 never applies. It is called only when a binding
    misses a value."""

    load_frame: Callable[[], ResolvedFrame | None]


def resolve_rolling(
    *,
    intent: Mapping[str, object],
    resolved_frame: ResolvedFrame,
    selectors: Sequence[Mapping[str, object]],
    stored_select: StoredSelect | None = None,
) -> ResolvedRolling | None:
    """Resolve the relative periods `intent` holds on `resolved_frame`.

    Returns `None` when `intent` holds no relative period. `stored_select` is set for a
    render of a stored revision, and the select binds by its rules.

    Raises:
        RollingPeriodInvalid: a window states one side as relative and omits the other, the
            frame's rows share one span, no period is the anchor, the periods have no single
            supported grain, a day-grain period is asked to move, a period is malformed or
            moves outside the years `0000` to `9999`, or the resolved window starts after
            it ends.
    """
    window = cast("Mapping[str, object] | None", intent.get("window"))
    change = cast("Mapping[str, object] | None", intent.get("change"))
    basis = cast("Sequence[object]", change["basis"]) if change is not None else ()
    window_relative = window is not None and any(_is_relative(side) for side in window.values())
    if not window_relative and not any(_is_relative(period) for period in basis):
        return None

    pointer = _WINDOW_POINTER if window_relative else _BASIS_POINTER
    if window is not None and window_relative and not {"start", "end"} <= window.keys():
        raise RollingPeriodInvalid(
            code=ROLLING_WINDOW_SIDE_OMITTED,
            pointer=_WINDOW_POINTER,
            detail=(
                "a window with a relative side states both start and end; state the omitted "
                "side as a period or a relative period"
            ),
        )
    span = resolved_frame.structural_view.span
    if span is not None:
        raise RollingPeriodInvalid(
            code=ROLLING_SPAN_FRAME,
            pointer=pointer,
            detail=(
                f"every row of this frame covers the one period {span!r}, which has no latest "
                "period to count from; leave the relative period out"
            ),
        )

    select = cast("Mapping[str, object]", intent["select"])
    selectors, absent = _without_absent_values(
        resolved_frame=resolved_frame,
        selectors=selectors,
        select=select,
        stored_select=stored_select,
    )
    if resolved_frame.anchor is not None:
        anchor = resolved_frame.anchor.period
        left_out = admitted_left_out(anchor=resolved_frame.anchor, select=select)
    else:
        anchor, left_out = _anchor_of_selection(
            resolved_frame=resolved_frame, selectors=selectors, pointer=pointer
        )

    resolved = dict(intent)
    if window is not None and window_relative:
        start, end = (
            _resolved_period(window[side], anchor=anchor, pointer=_WINDOW_POINTER)
            for side in ("start", "end")
        )
        if start > end:
            raise RollingPeriodInvalid(
                code=ROLLING_WINDOW_INVERTED,
                pointer=_WINDOW_POINTER,
                detail=f"the window resolves to {start}..{end}, which starts after it ends",
            )
        resolved["window"] = {"start": start, "end": end}
    if change is not None:
        resolved["change"] = {
            **change,
            "basis": [
                _resolved_period(period, anchor=anchor, pointer=_BASIS_POINTER) for period in basis
            ],
        }
    return ResolvedRolling(
        intent=resolved,
        selectors=selectors,
        anchor=anchor,
        left_out=left_out,
        absent=absent,
        window_end_relative=window is not None and _is_relative(window.get("end")),
    )


def _is_relative(period: object) -> bool:
    """Whether `period` is a relative period. The intent schema admits a period as a label
    string or a `relative_period` object."""
    return isinstance(period, Mapping)


def _resolved_period(period: object, *, anchor: str, pointer: str) -> str:
    """`period` as a label: `anchor` moved by a relative period's offset, or the label an
    intent states."""
    if not _is_relative(period):
        return cast(str, period)
    offset = cast("Mapping[str, int | float]", period)["offset"]
    try:
        # The schema's `integer` admits an integral float, and `shift_period` takes an `int`.
        return shift_period(anchor, int(offset))
    except (RollingError, FrameError) as exc:
        raise _refusal(exc, pointer=pointer) from exc


def _refusal(exc: RollingError | FrameError, *, pointer: str) -> RollingPeriodInvalid:
    code = exc.code if isinstance(exc, RollingError) else exc.category
    return RollingPeriodInvalid(
        code=code, pointer=pointer, detail=f"a relative period cannot be resolved: {exc}"
    )


def _anchor_of_selection(
    *,
    resolved_frame: ResolvedFrame,
    selectors: Sequence[Mapping[str, object]],
    pointer: str,
) -> tuple[str, tuple[NamedSeries, ...]]:
    """The anchor over the selected series, and those with no row at it. A frame keeping a
    selection in each period pools the selected series into one."""
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=selectors, window=None)
    population = {
        one.id: [
            cast(str, row[TIME_PERIOD_COLUMN]) for row in one.rows if row.get(STATUS_COLUMN) is None
        ]
        for one in series
    }
    if selects_per_period(resolved_frame=resolved_frame):
        # A series without a row in a period was not chosen there, so "latest" is the newest
        # period any selected series holds, and no series is left out at it.
        population = {"selected": [period for periods in population.values() for period in periods]}
    try:
        found = latest_anchor([population])
    except (RollingError, FrameError) as exc:
        raise _refusal(exc, pointer=pointer) from exc
    named_by_id = {
        one.id: NamedSeries(
            label=_label(resolved_frame=resolved_frame, coordinates=dict(one.coordinates)),
            coordinates={name: value for name, value in one.coordinates if isinstance(value, str)},
        )
        for one in series
    }
    return found.period, tuple(named_by_id[series_id] for _population, series_id in found.left_out)


def admitted_left_out(
    *, anchor: RollingAnchor, select: Mapping[str, object]
) -> tuple[NamedSeries, ...]:
    """The series of `anchor.left_out` that `select` admits, in their order.

    An entry is left unnamed when its coordinates hold a value for a dimension `select`
    binds and the binding does not name that value."""
    return tuple(
        NamedSeries(label=one.label, coordinates=one.coordinates)
        for one in _admitted(anchor=anchor, select=select)
    )


def _admitted(*, anchor: RollingAnchor, select: Mapping[str, object]) -> tuple[LeftOutSeries, ...]:
    named = {
        name: cast("list[object]", value) if isinstance(value, list) else [value]
        for name, value in select.items()
    }
    return tuple(
        one
        for one in anchor.left_out
        if all(
            one.coordinates[name] in values
            for name, values in named.items()
            if name in one.coordinates
        )
    )


def _without_absent_values(
    *,
    resolved_frame: ResolvedFrame,
    selectors: Sequence[Mapping[str, object]],
    select: Mapping[str, object],
    stored_select: StoredSelect | None = None,
) -> tuple[list[Mapping[str, object]], tuple[NamedSeries, ...]]:
    """`selectors`, normalised from `select`, without the bound values the frame holds no
    row for, and those values the frame anchor's left-out series that `select` admits do not
    already name, each by its label, dimension and value.

    A binding keeps its values when the frame holds none of them or lacks its dimension, so
    `resolve_selectors` refuses it as it refuses any selection. With `stored_select` each
    binding binds by its rules instead."""
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    dimensions = dimension_columns(columns=columns)
    anchor = resolved_frame.anchor
    named = _admitted(anchor=anchor, select=select) if anchor is not None else ()
    absent: dict[str, str] = {}
    kept_selectors: list[Mapping[str, object]] = []
    for selector in selectors:
        bindings: list[Mapping[str, object]] = []
        for binding in cast("Sequence[Mapping[str, object]]", selector.get("bindings", ())):
            name = str(binding["dimension"])
            values = cast("Sequence[str]", binding["values"])
            if name in dimensions:
                bound, dropped = _bound_values(
                    name=name,
                    values=values,
                    held=_held_values(resolved_frame=resolved_frame, name=name),
                    scalar=not isinstance(select.get(name), list),
                    stored_select=stored_select,
                )
                for value in dropped:
                    if not any(one.coordinates.get(name) == value for one in named):
                        absent.setdefault(value, name)
                if bound != values:
                    binding = {**binding, "values": bound}
            bindings.append(binding)
        kept_selectors.append({**selector, "bindings": bindings} if bindings else selector)
    labels = resolved_frame.labels
    if absent and stored_select is not None and (stored_frame := stored_select.load_frame()):
        # A value rule 2 names may be one only the stored frame labels.
        labels = {**stored_frame.labels, **labels}
    return kept_selectors, tuple(
        NamedSeries(label=labels.get(value, value), coordinates={name: value})
        for value, name in absent.items()
    )


def _bound_values(
    *,
    name: str,
    values: Sequence[str],
    held: Mapping[object, None],
    scalar: bool,
    stored_select: StoredSelect | None,
) -> tuple[Sequence[str], list[str]]:
    """The values a binding of `name` binds on a frame holding `held` for it, and the values
    it drops. `scalar` says `select` binds `name` to one value.
    `StoredSelect` states the rules a render binds by."""
    present = [value for value in values if value in held]
    if len(present) == len(values):
        return values, []
    if stored_select is not None:
        if scalar or name in facet_dimension_names():
            follows = not present and len(held) == 1
            return (cast("list[str]", list(held)), []) if follows else (values, [])
        stored_frame = stored_select.load_frame()
        stored_held = (
            _held_values(resolved_frame=stored_frame, name=name) if stored_frame is not None else {}
        )
        if stored_held and stored_held.keys() <= set(values):
            widened = [*present, *(cast(str, value) for value in held if value not in present)]
            return widened, [cast(str, value) for value in stored_held if value not in held]
    if not present:
        return values, []
    return present, [value for value in values if value not in held]


def _held_values(*, resolved_frame: ResolvedFrame, name: str) -> dict[object, None]:
    """The distinct values other than null the frame holds for dimension `name`, as keys in
    row order, or none when the frame lacks the dimension."""
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    if name not in columns:
        return {}
    index = columns.index(name)
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    return dict.fromkeys(hashable_cell(row[index]) for row in rows if row[index] is not None)
