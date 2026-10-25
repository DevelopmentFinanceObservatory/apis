"""Ten invariants of an answer, each checked on the cells of one grid.

A cell is a route (standard, or a role route with named or discovered destinations, read per
pair or in bulk), a row set, an answer kind, a unit request, a date request, the preview on
or off, a facet pin and an order of the two named places. Each cell runs the first door and
the resubmit door over the fixture world (`_answer_invariants_world`), and each door's
region is assembled into the part the wire returns, whose Specs carry their rendered
sentences. Every check compares two things the engine computed, or one of them with the rows
the world holds, and returns one message per disagreement naming both sides:

- `i1_plan_is_the_source`: the sentence, sources, scope, facets, dates, counts and the
  `year_unavailable` departure agree with each other and with the preview, and read the
  same with the preview off.
- `i2_door_equality`: an answer and its unchanged resubmit state the same things.
- `i3_one_election`: the answer, a datasets read of its Spec and every unit decision elect
  the same source.
- `i4_dates`: `latest` names the latest year served, and an answer holds data exactly when
  the world holds a point in the years asked.
- `i5_provenance_less_dispatch`: a top facet with no provenance is served, and a read that
  binds no provenance still refuses it.
- `i6_nearest_real_is_the_relaxed_answer`: a suggestion states what the relaxed query
  answers.
- `i7_one_sentence_names_the_answer`: the pipeline streams no sentence of its own, and the
  part's one sentence names the Spec its response states.
- `i8_place_order_independence`: both orders of the named places answer and request alike.
- `i9_latency`: no door makes a request or a serial step beyond the same door at c669cc874
  where that door gave the same answer, and beyond a bound of its own where it did not.
- `i11_download_is_the_datasets_read`: the rows and years a Spec states for its file are
  the ones a datasets read of its Spec reads.
- `i10_election_warnings`: a door warns an election warning exactly when it runs the
  election: a first answer warns `family_total_absent` for exactly the Specs it holds whose
  member is not its family's total, a resubmit warns it only where it elects again among
  the walked family, and no resubmit warns `slot_defaulted`.

The grid test runs a strength-4 covering array of the cells (`GRID`,
`_grid_cells.ANSWER_INVARIANTS`), and every cell under `DFO_FULL_GRIDS=1`. `i9_latency`
holds a door to a recorded bound, so it runs on the covering array's cells only
(`checks_for`), and every other invariant runs on every cell the test runs.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from code_index.shared import get_index
from datasets_api.adapter import to_materialisation_request
from datasets_api.shape import StatVarShape
from materialise import MaterialisationRequest, RequestedWindow, materialise_preview
from materialise.errors import DefaultFacetUndefinedError, MaterialisationError
from materialise.rate_tables import RateTableCache
from search.engine.departures import spec_departure_facts
from search.engine.region.result import RegionResult
from search.models import IdentityBindingValue, Spec
from tests._covering_array import Grid, uncovered
from tests.engine import _answer_invariants_record, _grid_cells
from tests.engine import _answer_invariants_world as fixture_world
from tests.engine._answer_invariants_world import (
    ALAND,
    ALT,
    BLAND,
    DATES,
    EUR,
    FACETS,
    IDS,
    KINDS,
    PLACE_NAMES,
    PROJ,
    PURPOSE,
    ROUTES,
    ROW_SETS,
    SOURCE_LABELS,
    UNIT_TEXT,
    USD,
    Answer,
    World,
)

_CLOCK_YEAR = 2026
PREVIEWS = (True, False)
PINS = ("none", "provenance", "unit")
ORDERS = ("dcid_order", "reversed")
# The source a provenance pin names in each row set: the one a dateless election passes
# over, or the only one.
_PIN_SOURCE = {
    "overlap": ALT,
    "projection": PROJ,
    "unsourced": ALT,
    "other_unit": ALT,
    "gap": ALT,
    "one_empty": IDS,
}


@dataclass(frozen=True)
class Cell:
    route: str
    row_set: str
    kind: str
    unit: str
    date: str
    preview: bool
    pin: str
    order: str

    def __str__(self) -> str:
        return "/".join(
            [
                self.route,
                self.row_set,
                self.kind,
                f"unit_{self.unit}",
                self.date,
                "preview_on" if self.preview else "preview_off",
                f"pin_{self.pin}",
                self.order,
            ]
        )


def cannot_exist(cell: Cell) -> str | None:
    """Why no request forms `cell`, or None when one does."""
    if cell.order == "reversed" and cell.route.startswith("discovered"):
        return "a discovered route names no place, so it has one order"
    if cell.pin == "unit" and cell.unit != "none":
        return "an identity's unit axis holds a request or a ref, never both"
    return None


# The cells each declared exemption passed over, by issue. A check records a cell here when
# it drops a disagreement that the issue describes.
PASSED_OVER: dict[str, set[Cell]] = {}


def _pass_over(issue: str, cell: Cell) -> None:
    PASSED_OVER.setdefault(issue, set()).add(cell)


# ---------------------------------------------------------------------------
# What an answer states.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    """One preview series: its variable, place, source, unit and dates."""

    stat_var: str
    place: str | None
    provenance: str | None
    unit: str | None
    dates: tuple[str, ...]
    converted: bool


@dataclass(frozen=True)
class Stated:
    """Everything one Spec states, in comparable form."""

    stat_vars: tuple[str, ...]
    sentence: str
    byline: str
    year_unavailable: tuple[str | None, ...]
    unserved: tuple[str, ...]
    sources: tuple[str, ...]
    facets: tuple[tuple[str, str | None, str | None], ...]
    scope: tuple[str, ...] | None
    window: tuple[int | None, int | None] | None
    observation_count: int | None
    dimensions: tuple[tuple[str, int], ...]
    headline_year: int | None
    date_ranges: tuple[tuple[str, str | None, str | None], ...]
    preview: tuple[Series, ...] | None
    preview_omitted: str | None
    preview_truncated: bool

    def facet(self, axis: str) -> tuple[str | None, str | None]:
        """The `(ref, selection)` stated on `axis`. A missing axis states neither."""
        return next(((ref, sel) for a, ref, sel in self.facets if a == axis), (None, None))

    def dimension(self, label: str) -> int | None:
        return dict(self.dimensions).get(label)


_PREVIEW_FIELDS = frozenset({"preview", "preview_omitted", "preview_truncated"})
_STATED: dict[int, tuple[Answer, tuple[Stated, ...]]] = {}


def statements(answer: Answer) -> tuple[Stated, ...]:
    """What each Spec of `answer` states on the wire, in the order of `answer.specs`, read
    once per answer."""
    if id(answer) not in _STATED:
        read = tuple(_read_statements(spec) for spec in answer.specs)
        _STATED[id(answer)] = (answer, read)
    return _STATED[id(answer)][1]


def _stat_vars(spec: Spec) -> tuple[str, ...]:
    return tuple(sv.ref.ref for sv in spec.resolved.members)


def _read_statements(spec: Spec) -> Stated:
    """One wire Spec's statements. Its sentence is the wire `rendered_sentence`, which is
    its token plan read in order; `year_unavailable` holds the years shown by each
    `year_unavailable` departure the Spec raises
    (`engine.departures.spec_departure_facts`)."""
    resolved = spec.resolved
    plan = resolved.sentence_plan or []
    assert resolved.rendered_sentence == "".join(token.text for token in plan)
    coverage = resolved.coverage
    window = getattr(coverage, "window", None)
    observations = resolved.observations
    series = None
    if observations is not None:
        series = tuple(
            Series(
                stat_var=s.stat_var.ref,
                place=next(
                    (e.ref for e in s.entities if e.role in ("destination", "observationAbout")),
                    None,
                ),
                provenance=s.facet["provenance"].ref if "provenance" in s.facet else None,
                unit=s.facet["unit"].ref if "unit" in s.facet else None,
                dates=tuple(p.date for p in s.points),
                converted=s.conversion is not None,
            )
            for s in observations.series
        )
    return Stated(
        stat_vars=_stat_vars(spec),
        sentence="".join(token.text for token in plan),
        byline="".join(token.text for token in plan if token.segment == "byline"),
        unserved=tuple(dcid for _role, dcid in resolved.unserved_entities),
        year_unavailable=tuple(
            fact.shown
            for fact in spec_departure_facts(spec, operations=())
            if fact.code == "year_unavailable"
        ),
        sources=tuple(ref.ref for ref in resolved.trace.resolved_sources),
        facets=tuple(
            sorted((axis, facet.ref, facet.selection) for axis, facet in resolved.facets.items())
        ),
        scope=tuple(m.ref for m in resolved.scope.members) if resolved.scope else None,
        window=(window.start_year, window.end_year) if window is not None else None,
        observation_count=getattr(coverage, "observation_count", None),
        dimensions=tuple((d.label, d.count) for d in getattr(coverage, "dimensions", None) or ()),
        headline_year=resolved.headline.year if resolved.headline else None,
        date_ranges=tuple(
            (
                sv.ref.ref,
                sv.data_date_range.start if sv.data_date_range else None,
                sv.data_date_range.end if sv.data_date_range else None,
            )
            for sv in resolved.members
        ),
        preview=series,
        preview_omitted=(
            observations.omitted.reason
            if observations is not None and observations.omitted is not None
            else None
        ),
        preview_truncated=bool(observations is not None and observations.truncated),
    )


def differences(a: Stated, b: Stated, *, ignoring: frozenset[str] = frozenset()) -> list[str]:
    """Each field `a` and `b` state differently, as `field: a's value against b's`."""
    return [
        f"{f.name}: {getattr(a, f.name)!r} against {getattr(b, f.name)!r}"
        for f in fields(Stated)
        if f.name not in ignoring and getattr(a, f.name) != getattr(b, f.name)
    ]


# Warnings about the query's wording. A resubmit posts the prices the first answer defaulted,
# so it has nothing to default.
_QUERY_WARNINGS = frozenset({"default_not_declared"})

# Warnings an election raises: a slot a declared default fixed, and a member that is not its
# family's total. A first answer elects, and so does a resubmit whose member walk settles a
# family wider than the members it posts. `i10_election_warnings` holds each door to them.
# `membership_unverified` describes the identity's member walk, which every door runs, so
# the doors' comparisons hold it equal.
_ELECTION_WARNINGS = frozenset({"family_total_absent", "slot_defaulted"})


def outcome(answer: Answer) -> tuple:
    """What an answer states outside its Specs: its status, reason, window-gap fact and
    warnings, or the refusal or error it raised."""
    if answer.result is None:
        return ("refused", answer.refusal, answer.error)
    result = answer.result
    return (
        result.status,
        result.no_data_reason,
        result.window_gap_fact,
        tuple(
            sorted(
                (w.code, w.message)
                for w in result.warnings
                if w.code not in _QUERY_WARNINGS | _ELECTION_WARNINGS
            )
        ),
    )


def _status(answer: Answer) -> str:
    return answer.result.status if answer.result is not None else "refused"


# ---------------------------------------------------------------------------
# The doors of a cell.
# ---------------------------------------------------------------------------

FIRST, PINNED, RESUBMIT, WINDOWED = (
    "the first answer",
    "the pinned resubmit",
    "the unchanged resubmit",
    "the resubmit asking the same years",
)


@cache
def _world(route: str, row_set: str, kind: str, relaxed: bool = False) -> World:
    return fixture_world.world(route, row_set, kind, relaxed=relaxed)


@cache
def _first_door(
    world: World,
    order: str,
    date: str,
    preview: bool,
    unit: str,
) -> Answer:
    places = [ALAND, BLAND] if order == "dcid_order" else [BLAND, ALAND]
    return world.first(
        places,
        DATES[date],
        preview=preview,
        unit_text=UNIT_TEXT[unit],
    )


_RESUBMITS: dict[tuple, Answer] = {}


def _resubmit_door(
    world: World,
    spec: Spec,
    *,
    preview: bool,
    pin: tuple[str, str] | None = None,
    window: str | None = None,
) -> Answer:
    """`spec`'s identity posted back, with `pin` (an axis and a ref) or the window of the
    date request `window` added. Two Specs that post the same payload share one answer."""
    payload = spec.definition.model_dump_json()
    key = (world, preview, spec.spec_id, payload, _stat_vars(spec), pin, window)
    if key not in _RESUBMITS:
        _RESUBMITS[key] = world.resubmit(
            spec,
            preview=preview,
            facets={pin[0]: IdentityBindingValue(ref=pin[1])} if pin else None,
            window=_identity_window(window) if window else None,
        )
    return _RESUBMITS[key]


def _identity_window(date: str) -> dict[str, object]:
    request = DATES[date]
    if request is None:
        return {"kind": "all"}
    if request.window is None:
        return {"kind": "latest"}
    return {
        "kind": "range",
        "start_year": request.window.start_year,
        "end_year": request.window.end_year,
    }


@dataclass(frozen=True)
class Doors:
    """The answers of one cell.

    `opened` holds the answers the cell is about: the first door's, or under a pin each
    resubmit that adds the pin to one of the first door's Specs. `again` pairs each Spec an
    opened answer states with the unchanged resubmit of it. When the first door states no
    Spec, `windowed` is the resubmit that asks the dateless answer's Spec for the cell's
    years, with the cell's pin."""

    cell: Cell
    world: World
    first: Answer
    opened: tuple[Answer, ...]
    again: tuple[tuple[Answer, int, Answer], ...]
    windowed: Answer | None

    def answers(self) -> Iterator[tuple[str, Answer]]:
        """Every answer of the cell, named by its door."""
        for answer in self.opened:
            yield (FIRST if self.cell.pin == "none" else PINNED), answer
        for _opened, _index, answer in self.again:
            yield RESUBMIT, answer
        if self.windowed is not None:
            yield WINDOWED, self.windowed


def _pin(cell: Cell) -> tuple[str, str] | None:
    if cell.pin == "provenance":
        return ("provenance", _PIN_SOURCE[cell.row_set])
    if cell.pin == "unit":
        return ("unit", EUR if cell.row_set == "other_unit" else USD)
    return None


def doors(cell: Cell) -> Doors:
    """The cell's answers. A world whose member walk fails opens the first door alone: a
    resubmit elects its members by that walk, so each of its doors fails retryably
    (`test_membership_authority`)."""
    world = _world(cell.route, cell.row_set, cell.kind)
    first = _first_door(world, cell.order, cell.date, cell.preview, cell.unit)
    if not world.walk_settles:
        opened_first = (first,) if cell.pin == "none" else ()
        return Doors(
            cell=cell, world=world, first=first, opened=opened_first, again=(), windowed=None
        )
    pin = _pin(cell)
    windowed = None
    if not first.specs:
        dateless = _first_door(world, cell.order, "dateless", cell.preview, cell.unit)
        if dateless.result is not None and dateless.result.specs:
            windowed = _resubmit_door(
                world, dateless.specs[0], preview=cell.preview, pin=pin, window=cell.date
            )
        opened: tuple[Answer, ...] = (first,) if pin is None else ()
    elif pin is None:
        opened = (first,)
    else:
        opened = tuple(
            _resubmit_door(world, spec, preview=cell.preview, pin=pin) for spec in first.specs
        )
    again = tuple(
        (answer, index, _resubmit_door(world, spec, preview=cell.preview))
        for answer in opened
        for index, spec in enumerate(answer.specs)
    )
    return Doors(cell=cell, world=world, first=first, opened=opened, again=again, windowed=windowed)


# ---------------------------------------------------------------------------
# What the world holds.
# ---------------------------------------------------------------------------

type Years = tuple[int | None, int | None]
_EVERY_YEAR: Years = (None, None)


def _asked(date: str) -> Years | None:
    """The years a date request asks for: every year for a query naming no date, None for
    `latest`."""
    request = DATES[date]
    if request is None:
        return _EVERY_YEAR
    window = request.window
    return None if window is None else (window.start_year, window.end_year)


def _read(date: str) -> Years | None:
    """The years a date request reads: an open end stops at the clock, and a query naming
    no date reads every year."""
    asked = _asked(date)
    if asked is None or asked == _EVERY_YEAR:
        return asked
    return (asked[0], asked[1] if asked[1] is not None else _CLOCK_YEAR)


def _within(year: int, years: Years | None) -> bool:
    return years is None or (
        (years[0] is None or year >= years[0]) and (years[1] is None or year <= years[1])
    )


def _held_years(
    world: World, stat_var: str, *, provenance: object = any, unit: object = any
) -> dict[str, set[int]]:
    """Each place's years of `stat_var`, on the facets of one source or unit when named."""
    held: dict[str, set[int]] = {}
    for row in world.rows:
        facet_provenance, facet_unit = FACETS[row.facet_id]
        if row.stat_var != stat_var:
            continue
        if provenance is not any and facet_provenance != provenance:
            continue
        if unit is not any and facet_unit != unit:
            continue
        held.setdefault(row.place, set()).update(row.years)
    return held


def _holds_points(world: World, stat_var: str, years: Years | None, **facets: object) -> bool:
    return any(
        _within(year, years)
        for held in _held_years(world, stat_var, **facets).values()
        for year in held
    )


def _admitted(cell: Cell) -> dict[str, object]:
    """The facets a cell's native unit request and pin admit, as `_held_years` filters."""
    admitted: dict[str, object] = {}
    if cell.unit == "native":
        admitted["unit"] = USD
    if (pin := _pin(cell)) is not None:
        admitted[pin[0]] = pin[1]
    return admitted


# ---------------------------------------------------------------------------
# I1. The plan is the source.
# ---------------------------------------------------------------------------

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_SCOPE_COUNT = re.compile(r"across the (\d+) (?:recipient|provider)s? with data")
_SCOPE_IN_SENTENCE = re.compile(r"across (\d+) (?:recipients|providers)")


def _main_clause(sentence: str) -> str:
    return sentence.split(". ")[0]


def _sources_named(byline: str) -> tuple[str, ...]:
    """The sources a sentence's byline names, in the order it names them."""
    found = [
        (match.start(), dcid)
        for dcid, label in SOURCE_LABELS.items()
        if (match := re.search(rf"\b{re.escape(label)}\b", byline))
    ]
    return tuple(dcid for _position, dcid in sorted(found))


def _asked_text(asked: Years) -> str:
    first, last = asked
    if first is not None and last is not None:
        return str(first) if first == last else f"{first} to {last}"
    if first is None and last is None:
        return "every year"
    return f"{first} or later" if first is not None else f"{last} or earlier"


def _sources_agree(spec: Stated) -> Iterator[str]:
    if _sources_named(spec.byline) != spec.sources:
        yield (
            f"the sentence names sources {_sources_named(spec.byline)} and "
            f"resolved_sources lists {spec.sources}: {spec.sentence!r}"
        )
    if (count := spec.dimension("sources")) is not None and count != len(spec.sources):
        yield f"coverage counts {count} sources and resolved_sources lists {spec.sources}"
    provenance, selection = spec.facet("provenance")
    if selection == "per_series":
        if len(spec.sources) < 2:
            yield f"resolved.facets states a source per series and resolved_sources {spec.sources}"
    elif spec.sources != ((provenance,) if provenance is not None else ()):
        yield f"resolved.facets states source {provenance} and resolved_sources {spec.sources}"


def _sentence_years(spec: Stated) -> list[int]:
    return sorted({int(year) for year in _YEAR.findall(_main_clause(spec.sentence))})


def _dates_agree(spec: Stated) -> Iterator[str]:
    """The years the sentence names against coverage's window, the span of the variables'
    date ranges and the headline. An answer to a query naming no date states no window, and
    its sentence dates the span its variables hold."""
    years = _sentence_years(spec)
    spans = {
        (int(start[:4]), int(end[:4])) for _sv, start, end in spec.date_ranges if start and end
    }
    if spec.window is None:
        held = {year for span in spans for year in span}
        if spans and years != sorted({min(held), max(held)}):
            yield f"the sentence dates {years} and data_date_range spans {sorted(spans)}"
        return
    if years != sorted({year for year in spec.window if year is not None}):
        yield (
            f"the sentence dates {years} and coverage's window is {spec.window}: {spec.sentence!r}"
        )
    if spans and (min(s for s, _e in spans), max(e for _s, e in spans)) != spec.window:
        yield f"data_date_range spans {sorted(spans)} and coverage's window is {spec.window}"
    if spec.headline_year is not None and spec.headline_year != spec.window[1]:
        yield f"the headline names {spec.headline_year} and coverage's window is {spec.window}"


def _scope_agrees(spec: Stated) -> Iterator[str]:
    if spec.scope is None:
        return
    for label in ("recipients", "providers"):
        if (count := spec.dimension(label)) is not None and count != len(spec.scope):
            yield f"coverage counts {count} {label} and resolved.scope lists {spec.scope}"
    match = _SCOPE_IN_SENTENCE.search(_main_clause(spec.sentence))
    if match is not None and int(match.group(1)) != len(spec.scope):
        yield f"the sentence counts {match.group(1)} and resolved.scope lists {spec.scope}"


def _year_departure_agrees(cell: Cell, spec: Stated) -> Iterator[str]:
    """Each `year_unavailable` departure's years shown against coverage's window, and a
    clipped closed range against whether the Spec raises one. The years a no-data answer's
    series holds are checked against the world by `_window_gap_fact_agrees`."""
    asked = _asked(cell.date)
    for shown in spec.year_unavailable:
        if shown is not None and spec.window is not None and None not in spec.window:
            start, end = spec.window
            served = str(start) if start == end else f"{start} to {end}"
            if shown != served:
                yield f"the departure shows {shown!r} and coverage's window is {spec.window}"
    first, last = asked or (None, None)
    served_first, served_last = spec.window or (None, None)
    if (
        first is not None
        and last is not None
        and served_first is not None
        and served_last is not None
    ):
        clipped = first < served_first or last > served_last
        if clipped != bool(spec.year_unavailable):
            yield (
                f"the years asked are {asked}, coverage's window is {spec.window}, and "
                f"year_unavailable is {'raised' if spec.year_unavailable else 'not raised'}"
            )


def _preview_agrees(cell: Cell, world: World, spec: Stated) -> Iterator[str]:
    """The statements of one Spec against the series its preview serves, and those series
    against the rows the world holds."""
    if spec.preview is None or spec.preview_omitted is not None or spec.preview_truncated:
        return
    served = [s for s in spec.preview if s.dates]
    sources = {s.provenance for s in served if s.provenance is not None}
    if sources != set(spec.sources):
        yield (
            f"the preview serves sources {sorted(sources)} and resolved_sources lists "
            f"{spec.sources}"
        )
    for axis in ("provenance", "unit"):
        refs = {getattr(s, axis) for s in served}
        ref, selection = spec.facet(axis)
        if (len(refs) == 1 and (ref, selection) != (next(iter(refs)), None)) or (
            len(refs) > 1 and selection != "per_series"
        ):
            yield (
                f"the preview serves {axis} {sorted(refs, key=str)} and resolved.facets "
                f"states {(ref, selection)}"
            )
    points = [(s, date) for s in served for date in s.dates]
    years = sorted({int(date[:4]) for _s, date in points})
    if years and spec.window is None and _asked(cell.date) == _EVERY_YEAR:
        # A query naming no date states no window: its sentence dates what it serves.
        if _sentence_years(spec) != sorted({years[0], years[-1]}):
            yield (
                f"the preview serves {years[0]} to {years[-1]} and the sentence dates "
                f"{_sentence_years(spec)}: {spec.sentence!r}"
            )
    elif years and spec.window != (years[0], years[-1]):
        yield (
            f"the preview serves {years[0]} to {years[-1]} and coverage's window is {spec.window}"
        )
    if spec.observation_count is not None and spec.observation_count != len(points):
        yield (
            f"coverage counts {spec.observation_count} observations and the preview serves "
            f"{len(points)}"
        )
    for stat_var, start, end in spec.date_ranges:
        dates = sorted(date for s, date in points if s.stat_var == stat_var)
        if dates and (start, end) != (dates[0], dates[-1]):
            yield (
                f"data_date_range of {stat_var} is {(start, end)} and the preview serves "
                f"{dates[0]} to {dates[-1]}"
            )
    if spec.scope is not None and {s.place for s in served} != set(spec.scope):
        yield (
            f"the preview serves {sorted(str(s.place) for s in served)} and resolved.scope "
            f"lists {spec.scope}"
        )
    read = _read(cell.date)
    for series in spec.preview:
        unit = {} if series.converted else {"unit": series.unit}
        held = _held_years(world, series.stat_var, provenance=series.provenance, **unit)
        in_years = {year for year in held.get(series.place or "", set()) if _within(year, read)}
        served_years = {int(date[:4]) for date in series.dates}
        # `latest` serves one point of a series, a range every point in its years.
        if not served_years <= in_years or (read is not None and served_years != in_years):
            yield (
                f"the preview serves {sorted(served_years)} of {series.provenance} at "
                f"{series.place} and the world holds {sorted(in_years)} there in the years read"
            )


def _spans(world: World, stat_vars: Iterable[str], **facets: object) -> set[tuple[int, int]]:
    """The span each facet of `stat_vars` runs across the places, on the facets of one
    source or unit when named."""
    spans = set()
    for stat_var, facet_id in itertools.product(stat_vars, FACETS):
        provenance, unit = FACETS[facet_id]
        if facets.get("provenance", provenance) != provenance or facets.get("unit", unit) != unit:
            continue
        years = {
            year
            for row in world.rows
            if (row.stat_var, row.facet_id) == (stat_var, facet_id)
            for year in row.years
        }
        if years:
            spans.add((min(years), max(years)))
    return spans


def _window_gap_fact_agrees(cell: Cell, world: World, door: str, answer: Answer) -> Iterator[str]:
    """A no-data answer's `year_unavailable` fact (`RegionResult.window_gap_fact`) against
    the years asked and the spans the world's series run. A resubmit that finds no
    observation in the years asked, of a variable whose admitted facets hold rows in other
    years, carries the fact."""
    result = answer.result
    asked = _asked(cell.date)
    if result is None or result.status != "no_data":
        return
    fact = result.window_gap_fact
    if fact is None:
        elected = [ref.ref for ref in result.elected_stat_vars or ()]
        spans = _spans(world, elected, **(_admitted(cell) if cell.unit != "converting" else {}))
        if door != FIRST and result.no_data_reason == "no_observations" and asked and spans:
            yield (
                f"the answer finds no observation in {_asked_text(asked)}, carries no "
                f"year_unavailable fact, and the world's admitted series run {sorted(spans)}"
            )
        return
    if asked is None:
        yield f"the fact names {fact.asked!r} and the date request is {cell.date}"
        return
    if fact.asked != _asked_text(asked):
        yield f"the fact names {fact.asked!r} and the years asked are {_asked_text(asked)!r}"
    if fact.data_years is None:
        return
    if fact.data_years not in (spans := _spans(world, world.stat_vars)):
        yield (
            f"the fact says the series runs {fact.data_years} and the world's series run "
            f"{sorted(spans)}"
        )


def _scope_warning_agrees(answer: Answer) -> Iterator[str]:
    """A scope count in a response's warning against the scope each of its Specs lists. No
    warning counts an empty scope: a response that lists no member resolved across none."""
    if answer.result is None:
        return
    scopes = {len(spec.scope) for spec in statements(answer) if spec.scope}
    for warning in answer.result.warnings:
        match = _SCOPE_COUNT.search(warning.message)
        if warning.code != "role_scope_defaulted" or match is None:
            continue
        if int(match.group(1)) == 0:
            yield f"the warning states a scope of no member: {warning.message!r}"
        elif scopes and scopes != {int(match.group(1))}:
            yield (
                f"the warning counts {match.group(1)} with data and the Specs' scopes list "
                f"{sorted(scopes)} members: {warning.message!r}"
            )


def i1_plan_is_the_source(run: Doors) -> list[str]:
    """Every statement an answer makes agrees with its other statements, with what its
    preview serves and with the rows the world holds, and the same answer with the preview
    off states the same."""
    cell, world = run.cell, run.world
    found: list[str] = []
    for door, answer in run.answers():
        if answer.error is not None:
            found.append(f"{door} raised {answer.error}")
        for spec in statements(answer):
            for check in (_sources_agree(spec), _dates_agree(spec), _scope_agrees(spec)):
                found += [f"in {door}, {message}" for message in check]
            found += [f"in {door}, {message}" for message in _year_departure_agrees(cell, spec)]
            found += [f"in {door}, {message}" for message in _preview_agrees(cell, world, spec)]
        found += [
            f"in {door}, {message}"
            for message in _window_gap_fact_agrees(cell, world, door, answer)
        ]
        found += [f"in {door}, {message}" for message in _scope_warning_agrees(answer)]
    if cell.preview:
        return found
    with_preview = doors(replace(cell, preview=True))
    for (door, off), (_door, on) in zip(run.answers(), with_preview.answers(), strict=False):
        if outcome(off)[:3] != outcome(on)[:3] or len(off.specs) != len(on.specs):
            found.append(
                f"{door} with the preview off is {outcome(off)[:3]} with {len(off.specs)} "
                f"Specs, and with it on {outcome(on)[:3]} with {len(on.specs)}"
            )
            continue
        for spec_off, spec_on in zip(statements(off), statements(on), strict=True):
            found += [
                f"{door} with the preview off against the preview on states {difference}"
                for difference in differences(spec_off, spec_on, ignoring=_PREVIEW_FIELDS)
            ]
    return found


# ---------------------------------------------------------------------------
# I2. Door equality.
# ---------------------------------------------------------------------------


def i2_door_equality(run: Doors) -> list[str]:
    """An answer and the unchanged resubmit of each Spec it states say the same things; an
    answer with no Spec and the resubmit asking the dateless Spec for the same years end
    alike, warnings included."""
    found: list[str] = []
    door = FIRST if run.cell.pin == "none" else PINNED
    for opened, index, again in run.again:
        if len(again.specs) != 1:
            found.append(
                f"{door} states Spec {index} and its unchanged resubmit is {outcome(again)}"
            )
            continue
        first, second = statements(opened)[index], statements(again)[0]
        # A candidates response carries no preview on its Specs.
        ignoring = _PREVIEW_FIELDS if first.preview is None else frozenset()
        found += [
            f"{door} against its unchanged resubmit states {difference}"
            for difference in differences(first, second, ignoring=ignoring)
        ]
        single = opened.result is not None and len(opened.result.specs) == 1
        if single and outcome(opened) != outcome(again):
            found.append(
                f"{door} ends {outcome(opened)} and its unchanged resubmit {outcome(again)}"
            )
    windowed = run.windowed if run.cell.pin == "none" else None
    if windowed is not None and outcome(run.first) != outcome(windowed):
        found.append(
            f"{FIRST} ends {outcome(run.first)} and the resubmit asking the dateless "
            f"Spec for the same years ends {outcome(windowed)}"
        )
    return found


# ---------------------------------------------------------------------------
# I3. One election.
# ---------------------------------------------------------------------------

_READS: dict[tuple, tuple[str | None, str | None, int, Years | None] | str] = {}


def _mint_request(world: World, stat_var: str, identity: dict) -> MaterialisationRequest:
    """The request a mint of an identity with no facet binding makes: its entities under
    the variable's declared roles, and its window, an open end stopping at the clock
    (`materialise.window.election_window`)."""
    # Imported here: the recorder runs this module on a copy of c669cc874, whose
    # `materialise.window` holds no `election_window`, and measures no datasets read there.
    from materialise.window import election_window

    window = identity["requested_window"]
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings={
            ("debtor" if role == "destination" else role): list(dcids)
            for role, dcids in identity["entities"].items()
        },
        declared_roles=world.declared_roles,
        facet_bindings={},
        window=election_window(
            RequestedWindow(
                kind=window["kind"],
                start_year=window.get("start_year"),
                end_year=window.get("end_year"),
            )
        ),
    )


def _datasets_read(
    world: World, spec: Spec
) -> tuple[str | None, str | None, int, Years | None] | str:
    """The source and unit a datasets read of `spec`'s identity elects, the rows it reads
    and the first and last years they span (None with no row), or the name of the refusal
    it raises.

    The read builds its request from the identity alone
    (`datasets_api.adapter.to_materialisation_request`) and previews it as a mint elects
    (`materialise.materialise_preview`). For an identity with no facet binding the request
    equals the one a mint makes (`_mint_request`)."""
    identity = spec.definition.identity.model_dump(mode="json", exclude_none=True)
    stat_var = spec.resolved.members[0].ref.ref
    key = (world, stat_var, json.dumps(identity, sort_keys=True))
    if key not in _READS:
        measured_property, purpose = world.stat_vars[stat_var]
        request = to_materialisation_request(
            identity,
            shape=StatVarShape(
                stat_var=f"dcid:{stat_var}",
                declared_roles=world.declared_roles,
                slots=((PURPOSE, purpose),) if purpose else (),
                measured_property=measured_property,
                population_type="FinancialFlow",
            ),
            code_index=get_index(),
        )
        if not identity.get("facets"):
            assert request == _mint_request(world, stat_var, identity)
        try:
            preview = materialise_preview(
                request,
                sdmx=world.materialise_sdmx,
                graph=world.materialise_graph,
                limit=10_000,
                coverage=world.index,
                rate_cache=RateTableCache(),
            )
        except MaterialisationError as refusal:
            _READS[key] = type(refusal).__name__
        else:
            served = preview.coverage.window
            _READS[key] = (
                preview.selections["provenance"].ref,
                preview.selections["unit"].ref,
                preview.total_rows,
                (served["start_year"], served["end_year"]) if served is not None else None,
            )
    return _READS[key]


def _is_undated_gap_election(
    cell: Cell, world: World, stat_vars: Iterable[str], elected: str | None
) -> bool:
    """Whether a datasets read electing `elected` is issue #1049, the declared limit of the
    shared election: a subject-only read probes spans with no dates, so it elects a facet
    whose span reaches the years the election ranks under (`_read`, an open end stopping at
    the clock) while it holds no point in them. A facet whose span lies wholly outside those
    years is no such election."""
    read = _read(cell.date)
    if world.role or read is None:
        return False
    reaches = any(
        _within(first, (None, read[1])) and _within(last, (read[0], None))
        for first, last in _spans(world, stat_vars, provenance=elected)
    )
    holds = any(_holds_points(world, stat_var, read, provenance=elected) for stat_var in stat_vars)
    return reaches and not holds


def _is_multi_member(spec: Stated) -> bool:
    """Whether a Spec is issue #1060: it lists more than one variable. The answer elects
    each member's group on its own and a mint elects one group across the members, so no
    one datasets read stands for the answer."""
    return len(spec.stat_vars) != 1


def _unserved_places_stated(world: World, spec: Stated) -> Iterator[str]:
    """An answer naming places serves one group across them, and states as unserved
    (`ResolvedSpec.unserved_entities`) exactly the named places its preview serves no
    point at."""
    if world.discovered or spec.preview is None:
        return
    if spec.preview_omitted is not None or spec.preview_truncated:
        return
    served = {s.place for s in spec.preview if s.dates}
    if not served:
        return
    unserved = set(PLACE_NAMES) - served
    if set(spec.unserved) != unserved:
        yield (
            f"its preview serves no point at {sorted(unserved)} and it states "
            f"{sorted(spec.unserved)} unserved"
        )


def i3_one_election(run: Doors) -> list[str]:
    """The source an answer cites is the source a datasets read of its Spec elects, the
    election a mint runs, and the source every unit decision behind the answer elects. The
    rows that read returns lie inside the years the answer states. A named place the one
    group does not serve is stated as unserved, at each door that previews.

    The two expected differences are issue #1049, where on the standard route a datasets
    read probes spans with no dates, so it can elect a facet with a gap across the whole
    window and read no row, while the answer read dates and serves a sibling facet
    (`_is_undated_gap_election`), and issue #1060, a Spec with more than one variable
    (`_is_multi_member`)."""
    cell, world = run.cell, run.world
    found: list[str] = []
    for door, answer in run.answers():
        for spec, said in zip(answer.specs, statements(answer), strict=True):
            found += [f"in {door}, {message}" for message in _unserved_places_stated(world, said)]
            if _is_multi_member(said):
                _pass_over("#1060", cell)
                continue
            read = _datasets_read(world, spec)
            if isinstance(read, str):
                # A provenance-less top facet refuses a read that binds no provenance (`i5`).
                if read != DefaultFacetUndefinedError.__name__ or said.sources:
                    found.append(
                        f"{door} cites {said.sources} and a datasets read of its Spec raises {read}"
                    )
            elif len(said.sources) > 1:
                found.append(
                    f"{door} cites a source per series, {said.sources}, and a datasets read of "
                    f"its Spec elects {read[0]} alone and reads {read[2]} rows"
                )
            elif said.sources != (read[0],):
                if _is_undated_gap_election(cell, world, said.stat_vars, read[0]):
                    _pass_over("#1049", cell)
                else:
                    found.append(
                        f"{door} cites {said.sources} and a datasets read of its Spec elects "
                        f"{read[0]} and reads {read[2]} rows"
                    )
            elif (
                read[3] is not None
                and said.window is not None
                and not (_within(read[3][0], said.window) and _within(read[3][1], said.window))
            ):
                found.append(
                    f"{door} states the years {said.window} and a datasets read of its Spec "
                    f"reads {read[3]}"
                )
            elif cell.unit == "none" and said.facet("unit")[0] not in (None, read[1]):
                found.append(
                    f"{door} states unit {said.facet('unit')[0]} and a datasets read of its "
                    f"Spec elects {read[1]}"
                )
        cited = {source for said in statements(answer) for source in said.sources}
        for decided in answer.recorder.unit_decisions:
            elected = {
                selections["provenance"].ref
                for combinations in decided.values()
                for selections in combinations
            }
            if answer.specs and not elected - {None} <= cited:
                found.append(
                    f"{door} cites {sorted(cited)} and its unit decision elects "
                    f"{sorted(elected, key=str)}"
                )
    return found


# ---------------------------------------------------------------------------
# I4. Dates.
# ---------------------------------------------------------------------------


def _is_unsourced_native_refusal(cell: Cell, first: Answer) -> bool:
    """Whether a first answer's refusal is issue #1055: over the `unsourced` rows a native
    unit request answers `facet_not_available`, while the facet with no provenance holds
    that unit."""
    return (
        cell.row_set == "unsourced"
        and cell.unit == "native"
        and outcome(first)[:2] == ("no_data", "facet_not_available")
    )


def i4_dates(run: Doors) -> list[str]:
    """`latest` names the latest year the cited sources hold at the places served, every
    cited source holds a point in the years read, and an answer holds data exactly when the
    world holds a point there, so a wholly future window answers from the facet holding
    those years.

    The one expected difference is issue #1055: a native unit request over the `unsourced`
    rows answers `facet_not_available` (`_is_unsourced_native_refusal`)."""
    cell, world = run.cell, run.world
    read = _read(cell.date)
    found: list[str] = []
    for door, answer in run.answers():
        for said in statements(answer):
            if cell.date == "latest" and said.window is not None:
                places = {s.place for s in said.preview if s.dates} if said.preview else None
                latest = max(
                    (
                        max(years)
                        for stat_var in said.stat_vars
                        for source in said.sources or (None,)
                        for place, years in _held_years(world, stat_var, provenance=source).items()
                        if places is None or place in places
                    ),
                    default=None,
                )
                if latest is not None and said.window[1] != latest:
                    found.append(
                        f"{door} states latest year {said.window[1]} and its sources "
                        f"{said.sources} hold {latest} at the places served"
                    )
            found += [
                f"{door} cites {source}, which holds no point in the years {cell.date} reads"
                for source in said.sources
                if not any(
                    _holds_points(world, stat_var, read, provenance=source)
                    for stat_var in said.stat_vars
                )
            ]
    if cell.unit == "converting" or cell.kind == "nearest_real":
        # A converting request serves whichever facets a recipe reaches, and a nearest-real
        # world's asked variable holds no row: `i1` and `i6` cover what those answers serve.
        return found
    admitted = _admitted(cell)
    if cell.pin == "none":
        with_points = [sv for sv in world.stat_vars if _holds_points(world, sv, read, **admitted)]
        expected = {0: "no_data", 1: "definite"}.get(len(with_points), "candidates")
        if _status(run.first) != expected and _is_unsourced_native_refusal(cell, run.first):
            _pass_over("#1055", cell)
        elif _status(run.first) != expected:
            found.append(
                f"{FIRST} is {outcome(run.first)[:3]} and {len(with_points)} variables "
                f"hold a point of {admitted or 'any facet'} in the years {cell.date} reads"
            )
    # Under a unit request a pinned source in another unit converts, so only an answer with
    # no unit request is held to the pinned facet's own points.
    pinned = run.opened if cell.pin != "none" and cell.unit == "none" else ()
    for said, answer in zip(statements(run.first), pinned, strict=False):
        holds = any(_holds_points(world, stat_var, read, **admitted) for stat_var in said.stat_vars)
        if holds != bool(answer.specs):
            found.append(
                f"{PINNED} is {outcome(answer)[:3]} and the world holds "
                f"{'a' if holds else 'no'} point of {admitted} in the years {cell.date} reads"
            )
    return found


# ---------------------------------------------------------------------------
# I5. Provenance-less top facets.
# ---------------------------------------------------------------------------


def i5_provenance_less_dispatch(run: Doors) -> list[str]:
    """A Spec that states no source, because the facet its plan elected carries no
    provenance, still serves that facet's points, by facet id, and names no source; a
    datasets read of the Spec, which binds no provenance, still refuses."""
    found: list[str] = []
    for door, answer in run.answers():
        for spec, said in zip(answer.specs, statements(answer), strict=True):
            if said.facet("provenance") != (None, None) or said.sources:
                continue
            if _sources_named(said.byline):
                found.append(
                    f"{door} cites no source and its sentence names one: {said.sentence!r}"
                )
            if said.preview is not None and {s.provenance for s in said.preview if s.dates} != {
                None
            }:
                found.append(
                    f"{door} cites no source and its preview serves "
                    f"{[(s.place, s.provenance, len(s.dates)) for s in said.preview]}"
                )
            if (read := _datasets_read(run.world, spec)) != DefaultFacetUndefinedError.__name__:
                found.append(
                    f"{door} cites no source and a datasets read binding none returns {read}"
                )
    return found


# ---------------------------------------------------------------------------
# I6. Nearest-real.
# ---------------------------------------------------------------------------


def i6_nearest_real_is_the_relaxed_answer(run: Doors) -> list[str]:
    """A nearest-real suggestion states what the same query answers with the relaxed
    binding left out, and an answer that found no observation at its named places suggests
    exactly the Specs that query answers with."""
    cell = run.cell
    if cell.kind != "nearest_real" or cell.pin != "none":
        return []
    suggestions = run.first.specs
    if not suggestions and (run.world.discovered or outcome(run.first)[1] != "no_observations"):
        # A relaxation probes the named places of an answer that found no observation.
        return []
    relaxed = _first_door(
        _world(cell.route, cell.row_set, cell.kind, True),
        cell.order,
        cell.date,
        cell.preview,
        cell.unit,
    )
    if len(suggestions) != len(relaxed.specs):
        return [
            f"{FIRST} suggests {len(suggestions)} Specs and the relaxed query is "
            f"{outcome(relaxed)[:3]} with {len(relaxed.specs)}"
        ]
    return [
        f"the suggestion against the relaxed query's answer states {difference}"
        for suggestion, direct in zip(statements(run.first), statements(relaxed), strict=True)
        for difference in differences(suggestion, direct)
    ]


# ---------------------------------------------------------------------------
# I7. The bind-stage sentence.
# ---------------------------------------------------------------------------


def i7_one_sentence_names_the_answer(run: Doors) -> list[str]:
    """The first door's pipeline streams no sentence: the part's one sentence goes out with
    the response, built from the part the response states
    (`sentence_events.response_sentence_events`)."""
    first = run.first
    if first.recorder.sentences:
        return [f"{FIRST} streamed sentences at requests {first.recorder.sentences}"]
    return []


# ---------------------------------------------------------------------------
# I8. Place order.
# ---------------------------------------------------------------------------


def _without_timings(value: Any) -> Any:
    """`value` without the step timings (`ms`) a trace records."""
    if isinstance(value, dict):
        return {k: _without_timings(v) for k, v in value.items() if k != "ms"}
    if isinstance(value, list):
        return [_without_timings(v) for v in value]
    return value


def i8_place_order_independence(run: Doors) -> list[str]:
    """The same named places in the other order give the same answer, warnings, requests and
    serial depth at every door."""
    cell = run.cell
    if cell.order == "dcid_order":
        return []
    other = doors(replace(cell, order="dcid_order"))
    found: list[str] = []
    pairs = zip(run.answers(), other.answers(), strict=False)
    for (door, swapped), (_door, answer) in pairs:
        if outcome(swapped) != outcome(answer):
            found.append(
                f"{door} ends {outcome(swapped)} with the places reversed and "
                f"{outcome(answer)} in dcid order"
            )
        dumps = [[_without_timings(s.model_dump()) for s in a.specs] for a in (swapped, answer)]
        if dumps[0] != dumps[1]:
            differing = [
                difference
                for a, b in zip(statements(swapped), statements(answer), strict=False)
                for difference in differences(a, b)
            ]
            found.append(
                f"{door} states different Specs with the places reversed: "
                f"{differing or 'outside the compared statements'}"
            )
        if swapped.recorder.counts() != answer.recorder.counts():
            found.append(
                f"{door} requests {dict(swapped.recorder.counts())} with the places "
                f"reversed and {dict(answer.recorder.counts())} in dcid order"
            )
        if swapped.recorder.depth() != answer.recorder.depth():
            found.append(
                f"{door} runs {swapped.recorder.depth()} serial steps with the places "
                f"reversed and {answer.recorder.depth()} in dcid order"
            )
    return found


# ---------------------------------------------------------------------------
# I9. Latency.
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_BASE_PATH = _FIXTURES / "answer_invariants_requests_c669cc874.json"
_BRANCH_PATH = _FIXTURES / "answer_invariants_requests_uncompared.json"
_DOOR_KEYS = {FIRST: "first", PINNED: "pinned", RESUBMIT: "resubmit", WINDOWED: "windowed"}
_GROUND_PHASES = {FIRST: "_ground_elected_answer", WINDOWED: "_phase_ground_build"}
_UNIT_DECISION_CLIENT = "materialise_graph."
_PAIR_READ = "materialise_sdmx.GET"
_UNIT_PLAN_READ = "materialise_graph.node_arcs_many"
_LABEL_READ = "graph.node_labels_batch"
_TYPE_READ = "graph.node_type"
_TYPE_LABEL_READ = "graph.node_label"
_TYPE_BATCH_READ = "graph.node_types_batch"
_BULK_READ = "sdmx.role_series"


def measured(answer: Answer) -> dict[str, Any]:
    """What I9 keeps of one answer: its outcome, the series, points and omission each Spec's
    preview serves, its requests by client method and its serial depth, the requests and
    depth of its nearest-real relaxations' member walks, the requests of its alternates'
    and re-election siblings' member walks, and the requests and depth of its Specs'
    facet-option builds."""
    served = []
    for spec in answer.specs:
        observations = spec.resolved.observations
        if observations is None:
            served.append([None, None, None])
            continue
        omitted = observations.omitted.reason if observations.omitted is not None else None
        points = sum(len(series.points) for series in observations.series)
        served.append([len(observations.series), points, omitted])
    reason = answer.result.no_data_reason if answer.result is not None else None
    walk = answer.recorder.relaxation_walk()
    alternate_walk = answer.recorder.alternate_walk()
    facet_options = answer.recorder.facet_options()
    return {
        "outcome": [_status(answer), reason, answer.refusal, answer.error],
        "served": served,
        "requests": dict(sorted(answer.recorder.counts().items())),
        "depth": answer.recorder.depth(),
        **({"relaxation_walk": walk} if walk is not None else {}),
        **({"alternate_walk": alternate_walk} if alternate_walk is not None else {}),
        **({"facet_options": facet_options} if facet_options is not None else {}),
    }


@cache
def _measurements(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _measurement(path: Path, cell: Cell, door: str, index: int) -> dict[str, Any] | None:
    """The measurement `path` holds for one door of `cell`, None when it holds none. A
    door's list holds None at each position the file does not hold."""
    held = _measurements(path)
    indices = held["doors"].get(str(cell), {}).get(_DOOR_KEYS[door], ())
    record = indices[index] if index < len(indices) else None
    return held["records"][record] if record is not None else None


def _same_answer(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["outcome"] == b["outcome"] and a["served"] == b["served"]


def _beyond(
    now: dict[str, Any],
    limit: dict[str, Any],
    source: str,
    *,
    granted: Mapping[str, int] | None = None,
) -> Iterator[str]:
    """Each way `now` requests more or runs deeper than `limit`, past the reads of the
    member walks its nearest-real relaxations, alternates and re-election siblings ran
    beyond those `limit` records, the reads of its facet-option builds beyond those
    `limit` records, the serial steps by which its deepest relaxation walk and its deepest
    facet-option build exceed `limit`'s (`measured`), and the reads `granted` adds. The
    walks of alternates and siblings run beside the elected shape's, so they earn no serial
    step."""
    none = {"requests": {}, "depth": 0}
    walk, held = now.get("relaxation_walk", none), limit.get("relaxation_walk", none)
    facets = now.get("facet_options", none)
    held_facets = limit.get("facet_options", none)
    extra = Counter(walk["requests"]) - Counter(held["requests"])
    extra += Counter(now.get("alternate_walk", none)["requests"]) - Counter(
        limit.get("alternate_walk", none)["requests"]
    )
    extra += Counter(facets["requests"]) - Counter(held_facets["requests"])
    allowed = Counter(limit["requests"]) + extra + Counter(granted or {})
    over = {
        request: f"{count} against {allowed[request]}"
        for request, count in now["requests"].items()
        if count > allowed[request]
    }
    if over:
        yield f"makes more requests than {source}: {over}"
    deeper = max(0, walk["depth"] - held["depth"])
    built = max(0, facets["depth"] - held_facets["depth"])
    if now["depth"] > limit["depth"] + deeper + built:
        yield (
            f"runs {now['depth']} serial steps against {limit['depth']} {source}, "
            f"{deeper} for its relaxations' walks and {built} for its facet options"
        )


def _rate_table_reads(now: dict[str, Any]) -> dict[str, int]:
    """The rate-table reads a door made, each counted once (`Recorder.counts`): the table
    load its bound-unit gate makes to check that every served row converts
    (`convertibility.unit_coverage`)."""
    return {
        request: count
        for request, count in now["requests"].items()
        if request.endswith(fixture_world.RATE_TABLE_READ)
    }


def _batched_type_read(now: dict[str, Any], limit: dict[str, Any]) -> dict[str, int]:
    """The batched reads a door makes in place of per-node reads `limit` holds. A resubmit
    reads the types of its posted places in one request
    (`promote_places._phase_where_places`), and grounding reads the types of a Spec's bound
    entities in one request and the labels of those types in one more
    (`grounding._ground_entities_and_sources`), where the recorded door read each node in
    turn. A door is granted at most one batched read for each per-node read it no longer
    makes, so it never makes more requests than `limit`. Empty for a door that makes as
    many per-node reads as `limit`."""
    granted = {
        batch: limit["requests"].get(single, 0) - now["requests"].get(single, 0)
        for single, batch in ((_TYPE_READ, _TYPE_BATCH_READ), (_TYPE_LABEL_READ, _LABEL_READ))
    }
    return {batch: count for batch, count in granted.items() if count > 0}


def _pair_read_in_place_of_bulk(now: dict[str, Any], limit: dict[str, Any]) -> dict[str, int]:
    """Grant a pair read only when a narrowed discovered scope drops a bulk read.

    A source pin can leave one recipient below the bulk cap. Each removed bulk
    request permits one per-pair request; retained bulk requests permit none.
    """
    count = min(
        now["requests"].get(_PAIR_READ, 0),
        max(0, limit["requests"].get(_BULK_READ, 0) - now["requests"].get(_BULK_READ, 0)),
    )
    return {_PAIR_READ: count} if count else {}


def _held_grants(now: dict[str, Any], limit: dict[str, Any]) -> dict[str, int]:
    """The reads a door may make beyond a recorded door with its answer: batched reads in
    place of per-node reads (`_batched_type_read`) and pair reads in place of bulk reads
    (`_pair_read_in_place_of_bulk`)."""
    return _batched_type_read(now, limit) | _pair_read_in_place_of_bulk(now, limit)


def _preview_reads(cell: Cell, answer: Answer) -> list[dict[str, Any]]:
    """The reads a door's preview may add to the same door with the preview off, as the
    request lists and serial steps it may choose between.

    A preview serving no series reads nothing. A per-pair read makes one GET per series and
    one label read after them. Under a unit request each series first plans its conversion
    in two reads of materialise's graph, six when it converts, and a converting preview
    loads the rate table in three reads, which then carry the label read. A bulk route may
    instead make one bulk read and one label read. Converting, it plans each StatVar from
    the reads its unit gate made, and between the two reads its request's view of the rate
    cache rereads the table's refresh date, so the label read carries the table."""
    series = [s for said in statements(answer) for s in said.preview or ()]
    if not series:
        return [{"requests": {}, "depth": 0}]
    rates = fixture_world.RATE_TABLE_READ
    converting = any(s.converted for s in series)
    plans = [0 if cell.unit == "none" else 6 if s.converted else 2 for s in series]
    requests = {_PAIR_READ: len(series), _UNIT_PLAN_READ: sum(plans)}
    if converting:
        requests |= {_UNIT_PLAN_READ + rates: 1, _PAIR_READ + rates: 1, _LABEL_READ + rates: 1}
    else:
        requests[_LABEL_READ] = 1
    per_pair = {"requests": requests, "depth": max(plans) + (3 if converting else 0) + 2}
    if not cell.route.endswith("bulk"):
        return [per_pair]
    bulk = (
        {
            "requests": {_BULK_READ: 1, _UNIT_PLAN_READ + rates: 1, _LABEL_READ + rates: 1},
            "depth": 3,
        }
        if converting
        else {"requests": {_BULK_READ: 1, _LABEL_READ: 1}, "depth": 2}
    )
    return [per_pair, bulk]


def _preview_bound(cell: Cell, answer: Answer, now: dict[str, Any], off: dict[str, Any]) -> str:
    """How the door requests beyond its preview-off door and every list its preview may
    add, empty when one list covers it."""
    found = ""
    for reads in _preview_reads(cell, answer):
        limit = {
            "requests": {
                request: off["requests"].get(request, 0) + reads["requests"].get(request, 0)
                for request in {*off["requests"], *reads["requests"]}
            },
            "depth": off["depth"] + reads["depth"],
            **({"facet_options": off["facet_options"]} if "facet_options" in off else {}),
        }
        found = "; ".join(_beyond(now, limit, "with the preview off and its preview's reads"))
        if not found:
            return ""
    return found


def _window_answer_reads(cell: Cell, door: str, answer: Answer) -> list[str]:
    """The requests a first or windowed door made once its ground phase began, when it
    answers that its window holds no point and suggests nothing: the answer states no label,
    so the phase may read only what its unit decision reads from materialise's graph."""
    result = answer.result
    if (
        door not in _GROUND_PHASES
        or result is None
        or result.no_data_reason != "no_observations"
        or result.window_gap_fact is None
        or result.nearest_real
    ):
        return []
    return [
        request
        for request in answer.recorder.requests_from(_GROUND_PHASES[door])
        if cell.unit == "none" or not request.startswith(_UNIT_DECISION_CLIENT)
    ]


def i9_latency(run: Doors) -> list[str]:
    """No door makes a request or runs a serial step it did not need.

    Each door is measured (`measured`) and held to the first of these that applies:

    - The door of the same cell, door and position at c669cc874, when that door gave the
      same outcome and served the same rows: no more requests of any kind and no deeper
      serial chain. `answer_invariants_requests_c669cc874.json` holds every door of the grid
      measured there.
    - For a door c669cc874 answered differently or not at all, the measurement recorded for
      it on this branch in `answer_invariants_requests_uncompared.json`, under the same
      condition. That file holds the covering array's doors with neither a comparable door
      at c669cc874 nor a preview-off door with their outcome.
      A re-recording keeps an entry while its door passes it and replaces one only with
      `--grow`. `_answer_invariants_record` regenerates both files and refuses a list that
      grows.

    A preview-on door is also held to its own preview-off door plus the reads its preview
    may make (`_preview_reads`), which is the whole bound of one neither file answers for.
    A door none of these bounds fails.

    A door whose nearest-real relaxations walk their members may exceed its bound by those
    walks' reads and by the serial steps of the deepest walk alone, since the walks run side
    by side (`grounding._walked_relaxations`).

    A door whose alternates or re-election siblings walk their members may exceed its bound
    by those walks' reads, and by no serial step: the walks start beside the elected
    shape's (`phase_place_materialise._start_alternate_walks`).

    A door may exceed its bound by the reads its Specs' facet options make
    (`option_coverage._maybe_build_facet_options`), built for every Spec whose coverage is
    not streamed, and by the serial steps of its deepest build.

    A first or windowed door answering that its window holds no point makes no request
    once its ground phase begins, beyond its unit decision's reads
    (`_window_answer_reads`).

    A door may make a batched read in place of each per-node type or type-label read its
    bound holds (`_batched_type_read`).

    A discovered bulk source pin may narrow the scope below the bulk cap, replacing each
    removed bulk read with at most one pair read (`_pair_read_in_place_of_bulk`).

    A door held to c669cc874 may exceed it by one load of each rate table it reads
    (`_rate_table_reads`). Its bound-unit gate loads them to refuse a conversion the tables
    cannot serve before the answer is definite, a check c669cc874 left to the preview.

    The grid holds one creditor and two debtors. An open role with 1, 5 and 40 members is
    covered by `test_read_plan_invariants.py::
    test_an_open_origin_preview_makes_one_read_per_debtor_however_many_creditors`."""
    cell = run.cell
    found: list[str] = []
    off_doors: dict[tuple[str, int], Answer] = {}
    if cell.preview:
        positions: dict[str, int] = {}
        for door, answer in doors(replace(cell, preview=False)).answers():
            off_doors[door, positions.setdefault(door, 0)] = answer
            positions[door] += 1
    positions = {}
    for door, answer in run.answers():
        index = positions.setdefault(door, 0)
        positions[door] += 1
        now = measured(answer)
        bounded = False
        for path, source in ((_BASE_PATH, "at c669cc874"), (_BRANCH_PATH, "as recorded")):
            limit = _measurement(path, cell, door, index)
            if limit is not None and _same_answer(limit, now):
                granted = _held_grants(now, limit)
                if path == _BASE_PATH:
                    granted |= _rate_table_reads(now)
                found += [
                    f"{door} ({_status(answer)}) {line}"
                    for line in _beyond(now, limit, source, granted=granted)
                ]
                bounded = True
                break
        off_door = off_doors.get((door, index))
        if off_door is not None and (off := measured(off_door))["outcome"] == now["outcome"]:
            if beyond := _preview_bound(cell, answer, now, off):
                found.append(f"{door} ({_status(answer)}) {beyond}")
            bounded = True
        if not bounded:
            found.append(
                f"{door} ({_status(answer)}) has no door to be held to: none at c669cc874 "
                "with its answer, none recorded and no preview-off door"
            )
        if reads := _window_answer_reads(cell, door, answer):
            found.append(
                f"{door} answers that its window holds no point and its ground phase "
                f"requests {reads}"
            )
    return found


# ---------------------------------------------------------------------------
# I10. Election warnings.
# ---------------------------------------------------------------------------


def _election_warnings(answer: Answer, code: str) -> list[str]:
    if answer.result is None:
        return []
    return sorted(w.message for w in answer.result.warnings if w.code == code)


def _is_window_answer(result: RegionResult) -> bool:
    """An answer for data outside the years asked: no Spec, `no_observations`, and the
    `year_unavailable` fact naming the years asked and the span the elected member's series
    runs."""
    return (
        not result.specs
        and result.no_data_reason == "no_observations"
        and result.window_gap_fact is not None
    )


def _expected_family_warnings(world: World, answer: Answer) -> list[str]:
    """The `family_total_absent` messages an answer's election raises: one for each Spec the
    part holds (its own, a flagged match, an alternate match) whose one member is not its
    family's total (`World.lacks_total`), and one for an answer for data outside the years
    asked (`_is_window_answer`) whose elected member is not. A
    nearest-real suggestion is a Spec of another query, whose own first answer raises its
    warning."""
    result = answer.result
    if result is None:
        return []
    held = [*result.specs, *(result.flagged_match or ())]
    if result.alternate_match is not None:
        held.append(result.alternate_match)
    named = [_stat_vars(spec) for spec in held]
    if _is_window_answer(result) and result.elected_stat_vars:
        named = [[ref.ref for ref in result.elected_stat_vars]]
    message = f"No total for this family of variables was found, so the answer shows {world.label}."
    return sorted(
        message for members in named if len(members) == 1 and world.lacks_total(members[0])
    )


def i10_election_warnings(run: Doors) -> list[str]:
    """A door warns an election warning exactly when it runs the election. A first answer
    warns `family_total_absent` for exactly the Specs it holds whose member is not its
    family's total, and so does a nearest-real suggestion's own first answer, the relaxed
    query's. A resubmit posts the member and any default. When the mint's member walk
    settles, the resubmit holds the walked family and elects again among it, so it warns
    `family_total_absent` where the first answer of the query its Spec answers does: the
    relaxed query's for a nearest-real suggestion. When the walk fails, no resubmit
    answers (`doors`). The world declares no slot default, so no door
    warns `slot_defaulted`. The comparisons between doors leave these warnings out
    (`outcome`), and this check states where each belongs.

    The first door warns `membership_unverified` in a world whose walk fails, and no door
    does in a world whose walk settles."""
    cell = run.cell
    first_answers = [(FIRST, run.world, run.first)] if cell.pin == "none" else []
    relaxed_world = (
        _world(cell.route, cell.row_set, cell.kind, True) if cell.kind == "nearest_real" else None
    )
    if relaxed_world is not None and cell.pin == "none":
        relaxed = _first_door(relaxed_world, cell.order, cell.date, cell.preview, cell.unit)
        first_answers.append(("the relaxed query's first answer", relaxed_world, relaxed))
    found: list[str] = []
    for door, world, answer in first_answers:
        expected = _expected_family_warnings(world, answer)
        actual = _election_warnings(answer, "family_total_absent")
        if actual != expected:
            found.append(f"{door} warns family_total_absent {actual} where it elects {expected}")
    resubmit_world = relaxed_world or run.world
    for door, answer in run.answers():
        if door == FIRST:
            continue
        expected = _expected_family_warnings(resubmit_world, answer)
        actual = _election_warnings(answer, "family_total_absent")
        if actual != expected:
            found.append(f"{door} warns family_total_absent {actual} where it elects {expected}")
    every_answer = [*run.answers(), *((d, a) for d, _w, a in first_answers[1:])]
    for door, answer in every_answer:
        if warned := _election_warnings(answer, "slot_defaulted"):
            found.append(f"{door} warns slot_defaulted {warned} in a world declaring none")
    for door, answer in every_answer:
        if answer.result is None:
            continue
        warned = bool(_election_warnings(answer, "membership_unverified"))
        if warned == run.world.walk_settles:
            walk = "settles" if run.world.walk_settles else "fails"
            says = "warns" if warned else "does not warn"
            found.append(f"{door} {says} membership_unverified where the member walk {walk}")
    return found


# ---------------------------------------------------------------------------
# I11. Download facts.
# ---------------------------------------------------------------------------


def i11_download_is_the_datasets_read(run: Doors) -> list[str]:
    """The rows and years a Spec states for its file (`ResolvedSpec.download`) are those a
    datasets read of its Spec reads: the same rows when stated exact, at least as many when
    stated as an upper bound. A Spec whose read refuses has no file to state. The read and
    the answer elect apart in issues #1049 and #1060, as `i3_one_election` passes over."""
    found: list[str] = []
    for door, answer in run.answers():
        for spec, said in zip(answer.specs, statements(answer), strict=True):
            if _is_multi_member(said):
                continue
            read = _datasets_read(run.world, spec)
            if isinstance(read, str):
                continue
            if said.sources != (read[0],) and _is_undated_gap_election(
                run.cell, run.world, said.stat_vars, read[0]
            ):
                continue
            download = spec.resolved.download
            if download is None or download.rows is None:
                found.append(f"{door} states no rows and a datasets read reads {read[2]}")
            elif download.rows != read[2] if download.rows_exact else download.rows < read[2]:
                exact = "exactly" if download.rows_exact else "up to"
                found.append(
                    f"{door} states {exact} {download.rows} rows and a datasets read of its "
                    f"Spec reads {read[2]}"
                )
            elif (
                download.rows_exact
                and read[3] is not None
                and download.window is not None
                and (download.window.start_year, download.window.end_year) != read[3]
            ):
                found.append(
                    f"{door} states the file's years {download.window} and a datasets read of "
                    f"its Spec reads {read[3]}"
                )
    return found


CHECKS: dict[str, Callable[[Doors], list[str]]] = {
    "I1": i1_plan_is_the_source,
    "I2": i2_door_equality,
    "I3": i3_one_election,
    "I4": i4_dates,
    "I5": i5_provenance_less_dispatch,
    "I6": i6_nearest_real_is_the_relaxed_answer,
    "I7": i7_one_sentence_names_the_answer,
    "I8": i8_place_order_independence,
    "I9": i9_latency,
    "I10": i10_election_warnings,
    "I11": i11_download_is_the_datasets_read,
}


# ---------------------------------------------------------------------------
# The grid.
# ---------------------------------------------------------------------------

BLOCKS = list(itertools.product(ROUTES, ROW_SETS, KINDS, UNIT_TEXT, DATES))


def cells(route: str, row_set: str, kind: str, unit: str, date: str) -> Iterator[Cell]:
    """Every cell of one route, row set, kind, unit request and date request: the product
    of previews, pins and place orders, the dcid order first."""
    for preview, pin, order in itertools.product(PREVIEWS, PINS, ORDERS):
        yield Cell(route, row_set, kind, unit, date, preview, pin, order)


GRID: Grid[Cell] = Grid(
    name="ANSWER_INVARIANTS",
    factors={
        "route": ROUTES,
        "row_set": tuple(ROW_SETS),
        "kind": KINDS,
        "unit": tuple(UNIT_TEXT),
        "date": tuple(DATES),
        "preview": PREVIEWS,
        "pin": PINS,
        "order": ORDERS,
    },
    strength=4,
    cell=Cell,
    cell_id=str,
    admits=lambda cell: cannot_exist(cell) is None,
)


def _cells_run() -> dict[tuple[str, ...], list[Cell]]:
    """The cells the grid test runs, by block in `BLOCKS` order, each block's cells in
    `cells` order: the committed covering array (`_grid_cells.ANSWER_INVARIANTS`), or every
    cell a request can form under `DFO_FULL_GRIDS=1`."""
    run: dict[tuple[str, ...], list[Cell]] = {}
    for cell in GRID.run(_grid_cells.ANSWER_INVARIANTS):
        run.setdefault((cell.route, cell.row_set, cell.kind, cell.unit, cell.date), []).append(cell)
    return run


CELLS_RUN = _cells_run()

# The cells `i9_latency` checks: the covering array its recorded bound holds.
_I9_CELLS = frozenset(_grid_cells.ANSWER_INVARIANTS)


def checks_for(cell: Cell) -> dict[str, Callable[[Doors], list[str]]]:
    """The checks the grid test runs on `cell`: every check in `CHECKS`, less I9 on a cell
    outside the covering array."""
    return {
        invariant: check
        for invariant, check in CHECKS.items()
        if invariant != "I9" or str(cell) in _I9_CELLS
    }


@pytest.fixture(scope="module")
def door_loop() -> Iterator[None]:
    with fixture_world.door_loop():
        yield


@pytest.fixture
def recorded_engine(monkeypatch, search_settings, door_loop) -> Callable[[str], None]:
    """The engine as the grid runs it: its hand-offs carry the recorder's logical clock,
    its unit decisions are logged, and the pair cap sends a bulk route to its bulk read.
    Returns the function that sets the cap for a route."""
    fixture_world.carry_logical_clock(monkeypatch)
    fixture_world.log_unit_decisions(monkeypatch)
    fixture_world.log_phases(monkeypatch)
    fixture_world.log_relaxation_walks(monkeypatch)
    fixture_world.log_alternate_walks(monkeypatch)
    fixture_world.log_facet_options(monkeypatch)
    return lambda route: search_settings(obs_preview_max_pairs=1 if route.endswith("bulk") else 120)


@pytest.mark.parametrize(
    ("route", "row_set", "kind", "unit", "date"), CELLS_RUN, ids=["-".join(b) for b in CELLS_RUN]
)
def test_every_cell_holds_every_invariant(
    recorded_engine, route: str, row_set: str, kind: str, unit: str, date: str
) -> None:
    """Each cell the grid runs (`CELLS_RUN`) runs its doors once and every invariant's
    check reads them. The cells of one block share their first doors. A
    failure lists one line per disagreement: the cell, the invariant and the two statements
    that disagree. The grid expects three differences, each passed over by one predicate
    tied to its issue: #1049, an undated datasets read electing a facet with a gap across
    the window (`i3_one_election`); #1055, a native unit request over the
    `unsourced` rows answering `facet_not_available` (`i4_dates`); and #1060, a Spec with
    more than one variable (`i3_one_election`). I9 runs on the covering array's cells only
    (`checks_for`)."""
    recorded_engine(route)
    failures: list[str] = []
    for cell in CELLS_RUN[route, row_set, kind, unit, date]:
        run = doors(cell)
        for invariant, check in checks_for(cell).items():
            failures += [f"{cell} | {invariant} | {message}" for message in check(run)]
    assert not failures, "\n".join(failures)


def test_a_bulk_read_substitution_does_not_excuse_extra_pair_reads_or_serial_depth() -> None:
    limit = {"requests": {_BULK_READ: 1}, "depth": 1}
    replaced = {"requests": {_PAIR_READ: 1}, "depth": 1}
    extra_pair = {"requests": {_PAIR_READ: 2}, "depth": 1}
    retained_bulk = {"requests": {_BULK_READ: 1, _PAIR_READ: 1}, "depth": 1}
    deeper = {"requests": {_PAIR_READ: 1}, "depth": 2}

    def excess(now: dict[str, Any]) -> list[str]:
        return list(
            _beyond(now, limit, "baseline", granted=_pair_read_in_place_of_bulk(now, limit))
        )

    assert excess(replaced) == []
    assert excess(extra_pair) == [
        "makes more requests than baseline: {'materialise_sdmx.GET': '2 against 1'}"
    ]
    assert excess(retained_bulk) == [
        "makes more requests than baseline: {'materialise_sdmx.GET': '1 against 0'}"
    ]
    assert excess(deeper) == [
        "runs 2 serial steps against 1 baseline, 0 for its relaxations' walks and 0 for its facet options"
    ]


def test_the_1049_exemption_needs_a_span_that_reaches_the_window_and_holds_no_point() -> None:
    """In the `gap` rows IDS runs 2010 to 2030 and holds no point from 2016 to 2026. An
    undated read electing it for 2019 to 2024 is issue #1049. Its span lies wholly outside
    1950 to 1955 and it holds points in 2027 to 2029, so neither of those elections is
    passed over, and none is on a role route, whose read probes dates."""

    def passed_over(route: str, date: str) -> bool:
        cell = Cell(route, "gap", "definite", "none", date, True, "none", "dcid_order")
        world = _world(route, "gap", "definite")
        return _is_undated_gap_election(cell, world, world.stat_vars, IDS)

    assert passed_over("standard", "clip_end")
    assert passed_over("standard", "since")
    assert not passed_over("standard", "unreached")
    assert not passed_over("standard", "future")
    assert not passed_over("standard", "dateless")
    assert not passed_over("named_pair", "clip_end")


def test_the_grid_skips_only_combinations_no_request_can_form() -> None:
    """Of the full product of routes, row sets, kinds, unit requests, date requests,
    previews, pins and place orders, the grid leaves out two kinds of combination, each
    for a stated reason: a reversed order on a route that names no place, and a unit pin
    beside a unit request."""
    skipped = [
        (cell, reason)
        for block in BLOCKS
        for cell in cells(*block)
        if (reason := cannot_exist(cell)) is not None
    ]
    assert {reason for _cell, reason in skipped} == {
        "a discovered route names no place, so it has one order",
        "an identity's unit axis holds a request or a ref, never both",
    }
    assert all(
        (cell.order == "reversed" and cell.route.startswith("discovered"))
        or (cell.pin == "unit" and cell.unit != "none")
        for cell, _reason in skipped
    )
    assert len(BLOCKS) * len(PREVIEWS) * len(PINS) * len(ORDERS) - len(skipped) == 21_504


def test_i9_checks_exactly_the_committed_cells() -> None:
    """Every committed cell runs I9 beside every other check, and every other cell runs
    every check but I9."""
    committed = frozenset(_grid_cells.ANSWER_INVARIANTS)
    every = [cell for _row, cell in GRID.cells()]
    assert committed <= {str(cell) for cell in every}
    for cell in every:
        expected = set(CHECKS) - (set() if str(cell) in committed else {"I9"})
        assert set(checks_for(cell)) == expected, cell
    assert set(_measurements(_BRANCH_PATH)["doors"]) <= committed


_HELD = {
    "outcome": ["definite", None, None, None],
    "served": [[1, 3, None]],
    "requests": {_LABEL_READ: 3},
    "depth": 2,
}


@pytest.mark.parametrize(
    ("old", "now", "grows"),
    [
        pytest.param(None, _HELD, True, id="added"),
        pytest.param(_HELD, {**_HELD, "outcome": ["no_data", None, None, None]}, True, id="answer"),
        pytest.param(_HELD, {**_HELD, "requests": {_LABEL_READ: 4}}, True, id="requests"),
        pytest.param(_HELD, {**_HELD, "depth": 3}, True, id="depth"),
        pytest.param(
            _HELD,
            {
                **_HELD,
                "requests": {_LABEL_READ: 4},
                "facet_options": {"requests": {_LABEL_READ: 1}, "depth": 1},
            },
            False,
            id="facet_read_within_bound",
        ),
        pytest.param(_HELD, {**_HELD, "requests": {_LABEL_READ: 2}, "depth": 1}, False, id="fewer"),
        pytest.param(_HELD, _HELD, False, id="identical"),
    ],
)
def test_a_rerecord_keeps_a_held_record_its_door_passes(
    old: dict[str, Any] | None, now: dict[str, Any], grows: bool
) -> None:
    """`_answer_invariants_record.rerecord` keeps a door's held record while the door passes
    I9 against it, so a re-record never loosens the bound: a facet-option read the old
    record lacks leaves the old record, which still refuses a fourth label read outside a
    facet build. A door added, answering differently or exceeding its record takes its
    measurement and grows the list."""
    key = ("cell", "resubmit", 0)
    held = {} if old is None else {key: old}
    doors, grown = _answer_invariants_record.rerecord(held, {key: now})
    assert doors == {key: now if grows else old}
    assert bool(grown) == grows


def test_the_recorder_lists_only_committed_doors_at_their_positions(tmp_path: Path) -> None:
    """`_answer_invariants_record` lists the doors of the committed cells alone, and a door
    whose earlier position is bounded keeps its own position in the written list."""
    committed, other = sorted(_grid_cells.ANSWER_INVARIANTS)[:2]
    same = {"outcome": ["definite", None, None, None], "served": [], "requests": {}, "depth": 1}
    changed = {**same, "served": [[1, 3, None]]}
    base = {(committed, "resubmit", 0): same, (committed, "resubmit", 1): same}
    branch = {
        (committed, "resubmit", 0): same,
        (committed, "resubmit", 1): changed,
        (other, "resubmit", 0): changed,
    }
    listed = _answer_invariants_record.uncompared(base, branch, cells=frozenset({committed}))
    assert listed == {(committed, "resubmit", 1): changed}
    path = tmp_path / "uncompared.json"
    path.write_text(_answer_invariants_record._dump(listed))
    cell = next(cell for _row, cell in GRID.cells() if str(cell) == committed)
    assert _measurement(path, cell, RESUBMIT, 0) is None
    assert _measurement(path, cell, RESUBMIT, 1) == changed


def test_the_committed_cells_cover_every_four_values_a_request_can_form() -> None:
    """Every four factor values that some cell a request can form holds, some committed
    cell holds too, and every committed cell is one a request can form. A factor value
    added to the grid fails this until `python -m tests._covering_array` regenerates
    `_grid_cells.ANSWER_INVARIANTS`."""
    assert not (gaps := uncovered(GRID, _grid_cells.ANSWER_INVARIANTS)), "\n".join(gaps)
