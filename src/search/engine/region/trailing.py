"""Settle a span counted back from now on the newest year an answer's admitted series hold.

A trailing request ("the last five years", `extract.DateRequest.trailing`) reads its
`extract.read_window` through materialisation and the unit and facet admission. That window
holds every N-year span ending in the judged years (`DateRequest.judged`), so the members,
units and facets of whichever span the answer settles on were confirmed in that one read. Once
admission has fixed what an answer serves, its newest year Y is the newest year any served
series holds (`read_plan.newest_served_year`), one year per Spec, and the answer reads the N
years ending there, `[Y - N + 1, Y]` (`extract.TrailingSpan.settled`), over the plan it already
built (`ReadPlan.windowed`), so the request resolves once. A standard answer that elected one
member starts after the newest year a member ahead of it in election order holds
(`Materialised.passed_over_newest_year`), so a resubmit of the range elects the same member,
and no span starts before the plan's own years; such a span can hold fewer than N years, and
one the passed-over member shortens departs `year_unavailable` showing the years it holds. Its
request becomes that plain range, which the Spec states and a resubmit reads. A Y more than
`trailing_stale_years` before the current year departs `year_unavailable` naming it
(`stale_departure`). An admitted answer whose newest year lies before the judged years reads the
judged years, where it holds no point, as an answer to the asked span alone would.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from search.engine.departures import shortened_trailing_fact, stale_trailing_fact
from search.engine.extract import DateRequest
from search.engine.region.codes import DATE_COVERAGE_PARTIAL
from search.engine.region.facets import FilteredAnswer, plan_answer
from search.engine.retrieval.read_plan import (
    current_year,
    data_years,
    has_points,
    held_stat_vars,
    newest_served_year,
    served_dates,
    year_of,
)
from search.models import TimeWindow, Warning
from search.scope_window import scope_window_settings

if TYPE_CHECKING:
    from search.engine.retrieval.read_plan import ReadPlan
    from search.engine.retrieval.retrieve import Materialised
    from search.heading.departure_fact import DepartureFact
    from series_coverage.index import CoverageIndex


@dataclass(frozen=True)
class SettledTrailing:
    """An answer read over the newest years of a trailing request: the materialised result
    and filtered answer over them, the plain range request that names them, the calendar
    years the request's words cover (`DateRequest.window`), which a later departure names as
    the years asked, and the departure a stale newest year raises."""

    mat: Materialised
    filtered: FilteredAnswer
    date_request: DateRequest
    asked: TimeWindow
    departure: tuple[Warning, DepartureFact] | None


def _assert_inside(plan: ReadPlan, window: TimeWindow) -> None:
    """`ReadPlan.windowed`'s precondition: `window` lies inside the years `plan` read."""
    years = plan.years
    assert isinstance(years, TimeWindow), years
    assert years.start_year is None or (
        window.start_year is not None and years.start_year <= window.start_year
    ), (years, window)
    assert years.end_year is None or (
        window.end_year is not None and window.end_year <= years.end_year
    ), (years, window)


def _plan_start(plan: ReadPlan) -> int | None:
    """The first of `plan`'s years, where a settled range starts at the earliest: a role read
    floors its years where coverage begins (`scope_window.role_read_years`), so no data
    precedes them, and a resubmit of the range floors there too."""
    years = plan.years
    return years.start_year if isinstance(years, TimeWindow) else None


def _serving(mat: Materialised, window: TimeWindow, answered: ReadPlan) -> Materialised:
    """`mat` listing the StatVars a read of `window` alone confirms, with their facets and
    count, by the evidence that read confirms them by, before unit and facet admission: on
    the standard path each confirmed member's facets at its places (`confirmed_facets`),
    elsewhere the plan the route read (`mat.read_plan`), already narrowed to `window`, else
    the answered plan `answered`. Unchanged when that evidence holds no point in `window`."""
    if mat.confirmed_facets is not None:
        confirmed = [
            sv for sv, facets in mat.confirmed_facets.items() if has_points(facets, window)
        ]
        held = frozenset(confirmed)
    else:
        held = held_stat_vars(mat.read_plan if mat.read_plan is not None else answered)
        confirmed = None
    if not held:
        return mat
    sv_dcids = [sv for sv in mat.sv_dcids if sv in held]
    dropped = [sv for sv in mat.sv_dcids if sv not in held]
    facets_by_sv = mat.facets_by_sv
    facets_by_entity = mat.facets_by_entity
    if facets_by_sv is not None and dropped:
        gone = {id(f) for sv in dropped for f in facets_by_sv.get(sv, ())}
        facets_by_sv = {sv: facets for sv, facets in facets_by_sv.items() if sv in held}
        if facets_by_entity is not None:
            facets_by_entity = {
                entity: [f for f in facets if id(f) not in gone]
                for entity, facets in facets_by_entity.items()
            }
    population = mat.population
    if population is not None:
        count = len(confirmed) if confirmed is not None else len(sv_dcids)
        population = replace(population, confirmed=count)
    return replace(
        mat,
        sv_dcids=sv_dcids,
        facets_by_sv=facets_by_sv,
        facets_by_entity=facets_by_entity,
        population=population,
    )


def _over(
    mat: Materialised,
    filtered: FilteredAnswer,
    window: TimeWindow,
    *,
    date_request: DateRequest,
    coverage_index: CoverageIndex,
) -> tuple[Materialised, FilteredAnswer]:
    """`mat` and `filtered` read over `window`, inside the years their plans read, `mat`
    listing the StatVars a read of `window` alone confirms (`_serving`)."""
    assert filtered.plan is not None
    _assert_inside(filtered.plan, window)
    plan = filtered.plan.windowed(window)
    if mat.read_plan is not None:
        _assert_inside(mat.read_plan, window)
        mat = replace(mat, read_plan=mat.read_plan.windowed(window))
    mat = _serving(mat, window, plan)
    answer = plan_answer(mat, plan, date_request=date_request, coverage_index=coverage_index)
    return mat, answer


def settle_trailing(
    mat: Materialised,
    filtered: FilteredAnswer,
    *,
    date_request: DateRequest | None,
    coverage_index: CoverageIndex,
) -> SettledTrailing | None:
    """`filtered`, the answer after unit and facet admission, read over the newest
    `TrailingSpan.years` years its plan serves, starting after the newest year a member passed
    over in election holds (`Materialised.passed_over_newest_year`). An answer whose newest
    year lies before the judged years is read over the judged years and keeps its request.
    None for a request that does not trail, or a plan that serves no dated point, which
    keeps its answer."""
    if date_request is None or date_request.trailing is None or filtered.plan is None:
        return None
    newest = newest_served_year(filtered.plan)
    if newest is None:
        return None
    judged = date_request.judged.window
    assert judged is not None and judged.start_year is not None and judged.end_year is not None
    asked = date_request.window
    assert asked is not None
    plan_start = _plan_start(filtered.plan)
    if newest < judged.start_year:
        start = judged.start_year if plan_start is None else max(judged.start_year, plan_start)
        window = TimeWindow(start_year=start, end_year=judged.end_year)
        mat, answer = _over(
            mat, filtered, window, date_request=date_request, coverage_index=coverage_index
        )
        return SettledTrailing(
            mat=mat, filtered=answer, date_request=date_request, asked=asked, departure=None
        )
    passed_over = mat.passed_over_newest_year
    passed_floor = None if passed_over is None else passed_over + 1
    floors = [floor for floor in (plan_start, passed_floor) if floor is not None]
    window = date_request.trailing.settled(newest, floor=max(floors, default=None))
    # The member passed over shortens the span only where its floor lies above both the
    # span's own start and the plan's first year, which would shorten it as much without it.
    shortened = passed_floor is not None and all(
        passed_floor > other
        for other in (newest - date_request.trailing.years + 1, plan_start)
        if other is not None
    )
    settled = DateRequest(window=window, latest=False)
    mat, answer = _over(mat, filtered, window, date_request=settled, coverage_index=coverage_index)
    return SettledTrailing(
        mat=mat,
        filtered=answer,
        date_request=settled,
        asked=asked,
        departure=stale_departure(date_request, answer.plan, newest=newest, shortened=shortened),
    )


def stale_departure(
    date_request: DateRequest, plan: ReadPlan | None, *, newest: int, shortened: bool
) -> tuple[Warning, DepartureFact] | None:
    """The `date_coverage_partial` warning and the `year_unavailable` fact it discloses for a
    trailing request answered from data whose newest year lies more than
    `trailing_stale_years` before the current year, or whose range a member passed over in
    election shortened (`shortened`): the years the words cover, the years shown, and for a
    stale year the years the data holds up to `newest`. A shortened range that is not stale
    states no data years (`departures.shortened_trailing_fact`), since the years before it
    can hold data. None for a newer year a full range shows."""
    now = current_year()
    stale = now - newest > scope_window_settings().trailing_stale_years
    if not stale and not shortened:
        return None
    asked = date_request.window
    assert asked is not None and asked.start_year is not None and asked.end_year is not None
    served = served_dates(plan).values() if plan is not None else ()
    firsts = [year for first, _last in served if (year := year_of(first)) is not None]
    shown = (min(firsts, default=newest), newest)
    requested = (asked.start_year, asked.end_year)
    held = data_years(plan)
    first = min(held[0], shown[0]) if held is not None else shown[0]
    if shortened:
        fact = shortened_trailing_fact(
            asked=requested, shown=shown, data_years=(first, newest) if stale else None
        )
    else:
        fact = stale_trailing_fact(asked=requested, data_years=(first, newest), shown=shown)
    reasons: list[str] = []
    if stale:
        reasons.append(
            f"the newest year the data holds is {newest}, {now - newest} years before {now}"
        )
    if shortened:
        reasons.append("earlier years would read a different series")
    warning = Warning(
        code=DATE_COVERAGE_PARTIAL,
        severity="warn",
        message=f"{' and '.join(reasons)}, so the answer shows {fact.shown}.",
    )
    return warning, fact


def passed_over_departure(
    settled: SettledTrailing, *, shown: str, first_year: int
) -> tuple[Warning, DepartureFact] | None:
    """The disclosure that a member passed over in election shortened `settled`, for an
    answer a conversion's period then narrowed to `shown`, starting in `first_year`
    (`phase_answer._over_conversion_period`): its `passed_over` fact over the years shown,
    stating no data years, which the conversion's own fact states. None where no member
    passed over shortened the span, or where the period starts after the year the passed
    over member's data ends, since the conversion's tables then set the first year shown."""
    if settled.departure is None or settled.departure[1].limited_by != "passed_over":
        return None
    window = settled.date_request.window
    assert window is not None, "a settled span reads a plain range"
    if first_year != window.start_year:
        return None
    fact = replace(settled.departure[1], shown=shown, data_years=None)
    warning = Warning(
        code=DATE_COVERAGE_PARTIAL,
        severity="warn",
        message=f"earlier years would read a different series, so the answer shows {shown}.",
    )
    return warning, fact
