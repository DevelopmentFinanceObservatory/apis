"""Settle a span counted back from now on the newest year an answer's admitted series hold.

A trailing request ("the last five years", `extract.DateRequest.trailing_years`) reads its
`window` through materialisation and the unit and facet admission. That window holds every
N-year span ending in the judged years (`DateRequest.judged`), so the members, units and facets
of whichever span the answer settles on were confirmed in that one read. Once admission has
fixed what an answer serves, its newest year Y is the newest year any served series holds
(`read_plan.newest_served_year`), one year per Spec, and the answer reads the N years ending
there, `[Y - N + 1, Y]`, over the plan it already built (`ReadPlan.windowed`), so the request
resolves once. A standard answer that elected one member starts after the newest year a member
ahead of it in election order holds (`Materialised.passed_over_newest_year`), so a resubmit of
the range elects the same member; such a span can hold fewer than N years. Its request becomes
that plain range, which the Spec states and a resubmit reads. A Y more than
`trailing_stale_years` before the current year departs `year_unavailable` naming it
(`stale_departure`). An admitted answer whose newest year lies before the judged years reads the
judged years, where it holds no point, as an answer to the asked span alone would.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from search.engine.departures import stale_trailing_fact
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
    years the request's words cover (`DateRequest.asked_window`), which a later departure
    names as the years asked, and the departure a stale newest year raises."""

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


def _from_plan_start(plan: ReadPlan, start: int, end: int) -> TimeWindow:
    """`start` to `end`, starting no earlier than `plan`'s years: a role read floors its
    years where coverage begins (`scope_window.role_read_years`), so no data precedes them,
    and a resubmit of the range floors there too."""
    years = plan.years
    if isinstance(years, TimeWindow) and years.start_year is not None:
        start = max(start, years.start_year)
    return TimeWindow(start_year=start, end_year=end)


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
    `trailing_years` years its plan serves, starting after the newest year a member passed
    over in election holds (`Materialised.passed_over_newest_year`). An answer whose newest
    year lies before the judged years is read over the judged years and keeps its request.
    None for a request that does not trail, or a plan that serves no dated point, which
    keeps its answer."""
    if date_request is None or date_request.trailing_years is None or filtered.plan is None:
        return None
    newest = newest_served_year(filtered.plan)
    if newest is None:
        return None
    judged = date_request.judged.window
    assert judged is not None and judged.start_year is not None and judged.end_year is not None
    asked = date_request.asked_window
    assert asked is not None
    if newest < judged.start_year:
        window = _from_plan_start(filtered.plan, judged.start_year, judged.end_year)
        mat, answer = _over(
            mat, filtered, window, date_request=date_request, coverage_index=coverage_index
        )
        return SettledTrailing(
            mat=mat, filtered=answer, date_request=date_request, asked=asked, departure=None
        )
    start = newest - date_request.trailing_years + 1
    if mat.passed_over_newest_year is not None:
        start = max(start, mat.passed_over_newest_year + 1)
    window = _from_plan_start(filtered.plan, start, newest)
    settled = DateRequest(window=window, latest=False)
    mat, answer = _over(mat, filtered, window, date_request=settled, coverage_index=coverage_index)
    return SettledTrailing(
        mat=mat,
        filtered=answer,
        date_request=settled,
        asked=asked,
        departure=stale_departure(date_request, answer.plan, newest=newest),
    )


def stale_departure(
    date_request: DateRequest, plan: ReadPlan | None, *, newest: int
) -> tuple[Warning, DepartureFact] | None:
    """The `date_coverage_partial` warning and the `year_unavailable` fact it discloses for a
    trailing request answered from data whose newest year lies more than
    `trailing_stale_years` before the current year: the years the words cover, the years the
    data holds up to `newest`, and the years shown. None for a newer year."""
    now = current_year()
    if now - newest <= scope_window_settings().trailing_stale_years:
        return None
    asked = date_request.asked_window
    assert asked is not None and asked.start_year is not None and asked.end_year is not None
    served = served_dates(plan).values() if plan is not None else ()
    firsts = [year for first, _last in served if (year := year_of(first)) is not None]
    shown = (min(firsts, default=newest), newest)
    held = data_years(plan)
    first = min(held[0], shown[0]) if held is not None else shown[0]
    fact = stale_trailing_fact(
        asked=(asked.start_year, asked.end_year), data_years=(first, newest), shown=shown
    )
    warning = Warning(
        code=DATE_COVERAGE_PARTIAL,
        severity="warn",
        message=(
            f"the newest year the data holds is {newest}, {now - newest} years before "
            f"{now}, so the answer shows {fact.shown}."
        ),
    )
    return warning, fact
