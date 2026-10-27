"""Decide once, at resolve time, what an answer's observation preview reads.

Every resolver builds one `ReadPlan` per answer and records it on `Materialised.read_plan`,
and every consumer reads that plan rather than derive its own:

- A `PairPlan` reads each series on its own: every candidate series `(stat_var, role
  tuple)` with its candidate facets and the group the preview dispatches for it. The
  members share one group across every place and role tuple the answer names, elected
  over every member's probe as a mint of the answer's Spec elects it (`PairPlan.probes`,
  `materialise.group_facets.elect_probes`): the group serving the most members. A series
  the group does not cover serves nothing, so a member the group leaves out serves nothing
  and its series are `PairPlan.uncovered`. A scope-defaulted read serves the one facet
  elected across every place (`facet_election.rank_scope_candidates`).
- A `BulkPlan` reads a role-bearing population whose bound pairs or series pass
  `obs_preview_max_pairs` (`search.observations.preview_mode`) in one read of one facet,
  elected over the merged role index as a group of its members
  (`facet_election.select_role_candidate`). Its pairs can run to 1.6M and are not listed.

Both carry the antichain-minimal members the preview reads and the years it reads
(`plan_years`). The years apply before anything is counted, ranked or built: a per-pair
plan builds only the series holding a point in them, and under `latest` serves each
StatVar's one latest period from the series holding it (`PairPlan.served`), as a mint of
the answer's Spec does. A role-bearing plan elects that year for each member once, over
the rows it holds before any reduction (`elected_years`), and each of its `latest`
reductions, the headline and the download read the year there. Every facet election ranks
a facet by the series it holds a point at in the years, then by the years of them its span
covers, then by its count (`materialise.facets.rank_candidates`, the election a mint of
the answer's Spec runs under
the same window). A request's facet filter narrows a plan once (`narrowed`), and
every consumer reads the narrowed plan. A unit request is decided over the same probes
(`region.convertibility.decide_spec_unit`), its elected units narrow the plan, and its
conversion preview dispatches the narrowed plan's groups. A conversion into each entity's
own currency then leaves out the rows of entities no currency names (`omitting`), so every
count, date, member and source an answer states is one of the rows it serves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

from code_index.shared import get_index
from materialise.facets import (
    MemberProbe,
    overlaps_window,
    read_variable_breadths,
    subject_probe,
)
from materialise.group_facets import elect_probes, is_populated
from materialise.window import RequestedWindow, latest_period, latest_periods
from search.engine import extract as _extract
from search.engine.clients.graph import Facet
from search.engine.reading_coverage import CoverageYears, ReadingCoverage, observed_years
from search.engine.retrieval.elected_dataset import elected_datasets, publishes
from search.engine.retrieval.facet_election import (
    RoleBulkCandidate,
    merge_scope_candidates,
    rank_scope_candidates,
    role_continuation,
    role_group_selection,
    scope_served,
    select_role_candidate,
    window_of,
)
from search.models import TimeWindow, in_window
from search.observations import PreviewMode, preview_mode
from search.scope_window import read_range as _read_range
from search.scope_window import role_read_years
from series_coverage.schema import is_sub_annual
from statvar_members import AbsentReading, antichain_minimal_ids

if TYPE_CHECKING:
    from search.engine.clients.sdmx import RoleFacetCandidate, RoleTuple
    from search.engine.extract import DateRequest
    from search.engine.retrieval.member_facts import MemberFacts
    from series_coverage.index import BoundRoles
    from series_coverage.index import Role as CoverageRole
    from series_coverage.schema import YearHorizon

type PlanYears = CoverageYears
"""The years a preview reads: a year range, `"latest"` for each StatVar's one latest period
(`materialise.window.latest_period`), or None for every date."""

type YearBounds = tuple[int | None, int | None]
"""A read's first and last years, None for an open bound."""

type Admits = Callable[[str, Facet | RoleFacetCandidate], bool]
"""A request's facet filter as one rule: whether it admits a candidate of a StatVar."""

type Omission = tuple[str, str, str]
"""`(stat_var, graph role, entity)`: the rows of a StatVar binding an entity on a role, which
a conversion into each entity's own currency leaves out because no single currency names the
entity (`region.convertibility.UnitCoverage.omitted`)."""

type Omissions = frozenset[Omission]

type Election = Callable[[ReadPlan], Mapping[str, int]]
"""The year each member of a role-bearing plan elects for a `latest` read, counted over the
rows that plan holds before any reduction (`retrieval.coverage.plan_election`). A member
that elects no year is absent and serves its own latest period."""


def _omitted_by_role(omitted: Omissions, stat_var: str) -> dict[str, frozenset[str]]:
    """The entities `omitted` leaves out of `stat_var`'s rows, by graph role."""
    by_role: dict[str, set[str]] = {}
    for sv, role, entity in omitted:
        if sv == stat_var:
            by_role.setdefault(role, set()).add(entity)
    return {role: frozenset(entities) for role, entities in by_role.items()}


def current_year() -> int:
    """Read through `search.engine.extract.date` so tests share its pinned clock."""
    return _extract.date.today().year


class NoYearsToRead(Exception):
    """A range request whose bounds, once floored and capped to the coverage horizon, leave
    no year to read (`read_range`). The resolver answers `no_observations`."""


def read_range(
    window: TimeWindow, *, first_year: int | None = None, last_year: int | None = None
) -> TimeWindow | None:
    """The years a range read covers for `window`, on every route, at this module's clock
    (`search.scope_window.read_range`). A plan's years are therefore the years a mint of the
    answer's Spec elects and reads under."""
    return _read_range(
        window, clock_year=current_year(), first_year=first_year, last_year=last_year
    )


def plan_years(
    date_request: DateRequest | None,
    *,
    first_year: int | None = None,
    last_year: int | None = None,
) -> PlanYears:
    """The years a preview for `date_request` reads, on every route: a range from
    `read_range`, whose bounds a role read passes from its coverage horizon. Raise
    `NoYearsToRead` when the range holds no year."""
    if date_request is None:
        return None
    if date_request.latest:
        return "latest"
    assert date_request.window is not None  # DateRequest's own either/or invariant
    years = read_range(date_request.window, first_year=first_year, last_year=last_year)
    if years is None:
        raise NoYearsToRead
    return years


def reads_dates(date_request: DateRequest | None) -> bool:
    """Whether a facet read that elects for `date_request` reads each facet's dates: under a
    range alone, the one window an election judges by dates
    (`materialise.facets.overlaps_window`), as a mint's probe does."""
    return date_request is not None and not date_request.latest


@dataclass(frozen=True)
class RoleYears:
    """A role read's years for one request over a coverage horizon (`role_years`): the
    plan's years, and for a range the SDMX year list (`listed`), every year in it."""

    years: PlanYears
    listed: tuple[str, ...] | None

    @property
    def bounds(self) -> tuple[int | None, int | None]:
        """The first and last year a coverage lookup confirms on; both None for a
        `latest` or dateless read, which confirms on every year."""
        if isinstance(self.years, TimeWindow):
            return self.years.start_year, self.years.end_year
        return None, None


def role_years(date_request: DateRequest | None, horizon: YearHorizon) -> RoleYears:
    """The years a role read confirms on, plans and lists for `date_request`, on every
    role route: for a range, `search.scope_window.role_read_years`, the rule the Datasets
    mint admits a window by. Raise `NoYearsToRead` when the range holds no year, and
    `search.scope_window.WindowTooLong` when it holds more than `scope_max_window_years`."""
    if date_request is None or date_request.latest:
        return RoleYears(years=plan_years(date_request), listed=None)
    assert date_request.window is not None  # DateRequest's own either/or invariant
    years = role_read_years(date_request.window, horizon=horizon, clock_year=current_year())
    if years is None:
        raise NoYearsToRead
    assert years.start_year is not None and years.end_year is not None  # both bounds set
    listed = tuple(str(y) for y in range(years.start_year, years.end_year + 1))
    return RoleYears(years=years, listed=listed)


def plan_window(years: PlanYears) -> RequestedWindow:
    """`years` as the window materialise ranks and reads a series with."""
    if years is None:
        return RequestedWindow(kind="all")
    if years == "latest":
        return RequestedWindow(kind="latest")
    return RequestedWindow(kind="range", start_year=years.start_year, end_year=years.end_year)


def _absent_readings() -> Mapping[str, AbsentReading]:
    """Map each code system to its `absent_reading` for `enumerate_members`.

    Read from the in-memory CodeIndex on each call. This keeps `statvar_members` independent
    of `code_index`.
    """
    return {
        code_system: declaration.absent_reading
        for code_system, declaration in get_index().declarations.items()
    }


def minimal_constraint_svs(
    sv_dcids: Sequence[str], member_constraints: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Return antichain-minimal members, preserving `sv_dcids` order.

    This keeps a family's total and part from appearing together. Members missing from
    `member_constraints` have unknown constraint sets, so the antichain keeps them. An empty
    default could remove a valid sibling.
    """
    readings = _absent_readings()
    keys_by_sv = {
        sv: frozenset(member_constraints[sv]) for sv in sv_dcids if sv in member_constraints
    }
    return list(antichain_minimal_ids(keys_by_sv, sv_dcids, readings=readings))


@dataclass(frozen=True)
class PlanRoles:
    """A role-bearing read's bound roles in coverage terms (`bound`), and the graph role
    each coverage role names on the wire (`role_names`), which the headline reads."""

    bound: BoundRoles
    role_names: Mapping[CoverageRole, str]


@dataclass(frozen=True)
class PlanSeries:
    """One series a per-series preview can read: `stat_var` at the entities `role_tuple`
    binds, its candidate facets, and the group the preview dispatches for it (`group`): the
    facets of its member's one group that the series holds, empty when it holds none.

    `rows` is the series' recipient and donor for the headline's served rows, `None` on a
    role its variable does not declare, and is itself None for a series that is not
    role-bearing. A role-bearing series binds every role the shape places, an open one
    included, so it is one series per full role tuple, as the bulk read serves it.
    """

    stat_var: str
    role_tuple: RoleTuple
    candidates: tuple[Facet, ...]
    rows: tuple[str | None, str | None] | None = None
    group: tuple[Facet, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        """`(stat_var, the role tuple's dcids joined by "|")`."""
        return self.stat_var, "|".join(dcid for _role, dcid in self.role_tuple)

    @property
    def group_ids(self) -> frozenset[str]:
        return frozenset(f.facet_id for f in self.group if f.facet_id is not None)

    @property
    def latest_period(self) -> str | None:
        """The newest period its group's facets state (`materialise.window.latest_period`
        over their latest dates), the one a read of the series alone serves under
        `latest`."""
        return latest_period(f.latest_date for f in self.group)

    @property
    def unservable(self) -> bool:
        """Whether the series holds data that no read can dispatch: no candidate carries a
        facet id, the key every read dispatches by. A series whose facets all lie outside
        its member's group is dispatchable and not served (`PairPlan.uncovered`)."""
        return any(f.obs_count > 0 for f in self.candidates) and not any(
            f.facet_id is not None for f in self.candidates
        )


def has_points(facets: Sequence[Facet], years: PlanYears) -> bool:
    """Whether any of `facets` holds a point under `years`, by the rule every election
    narrows a window with (`materialise.facets.overlaps_window`): its dates when it carries
    them all, else its span."""
    window = plan_window(years)
    return any(f.obs_count > 0 and overlaps_window(f, window=window) for f in facets)


def points_in_years(facet: Facet, years: PlanYears) -> int:
    """The observations `facet` holds under `years`, the count option coverage states.
    Inside a range: its dates there when it carries them all, else its whole count when it
    holds a point there (`has_points`), the most its points can reach, and zero without one.
    Under `latest` or every date: its whole count."""
    if not isinstance(years, TimeWindow):
        return facet.obs_count
    if not has_points([facet], years):
        return 0
    if facet.dates and facet.dates_complete:
        return sum(1 for d in facet.dates if in_window(d, years))
    return facet.obs_count


type ServedDates = tuple[str | None, str | None]
"""The first and last dates of a set of points, None for a bound no facet states."""


def _hull(spans: Iterable[ServedDates]) -> ServedDates | None:
    """The earliest first and latest last date across `spans`, or None when none states one."""
    spans = list(spans)
    firsts = [first for first, _last in spans if first]
    lasts = [last for _first, last in spans if last]
    if not firsts and not lasts:
        return None
    return min(firsts, default=None), max(lasts, default=None)


def facet_dates(facet: Facet, years: PlanYears) -> ServedDates | None:
    """The first and last dates `facet` holds under `years` by the point rule (`has_points`),
    or None when it holds no point there. Inside a range, a facet carrying all its dates
    gives the dates in the range, and one without gives its span clipped to the range, the
    most its points can reach. An every-date or `latest` read gives its whole span."""
    if not has_points([facet], years):
        return None
    if not isinstance(years, TimeWindow):
        return (
            facet.earliest_date or min(facet.dates, default=None),
            facet.latest_date or max(facet.dates, default=None),
        )
    if facet.dates and facet.dates_complete:
        inside = [d for d in facet.dates if in_window(d, years)]
        return min(inside), max(inside)
    first, last = facet.earliest_date, facet.latest_date
    if years.start_year is not None and (
        (year := year_of(first)) is None or year < years.start_year
    ):
        first = str(years.start_year)
    if years.end_year is not None and ((year := year_of(last)) is None or year > years.end_year):
        last = str(years.end_year)
    return first, last


@dataclass(frozen=True)
class WindowGap:
    """Data held at the place with no point in the years asked: those years (`asked_years`),
    and the first and last dates of the series an every-date read of the held facets serves
    (None when it serves none). A carrier holding one answers `no_observations` for the
    window, its `year_unavailable` fact built from it alone
    (`engine.departures.window_gap_fact`)."""

    years: TimeWindow
    first: str | None
    last: str | None


def asked_years(date_request: DateRequest | None) -> TimeWindow | None:
    """The years a range request asks for, read as the standard route reads them
    (`read_range` with no coverage floor or cap): an open start stays open, and an open end
    stops at the current year unless the request starts after it. Every window answer names
    these; a role read's horizon-clamped years (`role_years`) serve its read alone. None
    for a latest or dateless request, or a range holding no year."""
    if date_request is None or date_request.window is None:
        return None
    return read_range(date_request.window)


def window_gap(plan: ReadPlan | None, years: TimeWindow) -> WindowGap:
    """The gap the years asked (`asked_years`) leave in an answer whose read found no point
    in the years it read. Only a range leaves one, so `years` is a range: a `latest` or
    every-date read holds a point wherever data is held (`has_points`).

    `plan` is the plan the same answer builds with the window removed. The span is the
    first and last dates that plan serves (`served_dates`), so the departure names the years a
    dateless answer states and runs no election of its own. None states no span."""
    first, last = (_hull(served_dates(plan).values()) if plan is not None else None) or (
        None,
        None,
    )
    return WindowGap(years=years, first=first, last=last)


def plan_window_gap(plan: ReadPlan, date_request: DateRequest | None) -> WindowGap | None:
    """The gap the years a range request asks (`asked_years`) leave in what `plan` holds, for
    a plan that serves nothing (`serves_nothing`): the span the same series serve with the
    window removed (`window_gap` over `PairPlan.every_date`). A plan narrowed to a pinned
    source holds that source's candidates alone, so the gap names the span the pinned series
    runs. None for a `latest` or dateless request, for a bulk plan, whose candidates hold the
    plan's years alone, and for a plan holding no data."""
    years = asked_years(date_request)
    if years is None or not isinstance(plan, PairPlan):
        return None
    if not any(f.obs_count > 0 for s in plan.series for f in s.candidates):
        return None
    return window_gap(plan.every_date(), years)


def year_of(date: str | None) -> int | None:
    """The leading year of `date`, or None when it has none."""
    try:
        return int(str(date)[:4])
    except (ValueError, TypeError):
        return None


def _scope_group(
    series: Sequence[PlanSeries], *, members: Sequence[str], years: PlanYears
) -> tuple[tuple[str, ...], frozenset[str]]:
    """The facet ids a scope-defaulted read serves at every place, and the selection a mint
    of the answer's Spec serves (`facet_election.scope_served`). The read serves the one
    facet elected across every place, led by the places it holds a point at under `years`
    (`facet_election.rank_scope_candidates`), then the facets continuing the selection past
    each series' last year, which the mint dispatches too. Empty when no place holds a
    facet."""
    facets_by_entity: dict[str, list[Facet]] = {}
    for s in series:
        facets_by_entity.setdefault(s.role_tuple[0][1], []).extend(s.candidates)
    merged = merge_scope_candidates(facets_by_entity, list(facets_by_entity))
    if not merged:
        return (), frozenset()
    window = plan_window(years)
    lead = rank_scope_candidates(merged, stat_vars=members, window=window)[0]
    selection, continuing = scope_served(lead, merged, window=window)
    return (lead.facet_id, *continuing), frozenset(selection)


def series_probes(
    series: Sequence[PlanSeries],
    *,
    places: Sequence[str] = (),
    deadline: float | None = None,
) -> dict[str, MemberProbe]:
    """Each member's probe from its own series: their candidates merged by facet id and each
    facet's count of the series it holds a point at (`materialise.facets.subject_probe`, the
    probe a mint builds from its bound places). A standard plan's series are every place the
    answer names, so the plan and a mint rank the same candidates. When the answer names one
    place, `places`, each probe carries its variable's whole breadth, read by `deadline` only
    if a tie needs it, as a mint of the answer's Spec reads it
    (`materialise.facets.read_variable_breadths`)."""
    by_sv: dict[str, dict[str, tuple[Facet, ...]]] = {}
    for s in series:
        by_sv.setdefault(s.stat_var, {})[s.key[1]] = s.candidates
    breadths = read_variable_breadths(list(by_sv), places=places, deadline=deadline)
    return {
        sv: replace(subject_probe(by_series, list(by_series)), variable_breadth=breadths.get(sv))
        for sv, by_series in by_sv.items()
    }


def _member_groups(
    probes: Mapping[str, MemberProbe], *, years: PlanYears, index_sourced: bool
) -> tuple[dict[str, tuple[Facet, ...]], dict[str, frozenset[str]], tuple[str, ...]]:
    """The facets of the one group every member's preview dispatches at every series, in
    serving order, and each member's selection (`PairPlan.selections`): one election over
    every member's probe under `years` (`materialise.group_facets.elect_probes`), led by the
    series each facet holds a point at summed across the members, with a tie broken on the
    facets' places across the whole variables. The elected facets come first and the facets
    continuing them past their last year after (`materialise.facets.continuation`), which a
    mint of the answer's Spec dispatches too. A member the group does not serve holds no
    facet and no selection, and the populated ones among them are returned last, in member
    order (`GroupElection.unserved`). `index_sourced` lets an index-resolved family's members agree on
    provenance and unit alone, as its mint does (`MaterialisationRequest.index_sourced`)."""
    group = elect_probes(probes, window=plan_window(years), index_sourced=index_sourced)
    if group is None:
        return dict.fromkeys(probes, ()), {}, ()
    groups: dict[str, tuple[Facet, ...]] = {}
    selections: dict[str, frozenset[str]] = {}
    for member, (sv, probe) in enumerate(probes.items()):
        groups[sv] = group.served(member, probe.candidates)
        if selected := group.selected(member):
            selections[sv] = selected
    stat_vars = list(probes)
    return groups, selections, tuple(stat_vars[member] for member in group.unserved)


def _elect_groups(
    series: Sequence[PlanSeries],
    *,
    members: Sequence[str],
    years: PlanYears,
    scope: bool,
    probes: Mapping[str, MemberProbe],
    index_sourced: bool,
) -> tuple[tuple[PlanSeries, ...], str | None, dict[str, frozenset[str]], tuple[str, ...]]:
    """Each series with the group its preview dispatches, a scope read's one facet, each
    member's selection (`PairPlan.selections`), and the members the group leaves out
    (`PairPlan.unserved`), none on a scope read. A series holds its member's group where it
    has those facets, and no group elsewhere."""
    if scope:
        group, selection = _scope_group(series, members=members, years=years)
        return (
            tuple(replace(s, group=_group_of(s.candidates, group)) for s in series),
            group[0] if group else None,
            dict.fromkeys(members, selection),
            (),
        )
    groups, selections, unserved = _member_groups(probes, years=years, index_sourced=index_sourced)
    ids = {
        sv: tuple(f.facet_id for f in group if f.facet_id is not None)
        for sv, group in groups.items()
    }
    return (
        tuple(replace(s, group=_group_of(s.candidates, ids.get(s.stat_var, ()))) for s in series),
        None,
        selections,
        unserved,
    )


def _group_of(candidates: Sequence[Facet], group: Sequence[str]) -> tuple[Facet, ...]:
    """The `candidates` in `group`, in its serving order, so a series' group leads with the
    facet it was elected by (`series_facets`)."""
    position = {facet_id: index for index, facet_id in enumerate(group)}
    return tuple(
        sorted(
            (f for f in candidates if f.facet_id in position),
            key=lambda f: position[f.facet_id or ""],
        )
    )


def _elected_years(plan: ReadPlan) -> Mapping[str, int] | None:
    """The year each member of `plan` elects for a `latest` read (`plan.elect`), or None
    for a plan holding no election or reading a range or every date, which no election
    dates."""
    if plan.elect is None or plan.roles is None or plan.years != "latest":
        return None
    return plan.elect(plan)


def holds_year(series: PlanSeries, year: int) -> bool:
    """Whether `series`' group holds a point in `year` (`has_points`)."""
    return has_points(series.group, TimeWindow(start_year=year, end_year=year))


@dataclass(frozen=True)
class PairPlan:
    """A per-series read: every candidate series of `members` with its member's elected
    group.

    `held` keeps the series whose group has a point under `years`, and `served` those of them
    the preview dispatches. That is every one, except that under `latest` a StatVar serves
    one period, from the series holding it: the year it elected (`elected_years`), or
    without one the newest period its held series reach. `elect` is the election a
    role-bearing plan was built with, which every plan derived from it runs over its own
    rows. A scope-defaulted read (`scope`) serves `scope_facet_id` at every place.
    `selections` holds each member's facet ids a mint of the answer's Spec serves before any
    continuation: those matching the one group's selections at its cadence, absent for a
    member the group does not serve. A series' group holds all of them, a scope read's only
    `scope_facet_id`. `unserved` holds the populated members the one group leaves out, in
    member order (`GroupElection.unserved`), and `left_out` the members a unit decision
    leaves out (`leaving_out`), whose candidates the filter then removed: both serve nothing,
    and the answer names them (`unserved_members`). `index_sourced` marks an index-resolved
    family, whose group agrees on provenance and unit alone (`in_elected_datasets`).
    `roles` is set for a role-bearing read. `probes` holds each member's election input
    (`plan_pairs`): its candidates merged across every place and role tuple the answer
    names, and each facet's series count. A unit decision reads the same probes. `places`
    are the places a standard read was asked for, with or without a series
    (`unserved_entities`); a role-bearing read was asked for the entities `roles` binds.
    `omitted` are the rows a conversion leaves out (`omitting`), whose series the plan no
    longer holds.
    """

    members: tuple[str, ...]
    years: PlanYears
    series: tuple[PlanSeries, ...] = ()
    scope: bool = False
    scope_facet_id: str | None = None
    selections: Mapping[str, frozenset[str]] = field(default_factory=dict)
    unserved: tuple[str, ...] = ()
    left_out: frozenset[str] = frozenset()
    roles: PlanRoles | None = None
    places: tuple[str, ...] = ()
    omitted: Omissions = frozenset()
    index_sourced: bool = False
    probes: Mapping[str, MemberProbe] = field(default_factory=dict, compare=False)
    elect: Election | None = field(default=None, compare=False, repr=False)

    @property
    def held(self) -> tuple[PlanSeries, ...]:
        """The series whose group holds a point under `years`: the rows a read of every
        series fetches before a `latest` window reduces them."""
        return tuple(s for s in self.series if has_points(s.group, self.years))

    @cached_property
    def elected_years(self) -> Mapping[str, int] | None:
        """The year each member elects for a `latest` read of this plan, counted once over
        its held series less the rows it omits (`elect`). A member absent from it serves
        its own latest period. None for a plan that is not role-bearing, holds no election
        or reads a range or every date."""
        return _elected_years(self)

    def elected_year(self, sv: str) -> int | None:
        """The year this plan's `latest` read serves `sv` in by election (`elected_years`),
        or None when the plan reads other years or `sv` elected none."""
        if self.years != "latest":
            return None
        return (self.elected_years or {}).get(sv)

    @property
    def served(self) -> tuple[PlanSeries, ...]:
        """The series the preview reads and the answer states. Under `latest`, the held
        series at their StatVar's one period, as a mint keeps a StatVar's rows at that
        period alone (`materialise.columnar.latest_rows`). A member that elected its year
        (`elected_year`) serves the series whose group holds a point in it (`holds_year`).
        Any other serves the series at its latest period
        (`materialise.window.latest_periods` over each one's `PlanSeries.latest_period`): a
        series whose own latest period is earlier serves nothing, and a StatVar none of
        whose held series states a latest date keeps them all. Otherwise every held
        series."""
        held = self.held
        if self.years != "latest":
            return held
        elected = self.elected_years or {}
        latest = latest_periods(
            (s.stat_var, s.latest_period) for s in held if s.stat_var not in elected
        )
        return tuple(
            s
            for s in held
            if (
                holds_year(s, elected[s.stat_var])
                if s.stat_var in elected
                else latest.get(s.stat_var) in (None, s.latest_period)
            )
        )

    def series_dates(self, series: PlanSeries) -> ServedDates | None:
        """The first and last dates of the points this plan serves of `series`, one of its
        served series: the year its member elected (`elected_year`), else the dates
        `series_dates` reads under the plan's years."""
        year = self.elected_year(series.stat_var)
        if year is not None:
            return str(year), str(year)
        return series_dates(series, self.years)

    @property
    def unservable(self) -> tuple[PlanSeries, ...]:
        """The series holding a point under `years` that no facet can dispatch
        (`PlanSeries.unservable`). A scope read has none: a place lacking its one facet is
        outside the read by design."""
        if self.scope:
            return ()
        return tuple(
            s for s in self.series if s.unservable and has_points(s.candidates, self.years)
        )

    @property
    def uncovered(self) -> tuple[PlanSeries, ...]:
        """The series holding a point under `years` in a facet outside their member's group,
        which the answer names and serves no data for. A scope read has none, as it has no
        `unservable` series."""
        if self.scope:
            return ()
        return tuple(
            s
            for s in self.series
            if not s.unservable
            and not has_points(s.group, self.years)
            and has_points(s.candidates, self.years)
        )

    def omitting(self, omitted: Omissions) -> PairPlan:
        """This plan without the series binding an entity whose rows `omitted` leaves out of
        their StatVar. Every group stays as elected, as the read elects over the member's
        whole probe before it leaves those rows out. Under `latest` the rest then decide
        each StatVar's period (`elected_years`, `served`), as a read omits before it
        reduces."""
        if not omitted:
            return self
        return replace(
            self,
            series=tuple(
                s
                for s in self.series
                if not any((s.stat_var, role, dcid) in omitted for role, dcid in s.role_tuple)
            ),
            omitted=self.omitted | omitted,
        )

    def narrowed(self, admits: Admits) -> PairPlan:
        """This plan under a request's facet filter: each series and each member's probe
        keep the candidates `admits` accepts, and every group is elected again from them."""
        narrowed = plan_pairs(
            [
                replace(s, candidates=tuple(c for c in s.candidates if admits(s.stat_var, c)))
                for s in self.series
            ],
            members=self.members,
            years=self.years,
            scope=self.scope,
            roles=self.roles,
            places=self.places,
            probes={
                sv: replace(probe, candidates=tuple(c for c in probe.candidates if admits(sv, c)))
                for sv, probe in self.probes.items()
            }
            or None,
            index_sourced=self.index_sourced,
            elect=self.elect,
        )
        return replace(narrowed, omitted=self.omitted, left_out=self.left_out)

    def every_date(self) -> PairPlan:
        """This plan's series and probes read over every date: the plan the same answer
        builds with its window removed."""
        return self.windowed(None)

    def windowed(self, years: PlanYears) -> PairPlan:
        """This plan's series and probes read over `years`, every group elected again under
        them: the plan the same answer builds under that window, for a window inside the
        plan's own years, whose series it then holds."""
        dated = plan_pairs(
            self.series,
            members=self.members,
            years=years,
            scope=self.scope,
            roles=self.roles,
            places=self.places,
            probes=self.probes or None,
            index_sourced=self.index_sourced,
            elect=self.elect,
        )
        return replace(dated, omitted=self.omitted, left_out=self.left_out)

    def leaving_out(self, members: Collection[str]) -> PairPlan:
        """This plan naming `members` as left out by the answer's unit decision
        (`left_out`)."""
        return replace(self, left_out=self.left_out | frozenset(members))


@dataclass(frozen=True)
class BulkPlan:
    """A bulk role read: one group's facets across every pair of `members`.

    `candidates` are the facets merged over the role index the read builds for `members`
    inside `read_years`, the SDMX year list (None for `latest` and dateless reads, and for a
    range too long to list, which no bulk read can serve). `facet_id` leads the election
    over them (`facet_election.select_role_candidate`), and `selected` holds each served
    member's facet ids the group's selection serves, `facet_id` first where the member holds
    it (`facet_election.role_group_selection`): under `index_sourced` a member holding the
    group's provenance and unit under another method's facet serves that facet. `continuing`
    holds the facets continuing the lead past its last year
    (`facet_election.role_continuation`), which the read dispatches beside the selection, as
    a mint of the answer's Spec does (`role_election`). `unserved` holds the populated
    members the group leaves out (`GroupElection.unserved`), as `PairPlan.unserved` does. `omitted` are the
    rows a conversion leaves out (`omitting`), whose pairs the candidates no longer hold,
    and `omitted_served` the members one of whose omitted rows falls in their years
    (`member_years`). Under `latest` each member serves one year of its own
    (`member_year`), as a mint reduces each StatVar to its own period. `elect` is the
    election the plan was built with, which every plan derived from it runs over its own
    rows (`elected_years`). `index_sourced` marks an index-resolved family, whose members
    agree on provenance and unit alone (`in_elected_datasets`). `left_out` holds the members
    a unit decision leaves out (`leaving_out`), as `PairPlan.left_out` does.
    """

    members: tuple[str, ...]
    years: PlanYears
    read_years: tuple[str, ...] | None = None
    facet_id: str | None = None
    selected: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    unserved: tuple[str, ...] = ()
    continuing: tuple[str, ...] = ()
    candidates: Mapping[str, RoleBulkCandidate] = field(default_factory=dict, compare=False)
    roles: PlanRoles | None = None
    omitted: Omissions = frozenset()
    omitted_served: frozenset[str] = frozenset()
    index_sourced: bool = False
    left_out: frozenset[str] = frozenset()
    elect: Election | None = field(default=None, compare=False, repr=False)

    def leaving_out(self, members: Collection[str]) -> BulkPlan:
        """This plan naming `members` as left out by the answer's unit decision
        (`left_out`)."""
        return replace(self, left_out=self.left_out | frozenset(members))

    @cached_property
    def elected_years(self) -> Mapping[str, int] | None:
        """The year each member elects for a `latest` read of this plan, counted once over
        every pair of the facets it dispatches less the rows it omits (`elect`). A member
        absent from it serves its own latest year (`member_year`). None for a plan that
        holds no election or reads a range or every date."""
        return _elected_years(self)

    @property
    def elected(self) -> RoleBulkCandidate | None:
        """The candidate leading the group the bulk read serves, or None with no
        candidate."""
        return self.candidates.get(self.facet_id) if self.facet_id is not None else None

    @property
    def served_facet_ids(self) -> tuple[str, ...]:
        """Every facet id the group's selection serves at some member (`selected`), the
        lead first, then in member order."""
        return tuple(
            dict.fromkeys(facet_id for sv in self.members for facet_id in self.selected.get(sv, ()))
        )

    @property
    def served_facets(self) -> tuple[RoleBulkCandidate, ...]:
        """The candidates of `served_facet_ids` holding a pair with data in the years the
        read serves a member it is selected for (`member_years`): the facets the preview
        reads. Under `latest` a selected facet holding no pair in its member's one year
        serves nothing."""
        return tuple(
            candidate
            for facet_id in self.served_facet_ids
            if (candidate := self.candidates.get(facet_id)) is not None
            and any(
                c.pairs.n_with_data(first_year=first, last_year=last)
                for sv in self.served_members
                if facet_id in self.selected.get(sv, ())
                for first, last in (self.member_years(sv),)
                for c in candidate.by_sv.get(sv, ())
            )
        )

    def selected_candidates(self, sv: str) -> list[RoleFacetCandidate]:
        """`sv`'s pairs under the facets the group's selection serves it (`selected`)."""
        return [
            c
            for facet_id in self.selected.get(sv, ())
            if (candidate := self.candidates.get(facet_id)) is not None
            for c in candidate.by_sv.get(sv, ())
        ]

    @property
    def served_members(self) -> tuple[str, ...]:
        """The members the read serves: those holding pairs under a facet the group's
        selection serves them (`selected_candidates`), in member order. A filter narrowing
        the plan per StatVar can drop a member while the group stays elected, so coverage,
        the headline and the bulk read all take their members here."""
        return tuple(sv for sv in self.members if self.selected_candidates(sv))

    def member_year(self, sv: str) -> int | None:
        """The year a `latest` read serves `sv` in: the year it elected (`elected_years`),
        or for a member that elected none the year of its latest period under the elected
        facet, the leading year of `materialise.window.latest_period` over its pairs'
        latest dates. Both are read after any omission (`omitting`). Coverage knows only
        years, so the plan fixes the year and the bulk read reduces `sv`'s points within it
        by the same definition. None for a member the facets its group's selection serves
        it hold no dated pair of."""
        if self.elected is None:
            return None
        if (year := (self.elected_years or {}).get(sv)) is not None:
            return year
        return year_of(latest_period(c.latest_date for c in self.selected_candidates(sv)))

    def member_years(self, sv: str) -> YearBounds:
        """The years the read serves `sv` in: under `latest` its one year alone
        (`member_year`), as a mint reduces each StatVar to its own period, else the plan's
        range, unbounded for a dateless read."""
        if self.years == "latest":
            year = self.member_year(sv)
            return year, year
        if isinstance(self.years, TimeWindow):
            return self.years.start_year, self.years.end_year
        return None, None

    def served_candidates(self, sv: str) -> list[RoleFacetCandidate]:
        """`sv`'s candidates the read dispatches: those of the facets the group's selection
        serves it (`selected_candidates`), then those of the facets continuing the lead
        (`continuing`) that the selection does not already serve."""
        selected = self.selected.get(sv, ())
        return [
            *self.selected_candidates(sv),
            *(
                c
                for facet_id in self.continuing
                if facet_id not in selected
                for c in self.candidates[facet_id].by_sv.get(sv, ())
            ),
        ]

    def member_pairs(self) -> Iterator[tuple[str, RoleFacetCandidate, YearBounds]]:
        """Each served member's pairs under the facets the read dispatches
        (`served_candidates`), one candidate at a time, with the years the read serves that
        member in (`member_years`). Every member, entity and span a bulk plan states reads
        its pairs here. A pair two of the facets hold is yielded under each, so a count
        across them reads `counted_pairs`."""
        for sv in self.served_members:
            years = self.member_years(sv)
            for candidate in self.served_candidates(sv):
                yield sv, candidate, years

    @property
    def merges(self) -> bool:
        """Whether a member's pairs read more than one facet: a facet continuing the lead
        (`continuing`), or a selection of several (`selected`). Such a plan counts its pairs
        once across their facets (`counted_pairs`)."""
        return bool(self.continuing) or any(len(ids) > 1 for ids in self.selected.values())

    @cached_property
    def elected_pairs(self) -> Mapping[str, Mapping[RoleTuple, int]]:
        """Each served member's role tuples with an observation under the facets the group's
        selection serves it (`selected_candidates`) in its years, each with the observations
        it serves there: one facet's own, or across several the points a mint keeps of them
        (`_collapsed_counts`). Empty with no election."""
        counted: dict[str, Mapping[RoleTuple, int]] = {}
        for sv in self.served_members:
            candidates = self.selected_candidates(sv)
            years = self.member_years(sv)
            if len(self.selected.get(sv, ())) > 1:
                counted[sv] = _collapsed_counts(candidates, *years)
            else:
                counted[sv] = _tuple_counts(candidates, *years)
        return counted

    @property
    def counted_pairs(self) -> Mapping[str, Mapping[RoleTuple, int]]:
        """Each served member's role tuples with an observation the read serves in its
        years, each with the observations it serves there, counted once across the facets
        they read: `continued_pairs` with a continuation, else `elected_pairs`."""
        return self.continued_pairs if self.continuing else self.elected_pairs

    def continued_years(
        self, sv: str
    ) -> Iterator[tuple[str, RoleTuple, tuple[tuple[int, int], ...]]]:
        """Each role tuple of `sv` under a facet continuing the lead that the group's
        selection does not serve it, with the facet id and the observations by year it keeps
        there: those in the member's years (`member_years`) after the tuple's last year under
        the selected facets, every one where they hold none, as
        `materialise.columnar.continuation_mask` keeps a continuation's rows. Coverage counts
        by year, so a year the selected facets hold keeps none of the continuation. A tuple
        keeping no year is left out."""
        first, last = self.member_years(sv)
        lead = self.selected_candidates(sv)
        ends: dict[RoleTuple, int] = {}
        for c in lead:
            for role_tuple, by_year in c.pairs.year_counts(first_year=first, last_year=last):
                year = by_year[-1][0]
                ends[role_tuple] = max(year, ends.get(role_tuple, year))
        for c in self.served_candidates(sv)[len(lead) :]:
            for role_tuple, by_year in c.pairs.year_counts(first_year=first, last_year=last):
                end = ends.get(role_tuple)
                if kept := tuple((y, n) for y, n in by_year if end is None or y > end):
                    yield c.facet_id, role_tuple, kept

    @cached_property
    def continued_pairs(self) -> Mapping[str, Mapping[RoleTuple, int]]:
        """Each served member's role tuples with an observation the read serves in its
        years, each with the observations it serves there, under a plan with `continuing`
        facets: the selected facets' (`elected_pairs`), then those a continuing facet keeps
        after the selected facets' (`continued_years`). Each facet's tuples are read once
        with their observations by year (`RolePairs.year_counts`), as ranking the joined
        candidate decoded every tuple (`facet_election.rank_role_candidates`).
        Empty without `continuing`."""
        if not self.continuing:
            return {}
        counted: dict[str, Mapping[RoleTuple, int]] = {}
        for sv in self.served_members:
            counts = dict(self.elected_pairs[sv])
            for _facet_id, role_tuple, by_year in self.continued_years(sv):
                counts[role_tuple] = counts.get(role_tuple, 0) + sum(n for _y, n in by_year)
            counted[sv] = counts
        return counted

    @property
    def population(self) -> int:
        """The series the read can serve: one per full role tuple with an observation in its
        member's years under the facets the read dispatches (`member_pairs`, or
        `counted_pairs` when its pairs read several facets (`merges`)). The bulk read's
        truncation baseline."""
        if self.merges:
            return sum(len(counts) for counts in self.counted_pairs.values())
        return sum(
            c.pairs.n_with_data(first_year=first, last_year=last)
            for _sv, c, (first, last) in self.member_pairs()
        )

    @property
    def observation_count(self) -> int:
        """The observations the read serves across every pair in its member's years
        (`member_pairs`, or `counted_pairs` when its pairs read several facets)."""
        if self.merges:
            return sum(sum(counts.values()) for counts in self.counted_pairs.values())
        return sum(
            c.pairs.n_obs(first_year=first, last_year=last)
            for _sv, c, (first, last) in self.member_pairs()
        )

    @property
    def max_pair_observations(self) -> int:
        """The most observations the read serves at one full role tuple in its member's
        years (`member_pairs`, or `counted_pairs` when its pairs read several facets)."""
        if self.merges:
            return max(
                (n for counts in self.counted_pairs.values() for n in counts.values()),
                default=0,
            )
        return max(
            (
                c.pairs.max_n_obs(first_year=first, last_year=last)
                for _sv, c, (first, last) in self.member_pairs()
            ),
            default=0,
        )

    def narrowed(self, admits: Admits) -> BulkPlan:
        """This plan under a request's facet filter: each candidate keeps the pairs of the
        members whose facet `admits` accepts, and the read serves the top of the survivors'
        ranking, as a plan built from the admitted candidates would."""
        candidates: dict[str, RoleBulkCandidate] = {}
        for facet_id, candidate in self.candidates.items():
            by_sv = {
                sv: kept
                for sv, per_sv in candidate.by_sv.items()
                if (kept := [c for c in per_sv if admits(sv, c)])
            }
            if by_sv == candidate.by_sv:
                candidates[facet_id] = candidate
            elif by_sv:
                kept_all = [c for per_sv in by_sv.values() for c in per_sv]
                candidates[facet_id] = replace(
                    candidate,
                    by_sv=by_sv,
                    pair_count=sum(c.pair_count for c in kept_all),
                    latest_date=max(
                        (c.latest_date for c in kept_all if c.latest_date), default=None
                    ),
                )
        election = role_election(
            candidates,
            members=self.members,
            read_years=self.read_years,
            index_sourced=self.index_sourced,
        )
        return replace(self, **election._asdict(), candidates=candidates)

    def windowed(self, years: TimeWindow) -> BulkPlan:
        """This plan read over `years`, a range with both bounds inside the plan's own
        years: its year list is the range's, and the facet is elected again inside it, as a
        plan built under that window elects from the same candidates."""
        assert years.start_year is not None and years.end_year is not None
        listed = tuple(str(year) for year in range(years.start_year, years.end_year + 1))
        election = role_election(
            self.candidates,
            members=self.members,
            read_years=listed,
            index_sourced=self.index_sourced,
        )
        return replace(self, **election._asdict(), years=years, read_years=listed)

    def omitting(self, omitted: Omissions) -> BulkPlan:
        """This plan with each candidate's pairs leaving out the tuples that bind an entity
        whose rows `omitted` leaves out of their StatVar (`RolePairs.excluding`). The facet
        and its continuation stay elected, as the read elects them before it leaves those
        rows out. A member whose omitted entity held a pair in its years under the facets the
        read dispatches (`served_candidates`, `member_years`) joins `omitted_served`; under
        `latest` a member's year is the one its remaining pairs elect, so an omission in
        another year confirms nothing there."""
        if not omitted:
            return self
        candidates: dict[str, RoleBulkCandidate] = {}
        for facet_id, candidate in self.candidates.items():
            by_sv = {
                sv: [_without(c, _omitted_by_role(omitted, sv)) for c in per_sv]
                for sv, per_sv in candidate.by_sv.items()
            }
            if by_sv == candidate.by_sv:
                candidates[facet_id] = candidate
                continue
            kept = [c for per_sv in by_sv.values() for c in per_sv]
            candidates[facet_id] = replace(
                candidate,
                by_sv=by_sv,
                pair_count=sum(c.pair_count for c in kept),
                latest_date=max((c.latest_date for c in kept if c.latest_date), default=None),
            )
        narrowed = replace(self, candidates=candidates, omitted=self.omitted | omitted)
        if self.elected is None:
            return narrowed
        served: set[str] = set()
        for sv, role, entity in omitted:
            first, last = narrowed.member_years(sv)
            if any(
                entity in c.pairs.members_with_data(first_year=first, last_year=last, role=role)
                for c in self.served_candidates(sv)
            ):
                served.add(sv)
        if served <= self.omitted_served:
            return narrowed
        stated = replace(narrowed, omitted_served=self.omitted_served | served)
        if "elected_years" in vars(narrowed):
            # `omitted_served` selects no row, so the election `narrowed` ran stands.
            vars(stated)["elected_years"] = narrowed.elected_years
        return stated


class RoleElection(NamedTuple):
    """A bulk read's election (`role_election`), as the `BulkPlan` fields it sets."""

    facet_id: str | None
    selected: dict[str, tuple[str, ...]]
    unserved: tuple[str, ...]
    continuing: tuple[str, ...]


def role_election(
    candidates: Mapping[str, RoleBulkCandidate],
    *,
    members: Sequence[str],
    read_years: Sequence[str] | None,
    index_sourced: bool = False,
) -> RoleElection:
    """The election a bulk read of `members` serves from `candidates` inside `read_years`:
    the lead of the members' one group (`facet_election.select_role_candidate`), each
    member's facet ids its selection serves and the members it leaves out
    (`facet_election.role_group_selection`), and the facets continuing the lead there
    (`facet_election.role_continuation`), or None and nothing with no candidate."""
    best = select_role_candidate(
        candidates, stat_vars=members, time_period=read_years, index_sourced=index_sourced
    )
    if best is None:
        return RoleElection(None, {}, (), ())
    selected, unserved = role_group_selection(
        candidates,
        stat_vars=members,
        time_period=read_years,
        lead=best.facet_id,
        index_sourced=index_sourced,
    )
    return RoleElection(
        best.facet_id,
        selected,
        unserved,
        role_continuation(best, candidates, window=window_of(read_years)),
    )


def _tuple_counts(
    candidates: Iterable[RoleFacetCandidate], first: int | None, last: int | None
) -> dict[RoleTuple, int]:
    """Each role tuple `candidates` hold an observation for in `first`..`last`, with the
    observations they hold there."""
    counts: dict[RoleTuple, int] = {}
    for c in candidates:
        for role_tuple, n_obs in c.pairs.head(
            c.pair_count, first_year=first, last_year=last, with_data=True
        ):
            counts[role_tuple] = counts.get(role_tuple, 0) + n_obs
    return counts


def _collapsed_counts(
    candidates: Iterable[RoleFacetCandidate], first: int | None, last: int | None
) -> dict[RoleTuple, int]:
    """`_tuple_counts` across facets a mint collapses (`materialise.facet_collapse`), which
    keeps one row per date of facets sharing a unit, method and provenance; facets of
    different methods add. Coverage holds counts by year and no dates, so annual facets of
    one identity count a tuple's year once, at the larger of their counts there, which is
    exact, since each dates a year once. A sub-annual facet
    (`series_coverage.schema.is_sub_annual`) adds its own count (`_tuple_counts`): its
    dates within a year may or may not repeat another facet's, so the sum is an upper
    bound, the safe side of the row gate (`role_bulk_rank_max_rows`). Inside a window the
    index counts a row's years alone (`CoverageIndex.pairs`), for one facet as for
    several, so a sub-annual count there is the index's year count."""
    tuples: dict[RoleTuple, int] = {}
    annual: list[RoleFacetCandidate] = []
    for c in candidates:
        if is_sub_annual(c.observation_period):
            for role_tuple, n_obs in _tuple_counts([c], first, last).items():
                tuples[role_tuple] = tuples.get(role_tuple, 0) + n_obs
        else:
            annual.append(c)
    by_year: dict[tuple[RoleTuple, tuple[str | None, ...], int], int] = {}
    for c in annual:
        identity = (c.unit, c.measurement_method, c.provenance_id)
        for role_tuple, counts in c.pairs.year_counts(first_year=first, last_year=last):
            for year, n_obs in counts:
                key = (role_tuple, identity, year)
                by_year[key] = max(n_obs, by_year.get(key, 0))
    for (role_tuple, _identity, _year), n_obs in by_year.items():
        if n_obs:
            tuples[role_tuple] = tuples.get(role_tuple, 0) + n_obs
    return tuples


def _without(
    candidate: RoleFacetCandidate, by_role: Mapping[str, frozenset[str]]
) -> RoleFacetCandidate:
    """`candidate` without the tuples binding one of `by_role`'s entities on its role: its
    pairs, members, tuple count and latest year read only the remaining tuples."""
    dropped = {
        role: entities & candidate.members_by_role.get(role, frozenset())
        for role, entities in by_role.items()
    }
    if not any(dropped.values()):
        return candidate
    pairs = candidate.pairs
    for role, entities in dropped.items():
        if entities:
            pairs = pairs.excluding(role=role, dcids=entities)
    span = pairs.year_span(first_year=None, last_year=None)
    return replace(
        candidate,
        pairs=pairs,
        members_by_role={
            role: members - dropped.get(role, frozenset())
            for role, members in candidate.members_by_role.items()
        },
        pair_count=pairs.n_with_data(first_year=None, last_year=None),
        latest_date=str(span[1]) if span is not None else None,
    )


type ReadPlan = PairPlan | BulkPlan


def served_units(plan: ReadPlan) -> frozenset[str | None]:
    """The native unit of each facet `plan` serves, None for a facet with no unit: the
    group of each served series, whose facets share one unit, or in bulk the facets the
    group's selection serves (`BulkPlan.served_facets`) while they serve any pair."""
    if isinstance(plan, PairPlan):
        return frozenset(s.group[0].unit for s in plan.served if s.group)
    if plan.population == 0:
        return frozenset()
    return frozenset(facet.unit for facet in plan.served_facets)


def served_facet_keys(plan: ReadPlan) -> frozenset[tuple[str | None, str | None]]:
    """The native unit and provenance of each facet `plan` serves (`served_units`)."""
    if isinstance(plan, PairPlan):
        return frozenset((f.unit, f.provenance_id) for s in plan.served for f in s.group)
    if plan.population == 0:
        return frozenset()
    return frozenset((facet.unit, facet.provenance_id) for facet in plan.served_facets)


def plan_roles(plan: ReadPlan) -> tuple[str, ...]:
    """The graph roles `plan` reads its rows at: a role-bearing read's role names, else the
    roles its series bind."""
    if plan.roles is not None:
        return tuple(dict.fromkeys(plan.roles.role_names.values()))
    if isinstance(plan, BulkPlan):
        return ()
    return tuple(dict.fromkeys(role for s in plan.series for role, _dcid in s.role_tuple))


def served_role_entities(plan: ReadPlan, role: str) -> frozenset[str]:
    """The entities `plan` serves on graph role `role`: those its served series bind there,
    or in bulk those the served facets' pairs hold data for there in each member's years.
    Both read the plan and the coverage index in memory."""
    if isinstance(plan, PairPlan):
        return frozenset(dcid for s in plan.served for r, dcid in s.role_tuple if r == role)
    return frozenset().union(
        *(
            candidate.pairs.members_with_data(first_year=first, last_year=last, role=role)
            for _sv, candidate, (first, last) in plan.member_pairs()
        )
    )


def dated_without_complete_dates(facets: Iterable[Facet]) -> bool:
    """Whether any data-bearing facet lacks a complete dated read."""
    return any(f.obs_count > 0 and (not f.dates or not f.dates_complete) for f in facets)


def series_points(series: PlanSeries, read: TimeWindow | None) -> int:
    """The points a series serves inside `read` (every date when None): one per date across
    its group, since materialise keeps one row per date of facets sharing a provenance,
    method and unit, whatever their observation period (`materialise.facet_collapse`'s
    grain key). A group with a data-bearing facet whose dates are incomplete counts each
    facet's observations."""
    if dated_without_complete_dates(series.group):
        return sum(f.obs_count for f in series.group)
    return len({d for f in series.group for d in f.dates if read is None or in_window(d, read)})


def facets_served(plan: ReadPlan) -> list[Facet]:
    """The facets `plan` serves: each served series' group, or in bulk the facets the
    group's selection serves (`BulkPlan.served_facets`) and those continuing the lead
    (`BulkPlan.continuing`), each as one facet carrying its
    provenance. Coverage counts its sources from them and every
    answer cites them (`region.facets.cited_facets`), so the count, the sentence and
    `resolved_sources` state the sources the preview reads.

    A per-series plan lists them by source: most points served first (one per series under
    `latest`, else `series_points` in the plan's years, summed across the series a source
    serves), then by provenance dcid, then by facet id. The order is the plan's alone, so a
    first answer and its resubmit cite the same sources in the same order whatever order
    their places were probed in."""
    if isinstance(plan, PairPlan):
        read = plan.years if isinstance(plan.years, TimeWindow) else None
        points: dict[str | None, int] = {}
        facets: list[Facet] = []
        for s in plan.served:
            n = 1 if plan.years == "latest" else series_points(s, read)
            for provenance in {f.provenance_id for f in s.group}:
                points[provenance] = points.get(provenance, 0) + n
            facets.extend(s.group)
        return sorted(
            facets,
            key=lambda f: (-points[f.provenance_id], f.provenance_id or "", f.facet_id or ""),
        )
    if plan.elected is None:
        return []
    served = plan.served_facet_ids
    return [
        Facet(
            obs_count=c.pair_count,
            earliest_date=None,
            latest_date=c.latest_date,
            facet_id=c.facet_id,
            provenance_id=c.provenance_id,
        )
        for c in (
            *plan.served_facets,
            *(plan.candidates[facet_id] for facet_id in plan.continuing if facet_id not in served),
        )
    ]


def series_dates(series: PlanSeries, years: PlanYears) -> ServedDates | None:
    """The first and last dates of the points `series` serves under `years`, or None when
    its group states none. Under `latest` that is its group's latest period alone
    (`PlanSeries.latest_period`), which a served series shares with its StatVar unless the
    StatVar elected its year (`PairPlan.series_dates`), and otherwise the dates its group's
    facets hold under `years` (`facet_dates`)."""
    if years == "latest":
        latest = series.latest_period
        return (latest, latest) if latest is not None else None
    return _hull(d for f in series.group if (d := facet_dates(f, years)) is not None)


def served_dates(plan: ReadPlan) -> dict[str, ServedDates]:
    """Each StatVar's first and last dates among the points `plan` serves: across its
    served series (`series_dates`), or in bulk across the pairs of the facets the read
    dispatches (`BulkPlan.served_candidates`) inside the member's years
    (`BulkPlan.member_years`), at year grain, which under `latest` is the member's one
    year alone (`BulkPlan.member_year`). A StatVar the plan serves no dated point of is
    absent.

    Coverage states its window from these dates and every answer dates its StatVars from
    them (`spec_build.build_stat_vars`), so the years a sentence names are years the
    preview reads, on a first answer and on a resubmit alike."""
    if isinstance(plan, PairPlan):
        by_sv: dict[str, list[ServedDates]] = {}
        for s in plan.served:
            if (dates := plan.series_dates(s)) is not None:
                by_sv.setdefault(s.stat_var, []).append(dates)
        return {sv: hull for sv, spans in by_sv.items() if (hull := _hull(spans)) is not None}
    if plan.elected is None:
        return {}
    served: dict[str, ServedDates] = {}
    for sv in plan.served_members:
        first, last = plan.member_years(sv)
        if plan.years == "latest":
            if first is not None and any(
                c.pairs.n_with_data(first_year=first, last_year=last)
                for c in plan.selected_candidates(sv)
            ):
                served[sv] = (str(first), str(first))
            continue
        spans = [
            span
            for c in plan.served_candidates(sv)
            if (span := c.pairs.year_span(first_year=first, last_year=last)) is not None
        ]
        if spans:
            served[sv] = (
                str(min(start for start, _end in spans)),
                str(max(end for _start, end in spans)),
            )
    return served


def series_facets(plan: ReadPlan | None) -> list[Facet | RoleBulkCandidate]:
    """One facet per series `plan` serves, carrying the provenance, method and unit its
    preview dispatches: the elected facet that leads a served series' group, whose facets
    share all three save the estimated method continuing it past its last year, or in bulk
    each facet the group's selection serves (`BulkPlan.served_facets`), the lead first."""
    if plan is None:
        return []
    if isinstance(plan, BulkPlan):
        return list(plan.served_facets) if plan.population > 0 else []
    return [s.group[0] for s in plan.served]


def served_provenances(plan: ReadPlan | None) -> tuple[str, ...]:
    """The distinct provenances of the series `plan` serves (`series_facets`), in serving
    order."""
    return tuple(
        dict.fromkeys(facet.provenance_id for facet in series_facets(plan) if facet.provenance_id)
    )


def data_years(plan: ReadPlan | None) -> tuple[int, int] | None:
    """The first and last years the series `plan` serves hold over every date: the span
    of each served series' facets (`facet_dates` under no window), or in bulk the elected
    facet's pairs for each served StatVar holding data in its years (`member_pairs`),
    unclipped by them. `served_dates` clips to those years, so its first date in a range
    can sit after a gap the series holds. This span says where the data starts and ends.
    None for no plan, and when no served series states both bounds."""
    if plan is None:
        return None
    if isinstance(plan, PairPlan):
        hull = _hull(d for s in plan.served for f in s.group if (d := facet_dates(f, None)))
        if hull is None:
            return None
        first, last = year_of(hull[0]), year_of(hull[1])
        return (first, last) if first is not None and last is not None else None
    spans = [
        span
        for _sv, c, (first, last) in plan.member_pairs()
        if c.pairs.year_span(first_year=first, last_year=last) is not None
        and (span := c.pairs.year_span(first_year=None, last_year=None)) is not None
    ]
    if not spans:
        return None
    return min(start for start, _end in spans), max(end for _start, end in spans)


def serves_continuation(plan: ReadPlan | None) -> bool:
    """Whether `plan` serves a point from a facet continuing a series past its last period
    (`materialise.facets.continuation`). A facet in a served series' group outside its
    member's selection (`PairPlan.selections`) continues the series: it serves when it holds
    a date under the plan's years after the last date any facet of the selection holds at
    that series, the rows `materialise.columnar.continuation_mask` keeps. A scope read
    serves one facet of the selection, and a mint of its Spec all of them, so the last date
    is read from the series' candidates. In bulk a continuing facet serves when
    it adds an observation to a role tuple (`BulkPlan.continued_pairs`). False for no
    plan."""
    if plan is None:
        return False
    if isinstance(plan, BulkPlan):
        return any(
            sum(counts.values()) > sum(plan.elected_pairs[sv].values())
            for sv, counts in plan.continued_pairs.items()
        )
    for s in plan.served:
        selection = plan.selections.get(s.stat_var, frozenset())
        hull = _hull(
            d for f in s.candidates if f.facet_id in selection and (d := facet_dates(f, None))
        )
        last = hull[1] if hull is not None else None
        for f in s.group:
            if f.facet_id in selection:
                continue
            dates = facet_dates(f, plan.years)
            if dates is not None and dates[1] is not None and (last is None or dates[1] > last):
                return True
    return False


def plan_coverage(plan: ReadPlan) -> ReadingCoverage | None:
    """The years `plan` serves at each role tuple, keyed by its joined dcids
    (`PlanSeries.key`), across every StatVar it serves there (`PairPlan.served`), from each
    served series' group. None for a bulk plan, whose pairs carry year spans and not the
    years between, so no comparison can show it covers another."""
    if isinstance(plan, BulkPlan):
        return None
    by_key: dict[str, list[Facet]] = {}
    for s in plan.served:
        by_key.setdefault(s.key[1], []).extend(s.group)
    return {
        key: years
        for key, facets in by_key.items()
        if (years := observed_years(facets)) is not None
    }


def serves_nothing(plan: ReadPlan) -> bool:
    """Whether `plan`'s preview reads no point, so the answer is `no_observations`: no
    series served, or a bulk read whose served facets hold no pair with data in its years.
    A series holding a point in the plan's years that no facet can dispatch
    (`PairPlan.unservable`) keeps the answer, whose preview says why it serves nothing."""
    if isinstance(plan, BulkPlan):
        return plan.population == 0
    return not plan.served and not plan.unservable


def populated_members(plan: ReadPlan) -> frozenset[str]:
    """The members of `plan` a mint counts as populated (`materialise.group_facets.is_populated`):
    a per-series plan's members whose probe or series holds a candidate carrying a facet id,
    a bulk plan's those a candidate holds pairs for."""
    if isinstance(plan, BulkPlan):
        return frozenset(
            sv
            for sv in plan.members
            if is_populated(c for c in plan.candidates.values() if sv in c.by_sv)
        )
    return frozenset(
        sv for sv, probe in plan.probes.items() if is_populated(probe.candidates)
    ) | frozenset(s.stat_var for s in plan.series if is_populated(s.candidates))


def unserved_members(plan: ReadPlan | None) -> tuple[str, ...]:
    """The members of `plan` holding data that the answer's one group leaves out, in member
    order, as a mint of the answer's Spec names them in `coverage_below_scope`'s
    `unserved_members`: those the plan's election leaves out (`PairPlan.unserved`,
    `BulkPlan.unserved`), and
    those the answer's unit decision leaves out (`left_out`). A member holding no candidate
    is not named, as the mint names no unpopulated member. Empty for no plan and for a plan
    serving nothing, whose answer holds no data at all. The answer names them in a
    `member_unavailable` departure (`engine.departures.spec_departure_facts`)."""
    if plan is None or serves_nothing(plan):
        return ()
    named = {*plan.unserved, *plan.left_out}
    return tuple(sv for sv in plan.members if sv in named)


type RoleEntity = tuple[str, str]
"""An entity on one role: its role key as an identity keys it (`spec_id.entity_role_key`:
`observationAbout`, `origin` or `destination`) and its dcid."""

_AXIS_FOR_COVERAGE_ROLE: Mapping[str, str] = {"recipient": "destination", "donor": "origin"}
"""Each coverage role's flow axis, the role key an identity binds its entities under."""


def unserved_entities(plan: ReadPlan | None) -> frozenset[RoleEntity]:
    """The entities `plan` was asked to read on each role and serves no point at on that
    role: they hold no data in its years, or hold it only in a facet outside the one group,
    or the one bulk facet, the plan serves. An entity bound on two roles is served or not
    on each separately. A donor that is also a member of the recipient group is unserved as
    a recipient when it gives nothing to itself, and served as the donor. An answer naming
    one states so in an `entity_unavailable` departure
    (`engine.departures.spec_departure_facts`).

    A per-series plan was asked for its `places` as subjects and the entities its roles
    bind, and serves those a served series binds on the same role. Under `latest` an entity
    whose series end before their StatVar's latest period is unserved. A bulk plan was asked
    for the entities its roles bind, and serves on each role those its served facets hold
    a pair with data for in the member's years (`BulkPlan.member_pairs`). Empty for no plan,
    for a scope read, whose places the answer does not name, and for a plan serving nothing,
    whose answer holds no data at all. An entity whose rows a conversion left out of every
    StatVar (`omitting`) is unserved, since the plan reads none of its series."""
    if plan is None or serves_nothing(plan):
        return frozenset()
    bound = plan.roles.bound if plan.roles is not None else {}
    if isinstance(plan, PairPlan):
        if plan.scope:
            return frozenset()
        role_names = plan.roles.role_names if plan.roles is not None else {}
        key_by_role_name = {
            name: _AXIS_FOR_COVERAGE_ROLE.get(coverage_role, coverage_role)
            for coverage_role, name in role_names.items()
        }
        asked = {("observationAbout", dcid) for dcid in plan.places}
        asked.update(
            (_AXIS_FOR_COVERAGE_ROLE.get(coverage_role, coverage_role), dcid)
            for coverage_role, dcids in bound.items()
            for dcid in dcids
        )
        served = {
            (key_by_role_name.get(role, role), dcid)
            for s in plan.served
            for role, dcid in s.role_tuple
        }
        return frozenset(asked - served)
    if plan.elected is None or plan.roles is None:
        return frozenset()
    unserved: set[RoleEntity] = set()
    for coverage_role, asked_dcids in bound.items():
        if (role := plan.roles.role_names.get(coverage_role)) is None:
            continue
        served_dcids = frozenset().union(
            *(
                c.pairs.members_with_data(first_year=first, last_year=last, role=role)
                for _sv, c, (first, last) in plan.member_pairs()
            )
        )
        key = _AXIS_FOR_COVERAGE_ROLE.get(coverage_role, coverage_role)
        unserved.update((key, dcid) for dcid in asked_dcids - served_dcids)
    return frozenset(unserved)


def in_elected_datasets(plan: ReadPlan, member_facts: Mapping[str, MemberFacts]) -> ReadPlan:
    """`plan` with each member of an index-resolved family keeping only the candidates its
    elected dataset publishes (`elected_dataset.publishes`), and every group and bulk facet
    elected again from them (`narrowed`). A mint filters a member's candidates by the same
    test before it ranks (`materialise.facets.dataset_candidates`), so an answer and a mint
    of its Spec elect within one dataset, and a member holding data under another dataset
    alone serves nothing. Every resolver passes its plan through here. A plan every member of
    which has an elected dataset reads an index-resolved family, so its group agrees on
    provenance and unit alone (`PairPlan.index_sourced`, `BulkPlan.index_sourced`), as the
    mint's does (`MaterialisationRequest.index_sourced`). A plan none of whose members has an
    elected dataset is returned as it is."""
    datasets = elected_datasets(member_facts)
    if not datasets:
        return plan
    indexed = replace(plan, index_sourced=all(sv in datasets for sv in plan.members))
    return indexed.narrowed(lambda sv, candidate: publishes(datasets, sv, candidate))


def preview_years(plan: ReadPlan | None) -> PlanYears:
    """The years an answer reads (`plan.years`), which its unit election and facet options
    read too. An answer with no plan reads no facet: every date."""
    return plan.years if plan is not None else None


def plan_probes(plan: ReadPlan | None) -> Mapping[str, MemberProbe] | None:
    """Each member's election input in a per-series plan (`PairPlan.probes`), which a unit
    decision reads so it elects the group the plan serves. None for a bulk plan or no plan,
    whose decision probes for itself."""
    return plan.probes if isinstance(plan, PairPlan) else None


def plan_pairs(
    series: Iterable[PlanSeries],
    *,
    members: Sequence[str],
    years: PlanYears,
    scope: bool = False,
    roles: PlanRoles | None = None,
    probes: Mapping[str, MemberProbe] | None = None,
    places: Sequence[str] = (),
    deadline: float | None = None,
    index_sourced: bool = False,
    elect: Election | None = None,
) -> PairPlan:
    """A per-series plan over `series`, each member's group elected under `years` from its
    probe: `probes`, or with none each member's own series (`series_probes`), which a plan
    asked for one place and no scope builds with each variable's whole breadth, read by
    `deadline` if a tie needs it. A role-bearing
    plan passes the coverage probe a mint reads (`materialise.facets.coverage_probe`), since
    its series are only the role tuples holding a point in the years, and `elect`, the
    election of the year its `latest` read serves each member in. `index_sourced` marks an
    index-resolved family (`PairPlan.index_sourced`)."""
    series = list(series)
    if probes is None:
        probes = series_probes(
            series,
            places=() if scope else places,
            deadline=deadline,
        )
    elected, facet_id, selections, unserved = _elect_groups(
        series,
        members=members,
        years=years,
        scope=scope,
        probes=probes,
        index_sourced=index_sourced,
    )
    return PairPlan(
        members=tuple(members),
        years=years,
        series=elected,
        scope=scope,
        scope_facet_id=facet_id,
        selections=selections,
        unserved=unserved,
        roles=roles,
        probes=probes,
        places=tuple(places),
        index_sourced=index_sourced,
        elect=elect,
    )


def plan_role_read(
    sv_dcids: Sequence[str],
    entity_lists: Sequence[Sequence[str]],
    *,
    years: PlanYears,
    read_years: Sequence[str] | None,
    candidates: Callable[[Sequence[str]], Mapping[str, RoleBulkCandidate]],
    series_count: Callable[[Sequence[str]], int],
    pair_series: Callable[[Sequence[str]], Iterable[PlanSeries]],
    member_probes: Callable[[Sequence[str]], Mapping[str, MemberProbe]],
    roles: PlanRoles,
    elect: Election,
) -> ReadPlan:
    """Plan a role-bearing preview's read over `sv_dcids`.

    `sv_dcids` are the members the role-class election keeps
    (`role_resolve._elect_role_class`, the mint's `statvar_members.elect_role_class`), and
    the preview reads each of them at each role pair of `entity_lists`, one list per bound
    role. A series is one full role tuple, an open role's members
    included, as the bulk read serves it. The preview mode rule
    (`search.observations.preview_mode`) decides. The plan reads each series on its own when
    two counts fit `obs_preview_max_pairs`, the members times the bound pairs and
    `series_count(members)`, the series holding a point in the plan's years, counted from
    coverage. Only then does `pair_series(members)` build the series and
    `member_probes(members)` each member's election input, so neither is built for a plan
    that reads in bulk. Above the cap one bulk read serves the facet
    `facet_election.select_role_candidate` elects from `candidates(members)` inside
    `read_years`. Either plan holds `elect`, the one election of the year its `latest` read
    serves each member in (`PairPlan.elected_years`, `BulkPlan.elected_years`).
    """
    members = tuple(sv_dcids)
    mode = preview_mode(
        role_bearing=True,
        places_defaulted=False,
        bound_pairs=len(members) * math.prod(len(entities) for entities in entity_lists),
        series_count=lambda: series_count(members),
    )
    if mode is not PreviewMode.ROLE_BULK:
        return plan_pairs(
            pair_series(members),
            members=members,
            years=years,
            roles=roles,
            probes=member_probes(members),
            elect=elect,
        )
    merged = candidates(members)
    listed = tuple(read_years) if read_years is not None else None
    return BulkPlan(
        members=members,
        years=years,
        read_years=listed,
        **role_election(merged, members=members, read_years=listed)._asdict(),
        candidates=merged,
        roles=roles,
        elect=elect,
    )


def plan_standard(
    facets_by_series: Mapping[tuple[str, str], Sequence[Facet]],
    *,
    members: Sequence[str],
    years: PlanYears,
    scope: bool = False,
    places: Sequence[str] = (),
    deadline: float | None = None,
) -> PairPlan:
    """A standard preview's plan: each `(stat_var, place)` series of `members` read at
    `observationAbout`, or with `scope` one facet elected across every place. `places` are
    the places the read was asked for, including any holding no facet. A plan at one place
    reads each variable's whole breadth by `deadline` when a tie needs it (`plan_pairs`)."""
    kept = set(members)
    return plan_pairs(
        (
            PlanSeries(
                stat_var=sv, role_tuple=(("observationAbout", entity),), candidates=tuple(facets)
            )
            for (sv, entity), facets in facets_by_series.items()
            if sv in kept
        ),
        members=members,
        years=years,
        scope=scope,
        places=places,
        deadline=deadline,
    )
