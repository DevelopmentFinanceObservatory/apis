"""Fetch role-bearing previews when the pair product exceeds the per-pair cap.

The read plan (`read_plan.BulkPlan`) names the one facet the read serves, the facets
continuing it past its last year, and its years. One `sdmx.role_series` call reads the
points, and each pair serves a continuing facet's points only after its last elected one
(`_served_pairs`), as a mint of the answer's Spec does. Coverage counts the rows the answer
keeps before any read (`BulkPlan.observation_count`). When they fit `role_bulk_rank_max_rows`, the call
reads every series with the open role left unfiltered (or, with every role open, filtered
on the population's own values of its narrowest role), and the preview shows the first
`role_bulk_series_max` by `search.observations.series_rank` (`ranked`). Above the gate, or
under a conversion into each entity's own currency, the call reads only
`role_bulk_series_max` pairs chosen from coverage, StatVars newest year first and taken
round-robin (`fixed`), since the full population can reach 1.6M pairs. A range reads the
plan's year list. A `latest` read serves each member its one period alone, in the year
the plan dates it (`BulkPlan.member_year`). It reads each pair's latest point with SDMX
`LATEST`, then reads again the members one of whose pairs ends after the year its member
elected (`_ending_after_elected`), one read per year they elected naming that year alone
(`_by_elected_year`), as a mint reads a member again at its elected year. Those reads run
together, at most `sdmx_chunk_concurrency` at once (`_read_each_year`). A member that elected no year serves its own latest period
(`materialise.window.latest_period`). A member whose elected year the reads return no
point in (`_without_elected_points`) falls back under `ranked`, which reads every pair of
the member: it serves the newest period among the `LATEST` rows of every pair the read
returned for it, as a mint falls back (`materialise.columnar.elected_rows`). Under `fixed`
the read holds only some of the member's pairs, and the newest period can sit at one it did
not read, so the member serves no series. The truncation warning counts them as not shown,
or with no series left the preview is omitted as `OBSERVATIONS_NO_POINTS`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import NamedTuple

from code_index.shared import get_index
from dc_sdmx import SdmxCapExceededError, is_observed
from dfo_platform.executor import submit_with_context
from materialise.facet_collapse import (
    CollapseCell,
    FacetRank,
    FacetRankStats,
    facet_rank,
    keep_mask,
)
from materialise.method_status import status_of
from materialise.window import latest_periods
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import (
    EngineSdmxClient,
    RoleTimePeriod,
    RoleTuple,
)
from search.engine.config import settings
from search.engine.errors import (
    UPSTREAM_READ_ERRORS,
    GraphInfraError,
    RoleNotDeclaredError,
    RoleSignatureError,
)
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_NO_DISPATCHABLE_FACET,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_SERIES_TRUNCATED,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.region.facets import candidate_facet_refs
from search.engine.region.observations.bulk_conversion import (
    BulkConversion,
    ConvertedPairs,
    conversion_failed_omission,
    convert_pairs,
    left_out_omission,
    left_out_warning,
)
from search.engine.region.result import FacetAxisFilter
from search.engine.retrieval.facet_election import (
    RoleBulkCandidate,
)
from search.engine.retrieval.read_plan import BulkPlan, YearBounds, year_of
from search.models import (
    PreviewOrder,
    SpecObservations,
    Warning,
    omitted_observations,
)
from search.observations import (
    SeriesRead,
    assemble_role_bulk,
    fixed_selection,
    member_year_order,
    observation_settings,
    role_bulk_order,
    role_bulk_shown,
)

logger = logging.getLogger(__name__)

# One `role_series` read: each variable's facets by role tuple.
type _RoleRows = Mapping[str, Mapping[RoleTuple, Sequence[Facet]]]

# Bulk requests can hit role or facet limits that the bounded confirm read avoids.
_FETCH_FAILURE_EXCEPTIONS = (
    GraphInfraError,
    RoleNotDeclaredError,
    RoleSignatureError,
    SdmxCapExceededError,
)


def _bulk_fetch_timed_out(
    sv_dcids: list[str], where: str, facet_id: str | None = None
) -> tuple[SpecObservations, Warning]:
    """Return a timed-out omission and warning while preserving coverage.

    `where` names the read in the log and message. `facet_id` identifies the selected facet.
    """
    logger.warning(
        "bulk role-keyed %s read for sv_dcids=%r%s crossed the request deadline; "
        "delivering no observations",
        where,
        sv_dcids,
        f" facet_id={facet_id!r}" if facet_id is not None else "",
        exc_info=True,
    )
    _message = (
        "the bulk observation fetch for a role-bearing destination set did not finish "
        "before the request deadline; coverage stands, series were not populated."
    )
    return (
        omitted_observations(OBSERVATIONS_TIMED_OUT, _message),
        Warning(code=OBSERVATIONS_TIMED_OUT, severity="warn", message=_message),
    )


def _points_from_facets(facets: list[Facet]) -> list[tuple[str, float | None]]:
    """The date-value points of `facets` that hold an observation (`dc_sdmx.is_observed`),
    the rule a mint drops a row by (`materialise.columnar.survivor_mask`). `role_series`
    parses each row's value by it, so an empty or unparseable value holds none."""
    return [
        (d, v) for f in facets for d, v in zip(f.dates, f.values, strict=True) if is_observed(v)
    ]


_Point = tuple[str, float | None, str]
"""One observed point a bulk read returned: its date, its value and the facet id it is
read under, which the read keeps until it has fixed the points a pair serves."""

_PairRows = dict[tuple[str, RoleTuple], list[_Point]]


def _pair_rows(sv: str, facets: list[Facet], *, plan: BulkPlan) -> list[_Point]:
    """The observed points (`_points_from_facets`) the read returned for one pair of `sv`
    under the facets that can serve it: those the group's selection serves `sv`
    (`BulkPlan.selected`) and those continuing the lead (`BulkPlan.continuing`). A returned
    facet in neither, such as another member's selection, serves nothing here."""
    serving = {*plan.selected.get(sv, ()), *plan.continuing}
    return [
        (date, value, f.facet_id)
        for f in facets
        if f.facet_id in serving
        for date, value in _points_from_facets([f])
    ]


class _Served(NamedTuple):
    """The points each pair serves, the refs its series states, and the pairs holding a
    point of an estimated method (`_served_pairs`)."""

    points: dict[tuple[str, RoleTuple], list[tuple[str, float | None]]]
    stated: dict[tuple[str, RoleTuple], dict[str, str]]
    estimated: frozenset[tuple[str, RoleTuple]]


def _served_pairs(
    plan: BulkPlan,
    rows: Mapping[tuple[str, RoleTuple], list[_Point]],
    *,
    ranked_by_series: Collection[str],
    estimated: frozenset[str],
    fell_back: Collection[str] = (),
) -> _Served:
    """The points each pair of `rows` serves, as a mint of its member keeps its rows. Under
    `latest` `rows` arrive cut to each member's period (`_at_member_latest`), as a mint cuts
    its rows before it masks or collapses them.

    Each pair keeps its selected facets' points, then those of the facets continuing the
    lead dated after the last of them, or all of theirs where the selection holds none
    (`materialise.columnar.continuation_mask`). Each member's kept points then collapse
    across every pair the read holds at once (`materialise.facet_collapse.keep_mask`): on a
    date two facets contest at one pair under one unit, method and provenance, the
    best-ranked facet keeps it, and points of different methods both stay.

    A member in `ranked_by_series`, whose conversion runs arithmetic
    (`BulkConversion.converts_under`), ranks each facet over its points at the pair alone,
    as a mint ranks within each series then (`materialise.generation._convert_by_series`);
    the read holds every point of a pair it reads. Any other member ranks each facet over
    the rows a mint of the member keeps of it across all its pairs, counted from coverage
    (`_member_ranks`), so the winner does not depend on which pairs the read fetched. A
    member in `fell_back` serves its newest period in place of the year coverage elected,
    and ranks over coverage in that period's year.

    A series states the refs its pair's served facets share (`_series_refs`): the selected
    facets holding a served point, else the continuing ones. A pair holds estimates when a
    served point comes from one of the `estimated` facet ids. Refs and the flag read only
    the points served, so a point the period cut or the collapse drops shapes neither."""
    kept_by_sv: dict[str, list[tuple[RoleTuple, _Point]]] = {}
    for (sv, role_tuple), points in rows.items():
        ids = frozenset(plan.selected.get(sv, ()))
        selected = [point for point in points if point[2] in ids]
        last = max((point[0] for point in selected), default=None)
        kept = [
            *selected,
            *(p for p in points if p[2] not in ids and (last is None or p[0] > last)),
        ]
        kept_by_sv.setdefault(sv, []).extend((role_tuple, point) for point in kept)
    served: dict[tuple[str, RoleTuple], list[_Point]] = {}
    for sv, entries in kept_by_sv.items():
        ranks = None
        if sv not in ranked_by_series:
            if sv in fell_back:
                year = year_of(entries[0][1][0])
                years: YearBounds = (year, year)
            else:
                years = plan.member_years(sv)
            ranks = _member_ranks(plan, sv, {point[2] for _rt, point in entries}, years)
        mask = keep_mask(
            [
                _collapse_cell(sv, role_tuple, point, plan.candidates)
                for role_tuple, point in entries
            ],
            rank_within_series=ranks is None,
            ranks=ranks,
        )
        for (role_tuple, point), keep in zip(entries, mask, strict=True):
            if keep:
                served.setdefault((sv, role_tuple), []).append(point)
    stated: dict[tuple[str, RoleTuple], dict[str, str]] = {}
    for (sv, role_tuple), points in served.items():
        ids = list(dict.fromkeys(point[2] for point in points))
        stating = [i for i in ids if i in plan.selected.get(sv, ())] or ids
        stated[(sv, role_tuple)] = _series_refs([plan.candidates[i] for i in stating])
    return _Served(
        points={pair: [(d, v) for d, v, _f in points] for pair, points in served.items()},
        stated=stated,
        estimated=frozenset(
            pair for pair, points in served.items() if any(p[2] in estimated for p in points)
        ),
    )


def _member_ranks(
    plan: BulkPlan, sv: str, facet_ids: Collection[str], years: YearBounds
) -> dict[str | None, FacetRank]:
    """Each of `facet_ids`' collapse rank (`materialise.facet_collapse.facet_rank`) over the
    rows a mint of `sv` keeps of it across all `sv`'s pairs, counted from coverage in
    `years`: its rows the observations counted and its span the years between the first and
    last of them. A facet the selection serves `sv` counts every pair the plan holds there.
    A facet continuing the lead counts only what the continuation keeps after the selected
    facets' last year at each pair (`BulkPlan.continued_years`), as a mint ranks rows its
    continuation mask has already cut. Inside a window coverage counts a sub-annual facet's
    years, not its observations, so its rows are low there (#161). Coverage that holds no
    row of a facet in `years`, as when it disagrees with SDMX over a member's latest year,
    ranks it with no rows."""
    first, last = years
    selected = plan.selected.get(sv, ())
    continued: dict[str, list[tuple[int, int]]] = {}
    if any(facet_id not in selected for facet_id in facet_ids):
        for facet_id, _role_tuple, by_year in plan.continued_years(sv):
            continued.setdefault(facet_id, []).extend(by_year)
    ranks: dict[str | None, FacetRank] = {}
    for facet_id in facet_ids:
        if facet_id in selected:
            held = plan.candidates[facet_id].by_sv.get(sv, ())
            rows = sum(c.pairs.n_obs(first_year=first, last_year=last) for c in held)
            spans = [
                span
                for c in held
                if (span := c.pairs.year_span(first_year=first, last_year=last)) is not None
            ]
        else:
            by_year = continued.get(facet_id, [])
            rows = sum(n for _year, n in by_year)
            spans = [(year, year) for year, _n in by_year]
        ranks[facet_id] = facet_rank(
            FacetRankStats(
                facet_id=facet_id,
                has_period=plan.candidates[facet_id].observation_period is not None,
                rows=rows,
                span_years=(
                    max(end for _start, end in spans) - min(start for start, _end in spans)
                    if spans
                    else -1
                ),
            )
        )
    return ranks


def _collapse_cell(
    sv: str, role_tuple: RoleTuple, point: _Point, candidates: Mapping[str, RoleBulkCandidate]
) -> CollapseCell:
    """`point` of `sv` at `role_tuple` as the mint's collapse keys and ranks a row
    (`materialise.facet_collapse.CollapseCell`), its facet's unit, method, provenance and
    period read off its candidate. The pair's dcids are the row's role values, so its grain
    key keeps places apart."""
    date, _value, facet_id = point
    candidate = candidates[facet_id]
    return CollapseCell(
        variable=sv,
        role_values=tuple(dcid for _role, dcid in role_tuple),
        time_period=date,
        unit=candidate.unit,
        measurement_method=candidate.measurement_method,
        provenance=candidate.provenance_id,
        observation_period=candidate.observation_period,
        facet_id=facet_id,
    )


def _series_refs(stating: Sequence[RoleBulkCandidate]) -> dict[str, str]:
    """The axis refs every facet of `stating` declares alike (`candidate_facet_refs`), so a
    series served under two methods states its source and unit and no method."""
    refs = [candidate_facet_refs(c) for c in stating]
    if not refs:
        return {}
    return {axis: ref for axis, ref in refs[0].items() if all(r.get(axis) == ref for r in refs)}


def _preview_pairs(
    plan: BulkPlan, limit: int, *, member_year: Callable[[str], int | None]
) -> list[tuple[tuple[str, RoleTuple], int]]:
    """The first `limit` (sv, role_tuple) pairs `plan` serves with data inside their
    StatVar's years (`BulkPlan.member_years`), in `search.observations.fixed_selection`
    order, each with its coverage observation count there.

    StatVars go newest `member_year` first (`member_year_order`), each holding its role
    tuples in sorted order, and are taken round-robin. Only the first `limit` StatVars can
    contribute, and each reads at most `limit` of its own role tuples (RolePairs.head), so
    the population beyond the preview is never turned into Python objects. A plan whose
    plan whose pairs read several facets counts each tuple once across them
    (`BulkPlan.counted_pairs`), which has decoded them already."""
    columns: list[list[tuple[tuple[str, RoleTuple], int]]] = []
    ordered = member_year_order({sv: member_year(sv) for sv in plan.served_members})
    for sv in ordered[:limit]:
        first_year, last_year = plan.member_years(sv)
        counted: dict[RoleTuple, int] = {}
        if plan.merges:
            counted.update(plan.counted_pairs[sv])
        else:
            for c in plan.served_candidates(sv):
                for role_tuple, n_obs in c.pairs.head(
                    limit, first_year=first_year, last_year=last_year, with_data=True
                ):
                    counted.setdefault(role_tuple, n_obs)
        columns.append([((sv, rt), n) for rt, n in sorted(counted.items())[:limit]])
    return fixed_selection(columns, limit)


def _newest_year_with_data(plan: BulkPlan, sv: str) -> int | None:
    """The newest year `sv` holds an observation in its member's years under the facets the
    read dispatches (`BulkPlan.served_candidates`, `BulkPlan.member_years`), the year the
    Datasets observations door orders a snapshot's StatVars by. None when it holds none
    there."""
    first_year, last_year = plan.member_years(sv)
    spans = (
        c.pairs.year_span(first_year=first_year, last_year=last_year)
        for c in plan.served_candidates(sv)
    )
    return max((span[1] for span in spans if span is not None), default=None)


def _population_pairs(plan: BulkPlan) -> list[tuple[tuple[str, RoleTuple], int]]:
    """Every (sv, role_tuple) pair of `plan` with data in its member's years
    (`BulkPlan.member_pairs`, or `BulkPlan.counted_pairs` when its pairs read several
    facets), each with its coverage observation count there: the population
    `BulkPlan.population` counts."""
    if plan.merges:
        return [
            ((sv, role_tuple), n_obs)
            for sv, counts in plan.counted_pairs.items()
            for role_tuple, n_obs in counts.items()
        ]
    pairs: dict[tuple[str, RoleTuple], int] = {}
    for sv, c, (first_year, last_year) in plan.member_pairs():
        n_with_data = c.pairs.n_with_data(first_year=first_year, last_year=last_year)
        for role_tuple, n_obs in c.pairs.head(
            n_with_data, first_year=first_year, last_year=last_year, with_data=True
        ):
            pairs.setdefault((sv, role_tuple), n_obs)
    return list(pairs.items())


@dataclass(frozen=True)
class _RoleBulkFetchPlan:
    """The (stat_var, role_tuple) pairs `_bulk_role_series` dispatches for `best`'s
    facet_id, and how it chose them (`order`). `ranked` dispatches the whole
    coverage-confirmed population, `fixed` at most
    `observation_settings().role_bulk_series_max` of it, since the population can run to
    1.6M pairs on an open query. Also carries the population accounting
    assembly uses without a second read: `total_population` counts the series eligible for
    the answer, one per full role tuple with data in its member's years (under `latest` the
    member's own latest year), and the observation counts cover the same years.
    `population_n_obs` is also the row count `order` was gated on.
    """

    to_fetch: list[tuple[str, RoleTuple]]
    facet_svs: list[str]
    entities_for_series: Mapping[str, Sequence[str]]
    total_population: int
    population_n_obs: int
    preview_n_obs: int
    order: PreviewOrder


def _plan_role_bulk_fetch(
    plan: BulkPlan, *, converts_to_role_currency: bool = False
) -> _RoleBulkFetchPlan:
    """Choose the pairs `_bulk_role_series` reads from the coverage-confirmed population of
    the facets `plan` dispatches in each member's years (`BulkPlan.member_years`), under
    `search.observations.role_bulk_order` of the rows the answer keeps
    (`BulkPlan.observation_count`), the count the Datasets observations door gates its
    frame's rows on.

    `ranked` takes every series of the population (`_population_pairs`) and filters the
    read on the bound roles alone (`PlanRoles.bound`), leaving the open role unfiltered.
    When that filter comes out empty, as with every role open, the read filters on the
    narrowest role every pair carries (`_narrowest_common_role`), since SDMX refuses a read
    with no entity filter.
    `fixed` takes `role_bulk_series_max` pairs by `_preview_pairs`, StatVars ordered by
    their newest year with data in their member's years (`_newest_year_with_data`), read
    lazily rather than materialising the whole population, filtered on every role. Both
    pass over pairs with no observation in their member's years. The population and its
    observation count are the plan's (`BulkPlan.population`, `BulkPlan.observation_count`).
    """
    order = role_bulk_order(
        rows=plan.observation_count, converts_to_role_currency=converts_to_role_currency
    )
    population = plan.population
    ranked = order == "ranked"
    preview = (
        _population_pairs(plan)
        if ranked
        else _preview_pairs(
            plan,
            observation_settings().role_bulk_series_max,
            member_year=lambda sv: _newest_year_with_data(plan, sv),
        )
    )
    to_fetch = [pair for pair, _n_obs in preview]
    facet_svs = list(dict.fromkeys(sv for sv, _role_tuple in to_fetch))
    open_roles = (
        {name for role, name in plan.roles.role_names.items() if role not in plan.roles.bound}
        if ranked and plan.roles is not None
        else set()
    )
    narrowed_entities: dict[str, list[str]] = {}
    for _sv, role_tuple in to_fetch:
        for role, dcid in role_tuple:
            if role not in open_roles:
                narrowed_entities.setdefault(role, []).append(dcid)
    entities_for_series: Mapping[str, Sequence[str]] = {
        role: sorted(set(dcids)) for role, dcids in narrowed_entities.items()
    }
    if not entities_for_series and to_fetch:
        entities_for_series = _narrowest_common_role(to_fetch)
    return _RoleBulkFetchPlan(
        to_fetch=to_fetch,
        facet_svs=facet_svs,
        entities_for_series=entities_for_series,
        total_population=population,
        population_n_obs=plan.observation_count,
        preview_n_obs=sum(n_obs for _pair, n_obs in preview),
        order=order,
    )


def _narrowest_common_role(to_fetch: Sequence[tuple[str, RoleTuple]]) -> dict[str, list[str]]:
    """The read filter for pairs whose every role is open.

    It names the role every pair in `to_fetch` carries with the fewest distinct dcids there
    (ties by role name) and lists those dcids. Every pair holds one of them, so the read
    returns each pair's rows as an unfiltered read would. Raises `ValueError` when no role is
    common to every pair, since filtering on a role some pairs lack drops those pairs' rows."""
    dcids_by_role: dict[str, set[str]] = {}
    for _sv, role_tuple in to_fetch:
        for role, dcid in role_tuple:
            dcids_by_role.setdefault(role, set()).add(dcid)
    common = set(dcids_by_role)
    for _sv, role_tuple in to_fetch:
        common &= {role for role, _dcid in role_tuple}
    if not common:
        raise ValueError(f"no role is common to every dispatched pair: {to_fetch[:5]!r}")
    role = min(common, key=lambda r: (len(dcids_by_role[r]), r))
    return {role: sorted(dcids_by_role[role])}


_TRUNCATED_ORDER = {
    "ranked": "the most recent and largest first",
    "fixed": "a fixed selection, newest year first across variables, not ranked by value",
}


def _assemble_role_bulk_series(
    stated: Mapping[tuple[str, RoleTuple], Mapping[str, str]],
    plan: _RoleBulkFetchPlan,
    pair_points: Mapping[tuple[str, RoleTuple], list[tuple[str, float | None]]],
    *,
    member_constraints: Mapping[str, Mapping[str, str]] | None,
    graph: EngineGraphClient,
    converted: ConvertedPairs | None = None,
    target_unit_ref: str | None = None,
    estimated: frozenset[tuple[str, RoleTuple]] = frozenset(),
) -> tuple[SpecObservations, Warning | None]:
    """Build series from the dispatched pairs and return any warning. The pairs in
    `estimated` serve a point of an estimated method (`SeriesRead.holds_estimates`).

    `converted` holds each converted pair's conversion: its series states the conversion
    and is in `target_unit_ref`, or states no unit for a conversion into each entity's own
    currency, as the per-series path's do.

    No returned pairs yields `OBSERVATIONS_NO_POINTS` without a warning, or, when `converted`
    left out StatVars or pairs (`ConvertedPairs.leaves_out`), an
    `OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED` omission naming them. Otherwise each pair
    is a series stating the refs its pair's serving facets share (`stated`, `_series_refs`),
    assembled by `search.observations.assemble_role_bulk`,
    which shows at most `role_bulk_series_max` of them under `plan.order`. A population
    larger than the shown series warns, its `params.order` naming the order, and StatVars
    and pairs the conversion left out warn under
    `OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED`, which carries the truncation's message when
    both apply. `total_obs` adds the population's unread in-window coverage
    counts to the points read, so no second SDMX read is needed.
    """
    if not pair_points and converted is not None and converted.leaves_out:
        return left_out_omission(converted)
    if not pair_points:
        return (
            omitted_observations(
                OBSERVATIONS_NO_POINTS,
                f"the bulk observation fetch for a role-bearing destination set returned "
                f"no rows for any of the {len(plan.to_fetch)} (stat_var, role_tuple) "
                "pairs dispatched for this facet.",
                dispatched=len(plan.to_fetch),
                empty=len(plan.to_fetch),
            ),
            None,
        )

    conversions = converted.conversions if converted is not None else {}
    reads = {
        pair: SeriesRead(
            points=sorted(points, key=lambda p: p[0]),
            total_rows=len(points),
            facets=dict(stated.get(pair, {})),
            conversion=conversions.get(pair),
            holds_estimates=pair in estimated,
        )
        for pair, points in pair_points.items()
    }
    observations = assemble_role_bulk(
        reads,
        label_source=graph,
        index=get_index(),
        member_constraints=member_constraints or {},
        order=plan.order,
        unread_obs=plan.population_n_obs - plan.preview_n_obs,
        total_series=plan.total_population,
        target_currency_ref=target_unit_ref,
    )

    shown = len(role_bulk_shown(reads, order=plan.order))
    total_confirmed = plan.total_population
    warn: Warning | None = None
    if shown < total_confirmed:
        warn = Warning(
            code=OBSERVATIONS_SERIES_TRUNCATED,
            severity="warn",
            message=(
                f"showing {shown} of {total_confirmed} confirmed series "
                f"({_TRUNCATED_ORDER[plan.order]}); the rest are not shown."
            ),
            params={"shown": shown, "total": total_confirmed, "order": plan.order},
        )
    return observations, left_out_warning(converted, also=warn)


def _bulk_role_series(
    plan: BulkPlan,
    sv_dcids: list[str],
    *,
    sdmx: EngineSdmxClient,
    graph: EngineGraphClient,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    deadline: float | None = None,
    member_constraints: Mapping[str, Mapping[str, str]] | None = None,
    conversion: BulkConversion | None = None,
) -> tuple[SpecObservations, Warning | None]:
    """Serve the facet `plan` elected across its pairs, continued past each pair's last
    point by the facets `plan` names (`BulkPlan.continuing`), in one `role_series` call for every
    pair (`ranked`) or for at most `observation_settings().role_bulk_series_max` pairs
    (`fixed`, `_plan_role_bulk_fetch`), convert the points under a unit request
    (`conversion`, `bulk_conversion.convert_pairs`), then assemble the result. Each read logs
    the gate's row count, the order chosen and its elapsed time.

    The read elects nothing: `plan` already narrowed to any facet filter, and with no elected
    facet the read returns `OBSERVATIONS_NO_MATCHING_SOURCE` under `facet_filter`, else
    `OBSERVATIONS_NO_DISPATCHABLE_FACET`. A range reads the plan's year list
    (`BulkPlan.read_years`); `latest` reads the pairs with a point in their member's year
    (`BulkPlan.member_year`), each pair's latest point with SDMX `LATEST` and then the
    members a pair of which ends after its member's elected year (`_ending_after_elected`),
    one read per year they elected naming that year alone (`_by_elected_year`), and keeps
    each member's points at its one period alone
    (`_at_member_latest`). A member that elected a year the reads return no point in
    (`_without_elected_points`) costs no further read. Under `ranked` it serves the newest
    period among the `LATEST` rows of every pair returned for it, as a mint falls back.
    Under `fixed`, which reads only some of its pairs, it serves no series, which the
    truncation warning counts as not shown, or `OBSERVATIONS_NO_POINTS` when no series is
    left. A dateless read takes every year.

    A plan that dispatches no pair reads nothing and returns `OBSERVATIONS_NO_POINTS`.
    Read failures in `_FETCH_FAILURE_EXCEPTIONS` return `OBSERVATIONS_BULK_FETCH_FAILED`
    with a warning. Coverage stays valid and bound-unit handling escalates the failure.
    The shared SDMX deadline returns `OBSERVATIONS_TIMED_OUT` with a warning and bypasses
    503 escalation. The request deadline handles it. A read the conversion makes that fails
    or crosses the deadline omits the preview the same way
    (`bulk_conversion.conversion_failed_omission`). Every path returns a payload.

    `member_constraints` labels each series, falling back to the resolved StatVar label.
    A variable the origin catalogue places upstream raises `UpstreamVariableRoutedToSdmx`
    before the read.
    """
    best = plan.elected
    if best is None:
        if facet_filter:
            _message = "no observations available from the requested source(s)."
            return (
                omitted_observations(OBSERVATIONS_NO_MATCHING_SOURCE, _message),
                Warning(code=OBSERVATIONS_NO_MATCHING_SOURCE, severity="warn", message=_message),
            )
        _message = (
            "the observation preview was not fetched: no source across the "
            "role-bearing destination set could be dispatched as a single facet."
        )
        return (
            omitted_observations(OBSERVATIONS_NO_DISPATCHABLE_FACET, _message),
            Warning(code=OBSERVATIONS_NO_DISPATCHABLE_FACET, severity="warn", message=_message),
        )

    started = time.monotonic()
    fetch_plan = _plan_role_bulk_fetch(
        plan,
        converts_to_role_currency=conversion is not None and conversion.target_unit_ref is None,
    )
    try:
        if not fetch_plan.to_fetch:
            # No pair to read, and SDMX refuses a read with no entity filter.
            return _assemble_role_bulk_series(
                {},
                fetch_plan,
                {},
                member_constraints=member_constraints,
                graph=graph,
            )
        return _read_and_assemble(
            plan,
            fetch_plan,
            best,
            sv_dcids,
            sdmx=sdmx,
            graph=graph,
            deadline=deadline,
            member_constraints=member_constraints,
            conversion=conversion,
        )
    finally:
        logger.info(
            "bulk role read for sv_dcids=%r facet_id=%r: %d rows by coverage against a gate "
            "of %d, order=%s, %d pairs dispatched, %.3fs",
            sv_dcids,
            best.facet_id,
            fetch_plan.population_n_obs,
            observation_settings().role_bulk_rank_max_rows,
            fetch_plan.order,
            len(fetch_plan.to_fetch),
            time.monotonic() - started,
        )


def _read_and_assemble(
    plan: BulkPlan,
    fetch_plan: _RoleBulkFetchPlan,
    best: RoleBulkCandidate,
    sv_dcids: list[str],
    *,
    sdmx: EngineSdmxClient,
    graph: EngineGraphClient,
    deadline: float | None,
    member_constraints: Mapping[str, Mapping[str, str]] | None,
    conversion: BulkConversion | None,
) -> tuple[SpecObservations, Warning | None]:
    """`_bulk_role_series`'s read of `fetch_plan`, its conversion and its assembly."""
    latest = plan.years == "latest"
    # A dispatched pair names every role its variable declares, an open origin's creditor
    # included (the role index binds every role the shape places), so each row matches its
    # pair on the full role tuple, whether or not the read filtered on the open role.
    dispatched_pairs = set(fetch_plan.to_fetch)

    def read(variables: Sequence[str], time_period: RoleTimePeriod) -> _RoleRows:
        return sdmx.role_series(
            variables=variables,
            entities=fetch_plan.entities_for_series,
            facet_ids={*plan.served_facet_ids, *plan.continuing},
            time_period=time_period,
            max_rows=settings().sdmx_max_rows,
            max_bytes=settings().sdmx_max_bytes,
            deadline=deadline,
        )

    estimated_ids = frozenset(
        c.facet_id
        for c in (*plan.served_facets, *(plan.candidates[facet_id] for facet_id in plan.continuing))
        if status_of(c.measurement_method, c.provenance_id) == "estimated"
    )

    def pair_rows(rows: _RoleRows, *, every_pair_of: Collection[str] = ()) -> _PairRows:
        """Each dispatched pair's points in `rows` under the facets that can serve it
        (`_pair_rows`). A member in `every_pair_of` reads every pair `rows` holds for it
        that the plan does not omit (`_omitted`)."""
        found: _PairRows = {}
        for sv in fetch_plan.facet_svs:
            for role_tuple, facets in rows.get(sv, {}).items():
                if (sv, role_tuple) not in dispatched_pairs and (
                    sv not in every_pair_of or _omitted(plan, sv, role_tuple)
                ):
                    continue
                if points := _pair_rows(sv, facets, plan=plan):
                    found[(sv, role_tuple)] = points
        return found

    fell_back: Sequence[str] = ()
    try:
        if latest:
            newest = read(fetch_plan.facet_svs, "LATEST")
            rows = dict(newest)
            pair_points = pair_rows(rows)
            if lagging := _ending_after_elected(plan, pair_points):
                rows.update(_read_each_year(read, _by_elected_year(plan, lagging)))
                pair_points = pair_rows(rows)
            without = _without_elected_points(plan, fetch_plan.facet_svs, pair_points)
            if without and fetch_plan.order == "ranked":
                logger.warning(
                    "coverage index and the bulk role read disagree: %r returned no point "
                    "in the year coverage elects; serving their newest period",
                    without,
                )
                fell_back = without
                rows.update({sv: newest.get(sv, {}) for sv in fell_back})
                pair_points = pair_rows(rows, every_pair_of=fell_back)
            elif without:
                # A `fixed` read holds some of a member's pairs, and the newest period a
                # mint falls back to can sit at one it did not read, so it serves none.
                logger.warning(
                    "coverage index and the bulk role read disagree: %r returned no point "
                    "in the year coverage elects; omitting their series",
                    without,
                )
                pair_points = {
                    pair: points for pair, points in pair_points.items() if pair[0] not in without
                }
        else:
            pair_points = pair_rows(
                read(
                    fetch_plan.facet_svs,
                    list(plan.read_years) if plan.read_years is not None else None,
                )
            )
    except _FETCH_FAILURE_EXCEPTIONS:
        logger.warning(
            "bulk role-keyed series read failed for sv_dcids=%r facet_id=%r; "
            "delivering no observations",
            sv_dcids,
            best.facet_id,
            exc_info=True,
        )
        _message = (
            "the bulk observation fetch for a role-bearing destination set "
            "failed; coverage stands, series were not populated."
        )
        return (
            omitted_observations(OBSERVATIONS_BULK_FETCH_FAILED, _message),
            Warning(code=OBSERVATIONS_BULK_FETCH_FAILED, severity="warn", message=_message),
        )
    except TimeoutError:
        # Apply the same over-deadline degrade as the index read above. This
        # narrowed series read crossed this call's shared deadline.
        return _bulk_fetch_timed_out(sv_dcids, "series", facet_id=best.facet_id)

    if latest:
        at_latest = _at_member_latest(plan, pair_points, undated=fell_back)
        if short := sorted(
            pair for pair in pair_points.keys() - at_latest.keys() if pair[0] not in fell_back
        ):
            # The plan serves every dispatched pair in its member's latest year, so a pair
            # left short means coverage and SDMX disagree. The truncation warning counts it
            # as not shown. A member that fell back is logged above.
            logger.warning(
                "coverage index and the bulk role read disagree: %d dispatched pairs held "
                "no point at their member's latest period and are not served: %r",
                len(short),
                short,
            )
        pair_points = at_latest
    served = _served_pairs(
        plan,
        pair_points,
        ranked_by_series=(
            {sv for sv in plan.members if conversion.converts_under(sv, best)}
            if conversion is not None
            else ()
        ),
        estimated=estimated_ids,
        fell_back=fell_back,
    )
    converted = None
    if conversion is not None:
        try:
            converted = convert_pairs(served.points, facet=best, conversion=conversion)
        except (*UPSTREAM_READ_ERRORS, TimeoutError) as exc:
            return conversion_failed_omission(exc)
    return _assemble_role_bulk_series(
        served.stated,
        fetch_plan,
        converted.points if converted is not None else served.points,
        member_constraints=member_constraints,
        graph=graph,
        converted=converted,
        target_unit_ref=conversion.target_unit_ref if conversion is not None else None,
        estimated=served.estimated,
    )


def _ending_after_elected(plan: BulkPlan, pair_points: _PairRows) -> list[str]:
    """The members that elected a year (`BulkPlan.elected_years`) one of whose pairs in
    `pair_points`, each pair's `LATEST` row, is dated after: the `LATEST` read cannot
    return that pair's point in the elected year, so the member is read again there. In
    member order."""
    elected = plan.elected_years or {}
    after = {
        sv
        for (sv, _role_tuple), points in pair_points.items()
        if (year := elected.get(sv)) is not None
        and any((held := year_of(point[0])) is not None and held > year for point in points)
    }
    return [sv for sv in plan.members if sv in after]


def _by_elected_year(plan: BulkPlan, variables: Sequence[str]) -> list[tuple[int, list[str]]]:
    """`variables` grouped by the year each elected (`BulkPlan.elected_years`), oldest year
    first, each group in `variables` order: the reads that name each variable its own year
    alone. SDMX filters variables and periods as a product, so one read naming several
    years would return each variable's rows in every one of them. The reads number the
    distinct years `variables` elected, at most one per variable."""
    elected = plan.elected_years or {}
    grouped: dict[int, list[str]] = {}
    for sv in variables:
        grouped.setdefault(elected[sv], []).append(sv)
    return sorted(grouped.items())


def _read_each_year(
    read: Callable[[Sequence[str], RoleTimePeriod], _RoleRows],
    groups: Sequence[tuple[int, list[str]]],
) -> dict[str, Mapping[RoleTuple, Sequence[Facet]]]:
    """The rows of one `read` per group of `groups` (`_by_elected_year`), each naming its
    variables that one year, merged by variable. The reads run together in a pool of at
    most `sdmx_chunk_concurrency` workers, the width one SDMX call spreads its own chunks
    over, and every GET they make holds a slot of the process-wide `SDMX_TOTAL_SEMAPHORE`
    (`search.engine.clients.sdmx`). Each carries the caller's deadline. The first read to
    fail raises its error here: the reads not yet started are cancelled and those in
    flight are waited for, so none outlives the call and no partial result is returned."""
    merged: dict[str, Mapping[RoleTuple, Sequence[Facet]]] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(settings().sdmx_chunk_concurrency, len(groups))),
        thread_name_prefix="bulk-role-year",
    ) as pool:
        futures = [
            submit_with_context(pool, read, variables, [str(year)]) for year, variables in groups
        ]
        try:
            for future in as_completed(futures):
                merged.update(future.result())
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return merged


def _omitted(plan: BulkPlan, sv: str, role_tuple: RoleTuple) -> bool:
    """Whether `role_tuple` binds an entity whose rows the plan leaves out of `sv`
    (`BulkPlan.omitted`)."""
    return any((sv, role, dcid) in plan.omitted for role, dcid in role_tuple)


def _without_elected_points(
    plan: BulkPlan, variables: Sequence[str], pair_points: _PairRows
) -> list[str]:
    """Those of `variables` that elected a year (`BulkPlan.elected_years`) in which none of
    their pairs in `pair_points` serves a point: coverage dates them a year SDMX holds no
    row in. A member one of whose pairs holds the year keeps it."""
    elected = plan.elected_years or {}
    held = {
        sv
        for (sv, _role_tuple), points in pair_points.items()
        if any(year_of(point[0]) == elected.get(sv) for point in points)
    }
    return [sv for sv in variables if sv in elected and sv not in held]


def _at_member_latest(
    plan: BulkPlan, pair_points: _PairRows, *, undated: Collection[str] = ()
) -> _PairRows:
    """`pair_points` at each member's one period, as a mint keeps a StatVar's rows
    (`materialise.columnar.latest_rows`): the plan fixes the member's year
    (`BulkPlan.member_year`, the year it elected or else the year coverage dates its
    latest period in), and the newest period its points hold inside that year
    (`materialise.window.latest_periods`) is the one served. A member in `undated` serves
    the newest period its points hold in any year. A pair left with no point is dropped,
    and so is every other pair of a member coverage dates no year for."""
    years = {
        sv: str(year)
        for sv in {sv for sv, _rt in pair_points}
        if (year := plan.member_year(sv)) is not None
    }
    in_year = {
        pair: [
            point for point in points if pair[0] in undated or point[0][:4] == years.get(pair[0])
        ]
        for pair, points in pair_points.items()
    }
    latest = latest_periods(
        (sv, point[0]) for (sv, _rt), points in in_year.items() for point in points
    )
    return {
        pair: kept
        for pair, points in in_year.items()
        if (kept := [point for point in points if point[0] == latest.get(pair[0])])
    }
