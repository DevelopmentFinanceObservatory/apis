"""Fetch role-bearing previews when the pair product exceeds the per-pair cap.

The read plan (`read_plan.BulkPlan`) names the one facet the read serves and its years. One
`sdmx.role_series` call reads the points. Coverage counts the rows the answer keeps before
any read (`BulkPlan.observation_count`). When they fit `role_bulk_rank_max_rows`, the call
reads every series with the open role left unfiltered, and the preview shows the first
`role_bulk_series_max` by `search.observations.series_rank` (`ranked`). Above the gate, or
under a conversion into each entity's own currency, the call reads only
`role_bulk_series_max` pairs chosen from coverage, StatVars newest year first and taken
round-robin (`fixed`), since the full population can reach 1.6M pairs. A range reads the
plan's year list. A `latest` read reads each pair's latest point with SDMX `LATEST` and
serves each member its own latest period alone (`materialise.window.latest_period`), in the
year coverage dates it (`BulkPlan.member_year`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from dc_sdmx import SdmxCapExceededError
from materialise.window import latest_periods
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import (
    EngineSdmxClient,
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
from search.engine.retrieval.read_plan import BulkPlan, YearBounds
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
    """Return all date-value points. `role_series` normally supplies one facet."""
    return [(d, v) for f in facets for d, v in zip(f.dates, f.values, strict=True)]


def _preview_pairs(
    candidate: RoleBulkCandidate,
    limit: int,
    *,
    years: Callable[[str], YearBounds],
    member_year: Callable[[str], int | None],
) -> list[tuple[tuple[str, RoleTuple], int]]:
    """The first `limit` (sv, role_tuple) pairs of `candidate` with data inside their
    StatVar's `years`, in `search.observations.fixed_selection` order, each with its
    coverage observation count there.

    StatVars go newest `member_year` first (`member_year_order`), each holding its role
    tuples in sorted order, and are taken round-robin. Only the first `limit` StatVars can
    contribute, and each reads at most `limit` of its own role tuples (RolePairs.head), so
    the population beyond the preview is never turned into Python objects."""
    columns: list[list[tuple[tuple[str, RoleTuple], int]]] = []
    ordered = member_year_order({sv: member_year(sv) for sv in candidate.by_sv})
    for sv in ordered[:limit]:
        first_year, last_year = years(sv)
        counted: dict[RoleTuple, int] = {}
        for c in candidate.by_sv[sv]:
            for role_tuple, n_obs in c.pairs.head(
                limit, first_year=first_year, last_year=last_year, with_data=True
            ):
                counted.setdefault(role_tuple, n_obs)
        columns.append([((sv, rt), n) for rt, n in sorted(counted.items())[:limit]])
    return fixed_selection(columns, limit)


def _newest_year_with_data(plan: BulkPlan, sv: str) -> int | None:
    """The newest year `sv` holds an observation in its member's years under the elected
    facet (`BulkPlan.member_years`), the year the Datasets observations door orders a
    snapshot's StatVars by. None when it holds none there."""
    best = plan.elected
    first_year, last_year = plan.member_years(sv)
    spans = (
        c.pairs.year_span(first_year=first_year, last_year=last_year)
        for c in (best.by_sv.get(sv, ()) if best is not None else ())
    )
    return max((span[1] for span in spans if span is not None), default=None)


def _population_pairs(plan: BulkPlan) -> list[tuple[tuple[str, RoleTuple], int]]:
    """Every (sv, role_tuple) pair of `plan` with data in its member's years
    (`BulkPlan.member_pairs`), each with its coverage observation count there: the
    population `BulkPlan.population` counts."""
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
    plan: BulkPlan, best: RoleBulkCandidate, *, converts_to_role_currency: bool = False
) -> _RoleBulkFetchPlan:
    """Choose the pairs `_bulk_role_series` reads from `best`'s coverage-confirmed
    population in each member's years (`BulkPlan.member_years`), under
    `search.observations.role_bulk_order` of the rows the answer keeps
    (`BulkPlan.observation_count`), the count the Datasets observations door gates its
    frame's rows on.

    `ranked` takes every series of the population (`_population_pairs`) and filters the
    read on the bound roles alone (`PlanRoles.bound`), leaving the open role unfiltered.
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
            best,
            observation_settings().role_bulk_series_max,
            years=plan.member_years,
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
    return _RoleBulkFetchPlan(
        to_fetch=to_fetch,
        facet_svs=facet_svs,
        entities_for_series=entities_for_series,
        total_population=population,
        population_n_obs=plan.observation_count,
        preview_n_obs=sum(n_obs for _pair, n_obs in preview),
        order=order,
    )


_TRUNCATED_ORDER = {
    "ranked": "the most recent and largest first",
    "fixed": "a fixed selection, newest year first across variables, not ranked by value",
}


def _assemble_role_bulk_series(
    best: RoleBulkCandidate,
    plan: _RoleBulkFetchPlan,
    pair_points: Mapping[tuple[str, RoleTuple], list[tuple[str, float | None]]],
    *,
    member_constraints: Mapping[str, Mapping[str, str]] | None,
    graph: EngineGraphClient,
    converted: ConvertedPairs | None = None,
    target_unit_ref: str | None = None,
) -> tuple[SpecObservations, Warning | None]:
    """Build series from the dispatched pairs and return any warning.

    `converted` holds each converted pair's conversion: its series states the conversion
    and is in `target_unit_ref`, or states no unit for a conversion into each entity's own
    currency, as the per-series path's do.

    No returned pairs yields `OBSERVATIONS_NO_POINTS` without a warning, or, when `converted`
    left out StatVars or pairs (`ConvertedPairs.leaves_out`), an
    `OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED` omission naming them. Otherwise each pair
    is a series of `best`'s facet, assembled by `search.observations.assemble_role_bulk`,
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

    facets = candidate_facet_refs(best)
    conversions = converted.conversions if converted is not None else {}
    reads = {
        pair: SeriesRead(
            points=sorted(points, key=lambda p: p[0]),
            total_rows=len(points),
            facets=facets,
            conversion=conversions.get(pair),
        )
        for pair, points in pair_points.items()
    }
    observations = assemble_role_bulk(
        reads,
        label_source=graph,
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
    """Serve the facet `plan` elected across its pairs, in one `role_series` call for every
    pair (`ranked`) or for at most `observation_settings().role_bulk_series_max` pairs
    (`fixed`, `_plan_role_bulk_fetch`), convert the points under a unit request
    (`conversion`, `bulk_conversion.convert_pairs`), then assemble the result. Each read logs
    the gate's row count, the order chosen and its elapsed time.

    The read elects nothing: `plan` already narrowed to any facet filter, and with no elected
    facet the read returns `OBSERVATIONS_NO_MATCHING_SOURCE` under `facet_filter`, else
    `OBSERVATIONS_NO_DISPATCHABLE_FACET`. A range reads the plan's year list
    (`BulkPlan.read_years`); `latest` reads each dispatched pair's latest point with SDMX
    `LATEST` for pairs with a point in their member's latest year (`BulkPlan.member_year`)
    and keeps each member's points at its latest period alone (`_at_member_latest`); a
    dateless read takes every year.

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
        best,
        converts_to_role_currency=conversion is not None and conversion.target_unit_ref is None,
    )
    try:
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
    try:
        series_by_variable = sdmx.role_series(
            variables=fetch_plan.facet_svs,
            entities=fetch_plan.entities_for_series,
            facet_ids={best.facet_id},
            time_period=(
                "LATEST"
                if latest
                else list(plan.read_years)
                if plan.read_years is not None
                else None
            ),
            max_rows=settings().sdmx_max_rows,
            max_bytes=settings().sdmx_max_bytes,
            deadline=deadline,
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

    pair_points = {
        (sv, role_tuple): points
        for sv in fetch_plan.facet_svs
        for role_tuple, facets in series_by_variable.get(sv, {}).items()
        if (sv, role_tuple) in dispatched_pairs
        if (points := _points_from_facets(facets))
    }
    if latest:
        at_latest = _at_member_latest(plan, pair_points)
        if short := sorted(pair_points.keys() - at_latest.keys()):
            # The plan serves every dispatched pair in its member's latest year, so a pair
            # left short means coverage and SDMX disagree. The truncation warning counts it
            # as not shown.
            logger.warning(
                "coverage index and the bulk role read disagree: %d dispatched pairs held "
                "no point at their member's latest period and are not served: %r",
                len(short),
                short,
            )
        pair_points = at_latest
    converted = None
    if conversion is not None:
        try:
            converted = convert_pairs(pair_points, facet=best, conversion=conversion)
        except (*UPSTREAM_READ_ERRORS, TimeoutError) as exc:
            return conversion_failed_omission(exc)
        pair_points = converted.points
    return _assemble_role_bulk_series(
        best,
        fetch_plan,
        pair_points,
        member_constraints=member_constraints,
        graph=graph,
        converted=converted,
        target_unit_ref=conversion.target_unit_ref if conversion is not None else None,
    )


def _at_member_latest(
    plan: BulkPlan, pair_points: Mapping[tuple[str, RoleTuple], list[tuple[str, float | None]]]
) -> dict[tuple[str, RoleTuple], list[tuple[str, float | None]]]:
    """`pair_points` at each member's latest period, as a mint keeps a StatVar's rows
    (`materialise.columnar.latest_rows`): coverage fixes the member's year
    (`BulkPlan.member_year`), and the newest period its points hold inside that year
    (`materialise.window.latest_periods`) is the one served. A pair left with no point is
    dropped, and so is every pair of a member coverage dates no year for."""
    years = {
        sv: str(year)
        for sv in {sv for sv, _rt in pair_points}
        if (year := plan.member_year(sv)) is not None
    }
    in_year = {
        pair: [point for point in points if point[0][:4] == years.get(pair[0])]
        for pair, points in pair_points.items()
    }
    latest = latest_periods(
        (sv, date) for (sv, _rt), points in in_year.items() for date, _value in points
    )
    return {
        pair: kept
        for pair, points in in_year.items()
        if (kept := [point for point in points if point[0] == latest.get(pair[0])])
    }
