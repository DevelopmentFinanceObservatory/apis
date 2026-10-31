"""Fetch one shared facet for a scope-defaulted standard query.

The read plan (`read_plan.PairPlan` with `scope`) names the facet every place serves. One
`subject_observations` call reads it, or for UN System Data Commons variables one dated facet
read through the origin graph client. The read fetches each place's latest point of the
facet. Under `latest` each variable serves its one latest period
(`materialise.window.latest_period`) at the places its plan serves (`PairPlan.served`) that
hold it, as a mint of the same scope does. A dateless read serves each place's latest point.
Under a unit request the served points convert as a bulk role read's do (`bulk_conversion`).
Coverage names the span the served points fall in (`retrieval.coverage`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from code_index.shared import get_index
from dc_graph import GraphCapExceededError
from dc_sdmx import SdmxCapExceededError, SdmxObservation
from materialise.place_scope import is_worldwide
from materialise.window import latest_periods
from origin_catalogue.origins import UN_ID_PREFIX
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import UPSTREAM_READ_ERRORS, GraphInfraError
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_NO_DISPATCHABLE_FACET,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_SCOPE_PARTIAL,
)
from search.engine.region.facets import candidate_facet_refs
from search.engine.region.observations.bulk_conversion import (
    BulkConversion,
    Pair,
    Points,
    conversion_failed_omission,
    convert_pairs,
    left_out_omission,
    left_out_warning,
)
from search.engine.region.result import FacetAxisFilter
from search.engine.retrieval.read_plan import PairPlan
from search.models import (
    ObservationPoint,
    SpecObservations,
    Warning,
    omitted_observations,
)
from search.observations import SeriesRead, assemble_scope_bulk

if TYPE_CHECKING:
    from search.engine.region.place_context import PlaceContext
    from search.engine.retrieval.retrieve import Materialised

logger = logging.getLogger(__name__)


@dataclass
class BulkScope:
    """Inputs for the scope-wide bulk branch. `places` is the defaulted scope, every place
    with data or the countries fallback, which the bulk read names, keeping it aligned with
    coverage."""

    places: tuple[str, ...]
    sdmx: EngineSdmxClient


def bulk_scope_for(
    place: PlaceContext, mat: Materialised, *, sdmx: EngineSdmxClient
) -> BulkScope | None:
    """The bulk read of an answer over a default scope, else `None`.

    A default scope whose probe was plural (`mat.facets_by_entity` is set) reads in bulk,
    whether it is every place with data, of any level and type, or the countries fallback:
    the read names its places in URL-budget slices (`EngineSdmxClient.subject_observations`).
    An empty scope, Earth alone, one place, or bound or discovered places read per pair.
    """
    scope = place.default_scope
    if (
        scope is None
        or not scope.places
        or is_worldwide(scope.places)
        or mat.facets_by_entity is None
    ):
        return None
    return BulkScope(places=scope.places, sdmx=sdmx)


def _sdmx_points(rows: Sequence[SdmxObservation]) -> list[ObservationPoint]:
    """Dated rows as points, oldest first."""
    return [
        ObservationPoint(date=r.time_period, value=r.value_as_float())
        for r in sorted(rows, key=lambda r: r.time_period or "")
        if r.time_period
    ]


def _facet_rows(rows: Sequence[SdmxObservation], facet_id: str) -> list[SdmxObservation]:
    """The rows facet `facet_id` carries.

    The read filters on `c[facetId]`, so a row without a facet id is malformed and raises
    `GraphInfraError`, which the caller reports as a failed fetch. A row from another facet
    does not serve under this one's attribution.
    """
    if any(r.facet_id is None for r in rows):
        raise GraphInfraError(
            f"a bulk SDMX row filtered on facetId {facet_id!r} carries no facetId, so its "
            "admission at its place cannot be checked"
        )
    return [r for r in rows if r.facet_id == facet_id]


def _latest_facet_point(facets: Sequence[Facet], facet_id: str) -> list[ObservationPoint]:
    """The newest dated observation of the facet `facet_id` among `facets`, as one point."""
    for facet in facets:
        if facet.facet_id == facet_id and facet.dates:
            date, value = max(zip(facet.dates, facet.values, strict=True), key=lambda o: o[0])
            return [ObservationPoint(date=date, value=value)]
    return []


def _latest_scope_points(
    sv_dcids: Sequence[str],
    entity_dcids: Sequence[str],
    *,
    scope: BulkScope,
    graph: EngineGraphClient,
    facet_id: str,
) -> dict[str, dict[str, list[ObservationPoint]]]:
    """The latest point of facet `facet_id` per variable and place in the scope, read only
    from rows that facet carries.

    UN System Data Commons variables are absent from our mixer's SDMX service, so they take
    one dated facet read through the origin graph client at the scope's places, which also
    notes places the UN front door refuses. Every other variable takes one
    `subject_observations` read naming the scope's places.
    """
    un = [sv for sv in sv_dcids if sv.startswith(UN_ID_PREFIX)]
    ours = [sv for sv in sv_dcids if not sv.startswith(UN_ID_PREFIX)]
    points: dict[str, dict[str, list[ObservationPoint]]] = {}
    if ours:
        rows_by_variable = scope.sdmx.subject_observations(
            variables=ours,
            places=scope.places,
            facet_ids=[facet_id],
            latest_only=True,
            max_rows=settings().sdmx_max_rows,
            max_bytes=settings().sdmx_max_bytes,
        )
        for sv, rows_by_entity in rows_by_variable.items():
            points[sv] = {
                entity: _sdmx_points(_facet_rows(rows, facet_id))
                for entity, rows in rows_by_entity.items()
            }
    if un:
        facets = graph.observation_facets_batch(
            un, list(entity_dcids), True, max_bytes=settings().sdmx_max_bytes
        )
        for sv, facets_by_entity in facets.items():
            points[sv] = {
                entity: _latest_facet_point(entity_facets, facet_id)
                for entity, entity_facets in facets_by_entity.items()
            }
    return points


def _bulk_scope_series(
    plan: PairPlan,
    sv_dcids: list[str],
    entity_dcids: Sequence[str],
    *,
    scope: BulkScope,
    graph: EngineGraphClient,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    member_constraints: Mapping[str, Mapping[str, str]] | None = None,
    conversion: BulkConversion | None = None,
) -> tuple[SpecObservations, Warning | None]:
    """Fetch the scope-wide facet `plan` elected for all requested StatVars and entities.

    The read elects nothing: `plan` already narrowed to any facet filter, and with no elected
    facet the read returns `OBSERVATIONS_NO_MATCHING_SOURCE` under `facet_filter`, else
    `OBSERVATIONS_NO_DISPATCHABLE_FACET`; both omit observations and issue no read. Otherwise
    read the latest points (`_latest_scope_points`).

    `GraphInfraError`, `SdmxCapExceededError` and `GraphCapExceededError` return
    `OBSERVATIONS_BULK_FETCH_FAILED` with a warning. Coverage stands, and the code avoids
    probe-failure escalation to 503. An empty fetch returns `OBSERVATIONS_NO_POINTS` without a
    warning. Every path returns a `SpecObservations` payload. LATEST supplies one point per
    entity, and `assemble_scope_bulk` sets `truncated` when that hides a point. A fetched row
    serves only when it carries the chosen facet and the probe admitted that facet for its
    StatVar at its entity; `total_obs` sums those pairs' probe counts. Under `latest` a place
    whose point is not at its variable's latest period across the served pairs
    (`materialise.window.latest_periods`) does not serve either. `conversion` converts the
    served points to its unit (`bulk_conversion.convert_pairs`); a StatVar it leaves out, and
    a place whose point it leaves out for a missing deflator or currency, is named in an
    `OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED` warning, or in an omission with that reason
    when nothing converts. A read the conversion makes that fails or crosses the deadline
    omits the preview (`bulk_conversion.conversion_failed_omission`). Places with no data
    are omitted, with `OBSERVATIONS_SCOPE_PARTIAL`; a place whose served point the conversion
    left out holds data and is not counted there.

    `member_constraints` supplies row labels, falling back to the resolved StatVar label.
    Rows sort by `search.observations.series_rank`: latest year, then latest value, both
    descending, then `(stat_var, entity)`, within groups of one unit.
    """
    groups = [s for s in plan.served if s.group]
    if plan.scope_facet_id is None or not groups:
        if facet_filter:
            _message = "no observations available from the requested source(s)."
            return (
                omitted_observations(OBSERVATIONS_NO_MATCHING_SOURCE, _message),
                Warning(code=OBSERVATIONS_NO_MATCHING_SOURCE, severity="warn", message=_message),
            )
        _message = (
            "the observation preview was not fetched: no source across the "
            "resolved scope could be dispatched as a single facet."
        )
        return (
            omitted_observations(OBSERVATIONS_NO_DISPATCHABLE_FACET, _message),
            Warning(code=OBSERVATIONS_NO_DISPATCHABLE_FACET, severity="warn", message=_message),
        )
    facet_id = plan.scope_facet_id
    # Every place's facet carries the elected id, so its provenance, method and unit.
    best = groups[0].group[0]
    obs_by_pair: dict[tuple[str, str], int] = {}
    for s in groups:
        pair = (s.stat_var, s.role_tuple[0][1])
        obs_by_pair[pair] = obs_by_pair.get(pair, 0) + sum(f.obs_count for f in s.group)

    try:
        points_by_variable = _latest_scope_points(
            sv_dcids, entity_dcids, scope=scope, graph=graph, facet_id=facet_id
        )
    except (GraphInfraError, SdmxCapExceededError, GraphCapExceededError):
        logger.warning(
            "bulk scope-defaulted observations fetch failed for sv_dcids=%r "
            "facet_id=%r; delivering coverage without series",
            sv_dcids,
            facet_id,
            exc_info=True,
        )
        _message = (
            "the bulk observation fetch for a defaulted place scope failed; "
            "coverage stands, series were not populated."
        )
        return (
            omitted_observations(OBSERVATIONS_BULK_FETCH_FAILED, _message),
            Warning(code=OBSERVATIONS_BULK_FETCH_FAILED, severity="warn", message=_message),
        )

    # Only pairs the plan serves set a variable's latest period, so a place the probe never
    # admitted cannot push the period past every served place.
    latest = (
        latest_periods(
            (sv, point.date)
            for sv, by_entity in points_by_variable.items()
            for entity, points in by_entity.items()
            if (sv, entity) in obs_by_pair
            for point in points
        )
        if plan.years == "latest"
        else None
    )

    facets = candidate_facet_refs(best)
    served: dict[Pair, Points] = {}
    total_rows: dict[Pair, int] = {}
    short_of_latest: list[tuple[str, str]] = []
    for sv in sv_dcids:
        entities_for_sv = points_by_variable.get(sv, {})
        for entity in entity_dcids:
            points = entities_for_sv.get(entity)
            if not points:
                continue
            pair_obs_count = obs_by_pair.get((sv, entity))
            if pair_obs_count is None:
                # The plan serves this facet_id for no group of this StatVar at
                # this entity: the probe never admitted it there, or the graph
                # and SDMX disagree. Either way the row does not serve.
                continue
            if latest is not None and all(p.date != latest.get(sv) for p in points):
                # The served place's latest point is older than its variable's latest
                # period: coverage and SDMX disagree. The scope-partial warning counts it.
                short_of_latest.append((sv, entity))
                continue
            key = (sv, (("observationAbout", entity),))
            served[key] = [(p.date, p.value) for p in points]
            total_rows[key] = pair_obs_count

    if short_of_latest:
        logger.warning(
            "coverage index and the scope read disagree: %d served places held no point at "
            "their variable's latest period and are not served: %r",
            len(short_of_latest),
            short_of_latest,
        )
    converted = None
    left_out_places: set[str] = set()
    if conversion is not None:
        try:
            converted = convert_pairs(served, facet=best, conversion=conversion)
        except (*UPSTREAM_READ_ERRORS, TimeoutError) as exc:
            return conversion_failed_omission(exc)
        left_out_places = {entity for _sv, ((_role, entity),) in converted.left_out_pairs(served)}
        served = converted.points
    conversions = converted.conversions if converted is not None else {}
    reads = {
        pair: SeriesRead(
            points=points,
            total_rows=total_rows[pair],
            facets=facets,
            conversion=conversions.get(pair),
        )
        for pair, points in served.items()
    }
    if not reads and converted is not None and converted.leaves_out:
        return left_out_omission(converted)
    if not reads:
        # Every entity the probe attributed this facet_id to disagreed with
        # the actual fetch (or the fetch returned nothing at all) -- a
        # fetched-and-empty result, the same condition the coverage probe
        # already reported a nonzero count for. It mirrors
        # fetch.py's own OBSERVATIONS_NO_POINTS branch -- omitted.reason
        # alone carries this, not the diagnostics channel too.
        return (
            omitted_observations(
                OBSERVATIONS_NO_POINTS,
                f"the bulk observation fetch for a defaulted place scope returned no rows "
                f"for any of the {len(entity_dcids)} places the coverage probe confirmed "
                "on this facet.",
                dispatched=len(entity_dcids),
                empty=len(entity_dcids),
            ),
            None,
        )
    observations = assemble_scope_bulk(
        reads,
        label_source=graph,
        on_names_unread=partial(note_degraded_read, "provenance names"),
        index=get_index(),
        member_constraints=member_constraints or {},
        target_currency_ref=conversion.target_unit_ref if conversion is not None else None,
    )
    delivered_entities = {entity for _sv, ((_role, entity),) in reads}
    # A place whose series the conversion left out holds data; the conversion warning names it.
    uncovered = set(entity_dcids) - delivered_entities - left_out_places

    warn: Warning | None = None
    if uncovered:
        warn = Warning(
            code=OBSERVATIONS_SCOPE_PARTIAL,
            severity="warn",
            message=(
                f"series for {len(delivered_entities)} of {len(entity_dcids)} places in the "
                f"resolved scope; {len(uncovered)} "
                + ("place has" if len(uncovered) == 1 else "places have")
                + " no data on the selected observation facet"
                + (" in its latest period." if latest is not None else ".")
            ),
        )
    return observations, left_out_warning(converted, also=warn)
