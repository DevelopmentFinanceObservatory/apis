"""The trailing probes of a resubmitted definite part, held apart from the request's clients.

A resubmit resolves in a worker thread, so its definite region carries relaunch closures that
the event loop calls once the region returns (`enrichment._launch_trailing_tasks`). The resolution
fixes the facts those closures read. The clients, deadline, pools and memos they read belong to
the request. `PromoteRelaunch` holds the facts, and `bind_relaunches` builds the
closures over one request's clients, so a part the response cache's memo stores holds no
closure and a hit binds the closures of the request it serves (`response_cache_memo`).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from materialise.facets import FacetSelection
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.events import PlaceOptionsEvent
from search.engine.extract import DateRequest
from search.engine.region.codes import _OptionProbeResult
from search.engine.region.facet_probe import FacetProbe
from search.engine.region.observations.bulk_scope import BulkScope
from search.engine.region.observations.dispatch import _observations_relaunch_closure
from search.engine.region.option_coverage import _launch_slot_options_task
from search.engine.region.result import FacetAxisFilter, RegionResult
from search.engine.region.state import EngineOptions
from search.engine.retrieval.read_plan import ReadPlan
from search.engine.shape import ShapeDraft
from search.models import (
    EntityOptions,
    IdentityUnitRequest,
    RoleAxis,
    SlotOptions,
    Spec,
)
from series_coverage.shared import CoverageSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class PromoteRelaunch:
    """What a resubmitted definite part's slot-options, place-options and observations
    probes read, apart from the request's clients.

    The slot-options probe reads the answer's shape, bindings, role entities, open flow axes,
    window, probe places, raw entity options and facet probe, and whether the identity binds a
    facet. The
    place-options probe reads the Spec it is launched for. The observations probe reads its
    members, observation entities, facet filter, the bulk scope's places (None for a
    per-pair read), the posted request-form unit, the elected unit and the narrowed plan.
    The slot-options and observations probes name the UN variables of a failed read from
    `variable_labels`, the names of the Spec's members (`un_system.member_labels`)."""

    shape_draft: ShapeDraft
    bindings: Sequence[SlotBindingDraft]
    entities: Mapping[str, list[str]]
    open_axes: frozenset[RoleAxis]
    date_request: DateRequest | None
    place_dcids: Sequence[str]
    entity_options: Sequence[EntityOptions] | None
    facet_probe: FacetProbe
    facets_pinned: bool
    sv_dcids: Sequence[str]
    obs_entities: Mapping[str, Sequence[str]]
    facet_filter: Mapping[str, FacetAxisFilter] | None
    bulk_places: tuple[str, ...] | None
    unit_request: IdentityUnitRequest | None
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None
    plan: ReadPlan | None
    variable_labels: Mapping[str, str]


def bind_relaunches(
    region: RegionResult,
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    coverage: CoverageSnapshot,
    options: EngineOptions,
) -> RegionResult:
    """`region` carrying the relaunch closures of its `promote_relaunch` over this request's
    clients, each where `options` streams it: slot options with `stream_slot_options`, place
    options with that flag, a materialise graph and a place-options pool, and observations
    with `stream_observations`. A region without `promote_relaunch` returns unchanged."""
    facts = region.promote_relaunch
    if facts is None:
        return region

    def _relaunch_slot_options(
        spec_id: str, opts: list[SlotOptions], facet_probe: FacetProbe | None = None
    ) -> asyncio.Task[_OptionProbeResult]:
        """A facet probe override lets a merged result judge its options over every member."""
        return _launch_slot_options_task(
            spec_id,
            facts.shape_draft,
            list(facts.bindings),
            opts,
            entities=facts.entities,
            open_axes=facts.open_axes,
            date_request=facts.date_request,
            graph=graph,
            sdmx=sdmx,
            materialise_graph=options.materialise_graph,
            place_dcids=list(facts.place_dcids),
            entity_options=list(facts.entity_options) if facts.entity_options is not None else None,
            facet_probe=facet_probe if facet_probe is not None else facts.facet_probe,
            coverage=coverage.index,
            deadline=options.deadline,
            facets_pinned=facts.facets_pinned,
            variable_labels=facts.variable_labels,
        )

    def _relaunch_place_options(spec: Spec) -> asyncio.Task[list[PlaceOptionsEvent]]:
        """The task reads `spec`'s own identity (`place_options.place_options_task`), so a
        same-shape merge's Spec needs no new inputs. A local import defers builder loading
        until region initialisation finishes."""
        from search.engine.region.place_options import place_options_task

        assert options.materialise_graph is not None and options.place_options_pool is not None
        return place_options_task(
            spec,
            graph=graph,
            sdmx=sdmx,
            materialise_graph=options.materialise_graph,
            materialise_sdmx=options.materialise_sdmx,
            pool=options.place_options_pool,
            coverage=coverage,
            population_memo=options.population_memo,
        )

    stream_place_options = (
        options.stream_slot_options
        and options.materialise_graph is not None
        and options.place_options_pool is not None
    )
    return dataclasses.replace(
        region,
        slot_options_relaunch=_relaunch_slot_options if options.stream_slot_options else None,
        place_options_relaunch=_relaunch_place_options if stream_place_options else None,
        observations_relaunch=(
            _observations_relaunch_closure(
                list(facts.sv_dcids),
                facts.obs_entities,
                graph=graph,
                materialise_graph=options.materialise_graph,
                materialise_sdmx=options.materialise_sdmx,
                facet_filter=facts.facet_filter,
                scope=(
                    BulkScope(places=facts.bulk_places, sdmx=sdmx)
                    if facts.bulk_places is not None
                    else None
                ),
                sdmx=sdmx,
                deadline=options.deadline,
                rate_cache=options.rate_cache,
                unit_request=facts.unit_request,
                elected=facts.elected,
                coverage=coverage,
                plan=facts.plan,
                variable_labels=facts.variable_labels,
            )
            if options.stream_observations
            else None
        ),
    )


def without_relaunches(region: RegionResult) -> RegionResult:
    """`region` holding no relaunch closure and no task, as the memo stores it."""
    assert (
        region.slot_options_task is None
        and region.place_options_task is None
        and region.observations_task is None
    ), "a resubmit region is stored before its trailing tasks launch"
    return dataclasses.replace(
        region,
        slot_options_relaunch=None,
        place_options_relaunch=None,
        observations_relaunch=None,
    )
