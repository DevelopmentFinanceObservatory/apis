"""The trailing probes of a resubmitted definite part, held apart from the request's clients.

A resubmit resolves in a worker thread, so its definite region carries relaunch closures that
the event loop calls once the region returns (`enrichment._launch_trailing_tasks`). The resolution
fixes the facts those closures read. The clients, deadline, pools and memos they read belong to
the request. `PromoteRelaunch` holds the options probes' facts and the region's
`preview_read` the preview's (`dispatch.PreviewRead`), and `bind_relaunches` builds the
closures over one request's clients, so a part the response cache's memo stores holds no
closure and a hit binds the closures of the request it serves (`response_cache_memo`).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.events import PlaceOptionsEvent
from search.engine.extract import DateRequest
from search.engine.region.codes import _OptionProbeResult
from search.engine.region.facet_probe import FacetProbe
from search.engine.region.observations.dispatch import PreviewReader
from search.engine.region.option_coverage import _launch_slot_options_task
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.shape import ShapeDraft
from search.models import (
    EntityOptions,
    RoleAxis,
    SlotOptions,
    Spec,
)
from series_coverage.shared import CoverageSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class PromoteRelaunch:
    """What a resubmitted definite part's slot-options and place-options probes read, apart
    from the request's clients. Its observations probe reads the region's `preview_read`.

    The slot-options probe reads the answer's shape, bindings, role entities, open flow axes,
    window, probe places, raw entity options and facet probe, and whether the identity binds a
    facet. The place-options probe reads the Spec it is launched for.
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
    with `stream_observations`, reading the region's `preview_read` (`PreviewReader`). A
    region without `promote_relaunch` returns unchanged."""
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
            PreviewReader(
                read=region.preview_read,
                graph=graph,
                sdmx=sdmx,
                materialise_graph=options.materialise_graph,
                materialise_sdmx=options.materialise_sdmx,
                rate_cache=options.rate_cache,
                deadline=options.deadline,
                coverage=coverage,
            ).relaunch(facts.variable_labels)
            if options.stream_observations and region.preview_read is not None
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
