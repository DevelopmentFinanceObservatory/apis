"""Shared slot- and place-options relaunch closures for definite region results.

Callers bind each region's probe context; relaunches can target a merged spec. Resubmits
build separate closures because they invoke them from the event loop after a worker-thread run."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from materialise import GraphReader
from search.engine.bind import SlotBindingDraft
from search.engine.concurrency import create_owned_task
from search.engine.envelope import now_ms
from search.engine.events import PlaceOptionsEvent
from search.engine.region.codes import _OptionProbeResult
from search.engine.region.facet_probe import FacetProbe
from search.engine.region.option_coverage import _launch_slot_options_task
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from search.models import BaseIdentity, EntityOptions, RoleAxis, SlotOptions


def _relaunch_slot_options(
    spec_id: str,
    opts: list[SlotOptions],
    facet_probe: FacetProbe | None = None,
    *,
    state: ResolveState,
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    entities: Mapping[str, list[str]],
    open_axes: frozenset[RoleAxis],
    place_dcids: list[str] | None,
    entity_options: list[EntityOptions] | None,
    default_facet_probe: FacetProbe,
    facets_pinned: bool,
    variable_labels: Mapping[str, str],
) -> asyncio.Task[_OptionProbeResult]:
    """Relaunch with this survivor's probe context and supplied spec id and slot options.

    A supplied facet probe overrides the survivor's own, letting a merged result judge its
    options over every member. Each caller binds the other context when building the result,
    `facets_pinned` (whether the survivor's identity binds a facet) and `variable_labels`
    (the survivor's member names) among it."""
    return _launch_slot_options_task(
        spec_id,
        shape_draft,
        bindings,
        opts,
        entities=entities,
        open_axes=open_axes,
        date_request=state.date_request,
        graph=state.graph,
        sdmx=state.sdmx,
        materialise_graph=state.materialise_graph,
        place_dcids=place_dcids,
        entity_options=entity_options,
        facet_probe=facet_probe if facet_probe is not None else default_facet_probe,
        coverage=state.coverage.index,
        deadline=state.deadline,
        facets_pinned=facets_pinned,
        variable_labels=variable_labels,
    )


def _relaunch_place_options(
    spec_id: str,
    *,
    state: ResolveState,
    spec_identity: BaseIdentity,
    place_options_materialise_graph: GraphReader,
) -> asyncio.Task[list[PlaceOptionsEvent]]:
    """Relaunch place options for this identity under a supplied spec id.

    Same-shape merges preserve bound identity axes, so the confirmed identity is valid. The
    caller binds the context and ensures the place-options pool exists. Import the builder
    locally after package initialisation to avoid the import cycle; create its task on the
    event loop."""
    from search.engine.region.place_options import build_place_options_events

    assert state.place_options_pool is not None
    return create_owned_task(
        build_place_options_events(
            spec_id,
            spec_identity,
            graph=state.graph,
            sdmx=state.sdmx,
            materialise_graph=place_options_materialise_graph,
            materialise_sdmx=state.materialise_sdmx,
            pool=state.place_options_pool,
            coverage=state.coverage,
            population_memo=state.population_memo,
            start_ms=now_ms(),
        )
    )
