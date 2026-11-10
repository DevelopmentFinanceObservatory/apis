"""Probe options only for surviving candidates, using their grounded metadata."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Mapping

from search.engine.concurrency import create_owned_task
from search.engine.region.candidate_grounding import _GroundedCandidates
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA, _OptionProbeResult
from search.engine.region.convertibility import option_pins
from search.engine.region.facet_probe import FacetProbe, facet_probe
from search.engine.region.option_coverage import (
    _build_and_enrich_entity_options,
    _gather_slot_options_events,
    _maybe_build_facet_options,
    _maybe_enrich_slot_options,
    _SlotOptionsProbe,
)
from search.engine.region.relaunch import _relaunch_slot_options
from search.engine.region.state import ResolveState
from search.engine.region.un_system import member_labels
from search.engine.retrieval.elected_dataset import elected_datasets
from search.engine.retrieval.read_plan import preview_years
from search.engine.retrieval.retrieve import NoDataDraft
from search.models import EntityOptions, SlotOptions, Spec, Warning


def _candidate_facet_probe(
    state: ResolveState, grounded: _GroundedCandidates, spec: Spec
) -> FacetProbe:
    """What a surviving candidate's facet options are judged over: its captured StatVars
    and probe entities, its plan's years, and its identity's facet pins admitted per StatVar by
    the candidate's elected units."""
    entry = grounded[spec.spec_id]
    return facet_probe(
        entry.sv_dcids,
        entry.probe_entities,
        years=preview_years(entry.plan),
        pins=option_pins(spec.definition.identity.facets, entry.elected),
        datasets=elected_datasets(entry.candidate.mat.member_facts),
    )


def _make_slot_options_task(
    state: ResolveState,
    grounded: _GroundedCandidates,
    surviving: list[Spec],
    *,
    entity_options_raw: Mapping[str, list[EntityOptions]],
) -> asyncio.Task[_OptionProbeResult] | None:
    """Launch slot-option events for surviving specs with captured probe inputs.

    Per-spec entities and windows preserve buffered/SSE parity. `entity_options_raw` holds
    the raw entity options `_attach_candidate_entity_options` built, by `spec_id`, present
    only for the single-survivor definite path.
    """
    if not state.stream_slot_options:
        return None
    probes: list[_SlotOptionsProbe] = []
    for s in surviving:
        entry = grounded[s.spec_id]
        if (inputs := entry.probe_inputs) is None:
            continue
        _cand_shape, _cand_bindings, _cand_slot_options, _cand_place_dcids = inputs
        probes.append(
            (
                s.spec_id,
                _cand_shape,
                _cand_bindings,
                _cand_slot_options,
                entry.candidate.place.roles.resolved_entities,
                entry.candidate.place.roles.open_axes,
                _cand_place_dcids,
                entity_options_raw.get(s.spec_id),
                _candidate_facet_probe(state, grounded, s),
                bool(s.definition.identity.facets),
                member_labels((s,)),
                entry.candidate.date_request,
            )
        )
    if not probes:
        return None
    # The request scope owns this detached task and reclaims it if no caller consumes it.
    return create_owned_task(
        _gather_slot_options_events(
            probes,
            graph=state.graph,
            sdmx=state.sdmx,
            materialise_graph=state.materialise_graph,
            coverage=state.coverage.index,
            deadline=state.deadline,
        )
    )


def _make_slot_options_relaunch(
    state: ResolveState,
    grounded: _GroundedCandidates,
    original_spec: Spec,
    *,
    entity_options_raw: list[EntityOptions] | None,
) -> Callable[[str, list[SlotOptions], FacetProbe | None], asyncio.Task] | None:
    """Bind the surviving spec's probe context to the public relaunch callable, with the raw
    entity options `_attach_candidate_entity_options` built for it.

    Expose the same optional facet probe override as the single-shape path.
    """
    entry = grounded[original_spec.spec_id]
    if not state.stream_slot_options or (inputs := entry.probe_inputs) is None:
        return None
    _shape, _bindings, _, _place_dcids = inputs
    return functools.partial(
        _relaunch_slot_options,
        state=state,
        shape_draft=_shape,
        bindings=_bindings,
        entities=entry.candidate.place.roles.resolved_entities,
        open_axes=entry.candidate.place.roles.open_axes,
        place_dcids=_place_dcids,
        entity_options=entity_options_raw,
        default_facet_probe=_candidate_facet_probe(state, grounded, original_spec),
        facets_pinned=bool(original_spec.definition.identity.facets),
        variable_labels=member_labels((original_spec,)),
        date_request=entry.candidate.date_request,
    )


def _elected_sibling_warning(*, no_data: list[str], elected: list[str]) -> Warning:
    """Warn once per election that a sibling reading with data replaced readings without.

    ``no_data`` and ``elected`` are shape labels, each listed once in first-seen order.
    """
    return Warning(
        code=ELECTED_SHAPE_NO_DATA,
        severity="info",
        message=(
            "the originally chosen reading had no observations at "
            "this place; a sibling reading confirmed with real data "
            "was elected instead."
        ),
        params={"no_data": list(dict.fromkeys(no_data)), "elected": list(dict.fromkeys(elected))},
    )


def _elected_no_data_warning(state: ResolveState, alternates: list[Spec]) -> Warning:
    _elected_reason = (
        state.mat_result.reason if isinstance(state.mat_result, NoDataDraft) else "no_observations"
    )
    return Warning(
        code=ELECTED_SHAPE_NO_DATA,
        severity="warn",
        message=(
            f"the model's elected shape {state.shape_draft.label!r} returned no data "
            f"({_elected_reason}); showing the also-plausible alternate(s) instead "
            "of a single confident answer"
        ),
        params={
            "no_data": [state.shape_draft.label],
            "elected": list(dict.fromkeys(spec.resolved.shape.label for spec in alternates)),
        },
    )


def _elected_no_data_twin_substituted_warning(state: ResolveState, served_label: str) -> Warning:
    """Note that a registry twin of the elected shape (`EquivalentCluster`) answers in its
    place. The twin reads the same indicator, so the note is informational."""
    _elected_reason = (
        state.mat_result.reason if isinstance(state.mat_result, NoDataDraft) else "no_observations"
    )
    return Warning(
        code=ELECTED_SHAPE_NO_DATA,
        severity="info",
        message=(
            f"the model's elected shape {state.shape_draft.label!r} returned no data "
            f"({_elected_reason}); answering on {served_label!r} instead, another series "
            "of the same indicator that holds data here"
        ),
        params={"no_data": [state.shape_draft.label], "elected": [served_label]},
    )


async def _attach_candidate_facet_options(
    state: ResolveState, grounded: _GroundedCandidates, survivors: list[Spec]
) -> list[Spec]:
    """Attach facet options to each surviving spec using its captured entities.

    A streamed request with option coverage delivers them in trailing events
    (`_make_slot_options_task`) instead. Discarded candidates never trigger a read.
    """
    if state.stream_slot_options:
        return survivors
    _results = await asyncio.gather(
        *(
            asyncio.to_thread(
                _maybe_build_facet_options,
                _candidate_facet_probe(state, grounded, s),
                graph=state.graph,
                coverage=state.coverage.index,
                deadline=state.deadline,
            )
            for s in survivors
        )
    )
    _updated: list[Spec] = []
    for s, (_facet_options, _facet_warn) in zip(survivors, _results, strict=True):
        if _facet_warn is not None:
            state.warnings.append(_facet_warn.model_copy(update={"spec_id": s.spec_id}))
        if _facet_options:
            s = s.model_copy(
                update={"resolved": s.resolved.model_copy(update={"facet_options": _facet_options})}
            )
        _updated.append(s)
    return _updated


async def _attach_candidate_slot_options(
    state: ResolveState, grounded: _GroundedCandidates, spec: Spec
) -> Spec:
    """Enrich one surviving spec's slot options from its captured probe inputs."""
    entry = grounded[spec.spec_id]
    if not state.include_option_coverage or (inputs := entry.probe_inputs) is None:
        return spec
    _shape, _bindings, _raw_options, _place_dcids = inputs
    _enriched, _warn = await asyncio.to_thread(
        _maybe_enrich_slot_options,
        _shape,
        _bindings,
        _raw_options,
        entities=entry.candidate.place.roles.resolved_entities,
        open_axes=entry.candidate.place.roles.open_axes,
        date_request=entry.candidate.date_request,
        graph=state.graph,
        materialise_graph=state.materialise_graph,
        place_dcids=_place_dcids,
        include_option_coverage=True,
        facets_pinned=bool(spec.definition.identity.facets),
    )
    if _warn is not None:
        state.warnings.append(_warn.model_copy(update={"spec_id": spec.spec_id}))
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"slot_options": _enriched})}
    )


async def _attach_candidate_entity_options(
    state: ResolveState, grounded: _GroundedCandidates, spec: Spec
) -> tuple[Spec, list[EntityOptions] | None]:
    """Build entity options for the single surviving Spec, reusing its role-axis facts.

    Membership and counts are unconditional. Labels follow the coverage gate. Return the
    Spec with its options and the raw options a later slot-options probe or relaunch reads,
    None when its shape has no role-axis facts.
    """
    entry = grounded[spec.spec_id]
    if not entry.role_axis_facts:
        return spec, None
    _entity_options_raw, _entity_options, _entity_warn = await asyncio.to_thread(
        _build_and_enrich_entity_options,
        entry.candidate.shape,
        entry.candidate.place.roles.resolved_entities,
        state.entity_objects,
        entry.sv_dcids,
        date_request=entry.candidate.date_request,
        graph=state.graph,
        sdmx=state.sdmx,
        include_option_coverage=state.include_option_coverage,
        role_axis_facts=entry.role_axis_facts,
        place_dcids=entry.candidate.place.probe_places,
    )
    if _entity_warn is not None:
        state.warnings.append(_entity_warn.model_copy(update={"spec_id": spec.spec_id}))
    spec = spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"entity_options": _entity_options})}
    )
    return spec, _entity_options_raw
