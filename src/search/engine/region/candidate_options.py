"""Probe options only for surviving candidates, using their grounded metadata."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable

from search.engine.closest_match import substituted_measure_fact
from search.engine.concurrency import create_owned_task
from search.engine.departures import warning_key
from search.engine.region.candidate_grounding import _GroundedCandidates
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA, _OptionProbeResult
from search.engine.region.convertibility import option_pins
from search.engine.region.facet_probe import FacetProbe, facet_probe
from search.engine.region.grounding import _place_confidence
from search.engine.region.option_coverage import (
    _build_and_enrich_entity_options,
    _gather_slot_options_events,
    _maybe_build_facet_options,
    _maybe_enrich_slot_options,
    _SlotOptionsProbe,
)
from search.engine.region.relaunch import _relaunch_slot_options
from search.engine.region.state import ResolveState
from search.engine.retrieval.elected_dataset import elected_datasets
from search.engine.retrieval.read_plan import preview_years
from search.engine.retrieval.retrieve import NoDataDraft
from search.models import SlotOptions, Spec, Warning
from search.render.heading import heading_for


def _candidate_facet_probe(
    state: ResolveState, grounded: _GroundedCandidates, spec: Spec
) -> FacetProbe:
    """What a surviving candidate's facet options are judged over: its captured StatVars
    and probe entities, its plan's years, and its identity's facet pins admitted per StatVar by
    the candidate's elected units."""
    sv_dcids, _ = grounded.cand_obs_entities.get(spec.spec_id, ([], {}))
    return facet_probe(
        sv_dcids,
        grounded.cand_probe_entities.get(spec.spec_id, {}),
        years=preview_years(grounded.cand_plans.get(spec.spec_id)),
        pins=option_pins(spec.definition.identity.facets, grounded.cand_elected.get(spec.spec_id)),
        datasets=(
            elected_datasets(grounded.cand_scopes[spec.spec_id][0].member_facts)
            if spec.spec_id in grounded.cand_scopes
            else None
        ),
    )


def _make_slot_options_task(
    state: ResolveState, grounded: _GroundedCandidates, surviving: list[Spec]
) -> asyncio.Task[_OptionProbeResult] | None:
    """Launch slot-option events for surviving specs with captured probe inputs.

    Per-spec entities and windows preserve buffered/SSE parity. Raw entity options are
    present only for the single-survivor definite path.
    """
    if not state.stream_slot_options:
        return None
    probes: list[_SlotOptionsProbe] = []
    for s in surviving:
        if s.spec_id not in grounded.cand_probe_inputs:
            continue
        _cand_shape, _cand_bindings, _cand_slot_options, _cand_place_dcids = (
            grounded.cand_probe_inputs[s.spec_id]
        )
        probes.append(
            (
                s.spec_id,
                _cand_shape,
                _cand_bindings,
                _cand_slot_options,
                grounded.cand_places[s.spec_id].roles.resolved_entities,
                _cand_place_dcids,
                grounded.cand_entity_options_raw.get(s.spec_id),
                _candidate_facet_probe(state, grounded, s),
                bool(s.definition.identity.facets),
            )
        )
    if not probes:
        return None
    # The request scope owns this detached task and reclaims it if no caller consumes it.
    return create_owned_task(
        _gather_slot_options_events(
            probes,
            date_request=state.date_request,
            graph=state.graph,
            sdmx=state.sdmx,
            materialise_graph=state.materialise_graph,
            coverage=state.coverage.index,
            deadline=state.deadline,
        )
    )


def _make_slot_options_relaunch(
    state: ResolveState, grounded: _GroundedCandidates, original_spec: Spec
) -> Callable[[str, list[SlotOptions], FacetProbe | None], asyncio.Task] | None:
    """Bind the surviving spec's probe context to the public relaunch callable.

    Expose the same optional facet probe override as the single-shape path.
    """
    if not state.stream_slot_options or original_spec.spec_id not in grounded.cand_probe_inputs:
        return None
    _shape, _bindings, _, _place_dcids = grounded.cand_probe_inputs[original_spec.spec_id]
    return functools.partial(
        _relaunch_slot_options,
        state=state,
        shape_draft=_shape,
        bindings=_bindings,
        entities=grounded.cand_places[original_spec.spec_id].roles.resolved_entities,
        place_dcids=_place_dcids,
        entity_options=grounded.cand_entity_options_raw.get(original_spec.spec_id),
        default_facet_probe=_candidate_facet_probe(state, grounded, original_spec),
        facets_pinned=bool(original_spec.definition.identity.facets),
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


def depart_to_measure_sibling(state: ResolveState, served: Spec) -> None:
    """Record on `state` that the region answers with `served`, a Spec on the elected
    shape's measure sibling. The record holds the `elected_shape_no_data` warning naming its
    `measure_label` and the `closest_match` departure it discloses, which states the
    measure noun its heading shows. A `measure_not_applied` departure on a property `served`
    binds goes with its warning, since the sibling applies what the elected shape dropped."""
    assert served.resolved.measure_label is not None, "a measure sibling carries its label"
    applied = {
        f"how:{slot.key.property.dcid}"
        for slot in served.resolved.slots
        if slot.key.property is not None and slot.binding.kind != "unbound"
    }
    retracted = {
        key
        for key, fact in state.departure_facts.items()
        if fact.code == "measure_not_applied" and fact.binding_ref in applied
    }
    state.warnings[:] = [w for w in state.warnings if warning_key(w) not in retracted]
    for key in retracted:
        del state.departure_facts[key]
    state.depart(
        _elected_no_data_measure_substituted_warning(state, served.resolved.measure_label),
        substituted_measure_fact(
            variable_text=state.variable,
            measure_noun=heading_for(served).heading.noun.inline_text,
        ),
    )


def _elected_no_data_measure_substituted_warning(
    state: ResolveState, served_measure_label: str
) -> Warning:
    _elected_reason = (
        state.mat_result.reason if isinstance(state.mat_result, NoDataDraft) else "no_observations"
    )
    return Warning(
        code=ELECTED_SHAPE_NO_DATA,
        severity="warn",
        message=(
            f"the model's elected shape {state.shape_draft.label!r} returned no data "
            f"({_elected_reason}); answering on its own measure sibling "
            f"{served_measure_label!r} instead, the only measure of this shape the "
            "graph has data for"
        ),
        params={"no_data": [state.shape_draft.label], "elected": [served_measure_label]},
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

    The probe controls the coverage gate and warning behaviour. Discarded candidates never
    trigger a read.
    """
    _results = await asyncio.gather(
        *(
            asyncio.to_thread(
                _maybe_build_facet_options,
                _candidate_facet_probe(state, grounded, s),
                graph=state.graph,
                coverage=state.coverage.index,
                include_option_coverage=state.include_option_coverage,
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
    if not state.include_option_coverage or spec.spec_id not in grounded.cand_probe_inputs:
        return spec
    _shape, _bindings, _raw_options, _place_dcids = grounded.cand_probe_inputs[spec.spec_id]
    _enriched, _warn = await asyncio.to_thread(
        _maybe_enrich_slot_options,
        _shape,
        _bindings,
        _raw_options,
        entities=grounded.cand_places[spec.spec_id].roles.resolved_entities,
        date_request=state.date_request,
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
) -> Spec:
    """Build entity options for the single surviving Spec, reusing its role-axis facts.

    Membership and counts are unconditional. Labels follow the coverage gate. Save raw
    options for a later slot-options relaunch.
    """
    if spec.spec_id not in grounded.cand_role_axis_facts:
        return spec
    _shape, _role_axis_facts, _place_dcids = grounded.cand_role_axis_facts[spec.spec_id]
    if not _role_axis_facts:
        return spec
    _sv_dcids, _ = grounded.cand_obs_entities.get(spec.spec_id, ([], {}))
    _entity_options_raw, _entity_options, _entity_warn = await asyncio.to_thread(
        _build_and_enrich_entity_options,
        _shape,
        grounded.cand_places[spec.spec_id].roles.resolved_entities,
        state.entity_objects,
        _sv_dcids,
        date_request=state.date_request,
        graph=state.graph,
        sdmx=state.sdmx,
        include_option_coverage=state.include_option_coverage,
        role_axis_facts=_role_axis_facts,
        place_dcids=_place_dcids,
    )
    grounded.cand_entity_options_raw[spec.spec_id] = _entity_options_raw
    if _entity_warn is not None:
        state.warnings.append(_entity_warn.model_copy(update={"spec_id": spec.spec_id}))
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"entity_options": _entity_options})}
    )


def _without_near_neighbor_confidence(
    state: ResolveState, grounded: _GroundedCandidates, spec: Spec
) -> Spec:
    """Regrade confidence without the near-neighbour penalty for a nested equivalent."""
    _available_at_place, _confidence = _place_confidence(
        destination_dcid=state.destination_dcid,
        available_at_place=spec.resolved.available_at_place,
        representative_score=grounded.cand_scores[spec.spec_id],
        candidate=True,
        retrieved=grounded.cand_retrieved[spec.spec_id],
    )
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"confidence": _confidence})}
    )
