"""Collapse grounded alternates into a `RegionResult` after unit filtering, measure folding,
equivalent-reading diversion and no-data checks.
"""

from __future__ import annotations

import asyncio
import functools
from typing import cast

from materialise import GraphReader
from materialise.facets import native_currency_iso
from materialise.rate_tables import RateTableCache
from search.engine.clients.graph import Facet
from search.engine.departures import measure_not_applied, qualifier_property
from search.engine.envelope import _emit_step, now_ms
from search.engine.reading_coverage import covers
from search.engine.region.bound_unit_gate import bound_unit_gate_reason
from search.engine.region.candidate_admission import _AlternatesAdmission
from search.engine.region.candidate_grounding import (
    _GroundedCandidates,
    ground_candidates,
    restate_scope_warning,
)
from search.engine.region.candidate_measure_fold import (
    _facet_provenance_ids,
    _fold_measure_sibling_candidates,
    _shapes_are_measure_siblings,
)
from search.engine.region.candidate_options import (
    _attach_candidate_entity_options,
    _attach_candidate_facet_options,
    _attach_candidate_slot_options,
    _candidate_facet_probe,
    _elected_no_data_twin_substituted_warning,
    _elected_no_data_warning,
    _make_slot_options_relaunch,
    _make_slot_options_task,
    _without_near_neighbor_confidence,
    depart_to_measure_sibling,
)

# _UNIT_CHECK_NOT_PROBED/_unbound_query_qualifiers live in region/candidates.py,
# which imports assemble_candidates (this module) locally, inside
# resolve_candidates' own body, to avoid the cycle a module-level import
# would make here: by the time resolve_candidates first runs and takes
# that local import, region/candidates.py has already finished its own
# top-level execution, so this module-level import back into it is safe.
from search.engine.region.candidates import (
    _UNIT_CHECK_NOT_PROBED,
    _Candidate,
    _unbound_query_qualifiers,
)
from search.engine.region.codes import (
    NO_MEMBER_HAS_DATA_AT_PLACE,
    ONLY_ALTERNATE_HAS_DATA,
    QUERY_QUALIFIER_DROPPED,
    _elect_no_data_reason,
)
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.equivalent_series import shared_cluster
from search.engine.region.observations.dispatch import (
    _attach_observations,
    _resolve_spec_observations,
)
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.phase_shared import (
    UNIT_CONVERSION_REFUSALS,
    UnitConversionRefusal,
    unit_conversion_refusal,
)
from search.engine.region.place_context import (
    install_place_warnings,
    stamp_offered_place_warnings,
)
from search.engine.region.related_read import (
    related_read_for,
    start_elected_related_read,
    with_related,
)
from search.engine.region.relaunch import _relaunch_place_options
from search.engine.region.result import RegionResult
from search.engine.region.sentence_events import _build_definite_sentence_event
from search.engine.region.state import ResolveState, raised_facts
from search.engine.retrieval.read_plan import ReadPlan, plan_coverage, serves_nothing
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.unit.bind import with_unit_lexicon_version
from search.models import (
    IdentityUnitRequest,
    Spec,
    Warning,
)


def _offer_place_warnings(
    state: ResolveState, grounded: _GroundedCandidates, offered: list[Spec]
) -> None:
    """Stamp each spec in `offered`, in menu order, with the place warnings of its own
    context that the shape in play's do not hold (`stamp_offered_place_warnings`)."""
    stamp_offered_place_warnings(
        state.warnings,
        [(spec.spec_id, grounded.cand_places[spec.spec_id]) for spec in offered],
        departure_facts=state.departure_facts,
    )


def _apply_bound_unit_filter(
    state: ResolveState, unique_specs: list[Spec], grounded: _GroundedCandidates
) -> list[Spec] | RegionResult:
    """Filter candidates against a bound currency, price basis or base year after deduplication and
    before measure folding. This order keeps a compatible sibling available as its own candidate.
    An empty input proceeds to the elected shape's no-data reason.

    Each candidate passes the single-shape gate over its own read plan, which fetches nothing:
    the plan serves every StatVar under its elected unit (`bound_unit_gate.bound_unit_gate_reason`)
    and every row it serves converts, as grounding found it (`cand_unit_coverage`). A candidate
    whose rows do not convert is dropped, and its refusal is recorded beside its election's
    (`cand_unit_refusals`). A passing candidate whose conversion omits the rows of entities
    no currency names records its warning (`cand_unit_omitted`). Return the surviving specs
    or a terminal no-match result.
    """
    if state.bound_unit_outcome is None or not unique_specs:
        return unique_specs
    available: list[Spec] = []
    _matched_specs: list[Spec] = []
    for s in unique_specs:
        if s.resolved.available_at_place is False:
            # Preserve the carrier for the `NO_MEMBER_HAS_DATA_AT_PLACE` refusal below.
            _matched_specs.append(s)
            continue
        _s_svs, _s_ents = grounded.cand_obs_entities.get(s.spec_id, ([], {}))
        _gate_plan = grounded.cand_gate_plans.get(s.spec_id, grounded.cand_plans.get(s.spec_id))
        if bound_unit_gate_reason(_gate_plan, _s_ents, _s_svs) is None:
            available.append(s)
    for s in available:
        coverage = grounded.cand_unit_coverage.get(s.spec_id, UnitCoverage())
        if coverage.refusal is None:
            _matched_specs.append(s)
            if (omitted := coverage.omitted_warning(spec_id=s.spec_id)) is not None:
                grounded.cand_unit_omitted[s.spec_id] = omitted
            continue
        sv, exc = coverage.refusal
        grounded.cand_unit_refusals[s.spec_id] = {
            **grounded.cand_unit_refusals.get(s.spec_id, {}),
            sv: exc,
        }
    _matched_specs.sort(key=unique_specs.index)
    if not _matched_specs:
        _refusal = _first_conversion_refusal(unique_specs, grounded)
        if _refusal is not None:
            _refusing, _exc = _refusal
            # A bound outcome sets `unit_request`. Ty treats attributes as optional.
            return unit_conversion_refusal(
                state,
                _exc,
                unit_request=cast(IdentityUnitRequest, state.unit_request),
                candidate=_refusing,
            )
        return state.no_data("facet_not_available", llm_usage=state.var_usage)
    grounded.cand_scores = {s.spec_id: grounded.cand_scores[s.spec_id] for s in _matched_specs}
    grounded.cand_retrieved = {
        s.spec_id: grounded.cand_retrieved[s.spec_id] for s in _matched_specs
    }
    return _matched_specs


def _first_conversion_refusal(
    unique_specs: list[Spec], grounded: _GroundedCandidates
) -> tuple[_Candidate, UnitConversionRefusal] | None:
    """Return the conversion refusal that answers when no candidate serves the bound unit, with
    the candidate that raised it.

    The first candidate in candidate order, which puts the elected shape first, supplies it,
    from its first refused StatVar. A candidate refused for conversion holds data the request
    cannot convert, so its refusal outranks `facet_not_available`. A candidate whose refusals
    are all missing facets is passed over. The answer describes the returned candidate, so a
    resubmit of its definition meets the same refusal.
    """
    for s in unique_specs:
        for exc in grounded.cand_unit_refusals.get(s.spec_id, {}).values():
            if isinstance(exc, UNIT_CONVERSION_REFUSALS):
                return grounded.cand_candidates[s.spec_id], exc
    return None


def _handle_sparse_survivors(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    served_twin_id: str | None,
) -> RegionResult | None:
    """Handle zero survivors and a lone non-elected survivor after twin diversion. Return a
    terminal result or `None` when at least two candidates, the elected survivor, or the twin
    answering for an elected shape with no data (`served_twin_id`) remain.
    """
    if len(unique_specs) == 0:
        # Prefer the elected shape's earlier-ranked failure, such as `no_matching_slice`, over the
        # `no_observations` fallback. No candidate survived to attach as a match.
        _elected_reason = _elect_no_data_reason(
            state.mat_result.reason if isinstance(state.mat_result, NoDataDraft) else None,
            fallback="no_observations",
        )
        return state.no_data(_elected_reason, llm_usage=state.var_usage)
    if len(unique_specs) == 1 and unique_specs[0].resolved.available_at_place is False:
        # Flagged-unavailable carriers belong in no-data responses.
        _offer_place_warnings(state, grounded, unique_specs)
        return state.no_data(
            "no_observations",
            flagged_match=(unique_specs[0],),
            gate=NO_MEMBER_HAS_DATA_AT_PLACE,
            llm_usage=state.var_usage,
        )
    if len(unique_specs) == 1 and not state.elected_has_data:
        if unique_specs[0].spec_id == served_twin_id:
            # A registry twin reads the elected indicator itself (`_apply_twin_diversion`).
            state.warnings.append(
                _elected_no_data_twin_substituted_warning(
                    state, unique_specs[0].resolved.shape.label
                )
            )
            return None
        _survivor_shape_for_sibling_check, _, _ = grounded.cand_role_axis_facts[
            unique_specs[0].spec_id
        ]
        _survivor_is_measure_sibling_of_elected = (
            _shapes_are_measure_siblings(state.shape_draft, _survivor_shape_for_sibling_check)
            and unique_specs[0].resolved.measure_label is not None
        )
        if _survivor_is_measure_sibling_of_elected:
            # A real measure sibling can answer when the elected shape has no data, since no elected
            # Spec exists to carry it as an alternative measure. Require `measure_label`.
            # The warning and its closest_match departure record the substitution.
            depart_to_measure_sibling(state, unique_specs[0])
            return None
        # `CandidateSet` requires at least two specs. A lone unelected survivor yields no-data with
        # the elected shape's failure and the materialised alternate. `grounded_near_match` keeps
        # unresolved-entity and unresolved-variable reasons from contradicting that match.
        _elected_reason = _elect_no_data_reason(
            state.mat_result.reason if isinstance(state.mat_result, NoDataDraft) else None,
            fallback="no_observations",
            grounded_near_match=True,
        )
        _offer_place_warnings(state, grounded, unique_specs)
        return state.no_data(
            _elected_reason,
            alternate_match=unique_specs[0],
            gate=ONLY_ALTERNATE_HAS_DATA,
            llm_usage=state.var_usage,
        )
    return None


def _in_currency(facets: list[Facet], currency: str | None) -> bool:
    """Whether no facet reports a unit other than `currency`, the one a twin cluster's
    members are equivalent in (`EquivalentCluster.currency`). Always true for a cluster
    naming none. A facet stating no unit passes, as nothing shows it reads another.
    Grounding keeps a twin's facets in that currency where it reports others too
    (`candidate_grounding._twin_unit_filters`)."""
    return currency is None or all(
        f.unit is None or native_currency_iso(f.unit) == currency for f in facets
    )


def _covers(preferred: ReadPlan | None, elected: ReadPlan | None) -> bool:
    """Whether the `preferred` twin's plan serves a point in every requested year the
    `elected` twin's may serve one in, at every place it serves (`reading_coverage.covers`
    over `plan_coverage`, in the elected plan's years). Unknown plans, and a preferred plan
    serving nothing, cover nothing."""
    if preferred is None or elected is None or serves_nothing(preferred):
        return False
    return covers(plan_coverage(preferred), plan_coverage(elected), elected.years)


def _preferred_twin(
    grounded: _GroundedCandidates, elected_spec_id: str, twins: list[Spec]
) -> Spec | None:
    """The twin the registry prefers to the elected spec (`EquivalentCluster.preferred`),
    when it covers the elected spec at the question's places and dates (`_covers`); else
    `None`, and the elected spec answers with the preferred twin among its equivalent
    readings."""
    elected_dcids = grounded.cand_obs_entities[elected_spec_id][0]
    for twin in twins:
        twin_dcids = grounded.cand_obs_entities[twin.spec_id][0]
        cluster = shared_cluster(elected_dcids, twin_dcids)
        if (
            cluster is not None
            and cluster.preferred in twin_dcids
            and cluster.preferred not in elected_dcids
            and _covers(
                grounded.cand_plans.get(twin.spec_id), grounded.cand_plans.get(elected_spec_id)
            )
        ):
            return twin
    return None


def _apply_twin_diversion(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    near_neighbor_fired_spec_id: str | None,
) -> tuple[list[Spec], str | None, str | None]:
    """Move registry-confirmed encodings of the elected indicator into `equivalent_readings`.
    Return the remaining specs, any surviving near-neighbour fired-spec id, and the id of the
    twin that answers for an elected shape with no data, if one does.

    When the elected spec survives with data, the registry's preferred twin (`_preferred_twin`)
    answers in its place, with the elected spec listed among its equivalent readings, when it
    holds data in every requested year the elected spec does, at each place. When the
    elected shape has no data at the question's places, a surviving twin with data answers
    for it, the cluster's preferred member first, with the other twins listed; the elected
    shape holds nothing to read, so it is not listed. A twin either side of which reports a
    unit outside the cluster's currency (`_in_currency`) is dropped, because it was admitted
    as the same concept and does not read it in the served unit.
    """
    _elected_spec = next((s for s in unique_specs if s.spec_id == grounded.elected_spec_id), None)
    if _elected_spec is not None:
        _elected_sv_dcids = grounded.cand_obs_entities[_elected_spec.spec_id][0]
        _elected_facets = grounded.cand_confirmed_facets.get(_elected_spec.spec_id, [])
    elif not state.elected_has_data and isinstance(state.mat_result, Materialised | NoDataDraft):
        _elected_sv_dcids = state.mat_result.sv_dcids or []
        _elected_facets = []
    else:
        return unique_specs, near_neighbor_fired_spec_id, None
    _elected_provenance_ids = _facet_provenance_ids(_elected_facets)
    _twins: list[Spec] = []
    _preferred_members: set[str] = set()
    _dropped_ids: set[str] = set()
    for s in unique_specs:
        if s is _elected_spec or s.resolved.available_at_place is False:
            continue
        _candidate_sv_dcids = grounded.cand_obs_entities[s.spec_id][0]
        cluster = shared_cluster(_elected_sv_dcids, _candidate_sv_dcids)
        if cluster is None:
            continue
        _candidate_facets = grounded.cand_confirmed_facets.get(s.spec_id, [])
        if not all(
            _in_currency(facets, cluster.currency)
            for facets in (_elected_facets, _candidate_facets)
        ):
            _dropped_ids.add(s.spec_id)
            continue
        # With data on both sides, require known, disjoint provenance sets. This blocks
        # same-source twins split across tables. A stale registry entry with disjoint
        # provenance can pass. An elected shape with no data has no reading to duplicate.
        _candidate_provenance_ids = _facet_provenance_ids(_candidate_facets)
        if _elected_spec is None or (
            _elected_provenance_ids
            and _candidate_provenance_ids
            and _elected_provenance_ids.isdisjoint(_candidate_provenance_ids)
        ):
            _twins.append(s)
            if cluster.preferred in _candidate_sv_dcids:
                _preferred_members.add(s.spec_id)
    if not _twins and not _dropped_ids:
        return unique_specs, near_neighbor_fired_spec_id, None
    # Preserve the elected spec's real bind result; only non-elected survivors can become twins.
    _twin_ids = {t.spec_id for t in _twins}
    # Retract the warning when its fired shape becomes an `equivalent_readings` annotation, an
    # answer in the elected shape's place, or is dropped. Warnings for remaining candidates
    # stay valid.
    if near_neighbor_fired_spec_id in _twin_ids | _dropped_ids:
        # Clear the fired index and warning, then regrade both readings without the gate's penalty.
        if state.near_neighbor_warning is not None:
            state.warnings.remove(state.near_neighbor_warning)
        state.near_neighbor_warning = None
        state.near_neighbor_fired_index = None
        near_neighbor_fired_spec_id = None
        if _elected_spec is not None:
            _elected_spec = _without_near_neighbor_confidence(state, grounded, _elected_spec)
        _twins = [_without_near_neighbor_confidence(state, grounded, twin) for twin in _twins]
    # The answer takes the place of the first reading: the elected spec, or the first twin
    # standing for an elected shape with no data.
    _served_twin_id: str | None = None
    _listed: list[Spec] = []
    if _elected_spec is not None or _twins:
        if _elected_spec is not None:
            _readings = [_elected_spec, *_twins]
            _primary = _preferred_twin(grounded, _elected_spec.spec_id, _twins) or _elected_spec
        else:
            _readings = _twins
            _primary = next((t for t in _twins if t.spec_id in _preferred_members), _twins[0])
            _served_twin_id = _primary.spec_id
        _listed = [r for r in _readings if r is not _primary]
        if _listed:
            _primary = _primary.model_copy(
                update={
                    "resolved": _primary.resolved.model_copy(
                        update={"equivalent_readings": _listed}
                    )
                }
            )
        _anchor_id = _readings[0].spec_id
        unique_specs = [
            _primary if s.spec_id == _anchor_id else s
            for s in unique_specs
            if s.spec_id == _anchor_id or s.spec_id not in _twin_ids | _dropped_ids
        ]
    else:
        # Every twin of an elected shape with no data read another currency.
        unique_specs = [s for s in unique_specs if s.spec_id not in _dropped_ids]
    _listed_ids = {t.spec_id for t in _listed} | _dropped_ids
    grounded.cand_scores = {k: v for k, v in grounded.cand_scores.items() if k not in _listed_ids}
    grounded.cand_retrieved = {
        k: v for k, v in grounded.cand_retrieved.items() if k not in _listed_ids
    }
    return unique_specs, near_neighbor_fired_spec_id, _served_twin_id


async def _assemble_definite(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    rate_cache: RateTableCache | None,
) -> RegionResult:
    """Build the definite response for one surviving candidate. An unbound query qualifier returns
    no-data. A no-data elected shape reaches here only through measure-sibling or registry-twin
    substitution.
    A survivor from an alternate answers over its own place context, so its place warnings
    become the answer's own (`install_place_warnings`).
    """
    if unique_specs[0].spec_id != grounded.elected_spec_id:
        install_place_warnings(
            state.warnings,
            grounded.cand_places[unique_specs[0].spec_id],
            departure_facts=state.departure_facts,
        )
    # Qualifier binding is computed against the elected shape's declared properties.
    _survivor_unbound_qualifiers = _unbound_query_qualifiers(
        state.query_qualifiers, unapplied=state.unapplied_qualifier_phrases
    )
    if _survivor_unbound_qualifiers:
        return state.no_data(
            "qualifier_not_bound",
            flagged_match=(unique_specs[0],),
            gate=None,
            llm_usage=state.var_usage,
            unbound_qualifiers=_survivor_unbound_qualifiers,
        )
    # A `has_data` result with no StatVars signals a missing series.
    if not unique_specs[0].resolved.members:
        return state.no_data("no_observations", llm_usage=state.var_usage)
    _survivor_spec = unique_specs[0]
    if (omitted := grounded.cand_unit_omitted.get(_survivor_spec.spec_id)) is not None:
        state.warnings.append(omitted)
    _obs_svs, _obs_entities = grounded.cand_obs_entities.get(_survivor_spec.spec_id, ([], {}))
    # The related reads run beside the streamed preview and the options below.
    _survivor_id = _survivor_spec.spec_id
    _related_read = related_read_for(
        state,
        stat_vars=_obs_svs,
        measure_partition=grounded.cand_measure_partitions.get(_survivor_id),
        places=grounded.cand_places[_survivor_id].roles.roles,
        arcs=grounded.cand_role_axis_facts[_survivor_id][0].sv_arc_facts,
    )
    if state.bound_unit_outcome is not None:
        # State the unit lexicon version. A streamed answer also states its preview's facets
        # and warning with its immediate response; a flagged-unavailable carrier holds nothing
        # to read and states `_UNIT_CHECK_NOT_PROBED`.
        _survivor_spec = with_unit_lexicon_version(_survivor_spec, state.bound_unit_outcome)
        if state.stream_observations:
            # An inline preview states its own warning (`_resolve_spec_observations`).
            _streamed = (
                _UNIT_CHECK_NOT_PROBED
                if _survivor_spec.resolved.available_at_place is False
                else await asyncio.to_thread(
                    _build_observations,
                    grounded.cand_plans.get(_survivor_spec.spec_id),
                    _obs_svs,
                    _obs_entities,
                    graph=state.graph,
                    materialise_graph=state.materialise_graph,
                    materialise_sdmx=state.materialise_sdmx,
                    # A ref-form unit filters literally; a request converts.
                    facet_filter=(
                        None if state.unit_request is not None else state.unit_facet_filter
                    ),
                    scope=grounded.cand_bulk_scope.get(_survivor_spec.spec_id),
                    sdmx=state.sdmx,
                    deadline=state.deadline,
                    rate_cache=rate_cache,
                    unit_request=state.unit_request,
                    elected=grounded.cand_elected.get(_survivor_spec.spec_id),
                    coverage=state.coverage,
                    member_facts=grounded.cand_candidates[_survivor_id].mat.member_facts,
                )
            )
            _survivor_spec = _attach_observations(
                _survivor_spec, _streamed, state.warnings, include_points=False
            )
    # Keep this timing key identical to the single-shape path.
    _options_t0 = now_ms()
    _survivor_spec = await _attach_candidate_slot_options(state, grounded, _survivor_spec)
    (_survivor_spec,) = await _attach_candidate_facet_options(state, grounded, [_survivor_spec])
    _survivor_spec = await _attach_candidate_entity_options(state, grounded, _survivor_spec)
    state.timing["options"] = now_ms() - _options_t0
    _family = state.family_total_absent_departures((_survivor_spec,))
    # Emit the final sentence after facet options, matching the single-shape response. Rendering
    # uses no observations, which SSE sends in a later event.
    if state.emit is not None:
        await state.emit(
            _build_definite_sentence_event(_survivor_spec, variable_index=state.variable_index)
        )
    # Place-option tasks run only when `materialise_graph` is available.
    _place_options_materialise_graph = state.materialise_graph
    # Bind the identity confirmed by materialisation. Candidate merging preserves that identity.
    _relaunch_place_options_bound = functools.partial(
        _relaunch_place_options,
        state=state,
        spec_identity=_survivor_spec.definition.identity,
        place_options_materialise_graph=cast(GraphReader, _place_options_materialise_graph),
    )
    _stream_place_options = (
        state.stream_slot_options
        and _place_options_materialise_graph is not None
        and state.place_options_pool is not None
    )
    _place_options_task = (
        _relaunch_place_options_bound(_survivor_spec.spec_id) if _stream_place_options else None
    )
    _survivor_spec = await with_related(_survivor_spec, _related_read)
    # This is the final await. Passing a live observation task to `RegionResult` keeps it reachable
    # if cancellation follows.
    _survivor_spec, _obs_task, _obs_relaunch = await _resolve_spec_observations(
        _survivor_spec,
        _obs_svs,
        _obs_entities,
        graph=state.graph,
        materialise_graph=state.materialise_graph,
        materialise_sdmx=state.materialise_sdmx,
        include_observations=state.include_observations,
        stream_observations=state.stream_observations,
        warnings=state.warnings,
        facet_filter=(None if state.unit_request is not None else state.unit_facet_filter),
        scope=grounded.cand_bulk_scope.get(_survivor_spec.spec_id),
        sdmx=state.sdmx,
        deadline=state.deadline,
        rate_cache=rate_cache,
        unit_request=state.unit_request,
        elected=grounded.cand_elected.get(_survivor_spec.spec_id),
        coverage=state.coverage,
        plan=grounded.cand_plans.get(_survivor_spec.spec_id),
        member_facts=grounded.cand_candidates[_survivor_id].mat.member_facts,
    )
    return RegionResult(
        variable_text=state.variable,
        status="definite",
        specs=(_survivor_spec,),
        no_data_reason=None,
        warnings=(*state.warnings, *(warning for warning, _fact in _family)),
        departure_facts={**state.departure_facts, **raised_facts(_family)},
        timing_by_step=dict(state.timing),
        llm_usage=state.var_usage,
        slot_options_task=_make_slot_options_task(state, grounded, unique_specs),
        facet_probe=_candidate_facet_probe(state, grounded, unique_specs[0]),
        slot_options_relaunch=_make_slot_options_relaunch(state, grounded, unique_specs[0]),
        place_options_task=_place_options_task,
        place_options_relaunch=(_relaunch_place_options_bound if _stream_place_options else None),
        observations_task=_obs_task,
        observations_relaunch=_obs_relaunch,
        member_population=grounded.cand_populations.get(_survivor_spec.spec_id),
        answered_roles=grounded.cand_answered_roles[_survivor_spec.spec_id],
        measure_partition=grounded.cand_measure_partitions.get(_survivor_spec.spec_id),
        measure_siblings=grounded.cand_measure_siblings.get(_survivor_spec.spec_id, ()),
        open_axis_facts=grounded.cand_open_axis_facts.get(_survivor_spec.spec_id, ()),
        role_axis_facts=tuple(grounded.cand_role_axis_facts[_survivor_id][1]),
    )


async def _assemble_candidate_set(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    near_neighbor_fired_spec_id: str | None,
    candidate_budget_truncated: bool,
) -> RegionResult:
    """Build a candidate response from two or more distinct survivors. When the elected shape has no
    data, add a warning describing its failure; assembly states its `closest_match`
    departure (`assemble._with_readings_substituted`).
    """
    _offer_place_warnings(state, grounded, unique_specs)
    if not state.elected_has_data:
        state.warnings.append(_elected_no_data_warning(state, unique_specs))
    # Plural answers disclose each unbound qualifier with `QUERY_QUALIFIER_DROPPED`. The client
    # chooses among specs and can weigh the omission.
    # The shared warning form supports deduplication.
    for _unbound in _unbound_query_qualifiers(
        state.query_qualifiers, unapplied=state.unapplied_qualifier_phrases
    ):
        state.depart(
            Warning(
                code=QUERY_QUALIFIER_DROPPED,
                severity="degraded",
                message=f'The requested qualifier "{_unbound.text}" was not applied.',
            ),
            measure_not_applied(
                asked=_unbound.text, property_dcid=qualifier_property(_unbound.named)
            ),
        )
    # State each candidate's unit lexicon version. Its identity already carries the binding,
    # and the outcome's warnings were merged earlier.
    if state.bound_unit_outcome is not None:
        unique_specs = [
            with_unit_lexicon_version(s, state.bound_unit_outcome) for s in unique_specs
        ]
    # Deduplication precedes option probes, so only surviving candidates incur those reads.
    unique_specs = list(
        await asyncio.gather(
            *(_attach_candidate_slot_options(state, grounded, s) for s in unique_specs)
        )
    )
    _candidate_specs = await _attach_candidate_facet_options(state, grounded, unique_specs)
    state.warnings.extend(
        omitted
        for spec in _candidate_specs
        if (omitted := grounded.cand_unit_omitted.get(spec.spec_id)) is not None
    )
    _surviving_candidate_ids = {spec.spec_id for spec in _candidate_specs}
    candidate_near_neighbor_spec_ids = (
        frozenset(
            spec_id
            for spec_id in (grounded.elected_spec_id, near_neighbor_fired_spec_id)
            if spec_id is not None and spec_id in _surviving_candidate_ids
        )
        if state.near_neighbor_fired_index is not None
        else frozenset()
    )
    _family = state.family_total_absent_departures(_candidate_specs)
    return RegionResult(
        variable_text=state.variable,
        status="candidates",
        specs=tuple(_candidate_specs),
        no_data_reason=None,
        warnings=(*state.warnings, *(warning for warning, _fact in _family)),
        departure_facts={**state.departure_facts, **raised_facts(_family)},
        timing_by_step=dict(state.timing),
        llm_usage=state.var_usage,
        candidate_scores=grounded.cand_scores,
        candidate_near_neighbor_spec_ids=candidate_near_neighbor_spec_ids,
        candidate_budget_truncated=candidate_budget_truncated,
        near_neighbor_fired=state.near_neighbor_fired_index is not None,
        slot_options_task=_make_slot_options_task(state, grounded, unique_specs),
        answered_roles=grounded.cand_answered_roles[_candidate_specs[0].spec_id],
    )


async def assemble_candidates(
    state: ResolveState,
    admission: _AlternatesAdmission,
    *,
    rate_cache: RateTableCache | None,
) -> RegionResult | None:
    """Ground admitted candidates and return a definite or multi-candidate result.
    Return `None` when no candidate survives beside an elected shape holding data only outside
    the years read, which the single-shape gates answer for the window.
    """
    candidates: list[_Candidate] = []
    if state.elected_has_data:
        # `elected_has_data` guarantees a materialised result. Ty treats attributes as optional.
        candidates.append(
            _Candidate(
                state.shape_draft,
                state.bindings,
                cast(Materialised, state.mat_result),
                state.place,
                state.walked,
            )
        )
    candidates.extend(admission.admitted)
    if state.elected_has_data:
        # The elected shape most often survives, so its related measures are read beside
        # the grounding and assembly of every candidate (`related_read_for`).
        start_elected_related_read(state, cast(Materialised, state.mat_result))
    if not candidates and (
        isinstance(state.mat_result, Materialised) and state.mat_result.available_at_place is False
    ):
        # Retain the elected shape's flagged-unavailable carrier so sparse-survivor handling returns
        # the matching no-data response.
        candidates.append(
            _Candidate(
                state.shape_draft, state.bindings, state.mat_result, state.place, state.walked
            )
        )

    # Grounding reads the graph synchronously for every scope member, so it runs off the event
    # loop, where the request deadline and other requests keep running.
    grounded = await asyncio.to_thread(ground_candidates, state, candidates)
    unique_specs = grounded.specs

    _filtered = _apply_bound_unit_filter(state, unique_specs, grounded)
    if isinstance(_filtered, RegionResult):
        return _filtered
    unique_specs = _filtered

    unique_specs = _fold_measure_sibling_candidates(
        unique_specs,
        {spec_id: shape for spec_id, (shape, _, _) in grounded.cand_role_axis_facts.items()},
        grounded.cand_measure_partitions,
    )

    # Ambiguity warnings describe competing candidates. With fewer than two survivors, remove the
    # warning before returning no-data or definite. A tie beside an unevaluated alternate stays.
    if (
        state.near_neighbor_fired_index is not None
        and len(unique_specs) < 2
        and state.near_neighbor_warning is not None
        and not state.tie_unresolved
    ):
        state.warnings.remove(state.near_neighbor_warning)
        state.near_neighbor_warning = None
        state.near_neighbor_fired_index = None

    if (
        not unique_specs
        and isinstance(state.mat_result, Materialised)
        and state.mat_result.window_gap is not None
    ):
        # The window answer is the elected shape's alone (`phase_answer._phase_ground_answer`).
        return None

    state.timing["answer"] = now_ms() - grounded.answer_t0
    await _emit_step(
        state.emit,
        state.variable_index,
        "answer",
        ms=state.timing["answer"],
    )

    unique_specs, near_neighbor_fired_spec_id, _served_twin_id = _apply_twin_diversion(
        state, unique_specs, grounded, grounded.near_neighbor_fired_spec_id
    )
    _sparse = _handle_sparse_survivors(state, unique_specs, grounded, _served_twin_id)
    if _sparse is not None:
        return _sparse

    # The discovered-scope warning speaks for the candidates the response keeps.
    restate_scope_warning(
        state,
        [
            grounded.cand_scopes[s.spec_id]
            for s in unique_specs
            if s.spec_id in grounded.cand_scopes
        ],
    )
    if len(unique_specs) == 1:
        return await _assemble_definite(state, unique_specs, grounded, rate_cache)
    return await _assemble_candidate_set(
        state,
        unique_specs,
        grounded,
        near_neighbor_fired_spec_id,
        admission.candidate_budget_truncated,
    )
