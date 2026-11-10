"""Collapse grounded alternates into a `RegionResult` after unit filtering, measure folding,
equivalent-reading diversion and no-data checks.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import cast

from materialise import GraphReader
from materialise.facets import nominal_currency
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.binders.election import mentioned_drafts
from search.engine.clients.graph import Facet
from search.engine.envelope import _emit_step, now_ms
from search.engine.reading_coverage import covers
from search.engine.region.candidate_admission import _AlternatesAdmission
from search.engine.region.candidate_grounding import (
    _GroundedCandidates,
    ground_candidates,
    restate_scope_warning,
)
from search.engine.region.candidate_measure_fold import (
    _facet_provenance_ids,
    _fold_measure_sibling_candidates,
)
from search.engine.region.candidate_options import (
    _attach_candidate_entity_options,
    _attach_candidate_facet_options,
    _attach_candidate_slot_options,
    _candidate_facet_probe,
    _elected_no_data_twin_substituted_warning,
    _elected_no_data_warning,
    _elected_sibling_warning,
    _make_slot_options_relaunch,
    _make_slot_options_task,
)

# _UNIT_CHECK_NOT_PROBED and _Candidate live in region/candidates.py,
# which imports assemble_candidates (this module) locally, inside
# resolve_candidates' own body, to avoid the cycle a module-level import
# would make here: by the time resolve_candidates first runs and takes
# that local import, region/candidates.py has already finished its own
# top-level execution, so this module-level import back into it is safe.
from search.engine.region.candidates import (
    _UNIT_CHECK_NOT_PROBED,
    _Candidate,
)
from search.engine.region.codes import (
    NO_MEMBER_HAS_DATA_AT_PLACE,
    ONLY_ALTERNATE_HAS_DATA,
    _elect_no_data_reason,
    _ObsProbeResult,
)
from search.engine.region.equivalent_series import shared_cluster
from search.engine.region.materialise_step import lone_breakdown_absent
from search.engine.region.observations.dispatch import (
    PreviewRead,
    PreviewReader,
    _attach_observations,
)
from search.engine.region.phase_election import judge_answer_coverage
from search.engine.region.phase_place_materialise import (
    note_denominator_unverified,
    refuse_unserved_quantity,
    serves_leg_quantity,
)
from search.engine.region.phase_rank_bind import judge_value_requirements
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
from search.engine.region.requirements import (
    meets_stray_values,
    settle_answer,
    settle_candidate_set,
    settle_flagged,
)
from search.engine.region.result import RegionResult
from search.engine.region.scope_decision import decide_guarded
from search.engine.region.state import (
    Requirement,
    ResolveState,
    ValueRule,
    raised_facts,
    unapplied_departure,
    with_requirement,
)
from search.engine.region.un_system import member_labels
from search.engine.region.value_labels import draft_value_labels, model_only_values
from search.engine.region.whole_totals import whole_total_warnings
from search.engine.retrieval.read_plan import ReadPlan, plan_coverage, serves_nothing
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.retrieval.standard import Narrow
from search.engine.shape import ShapeDraft
from search.engine.unit.bind import with_unit_lexicon_version
from search.engine.unit.conversion import served_prices
from search.engine.unit.conversion_catalogue import current_conversion_catalogue
from search.models import (
    CandidateBasis,
    GraphRef,
    IdentityUnitRequest,
    ResolvedQualifier,
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
        [(spec.spec_id, grounded[spec.spec_id].candidate.place) for spec in offered],
        departure_facts=state.departure_facts,
    )


def _apply_bound_unit_filter(
    state: ResolveState, unique_specs: list[Spec], grounded: _GroundedCandidates
) -> list[Spec] | RegionResult:
    """Filter candidates against a bound currency, price basis or base year after deduplication and
    before measure folding. This order keeps a compatible sibling available as its own candidate.
    An empty input proceeds to the elected shape's no-data reason.

    A candidate passes when grounding found its answer passes the single-shape gate
    (`bound_unit_gate.bound_unit_gate_reason`) and every row it serves converts
    (`_GroundedCandidate.serves_bound_unit`). A flagged-unavailable carrier passes unchecked,
    for the `NO_MEMBER_HAS_DATA_AT_PLACE` refusal. Return the surviving specs, or a terminal
    no-match result when none survives.
    """
    if state.bound_unit_outcome is None or not unique_specs:
        return unique_specs
    matched = [
        s
        for s in unique_specs
        if s.resolved.available_at_place is False or grounded[s.spec_id].serves_bound_unit
    ]
    if matched:
        return matched
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
        entry = grounded[s.spec_id]
        for exc in entry.unit_refusals.values():
            if isinstance(exc, UNIT_CONVERSION_REFUSALS):
                return entry.candidate, exc
    return None


def _unit_omitted_warning(grounded: _GroundedCandidates, spec_id: str) -> Warning | None:
    """The `currency_unmapped_omitted` warning of a kept Spec whose conversion omits the rows
    of entities no currency names (`UnitCoverage.omitted_warning`), stamped with its Spec.
    Every Spec kept under a bound unit passed the bound-unit filter, and with none bound its
    coverage is empty."""
    return grounded[spec_id].unit_coverage.omitted_warning(
        catalogue=current_conversion_catalogue(), spec_id=spec_id
    )


def _handle_sparse_survivors(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    served_twin: Spec | None,
) -> list[Spec] | RegionResult:
    """Settle the part under status rule 1 when the elected shape has no data, and handle
    zero survivors and a flagged-unavailable lone survivor. Return a terminal result, or the
    specs the part answers with.

    With no data on the elected shape, a registry twin that answers for it
    (`served_twin`) answers. Else the stand-ins re-election offered answer: one alone,
    several as candidates (`_Candidate.basis` `structural`). Else the part refuses with the
    elected shape's reason, and the first nominated reading with data rides on it as its
    alternate match.
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
        _settle_lone_survivor(state, grounded, unique_specs[0])
        return state.no_data(
            "no_observations",
            flagged_match=(unique_specs[0],),
            gate=NO_MEMBER_HAS_DATA_AT_PLACE,
            llm_usage=state.var_usage,
        )
    if state.elected_has_data:
        return unique_specs
    if served_twin is not None:
        # A registry twin reads the elected indicator itself (`_apply_twin_diversion`).
        state.warnings.append(
            _elected_no_data_twin_substituted_warning(state, served_twin.resolved.shape.label)
        )
        return [served_twin]
    # The elected reading of a family with no total is one of its breakdowns, so the others
    # with data answer in its place: one alone, several as candidates.
    breakdowns = [
        spec for spec in unique_specs if grounded[spec.spec_id].candidate.basis == "breakdown"
    ]
    if breakdowns:
        return breakdowns
    # Rule 1 installs only re-election's siblings as stand-ins (`_offered`). A registry twin
    # the diversion did not serve and a nominated rider stand in for nothing.
    stand_ins = [
        spec for spec in unique_specs if grounded[spec.spec_id].candidate.basis == "structural"
    ]
    if len(stand_ins) == 1:
        # One stand-in answers in the elected shape's place, as re-election installs one.
        # Definite assembly judges every requirement on its own shape and bindings
        # (`_assemble_definite`).
        state.warnings.append(
            _elected_sibling_warning(
                no_data=[state.shape_draft.label], elected=[stand_ins[0].resolved.shape.label]
            )
        )
        return stand_ins
    if stand_ins:
        return stand_ins
    # Nothing answers for the elected shape: the part refuses with its failure, and the
    # first nominated reading with data rides on it. `grounded_near_match` keeps
    # unresolved-entity and unresolved-variable reasons from contradicting that match.
    _elected_reason = _elect_no_data_reason(
        state.mat_result.reason if isinstance(state.mat_result, NoDataDraft) else None,
        fallback="no_observations",
        grounded_near_match=True,
    )
    _offer_place_warnings(state, grounded, unique_specs[:1])
    _settle_lone_survivor(state, grounded, unique_specs[0])
    return state.no_data(
        _elected_reason,
        alternate_match=unique_specs[0],
        gate=ONLY_ALTERNATE_HAS_DATA,
        llm_usage=state.var_usage,
    )


def _note_lone_breakdown(state: ResolveState, grounded: _GroundedCandidates, spec: Spec) -> None:
    """Record that a family with no total answers by one breakdown, narrower than asked
    (`standard.Narrow.lone_signature`), when `spec`, the lone survivor, is the elected
    breakdown or one offered beside it (`ResolveState.breakdown_rivals`). The survivor then
    carries the `family_total_absent` fact naming that breakdown (`lone_breakdown_absent`)
    unless its own fact names its single member."""
    survivor = grounded[spec.spec_id].candidate
    if spec.spec_id != grounded.elected_spec_id and survivor.basis != "breakdown":
        return
    if state.breakdown_rivals:
        state.breakdown = Narrow(
            frozenset(
                slot.property_dcid
                for slot in survivor.shape.slot_keys
                if slot.axis in ("what", "how") and slot.property_dcid is not None
            ),
            lone_signature=True,
        )
    fact = lone_breakdown_absent(
        state.breakdown,
        shape=survivor.shape,
        bindings=survivor.bindings,
        served=survivor.mat.sv_dcids,
    )
    if fact is not None:
        state.family_total_absent.setdefault(spec.spec_id, fact)


def _settle_lone_survivor(state: ResolveState, grounded: _GroundedCandidates, spec: Spec) -> None:
    """Judge the lone survivor a refusal carries on its own shape and bindings, the ones its
    trace records (`requirements.settle_flagged`)."""
    candidate = grounded[spec.spec_id].candidate
    settle_flagged(state, candidate.shape, candidate.bindings, uncovered=candidate.uncovered)


def _in_currency(facets: Sequence[Facet], currency: str | None) -> bool:
    """Whether no facet reports a unit other than `currency`, the one a twin cluster's
    members are equivalent in (`EquivalentCluster.currency`). Always true for a cluster
    naming none. A facet stating no unit passes, as nothing shows it reads another.
    Grounding keeps a twin's facets in that currency where it reports others too
    (`candidate_grounding._twin_unit_filters`)."""
    return currency is None or all(
        f.unit is None or nominal_currency(f.unit) == currency for f in facets
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
    elected = grounded[elected_spec_id]
    for twin in twins:
        twin_dcids = grounded[twin.spec_id].sv_dcids
        cluster = shared_cluster(elected.sv_dcids, twin_dcids)
        if (
            cluster is not None
            and cluster.preferred in twin_dcids
            and cluster.preferred not in elected.sv_dcids
            and _covers(grounded[twin.spec_id].plan, elected.plan)
        ):
            return twin
    return None


def _meeting_stray_values(
    state: ResolveState, unique_specs: list[Spec], grounded: _GroundedCandidates
) -> list[Spec]:
    """`unique_specs` without each answer that leaves unmet a value the bind drafted on a
    slot the elected reading lacks (`requirements.meets_stray_values`), the elected reading
    among them. When none meets them, the first answer alone, the elected one when it has
    data, which the definite path then refuses `qualifier_not_bound` on
    (`requirements.settle_answer`)."""
    meeting = [
        spec
        for spec in unique_specs
        if meets_stray_values(
            state.requirements,
            grounded[spec.spec_id].candidate.shape,
            grounded[spec.spec_id].candidate.bindings,
        )
    ]
    return meeting or unique_specs[:1]


def answers_for_the_elected_reading(basis: CandidateBasis | None) -> bool:
    """Whether a lone survivor offered on `basis` answers when the elected reading, which has
    data, leaves for a stray value (`_meeting_stray_values`): when it reads the elected
    measure in the asked direction. A `structural` reading differs from it only on an axis
    the question leaves open, and a `twin` is a registry encoding of the elected indicator
    itself. A `nominated` reading reads another measure, a `direction` reading the other
    direction, and a survivor offered on no basis was offered on no reading of the elected
    one, so none of these answers by elimination."""
    return basis in ("structural", "twin")


def _apply_twin_diversion(
    state: ResolveState, unique_specs: list[Spec], grounded: _GroundedCandidates
) -> tuple[list[Spec], Spec | None]:
    """Move registry-confirmed encodings of the elected indicator into `equivalent_readings`.
    Return the remaining specs and the twin, among them, that answers for an elected shape
    with no data, if one does.

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
        _elected_sv_dcids = grounded[_elected_spec.spec_id].sv_dcids
        _elected_facets = grounded[_elected_spec.spec_id].confirmed_facets
    elif not state.elected_has_data and isinstance(state.mat_result, Materialised | NoDataDraft):
        _elected_sv_dcids = state.mat_result.sv_dcids or []
        _elected_facets = ()
    else:
        return unique_specs, None
    _elected_provenance_ids = _facet_provenance_ids(list(_elected_facets))
    _twins: list[Spec] = []
    _preferred_members: set[str] = set()
    _dropped_ids: set[str] = set()
    for s in unique_specs:
        if s is _elected_spec or s.resolved.available_at_place is False:
            continue
        _candidate_sv_dcids = grounded[s.spec_id].sv_dcids
        cluster = shared_cluster(_elected_sv_dcids, _candidate_sv_dcids)
        if cluster is None:
            continue
        _candidate_facets = grounded[s.spec_id].confirmed_facets
        if not all(
            _in_currency(facets, cluster.currency)
            for facets in (_elected_facets, _candidate_facets)
        ):
            _dropped_ids.add(s.spec_id)
            continue
        # With data on both sides, require known, disjoint provenance sets. This blocks
        # same-source twins split across tables. A stale registry entry with disjoint
        # provenance can pass. An elected shape with no data has no reading to duplicate.
        _candidate_provenance_ids = _facet_provenance_ids(list(_candidate_facets))
        if _elected_spec is None or (
            _elected_provenance_ids
            and _candidate_provenance_ids
            and _elected_provenance_ids.isdisjoint(_candidate_provenance_ids)
        ):
            _twins.append(s)
            if cluster.preferred in _candidate_sv_dcids:
                _preferred_members.add(s.spec_id)
    if not _twins and not _dropped_ids:
        return unique_specs, None
    # Preserve the elected spec's real bind result; only non-elected survivors can become twins.
    _twin_ids = {t.spec_id for t in _twins}
    # The answer takes the place of the first reading: the elected spec, or the first twin
    # standing for an elected shape with no data.
    _served_twin: Spec | None = None
    _listed: list[Spec] = []
    if _elected_spec is not None or _twins:
        if _elected_spec is not None:
            _readings = [_elected_spec, *_twins]
            _primary = _preferred_twin(grounded, _elected_spec.spec_id, _twins) or _elected_spec
        else:
            _readings = _twins
            _primary = next((t for t in _twins if t.spec_id in _preferred_members), _twins[0])
        _listed = [r for r in _readings if r is not _primary]
        if _listed:
            _primary = _primary.model_copy(
                update={
                    "resolved": _primary.resolved.model_copy(
                        update={"equivalent_readings": _listed}
                    )
                }
            )
        if _elected_spec is None:
            _served_twin = _primary
        _anchor_id = _readings[0].spec_id
        unique_specs = [
            _primary if s.spec_id == _anchor_id else s
            for s in unique_specs
            if s.spec_id == _anchor_id or s.spec_id not in _twin_ids | _dropped_ids
        ]
    else:
        # Every twin of an elected shape with no data read another currency.
        unique_specs = [s for s in unique_specs if s.spec_id not in _dropped_ids]
    return unique_specs, _served_twin


async def _assemble_definite(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    rate_cache: RateTableCache | None,
) -> RegionResult:
    """Build the definite response for one surviving candidate. A requirement it leaves
    unmet returns no-data. A no-data elected shape reaches here only through measure-sibling
    or registry-twin substitution.
    A survivor from an alternate answers over its own place context, so its place warnings
    become the answer's own (`install_place_warnings`). The survivor is judged on every
    requirement on its own shape, bindings and served unit request
    (`requirements.settle_answer`), since a stand-in or an alternate binds its values from
    its own representative and can hold a code the elected shape dropped, bind another code
    on the same property, or, as a sibling every member of which holds Female, apply
    "female". Its refusal names the terms
    its flagged match's trace records (`requirements.answer_terms`): the elected reading's
    unbound query terms other than its coverage records, its own coverage records
    (`_Candidate.uncovered`), and each code and value requirement its own answer leaves
    unmet.
    """
    _entry = grounded[unique_specs[0].spec_id]
    _survivor = _entry.candidate
    if not _entry.is_elected:
        install_place_warnings(
            state.warnings, _survivor.place, departure_facts=state.departure_facts
        )
    _survivor_unbound_qualifiers = settle_answer(
        state,
        _survivor.shape,
        _survivor.bindings,
        served=served_prices(unique_specs[0]),
        uncovered=_survivor.uncovered,
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
    note_denominator_unverified(state, _survivor.shape, spec_id=_survivor_spec.spec_id)
    if (omitted := _unit_omitted_warning(grounded, _survivor_spec.spec_id)) is not None:
        state.warnings.append(omitted)
    _preview = PreviewRead(
        sv_dcids=tuple(_entry.sv_dcids),
        obs_entities=_entry.obs_entities,
        plan=_entry.plan,
        # A ref-form unit filters literally; a request converts.
        facet_filter=None if state.unit_request is not None else state.unit_facet_filter,
        bulk_places=_entry.bulk_scope.places if _entry.bulk_scope is not None else None,
        member_facts=_survivor.mat.member_facts,
        unit_request=state.unit_request,
        elected=_entry.elected,
    )
    _reader = PreviewReader(
        read=_preview,
        graph=state.graph,
        sdmx=state.sdmx,
        materialise_graph=state.materialise_graph,
        materialise_sdmx=state.materialise_sdmx,
        rate_cache=rate_cache,
        deadline=state.deadline,
        coverage=state.coverage,
    )
    # The scope is decided from the survivor's Spec beside the reads below and before its
    # preview launches, so an answer restated as a composite reads no preview of the one
    # table it resolved. `pipeline._with_scope_choices` states the decision.
    _decision_task = asyncio.create_task(
        asyncio.to_thread(
            decide_guarded,
            _survivor_spec,
            selection=state.scope_selection,
            operations=state.operations,
            coverage=state.coverage,
        )
    )
    try:
        # The related reads run beside the streamed preview and the options below.
        _related_read = related_read_for(
            state,
            stat_vars=_entry.sv_dcids,
            measure_partition=_entry.measure_partition,
            places=_survivor.place.roles.roles,
            arcs=_survivor.shape.sv_arc_facts,
        )
        if state.bound_unit_outcome is not None:
            # State the unit lexicon version. A streamed answer also states its preview's
            # facets and warning with its immediate response. A flagged-unavailable carrier
            # holds nothing to read and states `_UNIT_CHECK_NOT_PROBED`, and a composite
            # reads its components' preview once restated.
            _survivor_spec = with_unit_lexicon_version(_survivor_spec, state.bound_unit_outcome)
            if state.stream_observations:
                # An inline preview states its own warning, below.
                _streamed = (
                    _UNIT_CHECK_NOT_PROBED
                    if _survivor_spec.resolved.available_at_place is False
                    or (await _decision_task).composite
                    else await asyncio.to_thread(_reader.observations)
                )
                _survivor_spec = _attach_observations(
                    _survivor_spec, _streamed, state.warnings, include_points=False
                )
        # Keep this timing key identical to the single-shape path.
        _options_t0 = now_ms()
        _survivor_spec = await _attach_candidate_slot_options(state, grounded, _survivor_spec)
        (_survivor_spec,) = await _attach_candidate_facet_options(state, grounded, [_survivor_spec])
        _survivor_spec, _entity_options_raw = await _attach_candidate_entity_options(
            state, grounded, _survivor_spec
        )
        state.timing["options"] = now_ms() - _options_t0
        _family = state.family_total_absent_departures((_survivor_spec,))
        # Place-option tasks run only when `materialise_graph` is available.
        _place_options_materialise_graph = state.materialise_graph
        # Each launch reads the identity of the Spec it is given, a merged one included.
        _relaunch_place_options_bound = functools.partial(
            _relaunch_place_options,
            state=state,
            place_options_materialise_graph=cast(GraphReader, _place_options_materialise_graph),
        )
        _stream_place_options = (
            state.stream_slot_options
            and _place_options_materialise_graph is not None
            and state.place_options_pool is not None
        )
        _place_options_task = (
            _relaunch_place_options_bound(_survivor_spec) if _stream_place_options else None
        )
        _survivor_spec = await with_related(_survivor_spec, _related_read)
        # No await follows the observation task's launch. Passing it live to `RegionResult`
        # keeps it reachable if cancellation follows.
        _obs_task: asyncio.Task[_ObsProbeResult] | None = None
        _obs_relaunch: Callable[[str], asyncio.Task[_ObsProbeResult]] | None = None
        # A composite's preview reads each component once the answer is restated
        # (`pipeline._with_composite_preview`).
        state.scope_decision = await _decision_task
        composite = state.scope_decision.composite
        if not composite and state.stream_observations:
            _obs_relaunch = _reader.relaunch(member_labels((_survivor_spec,)))
            _obs_task = _obs_relaunch(_survivor_spec.spec_id)
        elif not composite and state.include_observations:
            _survivor_spec = _attach_observations(
                _survivor_spec, await asyncio.to_thread(_reader.observations), state.warnings
            )
    finally:
        # Cancel and await the decision so cancellation cannot detach it.
        if not _decision_task.done():
            _decision_task.cancel()
            await asyncio.gather(_decision_task, return_exceptions=True)
    return RegionResult(
        variable_text=state.variable,
        status="definite",
        specs=(_survivor_spec,),
        no_data_reason=None,
        warnings=(
            *state.warnings,
            *(warning for warning, _fact in _family),
            *whole_total_warnings(_survivor_spec, coverage=state.coverage.index),
        ),
        departure_facts={**state.departure_facts, **raised_facts(_family)},
        timing_by_step=dict(state.timing),
        llm_usage=state.var_usage,
        slot_options_task=_make_slot_options_task(
            state,
            grounded,
            unique_specs,
            entity_options_raw=(
                {_entry.spec_id: _entity_options_raw} if _entity_options_raw is not None else {}
            ),
        ),
        facet_probe=_candidate_facet_probe(state, grounded, unique_specs[0]),
        slot_options_relaunch=_make_slot_options_relaunch(
            state, grounded, unique_specs[0], entity_options_raw=_entity_options_raw
        ),
        place_options_task=_place_options_task,
        place_options_relaunch=(_relaunch_place_options_bound if _stream_place_options else None),
        observations_task=_obs_task,
        observations_relaunch=_obs_relaunch,
        preview_read=_preview,
        member_population=_entry.population,
        answered_roles=_entry.answered_roles,
        measure_partition=_entry.measure_partition,
        measure_siblings=_entry.measure_siblings,
        open_axis_facts=_entry.open_axis_facts,
        role_axis_facts=_entry.role_axis_facts,
        # The members and plan its conversion offer reads, which a same-shape merge offers
        # conversions over (`RegionResult.served_members`).
        served_members=_entry.served_members,
        served_plans=_entry.served_plans,
    )


async def _record_unheld_values(
    state: ResolveState, answers: Sequence[tuple[ShapeDraft, Sequence[SlotBindingDraft]]]
) -> None:
    """Record in the part's ledger each value the request's words name, drafted on the
    elected shape (`election.mentioned_drafts`), that one of `answers`, the candidate set's
    shapes with their own bindings, leaves unmet: one requirement per (property, value),
    however many answers leave it unmet, judged as a value the request names (`ValueRule`,
    origin ``unheld``) by the rule that settles it on each answer
    (`phase_rank_bind.judge_value_requirements`), a value drafted in a set judged with
    that set (`ValueRule.drafted_set`), and named by the value's label
    (`value_labels.draft_value_labels`). An answer binding the value on its property meets
    it, so a set whose every answer binds each drafted value reads no labels. A value no
    recorded term supports is the bind model's reading of the measure the phrase names,
    not a value the user typed (`value_labels.model_only_values`), as LiveBirth is of
    "birth rate", and records nothing. The requirement refuses nothing, so the set offers
    each candidate and departs the value once unless every candidate applies it
    (`requirements.settle_candidate_set`)."""
    drafted_sets: dict[tuple[str, str], frozenset[tuple[str, str]]] = {}
    for draft in mentioned_drafts(state.bindings):
        if (prop := draft.property_dcid) is None:
            continue
        members = frozenset((prop, value) for value in draft.value_dcids)
        for value in draft.value_dcids:
            drafted_sets.setdefault((prop, value), members if draft.kind == "set" else frozenset())
    drafted = list(drafted_sets)
    bound = [
        {
            (binding.property_dcid, value)
            for binding in bindings
            if binding.kind in ("value", "set")
            for value in binding.value_dcids
        }
        for _shape, bindings in answers
    ]
    pending = [pair for pair in drafted if not all(pair in held for held in bound)]
    if not pending:
        return
    labels = await draft_value_labels(state, [value for _prop, value in pending])
    requirements: list[Requirement] = []
    for prop in dict.fromkeys(prop for prop, _value in pending):
        values = [value for on, value in pending if on == prop]
        model = model_only_values(state, prop, values, labels)
        requirements += [
            _unheld_requirement(prop, value, labels[value], drafted_sets[prop, value])
            for value in values
            if value not in model
        ]
    unmet = {
        id(requirement)
        for shape, bindings in answers
        for requirement in judge_value_requirements(requirements, shape, bindings, state)
    }
    for requirement in requirements:
        if id(requirement) in unmet:
            state.requirements = with_requirement(state.requirements, requirement)


def _unheld_requirement(
    prop: str, value: str, label: str, drafted_set: frozenset[tuple[str, str]]
) -> Requirement:
    """The requirement that an answer apply `value`, named `label`, on `prop`, a value the
    request's words name that a candidate may leave unheld, drafted in `drafted_set` when
    the bind drafted it in a set; it refuses nothing."""
    return Requirement(
        term=ResolvedQualifier(
            text=label,
            named=[GraphRef(ref=value, label=label)],
            slot=None,
            disposition="unbound",
            evidence="model",
        ),
        origin="unheld",
        rule=ValueRule(values=frozenset({(prop, value)}), drafted_set=drafted_set),
        departure=unapplied_departure(label, asked=label, property_dcid=prop),
        refuses=False,
    )


async def _assemble_candidate_set(
    state: ResolveState,
    unique_specs: list[Spec],
    grounded: _GroundedCandidates,
    candidate_budget_truncated: bool,
) -> RegionResult:
    """Build a candidate response from two or more distinct survivors, each but the elected
    one stating why the status rule offers it (`ResolvedSpec.basis`). When the elected shape
    has no data, add a warning describing its failure; assembly states its `closest_match`
    departure (`assemble._with_readings_substituted`). Each candidate the search could not
    check against the denominator the question names warns under its own spec_id
    (`note_denominator_unverified`), as each one listing a side whose total its dataset
    publishes does (`whole_totals.whole_total_warnings`).
    """
    unique_specs = [
        spec.model_copy(
            update={
                "resolved": spec.resolved.model_copy(
                    update={"basis": grounded[spec.spec_id].candidate.basis}
                )
            }
        )
        for spec in unique_specs
    ]
    _offer_place_warnings(state, grounded, unique_specs)
    if not state.elected_has_data:
        state.warnings.append(_elected_no_data_warning(state, unique_specs))
    await _record_unheld_values(
        state,
        [
            (
                grounded.cand_candidates[spec.spec_id].shape,
                grounded.cand_candidates[spec.spec_id].bindings,
            )
            for spec in unique_specs
        ],
    )
    # Each survivor is judged on its own shape and bindings, the unit request its Spec is
    # served under and its own coverage records, and a requirement departs unless every
    # survivor meets it (`requirements.settle_candidate_set`).
    settle_candidate_set(
        state,
        [
            (grounded[spec.spec_id].candidate.shape, grounded[spec.spec_id].candidate.bindings)
            for spec in unique_specs
        ],
        served=[served_prices(spec) for spec in unique_specs],
        uncovered=[grounded[spec.spec_id].candidate.uncovered for spec in unique_specs],
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
    for spec in _candidate_specs:
        note_denominator_unverified(
            state, grounded[spec.spec_id].candidate.shape, spec_id=spec.spec_id
        )
    state.warnings.extend(
        omitted
        for spec in _candidate_specs
        if (omitted := _unit_omitted_warning(grounded, spec.spec_id)) is not None
    )
    _family = state.family_total_absent_departures(_candidate_specs)
    return RegionResult(
        variable_text=state.variable,
        status="candidates",
        specs=tuple(_candidate_specs),
        no_data_reason=None,
        warnings=(
            *state.warnings,
            *(warning for warning, _fact in _family),
            *(
                warning
                for spec in _candidate_specs
                for warning in whole_total_warnings(spec, coverage=state.coverage.index)
            ),
        ),
        departure_facts={**state.departure_facts, **raised_facts(_family)},
        timing_by_step=dict(state.timing),
        llm_usage=state.var_usage,
        candidate_budget_truncated=candidate_budget_truncated,
        slot_options_task=_make_slot_options_task(
            state, grounded, unique_specs, entity_options_raw={}
        ),
        answered_roles=grounded[_candidate_specs[0].spec_id].answered_roles,
        candidate_answered_roles=tuple(
            grounded[spec.spec_id].answered_roles for spec in _candidate_specs
        ),
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
    # Each candidate beside the elected answer is judged on the members its own read
    # served (`judge_answer_coverage`), before grounding records each one's trace.
    uncovered = await judge_answer_coverage(
        state, [(cand.shape, cand.bindings, cand.mat) for cand in admission.admitted]
    )
    candidates.extend(
        dataclasses.replace(cand, uncovered=own)
        for cand, own in zip(admission.admitted, uncovered, strict=True)
    )
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
    # The stand-in that re-election installed is the elected candidate here.
    for spec in unique_specs:
        if spec.spec_id == grounded.elected_spec_id:
            state.state_reelected_reading(spec)

    _filtered = _apply_bound_unit_filter(state, unique_specs, grounded)
    if isinstance(_filtered, RegionResult):
        return _filtered
    unique_specs = _filtered

    # The stray filter runs before the fold, so each reading is judged on its own shape and
    # a sibling that applies the value is not folded into one that leaves.
    _met = _meeting_stray_values(state, unique_specs, grounded)
    # The elected Spec, when it left for a stray value another reading meets.
    _elected_left = next(
        (s for s in unique_specs if s.spec_id == grounded.elected_spec_id and s not in _met),
        None,
    )
    unique_specs = _met

    unique_specs = _fold_measure_sibling_candidates(
        unique_specs,
        {spec_id: entry.candidate.shape for spec_id, entry in grounded.entries.items()},
        {spec_id: entry.measure_partition for spec_id, entry in grounded.entries.items()},
    )

    if (
        not unique_specs
        and isinstance(state.mat_result, Materialised)
        and state.mat_result.window_gap is not None
    ):
        # The window answer is the elected shape's alone (`phase_answer._ground_elected_answer`).
        return None

    state.timing["answer"] = now_ms() - grounded.answer_t0
    await _emit_step(
        state.emit,
        state.variable_index,
        "answer",
        ms=state.timing["answer"],
    )

    if state.quantity_refusal is not None:
        # The elected shape does not serve the quantity, so the part offers only the readings
        # that do, and refuses when none survived. The filter runs before the twin diversion,
        # so a twin that misses the quantity, as the elected shape's registry twins do, is
        # neither served nor listed as an equivalent reading.
        unique_specs = [
            s
            for s in unique_specs
            if serves_leg_quantity(state, grounded[s.spec_id].candidate.shape)
        ]
        if not unique_specs:
            return await refuse_unserved_quantity(state)
    unique_specs, _served_twin = _apply_twin_diversion(state, unique_specs, grounded)
    _settled = _handle_sparse_survivors(state, unique_specs, grounded, _served_twin)
    if isinstance(_settled, RegionResult):
        return _settled
    unique_specs = _settled
    if (
        len(unique_specs) == 1
        and _elected_left is not None
        and not answers_for_the_elected_reading(grounded[unique_specs[0].spec_id].candidate.basis)
    ):
        # A lone survivor reading another measure or direction does not answer by
        # elimination. The elected reading stands alone, and the definite path refuses
        # `qualifier_not_bound` on the stray value it leaves unmet, with it as the flagged
        # match (`requirements.settle_answer`).
        unique_specs = [_elected_left]

    # The discovered-scope warning speaks for the candidates the response keeps.
    restate_scope_warning(
        state,
        [
            (grounded[s.spec_id].candidate.mat, grounded[s.spec_id].scope_members)
            for s in unique_specs
        ],
    )
    if len(unique_specs) == 1:
        _note_lone_breakdown(state, grounded, unique_specs[0])
        # The lone reading, a twin or a nominated rival the quantity or unit filter left
        # alone included, discloses each typed value its shape cannot hold, as a candidate
        # set does. A reading whose bindings bind every typed value, as the elected shape
        # and an installed stand-in bind their own drafts, records nothing and reads no
        # labels.
        lone = grounded.cand_candidates[unique_specs[0].spec_id]
        await _record_unheld_values(state, [(lone.shape, lone.bindings)])
        return await _assemble_definite(state, unique_specs, grounded, rate_cache)
    return await _assemble_candidate_set(
        state,
        unique_specs,
        grounded,
        admission.candidate_budget_truncated or state.breakdown_truncated,
    )
