"""Mint the plan and set the part's status over the evidence materialisation read
(`_phase_outcome`).

Candidates come first, where the status rule offers readings beside the elected shape
(`candidates.resolve_candidates`). Else the elected shape's own read answers: no data or
unsupported (`_no_data_answer`), or grounding (`_ground_elected_answer`) and the definite
Spec (`_assemble_definite_answer`). A bound unit is checked before grounding. Assembly
builds the Spec, applies no-data carve-outs, and attaches buffered or streamed options
and observations.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
from collections.abc import Callable, Mapping
from typing import cast

from materialise import GraphReader
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from search.engine.envelope import _emit_step, now_ms
from search.engine.errors import EngineInfraError
from search.engine.extract import DateRequest
from search.engine.region.arcs import (
    inapplicable_slots_from_member_facts,
    unresolved_slots_from_member_facts,
)
from search.engine.region.bound_unit_gate import bound_unit_gate_reason
from search.engine.region.candidates import resolve_candidates
from search.engine.region.codes import (
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
    _ObsProbeResult,
)
from search.engine.region.convertibility import (
    conversion_period_departure,
    convertible_answer,
    decide_part_unit,
    deflator_omission_departure,
    elected_unit_refs,
    option_pins,
    over_conversion_period,
    survivor_declared_roles,
    unit_gate,
    unit_request_period,
)
from search.engine.region.facet_probe import FacetProbe, facet_probe
from search.engine.region.facets import (
    FacetAdmission,
    FilteredAnswer,
    answer_facets,
    cluster_currency_binding,
    cluster_currency_filter,
    filter_answer,
    narrow_facets_by_sv_to_filter,
)
from search.engine.region.grounding import (
    _build_entity,
    _ground_answer,
    _place_confidence,
    build_identity_entities,
    requested_window_from_date_request,
    restate_role_scope_warnings,
)
from search.engine.region.materialise_step import family_total_absent_departure
from search.engine.region.observations.bulk_scope import BulkScope, bulk_scope_for
from search.engine.region.observations.dispatch import (
    _attach_observations,
    _resolve_spec_observations,
)
from search.engine.region.observations.entities import (
    _flat_obs_entities,
    _observation_entities,
    coverage_probe_role_entities,
    open_role_axis,
    scope_axis_dcids,
)
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.option_coverage import (
    CountedSlotOptions,
    RoleServedFilter,
    _build_and_enrich_entity_options,
    _maybe_build_facet_options,
    _maybe_enrich_slot_options,
    counted_slot_options,
    measure_served_counter,
    role_axis_facts_for_materialised,
    role_served_filter,
)
from search.engine.region.phase_place_materialise import (
    _nearest_real_for,
    note_denominator_unverified,
    refuse_unserved_quantity,
)
from search.engine.region.phase_shared import (
    UNIT_CONVERSION_REFUSALS,
    _unsupported,
    unit_conversion_refusal,
    withdraw_typed_unit,
)
from search.engine.region.place_roles import request_named_groups
from search.engine.region.related_read import start_elected_related_read, with_related
from search.engine.region.relaunch import _relaunch_place_options, _relaunch_slot_options
from search.engine.region.requirements import answer_terms, refusal_terms, settle_answer
from search.engine.region.result import FacetAxisFilter, RegionResult, answered_roles
from search.engine.region.scope import (
    build_scope,
    fetch_scope_members,
    scope_axis,
    stated_scope_members,
)
from search.engine.region.state import MeasureFamily, ResolveState, raised_facts
from search.engine.region.un_system import member_labels
from search.engine.region.whole_totals import whole_total_warnings
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    OpenAxisFact,
    OptionServedFact,
    apply_declaration_version,
    apply_slot_expansions,
    bound_values_from_bindings,
    open_constraint_axes,
    slot_expansions,
)
from search.engine.retrieval.elected_dataset import elected_datasets
from search.engine.retrieval.read_plan import ReadPlan, plan_probes
from search.engine.retrieval.retrieve import Materialised, NoDataDraft, UnsupportedDraft
from search.engine.spec_build import (
    RoleAxisFactInput,
    bind_when_slot,
    build_shape_model,
    build_spec,
    build_stat_vars,
)
from search.engine.unit.bind import with_unit_lexicon_version
from search.engine.unit.conversion import (
    ConversionPeriod,
    counted_conversion,
    pins_facets,
    with_unit_conversion,
)
from search.engine.unit.conversion_catalogue import current_conversion_catalogue
from search.engine.unit.refs import unit_currency
from search.models import (
    Confidence,
    Entity,
    EntityOptions,
    FacetOptions,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    NearestRealSpec,
    Slot,
    SlotOptions,
    Spec,
    SpecObservations,
    Warning,
)

logger = logging.getLogger(__name__)


def _answer_facet_probe(
    state: ResolveState, mat_result: Materialised, probe_role_entities: dict[str, list[str]]
) -> FacetProbe:
    """What this answer's facet options are judged over. Its one possible pin is the bound
    unit request the spec's identity carries, admitted per StatVar by the elected units."""
    bindings = (
        {"unit": IdentityFacetValueRequest(request=cast(IdentityUnitRequest, state.unit_request))}
        if state.bound_unit_outcome is not None
        else None
    )
    return facet_probe(
        mat_result.sv_dcids,
        probe_role_entities,
        years=mat_result.preview_years,
        pins=option_pins(bindings, state.unit_elected_selections),
        datasets=elected_datasets(mat_result.member_facts),
    )


async def _decide_bound_unit_convertibility(
    state: ResolveState, mat_result: Materialised
) -> RegionResult | None:
    """Check a request-form unit once per StatVar before grounding or observation fetch.

    Map ``RecipeNotDeclaredError`` to ``recipe_not_declared``, ``RateCoverageGapError`` to
    ``rate_coverage_gap``, ``NonCurrencyUnitError`` to ``facet_binding_unsupported``, and
    missing-facet errors to ``facet_not_available``. A conversion refusal of the raw-text
    door's typed unit phrase withdraws the request instead (`withdraw_typed_unit`), and the
    leg answers unconverted. Missing clients raise
    ``EngineInfraError``. Ref-form and unbound units are no-ops. Success stores selections
    on ``state``. Recipe errors retain their cause.
    """
    if state.unit_request is None:
        return None
    if state.materialise_graph is None or state.materialise_sdmx is None:
        raise EngineInfraError(
            "materialise graph/sdmx client not initialised; cannot decide this "
            "bound unit request's own convertibility"
        )
    # Reuse discovered members so unit decisions and observation fetches probe the same scope.
    _decision_destination_dcids, _decision_origin_dcids = scope_axis_dcids(
        mat_result, state.place.probe_places
    )
    _decision_role_bindings, _ = _observation_entities(
        state.shape_draft,
        destination_dcids=_decision_destination_dcids,
        destination_dcid=state.place.roles.obs_destination_dcid,
        origin_dcids=_decision_origin_dcids or state.place.roles.origin_dcids,
    )
    # With no probe entity, the later bound-unit gate returns ``facet_not_available``.
    if not _decision_role_bindings:
        return None
    _declared_roles = survivor_declared_roles(mat_result)
    try:
        _decided_selections = await asyncio.to_thread(
            decide_part_unit,
            mat_result.sv_dcids,
            unit_request=state.unit_request,
            role_bindings=_decision_role_bindings,
            declared_roles=_declared_roles,
            # The plan's years; unit-filtered coverage is computed afterward.
            years=mat_result.preview_years,
            graph=state.materialise_graph,
            sdmx=state.materialise_sdmx,
            coverage=state.coverage.index,
            # The decision ranks the candidates the answer's plan elects from.
            probes=plan_probes(mat_result.read_plan),
            datasets=elected_datasets(mat_result.member_facts),
            deadline=state.deadline,
        )
        # Share this election with coverage narrowing and observation fetches. Repeating it
        # would repeat the graph/SDMX read.
        state.unit_elected_selections = _decided_selections
    except UNIT_CONVERSION_REFUSALS as exc:
        if withdraw_typed_unit(state, exc):
            return None
        return unit_conversion_refusal(state, exc, unit_request=state.unit_request)
    except (FacetNotAvailableError, DefaultFacetUndefinedError):
        # No eligible facet exists on at least one axis.
        return state.no_data("facet_not_available", llm_usage=state.var_usage)
    return None


def _over_conversion_period(
    state: ResolveState,
    mat_result: Materialised,
    filtered: FilteredAnswer,
    admission: FacetAdmission,
) -> tuple[Materialised, FilteredAnswer]:
    """The answer a unit request shows when its conversion tables cover only part of the
    years `filtered` serves (`convertibility.unit_request_period`): the request reads the
    period's window from here on, the plan is read over it and filtered again, and the
    answer departs `year_unavailable` naming the table. Otherwise both as they are. Either
    way an answer whose deflators leave out some entities' rows departs `year_unavailable`
    naming them (`convertibility.deflator_omission_departure`)."""
    if state.unit_request is None or state.unit_elected_selections is None:
        return mat_result, filtered
    catalogue = current_conversion_catalogue()
    period = unit_request_period(
        filtered.plan,
        state.unit_request,
        state.unit_elected_selections,
        declared_roles=survivor_declared_roles(mat_result),
        catalogue=catalogue,
    )
    if period is not None:
        mat_result, filtered = _narrowed_to_period(state, mat_result, admission, period)
    omission = deflator_omission_departure(
        filtered.plan,
        state.unit_request,
        coverage_index=state.coverage.index,
        catalogue=catalogue,
    )
    if omission is not None:
        state.depart(*omission)
    return mat_result, filtered


def _narrowed_to_period(
    state: ResolveState,
    mat_result: Materialised,
    admission: FacetAdmission,
    period: ConversionPeriod,
) -> tuple[Materialised, FilteredAnswer]:
    """The answer read over `period`'s window, departing `year_unavailable` naming the
    table that leaves the other years out (`_over_conversion_period`)."""
    requested = state.date_request.window if state.date_request is not None else None
    state.date_request = DateRequest(window=period.window, latest=False)
    mat_result = over_conversion_period(mat_result, period)
    state.mat_result = mat_result
    filtered = filter_answer(
        mat_result,
        admission,
        date_request=state.date_request,
        coverage_index=state.coverage.index,
    )
    state.depart(*conversion_period_departure(period, requested=requested, plan=filtered.plan))
    return mat_result, filtered


def _answer_unit_filter(
    state: ResolveState, mat_result: Materialised
) -> dict[str, FacetAxisFilter] | None:
    """The literal unit filter, else the answer's twin cluster currency
    (`cluster_currency_filter`) when the request binds no unit."""
    if state.unit_facet_filter is not None or state.unit_request is not None:
        return state.unit_facet_filter
    return cluster_currency_filter(mat_result)


def _install_filtered_answer(
    state: ResolveState, mat_result: Materialised, filtered: FilteredAnswer
) -> None:
    """Record the answer's coverage, headline and read plan, and restate the discovered-scope
    warning to count the members that plan serves, as the scope list and coverage do."""
    state.elected_coverage = filtered.coverage
    state.elected_headline = filtered.headline
    state.elected_plan = filtered.plan
    if state.opposite_labels is not None:
        restate_role_scope_warnings(
            state.warnings,
            mat_result,
            members=stated_scope_members(mat_result, filtered.plan),
            boundary_roles=state.place.roles.roles,
            label_of=state.opposite_labels,
            unnamed_wholes=state.place.roles.unnamed_wholes,
        )


def _gate_carve_out(state: ResolveState, mat_result: Materialised) -> bool:
    """Whether a carve-out answers before the bound-unit gate is read: no member with data at
    the place, no place and no data, or a requirement the answer leaves unmet
    (`requirements.refusal_terms`, `_assemble_carve_outs`)."""
    return (
        mat_result.available_at_place is False
        or (state.destination_dcid is None and not mat_result.has_data)
        or bool(refusal_terms(state, state.shape_draft, state.bindings))
    )


def _answer_observation_entities(
    state: ResolveState, mat_result: Materialised
) -> tuple[dict[str, list[str]], Warning | None]:
    """The entities the answer's observations bind by role, discovered members on their
    declared flow axis, with the warning their derivation raises (`_observation_entities`)."""
    roles = state.place.roles
    scope_destination_dcids, scope_origin_dcids = scope_axis_dcids(
        mat_result, state.place.probe_places
    )
    return _observation_entities(
        state.shape_draft,
        destination_dcids=scope_destination_dcids,
        destination_dcid=roles.obs_destination_dcid,
        origin_dcids=scope_origin_dcids or roles.origin_dcids,
    )


def _runs_unit_conversion(
    state: ResolveState, mat_result: Materialised, plan: ReadPlan | None
) -> bool:
    """Whether the bound-unit gate will read the answer's conversion: a bound unit, no
    carve-out ahead of the gate (`_gate_carve_out`), a StatVar to convert, and a plan that
    serves every StatVar under its elected unit (`convertibility.unit_gate`). The plan the
    gate read is recorded for the final gate (`ResolveState.unit_gate_plan`)."""
    if state.bound_unit_outcome is None or not mat_result.sv_dcids:
        return False
    if _gate_carve_out(state, mat_result):
        return False
    obs_entities, _warning = _answer_observation_entities(state, mat_result)
    gate = unit_gate(
        mat_result,
        plan,
        state.unit_elected_selections,
        obs_entities,
        graph=state.materialise_graph,
        deadline=state.deadline,
    )
    state.unit_gate_plan = gate.plan
    return gate.reason is None


async def _ground_elected_answer(state: ResolveState) -> RegionResult | None:
    """Check bound-unit convertibility, then ground facets, slots, refs, and entities.

    Recompute slot status from resolved member facts. Assembly builds the Spec.
    """
    # The phase spine narrows this result to ``Materialised``. The attribute loses that
    # narrowing here.
    mat_result = cast(Materialised, state.mat_result)
    start_elected_related_read(state, mat_result)

    early = await _decide_bound_unit_convertibility(state, mat_result)
    if early is not None:
        return early

    t0 = now_ms()

    # Coverage, the headline and the preview's read plan take the same filter once, before
    # any election (`filter_answer`): per-StatVar unit elections for a unit request, else the
    # literal unit filter, else the twin cluster's currency.
    if state.unit_request is not None:
        _admission = FacetAdmission(
            elected_units=elected_unit_refs(state.unit_elected_selections or {})
        )
    else:
        _admission = FacetAdmission(axes=_answer_unit_filter(state, mat_result))
    _filtered = filter_answer(
        mat_result,
        _admission,
        date_request=state.date_request,
        coverage_index=state.coverage.index,
    )
    mat_result, _filtered = _over_conversion_period(state, mat_result, _filtered, _admission)
    _install_filtered_answer(state, mat_result, _filtered)

    # A member holds data at the place but no point in the years read: the answer is
    # ``no_observations`` for the window, and `state.no_data` carries the `year_unavailable`
    # fact the carrier's gap gives, naming the years asked and the span the series runs. The
    # registry's gates name no window case, so the answer carries no flagged match. The answer
    # reads the member held, so a member that is not its family's total warns as an answer
    # showing it does. An unbound qualifier ranks first, and its refusal flags the grounded
    # Spec (`_assemble_carve_outs`).
    if mat_result.window_gap is not None and not refusal_terms(
        state, state.shape_draft, state.bindings
    ):
        if mat_result.family_total_absent is not None:
            state.depart(
                *family_total_absent_departure(
                    mat_result.family_total_absent,
                    spec=None,
                    variable_text=state.variable,
                    label_of=state.label_dcids,
                )
            )
        return state.no_data(
            "no_observations",
            nearest_real=await _nearest_real_for(state),
            llm_usage=state.var_usage,
        )

    # A bound unit's conversion runs over the filtered plan once, where the bound-unit gate
    # will read it (`_runs_unit_conversion`), so coverage, the headline and the plan leave
    # out the rows it omits before grounding reads them (`convertible_answer`). The gate
    # reports its outcome; a carve-out answers from the plan as filtered.
    if await asyncio.to_thread(_runs_unit_conversion, state, mat_result, _filtered.plan):
        _converted, state.unit_coverage = await asyncio.to_thread(
            convertible_answer,
            mat_result,
            _filtered,
            state.unit_elected_selections,
            graph=state.materialise_graph,
            sdmx=state.materialise_sdmx,
            rate_cache=state.rate_cache,
            dated_facets=state.graph,
            deadline=state.deadline,
            date_request=state.date_request,
            coverage_index=state.coverage.index,
        )
        if _converted is not _filtered:
            _filtered = _converted
            state.unit_gate_plan = _filtered.plan
            _install_filtered_answer(state, mat_result, _filtered)

    # The preview converts by the unit request once units are elected per StatVar.
    _converted_by = state.unit_request if state.unit_elected_selections is not None else None
    grounding_result = await asyncio.to_thread(
        _ground_answer,
        state.shape_draft,
        state.bindings,
        mat_result.sv_dcids,
        state.place.roles.roles,
        state.graph,
        # The filtered plan names the sources: under a unit request, the facets it serves
        # in the elected units. A conversion preview may read facets outside it.
        _filtered.plan,
        state.entity_label_text_by_dcid,
        unit_request=_converted_by,
        member_facts=mat_result.member_facts,
        open_axes=state.place.roles.open_axes,
    )

    (
        five_tuple_refs,
        role_refs,
        slots,
        slot_key_models,
        sv_refs,
        entity_objects_data,
        source_refs,
        slot_options,
        _draft_inapplicable_slots,  # unused: draft-kind-derived, see below for why
        _ground_warnings,
        labels,
    ) = grounding_result
    state.slots = slots
    state.source_refs = source_refs
    # A unit outside the `Currency<ISO>` convention reads the graph's currency units.
    state.answer_facets = await asyncio.to_thread(
        answer_facets,
        _filtered.plan,
        labels,
        bound_axes=state.bound_facet_axes,
        request_basis=state.unit_request_basis,
        unit_request=_converted_by,
        currency_of=functools.partial(unit_currency, state.graph, deadline=state.deadline),
    )
    state.slot_options = slot_options
    state.warnings.extend(_ground_warnings)

    # Resolved member facts describe the materialised population. Index-sourced shapes can
    # include members absent from ``shape_draft.sv_arc_facts``. Empty ``sv_dcids`` yields no
    # inapplicable slots. Add dropped measure axes after member-derived slots.
    inapplicable_slots = (
        inapplicable_slots_from_member_facts(
            slots=slots,
            sv_dcids=mat_result.sv_dcids,
            member_facts=mat_result.member_facts or {},
        )
        if mat_result.sv_dcids
        else []
    )
    already_inapplicable = set(inapplicable_slots)
    inapplicable_slots = inapplicable_slots + [
        axis for axis in state.measure_dropped_axes if axis not in already_inapplicable
    ]
    state.inapplicable_slots = inapplicable_slots
    unresolved_slots = (
        unresolved_slots_from_member_facts(
            slots=slots,
            sv_dcids=mat_result.sv_dcids,
            member_facts=mat_result.member_facts or {},
        )
        if mat_result.sv_dcids
        else []
    )
    state.unresolved_slots = unresolved_slots

    # Shape counts use admitted members. Elected StatVar refs are representatives. A missing
    # population occurs only for hand-built ``Materialised`` results and uses the fallback.
    shape_model = build_shape_model(
        state.shape_draft,
        slot_key_models,
        five_tuple_refs,
        member_count=(
            mat_result.population.admitted if mat_result.population is not None else len(sv_refs)
        ),
        members=mat_result.served_members,
    )
    state.shape_model = shape_model

    # Build from resolved member facts, which can include index-sourced StatVars.
    stat_vars = build_stat_vars(
        sv_refs,
        state.shape_draft.shape_id,
        slots,
        # The filtered plan dates each StatVar, as it names the sources.
        plan=_filtered.plan,
        destination_confirmed=mat_result.destination_confirmed,
        member_facts=mat_result.member_facts or {},
    )
    state.stat_vars = stat_vars

    entity_objects: list[Entity] = [
        _build_entity(role_draft, entity_ref, entity_type_ref, role_refs)
        for role_draft, entity_ref, entity_type_ref in entity_objects_data
    ]
    state.entity_objects = entity_objects
    # Hash the same role-dispatched entities presented in the response.
    identity_entities = build_identity_entities(entity_objects_data, role_refs)
    state.identity_entities = identity_entities

    state.timing["answer"] = now_ms() - t0
    await _emit_step(
        state.emit,
        state.variable_index,
        "answer",
        ms=state.timing["answer"],
    )


@dataclasses.dataclass(frozen=True)
class _DefiniteSetup:
    """Read-only setup shared by one definite assembly."""

    available_at_place: bool | None
    confidence: Confidence | None
    obs_entities: dict[str, list[str]]
    probe_role_entities: dict[str, list[str]]
    flat_probe_entities: list[str]
    bulk_scope: BulkScope | None
    role_axis_facts: list[RoleAxisFactInput]
    obs_fetch_facet_filter: Mapping[str, FacetAxisFilter] | None
    entity_task: asyncio.Task[tuple[list[EntityOptions], list[EntityOptions], Warning | None]]
    slot_options_raw: list[SlotOptions]
    facet_probe: FacetProbe
    t0: int
    # The facets the answer's identity binds (`_answer_facet_bindings`), None when it binds
    # none.
    facet_bindings: dict[str, IdentityFacetBinding] | None
    # The answer's served test for counting options in memory; None for a subject-place
    # family (`option_coverage.role_served_filter`).
    served_filter: RoleServedFilter | None = None
    # What `slot_options_raw` was counted from (`option_coverage.option_axis_facts`).
    option_axis_facts: tuple[OptionServedFact, ...] = ()

    @property
    def facets_pinned(self) -> bool:
        """Whether the answer's identity binds a facet, which makes every option count the
        coverage probe states an upper bound."""
        return self.facet_bindings is not None

    @property
    def counts_pinned(self) -> bool:
        """Whether the identity binds a facet value the in-memory counts do not apply
        (`unit.conversion.pins_facets`), which makes each an upper bound."""
        return pins_facets(self.facet_bindings)


def _start_assemble_definite(state: ResolveState, mat_result: Materialised) -> _DefiniteSetup:
    """Build shared assembly inputs and launch entity options before other coverage probes."""
    # No-place results keep ``available_at_place=None``. A no-data match has no confidence.
    # Otherwise confidence uses the representative score.
    available_at_place, confidence = _place_confidence(
        destination_dcid=state.destination_dcid,
        available_at_place=mat_result.available_at_place,
        representative_score=state.shape_draft.representative_score,
        has_data=mat_result.has_data,
        retrieved=state.shape_draft.retrieved,
        sole_reading=state.status_plan.sole,
    )

    # Probe the region members confirmed during materialisation.
    destination_dcids = state.place.probe_places
    roles = state.place.roles
    # Build observation entities before coverage tasks so their fetch can overlap.
    obs_entities, obs_entities_warn = _answer_observation_entities(state, mat_result)
    if obs_entities_warn is not None:
        state.warnings.append(obs_entities_warn)
    # Option probes use a narrower entity set than observations and exclude discovered-scope
    # members. This keeps candidate and single-shape probes aligned.
    probe_role_entities = coverage_probe_role_entities(
        state.shape_draft,
        mat_result,
        destination_dcids=destination_dcids,
        destination_dcid=roles.obs_destination_dcid,
        origin_dcids=roles.origin_dcids,
    )
    # Bulk reads apply only to a plural, defaulted country scope. Single-place scopes use
    # the per-pair path.
    bulk_scope = bulk_scope_for(state.place, mat_result, sdmx=state.sdmx)
    # Entity options have no bound-unit gate, so launch them immediately. Their raw
    # values feed relaunch. Shared role-axis facts keep options and Spec aligned.
    t0 = now_ms()
    role_axis_facts = role_axis_facts_for_materialised(mat_result, shape=state.shape_draft)
    entity_task = asyncio.create_task(
        asyncio.to_thread(
            _build_and_enrich_entity_options,
            state.shape_draft,
            roles.resolved_entities,
            state.entity_objects,
            mat_result.sv_dcids,
            date_request=state.date_request,
            graph=state.graph,
            sdmx=state.sdmx,
            include_option_coverage=state.include_option_coverage,
            role_axis_facts=role_axis_facts,
            place_dcids=destination_dcids,
        )
    )

    # Request-form elections filter observations through the elected selections. A flat
    # literal-unit predicate would reject converted series, whose facets report native units.
    obs_fetch_facet_filter = (
        None if state.unit_request is not None else _answer_unit_filter(state, mat_result)
    )

    facet_bindings = _answer_facet_bindings(state, mat_result)
    return _DefiniteSetup(
        available_at_place=available_at_place,
        confidence=confidence,
        obs_entities=obs_entities,
        probe_role_entities=probe_role_entities,
        flat_probe_entities=_flat_obs_entities(probe_role_entities),
        bulk_scope=bulk_scope,
        role_axis_facts=role_axis_facts,
        obs_fetch_facet_filter=obs_fetch_facet_filter,
        entity_task=entity_task,
        slot_options_raw=state.slot_options,
        facet_probe=_answer_facet_probe(state, mat_result, probe_role_entities),
        t0=t0,
        facet_bindings=facet_bindings,
        served_filter=role_served_filter(
            state.shape_draft,
            probe_role_entities,
            open_axes=state.place.roles.open_axes,
            years=mat_result.preview_years,
            coverage=state.coverage.index,
            facets_pinned=pins_facets(facet_bindings),
            plan=state.elected_plan,
            conversion=counted_conversion(
                state.elected_plan, facet_bindings, catalogue=current_conversion_catalogue()
            ),
        ),
    )


async def _fetch_definite_observations(
    mat_result: Materialised, setup: _DefiniteSetup, state: ResolveState
) -> tuple[SpecObservations, Warning | None]:
    """Fetch observations with the shared arguments used by buffered and SSE paths."""
    return await asyncio.to_thread(
        _build_observations,
        state.elected_plan,
        list(mat_result.sv_dcids),
        setup.obs_entities,
        graph=state.graph,
        materialise_graph=state.materialise_graph,
        materialise_sdmx=state.materialise_sdmx,
        facet_filter=setup.obs_fetch_facet_filter,
        scope=setup.bulk_scope,
        sdmx=state.sdmx,
        deadline=state.deadline,
        member_facts=mat_result.member_facts,
        rate_cache=state.rate_cache,
        unit_request=state.unit_request,
        elected=state.unit_elected_selections,
        coverage=state.coverage,
    )


@dataclasses.dataclass(frozen=True)
class _DefiniteCoverage:
    """Coverage results shared by Spec construction and finalisation."""

    # The answer's counted menus and what they were counted from (`counted_slot_options`).
    counted: CountedSlotOptions
    slot_options: list[SlotOptions]
    facet_options: list[FacetOptions]
    entity_options_raw: list[EntityOptions]
    entity_options: list[EntityOptions]
    obs_result: tuple[SpecObservations, Warning | None] | None
    eager_obs_availability: tuple[SpecObservations, Warning | None] | None


def _bound_unit_refusal(
    state: ResolveState, mat_result: Materialised, setup: _DefiniteSetup
) -> RegionResult | None:
    """The answer a bound unit request refuses with at the gate, or None when the elected
    plan serves every StatVar under its unit and every served row converts, as grounding
    found it (`ResolveState.unit_coverage`). A pass whose conversion omits the rows of
    entities no currency names adds `currency_unmapped_omitted` to the answer's warnings."""
    gate_plan = state.unit_gate_plan if state.unit_gate_plan is not None else state.elected_plan
    reason = bound_unit_gate_reason(gate_plan, setup.obs_entities, mat_result.sv_dcids)
    if reason is not None:
        return state.no_data(reason, llm_usage=state.var_usage)
    # Grounding ran the conversion on every path that reaches here (`_runs_unit_conversion`).
    coverage = state.unit_coverage
    if coverage.refusal is None:
        if (omitted := coverage.omitted_warning()) is not None:
            state.warnings.append(omitted)
        return None
    # A bound outcome guarantees the unit request.
    return unit_conversion_refusal(
        state, coverage.refusal[1], unit_request=cast(IdentityUnitRequest, state.unit_request)
    )


async def _gather_definite_coverage(
    state: ResolveState,
    mat_result: Materialised,
    setup: _DefiniteSetup,
    count_task: asyncio.Task[CountedSlotOptions],
) -> RegionResult | _DefiniteCoverage:
    """Check bound-unit availability, run eligible coverage probes, and await their tasks.

    `count_task` counts the answer's menus (`counted_slot_options`) beside the entity,
    facet and observation reads; the slot probe starts from its counted menus once it
    ends. Unit refusals return a terminal result before slot/facet probes launch. Record
    option probe timing and return the coverage bundle otherwise.
    """
    # Skip option probes when a known carve-out guarantees no definite result.
    skip_coverage_probe = _gate_carve_out(state, mat_result) or not state.stat_vars

    # The bound unit is decided from the answer's read plan before any fetch, so a request
    # without a preview reads none: the plan must serve every StatVar under its elected unit
    # (`bound_unit_gate`), and every row it serves must convert (`unit_coverage`). A
    # known carve-out decides first.
    if state.bound_unit_outcome is not None and not skip_coverage_probe:
        refusal = _bound_unit_refusal(state, mat_result, setup)
        if refusal is not None:
            setup.entity_task.cancel()
            count_task.cancel()
            await asyncio.gather(setup.entity_task, count_task, return_exceptions=True)
            return refusal

    # Launch slot/facet probes with the buffered observations fetch to overlap network waits.
    # Facet options are built for every answer; a streamed request with option coverage
    # reads them in its trailing probe instead.
    slot_task: asyncio.Task[tuple[list[SlotOptions], Warning | None]] | None = None
    facet_task: asyncio.Task[tuple[list[FacetOptions], Warning | None]] | None = None
    if state.include_option_coverage and not skip_coverage_probe:
        slot_task = asyncio.create_task(_probe_counted_slot_options(state, setup, count_task))
    if not state.stream_slot_options and not skip_coverage_probe:
        facet_task = asyncio.create_task(
            asyncio.to_thread(
                _maybe_build_facet_options,
                setup.facet_probe,
                graph=state.graph,
                coverage=state.coverage.index,
                deadline=state.deadline,
            )
        )
    obs_fetch_task: asyncio.Task[tuple[SpecObservations, Warning | None]] | None = None
    if state.include_observations and not state.stream_observations:
        obs_fetch_task = asyncio.create_task(_fetch_definite_observations(mat_result, setup, state))
    # Streaming defers points, but status and ``resolved.facets`` belong on the immediate
    # response. Fetch eagerly for every streamed preview. A later task sends points.
    eager_obs_availability: tuple[SpecObservations, Warning | None] | None = None
    if state.stream_observations:
        eager_obs_availability = await _fetch_definite_observations(mat_result, setup, state)
    facet_options: list[FacetOptions] = []
    obs_result: tuple[SpecObservations, Warning | None] | None = None
    try:
        counted = await count_task
        # Keep the counted options unless a coverage probe ran.
        slot_options = counted.options
        entity_options_raw, entity_options, entity_coverage_warn = await setup.entity_task
        if entity_coverage_warn is not None:
            state.warnings.append(entity_coverage_warn)
        obs_result = await obs_fetch_task if obs_fetch_task is not None else None
        if slot_task is not None:
            slot_options, coverage_warn = await slot_task
            if coverage_warn is not None:
                state.warnings.append(coverage_warn)
        if facet_task is not None:
            facet_options, facet_coverage_warn = await facet_task
            if facet_coverage_warn is not None:
                state.warnings.append(facet_coverage_warn)
    finally:
        # Cancel and await pending worker tasks so cancellation cannot detach them.
        pending = [
            t
            for t in (setup.entity_task, count_task, obs_fetch_task, slot_task, facet_task)
            if t is not None and not t.done()
        ]
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    # Time inline enrichment only. Trailing SSE probes are excluded from response latency.
    state.timing["options"] = now_ms() - setup.t0

    return _DefiniteCoverage(
        counted=counted,
        slot_options=slot_options,
        facet_options=facet_options,
        entity_options_raw=entity_options_raw,
        entity_options=entity_options,
        obs_result=obs_result,
        eager_obs_availability=eager_obs_availability,
    )


async def _probe_counted_slot_options(
    state: ResolveState, setup: _DefiniteSetup, count_task: asyncio.Task[CountedSlotOptions]
) -> tuple[list[SlotOptions], Warning | None]:
    """Probe the menus `count_task` leaves uncounted (`_maybe_enrich_slot_options`) once it
    has counted them."""
    counted = await count_task
    return await asyncio.to_thread(
        _maybe_enrich_slot_options,
        state.shape_draft,
        state.bindings,
        counted.options,
        entities=state.place.roles.resolved_entities,
        open_axes=state.place.roles.open_axes,
        date_request=state.date_request,
        materialise_graph=state.materialise_graph,
        # Option probes use filtered entities. Observation fetches use the resolved scope.
        place_dcids=setup.flat_probe_entities,
        include_option_coverage=True,
        facets_pinned=setup.facets_pinned,
    )


def _answer_facet_bindings(
    state: ResolveState, mat_result: Materialised
) -> dict[str, IdentityFacetBinding] | None:
    """The facets the answer's identity binds: its bound unit request, else its twin
    cluster's currency (`_cluster_currency_bindings`); None when it binds none."""
    if state.bound_unit_outcome is not None:
        # A bound outcome guarantees a unit request. The type checker cannot carry that
        # attribute invariant through this access.
        return {
            "unit": IdentityFacetValueRequest(request=cast(IdentityUnitRequest, state.unit_request))
        }
    return _cluster_currency_bindings(state, mat_result)


def _cluster_currency_bindings(
    state: ResolveState, mat_result: Materialised
) -> dict[str, IdentityFacetBinding] | None:
    """The identity's unit binding for an answer narrowed to its twin cluster's currency
    (`cluster_currency_binding`), so a mint of the Spec serves the same facets; else None."""
    if state.unit_facet_filter is not None or state.unit_request is not None:
        return None
    facet_filter = cluster_currency_filter(mat_result)
    if facet_filter is None:
        return None
    narrowed = narrow_facets_by_sv_to_filter(mat_result.facets_by_sv, facet_filter) or {}
    return {"unit": cluster_currency_binding([f for fs in narrowed.values() for f in fs])}


@dataclasses.dataclass(frozen=True)
class _DefiniteSpec:
    """The assembled Spec and its measure-partition/open-axis facts,
    `_finalize_definite_result`'s own remaining inputs from `_build_definite_spec`."""

    spec: Spec
    measure_partition: MeasurePartition | None
    measure_siblings: tuple[MeasurePartition, ...]
    open_axis_facts: tuple[OpenAxisFact, ...]


def _build_definite_spec(
    state: ResolveState,
    mat_result: Materialised,
    slots: list[Slot],
    setup: _DefiniteSetup,
    coverage: _DefiniteCoverage,
) -> _DefiniteSpec:
    """Build the scope and Spec from materialised data, coverage, and resolve state."""
    # A defaulted or discovered scope lists the members the filtered plan serves, the ones
    # coverage counts.
    scope = None
    scope_member_dcids = stated_scope_members(mat_result, state.elected_plan)
    if scope_member_dcids is not None:
        scope_labels, scope_types = fetch_scope_members(scope_member_dcids, graph=state.graph)
        scope = build_scope(scope_axis(mat_result), scope_member_dcids, scope_labels, scope_types)

    # Assemble the Spec
    slots = apply_slot_expansions(slots, slot_expansions(state.bindings))
    measure_bound_values = bound_values_from_bindings(state.bindings)
    # Grounding reads the family before assembly starts.
    measure_partition, measure_siblings = cast(MeasureFamily, state.measure_family)
    served_member_facts = mat_result.member_facts or {}
    served_members = [served_member_facts[dcid] for dcid in mat_result.sv_dcids]
    # Record served values for every open constraint axis.
    open_axis_facts = open_constraint_axes(
        measure_bound_values, dict(zip(mat_result.sv_dcids, served_members, strict=True))
    )
    spec = build_spec(
        shape=state.shape_model,
        slots=slots,
        stat_vars=state.stat_vars,
        entities=state.entity_objects,
        coverage=state.elected_coverage,
        plan=state.elected_plan,
        pin_served_provenance=state.pin_served_provenance,
        headline=state.elected_headline,
        coverage_index=state.coverage.index,
        coverage_stamp=mat_result.coverage_stamp,
        scope=scope,
        default_reading=state.default_reading,
        stated_form=state.stated_form,
        resolved_sources=state.source_refs,
        facets=state.answer_facets,
        # The confirmed count is exact unless measured over a capped recall sample. A
        # missing population occurs only on hand-built materialised results.
        stat_vars_member_count=(
            mat_result.population.confirmed if mat_result.population is not None else None
        ),
        stat_vars_member_count_exact=(
            mat_result.population.member_count_exact if mat_result.population is not None else True
        ),
        walked=state.walked,
        slot_options=coverage.slot_options,
        entity_options=coverage.entity_options,
        facet_options=coverage.facet_options,
        available_at_place=setup.available_at_place,
        confidence=setup.confidence,
        requested_window=requested_window_from_date_request(state.date_request),
        identity_entities=state.identity_entities,
        open_roles=mat_result.open_axes,
        inapplicable_slots=state.inapplicable_slots,
        unresolved_slots=state.unresolved_slots,
        # Use qualifier bindings after election vetting and repair.
        text_recognized_properties=state.text_recognized_properties,
        query_qualifiers=answer_terms(state, state.shape_draft, state.bindings),
        facet_bindings=setup.facet_bindings,
        # Record properties forced by measure election.
        measure_elected_properties=state.measure_elected_properties,
        defaulted_properties=state.defaulted_properties,
        measure_partition=measure_partition,
        measure_siblings=measure_siblings,
        open_axis_facts=open_axis_facts,
        role_axis_facts=setup.role_axis_facts,
        named_groups=request_named_groups(state, state.place.roles),
        measure_served_rows=measure_served_counter(setup.served_filter),
    )
    spec = with_unit_conversion(
        spec,
        members=served_members,
        plans=(state.elected_plan,) if state.elected_plan is not None else (),
    )

    # The request, warnings, and basis are already on the Spec. State only the lexicon
    # version.
    if state.bound_unit_outcome is not None:
        spec = with_unit_lexicon_version(spec, state.bound_unit_outcome)
    spec = apply_declaration_version(spec, state.bindings)
    if mat_result.family_total_absent is not None:
        state.family_total_absent[spec.spec_id] = mat_result.family_total_absent

    return _DefiniteSpec(
        spec=spec,
        measure_partition=measure_partition,
        measure_siblings=measure_siblings,
        open_axis_facts=open_axis_facts,
    )


async def _assemble_carve_outs(
    state: ResolveState, mat_result: Materialised, spec: Spec
) -> RegionResult | None:
    """Apply the remaining no-data gates after Spec construction. ``None`` means definite.

    Qualifier refusal precedes missing-slice and observation checks. Bound-unit availability
    is already verified by ``bound_unit_gate_reason``.
    """
    # Judge the final answer on every requirement (`requirements.settle_answer`). One it
    # leaves unmet refuses before any broader answer can pass as definite, naming the terms
    # its trace records. The candidates path judges its answers the same way.
    unbound_qualifiers = settle_answer(state, state.shape_draft, state.bindings)
    if unbound_qualifiers:
        return state.no_data(
            "qualifier_not_bound",
            flagged_match=(spec,),
            gate=None,
            llm_usage=state.var_usage,
            unbound_qualifiers=unbound_qualifiers,
        )

    # Keep the matched Spec as ``flagged_match`` when no group member has data at the
    # place. Read the materialised value directly. Confidence masks it when no place is
    # named. A defaulted country scope can establish ``False``.
    if mat_result.available_at_place is False:
        return state.no_data(
            "no_observations",
            flagged_match=(spec,),
            gate=NO_MEMBER_HAS_DATA_AT_PLACE,
            nearest_real=await _nearest_real_for(state),
            llm_usage=state.var_usage,
        )

    # Without a named place, ``has_data=False`` means no member asserted every bound
    # property. Report the matched Spec as ``flagged_match``.
    if state.destination_dcid is None and not mat_result.has_data:
        return state.no_data(
            "no_matching_slice",
            flagged_match=(spec,),
            gate=NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
            llm_usage=state.var_usage,
        )

    # A data-bearing result must have at least one StatVar.
    if not state.stat_vars:
        return state.no_data("no_observations", llm_usage=state.var_usage)
    return None


async def _finalize_definite_result(
    state: ResolveState,
    mat_result: Materialised,
    setup: _DefiniteSetup,
    coverage: _DefiniteCoverage,
    definite_spec: _DefiniteSpec,
) -> RegionResult:
    """Attach eager SSE data and related measures, launch relaunch tasks, and return the
    definite result."""
    spec = await with_related(definite_spec.spec, state.related_read)
    # Include facet data on the immediate SSE response, including for unbound axes.
    if state.stream_observations and coverage.eager_obs_availability is not None:
        spec = _attach_observations(
            spec,
            coverage.eager_obs_availability,
            state.warnings,
            include_points=False,
        )

    family = state.family_total_absent_departures((spec,))
    relaunch_slot_options = functools.partial(
        _relaunch_slot_options,
        state=state,
        shape_draft=state.shape_draft,
        bindings=state.bindings,
        entities=state.place.roles.resolved_entities,
        open_axes=state.place.roles.open_axes,
        # Option relaunches use the same probe entities as the initial task.
        place_dcids=setup.flat_probe_entities,
        entity_options=coverage.entity_options_raw,
        default_facet_probe=setup.facet_probe,
        facets_pinned=setup.facets_pinned,
        variable_labels=member_labels((spec,)),
    )
    slot_options_task = (
        relaunch_slot_options(spec.spec_id, setup.slot_options_raw)
        if state.stream_slot_options
        else None
    )

    # Place options require both the graph reader and their dedicated executor. Omit the
    # task when either dependency is absent.
    place_options_materialise_graph = state.materialise_graph
    stream_place_options = (
        state.stream_slot_options
        and place_options_materialise_graph is not None
        and state.place_options_pool is not None
    )
    relaunch_place_options = functools.partial(
        _relaunch_place_options,
        state=state,
        spec_identity=spec.definition.identity,
        place_options_materialise_graph=cast(GraphReader, place_options_materialise_graph),
    )
    place_options_task = relaunch_place_options(spec.spec_id) if stream_place_options else None

    # Create the deferred SSE task only after the Spec has an ID for correlation.
    observations_task: asyncio.Task[_ObsProbeResult] | None = None
    observations_relaunch: Callable[[str], asyncio.Task[_ObsProbeResult]] | None = None
    if state.stream_observations:
        spec, observations_task, observations_relaunch = await _resolve_spec_observations(
            spec,
            list(mat_result.sv_dcids),
            setup.obs_entities,
            graph=state.graph,
            materialise_graph=state.materialise_graph,
            materialise_sdmx=state.materialise_sdmx,
            include_observations=state.include_observations,
            stream_observations=state.stream_observations,
            warnings=state.warnings,
            facet_filter=setup.obs_fetch_facet_filter,
            scope=setup.bulk_scope,
            sdmx=state.sdmx,
            deadline=state.deadline,
            member_facts=mat_result.member_facts,
            rate_cache=state.rate_cache,
            unit_request=state.unit_request,
            elected=state.unit_elected_selections,
            coverage=state.coverage,
            plan=state.elected_plan,
        )
    elif coverage.obs_result is not None and state.include_observations:
        # A bound-unit check fetches observations whether or not a preview is asked for.
        # Without one the answer states nothing from that fetch, as its resubmit, which
        # fetches only for a preview, states nothing.
        spec = _attach_observations(spec, coverage.obs_result, state.warnings)

    return RegionResult(
        variable_text=state.variable,
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(
            *state.warnings,
            *(warning for warning, _fact in family),
            *whole_total_warnings(spec, coverage=state.coverage.index),
        ),
        departure_facts={**state.departure_facts, **raised_facts(family)},
        timing_by_step=dict(state.timing),
        llm_usage=state.var_usage,
        slot_options_task=slot_options_task,
        facet_probe=setup.facet_probe,
        slot_options_relaunch=relaunch_slot_options if state.stream_slot_options else None,
        place_options_task=place_options_task,
        place_options_relaunch=relaunch_place_options if stream_place_options else None,
        observations_task=observations_task,
        observations_relaunch=observations_relaunch,
        measure_partition=definite_spec.measure_partition,
        measure_siblings=definite_spec.measure_siblings,
        open_axis_facts=definite_spec.open_axis_facts,
        option_axis_facts=setup.option_axis_facts,
        served_filter=setup.served_filter,
        served_members=tuple((mat_result.member_facts or {})[dcid] for dcid in mat_result.sv_dcids),
        served_plans=(state.elected_plan,) if state.elected_plan is not None else (),
        role_axis_facts=tuple(setup.role_axis_facts),
        member_population=mat_result.population,
        answered_roles=answered_roles(state.shape_draft),
    )


async def _assemble_definite_answer(state: ResolveState) -> RegionResult:
    """Build the Spec, apply no-data gates, and return the terminal result."""
    mat_result = cast(Materialised, state.mat_result)

    # Bind the when-slot to the extracted window before assembling the Spec
    slots = bind_when_slot(state.slots, window=state.elected_coverage.window)

    setup = _start_assemble_definite(state, mat_result)
    # Every menu the answer, its coverage probe and its trailing events offer is counted
    # first, so an option that would serve nothing is dropped everywhere. The count runs beside
    # the coverage reads, which wait on the network.
    count_task = asyncio.create_task(
        asyncio.to_thread(
            counted_slot_options,
            state.slot_options,
            slots=state.slots,
            shape=state.shape_draft,
            bindings=state.bindings,
            plan=state.elected_plan,
            member_facts=mat_result.member_facts or {},
            served_filter=setup.served_filter,
            facets_pinned=setup.counts_pinned,
            conversion=counted_conversion(
                state.elected_plan, setup.facet_bindings, catalogue=current_conversion_catalogue()
            ),
        )
    )

    gathered = await _gather_definite_coverage(state, mat_result, setup, count_task)
    if isinstance(gathered, RegionResult):
        return gathered
    coverage = gathered
    state.slot_options = coverage.counted.options
    setup = dataclasses.replace(
        setup,
        slot_options_raw=coverage.counted.options,
        option_axis_facts=coverage.counted.facts,
    )

    # The scope's label and type reads run off the event loop so the request deadline can fire.
    definite_spec = await asyncio.to_thread(
        _build_definite_spec, state, mat_result, slots, setup, coverage
    )
    state.state_reelected_reading(definite_spec.spec)

    early = await _assemble_carve_outs(state, mat_result, definite_spec.spec)
    if early is not None:
        return early
    note_denominator_unverified(state, state.shape_draft, spec_id=definite_spec.spec.spec_id)

    return await _finalize_definite_result(state, mat_result, setup, coverage, definite_spec)


async def _phase_outcome(state: ResolveState) -> RegionResult:
    """The part's terminal result over the evidence materialisation read.

    The status rule offers candidates when ambiguity yields data
    (`candidates.resolve_candidates`). Otherwise the elected shape answers alone: its
    no-data or unsupported read refuses (`_no_data_answer`), a bound unit it cannot convert
    to or a window holding no point refuses at grounding (`_ground_elected_answer`), and
    else the definite Spec is minted (`_assemble_definite_answer`).
    """
    result = await resolve_candidates(state, rate_cache=state.rate_cache)
    if result is None:
        result = await _no_data_answer(state)
    if result is None:
        result = await _ground_elected_answer(state)
    if result is None:
        result = await _assemble_definite_answer(state)
    return result


async def _no_data_answer(state: ResolveState) -> RegionResult | None:
    """The elected shape's no-data or unsupported result, ``None`` when materialisation
    found data.

    Attach nearest-real suggestions for ``no_observations``. An elected shape the quantity
    gate found does not serve the leg's quantity answers nothing alone, so the refusal the
    gate deferred to the fan-out stands when the fan-out left the elected shape's path
    (`refuse_unserved_quantity`).
    """
    if (refused := await refuse_unserved_quantity(state)) is not None:
        return refused
    mat_result = state.mat_result

    if isinstance(mat_result, NoDataDraft):
        # Relax constraint bindings to suggest nearby data when observations are missing.
        nearest_real: tuple[NearestRealSpec, ...] | None = None
        if mat_result.reason == "no_observations":
            nearest_real = await _nearest_real_for(state)
        if (
            refused := state.source_code_no_data(mat_result.reason, llm_usage=state.var_usage)
        ) is not None:
            return refused
        return RegionResult(
            variable_text=state.variable,
            status="no_data",
            specs=(),
            no_data_reason=mat_result.reason,
            warnings=tuple(state.warnings),
            departure_facts=dict(state.departure_facts),
            timing_by_step=dict(state.timing),
            settled_dataset=state.settled_dataset(),
            nearest_real=nearest_real,
            llm_usage=state.var_usage,
            no_data_role_axis=(
                open_role_axis(state.shape_draft, state.place.roles.resolved_entities)
                if mat_result.reason == "no_observations"
                else None
            ),
        )

    if isinstance(mat_result, UnsupportedDraft):
        return _unsupported(
            state,
            mat_result.code,
            pointer=mat_result.pointer,
            detail=mat_result.detail,
            llm_usage=state.var_usage,
            facets=mat_result.facets,
        )

    # Treat unexpected resolver outcomes as ``no_observations``.
    if not isinstance(mat_result, Materialised):
        return state.no_data("no_observations", llm_usage=state.var_usage)

    return None
