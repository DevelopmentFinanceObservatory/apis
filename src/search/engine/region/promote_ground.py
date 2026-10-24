"""Ground the resolved answer, build its Spec, and apply no-data and facet gates."""

from __future__ import annotations

import functools
from typing import cast

from code_index.shared import get_index, get_measure_index
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.facets import role_bearing_facet_candidates
from materialise.recipes import (
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    RoleAxisConflictError,
    RoleAxisNotDeclaredError,
    cause_from_recipe_error,
)
from search.engine.clients.graph import Facet
from search.engine.departures import window_gap_fact
from search.engine.envelope import now_ms
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.extract import DateRequest
from search.engine.region.arcs import inapplicable_slots_from_arcs, unresolved_slots_from_arcs
from search.engine.region.codes import (
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
)
from search.engine.region.convertibility import (
    BoundFacetCombinationsCeilingExceededError,
    UnitCoverageRefusal,
    bound_facet_refs,
    conversion_period_departure,
    convertible_answer,
    decide_part_unit,
    deflator_omission_departure,
    elected_unit_refs,
    over_conversion_period,
    survivor_declared_roles,
    unit_gate,
    unit_request_period,
)
from search.engine.region.facets import (
    FacetAdmission,
    FilteredAnswer,
    answer_facets,
    filter_answer,
)
from search.engine.region.grounding import (
    _build_entity,
    _ground_answer,
    _place_confidence,
    restate_role_scope_warnings,
)
from search.engine.region.materialise_step import (
    resubmit_family_total_absent_departure,
)
from search.engine.region.observations.entities import (
    _flat_obs_entities,
    _observation_entities,
    coverage_probe_role_entities,
    scope_axis_dcids,
)
from search.engine.region.option_coverage import (
    counted_slot_options,
    measure_served_counter,
    role_axis_facts_for_materialised,
    role_served_filter,
)
from search.engine.region.promote_materialise import _graph_labels
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.related_read import submit_related_read
from search.engine.region.requirements import posted_refusal_terms, settle_posted
from search.engine.region.resubmit_request import (
    _facet_filter_from_bindings,
    _rebind_source_slot,
    _requested_window_for_id,
    _unit_request_basis_from_facets,
    resubmit_named_groups,
)
from search.engine.region.result import RegionResult
from search.engine.region.scope import (
    build_scope,
    fetch_scope_members,
    scope_axis,
    stated_scope_members,
)
from search.engine.region.state import _elected_stat_vars
from search.engine.region.whole_totals import whole_total_warnings
from search.engine.retrieval.bound_values import (
    apply_declaration_version,
    apply_slot_expansions,
    bound_values_from_bindings,
    entity_role_keys,
    observed_measure_family,
    open_constraint_axes,
    shape_query,
    slot_expansions,
)
from search.engine.retrieval.elected_dataset import elected_datasets, published
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import plan_probes, serves_nothing
from search.engine.retrieval.retrieve import Materialised
from search.engine.shape import is_role_bearing
from search.engine.spec_build import (
    bind_when_slot,
    build_shape_model,
    build_spec,
    build_stat_vars,
)
from search.engine.spec_id import bound_facet_axes
from search.engine.unit.conversion import (
    ConversionPeriod,
    counted_conversion,
    pins_facets,
    with_unit_conversion,
)
from search.engine.unit.conversion_catalogue import current_conversion_catalogue
from search.engine.unit.refs import unit_currency
from search.models import (
    EntityRoleKey,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    NoDataGate,
    NoDataReason,
    ResolvedQualifier,
    Spec,
    TimeWindow,
)


def _decide_unit_convertibility(state: _PromoteState, mat_result: Materialised) -> None:
    """Decide convertibility once per resolved StatVar, before facet refusal.

    Only request-form unit bindings need a decision; validation has already rejected units
    without graph refs. Store elected facets on `state.elected_unit`, then narrow facets and
    role indexes so grounding, coverage, availability, and observations use the same choices.
    Ref/set bindings and empty role maps are no-ops. Missing materialise clients raise
    `EngineInfraError`. Input errors include `facet_not_available`, `request_invalid` (422) for
    undeclared or conflicting roles, `facet_probe_ceiling` (422) when the Set-bound
    provenance/method product exceeds `_BOUND_FACET_COMBINATIONS_MAX`, and
    `recipe_not_declared` or `rate_coverage_gap` with cause."""
    _unit_binding = (state.inp.identity.facets or {}).get("unit")
    _convertibility_request = (
        _unit_binding.request if isinstance(_unit_binding, IdentityFacetValueRequest) else None
    )
    if _convertibility_request is None:
        return
    if state.materialise_graph is None or state.materialise_sdmx is None:
        # A bound unit request always needs this decision.
        raise EngineInfraError(
            "materialise graph/sdmx client not initialised; cannot decide this "
            "bound unit request's own convertibility"
        )
    # _phase_materialise settles the place context on discovered role scope members; the
    # observations preview probes the same set.
    place = state.place
    _decision_destination_dcids, _decision_origin_dcids = scope_axis_dcids(
        mat_result, place.probe_places
    )
    _decision_role_bindings, _ = _observation_entities(
        state.shape_draft,
        destination_dcids=_decision_destination_dcids,
        destination_dcid=place.roles.obs_destination_dcid,
        origin_dcids=_decision_origin_dcids or place.roles.origin_dcids,
    )
    # Empty role bindings leave no entity for convertibility; facet-refusal gates handle it.
    if not _decision_role_bindings:
        return
    _bound_facets = bound_facet_refs(state.inp.identity.facets, state.source_dcids)
    _declared_roles = survivor_declared_roles(mat_result)
    try:
        state.elected_unit = decide_part_unit(
            mat_result.sv_dcids,
            unit_request=_convertibility_request,
            role_bindings=_decision_role_bindings,
            declared_roles=_declared_roles,
            bound_facets=_bound_facets,
            years=mat_result.preview_years,
            graph=state.materialise_graph,
            sdmx=state.materialise_sdmx,
            coverage=state.coverage.index,
            # The decision ranks the candidates the answer's plan elects from.
            probes=plan_probes(mat_result.read_plan),
            datasets=elected_datasets(mat_result.member_facts),
            deadline=state.deadline,
        )
    except NonCurrencyUnitError as exc:
        raise unit_refusal_input_error(exc) from exc
    except (RoleAxisNotDeclaredError, RoleAxisConflictError) as exc:
        # Both errors reject axes that are undeclared or shared by multiple roles.
        detail = str(exc)
        raise EngineInputError(
            detail,
            code="request_invalid",
            status=422,
            pointer="/input/facets/unit",
        ) from exc
    except BoundFacetCombinationsCeilingExceededError as exc:
        # Check the Set-bound provenance/method product against its limit before probing.
        detail = str(exc)
        raise EngineInputError(
            detail,
            code="facet_probe_ceiling",
            status=422,
            pointer="/input/facets/unit",
        ) from exc
    except (RecipeNotDeclaredError, RateCoverageGapError) as exc:
        raise unit_refusal_input_error(exc) from exc
    except (FacetNotAvailableError, DefaultFacetUndefinedError) as exc:
        # No facet a mint would elect (no candidate on some axis, or
        # no Set member had one). Raised ahead of the observations fetch,
        # same as facet_refusal's facet_not_available.
        detail = exc.detail or str(exc)
        raise EngineInputError(
            detail,
            code="facet_not_available",
            status=422,
            pointer="/input/facets/unit",
        ) from exc


def unit_refusal_input_error(exc: UnitCoverageRefusal) -> EngineInputError:
    """The input error a resubmit raises when its unit cannot be converted: the error's own
    code and status at the unit pointer, with the recipe or rate cause it carries."""
    if isinstance(exc, NonCurrencyUnitError):
        detail = str(exc)
        return EngineInputError(
            detail,
            code=exc.code,
            status=exc.status,
            pointer="/input/facets/unit",
        )
    detail = exc.detail or str(exc)
    return EngineInputError(
        detail,
        code=exc.code,
        status=exc.status,
        pointer="/input/facets/unit",
        cause=cause_from_recipe_error(exc),
    )


def _narrow_ground_facets(state: _PromoteState, mat_result: Materialised) -> None:
    """Rebuild the facet filter after unit convertibility runs and apply it to the answer.

    Coverage, the headline and the preview's read plan take the full filter, provenance
    and per-StatVar unit choices included, once before any election (`filter_answer`).
    Grounding cites the sources that plan serves, dates each StatVar from it and lists its
    scope members, so a resubmit states what its first answer did. Without elected units,
    the filter matches the one cached during materialisation."""
    _elected_units = state.elected_units = (
        elected_unit_refs(state.elected_unit) if state.elected_unit is not None else None
    )
    _facet_filter = state.facet_filter = _facet_filter_from_bindings(
        state.graph,
        state.source_dcids,
        state.inp.identity.facets,
        elected_units=_elected_units,
        deadline=state.deadline,
    )
    _filtered = filter_answer(
        mat_result,
        FacetAdmission(axes=_facet_filter, elected_units=_elected_units),
        date_request=state.date_request,
        coverage_index=state.coverage.index,
    )
    _install_narrowed_answer(state, mat_result, _filtered)


def _over_conversion_period(state: _PromoteState, mat_result: Materialised) -> Materialised:
    """The answer the unit request shows when its conversion tables cover only part of the
    years the narrowed plan serves (`convertibility.unit_request_period`): the identity reads
    the period's window from here on (`_requested_window_for_id`), the plan is read over it
    and narrowed again, and the answer departs `year_unavailable` naming the table, as a
    fresh request's does. Otherwise `mat_result` as it is. Either way an answer whose
    deflators leave out some entities' rows departs `year_unavailable` naming them
    (`convertibility.deflator_omission_departure`)."""
    request = _converted_by(state)
    if request is None:
        return mat_result
    catalogue = current_conversion_catalogue()
    period = unit_request_period(
        state.narrowed_plan,
        request,
        state.elected_unit or {},
        declared_roles=survivor_declared_roles(mat_result),
        catalogue=catalogue,
    )
    if period is not None:
        mat_result = _narrowed_to_period(state, mat_result, period)
    omission = deflator_omission_departure(
        state.narrowed_plan, request, coverage_index=state.coverage.index, catalogue=catalogue
    )
    if omission is not None:
        state.warnings.append(state.depart(*omission))
    return mat_result


def _narrowed_to_period(
    state: _PromoteState, mat_result: Materialised, period: ConversionPeriod
) -> Materialised:
    """The answer read over `period`'s window, departing `year_unavailable` naming the
    table that leaves the other years out (`_over_conversion_period`)."""
    requested = state.date_request.window if state.date_request is not None else None
    state.date_request = DateRequest(window=period.window, latest=False)
    mat_result = over_conversion_period(mat_result, period)
    state.mat_result = mat_result
    _narrow_ground_facets(state, mat_result)
    state.warnings.append(
        state.depart(
            *conversion_period_departure(period, requested=requested, plan=state.narrowed_plan)
        )
    )
    return mat_result


def _install_narrowed_answer(
    state: _PromoteState, mat_result: Materialised, filtered: FilteredAnswer
) -> None:
    """Record the resubmit's coverage, headline and read plan, and restate the
    discovered-scope warning to count the members that plan serves, as the scope list and
    coverage do."""
    state.narrowed_coverage = filtered.coverage
    state.narrowed_headline = filtered.headline
    state.narrowed_plan = filtered.plan
    if state.opposite_labels is not None:
        restate_role_scope_warnings(
            state.warnings,
            mat_result,
            members=stated_scope_members(mat_result, filtered.plan),
            boundary_roles=state.place.roles.roles,
            label_of=state.opposite_labels,
        )


def _ground_answer_and_models(
    state: _PromoteState, mat_result: Materialised, roles: list[tuple[str, EntityRoleKey]]
) -> int:
    """Ground the answer, rebuild slot states from confirmed arcs, and assemble response models.

    Return the start time so the caller can time the full answer step, including scope and
    Spec assembly. The observation and probe entities are derived before it
    (`_derive_observation_entities`)."""
    t0 = now_ms()
    grounding = _ground_answer(
        state.shape_draft,
        state.bindings,
        mat_result.sv_dcids,
        roles,
        state.graph,
        state.narrowed_plan,
        unit_request=_converted_by(state),
        member_facts=mat_result.member_facts,
        open_axes=state.place.roles.open_axes,
    )
    (
        five_tuple_refs,
        role_refs,
        slots,
        slot_key_models,
        sv_refs,
        entity_data,
        source_refs,
        slot_options,
        _draft_inapplicable_slots,  # unused: draft-kind-derived, see below for why
        _ground_warnings,
        labels,
    ) = grounding
    state.slots = slots
    state.source_refs = source_refs
    state.labels = labels
    state.slot_options = slot_options
    state.ground_warnings = list(_ground_warnings)
    # Stamp grounding warnings with spec_id only on definite results.

    # Derive inapplicable and unresolved slot states from confirmed arcs. The wire's unbound
    # state represents both widening and prior inapplicability, so draft kinds are ambiguous.
    # Reuse arcs already read for the resolved members.
    state.inapplicable_slots = inapplicable_slots_from_arcs(
        slots=slots,
        sv_dcids=state.resolved_stat_var_dcids,
        arcs_by_sv=state.resolved_arcs,
    )
    # `unresolved_slots` uses the same confirmed arcs.
    state.unresolved_slots = unresolved_slots_from_arcs(
        slots=slots,
        sv_dcids=state.resolved_stat_var_dcids,
        arcs_by_sv=state.resolved_arcs,
    )

    # Use the admitted population, which can exceed the final winner count. The
    # flagged-unavailable carrier has no population record, so use its sole grounded member.
    state.shape_model = build_shape_model(
        state.shape_draft,
        slot_key_models,
        five_tuple_refs,
        member_count=(
            mat_result.population.admitted if mat_result.population is not None else len(sv_refs)
        ),
        members=mat_result.served_members,
    )
    state.stat_vars = build_stat_vars(
        sv_refs,
        state.shape_draft.shape_id,
        slots,
        plan=state.narrowed_plan,
        destination_confirmed=mat_result.destination_confirmed,
        arcs_by_sv=state.resolved_arcs,
    )
    state.entities = [_build_entity(rd, er, etr, role_refs) for rd, er, etr in entity_data]
    slots = state.slots = bind_when_slot(slots, window=state.narrowed_coverage.window)
    # state.source_dcids/facet_filter are already computed above, ahead of
    # grounding; only the actual source-slot rebind (labels read from the
    # graph) waits until slots exists.
    slots = state.slots = _rebind_source_slot(slots, state.source_dcids, state.graph)
    return t0


def _derive_observation_entities(state: _PromoteState, mat_result: Materialised) -> None:
    """Derive the observation and coverage-probe entities, before grounding, since the facet
    check, the conversion and the slot-options relaunch read them. Discovered members use
    destination by default, or origin when the shape declares only that role and the request
    has no provider ref."""
    place = state.place
    _scope_destination_dcids, _scope_origin_dcids = scope_axis_dcids(mat_result, place.probe_places)
    _obs_entities, _obs_entities_warn = _observation_entities(
        state.shape_draft,
        destination_dcids=_scope_destination_dcids,
        destination_dcid=place.roles.obs_destination_dcid,
        origin_dcids=_scope_origin_dcids or place.roles.origin_dcids,
    )
    state.obs_entities = _obs_entities
    state.obs_entities_warning = _obs_entities_warn
    # Coverage probes do not fan out across recipients; observations use every discovered
    # donor or recipient.
    state.probe_role_entities = coverage_probe_role_entities(
        state.shape_draft,
        mat_result,
        destination_dcids=place.probe_places,
        destination_dcid=place.roles.obs_destination_dcid,
        origin_dcids=place.roles.origin_dcids,
    )
    state.flat_probe_entities = _flat_obs_entities(state.probe_role_entities)


def _converted_by(state: _PromoteState) -> IdentityUnitRequest | None:
    """The unit request the preview converts by: the identity's request-form unit once
    units are elected per StatVar (`state.elected_unit`), as the preview dispatch reads it."""
    unit = (state.inp.identity.facets or {}).get("unit")
    if state.elected_unit is None or not isinstance(unit, IdentityFacetValueRequest):
        return None
    return unit.request


def _ground_and_build_spec(
    state: _PromoteState, mat_result: Materialised, roles: list[tuple[str, EntityRoleKey]]
) -> Spec:
    """Ground the answer, derive scope and axis facts, and build the Spec.

    Option-coverage probes wait until finalisation confirms a definite result."""
    t0 = _ground_answer_and_models(state, mat_result, roles)
    slots = state.slots
    inapplicable_slots = state.inapplicable_slots
    unresolved_slots = state.unresolved_slots
    shape_model = state.shape_model
    stat_vars = state.stat_vars
    entities = state.entities
    source_refs = state.source_refs
    slot_options = state.slot_options
    state.timing["answer"] = now_ms() - t0
    # Defer option-coverage probes until finalisation confirms a definite result.
    facet_options: list = []
    # Flagged-unavailable Specs use `available_at_place=False` and low confidence. Keep
    # confidence unset for no-place non-matches; entity-less definite answers use the
    # representative score.
    _available_at_place, _confidence = _place_confidence(
        destination_dcid=state.entity_dcid,
        available_at_place=mat_result.available_at_place,
        representative_score=state.shape_draft.representative_score,
        has_data=mat_result.has_data,
        retrieved=state.shape_draft.retrieved,
    )
    # A defaulted or discovered scope lists the members the filtered plan serves, the ones
    # coverage counts.
    _scope = None
    _scope_member_dcids = stated_scope_members(mat_result, state.narrowed_plan)
    if _scope_member_dcids is not None:
        _scope_labels, _scope_types = fetch_scope_members(_scope_member_dcids, graph=state.graph)
        _scope = build_scope(
            scope_axis(mat_result), _scope_member_dcids, _scope_labels, _scope_types
        )
    slots = state.slots = apply_slot_expansions(slots, slot_expansions(state.bindings))
    # Grounding starts with the measure family (`_start_related_read`).
    _measure_bound_values = bound_values_from_bindings(state.bindings)
    _served_members = _served_member_facts(state)
    # Open constraint axes come from the same final survivor facts.
    _open_axis_facts = open_constraint_axes(
        _measure_bound_values,
        dict(zip(state.resolved_stat_var_dcids, _served_members, strict=True)),
        value_labels=state.shape_draft.value_labels or {},
    )
    state.open_axis_facts = _open_axis_facts
    # Count the menus before the Spec takes them, so the finalisation probe and the
    # trailing events start from the counted menus.
    _conversion = counted_conversion(
        state.narrowed_plan, state.inp.identity.facets, catalogue=current_conversion_catalogue()
    )
    _served_filter = role_served_filter(
        state.shape_draft,
        state.probe_role_entities,
        open_axes=state.place.roles.open_axes,
        years=mat_result.preview_years,
        coverage=state.coverage.index,
        facets_pinned=pins_facets(state.inp.identity.facets),
        plan=state.narrowed_plan,
        conversion=_conversion,
    )
    _counted = counted_slot_options(
        state.slot_options,
        slots=slots,
        shape=state.shape_draft,
        bindings=state.bindings,
        plan=state.narrowed_plan,
        member_facts=dict(zip(state.resolved_stat_var_dcids, _served_members, strict=True)),
        served_filter=_served_filter,
        facets_pinned=pins_facets(state.inp.identity.facets),
        conversion=_conversion,
    )
    slot_options = state.slot_options = _counted.options
    state.option_axis_facts = _counted.facts
    state.served_filter = _served_filter
    state.served_members = tuple(_served_members)
    # Share role-axis facts between the Spec and entity-option builder.
    _role_axis_facts = role_axis_facts_for_materialised(
        mat_result, shape=state.shape_draft, plan=state.narrowed_plan
    )
    state.role_axis_facts = tuple(_role_axis_facts)
    state.bound_facet_axes = dict.fromkeys(
        bound_facet_axes(slots, state.inp.identity.facets), "bound"
    )
    state.unit_request_basis = _unit_request_basis_from_facets(state.inp.identity.facets)
    named_groups = resubmit_named_groups(state.inp, graph=state.graph, deadline=state.deadline)
    spec = state.spec = build_spec(
        shape=shape_model,
        slots=slots,
        stat_vars=stat_vars,
        entities=entities,
        coverage=state.narrowed_coverage,
        plan=state.narrowed_plan,
        headline=state.narrowed_headline,
        coverage_index=state.coverage.index,
        coverage_stamp=mat_result.coverage_stamp,
        scope=_scope,
        resolved_sources=source_refs,
        facets=answer_facets(
            state.narrowed_plan,
            state.labels,
            bound_axes=state.bound_facet_axes,
            request_basis=state.unit_request_basis,
            unit_request=_converted_by(state),
            currency_of=functools.partial(unit_currency, state.graph, deadline=state.deadline),
        ),
        slot_options=slot_options,
        facet_options=facet_options,
        available_at_place=_available_at_place,
        confidence=_confidence,
        requested_window=_requested_window_for_id(state.date_request),
        open_roles=mat_result.open_axes,
        inapplicable_slots=inapplicable_slots,
        unresolved_slots=unresolved_slots,
        # Same population record's confirmed half. See the member_count
        # comment on shape_model above. exact carries that record's own
        # exact flag, False when population was measured over a capped
        # recall sample.
        stat_vars_member_count=(
            mat_result.population.confirmed if mat_result.population is not None else None
        ),
        stat_vars_member_count_exact=(
            mat_result.population.member_count_exact if mat_result.population is not None else True
        ),
        facet_bindings=state.inp.identity.facets,
        # No text is re-parsed during resubmit, so posted and arc-derived bindings retain
        # binder provenance. Named-family elections supply measure-election provenance.
        measure_elected_properties=state.measure_elected_properties,
        measure_partition=state.measure_partition,
        measure_siblings=state.measure_siblings,
        open_axis_facts=_open_axis_facts,
        role_axis_facts=_role_axis_facts,
        named_groups=named_groups,
        measure_served_rows=measure_served_counter(_served_filter),
        walked=state.walked,
    )
    spec = with_unit_conversion(
        spec,
        members=_served_members,
        plans=(state.narrowed_plan,) if state.narrowed_plan is not None else (),
    )
    spec = state.spec = apply_declaration_version(spec, state.bindings)
    # The resubmit states the `family_total_absent` fact its member read states. The slots
    # its posted identity leaves open count as named, so every signature of the family it
    # admits lies inside them and no lone breakdown is stated (`lone_breakdown_absent`).
    if (family_total_absent := mat_result.family_total_absent) is not None:
        state.ground_warnings.append(
            state.depart(
                *resubmit_family_total_absent_departure(
                    family_total_absent,
                    spec=spec,
                    label_of=lambda dcids: _graph_labels(state, dcids),
                )
            )
        )
    return spec


def _no_data_gate_holds(
    state: _PromoteState, mat_result: Materialised, required: tuple[ResolvedQualifier, ...]
) -> bool:
    """Whether a no-data gate answers the resubmit with a flagged Spec (`_ground_build_gates`),
    `required` being the terms its refusal on the posted values names
    (`requirements.posted_refusal_terms`)."""
    return (
        bool(required)
        or mat_result.available_at_place is False
        or (state.entity_dcid is None and not mat_result.has_data)
    )


def _ground_build_gates(
    state: _PromoteState,
    mat_result: Materialised,
    spec: Spec,
    *,
    required: tuple[ResolvedQualifier, ...],
    facet_checked: bool,
) -> RegionResult | None:
    """Apply no-data gates, then check bound facet availability.

    Return `qualifier_not_bound` with a flagged Spec when the answer leaves a posted value
    unapplied on a property the shape holds (`required`, `requirements.settle_posted`),
    naming each such value once, as a term it could not bind, in place of its departure. Return
    `no_observations` with a flagged Spec when no asserting member has data at the
    place. Return `no_matching_slice` with a flagged Spec when no place is bound and no member
    asserts every bound property. Both omit observations, after a bound source pin is checked
    against globally admissible role candidates. Otherwise raise any facet refusal
    (`_raise_facet_refusal`), unless it ran before the conversion (`facet_checked`), or
    return `None` for finalisation."""
    if state.source_dcids and is_role_bearing(state.shape_draft) and not facet_checked:
        _raise_facet_refusal(state, mat_result)
        facet_checked = True
    settle_posted(state)
    if required:
        return _flagged_refusal(state, spec, "qualifier_not_bound", unbound_qualifiers=required)
    if mat_result.available_at_place is False:
        return _flagged_refusal(state, spec, "no_observations", gate=NO_MEMBER_HAS_DATA_AT_PLACE)
    if state.entity_dcid is None and not mat_result.has_data:
        return _flagged_refusal(
            state, spec, "no_matching_slice", gate=NO_MEMBER_ASSERTS_BOUND_PROPERTIES
        )
    if not facet_checked:
        _raise_facet_refusal(state, mat_result)
    return None


def _flagged_refusal(
    state: _PromoteState,
    spec: Spec,
    reason: NoDataReason,
    *,
    gate: NoDataGate | None = None,
    unbound_qualifiers: tuple[ResolvedQualifier, ...] = (),
) -> RegionResult:
    """Refuse `reason` with `spec` as the flagged match. The part holds `spec`, so it states
    `whole_total_available` for it (`whole_totals.whole_total_warnings`), as a fresh
    request's refusal does for its flagged match (`ResolveState.no_data`)."""
    return _no_data_region(
        state.variable_text,
        reason,
        warnings=(
            *state.refusal_warnings(),
            *whole_total_warnings(spec, coverage=state.coverage.index),
        ),
        flagged_match=(spec,),
        gate=gate,
        elected_stat_vars=_elected_stat_vars(state.mat_result, state.shape_draft),
        unbound_qualifiers=unbound_qualifiers,
    )


def _role_candidate_facets(
    state: _PromoteState, mat_result: Materialised
) -> tuple[dict[str, list[Facet]], dict[str, list[Facet]]]:
    """Return the windowed and lifetime role candidates for each materialised member."""
    datasets = elected_datasets(mat_result.member_facts)
    years = mat_result.preview_years
    bounds = (years.start_year, years.end_year) if isinstance(years, TimeWindow) else None
    windowed: dict[str, list[Facet]] = {}
    lifetime: dict[str, list[Facet]] = {}
    for sv in mat_result.sv_dcids:
        candidates = role_bearing_facet_candidates(
            stat_var=sv,
            role_bindings=state.probe_role_entities,
            coverage=state.coverage.index,
        )
        if candidates is None:
            continue
        lifetime[sv] = published(datasets, sv, candidates)
        windowed[sv] = (
            lifetime[sv]
            if bounds is None
            else published(
                datasets,
                sv,
                role_bearing_facet_candidates(
                    stat_var=sv,
                    role_bindings=state.probe_role_entities,
                    coverage=state.coverage.index,
                    first_year=bounds[0],
                    last_year=bounds[1],
                )
                or [],
            )
        )
    return windowed, lifetime


def _raise_facet_refusal(state: _PromoteState, mat_result: Materialised) -> None:
    """Raise the refusal of a bound facet the resubmit's members do not hold.

    A source pin calls this before the scoped no-data gate. Role-bearing method and unit
    pins use globally admitted candidates in the requested window; provenance uses their
    lifetime candidates. The subject path reads entity-keyed facets unwindowed.
    """
    # Import locally to avoid a cycle during region package initialisation.
    from search.engine.facet_refusals import (
        candidate_facet_refusal,
        facet_refusal,
    )

    if is_role_bearing(state.shape_draft):
        candidate_facets_by_sv, lifetime_facets_by_sv = _role_candidate_facets(state, mat_result)
        _facet_refusal = candidate_facet_refusal(
            state.graph,
            candidate_facets_by_sv,
            state.inp.identity.facets,
            provenance_facets_by_sv=lifetime_facets_by_sv,
            elected_units=state.elected_units,
            deadline=state.deadline,
        )
    else:
        _facet_refusal = facet_refusal(
            state.graph,
            list(mat_result.sv_dcids),
            state.flat_probe_entities,
            state.inp.identity.facets,
            elected_units=state.elected_units,
            datasets=elected_datasets(mat_result.member_facts),
            deadline=state.deadline,
        )
    if _facet_refusal is not None:
        raise _facet_refusal


def _phase_ground_build(state: _PromoteState) -> RegionResult | None:
    """Decide unit convertibility, narrow facets, build the Spec, and apply gates.

    Return `no_observations` for a window holding no point before any Spec is grounded, and
    for `no_observations` at the place or `no_matching_slice` after; facet and convertibility
    failures raise. A clean pass leaves option probing and observation previews to finalisation."""
    mat_result = cast(Materialised, state.mat_result)
    _start_related_read(state)

    _decide_unit_convertibility(state, mat_result)

    _narrow_ground_facets(state, mat_result)
    mat_result = _over_conversion_period(state, mat_result)
    required = posted_refusal_terms(state)

    # Data at the place in other years only: `no_observations` for the window, with the
    # `year_unavailable` fact the carrier's gap gives, naming the years read and the span the
    # series runs. The answer carries no Spec, so it returns before one is grounded. The
    # answer reads the member held, so a member that is not its family's total warns as an answer
    # showing it does. A dropped posted value ranks first, and its refusal flags the grounded
    # Spec (`_ground_build_gates`).
    if (gap := mat_result.window_gap) is not None and not required:
        family = mat_result.family_total_absent
        return _no_data_region(
            state.variable_text,
            "no_observations",
            warnings=(
                *state.warnings,
                *(
                    [
                        state.depart(
                            *resubmit_family_total_absent_departure(
                                family,
                                spec=None,
                                label_of=lambda dcids: _graph_labels(state, dcids),
                            )
                        )
                    ]
                    if family is not None
                    else []
                ),
            ),
            elected_stat_vars=_elected_stat_vars(state.mat_result, state.shape_draft),
            window_gap_fact=window_gap_fact(gap),
        )

    _derive_observation_entities(state, mat_result)
    # The conversion runs only where finalisation's bound-unit gate reads it, and after the
    # facet check, which follows both no-data gates; the gates then skip that check.
    facet_checked = _may_convert(state, mat_result, required)
    if facet_checked:
        _raise_facet_refusal(state, mat_result)
        if _unit_gate_passes(state, mat_result):
            _convert_narrowed_answer(state, mat_result)

    spec = _ground_and_build_spec(state, mat_result, state.place.roles.roles)
    return _ground_build_gates(
        state, mat_result, spec, required=required, facet_checked=facet_checked
    )


def _served_member_facts(state: _PromoteState) -> list[ArcsMemberFacts]:
    """The final survivors' facts. Flagged-unavailable carriers omit member_facts, while
    resolved_arcs is populated on every path."""
    return [
        ArcsMemberFacts(arcs=state.resolved_arcs[dcid]) for dcid in state.resolved_stat_var_dcids
    ]


def _start_related_read(state: _PromoteState) -> None:
    """Read the survivors' measure partition and siblings, then start their related-measures
    reads on a thread of their own (`related_read.submit_related_read`), which run beside
    grounding and finalisation."""
    role_keys = entity_role_keys(
        state.place.roles.resolved_entities,
        open_axes=state.place.roles.open_axes,
    )
    state.measure_partition, state.measure_siblings = observed_measure_family(
        state.shape_draft,
        get_index(),
        get_measure_index().axes,
        shape_query(
            state.shape_draft,
            state.bindings,
            role_keys.bound,
            open_role_keys=role_keys.open,
        ),
        _served_member_facts(state),
    )
    state.related_read = submit_related_read(
        stat_vars=state.resolved_stat_var_dcids,
        measure_partition=state.measure_partition,
        places=state.place.roles.roles,
        arcs=state.resolved_arcs,
        window=_requested_window_for_id(state.date_request),
        graph=state.graph,
        request_deadline=state.deadline,
    )


def _may_convert(
    state: _PromoteState, mat_result: Materialised, required: tuple[ResolvedQualifier, ...]
) -> bool:
    """Whether finalisation reaches its bound-unit gate: no no-data gate holds
    (`_no_data_gate_holds`, `required` the terms a refusal on the posted values names), the unit is
    request-form, and the narrowed plan serves a point."""
    plan = state.narrowed_plan
    return not (
        _no_data_gate_holds(state, mat_result, required)
        or not isinstance((state.inp.identity.facets or {}).get("unit"), IdentityFacetValueRequest)
        or plan is None
        or serves_nothing(plan)
    )


def _unit_gate_passes(state: _PromoteState, mat_result: Materialised) -> bool:
    """Whether the bound-unit gate passes before the conversion runs: the narrowed plan
    serves every StatVar under its elected unit (`convertibility.unit_gate`). The plan the
    gate read is recorded for finalisation's gate (`_PromoteState.unit_gate_plan`)."""
    gate = unit_gate(
        mat_result,
        state.narrowed_plan,
        state.elected_unit,
        state.obs_entities,
        graph=state.materialise_graph,
        deadline=state.deadline,
    )
    state.unit_gate_plan = gate.plan
    return gate.reason is None


def _convert_narrowed_answer(state: _PromoteState, mat_result: Materialised) -> None:
    """Run the elected conversion over the narrowed plan (`_unit_gate_passes`), so
    coverage, the headline and the plan leave out the rows it omits before grounding reads
    them (`convertible_answer`), as a fresh request's do; finalisation's bound-unit gate
    reports its outcome (`_PromoteState.unit_coverage`)."""
    plan = state.narrowed_plan
    assert plan is not None  # `_may_convert`
    filtered = FilteredAnswer(state.narrowed_coverage, state.narrowed_headline, plan)
    converted, state.unit_coverage = convertible_answer(
        mat_result,
        filtered,
        state.elected_unit,
        graph=state.materialise_graph,
        sdmx=state.materialise_sdmx,
        rate_cache=state.rate_cache,
        dated_facets=state.graph,
        deadline=state.deadline,
        date_request=state.date_request,
        coverage_index=state.coverage.index,
    )
    if converted is not filtered:
        state.unit_gate_plan = converted.plan
        _install_narrowed_answer(state, mat_result, converted)
