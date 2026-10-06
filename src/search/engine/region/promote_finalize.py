"""Finalise the definite result, including option relaunches and observations."""

from __future__ import annotations

from typing import cast

from materialise.recipes import RateCoverageGapError, cause_from_recipe_error
from search.engine.departures import window_gap_fact
from search.engine.envelope import now_ms
from search.engine.region.bound_unit_gate import bound_unit_gate_reason
from search.engine.region.convertibility import option_pins
from search.engine.region.facet_probe import FacetProbe, facet_probe
from search.engine.region.observations.bulk_scope import bulk_scope_for
from search.engine.region.observations.dispatch import _attach_observations
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.option_coverage import (
    _build_and_enrich_entity_options,
    _maybe_build_facet_options,
    _maybe_enrich_slot_options,
)
from search.engine.region.promote_ground import unit_refusal_input_error
from search.engine.region.promote_relaunch import PromoteRelaunch, bind_relaunches
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.related_read import joined_related, stating_related
from search.engine.region.result import RegionResult, answered_roles
from search.engine.region.state import EngineOptions, _elected_stat_vars
from search.engine.region.un_system import member_labels
from search.engine.retrieval.elected_dataset import elected_datasets
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import plan_window_gap, serves_nothing
from search.engine.retrieval.retrieve import Materialised
from search.models import (
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    Spec,
    SpecObservations,
    Warning,
)


def _promote_facet_probe(state: _PromoteState, mat_result: Materialised) -> FacetProbe:
    """What the resubmitted spec's facet options are judged over: its posted facets,
    admitted per StatVar by the elected units."""
    return facet_probe(
        mat_result.sv_dcids,
        state.probe_role_entities,
        years=mat_result.preview_years,
        pins=option_pins(state.inp.identity.facets, state.elected_unit),
        datasets=elected_datasets(mat_result.member_facts),
    )


def _request_form_unit(state: _PromoteState) -> IdentityUnitRequest | None:
    """The posted identity's request-form unit, None for a ref-form or absent unit."""
    _unit_binding = (state.inp.identity.facets or {}).get("unit")
    return _unit_binding.request if isinstance(_unit_binding, IdentityFacetValueRequest) else None


def _fetch_finalize_observations(
    state: _PromoteState,
    mat_result: Materialised,
    *,
    unit_request: IdentityUnitRequest | None,
) -> tuple[SpecObservations, Warning | None] | None:
    """Fetch the preview when streamed or included observations are requested.

    Return the fetch with its own warning, or None when no preview is requested."""
    state.fetched_obs = None
    if not (state.stream_observations or state.include_observations):
        return None
    # Fetch facets for the immediate response; the relaunch sends streamed points. Read
    # declared roles from resolved arcs because flagged carriers omit `member_facts`.
    _obs_member_facts = {
        dcid: ArcsMemberFacts(arcs=arcs) for dcid, arcs in state.resolved_arcs.items()
    }
    _fetch = _build_observations(
        state.narrowed_plan,
        list(mat_result.sv_dcids),
        state.obs_entities,
        graph=state.graph,
        materialise_graph=state.materialise_graph,
        materialise_sdmx=state.materialise_sdmx,
        facet_filter=state.facet_filter,
        scope=bulk_scope_for(state.place, mat_result, sdmx=state.sdmx),
        sdmx=state.sdmx,
        deadline=state.deadline,
        rate_cache=state.rate_cache,
        unit_request=unit_request,
        elected=state.elected_unit,
        member_facts=_obs_member_facts,
        coverage=state.coverage,
    )
    state.fetched_obs = _fetch[0]
    return _fetch


def _bound_unit_refusal(
    state: _PromoteState,
    mat_result: Materialised,
) -> RegionResult | None:
    """The refusal a fresh request returns at the bound-unit gate for the same narrowed plan,
    or None when the gate passes. The refusal holds no Spec, so it carries the warnings a
    fresh refusal does (`_PromoteState.refusal_warnings`). A served row the elected
    conversion cannot convert (`convertibility.unit_coverage`) answers
    `rate_coverage_gap` with its cause, as a fresh request does; a recipe or unit refusal
    there raises the input error the resubmit's unit decision raises
    (`unit_refusal_input_error`). The conversion ran once over the narrowed plan
    (`_PromoteState.unit_coverage`). A pass whose conversion omits the rows of entities no
    currency names adds `currency_unmapped_omitted` to the state's warnings, as a fresh
    request does."""
    gate_plan = state.unit_gate_plan if state.unit_gate_plan is not None else state.narrowed_plan
    reason = bound_unit_gate_reason(gate_plan, state.obs_entities, mat_result.sv_dcids)
    if reason is not None:
        return _no_data_region(
            state.variable_text,
            reason,
            warnings=state.refusal_warnings(),
            elected_stat_vars=_elected_stat_vars(state.mat_result, state.shape_draft),
        )
    coverage = state.unit_coverage
    if coverage.refusal is None:
        if (omitted := coverage.omitted_warning()) is not None:
            state.warnings.append(omitted)
        return None
    exc = coverage.refusal[1]
    if not isinstance(exc, RateCoverageGapError):
        raise unit_refusal_input_error(exc) from exc
    return _no_data_region(
        state.variable_text,
        "rate_coverage_gap",
        warnings=state.refusal_warnings(),
        elected_stat_vars=_elected_stat_vars(state.mat_result, state.shape_draft),
        cause=cause_from_recipe_error(exc),
    )


def _enrich_finalize_options(
    state: _PromoteState,
    mat_result: Materialised,
    spec: Spec,
    resubmit_warnings: tuple[Warning, ...],
) -> tuple[Spec, tuple[Warning, ...]]:
    """Probe option coverage and build facet options only after both phases confirm a
    definite result. A streamed request with option coverage reads facet options in its
    trailing probe instead.

    Always record an `options` timing entry. Entity options return raw counts without coverage
    probes when labels are disabled."""
    _opts_t0 = now_ms()
    if state.include_option_coverage:
        _slot_options, _coverage_warn = _maybe_enrich_slot_options(
            state.shape_draft,
            state.bindings,
            state.slot_options,
            entities=state.place.roles.resolved_entities,
            open_axes=state.place.roles.open_axes,
            date_request=state.date_request,
            materialise_graph=state.materialise_graph,
            place_dcids=state.flat_probe_entities,
            include_option_coverage=True,
            facets_pinned=bool(state.inp.identity.facets),
        )
        if _coverage_warn is not None:
            resubmit_warnings = (
                *resubmit_warnings,
                _coverage_warn.model_copy(update={"spec_id": spec.spec_id}),
            )
        spec = state.spec = spec.model_copy(
            update={"resolved": spec.resolved.model_copy(update={"slot_options": _slot_options})}
        )
    if not state.stream_slot_options:
        _facet_options, _facet_coverage_warn = _maybe_build_facet_options(
            _promote_facet_probe(state, mat_result),
            graph=state.graph,
            coverage=state.coverage.index,
            deadline=state.deadline,
        )
        if _facet_coverage_warn is not None:
            resubmit_warnings = (
                *resubmit_warnings,
                _facet_coverage_warn.model_copy(update={"spec_id": spec.spec_id}),
            )
        spec = state.spec = spec.model_copy(
            update={"resolved": spec.resolved.model_copy(update={"facet_options": _facet_options})}
        )
    if state.role_axis_facts:
        # Reusing role-axis facts from grounding avoids a materialise read.
        _entity_options_raw, _entity_options, _entity_warn = _build_and_enrich_entity_options(
            state.shape_draft,
            state.place.roles.resolved_entities,
            state.entities,
            list(mat_result.sv_dcids),
            date_request=state.date_request,
            graph=state.graph,
            sdmx=state.sdmx,
            include_option_coverage=state.include_option_coverage,
            role_axis_facts=state.role_axis_facts,
            place_dcids=state.flat_probe_entities,
        )
        state.entity_options_raw = _entity_options_raw
        if _entity_warn is not None:
            resubmit_warnings = (
                *resubmit_warnings,
                _entity_warn.model_copy(update={"spec_id": spec.spec_id}),
            )
        spec = state.spec = spec.model_copy(
            update={
                "resolved": spec.resolved.model_copy(update={"entity_options": _entity_options})
            }
        )
    # Timing covers inline enrichment and excludes the trailing SSE probe.
    state.timing["options"] = now_ms() - _opts_t0
    return spec, resubmit_warnings


def _attach_finalize_observations(
    state: _PromoteState,
    spec: Spec,
    fetched_obs: SpecObservations | None,
    resubmit_warnings: tuple[Warning, ...],
) -> tuple[Spec, tuple[Warning, ...]]:
    """Attach the requested preview. Streamed points arrive in the trailing event.

    Return the updated Spec and warnings."""
    if fetched_obs is not None and (state.stream_observations or state.include_observations):
        # Fetch warnings precede refusal; streamed points arrive in the trailing event.
        _resubmit_warnings_list = list(resubmit_warnings)
        spec = state.spec = _attach_observations(
            spec,
            (fetched_obs, None),
            _resubmit_warnings_list,
            include_points=state.include_observations and not state.stream_observations,
        )
        resubmit_warnings = tuple(_resubmit_warnings_list)
    return spec, resubmit_warnings


def _promote_relaunch(
    state: _PromoteState,
    mat_result: Materialised,
    spec: Spec,
    unit_request: IdentityUnitRequest | None,
) -> PromoteRelaunch:
    """The facts the definite answer's trailing probes read, apart from the request's clients.
    Built after options are enriched, since the slot-options probe carries the raw entity
    options finalisation fills."""
    bulk = bulk_scope_for(state.place, mat_result, sdmx=state.sdmx)
    return PromoteRelaunch(
        shape_draft=state.shape_draft,
        bindings=state.bindings,
        entities=state.place.roles.resolved_entities,
        open_axes=state.place.roles.open_axes,
        date_request=state.date_request,
        place_dcids=state.flat_probe_entities,
        entity_options=state.entity_options_raw,
        facet_probe=_promote_facet_probe(state, mat_result),
        facets_pinned=bool(state.inp.identity.facets),
        identity=spec.definition.identity,
        sv_dcids=list(mat_result.sv_dcids),
        obs_entities=state.obs_entities,
        facet_filter=state.facet_filter,
        bulk_places=bulk.places if bulk is not None else None,
        unit_request=unit_request,
        elected=state.elected_unit,
        plan=state.narrowed_plan,
        variable_labels=member_labels((spec,)),
    )


def _phase_finalize(state: _PromoteState) -> RegionResult:
    """Build relaunches, resolve the preview and option coverage, and return the definite result.

    A narrowed plan that serves nothing returns `no_observations` first, with the
    `year_unavailable` fact a range's gap in the plan's candidates gives
    (`read_plan.plan_window_gap`)."""
    mat_result = cast(Materialised, state.mat_result)
    spec = state.spec

    # A resubmit whose narrowed plan serves no point in its years, under its window or its
    # facet pins, answers `no_observations` (`read_plan.serves_nothing`) before any preview
    # is fetched. Under a range its `year_unavailable` fact names the years asked and the span
    # the series the plan holds runs, as an unpinned window answer's does.
    if state.narrowed_plan is not None and serves_nothing(state.narrowed_plan):
        return _no_data_region(
            state.variable_text,
            "no_observations",
            warnings=state.refusal_warnings(),
            elected_stat_vars=_elected_stat_vars(state.mat_result, state.shape_draft),
            window_gap_fact=window_gap_fact(
                plan_window_gap(state.narrowed_plan, state.date_request)
            ),
        )

    # A request-form unit passes the fresh request's bound-unit gate, decided from the
    # narrowed plan, before any preview is fetched.
    _unit_request = _request_form_unit(state)
    if _unit_request is not None and (
        (refusal := _bound_unit_refusal(state, mat_result)) is not None
    ):
        return refusal
    _fetch = _fetch_finalize_observations(state, mat_result, unit_request=_unit_request)
    # A requested preview's warning joins the answer, stamped with its Spec.
    resubmit_warnings = state.definite_warnings(spec.spec_id)
    _fetched_obs = None
    if _fetch is not None:
        _fetched_obs, _fetch_warn = _fetch
        if _fetch_warn is not None:
            resubmit_warnings = (
                *resubmit_warnings,
                _fetch_warn.model_copy(update={"spec_id": spec.spec_id}),
            )

    spec, resubmit_warnings = _enrich_finalize_options(state, mat_result, spec, resubmit_warnings)
    spec, resubmit_warnings = _attach_finalize_observations(
        state, spec, _fetched_obs, resubmit_warnings
    )
    spec = stating_related(spec, joined_related(state.related_read))

    region = RegionResult(
        variable_text=state.variable_text,
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=resubmit_warnings,
        timing_by_step=dict(state.timing),
        facet_probe=_promote_facet_probe(state, mat_result),
        promote_relaunch=_promote_relaunch(state, mat_result, spec, _unit_request),
        observation_entities=state.obs_entities,
        facet_filter=state.facet_filter,
        measure_partition=state.measure_partition,
        measure_siblings=state.measure_siblings,
        open_axis_facts=state.open_axis_facts,
        option_axis_facts=state.option_axis_facts,
        served_filter=state.served_filter,
        served_members=state.served_members,
        served_plans=(state.narrowed_plan,) if state.narrowed_plan is not None else (),
        role_axis_facts=state.role_axis_facts,
        member_population=mat_result.population,
        answered_roles=answered_roles(state.shape_draft),
    )
    return bind_relaunches(
        region,
        graph=state.graph,
        sdmx=state.sdmx,
        coverage=state.coverage,
        options=EngineOptions(
            include_option_coverage=state.include_option_coverage,
            stream_slot_options=state.stream_slot_options,
            include_observations=state.include_observations,
            stream_observations=state.stream_observations,
            materialise_graph=state.materialise_graph,
            materialise_sdmx=state.materialise_sdmx,
            rate_cache=state.rate_cache,
            population_memo=state.population_memo,
            place_options_pool=state.place_options_pool,
            deadline=state.deadline,
        ),
    )
