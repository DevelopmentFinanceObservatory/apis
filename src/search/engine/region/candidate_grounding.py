"""Ground each admitted candidate into a Spec and retain its collapse metadata.

Candidate order is preserved. Metadata is keyed by ``spec_id`` for later probes and folds.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import cast

from code_index.shared import get_index, get_measure_index
from dfo_platform.executor import submit_with_context
from materialise.errors import DatasetDisagreementError, MaterialisationError
from materialise.facets import FacetSelection
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.envelope import now_ms
from search.engine.errors import EngineInfraError
from search.engine.region.arcs import (
    inapplicable_slots_from_member_facts,
    unresolved_slots_from_member_facts,
)
from search.engine.region.candidate_unit_election import (
    _candidate_unit_elections,
    _CandidateUnitElectionMemo,
)
from search.engine.region.candidates import _Candidate
from search.engine.region.codes import MEMBERSHIP_UNVERIFIED
from search.engine.region.convertibility import (
    UnitCoverage,
    convertible_answer,
    elected_unit_refs,
    survivor_declared_roles,
    unit_gate,
)
from search.engine.region.facets import (
    FacetAdmission,
    FilteredAnswer,
    answer_facets,
    cluster_currency_binding,
    cluster_currency_filter,
    filter_answer,
    narrow_by_elected_unit,
    narrow_facets_by_sv_to_filter,
)
from search.engine.region.grounding import (
    _build_entity,
    _ground_answer,
    _place_confidence,
    build_identity_entities,
    carrier_flag_warnings,
    check_breakdown_settled,
    requested_window_from_date_request,
    restate_role_scope_warnings,
)
from search.engine.region.observations.bulk_scope import BulkScope, bulk_scope_for
from search.engine.region.observations.entities import (
    _flat_obs_entities,
    _observation_entities,
    coverage_probe_role_entities,
    scope_axis_dcids,
)
from search.engine.region.option_coverage import (
    RoleServedFilter,
    counted_slot_options,
    measure_served_counter,
    role_axis_facts_for_materialised,
    role_served_filter,
)
from search.engine.region.place_context import PlaceContext
from search.engine.region.place_roles import request_named_groups
from search.engine.region.requirements import answer_terms
from search.engine.region.result import FacetAxisFilter, answered_roles
from search.engine.region.scope import (
    build_scope,
    fetch_scope_members,
    scope_axis,
    stated_scope_members,
)
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    MeasureFamilyCacheKey,
    MeasurePartition,
    OpenAxisFact,
    apply_declaration_version,
    apply_slot_expansions,
    entity_role_keys,
    observed_measure_family,
    served_open_axes,
    shape_query,
    slot_expansions,
)
from search.engine.retrieval.elected_dataset import elected_datasets
from search.engine.retrieval.read_plan import ReadPlan, plan_probes
from search.engine.retrieval.retrieve import Materialised, MemberPopulation
from search.engine.shape import ShapeDraft
from search.engine.spec_build import (
    RoleAxisFactInput,
    bind_when_slot,
    build_shape_model,
    build_spec,
    build_stat_vars,
)
from search.engine.unit.conversion import bound_prices
from search.engine.unit.refs import unit_currency
from search.models import (
    Coverage,
    Entity,
    EntityOptions,
    EntityRoleKey,
    Headline,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ProvenanceRef,
    ResolvedFacet,
    Shape,
    Slot,
    SlotOptions,
    Spec,
    StatVar,
    Warning,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CandidatePrep:
    """Shared per-candidate unit elections, filtered answers and scope-member lists, for
    the candidates that survive their unit elections (`candidates`), in input order."""

    candidates: list[_Candidate]

    elected_raw: list[Mapping[str, tuple[Mapping[str, FacetSelection], ...]]]
    # Each candidate's definite unit refusals by StatVar (`CandidateUnitElections.refusals`).
    unit_refusals: list[Mapping[str, MaterialisationError]]
    elected_units: list[Mapping[str, frozenset[str]]]
    # Each candidate's StatVar facets in its elected units, for a request-form unit.
    facets_in_elected_units: list[dict[str, list[Facet]]] | None
    # Each candidate under the request's facet filter (`filter_answer`), leaving out the rows
    # its conversion omits (`convertible_answer`).
    filtered: list[FilteredAnswer]
    # What each candidate's elected conversion does to its rows, which the collapse's
    # bound-unit filter reports, and the plan that filter's gate reads
    # (`convertibility.unit_gate`).
    unit_coverage: list[UnitCoverage]
    gate_plans: list[ReadPlan | None]
    scope_members: list[list[str] | None]
    scope_labels: dict[str, str]
    scope_types: dict[str, str]
    # Each candidate's unit filter as a twin cluster member (`_twin_unit_filters`), else None.
    twin_unit_filters: list[Mapping[str, FacetAxisFilter] | None]


def _twin_unit_filters(
    state: ResolveState, candidates: Sequence[_Candidate]
) -> list[Mapping[str, FacetAxisFilter] | None]:
    """Each candidate's unit filter as a member of a twin cluster naming a currency
    (`facets.cluster_currency_filter`), whatever else is on the menu, so a twin answering
    for an elected shape with no data serves the cluster's currency. Every candidate takes
    None when the request binds a unit, which then decides the facets."""
    if state.unit_request is not None or state.unit_facet_filter is not None:
        return [None] * len(candidates)
    return [cluster_currency_filter(cand.mat) for cand in candidates]


def _prepare_candidate_context(state: ResolveState, candidates: list[_Candidate]) -> _CandidatePrep:
    """Compute candidate-specific unit elections and scope members once.

    Request-form units require graph and SDMX clients or raise ``EngineInfraError``.
    Ref-form and unbound units use the literal filter. Coverage, the headline and the
    preview's read plan take that filter once per candidate, before any election
    (`filter_answer`). Under a bound unit each candidate's conversion then runs over its
    filtered plan (`_candidate_conversions`), and all three leave out the rows it omits.
    Each scope lists the members its candidate's filtered plan serves, then one label/type
    batch covers their union.

    An alternate whose unit election raises `DatasetDisagreementError` is dropped with a
    warning, as an alternate whose mint raises it is; the elected shape's propagates.
    """
    if state.unit_request is not None:
        if state.materialise_graph is None or state.materialise_sdmx is None:
            raise EngineInfraError(
                "materialise graph/sdmx client not initialised; cannot decide "
                "this bound unit request's own convertibility for the "
                "also_plausible candidates fan-out"
            )
        # Each election uses its candidate's probe places, a discovered scope's members once
        # settled, and its role bindings.
        _cand_role_bindings: list[dict[str, list[str]]] = []
        for cand in candidates:
            _dest_dcids, _origin_dcids = scope_axis_dcids(cand.mat, cand.place.probe_places)
            _role_bindings, _ = _observation_entities(
                cand.shape,
                destination_dcids=_dest_dcids,
                destination_dcid=cand.place.roles.obs_destination_dcid,
                origin_dcids=_origin_dcids or cand.place.roles.origin_dcids,
            )
            _cand_role_bindings.append(_role_bindings)
        _unit_election_memo: _CandidateUnitElectionMemo = {}
        # Share the request's coverage snapshot across every candidate election.
        _coverage_index = state.coverage.index
        _kept: list[_Candidate] = []
        _cand_elections = []
        for cand_index, (cand, _role_bindings) in enumerate(
            zip(candidates, _cand_role_bindings, strict=True)
        ):
            try:
                _election = _candidate_unit_elections(
                    cand.mat.sv_dcids,
                    unit_request=state.unit_request,
                    role_bindings=_role_bindings,
                    declared_roles=survivor_declared_roles(cand.mat),
                    probes=plan_probes(cand.mat.read_plan),
                    years=cand.mat.preview_years,
                    graph=state.materialise_graph,
                    sdmx=state.materialise_sdmx,
                    coverage=_coverage_index,
                    memo=_unit_election_memo,
                    deadline=state.deadline,
                    datasets=elected_datasets(cand.mat.member_facts),
                )
            except DatasetDisagreementError:
                if cand_index == 0 and state.elected_has_data:
                    raise
                # The mint refuses this alternate's identity, so it is dropped as an
                # alternate whose walk the mint refuses is (`candidates._materialise_alt_shapes`);
                # the elected answer stands.
                logger.warning(
                    "alternate %r dropped: a member's dataset disagrees with the code index",
                    cand.shape.shape_id,
                    exc_info=True,
                )
                continue
            _kept.append(cand)
            _cand_elections.append(_election)
        candidates = _kept
        _cand_elected_raw: list[Mapping[str, tuple[Mapping[str, FacetSelection], ...]]] = [
            e.elected for e in _cand_elections
        ]
        _cand_unit_refusals: list[Mapping[str, MaterialisationError]] = [
            e.refusals for e in _cand_elections
        ]
        _cand_elected_units: list[Mapping[str, frozenset[str]]] = [
            elected_unit_refs(e) for e in _cand_elected_raw
        ]
        _cand_facets_in_elected_units: list[dict[str, list[Facet]]] | None = [
            narrow_by_elected_unit(cand.mat.facets_by_sv or {}, elected)
            for cand, elected in zip(candidates, _cand_elected_units, strict=True)
        ]
    else:
        _cand_elected_raw: list[Mapping[str, tuple[Mapping[str, FacetSelection], ...]]] = [
            {} for _ in candidates
        ]
        _cand_unit_refusals: list[Mapping[str, MaterialisationError]] = [{} for _ in candidates]
        _cand_elected_units: list[Mapping[str, frozenset[str]]] = [{} for _ in candidates]
        _cand_facets_in_elected_units = None

    # The candidate's per-StatVar unit elections for a request-form unit, else the literal
    # unit filter, else its twin's currency.
    _cand_twin_unit_filters = _twin_unit_filters(state, candidates)
    _cand_filtered: list[FilteredAnswer] = []
    for cand, elected, twin_filter in zip(
        candidates, _cand_elected_units, _cand_twin_unit_filters, strict=True
    ):
        if state.unit_request is not None:
            _admission = FacetAdmission(elected_units=elected)
        else:
            _admission = FacetAdmission(axes=twin_filter or state.unit_facet_filter)
        _cand_filtered.append(
            filter_answer(
                cand.mat,
                _admission,
                date_request=state.date_request,
                coverage_index=state.coverage.index,
            )
        )

    _cand_unit_coverage = [UnitCoverage() for _ in candidates]
    _cand_gate_plans = [filtered.plan for filtered in _cand_filtered]
    if state.bound_unit_outcome is not None:
        converted = _candidate_conversions(state, candidates, _cand_filtered, _cand_elected_raw)
        _cand_filtered = [c.answer for c in converted]
        _cand_unit_coverage = [c.coverage for c in converted]
        _cand_gate_plans = [c.gate_plan for c in converted]

    _cand_scope_members = [
        stated_scope_members(cand.mat, filtered.plan)
        for cand, filtered in zip(candidates, _cand_filtered, strict=True)
    ]
    _cand_scope_universe = sorted({d for members in _cand_scope_members for d in (members or ())})
    _cand_scope_labels, _cand_scope_types = fetch_scope_members(
        _cand_scope_universe, graph=state.graph
    )

    return _CandidatePrep(
        candidates=candidates,
        elected_raw=_cand_elected_raw,
        unit_refusals=_cand_unit_refusals,
        elected_units=_cand_elected_units,
        facets_in_elected_units=_cand_facets_in_elected_units,
        filtered=_cand_filtered,
        unit_coverage=_cand_unit_coverage,
        gate_plans=_cand_gate_plans,
        scope_members=_cand_scope_members,
        scope_labels=_cand_scope_labels,
        scope_types=_cand_scope_types,
        twin_unit_filters=_cand_twin_unit_filters,
    )


def _candidate_observation_entities(
    cand: _Candidate,
) -> tuple[dict[str, list[str]], Warning | None]:
    """The entities a candidate's observations bind by role, read on its discovered members
    and flow axis once settled, with the warning their derivation raises."""
    destination_dcids, origin_dcids = scope_axis_dcids(cand.mat, cand.place.probe_places)
    return _observation_entities(
        cand.shape,
        destination_dcids=destination_dcids,
        destination_dcid=cand.place.roles.obs_destination_dcid,
        origin_dcids=origin_dcids or cand.place.roles.origin_dcids,
    )


@dataclass(frozen=True)
class _CandidateConversion:
    """One candidate's answer once its elected conversion has run, what that conversion did
    to its rows, and the plan the collapse's bound-unit gate reads: the answer's plan once
    the conversion narrows it, else the plan `convertibility.unit_gate` read."""

    answer: FilteredAnswer
    coverage: UnitCoverage
    gate_plan: ReadPlan | None


def _candidate_conversions(
    state: ResolveState,
    candidates: Sequence[_Candidate],
    filtered: Sequence[FilteredAnswer],
    elected: Sequence[Mapping[str, tuple[Mapping[str, FacetSelection], ...]]],
) -> list[_CandidateConversion]:
    """Each candidate's conversion (`_CandidateConversion`), in input order.

    A conversion runs only where the collapse's bound-unit filter reads it: for a candidate
    with data at the place whose plan serves every StatVar under its elected unit
    (`convertibility.unit_gate`). Any other candidate keeps its answer and an empty coverage,
    so a candidate the filter drops or passes unchecked makes no conversion read. Candidates
    read nothing another writes, so several run side by side."""

    def convert(index: int) -> _CandidateConversion:
        cand = candidates[index]
        unconverted = _CandidateConversion(filtered[index], UnitCoverage(), filtered[index].plan)
        if cand.mat.available_at_place is False:
            return unconverted
        obs_entities, _warning = _candidate_observation_entities(cand)
        gate = unit_gate(
            cand.mat,
            filtered[index].plan,
            elected[index],
            obs_entities,
            graph=state.materialise_graph,
            deadline=state.deadline,
        )
        if gate.reason is not None:
            return replace(unconverted, gate_plan=gate.plan)
        answer, coverage = convertible_answer(
            cand.mat,
            filtered[index],
            elected[index],
            graph=state.materialise_graph,
            sdmx=state.materialise_sdmx,
            rate_cache=state.rate_cache,
            dated_facets=state.graph,
            deadline=state.deadline,
            date_request=state.date_request,
            coverage_index=state.coverage.index,
        )
        gate_plan = answer.plan if answer is not filtered[index] else gate.plan
        return _CandidateConversion(answer, coverage, gate_plan)

    if len(candidates) < 2:
        return [convert(i) for i in range(len(candidates))]
    with ThreadPoolExecutor(
        max_workers=len(candidates), thread_name_prefix="candidate-conversion"
    ) as pool:
        futures = [submit_with_context(pool, convert, i) for i in range(len(candidates))]
        return [future.result() for future in futures]


@dataclass(frozen=True)
class _CandidateGroundwork:
    """Fields for assembling one candidate Spec."""

    # Every facet the candidate's StatVars hold under the unit filter, served or not.
    confirmed_facets: list[Facet]
    coverage: Coverage
    headline: Headline | None
    plan: ReadPlan | None
    shape_model: Shape
    stat_vars: list[StatVar]
    entities: list[Entity]
    identity_entities: list[Entity]
    slots: list[Slot]
    inapplicable_slots: list[str]
    unresolved_slots: list[str]
    source_refs: list[ProvenanceRef]
    facets: dict[str, ResolvedFacet]
    slot_options_raw: list[SlotOptions]
    member_count: int
    stat_vars_member_count: int | None
    stat_vars_member_count_exact: bool
    entity_warnings: list[Warning]
    # The candidate's in-memory served test, None for a subject-place family.
    served_filter: RoleServedFilter | None


def _ground_one_candidate(
    state: ResolveState, cand_index: int, cand: _Candidate, prep: _CandidatePrep
) -> _CandidateGroundwork:
    """Ground one candidate from its filtered answer (`_CandidatePrep.filtered`)."""
    cand_shape, cand_bindings, cand_mat = cand.shape, cand.bindings, cand.mat
    # Materialisation returns unfiltered facets. Request-form units narrow per StatVar to
    # that candidate's election. A pooled predicate could admit another StatVar's facet.
    # Ref-form or unbound requests use the literal unit filter, and a twin its currency.
    _twin_unit_filter = prep.twin_unit_filters[cand_index]
    if prep.facets_in_elected_units is not None:
        _cand_facets_by_sv = (
            prep.facets_in_elected_units[cand_index] if cand_mat.facets_by_sv is not None else None
        )
    else:
        _cand_facets_by_sv = narrow_facets_by_sv_to_filter(
            cand_mat.facets_by_sv, _twin_unit_filter or state.unit_facet_filter
        )
    _cand_filtered = prep.filtered[cand_index]
    _cand_coverage = _cand_filtered.coverage
    (
        ft_refs,
        role_refs,
        cand_slots,
        _cand_slot_key_models,
        cand_sv_refs,
        cand_entity_data,
        cand_source_refs,
        cand_slot_options,
        _cand_draft_inapplicable_slots,  # unused: draft-kind-derived, see below
        _cand_ground_warnings,
        cand_labels,
    ) = _ground_answer(
        cand_shape,
        cand_bindings,
        cand_mat.sv_dcids,
        cand.place.roles.roles,
        state.graph,
        # The candidate's filtered plan names its sources and dates its StatVars, as the
        # elected answer's does.
        _cand_filtered.plan,
        state.entity_label_text_by_dcid,
        # A candidate's preview converts by the request against its own elections.
        unit_request=state.unit_request,
        member_facts=cand_mat.member_facts,
        open_axes=cand.place.roles.open_axes,
    )
    cand_facets = answer_facets(
        _cand_filtered.plan,
        cand_labels,
        bound_axes=(state.bound_facet_axes if _twin_unit_filter is None else {"unit": "bound"}),
        request_basis=state.unit_request_basis,
        unit_request=state.unit_request,
        currency_of=functools.partial(unit_currency, state.graph, deadline=state.deadline),
    )

    # Recompute each candidate's slot status from its resolved member facts. Index-sourced
    # shapes can cover a different population from ``shape.sv_arc_facts``.
    cand_inapplicable_slots = (
        inapplicable_slots_from_member_facts(
            slots=cand_slots, sv_dcids=cand_mat.sv_dcids, member_facts=cand_mat.member_facts or {}
        )
        if cand_mat.sv_dcids
        else []
    )
    cand_unresolved_slots = (
        unresolved_slots_from_member_facts(
            slots=cand_slots, sv_dcids=cand_mat.sv_dcids, member_facts=cand_mat.member_facts or {}
        )
        if cand_mat.sv_dcids
        else []
    )

    # Use the admitted population for broadest-first Spec ordering. The grounded StatVar
    # representative count measures a different quantity. The fallback handles hand-built results.
    cand_member_count = (
        cand_mat.population.admitted
        if cand_mat.population is not None
        else (len(cand_mat.member_facts) if cand_mat.member_facts else 1)
    )
    # ``confirmed`` counts admitted members with observations at the probe entities.
    # ``exact`` marks whether a capped recall sample makes that count a floor.
    cand_stat_vars_member_count = (
        cand_mat.population.confirmed if cand_mat.population is not None else None
    )
    cand_stat_vars_member_count_exact = (
        cand_mat.population.member_count_exact if cand_mat.population is not None else True
    )
    cand_shape_model = build_shape_model(
        cand_shape,
        _cand_slot_key_models,
        ft_refs,
        member_count=cand_member_count,
        members=cand_mat.served_members,
    )
    cand_stat_vars = build_stat_vars(
        cand_sv_refs,
        cand_shape.shape_id,
        cand_slots,
        plan=_cand_filtered.plan,
        destination_confirmed=cand_mat.destination_confirmed,
        member_facts=cand_mat.member_facts or {},
    )
    cand_entities = [_build_entity(rd, er, etr, role_refs) for rd, er, etr in cand_entity_data]
    # Hash the same role-dispatched entities presented in the response.
    cand_identity_entities = build_identity_entities(cand_entity_data, role_refs)
    cand_slots = bind_when_slot(cand_slots, window=_cand_coverage.window)
    cand_slots = apply_slot_expansions(cand_slots, slot_expansions(cand_bindings))
    # Count the menus over the candidate's own probe entities, as its probes read them. Its
    # identity binds a unit for a bound unit request or a twin's currency (`_finish_candidate`).
    facets_pinned = state.bound_unit_outcome is not None or _twin_unit_filter is not None
    cand_served_filter = role_served_filter(
        cand_shape,
        coverage_probe_role_entities(
            cand_shape,
            cand_mat,
            destination_dcids=cand.place.probe_places,
            destination_dcid=cand.place.roles.obs_destination_dcid,
            origin_dcids=cand.place.roles.origin_dcids,
        ),
        open_axes=cand.place.roles.open_axes,
        years=cand_mat.preview_years,
        coverage=state.coverage.index,
        facets_pinned=facets_pinned,
        plan=_cand_filtered.plan,
    )
    cand_slot_options = counted_slot_options(
        cand_slot_options,
        slots=cand_slots,
        shape=cand_shape,
        bindings=cand_bindings,
        plan=_cand_filtered.plan,
        member_facts=cand_mat.member_facts or {},
        served_filter=cand_served_filter,
        facets_pinned=facets_pinned,
    ).options

    return _CandidateGroundwork(
        confirmed_facets=[f for facets in (_cand_facets_by_sv or {}).values() for f in facets],
        coverage=_cand_coverage,
        headline=_cand_filtered.headline,
        plan=_cand_filtered.plan,
        shape_model=cand_shape_model,
        stat_vars=cand_stat_vars,
        entities=cand_entities,
        identity_entities=cand_identity_entities,
        slots=cand_slots,
        inapplicable_slots=cand_inapplicable_slots,
        unresolved_slots=cand_unresolved_slots,
        source_refs=cand_source_refs,
        facets=cand_facets,
        slot_options_raw=cand_slot_options,
        member_count=cand_member_count,
        stat_vars_member_count=cand_stat_vars_member_count,
        stat_vars_member_count_exact=cand_stat_vars_member_count_exact,
        entity_warnings=_cand_ground_warnings,
        served_filter=cand_served_filter,
    )


def _ground_each_candidate(
    state: ResolveState, candidates: Sequence[_Candidate], prep: _CandidatePrep
) -> list[_CandidateGroundwork]:
    """Ground every candidate (`_ground_one_candidate`), in input order. Each makes its own
    label and type reads and reads nothing another writes, so several are grounded side by
    side and their reads take the steps of one."""
    if len(candidates) < 2:
        return [_ground_one_candidate(state, i, cand, prep) for i, cand in enumerate(candidates)]
    with ThreadPoolExecutor(
        max_workers=len(candidates), thread_name_prefix="candidate-grounding"
    ) as pool:
        futures = [
            submit_with_context(pool, _ground_one_candidate, state, i, cand, prep)
            for i, cand in enumerate(candidates)
        ]
        return [future.result() for future in futures]


@dataclass
class _GroundedCandidate:
    """A finished Spec and its candidate-specific collapse metadata."""

    spec: Spec
    is_elected: bool
    score: float | None
    retrieved: bool
    population: MemberPopulation | None
    confirmed_facets: list[Facet]
    role_axis_facts: tuple[ShapeDraft, list[RoleAxisFactInput], list[str] | None]
    measure_partition: MeasurePartition | None
    measure_siblings: tuple[MeasurePartition, ...]
    open_axis_facts: tuple[OpenAxisFact, ...]
    probe_entities: dict[str, list[str]]
    probe_inputs: (
        tuple[ShapeDraft, list[SlotBindingDraft], list[SlotOptions], list[str] | None] | None
    )
    obs_entities: tuple[list[str], dict[str, list[str]]]
    elected_raw: Mapping[str, tuple[Mapping[str, FacetSelection], ...]]
    bulk_scope: BulkScope | None
    plan: ReadPlan | None
    role_declaration_unverified: bool
    open_origin_defaulted: bool
    membership_unverified: bool
    obs_warning: Warning | None
    place: PlaceContext


def _finish_candidate(
    state: ResolveState,
    cand_index: int,
    cand: _Candidate,
    groundwork: _CandidateGroundwork,
    prep: _CandidatePrep,
    measure_family_cache: dict[MeasureFamilyCacheKey, tuple[MeasurePartition, ...]],
    *,
    text_recognized_properties: frozenset[str],
    measure_elected_properties: frozenset[str],
    defaulted_properties: frozenset[str],
    rivalled: bool,
) -> _GroundedCandidate:
    """Assemble one candidate's Spec and the metadata used by candidate collapse.

    `rivalled` holds when the status rule offers a structural or nominated rival beside
    the elected reading.

    The candidate's probes, observation reads and bulk read follow its own place context.
    """
    cand_shape, cand_bindings, cand_mat = cand.shape, cand.bindings, cand.mat
    # Candidate triples contain data. Flagged-unavailable results may have
    # ``available_at_place=False``. Place-less coverage uses ``None``.
    cand_is_elected = cand_index == 0 and state.elected_has_data
    # A reading the status rule offers on a structural or nominated basis is a rival, and
    # so are the elected reading and its other directions and breakdowns beside one
    # (`rivalled`).
    cand_has_rival = cand.basis in ("structural", "nominated") or (
        rivalled and (cand_is_elected or cand.basis in ("direction", "breakdown"))
    )
    cand_available_at_place, cand_confidence = _place_confidence(
        destination_dcid=state.destination_dcid,
        available_at_place=cand_mat.available_at_place,
        representative_score=cand_shape.representative_score,
        candidate=True,
        rival=cand_has_rival,
        sole_reading=cand_is_elected and state.status_plan.sole,
        retrieved=cand_shape.retrieved,
    )
    # Probe the candidate's own places: its resolved region members, default scope or
    # discovered members.
    cand_destination_dcids = cand.place.probe_places
    # Keep uncovered slot options here. Probe coverage after dedupe and no-data filtering.
    cand_scope_member_dcids = prep.scope_members[cand_index]
    cand_scope = (
        build_scope(
            scope_axis(cand_mat), cand_scope_member_dcids, prep.scope_labels, prep.scope_types
        )
        if cand_scope_member_dcids is not None
        else None
    )
    # Served member facts determine each candidate's measure partition. A missing StatVar
    # entry violates the materialisation contract. Share the partition cache across candidates.
    cand_served_member_facts = cand_mat.member_facts or {}
    cand_served_members = [cand_served_member_facts[dcid] for dcid in cand_mat.sv_dcids]
    cand_role_keys = entity_role_keys(
        cand.place.roles.resolved_entities, open_axes=cand.place.roles.open_axes
    )
    cand_measure_partition, cand_measure_siblings = observed_measure_family(
        cand_shape,
        get_index(),
        get_measure_index().axes,
        shape_query(
            cand_shape,
            cand_bindings,
            cand_role_keys.bound,
            open_role_keys=cand_role_keys.open,
        ),
        cand_served_members,
        cache=measure_family_cache,
    )
    cand_open_axis_facts = served_open_axes(
        cand_bindings,
        cand_mat.sv_dcids,
        cand_served_member_facts,
        value_labels=cand_shape.value_labels or {},
    )
    # Reuse materialised role facts without extra SDMX reads. Labelled entity options are
    # built only for a surviving definite answer.
    cand_role_axis_facts = role_axis_facts_for_materialised(
        cand_mat, shape=cand_shape, plan=groundwork.plan
    )
    cand_facet_bindings = (
        # A bound outcome guarantees a unit request. The type checker cannot carry that
        # attribute invariant through this access.
        {"unit": IdentityFacetValueRequest(request=cast(IdentityUnitRequest, state.unit_request))}
        if state.bound_unit_outcome is not None
        else (
            {"unit": cluster_currency_binding(groundwork.confirmed_facets)}
            if prep.twin_unit_filters[cand_index] is not None
            else None
        )
    )
    cand_spec = build_spec(
        shape=groundwork.shape_model,
        slots=groundwork.slots,
        stat_vars=groundwork.stat_vars,
        entities=groundwork.entities,
        coverage=groundwork.coverage,
        plan=groundwork.plan,
        pin_served_provenance=state.pin_served_provenance,
        headline=groundwork.headline,
        coverage_index=state.coverage.index,
        coverage_stamp=cand_mat.coverage_stamp,
        scope=cand_scope,
        default_reading=state.default_reading,
        stated_form=state.stated_form,
        resolved_sources=groundwork.source_refs,
        facets=groundwork.facets,
        stat_vars_member_count=groundwork.stat_vars_member_count,
        stat_vars_member_count_exact=groundwork.stat_vars_member_count_exact,
        walked=cand.walked,
        slot_options=groundwork.slot_options_raw,
        available_at_place=cand_available_at_place,
        confidence=cand_confidence,
        requested_window=requested_window_from_date_request(state.date_request),
        identity_entities=groundwork.identity_entities,
        open_roles=cand_mat.open_axes,
        inapplicable_slots=groundwork.inapplicable_slots,
        unresolved_slots=groundwork.unresolved_slots,
        measure_partition=cand_measure_partition,
        measure_siblings=cand_measure_siblings,
        open_axis_facts=cand_open_axis_facts,
        role_axis_facts=cand_role_axis_facts,
        facet_bindings=cand_facet_bindings,
        text_recognized_properties=(
            text_recognized_properties if cand_bindings is state.bindings else frozenset()
        ),
        # Judged under the unit request this candidate is served under, as the definite
        # answer's trace is.
        query_qualifiers=answer_terms(
            state,
            cand_shape,
            cand_bindings,
            served=bound_prices(cand_facet_bindings),
            uncovered=cand.uncovered,
        ),
        measure_elected_properties=(
            measure_elected_properties if cand_bindings is state.bindings else frozenset()
        ),
        defaulted_properties=(
            defaulted_properties if cand_bindings is state.bindings else frozenset()
        ),
        named_groups=request_named_groups(state, cand.place.roles),
        measure_served_rows=measure_served_counter(groundwork.served_filter),
    )
    check_breakdown_settled(cand_shape, cand.walked, cand_spec)
    cand_spec = apply_declaration_version(cand_spec, cand_bindings)
    cand_role_declaration_unverified = (
        False if cand_is_elected else cand_mat.role_declaration_unverified
    )
    cand_open_origin_defaulted = (
        not cand_is_elected
        and cand_mat.open_origin_dcids is not None
        and "origin" not in cand.place.roles.unnamed_wholes
    )

    cand_places = cand.place.roles
    _cand_probe_role_entities = coverage_probe_role_entities(
        cand_shape,
        cand_mat,
        destination_dcids=cand_destination_dcids,
        destination_dcid=cand_places.obs_destination_dcid,
        origin_dcids=cand_places.origin_dcids,
    )
    # Retain inputs for both buffered and SSE probes.
    cand_probe_inputs = (
        (
            cand_shape,
            cand_bindings,
            groundwork.slot_options_raw,
            _flat_obs_entities(_cand_probe_role_entities),
        )
        if state.stream_slot_options or state.include_option_coverage
        else None
    )
    cand_obs_entities, cand_obs_warn = _candidate_observation_entities(cand)
    cand_bulk_scope = bulk_scope_for(cand.place, cand_mat, sdmx=state.sdmx)

    return _GroundedCandidate(
        spec=cand_spec,
        is_elected=cand_is_elected,
        score=cand_shape.representative_score,
        retrieved=cand_shape.retrieved,
        population=cand_mat.population,
        confirmed_facets=groundwork.confirmed_facets,
        role_axis_facts=(cand_shape, cand_role_axis_facts, cand_destination_dcids),
        measure_partition=cand_measure_partition,
        measure_siblings=cand_measure_siblings,
        open_axis_facts=cand_open_axis_facts,
        probe_entities=_cand_probe_role_entities,
        probe_inputs=cand_probe_inputs,
        obs_entities=(list(cand_mat.sv_dcids), cand_obs_entities),
        elected_raw=prep.elected_raw[cand_index],
        bulk_scope=cand_bulk_scope,
        plan=groundwork.plan,
        role_declaration_unverified=cand_role_declaration_unverified,
        open_origin_defaulted=cand_open_origin_defaulted,
        membership_unverified=False if cand_is_elected else cand_mat.membership_unverified,
        obs_warning=cand_obs_warn,
        place=cand.place,
    )


@dataclass
class _GroundedCandidates:
    """Deduped candidate Specs and ``spec_id``-keyed data for collapse and option probes.

    Spec order follows the candidates'. ``answer_t0`` excludes context preparation
    from answer timing.
    """

    specs: list[Spec]
    elected_spec_id: str | None
    cand_scores: dict[str, float | None]
    cand_retrieved: dict[str, bool]
    cand_populations: dict[str, MemberPopulation | None]
    cand_probe_inputs: dict[
        str, tuple[ShapeDraft, list[SlotBindingDraft], list[SlotOptions], list[str] | None]
    ]
    cand_role_axis_facts: dict[str, tuple[ShapeDraft, list[RoleAxisFactInput], list[str] | None]]
    cand_entity_options_raw: dict[str, list[EntityOptions]]
    cand_measure_partitions: dict[str, MeasurePartition | None]
    # Each candidate's sibling partitions and open constraint axes, which a definite
    # survivor's region carries so a same-shape merge re-mints its measure label.
    cand_measure_siblings: dict[str, tuple[MeasurePartition, ...]]
    cand_open_axis_facts: dict[str, tuple[OpenAxisFact, ...]]
    cand_obs_entities: dict[str, tuple[list[str], dict[str, list[str]]]]
    cand_probe_entities: dict[str, dict[str, list[str]]]
    cand_elected: dict[str, Mapping[str, tuple[Mapping[str, FacetSelection], ...]]]
    cand_bulk_scope: dict[str, BulkScope]
    cand_plans: dict[str, ReadPlan]
    # Each candidate's confirmed facets (`_CandidateGroundwork.confirmed_facets`), which the
    # collapse compares provenance sets from. No answer cites them.
    cand_confirmed_facets: dict[str, list[Facet]]
    cand_answered_roles: dict[str, tuple[str, ...]]
    """The roles each Spec answers under: those its candidate shape declares."""
    # Each candidate's place context, settled on its materialised result.
    cand_places: dict[str, PlaceContext]
    answer_t0: int
    # Each candidate's carrier and the scope its Spec lists (`_CandidateContext.scope_members`),
    # which the discovered-scope warning counts for the candidates a response keeps.
    cand_scopes: dict[str, tuple[Materialised, list[str] | None]] = field(default_factory=dict)
    # Each Spec's candidate (shape, bindings, materialised result and place context) and its
    # definite unit refusals by StatVar, which answer a bound unit no candidate serves.
    cand_candidates: dict[str, _Candidate] = field(default_factory=dict)
    cand_unit_refusals: dict[str, Mapping[str, MaterialisationError]] = field(default_factory=dict)
    # What each Spec's elected conversion does to its rows (`_CandidatePrep.unit_coverage`),
    # and the plan its bound-unit gate reads (`_CandidatePrep.gate_plans`).
    cand_unit_coverage: dict[str, UnitCoverage] = field(default_factory=dict)
    cand_gate_plans: dict[str, ReadPlan | None] = field(default_factory=dict)
    # Each passing candidate's `currency_unmapped_omitted` warning, stamped with its Spec, for
    # the response that keeps it.
    cand_unit_omitted: dict[str, Warning] = field(default_factory=dict)


def scopes_on_axis(
    candidates: Sequence[Materialised],
    scope_members: Sequence[list[str] | None],
    *,
    axis: EntityRoleKey,
) -> set[frozenset[str]]:
    """The distinct scopes the candidates list on `axis`, the axis a discovered-scope warning
    counts. A candidate with no scope, or with a scope on another axis, has none here: a
    subject-only alternate's places say nothing about a role-bearing candidate's
    recipients."""
    return {
        frozenset(members)
        for candidate, members in zip(candidates, scope_members, strict=True)
        if members is not None and scope_axis(candidate) == axis
    }


def restate_scope_warning(
    state: ResolveState, scopes: Sequence[tuple[Materialised, list[str] | None]]
) -> None:
    """Restate the elected shape's discovered-scope warning for the candidates a response
    states, each given as its carrier and the scope its Spec lists
    (`restate_role_scope_warnings`). The warning states a count only when every scope on its
    axis lists the same members (`scopes_on_axis`), and otherwise names every member with
    data. Grounding restates it over every candidate, and the collapse over the candidates
    the unit filter keeps, so a lone survivor's warning counts its own scope, as its
    resubmit does."""
    if state.opposite_labels is None or not isinstance(state.mat_result, Materialised):
        return
    axis = scope_axis(state.mat_result)
    on_axis = [
        members
        for candidate, members in scopes
        if members is not None and scope_axis(candidate) == axis
    ]
    if not on_axis:
        return
    distinct = scopes_on_axis(
        [candidate for candidate, _members in scopes],
        [members for _candidate, members in scopes],
        axis=axis,
    )
    own = next((members for candidate, members in scopes if candidate is state.mat_result), None)
    restate_role_scope_warnings(
        state.warnings,
        state.mat_result,
        members=own if own is not None else on_axis[0],
        boundary_roles=state.place.roles.roles,
        label_of=state.opposite_labels,
        unnamed_wholes=state.place.roles.unnamed_wholes,
        state_count=len(distinct) <= 1,
    )


def ground_candidates(state: ResolveState, candidates: list[_Candidate]) -> _GroundedCandidates:
    """Ground candidates in input order, dedupe Specs by ID, and fold alternate warnings.

    The elected shape's carrier warnings already come from ``_phase_materialise``. Its
    discovered-scope warning is restated to count the members its filtered plan serves
    (`restate_role_scope_warnings`), as its scope list and coverage do. The warning speaks
    for every candidate with a scope on its axis, so it states that count only when each of
    those scopes lists the same members; otherwise it names every member with data, and
    each candidate's scope lists its own. A scope on another axis, a subject-only
    alternate's places beside a role-bearing candidate's recipients, is no part of the
    comparison. A ``family_total_absent`` fact names one Spec's member, so each is recorded
    on the state by ``spec_id`` for the terminal that holds that Spec.
    """
    prep = _prepare_candidate_context(state, candidates)
    candidates = prep.candidates
    restate_scope_warning(
        state,
        [(cand.mat, members) for cand, members in zip(candidates, prep.scope_members, strict=True)],
    )
    answer_t0 = now_ms()
    measure_family_cache: dict[MeasureFamilyCacheKey, tuple[MeasurePartition, ...]] = {}
    # These forced qualifier, measure, and default bindings belong only to candidates that
    # reuse the elected ``state.bindings`` list.
    text_recognized_properties = state.text_recognized_properties
    measure_elected_properties = state.measure_elected_properties
    defaulted_properties = state.defaulted_properties
    rivalled = any(
        alternative.basis in ("structural", "nominated") for alternative in state.status_plan.rivals
    )

    specs: list[Spec] = []
    seen_spec_ids: set[str] = set()
    elected_spec_id: str | None = None
    cand_scores: dict[str, float | None] = {}
    cand_retrieved: dict[str, bool] = {}
    cand_populations: dict[str, MemberPopulation | None] = {}
    cand_confirmed_facets: dict[str, list[Facet]] = {}
    cand_role_axis_facts: dict[
        str, tuple[ShapeDraft, list[RoleAxisFactInput], list[str] | None]
    ] = {}
    cand_measure_partitions: dict[str, MeasurePartition | None] = {}
    cand_measure_siblings: dict[str, tuple[MeasurePartition, ...]] = {}
    cand_open_axis_facts: dict[str, tuple[OpenAxisFact, ...]] = {}
    cand_probe_entities: dict[str, dict[str, list[str]]] = {}
    cand_probe_inputs: dict[
        str, tuple[ShapeDraft, list[SlotBindingDraft], list[SlotOptions], list[str] | None]
    ] = {}
    cand_obs_entities: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    cand_elected: dict[str, Mapping[str, tuple[Mapping[str, FacetSelection], ...]]] = {}
    cand_bulk_scope: dict[str, BulkScope] = {}
    cand_answered_roles: dict[str, tuple[str, ...]] = {}
    cand_plans: dict[str, ReadPlan] = {}
    cand_scopes: dict[str, tuple[Materialised, list[str] | None]] = {}
    cand_places: dict[str, PlaceContext] = {}
    cand_candidates: dict[str, _Candidate] = {}
    cand_unit_refusals: dict[str, Mapping[str, MaterialisationError]] = {}
    cand_unit_coverage: dict[str, UnitCoverage] = {}
    cand_gate_plans: dict[str, ReadPlan | None] = {}
    role_unverified = False
    membership_unverified = False
    open_origin_defaulted = False
    obs_warnings: list[Warning] = []
    entity_warnings: list[Warning] | None = None

    groundworks = _ground_each_candidate(state, candidates, prep)
    for cand_index, (cand, groundwork) in enumerate(zip(candidates, groundworks, strict=True)):
        grounded = _finish_candidate(
            state,
            cand_index,
            cand,
            groundwork,
            prep,
            measure_family_cache,
            text_recognized_properties=text_recognized_properties,
            measure_elected_properties=measure_elected_properties,
            defaulted_properties=defaulted_properties,
            rivalled=rivalled,
        )
        spec_id = grounded.spec.spec_id
        if entity_warnings is None:
            entity_warnings = groundwork.entity_warnings
        if grounded.is_elected:
            elected_spec_id = spec_id
        else:
            role_unverified = role_unverified or grounded.role_declaration_unverified
            open_origin_defaulted = open_origin_defaulted or grounded.open_origin_defaulted
            membership_unverified = membership_unverified or grounded.membership_unverified
        if grounded.obs_warning is not None:
            obs_warnings.append(grounded.obs_warning)
        if spec_id not in seen_spec_ids:
            seen_spec_ids.add(spec_id)
            specs.append(grounded.spec)
            cand_answered_roles[spec_id] = answered_roles(cand.shape)
            if cand.mat.family_total_absent is not None:
                state.family_total_absent[spec_id] = cand.mat.family_total_absent
        cand_scores[spec_id] = grounded.score
        cand_retrieved[spec_id] = grounded.retrieved
        cand_populations[spec_id] = grounded.population
        cand_confirmed_facets[spec_id] = grounded.confirmed_facets
        cand_role_axis_facts[spec_id] = grounded.role_axis_facts
        cand_measure_partitions[spec_id] = grounded.measure_partition
        cand_measure_siblings[spec_id] = grounded.measure_siblings
        cand_open_axis_facts[spec_id] = grounded.open_axis_facts
        cand_probe_entities[spec_id] = grounded.probe_entities
        if grounded.probe_inputs is not None:
            cand_probe_inputs[spec_id] = grounded.probe_inputs
        cand_obs_entities[spec_id] = grounded.obs_entities
        cand_elected[spec_id] = grounded.elected_raw
        if grounded.bulk_scope is not None:
            cand_bulk_scope[spec_id] = grounded.bulk_scope
        if grounded.plan is not None:
            cand_plans[spec_id] = grounded.plan
        cand_scopes[spec_id] = (cand.mat, prep.scope_members[cand_index])
        cand_places[spec_id] = grounded.place
        cand_candidates[spec_id] = cand
        cand_unit_refusals[spec_id] = prep.unit_refusals[cand_index]
        cand_unit_coverage[spec_id] = prep.unit_coverage[cand_index]
        cand_gate_plans[spec_id] = prep.gate_plans[cand_index]

    state.warnings.extend(obs_warnings)
    if entity_warnings:
        state.warnings.extend(entity_warnings)
    # Aggregate alternate carrier flags so each warning appears once.
    state.warnings.extend(
        carrier_flag_warnings(
            role_declaration_unverified=role_unverified,
            open_origin_defaulted=open_origin_defaulted,
            # The elected shape's own unsettled membership already warned at binding.
            membership_unverified=membership_unverified
            and all(warning.code != MEMBERSHIP_UNVERIFIED for warning in state.warnings),
        )
    )

    return _GroundedCandidates(
        specs=specs,
        elected_spec_id=elected_spec_id,
        cand_scores=cand_scores,
        cand_retrieved=cand_retrieved,
        cand_populations=cand_populations,
        cand_probe_inputs=cand_probe_inputs,
        cand_role_axis_facts=cand_role_axis_facts,
        cand_entity_options_raw={},
        cand_measure_partitions=cand_measure_partitions,
        cand_measure_siblings=cand_measure_siblings,
        cand_open_axis_facts=cand_open_axis_facts,
        cand_obs_entities=cand_obs_entities,
        cand_probe_entities=cand_probe_entities,
        cand_elected=cand_elected,
        cand_bulk_scope=cand_bulk_scope,
        cand_plans=cand_plans,
        cand_confirmed_facets=cand_confirmed_facets,
        cand_answered_roles=cand_answered_roles,
        cand_places=cand_places,
        answer_t0=answer_t0,
        cand_scopes=cand_scopes,
        cand_candidates=cand_candidates,
        cand_unit_refusals=cand_unit_refusals,
        cand_unit_coverage=cand_unit_coverage,
        cand_gate_plans=cand_gate_plans,
    )
