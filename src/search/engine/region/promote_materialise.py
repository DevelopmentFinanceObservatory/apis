"""Materialise the resubmit over its place context and settle that context."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from search.engine.bind import SlotBindingDraft
from search.engine.envelope import now_ms
from search.engine.region.grounding import _build_entity, memoised_labels, role_bindings
from search.engine.region.materialise_step import (
    NoDataMaterialise,
    UnsupportedMaterialise,
    interpret_materialised,
    run_materialise,
)
from search.engine.region.observations.entities import open_role_axis
from search.engine.region.place_context import settle_place
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import (
    _facet_filter_from_bindings,
    _requested_window_for_id,
    _source_dcids_from_facets_provenance,
)
from search.engine.region.result import RegionResult
from search.engine.region.unrealisable import unbind_posted
from search.engine.retrieval.facts import read_constraints
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
)
from search.engine.shape import identity_shape_for
from search.engine.spec_build import build_slot, build_unsupported_definition, pair_slot_drafts
from search.models import CoverageBare, GraphRef, IdentityFacetBinding, Unsupported

type MaterialiseResult = Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft


def _graph_labels(state: _PromoteState, dcids: Sequence[str]) -> list[str]:
    """Label dcids in one batch; an unlabelled dcid displays as itself."""
    graph_labels = state.graph.node_labels_batch(list(dcids))
    return [graph_labels.get(dcid, dcid) for dcid in dcids]


def _draft_grounded_values(binding: SlotBindingDraft | None) -> list[GraphRef]:
    """Use dcids as labels; identity projection reads only `.ref.dcid`."""
    if binding is None or binding.kind not in ("value", "set"):
        return []
    return [GraphRef(dcid=d, label=d) for d in binding.value_dcids]


def _resubmit_unsupported_definition(
    state: _PromoteState, facets: dict[str, IdentityFacetBinding] | None = None
) -> dict[str, object]:
    """Build the unsupported definition from the resubmit's resolved shape, bindings, and roles.

    Display labels are ignored by identity hashing, so no graph reads are needed. `facets`
    keeps a completed facet request, so an unsupported pointer at it resolves in the
    definition."""
    roles = state.place.roles.roles
    id_slots = [
        build_slot(
            slot_draft,
            binding,
            _draft_grounded_values(binding),
            property_ref=(
                GraphRef(dcid=slot_draft.property_dcid, label=slot_draft.property_dcid)
                if slot_draft.property_dcid
                else None
            ),
        )
        for slot_draft, binding in pair_slot_drafts(state.shape_draft.slot_keys, state.bindings)
    ]
    id_entities = [
        _build_entity(role, GraphRef(dcid=dcid, label=dcid), None, {})
        for dcid, role in role_bindings(roles)
    ]
    return build_unsupported_definition(
        identity_shape_for(
            pop_type_dcid=state.shape_draft.pop_type_dcid,
            meas_prop_dcid=state.shape_draft.meas_prop_dcid,
            stat_type_dcid=state.shape_draft.stat_type_dcid,
            meas_qual_dcid=state.shape_draft.meas_qual_dcid,
            meas_denom_dcid=state.shape_draft.meas_denom_dcid,
            meas_denom_unresolved=state.shape_draft.meas_denom_unresolved,
        ),
        id_slots,
        entities=id_entities,
        requested_window=_requested_window_for_id(state.date_request),
        explicit_facets=facets,
    )


def _unsupported_region(
    state: _PromoteState,
    code: str,
    *,
    pointer: str,
    detail: str,
    facets: dict[str, IdentityFacetBinding] | None = None,
) -> RegionResult:
    """Build the unsupported result from the resubmit's resolved shape, bindings, and roles."""
    return RegionResult(
        variable_text=state.variable_text,
        status="unsupported",
        specs=(),
        no_data_reason=None,
        warnings=state.refusal_warnings(),
        timing_by_step={},
        unsupported=Unsupported(
            code=code,
            pointer=pointer,
            detail=detail,
            definition=_resubmit_unsupported_definition(state, facets),
        ),
    )


def _bound_facet_filter(state: _PromoteState) -> None:
    """Derive this resubmit's source refs and facet filter once from facets.provenance and
    facets.measurementMethod/unit, and cache them on state.source_dcids and
    state.facet_filter for `_phase_ground_build`.
    """
    _source_dcids = state.source_dcids = _source_dcids_from_facets_provenance(
        state.inp.identity.facets
    )
    state.facet_filter = _facet_filter_from_bindings(
        state.graph, _source_dcids, state.inp.identity.facets, deadline=state.deadline
    )


def _phase_materialise(state: _PromoteState) -> RegionResult | None:
    """Materialise the representative StatVar over the place context.

    Return the draft's no-data reason, or `no_data` for candidate-only results. A capped
    discovered-scope facet index raises `scope_too_broad` when a method or unit is bound. A
    discovered role scope, or a default scope narrowed to the StatVars the answer serves,
    settles the place context (`PlaceContext.settled`). Restrict probes to
    members asserting every bound property, then fall back to a flagged-unavailable carrier
    when no asserting member has data. Bound values that together match no member
    (`retrieve.BoundValuesUnrealisable`) lose a minimal set, each value unbound departing
    under its name, and the shape is read again (`unrealisable.unbind_posted`), as a fresh
    request does."""
    resolved_entities = state.place.roles.resolved_entities

    def materialise() -> MaterialiseResult:
        return run_materialise(
            state.shape_draft,
            state.bindings,
            resolved_entities,
            graph=state.graph,
            sdmx=state.sdmx,
            date_request=state.date_request,
            deadline=state.deadline,
            place_dcids=state.place.probe_places,
            default_presence=state.place.presence_for(state.default_presence),
            coverage=state.coverage,
        )

    # The standard materialise probe uses the representative StatVar.
    t0 = now_ms()
    mat_result = materialise()
    while isinstance(mat_result, BoundValuesUnrealisable):
        unbind_posted(state, mat_result)
        mat_result = materialise()
    state.mat_result = mat_result
    # Preserve timings recorded by earlier phases.
    state.timing["materialise"] = now_ms() - t0

    # Interpret draft outcomes and collect warnings derived from a Materialised result.
    state.opposite_labels = memoised_labels(lambda dcids: _graph_labels(state, dcids))
    outcome = interpret_materialised(
        mat_result,
        boundary_roles=state.place.roles.roles,
        label_of=state.opposite_labels,
        warnings=state.warnings,
    )
    # A discovered scope's own bounds refuse the read
    # (retrieval/role_scope.py's discover_role_scope). Terminal here, ahead
    # of every other check below, none of which is meaningful against an
    # UnsupportedDraft.
    if isinstance(outcome, UnsupportedMaterialise):
        return _unsupported_region(
            state,
            outcome.code,
            pointer=outcome.pointer,
            detail=outcome.detail,
        )
    if isinstance(outcome, NoDataMaterialise):
        # Keep warnings collected before materialisation.
        return RegionResult(
            variable_text=state.variable_text,
            status="no_data",
            specs=(),
            no_data_reason=outcome.reason,
            warnings=tuple(state.warnings),
            timing_by_step={},
            no_data_role_axis=(
                open_role_axis(state.shape_draft, resolved_entities)
                if outcome.reason == "no_observations"
                else None
            ),
        )
    if not isinstance(mat_result, Materialised):
        # Standard promote accepts one fixed SV; candidate-only results map to no_data.
        return _no_data_region(state.variable_text, warnings=tuple(state.warnings))

    # Derived here, once, so _phase_ground_build's own coverage narrowing
    # reads state.source_dcids/facet_filter without computing it fresh.
    _bound_facet_filter(state)

    # Read the answer over the context its result settles: a discovered role scope's members,
    # or the default scope of the StatVars the answer serves.
    state.place = settle_place(state.place, mat_result, state.warnings)

    # Recheck edit-reshaped survivor sets because they can differ from materialise's filtered
    # members. Materialise already enforces this for its own probe, so skip when it returned
    # unavailable. Keep bindings unchanged when restricting data members; identity and spec id
    # remain based on the posted definition, while this check controls which data can answer.
    bound_props = {
        b.property_dcid
        for b in state.bindings
        if b.property_dcid and b.kind in ("value", "set") and b.value_dcids
    }
    if bound_props and mat_result.has_data:
        winner_constraints = [
            read_constraints(state.resolved_arcs[sv]) for sv in mat_result.sv_dcids
        ]
        if not all(all(prop in wc for wc in winner_constraints) for prop in bound_props):
            # Probe only members that assert every bound property. This preserves data from
            # valid members in mixed-presence groups without serving unsupported claims.
            asserting = [
                sv
                for sv in state.resolved_stat_var_dcids
                if all(prop in state.resolved_constraints_by_sv[sv] for prop in bound_props)
            ]
            asserting_result: Materialised | None = None
            if asserting:
                restricted_shape = dataclasses.replace(
                    state.shape_draft,
                    sv_arc_facts={sv: state.resolved_arcs[sv] for sv in asserting},
                )
                candidate = run_materialise(
                    restricted_shape,
                    state.bindings,
                    resolved_entities,
                    graph=state.graph,
                    sdmx=state.sdmx,
                    date_request=state.date_request,
                    deadline=state.deadline,
                    place_dcids=state.place.probe_places,
                    default_presence=state.place.presence_for(state.default_presence),
                    coverage=state.coverage,
                )
                # The asserting members' own answer stands when they hold data, in the years
                # read or, as a window carrier, outside them.
                if isinstance(candidate, Materialised) and (
                    candidate.has_data or candidate.window_gap is not None
                ):
                    asserting_result = candidate
            if asserting_result is not None:
                mat_result = state.mat_result = asserting_result
                # The asserting members' own default scope replaces the winner's.
                if asserting_result.default_scope is not None:
                    state.place = settle_place(state.place, asserting_result, state.warnings)
            else:
                # Use a bound-property asserter for the flagged identity, separate from the
                # unrelated winner's data.
                explicit_sv = asserting[0] if asserting else state.resolved_stat_var_dcids[0]
                explicit_facts = (mat_result.member_facts or {}).get(explicit_sv) or (
                    ArcsMemberFacts(arcs=state.resolved_arcs[explicit_sv])
                )
                mat_result = state.mat_result = Materialised(
                    sv_dcids=[explicit_sv],
                    member_facts={explicit_sv: explicit_facts},
                    has_data=False,
                    coverage=CoverageBare(
                        has_data=False,
                        window=state.date_request.window if state.date_request else None,
                    ),
                    available_at_place=False,
                )

    # A resubmit whose family the mint's walk settled nothing for reads its posted members
    # alone and runs no election, so it raises no election warning. One whose family the
    # index or the walk settled elects among it and warns where the first answer does.
    if mat_result.family_total_absent is not None and not state.family_settled:
        mat_result = state.mat_result = dataclasses.replace(mat_result, family_total_absent=None)
    return None
