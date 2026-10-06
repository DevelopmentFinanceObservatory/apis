"""Materialise the resubmit over its place context and settle that context."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from materialise.roles import axis_noun
from search.engine.bind import SlotBindingDraft
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.region.grounding import _build_entity, memoised_labels, role_bindings
from search.engine.region.materialise_step import (
    NoDataMaterialise,
    UnsupportedMaterialise,
    interpret_materialised,
    run_materialise,
)
from search.engine.region.observations.entities import open_role_axis
from search.engine.region.place_context import settle_place, settled_shape
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import (
    _facet_filter_from_bindings,
    _requested_window_for_id,
    _source_dcids_from_facets_provenance,
)
from search.engine.region.result import RegionResult
from search.engine.region.unrealisable import unbind_posted
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
)
from search.engine.shape import identity_shape_for
from search.engine.spec_build import build_slot, build_unsupported_definition, pair_slot_drafts
from search.models import GraphRef, IdentityFacetBinding, Unsupported

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
    """Build the unsupported definition from the resubmit's resolved shape, bindings, roles,
    and posted open roles.

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
        open_roles=state.inp.identity.open_roles or (),
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

    Return a terminal `RegionResult`, or `None` to continue to `_phase_ground_build`.
    `run_materialise` reads the shape and bindings over the place context. Bound values that
    together match no member (`retrieve.BoundValuesUnrealisable`) lose a minimal set, each
    value unbound departing under its name, and the shape is read again
    (`unrealisable.unbind_posted`), as a fresh request does. A discovered role scope's own
    refusal returns an `unsupported` part, and a no-data outcome or a candidate-only result
    returns `no_data`. A `Materialised` answer derives the source refs and facet filter
    (`_bound_facet_filter`), then settles the place context (`PlaceContext.settled`) and the
    shape over the role scope's members or the StatVars the default scope serves. A
    resubmit whose family neither the index nor the walk settled (`membership_unverified`)
    cannot name the family's total, so it drops `family_total_absent`. The answer refuses
    422 `open_role_unavailable` for any open role the posted identity names and the elected
    members do not serve (`_refuse_unserved_open_roles`)."""
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
            open_axes=state.place.roles.open_axes,
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
    # The open axes narrow to those the elected members serve, and the shape declares their
    # roles.
    state.place = settle_place(state.place, mat_result, state.warnings)
    state.shape_draft = settled_shape(state.shape_draft, mat_result)

    # A resubmit whose family the walk did not settle (`membership_unverified`) cannot name
    # the family's total, so it raises no election warning. One whose family the index or
    # the walk settled elects among it and warns where the first answer does.
    if mat_result.family_total_absent is not None and not state.family_settled:
        mat_result = state.mat_result = dataclasses.replace(mat_result, family_total_absent=None)
    _refuse_unserved_open_roles(state, mat_result)
    return None


def _refuse_unserved_open_roles(state: _PromoteState, mat_result: Materialised) -> None:
    """Refuse `open_role_unavailable` when the answer drops an open role the identity names.

    An open role in an identity is a promise of one series per member on that axis, and a
    posted identity is the whole definition, so one the elected members declare no role on
    (`Materialised.open_axes`) refuses the resubmit. An answer without the role would carry
    another `spec_id` than the identity posted. The parsed door's `open_roles` are a request,
    and their drop warns (`result.with_dropped_open_roles`)."""
    unserved = sorted(frozenset(state.inp.identity.open_roles or ()) - mat_result.open_axes)
    if not unserved:
        return
    nouns = " or ".join(f"every {axis_noun(axis)}" for axis in unserved)
    raise EngineInputError(
        f"open role(s) {unserved} unavailable",
        code="open_role_unavailable",
        status=422,
        pointer="/input/identity/open_roles",
        detail=(
            f"The identity asks for {nouns}, and the StatVars this definition elects "
            "give no series per member on that axis. A fresh search that asks "
            "for the role elects whatever definition serves it, if one does."
        ),
    )
