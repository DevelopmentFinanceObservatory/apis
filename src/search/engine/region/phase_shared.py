"""Terminal results shared by the region pipeline's materialise and answer phases."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from dfo_platform import Cause
from materialise.recipes import (
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    cause_from_recipe_error,
)
from search.engine.bind import SlotBindingDraft
from search.engine.departures import measure_not_applied, price_basis_not_applied
from search.engine.qualifier_match import QualifierMatch
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.grounding import (
    _build_entity,
    _ground_answer,
    build_identity_entities,
    requested_window_from_date_request,
    role_bindings,
)
from search.engine.region.place_roles import request_named_groups, untold_direction_names
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    apply_declaration_version,
    apply_slot_expansions,
    slot_expansions,
)
from search.engine.shape import identity_shape_for
from search.engine.spec_build import (
    build_shape_model,
    build_slot,
    build_spec,
    build_stat_vars,
    build_unsupported_definition,
    pair_slot_drafts,
)
from search.models import (
    CoverageBare,
    GraphRef,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    Spec,
    Unsupported,
    Warning,
)

if TYPE_CHECKING:
    from search.engine.region.candidates import _Candidate


def _grounded_values(binding: SlotBindingDraft | None) -> list[GraphRef]:
    """Use dcids as labels; identity hashes read only `.ref.dcid`."""
    if binding is None or binding.kind not in ("value", "set"):
        return []
    return [GraphRef(dcid=d, label=d) for d in binding.value_dcids]


def _admitted_qualifier_properties(matches: tuple[QualifierMatch, ...]) -> frozenset[str]:
    """Constraint property dcids `_force_qualifier_bindings` forced a value
    onto, for build_spec's text_recognized_properties, the same code_system
    key it groups admitted matches by.
    """
    return frozenset(m.code_system for m in matches)


def _unsupported_definition(
    state: ResolveState,
    facets: dict[str, IdentityFacetBinding] | None = None,
    *,
    candidate: _Candidate | None = None,
) -> dict[str, object]:
    """Build the unsupported definition with the identity's shape, bindings, roles, and window.

    The shape, bindings and roles are `candidate`'s when given, else the elected shape's.
    No election confirms an open axis, so the definition keeps the axes the part leaves
    open (`PlaceRoles.open_axes`), as the resubmit door's keeps the posted `open_roles`.
    Graph reads add display labels, which identity hashing ignores. Preserve `facets` so an
    unsupported pointer can resolve against a completed facet request."""
    if candidate is None:
        shape, bindings, place_roles = state.shape_draft, state.bindings, state.place.roles
    else:
        shape, bindings, place_roles = candidate.shape, candidate.bindings, candidate.place.roles
    id_slots = [
        build_slot(
            slot_draft,
            binding,
            _grounded_values(binding),
            property_ref=(
                GraphRef(dcid=slot_draft.property_dcid, label=slot_draft.property_dcid)
                if slot_draft.property_dcid
                else None
            ),
        )
        for slot_draft, binding in pair_slot_drafts(shape.slot_keys, bindings)
    ]
    id_entities = [
        _build_entity(role, GraphRef(dcid=dcid, label=dcid), None, {})
        for dcid, role in role_bindings(place_roles.roles)
    ]
    return build_unsupported_definition(
        identity_shape_for(
            pop_type_dcid=shape.pop_type_dcid,
            meas_prop_dcid=shape.meas_prop_dcid,
            stat_type_dcid=shape.stat_type_dcid,
            meas_qual_dcid=shape.meas_qual_dcid,
            meas_denom_dcid=shape.meas_denom_dcid,
            meas_denom_unresolved=shape.meas_denom_unresolved,
        ),
        id_slots,
        entities=id_entities,
        open_roles=place_roles.open_axes,
        requested_window=requested_window_from_date_request(state.date_request),
        explicit_facets=facets,
    )


def _unsupported(
    state: ResolveState,
    code: str,
    *,
    pointer: str,
    detail: str,
    llm_usage: dict | None = None,
    facets: dict[str, IdentityFacetBinding] | None = None,
    cause: Cause | None = None,
    candidate: _Candidate | None = None,
) -> RegionResult:
    """`cause` carries through to `Unsupported.cause` unchanged (populated
    only for code=="recipe_not_declared"). The definition describes `candidate` when given,
    else the elected shape."""
    return RegionResult(
        variable_text=state.variable,
        status="unsupported",
        specs=(),
        no_data_reason=None,
        warnings=tuple(state.warnings),
        departure_facts=dict(state.departure_facts),
        timing_by_step=dict(state.timing),
        llm_usage=llm_usage,
        unsupported=Unsupported(
            code=code,
            pointer=pointer,
            detail=detail,
            cause=cause,
            definition=_unsupported_definition(state, facets, candidate=candidate),
        ),
    )


# Unit elections refused because the data cannot be converted to the requested unit. Each maps
# to its own answer code in `unit_conversion_refusal`.
UNIT_CONVERSION_REFUSALS = (NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError)
type UnitConversionRefusal = NonCurrencyUnitError | RateCoverageGapError | RecipeNotDeclaredError


def unit_conversion_refusal(
    state: ResolveState,
    exc: UnitConversionRefusal,
    *,
    unit_request: IdentityUnitRequest,
    candidate: _Candidate | None = None,
) -> RegionResult:
    """Answer a unit election refused for conversion, on the single-shape and candidates doors.

    `RecipeNotDeclaredError` answers `recipe_not_declared` with its cause,
    `RateCoverageGapError` answers no-data `rate_coverage_gap` with its cause, and
    `NonCurrencyUnitError` answers `facet_binding_unsupported`. Every shape-derived field
    describes the shape that refused: `candidate` when the candidates door answers with a
    candidate's refusal, else the elected shape. An unsupported definition carries the unit
    request on that shape's identity, so resubmitting it meets the same refusal.
    """
    if isinstance(exc, RateCoverageGapError):
        return state.no_data(
            "rate_coverage_gap",
            cause=cause_from_recipe_error(exc),
            llm_usage=state.var_usage,
            candidate=candidate,
        )
    facets: dict[str, IdentityFacetBinding] = {
        "unit": IdentityFacetValueRequest(request=unit_request)
    }
    if isinstance(exc, RecipeNotDeclaredError):
        return _unsupported(
            state,
            "recipe_not_declared",
            pointer="/identity/facets/unit",
            detail=str(exc),
            llm_usage=state.var_usage,
            cause=cause_from_recipe_error(exc),
            facets=facets,
            candidate=candidate,
        )
    return _unsupported(
        state,
        "facet_binding_unsupported",
        pointer="/identity/facets/unit",
        detail=(
            "the requested unit's currency/prices/base_year combination is "
            "not expressed by any unit this deployment's data carries."
        ),
        llm_usage=state.var_usage,
        facets=facets,
        candidate=candidate,
    )


def withdraw_typed_unit(state: ResolveState, exc: UnitConversionRefusal) -> bool:
    """Whether the leg answers without its unit request instead of refusing it, for a unit
    election refused for conversion (`UNIT_CONVERSION_REFUSALS`).

    The raw-text door's typed unit phrase is best effort (`ResolveState.unit_text_best_effort`):
    the leg drops the request, reads as if no unit were asked, and departs
    `measure_not_applied` for the phrase. Currency data that cannot convert states its unit on
    the byline, whose token clears the departure (`price_basis_not_applied`); data in no
    currency (`NonCurrencyUnitError`) has no unit token, so no one slot clears it. Any other
    unit request (the parsed door's, a resubmit's) is explicit, returns False, and refuses
    through `unit_conversion_refusal`."""
    if not state.unit_text_best_effort or state.unit_text is None:
        return False
    state.unit_request = None
    state.bound_unit_outcome = None
    state.unit_facet_filter = None
    state.bound_facet_axes = {}
    state.unit_request_basis = None
    state.unit_elected_selections = None
    state.depart(
        Warning(
            code=QUERY_QUALIFIER_DROPPED,
            severity="degraded",
            message=f'The requested qualifier "{state.unit_text}" was not applied.',
        ),
        measure_not_applied(asked=state.unit_text, property_dcid=None)
        if isinstance(exc, NonCurrencyUnitError)
        else price_basis_not_applied(asked=state.unit_text, shown=None),
    )
    return True


async def _role_incompatible_flagged_match(state: ResolveState) -> Spec:
    """Build the flagged identity without probing availability.

    Preserve an established entity mapping on materialise-stage refusals so the where slot,
    entities, trace, and identity agree. Early refusals have no mapping to retain."""
    sv_dcids = list(state.shape_draft.sv_arc_facts or ())
    (
        five_tuple_refs,
        role_refs,
        slots,
        slot_key_models,
        labelled_sv_refs,
        entity_objects_data,
        _,
        _,
        _,
        grounding_warnings,
        _,
    ) = await asyncio.to_thread(
        _ground_answer,
        state.shape_draft,
        state.bindings,
        sv_dcids,
        state.place.roles.roles,
        state.graph,
        query_entity_text_by_dcid=state.entity_label_text_by_dcid,
        open_axes=state.place.roles.open_axes,
    )
    state.warnings.extend(grounding_warnings)

    # A ShapeDraft's members were confirmed before bind. Preserve every one
    # even if its display label is missing, matching _elected_stat_vars' own
    # dcid-label fallback for no-data outcomes.
    labels_by_dcid = {ref.dcid: ref for ref in labelled_sv_refs}
    sv_refs = [labels_by_dcid.get(dcid, GraphRef(dcid=dcid, label=dcid)) for dcid in sv_dcids]
    shape = build_shape_model(
        state.shape_draft,
        slot_key_models,
        five_tuple_refs,
        member_count=len(sv_refs),
        members=sv_dcids,
    )
    stat_vars = build_stat_vars(
        sv_refs,
        state.shape_draft.shape_id,
        slots,
        arcs_by_sv=state.shape_draft.sv_arc_facts or {},
    )
    entities = [
        _build_entity(role, entity_ref, entity_type_ref, role_refs)
        for role, entity_ref, entity_type_ref in entity_objects_data
    ]
    identity_entities = build_identity_entities(entity_objects_data, role_refs)
    requested_window = requested_window_from_date_request(state.date_request)
    slots = apply_slot_expansions(slots, slot_expansions(state.bindings))
    spec = build_spec(
        shape=shape,
        slots=slots,
        stat_vars=stat_vars,
        entities=entities,
        coverage=CoverageBare(
            has_data=False,
            window=state.date_request.window if state.date_request is not None else None,
        ),
        default_reading=state.default_reading,
        resolved_sources=[],
        slot_options=[],
        entity_options=[],
        facet_options=[],
        available_at_place=None,
        confidence="low",
        requested_window=requested_window,
        identity_entities=identity_entities,
        # No member serving the bound roles is elected, so no axis is open
        # (`phase_election._query_role_keys`).
        open_roles=(),
        # state.bindings (fed to slots above via slot_expansions) is the same
        # bindings list _phase_vet_election may have forced a qualifier-match
        # value onto. This flagged_match runs after that phase (see
        # _role_incompatible_no_data's own caller), so the forced set is final.
        text_recognized_properties=_admitted_qualifier_properties(state.admitted_qualifiers),
        # Every call site of _role_incompatible_no_data runs from
        # _phase_place_bindings or later, and all of those phases run after
        # _phase_repair_bindings' measure-election splice, so
        # state.measure_elected_properties already carries any forced
        # pairs, as on the definite path below.
        measure_elected_properties=state.measure_elected_properties,
        defaulted_properties=state.defaulted_properties,
        named_groups=request_named_groups(state, state.place.roles),
        walked=state.walked,
    )
    return apply_declaration_version(spec, state.bindings)


async def _role_incompatible_no_data(state: ResolveState) -> RegionResult:
    """Return the common role-incompatibility outcome without a new probe."""
    flagged_match = await _role_incompatible_flagged_match(state)
    return state.no_data(
        "role_incompatible",
        flagged_match=(flagged_match,),
        llm_usage=state.var_usage,
    )


async def _role_not_resolved_no_data(state: ResolveState) -> RegionResult:
    """Return `role_not_resolved` when place roles cannot be mapped.

    This covers multiple places without resolvable axes, a solitary place without an axis, or
    wording that contradicts a supplied axis. A `flagged_match` requires a resolved role map,
    which this refusal lacks. The names whose direction it could not tell
    (`untold_direction_names`) ride as the result's `unresolved_entity_names`."""
    return state.no_data(
        "role_not_resolved",
        llm_usage=state.var_usage,
        unresolved_entity_names=untold_direction_names(state),
    )
