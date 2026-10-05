"""Validate the posted identity and read its bound values for standard resubmit."""

from __future__ import annotations

from collections.abc import Sequence

from code_index.shared import get_measure_index
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.identity_edit import (
    check_edit_incomplete,
    check_no_exact_bindings,
    check_shape_unchanged,
    check_slots_declared,
    validate_edited_identity,
    validate_removed,
    verify_base_identity,
)
from search.engine.region.place_roles import place_order
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import _describe_shape, _reject_inexpressible_unit
from search.engine.retrieval.bound_values import (
    canonicalise_identity,
    identity_names_measure_family,
    refuse_source_outside_family,
)
from search.models import (
    IdentityBindingSet,
    IdentityBindingValue,
    IdentitySlot,
)


def _raw_bound_values_from_identity_slots(slots: Sequence[IdentitySlot]) -> dict[str, list[str]]:
    """Return each bound what/how property's literal dcids from the posted identity.

    Unbound properties are absent; this reader uses no index expansion or graph read."""
    values: dict[str, list[str]] = {}
    for slot in slots:
        binding = slot.binding
        if isinstance(binding, IdentityBindingValue):
            values[slot.property] = [binding.ref]
        elif isinstance(binding, IdentityBindingSet):
            values[slot.property] = list(binding.refs)
    return values


def _phase_validate(state: _PromoteState) -> None:
    """Validate the base and posted identity, then derive place roles.

    Require `base_identity`; standard shapes also require a StatVar seed, while named families
    elect membership and retain the seed only as an empty-election fallback. Verify the base
    hash, then read both identities in their canonical form (`canonicalise_identity`), so a
    pin posted bound and one posted unbound mint one spec_id and count as no edit. Refuse a
    provenance that does not publish its five-tuple (`refuse_source_outside_family`).
    Verify the shape, validate removals and unit facets, and check that every base-bound
    facet, role, and what/how slot is reposted or removed. Reject `exact` bindings on both
    identities. Failures raise `EngineInputError` unchanged."""
    t0 = now_ms()
    try:
        validate_edited_identity(state.inp.identity)
        if state.inp.base_identity is not None:
            check_no_exact_bindings(state.inp.base_identity, pointer="/input/base_identity")

        # The shape phase assigns the verified slug to variable_text.
        identity = state.inp.identity
        state.variable_text = _describe_shape(identity.shape)

        verify_base_identity(state.inp.base_spec_id, state.inp.base_identity)
        if state.inp.base_identity is None:
            detail = "standard resubmit requires a base_identity (alongside base_spec_id)"
            raise EngineInputError(
                detail,
                code="request_invalid",
                pointer="/input/base_identity",
                detail=detail,
            )
        # Each invariant of a dataset an identity names reads bound on both identities, as a
        # first answer mints it, so one family keeps one spec_id whichever door minted it.
        identity = canonicalise_identity(state.inp.identity)
        base_identity = canonicalise_identity(state.inp.base_identity)
        state.inp = state.inp.model_copy(
            update={"identity": identity, "base_identity": base_identity}
        )
        refuse_source_outside_family(identity, pointer="/input/identity/facets/provenance")
        # Named-family membership is elected from the packaged index.
        names_family = identity_names_measure_family(identity, get_measure_index().axes)
        posted_seed = state.inp.resolved_stat_var_dcids or []
        if not names_family and not posted_seed:
            # Standard shapes need a seed because the identity alone cannot recover membership.
            detail = "standard resubmit requires resolved_stat_var_dcids"
            raise EngineInputError(
                detail,
                code="request_invalid",
                pointer="/input/resolved_stat_var_dcids",
                detail=detail,
            )
        # Empty for a named family (_phase_reread_arcs elects membership). A
        # non-family shape has no election, so its posted seed is membership.
        state.stat_var_dcids = [] if names_family else posted_seed
        check_shape_unchanged(base_identity, identity.shape)
        removed_facets, removed_entities = validate_removed(
            state.inp.removed, state.inp.base_spec_id
        )
        _reject_inexpressible_unit(identity.facets)

        # The posted identity is the complete live definition; the base identity supplies
        # only the hash check and edit-accounting baseline.
        state.identity_entities = {role: list(dcids) for role, dcids in identity.entities.items()}
        destinations = state.identity_entities.get("destination", [])
        subjects = state.identity_entities.get("observationAbout", [])
        state.standard_where_dcids = place_order(destinations or subjects)
        state.entity_dcid = state.standard_where_dcids[0] if state.standard_where_dcids else None
        state.origin_dcids = state.identity_entities.get("origin", [])
        check_edit_incomplete(
            base_identity,
            posted_facets=identity.facets,
            posted_entity_roles=set(identity.entities),
            removed_facets=removed_facets,
            removed_entities=removed_entities,
        )
        # Reject a dropped base slot before graph reads; the posted identity defines this
        # resubmit's declared slot set.
        check_slots_declared(base_identity.slots, identity.slots)
    finally:
        state.timing["validate"] = now_ms() - t0
