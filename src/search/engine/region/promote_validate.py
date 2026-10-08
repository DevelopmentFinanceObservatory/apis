"""Validate the posted identity and read its bound values for standard resubmit."""

from __future__ import annotations

from search.engine.envelope import now_ms
from search.engine.identity_edit import refuse_over_place_bound, validate_edited_identity
from search.engine.region.place_roles import place_order
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import _describe_shape, _reject_inexpressible_unit
from search.engine.retrieval.bound_values import (
    canonicalise_identity,
    refuse_source_outside_family,
)


def _phase_validate(state: _PromoteState) -> None:
    """Validate the posted identity, then derive place roles.

    Read the identity in its canonical form (`canonicalise_identity`), so a pin posted
    bound and one posted unbound mint one spec_id. Refuse a provenance that does not
    publish its five-tuple (`refuse_source_outside_family`), an `exact` binding, an
    inexpressible unit request and an identity over the bound on the places a Spec binds
    (`refuse_over_place_bound`). Failures raise `EngineInputError` unchanged."""
    t0 = now_ms()
    try:
        validate_edited_identity(state.inp.identity)
        refuse_over_place_bound(state.inp.identity.entities)
        # The shape phase assigns the verified slug to variable_text.
        state.variable_text = _describe_shape(state.inp.identity.shape)
        # Each invariant of a dataset an identity names reads bound, as a first answer
        # mints it, so one family keeps one spec_id whichever door minted it.
        identity = canonicalise_identity(state.inp.identity)
        state.inp = state.inp.model_copy(update={"identity": identity})
        refuse_source_outside_family(identity, pointer="/input/identity/facets/provenance")
        _reject_inexpressible_unit(identity.facets)

        state.identity_entities = {role: list(dcids) for role, dcids in identity.entities.items()}
        destinations = state.identity_entities.get("destination", [])
        subjects = state.identity_entities.get("observationAbout", [])
        state.standard_where_dcids = place_order(destinations or subjects)
        state.entity_dcid = state.standard_where_dcids[0] if state.standard_where_dcids else None
        state.origin_dcids = state.identity_entities.get("origin", [])
    finally:
        state.timing["validate"] = now_ms() - t0
