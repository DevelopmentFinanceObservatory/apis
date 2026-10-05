"""Re-read member arcs and verify their shared five-tuple and shape."""

from __future__ import annotations

from typing import cast

from code_index.shared import get_index, get_measure_index
from materialise.graph import UNRESOLVED_NODE
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import _identity_slots_to_wire
from search.engine.region.result import RegionResult
from search.engine.retrieval.bound_values import (
    elect_measure_family,
    identity_query,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.derive import group_shape_id, member_group_class
from search.engine.retrieval.facts import _value_for, read_five_tuple
from search.engine.shape import identity_shape_for
from search.models import IdentityShapeSchema


def _family_measurement_qualifier(arcs: dict) -> str | None:
    """Return the qualifier value that the family arc reader will expose."""
    value = _value_for(arcs, "measurementQualifier")
    if value is not None:
        return value
    nodes = arcs.get("measurementQualifier", {}).get("nodes", [])
    return UNRESOLVED_NODE if nodes else None


def _phase_reread_arcs(state: _PromoteState) -> RegionResult | None:
    """Re-read member arcs, verify the shared five-tuple, and regenerate the shape.

    Named families elect members from the packaged index. An election that matches no row
    or refuses (the named dataset publishes no matching row, or several datasets match and
    none is the default) returns `no_data`, with or without a posted seed: the Datasets mint
    refuses the same identity, so no posted member stands in for the family. Drop missing
    elected members, but require every seeded member to resolve. Return `no_data` if no members remain. Mixed five-tuples raise
    `request_invalid`; a regenerated shape mismatch raises `shape_mismatch` (422). The shape
    slug supplies `variable_text`; errors propagate unchanged."""
    t0 = now_ms()
    try:
        if not state.stat_var_dcids:
            identity_shape = cast(IdentityShapeSchema, state.inp.identity.shape)
            synthetic_shape = synthetic_shape_for_identity(identity_shape)
            election = elect_measure_family(
                synthetic_shape,
                get_index(),
                get_measure_index().axes,
                identity_query(state.inp.identity),
            )
            if election is None:
                return _no_data_region(state.variable_text)
            state.stat_var_dcids = list(election.elected.members)
            state.stat_var_dcids_from_election = True
            # Record forced axes; bindings are rebuilt from the elected members' arcs.
            state.measure_elected_properties = frozenset(prop for prop, _value in election.forced)

        # Re-read arcs for every posted (or recovered) SV.
        arcs_map = state.graph.node_arcs_batch(state.stat_var_dcids)
        absent_svs = [sv for sv, arcs in arcs_map.items() if arcs is None]
        if absent_svs:
            if state.stat_var_dcids_from_election:
                # Drop absent elected members; a posted seed requires every member to resolve.
                state.stat_var_dcids = [sv for sv in state.stat_var_dcids if sv not in absent_svs]
                if not state.stat_var_dcids:
                    return _no_data_region(state.variable_text)
            else:
                return _no_data_region(state.variable_text)
        # Every remaining member has confirmed arcs.
        confirmed_arcs: dict[str, dict] = {sv: a for sv, a in arcs_map.items() if a is not None}
        state.confirmed_arcs = confirmed_arcs

        # Extract a five-tuple for every SV and require them to share one.
        five_tuples = {sv: read_five_tuple(confirmed_arcs[sv]) for sv in state.stat_var_dcids}
        unique_fts = set(five_tuples.values())
        if len(unique_fts) > 1:
            detail = (
                "base's resolved_stat_var_dcids span multiple five-tuple shapes; "
                "a candidate is one shape"
            )
            raise EngineInputError(
                detail,
                code="request_invalid",
                detail=detail,
            )

        ft = state.ft = next(iter(unique_fts))
        state.family_measurement_qualifiers = frozenset(
            _family_measurement_qualifier(arcs) for arcs in confirmed_arcs.values()
        )

        # Rebuild the identity shape from its five-tuple. Structural equality verifies the
        # identity. The shape id is the one derive gives the members' group, read from the
        # members alone (`derive.group_shape_id`), so a resubmit keeps its answer's id.
        classes = [member_group_class(ft, arcs) for arcs in confirmed_arcs.values()]
        regenerated_shape_id = state.regenerated_shape_id = group_shape_id(
            ft,
            role_bearing=any(role_bearing for role_bearing, _indexed in classes),
            indexed=all(indexed for _role_bearing, indexed in classes),
        )
        regenerated_shape = identity_shape_for(
            pop_type_dcid=ft.pop_type_dcid,
            meas_prop_dcid=ft.meas_prop_dcid,
            stat_type_dcid=ft.stat_type_dcid,
            meas_qual_dcid=ft.meas_qual_dcid,
            meas_denom_dcid=ft.meas_denom_dcid,
            meas_denom_unresolved=ft.meas_denom_unresolved,
        )

        if regenerated_shape != state.inp.identity.shape.model_dump(exclude_none=True):
            detail = "shape does not match the five-tuple derived from the base's resolved_stat_var_dcids"
            raise EngineInputError(
                detail,
                code="shape_mismatch",
                status=422,
                pointer="/input/identity/shape",
                detail=detail,
            )
        # Respell variable_text from the re-read draft's own display slug,
        # superseding the fallback rendering _phase_validate set.
        state.variable_text = regenerated_shape_id
        state.posted_slots = _identity_slots_to_wire(state.inp.identity.slots, confirmed_arcs)
        return None
    finally:
        state.timing["reread_arcs"] = now_ms() - t0
