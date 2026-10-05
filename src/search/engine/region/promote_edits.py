"""Compare posted bindings with the base identity and record edits."""

from __future__ import annotations

from typing import cast

from code_index.shared import get_index
from search.engine.envelope import now_ms
from search.engine.region.promote_state import _PromoteState
from search.engine.region.promote_validate import _raw_bound_values_from_identity_slots
from search.engine.region.resubmit_seed import (
    _check_seed_matches_base_identity,
    _seed_effective_constraints,
    _seed_unresolved_props,
)
from search.engine.retrieval.bound_values import _expand_values
from search.engine.retrieval.facts import read_constraints
from search.models import BaseIdentity, BindingSet, BindingValue


def _phase_detect_edits(state: _PromoteState) -> None:
    """Compare posted bindings with the base and verify a client-posted seed.

    Record each posted StatVar's constraints, bound what/how values, edits, and unbound
    widenings. Seed edits compare literal values with the base, so rollup-to-leaf changes
    count; election-backed edits use descendant coverage because rollups are absent from
    member arcs. Check seed constraints against the base only for client-posted membership.
    Contradictions raise `request_invalid`."""
    t0 = now_ms()
    try:
        # Union constraints feed election coverage checks and survivor filtering.
        union_constraints: dict[str, set[str]] = {}
        state.union_constraints = union_constraints
        for sv in state.stat_var_dcids:
            sv_constraints = read_constraints(state.confirmed_arcs[sv])
            for prop, val in sv_constraints.items():
                union_constraints.setdefault(prop, set()).add(val)

        # The loop compares each posted binding with the base identity's literal value.
        base_identity = cast(BaseIdentity, state.inp.base_identity)
        base_bound_values = _raw_bound_values_from_identity_slots(base_identity.slots)

        # Literal comparison catches rollup/leaf edits in both directions; rollup identity
        # values correspond to leaves in member arcs. Where, when, source, and property-less
        # slots are outside edit detection.
        # Posted-unbound properties are widenings unless the family election forced them.
        edits: dict[str, list[str]] = {}
        state.edits = edits
        widened_props: set[str] = set()
        state.widened_props = widened_props
        # Every bound what/how property, except widened axes, mapped to its requested values.
        # The unsupported-property check below (after the edits block resolves the
        # survivor set) needs the untouched, non-edited props' own posted values
        # too, to route them through refine if no survivor asserts them.
        posted_bound_values: dict[str, list[str]] = {}
        state.posted_bound_values = posted_bound_values
        for slot in state.posted_slots:
            if slot.key.axis not in ("what", "how"):
                continue
            prop_ref = slot.key.property
            prop = prop_ref.dcid if prop_ref is not None else None
            if prop is None:
                continue
            binding = slot.binding
            if isinstance(binding, BindingValue) and binding.value.ref:
                requested = [binding.value.ref.dcid]
            elif isinstance(binding, BindingSet):
                requested = [v.ref.dcid for v in binding.values if v.ref is not None]
            else:
                # Unbound values widen the axis unless the family election forced it.
                if prop not in state.measure_elected_properties:
                    widened_props.add(prop)
                continue
            if not requested:
                continue
            posted_bound_values[prop] = requested
            if state.stat_var_dcids_from_election:
                # stat_var_dcids was selected by this exact posted value,
                # widened through the packaged index's descendant walk
                # (elect_measure_family, via identity_query).
                # Union constraints are guaranteed a subset of that widening.
                # Compare descendant coverage; rollup values appear as leaves in member arcs.
                covers = set(_expand_values(get_index(), requested))
                if not union_constraints.get(prop, set()) <= covers:
                    edits[prop] = requested
            elif set(requested) != set(base_bound_values.get(prop, [])):
                edits[prop] = requested

        # Hash and five-tuple checks establish the base identity, while seed constraints must
        # match its bound values. Widened axes are exempt because the request leaves them
        # unbound; election-backed members have no client seed.
        if not state.stat_var_dcids_from_election:
            seed_constraints = _seed_effective_constraints(
                state.stat_var_dcids, state.confirmed_arcs
            )
            seed_unresolved_props = _seed_unresolved_props(
                state.stat_var_dcids, state.confirmed_arcs
            )
            _check_seed_matches_base_identity(
                base_identity, seed_constraints, widened_props, seed_unresolved_props
            )
    finally:
        state.timing["detect_edits"] = now_ms() - t0
