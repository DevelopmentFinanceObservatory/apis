"""Read the posted identity's bound values against its elected members."""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.envelope import now_ms
from search.engine.region.promote_state import _PromoteState
from search.engine.retrieval.bound_values import _expand_values
from search.engine.retrieval.facts import read_constraints
from search.models import BindingSet, BindingValue


def _phase_detect_edits(state: _PromoteState) -> None:
    """Record the elected members' constraints and the identity's bound what/how values.

    Each bound property's requested values are kept. A property whose elected members
    assert a value outside the requested values and their descendants is an edit, and the
    survivor phase narrows the members to those carrying a requested value. Rollup values
    appear as leaves in member arcs, so coverage reads the descendant walk. A property
    posted unbound widens the axis, unless the family election forced it."""
    t0 = now_ms()
    try:
        union_constraints: dict[str, set[str]] = {}
        state.union_constraints = union_constraints
        for sv in state.stat_var_dcids:
            for prop, val in read_constraints(state.confirmed_arcs[sv]).items():
                union_constraints.setdefault(prop, set()).add(val)

        edits: dict[str, list[str]] = {}
        state.edits = edits
        widened_props: set[str] = set()
        state.widened_props = widened_props
        posted_bound_values: dict[str, list[str]] = {}
        state.posted_bound_values = posted_bound_values
        # Where, when, source, and property-less slots are outside edit detection.
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
                if prop not in state.measure_elected_properties:
                    widened_props.add(prop)
                continue
            if not requested:
                continue
            posted_bound_values[prop] = requested
            covers = set(_expand_values(get_index(), requested))
            if not union_constraints.get(prop, set()) <= covers:
                edits[prop] = requested
    finally:
        state.timing["detect_edits"] = now_ms() - t0
