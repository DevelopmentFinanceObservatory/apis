"""``_resolve_standard_promote``'s fourth phase: resolve the answering StatVar
survivor set from the elected members (region/resubmit_promote.py).
"""

from __future__ import annotations

from search.engine.envelope import now_ms
from search.engine.region.promote_state import _PromoteState
from search.engine.region.result import RegionResult
from search.engine.retrieval.bound_values import members_by_value
from search.engine.retrieval.facts import read_constraints


def _narrowed_members(state: _PromoteState) -> list[str]:
    """The elected members that carry, for every edited property, one of its requested
    values or a descendant of one, as a binding widens a value (`members_by_value`) and as
    the menu counted the option, and assert every untouched bound property. Empty when a
    requested value is carried, with its descendants, by no member, or no member
    qualifies."""
    constraints = {sv: read_constraints(state.confirmed_arcs[sv]) for sv in state.stat_var_dcids}
    admitted = set(constraints)
    for prop, vals in state.edits.items():
        by_value = members_by_value({sv: c.get(prop) for sv, c in constraints.items()}, vals)
        if not all(by_value.values()):
            return []
        admitted &= {sv for svs in by_value.values() for sv in svs}
    untouched = [
        prop
        for prop in state.posted_bound_values
        if prop not in state.edits and prop not in state.widened_props
    ]
    return [
        sv
        for sv in state.stat_var_dcids
        if sv in admitted and all(prop in constraints[sv] for prop in untouched)
    ]


def _phase_resolve_survivors(state: _PromoteState) -> RegionResult | None:
    """Resolve the members that answer the posted identity.

    These are the elected members, narrowed to those carrying a requested value when some
    member asserts a value outside them (`_narrowed_members`). Return `None` once the
    survivor set is stored on state."""
    t0 = now_ms()
    try:
        state.resolved_stat_var_dcids = state.stat_var_dcids
        state.resolved_arcs = state.confirmed_arcs
        if state.edits and (narrowed := _narrowed_members(state)):
            state.resolved_stat_var_dcids = narrowed
            state.resolved_arcs = {sv: state.confirmed_arcs[sv] for sv in narrowed}
        return None
    finally:
        state.timing["resolve_survivors"] = now_ms() - t0
