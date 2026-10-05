"""``_resolve_standard_promote``'s fourth phase: resolve the answering StatVar
survivor set from the detected edits (region/resubmit_promote.py).
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from search.engine.envelope import now_ms
from search.engine.errors import EngineInfraError, FamilyNotResolvableError, RoleSignatureError
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.result import RegionResult
from search.engine.retrieval.bound_values import (
    identity_query,
    index_decides_membership,
    members_by_value,
    shape_member_facts,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.facts import read_constraints
from search.models import IdentityShapeSchema


def _narrowed_seed(state: _PromoteState) -> list[str]:
    """The posted members that answer the edited identity without a graph read: each
    carries, for every edited property, one of its edited values or a descendant of one,
    as a binding widens a value (`members_by_value`) and as the menu counted the option,
    and asserts every untouched bound property. Empty when an edited value is carried, with
    its descendants, by no posted member, or no posted member qualifies."""
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


def _elected_members(state: _PromoteState) -> list[str] | RegionResult:
    """The members of the family the posted identity resolves, as the Datasets mint elects
    them: the code index's family when it decides the identity, else the mint's graph walk
    (`RoleClassWalks.family`) seeded with the posted members. A walk that settles nothing
    returns `no_matching_slice`; role sets the election cannot order and a member at fault
    return the mint's refusals as `no_data`."""
    identity = state.inp.identity
    query = identity_query(identity)
    if index_decides_membership(identity):
        # A synthetic shape holds no posted members, so it must not read as an exact posted
        # set, which `shape_member_facts` would return as its (empty) members.
        shape = replace(
            synthetic_shape_for_identity(cast(IdentityShapeSchema, identity.shape)),
            sv_arc_facts_complete=False,
        )
        members, _index_sourced = shape_member_facts(shape, query)
        return list(members)
    if state.role_class_walks is None:
        raise EngineInfraError(
            "materialise graph client not initialised; cannot elect the members of an "
            "edited standard identity"
        )
    try:
        walked = state.role_class_walks.family(
            query, tuple(state.stat_var_dcids), deadline=state.deadline
        )
    except RoleSignatureError:
        return _no_data_region(state.variable_text, "role_signature_incompatible")
    except FamilyNotResolvableError:
        return _no_data_region(state.variable_text, "variable_not_resolved")
    if walked.value is None:
        return _no_data_region(state.variable_text, "no_matching_slice")
    return list(walked.value.group.family)


def _phase_resolve_survivors(state: _PromoteState) -> RegionResult | None:
    """Resolve the members that answer the posted identity.

    No edits keep the posted set. An edit the posted members answer alone keeps those that
    carry every edited value and assert every untouched bound property (`_narrowed_seed`).
    Any other edit elects the family of the edited identity on the server, as the Datasets
    mint does (`_elected_members`), so the answer does not depend on which members the
    client posted, and a menu option counted from members the posted set lacks is still
    reached. An elected family with no member whose arcs resolve returns
    `no_matching_slice`. Return `None` when the survivor set is stored on state."""
    t0 = now_ms()
    try:
        state.resolved_stat_var_dcids = state.stat_var_dcids
        state.resolved_arcs = state.confirmed_arcs
        if not state.edits:
            return None

        if narrowed := _narrowed_seed(state):
            state.resolved_stat_var_dcids = narrowed
            state.resolved_arcs = {sv: state.confirmed_arcs[sv] for sv in narrowed}
            return None

        elected = _elected_members(state)
        if isinstance(elected, RegionResult):
            return elected
        unread = [sv for sv in elected if sv not in state.confirmed_arcs]
        read = state.graph.node_arcs_batch(unread) if unread else {}
        arcs = {
            sv: found
            for sv in elected
            if (found := state.confirmed_arcs.get(sv) or read.get(sv)) is not None
        }
        if not arcs:
            return _no_data_region(state.variable_text, reason="no_matching_slice")
        state.resolved_stat_var_dcids = list(arcs)
        state.resolved_arcs = arcs
        return None
    finally:
        state.timing["resolve_survivors"] = now_ms() - t0
