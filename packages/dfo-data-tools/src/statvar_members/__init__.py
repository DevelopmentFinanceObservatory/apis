"""Enumerate StatVar members for a five-tuple and slot bindings.

The enumerator follows reverse arcs, applies slot constraints, then returns the
minimal-constraint antichain. `measured_property` anchors the query. This package
defines its own structural graph protocol, independent of `materialise`."""

from __future__ import annotations

from statvar_members.cache import MemberEnumerator
from statvar_members.digest import member_digest
from statvar_members.enumerate import (
    CANDIDATE_CAP,
    DEFAULT_MEMBER_CAP,
    POSTED_MEMBER_CAP,
    DegenerateObservationProperties,
    Member,
    MemberEnumerationTooLarge,
    MemberGroup,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    RoleClassElection,
    SlotBinding,
    antichain_minimal_ids,
    constraints_resolve,
    covers_role_keys,
    declares_only_placing_roles,
    dominates_constraint_keys,
    elect_role_class,
    enumerate_members,
    five_tuple_arcs_match,
    five_tuple_from_identity,
    five_tuple_reverse_filters,
    root_can_prove_complete,
    slot_bindings_from_identity,
)
from statvar_members.types import (
    UNRESOLVED_NODE,
    AbsentReading,
    FiveTuple,
    MemberArcReader,
    SubjectPage,
)

__all__ = [
    "CANDIDATE_CAP",
    "DEFAULT_MEMBER_CAP",
    "POSTED_MEMBER_CAP",
    "UNRESOLVED_NODE",
    "AbsentReading",
    "DegenerateObservationProperties",
    "FiveTuple",
    "Member",
    "MemberArcReader",
    "MemberEnumerationTooLarge",
    "MemberEnumerator",
    "MemberGroup",
    "MemberQueryUnanchored",
    "MemberRoleDisagreement",
    "MultiValuedConstraint",
    "NotAStatisticalVariable",
    "RoleClassElection",
    "SlotBinding",
    "SubjectPage",
    "antichain_minimal_ids",
    "constraints_resolve",
    "covers_role_keys",
    "declares_only_placing_roles",
    "dominates_constraint_keys",
    "elect_role_class",
    "enumerate_members",
    "five_tuple_arcs_match",
    "five_tuple_from_identity",
    "five_tuple_reverse_filters",
    "member_digest",
    "root_can_prove_complete",
    "slot_bindings_from_identity",
]
