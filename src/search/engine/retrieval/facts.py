"""Read shape identity, constraint, and role facts from fetched graph arcs.

An answer the catalogue's tables give is a read-only `FrozenArcs` shared by every request
of its snapshot (`origin_catalogue.frozen_arcs`), so each reader here that returns an
immutable value parses such an answer once and keeps the result on it
(`FrozenArcs.derived`). Every phase of a resolve that reads a member's facts then reads
the kept value. A live answer is parsed on every call.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from dc_graph import label_or_self_name
from materialise.roles import declared_roles, roles_by_axis
from origin_catalogue.frozen_arcs import FrozenArcs
from search.engine.errors import RoleSignatureError

if TYPE_CHECKING:
    from search.engine.retrieval.member_facts import MemberFacts
    from search.engine.shape import SlotKeyDraft

_OBSERVATION_ABOUT = "observationAbout"


class FiveTuple(NamedTuple):
    """The five-tuple that identifies a shape group."""

    pop_type_dcid: str
    meas_prop_dcid: str
    stat_type_dcid: str
    meas_qual_dcid: str | None
    meas_denom_dcid: str | None
    # Distinguishes an absent denominator from a dangling ProvisionalNode,
    # including one with a dcid. The dcid remains available for grouping.
    meas_denom_unresolved: bool = False


def _kept_on_answer[T](read: Callable[[dict], T]) -> Callable[[dict], T]:
    """`read`, whose result on a `FrozenArcs` is computed once and kept on it. `read` must
    return an immutable value."""

    @functools.wraps(read)
    def kept(arcs: dict) -> T:
        if isinstance(arcs, FrozenArcs):
            return arcs.derived(read)
        return read(arcs)

    return kept


def _dcids_for(arcs: dict, prop: str) -> list[str]:
    """Extract dcid values from an arcs entry."""
    nodes = arcs.get(prop, {}).get("nodes", [])
    return [n["dcid"] for n in nodes if "dcid" in n]


def read_name(arcs: dict, dcid: str) -> str | None:
    """Return the label of node `dcid` from its already-fetched `->name` arcs as the graph
    client's `node_label` reads it (`label_or_self_name`), without another graph call."""
    names = [value for n in arcs.get("name", {}).get("nodes", []) if (value := n.get("value"))]
    return label_or_self_name(names, dcid)


def _value_for(arcs: dict, prop: str) -> str | None:
    """Extract a single string value from an arcs entry (for literal props)."""
    nodes = arcs.get(prop, {}).get("nodes", [])
    for n in nodes:
        if "value" in n:
            return n["value"]
    dcids = _dcids_for(arcs, prop)
    return dcids[0] if dcids else None


def _is_unresolved(arcs: dict, prop: str) -> bool:
    """True when the arc has nodes but none yields a value or dcid."""
    nodes = arcs.get(prop, {}).get("nodes", [])
    return bool(nodes) and _value_for(arcs, prop) is None


def _is_provisional_target(arcs: dict, prop: str) -> bool:
    """True when every arc target has exactly the ProvisionalNode type.

    A ProvisionalNode target may carry a dangling dcid. The exact-type check
    avoids treating a target with an additional real type as unresolved. This
    applies to denominators and qualifiers, whose dangling targets affect shape
    semantics. Constraint values with dcids remain resolved and filterable.
    """
    nodes = arcs.get(prop, {}).get("nodes", [])
    return bool(nodes) and all(n.get("types") == ["ProvisionalNode"] for n in nodes)


@_kept_on_answer
def read_five_tuple(sv_arcs: dict) -> FiveTuple:
    """Read the five shape dimensions from fetched arcs.

    Missing qualifiers and denominators are ``None``. A ProvisionalNode
    qualifier is also ``None``. An unresolved denominator sets
    ``meas_denom_unresolved``, which can coexist with its dcid for grouping.
    """
    pop_type = _value_for(sv_arcs, "populationType") or ""
    meas_prop = _value_for(sv_arcs, "measuredProperty") or ""
    stat_type = _value_for(sv_arcs, "statType") or ""
    meas_qual = (
        None
        if _is_provisional_target(sv_arcs, "measurementQualifier")
        else _value_for(sv_arcs, "measurementQualifier")
    )
    meas_denom = _value_for(sv_arcs, "measurementDenominator")
    meas_denom_unresolved = _is_unresolved(
        sv_arcs, "measurementDenominator"
    ) or _is_provisional_target(sv_arcs, "measurementDenominator")
    return FiveTuple(
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid=stat_type,
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=meas_denom,
        meas_denom_unresolved=meas_denom_unresolved,
    )


def standard_shape_id_for(five_tuple: FiveTuple) -> str:
    """Build the display slug. Identity uses the separate shape union member.

    Denominators map to no suffix when absent, ``_per_unresolved`` when
    unresolved, or ``_per_<dcid>`` when resolved. Unresolved takes precedence
    over a dcid, keeping dangling per-capita shapes distinct from aggregates.
    """
    if five_tuple.meas_denom_unresolved:
        denom_suffix = "_per_unresolved"
    elif five_tuple.meas_denom_dcid:
        denom_suffix = f"_per_{five_tuple.meas_denom_dcid}"
    else:
        denom_suffix = ""
    return (
        f"{five_tuple.pop_type_dcid}_{five_tuple.meas_prop_dcid}"
        f"_{five_tuple.stat_type_dcid}"
        + (f"_{five_tuple.meas_qual_dcid}" if five_tuple.meas_qual_dcid else "")
        + denom_suffix
    ).lower()


def read_constraints(sv_arcs: dict) -> dict[str, str]:
    """Map declared constraint properties to values, omitting unresolved arcs.

    A dcid-bearing ProvisionalNode remains a filterable value. Grounding checks
    the elected shape's taxonomy before echoing it. Each call returns a dict of its own.
    """
    return dict(_constraint_pairs(sv_arcs))


@_kept_on_answer
def _constraint_pairs(sv_arcs: dict) -> tuple[tuple[str, str], ...]:
    """`read_constraints` as (property, value) pairs, in declaration order."""
    result: dict[str, str] = {}
    for prop_dcid in _dcids_for(sv_arcs, "constraintProperties"):
        value = _value_for(sv_arcs, prop_dcid)
        if value is not None:
            result[prop_dcid] = value
    return tuple(result.items())


@_kept_on_answer
def read_declared_constraint_properties(sv_arcs: dict) -> tuple[str, ...]:
    """Return the constraint properties a member lists in `constraintProperties`, whether
    or not it carries a value arc for each."""
    return tuple(_dcids_for(sv_arcs, "constraintProperties"))


@_kept_on_answer
def read_unresolved_constraints(sv_arcs: dict) -> frozenset[str]:
    """Return properties with asserted arcs whose values lack dcids.

    Absent properties are omitted. ``read_constraints`` also omits unresolved
    values so presence checks cannot mistake them for bound values. Consumers
    treat this set as distinct from both a match and silence.
    """
    constraint_prop_dcids = _dcids_for(sv_arcs, "constraintProperties")
    return frozenset(p for p in constraint_prop_dcids if _is_unresolved(sv_arcs, p))


class GroupRoles(NamedTuple):
    """The roles a five-tuple group's members declare."""

    # Every role some member declares.
    declared: tuple[str, ...]
    # The declared roles some member does not declare.
    optional: frozenset[str]


def group_roles(arcs_by_sv: Mapping[str, dict]) -> GroupRoles:
    """Return the union of a five-tuple group's declared roles.

    Members that declare one role set, compared without regard to order, give
    it in the first member's arc order, with no optional role. Members that
    differ, such as a total and its by-creditor breakdown, give the union in the
    widest member's arc order, so input order does not change it; roles outside
    some member are optional, and the role class the query's bound roles elect
    (`statvar_members.elect_role_class`) picks the members that answer. An empty
    mapping gives ``("observationAbout",)``.

    Raises ``RoleSignatureError`` when no single axis mapping covers the group:
    a subject-only member beside a role-bearing one, or two declared roles on
    one flow axis. Delegated role parsing can raise
    ``DegenerateObservationProperties``.
    """
    roles_by_sv = {sv: read_declared_roles(arcs) for sv, arcs in arcs_by_sv.items()}
    optional: frozenset[str] = frozenset()
    if len({frozenset(roles) for roles in roles_by_sv.values()}) <= 1:
        declared = next(iter(roles_by_sv.values()), (_OBSERVATION_ABOUT,))
    else:
        if any(roles == (_OBSERVATION_ABOUT,) for roles in roles_by_sv.values()):
            raise RoleSignatureError(
                "group mixes subject-only and role-bearing members: "
                f"{dict(sorted(roles_by_sv.items()))!r}"
            )
        # Widest member first, so the union's order does not depend on member order.
        ordered = sorted(roles_by_sv.values(), key=lambda roles: (-len(roles), roles))
        declared = tuple(dict.fromkeys(role for roles in ordered for role in roles))
        shared = frozenset.intersection(*(frozenset(roles) for roles in roles_by_sv.values()))
        optional = frozenset(declared) - shared
    try:
        roles_by_axis(declared)
    except ValueError as exc:
        raise RoleSignatureError(f"group roles cannot share one axis mapping: {exc}") from exc
    return GroupRoles(declared=declared, optional=optional)


@_kept_on_answer
def read_declared_roles(sv_arcs: dict) -> tuple[str, ...]:
    """Read declared roles from fetched arcs via the shared role parser.

    An absent or empty arc defaults to ``("observationAbout",)``. A non-empty
    arc that names ``observationAbout`` raises ``DegenerateObservationProperties``.
    """
    return declared_roles(tuple(_dcids_for(sv_arcs, "observationProperties")))


def read_slot_taxonomy(
    *,
    slot_keys: Sequence[SlotKeyDraft],
    member_facts: Mapping[str, MemberFacts],
) -> dict[str, list[str]]:
    """Build bindable constraint values for the caller-selected members.

    For each slot with a property dcid, values are deduplicated in first-seen
    order across ``member_facts``. The caller chooses the source and scope.
    ``core.py`` adds the recipient where-slot after entity resolution.
    """
    # Values keep first-seen order; a dict per property dedupes them in constant time.
    observed: dict[str, dict[str, None]] = {}
    for member in member_facts.values():
        for prop_dcid, value_dcid in member.read_constraints().items():
            observed.setdefault(prop_dcid, {})[value_dcid] = None

    taxonomy = {}
    for slot in slot_keys:
        prop = slot.property_dcid
        if prop is None:
            continue
        values = observed.get(prop)
        if values:
            taxonomy[f"{slot.axis}:{prop}"] = list(values)

    return taxonomy
