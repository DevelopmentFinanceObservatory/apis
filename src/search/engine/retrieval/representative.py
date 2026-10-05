"""Select representatives that do not assert unbound disagreement values."""

from __future__ import annotations

from search.engine.retrieval.facts import read_constraints, read_unresolved_constraints
from search.engine.shape import ShapeDraft


def _shape_disagreement_props(shape_draft: ShapeDraft) -> set[str]:
    """Return constraint properties with multiple values across all members.

    Use the full group in ``sv_arc_facts``; place-scoped taxonomy can omit a
    member that carries a distinct value. Shared single values are fixed facts,
    while only properties with multiple values are open disagreement axes.
    """
    sv_arc_facts = shape_draft.sv_arc_facts or {}
    observed: dict[str, set[str]] = {}
    for arcs in sv_arc_facts.values():
        for prop_dcid, value_dcid in read_constraints(arcs).items():
            observed.setdefault(prop_dcid, set()).add(value_dcid)
    return {
        slot.property_dcid
        for slot in shape_draft.slot_keys
        if slot.property_dcid is not None and len(observed.get(slot.property_dcid, set())) > 1
    }


def _is_honest_representative(
    constraints: dict[str, str],
    disagreement_props: set[str],
    elected_bound_values: dict[str, list[str]],
    unresolved: frozenset[str] = frozenset(),
) -> bool:
    """Allow silence on disagreement axes and bound values that match.

    Any asserted value on an unbound axis is fabricated specificity. An
    unresolved arc also counts as an assertion, since the member is qualified
    even though its value has no dcid.
    """
    for prop in disagreement_props:
        if prop in unresolved:
            return False
        value = constraints.get(prop)
        if value is None:
            continue
        allowed = elected_bound_values.get(prop)
        if allowed is None or value not in allowed:
            return False
    return True


def shape_bare_representative(shape_draft: ShapeDraft) -> str | None:
    """Return an unqualified member, or ``None`` if none exists.

    A member qualifies when silent on every disagreement axis. This is used
    when no bind call ran for an alternative shape. Empty member groups also
    return ``None``.
    """
    sv_arc_facts = shape_draft.sv_arc_facts or {}
    if not sv_arc_facts:
        return None

    disagreement_props = _shape_disagreement_props(shape_draft)
    if not disagreement_props:
        return next(iter(sv_arc_facts))

    for sv_dcid, arcs in sv_arc_facts.items():
        if _is_honest_representative(
            read_constraints(arcs), disagreement_props, {}, read_unresolved_constraints(arcs)
        ):
            return sv_dcid
    return None


def shape_agreement_representative(
    shape_draft: ShapeDraft,
    elected_bound_values: dict[str, list[str]],
) -> str | None:
    """Choose a member that agrees with the elected shape without adding claims.

    On shared disagreement axes, explicit values must match the elected
    bindings. Every other disagreement axis must be silent. If no such member
    exists, or no shared axis is bound, return a bare representative; return
    ``None`` if the shape has no honest bare member. This also handles sibling
    shapes whose value domains differ from the elected shape's.
    """
    sv_arc_facts = shape_draft.sv_arc_facts or {}
    if not sv_arc_facts:
        return None

    disagreement_props = _shape_disagreement_props(shape_draft)
    if not disagreement_props:
        return next(iter(sv_arc_facts))

    shared = disagreement_props & set(elected_bound_values)
    if shared:
        for sv_dcid, arcs in sv_arc_facts.items():
            constraints = read_constraints(arcs)
            if shared & set(constraints) and _is_honest_representative(
                constraints,
                disagreement_props,
                elected_bound_values,
                read_unresolved_constraints(arcs),
            ):
                return sv_dcid

    for sv_dcid, arcs in sv_arc_facts.items():
        if _is_honest_representative(
            read_constraints(arcs), disagreement_props, {}, read_unresolved_constraints(arcs)
        ):
            return sv_dcid
    return None
