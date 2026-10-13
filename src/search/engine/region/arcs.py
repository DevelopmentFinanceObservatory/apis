"""Build standard-shape bindings and sibling StatVar candidates from graph facts.

Applicability helpers partition unbound what/how slots into absent, unresolved,
and unclassified states. Arc-dict and ``MemberFacts`` inputs share one
implementation. Sibling candidates require arc confirmation before use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import shape_population_arcs
from search.engine.retrieval.facts import read_constraints
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.shape import ShapeDraft
from search.models import Slot

if TYPE_CHECKING:
    from search.engine.retrieval.member_facts import MemberFacts


def standard_bindings_from_arcs(
    *,
    shape: ShapeDraft,
    rep_sv_dcid: str,
) -> list[SlotBindingDraft]:
    """Build bindings for constraint slots from the representative StatVar's arcs.

    Bound values produce ``kind="value"``. Missing values produce
    ``kind="absent"``. Skips when/source slots and returns ``[]`` when no
    constraint slots exist.
    """
    constraint_slots = [
        slot
        for slot in shape.slot_keys
        if slot.property_dcid is not None and slot.axis not in ("when", "source")
    ]
    if not constraint_slots:
        return []

    # The representative can be any member the shape stands for.
    rep_arcs = shape_population_arcs(shape).get(rep_sv_dcid, {})
    sv_constraints = read_constraints(rep_arcs)

    result: list[SlotBindingDraft] = []
    for slot in constraint_slots:
        prop = slot.property_dcid
        value = sv_constraints.get(prop)
        if value is not None:
            result.append(
                SlotBindingDraft(
                    axis=slot.axis,
                    property_dcid=prop,
                    kind="value",
                    value_dcids=[value],
                )
            )
        else:
            result.append(
                SlotBindingDraft(
                    axis=slot.axis,
                    property_dcid=prop,
                    kind="absent",
                    value_dcids=[],
                )
            )

    return result


def _declared_props_from_facts(
    member_facts: Mapping[str, MemberFacts],
    sv_dcids: Sequence[str],
) -> set[str]:
    """Collect resolved and unresolved constraint properties from each member.

    Missing members contribute nothing.
    """
    declared: set[str] = set()
    for sv in sv_dcids:
        member = member_facts.get(sv)
        if member is None:
            continue
        declared.update(member.read_constraints())
        declared.update(member.read_unresolved_constraints())
    return declared


def declared_props_from_arcs(
    *,
    sv_dcids: list[str],
    arcs_by_sv: Mapping[str, dict | None],
) -> set[str]:
    """Return properties any listed member declares, resolved or unresolved.

    Missing or ``None`` arc entries contribute nothing. This wrapper reads
    the resubmit path's raw arcs, scoped to its posted StatVars.
    """
    member_facts = {sv: ArcsMemberFacts(arcs=a) for sv, a in arcs_by_sv.items() if a is not None}
    return _declared_props_from_facts(member_facts, sv_dcids)


def inapplicable_slots_from_arcs(
    *,
    slots: list[Slot],
    sv_dcids: list[str],
    arcs_by_sv: Mapping[str, dict | None],
) -> list[str]:
    """Return unbound what/how properties no resolved member declares.

    The wire represents absent constraints and deliberate widens as unbound.
    Confirmed arcs distinguish the two.
    The resubmit path supplies confirmed arcs, fetched in one bounded batch when
    needed. A property is inapplicable only when no member declares it, resolved
    or unresolved. Bound slots and where/when/source slots are excluded. Results
    follow ``slots`` order. Missing arc entries contribute nothing.

    Pass the grounded ``slots``, so a property reported here is also unbound on the wire.
    The slot keys may include a base-declared property absent from every
    resolved member. Such a property is unbound and inapplicable. A widened
    property declared by a member is unbound but applicable.
    """
    member_facts = {sv: ArcsMemberFacts(arcs=a) for sv, a in arcs_by_sv.items() if a is not None}
    return _inapplicable_slots_from_facts(slots=slots, sv_dcids=sv_dcids, member_facts=member_facts)


def _inapplicable_slots_from_facts(
    *,
    slots: list[Slot],
    sv_dcids: Sequence[str],
    member_facts: Mapping[str, MemberFacts],
) -> list[str]:
    """Shared implementation for raw-arc and ``MemberFacts`` inputs."""
    applies = _declared_props_from_facts(member_facts, sv_dcids)
    result: list[str] = []
    for slot in slots:
        if slot.key.axis not in ("what", "how") or slot.key.property is None:
            continue
        if slot.binding.kind != "unbound":
            continue
        prop = slot.key.property.ref
        if prop not in applies:
            result.append(prop)
    return result


def inapplicable_slots_from_member_facts(
    *,
    slots: list[Slot],
    sv_dcids: list[str],
    member_facts: Mapping[str, MemberFacts],
) -> list[str]:
    """Return inapplicable slots from a resolve's ``MemberFacts``."""
    return _inapplicable_slots_from_facts(slots=slots, sv_dcids=sv_dcids, member_facts=member_facts)


def unresolved_slots_from_arcs(
    *,
    slots: list[Slot],
    sv_dcids: list[str],
    arcs_by_sv: Mapping[str, dict | None],
) -> list[str]:
    """Return unbound what/how properties with unresolved values on any member.

    The graph asserts these properties but cannot resolve their values.
    Resolved and unresolved declarations both count as applicable. Results
    are excluded from ``inapplicable_slots_from_arcs``. Missing arc entries,
    bound slots, and non-what/how slots are ignored.
    """
    member_facts = {sv: ArcsMemberFacts(arcs=a) for sv, a in arcs_by_sv.items() if a is not None}
    return _unresolved_slots_from_facts(slots=slots, sv_dcids=sv_dcids, member_facts=member_facts)


def _unresolved_slots_from_facts(
    *,
    slots: list[Slot],
    sv_dcids: Sequence[str],
    member_facts: Mapping[str, MemberFacts],
) -> list[str]:
    """Shared implementation for raw-arc and ``MemberFacts`` inputs."""
    unresolved: set[str] = set()
    for sv in sv_dcids:
        member = member_facts.get(sv)
        if member is None:
            continue
        unresolved.update(member.read_unresolved_constraints())
    result: list[str] = []
    for slot in slots:
        if slot.key.axis not in ("what", "how") or slot.key.property is None:
            continue
        if slot.binding.kind != "unbound":
            continue
        prop = slot.key.property.ref
        if prop in unresolved:
            result.append(prop)
    return result


def unresolved_slots_from_member_facts(
    *,
    slots: list[Slot],
    sv_dcids: list[str],
    member_facts: Mapping[str, MemberFacts],
) -> list[str]:
    """Return unresolved slots from a resolve's ``MemberFacts``."""
    return _unresolved_slots_from_facts(slots=slots, sv_dcids=sv_dcids, member_facts=member_facts)
