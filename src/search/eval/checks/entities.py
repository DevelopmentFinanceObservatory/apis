"""``entities``: the exact set of (dcid, role) pairs the answer binds.

Reads ``definition.identity.entities`` (role to dcids) and, for an entity marked
``expands``, ``resolved.entity_groups``: a group read as members passes when the answer
records that group's node on its role and binds at least one member there. A group named by
``label`` matches a recorded group with no node (a null dcid) whose label is the same,
ignoring case and surrounding space. An unsupported answer has no ``resolved`` half and
records no group. The members on a group's role are left out of the exact comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import ExpectedEntity, StatusExpect
from search.eval.checks.answers import Answer, over_answers
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    all_of,
    fail,
    register_part,
    unconfirmed,
    unverifiable,
)


def _same_label(a: object, b: str) -> bool:
    return isinstance(a, str) and a.strip().casefold() == b.strip().casefold()


def _group(entity: ExpectedEntity, answer: Answer, ctx: ScoreContext) -> Result:
    role = str(entity.role)
    recorded = answer.read("resolved.entity_groups", ctx) if answer.is_spec else None
    group = (recorded or {}).get(role) or {}
    dcid = group.get("ref")
    if entity.label is not None:
        name = entity.label
        label = group.get("label")
        if dcid is not None or not _same_label(label, name):
            return fail(f"group {name!r} on {role}: the answer names group {dcid or label!r}")
    else:
        name = str(entity.dcid)
        if dcid != entity.dcid:
            return fail(f"group {name} on {role}: the answer names group {dcid!r}")
    if not (answer.identity("entities", ctx) or {}).get(role):
        return fail(f"group {name} on {role}: no member is bound")
    return PASS


def match_entities(expected: list[ExpectedEntity], answer: Answer, ctx: ScoreContext) -> Result:
    if any(unconfirmed(e.dcid, e.role) for e in expected):
        return unverifiable("entities are unconfirmed")
    groups = [e for e in expected if e.expands]
    group_roles = {str(e.role) for e in groups}
    bound = answer.identity("entities", ctx) or {}
    got = {
        (dcid, role) for role, dcids in bound.items() if role not in group_roles for dcid in dcids
    }
    want = {(str(e.dcid), str(e.role)) for e in expected if not e.expands}
    exact = (
        PASS
        if got == want
        else fail(f"entities: extra {sorted(got - want)}, missing {sorted(want - got)}")
    )
    return all_of([exact, *(_group(e, answer, ctx) for e in groups)])


@register_part("entities")
def entities(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The answer binds exactly the expected entities on their roles."""
    expected = block.entities
    if expected is None:
        return NA
    return over_answers(
        part, ctx, lambda answer: match_entities(expected, answer, ctx), reading=False
    )
