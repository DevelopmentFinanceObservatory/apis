"""Bind a part its series code alone names without a model call.

A part whose variable text writes only its series code (`source_code.names_only_code`)
has its StatVars fixed by the code (`ResolveState.source_code`). When the bind menu holds
one shape and the code's members agree on every constraint value, the model has nothing
to choose. The election is that shape, and every slot the members fix is bound to their
value after the bind (`bound_values.bind_member_values`), over whatever the bind drafted.
`binds_on_code` checks those premises on the request, and `code_pinned_bind` drafts each
listed slot from what fixes it before any reading of the phrase
(`assembly.anchored_draft`): a fixed qualifier match bound, a one-member taxonomy bound as
its sole option, the rest unbound. It grounds the drafts as every binder's are
(`bind.ground_bind_output`), runs the house policies once over a pool of the one shape,
and assembles the result as the model binders do (`assembly.elected_result`).
"""

from __future__ import annotations

from collections.abc import Collection

from search.engine.bind import (
    SlotBindingDraft,
    _listed_slots,
    _qualifier_menu,
    _UniversalBindOutput,
    ground_bind_output,
)
from search.engine.binders.assembly import anchored_draft, elected_result
from search.engine.binders.base import BindRequest, BindResult
from search.engine.binders.house_policy import PolicyPool
from search.engine.retrieval.bound_values import member_constraint_values

CODE_PINNED_BINDER = "code"
"""The binder name a code-pinned bind states (`BindResult.binder`)."""


def binds_on_code(request: BindRequest, *, members: Collection[str]) -> bool:
    """Whether `request`, the bind call of a part whose variable text writes only its
    series code, whose StatVars are `members`, leaves the model nothing to decide. That
    holds when the menu has one shape, no reading is refused on the pinned part, no shape
    offers a tie, a candidate menu or a nominated word, the part drops no qualifier phrase
    (a model would read one as asking for every value of a slot), and the members agree on
    every constraint value (`members_agree`)."""
    return (
        len(request.shapes) == 1
        and not request.refused
        and not any(request.admitted_ties)
        and not any(request.candidate_menus)
        and not any(request.nominated_words)
        and not request.dropped_qualifier_phrases
        and members_agree(members)
    )


def members_agree(members: Collection[str]) -> bool:
    """Whether the code index holds a row for each StatVar of `members` and every row
    states the same constraint values (`bound_values.member_constraint_values`). Each slot
    of the code's shape is then one the members fix, which `bound_values.bind_member_values`
    binds to their value over any draft, or one every member leaves open, which the bind
    leaves unbound unless a qualifier match or a one-member taxonomy fixes it. The code
    names the series over every value of an open slot, as DT.DOD.DECT.CD's StatVars leave
    the debtor type and maturity open."""
    rows = member_constraint_values(members)
    return rows is not None and all(row == rows[0] for row in rows[1:])


def code_pinned_bind(request: BindRequest) -> BindResult:
    """The bind result electing `request`'s one menu shape under the binder name
    `CODE_PINNED_BINDER`, with no election score. Each slot the bind prompt lists
    (`bind._listed_slots`) gets the draft its fixed qualifier matches and taxonomy give
    (`assembly.anchored_draft`), unbound where they give none. The house policies run once,
    over a pool of that shape."""
    (shape,) = request.shapes
    matches = request.admitted_qualifiers[0] if request.admitted_qualifiers else ()
    fixed = _qualifier_menu(shape, tuple(matches), (), request.withheld_measure_axes).fixed
    drafts: list[SlotBindingDraft] = []
    for slot in _listed_slots(shape, request.withheld_measure_axes):
        assert slot.property_dcid is not None  # `_listed_slots` keeps none without one
        anchors = [m for m in fixed if m.code_system == slot.property_dcid]
        drafted = None
        if slot.axis != "source":
            drafted = anchored_draft(shape, slot.axis, slot.property_dcid, anchors)
        drafts.append(
            drafted
            or SlotBindingDraft(axis=slot.axis, property_dcid=slot.property_dcid, kind="unbound")
        )
    grounded = ground_bind_output(
        _UniversalBindOutput(chosen_shape_index=0, bindings=drafts), [shape]
    )
    pool = PolicyPool(measuring=(0,))
    choice = request.apply_policies(pool, holds={0: True}, drafts=grounded.bindings)
    return elected_result(
        request,
        elected=0,
        drafts=grounded.bindings,
        binder=CODE_PINNED_BINDER,
        pool=pool,
        choice=choice,
    )
