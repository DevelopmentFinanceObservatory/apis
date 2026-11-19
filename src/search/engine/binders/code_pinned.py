"""Bind a part its series code alone names without a model call.

A part whose variable text writes only its series code (`source_code.names_only_code`)
has its StatVars fixed by the code (`ResolveState.source_code`). When the bind menu holds
one shape and the code's members agree on every constraint value, the model has nothing
to choose. The election is that shape, and every slot the members fix is bound to their
value after the bind (`bound_values.bind_member_values`), over whatever the model
drafted. `binds_on_code` checks those premises on the request, and `code_pinned_bind`
answers it through Jev's own assembler (`jev_assemble.bind_result`) as Jev would on
answers electing the one shape, so the house policies run once and the drafts are
grounded as on a model reply. The assembler binds the slots no answer names as it does
on a model reply: a fixed qualifier match bound, a one-member taxonomy bound as its sole
option, the rest unbound.
"""

from __future__ import annotations

from collections.abc import Collection

from search.engine.binders.base import BindRequest, BindResult
from search.engine.binders.jev_assemble import JevAnswers, bind_result, frame_stated_form
from search.engine.binders.jev_menu import jev_menu
from search.engine.retrieval.bound_values import member_constraint_values

CODE_PINNED_BINDER = "code"
"""The binder name a code-pinned bind states (`BindResult.binder`)."""

# Jev's answers electing the menu's one shape. They measure the phrase, and the shape is
# the one the query asks for (`jev_assemble._measure_scores`, `jev_assemble._election`).
_ELECT_THE_ONE_SHAPE = JevAnswers(nouls={"m|0": 1.0}, choices={"electq": {"Shape 0": 1.0}})


def binds_on_code(request: BindRequest, *, members: Collection[str]) -> bool:
    """Whether `request`, the bind call of a part whose variable text writes only its
    series code, whose StatVars are `members`, leaves the model nothing to decide. That
    holds when the menu has one shape, no reading is refused on the pinned part, no shape
    offers a tie, a candidate menu or a nominated word, the part drops no qualifier phrase
    (Jev asks on one whether it wants every value of a slot, `requested_all`), and the
    members agree on every constraint value (`members_agree`)."""
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
    leaves as Jev's assembler does on an answer naming no value there. The code names the
    series over every value of an open slot, as DT.DOD.DECT.CD's StatVars leave the debtor
    type and maturity open."""
    rows = member_constraint_values(members)
    return rows is not None and all(row == rows[0] for row in rows[1:])


def code_pinned_bind(request: BindRequest) -> BindResult:
    """The bind result electing `request`'s one menu shape, assembled as Jev assembles
    answers that elect it (`jev_assemble.bind_result`), under the binder name
    `CODE_PINNED_BINDER`. The house policies run once, over a pool of that shape."""
    result, _confidence = bind_result(
        jev_menu(request), _ELECT_THE_ONE_SHAPE, stated_form=frame_stated_form(request.frame)
    )
    return BindResult(
        output=result.output,
        binder=CODE_PINNED_BINDER,
        policy_pool=result.policy_pool,
        policy_choice=result.policy_choice,
    )
