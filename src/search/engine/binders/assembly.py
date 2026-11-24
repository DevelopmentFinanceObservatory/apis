"""The bind result every binder assembles once a shape is elected.

`anchored_draft` is the draft a slot's admitted qualifier matches and its taxonomy fix
before any reading of the phrase. `elected_result` turns an elected menu shape and its
grounded slot drafts (`bind.ground_bind_output`) into the `BindResult` the engine reads,
holding the election to the shape the house policies chose. The Jev and Gemini binders
call it on a reply that elects a shape, and the code-pinned bind
(`code_pinned.code_pinned_bind`) on the one shape a series code fixes.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from search.engine.bind import DraftBindingEvidence, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders.base import BindRequest, BindResult, PolicyContractError

if TYPE_CHECKING:
    from search.engine.binders.house_policy import PolicyChoice, PolicyPool
    from search.engine.qualifier_match import QualifierMatch
    from search.engine.shape import ShapeDraft
    from search.models import Axis


def anchored_draft(
    shape: ShapeDraft,
    axis: Axis,
    property_dcid: str,
    anchors: Sequence[QualifierMatch],
    *,
    named: Collection[str] = (),
) -> SlotBindingDraft | None:
    """The draft of `shape`'s slot on `property_dcid` that its `anchors` (the admitted
    qualifier matches binding the slot) and its taxonomy fix, None when neither does.

    A one-member taxonomy binds its member when the anchors name it or none: ``mentioned``
    when the anchors set it or the phrase names it (`named`), else ``sole_option``.
    Otherwise the anchors' codes bind, a value for one and a set for more."""
    forced = list(dict.fromkeys(m.code_dcid for m in anchors))
    taxonomy = (shape.slot_taxonomy or {}).get(f"{axis}:{property_dcid}", [])
    if len(taxonomy) == 1 and forced in ([], taxonomy):
        evidence: DraftBindingEvidence = (
            "mentioned" if anchors or taxonomy[0] in named else "sole_option"
        )
        return SlotBindingDraft(
            axis=axis,
            property_dcid=property_dcid,
            kind="value",
            value_dcids=list(taxonomy),
            evidence=evidence,
        )
    if forced:
        return SlotBindingDraft(
            axis=axis,
            property_dcid=property_dcid,
            kind="value" if len(forced) == 1 else "set",
            value_dcids=forced,
        )
    return None


def elected_result(
    request: BindRequest,
    *,
    elected: int,
    drafts: Sequence[SlotBindingDraft],
    binder: str,
    pool: PolicyPool,
    choice: PolicyChoice,
    reply: _UniversalBindOutput | None = None,
) -> BindResult:
    """The result of `binder` electing `request`'s menu shape `elected` with `drafts`, its
    slot drafts with the engine's rules already applied (`bind.ground_bind_output`), from
    the house policies' `pool` and `choice`.

    `reply` is a model's reply in the bind schema, whose other fields (`also_plausible`,
    `unplaced`, `measuring`) the result keeps; None for a bind no model wrote. Raises
    `PolicyContractError` when `elected` is not the shape the policies chose."""
    if elected != choice.survivor or not 0 <= elected < len(request.shapes):
        raise PolicyContractError(
            f"{binder} elected menu shape {elected}, the house policies chose {choice.survivor}"
        )
    base = reply if reply is not None else _UniversalBindOutput(chosen_shape_index=elected)
    output = base.model_copy(update={"chosen_shape_index": elected, "bindings": list(drafts)})
    return BindResult(output=output, binder=binder, policy_pool=pool, policy_choice=choice)
