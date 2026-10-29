"""The checks every binder runs to move an election: whether a shape holds the values a
reply drafted on another (`holds_mentioned_values`), and the reply re-drafted onto the shape
the house policies elect (`reelect`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from search.engine.bind import (
    SlotBindingDraft,
    _grounded_in_taxonomy,
    _UniversalBindOutput,
    _without_partly_stated_values,
    covered_leaves,
    reads_bound_values_alike,
)
from search.engine.bind_repair import _label_named_in_text
from search.engine.binders.base import PolicyContractError

if TYPE_CHECKING:
    from search.engine.binders.house_policy import PolicyChoice
    from search.engine.qualifier_match import QualifierMatch
    from search.engine.shape import ShapeDraft


def holds_mentioned_values(
    chosen: ShapeDraft, shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> bool:
    """Whether `shape` holds each value of `bindings`, drafted on `chosen`, that the phrase
    names (`holds_drafted_values` over `mentioned_drafts`), and every value `chosen`'s slot
    holds for each slot `bindings` ask for in full (`holds_requested_slots`), so "across all
    flows" keeps every flow type."""
    return holds_drafted_values(chosen, shape, mentioned_drafts(bindings)) and (
        holds_requested_slots(chosen, shape, bindings)
    )


def holds_requested_slots(
    chosen: ShapeDraft, shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]
) -> bool:
    """Whether, for each ``requested_all`` draft of `bindings`, `shape`'s slot on its
    property covers every leaf code `chosen`'s slot covers (`bind.covered_leaves`)."""
    own, held = chosen.slot_taxonomy or {}, shape.slot_taxonomy or {}
    keys = [f"{d.axis}:{d.property_dcid}" for d in bindings if d.evidence == "requested_all"]
    return all(
        covered_leaves(own.get(key, ())) <= covered_leaves(held.get(key, ())) for key in keys
    )


def holds_drafted_values(
    chosen: ShapeDraft, shape: ShapeDraft, drafts: Sequence[SlotBindingDraft]
) -> bool:
    """Whether `shape` holds each value `drafts` bind on `chosen` as `chosen` does: it has
    a slot for each drafted property, and each value applies to the same taxonomy codes of
    its slot as of `chosen`'s (`bind.reads_bound_values_alike`)."""
    return {draft.property_dcid for draft in drafts} <= _slot_property_dcids(
        shape
    ) and reads_bound_values_alike(chosen, shape, drafts)


def reelect(
    output: _UniversalBindOutput,
    choice: PolicyChoice,
    shapes: Sequence[ShapeDraft],
    admitted_qualifiers: Sequence[Sequence[QualifierMatch]] = (),
) -> _UniversalBindOutput:
    """`output` electing the shape the house policies chose (`choice`), without the shapes
    they set aside in `also_plausible`.

    The reply drafted its bindings and judged its `unplaced` phrases on the shape it
    elected. On a move, a ``sole_option`` draft leaves unless the new shape's slot holds
    its value, since it bound only because the old slot held one member. The rest are
    grounded in the new shape's taxonomy (`bind._grounded_in_taxonomy`,
    `bind._without_partly_stated_values`). The pool admitted the new shape only if it
    holds each value the phrase names (`holds_mentioned_values`), and a move to one that
    does not raises `PolicyContractError`. An unplaced phrase leaves `unplaced` when the
    new shape admits a qualifier read from its words (`admitted_qualifiers`, each menu
    shape's own, as the bind request carries them), since that shape's vocabulary reads it.
    An output the policies leave as it was is returned as it is."""
    elected = choice.survivor
    old = output.chosen_shape_index
    moved = elected != old
    if not moved and not choice.dropped:
        return output
    update: dict[str, object] = {"chosen_shape_index": elected}
    if output.also_plausible is not None:
        update["also_plausible"] = [
            index
            for index in output.also_plausible
            if index != elected and index not in choice.dropped
        ]
    if moved:
        shape = shapes[elected]
        if not holds_mentioned_values(shapes[old], shape, output.bindings):
            raise PolicyContractError(
                f"house policies moved the election to {shape.shape_id!r}, which does not "
                "hold a value the phrase names"
            )
        slots = _slot_property_dcids(shape)
        kept = [
            draft
            for draft in output.bindings
            if draft.evidence != "sole_option"
            or (
                draft.property_dcid in slots
                and reads_bound_values_alike(shapes[old], shape, [draft])
            )
        ]
        update["bindings"] = _without_partly_stated_values(
            _grounded_in_taxonomy(kept, shape), shape
        )
        admitted = admitted_qualifiers[elected] if admitted_qualifiers else ()
        update["unplaced"] = [
            phrase
            for phrase in output.unplaced
            if not any(_label_named_in_text(match.matched_text, phrase) for match in admitted)
        ]
    return output.model_copy(update=update)


def mentioned_drafts(bindings: Sequence[SlotBindingDraft]) -> list[SlotBindingDraft]:
    """The value and set drafts of `bindings` the phrase names (evidence ``mentioned``)."""
    return [
        draft
        for draft in bindings
        if draft.kind in ("value", "set")
        and draft.evidence == "mentioned"
        and draft.property_dcid is not None
    ]


def _slot_property_dcids(shape: ShapeDraft) -> frozenset[str | None]:
    return frozenset(slot.property_dcid for slot in shape.slot_keys)
