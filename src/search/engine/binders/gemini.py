"""The Gemini binder: the engine's own quantity read and choose-and-bind call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from search.engine.bind import (
    SlotBindingDraft,
    _grounded_in_taxonomy,
    _UniversalBindOutput,
    _without_partly_stated_values,
    bind_universal,
    reads_bound_values_alike,
    stated_reading,
)
from search.engine.bind_repair import _label_named_in_text
from search.engine.binders.base import BindRequest, BindResult
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.config import settings
from search.engine.quantity_read import read_stated_quantity

if TYPE_CHECKING:
    from search.engine.clients.llm import SupportsLLM
    from search.engine.qualifier_match import QualifierMatch
    from search.engine.quantity_read import QuantityOutput
    from search.engine.shape import ShapeDraft
    from search.models import GraphRef

# The rated pool's bounds (`rated_pool`): a shape joins with a rating above
# `RATED_POOL_FLOOR` and at least (1 - `RATED_POOL_FRACTION`) times the best rating.
RATED_POOL_FLOOR = 0.1
RATED_POOL_FRACTION = 0.1

# The slack `near_best` gives the bound, wider than a float product's rounding and far
# narrower than the two decimals a reply writes a rating to.
_BOUND_SLACK = 1e-9


class GeminiBinder:
    """Reads the quantity with `quantity_read.read_stated_quantity` and elects and binds with
    `bind.bind_universal`, on the injected LLM client. A reply that elects a shape goes
    through the house policies over its pool, and a moved election takes the reply's drafts
    onto the shape the policies elect (`reelect`).

    The reply rates each shape that measures the phrase, and the pool is the shapes it rates
    near the best (`rated_pool`), over which all six policies run with the ratings."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return settings().engine_model

    async def read_quantity(
        self, variable: str, denominators: Sequence[GraphRef], *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        return await read_stated_quantity(variable, denominators, llm=llm)

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        output, usage = await bind_universal(
            request.variable,
            request.shapes,
            admitted_qualifiers=request.admitted_qualifiers,
            admitted_ties=request.admitted_ties,
            withheld_measure_axes=request.withheld_measure_axes,
            dropped_qualifier_phrases=request.dropped_qualifier_phrases,
            candidate_menus=request.candidate_menus,
            frame=request.frame,
            refused=request.refused,
            availability=request.availability,
            llm=llm,
        )
        if output.ask or not 0 <= output.chosen_shape_index < len(request.shapes):
            return BindResult(output=output, binder=self.name), usage
        ratings = shape_ratings(output, len(request.shapes))
        pool = rated_pool(request.shapes, output, ratings)
        choice = request.apply_policies(pool, ratings=ratings)
        result = BindResult(
            output=reelect(output, choice, request.shapes, request.admitted_qualifiers),
            binder=self.name,
            policy_pool=pool,
            policy_choice=choice,
        )
        return result, usage


def shape_ratings(output: _UniversalBindOutput, size: int) -> dict[int, float]:
    """The reply's rating of each menu shape of a menu of `size` shapes, by index, read
    from `output.measuring`.

    An entry whose index names no menu shape is left out, a rating outside 0 to 1 reads 0,
    and a repeated index keeps its last rating. A shape with no entry has no rating, which
    the pool and the house policies read as 0."""
    return {
        entry.index: entry.rating if 0.0 <= entry.rating <= 1.0 else 0.0
        for entry in output.measuring
        if 0 <= entry.index < size
    }


def rated_pool(
    shapes: Sequence[ShapeDraft], output: _UniversalBindOutput, ratings: Mapping[int, float]
) -> PolicyPool:
    """The rated pool of the shape `output` elects: that shape first, then in menu order
    every menu shape rated above `RATED_POOL_FLOOR` and near the best rating of the menu
    (`near_best` with `RATED_POOL_FRACTION`), that states the chosen shape's denominator
    (`bind.stated_reading`). The chosen shape stays whatever its rating.

    A shape joins only when it holds every value the phrase names as the chosen shape does (`holds_mentioned_values`), so the phrase's own words
    stand before any policy runs."""
    chosen = output.chosen_shape_index
    denominator = stated_reading(shapes[chosen]).denominator
    best = max(ratings.values(), default=0.0)
    others = (
        index
        for index, shape in enumerate(shapes)
        if index != chosen
        and (rating := ratings.get(index, 0.0)) > RATED_POOL_FLOOR
        and near_best(rating, best, RATED_POOL_FRACTION)
        and stated_reading(shape).denominator == denominator
        and holds_mentioned_values(shapes[chosen], shape, output)
    )
    return PolicyPool(measuring=(chosen, *others), rated=True)


def near_best(rating: float, best: float, fraction: float) -> bool:
    """Whether `rating` is at least (1 - `fraction`) times `best`, a rating exactly at that
    bound included whatever the float product rounds to."""
    return rating >= (1.0 - fraction) * best - _BOUND_SLACK


def holds_mentioned_values(
    chosen: ShapeDraft, shape: ShapeDraft, output: _UniversalBindOutput
) -> bool:
    """Whether `shape` holds each value the reply drafted on `chosen` that the phrase names
    (evidence ``mentioned``) as `chosen` does: it has a slot for each such property, and
    each value applies to the same taxonomy codes of its slot as of `chosen`'s
    (`bind.reads_bound_values_alike`)."""
    mentioned = _mentioned_drafts(output)
    return {draft.property_dcid for draft in mentioned} <= _slot_property_dcids(
        shape
    ) and reads_bound_values_alike(chosen, shape, mentioned)


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
    its value, since it bound only because the old slot held one member; the rest are
    grounded in the new shape's taxonomy (`bind._grounded_in_taxonomy`,
    `bind._without_partly_stated_values`), and the pool admitted the new shape only if it
    holds each value the phrase names (`holds_mentioned_values`). An unplaced phrase leaves
    `unplaced` when the new shape admits a qualifier read from its words
    (`admitted_qualifiers`, each menu shape's own, as the bind request carries them), since
    that shape's vocabulary reads it. An output the policies leave as it was is returned as
    it is."""
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
        assert holds_mentioned_values(shapes[old], shape, output), (
            f"house policies moved the election to {shape.shape_id!r}, which does not hold "
            "a value the phrase names"
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


def _mentioned_drafts(output: _UniversalBindOutput) -> list[SlotBindingDraft]:
    return [
        draft
        for draft in output.bindings
        if draft.kind in ("value", "set")
        and draft.evidence == "mentioned"
        and draft.property_dcid is not None
    ]


def _slot_property_dcids(shape: ShapeDraft) -> frozenset[str | None]:
    return frozenset(slot.property_dcid for slot in shape.slot_keys)
