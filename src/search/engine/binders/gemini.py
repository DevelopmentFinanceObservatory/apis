"""The Gemini binder: the engine's own quantity read and choose-and-bind call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from search.engine.bind import (
    SlotBindingDraft,
    _grounded_in_taxonomy,
    _states_same_reading,
    _UniversalBindOutput,
    _without_partly_stated_values,
    bind_universal,
    covered_leaves,
    reads_bound_values_alike,
    stated_reading,
    validate_requested_all_bindings,
)
from search.engine.bind_repair import _label_named_in_text
from search.engine.binders.base import (
    BindRequest,
    BindResult,
    PolicyContractError,
    QuantityRequest,
)
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.config import settings
from search.engine.quantity_read import read_stated_quantity

if TYPE_CHECKING:
    from search.engine.clients.llm import SupportsLLM
    from search.engine.qualifier_match import QualifierMatch
    from search.engine.quantity_read import QuantityOutput
    from search.engine.shape import ShapeDraft

# The rated pool's bounds (`rated_pool`): a shape joins with a rating above
# `RATED_POOL_FLOOR` and at least (1 - `RATED_POOL_FRACTION`) times the best rating.
RATED_POOL_FLOOR = 0.1
RATED_POOL_FRACTION = 0.1

# A reply that rates its own pick below `DECLINE_FLOOR` (`pick_rating`) declines
# (`declined`). The part asks, as a reply that asked does, and no policy runs.
DECLINE_FLOOR = 0.6

# The slack `near_best` gives the bound, wider than a float product's rounding and far
# narrower than the two decimals a reply writes a rating to.
_BOUND_SLACK = 1e-9


class GeminiBinder:
    """Reads the quantity with `quantity_read.read_stated_quantity` and elects and binds with
    `bind.bind_universal`, on the injected LLM client. A reply that elects a shape goes
    through the house policies over its pool, and a moved election takes the reply's drafts
    onto the shape the policies elect (`reelect`).

    The reply rates each shape that measures the phrase, and the pool is the shapes it rates
    near the best and the pick's twins (`rated_pool`), over which all six policies run with
    the ratings. Each pool shape holds the pick's constraints when it holds each value the
    reply drafted as named and every value of each slot the reply asks for in full
    (`holds_mentioned_values`), which the pool reads as it admits each shape, once the
    reply's ``requested_all`` claims are validated (`with_valid_requested_all`). A
    nomination in `also_plausible` stands only when the reply rates it near its best
    rating, whatever its denominator (`nominated_near_best`).

    A reply that rates neither its pick nor a twin of it in `measuring`, an empty one
    included, or rates it below `DECLINE_FLOOR` (`pick_rating`), says the pick measures
    something else, so it declines (`declined`) and no policy runs."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return settings().engine_model

    @property
    def reads_query_text(self) -> bool:
        return False

    async def read_quantity(
        self, request: QuantityRequest, *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        return await read_stated_quantity(request.variable, request.denominators, llm=llm)

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
        shapes = request.shapes
        output = with_valid_requested_all(
            output, shapes[output.chosen_shape_index], request.dropped_qualifier_phrases
        )
        ratings = shape_ratings(output, len(shapes))
        if pick_rating(shapes, output, ratings) < DECLINE_FLOOR:
            return BindResult(output=declined(output), binder=self.name), usage
        pool, holds = rated_pool(shapes, output, ratings)
        choice = request.apply_policies(pool, ratings=ratings, holds=holds)
        result = BindResult(
            output=nominated_near_best(
                reelect(output, choice, shapes, request.admitted_qualifiers), ratings
            ),
            binder=self.name,
            policy_pool=pool,
            policy_choice=choice,
        )
        return result, usage


# The `ask` a reply declined on its own rating carries (`declined`).
_UNRATED_PICK_ASK = "the reply rates its chosen shape as measuring something else"


def declined(output: _UniversalBindOutput) -> _UniversalBindOutput:
    """`output` as a decline: `ask` set, with no bindings, alternates or stand-in, so the
    part refuses as a reply that asked does. The reply's `measuring` stays as it wrote it."""
    return output.model_copy(
        update={
            "ask": _UNRATED_PICK_ASK,
            "bindings": [],
            "also_plausible": None,
            "nearest_refused_index": None,
            "unplaced": [],
        }
    )


def with_valid_requested_all(
    output: _UniversalBindOutput, pick: ShapeDraft, dropped_qualifier_phrases: Sequence[str]
) -> _UniversalBindOutput:
    """`output` with its ``requested_all`` claims validated on its `pick`
    (`bind.validate_requested_all_bindings`), before the pool and `reelect` read them, so
    a "total" that names no slot keeps no shape out."""
    bindings, _applied = validate_requested_all_bindings(
        output.bindings, dropped_qualifier_phrases, slot_properties=_slot_property_dcids(pick)
    )
    return output.model_copy(update={"bindings": bindings})


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


def pick_rating(
    shapes: Sequence[ShapeDraft], output: _UniversalBindOutput, ratings: Mapping[int, float]
) -> float:
    """The reply's rating of the shape `output` elects. A pick the reply leaves unrated takes
    the best rating of a shape the menu states as its reading (`bind._states_same_reading`),
    since the menu prints the two alike. A pick with neither reads 0."""
    chosen = output.chosen_shape_index
    if chosen in ratings:
        return ratings[chosen]
    return max(
        (
            rating
            for index, rating in ratings.items()
            if index != chosen
            and _states_same_reading(shapes[chosen], shapes[index], output.bindings)
        ),
        default=0.0,
    )


def rated_pool(
    shapes: Sequence[ShapeDraft], output: _UniversalBindOutput, ratings: Mapping[int, float]
) -> tuple[PolicyPool, dict[int, bool]]:
    """The rated pool of the shape `output` elects, and whether each pool shape holds the
    chosen shape's constraints (`holds_mentioned_values`). The pool is that shape first,
    then in menu order every menu shape rated above `RATED_POOL_FLOOR` and near the best
    rating of the menu (`near_best` with `RATED_POOL_FRACTION`), that states the chosen
    shape's denominator (`bind.stated_reading`). The chosen shape stays whatever its rating, and so does each
    shape the menu states as its reading (`bind._states_same_reading`), so the menu-rank
    tie-break (`bind.rank_elected`) can choose between the two.

    A rated shape joins only when it holds every value the phrase names as the chosen shape
    does, so the phrase's own words stand before any policy runs, and so holds them. A twin
    joins whether or not it holds them, and the house policies read which."""
    chosen = output.chosen_shape_index
    pick = shapes[chosen]
    denominator = stated_reading(pick).denominator
    best = max(ratings.values(), default=0.0)
    holds = {chosen: True}
    for index, shape in enumerate(shapes):
        if index == chosen:
            continue
        if _states_same_reading(pick, shape, output.bindings):
            holds[index] = holds_mentioned_values(pick, shape, output)
            continue
        rating = ratings.get(index, 0.0)
        if (
            rating > RATED_POOL_FLOOR
            and near_best(rating, best, RATED_POOL_FRACTION)
            and stated_reading(shape).denominator == denominator
            and holds_mentioned_values(pick, shape, output)
        ):
            holds[index] = True
    return PolicyPool(measuring=tuple(holds)), holds


def nominated_near_best(
    output: _UniversalBindOutput, ratings: Mapping[int, float]
) -> _UniversalBindOutput:
    """`output` nominating in `also_plausible` only the shapes `ratings` rates near the best
    rating of the menu (`near_best` with `RATED_POOL_FRACTION`), the bar a pool shape
    clears, without the pool's denominator test, so a reading of another denominator the
    reply rates as highly stays a rival. A shape the reply rates lower is a weaker reading
    it copied from its ratings, and an unrated one reads 0."""
    if not output.also_plausible:
        return output
    best = max(ratings.values(), default=0.0)
    kept = [
        index
        for index in output.also_plausible
        if near_best(ratings.get(index, 0.0), best, RATED_POOL_FRACTION)
    ]
    if kept == output.also_plausible:
        return output
    return output.model_copy(update={"also_plausible": kept})


def near_best(rating: float, best: float, fraction: float) -> bool:
    """Whether `rating` is at least (1 - `fraction`) times `best`, a rating exactly at that
    bound included whatever the float product rounds to."""
    return rating >= (1.0 - fraction) * best - _BOUND_SLACK


def holds_mentioned_values(
    chosen: ShapeDraft, shape: ShapeDraft, output: _UniversalBindOutput
) -> bool:
    """Whether `shape` holds each value the reply drafted on `chosen` that the phrase names
    (`holds_drafted_values` over `mentioned_drafts`), and every value `chosen`'s slot holds
    for each slot the reply asks for in full (`holds_requested_slots`), so "across all
    flows" keeps every flow type."""
    return holds_drafted_values(
        chosen, shape, mentioned_drafts(output.bindings)
    ) and holds_requested_slots(chosen, shape, output.bindings)


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
        if not holds_mentioned_values(shapes[old], shape, output):
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
