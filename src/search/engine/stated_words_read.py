"""Rate, in one small LLM call per pass, whether the elected shape's series states each word
of the variable the coverage readers leave unread on it.

The call sees the variable phrase, the elected shape's reading as the bind menu states it
(`bind.stated_reading`) with the values the final bindings restrict it to and the values
its measure's definition fixes, and the numbered words. It answers, per word number,
whether the series states what the word names (`StatedWordsOutput`): the total population
does not state "refugee", total revenue does not state "tax", and a completion rate bound to
Sex=Female states "girls". Its schema is fixed whatever the menu and the word count, so a
recorded answer is keyed by the prompt text alone, and bind's schema never carries the
rating. `GeminiBinder.rate_words_stated` makes the call.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from code_index.labels import without_property_suffix
from code_index.shared import get_index
from search.engine.bind import stated_reading
from search.engine.clients.llm import SupportsLLM, for_stage, generate_structured_async
from search.render.labels import _humanize_ref

if TYPE_CHECKING:
    from search.engine.bind import SlotBindingDraft
    from search.engine.shape import ShapeDraft


class StatedWord(BaseModel):
    """Whether the series states what one numbered word names."""

    index: int = Field(description="The number of the word in the WORDS list.")
    stated: bool = Field(
        description=(
            "False when the word narrows the request to a part of what the series counts "
            "that the series does not single out; true when the series states what the "
            "word names."
        )
    )


class StatedWordsOutput(BaseModel):
    """One answer per numbered word."""

    words: list[StatedWord] = Field(description="One entry for each word of the WORDS list.")


# The question and its reading of "states". A calibration run swaps this text to choose the
# wording on labelled pairs.
STATED_WORDS_SYSTEM_PROMPT: str = """\
A user asked for the quantity the variable phrase names, and the series below was chosen to
answer it. Each numbered word is a word of the variable phrase. For each word, say whether
the series states what the word names.

stated is true when the series measures what the word names, under any wording: the same
word, a synonym, an abbreviation, another spelling or word form, or a value the series is
restricted to. It is also true when the word names the series' form (a rate, a share, a
count, an index, an average), names the people or things the series is defined over though
its label leaves them implicit, or only describes, restates or links what the series
measures.

stated is false when the word names something narrower than, or different from, what the
series counts: a kind, a group, a party, a cause or a purpose the series does not single out,
so the series counts more than the word names, or something else.

Answer each word once, by its number in the WORDS list.
"""

WORDS_HEADER = "WORDS (words of the variable phrase):"


def build_stated_words_prompt(
    variable: str,
    shape: ShapeDraft,
    words: Sequence[str],
    *,
    bindings: Sequence[SlotBindingDraft],
) -> str:
    """The user prompt: the variable phrase, the series as the bind menu states the shape's
    reading (`bind.stated_reading`), the values `bindings` restrict it to
    (`_restricted_values`) and those its measure's definition fixes (`implied_values`),
    then the numbered WORDS list."""
    stated = stated_reading(shape)
    lines = [
        f"Variable phrase: {variable}",
        "",
        "SERIES:",
        f"  label: {stated.label}",
        f"  family: {stated.family}",
        "  five_tuple: populationType={}, measuredProperty={}, statType={}".format(
            stated.population_type or "(none)",
            stated.measured_property or "(none)",
            stated.stat_type or "(none)",
        ),
    ]
    if stated.qualifier or stated.denominator:
        lines.append(
            f"    measurementQualifier={stated.qualifier or '(none)'}, "
            f"measurementDenominator={stated.denominator or '(none)'}"
        )
    # The line says where the values sit: a bare "restricted to" reads to the model as all
    # the series singles out, so a word naming whom the series is defined over reads as
    # narrower than it.
    if restricted := _restricted_values(shape, bindings):
        lines.append(f"  restricted to, on its breakdowns: {restricted}")
    # The headline series states nothing on a property its measure's definition fixes, so
    # without the line a word naming that value reads as narrower than the series.
    if implied := implied_values(shape, bindings):
        lines.append(f"  fixed by the measure's definition: {implied}")
    lines.extend(["", WORDS_HEADER])
    lines.extend(f"  {number}: {word}" for number, word in enumerate(words))
    lines.append("")
    return "\n".join(lines)


def _restricted_values(shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]) -> str:
    """Each slot `bindings` restrict to values (`restricting`), as ``Property=Value``, a
    set's values joined by "or": "Sex=Female; Age=15 to 24 years". A value reads by
    `bound_value_label`, and a slot by its slot key's label. Empty when no binding
    restricts a slot."""
    return _property_values(shape, restricting(bindings).items())


def implied_values(shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]) -> str:
    """Each value the code index declares `shape`'s measure means on a property its headline
    series states nothing on (`CodeIndex.implied_values_of`), as `_restricted_values`
    prints a bound one, less those on a property `bindings` restrict, where the bound value
    holds instead, or open (``requested_all``), where the answer serves every value: "GDP
    by economic sector" is no total over all economic activities. Empty for a measure that
    declares none. Jev's stated-word questions quote the same values
    (`jev_menu.describe`)."""
    implied = get_index().implied_values_of(shape.pop_type_dcid, shape.meas_prop_dcid)
    overridden = restricting(bindings).keys() | {
        binding.property_dcid for binding in bindings if binding.evidence == "requested_all"
    }
    return _property_values(
        shape, ((prop, (value,)) for prop, value in implied.items() if prop not in overridden)
    )


def _property_values(shape: ShapeDraft, values: Iterable[tuple[str, Sequence[str]]]) -> str:
    """Each property of `values` with its values, as ``Property=Value``, a property's values
    joined by "or". A value reads by `bound_value_label`, and a property by its slot key's
    label on `shape`."""
    slot_labels = {key.property_dcid: key.label for key in shape.slot_keys}
    return "; ".join(
        f"{_humanize_ref(prop, slot_labels.get(prop))}="
        + " or ".join(bound_value_label(shape, prop, dcid) for dcid in dcids)
        for prop, dcids in values
    )


def restricting(bindings: Sequence[SlotBindingDraft]) -> dict[str, tuple[str, ...]]:
    """The values each value or set binding of `bindings` restricts its slot to, by property
    dcid. A source binding restricts nothing, since it picks where the series comes from
    and not what it counts, and neither does a binding with no property or no value."""
    return {
        prop: tuple(binding.value_dcids)
        for binding in bindings
        if binding.kind in ("value", "set")
        and binding.axis != "source"
        and (prop := binding.property_dcid) is not None
        and binding.value_dcids
    }


def bound_value_label(shape: ShapeDraft, prop: str, dcid: str) -> str:
    """Value `dcid` of `shape`'s slot `prop` as a prompt names it: its code row's label,
    else its graph label (`ShapeDraft.value_labels`) less the trailing name of its property
    (`without_property_suffix`), else its dcid split into words."""
    if (row := get_index().code_by_dcid(dcid)) is not None:
        return row.label
    labels = shape.value_labels or {}
    return without_property_suffix(_humanize_ref(dcid, labels.get(dcid)), labels.get(prop))


def word_ratings(output: StatedWordsOutput, words: Sequence[str]) -> dict[str, float]:
    """The rating of each of `words` `output` answers: 1.0 for stated, 0.0 for not. An
    answer whose index names no word is left out, and so is a word answered both ways, so
    a word with no clear answer stays unrated."""
    answered: dict[int, float | None] = {}
    for answer in output.words:
        if 0 <= answer.index < len(words):
            rating = 1.0 if answer.stated else 0.0
            answered[answer.index] = (
                rating if answered.get(answer.index, rating) == rating else None
            )
    return {words[index]: rating for index, rating in answered.items() if rating is not None}


async def read_stated_words(
    variable: str,
    shape: ShapeDraft,
    words: Sequence[str],
    *,
    bindings: Sequence[SlotBindingDraft],
    llm: SupportsLLM,
) -> tuple[dict[str, float], dict | None]:
    """Rate whether `shape`'s series, restricted to the values `bindings` bind, states each
    of `words` (`word_ratings`), and the call's usage. The call runs with minimal thinking. A call that fails or whose output fails
    validation raises `LLMInfraError`, which the caller reads as no rating. The prompt
    holds only the variable phrase, the shape's graph-sourced reading and the phrase's own
    words."""
    output, usage = await generate_structured_async(
        for_stage(llm, "stated_words"),
        prompt=build_stated_words_prompt(variable, shape, words, bindings=bindings),
        system=STATED_WORDS_SYSTEM_PROMPT,
        schema=StatedWordsOutput,
    )
    return word_ratings(output, words), usage
