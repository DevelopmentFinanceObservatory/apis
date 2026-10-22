"""Rate, in one small LLM call per pass, whether the elected shape's series states each word
of the variable the coverage readers leave unread on it.

The call sees the variable phrase, the elected shape's reading as the bind menu states it
(`bind.stated_reading`), and the numbered words. It answers, per word number, whether the
series states what the word names (`StatedWordsOutput`): the total population does not state
"refugee", and total revenue does not state "tax". Its schema is fixed whatever the menu
and the word count, so a recorded answer is keyed by the prompt text alone, and bind's
schema never carries the rating. `GeminiBinder.rate_words_stated` makes the call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from search.engine.bind import stated_reading
from search.engine.clients.llm import SupportsLLM, for_stage, generate_structured_async

if TYPE_CHECKING:
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
answer it. Each numbered word is a word of the variable phrase, which the series may state in
other words. For each word, say whether the series states what the word names.

stated is false when the word narrows the request to a part of what the series counts that
the series does not single out: a kind, a group, a type, a party or a source. Total revenue
counts more than tax, so it does not state "tax". The whole population counts more than
refugees, so it does not state "refugee". Debt owed to all multilateral creditors counts more
than debt owed to the IMF, so it does not state "IMF".

stated is true when the word names what the series measures, in other words or another word
form, or names its subject, its form, the reading the measure takes by convention, or a part
the series already singles out. A count of the people who can read states "literacy", and a
count of persons aged 0 to 14 states "children". When in doubt whether the word narrows the
series, it is stated.

Answer each word once, by its number in the WORDS list.
"""

WORDS_HEADER = "WORDS (words of the variable phrase):"


def build_stated_words_prompt(variable: str, shape: ShapeDraft, words: Sequence[str]) -> str:
    """The user prompt: the variable phrase, the series as the bind menu states the shape's
    reading (`bind.stated_reading`), then the numbered WORDS list."""
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
    lines.extend(["", WORDS_HEADER])
    lines.extend(f"  {number}: {word}" for number, word in enumerate(words))
    lines.append("")
    return "\n".join(lines)


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
    variable: str, shape: ShapeDraft, words: Sequence[str], *, llm: SupportsLLM
) -> tuple[dict[str, float], dict | None]:
    """Rate whether `shape`'s series states each of `words` (`word_ratings`), and the call's
    usage. The call runs with minimal thinking. A call that fails or whose output fails
    validation raises `LLMInfraError`, which the caller reads as no rating. The prompt
    holds only the variable phrase, the shape's graph-sourced reading and the phrase's own
    words."""
    output, usage = await generate_structured_async(
        for_stage(llm, "stated_words"),
        prompt=build_stated_words_prompt(variable, shape, words),
        system=STATED_WORDS_SYSTEM_PROMPT,
        schema=StatedWordsOutput,
    )
    return word_ratings(output, words), usage
