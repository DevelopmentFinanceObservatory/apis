"""Read the quantity a variable phrase asks for in one small LLM call per part, before bind.

The call sees the variable phrase, less the words that name the part's places, and the
numbered StatVar denominators the shapes admitted on their measure and roles divide by. It says which form
the phrase names and which denominator it divides by, quoting the words that name the base
(`QuantityOutput`). It elects nothing. It runs before the admissibility form rule, which
refuses on the form it states, and bind's own prompt and schema never carry the reading.
A call that fails or returns output that fails validation reads as a phrase that names
neither, and marks the answer degraded (`read_stated_quantity`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from search.engine.clients.llm import SupportsLLM, for_stage, generate_structured_async
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import LLMInfraError
from search.models import GraphRef, QuantityForm

logger = logging.getLogger(__name__)


class QuantityOutput(BaseModel):
    """The form and denominator the variable phrase names, each null when it names none,
    and the phrase's words that name the denominator."""

    base_words: str | None = Field(
        default=None,
        description=(
            "The words of the variable phrase that name the base a ratio or share divides "
            'by, copied exactly as the phrase spells them ("GDP" in "debt to GDP"); null '
            "when the phrase names no base, or only people or area."
        ),
    )
    denominator_index: int | None = Field(
        default=None,
        description=(
            "Index into the numbered DENOMINATORS list of the base base_words name; the "
            'index of "none of these" when they name a base no entry is; null when '
            "base_words is null."
        ),
    )
    stated_form: QuantityForm | None = Field(
        default=None,
        description=(
            "The normalised form the variable phrase asks for; null when it asks for a "
            "plain amount or count."
        ),
    )


_QUANTITY_SYSTEM_PROMPT: str = """\
PART 3 — read the quantity the variable phrase asks for, from the phrase's own words alone,
never from the shape you chose and never from the shapes the menu offers: a menu of ratio or
share series says nothing about what the phrase asks. Set a field only when words in the
phrase name it ("to GDP", "share of", "ratio", "per capita", "as a percentage of", "density").
Every field defaults to null. Set one only when you can quote the words of the phrase that name
it; a measure noun alone ("spending", "expenditure", "debt", "aid", "GDP") names a plain amount,
so every field stays null.

base_words: the words of the phrase that name the base a ratio or share divides by, copied
exactly as the phrase spells them: "GDP" in "debt to GDP", "GNI" in "aid as a share of GNI",
"exports" in "debt service as a percentage of exports". Null when the phrase names no such
base, or names only people or area ("per capita", "per 100,000 people", "per square km"),
which the shapes state on their own. A form word alone ("ratio", "rate", "share") names no
base.

denominator_index: the base base_words name.
  - null when base_words is null;
  - the index of the DENOMINATORS entry that is the base base_words name;
  - the index of "none of these" when they name a base that no entry is.

stated_form: the form the phrase asks for, or null when its words name none:
"per_person" (per capita, per person), "per_head_base" (per a number of people, "per 100,000
people"), "per_area" (per square km, density), "rate" (one quantity to another: "debt to GDP",
"a ratio") or "share" (a part of a whole: "% of GDP", "share of spending").

Example: "government education spending" names neither a base nor a form, so base_words,
denominator_index and stated_form are null, even when every shape on the menu is a share
of GDP or of government spending.
"""

DENOMINATORS_HEADER = (
    "DENOMINATORS (the bases the candidate series divide by, other than people and area):"
)
NONE_OF_THESE = "none of these"


def denominator_choices(denominators: Sequence[GraphRef]) -> list[str]:
    """The DENOMINATORS list entries in index order: each denominator's label and dcid,
    then "none of these", whose index is ``len(denominators)``."""
    return [*(f"{ref.label} ({ref.ref})" for ref in denominators), NONE_OF_THESE]


def build_quantity_prompt(variable: str, denominators: Sequence[GraphRef]) -> str:
    """The user prompt: the variable phrase, then the numbered DENOMINATORS list."""
    lines = [f"Variable phrase: {variable}", "", DENOMINATORS_HEADER]
    lines.extend(
        f"  {number}: {choice}" for number, choice in enumerate(denominator_choices(denominators))
    )
    lines.append("")
    return "\n".join(lines)


async def read_stated_quantity(
    variable: str, denominators: Sequence[GraphRef], *, llm: SupportsLLM
) -> tuple[QuantityOutput, dict | None]:
    """Ask which form and which of ``denominators`` the variable phrase names.

    ``denominator_index`` indexes ``denominators`` with "none of these" last. The call runs
    under bind's model settings. A call that fails or whose output fails validation
    (``LLMInfraError``) returns an output that names neither, with no usage, and notes the
    read as lost (`note_degraded_read`), so the response cache stores no answer read
    without it. The prompt holds only the variable phrase and graph-sourced labels.
    """
    try:
        return await generate_structured_async(
            for_stage(llm, "quantity"),
            prompt=build_quantity_prompt(variable, denominators),
            system=_QUANTITY_SYSTEM_PROMPT,
            schema=QuantityOutput,
        )
    except LLMInfraError:
        logger.warning("quantity read failed; the phrase reads as naming no form", exc_info=True)
        note_degraded_read("quantity read")
        return QuantityOutput(), None
