from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .base import StrictModel
from .scope import OptionId, ScopeAxis, TextOrigin

# ---------------------------------------------------------------------------
# StatedScopeRequest/StatedScopeResponse: the stated-scope door
# (POST /api/search/stated-scope)
# ---------------------------------------------------------------------------

# Caps are local literals, as for the parsed door, so importing search.models needs no engine.
STATED_SCOPE_MAX_TEXTS = 64
STATED_SCOPE_MAX_TEXT_CHARS = 2000


class StatedScopeRequest(StrictModel):
    """The texts whose stated scope to read."""

    texts: list[Annotated[str, Field(max_length=STATED_SCOPE_MAX_TEXT_CHARS)]] = Field(
        max_length=STATED_SCOPE_MAX_TEXTS,
        description="the texts to read, each a user message or a requirement's measure",
    )


class StatedScopeEntry(StrictModel):
    """One scope option a text states, with the words it was read from."""

    axis: ScopeAxis = Field(description="the axis the option settles")
    option_id: OptionId = Field(description="the option the text states")
    origin: TextOrigin = Field(description="the words the option was read from")


class StatedScopeReading(StrictModel):
    """What one text states of its scope."""

    entries: list[StatedScopeEntry] = Field(
        description="the options the text states, one per axis it states one option on"
    )
    total: bool = Field(description="whether the text asks for a total")
    mentions: list[OptionId] = Field(
        description=(
            "every provider option the text names, in order, including both options of a "
            "text that names two and so has no `provider` entry"
        )
    )


class StatedScopeResponse(StrictModel):
    """The reading of each posted text, in order."""

    readings: list[StatedScopeReading] = Field(description="one reading per posted text")
