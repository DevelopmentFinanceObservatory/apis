from __future__ import annotations

from pydantic import Field, field_validator

from .base import StrictModel
from .scope import OptionId, ScopeAxis, TextOrigin

# ---------------------------------------------------------------------------
# StatedScopeRequest/StatedScopeResponse: the stated-scope door
# (POST /api/search/stated-scope)
# ---------------------------------------------------------------------------

# The count cap is a local literal, as for the parsed door, so importing search.models needs no
# engine; a test checks it against `search.engine.config.SEARCH_MAX_STATED_SCOPE_TEXTS`. The
# route checks each text's length against the raw-text door's `max_query_chars`.
STATED_SCOPE_MAX_TEXTS = 64


class StatedScopeRequest(StrictModel):
    """The texts whose stated scope to read."""

    texts: list[str] = Field(
        max_length=STATED_SCOPE_MAX_TEXTS,
        description=(
            "the texts to read, each a user message or a requirement's measure, each at "
            "most the raw_text query character ceiling (capabilities `max_query_chars`)"
        ),
    )

    @field_validator("texts")
    @classmethod
    def _encode(cls, texts: list[str]) -> list[str]:
        """A text holding a lone surrogate does not encode as UTF-8, so it is refused here."""
        for text in texts:
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("a text must not hold a lone surrogate") from exc
        return texts


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
            "text that names two and so has no `provider` entry. Empty when the text names "
            '"multilateral" as a kind of provider ("multilateral development banks", or '
            "bilateral and imputed multilateral aid shown separately), which leaves the "
            "provider axis unstated"
        )
    )
    sums_providers: bool = Field(
        description=(
            "whether the provider entry sums the bilateral and multilateral lines naming no "
            'imputed share ("bilateral plus multilateral"); a dataset reads such words as its '
            "own option, provider:contributions on DAC1"
        )
    )


class StatedScopeResponse(StrictModel):
    """The reading of each posted text, in order."""

    readings: list[StatedScopeReading] = Field(description="one reading per posted text")
