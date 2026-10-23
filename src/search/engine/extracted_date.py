"""The date reference the extraction emits and `time_bind.parse_time_text` builds.

It lives in a leaf module so `extract` and `time_bind` can each import it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


def _is_none(value: object) -> bool:
    return value is None


class ExtractedDate(BaseModel):
    kind: Literal["point", "range", "latest"] = Field(
        description=(
            'Date reference type. "point" for a single year/month, "range" for a span '
            'with a start and/or end bound, "latest" ONLY when the query explicitly asks '
            'for the most recent / current value using a word like "current", "latest", '
            '"now", or "today" (e.g. "current population"). Do not use "latest" merely '
            "because a query omits a date."
        )
    )
    # The 32-character cap covers the 25-character ISO timestamp below.
    start: str | None = Field(
        default=None,
        max_length=32,
        description='ISO year string (e.g. "2010"); null when the start bound is open '
        'or kind is "latest".',
    )
    end: str | None = Field(
        default=None,
        max_length=32,
        description='ISO year string (e.g. "2020"); null when the end bound is open '
        'or kind is "latest".',
    )
    # `extract` replaces every date carrying it with the years it spans, so a dump of a
    # resolved date omits the field.
    relative: str | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "The query's own words for any time measured from now, a span or a single "
            "year ('over the last decade', 'in the past five years', 'recent years', "
            "'last year', 'this year', 'los últimos 10 años'); leave start and end null."
        ),
    )
    # A span counted back from now ("the last five years") asks for that many years ending
    # at the newest year the answer's data holds, at most the current year that ends
    # `end`. `time_bind.parse_time_text` sets it; the model's schema leaves it out, and a
    # dump states it only when set, so a cache key tells the span from its calendar years.
    trailing_years: SkipJsonSchema[int | None] = Field(default=None, exclude_if=_is_none)
