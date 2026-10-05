"""``request``: the window and unit the answer's identity records.

Reads ``definition.identity.requested_window.{kind,start_year,end_year}`` and
``definition.identity.facets.unit.request``, comparing only the unit fields the case sets.
``unit: none`` asserts no conversion request, which the wire shows by omitting the unit
facet, or the whole facets map.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import ExpectedUnit, StatusExpect, Window
from search.eval.checks.answers import Answer, over_answers
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    all_of,
    fail,
    register_part,
)


def _window(expected: Window, answer: Answer, ctx: ScoreContext) -> Result:
    got = tuple(
        answer.identity(f"requested_window.{name}", ctx)
        for name in ("kind", "start_year", "end_year")
    )
    want = (expected.kind, expected.start, expected.end)
    return PASS if got == want else fail(f"window: got {got}, expected {want}")


def _unit(expected: ExpectedUnit | str, answer: Answer, ctx: ScoreContext) -> Result:
    request = answer.identity("facets.unit.request", ctx)
    if isinstance(expected, str):
        return PASS if request is None else fail(f"unit: got {request}, expected none")
    want = expected.model_dump(mode="json", include=expected.model_fields_set)
    got = {name: request.get(name) for name in want} if request is not None else None
    return PASS if got == want else fail(f"unit: got {request}, expected {want}")


def match_request(block: StatusExpect, answer: Answer, ctx: ScoreContext) -> Result:
    return all_of(
        [
            _window(block.window, answer, ctx) if block.window is not None else NA,
            _unit(block.unit, answer, ctx) if block.unit is not None else NA,
        ]
    )


@register_part("request")
def request(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The answer's identity records the expected window and unit request."""
    if block.window is None and block.unit is None:
        return NA
    return over_answers(part, ctx, lambda answer: match_request(block, answer, ctx), reading=False)
