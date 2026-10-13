"""``disposition``: the kind of answer, its status, and a catalogue or declined scope.

Reads the root ``kind``. For a resolution it reads the root ``status`` (``mixed`` when
parts differ), or with ``expect.parts`` the part count and each ``parts[].status``. For a
catalogue or declined answer it reads ``scope.place.ref`` when the case sets
``scope_place``, where null asserts the answer names no place.

A body with no ``kind`` that carries a string ``code`` and an integer ``status`` is an RFC
9457 problem document (``dfo_platform.ProblemDetails``), the answer a door gives when it
refuses the request. Its kind is ``problem`` and its ``code`` is compared with the case's
``problem.code``. The problem document is the platform's contract, outside the Search
response schema, so its ``code`` is read without a schema lookup.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import Case
from search.eval.checks.base import (
    PASS,
    Result,
    ScoreContext,
    answer_kind,
    expected_parts,
    fail,
    problem_code,
    read_path,
    register,
    response_parts,
)


def _resolution(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    expect = case.expect
    if expect.parts is not None:
        got = len(response_parts(response, ctx))
        if got != len(expect.parts):
            return fail(f"parts: got {got}, expected {len(expect.parts)}")
    for e in expected_parts(case, response, ctx):
        if e.block is None:
            return fail(f"{e.missed}, expected one of {list(e.expectation.accepted_statuses)}")
    return PASS


@register("disposition")
def disposition(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The answer is an accepted kind, with an accepted status or scope place."""
    expect = case.expect
    code = problem_code(response)
    kind = answer_kind(response, ctx)
    if kind not in expect.kinds:
        detail = f" {code!r}" if code is not None else ""
        return fail(f"kind: got {kind!r}{detail}, expected one of {sorted(expect.kinds)}")
    if kind == "resolution":
        return _resolution(case, response, ctx)
    if expect.problem is not None and kind == "problem":
        if code != expect.problem.code:
            return fail(f"problem code: got {code!r}, expected {expect.problem.code!r}")
        return PASS
    if "scope_place" in expect.model_fields_set:
        place = read_path(response, "scope.place.ref", ctx)
        if place != expect.scope_place:
            return fail(f"scope place: got {place!r}, expected {expect.scope_place!r}")
    return PASS
