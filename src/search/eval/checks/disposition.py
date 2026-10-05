"""``disposition``: the kind of answer, its status, and a catalogue or declined scope.

Reads the root ``kind``. For a resolution it reads the root ``status`` (``mixed`` when
parts differ), or with ``expect.parts`` the part count and each ``parts[].status``. For a
catalogue or declined answer it reads ``scope.place.dcid`` when the case sets
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
    fail,
    read_path,
    register,
    response_parts,
)


def _resolution(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    expect = case.expect
    if expect.parts is None:
        status = read_path(response, "status", ctx)
        if status not in expect.accepted_statuses:
            return fail(f"status: got {status!r}, expected one of {list(expect.accepted_statuses)}")
        return PASS
    parts = response_parts(response, ctx)
    if len(parts) != len(expect.parts):
        return fail(f"parts: got {len(parts)}, expected {len(expect.parts)}")
    for part, part_expect in zip(parts, expect.parts, strict=True):
        if part.status not in part_expect.accepted_statuses:
            return fail(
                f"parts[{part.index}] status: got {part.status!r}, "
                f"expected one of {list(part_expect.accepted_statuses)}"
            )
    return PASS


def problem_code(response: Mapping[str, Any]) -> str | None:
    """The registry code of a problem document, or None for any other body."""
    code = response.get("code")
    if "kind" in response or not isinstance(code, str):
        return None
    return code if isinstance(response.get("status"), int) else None


@register("disposition")
def disposition(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The answer is an accepted kind, with an accepted status or scope place."""
    expect = case.expect
    code = problem_code(response)
    kind = "problem" if code is not None else read_path(response, "kind", ctx)
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
        place = read_path(response, "scope.place.dcid", ctx)
        if place != expect.scope_place:
            return fail(f"scope place: got {place!r}, expected {expect.scope_place!r}")
    return PASS
