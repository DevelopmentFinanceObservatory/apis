"""``parts``: each expected part, in query order, scored with every part check.

The response must be a resolution with one part per expected part, or another kind the case
accepts, which leaves the check n/a. Each part must return a
status its block accepts, and then pass every registered part check (``PART_CHECKS``, in
``CHECK_NAMES`` order) under that status's block. The first part that fails decides the
verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import CHECK_NAMES, Case
from search.eval.checks.base import (
    NA,
    PART_CHECKS,
    Result,
    ScoreContext,
    all_of,
    answer_kind,
    expected_parts,
    fail,
    register,
    response_parts,
    run_part,
)


@register("parts")
def parts(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """Every expected part returns, with an accepted status, and passes every part check."""
    expected = case.expect.parts
    if expected is None:
        return NA
    kind = answer_kind(response, ctx)
    if kind != "resolution":
        if kind in case.expect.kinds:
            return NA
        return fail(f"kind: got {kind!r}, expected resolution")
    got = len(response_parts(response, ctx))
    if got != len(expected):
        return fail(f"parts: got {got}, expected {len(expected)}")
    results = []
    for e in expected_parts(case, response, ctx):
        if e.block is None or e.part is None:
            accepted = list(e.expectation.accepted_statuses)
            results.append(fail(f"{e.missed}, expected one of {accepted}"))
            continue
        results.extend(
            run_part(PART_CHECKS[name], e.block, e.part, response, ctx).at(name)
            for name in CHECK_NAMES
            if name in PART_CHECKS
        )
    return all_of(results)
