"""``parts``: each expected part, in query order, scored with every part check.

The response must be a resolution with one part per expected part. Each part must return a
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
    fail,
    read_path,
    register,
    response_parts,
    run_part,
    status_block,
)


@register("parts")
def parts(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """Every expected part returns, with an accepted status, and passes every part check."""
    expected = case.expect.parts
    if expected is None:
        return NA
    kind = read_path(response, "kind", ctx)
    if kind != "resolution":
        return fail(f"kind: got {kind!r}, expected resolution")
    got = response_parts(response, ctx)
    if len(got) != len(expected):
        return fail(f"parts: got {len(got)}, expected {len(expected)}")
    results = []
    for part, part_expect in zip(got, expected, strict=True):
        block = status_block(part_expect, part.status)
        if block is None:
            accepted = list(part_expect.accepted_statuses)
            results.append(
                fail(f"parts[{part.index}] status: got {part.status!r}, expected one of {accepted}")
            )
            continue
        results.extend(
            run_part(PART_CHECKS[name], block, part, response, ctx).at(name)
            for name in CHECK_NAMES
            if name in PART_CHECKS
        )
    return all_of(results)
