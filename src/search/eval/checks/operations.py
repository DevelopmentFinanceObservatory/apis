"""``operations``: the part's operations in query order.

Reads ``parts[].operations[].{verb,over,target_year}``. Each expected operation asserts its
verb, and its ``over`` and ``target_year`` when the case sets them; ``[]`` asserts none.
The ``phrase`` an operation echoes is wording and is ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import StatusExpect
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    fail,
    read_path,
    register_part,
)

_FIELDS = ("verb", "over", "target_year")


@register_part("operations")
def operations(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The part carries the expected operations, in order."""
    expected = block.operations
    if expected is None:
        return NA
    at = f"{PartView.path}.operations[]"
    ops = read_path(part.body, "operations", ctx, at=PartView.path) or []
    got = [{f: read_path(op, f, ctx, at=at) for f in _FIELDS} for op in ops]
    want = [
        op.model_dump(include={"verb"} | (op.model_fields_set & set(_FIELDS))) for op in expected
    ]
    matched = len(got) == len(want) and all(
        all(g[f] == w[f] for f in w) for g, w in zip(got, want, strict=True)
    )
    return PASS if matched else fail(f"operations: got {got}, expected {want}")
