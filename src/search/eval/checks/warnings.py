"""``warnings``: the warning codes the part must carry. Extra warnings pass.

Reads ``parts[].warnings[].code`` and the request-level ``diagnostics.warnings[].code``.
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
    unconfirmed,
    unverifiable,
)


def _codes(warnings: list[Mapping[str, Any]] | None, at: str, ctx: ScoreContext) -> set[str]:
    return {read_path(w, "code", ctx, at=at) for w in warnings or []}


@register_part("warnings")
def warnings(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """Every required code is on the part or on the request."""
    required = block.warnings
    if required is None:
        return NA
    if unconfirmed(required):
        return unverifiable("warnings are unconfirmed")
    part_at = f"{PartView.path}.warnings[]"
    codes = _codes(read_path(part.body, "warnings", ctx, at=PartView.path), part_at, ctx)
    codes |= _codes(read_path(response, "diagnostics.warnings", ctx), "diagnostics.warnings[]", ctx)
    missing = sorted(set(map(str, required)) - codes)
    return PASS if not missing else fail(f"warnings missing {missing}, got {sorted(codes)}")
