"""``assumptions``: the part's assumptions in clause order.

Reads ``parts[].assumptions[].{kind,binding_ref,text}``. Each expected assumption asserts
its kind, and its ``binding_ref`` and ``text`` when the case sets them; ``[]`` asserts none.
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

_FIELDS = ("kind", "binding_ref", "text")


@register_part("assumptions")
def assumptions(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The part carries the expected assumptions, in order."""
    expected = block.assumptions
    if expected is None:
        return NA
    at = f"{PartView.path}.assumptions[]"
    items = read_path(part.body, "assumptions", ctx, at=PartView.path) or []
    got = [{f: read_path(item, f, ctx, at=at) for f in _FIELDS} for item in items]
    want = [
        item.model_dump(include={"kind"} | (item.model_fields_set & set(_FIELDS)))
        for item in expected
    ]
    matched = len(got) == len(want) and all(
        all(g[f] == w[f] for f in w) for g, w in zip(got, want, strict=True)
    )
    return PASS if matched else fail(f"assumptions: got {got}, expected {want}")
