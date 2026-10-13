"""``refusal``: why the answer refused.

A no-data part reads ``parts[].no_data.{reason,category,gate}`` against the case's
``no_data`` block (``reason`` may list accepted reasons), an unsupported part reads
``parts[].unsupported.code``, and a declined answer reads the root ``reason``. Each applies
only when that status or kind returns. Each is unreached when the answer pairs with no
block the case accepts (``base.expected_parts``) while the block it would have read asserts
it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import Case, NoDataExpect, StatusExpect, UnsupportedExpect
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    Verdict,
    all_of,
    answer_kind,
    expected_parts,
    fail,
    over_parts,
    read_path,
    register,
    register_part,
)


def _no_data(block: NoDataExpect, part: PartView, ctx: ScoreContext) -> Result:
    results = []
    for name in ("reason", "category", "gate"):
        if name not in block.model_fields_set:
            continue
        expected = getattr(block, name)
        accepted = expected if isinstance(expected, list) else [expected]
        got = read_path(part.body, f"no_data.{name}", ctx, at=PartView.path)
        results.append(
            PASS if got in accepted else fail(f"no_data {name}: got {got!r}, expected {expected!r}")
        )
    return all_of(results)


@register_part("refusal", case_check=False)
def part_refusal(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """A no-data part's reason, category and gate, or an unsupported part's code."""
    if isinstance(block, NoDataExpect) and part.status == "no_data":
        return _no_data(block, part, ctx)
    if isinstance(block, UnsupportedExpect) and part.status == "unsupported":
        if block.code is None:
            return NA
        code = read_path(part.body, "unsupported.code", ctx, at=PartView.path)
        return (
            PASS
            if code == block.code
            else fail(f"unsupported code: got {code!r}, expected {block.code!r}")
        )
    return NA


def _declined(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The declined reason; unreached when the answer pairs with no block the case accepts,
    whether its kind or every part's status is one the case does not accept, and n/a when
    it is another accepted answer."""
    expected = case.expect.declined
    if expected is None:
        return NA
    if answer_kind(response, ctx) == "declined":
        reason = read_path(response, "reason", ctx)
        if reason != expected.reason:
            return fail(f"declined reason: got {reason!r}, expected {expected.reason!r}")
        return PASS
    parts = expected_parts(case, response, ctx)
    if parts and all(part.block is None for part in parts):
        return Result(Verdict.UNREACHED, parts[0].missed)
    return NA


_parts_refusal = over_parts(part_refusal, name="refusal")


@register("refusal")
def refusal(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The declined reason, and each refusing part's reason or code."""
    return all_of([_declined(case, response, ctx), _parts_refusal(case, response, ctx)])
