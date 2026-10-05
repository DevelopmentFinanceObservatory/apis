"""``skills``: whether a definite part's spec matched any published skill, and which.

Reads ``parts[].spec.resolved.skills`` on each definite part the case's ``definite.skills``
block applies to, the envelope ``skills`` map, every spec's ``resolved.skills``
(``specs.response_specs``) and the warning codes in ``diagnostics.warnings[]`` and
``parts[].warnings[]``. A skills index warning fails the part, since an index that did not
load reads as no match. Every id any spec names must have an entry in the envelope map.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import DefiniteExpect, StatusExpect
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    fail,
    read_path,
    register_part,
    response_parts,
)
from search.eval.checks.specs import response_specs

# Warnings meaning ``resolved.skills`` may not reflect the published skills.
_SKILLS_INDEX_WARNING_CODES = frozenset({"skills_index_unavailable", "skills_index_incomplete"})


def _warning_codes(response: Mapping[str, Any], ctx: ScoreContext) -> set[str]:
    warnings = read_path(response, "diagnostics.warnings", ctx) or []
    codes = {read_path(w, "code", ctx, at="diagnostics.warnings[]") for w in warnings}
    for part in response_parts(response, ctx):
        for w in read_path(part.body, "warnings", ctx, at=PartView.path) or []:
            codes.add(read_path(w, "code", ctx, at=f"{PartView.path}.warnings[]"))
    return codes


@register_part("skills")
def skills(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The spec's skills are present or absent as expected, include every expected id, and
    every id a spec names is in the envelope's skills map."""
    if not isinstance(block, DefiniteExpect) or block.skills is None or part.status != "definite":
        return NA
    expected = block.skills
    index_warnings = sorted(_SKILLS_INDEX_WARNING_CODES & _warning_codes(response, ctx))
    if index_warnings:
        return fail(f"skills index warning: {', '.join(index_warnings)}")

    found = read_path(part.body, "spec.resolved.skills", ctx, at=PartView.path)
    if found is None:
        return fail("definite spec omitted resolved.skills")
    if expected.present != bool(found):
        expectation = "a matching skill" if expected.present else "no matching skill"
        return fail(f"expected {expectation}, found {', '.join(found) or 'none'}")
    missing = sorted(set(expected.ids or ()) - set(found))
    if missing:
        return fail(f"skills missing {missing}, found {sorted(found)}")

    named = {
        skill_id
        for spec in response_specs(response, ctx)
        for skill_id in spec.read("resolved.skills", ctx) or ()
    }
    orphans = sorted(named - set(read_path(response, "skills", ctx) or {}))
    if orphans:
        return fail(f"skills named by a spec but missing from the response skills map: {orphans}")
    return PASS
