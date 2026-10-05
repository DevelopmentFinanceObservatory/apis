"""``candidates``: the offered readings include every expected family, one each.

A family is included when a candidate spec matches its measure form (shape and fixed slots,
``measure.match_form``) and its provenance (``provenance.match_provenance``). The families
pair one to one with distinct candidates, so two families cannot both claim one spec. Reads
``parts[].candidates.specs[]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import CandidatesExpect, MeasureForm, StatusExpect
from search.eval.checks.answers import Answer, answers
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    Verdict,
    all_of,
    fail,
    pairs_one_to_one,
    register_part,
    unverifiable,
)
from search.eval.checks.measure import match_form
from search.eval.checks.provenance import match_provenance


def _family_match(form: MeasureForm, answer: Answer, ctx: ScoreContext) -> Result:
    provenance = match_provenance(form.provenance, answer, ctx) if form.provenance else NA
    return all_of([match_form(form, answer, ctx), provenance])


@register_part("candidates")
def candidates(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """Each expected family pairs with its own candidate."""
    if not isinstance(block, CandidatesExpect) or not block.include_families:
        return NA
    if part.status != "candidates":
        return NA
    families = list(block.family_forms.items())
    specs = answers(part, ctx)
    verdicts = [[_family_match(form, spec, ctx).verdict for spec in specs] for _, form in families]
    if pairs_one_to_one(len(families), len(specs), lambda f, s: verdicts[f][s] == Verdict.PASS):
        return PASS
    open_verdicts = {Verdict.PASS, Verdict.UNVERIFIABLE}
    if pairs_one_to_one(len(families), len(specs), lambda f, s: verdicts[f][s] in open_verdicts):
        return unverifiable("a family pairs only with a candidate whose match is unverifiable")
    unmatched = [
        family
        for (family, _), row in zip(families, verdicts, strict=True)
        if Verdict.PASS not in row
    ]
    detail = (
        f"no candidate matches {unmatched}" if unmatched else "two families claim one candidate"
    )
    return fail(f"{detail} among {len(specs)} candidates")
