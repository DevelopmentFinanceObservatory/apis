"""``agreement``: every fresh run of a case gives the same answer.

A case that sets ``repeat`` takes at least that many samples on each side of the gate
(``search.eval.runner.run_cases``), and the samples group by their answer as the runner's
vote groups them (``runner.modal_sample``): per part, the status with the spec id, served
count, member digest and entities of a definite part, the spec ids of candidates, the
no-data reason or unsupported code, and the operations. The driver scores each sample with
its run's ``ScoreContext.run_agreement``, so every sample of a run carries the run's verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import Case
from search.eval.checks.base import NA, PASS, Result, ScoreContext, fail, register


@register("agreement")
def agreement(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """Every one of at least ``case.repeat`` completed runs gives the modal answer. n/a for a
    case without ``repeat`` and for a sample scored alone, which has no runs to compare."""
    if case.repeat is None or ctx.run_agreement is None:
        return NA
    agreeing, completed = ctx.run_agreement
    if completed < case.repeat:
        return fail(f"{completed} of {case.repeat} runs answered")
    if agreeing < completed:
        return fail(f"{agreeing} of {completed} runs gave the modal answer")
    return PASS
