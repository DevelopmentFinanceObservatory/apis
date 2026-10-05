"""``entry_path``: a parsed or dataset_resubmit request skips extraction, and says so.

Reads ``query_echo.extract_skipped``, which must be true, and on every definite or
candidate spec ``resolved.trace.pipeline_trace[].{step, ran}``, which must record
``{step: extract, ran: false}``. A no-data or unsupported part carries no interpretation
trace. A raw_text case, and a problem document (``disposition.problem_code``), are n/a.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import Case
from search.eval.checks.base import (
    NA,
    PASS,
    Result,
    ScoreContext,
    fail,
    read_path,
    register,
    response_parts,
)
from search.eval.checks.disposition import problem_code
from search.eval.checks.specs import part_specs

_TRACED_STATUSES = ("definite", "candidates")


@register("entry_path")
def entry_path(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The echo reports skipped extraction and each traced spec records it."""
    if case.door not in ("parsed", "dataset_resubmit") or problem_code(response) is not None:
        return NA
    if read_path(response, "query_echo.extract_skipped", ctx) is not True:
        return fail(f"query_echo.extract_skipped is not true for door {case.door}")
    for part in response_parts(response, ctx):
        if part.status not in _TRACED_STATUSES:
            continue
        for spec in part_specs(part, ctx, twins=False):
            steps = spec.read("resolved.trace.pipeline_trace", ctx) or []
            at = f"{spec.at}.resolved.trace.pipeline_trace[]"
            if not any(
                read_path(s, "step", ctx, at=at) == "extract"
                and read_path(s, "ran", ctx, at=at) is False
                for s in steps
            ):
                return fail(f"{spec.pointer}: pipeline_trace has no {{step: extract, ran: false}}")
    return PASS
