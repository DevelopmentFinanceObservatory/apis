"""``task_outcome``: the runner got an answer for the case.

``search.eval.runner.run_cases`` records a sample that failed on infrastructure after
its retries as ``CaseRun.infra_error`` and leaves ``modal`` unset. The driver scores such a
case on ``{"task_error": run.infra_error}``, and this check fails on any response carrying a
top-level ``task_error``. Every other response passes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import Case
from search.eval.checks.base import PASS, Result, ScoreContext, fail, register


@register("task_outcome")
def task_outcome(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The response is an answer, not a recorded task error."""
    error = response.get("task_error")
    return PASS if error is None else fail(f"task error: {error}")
