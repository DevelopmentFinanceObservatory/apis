"""``entry_path``: a parsed or dataset_resubmit request skips extraction, and says so.

Reads ``query_echo.extract_skipped``, which must be true. A raw_text case, and a problem
document (``disposition.problem_code``), are n/a.
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
)
from search.eval.checks.disposition import problem_code


@register("entry_path")
def entry_path(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The echo reports skipped extraction."""
    if case.door not in ("parsed", "dataset_resubmit") or problem_code(response) is not None:
        return NA
    if read_path(response, "query_echo.extract_skipped", ctx) is not True:
        return fail(f"query_echo.extract_skipped is not true for door {case.door}")
    return PASS
