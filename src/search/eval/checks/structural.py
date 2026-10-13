"""``structural``: the response validates against the response models of this checkout.

A resolve response must carry ``search.models.SCHEMA_VERSION`` and validate with
``RESOLVE_RESPONSE_ADAPTER``. A problem document (``base.problem_code``) must validate
as ``dfo_platform.ProblemDetails``. These are the models of the checkout the driver runs
from, which is the head checkout in CI, so the check scores the head app against its own
models and is n/a on the base side (``ctx.side == "base"``). A driver run from another
checkout scores the head app against that checkout's models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from dfo_platform import ProblemDetails
from search.eval.cases import Case
from search.eval.checks.base import NA, PASS, Result, ScoreContext, fail, problem_code, register
from search.models import RESOLVE_RESPONSE_ADAPTER, SCHEMA_VERSION


@register("structural")
def structural(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """The head response validates: a problem document as one, any other body as a resolve
    response carrying this checkout's schema version."""
    if ctx.side == "base":
        return NA
    try:
        if problem_code(response) is not None:
            ProblemDetails.model_validate(response)
            return PASS
        version = response.get("schema_version")
        if version != SCHEMA_VERSION:
            return fail(f"schema_version {version!r}, expected {SCHEMA_VERSION!r}")
        RESOLVE_RESPONSE_ADAPTER.validate_python(response)
    except ValidationError as exc:
        first = exc.errors()[0]
        return fail(
            f"{exc.error_count()} validation error(s), first {first['loc']}: {first['msg']}"
        )
    return PASS
