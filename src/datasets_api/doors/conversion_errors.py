"""Classify conversion refusals raised while planning a mint or definition materialisation.

Recipe and rate-coverage errors return 422 with their shared `cause` extension. Non-currency
units return 422 without it. Invalid rate tables return `capability_unavailable` (501). The
classifier preserves each exception's status and code. The binding comes from the stored identity,
so the refusal carries no pointer.
"""

from __future__ import annotations

from dfo_platform import ProblemSpec
from materialise.rate_tables import RateTableInvalidError
from materialise.recipes import (
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    cause_from_recipe_error,
)


def classify_conversion_exception(exc: BaseException) -> ProblemSpec | None:
    """Return the error's status, code, and detail, plus `cause` where present. Otherwise return
    `None`.

    `RoleAxisNotDeclaredError` follows the `RecipeNotDeclaredError` arm and carries
    `role_undeclared`. Wire status and code come from the exception's `MaterialisationError`
    attributes.
    """
    if isinstance(exc, (RecipeNotDeclaredError, RateCoverageGapError)):
        detail = exc.detail or str(exc)
        return ProblemSpec(
            status=exc.status,
            code=exc.code,
            detail=detail,
            retryable=False,
            cause=cause_from_recipe_error(exc),
        )
    if isinstance(exc, NonCurrencyUnitError):
        detail = exc.detail or str(exc)
        return ProblemSpec(
            status=exc.status,
            code=exc.code,
            detail=detail,
            retryable=False,
        )
    if isinstance(exc, RateTableInvalidError):
        detail = exc.detail or str(exc)
        return ProblemSpec(
            status=exc.status,
            code=exc.code,
            detail=detail,
            retryable=exc.retryable,
        )
    return None
