"""Shared refusal renderers for assembler errors and FastAPI request validation.
Request validation runs before route handlers, so the surface composes its classifier outside
any individual door.
"""

from __future__ import annotations

from fastapi.exceptions import RequestValidationError

from assembler.errors import BindingInvalid, FrameUnavailable
from dfo_platform import Alternative, ProblemIssue, ProblemSpec
from dfo_platform.validation import validation_problem
from presentation_api.codes import retryable_for

__all__ = ["binding_invalid_spec", "frame_unavailable_spec", "request_validation_spec"]


def binding_invalid_spec(exc: BindingInvalid) -> ProblemSpec:
    """Render legal values as alternatives, using labels when available. A structural facet
    failure adds one issue per dimension.
    """
    issues = None
    if exc.structural_view is not None:
        issues = [
            ProblemIssue(
                axis=dimension.name,
                alternatives=[
                    Alternative(ref=value, label=exc.labels.get(value, value))
                    for value in dimension.values
                ],
                truncated=dimension.truncated,
            )
            for dimension in exc.structural_view.dimensions
        ]
    return ProblemSpec(
        status=400,
        code="binding_invalid",
        detail=exc.detail,
        pointer=exc.pointer,
        alternatives=[
            Alternative(ref=value, label=exc.labels.get(value, value)) for value in exc.legal_values
        ]
        or None,
        truncated=exc.truncated,
        issues=issues,
        retryable=retryable_for("binding_invalid"),
    )


# The code, status, and retryability each `FrameUnavailable` reason refuses with. `expired` and
# `ops_unsupported` can succeed after republishing. Revoked, missing, digest mismatch,
# quarantined, and unknown refs are permanent. A store that fails to answer (`io`) is
# `service_unavailable`.
_FRAME_UNAVAILABLE_OUTCOME: dict[str, tuple[str, int, bool]] = {
    "expired": ("frame_unavailable", 404, True),
    "revoked": ("frame_unavailable", 404, False),
    "ops_unsupported": ("frame_unavailable", 404, True),
    "missing_rendering": ("frame_unavailable", 404, False),
    "digest_mismatch": ("frame_unavailable", 404, False),
    "quarantined": ("frame_unavailable", 404, False),
    "io": ("service_unavailable", 503, True),
    "unknown": ("frame_unavailable", 404, False),
}


def frame_unavailable_spec(exc: FrameUnavailable) -> ProblemSpec:
    code, status, retryable = _FRAME_UNAVAILABLE_OUTCOME[exc.reason]
    return ProblemSpec(status=status, code=code, detail=exc.detail, retryable=retryable)


def request_validation_spec(exc: BaseException) -> ProblemSpec | None:
    """Return the platform validation spec unchanged, or `None` for other exceptions.
    Malformed requests inherit `retryable=False`.
    """
    if not isinstance(exc, RequestValidationError):
        return None
    return validation_problem(exc)
