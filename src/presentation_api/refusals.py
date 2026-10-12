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


# `expired` and `ops_unsupported` can succeed after republishing. Revoked, missing, digest
# mismatch, quarantined, and unknown refs are permanent. I/O maps to retryable 503.
_FRAME_UNAVAILABLE_OUTCOME: dict[str, tuple[int, bool]] = {
    "expired": (404, True),
    "revoked": (404, False),
    "ops_unsupported": (404, True),
    "missing_rendering": (404, False),
    "digest_mismatch": (404, False),
    "quarantined": (404, False),
    "io": (503, True),
    "unknown": (404, False),
}


def frame_unavailable_spec(exc: FrameUnavailable) -> ProblemSpec:
    status, retryable = _FRAME_UNAVAILABLE_OUTCOME.get(exc.reason, (503, True))
    return ProblemSpec(
        status=status, code="frame_unavailable", detail=exc.detail, retryable=retryable
    )


def request_validation_spec(exc: BaseException) -> ProblemSpec | None:
    """Return the platform validation spec unchanged, or `None` for other exceptions.
    Malformed requests inherit `retryable=False`.
    """
    if not isinstance(exc, RequestValidationError):
        return None
    return validation_problem(exc)
