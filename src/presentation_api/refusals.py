"""Shared refusal renderers for assembler errors and FastAPI request validation.
Request validation runs before route handlers, so the surface composes its classifier outside
any individual door.
"""

from __future__ import annotations

import math

from fastapi.exceptions import RequestValidationError

from assembler.errors import BindingInvalid, FrameUnavailable
from assembler.ports import FRAME_UNAVAILABLE_OUTCOME
from dfo_platform import Alternative, Cause, ProblemIssue, ProblemSpec, refused
from dfo_platform.codes import registered_status
from dfo_platform.validation import validation_problem
from presentation_api.codes import retryable_for

__all__ = [
    "EXPIRED_FRAME_FIX",
    "FRAME_UNAVAILABLE_EXEMPLARS",
    "NEWEST_NOT_READY_OUTCOME",
    "binding_invalid_spec",
    "frame_unavailable_spec",
    "request_validation_spec",
    "retry_after_headers",
]


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
    return refused(
        "binding_invalid",
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


# The reasons that name a fault to look into: every one that is no frame state but `busy`, which
# is ordinary load, and a stored artifact failing its digest. The frame store logs
# `missing_rendering` where bytes do not decode.
_LOGGED_REASONS: frozenset[str] = (
    frozenset(
        reason
        for reason, (code, _) in FRAME_UNAVAILABLE_OUTCOME.items()
        if code != "frame_unavailable"
    )
    - {"busy"}
) | {"digest_mismatch"}

# The code and retryability of a bare `spec_id` whose newest version is not ready.
NEWEST_NOT_READY_OUTCOME: tuple[str, bool] = ("service_unavailable", True)

# One exemplar of every reason, for each door whose routes load or pin a frame.
FRAME_UNAVAILABLE_EXEMPLARS: tuple[FrameUnavailable, ...] = tuple(
    FrameUnavailable(reason=reason, detail="exemplar") for reason in FRAME_UNAVAILABLE_OUTCOME
)


# An expired frame aged out of the working tier. A card made on it draws again from current
# data, which a fetch of its ref with latest=True returns.
EXPIRED_FRAME_FIX = (
    "the frame aged out of the 30-day working tier; fetch its ref with latest=True, as "
    "dfo.fetch(..., latest=True) does, to draw from current data"
)


def frame_unavailable_spec(exc: FrameUnavailable) -> ProblemSpec:
    """The refusal `FRAME_UNAVAILABLE_OUTCOME` gives `exc.reason`, at the bound endpoint's status,
    with the reason as `cause.code`, so a client tells a frame it can rebuild from one taken
    down. An expired frame's detail ends with `EXPIRED_FRAME_FIX`."""
    code, retryable = FRAME_UNAVAILABLE_OUTCOME[exc.reason]
    detail = f"{exc.detail}; {EXPIRED_FRAME_FIX}" if exc.reason == "expired" else exc.detail
    return refused(
        code,
        detail=detail,
        retryable=retryable,
        cause=Cause(code=exc.reason),
        headers=None if exc.retry_after_s is None else retry_after_headers(exc.retry_after_s),
        log=(
            f"Frame refused for reason {exc.reason}: {exc.detail}"
            if exc.reason in _LOGGED_REASONS
            else None
        ),
    )


def retry_after_headers(seconds: float) -> dict[str, str]:
    return {"Retry-After": str(math.ceil(seconds))}


def request_validation_spec(exc: BaseException) -> ProblemSpec | None:
    """Return the platform validation spec at the registry's status, or `None` for other
    exceptions. Malformed requests inherit `retryable=False`.
    """
    if not isinstance(exc, RequestValidationError):
        return None
    spec = validation_problem(exc)
    return spec._replace(status=registered_status(spec.code))
