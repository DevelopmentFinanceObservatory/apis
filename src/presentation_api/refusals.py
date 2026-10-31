"""Shared refusal renderers for assembler errors and FastAPI request validation.
Request validation runs before route handlers, so the surface composes its classifier outside
any individual door.
"""

from __future__ import annotations

import logging
from typing import get_args

from fastapi.exceptions import RequestValidationError

from assembler.errors import BindingInvalid, FrameUnavailable
from assembler.ports import FrameUnavailableReason
from dfo_platform import Alternative, Cause, ProblemIssue, ProblemSpec, refused
from dfo_platform.codes import registered_status
from dfo_platform.validation import validation_problem
from presentation_api.codes import retryable_for

__all__ = [
    "EXPIRED_FRAME_FIX",
    "FRAME_UNAVAILABLE_EXEMPLARS",
    "binding_invalid_spec",
    "frame_unavailable_spec",
    "request_validation_spec",
]

logger = logging.getLogger(__name__)


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


# The code and retryability each `FrameUnavailable` reason refuses with. `expired`,
# `ops_unsupported` and `recipe_pending` can succeed once the frame is republished or its recipe
# written. Revoked, missing, digest mismatch, quarantined, and unknown refs are permanent. A store
# that fails to answer (`io`) is `service_unavailable`. A recipe the store cannot read (`recipe_unreadable`) is write-once, so no
# retry clears it.
_FRAME_UNAVAILABLE_OUTCOME: dict[FrameUnavailableReason, tuple[str, bool]] = {
    "expired": ("frame_unavailable", True),
    "revoked": ("frame_unavailable", False),
    "ops_unsupported": ("frame_unavailable", True),
    "missing_rendering": ("frame_unavailable", False),
    "digest_mismatch": ("frame_unavailable", False),
    "quarantined": ("frame_unavailable", False),
    "recipe_pending": ("frame_unavailable", True),
    "unknown": ("frame_unavailable", False),
    "io": ("service_unavailable", True),
    "recipe_unreadable": ("internal_error", False),
}

if set(_FRAME_UNAVAILABLE_OUTCOME) != set(get_args(FrameUnavailableReason)):
    # Fail at import if a reason and its outcome drift apart.
    raise AssertionError(
        f"_FRAME_UNAVAILABLE_OUTCOME {sorted(_FRAME_UNAVAILABLE_OUTCOME)} does not match "
        f"FrameUnavailableReason {sorted(get_args(FrameUnavailableReason))}"
    )

# The reasons that name a server fault.
_LOGGED_REASONS: frozenset[str] = frozenset({"io", "recipe_unreadable"})

# One exemplar of every reason, for each door whose routes load or pin a frame.
FRAME_UNAVAILABLE_EXEMPLARS: tuple[FrameUnavailable, ...] = tuple(
    FrameUnavailable(reason=reason, detail="exemplar") for reason in _FRAME_UNAVAILABLE_OUTCOME
)


# An expired frame aged out of the working tier. A card made on it draws again from current
# data, which a fetch of its ref with latest=True returns.
EXPIRED_FRAME_FIX = (
    "the frame aged out of the 30-day working tier; fetch its ref with latest=True, as "
    "dfo.fetch(..., latest=True) does, to draw from current data"
)


def frame_unavailable_spec(exc: FrameUnavailable) -> ProblemSpec:
    """The refusal `_FRAME_UNAVAILABLE_OUTCOME` gives `exc.reason`, at the bound endpoint's status,
    with the reason as `cause.code`, so a client tells a frame it can rebuild from one taken
    down. An expired frame's detail ends with `EXPIRED_FRAME_FIX`."""
    code, retryable = _FRAME_UNAVAILABLE_OUTCOME[exc.reason]
    if exc.reason in _LOGGED_REASONS:
        logger.error("Frame refused for reason %s: %s", exc.reason, exc.detail)
    detail = f"{exc.detail}; {EXPIRED_FRAME_FIX}" if exc.reason == "expired" else exc.detail
    return refused(code, detail=detail, retryable=retryable, cause=Cause(code=exc.reason))


def request_validation_spec(exc: BaseException) -> ProblemSpec | None:
    """Return the platform validation spec at the registry's status, or `None` for other
    exceptions. Malformed requests inherit `retryable=False`.
    """
    if not isinstance(exc, RequestValidationError):
        return None
    spec = validation_problem(exc)
    return spec._replace(status=registered_status(spec.code))
