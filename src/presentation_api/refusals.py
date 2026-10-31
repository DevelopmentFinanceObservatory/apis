"""Shared refusal renderers for assembler errors and FastAPI request validation.
Request validation runs before route handlers, so the surface composes its classifier outside
any individual door.
"""

from __future__ import annotations

from fastapi.exceptions import RequestValidationError

from assembler.errors import BindingInvalid, FrameUnavailable
from dfo_platform import Alternative, Cause, ProblemIssue, ProblemSpec
from dfo_platform.codes import bound_registry_endpoint, registered_status
from dfo_platform.validation import validation_problem
from presentation_api.codes import retryable_for

__all__ = [
    "EXPIRED_FRAME_FIX",
    "WRITE_ENDPOINT",
    "binding_invalid_spec",
    "frame_unavailable_spec",
    "request_validation_spec",
]

# The code registry endpoint of the card writes. The registry maps no endpoint onto the read
# routes, which share the write routes' classifiers and so their statuses.
WRITE_ENDPOINT = "presentation.write"


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


def _outcome(code: str, *, retryable: bool) -> tuple[str, int, bool]:
    """`code` and `retryable` with the status the code registry gives `code` on the write
    routes."""
    with bound_registry_endpoint(WRITE_ENDPOINT):
        return code, registered_status(code), retryable


# The code, status, and retryability each `FrameUnavailable` reason refuses with. `expired` and
# `ops_unsupported` can succeed after republishing. Revoked, missing, digest mismatch,
# quarantined, and unknown refs are permanent. A store that fails to answer (`io`), a derived
# frame whose recipe is not written yet (`recipe_pending`), which a recipe write lets a retry pin,
# and a recipe the store cannot read (`recipe_unreadable`) are `service_unavailable`.
_FRAME_UNAVAILABLE_OUTCOME: dict[str, tuple[str, int, bool]] = {
    "expired": _outcome("frame_unavailable", retryable=True),
    "revoked": _outcome("frame_unavailable", retryable=False),
    "ops_unsupported": _outcome("frame_unavailable", retryable=True),
    "missing_rendering": _outcome("frame_unavailable", retryable=False),
    "digest_mismatch": _outcome("frame_unavailable", retryable=False),
    "quarantined": _outcome("frame_unavailable", retryable=False),
    "io": _outcome("service_unavailable", retryable=True),
    "recipe_pending": _outcome("service_unavailable", retryable=True),
    "recipe_unreadable": _outcome("service_unavailable", retryable=True),
    "unknown": _outcome("frame_unavailable", retryable=False),
}


# An expired frame aged out of the working tier. A card made on it draws again from current
# data, which a fetch of its ref with latest=True returns.
EXPIRED_FRAME_FIX = (
    "the frame aged out of the 30-day working tier; fetch its ref with latest=True, as "
    "dfo.fetch(..., latest=True) does, to draw from current data"
)


def frame_unavailable_spec(exc: FrameUnavailable) -> ProblemSpec:
    """The refusal `_FRAME_UNAVAILABLE_OUTCOME` gives `exc.reason`. A `frame_unavailable` names
    the reason as `cause.code`, so a client tells a frame it can rebuild from one taken down, and
    an expired frame's detail ends with `EXPIRED_FRAME_FIX`."""
    code, status, retryable = _FRAME_UNAVAILABLE_OUTCOME[exc.reason]
    detail = f"{exc.detail}; {EXPIRED_FRAME_FIX}" if exc.reason == "expired" else exc.detail
    return ProblemSpec(
        status=status,
        code=code,
        detail=detail,
        retryable=retryable,
        cause=Cause(code=exc.reason) if code == "frame_unavailable" else None,
    )


def request_validation_spec(exc: BaseException) -> ProblemSpec | None:
    """Return the platform validation spec unchanged, or `None` for other exceptions.
    Malformed requests inherit `retryable=False`.
    """
    if not isinstance(exc, RequestValidationError):
        return None
    return validation_problem(exc)
