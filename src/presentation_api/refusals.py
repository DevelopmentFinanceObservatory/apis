"""Shared refusal renderers for assembler errors and FastAPI request validation.
Request validation runs before route handlers, so the surface composes its classifier outside
any individual door.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from fastapi.exceptions import RequestValidationError

from assembler.errors import BindingInvalid, FrameUnavailable
from assembler.ports import FRAME_UNAVAILABLE_OUTCOME
from dfo_platform import Alternative, Cause, ProblemIssue, ProblemSpec, refused
from dfo_platform.codes import RetryMode, registered_status
from dfo_platform.validation import validation_problem
from presentation_api.codes import CODE_RETRY_MODE, UNCHANGED_RETRY_MODES, retryable_for

__all__ = [
    "EXPIRED_FRAME_FIX",
    "FRAME_UNAVAILABLE_EXEMPLARS",
    "NEWEST_NOT_READY_OUTCOME",
    "binding_invalid_spec",
    "frame_unavailable_spec",
    "request_validation_spec",
    "retry_after_headers",
    "retry_mode",
    "retry_mode_codes",
    "retry_mode_table",
    "with_retry_mode",
]

# The modes of the codes `CODE_RETRY_MODE` does not name: the shares door's codes, the daily
# allowance, and the platform's own refusals. Every code the surface can answer is named here
# or there, and none in both (`tests/presentation_api/test_refusal_registry.py`).
_STATED_MODES: Mapping[str, RetryMode] = MappingProxyType(
    {
        "payload_too_large": "correct_and_retry",
        # A replace or delete sent without its write token succeeds with the header.
        "write_token_required": "correct_and_retry",
        # A body carrying a top-level `schema_version` succeeds without it.
        "unsupported_schema_version": "correct_and_retry",
        # The allowance resets at the time the response's `Retry-After` names.
        "allowance_exhausted": "retry_after_delay",
        "write_token_invalid": "do_not_retry",
        "share_not_replaceable": "do_not_retry",
        "share_id_taken": "do_not_retry",
        "share_import_not_permitted": "do_not_retry",
        "method_not_allowed": "do_not_retry",
    }
)


def retry_mode(spec: ProblemSpec, *, stored_intent: bool) -> RetryMode:
    """What a caller does with the refusal `spec`, stated on every Presentation problem body as
    `retry_mode` (`with_retry_mode`). `stored_intent` is true for a refusal a render route or
    a share slot's read raised while replaying a revision's stored intent
    (`presentation_api.stored_intent.StoredIntentRefused`).

    A code states its headline mode (`CODE_RETRY_MODE`), or the mode `_STATED_MODES` gives a
    code outside it. A headline mode that repeats the request unchanged holds only where
    the occurrence is `retryable`, since `frame_unavailable` varies by cause. A replaying caller
    cannot change the stored intent it replays, so a refusal of that intent states
    `do_not_retry` where it would correct. Every refusal of a request itself, a render's body,
    path, `schema_version` and headers included, keeps its mode.

    Raises:
        KeyError: neither table names `spec.code`. Every code the surface answers is named
            (`tests/presentation_api/test_refusal_registry.py`), and the platform renders a
            refusal whose `finish` raises as it was classified."""
    mode = _STATED_MODES.get(spec.code) or CODE_RETRY_MODE[spec.code]
    if spec.code not in _STATED_MODES and mode in UNCHANGED_RETRY_MODES and not spec.retryable:
        return "do_not_retry"
    if stored_intent and mode == "correct_and_retry":
        return "do_not_retry"
    return mode


def retry_mode_codes() -> frozenset[str]:
    """Every code `retry_mode` states a mode for."""
    return frozenset(_STATED_MODES) | frozenset(CODE_RETRY_MODE)


def retry_mode_table() -> dict[str, RetryMode]:
    """`retry_mode`'s answer for every code it names, retryable or not, on a request and on a
    replayed stored intent, keyed `request|code|retryable` or `stored_intent|code|retryable`,
    the rule's only inputs. The capability version hashes it, so a change to the rule moves
    `capability_version`."""
    return {
        f"{'stored_intent' if stored_intent else 'request'}|{code}|{retryable}": retry_mode(
            ProblemSpec(400, code, retryable), stored_intent=stored_intent
        )
        for stored_intent in (False, True)
        for code in sorted(retry_mode_codes())
        for retryable in (False, True)
    }


def with_retry_mode(spec: ProblemSpec) -> ProblemSpec:
    """`spec` stating its `retry_mode`, the Presentation surface's `finish`, so every body it
    answers states the member. A refusal of a replayed stored intent arrives with the member
    set by the components door's classifier (`doors.components.classify_components_exception`)
    and keeps it. Any other states the mode of a refusal of the request itself."""
    if spec.retry_mode is not None:
        return spec
    return spec._replace(retry_mode=retry_mode(spec, stored_intent=False))


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
        for reason, (code, _, _) in FRAME_UNAVAILABLE_OUTCOME.items()
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
    code, retryable, _ = FRAME_UNAVAILABLE_OUTCOME[exc.reason]
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
