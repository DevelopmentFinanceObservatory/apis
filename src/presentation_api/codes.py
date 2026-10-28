"""Presentation refusal codes and retry modes. The surface defines assembler-specific codes
and uses the platform registry for shared codes. Unknown component types map to
`schema_invalid` because the catalogue type union is closed. presentation-mcp's
`transient_unavailable` maps to HTTP `service_unavailable`.

Malformed requests use `request_invalid` with `retryable=False`. An unknown `schema_version`
uses platform code `unsupported_schema_version` (422) with `retryable=False`, outside
`CODE_RETRY_MODE`. Schema, binding, and constraint refusals carry correction details.
`schema_invalid` and `frame_unavailable` vary by cause. Each response's `retryable` value is
authoritative. `request_invalid` has a retryable headline mode but always returns false.

`schema_invalid`, `binding_invalid`, `constraint_violation`, `request_invalid`, and
`resource_exhausted` return 400. `write_refused` returns 403, `edit_conflict` returns 409, and
`not_found` returns 404. The render routes refuse a caller other than the live-refresh
principal with `refresh_not_permitted` (403) and a target frame that may not stand in for the
stored one with `render_target_mismatch` (409). They serve `binding_invalid` and
`constraint_violation` at 422 and `frame_unavailable`, `release_mismatch`, and
`resource_exhausted` at 409 (`presentation_api.doors.render`). Their caller replays a stored
intent and cannot correct it, so there `retryable` is true only when the same request may
succeed later: `replay_retryable` keeps an occurrence's value for a `retry_after_delay` or
`refresh_then_retry` code and answers false for every other mode.
Frame unavailability returns 404 or transient 503. Release mismatch and service unavailability
return 503. Timeout returns 504, and internal errors return 500.

Missing-field `schema_invalid`, `binding_invalid`, and `constraint_violation` use
`correct_and_retry`, which `retryable_for` answers false: the request must change before it can
succeed, and a client retries a retryable refusal unchanged. Illegal fields use `do_not_retry`. Conflicts and some frame refusals need a
refresh. Transient frame, release, service, and timeout failures retry after a delay. `not_found`,
`write_refused`, `resource_exhausted`, `refresh_not_permitted`, `render_target_mismatch`, and
`internal_error` use `do_not_retry`. `CODE_RETRY_MODE` mirrors the
release manifest.

The shares door (`presentation_api.doors.shares`) serves `SHARE_CODE_CATALOGUE` beside these:
`write_token_required` and `write_token_invalid` at 403, `share_not_replaceable` at 409 and
`payload_too_large` at 413, none retryable, and the platform's `allowance_exhausted` at 429, retryable after `Retry-After`. The
component catalogue's release manifest lists no share code, so they stay out of
`CODE_RETRY_MODE`. `PRESENTATION_SERVED_CODES` is every code the surface serves.

Problem details expose fixed extension fields and no generic extra-data slot. Legal-value lists
use `alternatives`, structured facet failures use `issues`, and details such as `minimal_intent`,
constraint fixes, and limit names appear in `detail`.
"""

from __future__ import annotations

from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, CodeCatalogue, merge_catalogues

__all__ = [
    "CODE_CATALOG",
    "CODE_RETRY_MODE",
    "PRESENTATION_CODE_CATALOGUE",
    "PRESENTATION_SERVED_CODES",
    "PRESENTATION_SERVED_PLATFORM_CODES",
    "SHARE_CODE_CATALOGUE",
    "SHARE_SERVED_PLATFORM_CODES",
    "replay_retryable",
    "retryable_for",
]

PRESENTATION_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "schema_invalid": (
        "Schema Invalid",
        "The submitted intent does not satisfy the catalogue's schema for its type.",
    ),
    "binding_invalid": (
        "Binding Invalid",
        "A selector could not be resolved against the frame.",
    ),
    "constraint_violation": (
        "Constraint Violation",
        "The resolved component fails one of the catalogue's blocking constraints.",
    ),
    "edit_conflict": (
        "Edit Conflict",
        "The component or composition moved between this call's own read and its write.",
    ),
    "write_refused": (
        "Write Refused",
        "The x-dfo-write-token sent does not grant writing this component or composition.",
    ),
    "frame_unavailable": (
        "Frame Unavailable",
        "The referenced frame cannot be loaded right now.",
    ),
    "resource_exhausted": (
        "Resource Exhausted",
        "The request exceeds one of this deployment's resource maximums.",
    ),
    "release_mismatch": (
        "Release Mismatch",
        "This component's head was minted under a newer catalog_version than this deployment "
        "writes, or its stored payload carries a shape the release's schema does not "
        "recognise, a row a migration has not yet rewritten.",
    ),
    "render_target_mismatch": (
        "Render Target Mismatch",
        "The target frame may not stand in for the frame this revision is stored on: it is "
        "another dataset or rendering, or a derived frame of another recipe.",
    ),
}

# The shares door's own codes.
SHARE_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "write_token_required": (
        "Write Token Required",
        "A replace or delete of a share carries no x-dfo-write-token.",
    ),
    "write_token_invalid": (
        "Write Token Invalid",
        "The x-dfo-write-token sent is not the share's, or the share was made without one, "
        "which no token can change.",
    ),
    "share_not_replaceable": (
        "Share Not Replaceable",
        "The share is a snapshot, whose cards are fixed for good. Only a live share takes new "
        "cards.",
    ),
    "payload_too_large": (
        "Payload Too Large",
        "A share's request body is larger than the share routes accept.",
    ),
}

# This catalogue adds presentation codes to the shared platform catalogue.
CODE_CATALOG: dict[str, tuple[str, str]] = merge_catalogues(
    PLATFORM_CODE_CATALOGUE, PRESENTATION_CODE_CATALOGUE, SHARE_CODE_CATALOGUE
)

# Headline retry mode per served code, mirrored in the baked release manifest.
CODE_RETRY_MODE: dict[str, str] = {
    "schema_invalid": "correct_and_retry",
    "binding_invalid": "correct_and_retry",
    "constraint_violation": "correct_and_retry",
    "edit_conflict": "refresh_then_retry",
    "write_refused": "do_not_retry",
    "not_found": "do_not_retry",
    "request_invalid": "correct_and_retry",
    "resource_exhausted": "do_not_retry",
    "frame_unavailable": "refresh_then_retry",
    "release_mismatch": "retry_after_delay",
    "refresh_not_permitted": "do_not_retry",
    "render_target_mismatch": "do_not_retry",
    "service_unavailable": "retry_after_delay",
    "timeout": "retry_after_delay",
    "internal_error": "do_not_retry",
}

# Platform codes rendered by this surface, including validation's schema-version refusal.
PRESENTATION_SERVED_PLATFORM_CODES: frozenset[str] = frozenset(
    {
        "not_found",
        "request_invalid",
        "service_unavailable",
        "timeout",
        "internal_error",
        "unsupported_schema_version",
        # Shared with the Datasets frame refresh.
        "refresh_not_permitted",
    }
)

# Platform codes the shares door renders and the component catalogue does not.
SHARE_SERVED_PLATFORM_CODES: frozenset[str] = frozenset({"allowance_exhausted"})

PRESENTATION_SERVED_CODES: frozenset[str] = (
    frozenset(PRESENTATION_CODE_CATALOGUE)
    | PRESENTATION_SERVED_PLATFORM_CODES
    | frozenset(SHARE_CODE_CATALOGUE)
    | SHARE_SERVED_PLATFORM_CODES
)

_expected_codes = set(PRESENTATION_CODE_CATALOGUE) | (
    PRESENTATION_SERVED_PLATFORM_CODES - {"unsupported_schema_version"}
)
if set(CODE_RETRY_MODE) != _expected_codes:
    # Fail at import if a served code and its manifest retry mode drift apart.
    raise AssertionError(
        f"CODE_RETRY_MODE {sorted(CODE_RETRY_MODE)} does not match the expected code set "
        f"{sorted(_expected_codes)} (PRESENTATION_CODE_CATALOGUE plus the platform codes this "
        "surface renders, less unsupported_schema_version) -- every code either side declares "
        "needs a matching entry on the other"
    )

_served_platform_codes = PRESENTATION_SERVED_PLATFORM_CODES | SHARE_SERVED_PLATFORM_CODES
if not frozenset(PLATFORM_CODE_CATALOGUE) >= _served_platform_codes:
    raise AssertionError(
        f"the served platform codes name code(s) "
        f"{sorted(_served_platform_codes - frozenset(PLATFORM_CODE_CATALOGUE))} "
        "absent from PLATFORM_CODE_CATALOGUE"
    )


# The modes under which a request repeated unchanged may succeed.
_UNCHANGED_RETRY_MODES = frozenset({"retry_after_delay", "refresh_then_retry"})


def retryable_for(code: str) -> bool:
    """Return whether the same request, sent again, may succeed: true for a
    `retry_after_delay` or `refresh_then_retry` code. A `correct_and_retry` code answers false,
    since the caller must change its request first, and a client retries a retryable refusal
    unchanged. Classifiers handle `schema_invalid`, `frame_unavailable`, and `request_invalid`
    per occurrence.
    """
    return CODE_RETRY_MODE[code] in _UNCHANGED_RETRY_MODES


def replay_retryable(code: str, *, retryable: bool) -> bool:
    """Return `retryable` as a caller that cannot change its request reads it: the
    occurrence's own value when `code`'s mode is in `_UNCHANGED_RETRY_MODES`, else false.
    """
    return retryable and CODE_RETRY_MODE.get(code) in _UNCHANGED_RETRY_MODES


def _catalogue_covers(catalogue: CodeCatalogue) -> None:
    """Raise at import if a retry-mode code is absent from the merged catalogue."""
    missing = set(CODE_RETRY_MODE) - set(catalogue)
    if missing:
        raise AssertionError(
            f"CODE_RETRY_MODE names code(s) {sorted(missing)} absent from CODE_CATALOG"
        )


_catalogue_covers(CODE_CATALOG)
