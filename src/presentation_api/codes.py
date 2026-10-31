"""Presentation refusal codes and retry modes. The surface defines assembler-specific codes
and uses the platform registry for shared codes. Unknown component types map to
`schema_invalid` because the catalogue type union is closed. presentation-mcp's
`transient_unavailable` maps to HTTP `service_unavailable`.

Malformed requests use `request_invalid` with `retryable=False`. An unknown `schema_version`
uses platform code `unsupported_schema_version` (422) with `retryable=False`, outside
`CODE_RETRY_MODE`. Schema, binding, and constraint refusals carry correction details.
`schema_invalid` and `frame_unavailable` vary by cause. Each response's `retryable` value is
authoritative. `request_invalid` has a retryable headline mode but always returns false.

Every classifier reads a refusal's status from the code registry under the endpoint the platform
binds for the route (`served_status`), and `presentation_statuses` holds the registry's map for
the surface. `schema_invalid`, `binding_invalid`, `constraint_violation`, `request_invalid`, and
`resource_exhausted` are 400 on the write routes, and `frame_unavailable` 404. The render routes
serve `binding_invalid` and `constraint_violation` at 422 and `frame_unavailable`,
`release_mismatch`, and `resource_exhausted` at 409. Their caller replays a stored intent and
cannot correct it, so there `retryable` is true only when the same request may succeed later:
`replay_retryable` keeps an occurrence's value for a `retry_after_delay` or `refresh_then_retry`
code and answers false for every other mode. `presentation_api.refusals` names the code of each
`FrameUnavailable` reason.

Missing-field `schema_invalid`, `binding_invalid`, and `constraint_violation` use
`correct_and_retry`, which `retryable_for` answers false. The request must change before it can
succeed, and a client retries a retryable refusal unchanged. Illegal fields use `do_not_retry`. Conflicts and some frame refusals need a
refresh. Transient release, service, and timeout failures retry after a delay. `not_found`,
`write_refused`, `resource_exhausted`, `refresh_not_permitted`, `render_target_mismatch`, and
`internal_error` use `do_not_retry`. `CODE_RETRY_MODE` mirrors the
release manifest.

The shares door (`presentation_api.doors.shares`) serves `SHARE_CODE_CATALOGUE` beside these:
`write_token_required`, `write_token_invalid` and `share_import_not_permitted` at 403,
`share_not_replaceable` and `share_id_taken` at 409 and `payload_too_large` at 413, none
retryable. A card create or revise and a share create refuse
the platform's `allowance_exhausted`, not retryable either, with `Retry-After` naming
the reset (`WRITE_SERVED_PLATFORM_CODES`). The component catalogue's release manifest lists
none of these, so they stay out of `CODE_RETRY_MODE`. `PRESENTATION_SERVED_CODES` is every code the surface
serves.

Problem details expose fixed extension fields and no generic extra-data slot. Legal-value lists
use `alternatives`, structured facet failures use `issues`, and details such as `minimal_intent`,
constraint fixes, and limit names appear in `detail`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from dfo_spec_kit.registry import REGISTRY

from dfo_platform.codes import (
    PLATFORM_CODE_CATALOGUE,
    CodeCatalogue,
    merge_catalogues,
    registered_status,
)

__all__ = [
    "CODE_CATALOG",
    "CODE_RETRY_MODE",
    "COMPONENT_DATA_ENDPOINT",
    "COMPONENT_READ_ENDPOINT",
    "FRAMES_ENDPOINT",
    "PRESENTATION_CODE_CATALOGUE",
    "PRESENTATION_SERVED_CODES",
    "PRESENTATION_SERVED_PLATFORM_CODES",
    "RENDER_ENDPOINT",
    "SHARES_ENDPOINT",
    "SHARE_CODE_CATALOGUE",
    "WRITE_ENDPOINT",
    "WRITE_SERVED_PLATFORM_CODES",
    "presentation_statuses",
    "replay_retryable",
    "retryable_for",
    "served_status",
]

logger = logging.getLogger(__name__)

# The code registry endpoints the surface's routes serve.
WRITE_ENDPOINT = "presentation.write"
COMPONENT_READ_ENDPOINT = "presentation.component_read"
COMPONENT_DATA_ENDPOINT = "presentation.component_data"
RENDER_ENDPOINT = "presentation.render"
FRAMES_ENDPOINT = "presentation.frames"
SHARES_ENDPOINT = "presentation.shares"

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
        "The referenced frame is not held in a state Presentation can load.",
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
    "share_id_taken": (
        "Share Id Taken",
        "A share import names an id that holds another share, or holds this share under "
        "another write token digest.",
    ),
    "share_import_not_permitted": (
        "Share Import Not Permitted",
        "The share import is restricted to its first-party principal.",
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

# Platform codes the card and share writes render that the component catalogue's manifest
# does not list. Today that is the daily allowance's refusal.
WRITE_SERVED_PLATFORM_CODES: frozenset[str] = frozenset({"allowance_exhausted"})

PRESENTATION_SERVED_CODES: frozenset[str] = (
    frozenset(PRESENTATION_CODE_CATALOGUE)
    | PRESENTATION_SERVED_PLATFORM_CODES
    | frozenset(SHARE_CODE_CATALOGUE)
    | WRITE_SERVED_PLATFORM_CODES
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

_served_platform_codes = PRESENTATION_SERVED_PLATFORM_CODES | WRITE_SERVED_PLATFORM_CODES
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


@cache
def presentation_statuses() -> Mapping[str, Mapping[str, int]]:
    """The status the code registry gives each code's problem details, by `presentation.*`
    endpoint."""
    statuses: dict[str, dict[str, int]] = {}
    for entry in REGISTRY.entries:
        for mapping in entry.mappings:
            if (
                mapping.endpoint.startswith("presentation.")
                and mapping.channel == "problem_details"
                and mapping.http_status is not None
            ):
                statuses.setdefault(mapping.endpoint, {})[entry.code] = mapping.http_status
    return MappingProxyType({endpoint: statuses[endpoint] for endpoint in sorted(statuses)})


def served_status(code: str) -> int:
    """The status the code registry gives `code` on the bound endpoint. A code that endpoint
    does not map answers the status of the first Presentation endpoint that does,
    `presentation.write` before the others, and logs an error, so the client reads the refusal
    and not a 500. `tests/presentation_api/test_refusal_registry.py` fails on such a pair.

    Raises:
        LookupError: No Presentation endpoint maps `code`.
    """
    try:
        return registered_status(code)
    except LookupError:
        statuses = presentation_statuses()
        for endpoint in (WRITE_ENDPOINT, *statuses):
            if code in statuses[endpoint]:
                logger.error("The bound registry endpoint maps no status for %s", code)
                return statuses[endpoint][code]
        raise


def _catalogue_covers(catalogue: CodeCatalogue) -> None:
    """Raise at import if a retry-mode code is absent from the merged catalogue."""
    missing = set(CODE_RETRY_MODE) - set(catalogue)
    if missing:
        raise AssertionError(
            f"CODE_RETRY_MODE names code(s) {sorted(missing)} absent from CODE_CATALOG"
        )


_catalogue_covers(CODE_CATALOG)
