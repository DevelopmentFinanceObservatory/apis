"""Graph-owned `node_not_held` code and the shared platform codes used by Graph doors.

The platform already owns `not_found`, `method_not_allowed`, `timeout`,
and `internal_error`. Graph adds no registry mapping for them.
"""

from __future__ import annotations

from dfo_platform.codes import PLATFORM_CODE_CATALOGUE

GRAPH_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "node_not_held": (
        "Node Not Held",
        "This deployment's graph does not hold the requested node.",
    ),
}

# Node maps oversized dcids to `request_invalid` and DCP failures to
# `upstream_error` or `service_unavailable`. Resolve maps DCP failures the
# same way. Request validation yields `request_invalid` or, for a forbidden
# `schema_version`, `unsupported_schema_version`.
SHARED_PLATFORM_CODES: frozenset[str] = frozenset(
    {
        "request_invalid",
        "upstream_error",
        "service_unavailable",
        "unsupported_schema_version",
    }
)
if not frozenset(PLATFORM_CODE_CATALOGUE) >= SHARED_PLATFORM_CODES:
    raise AssertionError(
        f"SHARED_PLATFORM_CODES names code(s) "
        f"{sorted(SHARED_PLATFORM_CODES - frozenset(PLATFORM_CODE_CATALOGUE))} absent from "
        "PLATFORM_CODE_CATALOGUE"
    )
