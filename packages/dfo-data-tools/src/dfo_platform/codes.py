"""Shared refusal codes and the response body for `GET /codes/{code}`.

Each API owns its surface-specific catalogue. This module holds codes shared across
surfaces. The response combines catalogue titles and details with optional metadata from
`dfo_spec_kit.registry.REGISTRY`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

from dfo_spec_kit.registry import REGISTRY, Classification, CodeEntry, Lifecycle
from pydantic import BaseModel

# code -> (title, default_detail).
CodeCatalogue = Mapping[str, tuple[str, str]]

PLATFORM_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    # Clients depend on the titles and default details in `/codes/{code}` bodies.
    "request_invalid": ("Bad Request", "Bad request"),
    "unsupported_schema_version": (
        "Unsupported Schema Version",
        "This deployment does not support the requested schema_version.",
    ),
    "upstream_error": (
        "Bad Gateway",
        "Bad gateway: upstream returned an error",
    ),
    "service_unavailable": (
        "Service Unavailable",
        "Service temporarily unavailable",
    ),
    "timeout": (
        "Request Timeout",
        "Request timed out",
    ),
    "internal_error": (
        "Internal Server Error",
        "Internal error",
    ),
    "code_not_found": (
        "Code Not Found",
        "This registry has no entry for the requested code.",
    ),
    "not_found": ("Not Found", "No route exists for the requested path."),
    "method_not_allowed": (
        "Method Not Allowed",
        "The requested method is not allowed for this path; the Allow header lists the methods that are.",
    ),
    "facet_not_available": (
        "Facet Not Available",
        "The requested facet does not resolve to any observations.",
    ),
    "resolution_ambiguous": (
        "Resolution Ambiguous",
        "Several native facet tuples satisfy this definition, and none is a declared discriminator.",
    ),
    "facet_binding_unsupported": (
        "Facet Binding Unsupported",
        "This facet axis is recognised, but this deployment cannot honour it yet.",
    ),
    "invalid_role_ref": (
        "Invalid Role Reference",
        "A posted entity role cannot be bound: the matched StatVar declares no such role, "
        "its origin or destination alias names more than one declared role, or an entity "
        "posted under it is absent from the graph or of a kind the role does not take.",
    ),
    "role_incompatible": (
        "Role Incompatible",
        "The posted role cannot be bound on the matched StatVar(s): it is a graph property "
        "they do not declare, a flow axis or observationAbout their role shape does not "
        "take, or a role they pin to a fixed value.",
    ),
    "role_signature_incompatible": (
        "Role Signature Incompatible",
        "The matched StatVars disagree on (or this shape cannot host) the requested "
        "directional role signature.",
    ),
    "recipe_not_declared": (
        "Recipe Not Declared",
        "No declared conversion recipe satisfies the requested unit natively.",
    ),
    "rate_coverage_gap": (
        "Rate Coverage Gap",
        "The declared conversion recipe is missing rate or deflator inputs for part of this slice.",
    ),
    "capability_unavailable": (
        "Capability Unavailable",
        "This format, enrichment, or conversion class is not enabled on this deployment.",
    ),
    "upstream_version_changed": (
        "Upstream Version Changed",
        "The upstream source refreshed mid-materialisation and the retry did not converge.",
    ),
    # Served by the Datasets frame refresh and the Presentation render routes.
    "refresh_not_permitted": (
        "Refresh Not Permitted",
        "The authenticated caller is not the live-refresh principal, or live refresh is "
        "not configured on this deployment.",
    ),
}


class CodeMappingEntry(BaseModel):
    """One endpoint/channel combination a registered code is permitted on.
    Mirrors `dfo_spec_kit.registry.CodeMapping`'s fields. `doors` and
    `http_status` are null when the registry entry omits them.
    """

    endpoint: str
    channel: str
    doors: tuple[str, ...] | None
    http_status: int | None


class CodeRegistryEntry(BaseModel):
    """The wire shape of a successful `GET /codes/{code}` lookup, not the
    `ProblemDetails`/`Envelope` shape that carries `capability_version`,
    `trace_id`, and `type`. `dfo_platform.version` hashes this shape by
    calling `registry_entry_body` through the `codes` module attribute, so
    any change here moves `PLATFORM_CAPABILITY_VERSION`.

    `classification`, `definition`, `lifecycle` and `mappings` come from
    `dfo_spec_kit.registry.REGISTRY`. `title` and `detail` come from the
    merged refusal catalogue. Either source can lack the code. `title` and
    `detail` are null for a code the merged catalogue does not carry (a
    warning, or a no-data reason), while `classification`, `definition` and
    `lifecycle` are null and `mappings` is empty for a code absent from the
    registry. A client sees a missing source as null fields on a present
    code, not as a vanished key set.
    """

    code: str
    classification: Classification | None
    definition: str | None
    lifecycle: Lifecycle | None
    mappings: list[CodeMappingEntry]
    title: str | None
    detail: str | None


def registry_entry_body(
    *,
    code: str,
    registry_entry: CodeEntry | None,
    catalogue_entry: tuple[str, str] | None,
) -> CodeRegistryEntry:
    """Build a code entry from optional registry and catalogue data.

    An absent registry entry yields null registry fields and empty mappings. An absent catalogue
    entry yields null `title` and `detail`."""
    title, detail = catalogue_entry if catalogue_entry is not None else (None, None)
    if registry_entry is None:
        return CodeRegistryEntry(
            code=code,
            classification=None,
            definition=None,
            lifecycle=None,
            mappings=[],
            title=title,
            detail=detail,
        )
    mappings = [
        CodeMappingEntry(
            endpoint=mapping.endpoint,
            channel=mapping.channel,
            doors=mapping.doors,
            http_status=mapping.http_status,
        )
        for mapping in registry_entry.mappings
    ]
    return CodeRegistryEntry(
        code=code,
        classification=registry_entry.classification,
        definition=registry_entry.definition,
        lifecycle=registry_entry.lifecycle,
        mappings=mappings,
        title=title,
        detail=detail,
    )


class CodeCollision(Exception):
    """Raised when a code appears in multiple catalogues, even with identical entries.

    Put shared codes in `PLATFORM_CODE_CATALOGUE`. Surface catalogues hold
    surface-specific codes."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(
            f"code {code!r} is declared in more than one catalogue passed to "
            "merge_catalogues. A code minted by more than one API surface belongs in "
            "PLATFORM_CODE_CATALOGUE, not duplicated into a surface's own catalogue -- "
            "check that Surface.codes carries only that surface's own codes, not an "
            "already-merged view that also includes the platform's."
        )


def merge_catalogues(*catalogues: CodeCatalogue) -> dict[str, tuple[str, str]]:
    """Merge catalogues. Raise `CodeCollision` for any duplicate code."""
    merged: dict[str, tuple[str, str]] = {}
    for catalogue in catalogues:
        for code, entry in catalogue.items():
            if code in merged:
                raise CodeCollision(code)
            merged[code] = entry
    return merged


_registry_endpoint: ContextVar[str | None] = ContextVar("registry_endpoint", default=None)


@contextmanager
def bound_registry_endpoint(endpoint: str | None) -> Iterator[None]:
    """Bind the code registry endpoint, such as `datasets.mint`, whose statuses
    `registered_status` reads while the block runs. `None` binds no endpoint."""
    token = _registry_endpoint.set(endpoint)
    try:
        yield
    finally:
        _registry_endpoint.reset(token)


def bound_endpoint() -> str | None:
    """The code registry endpoint `bound_registry_endpoint` binds, or `None` where none is
    bound."""
    return _registry_endpoint.get()


@cache
def _problem_statuses(code: str) -> Mapping[str, frozenset[int]]:
    """The statuses the code registry gives `code` on each endpoint's problem details."""
    statuses: dict[str, set[int]] = {}
    for entry in REGISTRY.entries:
        if entry.code != code:
            continue
        for mapping in entry.mappings:
            if mapping.channel == "problem_details" and mapping.http_status is not None:
                statuses.setdefault(mapping.endpoint, set()).add(mapping.http_status)
    return {endpoint: frozenset(found) for endpoint, found in statuses.items()}


def registered_status(code: str) -> int:
    """The HTTP status the code registry gives `code` on the bound endpoint
    (`bound_registry_endpoint`), or, with no endpoint bound, the one status it gives `code` on
    every endpoint.

    Raises:
        LookupError: The registry maps `code` to no status or to more than one on the bound
            endpoint, or, with none bound, to no status or to more than one across endpoints.
    """
    by_endpoint = _problem_statuses(code)
    endpoint = _registry_endpoint.get()
    statuses = (
        by_endpoint.get(endpoint, frozenset())
        if endpoint is not None
        else frozenset().union(*by_endpoint.values())
    )
    if len(statuses) != 1:
        raise LookupError(
            f"the code registry gives {code!r} the statuses {sorted(statuses)}"
            f" (bound endpoint {endpoint!r})"
        )
    (status,) = statuses
    return status
