"""What the code registry maps, read the way the refusal sweeps compare a door against it."""

from __future__ import annotations

from dfo_spec_kit.registry import REGISTRY, CodeMapping

from dfo_platform.app import Surface
from dfo_platform.doors import RouteKey, route_keys


def _problem_mappings(code: str) -> tuple[CodeMapping, ...]:
    entry = REGISTRY[code] if code in REGISTRY else None  # noqa: SIM401 -- CodeRegistry has no .get()
    mappings = entry.mappings if entry is not None else ()
    return tuple(
        mapping
        for mapping in mappings
        if mapping.channel == "problem_details" and mapping.http_status is not None
    )


def problem_statuses(code: str, endpoint: str) -> set[int]:
    """The statuses the registry gives `code`'s problem details on `endpoint`."""
    return {
        status
        for mapping in _problem_mappings(code)
        if mapping.endpoint == endpoint and (status := mapping.http_status) is not None
    }


def registered_status_codes(endpoint: str) -> frozenset[tuple[str, int]]:
    """The active `problem_details` code-status pairs on `endpoint`. Manifest warnings use a
    separate channel without HTTP statuses."""
    return frozenset(
        (entry.code, status)
        for entry in REGISTRY.entries
        if entry.lifecycle == "active"
        for status in problem_statuses(entry.code, endpoint)
    )


def registered_endpoints(prefix: str) -> set[str]:
    """Every registry endpoint whose name starts with `prefix`."""
    return {
        mapping.endpoint
        for entry in REGISTRY.entries
        for mapping in entry.mappings
        if mapping.endpoint.startswith(prefix)
    }


def unbound_routes(surface: Surface) -> frozenset[RouteKey]:
    """The surface's routes that name no registry endpoint, one key per method."""
    return route_keys(surface.router) - set(surface.registry_endpoints)
