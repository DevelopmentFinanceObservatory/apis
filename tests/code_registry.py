"""What the code registry maps, read the way the refusal sweeps compare a door against it."""

from __future__ import annotations

from dfo_spec_kit.registry import load_registry

from dfo_platform.app import Surface
from dfo_platform.doors import RouteKey, route_keys


def _registered_pairs() -> frozenset[tuple[str, str, int]]:
    """Every active code's problem details as (endpoint, code, status) triples."""
    return frozenset(
        (mapping.endpoint, entry.code, mapping.http_status)
        for entry in load_registry().entries
        if entry.lifecycle == "active"
        for mapping in entry.mappings
        if mapping.channel == "problem_details" and mapping.http_status is not None
    )


def problem_statuses(code: str, endpoint: str) -> set[int]:
    """The statuses the registry gives active `code`'s problem details on `endpoint`."""
    return {
        status
        for mapped, mapped_code, status in _registered_pairs()
        if mapped == endpoint and mapped_code == code
    }


def registered_status_codes(endpoint: str) -> frozenset[tuple[str, int]]:
    """The active `problem_details` code-status pairs on `endpoint`. Manifest warnings use a
    separate channel without HTTP statuses."""
    return frozenset(
        (code, status) for mapped, code, status in _registered_pairs() if mapped == endpoint
    )


def registered_endpoints(prefix: str) -> set[str]:
    """Every registry endpoint whose name starts with `prefix` and maps a problem status."""
    return {endpoint for endpoint, _, _ in _registered_pairs() if endpoint.startswith(prefix)}


def unbound_routes(surface: Surface) -> frozenset[RouteKey]:
    """The surface's routes that name no registry endpoint, one key per method."""
    return route_keys(surface.router) - set(surface.registry_endpoints)
