"""What the code registry maps, read the way the refusal sweeps compare a door against it."""

from __future__ import annotations

from dfo_platform.app import Surface
from dfo_platform.codes import endpoint_statuses
from dfo_platform.doors import RouteKey, route_keys


def problem_statuses(code: str, endpoint: str) -> set[int]:
    """The statuses the registry gives active `code`'s problem details on `endpoint`."""
    return set(endpoint_statuses("").get(endpoint, {}).get(code, ()))


def registered_status_codes(endpoint: str) -> frozenset[tuple[str, int]]:
    """The active `problem_details` code-status pairs on `endpoint`. Manifest warnings use a
    separate channel without HTTP statuses."""
    return frozenset(
        (code, status)
        for code, statuses in endpoint_statuses("").get(endpoint, {}).items()
        for status in statuses
    )


def registered_endpoints(prefix: str) -> set[str]:
    """Every registry endpoint whose name starts with `prefix` and maps a problem status."""
    return set(endpoint_statuses(prefix))


def unbound_routes(surface: Surface) -> frozenset[RouteKey]:
    """The surface's routes that name no registry endpoint, one key per method."""
    return route_keys(surface.router) - set(surface.registry_endpoints)
