"""The surface's route paths and the code registry endpoint each route serves.

`ROUTE_ENDPOINTS` is the one record of which routes serve which endpoint. A door reads its
`registry_endpoints` from it (`door_registry_endpoints`), so a route with no row fails when the
door is built, and `compute_presentation_capability_version` hashes it, so a rebinding moves the
token. The release and meta routes serve no endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from fastapi import APIRouter

from dfo_platform.doors import RouteKey, route_keys

__all__ = [
    "BARE_FRAME_PATH",
    "COMPONENTS_PATH",
    "COMPONENT_DATA_ENDPOINT",
    "COMPONENT_DATA_PATH",
    "COMPONENT_PATH",
    "COMPONENT_READ_ENDPOINT",
    "FRAMES_ENDPOINT",
    "FRAME_PATH",
    "RENDER_DATA_PATH",
    "RENDER_ENDPOINT",
    "RENDER_PATH",
    "ROUTE_ENDPOINTS",
    "SHARES_ENDPOINT",
    "SHARES_PATH",
    "SHARE_HISTORY_PATH",
    "SHARE_IMPORT_PATH",
    "SHARE_PATH",
    "SHARE_SLOT_DATA_PATH",
    "SHARE_SLOT_PATH",
    "SHARE_SNAPSHOT_PATH",
    "WRITE_ENDPOINT",
    "door_registry_endpoints",
]

WRITE_ENDPOINT = "presentation.write"
COMPONENT_READ_ENDPOINT = "presentation.component_read"
COMPONENT_DATA_ENDPOINT = "presentation.component_data"
RENDER_ENDPOINT = "presentation.render"
FRAMES_ENDPOINT = "presentation.frames"
SHARES_ENDPOINT = "presentation.shares"

COMPONENTS_PATH = "/api/presentation/components"
COMPONENT_PATH = f"{COMPONENTS_PATH}/{{component_id}}"
COMPONENT_DATA_PATH = f"{COMPONENT_PATH}/revisions/{{revision}}/data"
RENDER_PATH = f"{COMPONENT_PATH}/revisions/{{revision}}/render"
RENDER_DATA_PATH = f"{RENDER_PATH}/data"
BARE_FRAME_PATH = "/api/presentation/frames/{spec_id}"
FRAME_PATH = f"{BARE_FRAME_PATH}/{{content_version}}"
SHARES_PATH = "/api/presentation/shares"
SHARE_PATH = f"{SHARES_PATH}/{{share_id}}"
SHARE_SNAPSHOT_PATH = f"{SHARE_PATH}/snapshot"
SHARE_IMPORT_PATH = f"{SHARE_PATH}/import"
SHARE_HISTORY_PATH = f"{SHARE_PATH}/history"
SHARE_SLOT_PATH = f"{SHARE_PATH}/slots/{{slot}}"
SHARE_SLOT_DATA_PATH = f"{SHARE_SLOT_PATH}/data"

ROUTE_ENDPOINTS: Mapping[RouteKey, str] = MappingProxyType(
    {
        ("POST", COMPONENTS_PATH): WRITE_ENDPOINT,
        ("PATCH", COMPONENT_PATH): WRITE_ENDPOINT,
        ("GET", COMPONENT_PATH): COMPONENT_READ_ENDPOINT,
        ("GET", COMPONENT_DATA_PATH): COMPONENT_DATA_ENDPOINT,
        ("POST", RENDER_PATH): RENDER_ENDPOINT,
        ("GET", RENDER_DATA_PATH): RENDER_ENDPOINT,
        ("GET", BARE_FRAME_PATH): FRAMES_ENDPOINT,
        ("GET", FRAME_PATH): FRAMES_ENDPOINT,
        ("POST", SHARES_PATH): SHARES_ENDPOINT,
        ("GET", SHARE_PATH): SHARES_ENDPOINT,
        ("PUT", SHARE_PATH): SHARES_ENDPOINT,
        ("DELETE", SHARE_PATH): SHARES_ENDPOINT,
        ("POST", SHARE_SNAPSHOT_PATH): SHARES_ENDPOINT,
        ("PUT", SHARE_IMPORT_PATH): SHARES_ENDPOINT,
        ("GET", SHARE_HISTORY_PATH): SHARES_ENDPOINT,
        ("GET", SHARE_SLOT_PATH): SHARES_ENDPOINT,
        ("GET", SHARE_SLOT_DATA_PATH): SHARES_ENDPOINT,
    }
)


def door_registry_endpoints(router: APIRouter) -> dict[RouteKey, str]:
    """Each route on `router` with its row of `ROUTE_ENDPOINTS`.

    Raises:
        KeyError: A route has no row.
    """
    return {key: ROUTE_ENDPOINTS[key] for key in route_keys(router)}
