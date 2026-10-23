"""The meta door serves health and capability documents. The surface computes both
versions from its pinned `CatalogueRelease` and passes them to this door.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from dfo_platform import HealthResponse, ProblemSpec, health_response
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution
from dfo_platform.static_response import freeze_public_json
from presentation_api.models import CapabilitiesResponse


def classify_meta_exception(_exc: BaseException) -> ProblemSpec | None:
    """Defer every exception to the next classifier."""
    return None


def build_meta_door(*, capability_version: str, catalog_version: str) -> DoorContribution:
    router = APIRouter()

    @router.get("/api/presentation/healthz", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        return health_response(capability_version=capability_version)

    capabilities_document = freeze_public_json(
        {
            "capability_version": capability_version,
            "catalog_version": catalog_version,
            "build": build_commit(),
        },
        max_age=300,
    )

    @router.get("/api/presentation/capabilities", responses={200: {"model": CapabilitiesResponse}})
    async def get_capabilities(request: Request) -> Response:
        return capabilities_document.respond(request)

    return DoorContribution(router=router, classify=classify_meta_exception)
