"""Health and capability endpoints for the Graph surface."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from dfo_platform import HealthResponse, ProblemSpec, health_response
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution
from graph_api.capabilities import Capabilities, build_capabilities
from graph_api.version import GRAPH_CAPABILITY_VERSION


def classify_health_exception(_exc: BaseException) -> ProblemSpec | None:
    """This door has no exceptions to classify."""
    return None


async def healthz() -> HealthResponse:
    return health_response(capability_version=GRAPH_CAPABILITY_VERSION)


# `response_model` documents the JSONResponse body in `/openapi.json`.
async def capabilities() -> JSONResponse:
    document = build_capabilities(capability_version=GRAPH_CAPABILITY_VERSION, build=build_commit())
    return JSONResponse(document.model_dump(mode="json"))


def build_health_door() -> DoorContribution:
    router = APIRouter()
    router.add_api_route(
        "/api/graph/healthz", healthz, methods=["GET"], response_model=HealthResponse
    )
    router.add_api_route(
        "/api/graph/capabilities",
        capabilities,
        methods=["GET"],
        response_model=Capabilities,
    )
    return DoorContribution(router=router, classify=classify_health_exception)
