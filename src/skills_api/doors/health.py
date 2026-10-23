"""Health check door for `GET /api/skills/healthz`."""

from __future__ import annotations

from fastapi import APIRouter

from dfo_platform import HealthResponse, ProblemSpec, health_response
from dfo_platform.doors import DoorContribution
from skills_api.version import SKILLS_CAPABILITY_VERSION


def classify_health_exception(_exc: BaseException) -> ProblemSpec | None:
    return None


async def get_health() -> HealthResponse:
    return health_response(capability_version=SKILLS_CAPABILITY_VERSION)


def build_health_door() -> DoorContribution:
    router = APIRouter()
    router.add_api_route(
        "/api/skills/healthz", get_health, methods=["GET"], response_model=HealthResponse
    )
    return DoorContribution(router=router, classify=classify_health_exception)
