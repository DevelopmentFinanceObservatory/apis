"""Health and capability endpoints for the Graph surface.

The capability document answers `public, max-age=300` with a strong `ETag` over its
body and honours `If-None-Match` with 304. It carries the data builds this process
serves, so its body and `ETag` move with them while `capability_version` stays put.
Before the surface lifespan publishes its dependencies, or while the graph generation is
unknown, the document serves those builds null and answers `no-store` (`gapped_public_json`).
Both gaps clear on a later read.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from code_index.shared import get_index
from dfo_platform import HealthResponse, ProblemSpec, health_response
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution
from dfo_platform.static_response import gapped_public_json
from graph_api.capabilities import Capabilities, DataBuilds, build_capabilities
from graph_api.deps import GRAPH_DEPS_SLOT, GraphDepsNotSet
from graph_api.version import GRAPH_CAPABILITY_VERSION
from origin_catalogue import current_catalogue


def classify_health_exception(_exc: BaseException) -> ProblemSpec | None:
    """This door has no exceptions to classify."""
    return None


async def healthz() -> HealthResponse:
    return health_response(capability_version=GRAPH_CAPABILITY_VERSION)


def _data_builds(request: Request) -> tuple[DataBuilds, bool]:
    """Read each build without I/O from the request's catalogue and the surface's deps, and
    say whether a null among them is transient: the deps are unpublished or the graph
    generation is unknown."""
    try:
        deps = GRAPH_DEPS_SLOT(request)
    except GraphDepsNotSet:
        deps = None
    builds = DataBuilds(
        graph_generation=deps.generation.stamp().token if deps is not None else None,
        origin_catalogue=current_catalogue().build_id,
        coverage_index=deps.coverage.manifest.build_id if deps is not None else None,
        code_index=get_index().stamp.content_hash,
    )
    transient = deps is None or builds.graph_generation is None
    return builds, transient


# `response_model` documents the response body in `/openapi.json`.
async def capabilities(request: Request) -> Response:
    data_builds, transient_gap = _data_builds(request)
    document = build_capabilities(
        capability_version=GRAPH_CAPABILITY_VERSION,
        build=build_commit(),
        data_builds=data_builds,
    )
    return gapped_public_json(
        document.model_dump(mode="json"), request, transient_gap=transient_gap
    )


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
