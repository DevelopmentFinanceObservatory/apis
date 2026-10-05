"""FastAPI-free response model for the shared health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dfo_platform.build import build_commit


class HealthResponse(BaseModel):
    """`GET /healthz`'s own response body, shared across every surface."""

    status: Literal["ok"] = Field(description='always "ok" for a 200 response')
    build: str = Field(
        description="the commit sha the image was built from, the same value the "
        'x-dfo-commit response header carries, "unknown" on a checkout run with '
        "no DFO_BUILD_COMMIT set"
    )
    capability_version: str = Field(description="this surface's own capability_version token")


def health_response(*, capability_version: str) -> HealthResponse:
    """Build a health response with the current build identity."""
    return HealthResponse(status="ok", build=build_commit(), capability_version=capability_version)
