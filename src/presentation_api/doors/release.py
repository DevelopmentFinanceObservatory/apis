"""The release door serves the pinned catalogue's schemas, member types, constraint IDs,
and retry modes. Clients resolve the envelope schema's component `$ref`s against
`component_schemas`.

The route is read-only and uses the in-memory release loaded at startup.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from assembler.catalogue import BakedReleaseRegistry, CatalogueRelease
from assembler.ports import ReleaseRegistry
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from presentation_api.codes import CODE_RETRY_MODE
from presentation_api.models import ReleaseDocumentResponse

__all__ = ["build_release_door", "classify_release_exception", "release_document"]


def classify_release_exception(_exc: BaseException) -> ProblemSpec | None:
    """Defer every exception to the next classifier."""
    return None


@dataclass
class ReleaseDeps:
    release_registry: ReleaseRegistry


def release_document(
    release: CatalogueRelease, *, retry_modes: Mapping[str, str] = CODE_RETRY_MODE
) -> dict[str, object]:
    """Build the release response from the resolved release, avoiding a second registry call."""
    return {
        "catalog_version": release.catalog_version,
        "schema_version": release.schema_version,
        "member_types": sorted(release.component_schemas),
        "constraint_ids": sorted({spec.constraint_id for spec in release.constraints}),
        "retry_modes": dict(retry_modes),
        "envelope_schema": dict(release.envelope_schema),
        "component_schemas": {
            member: dict(schema) for member, schema in release.component_schemas.items()
        },
        "intent_schemas_tolerant": dict(release.intent_schemas_tolerant),
        "minimal_intents": dict(release.minimal_intents),
    }


def build_release_door(*, release_registry: ReleaseRegistry | None = None) -> DoorContribution:
    """Load and verify a baked release in this door's lifespan when no registry is injected.
    The surface passes its already-resolved registry.
    """
    deps = ReleaseDeps(release_registry=cast(ReleaseRegistry, release_registry))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if release_registry is None:
            deps.release_registry = BakedReleaseRegistry()
        yield

    router = APIRouter()

    @router.get("/api/presentation/release", responses={200: {"model": ReleaseDocumentResponse}})
    async def get_release() -> JSONResponse:
        return JSONResponse(release_document(deps.release_registry.pinned()))

    return DoorContribution(router=router, classify=classify_release_exception, lifespan=lifespan)
