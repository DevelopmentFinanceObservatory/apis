"""Composition root for the Skills API surface.

The surface combines health, list, fetch, and resource doors. Its lifespan binds one shared
skills index and per-door response caches in a fresh `SkillsDeps` for each entry. Door
classifiers run in registration order, and the platform envelope renders RFC 9457 problem
responses. Callers reach the API through the website's proxy, which answers `429` once an IP
passes its allowance, 120 calls a minute on `/api/*` by default, counted in a fixed window that
every proxy instance shares in Postgres. A call whose count fails goes through. The proxy passes
`Authorization` upstream unread.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from dfo_platform.app import Surface
from dfo_platform.doors import (
    DoorContribution,
    check_endpoint_exemplars,
    combine_classifiers,
    endpoint_exemplars,
    flatten,
)
from skills_api.codes import SKILLS_CODE_CATALOGUE
from skills_api.deps import SKILLS_DEPS_SLOT, SkillsDeps
from skills_api.doors._response_cache import GenerationBodyCache
from skills_api.doors.fetch import build_fetch_door
from skills_api.doors.health import build_health_door
from skills_api.doors.list import build_list_door
from skills_api.doors.resource import build_resource_door
from skills_api.version import SKILLS_CAPABILITY_VERSION
from skills_index import SharedSkillsIndex, acquire_shared_index, release_shared_index


def skills_surface(
    *,
    skills_index: SharedSkillsIndex | None = None,
) -> Surface:
    """Build the Skills surface, injecting an index when supplied.

    Without an injected index, the lifespan acquires and releases the process-wide index.
    The lifespan owns only indexes it acquires. Production mounts Search first, so this
    surface acquires the already-bound index with `graph=None`.
    """
    doors: tuple[DoorContribution, ...] = (
        build_health_door(),
        build_list_door(),
        build_fetch_door(),
        build_resource_door(),
    )

    router = flatten(doors)
    classify = combine_classifiers(*(door.classify for door in doors))
    check_endpoint_exemplars(doors, classify)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        shared_index = skills_index
        if shared_index is None:
            shared_index = await acquire_shared_index(graph=None)

        SKILLS_DEPS_SLOT.set(
            _app,
            SkillsDeps(
                skills_index=shared_index,
                list_cache=GenerationBodyCache(),
                fetch_cache=GenerationBodyCache(),
                resource_cache=GenerationBodyCache(),
            ),
        )
        try:
            yield
        finally:
            SKILLS_DEPS_SLOT.clear(_app)
            # Search enters first, so this exits first and releases its reference before
            # Search closes the graph client used by the index refresh task.
            if skills_index is None:
                await release_shared_index(shared_index)

    return Surface(
        name="skills",
        description="List and fetch agent skills and their resources.",
        prefix="/api/skills",
        router=router,
        capability_version=SKILLS_CAPABILITY_VERSION,
        codes=SKILLS_CODE_CATALOGUE,
        classify=classify,
        exception_types=tuple(exc_type for door in doors for exc_type in door.exception_types),
        exception_exemplars=endpoint_exemplars(doors),
        lifespan=lifespan,
        registry_endpoints={
            key: endpoint for door in doors for key, endpoint in door.registry_endpoints.items()
        },
    )
