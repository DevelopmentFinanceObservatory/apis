"""Resource door for `GET /api/skills/{skill_id}/resources/{path}`.

It serves one file of a skill's manifest, `SKILL.md` included, resolving `path` by exact
match against that manifest. Unknown skills return `skill_not_found` (404). Unlisted paths
return `skill_resource_not_found` (404). Paths with `..`, a leading slash, or a backslash
match no entry. Unencoded `../` is normalised before routing. Both path parameters are checked
against their length ceilings before index lookup and return `request_invalid` (400).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from dfo_platform import ProblemSpec, refused
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from skills_api.deps import (
    CACHE_CONTROL,
    SkillIdTooLong,
    SkillNotFound,
    SkillsDepsDep,
    SkillsIndexNotBuilt,
    classify_skill_lookup_exception,
)
from skills_api.models import SkillResourceContent, media_type
from skills_api.routing import SKILL_ID_MAX_LENGTH
from skills_index import SkillsIndex
from skills_index.config import RESOURCE_PATH_MAX_LENGTH


class ResourcePathTooLong(Exception):
    """Raised before lookup when `path` exceeds its limit.

    Wire error is `request_invalid` (400).
    """

    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__(
            f"path length {length} exceeds the {RESOURCE_PATH_MAX_LENGTH}-character ceiling"
        )


class SkillResourceNotFound(Exception):
    """Known skill whose manifest lists no such path.

    Wire error is `skill_resource_not_found` (404).
    """

    def __init__(self, skill_id: str, path: str) -> None:
        self.skill_id = skill_id
        self.path = path
        super().__init__(f"skill {skill_id!r} publishes no resource at path {path!r}")


def classify_resource_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, ResourcePathTooLong):
        return refused(
            "request_invalid",
            detail=(
                f"path exceeds the {RESOURCE_PATH_MAX_LENGTH}-character ceiling "
                f"(received {exc.length} characters)"
            ),
            pointer="/path",
            retryable=False,
        )
    if isinstance(exc, SkillResourceNotFound):
        return refused(
            "skill_resource_not_found",
            detail=str(exc),
            pointer="/path",
            retryable=False,
        )
    return classify_skill_lookup_exception(exc)


def _build_response(index: SkillsIndex, skill_id: str, path: str) -> SkillResourceContent:
    entry = index.skills.get(skill_id)
    if entry is None:
        raise SkillNotFound(skill_id)

    file = next((file for file in entry.files if file.path == path), None)
    if file is None:
        raise SkillResourceNotFound(skill_id, path)
    return SkillResourceContent(
        skill_id=skill_id,
        path=path,
        size=file.size,
        digest=file.digest,
        media_type=media_type(path),
        content_text=file.content,
    )


async def get_resource(
    skill_id: str,
    path: str,
    request: Request,
    deps: SkillsDepsDep,
) -> object:
    if len(skill_id) > SKILL_ID_MAX_LENGTH:
        raise SkillIdTooLong(len(skill_id))
    if len(path) > RESOURCE_PATH_MAX_LENGTH:
        raise ResourcePathTooLong(len(path))

    shared = deps.skills_index
    index = shared.index
    if index is None:
        raise SkillsIndexNotBuilt()

    return deps.resource_cache.respond(
        generation=index,
        key=(skill_id, path),
        request=request,
        cache_control=CACHE_CONTROL,
        build_payload=lambda: _build_response(index, skill_id, path).model_dump(mode="json"),
    )


def build_resource_door() -> DoorContribution:
    router = APIRouter()
    router.add_api_route(
        "/api/skills/{skill_id:dfo_skill_id}/resources/{path:path}",
        get_resource,
        methods=["GET"],
        response_model=SkillResourceContent,
    )
    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "skills.resource"),
        classify=classify_resource_exception,
        exception_types=(
            SkillIdTooLong,
            ResourcePathTooLong,
            SkillNotFound,
            SkillResourceNotFound,
            SkillsIndexNotBuilt,
        ),
        endpoint_exemplars={
            "skills.resource": (
                SkillIdTooLong(SKILL_ID_MAX_LENGTH + 1),
                ResourcePathTooLong(RESOURCE_PATH_MAX_LENGTH + 1),
                SkillNotFound("exemplar-skill"),
                SkillResourceNotFound("exemplar-skill", "references/exemplar.md"),
                SkillsIndexNotBuilt(),
            )
        },
    )
