"""List door for `GET /api/skills`.

It serves skills and warnings from the shared index, ordered by skill id so unchanged
content keeps the same ETag. A warning naming no served skill, such as one for a skill left
out of the catalogue, joins the corpus warnings. `Cache-Control`
uses the seconds until the next refresh attempt, so a cached response expires by then.
An index awaiting its first build returns `service_unavailable` (503), avoiding a cached
empty catalogue.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from skills_api.deps import (
    CACHE_CONTROL,
    INDEX_NOT_BUILT_PROBLEM,
    SkillsDepsDep,
    SkillsIndexNotBuilt,
)
from skills_api.models import SkillListResponse, skill_summary, split_skill_warnings
from skills_index import SkillsIndex


def classify_list_exception(exc: BaseException) -> ProblemSpec | None:
    return INDEX_NOT_BUILT_PROBLEM if isinstance(exc, SkillsIndexNotBuilt) else None


def _build_response(index: SkillsIndex) -> SkillListResponse:
    corpus_warnings, skill_warnings = split_skill_warnings(index.issues, index.skills)

    skills = [
        skill_summary(skill_id, index.skills[skill_id], skill_warnings.get(skill_id, []))
        for skill_id in sorted(index.skills)
    ]

    return SkillListResponse(skills=skills, warnings=corpus_warnings)


async def list_endpoint(request: Request, deps: SkillsDepsDep) -> object:
    shared = deps.skills_index
    index = shared.index
    if index is None:
        raise SkillsIndexNotBuilt()
    return deps.list_cache.respond(
        generation=index,
        key=None,
        request=request,
        cache_control=CACHE_CONTROL,
        build_payload=lambda: _build_response(index).model_dump(mode="json"),
    )


def build_list_door() -> DoorContribution:
    router = APIRouter()
    router.add_api_route(
        "/api/skills", list_endpoint, methods=["GET"], response_model=SkillListResponse
    )
    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "skills.list"),
        classify=classify_list_exception,
        exception_types=(SkillsIndexNotBuilt,),
        exception_exemplars=(SkillsIndexNotBuilt(),),
    )
