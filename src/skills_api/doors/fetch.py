"""Fetch door for `GET /api/skills/{skill_id}`.

It serves a skill's list entry and its `SKILL.md` text from the shared index. An id past
`SKILL_ID_MAX_LENGTH` returns `request_invalid` (400) before lookup, and any other id outside
the catalogue returns `skill_not_found` (404).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from dfo_platform.doors import DoorContribution
from skills_api.deps import (
    CACHE_CONTROL,
    SkillIdTooLong,
    SkillNotFound,
    SkillsDepsDep,
    SkillsIndexNotBuilt,
    classify_skill_lookup_exception,
)
from skills_api.models import SkillBody, skill_body, split_skill_warnings
from skills_api.routing import SKILL_ID_MAX_LENGTH
from skills_index import SkillsIndex


def _build_response(index: SkillsIndex, skill_id: str) -> SkillBody:
    entry = index.skills.get(skill_id)
    if entry is None:
        raise SkillNotFound(skill_id)

    _, skill_warnings = split_skill_warnings(index.issues, index.skills)
    return skill_body(skill_id, entry, skill_warnings.get(skill_id, []))


async def fetch_endpoint(skill_id: str, request: Request, deps: SkillsDepsDep) -> object:
    if len(skill_id) > SKILL_ID_MAX_LENGTH:
        raise SkillIdTooLong(len(skill_id))

    shared = deps.skills_index
    index = shared.index
    if index is None:
        raise SkillsIndexNotBuilt()

    # Only published ids mint cache entries, leaving misses out of the cache.
    return deps.fetch_cache.respond(
        generation=index,
        key=skill_id,
        request=request,
        cache_control=CACHE_CONTROL,
        build_payload=lambda: _build_response(index, skill_id).model_dump(mode="json"),
    )


def build_fetch_door() -> DoorContribution:
    router = APIRouter()
    router.add_api_route(
        "/api/skills/{skill_id:dfo_skill_id}",
        fetch_endpoint,
        methods=["GET"],
        response_model=SkillBody,
    )
    return DoorContribution(
        router=router,
        classify=classify_skill_lookup_exception,
        exception_types=(SkillIdTooLong, SkillNotFound, SkillsIndexNotBuilt),
        exception_exemplars=(
            SkillIdTooLong(SKILL_ID_MAX_LENGTH + 1),
            SkillNotFound("exemplar-skill"),
            SkillsIndexNotBuilt(),
        ),
    )
