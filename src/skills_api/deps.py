"""Dependencies, exceptions, and refusals shared by skills API routes.

The lifespan builds one skills index and a fresh body cache for each door."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends

from dfo_platform import ProblemSpec
from dfo_platform.state import StateSlot
from skills_api.doors._response_cache import GenerationBodyCache
from skills_api.routing import SKILL_ID_MAX_LENGTH
from skills_index import SharedSkillsIndex


class SkillsIndexNotBuilt(Exception):
    """Raised when the shared index is unbuilt or the dependency bundle is unset.

    Doors map this exception to retryable `service_unavailable` (503)."""


class SkillIdTooLong(Exception):
    """Raised before lookup when `skill_id` exceeds its limit, carrying only its length.

    Wire error is `request_invalid` (400).
    """

    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__(
            f"skill_id length {length} exceeds the {SKILL_ID_MAX_LENGTH}-character ceiling"
        )


class SkillNotFound(Exception):
    """Unknown skill id. Wire error is `skill_not_found` (404)."""

    def __init__(self, skill_id: str) -> None:
        self.skill_id = skill_id
        super().__init__(f"skill {skill_id!r} is not published on this deployment")


INDEX_NOT_BUILT_PROBLEM = ProblemSpec(
    status=503,
    code="service_unavailable",
    detail="the skills index has not finished its first build yet; retry",
    retryable=True,
)


def classify_skill_lookup_exception(exc: BaseException) -> ProblemSpec | None:
    """Maps the refusals of a door that looks up one skill by id."""
    if isinstance(exc, SkillIdTooLong):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=(
                f"skill_id exceeds the {SKILL_ID_MAX_LENGTH}-character ceiling "
                f"(received {exc.length} characters)"
            ),
            pointer="/skill_id",
            retryable=False,
        )
    if isinstance(exc, SkillNotFound):
        return ProblemSpec(
            status=404,
            code="skill_not_found",
            detail=str(exc),
            pointer="/skill_id",
            retryable=False,
        )
    if isinstance(exc, SkillsIndexNotBuilt):
        return INDEX_NOT_BUILT_PROBLEM
    return None


def cache_control(shared: SharedSkillsIndex) -> str:
    """Returns a public `Cache-Control` that expires at the next refresh attempt."""
    return f"public, max-age={int(shared.seconds_until_next_rebuild())}"


@dataclass(frozen=True, slots=True)
class SkillsDeps:
    skills_index: SharedSkillsIndex
    list_cache: GenerationBodyCache
    fetch_cache: GenerationBodyCache
    resource_cache: GenerationBodyCache


# Separate app-state key; an unset slot maps to the same 503 as an unbuilt index.
SKILLS_DEPS_SLOT: StateSlot[SkillsDeps] = StateSlot(
    "skills_deps", unset_exception=SkillsIndexNotBuilt
)

# `Annotated` keeps the dependency out of argument defaults, avoiding Ruff B008.
SkillsDepsDep = Annotated[SkillsDeps, Depends(SKILLS_DEPS_SLOT)]
