"""Pins the skill-id convertor and routing to the health and skill doors."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from dfo_platform import HealthResponse
from dfo_platform.app import Surface, create_app
from dfo_platform.doors import DoorContribution, combine_classifiers, flatten
from skills_api.codes import SKILLS_CODE_CATALOGUE
from skills_api.deps import SKILLS_DEPS_SLOT, SkillsDeps
from skills_api.doors._response_cache import GenerationBodyCache
from skills_api.doors.fetch import build_fetch_door
from skills_api.doors.health import build_health_door
from skills_api.routing import SKILL_ID_MAX_LENGTH, SkillIdConvertor
from skills_api.version import SKILLS_CAPABILITY_VERSION
from skills_index.config import RESERVED_SKILL_IDS

from .conftest import NoGraphSharedIndex, index_of, skill_entry


def _deps(holder: object) -> SkillsDeps:
    """A `SkillsDeps` around `holder`, the shape `skills_surface`'s lifespan builds."""
    return SkillsDeps(
        skills_index=holder,
        list_cache=GenerationBodyCache(),
        fetch_cache=GenerationBodyCache(),
        resource_cache=GenerationBodyCache(),
    )


def _surface_from_doors(doors: tuple[DoorContribution, ...]) -> Surface:
    """A minimal `Surface` from `doors` in the order given, so a test can mount
    the skill door ahead of the health door."""
    return Surface(
        name="skills",
        prefix="/api/skills",
        router=flatten(doors),
        capability_version=SKILLS_CAPABILITY_VERSION,
        codes=SKILLS_CODE_CATALOGUE,
        classify=combine_classifiers(*(door.classify for door in doors)),
        exception_types=tuple(t for door in doors for t in door.exception_types),
        exception_exemplars=tuple(e for door in doors for e in door.exception_exemplars),
    )


def _client(doors: tuple[DoorContribution, ...]) -> TestClient:
    app = create_app([_surface_from_doors(doors)])
    SKILLS_DEPS_SLOT.set(app, _deps(NoGraphSharedIndex(index_of(skill_entry()))))
    return TestClient(app)


@pytest.mark.parametrize("skill_id", sorted(RESERVED_SKILL_IDS))
def test_convertor_refuses_reserved_names(skill_id: str) -> None:
    assert re.fullmatch(SkillIdConvertor.regex, skill_id) is None


@pytest.mark.parametrize(
    "skill_id",
    ["oda-concepts", "Oda_Concepts", "oda.concepts", "skill:oda", "healthz2", "s" * 600],
)
def test_convertor_admits_any_other_single_segment(skill_id: str) -> None:
    assert re.fullmatch(SkillIdConvertor.regex, skill_id)


@pytest.mark.parametrize("skill_id", ["", "skill/oda-concepts"])
def test_convertor_refuses_an_empty_or_slash_bearing_segment(skill_id: str) -> None:
    assert re.fullmatch(SkillIdConvertor.regex, skill_id) is None


def test_reserved_name_reaches_the_health_door_regardless_of_mount_order() -> None:
    with _client((build_fetch_door(), build_health_door())) as client:
        resp = client.get("/api/skills/healthz")

    assert resp.status_code == 200, resp.text
    body = HealthResponse.model_validate(resp.json())
    assert body.capability_version == SKILLS_CAPABILITY_VERSION


def test_catalogue_id_routes_to_the_skill_door() -> None:
    with _client((build_health_door(), build_fetch_door())) as client:
        resp = client.get("/api/skills/oda-concepts")

    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == "oda-concepts"


def test_oversized_id_reaches_the_handlers_ceiling() -> None:
    """The convertor admits an id of any length, so an oversized one refuses
    400 from the handler, not a routing 404."""
    with _client((build_health_door(), build_fetch_door())) as client:
        resp = client.get(f"/api/skills/{'s' * (SKILL_ID_MAX_LENGTH + 1)}")

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"
