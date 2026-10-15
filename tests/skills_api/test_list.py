"""Pins list entries, ordering, warning scope, cache headers, and the unbuilt-index response."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dfo_platform.app import create_app
from dfo_platform.problem import ProblemDetails
from skills_api.app import skills_surface
from skills_index.build import build_index
from skills_index.config import SKILLS_INDEX_REFRESH_SECONDS
from skills_index.types import SkillIssue
from tests.skills_index.conftest import FakeSkillsGraph, skill_node, stored

from .conftest import (
    ODA_CONCEPTS_DESCRIPTION,
    NoGraphSharedIndex,
    client_for,
    index_of,
    skill_entry,
    surface_client,
)


def test_list_entry_carries_frontmatter_attached_and_resources():
    entry = skill_entry(
        references={"references/core-oda.md": "# Core ODA\n"},
        url="https://github.com/DevelopmentFinanceObservatory/skills/tree/main/skills/oda-concepts",
        attached=True,
        statvar_count=4695,
    )

    with client_for(index_of(entry)) as client:
        resp = client.get("/api/skills")

    assert resp.status_code == 200, resp.text
    (skill,) = resp.json()["skills"]
    assert skill == {
        "id": "oda-concepts",
        "name": "oda-concepts",
        "description": ODA_CONCEPTS_DESCRIPTION,
        "frontmatter": {
            "name": "oda-concepts",
            "description": ODA_CONCEPTS_DESCRIPTION,
            "license": "CC-BY-4.0",
        },
        "content_version": entry.front_matter.content_version,
        "url": "https://github.com/DevelopmentFinanceObservatory/skills/tree/main/skills/oda-concepts",
        "attached": True,
        "excluded": False,
        "statvar_count": 4695,
        "statvar_count_truncated": False,
        "warnings": [],
        "resources": [
            {
                "path": "SKILL.md",
                "size": entry.files[0].size,
                "digest": entry.files[0].digest,
                "media_type": "text/markdown",
            },
            {
                "path": "references/core-oda.md",
                "size": len(b"# Core ODA\n"),
                "digest": entry.files[1].digest,
                "media_type": "text/markdown",
            },
        ],
    }
    assert skill["content_version"].startswith("sha256:")
    assert "content_text" not in skill


def test_unattached_and_excluded_skills_are_listed_unattached():
    index = index_of(
        skill_entry("oda-concepts"),
        skill_entry("sector-codes", excluded=True),
    )

    with client_for(index) as client:
        skills = {s["id"]: s for s in client.get("/api/skills").json()["skills"]}

    assert skills["oda-concepts"]["attached"] is False
    assert skills["oda-concepts"]["excluded"] is False
    assert skills["sector-codes"]["attached"] is False
    assert skills["sector-codes"]["excluded"] is True


@pytest.mark.parametrize(
    ("arcs", "attached"),
    [
        pytest.param({}, False, id="no-attachment"),
        pytest.param(
            {"skillAppliesToProvenance": ("provenance/OECDCreditorReportingSystem",)},
            True,
            id="provenance-only",
        ),
    ],
)
def test_a_built_skill_lists_attached_when_its_reach_names_a_node(arcs, attached):
    graph = FakeSkillsGraph(
        subjects={("AgentSkill", "typeOf"): ["skill/oda-crs"]},
        arcs={"skill/oda-crs": skill_node("oda-crs") | arcs},
    )

    with client_for(build_index(graph=graph)) as client:
        (skill,) = client.get("/api/skills").json()["skills"]

    assert skill["attached"] is attached
    assert skill["excluded"] is False
    assert skill["warnings"] == []


@pytest.mark.parametrize(
    "front_matter",
    [
        pytest.param(f"x: {'[' * 2000}{']' * 2000}\n", id="nested-2000-deep"),
        pytest.param('summary: "\\ud800"\n', id="lone-surrogate"),
    ],
)
def test_front_matter_the_wire_cannot_carry_leaves_only_its_skill_out(front_matter):
    bad_text = f"---\nname: bad-skill\ndescription: Does things.\n{front_matter}---\nBody.\n"
    graph = FakeSkillsGraph(
        subjects={("AgentSkill", "typeOf"): ["skill/bad-skill", "skill/oda-crs"]},
        arcs={
            "skill/bad-skill": {"text": (stored(bad_text),)},
            "skill/oda-crs": skill_node("oda-crs"),
        },
    )

    with client_for(build_index(graph=graph)) as client:
        resp = client.get("/api/skills")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [s["id"] for s in body["skills"]] == ["oda-crs"]
    assert [(w["code"], w["skill_id"]) for w in body["warnings"]] == [
        ("skill_frontmatter_invalid", "skill/bad-skill")
    ]


def test_two_skills_serve_in_deterministic_id_order():
    index = index_of(skill_entry("sector-codes"), skill_entry("oda-concepts"))

    with client_for(index) as client:
        resp = client.get("/api/skills")

    assert resp.status_code == 200, resp.text
    assert [s["id"] for s in resp.json()["skills"]] == ["oda-concepts", "sector-codes"]


def test_corpus_and_per_skill_warnings_split():
    index = index_of(
        skill_entry("oda-concepts"),
        skill_entry("sector-codes"),
        issues=(
            SkillIssue(
                skill_id=None,
                code="skills_roster_truncated",
                message="the roster read stopped at its result bound",
            ),
            SkillIssue(
                skill_id="sector-codes",
                code="skill_reach_too_deep",
                message="sector-codes's reach exceeds the depth bound",
            ),
        ),
    )

    with client_for(index) as client:
        body = client.get("/api/skills").json()

    assert [(w["code"], w["skill_id"], w["severity"]) for w in body["warnings"]] == [
        ("skills_roster_truncated", None, "warn")
    ]
    by_id = {s["id"]: s for s in body["skills"]}
    assert by_id["oda-concepts"]["warnings"] == []
    assert [
        (w["code"], w["skill_id"], w["severity"]) for w in by_id["sector-codes"]["warnings"]
    ] == [("skill_reach_too_deep", "sector-codes", "degraded")]


@pytest.mark.parametrize(
    "code",
    [
        "skills_roster_entry_unaddressable",
        "skill_content_absent",
        "skill_content_too_large",
        "skill_content_undecodable",
        "skill_frontmatter_invalid",
        "skills_corpus_budget_exhausted",
    ],
)
def test_a_left_out_skill_is_a_corpus_warning_naming_its_dcid(code: str):
    index = index_of(
        skill_entry(),
        issues=(SkillIssue(skill_id="skill/left-out", code=code, message="left out"),),
    )

    with client_for(index) as client:
        body = client.get("/api/skills").json()

    assert [s["id"] for s in body["skills"]] == ["oda-concepts"]
    assert body["skills"][0]["warnings"] == []
    assert [(w["code"], w["skill_id"]) for w in body["warnings"]] == [(code, "skill/left-out")]


def test_unbuilt_index_refuses_service_unavailable():
    """`seconds_until_next_rebuild=None` on the double raises if the door
    ever calls it, so this proves the door checks `index` and refuses 503
    first, the ordering the real holder requires.
    """
    with client_for(None, seconds_until_next_rebuild=None) as client:
        resp = client.get("/api/skills")

    assert resp.status_code == 503
    assert resp.headers["content-type"] == "application/problem+json"
    problem = ProblemDetails.model_validate(resp.json())
    assert problem.code == "service_unavailable"


def test_shared_index_unresolved_before_lifespan_refuses_service_unavailable():
    """Before the lifespan runs, `SKILLS_DEPS_SLOT` is unset and its
    `unset_exception` (`SkillsIndexNotBuilt`) refuses the same 503 an unbuilt
    index does, rather than an unclassified 500.
    """
    client = TestClient(create_app([skills_surface(skills_index=None)]))

    resp = client.get("/api/skills")

    assert resp.status_code == 503
    problem = ProblemDetails.model_validate(resp.json())
    assert problem.code == "service_unavailable"


def test_cache_control_carries_the_seconds_remaining_until_next_rebuild():
    with client_for(index_of()) as client:
        resp = client.get("/api/skills")

    assert resp.headers["cache-control"] == f"public, max-age={int(SKILLS_INDEX_REFRESH_SECONDS)}"


def test_a_response_served_later_in_a_generation_carries_a_smaller_max_age():
    index = index_of()

    with client_for(index, seconds_until_next_rebuild=SKILLS_INDEX_REFRESH_SECONDS) as client:
        early = client.get("/api/skills")
    with client_for(index, seconds_until_next_rebuild=1.0) as client:
        late = client.get("/api/skills")

    assert early.headers["cache-control"] == f"public, max-age={int(SKILLS_INDEX_REFRESH_SECONDS)}"
    assert late.headers["cache-control"] == "public, max-age=1"


def test_strong_etag_and_conditional_requests():
    with client_for(index_of(skill_entry())) as client:
        first = client.get("/api/skills")
        etag = first.headers["etag"]
        assert etag.startswith('"') and etag.endswith('"')

        matching = client.get("/api/skills", headers={"if-none-match": etag})
        assert matching.status_code == 304
        assert matching.content == b""

        star = client.get("/api/skills", headers={"if-none-match": "*"})
        assert star.status_code == 304
        assert star.content == b""

        non_matching = client.get("/api/skills", headers={"if-none-match": '"does-not-match"'})
        assert non_matching.status_code == 200


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity", "gzip, br", ""])
def test_never_sets_content_encoding(accept_encoding: str):
    with client_for(index_of(skill_entry())) as client:
        resp = client.get("/api/skills", headers={"accept-encoding": accept_encoding})

    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers


def test_unchanged_corpus_yields_the_same_etag_across_a_rebuilt_index():
    """The second generation holds the same skills in reversed insertion
    order, as a rebuild from scratch may. A door that relied on insertion
    order would mint a different ETag.
    """
    first, second = skill_entry("oda-concepts"), skill_entry("sector-codes")
    holder = NoGraphSharedIndex(index_of(first, second))

    with surface_client(holder) as client:
        first_etag = client.get("/api/skills").headers["etag"]
        holder.index = index_of(second, first)
        second_etag = client.get("/api/skills").headers["etag"]

    assert first_etag == second_etag


def test_body_encoded_once_across_repeated_requests_within_a_generation(monkeypatch):
    """`GenerationBodyCache` caches the encoded body and its ETag per
    generation, so two requests against one generation encode the payload
    once.
    """
    import skills_api.doors._response_cache as response_cache

    calls: list[object] = []
    original_encode_json_body = response_cache.encode_json_body

    def counting_encode_json_body(payload: object):
        calls.append(payload)
        return original_encode_json_body(payload)

    monkeypatch.setattr(response_cache, "encode_json_body", counting_encode_json_body)

    with client_for(index_of(skill_entry())) as client:
        first = client.get("/api/skills")
        second = client.get("/api/skills")

    assert first.status_code == second.status_code == 200
    assert first.json()["skills"][0]["id"] == "oda-concepts"
    assert first.content == second.content
    assert first.headers["etag"] == second.headers["etag"]
    assert len(calls) == 1


def test_generation_change_never_serves_the_previous_generations_body():
    holder = NoGraphSharedIndex(index_of(skill_entry(description="First wording.")))

    with surface_client(holder) as client:
        first = client.get("/api/skills")
        holder.index = index_of(skill_entry(description="Second wording."))
        second = client.get("/api/skills")

    assert first.json()["skills"][0]["description"] == "First wording."
    assert second.json()["skills"][0]["description"] == "Second wording."
    assert first.headers["etag"] != second.headers["etag"]
