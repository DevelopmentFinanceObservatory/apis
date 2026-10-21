"""Pins the skill door's response, errors, cache validators, and index access."""

from __future__ import annotations

import pytest

from dfo_platform.problem import ProblemDetails
from skills_api.routing import SKILL_ID_MAX_LENGTH
from skills_index.manifest import skill_file
from skills_index.types import SkillIssue

from .conftest import (
    ODA_CONCEPTS_DESCRIPTION,
    NoGraphSharedIndex,
    client_for,
    index_of,
    skill_entry,
    skill_markdown,
    surface_client,
)

_REFERENCES = {
    "references/core-oda.md": "# Core ODA\n\nGrant equivalents.\n",
    "references/channels/multilateral.md": "# Multilateral channels\n",
}


def test_skill_door_serves_the_list_entry_plus_skill_md_verbatim():
    entry = skill_entry(
        references=_REFERENCES,
        url="https://github.com/DevelopmentFinanceObservatory/skills/tree/main/skills/oda-concepts",
        attached=True,
        statvar_count=4695,
    )

    with client_for(index_of(entry)) as client:
        resp = client.get("/api/skills/oda-concepts")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "oda-concepts"
    assert body["name"] == "oda-concepts"
    assert body["description"] == ODA_CONCEPTS_DESCRIPTION
    assert body["frontmatter"] == {
        "name": "oda-concepts",
        "description": ODA_CONCEPTS_DESCRIPTION,
        "license": "CC-BY-4.0",
    }
    assert body["content_version"] == entry.front_matter.content_version
    assert body["attached"] is True
    assert body["excluded"] is False
    assert body["statvar_count"] == 4695
    assert body["content_text"] == skill_markdown(
        "oda-concepts", description=ODA_CONCEPTS_DESCRIPTION
    )
    assert body["resources"] == [
        {
            "path": file.path,
            "size": file.size,
            "digest": file.digest,
            "media_type": "text/markdown",
        }
        for file in entry.files
    ]
    assert [r["path"] for r in body["resources"]] == [
        "SKILL.md",
        "references/channels/multilateral.md",
        "references/core-oda.md",
    ]


def test_skill_md_entry_describes_the_served_text():
    with client_for(index_of(skill_entry())) as client:
        body = client.get("/api/skills/oda-concepts").json()

    expected = skill_file("SKILL.md", body["content_text"])
    assert body["resources"][0]["size"] == expected.size
    assert body["resources"][0]["digest"] == expected.digest


def test_corpus_level_warnings_do_not_leak_onto_a_skills_own_warnings():
    index = index_of(
        skill_entry(),
        issues=(
            SkillIssue(
                skill_id=None,
                code="skills_roster_truncated",
                message="the roster read stopped at its result bound",
            ),
            SkillIssue(
                skill_id="skill/left-out",
                code="skill_frontmatter_invalid",
                message="skill/left-out has no front matter",
            ),
        ),
    )

    with client_for(index) as client:
        resp = client.get("/api/skills/oda-concepts")

    assert resp.json()["warnings"] == []


def test_skill_warnings_ride_on_the_skill():
    index = index_of(
        skill_entry(),
        issues=(
            SkillIssue(
                skill_id="oda-concepts",
                code="skill_breadth_high",
                message="reach covers 5001 StatVars",
            ),
        ),
    )

    with client_for(index) as client:
        warnings = client.get("/api/skills/oda-concepts").json()["warnings"]

    assert [(w["code"], w["skill_id"]) for w in warnings] == [
        ("skill_breadth_high", "oda-concepts")
    ]


@pytest.mark.parametrize(
    "skill_id",
    ["skill-ghost", "Oda_Concepts", "oda.concepts", "skill:oda-concepts", "s" * 65],
)
def test_any_id_outside_the_catalogue_refuses_skill_not_found(skill_id: str):
    with client_for(index_of(skill_entry())) as client:
        resp = client.get(f"/api/skills/{skill_id}")

    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    problem = ProblemDetails.model_validate(resp.json())
    assert problem.code == "skill_not_found"


def test_unbuilt_index_refuses_service_unavailable():
    with client_for(None) as client:
        resp = client.get("/api/skills/oda-concepts")

    assert resp.status_code == 503
    problem = ProblemDetails.model_validate(resp.json())
    assert problem.code == "service_unavailable"


def test_skill_id_at_the_ceiling_reaches_the_lookup():
    with client_for(index_of()) as client:
        resp = client.get(f"/api/skills/{'s' * SKILL_ID_MAX_LENGTH}")

    assert resp.status_code == 404
    assert ProblemDetails.model_validate(resp.json()).code == "skill_not_found"


def test_skill_id_over_the_ceiling_refuses_request_invalid():
    with client_for(index_of()) as client:
        resp = client.get(f"/api/skills/{'s' * (SKILL_ID_MAX_LENGTH + 1)}")

    assert resp.status_code == 400
    assert ProblemDetails.model_validate(resp.json()).code == "request_invalid"


def test_strong_etag_and_conditional_requests():
    with client_for(index_of(skill_entry())) as client:
        first = client.get("/api/skills/oda-concepts")
        etag = first.headers["etag"]
        assert etag.startswith('"') and etag.endswith('"')

        matching = client.get("/api/skills/oda-concepts", headers={"if-none-match": etag})
        assert matching.status_code == 304
        assert matching.content == b""

        star = client.get("/api/skills/oda-concepts", headers={"if-none-match": "*"})
        assert star.status_code == 304
        assert star.content == b""


def test_unchanged_index_yields_the_same_etag_across_a_rebuilt_index():
    entry = skill_entry(references=_REFERENCES)
    holder = NoGraphSharedIndex(index_of(entry))

    with surface_client(holder) as client:
        first_etag = client.get("/api/skills/oda-concepts").headers["etag"]
        holder.index = index_of(entry)
        second_etag = client.get("/api/skills/oda-concepts").headers["etag"]

    assert first_etag == second_etag


def test_body_encoded_once_across_repeated_requests_within_a_generation(monkeypatch):
    """`GenerationBodyCache` caches the encoded body and its ETag per
    `(generation, skill_id)`, so two requests against one generation encode
    the payload once.
    """
    import skills_api.doors._response_cache as response_cache

    calls: list[object] = []
    original_encode_json_body = response_cache.encode_json_body

    def counting_encode_json_body(payload: object):
        calls.append(payload)
        return original_encode_json_body(payload)

    monkeypatch.setattr(response_cache, "encode_json_body", counting_encode_json_body)

    with client_for(index_of(skill_entry())) as client:
        first = client.get("/api/skills/oda-concepts")
        second = client.get("/api/skills/oda-concepts")

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == "oda-concepts"
    assert first.content == second.content
    assert first.headers["etag"] == second.headers["etag"]
    assert len(calls) == 1


def test_generation_change_never_serves_the_previous_generations_body():
    holder = NoGraphSharedIndex(index_of(skill_entry(description="First wording.")))

    with surface_client(holder) as client:
        first = client.get("/api/skills/oda-concepts")
        holder.index = index_of(skill_entry(description="Second wording."))
        second = client.get("/api/skills/oda-concepts")

    assert first.json()["description"] == "First wording."
    assert second.json()["description"] == "Second wording."
    assert first.json()["content_version"] != second.json()["content_version"]
    assert first.headers["etag"] != second.headers["etag"]


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity", "gzip, br", ""])
def test_never_sets_content_encoding(accept_encoding: str):
    with client_for(index_of(skill_entry())) as client:
        resp = client.get("/api/skills/oda-concepts", headers={"accept-encoding": accept_encoding})

    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers
