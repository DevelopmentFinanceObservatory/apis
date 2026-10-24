"""Pins the resources door: manifest lookup, SKILL.md, traversal refusals, length limits,
and cache validators."""

from __future__ import annotations

import pytest

from dfo_platform.problem import ProblemDetails
from skills_api.routing import SKILL_ID_MAX_LENGTH
from skills_index.config import RESOURCE_PATH_MAX_LENGTH
from skills_index.manifest import skill_file

from .conftest import NoGraphSharedIndex, client_for, index_of, skill_entry, surface_client

_CORE_ODA = "# Core ODA\n\nGrant equivalents, not flows. Ünïcode counts in bytes.\n"
_URL = "/api/skills/oda-concepts/resources/references/core-oda.md"


def _index():
    return index_of(skill_entry(references={"references/core-oda.md": _CORE_ODA}))


def test_reference_is_served_with_size_digest_and_media_type():
    with client_for(_index()) as client:
        resp = client.get(_URL)

    assert resp.status_code == 200, resp.text
    expected = skill_file("references/core-oda.md", _CORE_ODA)
    assert resp.json() == {
        "skill_id": "oda-concepts",
        "path": "references/core-oda.md",
        "size": expected.size,
        "digest": expected.digest,
        "media_type": "text/markdown",
        "content_text": _CORE_ODA,
    }
    assert expected.size == len(_CORE_ODA.encode("utf-8"))


def test_skill_md_is_served_with_the_skill_doors_text():
    with client_for(_index()) as client:
        skill = client.get("/api/skills/oda-concepts").json()
        resp = client.get("/api/skills/oda-concepts/resources/SKILL.md")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content_text"] == skill["content_text"]
    assert {k: body[k] for k in ("path", "size", "digest", "media_type")} == skill["resources"][0]


def test_every_listed_file_is_fetchable_with_its_listed_figures():
    index = index_of(
        skill_entry(
            references={
                "references/core-oda.md": _CORE_ODA,
                "references/channels/multilateral.md": "# Multilateral\n",
            }
        )
    )

    with client_for(index) as client:
        listed = client.get("/api/skills").json()["skills"][0]["resources"]
        served = [
            client.get(f"/api/skills/oda-concepts/resources/{file['path']}").json()
            for file in listed
        ]

    assert [{k: s[k] for k in ("path", "size", "digest", "media_type")} for s in served] == listed


def test_unlisted_path_refuses_skill_resource_not_found():
    with client_for(_index()) as client:
        resp = client.get("/api/skills/oda-concepts/resources/references/does-not-exist.md")

    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    assert ProblemDetails.model_validate(resp.json()).code == "skill_resource_not_found"


def test_unknown_skill_refuses_skill_not_found_rather_than_skill_resource_not_found():
    with client_for(index_of()) as client:
        resp = client.get("/api/skills/skill-ghost/resources/references/core-oda.md")

    assert resp.status_code == 404
    assert ProblemDetails.model_validate(resp.json()).code == "skill_not_found"


@pytest.mark.parametrize(
    "hostile_path",
    [
        # httpx removes an unencoded "../" client-side (RFC 3986 section
        # 5.2.4), so these percent-encode it to reach the handler as a
        # literal ".." in `path`.
        pytest.param("%2e%2e/references/core-oda.md", id="encoded-dot-dot-slash"),
        pytest.param("references%2f..%2f..%2fetc%2fpasswd", id="encoded-traversal"),
        pytest.param("references\\core-oda.md", id="backslash"),
        pytest.param("%2freferences/core-oda.md", id="encoded-leading-slash"),
        pytest.param("skill.md", id="case-folded-skill-md"),
    ],
)
def test_hostile_path_refuses_skill_resource_not_found(hostile_path: str):
    """The door matches `path` exactly against the manifest and never builds
    a dcid from it, so a hostile path is one more unlisted path."""
    with client_for(_index()) as client:
        resp = client.get(f"/api/skills/oda-concepts/resources/{hostile_path}")

    assert resp.status_code == 404
    assert ProblemDetails.model_validate(resp.json()).code == "skill_resource_not_found"


def test_unencoded_dot_dot_never_reaches_this_door():
    """httpx removes a raw "../" before sending, collapsing the `resources`
    segment, so the request matches no route and refuses through the
    platform's routing `not_found`."""
    with client_for(_index()) as client:
        resp = client.get("/api/skills/oda-concepts/resources/../references/core-oda.md")

    assert resp.status_code == 404
    assert ProblemDetails.model_validate(resp.json()).code == "not_found"


def test_resources_first_segment_matches_no_special_rule():
    """Everything after the literal `resources/` segment is the path, so a
    path whose own first segment is `resources` needs no special handling."""
    index = index_of(skill_entry(references={"resources/nested.md": "nested"}))

    with client_for(index) as client:
        resp = client.get("/api/skills/oda-concepts/resources/resources/nested.md")

    assert resp.status_code == 200, resp.text
    assert resp.json()["path"] == "resources/nested.md"


def test_unbuilt_index_refuses_service_unavailable():
    with client_for(None) as client:
        resp = client.get(_URL)

    assert resp.status_code == 503
    assert ProblemDetails.model_validate(resp.json()).code == "service_unavailable"


def test_skill_id_at_the_ceiling_reaches_the_lookup():
    with client_for(index_of()) as client:
        resp = client.get(f"/api/skills/{'s' * SKILL_ID_MAX_LENGTH}/resources/SKILL.md")

    assert resp.status_code == 404
    assert ProblemDetails.model_validate(resp.json()).code == "skill_not_found"


def test_skill_id_over_the_ceiling_refuses_request_invalid():
    with client_for(index_of()) as client:
        resp = client.get(f"/api/skills/{'s' * (SKILL_ID_MAX_LENGTH + 1)}/resources/SKILL.md")

    assert resp.status_code == 400
    assert ProblemDetails.model_validate(resp.json()).code == "request_invalid"


def test_path_at_the_ceiling_reaches_the_lookup():
    with client_for(_index()) as client:
        resp = client.get(f"/api/skills/oda-concepts/resources/{'a' * RESOURCE_PATH_MAX_LENGTH}")

    assert resp.status_code == 404
    assert ProblemDetails.model_validate(resp.json()).code == "skill_resource_not_found"


def test_path_over_the_ceiling_refuses_request_invalid():
    path = "a" * (RESOURCE_PATH_MAX_LENGTH + 1)
    with client_for(_index()) as client:
        resp = client.get(f"/api/skills/oda-concepts/resources/{path}")

    assert resp.status_code == 400
    assert ProblemDetails.model_validate(resp.json()).code == "request_invalid"


def test_strong_etag_and_conditional_requests():
    with client_for(_index()) as client:
        first = client.get(_URL)
        etag = first.headers["etag"]
        assert etag.startswith('"') and etag.endswith('"')

        matching = client.get(_URL, headers={"if-none-match": etag})
        assert matching.status_code == 304
        assert matching.content == b""

        star = client.get(_URL, headers={"if-none-match": "*"})
        assert star.status_code == 304
        assert star.content == b""


def test_unchanged_index_yields_the_same_etag_across_a_rebuilt_index():
    holder = NoGraphSharedIndex(_index())

    with surface_client(holder) as client:
        first_etag = client.get(_URL).headers["etag"]
        holder.index = _index()
        second_etag = client.get(_URL).headers["etag"]

    assert first_etag == second_etag


def test_body_encoded_once_across_repeated_requests_within_a_generation(monkeypatch):
    """`GenerationBodyCache` caches the encoded body and its ETag per
    `(generation, skill_id, path)`, so two requests against one generation
    encode the payload once.
    """
    import skills_api.doors._response_cache as response_cache

    calls: list[object] = []
    original_encode_json_body = response_cache.encode_json_body

    def counting_encode_json_body(payload: object):
        calls.append(payload)
        return original_encode_json_body(payload)

    monkeypatch.setattr(response_cache, "encode_json_body", counting_encode_json_body)

    with client_for(_index()) as client:
        first = client.get(_URL)
        second = client.get(_URL)

    assert first.status_code == second.status_code == 200
    assert first.json()["content_text"] == _CORE_ODA
    assert first.content == second.content
    assert first.headers["etag"] == second.headers["etag"]
    assert len(calls) == 1


def test_generation_change_never_serves_the_previous_generations_body():
    holder = NoGraphSharedIndex(
        index_of(skill_entry(references={"references/core-oda.md": "# Health\n"}))
    )

    with surface_client(holder) as client:
        first = client.get(_URL)
        holder.index = index_of(skill_entry(references={"references/core-oda.md": "# Education\n"}))
        second = client.get(_URL)

    assert first.json()["content_text"] == "# Health\n"
    assert second.json()["content_text"] == "# Education\n"
    assert first.json()["digest"] != second.json()["digest"]
    assert first.headers["etag"] != second.headers["etag"]


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity", "gzip, br", ""])
def test_never_sets_content_encoding(accept_encoding: str):
    with client_for(_index()) as client:
        resp = client.get(_URL, headers={"accept-encoding": accept_encoding})

    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers
