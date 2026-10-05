"""The components door reads `x-dfo-write-token` on revise and on a create into an existing
composition, answers a refused write with 403 `write_refused`, and serves reads to any caller.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assembler.fakes import FakeRevisionStore
from dfo_platform import ProblemDetails

_WRITE_TOKEN_HEADER = "x-dfo-write-token"

_LINE_CHART_INTENT = {
    "type": "line_chart",
    "frame_ref": {
        "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
        "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
    },
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
    "window": {"start": "2015", "end": "2024"},
    "title": "Malaria grant disbursements to Ethiopia",
}


def _create(client: TestClient, *, composition_id=None, write_token=None) -> dict:
    body: dict[str, object] = {"intent": _LINE_CHART_INTENT}
    if composition_id is not None:
        body["composition_id"] = composition_id
    headers = {_WRITE_TOKEN_HEADER: write_token} if write_token is not None else {}
    resp = client.post("/api/presentation/components", json=body, headers=headers)
    assert resp.status_code == 201, resp.json()
    return resp.json()


def _revise(client: TestClient, component_id: str, *, write_token=None, title="A new title"):
    headers = {_WRITE_TOKEN_HEADER: write_token} if write_token is not None else {}
    return client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": title}},
        headers=headers,
    )


def _assert_write_refused(resp) -> dict:
    assert resp.status_code == 403
    body = resp.json()
    assert body["code"] == "write_refused"
    assert body["retryable"] is False
    assert "create a new component" in body["detail"] or "composition_id" in body["detail"]
    ProblemDetails.model_validate(body)
    return body


def test_a_create_returns_the_write_token_as_a_top_level_field(client: TestClient):
    created = _create(client)

    assert created["write_token"].startswith("cmw_")
    assert "composition_write_token" not in created
    assert "cmw_" not in str(created["resolved_payload"])
    assert "cmw_" not in str(created["fact_summary"])


def test_a_create_starting_a_composition_returns_its_token_too(client: TestClient):
    created = _create(client, composition_id="new")

    assert created["write_token"].startswith("cmw_")
    assert created["composition_write_token"].startswith("cpw_")


def test_a_revise_with_the_write_token_succeeds(client: TestClient):
    created = _create(client)

    resp = _revise(client, created["component_id"], write_token=created["write_token"])

    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 2
    assert "write_token" not in body
    assert "composition_write_token" not in body


@pytest.mark.parametrize("case", ["missing", "wrong", "another_components", "empty"])
def test_a_revise_without_the_components_token_is_write_refused(client: TestClient, case):
    created = _create(client)
    other = _create(client)
    write_token = {
        "missing": None,
        "wrong": "cmw_" + "x" * 22,
        "another_components": other["write_token"],
        "empty": "",
    }[case]

    body = _assert_write_refused(_revise(client, created["component_id"], write_token=write_token))

    assert "create a new component" in body["detail"]
    assert (
        client.get(f"/api/presentation/components/{created['component_id']}").json()["revision"]
        == 1
    )


def test_a_component_with_no_stored_digest_refuses_revise(
    client: TestClient, revision_store: FakeRevisionStore
):
    created = _create(client)
    revision_store.forget_write_token(subject_kind="component", subject_id=created["component_id"])

    body = _assert_write_refused(
        _revise(client, created["component_id"], write_token=created["write_token"])
    )

    assert "predates write tokens" in body["detail"]


def test_reads_need_no_token_and_carry_none(client: TestClient):
    created = _create(client, composition_id="new")
    component_id = created["component_id"]
    revised = _revise(client, component_id, write_token=created["write_token"])
    assert revised.status_code == 200

    head = client.get(f"/api/presentation/components/{component_id}")
    first = client.get(f"/api/presentation/components/{component_id}", params={"revision": 1})

    for resp in (revised, head, first):
        assert resp.status_code == 200
        assert "cmw_" not in resp.text
        assert "cpw_" not in resp.text


def test_an_append_into_an_existing_composition_needs_the_composition_token(client: TestClient):
    first = _create(client, composition_id="new")
    composition_id = first["composition_id"]

    for write_token in (None, first["write_token"], "cpw_" + "x" * 22):
        resp = client.post(
            "/api/presentation/components",
            json={"intent": _LINE_CHART_INTENT, "composition_id": composition_id},
            headers={_WRITE_TOKEN_HEADER: write_token} if write_token is not None else {},
        )
        body = _assert_write_refused(resp)
        assert composition_id in body["detail"]

    appended = _create(
        client, composition_id=composition_id, write_token=first["composition_write_token"]
    )
    assert appended["composition_id"] == composition_id
    assert appended["composition_revision"] == 2
    assert "composition_write_token" not in appended


def test_a_composition_member_revises_with_only_its_own_token(client: TestClient):
    first = _create(client, composition_id="new")

    resp = _revise(client, first["component_id"], write_token=first["write_token"])
    assert resp.status_code == 200
    assert resp.json()["composition_revision"] == 2

    _assert_write_refused(
        _revise(client, first["component_id"], write_token=first["composition_write_token"])
    )
