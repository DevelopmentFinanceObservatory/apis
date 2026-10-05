"""Checks that route response models match successful and refusal wire bodies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assembler.ports import FrameRef
from dfo_platform import HealthResponse, ProblemDetails
from presentation_api.models import (
    CapabilitiesResponse,
    ComponentReadResponse,
    CreateReplyResponse,
    FrameDimensionValuesResponse,
    FrameFilteredViewResponse,
    FrameStructuralViewResponse,
    ReleaseDocumentResponse,
    WriteReplyResponse,
)

_FRAME_REF = {
    "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
    "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
}

_LINE_CHART_INTENT = {
    "type": "line_chart",
    "frame_ref": _FRAME_REF,
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
    "window": {"start": "2015", "end": "2024"},
    "title": "Malaria grant disbursements to Ethiopia",
}

_BAR_CHART_INTENT = {
    "type": "bar_chart",
    "frame_ref": _FRAME_REF,
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
    "window": {"start": "2015", "end": "2024"},
    "title": "Malaria grant disbursements to Ethiopia",
}

_TABLE_INTENT = {
    "type": "table",
    "frame_ref": _FRAME_REF,
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
    "window": {"start": "2015", "end": "2024"},
    "title": "Malaria grant disbursements to Ethiopia",
}

_STAT_CARD_INTENT = {
    "type": "stat_card",
    "frame_ref": _FRAME_REF,
    "select": {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/ETH",
    },
}

_NOTE_NARRATION_INTENT = {
    "type": "note_narration",
    "text": "Disbursements rose each year after 2020.",
}

_SOURCE_CARD_INTENT = {
    "type": "source_card",
    "frame_ref": _FRAME_REF,
}

_MEMBER_INTENTS = {
    "line_chart": _LINE_CHART_INTENT,
    "bar_chart": _BAR_CHART_INTENT,
    "table": _TABLE_INTENT,
    "stat_card": _STAT_CARD_INTENT,
    "note_narration": _NOTE_NARRATION_INTENT,
    "source_card": _SOURCE_CARD_INTENT,
}


def _create(client: TestClient, *, intent: dict):
    return client.post(
        "/api/presentation/components",
        json={"intent": intent},
    )


def test_create_reply_matches_create_reply_response(client: TestClient):
    resp = _create(client, intent=_LINE_CHART_INTENT)
    assert resp.status_code == 201
    CreateReplyResponse.model_validate(resp.json())


def test_a_substantive_revise_matches_write_reply_response(client: TestClient):
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    resp = client.patch(
        f"/api/presentation/components/{created['component_id']}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 200
    WriteReplyResponse.model_validate(resp.json())


def test_a_no_op_revise_matches_write_reply_response(client: TestClient):
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    resp = client.patch(
        f"/api/presentation/components/{created['component_id']}",
        json={"intent": {"title": _LINE_CHART_INTENT["title"]}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 200
    WriteReplyResponse.model_validate(resp.json())


def test_a_composition_members_create_revise_and_read_carry_non_null_membership(
    client: TestClient,
):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT, "composition_id": "new"},
    )
    assert resp.status_code == 201
    created = resp.json()
    create_reply = CreateReplyResponse.model_validate(created)
    assert create_reply.composition_id is not None
    assert create_reply.composition_revision is not None

    revise_resp = client.patch(
        f"/api/presentation/components/{created['component_id']}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert revise_resp.status_code == 200
    revise_reply = WriteReplyResponse.model_validate(revise_resp.json())
    assert revise_reply.composition_id is not None
    assert revise_reply.composition_revision is not None

    read_resp = client.get(
        f"/api/presentation/components/{created['component_id']}",
    )
    assert read_resp.status_code == 200
    read_reply = ComponentReadResponse.model_validate(read_resp.json())
    assert read_reply.composition_id is not None
    assert read_reply.composition_revision is not None


@pytest.mark.parametrize("member", sorted(_MEMBER_INTENTS))
def test_component_read_matches_component_read_response(client: TestClient, member: str):
    created = _create(client, intent=_MEMBER_INTENTS[member]).json()
    resp = client.get(
        f"/api/presentation/components/{created['component_id']}",
    )
    assert resp.status_code == 200
    ComponentReadResponse.model_validate(resp.json())


@pytest.mark.parametrize(
    ("params", "model"),
    [
        (None, FrameStructuralViewResponse),
        ({"dimension": "recipient"}, FrameDimensionValuesResponse),
        ({"query": "eth"}, FrameFilteredViewResponse),
    ],
    ids=["structural", "dimension", "filtered"],
)
def test_frame_view_matches_its_shape_model(
    client: TestClient, frame_ref: FrameRef, params: dict | None, model: type
):
    resp = client.get(
        f"/api/presentation/frames/{frame_ref.spec_id}/{frame_ref.content_version}",
        params=params,
    )
    assert resp.status_code == 200
    model.model_validate(resp.json())


def test_release_matches_release_document_response(client: TestClient):
    resp = client.get("/api/presentation/release")
    assert resp.status_code == 200
    ReleaseDocumentResponse.model_validate(resp.json())


def test_capabilities_matches_capabilities_response(client: TestClient):
    resp = client.get("/api/presentation/capabilities")
    assert resp.status_code == 200
    CapabilitiesResponse.model_validate(resp.json())


def test_healthz_matches_health_response(client: TestClient):
    resp = client.get("/api/presentation/healthz")
    assert resp.status_code == 200
    HealthResponse.model_validate(resp.json())


def test_a_schema_invalid_body_matches_problem_details(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": {"select": []}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "schema_invalid"
    ProblemDetails.model_validate(body)
    assert set(body) <= set(ProblemDetails.model_fields)


def test_a_not_found_body_matches_problem_details(client: TestClient):
    resp = client.get("/api/presentation/components/cmp_does_not_exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    ProblemDetails.model_validate(body)
    assert set(body) <= set(ProblemDetails.model_fields)


def test_an_edit_conflict_body_matches_problem_details(
    client: TestClient, revision_store, monkeypatch: pytest.MonkeyPatch
):
    created = _create(client, intent=_LINE_CHART_INTENT).json()
    component_id = created["component_id"]
    # A head that vanishes between the request and the pre-read is an edit conflict.
    monkeypatch.setattr(revision_store, "read_current", lambda *, component_id: None)
    resp = client.patch(
        f"/api/presentation/components/{component_id}",
        json={"intent": {"title": "x"}},
        headers={"x-dfo-write-token": created["write_token"]},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "edit_conflict"
    ProblemDetails.model_validate(body)
    assert set(body) <= set(ProblemDetails.model_fields)


def test_an_unsupported_schema_version_body_matches_problem_details(client: TestClient):
    resp = client.post(
        "/api/presentation/components",
        json={"intent": _LINE_CHART_INTENT, "schema_version": 3},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "unsupported_schema_version"
    ProblemDetails.model_validate(body)
    assert set(body) <= set(ProblemDetails.model_fields)
