"""Checks the release document, schema validation, and health response."""

from __future__ import annotations

import json

import jsonschema
from fastapi.testclient import TestClient

from assembler.catalogue import CatalogueRelease
from assembler.fakes import FakeRevisionStore
from assembler.ports import FrameRef
from dfo_platform.build import build_commit
from presentation_api.codes import CODE_RETRY_MODE
from presentation_api.version import compute_presentation_capability_version


def _schema_registry_from_release_body(component_schemas: dict[str, dict]):
    """Resolve the served envelope schema's references using its component schemas."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources = [
        Resource.from_contents(schema, default_specification=DRAFT202012)
        for schema in component_schemas.values()
    ]
    return Registry().with_resources(
        (resource.id(), resource) for resource in resources if resource.id() is not None
    )


def test_release_document_shape(client: TestClient):
    resp = client.get("/api/presentation/release")
    assert resp.status_code == 200
    body = resp.json()
    assert body["catalog_version"] == "2026.8.0"
    assert body["schema_version"] == 1
    _all_members = {
        "line_chart",
        "bar_chart",
        "table",
        "stat_card",
        "note_narration",
        "source_card",
    }
    assert set(body["member_types"]) == _all_members
    assert "log_scale_requires_positive_values" in body["constraint_ids"]
    assert body["retry_modes"] == CODE_RETRY_MODE
    assert set(body["intent_schemas_tolerant"]) == _all_members
    assert set(body["minimal_intents"]) == _all_members
    assert body["intent_schemas_tolerant"]["stat_card"]["type"] == "object"
    assert body["minimal_intents"]["stat_card"]["type"] == "stat_card"


def test_release_document_carries_the_envelope_schema_and_component_schemas_mint_validates_against(
    client: TestClient, release: CatalogueRelease
):
    resp = client.get("/api/presentation/release")
    assert resp.status_code == 200
    body = resp.json()
    assert body["envelope_schema"] == release.envelope_schema
    assert body["envelope_schema"]["$id"] == (
        "https://developmentfinance.org/presentation/schema/v1/envelope.schema.json"
    )
    assert body["component_schemas"] == dict(release.component_schemas)
    assert set(body["component_schemas"]) == set(body["member_types"])
    assert body["component_schemas"]["stat_card"]["$id"] == (
        "https://developmentfinance.org/presentation/schema/v1/component/stat_card.schema.json"
    )


def test_a_created_resolved_payload_validates_against_the_served_envelope_schema(
    client: TestClient, frame_ref: FrameRef, revision_store: FakeRevisionStore
):
    resp = client.post(
        "/api/presentation/components",
        json={
            "intent": {
                "type": "stat_card",
                "frame_ref": {
                    "spec_id": frame_ref.spec_id,
                    "content_version": frame_ref.content_version,
                },
                "select": {
                    "stat_var": "dcid:MalariaGrantDisbursements",
                    "recipient": "country/ETH",
                },
            }
        },
    )
    assert resp.status_code == 201
    resolved_payload = resp.json()["resolved_payload"]

    release_body = client.get("/api/presentation/release").json()
    registry = _schema_registry_from_release_body(release_body["component_schemas"])
    jsonschema.Draft202012Validator(release_body["envelope_schema"], registry=registry).validate(
        resolved_payload
    )

    record = revision_store.read_current(component_id=resp.json()["component_id"])
    assert record is not None
    assert record.resolved_payload == resolved_payload


def test_healthz(client: TestClient, release: CatalogueRelease):
    resp = client.get("/api/presentation/healthz")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    expected_body = {
        "status": "ok",
        "build": build_commit(),
        "capability_version": compute_presentation_capability_version(release=release),
    }
    assert resp.content == json.dumps(expected_body, separators=(",", ":")).encode()
