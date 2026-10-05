"""Malformed `/api/datasets/{id}` paths return `not_found` across route shapes and methods.
Valid ids and reserved literal paths reach their registered routes.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._snapshot_doubles import FakeSnapshotReader, FakeSpecReader
from .conftest import FakeDatasetStore
from .test_definition_door import _client, _graph

_MALFORMED_ID = "not_an_id"
_UNREGISTERED_ID = "spec_nonexistent_for_grammar_test"
_PIN_BODY = {"schema_version": 1, "artifact_digest": "sha256:" + "a" * 8}


class _UnusedReferenceWriter:
    """Reference-writer stub that fails if the pin door writes before checking permission."""

    def record_reference(self, **kwargs) -> datetime:
        raise AssertionError("not expected to be called: the permission check refuses first")

    def record_reference_unless_revoked(self, **kwargs) -> datetime:
        raise AssertionError("not expected to be called: the permission check refuses first")

    def remove_reference(self, **kwargs) -> None:
        raise AssertionError("not expected to be called: the permission check refuses first")


def test_get_malformed_id_refuses_not_found_naming_the_grammar():
    client = _client(graph=_graph())

    resp = client.get(f"/api/datasets/{_MALFORMED_ID}")

    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert "spec_" in body["detail"]
    assert "col_" in body["detail"]


def test_post_to_malformed_id_refuses_404_not_found():
    """POST to a malformed id returns `not_found` with the accepted id prefixes in `detail`."""
    client = _client(graph=_graph())

    resp = client.post(f"/api/datasets/{_MALFORMED_ID}")

    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert "spec_" in body["detail"]


def test_uppercase_prefixed_segment_still_fails_the_grammar():
    """The id grammar requires the lowercase `spec_` prefix."""
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/SPEC_uppercase")

    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert "spec_" in body["detail"]


def test_malformed_id_refuses_on_every_id_taking_route_shape():
    client = _client(graph=_graph())

    responses = {
        "GET /api/datasets/{id}/options": client.get(f"/api/datasets/{_MALFORMED_ID}/options"),
        "GET /api/datasets/{id}/{content_version}": client.get(f"/api/datasets/{_MALFORMED_ID}/v1"),
        "POST /api/datasets/{id}/{content_version}/pin": client.post(
            f"/api/datasets/{_MALFORMED_ID}/v1/pin", json=_PIN_BODY
        ),
    }
    for route, resp in responses.items():
        assert resp.status_code == 404, f"{route}: {resp.text}"
        assert resp.json()["code"] == "not_found", f"{route}: {resp.text}"


def test_well_formed_id_keeps_its_status_on_definition_and_snapshot_routes(version_store):
    """Well-formed unregistered ids reach each door and return `spec_not_found`."""
    client = _client(graph=_graph(), version_store=version_store)

    definition_resp = client.get(f"/api/datasets/{_UNREGISTERED_ID}")
    options_resp = client.get(f"/api/datasets/{_UNREGISTERED_ID}/options")
    snapshot_resp = client.get(f"/api/datasets/{_UNREGISTERED_ID}/v1")

    assert definition_resp.status_code == 404
    assert definition_resp.json()["code"] == "spec_not_found"
    assert options_resp.status_code == 404
    assert options_resp.json()["code"] == "spec_not_found"
    assert snapshot_resp.status_code == 404
    assert snapshot_resp.json()["code"] == "spec_not_found"


def test_well_formed_id_keeps_its_status_on_the_pin_route():
    """A well-formed id reaches the pin permission check before registry lookup."""
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=FakeSpecReader(),
        snapshot_reader=FakeSnapshotReader(),
        artifact_store=FakeArtifactStore(),
        reference_writer=_UnusedReferenceWriter(),
    )

    resp = client.post(f"/api/datasets/{_UNREGISTERED_ID}/v1/pin", json=_PIN_BODY)

    assert resp.status_code == 403
    assert resp.json()["code"] == "pin_not_permitted"


def test_wrong_method_on_a_well_formed_path_still_refuses_405():
    client = _client(graph=_graph())

    resp = client.put(f"/api/datasets/{_UNREGISTERED_ID}")

    assert resp.status_code == 405
    assert resp.json()["code"] == "method_not_allowed"


def test_literal_sibling_route_is_unaffected():
    """The reserved literal `/register` path reaches its route and returns 405 for GET."""
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/register")

    assert resp.status_code == 405
    assert resp.json()["code"] == "method_not_allowed"


def test_healthz_route_is_unaffected():
    """`healthz` is reserved at the single-segment shape too, so the real
    route answers 200."""
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/healthz")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/datasets/operations"),
        ("GET", "/api/datasets/register/options"),
        ("GET", "/api/datasets/register/v1"),
        ("POST", "/api/datasets/register/v1/pin"),
        ("GET", "/api/datasets/healthz/options"),
        ("GET", "/api/datasets/schema/v1"),
        ("GET", "/api/datasets/capabilities/v1"),
    ],
)
def test_meta_literal_used_off_its_own_shape_still_fails_the_grammar(method, path):
    """Reserved literals fail the id grammar when used in a route shape they do not define."""
    client = _client(graph=_graph())

    resp = client.post(path, json=_PIN_BODY) if method == "POST" else client.get(path)

    assert resp.status_code == 404, f"{method} {path}: {resp.text}"
    body = resp.json()
    assert body["code"] == "not_found", f"{method} {path}: {resp.text}"
    assert "spec_" in body["detail"]
    assert "col_" in body["detail"]


def test_get_known_operations_shape_still_reaches_the_operations_door():
    """The real operations route handles ids including the literal `options`."""
    client = _client(graph=_graph())

    op1_resp = client.get("/api/datasets/operations/op1")
    options_resp = client.get("/api/datasets/operations/options")

    assert op1_resp.status_code == 404
    assert op1_resp.json()["code"] == "operation_not_found"
    assert options_resp.status_code == 404
    assert options_resp.json()["code"] == "operation_not_found"


def test_post_to_operations_shape_still_refuses_405():
    """POST to the reserved operations route returns 405 for ordinary ids and `options`."""
    client = _client(graph=_graph())

    op1_resp = client.post("/api/datasets/operations/op1")
    options_resp = client.post("/api/datasets/operations/options")

    assert op1_resp.status_code == 405
    assert op1_resp.json()["code"] == "method_not_allowed"
    assert options_resp.status_code == 405
    assert options_resp.json()["code"] == "method_not_allowed"
