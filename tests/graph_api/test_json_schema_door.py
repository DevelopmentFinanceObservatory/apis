"""Tests for the schema door: `GET /api/graph/json-schema`."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from graph_api import version
from graph_api.doors.json_schema import build_json_schema_door, served_wire_schemas
from graph_api.node_models import NodesRequest
from graph_api.resolve_models import ResolveRequest
from graph_api.version import GRAPH_CAPABILITY_VERSION


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(build_json_schema_door().router)
    return TestClient(app)


def test_serves_every_wire_schema_and_the_capabilities_path():
    resp = _client().get("/api/graph/json-schema")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schemas"] == served_wire_schemas()
    assert body["capabilities"] == "/api/graph/capabilities"


def test_a_response_schema_requires_every_field_the_server_emits():
    schemas = served_wire_schemas()
    nodes_defs = schemas["nodes_response"]["$defs"]

    assert "status" in nodes_defs["NotHeldEntry"]["required"]
    envelopes = {name: d for name, d in nodes_defs.items() if name.endswith("Node")}
    assert envelopes
    assert all("kind" in d["required"] for d in envelopes.values())
    # No model of a response is left on a base that makes defaulted fields optional. The
    # `licence*` spellings alone stay optional, so bodies recorded before them still validate.
    for name, schema in schemas.items():
        if name.endswith("_response"):
            for node in [schema, *schema.get("$defs", {}).values()]:
                properties = set(node.get("properties", {}))
                optional = properties - set(node.get("required", []))
                assert optional <= {"licence", "licence_type"}, (name, node.get("title"))
    source_block = nodes_defs["SourceBlock"]
    assert {"licence", "licence_type"} <= set(source_block["properties"])
    assert source_block["properties"]["license"]["deprecated"] is True


def test_a_request_schema_keeps_a_defaulted_field_optional():
    schemas = served_wire_schemas()

    assert schemas["nodes_request"] == NodesRequest.model_json_schema()
    assert schemas["nodes_request"]["required"] == ["dcids"]
    assert schemas["resolve_request"] == ResolveRequest.model_json_schema()
    assert schemas["resolve_request"]["required"] == ["texts"]


def test_the_served_set_is_the_set_the_capability_token_hashes():
    token = version._compute_graph_capability_version(wire_schemas=served_wire_schemas())

    assert token == GRAPH_CAPABILITY_VERSION


def test_serves_public_gzip_with_an_etag_and_answers_304():
    client = _client()
    first = client.get("/api/graph/json-schema", headers={"accept-encoding": "gzip"})

    again = client.get(
        "/api/graph/json-schema",
        headers={"accept-encoding": "gzip", "if-none-match": first.headers["etag"]},
    )

    assert first.headers["cache-control"] == "public, max-age=300"
    assert first.headers["content-encoding"] == "gzip"
    assert first.headers["etag"]
    assert again.status_code == 304
