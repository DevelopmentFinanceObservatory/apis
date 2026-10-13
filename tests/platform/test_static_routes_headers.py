"""Checks static Search and Datasets routes through the composed app's middleware."""

from __future__ import annotations

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import datasets_api.schema
import search.engine.schema
from code_index.shared import get_index
from datasets_api.capabilities import build_capabilities as build_datasets_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_apis.app import create_app
from dfo_platform.build import build_commit
from dfo_platform.trace import TRACE_ID_HEADER
from search.engine.capabilities import build_capabilities as build_search_capabilities
from tests.engine._search_app import with_fake_search_deps


def _datasets_capabilities_payload() -> dict[str, object]:
    return build_datasets_capabilities(
        capability_version=DATASETS_CAPABILITY_VERSION,
        build=build_commit(),
        code_index_stamp=get_index().stamp,
        declarations=get_index().declarations,
        settings=DatasetsBuildSettings(),
    ).model_dump(mode="json")


_ROUTE_PAYLOAD_BUILDERS = {
    "/api/search/schema": search.engine.schema.schema_payload,
    "/api/search/capabilities": lambda: build_search_capabilities().model_dump(mode="json"),
    "/api/datasets/schema": datasets_api.schema.build_schema_document,
    "/api/datasets/capabilities": _datasets_capabilities_payload,
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    return TestClient(with_fake_search_deps(create_app()), raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("route", "payload_builder"), _ROUTE_PAYLOAD_BUILDERS.items(), ids=list(_ROUTE_PAYLOAD_BUILDERS)
)
def test_identity_body_matches_the_route_s_own_door_payload(client, route, payload_builder):
    resp = client.get(route, headers={"accept-encoding": "identity"})
    assert resp.status_code == 200
    assert resp.content == JSONResponse(payload_builder()).body


def test_vary_covers_accept_encoding_and_origin_with_no_cross_request_accumulation(
    monkeypatch: pytest.MonkeyPatch,
):
    """CORS adds `Origin` to `Vary` and keeps each request's origin separate."""
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.setenv("DFO_CORS_ORIGINS", "https://a.example.com,https://b.example.com")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    client = TestClient(with_fake_search_deps(create_app()), raise_server_exceptions=False)

    first = client.get(
        "/api/search/schema",
        headers={"accept-encoding": "identity", "origin": "https://a.example.com"},
    )
    vary_values = ",".join(first.headers.get_list("vary"))
    assert "Accept-Encoding" in vary_values
    assert "Origin" in vary_values
    assert first.headers.get_list("access-control-allow-origin") == ["https://a.example.com"]

    second = client.get(
        "/api/search/schema",
        headers={"accept-encoding": "identity", "origin": "https://b.example.com"},
    )
    assert second.headers.get_list("access-control-allow-origin") == ["https://b.example.com"]


def test_304_through_the_full_stack_has_no_body_and_no_content_length(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    client = TestClient(with_fake_search_deps(create_app()), raise_server_exceptions=False)

    fresh = client.get("/api/search/schema", headers={"accept-encoding": "identity"})
    etag = fresh.headers["etag"]

    conditional = client.get(
        "/api/search/schema",
        headers={"accept-encoding": "identity", "if-none-match": etag},
    )
    assert conditional.status_code == 304
    assert conditional.content == b""
    assert "content-length" not in conditional.headers


def test_static_cacheable_route_carries_no_trace_id_header_on_200_or_304(
    monkeypatch: pytest.MonkeyPatch,
):
    """Public cached responses omit trace IDs to prevent replaying one request's ID."""
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    client = TestClient(with_fake_search_deps(create_app()), raise_server_exceptions=False)

    fresh = client.get("/api/search/schema", headers={"accept-encoding": "identity"})
    assert fresh.status_code == 200
    assert TRACE_ID_HEADER not in fresh.headers

    conditional = client.get(
        "/api/search/schema",
        headers={"accept-encoding": "identity", "if-none-match": fresh.headers["etag"]},
    )
    assert conditional.status_code == 304
    assert TRACE_ID_HEADER not in conditional.headers


def test_ordinary_route_still_carries_a_distinct_trace_id_per_request(
    monkeypatch: pytest.MonkeyPatch,
):
    """Ordinary uncached routes receive a distinct trace ID on each request."""
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    client = TestClient(with_fake_search_deps(create_app()), raise_server_exceptions=False)

    first = client.get("/codes/request_invalid")
    second = client.get("/codes/request_invalid")

    assert first.status_code == 200
    assert "cache-control" not in first.headers
    assert TRACE_ID_HEADER in first.headers
    assert TRACE_ID_HEADER in second.headers
    assert first.headers[TRACE_ID_HEADER] != second.headers[TRACE_ID_HEADER]
