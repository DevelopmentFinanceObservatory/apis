"""Checks that every successful Datasets route declares a non-empty OpenAPI response schema.

The app uses the deployed route table with the Datasets surface enabled.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

import dfo_apis.app as dfo_apis_app
from datasets_api.schema import build_schema_document
from dfo_apis.app import create_app
from search.engine.app import search_surface
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})


def _fake_search_surface(**kwargs: object):
    # Same stub test_datasets_surface_gate.py uses: no live LLM/graph/SDMX
    # client, so building the app needs no GEMINI_API_KEY or network. Accepts
    # and drops whatever deployed_surfaces()/all_surfaces() pass through
    # (e.g. rate_cache).
    return search_surface(
        graph=FakeGraph(), llm=FakeLLM(), sdmx=FakeSdmxClient(observations={}, availability={})
    )


@pytest.fixture(autouse=True)
def _stub_search_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dfo_apis_app, "search_surface", _fake_search_surface)


@pytest.fixture(autouse=True)
def _clear_other_surface_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    return create_app()


def _datasets_operations(schema: dict[str, Any]):
    for path, path_item in schema["paths"].items():
        if not path.startswith("/api/datasets"):
            continue
        for method, operation in path_item.items():
            if method in _HTTP_METHODS:
                yield path, method, operation


def _two_xx_responses(operation: dict[str, Any]):
    for status, response in operation["responses"].items():
        if status.startswith("2"):
            yield status, response


def test_every_datasets_path_is_present(app: FastAPI):
    schema = app.openapi()
    datasets_paths = {path for path in schema["paths"] if path.startswith("/api/datasets")}
    assert datasets_paths == {
        "/api/datasets",
        "/api/datasets/register",
        "/api/datasets/healthz",
        "/api/datasets/schema",
        "/api/datasets/capabilities",
        "/api/datasets/operations/{operation_id}",
        "/api/datasets/{id}",
        "/api/datasets/{id}/options",
        "/api/datasets/{id}/members",
        "/api/datasets/{id}/{content_version}",
        "/api/datasets/{id}/{content_version}/pin",
        "/api/datasets/{id}/{content_version}/observations",
        "/api/datasets/{id}/{content_version}/file",
        "/api/datasets/{id}/{content_version}/files/{digest}",
        "/api/datasets/{id}/{content_version}/description",
        "/api/datasets/{id}/{content_version}/description/values",
        "/api/datasets/frames/publish",
        "/api/datasets/frames/refresh",
        "/api/datasets/frames/{spec_id}/{content_version}/recipe",
        "/api/datasets/recipes/replays/{replay_key}",
        "/api/datasets/{id}/versions",
    }


def test_every_datasets_operation_declares_its_runtime_response(app: FastAPI):
    """Each operation declares a 2xx response, or, for the download route, its 302 with the
    `Location` header it answers."""
    schema = app.openapi()
    operations = list(_datasets_operations(schema))
    assert operations, "expected at least one datasets operation in the schema"
    for path, method, operation in operations:
        two_xx = list(_two_xx_responses(operation))
        redirect = operation["responses"].get("302", {}).get("headers", {})
        assert two_xx or "Location" in redirect, f"{method.upper()} {path} declares no success"


def test_every_datasets_2xx_response_carries_a_non_empty_schema(app: FastAPI):
    schema = app.openapi()
    for path, method, operation in _datasets_operations(schema):
        for status, response in _two_xx_responses(operation):
            body_schema = response.get("content", {}).get("application/json", {}).get("schema")
            assert body_schema, f"{method.upper()} {path} {status} has no application/json schema"


def test_register_2xx_response_is_the_definition_response_model(app: FastAPI):
    schema = app.openapi()
    register_op = schema["paths"]["/api/datasets/register"]["post"]
    body_schema = register_op["responses"]["200"]["content"]["application/json"]["schema"]
    assert body_schema == {"$ref": "#/components/schemas/DefinitionResponse"}


def test_schema_route_2xx_response_matches_the_document_it_describes(app: FastAPI):
    schema = app.openapi()
    schema_op = schema["paths"]["/api/datasets/schema"]["get"]
    body_schema = schema_op["responses"]["200"]["content"]["application/json"]["schema"]
    assert set(body_schema["required"]) == set(build_schema_document())
    manifest_property = body_schema["properties"]["manifest"]
    assert manifest_property != schema["components"]["schemas"]["Manifest"]


@pytest.mark.parametrize(
    ("path", "method"), [("/api/datasets", "post"), ("/api/datasets/{id}", "get")]
)
def test_operation_submitting_routes_document_their_202(app: FastAPI, path: str, method: str):
    response = app.openapi()["paths"][path][method]["responses"]["202"]
    body_schema = response["content"]["application/json"]["schema"]
    assert body_schema == {"$ref": "#/components/schemas/OperationResource"}
    assert set(response["headers"]) == {"Location", "Retry-After"}
