"""Checks non-2xx OpenAPI responses for Search-only and Search-plus-Datasets app compositions."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from pydantic import BaseModel

import dfo_apis.app as dfo_apis_app
from dfo_apis.app import create_app
from dfo_platform.app import _install_problem_details_openapi
from dfo_platform.problem import PROBLEM_JSON, ProblemDetails
from search.engine.app import search_surface
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})


def _fake_search_surface(**kwargs: object):
    # Same stub test_datasets_surface_gate.py uses: no live LLM/graph/SDMX
    # client, so building the app needs no GEMINI_API_KEY or network. Accepts
    # and drops whatever deployed_surfaces()/all_surfaces() pass through
    # (e.g. rate_cache) -- this stub only cares about the three it injects.
    return search_surface(
        graph=FakeGraph(),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


@pytest.fixture(autouse=True)
def _stub_search_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dfo_apis_app, "search_surface", _fake_search_surface)


@pytest.fixture(params=["0", "1"], ids=["search-only", "search+datasets"])
def app(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("DFO_DATASETS_SURFACE", request.param)
    return create_app()


def _operations(schema: dict[str, Any]):
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method in _HTTP_METHODS:
                yield path, method, operation


def _takes_body_or_parameters(operation: dict[str, Any]) -> bool:
    return "requestBody" in operation or bool(operation.get("parameters"))


def test_no_stock_validation_error_schema_survives(app: FastAPI):
    schema = app.openapi()
    assert "HTTPValidationError" not in json.dumps(schema)


def test_no_operation_declares_a_422_response(app: FastAPI):
    schema = app.openapi()
    for path, method, operation in _operations(schema):
        assert "422" not in operation["responses"], f"{method.upper()} {path} still declares 422"


def test_every_operation_declares_a_problem_details_default_response(app: FastAPI):
    schema = app.openapi()
    operations = list(_operations(schema))
    assert operations, "expected at least one operation in the schema"
    for path, method, operation in operations:
        default = operation["responses"].get("default")
        assert default is not None, f"{method.upper()} {path} has no default response"
        assert default["content"][PROBLEM_JSON]["schema"] == {
            "$ref": "#/components/schemas/ProblemDetails"
        }


def test_every_body_or_parameter_operation_declares_400(app: FastAPI):
    schema = app.openapi()
    validatable_operations = [
        (path, method, operation)
        for path, method, operation in _operations(schema)
        if _takes_body_or_parameters(operation)
    ]
    assert validatable_operations, "expected at least one body- or parameter-taking operation"
    for path, method, operation in validatable_operations:
        response_400 = operation["responses"].get("400")
        assert response_400 is not None, (
            f"{method.upper()} {path} takes a body or parameters but declares no 400"
        )
        assert response_400["content"][PROBLEM_JSON]["schema"] == {
            "$ref": "#/components/schemas/ProblemDetails"
        }


def test_operation_without_body_or_parameters_declares_no_explicit_400(app: FastAPI):
    schema = app.openapi()
    bare_operations = [
        (path, method, operation)
        for path, method, operation in _operations(schema)
        if not _takes_body_or_parameters(operation)
    ]
    assert bare_operations, "expected at least one operation with neither a body nor parameters"
    for path, method, operation in bare_operations:
        assert "400" not in operation["responses"], (
            f"{method.upper()} {path} has no body or parameters but declares 400"
        )


def test_problem_details_component_schema_exists(app: FastAPI):
    schema = app.openapi()
    schemas = schema["components"]["schemas"]
    assert "ProblemDetails" in schemas
    assert "Alternative" in schemas
    assert "ProblemIssue" in schemas
    # Pydantic's own $defs indirection must be fully rewritten onto
    # components.schemas -- no leftover $defs pointer anywhere in the tree.
    assert "$defs" not in json.dumps(schemas["ProblemDetails"])


def test_problem_details_component_schema_round_trips_required_fields(app: FastAPI):
    schema = app.openapi()
    component = schema["components"]["schemas"]["ProblemDetails"]
    required_fields = {
        name for name, field in ProblemDetails.model_fields.items() if field.is_required()
    }
    assert set(component["required"]) == required_fields
    assert required_fields <= component["properties"].keys()


def test_component_name_collision_fails_closed():
    class Alternative(BaseModel):
        note: str

    collision_app = FastAPI()

    @collision_app.get("/collide", response_model=Alternative)
    def _collide() -> Alternative:
        return Alternative(note="x")

    _install_problem_details_openapi(collision_app)
    with pytest.raises(AssertionError, match="Alternative"):
        collision_app.openapi()


def test_openapi_schema_regenerates_when_a_route_is_added():
    minimal_app = FastAPI()
    _install_problem_details_openapi(minimal_app)

    first = minimal_app.openapi()
    assert "/probe" not in first["paths"]

    def _probe() -> dict[str, str]:
        return {"ok": "true"}

    minimal_app.add_api_route("/probe", _probe, methods=["GET"])

    second = minimal_app.openapi()
    assert "/probe" in second["paths"]
    default = second["paths"]["/probe"]["get"]["responses"]["default"]
    assert default["content"][PROBLEM_JSON]["schema"] == {
        "$ref": "#/components/schemas/ProblemDetails"
    }
