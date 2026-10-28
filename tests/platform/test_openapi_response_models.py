"""Checks the composed OpenAPI document across all mounted surfaces: a non-empty 2xx schema on
every JSON response, one declared tag and a `{surface}_{handler}` operation id per operation, and
no field of a datasets body dumped with `exclude_none` that is required and admits None.

Routes excluded from OpenAPI and handlers returning prebuilt `Response` objects require separate
wire tests. This check validates the served schema only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Any, get_args

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel

from datasets_api.manifest_models import Manifest, OptionsEnvelope
from datasets_api.schema import OperationResource
from dfo_apis.app import all_surfaces
from dfo_platform.app import Surface, create_app
from search.models.observations import SpecObservations

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})

_GRAPH_ENV_VARS = (
    "GRAPH_DCP_BASE_URL",
    "GRAPH_DCP_TOKEN",
    "GRAPH_NODE_POOL_SIZE",
    "GRAPH_RESOLVE_TIMEOUT_SECONDS",
    "GRAPH_NODE_TIMEOUT_SECONDS",
)

# Starlette keeps converters on routes, while OpenAPI strips `{id:dfo_spec_id}` to `{id}`.
_CONVERTER_SUFFIX_RE = re.compile(r"\{([^:}]+):[^}]+\}")


def _schema_path(path: str) -> str:
    return _CONVERTER_SUFFIX_RE.sub(r"{\1}", path)


# Operations allowed to omit a 2xx response. The Datasets download link answers 302.
_NO_2XX_EXEMPT: frozenset[tuple[str, str]] = frozenset(
    {
        ("/api/datasets/{id}/{content_version}/file", "get"),
        ("/api/datasets/{id}/{content_version}/files/{digest}", "get"),
    }
)

# Operations allowed to return an empty schema.
_EMPTY_SCHEMA_EXEMPT: frozenset[tuple[str, str, str]] = frozenset()


@pytest.fixture(scope="module")
def surfaces() -> list[Surface]:
    for var in _GRAPH_ENV_VARS:
        os.environ.pop(var, None)
    return all_surfaces()


@pytest.fixture(scope="module")
def app(surfaces: list[Surface]) -> FastAPI:
    return create_app(surfaces)


@pytest.fixture(scope="module")
def schema(app: FastAPI) -> dict[str, Any]:
    return app.openapi()


def _operations(schema: dict[str, Any]):
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method in _HTTP_METHODS:
                yield path, method, operation


def _two_xx_responses(operation: dict[str, Any]):
    for status, response in operation["responses"].items():
        if status.startswith("2"):
            yield status, response


def _declares_properties(schema_node: dict[str, Any], *, components: dict[str, Any]) -> bool:
    """Whether `schema_node` names at least one field: resolves a `$ref`
    into `components["schemas"]` first, and recurses into every branch of
    a `oneOf`/`anyOf` discriminated union, requiring all of them to declare
    properties.
    """
    ref = schema_node.get("$ref")
    if ref is not None:
        target_name = ref.rsplit("/", 1)[-1]
        return _declares_properties(components["schemas"][target_name], components=components)
    branches = schema_node.get("oneOf") or schema_node.get("anyOf")
    if branches is not None:
        return all(_declares_properties(branch, components=components) for branch in branches)
    return bool(schema_node.get("properties"))


def test_every_operation_declares_at_least_one_2xx_response(schema: dict[str, Any]):
    operations = list(_operations(schema))
    assert operations, "expected at least one operation in the composed schema"
    for path, method, operation in operations:
        if (path, method) in _NO_2XX_EXEMPT:
            assert not list(_two_xx_responses(operation)), (
                f"{method.upper()} {path} is expected to advertise no 2xx response"
            )
            continue
        assert list(_two_xx_responses(operation)), (
            f"{method.upper()} {path} declares no 2xx response"
        )


def test_every_json_2xx_response_carries_a_non_empty_schema(schema: dict[str, Any]):
    components = schema["components"]
    seen_empty: set[tuple[str, str, str]] = set()
    for path, method, operation in _operations(schema):
        for status, response in _two_xx_responses(operation):
            content = response.get("content", {})
            if "application/json" not in content:
                continue
            body_schema = content["application/json"].get("schema")
            if not body_schema or not _declares_properties(body_schema, components=components):
                seen_empty.add((path, method, status))
    assert seen_empty == _EMPTY_SCHEMA_EXEMPT


def test_search_resolve_response_marks_status_and_every_part_status_as_required(
    schema: dict[str, Any],
):
    components = schema["components"]["schemas"]
    op = schema["paths"]["/api/search/resolve"]["post"]
    union = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert union["discriminator"]["propertyName"] == "kind"
    ref = union["discriminator"]["mapping"]["resolution"]
    resolve_response = components[ref.rsplit("/", 1)[-1]]
    assert "status" in resolve_response["required"]

    part_union = resolve_response["properties"]["parts"]["items"]
    part_names = {branch["$ref"].rsplit("/", 1)[-1] for branch in part_union["oneOf"]}
    assert part_names, "expected at least one part variant"
    for part_name in part_names:
        assert "status" in components[part_name]["required"], part_name


@pytest.mark.parametrize("kind", ["resolution", "catalogue", "declined"])
def test_every_search_resolve_response_kind_requires_its_discriminator(
    schema: dict[str, Any], kind: str
):
    """Every resolve response carries `kind`, so each kind's served schema lists it as
    required while its constructor keeps the default. No other defaulted field joins it."""
    components = schema["components"]["schemas"]
    op = schema["paths"]["/api/search/resolve"]["post"]
    union = op["responses"]["200"]["content"]["application/json"]["schema"]
    ref = union["discriminator"]["mapping"][kind]
    required = components[ref.rsplit("/", 1)[-1]]["required"]
    assert "kind" in required
    assert "schema_version" not in required


def test_search_headline_requires_its_recent_years(schema: dict[str, Any]):
    components = schema["components"]["schemas"]
    headline = components["Headline"]
    assert "recent_years" in headline["required"]
    assert headline["properties"]["recent_years"]["items"]["$ref"].endswith("/HeadlineYear")
    assert set(components["HeadlineYear"]["required"]) == {
        "year",
        "observations",
        "series",
        "role_counts",
    }


def test_no_two_schemas_share_a_title(schema: dict[str, Any]):
    titles: dict[str, list[str]] = {}
    for name, schema_node in schema["components"]["schemas"].items():
        title = schema_node.get("title")
        if title is None:
            continue
        titles.setdefault(title, []).append(name)
    dupes = {title: names for title, names in titles.items() if len(names) > 1}
    assert dupes == {}


def test_every_object_schema_names_its_properties(schema: dict[str, Any]):
    """A model whose wrap serializer is annotated `dict[str, Any]` publishes an open object,
    which generated clients type as a free map."""
    untyped = [
        name
        for name, schema_node in schema["components"]["schemas"].items()
        if schema_node.get("type") == "object" and "properties" not in schema_node
    ]
    assert untyped == []


# The leading word of every handler name. A handler named for a noun or ending in
# `_endpoint` gives its operation an id that names no action.
_HANDLER_VERBS = frozenset(
    {
        "create",
        "delete",
        "get",
        "list",
        "mint",
        "post",
        "publish",
        "put",
        "read",
        "refresh",
        "register",
        "render",
        "replace",
        "resolve",
        "revise",
    }
)


def _schema_routes(surfaces: list[Surface]) -> Iterator[tuple[str, APIRoute]]:
    for surface in surfaces:
        for route in surface.router.routes:
            if isinstance(route, APIRoute) and route.include_in_schema:
                yield surface.name, route


def test_every_operation_id_is_its_surface_and_handler_and_unique(
    surfaces: list[Surface], schema: dict[str, Any]
):
    operation_ids = [schema["paths"]["/codes/{code}"]["get"]["operationId"]]
    assert operation_ids == ["codes_get_code"]
    for surface_name, route in _schema_routes(surfaces):
        for method in route.methods:
            operation = schema["paths"][_schema_path(route.path)][method.lower()]
            assert operation["operationId"] == f"{surface_name}_{route.name}", route.path
            operation_ids.append(operation["operationId"])
        assert route.name.split("_", 1)[0] in _HANDLER_VERBS, route.name
    assert len(operation_ids) == sum(len(ops) for ops in schema["paths"].values())
    duplicates = sorted({oid for oid in operation_ids if operation_ids.count(oid) > 1})
    assert duplicates == []


def test_every_operation_carries_exactly_one_declared_tag(schema: dict[str, Any]):
    declared = {tag["name"]: tag.get("description") for tag in schema["tags"]}
    assert all(declared.values()), declared
    used: set[str] = set()
    for path, method, operation in _operations(schema):
        tags = operation.get("tags", [])
        assert len(tags) == 1, f"{method.upper()} {path} carries tags {tags}"
        assert tags[0] in declared, f"{method.upper()} {path} carries undeclared tag {tags[0]}"
        used.add(tags[0])
    assert used == set(declared)


def _models_in(annotation: object) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
    for arg in get_args(annotation):
        yield from _models_in(arg)


def _admits_none(annotation: object) -> bool:
    return annotation is None or annotation is type(None) or type(None) in get_args(annotation)


# The datasets bodies the doors dump with `exclude_none`.
_EXCLUDE_NONE_DATASETS_MODELS: tuple[type[BaseModel], ...] = (
    Manifest,
    OptionsEnvelope,
    OperationResource,
    SpecObservations,
)


def test_no_datasets_response_field_is_required_and_admits_none():
    """An `exclude_none` dump drops the key of a None field, so a required field that admits
    None is a key the schema requires and the wire leaves out. Such a field defaults to None
    and declares `exclude_if`, which keeps it optional in the schema."""
    pending = list(_EXCLUDE_NONE_DATASETS_MODELS)
    seen: set[type[BaseModel]] = set()
    offenders: list[str] = []
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        for name, field in model.model_fields.items():
            if field.is_required() and _admits_none(field.annotation):
                offenders.append(f"{model.__name__}.{name}")
            pending.extend(_models_in(field.annotation))
    assert offenders == []
