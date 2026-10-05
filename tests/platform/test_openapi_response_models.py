"""Checks for a non-empty 2xx schema on every JSON response across all mounted surfaces.

Routes excluded from OpenAPI and handlers returning prebuilt `Response` objects require separate
wire tests. This check validates the served schema only.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi import FastAPI

from dfo_apis.app import all_surfaces
from dfo_platform.app import create_app

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})

_GRAPH_ENV_VARS = (
    "GRAPH_DCP_BASE_URL",
    "GRAPH_DCP_TOKEN",
    "GRAPH_NODE_POOL_SIZE",
    "GRAPH_RESOLVE_TIMEOUT_SECONDS",
    "GRAPH_NODE_TIMEOUT_SECONDS",
)

# Operations allowed to omit a 2xx response.
_NO_2XX_EXEMPT: frozenset[tuple[str, str]] = frozenset()

# Operations allowed to return an empty schema.
_EMPTY_SCHEMA_EXEMPT: frozenset[tuple[str, str, str]] = frozenset()


@pytest.fixture(scope="module")
def app() -> FastAPI:
    for var in _GRAPH_ENV_VARS:
        os.environ.pop(var, None)
    return create_app(all_surfaces())


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


def test_no_two_schemas_share_a_title(schema: dict[str, Any]):
    titles: dict[str, list[str]] = {}
    for name, schema_node in schema["components"]["schemas"].items():
        title = schema_node.get("title")
        if title is None:
            continue
        titles.setdefault(title, []).append(name)
    dupes = {title: names for title, names in titles.items() if len(names) > 1}
    assert dupes == {}
