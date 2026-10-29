"""Checks response schemas and modeled operations in the served OpenAPI document."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "options", "head", "trace"})

_MODELED_OPERATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("/api/presentation/components", "post"),
        ("/api/presentation/components/{component_id}", "patch"),
        ("/api/presentation/components/{component_id}", "get"),
        ("/api/presentation/components/{component_id}/revisions/{revision}/render", "post"),
        ("/api/presentation/healthz", "get"),
        ("/api/presentation/capabilities", "get"),
        ("/api/presentation/frames/{spec_id}", "get"),
        ("/api/presentation/frames/{spec_id}/{content_version}", "get"),
        ("/api/presentation/release", "get"),
    }
)

# Operations whose 2xx body is a file rather than JSON, with the media types each declares.
_FILE_MEDIA_TYPES = frozenset(
    {
        "application/zip",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.apache.parquet",
    }
)
_REVISION_PATH = "/api/presentation/components/{component_id}/revisions/{revision}"
_FILE_OPERATIONS: dict[tuple[str, str], frozenset[str]] = {
    (f"{_REVISION_PATH}/data", "get"): _FILE_MEDIA_TYPES,
    (f"{_REVISION_PATH}/render/data", "get"): _FILE_MEDIA_TYPES,
}


def _two_xx_responses(operation: dict[str, Any]):
    for status, response in operation["responses"].items():
        if status.startswith("2"):
            yield status, response


def _declares_properties(schema: dict[str, Any], *, components: dict[str, Any]) -> bool:
    """Whether `schema` names at least one field, resolving a `$ref` into
    `components["schemas"]` first and recursing into `anyOf`'s members for
    a union schema."""
    ref = schema.get("$ref")
    if ref is not None:
        target_name = ref.rsplit("/", 1)[-1]
        return _declares_properties(components["schemas"][target_name], components=components)
    if "anyOf" in schema:
        return all(
            _declares_properties(member, components=components) for member in schema["anyOf"]
        )
    return bool(schema.get("properties"))


def test_every_presentation_2xx_response_carries_a_non_empty_schema(client: TestClient):
    schema = client.get("/openapi.json").json()
    components = schema["components"]
    seen: set[tuple[str, str]] = set()
    for path, path_item in schema["paths"].items():
        if not path.startswith("/api/presentation"):
            continue
        for method, operation in path_item.items():
            if method not in _HTTP_METHODS:
                continue
            if (path, method) in _FILE_OPERATIONS:
                continue
            seen.add((path, method))
            for status, response in _two_xx_responses(operation):
                body_schema = response.get("content", {}).get("application/json", {}).get("schema")
                assert body_schema, (
                    f"{method.upper()} {path} {status} has no application/json schema"
                )
                assert _declares_properties(body_schema, components=components), (
                    f"{method.upper()} {path} {status}'s schema declares no properties"
                )
    assert seen == _MODELED_OPERATIONS


def test_every_presentation_operation_declares_a_2xx_response(client: TestClient):
    schema = client.get("/openapi.json").json()
    operations = [
        (path, method, operation)
        for path, path_item in schema["paths"].items()
        if path.startswith("/api/presentation")
        for method, operation in path_item.items()
        if method in _HTTP_METHODS
    ]
    assert operations, "expected at least one presentation operation in the schema"
    for path, method, operation in operations:
        assert list(_two_xx_responses(operation)), (
            f"{method.upper()} {path} declares no 2xx response"
        )


def test_release_schema_states_the_retryable_rule_on_retry_modes(client: TestClient):
    schema = client.get("/openapi.json").json()
    retry_modes = schema["components"]["schemas"]["ReleaseDocumentResponse"]["properties"][
        "retry_modes"
    ]
    description = retry_modes["description"]
    assert "authoritative" in description
    assert "schema_invalid" in description
    assert "frame_unavailable" in description
    assert "request_invalid" in description
    assert "false" in description


def test_create_reply_schema_names_affordance_and_revise_reply_does_not(client: TestClient):
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    assert "affordance" in components["CreateReplyResponse"]["properties"]
    assert "affordance" not in components["WriteReplyResponse"]["properties"]


def test_every_component_body_schema_names_resolved_payload(client: TestClient):
    components = client.get("/openapi.json").json()["components"]["schemas"]

    for model in ("CreateReplyResponse", "WriteReplyResponse", "ComponentReadResponse"):
        assert "resolved_payload" in components[model]["properties"], model
        assert "resolved_payload" in components[model]["required"], model


def test_every_file_operation_declares_its_media_types(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]
    for (path, method), media_types in _FILE_OPERATIONS.items():
        content = paths[path][method]["responses"]["200"]["content"]
        assert set(content) == media_types
