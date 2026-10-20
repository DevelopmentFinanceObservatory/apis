"""The document `GET /api/search/schema` serves, which the capability token also hashes."""

from __future__ import annotations

from code_index.declarations import DATASETS
from search.models import RESOLVE_RESPONSE_ADAPTER, SCHEMA_VERSION, ResolveRequest


def _with_declared_datasets(request_schema: dict) -> dict:
    """List the declared datasets (`code_index.declarations.DATASETS`), in rank order, as the
    values a parsed part's `provenance` takes: one `const` per dataset, titled with its label,
    described by its description and flagged with `x-opt-in`. The declarations are code, so
    the schema, and the capability version that hashes it, are fixed per process and read no
    code index, which needs a bound graph release under `GRAPH_RELEASE_COVERAGE`."""
    field = request_schema["$defs"]["ParsedPart"]["properties"]["provenance"]
    named = next(branch for branch in field["anyOf"] if branch.get("type") == "string")
    named["oneOf"] = [
        {
            "const": dataset.provenance,
            "title": dataset.label,
            "description": dataset.description,
            "x-opt-in": dataset.opt_in,
        }
        for dataset in sorted(DATASETS, key=lambda dataset: dataset.rank)
    ]
    return request_schema


def schema_payload() -> dict:
    """The versioned, unauthenticated contract schema. Capabilities are linked by relative
    path."""
    return {
        "schema_version": SCHEMA_VERSION,
        "request": _with_declared_datasets(ResolveRequest.model_json_schema()),
        "response": RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization"),
        "capabilities": "/api/search/capabilities",
    }
