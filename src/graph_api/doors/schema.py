"""Serve `GET /api/graph/schema`, the JSON Schema of every served request and response body.

`served_wire_schemas` names the models, the same set `graph_api.version` hashes into the
capability token. The document is encoded once when the door is built and answers
`public, max-age=300` with a strong `ETag` and a gzip representation, honouring
`If-None-Match` with 304. It changes only with a deploy.
"""

from __future__ import annotations

import functools
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from dfo_platform import HealthResponse, ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.static_response import FrozenJSON, freeze_public_json
from graph_api.doors.members import MembersPage, Roots
from graph_api.node_models import NodeResponse, NodesRequest, NodesResponse
from graph_api.resolve_models import ResolveRequest, ResolveResponse
from graph_api.source_models import SourceCatalogue, SourceDocumentation


def served_wire_schemas() -> dict[str, dict[str, Any]]:
    """The JSON Schema of each served request and response body, keyed by name.

    A request schema is built in validation mode, where a defaulted field is optional. A
    response schema is built in serialization mode, which describes the body the server emits.
    """
    return {
        "health_response": HealthResponse.model_json_schema(mode="serialization"),
        "resolve_request": ResolveRequest.model_json_schema(),
        "resolve_response": ResolveResponse.model_json_schema(mode="serialization"),
        "node_response": TypeAdapter(NodeResponse).json_schema(mode="serialization"),
        "nodes_request": NodesRequest.model_json_schema(),
        "nodes_response": NodesResponse.model_json_schema(mode="serialization"),
        "members_response": MembersPage.model_json_schema(mode="serialization"),
        "roots_response": Roots.model_json_schema(mode="serialization"),
        "source_documentation_response": SourceDocumentation.model_json_schema(
            mode="serialization"
        ),
        "source_catalogue_response": SourceCatalogue.model_json_schema(mode="serialization"),
    }


class GraphSchema(BaseModel):
    """The full `GET /api/graph/schema` document."""

    model_config = ConfigDict(title="GraphSchema")

    schemas: dict[str, dict[str, Any]] = Field(
        description="JSON Schema of each served request and response body, keyed by name"
    )
    capabilities: str = Field(description="path of this surface's capability document")


@functools.cache
def _frozen_schema() -> FrozenJSON:
    document = GraphSchema(schemas=served_wire_schemas(), capabilities="/api/graph/capabilities")
    return freeze_public_json(document.model_dump(mode="json"), max_age=300)


def classify_schema_exception(_exc: BaseException) -> ProblemSpec | None:
    """This door has no exceptions to classify."""
    return None


async def schema(request: Request) -> Response:
    return _frozen_schema().respond(request)


def build_schema_door() -> DoorContribution:
    # Encode the document now, so a model that cannot render a schema fails startup.
    _frozen_schema()
    router = APIRouter()
    # `response_model` documents the response body in `/openapi.json`.
    router.add_api_route("/api/graph/schema", schema, methods=["GET"], response_model=GraphSchema)
    return DoorContribution(router=router, classify=classify_schema_exception)
