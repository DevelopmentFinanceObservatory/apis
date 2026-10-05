"""Tests OpenAPI component merging for identical shapes and refusal for conflicting shapes."""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from dfo_apis.app import all_surfaces
from dfo_platform.app import Surface, _problem_details_component_schemas, create_app
from dfo_platform.problem import Cause


def _dummy_surface(*, prefix: str, response_model: type[BaseModel]) -> Surface:
    router = APIRouter()

    @router.get(f"{prefix}/thing", response_model=response_model)
    async def thing():  # pragma: no cover -- schema generation never calls this
        raise NotImplementedError

    return Surface(
        name="dummy",
        prefix=prefix,
        router=router,
        capability_version="v_dummy",
        codes={},
    )


def _different_cause_lookalike() -> type[BaseModel]:
    """A model literally named `Cause` at class-creation time (not renamed
    after the fact -- FastAPI/pydantic bake a model's `$defs` ref into its
    core schema when the class is created, so a post-hoc `__name__`
    reassignment would never actually collide with the real
    `dfo_platform.problem.Cause` component), with a deliberately different
    shape from it.
    """
    return create_model("Cause", only_field=(str, Field()))


def test_identical_named_component_merges_without_raising():
    # The real `Cause` class, reused as a route's own response_model --
    # exactly the scenario `Unsupported`/`NoData` create for real: the same
    # class reaches the schema through both FastAPI's own route walk and
    # `_problem_details_component_schemas`'s manual injection.
    app = create_app([_dummy_surface(prefix="/api/dummy", response_model=Cause)])
    schema = app.openapi()
    injected = _problem_details_component_schemas()
    assert schema["components"]["schemas"]["Cause"] == injected["Cause"]


def test_genuinely_different_named_component_still_raises():
    app = create_app(
        [_dummy_surface(prefix="/api/dummy", response_model=_different_cause_lookalike())]
    )
    with pytest.raises(AssertionError, match="DIFFERENT shape"):
        app.openapi()


@pytest.fixture(scope="module")
def composed_schema() -> dict[str, Any]:
    for var in (
        "GRAPH_DCP_BASE_URL",
        "GRAPH_DCP_TOKEN",
        "GRAPH_NODE_POOL_SIZE",
        "GRAPH_RESOLVE_TIMEOUT_SECONDS",
        "GRAPH_NODE_TIMEOUT_SECONDS",
    ):
        os.environ.pop(var, None)
    app = create_app(all_surfaces())
    return app.openapi()


class TestProblemDetailsComponentsShareOneCanonicalShape:
    """Whatever surface a `ProblemDetails`-nested component name reaches the
    merged schema through, its content is exactly `_problem_details_component_schemas`'s
    own canonical shape -- never a surface-specific model quietly standing in
    for it. Proves Datasets/Graph (and every other mounted surface) carry the
    unchanged, single-source ProblemDetails vocabulary.
    """

    def test_every_injected_name_matches_its_canonical_shape(self, composed_schema):
        injected = _problem_details_component_schemas()
        schemas = composed_schema["components"]["schemas"]
        for name, expected in injected.items():
            assert schemas[name] == expected

    def test_cause_appears_exactly_once_and_is_shared(self, composed_schema):
        schemas = composed_schema["components"]["schemas"]
        assert "Cause" in schemas
        assert schemas["Cause"] == Cause.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        # ProblemDetails' own `cause` field references it.
        problem_details_cause = schemas["ProblemDetails"]["properties"]["cause"]
        assert _refs_named(problem_details_cause, "Cause")
        # Search text-door parts share the `Cause` component.
        assert "Unsupported" in schemas
        assert _refs_named(schemas["Unsupported"]["properties"]["cause"], "Cause")
        assert "NoData" in schemas
        assert _refs_named(schemas["NoData"]["properties"]["cause"], "Cause")


def _refs_named(node: object, name: str) -> bool:
    """Whether `node` (a property schema, possibly `anyOf`-wrapped for an
    Optional field) contains a `$ref` to `#/components/schemas/{name}`."""
    if isinstance(node, dict):
        if node.get("$ref") == f"#/components/schemas/{name}":
            return True
        return any(_refs_named(value, name) for value in node.values())
    if isinstance(node, list):
        return any(_refs_named(item, name) for item in node)
    return False
