"""FastAPI matches routes in registration order. Meta routes must precede an unconstrained
`{id}` route so static paths such as `capabilities` remain reachable.

Tests also pin definition-door precedence for `/options`, confirm content versions reach the
snapshot door, and distinguish invalid ids from valid but unregistered ids. Mounting the
snapshot door first would turn a spec's `/options` request into `snapshot_not_found`.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.meta import build_meta_door
from datasets_api.doors.register import build_register_door
from dfo_platform.doors import DoorContribution, flatten
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from ._deps_doubles import fake_deps_cell
from .conftest import FakeDatasetStore
from .test_definition_door import _client, _graph, _mint_one_spec
from .test_output_options_doors import _body as _options_body
from .test_output_options_doors import _mint_and_snapshot_client


def test_meta_static_paths_precede_every_other_doors_routes() -> None:
    surface = datasets_surface(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    routes = surface.router.routes

    meta_paths = {
        route.path
        for route in build_meta_door(
            deps=fake_deps_cell(), settings=DatasetsBuildSettings.from_env()
        ).router.routes
    }
    meta_indices = [i for i, route in enumerate(routes) if route.path in meta_paths]
    other_indices = [i for i, route in enumerate(routes) if route.path not in meta_paths]

    assert meta_indices, "expected the meta door's own routes in the composed surface"
    assert other_indices, "expected at least one other door's route to compare order against"
    assert max(meta_indices) < min(other_indices)


def _stand_in_id_door() -> DoorContribution:
    """An unconstrained `{id}` route for precedence checks against the meta door."""
    router = APIRouter()

    @router.get("/api/datasets/{id}")
    async def catch_all(id: str) -> PlainTextResponse:
        return PlainTextResponse(f"caught:{id}")

    return DoorContribution(router=router, classify=lambda exc: None)


def _client_with_a_stand_in_id_door_after_register() -> TestClient:
    doors = (
        build_meta_door(deps=fake_deps_cell(), settings=DatasetsBuildSettings.from_env()),
        build_register_door(
            deps=fake_deps_cell(registry=LiveSpecRegistry(store=FakeDatasetStore()))
        ),
        _stand_in_id_door(),
    )
    router = flatten(doors)
    app = FastAPI()
    app.router.routes.extend(router.routes)
    return TestClient(app, raise_server_exceptions=False)


def test_a_stand_in_id_door_after_register_does_not_swallow_the_capabilities_path() -> None:
    client = _client_with_a_stand_in_id_door_after_register()

    resp = client.get("/api/datasets/capabilities")

    assert resp.status_code == 200
    assert "capability_version" in resp.json()
    assert resp.text != "caught:capabilities"


def test_the_stand_in_id_door_still_answers_for_an_unrecognized_segment() -> None:
    # The stand-in answers here to prove it is a real, reachable route,
    # so the assertion above tests precedence over an active competitor,
    # a stronger guarantee than an absent one would give.
    client = _client_with_a_stand_in_id_door_after_register()

    resp = client.get("/api/datasets/spec_something_unrecognized")

    assert resp.status_code == 200
    assert resp.text == "caught:spec_something_unrecognized"


def test_get_not_an_id_refuses_not_found_naming_the_grammar() -> None:
    """`SpecIdConvertor` routes malformed ids to the platform's `not_found` response, which
    explains the `spec_` prefix.
    """
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/not_an_id")

    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["detail"] == (
        "'not_an_id' is not a valid dataset id: an id is `spec_`-prefixed, case-sensitive"
    )


def test_get_unregistered_spec_id_refuses_spec_not_found() -> None:
    """A `spec_`-prefixed id matches the route's own grammar and reaches
    the definition door, which refuses `spec_not_found` for an id nothing
    durable ever claimed."""
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/spec_unregistered")

    assert resp.status_code == 404
    assert resp.json()["code"] == "spec_not_found"


def test_operations_route_precedes_the_snapshot_doors_two_segment_pattern() -> None:
    """`operations/op_123` reaches the meta route because the snapshot door accepts only
    `spec_` ids. The client leaves the meta lifespan inactive, so the response is
    `operation_not_found`.
    """
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/operations/op_123")

    assert resp.status_code == 404
    assert resp.json()["code"] == "operation_not_found"


def test_the_options_route_precedes_the_snapshot_doors_two_segment_pattern() -> None:
    """A literal `options` second segment reaches the definition door's own
    `GET /api/datasets/{id}/options` handler, serving `output_options`. The
    snapshot door's own unconstrained `{content_version}` route would
    otherwise treat the literal string `"options"` as a content_version
    and answer `snapshot_not_found` for one nothing ever minted.
    """
    client = _client(graph=_graph())
    spec_id = _mint_one_spec(client)

    resp = client.get(f"/api/datasets/{spec_id}/options")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "output_options" in body
    assert body["definition"]["spec_id"] == spec_id


def test_a_real_content_version_still_reaches_the_snapshot_door(version_store) -> None:
    """A non-`"options"`-named `content_version` segment reaches the
    snapshot door's own two-segment route. Only a literal `options`
    segment matches the options route's own fixed path suffix.
    """
    client = _mint_and_snapshot_client(version_store)
    mint_resp = client.post("/api/datasets", json=_options_body())
    assert mint_resp.status_code == 200, mint_resp.text
    manifest = mint_resp.json()
    spec_id = manifest["definition"]["spec_id"]
    content_version = manifest["content_version"]

    resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["content_version"] == content_version
