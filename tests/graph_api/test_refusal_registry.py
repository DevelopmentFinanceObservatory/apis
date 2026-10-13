"""Every Graph route that refuses binds the code registry endpoint it serves, and every exemplar a
Graph door declares goes out at the status the registry gives its code on that endpoint.

A code the platform owns for every surface, such as `timeout`, is mapped on `platform.any`, so
an endpoint that maps no status of its own for it takes that one.
"""

from __future__ import annotations

import pytest
from dfo_spec_kit.registry import REGISTRY

from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.doors import DoorContribution
from graph_api.app import graph_surface
from graph_api.doors.members import build_members_door
from graph_api.doors.node import build_node_door
from graph_api.doors.resolve import build_resolve_door
from graph_api.doors.sources import build_sources_door

from .conftest import FakeGraphNodeReader

# Routes that serve no registry endpoint: they refuse only with platform codes.
_UNBOUND_PATHS = frozenset(
    {"/api/graph/healthz", "/api/graph/capabilities", "/api/graph/schema", "/api/graph/sources"}
)


@pytest.fixture(scope="module")
def doors() -> tuple[DoorContribution, ...]:
    return build_resolve_door(), build_sources_door(), build_members_door(), build_node_door()


def _statuses(code: str, endpoint: str) -> set[int]:
    """The statuses the registry gives `code` on `endpoint`'s problem details, or on
    `platform.any`'s when `endpoint` maps none."""
    entry = REGISTRY[code] if code in REGISTRY else None  # noqa: SIM401 -- CodeRegistry has no .get()
    mappings = [
        mapping
        for mapping in (entry.mappings if entry is not None else ())
        if mapping.channel == "problem_details" and mapping.http_status is not None
    ]
    on_endpoint = {mapping.http_status for mapping in mappings if mapping.endpoint == endpoint}
    return on_endpoint or {
        mapping.http_status for mapping in mappings if mapping.endpoint == "platform.any"
    }


def test_every_graph_route_binds_the_registry_endpoint_it_serves():
    """The surface binds every Graph endpoint the registry maps, each to a route it serves, and
    leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = graph_surface(graph=FakeGraphNodeReader())
    registered = {
        mapping.endpoint
        for entry in REGISTRY.entries
        for mapping in entry.mappings
        if mapping.endpoint.startswith("graph.")
    }
    routed = {getattr(route, "path", None) for route in surface.router.routes}

    assert set(surface.registry_endpoints.values()) == registered
    assert routed - set(surface.registry_endpoints) == _UNBOUND_PATHS


def test_every_graph_exemplar_goes_out_at_its_registered_status(doors):
    """Each exemplar is classified by its own door with each of the door's endpoints bound, and
    goes out at the one status the registry gives its code there. Its code is mapped on at least
    one endpoint the door serves."""
    disagreements: set[tuple[str, str, str, int]] = set()
    unmapped: set[tuple[str, str]] = set()
    for door in doors:
        endpoints = set(door.registry_endpoints.values())
        assert endpoints, door.router.routes
        for exemplar in door.exception_exemplars:
            mapped = False
            for endpoint in sorted(endpoints):
                with bound_registry_endpoint(endpoint):
                    spec = door.classify(exemplar)
                if spec is None:
                    continue
                statuses = _statuses(spec.code, endpoint)
                if not statuses:
                    continue
                mapped = True
                if statuses != {spec.status}:
                    disagreements.add(
                        (type(exemplar).__qualname__, endpoint, spec.code, spec.status)
                    )
            if not mapped and door.classify(exemplar) is not None:
                unmapped.add((type(exemplar).__qualname__, ",".join(sorted(endpoints))))
    assert sorted(disagreements) == []
    assert sorted(unmapped) == []
