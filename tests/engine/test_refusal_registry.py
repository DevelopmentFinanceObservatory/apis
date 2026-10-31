"""Every Search route that refuses binds the code registry endpoint it serves, and every exemplar
the Search surface declares goes out at the status the registry gives its code on each Search
endpoint that maps it.
"""

from __future__ import annotations

from dfo_platform.codes import bound_registry_endpoint
from search.engine.app import search_surface
from tests.code_registry import problem_statuses, registered_endpoints, unbound_routes

# Routes that serve no registry endpoint: they refuse only with platform codes.
_UNBOUND_PATHS = frozenset(
    {
        "/api/search/healthz",
        "/api/search/schema",
        "/api/search/capabilities",
        "/api/search/catalogue",
    }
)


def test_every_search_route_binds_the_registry_endpoint_it_serves():
    """The surface binds every Search endpoint the registry maps, each to a route it serves, and
    leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = search_surface()
    registered = registered_endpoints("search.")

    assert set(surface.registry_endpoints.values()) == registered
    assert {path for _, path in unbound_routes(surface)} == _UNBOUND_PATHS


def test_every_search_exemplar_goes_out_at_its_registered_status():
    """Each exemplar, classified with each Search endpoint bound, goes out at the one status the
    registry gives its code there. Its code is mapped on at least one Search endpoint."""
    surface = search_surface()
    endpoints = sorted(set(surface.registry_endpoints.values()))
    assert endpoints
    disagreements: set[tuple[str, str, str, int]] = set()
    unmapped: set[str] = set()
    for exemplar in surface.exception_exemplars:
        mapped = False
        for endpoint in endpoints:
            with bound_registry_endpoint(endpoint):
                spec = surface.classify(exemplar)
            if spec is None:
                continue
            statuses = problem_statuses(spec.code, endpoint)
            if not statuses:
                continue
            mapped = True
            if statuses != {spec.status}:
                disagreements.add((type(exemplar).__qualname__, endpoint, spec.code, spec.status))
        if not mapped:
            unmapped.add(type(exemplar).__qualname__)
    assert sorted(disagreements) == []
    assert sorted(unmapped) == []
