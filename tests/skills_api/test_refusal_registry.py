"""Every Skills route that refuses binds the code registry endpoint it serves, and every exemplar
the Skills surface declares goes out at the status the registry gives its code on each Skills
endpoint that maps it.
"""

from __future__ import annotations

from dfo_spec_kit.registry import REGISTRY

from dfo_platform.codes import bound_registry_endpoint
from skills_api.app import skills_surface

# Routes that serve no registry endpoint: they refuse only with platform codes.
_UNBOUND_PATHS = frozenset({"/api/skills/healthz"})


def _statuses(code: str, endpoint: str) -> set[int]:
    """The statuses the registry gives `code`'s problem details on `endpoint`."""
    entry = REGISTRY[code] if code in REGISTRY else None  # noqa: SIM401 -- CodeRegistry has no .get()
    return {
        mapping.http_status
        for mapping in (entry.mappings if entry is not None else ())
        if mapping.channel == "problem_details"
        and mapping.http_status is not None
        and mapping.endpoint == endpoint
    }


def test_every_skills_route_binds_the_registry_endpoint_it_serves():
    """The surface binds every Skills endpoint the registry maps, each to a route it serves, and
    leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = skills_surface(skills_index=object())
    registered = {
        mapping.endpoint
        for entry in REGISTRY.entries
        for mapping in entry.mappings
        if mapping.endpoint.startswith("skills.")
    }
    routed = {getattr(route, "path", None) for route in surface.router.routes}

    assert set(surface.registry_endpoints.values()) == registered
    assert routed - set(surface.registry_endpoints) == _UNBOUND_PATHS


def test_every_skills_exemplar_goes_out_at_its_registered_status():
    """Each exemplar, classified with each Skills endpoint bound, goes out at the one status the
    registry gives its code there. Its code is mapped on at least one Skills endpoint."""
    surface = skills_surface(skills_index=object())
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
            statuses = _statuses(spec.code, endpoint)
            if not statuses:
                continue
            mapped = True
            if statuses != {spec.status}:
                disagreements.add((type(exemplar).__qualname__, endpoint, spec.code, spec.status))
        if not mapped:
            unmapped.add(type(exemplar).__qualname__)
    assert sorted(disagreements) == []
    assert sorted(unmapped) == []
