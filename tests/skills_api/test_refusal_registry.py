"""Every Skills route that refuses binds the code registry endpoint it serves, and every exemplar
each door declares goes out at the one status the registry gives its code on that door's endpoint.
"""

from __future__ import annotations

import pytest

from dfo_platform import codes
from dfo_platform.codes import bound_registry_endpoint
from skills_api.app import skills_surface
from skills_api.doors.fetch import build_fetch_door
from skills_api.doors.list import build_list_door
from skills_api.doors.resource import build_resource_door
from tests.code_registry import problem_statuses, registered_endpoints, unbound_routes

# Routes that serve no registry endpoint: they refuse only with platform codes.
_UNBOUND_PATHS = frozenset({"/api/skills/healthz"})


def test_every_skills_route_binds_the_registry_endpoint_it_serves():
    """The surface binds every Skills endpoint the registry maps, each to a route it serves, and
    leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = skills_surface(skills_index=object())

    assert set(surface.registry_endpoints.values()) == registered_endpoints("skills.")
    assert {path for _, path in unbound_routes(surface)} == _UNBOUND_PATHS


def test_every_skills_exemplar_goes_out_at_its_registered_status():
    """Each door's exemplars, classified with the door's own endpoint bound, go out at the one
    status the registry gives their code there."""
    disagreements: set[tuple[str, str, str, int]] = set()
    for door in (build_list_door(), build_fetch_door(), build_resource_door()):
        (endpoint,) = door.endpoint_exemplars
        for exemplar in door.endpoint_exemplars[endpoint]:
            with bound_registry_endpoint(endpoint):
                spec = door.classify(exemplar)
            assert spec is not None, repr(exemplar)
            if problem_statuses(spec.code, endpoint) != {spec.status}:
                disagreements.add((type(exemplar).__qualname__, endpoint, spec.code, spec.status))
    assert sorted(disagreements) == []


def test_a_registry_that_drops_a_raised_code_fails_the_surface_build(
    monkeypatch: pytest.MonkeyPatch,
):
    real = codes._problem_statuses
    monkeypatch.setattr(
        codes, "_problem_statuses", lambda code: {} if code == "skill_not_found" else real(code)
    )

    with pytest.raises(LookupError):
        skills_surface(skills_index=object())
