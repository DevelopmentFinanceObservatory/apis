"""Every Presentation route that serves a code registry endpoint binds it, and every exemplar goes
out at the status the registry gives its code on the endpoint whose routes raise it: a render
door exemplar on `presentation.render`, a shares door exemplar on `presentation.shares`, and every
other exemplar on `presentation.write`.

The registry maps no endpoint onto Presentation's read routes. They share the write routes'
classifiers, so they serve the write routes' statuses.
"""

from __future__ import annotations

from dfo_spec_kit.registry import REGISTRY

from assembler.errors import FrameUnavailable
from assembler.fakes import FakeReleaseRegistry
from dfo_platform.codes import bound_registry_endpoint
from presentation_api.app import presentation_surface
from presentation_api.doors.render import RENDER_EXCEPTION_EXEMPLARS
from presentation_api.doors.shares import SHARES_EXCEPTION_EXEMPLARS
from presentation_api.refusals import _FRAME_UNAVAILABLE_OUTCOME

_WRITE = "presentation.write"
_RENDER = "presentation.render"
_SHARES = "presentation.shares"

# Codes of the write routes alone. No render route raises them, and every other code a non-share
# exemplar classifies to must be mapped on `presentation.render`, whose routes restate it.
_WRITE_ONLY_CODES = frozenset({"write_refused", "allowance_exhausted"})

# Routes that serve no registry endpoint. The component read shares the revise route's path,
# so it is bound with it.
_UNBOUND_PATHS = frozenset(
    {
        "/api/presentation/healthz",
        "/api/presentation/capabilities",
        "/api/presentation/release",
        "/api/presentation/frames/{spec_id}",
        "/api/presentation/frames/{spec_id}/{content_version}",
        "/api/presentation/components/{component_id}/revisions/{revision}/data",
    }
)


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


def test_every_presentation_route_binds_the_registry_endpoint_it_serves(
    release_registry: FakeReleaseRegistry,
):
    """The surface binds every Presentation endpoint the registry maps, each to a route it
    serves, and leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = presentation_surface(release_registry=release_registry)
    registered = {
        mapping.endpoint
        for entry in REGISTRY.entries
        for mapping in entry.mappings
        if mapping.endpoint.startswith("presentation.")
    }
    routed = {getattr(route, "path", None) for route in surface.router.routes}

    assert set(surface.registry_endpoints.values()) == registered
    assert routed - set(surface.registry_endpoints) == _UNBOUND_PATHS


def test_every_presentation_exemplar_goes_out_at_its_registered_status(
    release_registry: FakeReleaseRegistry,
):
    """Each exemplar, classified with its routes' endpoint bound, goes out at the one status the
    registry gives its code there, and its code is mapped on that endpoint. So does every
    `FrameUnavailable` reason, on the write and share routes."""
    surface = presentation_surface(release_registry=release_registry)
    render_exemplars = {id(exemplar) for exemplar in RENDER_EXCEPTION_EXEMPLARS}
    share_exemplars = {id(exemplar) for exemplar in SHARES_EXCEPTION_EXEMPLARS}
    reasons = [
        FrameUnavailable(reason=reason, detail="exemplar") for reason in _FRAME_UNAVAILABLE_OUTCOME
    ]
    bound: list[tuple[str, BaseException]] = [
        *((_WRITE, exemplar) for exemplar in reasons),
        *((_SHARES, exemplar) for exemplar in reasons),
    ]
    for exemplar in surface.exception_exemplars:
        if id(exemplar) in render_exemplars:
            bound.append((_RENDER, exemplar))
        elif id(exemplar) in share_exemplars:
            bound.append((_SHARES, exemplar))
        else:
            bound.append((_WRITE, exemplar))
    disagreements: set[tuple[str, str, str, int]] = set()
    unmapped: set[tuple[str, str, str]] = set()
    unmapped_on_render: set[str] = set()
    checked: set[str] = set()
    for endpoint, exemplar in bound:
        with bound_registry_endpoint(endpoint):
            spec = surface.classify(exemplar)
        assert spec is not None, repr(exemplar)
        checked.add(endpoint)
        statuses = _statuses(spec.code, endpoint)
        if not statuses:
            unmapped.add((type(exemplar).__qualname__, endpoint, spec.code))
        elif statuses != {spec.status}:
            disagreements.add((type(exemplar).__qualname__, endpoint, spec.code, spec.status))
        if endpoint == _WRITE and not _statuses(spec.code, _RENDER):
            unmapped_on_render.add(spec.code)
    assert checked == {_WRITE, _RENDER, _SHARES}
    assert sorted(disagreements) == []
    assert sorted(unmapped) == []
    assert unmapped_on_render == _WRITE_ONLY_CODES
