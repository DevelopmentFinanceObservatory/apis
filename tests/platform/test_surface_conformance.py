"""Checks codes, classifications, and refusal bodies for every composed surface.
The app is built without entering its lifespan, so these checks need no live clients.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from datasets_api.codes import SHARED_PLATFORM_CODES as _DATASETS_SHARED_PLATFORM_CODES
from dfo_apis.app import all_surfaces
from dfo_platform.app import Surface, create_app
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, bound_registry_endpoint, merge_catalogues
from dfo_platform.problem import ProblemDetails, ProblemSpec
from presentation_api.codes import (
    PRESENTATION_SERVED_PLATFORM_CODES as _PRESENTATION_SHARED_PLATFORM_CODES,
)
from presentation_api.codes import (
    SHARE_SERVED_PLATFORM_CODES as _SHARE_SHARED_PLATFORM_CODES,
)
from search.engine.codes import SHARED_PLATFORM_CODES as _SEARCH_SHARED_PLATFORM_CODES
from skills_api.codes import SHARED_PLATFORM_CODES as _SKILLS_SHARED_PLATFORM_CODES

_SURFACES = all_surfaces()
_REGISTRY = merge_catalogues(PLATFORM_CODE_CATALOGUE, *(s.codes for s in _SURFACES))
_SURFACE_IDS = [s.name for s in _SURFACES]

# Codes each surface declares that it emits from the shared platform catalogue.
_DECLARED_SHARED_PLATFORM_CODES: dict[str, frozenset[str]] = {
    "search": _SEARCH_SHARED_PLATFORM_CODES,
    "datasets": _DATASETS_SHARED_PLATFORM_CODES,
    "presentation": _PRESENTATION_SHARED_PLATFORM_CODES | _SHARE_SHARED_PLATFORM_CODES,
    "skills": _SKILLS_SHARED_PLATFORM_CODES,
}


def _registered_codes_for(surface: Surface) -> frozenset[str]:
    """Return the surface catalogue plus its declared shared platform codes."""
    declared_shared = _DECLARED_SHARED_PLATFORM_CODES.get(surface.name)
    if declared_shared is None:
        return frozenset(_REGISTRY)
    return frozenset(surface.codes) | declared_shared


def _probe_path(surface: Surface, name: str) -> str:
    return f"{surface.prefix}/_conformance_probe_{name}"


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_app(all_surfaces())


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def app_with_probes() -> FastAPI:
    """Add prefixed refusal probes before surface routes to avoid fallback shadowing."""
    surfaces = all_surfaces()
    built = create_app(surfaces)

    async def _raise_validation():
        raise RequestValidationError(
            [{"loc": ("body", "probe"), "msg": "conformance probe", "type": "value_error"}]
        )

    async def _raise_timeout():
        raise TimeoutError("conformance probe timeout")

    async def _raise_unhandled():
        raise RuntimeError("conformance probe unhandled error")

    probe_router = APIRouter()
    for surface in surfaces:
        probe_router.add_api_route(
            _probe_path(surface, "validation"), _raise_validation, methods=["GET"]
        )
        probe_router.add_api_route(_probe_path(surface, "timeout"), _raise_timeout, methods=["GET"])
        probe_router.add_api_route(
            _probe_path(surface, "unhandled"), _raise_unhandled, methods=["GET"]
        )
    built.router.routes[0:0] = probe_router.routes
    return built


@pytest.fixture
def probe_client(app_with_probes: FastAPI) -> TestClient:
    return TestClient(app_with_probes, raise_server_exceptions=False)


def _assert_conforming_problem_body(resp, *, surface: Surface) -> None:
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.headers["cache-control"] == "no-store"
    problem = ProblemDetails.model_validate(resp.json())
    assert problem.code in _REGISTRY
    assert isinstance(problem.trace_id, str) and len(problem.trace_id) == 32
    int(problem.trace_id, 16)  # genuinely hex, not merely 32 characters
    assert problem.capability_version == surface.capability_version


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_every_surface_code_resolves_through_the_codes_endpoint(
    client: TestClient, surface: Surface
):
    for code in surface.codes:
        resp = client.get(f"/codes/{code}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"]
        assert body["detail"]


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_validation_refusal_conforms_for_its_own_surface(
    probe_client: TestClient, surface: Surface
):
    resp = probe_client.get(_probe_path(surface, "validation"))
    _assert_conforming_problem_body(resp, surface=surface)


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_timeout_refusal_conforms_for_its_own_surface(probe_client: TestClient, surface: Surface):
    resp = probe_client.get(_probe_path(surface, "timeout"))
    assert resp.status_code == 504
    _assert_conforming_problem_body(resp, surface=surface)


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_unhandled_refusal_conforms_for_its_own_surface(probe_client: TestClient, surface: Surface):
    resp = probe_client.get(_probe_path(surface, "unhandled"))
    assert resp.status_code == 500
    _assert_conforming_problem_body(resp, surface=surface)


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_routing_404_conforms_for_its_own_surface(client: TestClient, surface: Surface):
    """A two-segment unmatched path exercises the routing 404 under each prefix.

    One segment can match Skills' `/{skill_id}` route and return 503 without a lifespan.
    """
    resp = client.get(f"{surface.prefix}/_conformance_probe_nonexistent/route")
    assert resp.status_code == 404
    _assert_conforming_problem_body(resp, surface=surface)


def _route_missing_delete(surface: Surface) -> tuple[str, set[str]]:
    """Return a non-templated route and its allowed methods for a 405 probe."""
    for route in surface.router.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or path is None or "{" in path:
            continue
        if "DELETE" not in methods:
            return path, set(methods)
    pytest.fail(
        f"surface {surface.name!r} has no real, non-templated route usable for a 405 "
        "probe -- every route either allows DELETE or is templated"
    )


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_routing_405_conforms_for_its_own_surface(client: TestClient, surface: Surface):
    """The 405 `Allow` header matches the selected route's methods."""
    path, allowed_methods = _route_missing_delete(surface)
    resp = client.delete(path)
    assert resp.status_code == 405
    assert "allow" in resp.headers
    # Starlette preserves set iteration order, so compare methods as a set.
    assert {m.strip() for m in resp.headers["allow"].split(",")} == allowed_methods
    _assert_conforming_problem_body(resp, surface=surface)


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_every_exemplar_type_is_declared_in_exception_types(surface: Surface):
    """Each classified exemplar type appears in the surface's registered exception types."""
    declared = set(surface.exception_types)
    for exemplar in surface.exception_exemplars:
        exemplar_type = type(exemplar)
        assert any(issubclass(exemplar_type, declared_type) for declared_type in declared), (
            f"surface {surface.name!r} declares an exemplar of type "
            f"{exemplar_type.__qualname__}, absent from its own exception_types"
        )


def _reachable_exception_types(
    func: Callable[[BaseException], object], *, _seen: set[object] | None = None
) -> set[type[BaseException]]:
    """Find exception types recognized by `isinstance` checks without running the function.

    Follows composed-classifier closures and plain module-level calls recursively.
    Method, attribute, and dispatch-table indirection is outside this static scan.
    """
    seen = _seen if _seen is not None else set()
    if func in seen:
        return set()
    seen.add(func)
    types: set[type[BaseException]] = set()

    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None)
    if code is not None and closure is not None and "classifiers" in code.co_freevars:
        idx = code.co_freevars.index("classifiers")
        for classifier in closure[idx].cell_contents:
            types |= _reachable_exception_types(classifier, _seen=seen)
        return types

    source = inspect.getsource(func)
    tree = ast.parse(textwrap.dedent(source))
    globalns = getattr(func, "__globals__", {})
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "isinstance" and len(node.args) == 2:
            type_arg = node.args[1]
            names = type_arg.elts if isinstance(type_arg, ast.Tuple) else [type_arg]
            for name_node in names:
                if not isinstance(name_node, ast.Name):
                    continue
                candidate = globalns.get(name_node.id)
                if isinstance(candidate, type) and issubclass(candidate, BaseException):
                    types.add(candidate)
        else:
            callee = globalns.get(node.func.id)
            if inspect.isfunction(callee):
                types |= _reachable_exception_types(callee, _seen=seen)
    return types


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_classify_recognizes_no_type_absent_from_exception_types(surface: Surface):
    """Every statically recognized exception type has a registered base class.

    Unlike exemplar checks, this also catches types absent from both exemplar tuples.
    """
    found = _reachable_exception_types(surface.classify)
    declared = set(surface.exception_types)
    missing = {
        exc_type
        for exc_type in found
        if not any(issubclass(exc_type, declared_type) for declared_type in declared)
    }
    assert not missing, (
        f"surface {surface.name!r}'s classify() recognizes "
        f"{sorted(t.__qualname__ for t in missing)}, absent from its own "
        "declared exception_types"
    )


class _NeverDeclaredError(Exception):
    """An exception type no real surface declares in its own `exception_types`
    -- exercises `classify`'s own `None` branch, the case none of a
    surface's `exception_exemplars` can cover by construction.
    """


def _classify_on_some_endpoint(surface: Surface, exemplar: BaseException) -> ProblemSpec | None:
    """`surface.classify(exemplar)` with no registry endpoint bound, or else under the first of
    the surface's endpoints whose registry status for the code resolves. The app binds the
    endpoint of the route that raised (`codes.bound_registry_endpoint`), and a code whose
    status differs by endpoint resolves only under one; which routes raise which type is each
    surface's own sweep, as `tests/datasets_api/test_refusal_registry.py` is for Datasets."""
    for endpoint in (None, *sorted(set(surface.registry_endpoints.values()))):
        with bound_registry_endpoint(endpoint):
            try:
                return surface.classify(exemplar)
            except LookupError:
                continue
    raise AssertionError(f"{exemplar!r} has a registered status on none of its surface's endpoints")


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_classify_conforms_to_the_registered_catalogue_for_every_exemplar(surface: Surface):
    """Exemplars classify to `None` or to codes declared by their surface."""
    registry = _registered_codes_for(surface)
    for exemplar in (*surface.exception_exemplars, _NeverDeclaredError("never declared")):
        spec = _classify_on_some_endpoint(surface, exemplar)
        assert spec is None or spec.code in registry


@pytest.mark.parametrize("surface", _SURFACES, ids=_SURFACE_IDS)
def test_a_classify_that_raises_still_yields_a_conforming_platform_default(surface: Surface):
    """A raising classifier falls back to the platform refusal for its surface."""

    def _raising_classify(_exc: BaseException) -> ProblemSpec | None:
        raise RuntimeError("classify blew up")

    raising_surface = dataclasses.replace(surface, classify=_raising_classify)
    app = create_app([raising_surface])

    async def _raise_unhandled():
        raise RuntimeError("conformance probe unhandled error")

    # Keep fallback routes from shadowing the probe.
    probe_router = APIRouter()
    probe_router.add_api_route(
        _probe_path(raising_surface, "raising_classify"), _raise_unhandled, methods=["GET"]
    )
    app.router.routes[0:0] = probe_router.routes
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get(_probe_path(raising_surface, "raising_classify"))
    assert resp.status_code == 500
    _assert_conforming_problem_body(resp, surface=raising_surface)
