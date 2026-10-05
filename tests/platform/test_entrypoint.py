"""Checks Docker and Just use the same app factory and that its routes match mounted surfaces."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dfo_apis.app import create_app, deployed_surfaces
from dfo_platform.trace import TRACE_ID_HEADER
from tests.engine._search_app import with_fake_search_deps

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACTORY_RE = re.compile(r"--factory\s+(\S+)")

# FastAPI adds these routes to every app.
_BUILTIN_DOC_PATHS = frozenset({"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"})

# The platform registers this route on every composed app.
_PLATFORM_ROUTE_PATHS = frozenset({"/codes/{code}"})


def _expected_route_paths() -> frozenset[str]:
    """Returns mounted surface paths plus the platform registry path."""
    surface_paths = {
        route.path
        for surface in deployed_surfaces()
        for route in surface.router.routes
        if getattr(route, "path", None) is not None
    }
    return frozenset(surface_paths | _PLATFORM_ROUTE_PATHS)


def _factory_string(text: str) -> str:
    match = _FACTORY_RE.search(text)
    assert match is not None, "no --factory argument found"
    return match.group(1)


def _dockerfile_factory() -> str:
    return _factory_string((_REPO_ROOT / "Dockerfile").read_text())


def _justfile_dev_factory() -> str:
    return _factory_string((_REPO_ROOT / "justfile").read_text())


def _route_paths(app: FastAPI) -> set[str]:
    """Returns public OpenAPI paths, excluding FastAPI's documentation routes."""
    return {path for path in app.openapi()["paths"] if path not in _BUILTIN_DOC_PATHS}


def test_dockerfile_and_justfile_dev_target_name_the_same_factory():
    assert _dockerfile_factory() == _justfile_dev_factory()


def test_factory_resolves_to_the_expected_route_set():
    module_path, _, attr = _dockerfile_factory().partition(":")
    assert module_path and attr, f"malformed --factory string: {_dockerfile_factory()!r}"

    import importlib

    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    app = factory()
    assert isinstance(app, FastAPI)
    assert _route_paths(app) == _expected_route_paths()


def test_deployed_app_honors_dfo_cors_origins_on_a_real_preflight(monkeypatch):
    """Verifies CORS end-to-end through the actually-deployed composition
    (dfo_apis.app.create_app), which is what applies parse_cors_origins()
    app-wide in production -- distinct from search's own standalone factory
    (search.engine.app.create_app), which tests/engine/test_app.py covers
    separately.
    """
    monkeypatch.setenv("DFO_CORS_ORIGINS", "https://app.example.com")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.options(
        "/api/search/resolve",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_deployed_app_preflight_carries_trace_id_header(monkeypatch):
    """`_trace_id_dispatch` wraps `CORSMiddleware` from outside (see
    `dfo_platform.app._install_trace_id_outermost_middleware_stack`), so it
    runs -- and stamps `x-dfo-trace-id` -- even for a preflight, which
    `CORSMiddleware` itself answers directly without calling further into
    the stack it wraps.
    """
    monkeypatch.setenv("DFO_CORS_ORIGINS", "https://app.example.com")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.options(
        "/api/search/resolve",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert TRACE_ID_HEADER in resp.headers


def test_deployed_app_rejected_preflight_carries_trace_id_header(monkeypatch):
    """A preflight `CORSMiddleware` rejects -- a disallowed origin or a
    disallowed method -- is exactly the response someone debugging a CORS
    misconfiguration is looking at, and it carries a real diagnostic body
    (`CORSMiddleware`'s own plain-text reason), so it needs correlating
    just as much as an allowed preflight or an ordinary response does.
    """
    monkeypatch.setenv("DFO_CORS_ORIGINS", "https://app.example.com")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    disallowed_origin = client.options(
        "/api/search/resolve",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert disallowed_origin.status_code == 400
    assert TRACE_ID_HEADER in disallowed_origin.headers

    disallowed_method = client.options(
        "/api/search/resolve",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert disallowed_method.status_code == 400
    assert TRACE_ID_HEADER in disallowed_method.headers


def test_deployed_app_carries_cors_headers_on_a_bare_exception_500(monkeypatch):
    """The same guarantee as `test_unclassified_error_response_carries_cors_headers`
    in `tests/platform/test_classify_dispatch.py`, but through the actually-
    deployed composition (`dfo_apis.app.create_app`) rather than a dummy
    surface -- the real call site every other CORS test in this module
    covers for the happy and preflight paths. `FrozenJSON.respond` is
    patched to raise a plain `RuntimeError`, a type absent from
    `classify_search_exception`'s own recognized types and from Search's
    declared `exception_types`, so `GET /api/search/capabilities` takes the
    true bare-Exception path dispatched by `ServerErrorMiddleware`.
    """
    import dfo_platform.static_response

    def _raise(*_args: object, **_kwargs: object):
        raise RuntimeError("boom")

    monkeypatch.setattr(dfo_platform.static_response.FrozenJSON, "respond", _raise)
    monkeypatch.setenv("DFO_CORS_ORIGINS", "https://app.example.com")
    app = with_fake_search_deps(create_app())
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/search/capabilities", headers={"Origin": "https://app.example.com"})

    assert resp.status_code == 500
    assert resp.headers.get_list("access-control-allow-origin") == ["https://app.example.com"]


def test_deployed_app_with_no_cors_origins_configured_adds_no_cors_headers(monkeypatch):
    """`dfo_apis.app.create_app` passes `cors_origins=parse_cors_origins()`
    straight through to `dfo_platform.app.create_app` -- an unset
    `DFO_CORS_ORIGINS` yields an empty list there, and an empty
    `cors_origins` must add no CORS headers at all, with no wildcard
    fallback, on the bare-Exception 500 path just as on any other. Same
    guarantee as `test_no_cors_origins_configured_adds_no_cors_headers_and_no_wildcard`
    in `tests/platform/test_classify_dispatch.py`, but through the actually-
    deployed composition rather than a hand-built `Surface`. `FrozenJSON.respond`
    is patched to raise a plain `RuntimeError`, mirroring
    `test_deployed_app_carries_cors_headers_on_a_bare_exception_500` above, so
    `GET /api/search/capabilities` takes the same bare-Exception path.
    """
    monkeypatch.delenv("DFO_CORS_ORIGINS", raising=False)
    import dfo_platform.static_response

    def _raise(*_args: object, **_kwargs: object):
        raise RuntimeError("boom")

    monkeypatch.setattr(dfo_platform.static_response.FrozenJSON, "respond", _raise)
    app = with_fake_search_deps(create_app())
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/search/capabilities", headers={"Origin": "https://app.example.com"})

    assert resp.status_code == 500
    assert "access-control-allow-origin" not in resp.headers
