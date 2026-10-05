"""Checks the HSTS header on success, refusal, 500, preflight, and cacheable static responses."""

from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import HSTS_HEADER, HSTS_VALUE, Surface, create_app
from dfo_platform.problem import ProblemSpec


class _RefusedError(Exception):
    """A surface-classified exception type, dispatched via ExceptionMiddleware."""


def _make_app(*, cors_origins: tuple[str, ...] = ()) -> object:
    router = APIRouter()

    @router.get("/api/t/ping")
    async def ping():
        return {"ok": True}

    @router.get("/api/t/boom")
    async def boom():
        raise RuntimeError("unhandled")

    @router.get("/api/t/refuse")
    async def refuse():
        raise _RefusedError("handled")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _RefusedError):
            return ProblemSpec(status=409, code="refused", retryable=False)
        return None

    surface = Surface(
        name="t",
        prefix="/api/t",
        router=router,
        capability_version="v-t",
        codes={"refused": ("Refused", "detail")},
        classify=classify,
        exception_types=(_RefusedError,),
    )
    return create_app([surface], cors_origins=list(cors_origins))


def test_success_response_carries_the_hsts_header():
    client = TestClient(_make_app())

    resp = client.get("/api/t/ping")

    assert resp.status_code == 200
    assert resp.headers[HSTS_HEADER] == HSTS_VALUE


def test_surface_classified_refusal_carries_the_hsts_header():
    client = TestClient(_make_app(), raise_server_exceptions=False)

    resp = client.get("/api/t/refuse")

    assert resp.status_code == 409
    assert resp.headers[HSTS_HEADER] == HSTS_VALUE


def test_catchall_500_carries_the_hsts_header():
    client = TestClient(_make_app(), raise_server_exceptions=False)

    resp = client.get("/api/t/boom")

    assert resp.status_code == 500
    assert resp.headers[HSTS_HEADER] == HSTS_VALUE


def test_cors_preflight_carries_the_hsts_header():
    client = TestClient(_make_app(cors_origins=("https://app.example.com",)))

    resp = client.options(
        "/api/t/ping",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert resp.status_code == 200
    assert resp.headers[HSTS_HEADER] == HSTS_VALUE


@pytest.fixture
def _static_route_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    from dfo_apis.app import create_app as create_deployed_app
    from tests.engine._search_app import with_fake_search_deps

    return TestClient(with_fake_search_deps(create_deployed_app()), raise_server_exceptions=False)


def test_shared_cacheable_static_route_still_carries_the_hsts_header(_static_route_client):
    resp = _static_route_client.get("/api/search/schema", headers={"accept-encoding": "identity"})

    assert resp.status_code == 200
    assert resp.headers.get("cache-control", "").find("public") != -1
    assert resp.headers[HSTS_HEADER] == HSTS_VALUE
