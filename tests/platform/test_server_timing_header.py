"""Checks the platform timing fallback, preservation of custom headers, and stream/cache exclusions."""

from __future__ import annotations

import pytest
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from dfo_platform.app import SERVER_TIMING_HEADER, Surface, create_app
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

    @router.get("/api/t/own-timing")
    async def own_timing():
        return Response(
            content="{}",
            media_type="application/json",
            headers={SERVER_TIMING_HEADER: "total;dur=1.0, own;dur=0.5"},
        )

    @router.get("/api/t/public")
    async def public():
        return Response(
            content="{}", media_type="application/json", headers={"Cache-Control": "public"}
        )

    @router.get("/api/t/stream")
    async def stream():
        async def _body():
            yield b"data: hello\n\n"

        return StreamingResponse(_body(), media_type="text/event-stream")

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


def test_success_response_gets_total_timing():
    client = TestClient(_make_app())

    resp = client.get("/api/t/ping")

    assert resp.status_code == 200
    assert "total;dur=" in resp.headers[SERVER_TIMING_HEADER]


def test_surface_classified_refusal_gets_total_timing():
    client = TestClient(_make_app(), raise_server_exceptions=False)

    resp = client.get("/api/t/refuse")

    assert resp.status_code == 409
    assert "total;dur=" in resp.headers[SERVER_TIMING_HEADER]


def test_catchall_500_gets_total_timing():
    client = TestClient(_make_app(), raise_server_exceptions=False)

    resp = client.get("/api/t/boom")

    assert resp.status_code == 500
    assert "total;dur=" in resp.headers[SERVER_TIMING_HEADER]


def test_response_with_its_own_server_timing_is_left_byte_identical():
    client = TestClient(_make_app())

    resp = client.get("/api/t/own-timing")

    assert resp.status_code == 200
    assert resp.headers[SERVER_TIMING_HEADER] == "total;dur=1.0, own;dur=0.5"


def test_sse_response_carries_no_server_timing():
    client = TestClient(_make_app())

    with client.stream("GET", "/api/t/stream") as resp:
        assert resp.status_code == 200
        assert SERVER_TIMING_HEADER not in resp.headers


def test_cache_control_public_response_carries_no_server_timing():
    client = TestClient(_make_app())

    resp = client.get("/api/t/public")

    assert resp.status_code == 200
    assert SERVER_TIMING_HEADER not in resp.headers


@pytest.fixture
def _static_route_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DFO_DATASETS_SURFACE", "1")
    monkeypatch.delenv("DFO_GRAPH_SURFACE", raising=False)
    monkeypatch.delenv("DFO_PRESENTATION_SURFACE", raising=False)
    from dfo_apis.app import create_app as create_deployed_app
    from tests.engine._search_app import with_fake_search_deps

    return TestClient(with_fake_search_deps(create_deployed_app()), raise_server_exceptions=False)


def test_shared_cacheable_static_route_carries_no_server_timing(_static_route_client):
    resp = _static_route_client.get("/api/search/schema", headers={"accept-encoding": "identity"})

    assert resp.status_code == 200
    assert resp.headers.get("cache-control", "").find("public") != -1
    assert SERVER_TIMING_HEADER not in resp.headers
