"""Problem bodies carry the trace ID the endpoint saw before raising."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import Surface, create_app
from dfo_platform.problem import ProblemSpec
from dfo_platform.trace import get_trace_id


class _RefusedError(Exception):
    """A surface-classified exception type, dispatched via ExceptionMiddleware."""


def _make_app_and_seen() -> tuple:
    router = APIRouter()
    seen: dict[str, str] = {}

    @router.get("/api/t/boom")
    async def boom():
        seen["unhandled"] = get_trace_id()
        raise RuntimeError("unhandled")

    @router.get("/api/t/refuse")
    async def refuse():
        seen["refused"] = get_trace_id()
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
    app = create_app([surface])
    return app, seen


def test_catchall_500_body_carries_the_same_trace_id_the_endpoint_saw():
    app, seen = _make_app_and_seen()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/t/boom")

    assert resp.status_code == 500
    assert resp.json()["trace_id"] == seen["unhandled"]


def test_handled_refusal_body_carries_the_same_trace_id_the_endpoint_saw():
    app, seen = _make_app_and_seen()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/t/refuse")

    assert resp.status_code == 409
    assert resp.json()["trace_id"] == seen["refused"]
