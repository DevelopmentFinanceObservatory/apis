"""Ordinary responses carry `x-dfo-trace-id`, matching any problem body.
Static cacheable routes omit the header, as tested in `test_static_routes_headers.py`.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import Surface, create_app
from dfo_platform.problem import ProblemSpec
from dfo_platform.trace import TRACE_ID_HEADER


class _RefusedError(Exception):
    """A surface-classified exception type, dispatched via ExceptionMiddleware."""


def _make_app() -> object:
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
    return create_app([surface])


def _is_trace_id(value: str) -> bool:
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def test_success_response_carries_the_trace_id_header():
    client = TestClient(_make_app())

    resp = client.get("/api/t/ping")

    assert resp.status_code == 200
    assert _is_trace_id(resp.headers[TRACE_ID_HEADER])


def test_handled_refusal_carries_a_trace_id_header_matching_the_body():
    client = TestClient(_make_app(), raise_server_exceptions=False)

    resp = client.get("/api/t/refuse")

    assert resp.status_code == 409
    assert resp.headers[TRACE_ID_HEADER] == resp.json()["trace_id"]


def test_catchall_500_carries_a_trace_id_header_matching_the_body():
    client = TestClient(_make_app(), raise_server_exceptions=False)

    resp = client.get("/api/t/boom")

    assert resp.status_code == 500
    assert resp.headers[TRACE_ID_HEADER] == resp.json()["trace_id"]


class _UnregisteredCodeError(Exception):
    """A surface-classified exception whose spec names a code absent from
    that surface's own catalogue, forcing `_dispatch`'s `UnregisteredCode`
    fallback arm. Deliberately absent from the surface's own
    `exception_types` (see the `classify` below), the same way `boom`'s bare
    `RuntimeError` is absent from every surface's -- so it too has no
    specific handler and dispatches via Starlette's ServerErrorMiddleware,
    whose response is followed by an unconditional re-raise of the original
    exception. Exercises `_dispatch`'s own `UnregisteredCode` recovery path
    specifically, distinct from
    `test_catchall_500_carries_a_trace_id_header_matching_the_body` above."""


def test_unregistered_code_fallback_carries_a_trace_id_header_matching_the_body():
    router = APIRouter()

    @router.get("/api/t/unregistered")
    async def unregistered():
        raise _UnregisteredCodeError("boom")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _UnregisteredCodeError):
            return ProblemSpec(status=409, code="someone_elses_code", retryable=False)
        return None

    surface = Surface(
        name="t",
        prefix="/api/t",
        router=router,
        capability_version="v-t",
        codes={},  # "someone_elses_code" is deliberately not registered here
        classify=classify,
        # left at its default () -- see _UnregisteredCodeError's docstring
    )
    client = TestClient(create_app([surface]), raise_server_exceptions=False)

    resp = client.get("/api/t/unregistered")

    assert resp.status_code == 500
    assert resp.headers[TRACE_ID_HEADER] == resp.json()["trace_id"]


def test_two_requests_carry_distinct_trace_id_headers():
    client = TestClient(_make_app())

    first = client.get("/api/t/ping").headers[TRACE_ID_HEADER]
    second = client.get("/api/t/ping").headers[TRACE_ID_HEADER]

    assert first != second


def test_trace_id_header_and_server_timing_are_exposed_to_cors_clients():
    app = create_app(
        [
            Surface(
                name="t",
                prefix="/api/t",
                router=APIRouter(),
                capability_version="v-t",
                codes={},
            )
        ],
        cors_origins=["https://app.example.com"],
    )
    client = TestClient(app)
    resp = client.get("/codes/does-not-exist", headers={"Origin": "https://app.example.com"})

    exposed = resp.headers.get("access-control-expose-headers", "")
    assert TRACE_ID_HEADER in exposed
    assert "Server-Timing" in exposed
