"""Tests surface and platform refusal dispatch, including classifier fallthrough and CORS headers."""

from __future__ import annotations

import logging

import pytest
from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from dfo_platform import refusal
from dfo_platform.app import Surface, create_app
from dfo_platform.problem import ProblemSpec
from dfo_platform.trace import TRACE_ID_HEADER
from dfo_platform.version import PLATFORM_CAPABILITY_VERSION


class _ClassifiedError(Exception):
    """An exception a dummy surface's own `classify` recognizes."""


class _UnclaimedError(Exception):
    """An exception no surface's own `classify` recognizes."""


def test_surface_classify_returning_a_spec_renders_it_via_envelope_render():
    router = APIRouter()

    @router.get("/api/a/classified")
    async def classified():
        raise _ClassifiedError("boom")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ClassifiedError):
            return ProblemSpec(status=409, code="classified_code", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"classified_code": ("Classified", "classified detail")},
        classify=classify,
        exception_types=(_ClassifiedError,),
        exception_exemplars=(_ClassifiedError("exemplar"),),
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/classified")
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "classified_code"
    assert body["capability_version"] == "va"
    assert resp.headers["cache-control"] == "no-store"


def test_surface_classify_returning_none_falls_through_to_the_platform_default():
    router = APIRouter()

    @router.get("/api/a/unclaimed")
    async def unclaimed():
        raise _UnclaimedError("boom")

    def classify(_exc: BaseException) -> ProblemSpec | None:
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={},
        classify=classify,
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/unclaimed")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["capability_version"] == "va"


def test_surface_classify_that_raises_falls_through_to_the_platform_default(caplog):
    router = APIRouter()

    @router.get("/api/a/unclaimed")
    async def unclaimed():
        raise _UnclaimedError("boom")

    def classify(_exc: BaseException) -> ProblemSpec | None:
        raise RuntimeError("classify itself is broken")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={},
        classify=classify,
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="dfo_platform.app"):
        resp = client.get("/api/a/unclaimed")

    # The response uses the platform default; the classifier error is logged.
    assert resp.status_code == 500
    assert resp.json()["code"] == "internal_error"
    assert "classify()" in caplog.text
    assert "_UnclaimedError" in caplog.text


def test_surface_classify_returning_an_unregistered_code_falls_back_to_the_platform_default(
    caplog,
):
    """An unregistered classifier code falls back to `internal_error` and logs its source."""
    router = APIRouter()

    @router.get("/api/a/classified")
    async def classified():
        raise _ClassifiedError("boom")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ClassifiedError):
            return ProblemSpec(status=409, code="someone_elses_code", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={},  # "someone_elses_code" is deliberately not registered here
        classify=classify,
        exception_types=(_ClassifiedError,),
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="dfo_platform.app"):
        resp = client.get("/api/a/classified")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    # The platform envelope supplies the capability version.
    assert body["capability_version"] == PLATFORM_CAPABILITY_VERSION
    assert "'a'" in caplog.text
    assert "someone_elses_code" in caplog.text
    assert "_ClassifiedError" in caplog.text


def test_classify_recognizing_a_type_absent_from_its_own_exception_types_logs_a_warning(caplog):
    """An undeclared classified type renders and emits a registration warning."""
    router = APIRouter()

    @router.get("/api/a/classified")
    async def classified():
        raise _ClassifiedError("boom")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ClassifiedError):
            return ProblemSpec(status=409, code="classified_code", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"classified_code": ("Classified", "classified detail")},
        classify=classify,
        exception_types=(),  # deliberately omits _ClassifiedError
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.WARNING, logger="dfo_platform.app"):
        resp = client.get("/api/a/classified")

    assert resp.status_code == 409
    assert resp.json()["code"] == "classified_code"
    assert "'a'" in caplog.text
    assert "_ClassifiedError" in caplog.text


def test_surface_classified_error_response_carries_cors_headers():
    """Surface-classified refusals carry CORS headers."""
    router = APIRouter()

    @router.get("/api/a/classified")
    async def classified():
        raise _ClassifiedError("boom")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ClassifiedError):
            return ProblemSpec(status=409, code="classified_code", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"classified_code": ("Classified", "classified detail")},
        classify=classify,
        exception_types=(_ClassifiedError,),
    )
    app = create_app([surface], cors_origins=["https://app.example.com"])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/classified", headers={"Origin": "https://app.example.com"})

    assert resp.status_code == 409
    assert resp.json()["code"] == "classified_code"
    assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert resp.headers.get_list("access-control-allow-origin") == ["https://app.example.com"]
    assert resp.headers.get_list("vary") == ["Origin"]


def test_unclassified_error_response_carries_cors_headers():
    """The platform catch-all 500 carries CORS headers and the trace ID."""
    router = APIRouter()

    @router.get("/api/a/unclaimed")
    async def unclaimed():
        raise _UnclaimedError("boom")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={},
    )
    app = create_app([surface], cors_origins=["https://app.example.com"])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/unclaimed", headers={"Origin": "https://app.example.com"})

    assert resp.status_code == 500
    assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert TRACE_ID_HEADER in resp.headers.get("access-control-expose-headers", "")
    trace_id = resp.headers[TRACE_ID_HEADER]
    assert resp.json()["trace_id"] == trace_id


def test_server_error_middleware_still_reraises_the_bare_exception():
    """CORS middleware preserves `ServerErrorMiddleware`'s exception re-raise."""
    router = APIRouter()

    @router.get("/api/a/unclaimed")
    async def unclaimed():
        raise _UnclaimedError("boom")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={},
    )
    app = create_app([surface], cors_origins=["https://app.example.com"])
    client = TestClient(app)  # raise_server_exceptions=True (the default)

    with pytest.raises(_UnclaimedError):
        client.get("/api/a/unclaimed", headers={"Origin": "https://app.example.com"})


def test_cors_headers_are_not_duplicated_on_any_response():
    """Each response carries one CORS origin value and at most one `Origin` entry in `Vary`."""
    router = APIRouter()

    @router.get("/api/a/ok")
    async def ok():
        return {"status": "fine"}

    @router.get("/api/a/classified")
    async def classified():
        raise _ClassifiedError("boom")

    @router.get("/api/a/unclaimed")
    async def unclaimed():
        raise _UnclaimedError("boom")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ClassifiedError):
            return ProblemSpec(status=409, code="classified_code", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"classified_code": ("Classified", "classified detail")},
        classify=classify,
        exception_types=(_ClassifiedError,),
    )
    app = create_app([surface], cors_origins=["https://app.example.com"])
    client = TestClient(app, raise_server_exceptions=False)

    for path, expected_status in (
        ("/api/a/ok", 200),
        ("/api/a/classified", 409),
        ("/api/a/unclaimed", 500),
    ):
        resp = client.get(path, headers={"Origin": "https://app.example.com"})
        assert resp.status_code == expected_status
        assert resp.headers.get_list("access-control-allow-origin") == ["https://app.example.com"]
        vary_values = resp.headers.get_list("vary")
        assert sum(1 for v in vary_values if "Origin" in v) <= 1

    preflight = client.options(
        "/api/a/ok",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get_list("access-control-allow-origin") == ["https://app.example.com"]


def test_no_cors_origins_configured_adds_no_cors_headers_and_no_wildcard():
    """An empty CORS allowlist adds no CORS headers or wildcard fallback."""
    router = APIRouter()

    @router.get("/api/a/unclaimed")
    async def unclaimed():
        raise _UnclaimedError("boom")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={},
    )
    app = create_app([surface])  # cors_origins defaults to ()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/unclaimed", headers={"Origin": "https://app.example.com"})

    assert resp.status_code == 500
    assert "access-control-allow-origin" not in resp.headers


def test_classify_default_maps_request_validation_error():
    exc = RequestValidationError(
        [{"loc": ("body", "field"), "msg": "Field required", "type": "missing"}]
    )
    spec = refusal.classify_default(exc)
    assert spec.status == 400
    assert spec.code == "request_invalid"
    assert spec.retryable is False


def test_classify_default_maps_asyncio_timeout_error():
    spec = refusal.classify_default(TimeoutError("slow"))
    assert spec.status == 504
    assert spec.code == "timeout"
    assert spec.retryable is True


@pytest.mark.parametrize(
    ("status", "code"),
    [(404, "not_found"), (405, "method_not_allowed")],
)
def test_classify_default_maps_starlette_http_exception(status, code):
    exc = StarletteHTTPException(
        status_code=status, headers={"Allow": "GET"} if status == 405 else None
    )
    spec = refusal.classify_default(exc)
    assert spec.status == status
    assert spec.code == code
    assert spec.retryable is False
    if status == 405:
        assert spec.headers == {"Allow": "GET"}


def test_classify_default_maps_a_bare_exception_to_the_unhandled_default():
    spec = refusal.classify_default(RuntimeError("anything else"))
    assert spec.status == 500
    assert spec.code == "internal_error"
    assert spec.retryable is False
