"""Every request logs the caller's claimed identity from its Cloud Run token and omits the token."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse

from dfo_platform import caller
from dfo_platform.app import Surface, _trace_id_dispatch, create_app
from dfo_platform.caller import CallerIdentity, caller_identity_from_headers
from dfo_platform.cloud_logging import JsonFormatter
from dfo_platform.trace import get_cloud_trace_id, get_trace_id

_EMAIL = "website@dev-fin-observatory.iam.gserviceaccount.com"
_AUD = "https://dfo-dev-apis-abc123-nn.a.run.app"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _jwt(claims: object, *, signature: str = "") -> str:
    """A token as the container sees it, with the signature Cloud Run has already stripped."""
    header = _b64(b'{"alg":"RS256","typ":"JWT"}')
    return f"{header}.{_b64(json.dumps(claims).encode())}.{signature}"


_TOKEN = _jwt({"email": _EMAIL, "aud": _AUD, "sub": "1"})
_PRESENT = CallerIdentity(token_present=True)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, CallerIdentity(token_present=False)),
        ({"X-Serverless-Authorization": ""}, CallerIdentity(token_present=False)),
        (
            {"X-Serverless-Authorization": f"Bearer {_TOKEN}"},
            CallerIdentity(token_present=True, email=_EMAIL, aud=_AUD),
        ),
        (
            {"Authorization": f"bearer {_TOKEN}"},
            CallerIdentity(token_present=True, email=_EMAIL, aud=_AUD),
        ),
        (
            {
                "X-Serverless-Authorization": f"Bearer {_jwt({'email': 'a@b', 'aud': 'x'})}",
                "Authorization": f"Bearer {_TOKEN}",
            },
            CallerIdentity(token_present=True, email="a@b", aud="x"),
        ),
        (
            {"X-Serverless-Authorization": "", "Authorization": f"Bearer {_TOKEN}"},
            CallerIdentity(token_present=True, email=_EMAIL, aud=_AUD),
        ),
        (
            {"Authorization": f"Bearer {_jwt({'email': _EMAIL}, signature='c2ln')}"},
            CallerIdentity(token_present=True, email=_EMAIL),
        ),
        ({"Authorization": f"Bearer {_jwt({'email': 7, 'aud': ['x', 'y']})}"}, _PRESENT),
        ({"Authorization": f"Bearer {_jwt([1, 2])}"}, _PRESENT),
        ({"Authorization": f"Bearer {_jwt('text')}"}, _PRESENT),
        ({"Authorization": f"Bearer h.{_b64(b'not json')}."}, _PRESENT),
        ({"Authorization": f"Bearer h.{_b64(b'\xff\xfe')}."}, _PRESENT),
        ({"Authorization": "Bearer h.!!!!."}, _PRESENT),
        ({"Authorization": "Bearer a.é."}, _PRESENT),
        ({"Authorization": f"Bearer a.{_b64(b'[' * 10_000)}.s"}, _PRESENT),
        ({"Authorization": f"Bearer {_jwt({'email': _EMAIL, 'pad': 'a' * 5000})}"}, _PRESENT),
        ({"Authorization": "Bearer not-a-jwt"}, _PRESENT),
        ({"Authorization": "Bearer"}, _PRESENT),
        ({"Authorization": "Basic dXNlcjpwYXNz"}, _PRESENT),
        ({"Authorization": f"Basic {_TOKEN}"}, _PRESENT),
    ],
    ids=[
        "no-header",
        "empty-header",
        "serverless-header",
        "authorization-fallback",
        "serverless-preferred",
        "empty-serverless-falls-back",
        "signature-stripped-or-kept",
        "claims-of-the-wrong-type",
        "payload-is-a-list",
        "payload-is-a-string",
        "payload-is-not-json",
        "payload-is-not-utf8",
        "payload-is-not-base64",
        "payload-is-not-ascii",
        "payload-is-nested-past-the-recursion-limit",
        "credential-is-over-length",
        "not-a-jwt",
        "scheme-only",
        "basic-credentials",
        "non-bearer-scheme-with-a-jwt",
    ],
)
def test_the_caller_identity_comes_from_the_unverified_token(headers, expected):
    assert caller_identity_from_headers(Headers(headers)) == expected


def test_a_recursion_error_from_the_json_parser_yields_no_claims(monkeypatch):
    def too_deep(_):
        raise RecursionError

    monkeypatch.setattr(caller.json, "loads", too_deep)

    identity = caller_identity_from_headers(Headers({"Authorization": f"Bearer {_TOKEN}"}))

    assert identity == _PRESENT


def test_an_oversized_claim_is_clipped():
    token = _jwt({"email": "a" * 1_000})

    identity = caller_identity_from_headers(Headers({"Authorization": f"Bearer {token}"}))

    assert identity.email is not None
    assert len(identity.email) == 256


@pytest.fixture
def request_lines():
    stream = io.StringIO()
    log = logging.getLogger("dfo_platform.request")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(project="p-1"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    log.removeHandler(handler)
    log.setLevel(logging.NOTSET)
    log.propagate = True


def _client() -> TestClient:
    router = APIRouter()

    @router.get("/api/t/ok")
    async def ok():
        return {}

    @router.get("/api/t/boom")
    async def boom():
        raise RuntimeError("unhandled")

    surface = Surface(name="t", prefix="/api/t", router=router, capability_version="v-t", codes={})
    return TestClient(create_app([surface]), raise_server_exceptions=False)


def test_a_request_logs_one_line_with_the_callers_claimed_identity(request_lines):
    _client().get(
        "/api/t/ok?q=secret",
        headers={"X-Serverless-Authorization": f"Bearer {_TOKEN}", "User-Agent": "dfo-website/1"},
    )

    (entry,) = request_lines()
    assert entry["logger"] == "dfo_platform.request"
    assert entry["severity"] == "INFO"
    assert {k: v for k, v in entry.items() if k.startswith(("http_", "caller_", "user_"))} == {
        "http_method": "GET",
        "http_path": "/api/t/ok",
        "http_status": 200,
        "user_agent": "dfo-website/1",
        "caller_token_present": True,
        "caller_email": _EMAIL,
        "caller_aud": _AUD,
    }


@pytest.mark.parametrize(
    ("credential", "secret"),
    [
        (_jwt({"email": _EMAIL}, signature="c2lnbmF0dXJl"), "c2lnbmF0dXJl"),
        ("opaque-credential-value", "opaque-credential-value"),
        ("h.!!!!.sig-segment", "sig-segment"),
    ],
    ids=["signed", "opaque", "undecodable"],
)
def test_the_token_never_reaches_the_log(request_lines, credential, secret):
    _client().get("/api/t/ok", headers={"Authorization": f"Bearer {credential}"})

    output = json.dumps(request_lines())
    assert secret not in output
    assert credential not in output


def test_a_request_without_a_token_logs_it_absent(request_lines):
    _client().get("/api/t/ok")

    (entry,) = request_lines()
    assert entry["caller_token_present"] is False
    assert entry["caller_email"] is None
    assert entry["caller_aud"] is None


def test_a_malformed_header_logs_present_with_no_claims_and_still_serves(request_lines):
    response = _client().get("/api/t/ok", headers={"Authorization": "Bearer h.!!!!."})

    assert response.status_code == 200
    (entry,) = request_lines()
    assert entry["caller_token_present"] is True
    assert entry["caller_email"] is None
    assert entry["caller_aud"] is None


def test_an_unhandled_error_still_logs_the_request(request_lines):
    response = _client().get("/api/t/boom", headers={"Authorization": f"Bearer {_TOKEN}"})

    assert response.status_code == 500
    (entry,) = request_lines()
    assert entry["http_status"] == 500
    assert entry["caller_email"] == _EMAIL


def test_a_deeply_nested_token_still_serves_the_response(request_lines):
    nested = f"a.{_b64(b'[' * 10_000)}.s"

    response = _client().get("/api/t/ok", headers={"Authorization": f"Bearer {nested}"})

    assert response.status_code == 200
    (entry,) = request_lines()
    assert entry["caller_token_present"] is True
    assert entry["caller_email"] is None


def test_a_failing_request_log_neither_replaces_the_response_nor_skips_the_reset(
    monkeypatch, caplog
):
    secret = "secret-credential-value"

    def failing_log_request(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr("dfo_platform.app.log_request", failing_log_request)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/t/ok",
            "query_string": b"",
            "headers": [
                (b"authorization", f"Bearer {secret}".encode()),
                (b"x-cloud-trace-context", b"105445aa7843bc8bf206b12000100000/1;o=1"),
            ],
        }
    )
    seen: dict[str, str | None] = {}

    async def call_next(_):
        seen["trace_id"] = get_trace_id()
        seen["cloud_trace_id"] = get_cloud_trace_id()
        return JSONResponse({})

    async def dispatch():
        response = await _trace_id_dispatch(request, call_next, commit="c")
        return response, get_trace_id(), get_cloud_trace_id()

    with caplog.at_level(logging.WARNING, logger="dfo_platform.app"):
        response, trace_id_after, cloud_trace_id_after = asyncio.run(dispatch())

    assert response.status_code == 200
    assert seen["cloud_trace_id"] is not None
    assert trace_id_after != seen["trace_id"]
    assert cloud_trace_id_after is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert secret not in caplog.text
