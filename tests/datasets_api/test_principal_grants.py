"""Every route that needs a grant refuses each configured principal lacking it with 403, before
the request body is read. Three distinct secrets are configured, one per grant, so each route
sees two principals that lack its grant, plus a request naming none."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import Request
from starlette.types import Message

from datasets_api.config import DatasetsBuildSettings
from datasets_api.principals import Grant, Principals
from dfo_platform.problem import Refusal
from spec_registry import LiveSpecRegistry

from ._app_factories import make_app
from .conftest import FakeDatasetStore

_HEADER = "x-dfo-principal"
_SECRETS = {"publish": "publish-secret", "pin": "pin-secret", "refresh": "refresh-secret"}

# Each gated route, with the grant it needs and the code its refusal carries. The pin route
# admits every caller and reads the pin grant only to choose its budget (`test_pin_door`).
_GATED_ROUTES = (
    ("POST", "/api/datasets/frames/publish", "publish", "publish_not_permitted"),
    ("PUT", "/api/datasets/frames/ops_a/ocv_a/recipe", "publish", "publish_not_permitted"),
    ("GET", "/api/datasets/frames/ops_a/ocv_a/recipe", "publish", "publish_not_permitted"),
    ("GET", "/api/datasets/recipes/replays/rk_a", "publish", "publish_not_permitted"),
    ("POST", "/api/datasets/frames/refresh", "refresh", "refresh_not_permitted"),
)


def _cases() -> list[tuple[str, str, str, str, str | None]]:
    cases: list[tuple[str, str, str, str, str | None]] = []
    for method, path, grant, code in _GATED_ROUTES:
        lacking = [secret for held, secret in _SECRETS.items() if held != grant]
        cases.extend((method, path, code, grant, secret) for secret in [*lacking, None])
    return cases


@pytest.fixture(scope="module")
def app():
    settings = DatasetsBuildSettings(
        principal_header=_HEADER,
        first_party_principal=_SECRETS["publish"],
        pin_first_party_principal=_SECRETS["pin"],
        live_refresh_principal=_SECRETS["refresh"],
    )
    return make_app(registry=LiveSpecRegistry(store=FakeDatasetStore()), settings=settings)


def _call(app, *, method: str, path: str, principal: str | None) -> tuple[int, dict, int]:
    """Send one request whose body stream raises when read. Returns the status, the JSON body,
    and how many times the app pulled the request body."""
    reads = 0
    headers = [(b"content-type", b"application/json"), (b"content-length", b"2")]
    if principal is not None:
        headers.append((_HEADER.encode(), principal.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {},
    }
    sent: list[Message] = []

    async def receive() -> Message:
        nonlocal reads
        reads += 1
        raise AssertionError("the request body was read")

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], json.loads(body), reads


@pytest.mark.parametrize(
    ("method", "path", "code", "grant", "principal"),
    _cases(),
    ids=[f"{m} {p} {g} as {s or 'none'}" for m, p, _c, g, s in _cases()],
)
def test_a_principal_lacking_the_grant_is_refused_before_the_body_is_read(
    app, method: str, path: str, code: str, grant: str, principal: str | None
) -> None:
    status, body, reads = _call(app, method=method, path=path, principal=principal)

    assert (status, body["code"]) == (403, code)
    assert reads == 0


def test_the_table_names_every_gated_route(app) -> None:
    """A gated door gaining a route fails here until the table lists it."""
    gated_templates = {
        ("POST", "/api/datasets/frames/publish"),
        ("PUT", "/api/datasets/frames/{spec_id}/{content_version}/recipe"),
        ("GET", "/api/datasets/frames/{spec_id}/{content_version}/recipe"),
        ("GET", "/api/datasets/recipes/replays/{replay_key}"),
        ("POST", "/api/datasets/frames/refresh"),
    }
    markers = ("/recipe", "/replays/", "/frames/publish", "/frames/refresh")
    served = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        if any(marker in path for marker in markers)
        for method in operations
    }
    assert served == gated_templates
    assert len(_GATED_ROUTES) == len(gated_templates)


@pytest.mark.parametrize("grant", [Grant.publish, Grant.refresh])
def test_a_refused_principal_carries_no_secret_in_its_repr_or_refusal(grant: Grant) -> None:
    """A caller presenting another grant's secret leaves that secret on no repr or refusal a
    later log line could print."""
    held = next(other for other in Grant if other != grant)
    principals = Principals(header=_HEADER, secrets=((_SECRETS[held], held),))
    request = Request({"type": "http", "headers": [(_HEADER.encode(), _SECRETS[held].encode())]})
    principal = principals.parse(request)

    with pytest.raises(Refusal) as excinfo:
        principals.require(grant)(request)

    refusal = excinfo.value
    assert principal.grants == frozenset({held})
    assert getattr(refusal, "presented", None) is True
    for text in (repr(principal), repr(refusal), str(refusal), repr(vars(refusal))):
        assert _SECRETS[held] not in text
