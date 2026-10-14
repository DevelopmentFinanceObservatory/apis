"""Checks which responses the composed app gzips, and that streams pass through chunk by chunk."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from dfo_platform.app import COMMIT_HEADER, SERVER_TIMING_HEADER, Surface, create_app
from dfo_platform.static_response import dynamic_public_json, encode_json_body
from dfo_platform.trace import TRACE_ID_HEADER

_FIRST_CHUNK = b"a,b\n" * 1024
_LAST_CHUNK = b"1,2\n"
_ETAG_PAYLOAD = {"rows": ["value"] * 1000}


def _body(size: int) -> bytes:
    """Return a compressible JSON string document of exactly `size` bytes."""
    return b'"' + b"a" * (size - 2) + b'"'


def _make_app(*, cors_origins: tuple[str, ...] = ()) -> FastAPI:
    router = APIRouter()

    @router.get("/api/t/sized/{size}")
    async def sized(size: int, media_type: str = "application/json"):
        return Response(content=_body(size), media_type=media_type)

    @router.get("/api/t/etag")
    async def etag(request: Request):
        return dynamic_public_json(_ETAG_PAYLOAD, request)

    @router.get("/api/t/stream")
    async def stream(request: Request, media_type: str):
        async def _chunks():
            yield _FIRST_CHUNK
            # The route produces its last chunk only once the client holds the first.
            await request.app.state.delivered.wait()
            yield _LAST_CHUNK

        return StreamingResponse(_chunks(), media_type=media_type)

    @router.get("/api/t/late-events")
    async def late_events(request: Request):
        async def _events():
            # The route produces its first event only once the client holds the headers.
            await request.app.state.delivered.wait()
            yield _LAST_CHUNK

        return StreamingResponse(_events(), media_type="text/event-stream")

    @router.get("/api/t/two-vary-lines")
    async def two_vary_lines():
        response = Response(content=_body(20_000), media_type="application/json")
        response.headers.append("Vary", "Accept-Language")
        response.headers.append("Vary", "X-Custom")
        return response

    surface = Surface(name="t", prefix="/api/t", router=router, capability_version="v-t", codes={})
    return create_app([surface], cors_origins=list(cors_origins))


@pytest.mark.parametrize(
    ("size", "media_type", "accept_encoding", "encoded"),
    [
        pytest.param(20_000, "application/json", "gzip", True, id="json"),
        pytest.param(20_000, "application/json; charset=utf-8", "gzip, br", True, id="json-params"),
        pytest.param(20_000, "application/problem+json", "gzip", True, id="json-suffix"),
        pytest.param(20_000, "text/csv; charset=utf-8", "gzip", True, id="csv"),
        pytest.param(1024, "application/json", "gzip", True, id="at-minimum-size"),
        pytest.param(1023, "application/json", "gzip", False, id="below-minimum-size"),
        pytest.param(20_000, "application/json", "identity", False, id="identity-client"),
        pytest.param(20_000, "application/json", "gzip;q=0, br", False, id="gzip-refused"),
        pytest.param(20_000, "application/vnd.apache.parquet", "gzip", False, id="parquet"),
        pytest.param(20_000, "application/zip", "gzip", False, id="zip"),
    ],
)
def test_only_large_json_and_csv_bodies_are_gzipped_for_a_gzip_client(
    size: int, media_type: str, accept_encoding: str, encoded: bool
):
    client = TestClient(_make_app())

    resp = client.get(
        f"/api/t/sized/{size}",
        params={"media_type": media_type},
        headers={"accept-encoding": accept_encoding},
    )

    assert resp.status_code == 200
    # httpx decodes a gzip body, so equality holds whichever encoding crossed the wire.
    assert resp.content == _body(size)
    assert resp.headers["content-type"] == media_type
    if encoded:
        assert resp.headers["content-encoding"] == "gzip"
        assert int(resp.headers["content-length"]) < size
    else:
        assert "content-encoding" not in resp.headers
        assert int(resp.headers["content-length"]) == size


@pytest.mark.parametrize(
    ("size", "accept_encoding", "varies"),
    [
        pytest.param(20_000, "gzip", True, id="encoded"),
        pytest.param(20_000, "identity", True, id="eligible-identity-client"),
        pytest.param(1023, "gzip", False, id="below-minimum-size"),
    ],
)
def test_vary_names_accept_encoding_on_every_body_that_gzip_could_change(
    size: int, accept_encoding: str, varies: bool
):
    client = TestClient(_make_app())

    resp = client.get(f"/api/t/sized/{size}", headers={"accept-encoding": accept_encoding})

    vary = [token.strip() for token in ",".join(resp.headers.get_list("vary")).split(",")]
    assert ("Accept-Encoding" in vary) is varies


def test_encoded_response_keeps_the_cors_and_trace_headers():
    origin = "https://app.example.com"
    client = TestClient(_make_app(cors_origins=(origin,)))

    resp = client.get("/api/t/sized/20000", headers={"accept-encoding": "gzip", "origin": origin})

    assert resp.headers["content-encoding"] == "gzip"
    vary = [token.strip() for token in ",".join(resp.headers.get_list("vary")).split(",")]
    assert sorted(vary) == ["Accept-Encoding", "Origin"]
    assert resp.headers["access-control-allow-origin"] == origin
    assert TRACE_ID_HEADER in resp.headers
    assert COMMIT_HEADER in resp.headers
    assert SERVER_TIMING_HEADER in resp.headers


def test_encoded_response_weakens_the_route_s_etag_and_keeps_its_vary():
    client = TestClient(_make_app())
    _, strong = encode_json_body(_ETAG_PAYLOAD)

    identity = client.get("/api/t/etag", headers={"accept-encoding": "identity"})
    encoded = client.get("/api/t/etag", headers={"accept-encoding": "gzip"})

    assert encoded.headers["content-encoding"] == "gzip"
    assert encoded.json() == _ETAG_PAYLOAD
    assert identity.headers["etag"] == strong
    assert encoded.headers["etag"] == f"W/{strong}"
    # The route already names `Accept-Encoding`, so the header gains no second entry.
    assert encoded.headers.get_list("vary") == identity.headers.get_list("vary")


@pytest.mark.parametrize("weak", [True, False], ids=["weak", "strong"])
@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_revalidation_answers_304_with_the_form_of_the_etag_the_client_sent(
    accept_encoding: str, weak: bool
):
    client = TestClient(_make_app())
    _, strong = encode_json_body(_ETAG_PAYLOAD)
    etag = f"W/{strong}" if weak else strong

    resp = client.get(
        "/api/t/etag", headers={"accept-encoding": accept_encoding, "if-none-match": etag}
    )

    assert resp.status_code == 304
    assert resp.headers["etag"] == etag


def test_encoded_response_without_an_etag_gains_none():
    client = TestClient(_make_app())

    resp = client.get("/api/t/sized/20000", headers={"accept-encoding": "gzip"})

    assert resp.headers["content-encoding"] == "gzip"
    assert "etag" not in resp.headers


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_every_vary_value_shares_one_line_that_survives_cors(accept_encoding: str):
    origin = "https://app.example.com"
    client = TestClient(_make_app(cors_origins=(origin,)))

    resp = client.get(
        "/api/t/two-vary-lines", headers={"accept-encoding": accept_encoding, "origin": origin}
    )

    assert resp.headers["access-control-allow-origin"] == origin
    (line,) = resp.headers.get_list("vary")
    vary = [value.strip() for value in line.split(",")]
    assert sorted(vary) == ["Accept-Encoding", "Accept-Language", "Origin", "X-Custom"]


def _stream(path: str, *, query_string: str = "", delivered: Callable[[dict], bool]) -> list[dict]:
    """Return the messages the composed app sends for a gzip client's `GET`.

    The route waits on `app.state.delivered`, which is set once `delivered` matches a
    message the client has received. A layer that held that message back stalls the
    response until the timeout."""
    app = _make_app()
    messages: list[dict] = []

    async def run() -> None:
        app.state.delivered = asyncio.Event()

        async def receive() -> dict:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(message: dict) -> None:
            messages.append(message)
            if delivered(message):
                app.state.delivered.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query_string.encode(),
            "headers": [(b"host", b"testserver"), (b"accept-encoding", b"gzip")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=5)

    asyncio.run(run())
    return messages


def test_event_stream_headers_reach_the_client_before_the_first_event_is_produced():
    messages = _stream(
        "/api/t/late-events",
        delivered=lambda message: message["type"] == "http.response.start",
    )

    assert messages[0]["type"] == "http.response.start"
    assert [message["body"] for message in messages[1:] if message.get("body")] == [_LAST_CHUNK]


@pytest.mark.parametrize("media_type", ["text/event-stream", "text/csv"])
def test_streamed_response_reaches_the_client_chunk_by_chunk_and_unencoded(media_type: str):
    messages = _stream(
        "/api/t/stream",
        query_string=f"media_type={media_type}",
        delivered=lambda message: message.get("body") == _FIRST_CHUNK,
    )

    start = messages[0]
    assert start["type"] == "http.response.start"
    assert start["status"] == 200
    assert b"content-encoding" not in {name.lower() for name, _ in start["headers"]}
    bodies = [message["body"] for message in messages[1:] if message.get("body")]
    assert bodies == [_FIRST_CHUNK, _LAST_CHUNK]
