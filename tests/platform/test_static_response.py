"""Tests frozen and dynamic JSON responses, including compression, ETags, and cache headers."""

from __future__ import annotations

import gzip as gzip_module
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from dfo_platform.static_response import (
    FrozenJSON,
    accepts_gzip,
    dynamic_public_json,
    encode_json_body,
    freeze_public_json,
    gapped_public_json,
    serve_encoded_json,
)

_COMPRESSIBLE_PAYLOAD = {"items": ["repeat-me"] * 500}
_TINY_PAYLOAD = {"a": 1}


def _make_app() -> FastAPI:
    compressible = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=3600)
    tiny = freeze_public_json(_TINY_PAYLOAD, max_age=60)

    app = FastAPI()

    @app.get("/compressible")
    async def get_compressible(request: Request) -> object:
        return compressible.respond(request)

    @app.get("/tiny")
    async def get_tiny(request: Request) -> object:
        return tiny.respond(request)

    return app


def _client() -> TestClient:
    return TestClient(_make_app())


def _make_dynamic_app(current: list[Any]) -> FastAPI:
    """Builds a route that reads the payload from a mutable list on each request."""
    app = FastAPI()

    @app.get("/dynamic")
    async def get_dynamic(request: Request) -> object:
        return dynamic_public_json(current[0], request)

    return app


def _dynamic_client(payload: Any) -> tuple[TestClient, list[Any]]:
    current = [payload]
    return TestClient(_make_dynamic_app(current)), current


def test_identity_bytes_match_jsonresponse_body():
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    assert frozen.identity == JSONResponse(_COMPRESSIBLE_PAYLOAD).body


def test_gzip_representation_decodes_to_identity_bytes():
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    assert frozen.gzip is not None
    assert gzip_module.decompress(frozen.gzip) == frozen.identity


def test_etags_are_distinct_and_stable_across_freeze_invocations():
    frozen_a = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    frozen_b = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    assert frozen_a.etag_gzip is not None
    assert frozen_a.etag_identity != frozen_a.etag_gzip
    assert frozen_a.etag_identity == frozen_b.etag_identity
    assert frozen_a.etag_gzip == frozen_b.etag_gzip


def test_304_for_identity_etag_when_client_does_not_accept_gzip():
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    client = _client()
    resp = client.get(
        "/compressible",
        headers={"if-none-match": frozen.etag_identity, "accept-encoding": "identity"},
    )
    assert resp.status_code == 304
    assert resp.content == b""


def test_if_none_match_star_yields_304():
    client = _client()
    resp = client.get(
        "/compressible",
        headers={"if-none-match": "*", "accept-encoding": "identity"},
    )
    assert resp.status_code == 304
    assert resp.content == b""


def test_304_for_gzip_etag_when_client_accepts_gzip():
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    assert frozen.etag_gzip is not None
    client = _client()
    resp = client.get(
        "/compressible",
        headers={"if-none-match": frozen.etag_gzip, "accept-encoding": "gzip"},
    )
    assert resp.status_code == 304
    assert resp.content == b""


def test_identity_etag_with_gzip_accepting_client_yields_200_gzip_not_304():
    """A gzip client compares `If-None-Match` with the gzip ETag."""
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    client = _client()
    resp = client.get(
        "/compressible",
        headers={"if-none-match": frozen.etag_identity, "accept-encoding": "gzip"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.headers["etag"] == frozen.etag_gzip
    assert resp.json() == _COMPRESSIBLE_PAYLOAD


def test_gzip_q0_yields_identity_representation():
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    assert frozen.etag_gzip is not None
    client = _client()
    resp = client.get(
        "/compressible",
        headers={"accept-encoding": "gzip;q=0", "if-none-match": frozen.etag_gzip},
    )
    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers
    assert resp.headers["etag"] == frozen.etag_identity


def test_tiny_payload_gzip_not_smaller_always_serves_identity():
    frozen = freeze_public_json(_TINY_PAYLOAD, max_age=60)
    assert frozen.gzip is None
    assert frozen.etag_gzip is None

    client = _client()
    resp = client.get("/tiny", headers={"accept-encoding": "gzip"})
    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers
    assert resp.headers["etag"] == frozen.etag_identity


def test_headers_present_on_200():
    client = _client()
    resp = client.get("/compressible", headers={"accept-encoding": "identity"})
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=3600"
    assert resp.headers["vary"] == "Accept-Encoding, Origin"
    assert "etag" in resp.headers
    assert "content-encoding" not in resp.headers


def test_headers_present_on_304():
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    client = _client()
    resp = client.get(
        "/compressible",
        headers={"if-none-match": frozen.etag_identity, "accept-encoding": "identity"},
    )
    assert resp.status_code == 304
    assert resp.headers["etag"] == frozen.etag_identity
    assert resp.headers["cache-control"] == "public, max-age=3600"
    assert resp.headers["vary"] == "Accept-Encoding, Origin"


def test_sequential_requests_do_not_accumulate_headers():
    """Each call creates a fresh response; CORS appends one `Origin` token to its `Vary` header."""
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["https://example.org"])
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)

    @app.get("/doc")
    async def get_doc(request: Request) -> object:
        return frozen.respond(request)

    client = TestClient(app)
    headers = {"origin": "https://example.org", "accept-encoding": "identity"}
    first = client.get("/doc", headers=headers)
    second = client.get("/doc", headers=headers)

    assert first.headers["vary"] == second.headers["vary"] == "Accept-Encoding, Origin, Origin"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("gzip", True),
        ("x-gzip", True),
        ("gzip;q=0", False),
        ("gzip;q=0.5", True),
        ("br, gzip ; q=1.0", True),
        ("identity", False),
        ("", False),
        ("*;q=1, identity;q=0", True),
        ("*;q=0", False),
        ("*", True),
        ("gzip;q=0, *;q=1", False),
    ],
)
def test_accepts_gzip(header: str, expected: bool):
    assert accepts_gzip(header) is expected


def test_second_if_none_match_line_participates_in_the_conditional_check():
    """A matching ETag on a later `If-None-Match` line yields 304."""
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    client = _client()
    resp = client.get(
        "/compressible",
        headers=[
            ("if-none-match", '"does-not-match"'),
            ("if-none-match", frozen.etag_identity),
            ("accept-encoding", "identity"),
        ],
    )
    assert resp.status_code == 304
    assert resp.content == b""


def test_second_accept_encoding_line_participates_in_negotiation():
    """A `gzip` token on a later `Accept-Encoding` line selects gzip."""
    frozen = freeze_public_json(_COMPRESSIBLE_PAYLOAD, max_age=60)
    client = _client()
    resp = client.get(
        "/compressible",
        headers=[
            ("accept-encoding", "br"),
            ("accept-encoding", "gzip"),
        ],
    )
    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.headers["etag"] == frozen.etag_gzip


def test_mismatched_gzip_pairing_raises_value_error():
    with pytest.raises(ValueError, match="gzip and etag_gzip"):
        FrozenJSON(
            identity=b"{}",
            gzip=b"whatever",
            etag_identity='"abc"',
            etag_gzip=None,
            cache_control="public, max-age=60",
        )


# -- dynamic_public_json ----------------------------------------------------


def test_dynamic_200_carries_a_strong_etag():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    resp = client.get("/dynamic")
    assert resp.status_code == 200
    assert resp.json() == _TINY_PAYLOAD
    assert resp.headers["etag"].startswith('"') and resp.headers["etag"].endswith('"')


def test_dynamic_matching_if_none_match_yields_304_with_no_body():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    etag = client.get("/dynamic").headers["etag"]
    resp = client.get("/dynamic", headers={"if-none-match": etag})
    assert resp.status_code == 304
    assert resp.content == b""


def test_dynamic_if_none_match_star_yields_304():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    resp = client.get("/dynamic", headers={"if-none-match": "*"})
    assert resp.status_code == 304
    assert resp.content == b""


def test_dynamic_non_matching_if_none_match_yields_200():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    resp = client.get("/dynamic", headers={"if-none-match": '"does-not-match"'})
    assert resp.status_code == 200
    assert resp.json() == _TINY_PAYLOAD


def test_dynamic_different_payloads_yield_different_etags():
    client, current = _dynamic_client({"a": 1})
    first_etag = client.get("/dynamic").headers["etag"]
    current[0] = {"a": 2}
    second_etag = client.get("/dynamic").headers["etag"]
    assert first_etag != second_etag


def test_dynamic_same_payload_yields_same_etag_across_calls():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    first_etag = client.get("/dynamic").headers["etag"]
    second_etag = client.get("/dynamic").headers["etag"]
    assert first_etag == second_etag


def test_dynamic_vary_header():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    resp = client.get("/dynamic")
    assert resp.headers["vary"] == "Accept-Encoding, Origin"


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity", "gzip, br", ""])
def test_dynamic_never_sets_content_encoding(accept_encoding: str):
    client, _ = _dynamic_client(_COMPRESSIBLE_PAYLOAD)
    resp = client.get("/dynamic", headers={"accept-encoding": accept_encoding})
    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers
    assert resp.json() == _COMPRESSIBLE_PAYLOAD


def test_dynamic_default_cache_control():
    client, _ = _dynamic_client(_TINY_PAYLOAD)
    resp = client.get("/dynamic")
    assert resp.headers["cache-control"] == "public, no-cache"


def test_dynamic_caller_supplied_cache_control():
    app = FastAPI()

    @app.get("/dynamic")
    async def get_dynamic(request: Request) -> object:
        return dynamic_public_json(_TINY_PAYLOAD, request, cache_control="public, max-age=5")

    resp = TestClient(app).get("/dynamic")
    assert resp.headers["cache-control"] == "public, max-age=5"


# -- encode_json_body / serve_encoded_json ----------------------------------
#
# The dynamic helper composes body encoding and response serving.


def test_encode_json_body_matches_jsonresponse_body():
    body, _etag = encode_json_body(_TINY_PAYLOAD)
    assert body == bytes(JSONResponse(_TINY_PAYLOAD).body)


def test_encode_json_body_etag_is_strong_and_stable():
    first_body, first_etag = encode_json_body(_TINY_PAYLOAD)
    second_body, second_etag = encode_json_body(_TINY_PAYLOAD)
    assert first_body == second_body
    assert first_etag == second_etag
    assert first_etag.startswith('"') and first_etag.endswith('"')


def test_encode_json_body_different_payloads_yield_different_etags():
    _, first_etag = encode_json_body({"a": 1})
    _, second_etag = encode_json_body({"a": 2})
    assert first_etag != second_etag


def _serve_encoded_client(
    body: bytes, etag: str, *, cache_control: str = "public, max-age=60"
) -> TestClient:
    app = FastAPI()

    @app.get("/encoded")
    async def get_encoded(request: Request) -> object:
        return serve_encoded_json(request, body=body, etag=etag, cache_control=cache_control)

    return TestClient(app)


def test_serve_encoded_json_200_serves_the_given_body_and_etag():
    body, etag = encode_json_body(_TINY_PAYLOAD)
    resp = _serve_encoded_client(body, etag).get("/encoded")
    assert resp.status_code == 200
    assert resp.json() == _TINY_PAYLOAD
    assert resp.headers["etag"] == etag
    assert resp.headers["cache-control"] == "public, max-age=60"


def test_serve_encoded_json_matching_if_none_match_yields_304():
    body, etag = encode_json_body(_TINY_PAYLOAD)
    resp = _serve_encoded_client(body, etag).get("/encoded", headers={"if-none-match": etag})
    assert resp.status_code == 304
    assert resp.content == b""


def test_serve_encoded_json_star_yields_304():
    body, etag = encode_json_body(_TINY_PAYLOAD)
    resp = _serve_encoded_client(body, etag).get("/encoded", headers={"if-none-match": "*"})
    assert resp.status_code == 304


def test_serve_encoded_json_never_sets_content_encoding():
    body, etag = encode_json_body(_COMPRESSIBLE_PAYLOAD)
    resp = _serve_encoded_client(body, etag).get("/encoded", headers={"accept-encoding": "gzip"})
    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers


def test_dynamic_public_json_is_encode_json_body_then_serve_encoded_json():
    """The helper response matches sequential body encoding and response serving."""
    app = FastAPI()

    @app.get("/dynamic")
    async def get_dynamic(request: Request) -> object:
        return dynamic_public_json(_TINY_PAYLOAD, request, cache_control="public, max-age=42")

    resp = TestClient(app).get("/dynamic")

    body, etag = encode_json_body(_TINY_PAYLOAD)
    assert resp.content == body
    assert resp.headers["etag"] == etag
    assert resp.headers["cache-control"] == "public, max-age=42"


def _gapped_client(*, transient_gap: bool) -> TestClient:
    app = FastAPI()

    @app.get("/gapped")
    async def get_gapped(request: Request) -> object:
        return gapped_public_json(_TINY_PAYLOAD, request, transient_gap=transient_gap)

    return TestClient(app)


def test_gapped_permanent_gaps_serve_public_with_an_etag_over_the_body():
    resp = _gapped_client(transient_gap=False).get("/gapped")

    body, etag = encode_json_body(_TINY_PAYLOAD)
    assert resp.status_code == 200
    assert resp.content == body
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert resp.headers["etag"] == etag


def test_gapped_permanent_gaps_honour_if_none_match():
    client = _gapped_client(transient_gap=False)
    etag = client.get("/gapped").headers["etag"]

    resp = client.get("/gapped", headers={"if-none-match": etag})

    assert resp.status_code == 304
    assert resp.content == b""


def test_gapped_transient_gap_serves_no_store_without_an_etag():
    client = _gapped_client(transient_gap=True)
    _body, etag = encode_json_body(_TINY_PAYLOAD)

    resp = client.get("/gapped", headers={"if-none-match": etag})

    assert resp.status_code == 200
    assert resp.json() == _TINY_PAYLOAD
    assert resp.headers["cache-control"] == "no-store"
    assert "etag" not in resp.headers
