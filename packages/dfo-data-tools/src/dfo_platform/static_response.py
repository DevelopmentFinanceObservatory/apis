"""Serve public JSON with ETags, either pre-encoded or generated per request.

`freeze_public_json` is for process-wide, unauthenticated content. Use `dynamic_public_json`
for changing payloads. Both support conditional requests and vary on encoding and origin.
`gapped_public_json` serves a payload whose gaps are permanent as public and one with a
transient gap as `no-store`.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse


def _q(parts: list[str]) -> float:
    """Return the coding's q-value, defaulting to 1.0 when absent or invalid."""
    for param in parts[1:]:
        key, _, value = param.strip().partition("=")
        if key.strip().lower() == "q":
            try:
                return float(value.strip())
            except ValueError:
                return 1.0
    return 1.0


def accepts_gzip(header: str) -> bool:
    """Use a `gzip`/`x-gzip` q-value, then `*`.

    Missing entries mean 0."""
    explicit_q: float | None = None
    star_q: float | None = None
    for token in header.split(","):
        parts = token.split(";")
        coding = parts[0].strip().lower()
        if coding in ("gzip", "x-gzip"):
            explicit_q = _q(parts)
        elif coding == "*":
            star_q = _q(parts)
    effective_q = explicit_q if explicit_q is not None else star_q if star_q is not None else 0.0
    return effective_q > 0


def _strong_etag(body: bytes) -> str:
    """Return a quoted strong ETag using the first 16 sha256 hex characters (64 bits)."""
    digest = hashlib.sha256(body).hexdigest()[:16]
    return f'"{digest}"'


def _conditional_response(
    request: Request,
    *,
    body: bytes,
    etag: str,
    cache_control: str,
    content_encoding: str | None,
) -> Response:
    """Serve a representation with conditional request handling.

    Return 304 for matching `If-None-Match` tokens, including `*`. Otherwise return 200.
    The ETag check accepts weak `W/` entries. `Vary` includes `Accept-Encoding` because gzip
    changes the representation and `Origin` because CORS headers vary by request. Build a fresh
    `Response` each time because Starlette's CORS middleware mutates headers.
    """
    headers = {
        "ETag": etag,
        "Cache-Control": cache_control,
        "Vary": "Accept-Encoding, Origin",
    }

    # The fixed-length quoted ETag matches as a token, including inside weak `W/` entries.
    # A bare `*` matches because this response always has a current representation.
    if_none_match = ", ".join(request.headers.getlist("if-none-match"))
    tokens = [token.strip() for token in if_none_match.split(",")]
    if "*" in tokens or etag in if_none_match:
        return Response(status_code=304, headers=headers)

    if content_encoding is not None:
        headers["Content-Encoding"] = content_encoding
    return Response(content=body, media_type="application/json", headers=headers)


@dataclass(frozen=True, slots=True)
class FrozenJSON:
    """Pre-encoded representations for one JSON document.

    `gzip` and `etag_gzip` are set when compression saves bytes. Otherwise `respond` uses
    `identity`."""

    identity: bytes
    gzip: bytes | None
    etag_identity: str
    etag_gzip: str | None
    cache_control: str

    def __post_init__(self) -> None:
        """Require `gzip` and `etag_gzip` to be both set or both absent."""
        if (self.gzip is None) != (self.etag_gzip is None):
            raise ValueError("gzip and etag_gzip must both be None or both be set")

    def respond(self, request: Request) -> Response:
        """Serve the selected representation, or return 304 for its matching ETag.

        Select gzip before checking `If-None-Match`, so gzip clients use its ETag. When
        `Accept-Encoding` names neither supported encoding, serve identity under RFC 9110's
        available-representation rule."""
        accept_encoding = ", ".join(request.headers.getlist("accept-encoding"))
        use_gzip = self.gzip is not None and accepts_gzip(accept_encoding)
        body = self.gzip if use_gzip else self.identity
        etag = self.etag_gzip if use_gzip else self.etag_identity
        assert body is not None
        assert etag is not None

        return _conditional_response(
            request,
            body=body,
            etag=etag,
            cache_control=self.cache_control,
            content_encoding="gzip" if use_gzip else None,
        )


def freeze_public_json(payload: Any, *, max_age: int) -> FrozenJSON:
    """Encode `payload` once and prepare public response representations.

    Keep gzip only when smaller than identity. `mtime=0` gives identical gzip bytes and ETags
    across processes serving the same payload."""
    identity = bytes(JSONResponse(payload).body)
    gzip_candidate = gzip.compress(identity, compresslevel=6, mtime=0)
    gzip_body = gzip_candidate if len(gzip_candidate) < len(identity) else None
    return FrozenJSON(
        identity=identity,
        gzip=gzip_body,
        etag_identity=_strong_etag(identity),
        etag_gzip=_strong_etag(gzip_body) if gzip_body is not None else None,
        cache_control=f"public, max-age={max_age}",
    )


_DEFAULT_DYNAMIC_CACHE_CONTROL = "public, no-cache"


def encode_json_body(payload: Any) -> tuple[bytes, str]:
    """Return identity-encoded JSON bytes and their ETag for reuse with `serve_encoded_json`."""
    body = bytes(JSONResponse(payload).body)
    return body, _strong_etag(body)


def serve_encoded_json(
    request: Request,
    *,
    body: bytes,
    etag: str,
    cache_control: str,
) -> Response:
    """Serve identity-encoded JSON with ETag checks and `Vary` headers.

    `Content-Encoding` is omitted. `body` and `etag` can come from `encode_json_body`."""
    return _conditional_response(
        request,
        body=body,
        etag=etag,
        cache_control=cache_control,
        content_encoding=None,
    )


def dynamic_public_json(
    payload: Any,
    request: Request,
    *,
    cache_control: str | None = None,
) -> Response:
    """Encode a changing payload per request and serve it with a strong ETag.

    The body leaves here identity-encoded, and the `ETag` covers those identity bytes.
    `dfo_platform.compression` gzips the body for a client that accepts gzip and sends that
    `ETag` as a weak validator (`W/` prefix), on the gzip body and on the 304 that answers
    it. `cache_control` is caller-defined. The default `public, no-cache` permits storage but
    requires revalidation before reuse."""
    body, etag = encode_json_body(payload)
    resolved_cache_control = (
        cache_control if cache_control is not None else _DEFAULT_DYNAMIC_CACHE_CONTROL
    )
    return serve_encoded_json(
        request,
        body=body,
        etag=etag,
        cache_control=resolved_cache_control,
    )


NO_STORE = "no-store"


def gapped_public_json(
    payload: Any,
    request: Request,
    *,
    transient_gap: bool,
    max_age: int = 300,
) -> Response:
    """Serve a payload that may carry gaps, choosing its cache policy from where they came from.

    The caller sets `transient_gap` when any gap in `payload` comes from a failed read or an
    unloaded index. Such a payload answers `no-store` with no `ETag`, so a shared cache never
    holds a transient fault. Otherwise every gap is permanent, and the payload answers
    `public, max-age=<max_age>` with a strong `ETag` over the identity body, honouring
    `If-None-Match` with 304."""
    if transient_gap:
        return JSONResponse(payload, headers={"Cache-Control": NO_STORE})
    return dynamic_public_json(payload, request, cache_control=f"public, max-age={max_age}")
