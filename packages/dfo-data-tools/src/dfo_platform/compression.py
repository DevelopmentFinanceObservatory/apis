"""Gzip buffered JSON and CSV responses for clients that accept gzip.

A response is eligible when its whole body arrives in one ASGI message, its media type is
JSON or CSV, it carries no `Content-Encoding`, and its body is at least `MINIMUM_SIZE` bytes.
Whatever the client accepts, an eligible response leaves with one `Vary` line holding every
value the route set plus `Accept-Encoding`, so an outer layer that rewrites `Vary` from its
first line keeps them all.

A gzip body carries the route's strong `ETag` as a weak validator (`W/` prefix), because a
strong one identifies the identity bytes alone (RFC 9110 section 8.8.3.3). The identity body
keeps the strong form. A 304 carries the weak form when the request's `If-None-Match` named
it, so a client holding the gzip body sees one validator on 200 and 304.

A streamed response (`more_body` set on its first body message) passes through untouched,
so each chunk reaches the client as the route produces it. The response start of a JSON or
CSV response waits for its first body message. Every other start, a server-sent-event
stream's included, is sent as the route produces it. `BaseHTTPMiddleware` re-emits every
response as a stream, so this layer sits inside every such layer.
"""

from __future__ import annotations

import gzip
import re

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dfo_platform.static_response import accepts_gzip

COMPRESS_LEVEL = 1
"""Level 1 costs about a sixth of level 9's CPU time on these JSON bodies. Compression runs
on the event loop, so the cheap level bounds how long one large response stalls it."""

MINIMUM_SIZE = 1024
"""Bodies below this many bytes stay identity-encoded."""


def _is_compressible(content_type: str) -> bool:
    """Return whether the media type is CSV, JSON, or a `+json` structured syntax."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type in ("text/csv", "application/json") or media_type.endswith("+json")


def _is_candidate(start: Message) -> bool:
    """Return whether the response start has a compressible media type and no encoding."""
    headers = Headers(raw=start["headers"])
    return "content-encoding" not in headers and _is_compressible(headers.get("content-type", ""))


_ENTITY_TAG = re.compile(r'(?:W/)?"[^"]*"')


def _vary_on_accept_encoding(headers: MutableHeaders) -> None:
    """Fold the `Vary` lines into one that names `Accept-Encoding`."""
    lines = ",".join(headers.getlist("vary")).split(",")
    values = [value.strip() for value in lines if value.strip()]
    if not any(value.lower() == "accept-encoding" for value in values):
        values.append("Accept-Encoding")
    # Assignment replaces every `Vary` line with this one.
    headers["Vary"] = ", ".join(values)


def _weaken_etag(headers: MutableHeaders) -> None:
    """Rewrite a strong `ETag` as a weak one, leaving a weak or absent one alone."""
    etag = headers.get("etag")
    if etag and not etag.startswith("W/"):
        headers["ETag"] = f"W/{etag}"


def _match_revalidated_etag(start: Message, *, if_none_match: str) -> None:
    """Weaken a 304's strong `ETag` when `if_none_match` names its weak form.

    Mutates `start`'s headers in place."""
    headers = MutableHeaders(raw=start["headers"])
    if f"W/{headers.get('etag')}" in _ENTITY_TAG.findall(if_none_match):
        _weaken_etag(headers)


def _encode(start: Message, body_message: Message, *, client_accepts_gzip: bool) -> Message:
    """Return the body message to send for a candidate start, gzip-encoded when eligible.

    Mutates `start`'s headers in place to match the returned body."""
    body = body_message.get("body", b"")
    if len(body) < MINIMUM_SIZE:
        return body_message

    headers = MutableHeaders(raw=start["headers"])
    _vary_on_accept_encoding(headers)
    if not client_accepts_gzip:
        return body_message

    # `mtime=0` keeps the bytes identical for identical bodies.
    compressed = gzip.compress(body, compresslevel=COMPRESS_LEVEL, mtime=0)
    if len(compressed) >= len(body):
        return body_message

    _weaken_etag(headers)
    headers["Content-Encoding"] = "gzip"
    headers["Content-Length"] = str(len(compressed))
    return {**body_message, "body": compressed}


class BufferedGzipMiddleware:
    """Gzip eligible buffered responses and pass every other message through unchanged."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_headers = Headers(scope=scope)
        client_accepts_gzip = accepts_gzip(", ".join(request_headers.getlist("accept-encoding")))
        if_none_match = ", ".join(request_headers.getlist("if-none-match"))
        held_start: Message | None = None

        async def send_encoded(message: Message) -> None:
            # Hold a candidate's response start until the first body message shows whether
            # the response is buffered, since encoding rewrites the start's headers.
            nonlocal held_start
            if message["type"] == "http.response.start":
                if message["status"] == 304:
                    _match_revalidated_etag(message, if_none_match=if_none_match)
                elif _is_candidate(message):
                    held_start = message
                    return
            if held_start is None:
                await send(message)
                return

            start, held_start = held_start, None
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                message = _encode(start, message, client_accepts_gzip=client_accepts_gzip)
            await send(start)
            await send(message)

        await self.app(scope, receive, send_encoded)
