"""The composed app's CORS layer: Starlette's `CORSMiddleware` with public paths and merged
expose lists.

A public path is a route any origin may read with `GET`. Every `GET` or `HEAD` reply on one
carries `Access-Control-Allow-Origin: *`, whether the route sent it or a layer refused before
routing (an admission check, a 404), so a page tells a share taken down from a retryable fault.
A reply to any other method on the same path allows only what `CORSMiddleware` allows. A
preflight for a `GET` of a public path answers for any origin and allows `If-None-Match` beside the platform's request headers, so a page's conditional
fetch passes whatever `DFO_CORS_ORIGINS` lists. Every other preflight, such as one for a `PUT`
to the same path, answers as `CORSMiddleware` does.

Starlette replaces a response's `Access-Control-Expose-Headers` with the platform's list. This
layer keeps the response's own names after the platform's, so a route that exposes its own
headers, such as a share slot's `ETag` and freshness, stays readable to browser scripts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse, Response
from starlette.routing import compile_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["PlatformCORSMiddleware"]

_EXPOSE = "Access-Control-Expose-Headers"
_PUBLIC_METHODS = ("GET",)
# A `GET` route also answers `HEAD`, so a public path's `HEAD` reply allows any origin too.
_PUBLIC_READ_METHODS = frozenset({"GET", "HEAD"})
_PUBLIC_REQUEST_HEADER = "if-none-match"


class PlatformCORSMiddleware(CORSMiddleware):
    """`CORSMiddleware` that allows any origin on a `GET` of `public_paths` (route paths as a
    router declares them) and answers its preflight for any origin, and merges a response's own
    expose list into the platform's."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        public_paths: Sequence[str] = (),
        allow_origins: Sequence[str] = (),
        allow_methods: Sequence[str] = ("GET",),
        allow_headers: Sequence[str] = (),
        expose_headers: Sequence[str] = (),
        max_age: int = 600,
    ) -> None:
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            expose_headers=expose_headers,
            max_age=max_age,
        )
        self._public_paths: tuple[re.Pattern[str], ...] = tuple(
            compile_path(path)[0] for path in public_paths
        )
        # `CORSMiddleware` keeps its allowed request headers lowercased, the safelisted included.
        self._public_request_headers = sorted({*self.allow_headers, _PUBLIC_REQUEST_HEADER})
        self._public_preflight_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": ", ".join(_PUBLIC_METHODS),
            "Access-Control-Allow-Headers": ", ".join(self._public_request_headers),
            "Access-Control-Max-Age": str(max_age),
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._is_public(scope["path"]):
            await super().__call__(scope, receive, send)
            return
        if scope["method"] == "OPTIONS":
            headers = Headers(scope=scope)
            if "origin" in headers and headers.get("access-control-request-method") in (
                _PUBLIC_METHODS
            ):
                await self._public_preflight(headers)(scope, receive, send)
                return
        if scope["method"] in _PUBLIC_READ_METHODS:
            await super().__call__(scope, receive, _allowing_any_origin(send))
            return
        await super().__call__(scope, receive, send)

    def _is_public(self, path: str) -> bool:
        return any(public.match(path) for public in self._public_paths)

    def _public_preflight(self, request_headers: Headers) -> Response:
        """Allow any origin a `GET` sending only allowed request headers."""
        requested = request_headers.get("access-control-request-headers")
        if requested is not None and any(
            name.lower() not in self._public_request_headers for name in _names(requested)
        ):
            return PlainTextResponse(
                "Disallowed CORS headers", status_code=400, headers=self._public_preflight_headers
            )
        return PlainTextResponse("OK", status_code=200, headers=self._public_preflight_headers)

    async def send(self, message: Message, send: Send, request_headers: Headers) -> None:
        if message["type"] != "http.response.start":
            await send(message)
            return
        message.setdefault("headers", [])
        own = MutableHeaders(scope=message).get(_EXPOSE)
        if not own:
            await super().send(message, send, request_headers)
            return

        async def merged(start: Message) -> None:
            headers = MutableHeaders(scope=start)
            names = _names(headers.get(_EXPOSE, ""))
            seen = {name.lower() for name in names}
            names += [name for name in _names(own) if name.lower() not in seen]
            headers[_EXPOSE] = ", ".join(names)
            await send(start)

        await super().send(message, merged, request_headers)


def _allowing_any_origin(send: Send) -> Send:
    """`send` with `Access-Control-Allow-Origin: *` on the response start, in place of the
    listed origin `CORSMiddleware` may have set."""

    async def send_allowing_any_origin(message: Message) -> None:
        if message["type"] == "http.response.start":
            message.setdefault("headers", [])
            MutableHeaders(scope=message)["Access-Control-Allow-Origin"] = "*"
        await send(message)

    return send_allowing_any_origin


def _names(header: str) -> list[str]:
    """The header names a comma-separated list holds."""
    return [name.strip() for name in header.split(",") if name.strip()]
