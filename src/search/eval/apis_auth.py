"""Cloud Run ID tokens for the eval and the scripts that call a deployed apis service.

A deployed apis service admits only callers holding ``roles/run.invoker``. Cloud Run reads
the caller's ID token from ``X-Serverless-Authorization``, and the token's audience must be
the service's origin. ``apis_auth()`` returns the one ``httpx.Auth`` every eval client and
script shares. It sends a token on every ``https`` request and none on plain ``http``, so a
local app, an in-process app, or ``gcloud run services proxy`` on localhost gets none.

The token comes from ``DFO_APIS_ID_TOKEN`` when it is set. Otherwise the process's one
``dfo_platform.cloud_run_auth.IdTokenMinter`` mints it for the request's origin from
Application Default Credentials and refreshes it before it expires, so a run can outlast one
token. That module lists the credentials that can mint.

A token that cannot be had, or a tokened request answered with 401 or 403, which is Cloud Run
refusing the caller, raises ``ApisAuthError``. It subclasses no ``httpx`` error, so no eval
path scores it as an infrastructure loss.
"""

from __future__ import annotations

import asyncio
import functools
import os
from collections.abc import AsyncGenerator, Generator
from typing import override

import httpx

from dfo_platform.cloud_run_auth import (
    TOKEN_HEADER,
    CloudRunAuthError,
    IdTokenMinter,
    id_token_minter,
)

TOKEN_ENV = "DFO_APIS_ID_TOKEN"
_REFUSED_STATUSES = frozenset({401, 403})
_NO_IDENTITY_HINT = (
    f"set {TOKEN_ENV}, authenticate as a service account, or reach the service through "
    "`gcloud run services proxy` and an http://localhost URL"
)


class ApisAuthError(RuntimeError):
    """No ID token could be had for an apis origin, or the service refused the one sent."""


class ApisAuth(httpx.Auth):
    """Adds a Cloud Run ID token for the request's origin to every ``https`` request, and
    raises ``ApisAuthError`` when no token can be had or such a request gets 401 or 403.

    ``minter`` defaults to the process's one minter.
    """

    def __init__(
        self, *, static_token: str | None = None, minter: IdTokenMinter | None = None
    ) -> None:
        self._static_token = static_token
        self._minter = minter or id_token_minter()

    @override
    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        origin = _tokened_origin(request.url)
        if origin is not None:
            request.headers[TOKEN_HEADER] = f"Bearer {self._token(origin, request)}"
        response = yield request
        _check(response, origin)

    @override
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        origin = _tokened_origin(request.url)
        if origin is not None:
            # Minting and refreshing block on HTTP, so they leave the event loop.
            token = (
                self._static_token
                or self._minter.fresh(origin)
                or await asyncio.to_thread(self._token, origin, request)
            )
            request.headers[TOKEN_HEADER] = f"Bearer {token}"
        response = yield request
        _check(response, origin)

    def _token(self, origin: str, request: httpx.Request) -> str:
        if self._static_token:
            return self._static_token
        try:
            return self._minter.token(origin, request=request)
        except CloudRunAuthError as exc:
            raise ApisAuthError(f"{exc}; {_NO_IDENTITY_HINT}") from exc


@functools.cache
def apis_auth() -> ApisAuth:
    """The process's one ``ApisAuth``.

    ``DFO_APIS_ID_TOKEN`` is read on the first call.
    """
    return ApisAuth(static_token=os.environ.get(TOKEN_ENV) or None)


def _tokened_origin(url: httpx.URL) -> str | None:
    """The audience a request to ``url`` needs, or None when it gets no token."""
    if url.scheme != "https":
        return None
    port = f":{url.port}" if url.port is not None else ""
    return f"https://{url.host}{port}"


def _check(response: httpx.Response, origin: str | None) -> None:
    if origin is not None and response.status_code in _REFUSED_STATUSES:
        raise ApisAuthError(
            f"{origin} refused the ID token with HTTP {response.status_code}: the caller "
            "needs roles/run.invoker on the service, and the token's audience must be "
            f"{origin}"
        )
