"""Google-signed ID tokens for requests to Cloud Run services.

A Cloud Run service that refuses unauthenticated calls admits a caller holding
``roles/run.invoker`` whose ID token names the service's origin as its audience. Cloud Run
reads the token from ``X-Serverless-Authorization: Bearer <token>``, checks it, and strips the
header, so the service's container never sees it. ``CloudRunIdTokenAuth`` adds that header to a
request exactly when the request's host ends in ``.run.app`` (a trailing dot ignored), with
``https://<host>`` as the audience, and refuses such a request over plain ``http``. Every other
host (base Data Commons, the UN Data Commons, localhost, a test transport's placeholder) gets no
header, so the same client reads any of them unchanged.

``DCP_ID_TOKEN``, when set, is checked first. GitHub Actions sets it, because its external
account credentials cannot mint an ID token. A token whose ``aud`` is an ``https://`` origin is
sent only to that origin. A token whose ``aud`` is not a URL, as from ``gcloud auth
print-identity-token`` for a user, is sent to every ``.run.app`` host. Any other host's token is
minted per audience with ``google.oauth2.id_token.fetch_id_token``, which works from the Cloud
Run metadata server, a service-account key file, or an impersonated-service-account ADC file.
User ADC cannot mint. A minted token is cached and replaced once less than
``REFRESH_MARGIN_S`` of its lifetime remains. A failed mint is remembered for
``FAILED_MINT_RETRY_S``, so concurrent reads fail at once. A ``.run.app`` request that can get
no token raises ``CloudRunAuthError`` before it is sent.
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import os
import threading
import time
from collections.abc import Callable, Generator
from typing import override

import httpx

TOKEN_ENV = "DCP_ID_TOKEN"
TOKEN_HEADER = "X-Serverless-Authorization"
REFRESH_MARGIN_S = 300.0
FAILED_MINT_RETRY_S = 60.0
_CLOUD_RUN_HOST_SUFFIX = ".run.app"

type TokenFetcher = Callable[[str], str]
"""Mints a Google-signed ID token for the audience it is given."""


class CloudRunAuthError(RuntimeError):
    """No ID token could be had for a Cloud Run host."""


class CloudRunIdTokenAuth(httpx.Auth):
    """Adds ``X-Serverless-Authorization: Bearer <ID token>`` to every request whose host ends in
    ``.run.app``, and nothing to any other request.

    ``fetch_token`` mints a token for an audience and defaults to Application Default
    Credentials. ``clock`` returns epoch seconds and is compared with a token's ``exp``.
    The cache is shared across threads. Minting blocks, so an async client using this auth
    blocks its event loop for the one mint per audience per hour.
    """

    def __init__(
        self,
        *,
        fetch_token: TokenFetcher | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._fetch_token = fetch_token or _fetch_with_google_auth
        self._clock = clock
        self._cache: dict[str, tuple[str, float]] = {}
        self._failures: dict[str, tuple[CloudRunAuthError, float]] = {}
        self._lock = threading.Lock()

    @override
    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        host = request.url.host.removesuffix(".")
        if host.endswith(_CLOUD_RUN_HOST_SUFFIX):
            if request.url.scheme != "https":
                raise CloudRunAuthError(
                    f"refusing to send an ID token for {host} over {request.url.scheme}; use https"
                )
            request.headers[TOKEN_HEADER] = f"Bearer {self.token_for(host)}"
        yield request

    def token_for(self, host: str) -> str:
        """The ID token for ``https://<host>``: ``DCP_ID_TOKEN`` when its audience fits, else a
        cached or newly minted one."""
        audience = f"https://{host}"
        supplied = os.environ.get(TOKEN_ENV)
        if not supplied:
            return self._minted(audience)
        supplied_audience = _supplied_audience(supplied)
        if supplied_audience == audience or not supplied_audience.startswith("https://"):
            return supplied
        try:
            return self._minted(audience)
        except CloudRunAuthError as exc:
            raise CloudRunAuthError(
                f"{TOKEN_ENV} is for {supplied_audience}, not {audience}, and {exc}"
            ) from exc

    def _minted(self, audience: str) -> str:
        with self._lock:
            now = self._clock()
            cached = self._cache.get(audience)
            if cached is not None and cached[1] - now > REFRESH_MARGIN_S:
                return cached[0]
            failed = self._failures.get(audience)
            if failed is not None and now - failed[1] < FAILED_MINT_RETRY_S:
                raise CloudRunAuthError(str(failed[0])) from failed[0]
            try:
                token = self._fetch_token(audience)
                expiry = float(_claims(token, source=f"the ID token minted for {audience}")["exp"])
            except CloudRunAuthError as exc:
                self._failures[audience] = (exc, now)
                raise
            except (KeyError, TypeError, ValueError) as exc:
                error = CloudRunAuthError(f"the ID token minted for {audience} has no readable exp")
                self._failures[audience] = (error, now)
                raise error from exc
            self._failures.pop(audience, None)
            self._cache[audience] = (token, expiry)
            return token


@functools.cache
def cloud_run_auth() -> CloudRunIdTokenAuth:
    """The process's one ``CloudRunIdTokenAuth``, so every client shares its minted tokens."""
    return CloudRunIdTokenAuth()


def _fetch_with_google_auth(audience: str) -> str:
    import google.auth.exceptions
    import google.auth.transport.requests
    import google.oauth2.id_token

    try:
        return google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), audience
        )
    except google.auth.exceptions.GoogleAuthError as exc:
        raise CloudRunAuthError(
            f"cannot mint an ID token for {audience}: {exc}. User credentials cannot mint one. "
            "Either run `gcloud auth application-default login "
            "--impersonate-service-account=<service account>`, which mints and refreshes "
            f'tokens, or set {TOKEN_ENV}="$(gcloud auth print-identity-token)", which expires '
            "after an hour"
        ) from exc


@functools.cache
def _supplied_audience(token: str) -> str:
    """The ``aud`` claim of ``DCP_ID_TOKEN``."""
    try:
        return str(_claims(token, source=TOKEN_ENV)["aud"])
    except KeyError as exc:
        raise CloudRunAuthError(f"{TOKEN_ENV} has no aud claim") from exc


def _claims(token: str, *, source: str) -> dict:
    """The claims of a JWT, read without verifying its signature."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError, binascii.Error) as exc:
        raise CloudRunAuthError(f"{source} is not a readable JWT") from exc
    if not isinstance(claims, dict):
        raise CloudRunAuthError(f"{source} is not a readable JWT")
    return claims
