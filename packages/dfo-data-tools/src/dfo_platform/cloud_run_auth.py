"""Google-signed ID tokens for requests to Cloud Run services.

A Cloud Run service that refuses unauthenticated calls admits a caller holding
``roles/run.invoker`` whose ID token names the service's origin as its audience. Cloud Run
reads the token from ``X-Serverless-Authorization: Bearer <token>``, checks it, and strips the
header, so the service's container never sees it.

``IdTokenMinter`` holds one token per audience. ``id_token_minter()`` is the process's one
minter, shared by ``CloudRunIdTokenAuth`` here and ``search.eval.apis_auth.ApisAuth``. By
default it mints from Application Default Credentials, by their kind:

- the metadata server (Cloud Run, Compute Engine) answers from its identity endpoint;
- a service-account key file signs the token request with its key;
- impersonated-service-account ADC, as ``gcloud auth application-default login
  --impersonate-service-account=<service account>`` writes it, asks the IAM Credentials API
  for a token of the impersonated account;
- any other ADC that names a service account, such as the external-account file
  ``google-github-actions/auth`` writes for Workload Identity Federation, asks the IAM
  Credentials API for a token of that account acting on itself, which needs
  ``roles/iam.serviceAccountOpenIdTokenCreator`` on itself.

User ADC names no service account and cannot mint. A minted token is cached and replaced once
less than ``REFRESH_MARGIN_S`` of its lifetime remains. A failed mint is remembered for
``FAILED_MINT_RETRY_S``, so concurrent reads fail at once, and a held token keeps serving until
its own ``exp`` while minting fails. Each audience mints under its own lock.

``CloudRunIdTokenAuth`` adds the header to a request exactly when the request's host ends in
``.run.app`` (a trailing dot ignored), with ``https://<host>`` as the audience, and refuses
such a request over plain ``http``. Every other host (base Data Commons, the UN Data Commons,
localhost, a test transport's placeholder) gets no header, so the same client reads any of them
unchanged. A ``.run.app`` request that can get no token raises ``CloudRunAuthError`` before it
is sent. The error is an ``httpx.RequestError`` carrying the request, so a client retries it
and maps it to its read error as it does a failed connection.
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import threading
import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Protocol, override

import httpx

if TYPE_CHECKING:
    from google.auth.transport import Request

TOKEN_HEADER = "X-Serverless-Authorization"
REFRESH_MARGIN_S = 300.0
FAILED_MINT_RETRY_S = 60.0
_CLOUD_RUN_HOST_SUFFIX = ".run.app"
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
MINTING_ROUTES = (
    "ID tokens mint from the metadata server, a service-account key file in "
    "GOOGLE_APPLICATION_CREDENTIALS, `gcloud auth application-default login "
    "--impersonate-service-account=<service account>`, or external-account credentials that "
    "impersonate a service account holding roles/iam.serviceAccountOpenIdTokenCreator on itself"
)

type TokenFetcher = Callable[[str], str]
"""Mints a Google-signed ID token for the audience it is given, or raises
``CloudRunAuthError``."""


class CloudRunAuthError(httpx.RequestError):
    """No ID token could be had for a Cloud Run audience."""


class IdTokenCredentials(Protocol):
    """The part of ``google.auth`` ID token credentials the minter uses."""

    token: str | None

    def refresh(self, request: Request) -> None: ...


class IdTokenMinter:
    """Mints, caches and refreshes one ID token per audience, shared across threads.

    ``fetch_token`` mints a token for an audience and defaults to Application Default
    Credentials. ``clock`` returns epoch seconds and is compared with a token's ``exp``.
    """

    def __init__(
        self,
        *,
        fetch_token: TokenFetcher | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._fetch_token = fetch_token or mint_from_adc
        self._clock = clock
        self._held: dict[str, tuple[str, float]] = {}
        self._failures: dict[str, tuple[str, float]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def fresh(self, audience: str) -> str | None:
        """The held token for ``audience`` while it needs no refresh. Never blocks."""
        held = self._held.get(audience)
        if held is not None and held[1] - self._clock() > REFRESH_MARGIN_S:
            return held[0]
        return None

    def token(self, audience: str, *, request: httpx.Request) -> str:
        """A token for ``audience``, minted when none is fresh. Raises ``CloudRunAuthError``
        carrying ``request`` when none can be had."""
        if (token := self.fresh(audience)) is not None:
            return token
        with self._lock_for(audience):
            if (token := self.fresh(audience)) is not None:
                return token
            now = self._clock()
            failed = self._failures.get(audience)
            if failed is None or now - failed[1] >= FAILED_MINT_RETRY_S:
                try:
                    token = self._fetch_token(audience)
                    expiry = _expiry(token, audience=audience)
                except CloudRunAuthError as exc:
                    failed = self._failures[audience] = (str(exc), now)
                else:
                    self._failures.pop(audience, None)
                    self._held[audience] = (token, expiry)
                    return token
            held = self._held.get(audience)
            if held is not None and held[1] > now:
                return held[0]
            raise CloudRunAuthError(failed[0], request=request)

    def _lock_for(self, audience: str) -> threading.Lock:
        with self._locks_lock:
            return self._locks.setdefault(audience, threading.Lock())


@functools.cache
def id_token_minter() -> IdTokenMinter:
    """The process's one ``IdTokenMinter``, so every auth shares its minted tokens."""
    return IdTokenMinter()


class CloudRunIdTokenAuth(httpx.Auth):
    """Adds ``X-Serverless-Authorization: Bearer <ID token>`` to every request whose host ends in
    ``.run.app``, and nothing to any other request.

    ``minter`` defaults to the process's one minter. Minting blocks, so an async client using
    this auth blocks its event loop for the one mint per audience per hour.
    """

    def __init__(self, *, minter: IdTokenMinter | None = None) -> None:
        self._minter = minter or id_token_minter()

    @override
    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        host = request.url.host.removesuffix(".")
        if host.endswith(_CLOUD_RUN_HOST_SUFFIX):
            if request.url.scheme != "https":
                raise CloudRunAuthError(
                    f"refusing to send an ID token for {host} over {request.url.scheme}; use https",
                    request=request,
                )
            token = self._minter.token(f"https://{host}", request=request)
            request.headers[TOKEN_HEADER] = f"Bearer {token}"
        yield request


@functools.cache
def cloud_run_auth() -> CloudRunIdTokenAuth:
    """The process's one ``CloudRunIdTokenAuth``."""
    return CloudRunIdTokenAuth()


def mint_from_adc(audience: str) -> str:
    """An ID token for ``audience`` from Application Default Credentials.

    The impersonation route posts to the IAM Credentials API without wrapping a transport
    failure, and reads a non-JSON error body as JSON, so a ``requests`` error and a
    ``ValueError`` raise ``CloudRunAuthError`` as a ``GoogleAuthError`` does."""
    import google.auth.exceptions
    import google.auth.transport.requests
    import requests

    request = google.auth.transport.requests.Request()
    try:
        credentials = adc_id_token_credentials(audience, request)
        credentials.refresh(request)
    except (
        google.auth.exceptions.GoogleAuthError,
        requests.RequestException,
        ValueError,
    ) as exc:
        raise CloudRunAuthError(
            f"cannot mint an ID token for {audience}: {exc}. {MINTING_ROUTES}"
        ) from exc
    if not credentials.token:
        raise CloudRunAuthError(f"minting an ID token for {audience} returned none")
    return credentials.token


def adc_id_token_credentials(audience: str, request: Request) -> IdTokenCredentials:
    """ID token credentials for ``audience`` that suit the kind of Application Default
    Credentials found. Raises ``CloudRunAuthError`` when those name no service account."""
    import google.auth
    from google.auth import compute_engine, impersonated_credentials
    from google.oauth2 import service_account

    # Unscoped, because impersonation scopes the source itself and a scoped external
    # account would spend a token on looking up its project here.
    source, _ = google.auth.default(request=request)
    if isinstance(source, compute_engine.Credentials):
        return compute_engine.IDTokenCredentials(
            request, audience, use_metadata_identity_endpoint=True
        )
    if isinstance(source, service_account.Credentials):
        return service_account.IDTokenCredentials(
            signer=source.signer,
            service_account_email=source.service_account_email,
            # The key file's token endpoint, which the credentials keep only privately.
            token_uri=source._token_uri,
            target_audience=audience,
        )
    if isinstance(source, impersonated_credentials.Credentials):
        target = source
    elif email := getattr(source, "service_account_email", None):
        target = impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=email,
            # The constructor requires scopes. No access token is minted for the target.
            target_scopes=[_CLOUD_PLATFORM_SCOPE],
        )
    else:
        raise CloudRunAuthError(
            f"cannot mint an ID token for {audience}: the default credentials name no "
            f"service account. {MINTING_ROUTES}"
        )
    return impersonated_credentials.IDTokenCredentials(
        target_credentials=target, target_audience=audience, include_email=True
    )


def _expiry(token: str, *, audience: str) -> float:
    """The ``exp`` claim of a JWT, read without verifying its signature."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(claims["exp"])
    except (IndexError, KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise CloudRunAuthError(f"the ID token minted for {audience} has no readable exp") from exc
