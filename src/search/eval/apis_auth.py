"""Cloud Run ID tokens for the eval and the scripts that call a deployed apis service.

A deployed apis service admits only callers holding ``roles/run.invoker``. Cloud Run reads
the caller's ID token from ``X-Serverless-Authorization``, and the token's audience must be
the service's origin. ``apis_auth()`` returns the one ``httpx.Auth`` every eval client and
script shares. It sends a token on every ``https`` request and none on plain ``http``, so a
local app, an in-process app, or ``gcloud run services proxy`` on localhost gets none.

The token comes from ``DFO_APIS_ID_TOKEN`` when it is set. Otherwise it is minted for the
request's origin from Application Default Credentials that name a service account, as the
``google-github-actions/auth`` step writes them: the account asks the IAM Credentials API
for an ID token for itself, which needs ``roles/iam.serviceAccountOpenIdTokenCreator`` on
itself. A minted token is refreshed before it expires, so a run can outlast one token.

A tokened request answered with 401 or 403 is Cloud Run refusing the caller. It raises
``ApisAuthError``, which subclasses no ``httpx`` error, so no eval path scores it as an
infrastructure loss.
"""

from __future__ import annotations

import asyncio
import functools
import os
import threading
from collections.abc import AsyncGenerator, Callable, Generator
from typing import TYPE_CHECKING, Protocol, override

import httpx

if TYPE_CHECKING:
    from google.auth.transport import Request

TOKEN_ENV = "DFO_APIS_ID_TOKEN"
TOKEN_HEADER = "X-Serverless-Authorization"
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_REFUSED_STATUSES = frozenset({401, 403})
_NO_IDENTITY_HINT = (
    f"set {TOKEN_ENV}, authenticate as a service account, or reach the service through "
    "`gcloud run services proxy` and an http://localhost URL"
)


class ApisAuthError(RuntimeError):
    """No ID token could be had for an apis origin, or the service refused the one sent."""


class IdTokenCredentials(Protocol):
    """The part of ``google.auth`` ID token credentials the auth uses."""

    token: str | None

    @property
    def valid(self) -> bool: ...

    def refresh(self, request: Request) -> None: ...


class ApisAuth(httpx.Auth):
    """Adds a Cloud Run ID token, minted per origin and refreshed before expiry, to every
    ``https`` request, and raises ``ApisAuthError`` when such a request gets 401 or 403.

    ``credentials`` builds ID token credentials for an audience, and the default mints from
    Application Default Credentials. It is called once per origin, on first use.
    """

    def __init__(
        self,
        *,
        static_token: str | None = None,
        credentials: Callable[[str], IdTokenCredentials] | None = None,
    ) -> None:
        self._static_token = static_token
        self._credentials = credentials or _adc_id_token_credentials
        self._by_origin: dict[str, IdTokenCredentials] = {}
        self._lock = threading.Lock()

    @override
    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        origin = _tokened_origin(request.url)
        if origin is not None:
            request.headers[TOKEN_HEADER] = f"Bearer {self._token(origin)}"
        response = yield request
        _check(response, origin)

    @override
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        origin = _tokened_origin(request.url)
        if origin is not None:
            # Minting and refreshing block on HTTP, so they leave the event loop.
            token = self._cached(origin) or await asyncio.to_thread(self._token, origin)
            request.headers[TOKEN_HEADER] = f"Bearer {token}"
        response = yield request
        _check(response, origin)

    def _cached(self, origin: str) -> str | None:
        if self._static_token:
            return self._static_token
        held = self._by_origin.get(origin)
        return held.token if held is not None and held.valid else None

    def _token(self, origin: str) -> str:
        if self._static_token:
            return self._static_token
        with self._lock:
            held = self._by_origin.get(origin)
            if held is None:
                held = self._by_origin[origin] = self._credentials(origin)
            if not held.valid:
                held.refresh(_transport_request())
            if not held.token:
                raise ApisAuthError(f"minting an ID token for {origin} returned none")
            return held.token


@functools.cache
def apis_auth() -> ApisAuth:
    """The process's one ``ApisAuth``, so every client shares its minted tokens.

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


def _transport_request() -> Request:
    import google.auth.transport.requests

    return google.auth.transport.requests.Request()


def _adc_id_token_credentials(audience: str) -> IdTokenCredentials:
    """ID token credentials for ``audience``, minted by the service account that Application
    Default Credentials name, acting on itself through the IAM Credentials API."""
    import google.auth
    from google.auth import exceptions, impersonated_credentials

    try:
        # Unscoped, because impersonation scopes the source itself and a scoped external
        # account would spend a token on looking up its project here.
        source, _ = google.auth.default()
    except exceptions.DefaultCredentialsError as exc:
        raise ApisAuthError(
            f"no credentials to mint an ID token for {audience}: {exc}; {_NO_IDENTITY_HINT}"
        ) from exc
    email = getattr(source, "service_account_email", None)
    if not email:
        raise ApisAuthError(
            f"the default credentials name no service account to mint an ID token for "
            f"{audience}; {_NO_IDENTITY_HINT}"
        )
    target = impersonated_credentials.Credentials(
        source_credentials=source,
        target_principal=email,
        # The constructor requires scopes. No access token is minted for the target.
        target_scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    return impersonated_credentials.IDTokenCredentials(
        target_credentials=target, target_audience=audience, include_email=True
    )
