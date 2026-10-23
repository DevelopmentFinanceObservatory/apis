"""Which requests carry a Cloud Run ID token, its audience, caching and refresh, the
`DCP_ID_TOKEN` override, and the loud failure when no token can be had."""

from __future__ import annotations

import base64
import json

import google.oauth2.id_token
import httpx
import pytest
from google.auth.exceptions import DefaultCredentialsError

from dfo_platform.cloud_run_auth import (
    FAILED_MINT_RETRY_S,
    REFRESH_MARGIN_S,
    TOKEN_ENV,
    TOKEN_HEADER,
    CloudRunAuthError,
    CloudRunIdTokenAuth,
)

_DCP = "https://dfo-dev-dc-datacommons-service-abc123-nn.a.run.app"
_ISSUED_AT = 1_000_000.0
_LIFETIME_S = 3600.0


def _jwt(*, audience: str, exp: float, serial: int) -> str:
    """An unsigned JWT carrying `exp`, enough for the auth's unverified read."""
    claims = {"aud": audience, "exp": exp, "serial": serial}
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


class _Fetcher:
    """Mints a token valid for an hour from the clock's current time, recording each audience."""

    def __init__(self, clock: _Clock) -> None:
        self.audiences: list[str] = []
        self._clock = clock

    def __call__(self, audience: str) -> str:
        self.audiences.append(audience)
        return _jwt(audience=audience, exp=self._clock() + _LIFETIME_S, serial=len(self.audiences))


class _Clock:
    def __init__(self) -> None:
        self.now = _ISSUED_AT

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _no_supplied_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)


def _sent_token(auth: CloudRunIdTokenAuth, url: str) -> str | None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "token": request.headers.get(TOKEN_HEADER),
                "plain": "Authorization" in request.headers,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as client:
        body = client.get(url).json()
    assert body["plain"] is False
    return body["token"]


def _claims(sent: str | None) -> dict:
    assert sent is not None
    scheme, token = sent.split(" ")
    assert scheme == "Bearer"
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


@pytest.mark.parametrize(
    "url",
    [f"{_DCP}/core/api/v2/node?nodes=geoId/06", f"{_DCP}./core/api/v2/node"],
    ids=["plain", "trailing-dot"],
)
def test_run_app_request_carries_a_token_for_its_origin(url: str) -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    auth = CloudRunIdTokenAuth(fetch_token=fetcher, clock=clock)

    claims = _claims(_sent_token(auth, url))

    assert claims["aud"] == _DCP
    assert fetcher.audiences == [_DCP]


@pytest.mark.parametrize(
    "url",
    [
        "https://api.datacommons.org/v2/node",
        "https://unsd-datacommons.gcp.un-icc.cloud/core/api/v2/node",
        "http://localhost:8080/core/api/v2/node",
        "https://mixer.example.org/core/api/v2/node",
        "https://run.app.example.org/core/api",
    ],
)
def test_other_hosts_get_no_token(url: str) -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)

    assert _sent_token(CloudRunIdTokenAuth(fetch_token=fetcher, clock=clock), url) is None
    assert fetcher.audiences == []


def test_plain_http_run_app_request_is_refused_before_minting() -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    auth = CloudRunIdTokenAuth(fetch_token=fetcher, clock=clock)

    with pytest.raises(CloudRunAuthError):
        _sent_token(auth, _DCP.replace("https://", "http://") + "/core/api")
    assert fetcher.audiences == []


def test_token_is_reused_and_refreshed_inside_the_margin() -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    auth = CloudRunIdTokenAuth(fetch_token=fetcher, clock=clock)
    url = f"{_DCP}/core/api/v2/node"

    first = _claims(_sent_token(auth, url))
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S - 1
    reused = _claims(_sent_token(auth, url))
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S + 1
    refreshed = _claims(_sent_token(auth, url))

    assert first["serial"] == reused["serial"] == 1
    assert refreshed["serial"] == 2


_OTHER = "https://dfo-prod-dc-datacommons-service-xyz789-nn.a.run.app"


def test_each_origin_gets_its_own_token() -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    auth = CloudRunIdTokenAuth(fetch_token=fetcher, clock=clock)

    assert _claims(_sent_token(auth, f"{_DCP}/core/api"))["aud"] == _DCP
    assert _claims(_sent_token(auth, f"{_OTHER}/core/api"))["aud"] == _OTHER
    assert fetcher.audiences == [_DCP, _OTHER]


_GCLOUD_CLIENT_ID = "32555940559.apps.googleusercontent.com"


@pytest.mark.parametrize(
    ("supplied_audience", "url", "uses_supplied", "minted"),
    [
        (_DCP, f"{_DCP}/core/api", True, []),
        (_DCP, f"{_OTHER}/core/api", False, [_OTHER]),
        (_GCLOUD_CLIENT_ID, f"{_DCP}/core/api", True, []),
        (_GCLOUD_CLIENT_ID, f"{_OTHER}/core/api", True, []),
        (_DCP, "https://api.datacommons.org/v2/node", None, []),
    ],
    ids=["url-aud-match", "url-aud-other-origin", "user-aud", "user-aud-other", "non-run-app"],
)
def test_supplied_token_is_sent_where_its_audience_fits(
    monkeypatch: pytest.MonkeyPatch,
    supplied_audience: str,
    url: str,
    uses_supplied: bool | None,
    minted: list[str],
) -> None:
    supplied = _jwt(audience=supplied_audience, exp=_ISSUED_AT + _LIFETIME_S, serial=0)
    monkeypatch.setenv(TOKEN_ENV, supplied)
    clock = _Clock()
    fetcher = _Fetcher(clock)

    sent = _sent_token(CloudRunIdTokenAuth(fetch_token=fetcher, clock=clock), url)

    if uses_supplied is None:
        assert sent is None
    elif uses_supplied:
        assert sent == f"Bearer {supplied}"
    else:
        assert _claims(sent)["serial"] == 1
    assert fetcher.audiences == minted


def test_supplied_token_for_another_origin_raises_when_minting_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, _jwt(audience=_DCP, exp=_ISSUED_AT + _LIFETIME_S, serial=0))

    def refuse(audience: str) -> str:
        raise CloudRunAuthError(f"cannot mint an ID token for {audience}")

    with pytest.raises(CloudRunAuthError) as raised:
        _sent_token(CloudRunIdTokenAuth(fetch_token=refuse), f"{_OTHER}/core/api")

    assert _DCP in str(raised.value)
    assert _OTHER in str(raised.value)


def test_unreadable_supplied_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV, "not-a-jwt")
    clock = _Clock()

    with pytest.raises(CloudRunAuthError):
        _sent_token(CloudRunIdTokenAuth(fetch_token=_Fetcher(clock), clock=clock), _DCP)


def test_failure_to_mint_raises_naming_the_audience_and_both_fixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: object, audience: str) -> str:
        raise DefaultCredentialsError("no service account credentials")

    monkeypatch.setattr(google.oauth2.id_token, "fetch_id_token", refuse)

    with pytest.raises(CloudRunAuthError) as raised:
        _sent_token(CloudRunIdTokenAuth(), f"{_DCP}/core/api")

    assert _DCP in str(raised.value)
    assert TOKEN_ENV in str(raised.value)
    assert "--impersonate-service-account" in str(raised.value)


def test_a_failed_mint_is_remembered_for_the_retry_window() -> None:
    clock = _Clock()
    attempts: list[str] = []

    def refuse_then_mint(audience: str) -> str:
        attempts.append(audience)
        if len(attempts) == 1:
            raise CloudRunAuthError(f"cannot mint an ID token for {audience}")
        return _jwt(audience=audience, exp=clock() + _LIFETIME_S, serial=len(attempts))

    auth = CloudRunIdTokenAuth(fetch_token=refuse_then_mint, clock=clock)
    url = f"{_DCP}/core/api"

    for _ in range(2):
        with pytest.raises(CloudRunAuthError):
            _sent_token(auth, url)
    assert len(attempts) == 1

    clock.now = _ISSUED_AT + FAILED_MINT_RETRY_S
    assert _claims(_sent_token(auth, url))["serial"] == 2
    assert len(attempts) == 2


def test_a_token_without_exp_raises() -> None:
    auth = CloudRunIdTokenAuth(fetch_token=lambda audience: "not-a-jwt")

    with pytest.raises(CloudRunAuthError):
        _sent_token(auth, f"{_DCP}/core/api")
