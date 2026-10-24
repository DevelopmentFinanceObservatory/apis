"""Which requests carry a Cloud Run ID token, its audience, caching and refresh, the held
token after a failed refresh, per-audience locking, the credentials each kind of ADC mints
with, and the loud failure when no token can be had."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
import httpx
import pytest
import requests
from google.auth import _cloud_sdk, impersonated_credentials
from google.auth.exceptions import DefaultCredentialsError

from dfo_platform import cloud_run_auth
from dfo_platform.cloud_run_auth import (
    FAILED_MINT_RETRY_S,
    REFRESH_MARGIN_S,
    TOKEN_HEADER,
    CloudRunAuthError,
    CloudRunIdTokenAuth,
    IdTokenMinter,
    adc_id_token_credentials,
    mint_from_adc,
)

_DCP = "https://dfo-dev-dc-datacommons-service-abc123-nn.a.run.app"
_OTHER = "https://dfo-prod-dc-datacommons-service-xyz789-nn.a.run.app"
_ISSUED_AT = 1_000_000.0
_LIFETIME_S = 3600.0
_EMAIL = "reader@project.iam.gserviceaccount.com"


def _jwt(*, audience: str, exp: float, serial: int) -> str:
    """An unsigned JWT carrying `exp`, enough for the minter's unverified read."""
    claims = {"aud": audience, "exp": exp, "serial": serial}
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


class _Clock:
    def __init__(self) -> None:
        self.now = _ISSUED_AT

    def __call__(self) -> float:
        return self.now


class _Fetcher:
    """Mints a token valid for an hour from the clock's current time, recording each audience.
    Refuses while `refusing` is set."""

    def __init__(self, clock: _Clock) -> None:
        self.audiences: list[str] = []
        self.refusing = False
        self._clock = clock

    def __call__(self, audience: str) -> str:
        self.audiences.append(audience)
        if self.refusing:
            raise CloudRunAuthError(f"cannot mint an ID token for {audience}")
        return _jwt(audience=audience, exp=self._clock() + _LIFETIME_S, serial=len(self.audiences))


def _auth(fetch_token, clock: _Clock | None = None) -> CloudRunIdTokenAuth:
    return CloudRunIdTokenAuth(
        minter=IdTokenMinter(fetch_token=fetch_token, clock=clock or _Clock())
    )


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

    claims = _claims(_sent_token(_auth(fetcher, clock), url))

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
    fetcher = _Fetcher(_Clock())

    assert _sent_token(_auth(fetcher), url) is None
    assert fetcher.audiences == []


def test_plain_http_run_app_request_is_refused_before_minting() -> None:
    fetcher = _Fetcher(_Clock())
    url = _DCP.replace("https://", "http://") + "/core/api"

    with pytest.raises(CloudRunAuthError) as raised:
        _sent_token(_auth(fetcher), url)

    assert raised.value.request.url == url
    assert fetcher.audiences == []


def test_token_is_reused_and_refreshed_inside_the_margin() -> None:
    clock = _Clock()
    auth = _auth(_Fetcher(clock), clock)
    url = f"{_DCP}/core/api/v2/node"

    first = _claims(_sent_token(auth, url))
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S - 1
    reused = _claims(_sent_token(auth, url))
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S + 1
    refreshed = _claims(_sent_token(auth, url))

    assert first["serial"] == reused["serial"] == 1
    assert refreshed["serial"] == 2


def test_each_origin_gets_its_own_token() -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    auth = _auth(fetcher, clock)

    assert _claims(_sent_token(auth, f"{_DCP}/core/api"))["aud"] == _DCP
    assert _claims(_sent_token(auth, f"{_OTHER}/core/api"))["aud"] == _OTHER
    assert fetcher.audiences == [_DCP, _OTHER]


def test_a_failed_mint_is_remembered_for_the_retry_window() -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    fetcher.refusing = True
    auth = _auth(fetcher, clock)
    url = f"{_DCP}/core/api"

    for _ in range(2):
        with pytest.raises(CloudRunAuthError) as raised:
            _sent_token(auth, url)
        assert isinstance(raised.value, httpx.RequestError)
        assert raised.value.request.url == url
    assert fetcher.audiences == [_DCP]

    fetcher.refusing = False
    clock.now = _ISSUED_AT + FAILED_MINT_RETRY_S
    assert _claims(_sent_token(auth, url))["serial"] == 2


def test_a_held_token_serves_until_its_exp_when_refreshing_fails() -> None:
    clock = _Clock()
    fetcher = _Fetcher(clock)
    auth = _auth(fetcher, clock)
    url = f"{_DCP}/core/api"
    _sent_token(auth, url)

    fetcher.refusing = True
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S + 1
    inside_margin = _claims(_sent_token(auth, url))
    clock.now = _ISSUED_AT + _LIFETIME_S - 1
    last_second = _claims(_sent_token(auth, url))
    clock.now = _ISSUED_AT + _LIFETIME_S
    with pytest.raises(CloudRunAuthError):
        _sent_token(auth, url)

    assert inside_margin["serial"] == last_second["serial"] == 1
    assert len(fetcher.audiences) == 3


def test_a_slow_mint_does_not_block_another_audience() -> None:
    clock = _Clock()
    release = threading.Event()
    minting_dcp = threading.Event()

    def fetch(audience: str) -> str:
        if audience == _DCP:
            minting_dcp.set()
            assert release.wait(timeout=5)
        return _jwt(audience=audience, exp=clock() + _LIFETIME_S, serial=0)

    auth = _auth(fetch, clock)
    sent: list[str | None] = []
    blocked = threading.Thread(target=_sent_token, args=(auth, f"{_DCP}/core/api"))
    other = threading.Thread(target=lambda: sent.append(_sent_token(auth, f"{_OTHER}/core/api")))
    blocked.start()
    try:
        assert minting_dcp.wait(timeout=5)
        other.start()
        other.join(timeout=2)
        assert not other.is_alive()
    finally:
        release.set()
        blocked.join()
        other.join()
    assert _claims(sent[0])["aud"] == _OTHER


def test_a_token_without_exp_raises() -> None:
    with pytest.raises(CloudRunAuthError):
        _sent_token(_auth(lambda audience: "not-a-jwt"), f"{_DCP}/core/api")


def _request() -> google.auth.transport.requests.Request:
    return google.auth.transport.requests.Request()


def test_impersonated_gcloud_adc_mints_as_the_impersonated_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gcloud auth application-default login --impersonate-service-account` writes the
    well-known ADC file, which GOOGLE_APPLICATION_CREDENTIALS does not name."""
    (tmp_path / "application_default_credentials.json").write_text(
        json.dumps(
            {
                "type": "impersonated_service_account",
                "service_account_impersonation_url": "https://iamcredentials.googleapis.com/"
                f"v1/projects/-/serviceAccounts/{_EMAIL}:generateAccessToken",
                "source_credentials": {
                    "type": "authorized_user",
                    "client_id": "client",
                    "client_secret": "secret",
                    "refresh_token": "refresh",
                },
            }
        )
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path))
    monkeypatch.setattr(_cloud_sdk, "get_project_id", lambda: "project")

    credentials = adc_id_token_credentials(_DCP, _request())

    assert isinstance(credentials, impersonated_credentials.IDTokenCredentials)
    assert credentials._target_audience == _DCP
    assert credentials._target_credentials.signer_email == _EMAIL


def test_external_account_adc_mints_as_its_service_account_on_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """google-github-actions/auth writes an external_account file that impersonates a
    service account. The ID token is asked of the IAM Credentials API for that account, for
    the audience, with the email claim Cloud Run and the request log read."""
    credentials_file = tmp_path / "gha-creds.json"
    credentials_file.write_text(
        json.dumps(
            {
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/1/locations/global/"
                "workloadIdentityPools/github-pool/providers/github",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": "https://sts.googleapis.com/v1/token",
                "service_account_impersonation_url": "https://iamcredentials.googleapis.com/"
                f"v1/projects/-/serviceAccounts/{_EMAIL}:generateAccessToken",
                "credential_source": {
                    "url": "https://token.actions.example/?audience=x",
                    "headers": {"Authorization": "Bearer request-token"},
                    "format": {"type": "json", "subject_token_field_name": "value"},
                },
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))

    credentials = adc_id_token_credentials(_DCP, _request())

    assert isinstance(credentials, impersonated_credentials.IDTokenCredentials)
    assert credentials._target_audience == _DCP
    assert credentials._include_email is True
    assert credentials._target_credentials.signer_email == _EMAIL
    # Unscoped from the file, the source is scoped for the IAM Credentials call.
    assert credentials._target_credentials._source_credentials.scopes == [
        "https://www.googleapis.com/auth/iam"
    ]


def _user_adc(**_: object) -> tuple[object, str]:
    return google.oauth2.credentials.Credentials(token="user"), "project"


def _no_adc(**_: object) -> tuple[object, str]:
    raise DefaultCredentialsError("no credentials found")


@pytest.mark.parametrize("default", [_user_adc, _no_adc], ids=["user-adc", "no-adc"])
def test_adc_that_cannot_mint_raises_naming_the_audience_and_the_routes(
    monkeypatch: pytest.MonkeyPatch, default
) -> None:
    monkeypatch.setattr(google.auth, "default", default)

    with pytest.raises(CloudRunAuthError) as raised:
        mint_from_adc(_DCP)

    assert _DCP in str(raised.value)
    assert "--impersonate-service-account" in str(raised.value)


class _FailingRefresh:
    """ID token credentials whose refresh raises `error`, as the impersonation route's
    unwrapped IAM Credentials call does."""

    token = None

    def __init__(self, error: Exception) -> None:
        self._error = error

    def refresh(self, request: object) -> None:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("connection reset"),
        requests.JSONDecodeError("Expecting value", "<html>502</html>", 0),
    ],
    ids=["connection-error", "html-error-body"],
)
def test_a_refresh_failing_outside_google_auth_keeps_the_held_token(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """The first mint succeeds, then ADC minting fails: the failure is a `CloudRunAuthError`
    naming the audience, and the held token keeps serving inside the refresh margin."""
    held = [_jwt(audience=_DCP, exp=_ISSUED_AT + _LIFETIME_S, serial=1)]
    clock = _Clock()
    auth = _auth(lambda audience: held.pop() if held else mint_from_adc(audience), clock)
    url = f"{_DCP}/core/api"
    _sent_token(auth, url)
    monkeypatch.setattr(
        cloud_run_auth, "adc_id_token_credentials", lambda audience, request: _FailingRefresh(error)
    )

    with pytest.raises(CloudRunAuthError, match=_DCP):
        mint_from_adc(_DCP)
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S + 1
    assert _claims(_sent_token(auth, url))["serial"] == 1
