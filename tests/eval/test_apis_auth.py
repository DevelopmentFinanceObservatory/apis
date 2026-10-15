"""The eval's Cloud Run ID token source: which requests carry a token, minting per origin,
refresh before expiry, the env override, and refusals that stop the run."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import google.auth
import httpx
import pytest
from google.auth import impersonated_credentials
from google.auth.exceptions import DefaultCredentialsError

from search.eval import apis_auth as module
from search.eval.apis_auth import TOKEN_HEADER, ApisAuth, ApisAuthError


class _Credentials:
    """ID token credentials that hand out ``<audience>#<n>`` and expire on demand."""

    def __init__(self, audience: str, *, refresh_delay_s: float = 0.0) -> None:
        self.audience = audience
        self.token: str | None = None
        self.refreshes = 0
        self.expired = True
        self._delay_s = refresh_delay_s

    @property
    def valid(self) -> bool:
        return self.token is not None and not self.expired

    def refresh(self, request: object) -> None:
        time.sleep(self._delay_s)
        self.refreshes += 1
        self.token = f"{self.audience}#{self.refreshes}"
        self.expired = False


class _Minter:
    """Records every audience the auth asks credentials for."""

    def __init__(self, *, refresh_delay_s: float = 0.0) -> None:
        self.made: dict[str, _Credentials] = {}
        self._delay_s = refresh_delay_s

    def __call__(self, audience: str) -> _Credentials:
        assert audience not in self.made, f"credentials for {audience} built twice"
        self.made[audience] = _Credentials(audience, refresh_delay_s=self._delay_s)
        return self.made[audience]


@pytest.fixture(autouse=True)
def _no_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake credentials ignore the transport request, so skip building a real one."""
    monkeypatch.setattr(module, "_transport_request", lambda: None)


def _echo(status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"token": request.headers.get(TOKEN_HEADER)})

    return handler


def _send(auth: ApisAuth, url: str, *, status: int = 200) -> str | None:
    with httpx.Client(transport=httpx.MockTransport(_echo(status)), auth=auth) as client:
        return client.get(url).json()["token"]


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8083/api/search/schema", "http://in-process/api/search/resolve"],
)
def test_plain_http_gets_no_token_and_builds_no_credentials(url):
    minter = _Minter()
    assert _send(ApisAuth(credentials=minter), url) is None
    assert minter.made == {}


def test_an_https_request_carries_a_token_minted_for_its_origin():
    minter = _Minter()
    auth = ApisAuth(credentials=minter)

    token = _send(auth, "https://apis.example/api/search/resolve?x=1")

    assert token == "Bearer https://apis.example#1"
    assert list(minter.made) == ["https://apis.example"]


def test_each_origin_gets_its_own_audience_and_an_explicit_port_is_kept():
    minter = _Minter()
    auth = ApisAuth(credentials=minter)

    _send(auth, "https://head.example/a")
    _send(auth, "https://base.example:8443/a")

    assert list(minter.made) == ["https://head.example", "https://base.example:8443"]


def test_a_token_is_reused_until_it_expires_then_refreshed():
    minter = _Minter()
    auth = ApisAuth(credentials=minter)

    first = _send(auth, "https://apis.example/a")
    again = _send(auth, "https://apis.example/b")
    minter.made["https://apis.example"].expired = True
    refreshed = _send(auth, "https://apis.example/c")

    assert (first, again, refreshed) == (
        "Bearer https://apis.example#1",
        "Bearer https://apis.example#1",
        "Bearer https://apis.example#2",
    )


def test_the_static_token_overrides_minting():
    minter = _Minter()
    auth = ApisAuth(static_token="from-env", credentials=minter)

    assert _send(auth, "https://apis.example/a") == "Bearer from-env"
    assert minter.made == {}


def test_the_process_auth_reads_the_override_from_the_environment(monkeypatch):
    monkeypatch.setenv(module.TOKEN_ENV, "from-env")
    module.apis_auth.cache_clear()
    try:
        assert _send(module.apis_auth(), "https://apis.example/a") == "Bearer from-env"
    finally:
        module.apis_auth.cache_clear()


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_token_raises(status):
    auth = ApisAuth(static_token="t")
    with pytest.raises(ApisAuthError, match=f"HTTP {status}"):
        _send(auth, "https://apis.example/a", status=status)


def test_a_401_from_a_local_app_is_returned_as_the_response():
    with httpx.Client(transport=httpx.MockTransport(_echo(401)), auth=ApisAuth()) as client:
        assert client.get("http://localhost/a").status_code == 401


def test_concurrent_async_requests_mint_once():
    minter = _Minter(refresh_delay_s=0.05)
    auth = ApisAuth(credentials=minter)

    async def go() -> list[str]:
        transport = httpx.MockTransport(_echo())
        async with httpx.AsyncClient(transport=transport, auth=auth) as client:
            replies = await asyncio.gather(
                *(client.get(f"https://apis.example/{i}") for i in range(8))
            )
        return [reply.json()["token"] for reply in replies]

    tokens = asyncio.run(go())

    assert set(tokens) == {"Bearer https://apis.example#1"}
    assert minter.made["https://apis.example"].refreshes == 1


def test_concurrent_threads_mint_once():
    minter = _Minter(refresh_delay_s=0.05)
    auth = ApisAuth(credentials=minter)
    tokens: list[str | None] = []

    def send() -> None:
        tokens.append(_send(auth, "https://apis.example/a"))

    threads = [threading.Thread(target=send) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tokens == ["Bearer https://apis.example#1"] * 6
    assert minter.made["https://apis.example"].refreshes == 1


def test_default_credentials_without_a_service_account_refuse(monkeypatch):
    class _UserCredentials:
        pass

    monkeypatch.setattr(google.auth, "default", lambda: (_UserCredentials(), "project"))
    with pytest.raises(ApisAuthError, match="DFO_APIS_ID_TOKEN"):
        _send(ApisAuth(), "https://apis.example/a")


def test_missing_default_credentials_refuse(monkeypatch):
    def missing():
        raise DefaultCredentialsError("none found")

    monkeypatch.setattr(google.auth, "default", missing)
    with pytest.raises(ApisAuthError, match="none found"):
        _send(ApisAuth(), "https://apis.example/a")


def test_the_auth_actions_credentials_mint_as_their_service_account(tmp_path, monkeypatch):
    """google-github-actions/auth writes an external_account file that impersonates the
    eval reader. The ID token is asked of the IAM Credentials API for that account, for
    the origin, with the email claim Cloud Run and the request log read."""
    email = "eval-reader@project.iam.gserviceaccount.com"
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
                f"v1/projects/-/serviceAccounts/{email}:generateAccessToken",
                "credential_source": {
                    "url": "https://token.actions.example/?audience=x",
                    "headers": {"Authorization": "Bearer request-token"},
                    "format": {"type": "json", "subject_token_field_name": "value"},
                },
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))

    credentials = module._adc_id_token_credentials("https://apis.example")

    assert isinstance(credentials, impersonated_credentials.IDTokenCredentials)
    assert credentials._target_audience == "https://apis.example"
    assert credentials._include_email is True
    assert credentials._target_credentials.signer_email == email
    # Unscoped from the file, the source is scoped for the IAM Credentials call.
    assert credentials._target_credentials._source_credentials.scopes == [
        "https://www.googleapis.com/auth/iam"
    ]
