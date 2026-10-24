"""The eval's Cloud Run ID token source: which requests carry a token, minting per origin,
refresh before expiry, the env override, and refusals that stop the run."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

import httpx
import pytest

from dfo_platform.cloud_run_auth import REFRESH_MARGIN_S, CloudRunAuthError, IdTokenMinter
from search.eval import apis_auth as module
from search.eval.apis_auth import TOKEN_HEADER, ApisAuth, ApisAuthError

_ISSUED_AT = 1_000_000.0
_LIFETIME_S = 3600.0


class _Clock:
    def __init__(self) -> None:
        self.now = _ISSUED_AT

    def __call__(self) -> float:
        return self.now


class _Fetcher:
    """Mints `<audience>#<n>` tokens valid for an hour, counting mints per audience."""

    def __init__(self, clock: _Clock, *, delay_s: float = 0.0) -> None:
        self.mints: dict[str, int] = {}
        self._clock = clock
        self._delay_s = delay_s

    def __call__(self, audience: str) -> str:
        time.sleep(self._delay_s)
        self.mints[audience] = self.mints.get(audience, 0) + 1
        claims = {"exp": self._clock() + _LIFETIME_S, "id": f"{audience}#{self.mints[audience]}"}
        body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        return f"header.{body}.signature"


def _auth(fetch_token, clock: _Clock | None = None) -> ApisAuth:
    return ApisAuth(minter=IdTokenMinter(fetch_token=fetch_token, clock=clock or _Clock()))


def _token_id(sent: str | None) -> str | None:
    """The `id` claim of a sent bearer token, or the sent value when it is not a JWT."""
    if sent is None:
        return None
    token = sent.removeprefix("Bearer ")
    if token.count(".") != 2:
        return sent
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["id"]


def _echo(status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"token": request.headers.get(TOKEN_HEADER)})

    return handler


def _send(auth: ApisAuth, url: str, *, status: int = 200) -> str | None:
    with httpx.Client(transport=httpx.MockTransport(_echo(status)), auth=auth) as client:
        return _token_id(client.get(url).json()["token"])


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8083/api/search/schema", "http://in-process/api/search/resolve"],
)
def test_plain_http_gets_no_token_and_mints_none(url):
    fetcher = _Fetcher(_Clock())
    assert _send(_auth(fetcher), url) is None
    assert fetcher.mints == {}


def test_an_https_request_carries_a_token_minted_for_its_origin():
    fetcher = _Fetcher(_Clock())

    assert _send(_auth(fetcher), "https://apis.example/api/search/resolve?x=1") == (
        "https://apis.example#1"
    )
    assert list(fetcher.mints) == ["https://apis.example"]


def test_each_origin_gets_its_own_audience_and_an_explicit_port_is_kept():
    fetcher = _Fetcher(_Clock())
    auth = _auth(fetcher)

    _send(auth, "https://head.example/a")
    _send(auth, "https://base.example:8443/a")

    assert list(fetcher.mints) == ["https://head.example", "https://base.example:8443"]


def test_a_token_is_reused_until_it_nears_expiry_then_refreshed():
    clock = _Clock()
    auth = _auth(_Fetcher(clock), clock)

    first = _send(auth, "https://apis.example/a")
    again = _send(auth, "https://apis.example/b")
    clock.now = _ISSUED_AT + _LIFETIME_S - REFRESH_MARGIN_S + 1
    refreshed = _send(auth, "https://apis.example/c")

    assert (first, again, refreshed) == (
        "https://apis.example#1",
        "https://apis.example#1",
        "https://apis.example#2",
    )


def test_the_static_token_overrides_minting():
    fetcher = _Fetcher(_Clock())
    auth = ApisAuth(static_token="from-env", minter=IdTokenMinter(fetch_token=fetcher))

    assert _send(auth, "https://apis.example/a") == "Bearer from-env"
    assert fetcher.mints == {}


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


@pytest.mark.parametrize("flow", ["sync", "async"])
def test_a_failed_mint_stops_the_run_with_the_routes_to_a_token(flow):
    """A mint failure is an `httpx.RequestError` for the data clients. The eval must stop on
    it, so it surfaces as `ApisAuthError`, which no eval path scores as infrastructure."""

    def refuse(audience: str) -> str:
        raise CloudRunAuthError(f"cannot mint an ID token for {audience}")

    async def send_async(auth: ApisAuth) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_echo()), auth=auth) as client:
            await client.get("https://apis.example/a")

    with pytest.raises(ApisAuthError, match=module.TOKEN_ENV) as raised:
        if flow == "sync":
            _send(_auth(refuse), "https://apis.example/a")
        else:
            asyncio.run(send_async(_auth(refuse)))
    assert not isinstance(raised.value, httpx.HTTPError)


def test_concurrent_async_requests_mint_once():
    fetcher = _Fetcher(_Clock(), delay_s=0.05)
    auth = _auth(fetcher)

    async def go() -> list[str | None]:
        transport = httpx.MockTransport(_echo())
        async with httpx.AsyncClient(transport=transport, auth=auth) as client:
            replies = await asyncio.gather(
                *(client.get(f"https://apis.example/{i}") for i in range(8))
            )
        return [_token_id(reply.json()["token"]) for reply in replies]

    tokens = asyncio.run(go())

    assert set(tokens) == {"https://apis.example#1"}
    assert fetcher.mints == {"https://apis.example": 1}


def test_concurrent_threads_mint_once():
    fetcher = _Fetcher(_Clock(), delay_s=0.05)
    auth = _auth(fetcher)
    tokens: list[str | None] = []

    def send() -> None:
        tokens.append(_send(auth, "https://apis.example/a"))

    threads = [threading.Thread(target=send) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tokens == ["https://apis.example#1"] * 6
    assert fetcher.mints == {"https://apis.example": 1}
