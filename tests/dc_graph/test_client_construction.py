"""Pins `GraphClient` constructor bounds and exception-class validation without network calls."""

from __future__ import annotations

import httpx
import pytest

import dc_graph.client as graph_client_module
from dc_graph import (
    GraphCapExceededError,
    GraphClient,
    GraphDeadlineExceededError,
    GraphReadError,
)
from dfo_platform.cloud_run_auth import CloudRunAuthError, CloudRunIdTokenAuth, IdTokenMinter


class _UnrelatedError(Exception):
    pass


def test_read_error_cls_must_be_a_graph_read_error_subclass() -> None:
    with pytest.raises(TypeError, match="read_error_cls"):
        GraphClient(base_url="http://fake/core/api", read_error_cls=_UnrelatedError)


def test_deadline_error_cls_must_be_a_graph_deadline_exceeded_error_subclass() -> None:
    with pytest.raises(TypeError, match="deadline_error_cls"):
        GraphClient(base_url="http://fake/core/api", deadline_error_cls=GraphReadError)


def test_cap_error_cls_must_be_a_graph_cap_exceeded_error_subclass() -> None:
    with pytest.raises(TypeError, match="cap_error_cls"):
        GraphClient(base_url="http://fake/core/api", cap_error_cls=_UnrelatedError)


def test_a_deadline_error_cls_subclass_is_accepted_as_a_read_error_cls_too() -> None:
    """A `GraphReadError` subclass can serve as both read and deadline error type."""
    GraphClient(
        base_url="http://fake/core/api",
        read_error_cls=GraphDeadlineExceededError,
        deadline_error_cls=GraphDeadlineExceededError,
        cap_error_cls=GraphCapExceededError,
    )


_DCP = "https://dfo-dev-dc-datacommons-service-abc123-nn.a.run.app"
# An unsigned JWT whose `exp` is far in the future; the minter reads `exp` unverified.
_ID_TOKEN = "h.eyJleHAiOiA0MTAyNDQ0ODAwfQ.s"


def _use_minter(monkeypatch: pytest.MonkeyPatch, fetch_token) -> None:
    auth = CloudRunIdTokenAuth(minter=IdTokenMinter(fetch_token=fetch_token))
    monkeypatch.setattr(graph_client_module, "cloud_run_auth", lambda: auth)


@pytest.mark.parametrize(
    ("base_url", "sent"),
    [(f"{_DCP}/core/api", f"Bearer {_ID_TOKEN}"), ("https://api.datacommons.org", None)],
)
def test_run_app_hosts_get_a_cloud_run_id_token(
    monkeypatch: pytest.MonkeyPatch, base_url: str, sent: str | None
) -> None:
    audiences: list[str] = []

    def mint(audience: str) -> str:
        audiences.append(audience)
        return _ID_TOKEN

    _use_minter(monkeypatch, mint)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    with GraphClient(base_url=base_url, transport=httpx.MockTransport(handler)) as client:
        client.post("/v2/node", {"nodes": ["geoId/06"]})

    assert [request.headers.get("x-serverless-authorization") for request in requests] == [sent]
    assert all("authorization" not in request.headers for request in requests)
    assert audiences == ([_DCP] if sent else [])


def test_a_failed_mint_is_retried_then_raised_as_the_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mint failure is a transport error. The minter remembers it, so the retries fail
    without asking again, and no request is sent."""
    audiences: list[str] = []

    def refuse(audience: str) -> str:
        audiences.append(audience)
        raise CloudRunAuthError(f"cannot mint an ID token for {audience}")

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a request without a token was sent")

    _use_minter(monkeypatch, refuse)
    monkeypatch.setattr(graph_client_module, "_RETRY_BACKOFF_SECONDS", (0.0,))
    client = GraphClient(base_url=f"{_DCP}/core/api", transport=httpx.MockTransport(unreachable))
    attempts: list[int] = []
    await_retry = client._await_retry

    def counted_retry(attempt: int, deadline: float | None) -> bool:
        attempts.append(attempt)
        return await_retry(attempt, deadline)

    monkeypatch.setattr(client, "_await_retry", counted_retry)

    with client, pytest.raises(GraphReadError, match="cannot mint an ID token"):
        client.post("/v2/node", {"nodes": ["geoId/06"]})

    assert attempts == [1, 2, 3]
    assert audiences == [_DCP]
