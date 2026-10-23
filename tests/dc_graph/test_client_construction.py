"""Pins `GraphClient` constructor bounds and exception-class validation without network calls."""

from __future__ import annotations

import httpx
import pytest

from dc_graph import (
    GraphCapExceededError,
    GraphClient,
    GraphDeadlineExceededError,
    GraphReadError,
)


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


# An unsigned JWT whose `aud` is the run.app origin below; the auth reads `aud` unverified.
_ID_TOKEN = "h.eyJhdWQiOiJodHRwczovL2Rmby1kZXYtZGMtZGF0YWNvbW1vbnMtc2VydmljZS1hYmMxMjMtbm4uYS5ydW4uYXBwIn0.s"


@pytest.mark.parametrize(
    ("base_url", "sent"),
    [
        (
            "https://dfo-dev-dc-datacommons-service-abc123-nn.a.run.app/core/api",
            f"Bearer {_ID_TOKEN}",
        ),
        ("https://api.datacommons.org", None),
    ],
)
def test_run_app_hosts_get_a_cloud_run_id_token(
    monkeypatch: pytest.MonkeyPatch, base_url: str, sent: str | None
) -> None:
    monkeypatch.setenv("DCP_ID_TOKEN", _ID_TOKEN)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    with GraphClient(base_url=base_url, transport=httpx.MockTransport(handler)) as client:
        client.post("/v2/node", {"nodes": ["geoId/06"]})

    assert [request.headers.get("x-serverless-authorization") for request in requests] == [sent]
    assert all("authorization" not in request.headers for request in requests)
