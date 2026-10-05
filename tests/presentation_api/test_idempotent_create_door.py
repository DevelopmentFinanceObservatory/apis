"""The create body's `write_token` keys the create: a retry with the same token and request
answers 201 with the first create's reply, and every refusal of the token is 400
`request_invalid` at `/write_token`.
"""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from dfo_platform import ProblemDetails

from .test_write_token_door import _LINE_CHART_INTENT

_CREATE_ROUTE = "/api/presentation/components"


def _client_token() -> str:
    return f"cmw_{secrets.token_urlsafe(16)}"


def _post(client: TestClient, **body):
    return client.post(_CREATE_ROUTE, json={"intent": _LINE_CHART_INTENT, **body})


def _assert_request_invalid(resp, *, pointer: str) -> dict:
    assert resp.status_code == 400, resp.json()
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == pointer
    assert body["retryable"] is False
    ProblemDetails.model_validate(body)
    return body


def test_a_retried_create_answers_201_with_the_first_reply(client: TestClient):
    token = _client_token()

    first = _post(client, write_token=token)
    again = _post(client, write_token=token)

    assert first.status_code == 201, first.json()
    assert again.status_code == 201, again.json()
    assert again.json() == first.json()
    assert first.json()["write_token"] == token


def test_the_body_token_grants_revise(client: TestClient):
    token = _client_token()
    created = _post(client, write_token=token).json()

    resp = client.patch(
        f"{_CREATE_ROUTE}/{created['component_id']}",
        json={"intent": {"title": "A new title"}},
        headers={"x-dfo-write-token": token},
    )

    assert resp.status_code == 200, resp.json()


def test_the_same_token_with_another_intent_is_refused(client: TestClient):
    token = _client_token()
    assert _post(client, write_token=token).status_code == 201

    resp = client.post(
        _CREATE_ROUTE,
        json={"intent": {**_LINE_CHART_INTENT, "title": "Another title"}, "write_token": token},
    )

    body = _assert_request_invalid(resp, pointer="/write_token")
    assert "mint a new write_token" in body["detail"]


def test_a_token_with_a_new_composition_is_refused(client: TestClient):
    resp = _post(client, write_token=_client_token(), composition_id="new")

    _assert_request_invalid(resp, pointer="/write_token")


@pytest.mark.parametrize("token", ["cmw_short", "cpw_" + "A" * 22, "cmw_" + "A" * 21 + "="])
def test_a_malformed_token_is_refused(client: TestClient, token: str):
    _assert_request_invalid(_post(client, write_token=token), pointer="/write_token")


def test_a_non_string_token_is_refused(client: TestClient):
    _assert_request_invalid(_post(client, write_token=123), pointer="/write_token")


def test_a_revise_body_carrying_a_token_is_refused(client: TestClient):
    token = _client_token()
    created = _post(client, write_token=token).json()

    resp = client.patch(
        f"{_CREATE_ROUTE}/{created['component_id']}",
        json={"intent": {"title": "A new title"}, "write_token": token},
        headers={"x-dfo-write-token": token},
    )

    _assert_request_invalid(resp, pointer="/write_token")
