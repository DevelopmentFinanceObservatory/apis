"""Pins `POST /api/search/stated-scope`: reading order, shape and the size caps."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from search.engine.app import create_app
from search.models.stated_scope_door import STATED_SCOPE_MAX_TEXT_CHARS, STATED_SCOPE_MAX_TEXTS
from tests.engine._harness import EMPTY_SDMX
from tests.fixtures import FakeGraph, FakeLLM

_ROUTE = "/api/search/stated-scope"


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_each_text_reads_in_order(client: TestClient) -> None:
    resp = client.post(
        _ROUTE,
        json={
            "texts": [
                "total US ODA",
                "US bilateral aid to Kenya as a share of US ODA to Kenya including imputed "
                "multilateral",
                "net bilateral ODA",
            ]
        },
    )

    assert resp.status_code == 200, resp.text
    readings = resp.json()["readings"]
    assert [(r["total"], r["mentions"]) for r in readings] == [
        (True, []),
        (False, ["provider:bilateral", "provider:inclusive"]),
        (False, ["provider:bilateral"]),
    ]
    assert readings[1]["entries"] == []
    assert readings[2]["entries"] == [
        {
            "axis": "measure_basis",
            "option_id": "measure_basis:net",
            "origin": {"kind": "text", "text": "net", "start": 0, "end": 3},
        },
        {
            "axis": "provider",
            "option_id": "provider:bilateral",
            "origin": {"kind": "text", "text": "bilateral", "start": 4, "end": 13},
        },
    ]


@pytest.mark.parametrize(
    "texts",
    [
        ["x"] * (STATED_SCOPE_MAX_TEXTS + 1),
        ["x" * (STATED_SCOPE_MAX_TEXT_CHARS + 1)],
    ],
    ids=["too_many_texts", "text_too_long"],
)
def test_oversize_input_is_refused(client: TestClient, texts: list[str]) -> None:
    resp = client.post(_ROUTE, json={"texts": texts})

    assert resp.status_code == 400, resp.text
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "request_invalid"
