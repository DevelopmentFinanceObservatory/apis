"""The buffered resolve route reports a degraded leg in a header, outside the body and SSE."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from search.engine import fanout
from search.engine.app import DEGRADED_LEG_HEADER
from tests.engine._harness import EMPTY_SDMX, PINNED_DATE, parse_sse
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_REQUEST = {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}}


@pytest.fixture
def client() -> Iterator[TestClient]:
    graph = FakeGraph()
    app = search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=EMPTY_SDMX)
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        yield TestClient(app)


@pytest.fixture
def degrade(monkeypatch) -> None:
    """Run the real raw-text pipeline with its leg reporting a swallowed read failure."""
    real_leg = fanout.resolve_variable

    async def degraded_leg(*args, **kwargs):
        return dataclasses.replace(await real_leg(*args, **kwargs), degraded=True)

    monkeypatch.setattr(fanout, "resolve_variable", degraded_leg)


def test_a_healthy_answer_carries_no_degraded_header(client):
    response = client.post("/api/search/resolve", json=_REQUEST)

    assert response.status_code == 200
    assert DEGRADED_LEG_HEADER not in response.headers


@pytest.mark.usefixtures("degrade")
@pytest.mark.parametrize(
    "request_body",
    [
        pytest.param(_REQUEST, id="raw_text"),
        pytest.param(
            {
                "input": {
                    "kind": "parsed",
                    "parts": [{"variable_text": "fertility rate", "entities": [{"text": "Kenya"}]}],
                }
            },
            id="parsed",
        ),
    ],
)
def test_a_degraded_leg_sets_the_header_and_leaves_the_body_unchanged(client, request_body):
    response = client.post("/api/search/resolve", json=request_body)

    assert response.status_code == 200
    assert response.headers[DEGRADED_LEG_HEADER] == "1"
    assert response.json()["parts"][0]["status"] == "definite"
    assert "degraded_leg" not in response.text


@pytest.mark.usefixtures("degrade")
def test_a_degraded_leg_stays_out_of_the_sse_done_payload(client):
    response = client.post(
        "/api/search/resolve", json=_REQUEST, headers={"Accept": "text/event-stream"}
    )

    frames = parse_sse(response.text)
    (done,) = [data for kind, data in frames if kind == "done"]
    assert "degraded_leg" not in done
    assert DEGRADED_LEG_HEADER not in response.headers
