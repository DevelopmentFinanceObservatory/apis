"""The buffered resolve route reports a degraded leg in a header, outside the body and SSE."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from search.engine.app import DEGRADED_LEG_HEADER
from search.engine.entry import raw_text
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
    """Run the real raw-text pipeline after noting a degraded leg on its cache scope."""
    real_handle = raw_text.handle

    def degrading_handle(*args, scope, **kwargs):
        scope.note_health(timed_out=False, degraded_leg=True)
        return real_handle(*args, scope=scope, **kwargs)

    monkeypatch.setattr(raw_text, "handle", degrading_handle)


def test_a_healthy_answer_carries_no_degraded_header(client):
    response = client.post("/api/search/resolve", json=_REQUEST)

    assert response.status_code == 200
    assert DEGRADED_LEG_HEADER not in response.headers


@pytest.mark.usefixtures("degrade")
def test_a_degraded_leg_sets_the_header_and_leaves_the_body_unchanged(client):
    response = client.post("/api/search/resolve", json=_REQUEST)

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
