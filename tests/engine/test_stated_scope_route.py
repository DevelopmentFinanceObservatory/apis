"""Pins `POST /api/search/stated-scope`: reading order, shape and the size caps."""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from search.engine.app import create_app, read_stated_scopes
from search.engine.config import SEARCH_MAX_STATED_SCOPE_TEXTS, settings
from search.models.stated_scope_door import STATED_SCOPE_MAX_TEXTS
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
    assert [r["sums_providers"] for r in readings] == [False, False, False]
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
        ["x" * (settings().max_query_chars + 1)],
    ],
    ids=["too_many_texts", "text_too_long"],
)
def test_oversize_input_is_refused(client: TestClient, texts: list[str]) -> None:
    resp = client.post(_ROUTE, json={"texts": texts})

    assert resp.status_code == 400, resp.text
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "request_invalid"


def test_the_text_cap_is_the_raw_text_query_limit(client: TestClient) -> None:
    at_limit = client.post(_ROUTE, json={"texts": ["x" * settings().max_query_chars]})

    assert at_limit.status_code == 200, at_limit.text


def test_the_text_count_literal_matches_the_config_cap() -> None:
    assert STATED_SCOPE_MAX_TEXTS == SEARCH_MAX_STATED_SCOPE_TEXTS


def test_a_text_is_read_normalised(client: TestClient) -> None:
    resp = client.post(
        _ROUTE,
        json={
            "texts": [
                "US  aid,  including   imputed  multilateral",
                "\uff42\uff49\uff4c\uff41\uff54\uff45\uff52\uff41\uff4c aid",
                "aid including" + " " * 300 + "imputed multilateral",
            ]
        },
    )

    assert resp.status_code == 200, resp.text
    first, second, third = (r["entries"][0] for r in resp.json()["readings"])
    # Offsets index the single-spaced text: "US aid, including imputed multilateral".
    assert first["origin"] == {
        "kind": "text",
        "text": "including imputed multilateral",
        "start": 8,
        "end": 38,
    }
    assert second["option_id"] == "provider:bilateral"
    assert second["origin"]["text"] == "bilateral"
    assert third["option_id"] == "provider:inclusive"


def test_a_scope_phrase_too_long_to_quote_is_refused(client: TestClient) -> None:
    resp = client.post(_ROUTE, json={"texts": ["fine", "not " + "m" * 300 + " multilateral aid"]})

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"


def test_summed_provider_words_are_marked(client: TestClient) -> None:
    resp = client.post(_ROUTE, json={"texts": ["bilateral plus multilateral ODA"]})

    assert resp.json()["readings"][0]["sums_providers"] is True


def test_the_read_runs_off_the_event_loop() -> None:
    assert not inspect.iscoroutinefunction(read_stated_scopes)


def test_the_projection_chip_prompt_reads_as_a_projection_over_its_years(
    client: TestClient,
) -> None:
    resp = client.post(
        _ROUTE, json={"texts": ["Show projected external debt service for 2026 to 2030"]}
    )

    assert resp.status_code == 200, resp.text
    (reading,) = resp.json()["readings"]
    assert (reading["named_years"], reading["projection"]) == ([2026, 2030], True)
