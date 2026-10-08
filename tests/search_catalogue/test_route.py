"""Pins `GET /api/search/catalogue` through the search surface's lifespan,
with `SEARCH_CATALOGUE_URI` set, pointing at nothing, and unset."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dfo_platform.named import MAX_VALUE_CHARS
from search.engine.app import create_app
from search.models import ResolveRequest, ResolveResponse
from tests.engine._harness import EMPTY_SDMX
from tests.fixtures import FakeGraph, FakeLLM
from tests.search_catalogue.support import definite, publish_catalogue

_ROUTE = "/api/search/catalogue"


def _client(monkeypatch: pytest.MonkeyPatch, uri: str | None) -> Iterator[TestClient]:
    if uri is None:
        monkeypatch.delenv("SEARCH_CATALOGUE_URI", raising=False)
    else:
        monkeypatch.setenv("SEARCH_CATALOGUE_URI", uri)
    app = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    uri = tmp_path.as_uri()
    publish_catalogue(uri)
    yield from _client(monkeypatch, uri)


@pytest.fixture
def empty_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(monkeypatch, tmp_path.as_uri())


@pytest.fixture
def unset_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(monkeypatch, None)


def _assert_problem(resp, *, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["code"] == code
    return body


def test_serves_the_global_counts_and_every_example(client: TestClient) -> None:
    resp = client.get(_ROUTE)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "catalogue_version": "20260930T000000Z-00000001",
        "counts": {
            "indicators": 9,
            "datasets": 5,
            "publishers": 4,
            "countries": 2,
            "aid": {"donors": 3, "recipients": 5, "years": {"first": 1990, "last": 2024}},
        },
        "examples": [
            {
                "text": "Population of Ethiopia",
                "template": "Population of {place}",
                "stat_var": "Count_Person",
                "place": "country/ETH",
            },
            {
                "text": "Health aid to Ghana",
                "template": "Health aid to {place}",
                "stat_var": "Amount_Health",
                "place": "country/GHA",
            },
        ],
    }


def test_sends_a_five_minute_cache_control(client: TestClient) -> None:
    assert client.get(_ROUTE).headers["cache-control"] == "public, max-age=300"
    assert (
        client.get(_ROUTE, params={"place": "country/KEN"}).headers["cache-control"]
        == "public, max-age=300"
    )


def test_place_scopes_the_counts_and_fills_its_examples(client: TestClient) -> None:
    resp = client.get(_ROUTE, params={"place": "country/KEN"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "catalogue_version": "20260930T000000Z-00000001",
        "counts": {
            "publishers": 2,
            "variables": 7,
            "aid": {"donors": 2, "recipients": 0, "years": {"first": 2000, "last": 2024}},
        },
        "examples": [
            {
                "text": "Population of Kenya",
                "template": "Population of {place}",
                "stat_var": "Count_Person",
                "place": "country/KEN",
            }
        ],
    }


def test_a_country_without_a_name_has_counts_and_no_examples(client: TestClient) -> None:
    body = client.get(_ROUTE, params={"place": "country/TUV"}).json()
    assert body["counts"]["variables"] == 1
    assert body["counts"]["aid"]["years"] is None
    assert body["examples"] == []


def test_a_country_the_catalogue_does_not_hold_is_request_invalid(client: TestClient) -> None:
    resp = client.get(_ROUTE, params={"place": "country/ZZZ"})
    body = _assert_problem(resp, status=400, code="request_invalid")
    assert body["detail"] == "The catalogue holds no data for place 'country/ZZZ'."
    assert body["retryable"] is False


def test_a_long_place_the_catalogue_does_not_hold_is_not_echoed_whole(client: TestClient) -> None:
    place = "country/" + "Z" * 4000
    resp = client.get(_ROUTE, params={"place": place})
    _assert_problem(resp, status=400, code="request_invalid")
    assert place[:MAX_VALUE_CHARS] in resp.text
    assert place[: MAX_VALUE_CHARS + 1] not in resp.text


@pytest.mark.parametrize("place", ["geoId/06", "africa", "country/", ""])
def test_a_place_that_is_not_a_country_dcid_is_request_invalid(
    client: TestClient, place: str
) -> None:
    resp = client.get(_ROUTE, params={"place": place})
    _assert_problem(resp, status=400, code="request_invalid")


def test_nothing_published_is_a_retryable_503(empty_client: TestClient) -> None:
    resp = empty_client.get(_ROUTE)
    body = _assert_problem(resp, status=503, code="service_unavailable")
    assert body["retryable"] is True
    assert resp.headers.get("cache-control") != "public, max-age=300"


def test_unset_variable_serves_resolve_and_refuses_the_catalogue(
    unset_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def resolve_async(request: ResolveRequest, **_deps: object) -> ResolveResponse:
        return definite("Count_Person")

    monkeypatch.setattr("search.engine.app.resolve_async", resolve_async)
    resp = unset_client.post(
        "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "population"}}
    )
    assert resp.status_code == 200, resp.text
    assert unset_client.get("/api/search/healthz").status_code == 200
    body = _assert_problem(unset_client.get(_ROUTE), status=503, code="service_unavailable")
    assert body["retryable"] is True


def test_openapi_describes_the_catalogue_body(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"][_ROUTE]["get"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/CatalogueResponse"}
    assert [parameter["name"] for parameter in operation["parameters"]] == ["place"]
