"""A catalogue or declined question naming many places answers with its kind, never a 400.
Past `max_entities` distinct places extract clears its entities, so it makes no resolver
call for its `scope.place`."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from search.engine.config import settings
from search.models import CatalogueAnswer
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.engine.test_dispositions import (
    _events,
    _extraction,
    _ExtractOnlyLLM,
    _response,
    _StubResolver,
)
from tests.fixtures import FakeGraph


def _places(count: int) -> list[str]:
    return [f"Place {i}" for i in range(count)]


def _post(kind: str, entities: list[str], resolver: _StubResolver):
    llm = _ExtractOnlyLLM(_extraction(kind, entities=entities))
    client = TestClient(
        search_app(graph=FakeGraph(), resolver=resolver, llm=llm, sdmx=FakeSdmxClient()),
        raise_server_exceptions=True,
    )
    return client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "What about all of these places?"}},
    )


@pytest.mark.parametrize(
    ("kind", "wire_kind"), [("out_of_scope", "declined"), ("catalogue", "catalogue")]
)
def test_a_question_past_the_place_cap_answers_200_with_no_place_and_no_resolver_call(
    kind: str, wire_kind: str
):
    resolver = _StubResolver()

    resp = _post(kind, _places(settings().max_entities + 1), resolver)

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == wire_kind
    assert body["scope"].get("place") is None
    assert resolver.calls == []


def test_a_question_at_the_place_cap_still_sends_its_places_to_the_resolver():
    resolver = _StubResolver()
    places = _places(settings().max_entities)

    resp = _post("out_of_scope", places, resolver)

    assert resp.status_code == 200
    assert resolver.calls == [places]


def test_a_catalogue_question_naming_two_countries_has_no_place():
    llm = _ExtractOnlyLLM(_extraction("catalogue", entities=["Kenya", "Uganda"]))

    response = _response(
        _events("What data do you have on Kenya and Uganda?", llm, _StubResolver())
    )

    assert isinstance(response, CatalogueAnswer)
    assert response.scope.place is None
