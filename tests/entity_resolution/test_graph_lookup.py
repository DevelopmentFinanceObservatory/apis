"""GraphClient satisfies GraphLookup and sends the expected resolve_by_arc wire shape."""

from __future__ import annotations

import json

import httpx

from dc_graph import GraphClient
from entity_resolution.types import GraphLookup


def test_graph_client_satisfies_graph_lookup():
    reader = GraphClient(base_url="http://x")
    assert isinstance(reader, GraphLookup)


def test_resolve_by_arc_request_and_response_shape():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = {
            "entities": [
                {"node": "Kenya", "candidates": [{"dcid": "country/KEN"}]},
                {"node": "Nowhere", "candidates": []},
            ]
        }
        return httpx.Response(200, json=body)

    reader = GraphClient(
        base_url="https://graph.example.org/core/api", transport=httpx.MockTransport(handler)
    )

    result = reader.resolve_by_arc(["Kenya", "Nowhere"], arc="<-description{typeOf:Country}->dcid")

    assert result == {"Kenya": ["country/KEN"], "Nowhere": []}
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/resolve"
    payload = json.loads(request.content)
    assert payload == {
        "nodes": ["Kenya", "Nowhere"],
        "property": "<-description{typeOf:Country}->dcid",
    }


def test_resolve_by_arc_keys_every_requested_name_even_when_response_omits_one():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"entities": []})

    reader = GraphClient(base_url="https://x", transport=httpx.MockTransport(handler))

    result = reader.resolve_by_arc(["Ghost"], arc="<-description->dcid")

    assert result == {"Ghost": []}


def test_resolve_by_arc_empty_names_makes_no_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("resolve_by_arc must not call the transport for empty names")

    reader = GraphClient(base_url="https://x", transport=httpx.MockTransport(handler))

    assert reader.resolve_by_arc([], arc="<-description->dcid") == {}
