"""Batched `/v2/node` reads that meet the UN front door's refusal.

The UN System Data Commons front door refuses some POST bodies naming several dcids that it
serves one by one: `undata/sdg/SERIES-EN_TWT_GENV`, `undata/sdg/SERIES-EN_HAZ_GENV` and
`undata/sdg/SERIES-EN_MAT_FTPRPC` together draw its 403 page, each alone does not.
"""

from __future__ import annotations

import json

import httpx
import pytest

from dc_graph import GraphClient, GraphReadError
from graph_generation import FakeGeneration

_BASE_URL = "https://graph.example.org/core/api"
_NODE_PATH = "/core/api/v2/node"

# The UN front door's refusal page.
_FRONT_DOOR_PAGE = (
    '<!doctype html><meta charset="utf-8"><meta name=viewport '
    'content="width=device-width, initial-scale=1"><title>403</title>403 Forbidden'
)

_REFUSED_TOGETHER = frozenset(
    {
        "undata/sdg/SERIES-EN_TWT_GENV",
        "undata/sdg/SERIES-EN_HAZ_GENV",
        "undata/sdg/SERIES-EN_MAT_FTPRPC",
    }
)
_DCIDS = [
    "undataProduct",
    "undata/sdg/SERIES-EN_TWT_GENV",
    "undata/sdg/SERIES-EN_ATM_CO2",
    "undata/sdg/SERIES-EN_HAZ_GENV",
    "undata/sdg/SERIES-EN_ATM_GHGT_WLU",
    "undata/sdg/SERIES-EN_MAT_FTPRPC",
]


class _NodeServer:
    """Answer `/v2/node` POSTs with a name and a type per dcid, and refuse any body naming every
    dcid in `refused` with `status`, `page` and `content_type`."""

    def __init__(
        self,
        refused: frozenset[str],
        *,
        status: int = 403,
        page: str = _FRONT_DOOR_PAGE,
        content_type: str = "text/html; charset=UTF-8",
    ) -> None:
        self.refused = refused
        self.status = status
        self.page = page
        self.content_type = content_type
        self.posted: list[list[str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == _NODE_PATH
        nodes = json.loads(request.content)["nodes"]
        self.posted.append(nodes)
        if self.refused <= set(nodes):
            return httpx.Response(
                self.status, text=self.page, headers={"content-type": self.content_type}
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    dcid: {
                        "arcs": {
                            "name": {"nodes": [{"value": f"name of {dcid}"}]},
                            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                        }
                    }
                    for dcid in nodes
                }
            },
        )


def _client(server: _NodeServer) -> GraphClient:
    return GraphClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(server.handler),
        generation=FakeGeneration(token="rev-1"),
    )


def test_a_refused_label_batch_halves_until_every_label_is_read():
    server = _NodeServer(_REFUSED_TOGETHER)

    labels = _client(server).node_labels_batch(_DCIDS)

    assert labels == {dcid: f"name of {dcid}" for dcid in _DCIDS}
    assert server.posted[0] == _DCIDS
    assert all(not set(nodes) >= _REFUSED_TOGETHER for nodes in server.posted[1:])


def test_a_refused_arc_read_halves_until_every_node_is_read():
    server = _NodeServer(_REFUSED_TOGETHER)

    arcs = _client(server).node_arcs_many(_DCIDS, properties=["typeOf"])

    assert arcs == {dcid: {"typeOf": ("StatisticalVariable",)} for dcid in _DCIDS}
    assert len(server.posted) > 1


def test_a_refusal_naming_one_dcid_fails_the_read():
    server = _NodeServer(frozenset({"undata/sdg/SERIES-EN_ATM_CO2"}))

    with pytest.raises(GraphReadError) as excinfo:
        _client(server).node_labels_batch(_DCIDS)
    assert excinfo.value.upstream_status == 403
    assert server.posted[-1] == ["undata/sdg/SERIES-EN_ATM_CO2"]


@pytest.mark.parametrize(
    ("status", "page", "content_type"),
    [
        (403, '{"error": "forbidden"}', "application/json"),
        (403, _FRONT_DOOR_PAGE, "text/plain"),
        (401, _FRONT_DOOR_PAGE, "text/html"),
    ],
)
def test_any_other_error_fails_the_read_without_halving(status: int, page: str, content_type: str):
    server = _NodeServer(_REFUSED_TOGETHER, status=status, page=page, content_type=content_type)

    with pytest.raises(GraphReadError) as excinfo:
        _client(server).node_labels_batch(_DCIDS)
    assert excinfo.value.upstream_status == status
    assert server.posted == [_DCIDS]
