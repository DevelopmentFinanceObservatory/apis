"""Every graph read of a role-class walk, and of its members' arcs after it, ends by the
walk's bound.

The reads go through the client the served app builds (`LiveGraphClient`) over a mocked
transport that records when each request was sent and the timeout it carried.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest

import search.engine.clients.graph_client as graph_client_module
from origin_catalogue.graph import build_graph_client
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.retrieval.bound_values import (
    MembershipAuthority,
    RoleClassWalks,
    elected_role_class_shape,
    shape_query,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine.test_role_class_shapes import (
    _DEBT_NODES,
    _TOTAL,
    _TOTAL_GUARANTEED,
    _debt_shape,
    _walk_arcs,
)

_HTTP_TIMEOUT_S = 30.0
_PAGE_LATENCY_S = 0.02


@dataclass(frozen=True)
class _Sent:
    at: float
    timeout: float
    property: str


class _Transport:
    """Answers `/v2/node`: a reverse query in four pages, each `_PAGE_LATENCY_S` late, and
    `->*` reads from the debt fixture."""

    def __init__(self) -> None:
        self.sent: list[_Sent] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.sent.append(
            _Sent(
                at=time.monotonic(),
                timeout=request.extensions["timeout"]["read"],
                property=payload["property"],
            )
        )
        if payload["property"].startswith("<-"):
            time.sleep(_PAGE_LATENCY_S)
            page = int(payload.get("nextToken", "0"))
            (lead,) = payload["nodes"]
            body: dict = {
                "data": {
                    lead: {"arcs": {"measuredProperty": {"nodes": [{"dcid": f"Page{page}_SV"}]}}}
                }
            }
            if page < 3:
                body["nextToken"] = str(page + 1)
            return httpx.Response(200, json=body)
        if payload["property"] == "->*":
            return httpx.Response(
                200,
                json={
                    "data": {
                        dcid: {"arcs": _DEBT_NODES[dcid]["arcs"]}
                        for dcid in payload["nodes"]
                        if dcid in _DEBT_NODES
                    }
                },
            )
        raise AssertionError(f"unexpected read {payload!r}")


@pytest.fixture
def transport() -> _Transport:
    return _Transport()


@pytest.fixture
def live_graph(transport: _Transport, monkeypatch: pytest.MonkeyPatch) -> Iterator[LiveGraphClient]:
    monkeypatch.setattr(
        graph_client_module,
        "build_graph_client",
        functools.partial(build_graph_client, transport=httpx.MockTransport(transport.handler)),
    )
    client = LiveGraphClient(base="https://graph.test", timeout=_HTTP_TIMEOUT_S, token="")
    yield client
    client.close()


def _assert_bounded(sent: list[_Sent], *, started: float, bound: float) -> None:
    """Every request was sent before `bound`, with a timeout no longer than the time that
    remained when the read began at `started`."""
    for request in sent:
        assert request.at < bound
        assert request.timeout <= bound - started


def test_a_paginated_reverse_query_sends_no_page_past_the_walks_bound(live_graph, transport):
    """A 5 ms bound against pages that each take 20 ms: the first page carries a timeout
    within the bound and no later page is sent, so the walk settles nothing."""
    shape = _debt_shape(_TOTAL_GUARANTEED)
    started = time.monotonic()
    bound = started + 0.005

    resolved = RoleClassWalks(live_graph).family(
        shape_query(shape, [], ["destination"]), (), deadline=bound
    )

    assert resolved.authority is MembershipAuthority.FAILED
    assert len(transport.sent) == 1
    _assert_bounded(transport.sent, started=started, bound=bound)


def _walk_reader() -> FakeMintGraphReader:
    return FakeMintGraphReader(arcs={dcid: _walk_arcs(node) for dcid, node in _DEBT_NODES.items()})


def test_the_read_of_the_walked_familys_arcs_carries_the_walks_bound(live_graph, transport):
    shape = _debt_shape(_TOTAL_GUARANTEED)
    started = time.monotonic()
    bound = started + 2.0

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], ["destination"]),
        walks=RoleClassWalks(_walk_reader()),
        arcs_graph=live_graph,
        deadline=bound,
    )

    assert resolution.authority is MembershipAuthority.AUTHORITATIVE
    assert set(resolution.value.sv_arc_facts or {}) == {_TOTAL, _TOTAL_GUARANTEED}
    assert [request.property for request in transport.sent] == ["->*"]
    _assert_bounded(transport.sent, started=started, bound=bound)


def test_a_walk_whose_bound_has_passed_reads_no_arcs_after_it(live_graph, transport):
    """The walk settles inside its bound and is kept for the request. Asked again once the
    bound has passed, the family's arcs are not read and the sample answers."""
    shape = _debt_shape(_TOTAL_GUARANTEED)
    query = shape_query(shape, [], ["destination"])
    walks = RoleClassWalks(_walk_reader())
    bound = time.monotonic() + 0.05
    walked = walks.family(query, (_TOTAL_GUARANTEED,), deadline=bound)
    assert walked.authority is MembershipAuthority.AUTHORITATIVE
    time.sleep(max(0.0, bound - time.monotonic()) + 0.01)

    resolution = elected_role_class_shape(
        shape, query, walks=walks, arcs_graph=live_graph, deadline=bound
    )

    assert resolution.authority is MembershipAuthority.FAILED
    assert resolution.value is shape
    assert transport.sent == []
