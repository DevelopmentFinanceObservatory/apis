"""Tests for `GET /api/graph/schema/node/{dcid}`: which deployment each dcid is read from, the
refusals, and the cached body's HTTP caching."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence

import httpx
import pytest

from dc_graph import ArcTarget, GraphClient
from graph_api import app as graph_app
from graph_api.deps import DcpExecutorSaturated
from materialise.graph import GraphDeadlineExceededError
from origin_catalogue.graph import UnGeneration
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import make_client
from .test_node_view import _ARCS, _BASE, _NAMES, _PROVENANCES, _FakeGraph

_ROUTE = "/api/graph/schema/node/"
_IDENTITY = {"Accept-Encoding": "identity"}

# The mixer holds the base and DFO terms. The UN deployment holds its own nodes and `unCode`,
# a UN property with no `undata` prefix.
_MIXER_ARCS = tuple(arc for arc in _ARCS if not arc[0].startswith("undata"))
_UN_ARCS = (
    *(arc for arc in _ARCS if arc[0].startswith("undata")),
    ("Thing", "typeOf", "Class"),
    ("Intangible", "typeOf", "Class"),
    ("Intangible", "subClassOf", "Thing"),
    ("Enumeration", "typeOf", "Class"),
    ("Enumeration", "subClassOf", "Intangible"),
    ("unCode", "typeOf", "Property"),
)


def _out_reads(graph: _FakeGraph) -> int:
    """The `->*` reads `graph` answered. A view of a node the graph lacks cancels or drops its
    other round-one reads, so only these count the same on every run."""
    return sum(method == "->*" for method, _, _ in graph.calls)


def _refusal(response) -> tuple[int, str, bool, str, str]:
    body = response.json()
    return (
        response.status_code,
        body["code"],
        body["retryable"],
        body["cause"]["code"],
        response.headers["Cache-Control"],
    )


def test_a_base_class_is_read_from_the_mixer_and_cached(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "Person", headers=_IDENTITY)
        reads = len(mixer.calls)
        gzipped = client.get(_ROUTE + "Person", headers={"Accept-Encoding": "gzip"})
        not_modified = client.get(
            _ROUTE + "Person", headers={**_IDENTITY, "If-None-Match": response.headers["ETag"]}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert sorted(body) == ["capped", "centre", "generation", "nodes"]
    assert body["centre"]["ref"] == "Person"
    assert body["centre"]["declared_by"] == "base"
    assert [node["ref"] for node in body["nodes"]] == ["Class", "Student", "Thing", "age", "gender"]
    assert body["capped"] == []
    assert response.headers["Cache-Control"] == "public, max-age=300"
    assert gzipped.headers["Content-Encoding"] == "gzip"
    assert gzipped.json() == body
    assert not_modified.status_code == 304
    # Three reads build the view once. The later requests read nothing.
    assert reads == 3
    assert len(mixer.calls) == reads
    assert un.calls == []


def test_a_un_series_is_read_from_the_un_deployment(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "undata/eclac/SI_POV")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["generation"] is None
    assert body["centre"]["declared_by"] == "un_system"
    series = next(node for node in body["nodes"] if node["ref"] == "undata/eclac/SERIES")
    assert series["instances"] == {"refs": ["undata/eclac/SI_POV"], "complete": False}
    assert mixer.calls == []
    assert un.calls


@pytest.mark.parametrize("dcid", ["undata/eclac/SI_POV", "unCode"])
@pytest.mark.parametrize("un_reader", [True, False], ids=["un-off", "no-un-reader"])
def test_the_un_deployment_is_never_read_where_the_un_origin_is_not_served(
    serve_catalogue, dcid: str, un_reader: bool
):
    """A UN id is refused `origin_not_served` with no read; a non-UN id the mixer lacks is
    `node_not_held` without falling to the UN deployment."""
    serve_catalogue(FakeCatalogue(un_system_enabled=not un_reader))
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un if un_reader else None) as client:
        response = client.get(_ROUTE + dcid)

    un_id = dcid.startswith("undata/")
    assert _refusal(response) == (
        404,
        "not_found",
        False,
        "origin_not_served" if un_id else "node_not_held",
        "no-store",
    )
    assert un.calls == []
    assert (mixer.calls == []) is un_id


def test_a_node_the_mixer_lacks_is_read_from_the_un_deployment(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "unCode")
        again = client.get(_ROUTE + "unCode")

    assert response.status_code == again.status_code == 200, response.text
    assert response.json()["centre"]["types"] == ["Property"]
    assert response.json()["generation"] is None
    # The UN view is kept, and so is the mixer's miss, so neither is read again.
    assert (_out_reads(mixer), _out_reads(un)) == (1, 1)


def test_a_node_no_deployment_holds_is_refused_and_not_kept(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "NoSuchNode")
        again = client.get(_ROUTE + "NoSuchNode")

    assert _refusal(response) == _refusal(again)
    assert _refusal(response) == (404, "not_found", False, "node_not_held", "no-store")
    # The mixer's miss is kept under its generation. The UN deployment's refusal is not.
    assert (_out_reads(mixer), _out_reads(un)) == (1, 2)


def _wire_entry(subject: str, value: str, arcs: Sequence[tuple[str, str, str]]) -> dict:
    return {
        "dcid": value,
        "name": _NAMES.get(value, value),
        "types": sorted(o for s, p, o in arcs if s == value and p == "typeOf"),
        "provenanceId": _PROVENANCES.get(subject, _BASE),
    }


def _wire(arcs: Sequence[tuple[str, str, str]]) -> httpx.MockTransport:
    """`arcs` as the `/v2/node` endpoint answers `->*`, `->p`, `->p+` and `<-[p]`, in one
    page."""

    def out(node: str, props: Sequence[str] | None) -> dict:
        found: dict[str, list[dict]] = {}
        for s, p, o in arcs:
            if s == node and (props is None or p in props):
                found.setdefault(p, []).append(_wire_entry(s, o, arcs))
        if node in _NAMES and props is None:
            found["name"] = [{"value": _NAMES[node]}]
        return found

    def closure(node: str, prop: str) -> list[dict]:
        seen: list[str] = []
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for s, p, o in arcs:
                if s == current and p == prop and o not in seen:
                    seen.append(o)
                    frontier.append(o)
        return [_wire_entry(node, value, arcs) for value in seen]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        spec: str = payload["property"]
        data: dict = {}
        for node in payload["nodes"]:
            if spec == "->*":
                found = out(node, None)
            elif spec.startswith("->") and spec.endswith("+"):
                found = {spec[2:]: closure(node, spec[2:-1])}
            elif spec.startswith("->"):
                found = out(node, spec[2:].strip("[]").split(","))
            elif spec.startswith("<-["):
                found = {
                    prop: [_wire_entry(s, s, arcs) for s, p, o in arcs if p == prop and o == node]
                    for prop in spec[3:-1].split(",")
                }
            else:
                raise AssertionError(f"unexpected property expression {spec!r}")
            found = {prop: {"nodes": nodes} for prop, nodes in found.items() if nodes}
            data[node] = {"arcs": found} if found else {}
        return httpx.Response(200, json={"data": data})

    return httpx.MockTransport(handler)


def test_a_un_class_below_enumeration_reads_through_the_un_client(serve_catalogue):
    """The UN deployment's own client stamps its cache with the bound catalogue's build, so each
    read runs under the request's context. The class undata/eclac/SERIES reaches `Enumeration` through
    undata/Series, and the chain's edges read through the client's generation-stamped cache."""
    serve_catalogue(FakeCatalogue())
    un = GraphClient(
        base_url="https://un.example.org/core/api",
        transport=_wire(_UN_ARCS),
        generation=UnGeneration(),
    )

    with un, make_client(view_graph=_FakeGraph(_MIXER_ARCS), un_view_graph=un) as client:
        response = client.get(_ROUTE + "undata/eclac/SERIES")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["centre"]["instances"] == {"refs": ["undata/eclac/SI_POV"], "complete": True}
    chain = next(node for node in body["nodes"] if node["ref"] == "undata/Series")
    assert chain["sub_class_of"] == ["Enumeration"]


class _Failing(_FakeGraph):
    """`_FakeGraph` whose `->*` read raises `error`."""

    def __init__(self, error: Exception) -> None:
        super().__init__(_MIXER_ARCS)
        self.error = error

    def node_all_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls.append(("->*", tuple(dcids), ()))
        raise self.error


@pytest.mark.parametrize(
    ("dcid", "error", "status", "code"),
    [
        pytest.param("a" * 513, None, 400, "request_invalid", id="dcid-too-long"),
        pytest.param("Person", GraphDeadlineExceededError("late"), 504, "timeout", id="deadline"),
        pytest.param("Person", DcpExecutorSaturated(), 503, "service_unavailable", id="busy"),
    ],
)
def test_a_failed_view_answers_its_refusal_and_is_not_kept(
    serve_catalogue, dcid: str, error: Exception | None, status: int, code: str
):
    serve_catalogue(FakeCatalogue())
    mixer = _FakeGraph(_MIXER_ARCS) if error is None else _Failing(error)

    with make_client(view_graph=mixer, un_view_graph=_FakeGraph(_UN_ARCS)) as client:
        response = client.get(_ROUTE + dcid)
        reads = _out_reads(mixer)
        client.get(_ROUTE + dcid)

    assert (response.status_code, response.json()["code"]) == (status, code)
    assert response.headers["Cache-Control"] == "no-store"
    # A failed read is not kept, so the next request reads again. A long dcid reads nothing.
    assert reads == (0 if error is None else 1)
    assert _out_reads(mixer) == 2 * reads


class _Blocking(_FakeGraph):
    """`_FakeGraph` whose `->*` read of `Person` waits for `release`."""

    def __init__(self) -> None:
        super().__init__(_MIXER_ARCS)
        self.entered = threading.Event()
        self.release = threading.Event()

    def node_all_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        if "Person" in dcids:
            self.entered.set()
            assert self.release.wait(5), "the blocked read was never released"
        return super().node_all_arc_targets_many(dcids, max_bytes=max_bytes, deadline=deadline)


def test_a_miss_past_every_flight_worker_is_refused_at_once(serve_catalogue, monkeypatch):
    """With every flight worker building a view, a miss for another node answers 503 without
    queueing, and the running view still answers."""
    monkeypatch.setattr(graph_app, "NODE_VIEW_FLIGHT_WORKERS", 1)
    serve_catalogue(FakeCatalogue())
    mixer = _Blocking()

    with make_client(view_graph=mixer, un_view_graph=_FakeGraph(_UN_ARCS)) as client:
        first: list[httpx.Response] = []
        running = threading.Thread(target=lambda: first.append(client.get(_ROUTE + "Person")))
        running.start()
        try:
            assert mixer.entered.wait(5)
            refused = client.get(_ROUTE + "GenderType")
        finally:
            mixer.release.set()
            running.join(5)

    assert (refused.status_code, refused.json()["code"]) == (503, "service_unavailable")
    assert refused.json()["retryable"] is True
    assert ("->*", ("GenderType",), ()) not in mixer.calls
    assert first[0].status_code == 200, first[0].text
