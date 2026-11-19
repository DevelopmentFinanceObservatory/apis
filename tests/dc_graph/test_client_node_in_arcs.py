"""Tests the per-property bracket in-arc read `node_in_arcs` and the transitive read `node_closure`."""

import pytest

from dc_graph import ArcTarget, GraphClient, InArcs, InArcSubject
from dc_graph.errors import GraphReadError

_PROPS = ["subClassOf", "rangeIncludes"]


def _client(monkeypatch, pages: list[dict], sent: list[dict]) -> GraphClient:
    """A client whose `/v2/node` POSTs answer `pages` in order and record each payload."""
    answers = iter(pages)

    def fake_post(self, path, payload, *, deadline=None):
        sent.append(payload)
        return next(answers)

    monkeypatch.setattr(GraphClient, "post", fake_post)
    return GraphClient(base_url="http://fake/core/api")


def _entry(dcid: str, name: str | None = None, types=("Class",), prov: str | None = "dc/p1"):
    entry: dict = {"dcid": dcid, "types": list(types)}
    if name is not None:
        entry["name"] = name
    if prov is not None:
        entry["provenanceId"] = prov
    return entry


def _arcs(node: str, **by_property: list[dict]) -> dict:
    return {"data": {node: {"arcs": {p: {"nodes": n} for p, n in by_property.items()}}}}


def test_node_in_arcs_pages_and_carries_name_types_and_property(monkeypatch) -> None:
    sent: list[dict] = []
    pages = [
        {**_arcs("Person", subClassOf=[_entry("Patient", "A patient")]), "nextToken": "tok1"},
        _arcs("Person", subClassOf=[_entry("Adult", "An adult", prov="dc/p2")]),
        _arcs("Person", rangeIncludes=[_entry("spouse", "Spouse", ("Property",))]),
    ]
    client = _client(monkeypatch, pages, sent)
    result = client.node_in_arcs(["Person"], properties=_PROPS, max_results=10)
    assert sent == [
        {"nodes": ["Person"], "property": "<-[subClassOf]"},
        {"nodes": ["Person"], "property": "<-[subClassOf]", "nextToken": "tok1"},
        {"nodes": ["Person"], "property": "<-[rangeIncludes]"},
    ]
    assert result == {
        "Person": InArcs(
            subjects=(
                InArcSubject("Patient", "subClassOf", "A patient", ("Class",), ("dc/p1",)),
                InArcSubject("Adult", "subClassOf", "An adult", ("Class",), ("dc/p2",)),
                InArcSubject("spouse", "rangeIncludes", "Spouse", ("Property",), ("dc/p1",)),
            ),
            capped=frozenset(),
        )
    }


def test_node_in_arcs_node_without_arcs_or_absent_is_empty(monkeypatch) -> None:
    client = _client(monkeypatch, [{"data": {"leaf": {"arcs": {}}}}] * 2, [])
    result = client.node_in_arcs(["leaf", "absent"], properties=_PROPS, max_results=10)
    assert result == {"leaf": InArcs((), frozenset()), "absent": InArcs((), frozenset())}


def test_node_in_arcs_stops_each_property_at_its_cap(monkeypatch) -> None:
    """A capped property reads no page past its cap while another property is still listed.
    `subClassOf` caps on its first of five pages, and `rangeIncludes` has one page."""
    pages_by_property = {
        "subClassOf": [["x", "y", "x"], ["z"], ["w"], ["v"], ["u"]],
        "rangeIncludes": [["r1"]],
    }
    sent: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        sent.append(payload)
        spec = payload["property"]
        assert spec.startswith("<-[") and spec.endswith("]")
        props = spec[3:-1].split(",")
        page = int(payload.get("nextToken", "t0")[1:])
        arcs = {
            prop: {"nodes": [_entry(dcid) for dcid in pages_by_property[prop][page]]}
            for prop in props
            if page < len(pages_by_property[prop])
        }
        more = any(page + 1 < len(pages_by_property[prop]) for prop in props)
        return {"data": {"A": {"arcs": arcs}}, **({"nextToken": f"t{page + 1}"} if more else {})}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")
    result = client.node_in_arcs(["A"], properties=_PROPS, max_results=1)
    assert [(s.property, s.dcid) for s in result["A"].subjects] == [
        ("subClassOf", "x"),
        ("rangeIncludes", "r1"),
    ]
    assert result["A"].capped == {"subClassOf"}
    assert [payload["property"] for payload in sent] == ["<-[subClassOf]", "<-[rangeIncludes]"]


def test_node_in_arcs_repeated_token_raises_read_error(monkeypatch) -> None:
    page = {**_arcs("A"), "nextToken": "tok1"}
    client = _client(monkeypatch, [page, page], [])
    with pytest.raises(GraphReadError):
        client.node_in_arcs(["A"], properties=_PROPS, max_results=10)


def test_node_in_arcs_rejects_unsafe_property() -> None:
    client = GraphClient(base_url="http://fake/core/api")
    with pytest.raises(ValueError):
        client.node_in_arcs(["A"], properties=["a,b"], max_results=10)


def test_node_closure_pages_and_parses_targets(monkeypatch) -> None:
    sent: list[dict] = []
    pages = [
        {
            **_arcs("GenderType", **{"subClassOf+": [_entry("Enumeration", "Enumeration type")]}),
            "nextToken": "tok1",
        },
        _arcs("GenderType", **{"subClassOf+": [_entry("Thing")]}),
    ]
    client = _client(monkeypatch, pages, sent)
    result = client.node_closure("GenderType", property="subClassOf")
    assert sent[0] == {"nodes": ["GenderType"], "property": "->subClassOf+"}
    assert result == (
        ArcTarget("Enumeration", "Enumeration type", ("Class",), ("dc/p1",)),
        ArcTarget("Thing", None, ("Class",), ("dc/p1",)),
    )


def test_node_closure_of_node_without_arc_is_empty(monkeypatch) -> None:
    client = _client(monkeypatch, [_arcs("age")], [])
    assert client.node_closure("age", property="subClassOf") == ()
