"""The resolve regression corpus (`tests/entity_resolution/corpus/resolve_cases.yaml`) runs every
case through `POST /api/graph/resolve` over the shared resolver, the real membership
policy it publishes, and a fake graph, and checks every expectation a case states on the wire.
The fake graph types each node as the case declares, so a case catches a wrong kind only for
the types it records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from graph_api.doors.node import NODE_MEMBERSHIP_SLOT

from .conftest import FakeGraphNodeReader, make_client, make_resolver

_CORPUS = Path(__file__).parents[1] / "entity_resolution" / "corpus" / "resolve_cases.yaml"
_LAST_RUNG_ARC = "<-description->dcid"
_EXPECTATION_KEYS = frozenset(
    {"status", "ref", "candidates", "kind", "matched_as", "reading", "member_count"}
)
_STATUSES = frozenset({"resolved", "ambiguous", "not_found", "not_held"})


def _cases() -> list[dict[str, Any]]:
    return yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))["cases"]


def _case_id(case: dict[str, Any]) -> str:
    return case["text"] + (f"[{','.join(case['kinds'])}]" if case.get("kinds") else "")


def _params() -> list[Any]:
    return [pytest.param(case, id=_case_id(case)) for case in _cases()]


def _graph(case: dict[str, Any]) -> FakeGraphNodeReader:
    graph = case.get("graph") or {}
    last_rung = graph.get("last_rung") or []
    nodes = {
        dcid: {
            "typeOf": (graph_type,),
            "name": (case["text"],) if dcid in last_rung else ("graph-name-unused",),
        }
        for dcid, graph_type in (graph.get("held") or {}).items()
    }
    arcs = {(_LAST_RUNG_ARC, case["text"]): list(last_rung)} if last_rung else {}
    return FakeGraphNodeReader(nodes=nodes, arcs=arcs)


def test_the_corpus_is_well_formed():
    """Every case names a known status and a note, every case that is not `not_found` states a
    ref or candidates, and no expectation key is misspelt."""
    cases = _cases()

    assert len(cases) >= 84
    for case in cases:
        expect = case["expect"]
        assert expect["status"] in _STATUSES, case
        assert set(expect) <= _EXPECTATION_KEYS, case
        assert case.get("note"), case
        if expect["status"] != "not_found":
            assert expect.get("ref") or expect.get("candidates"), case


@pytest.mark.parametrize("case", _params())
def test_a_case_resolves_as_the_corpus_expects(shared_resolver, case: dict[str, Any]):
    graph = _graph(case)
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    # An injected resolver leaves the policy to its caller; publish the shared one, as the
    # live lifespan does, so a group or region carries its count.
    NODE_MEMBERSHIP_SLOT.set(client.app, shared_resolver.policy)
    body: dict[str, Any] = {"texts": [case["text"]]}
    if case.get("kinds") is not None:
        body["kinds"] = case["kinds"]
    if case.get("code_system") is not None:
        body["code_system"] = case["code_system"]

    with client:
        response = client.post("/api/graph/resolve", json=body)

    assert response.status_code == 200, response.text
    (result,) = response.json()["results"]
    expect = case["expect"]
    assert result["input"] == case["text"]
    assert result["status"] == expect["status"]
    if expect.get("ref") is not None:
        assert result["entity"]["ref"] == expect["ref"]
    if expect.get("kind") is not None:
        assert result["entity"]["kind"] == expect["kind"]
    if expect.get("candidates") is not None:
        assert {c["ref"] for c in result["candidates"]} == set(expect["candidates"])
    if expect.get("matched_as") is not None:
        assert result["matched"]["as"] == expect["matched_as"]
    if expect.get("reading") is not None:
        assert result["reading"] == expect["reading"]
    if expect.get("member_count") is not None:
        assert result["member_count"] == expect["member_count"]
