"""Check that graph readers and test doubles deduplicate repeated values in first-occurrence order.
Cases cover arc reads, subjects, resolve results, child lists, and names, including cross-arc duplicates.
Distinct values must remain in the result.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import httpx
import pytest

import dc_graph
from entity_resolution.types import GraphLookup
from graph_api.reader import GraphNodeReader
from materialise.graph import GraphReader, SubjectPage
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.graph_client import LiveGraphClient
from statvar_members.types import MemberArcReader
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.entity_resolution.conftest import FakeGraphLookup
from tests.fixtures import FakeGraph
from tests.graph_api.conftest import FakeGraphNodeReader
from tests.materialise.conftest import FakeGraphReader
from tests.statvar_members.conftest import FakeMemberArcReader, FlakyMemberArcReader

# The cases exercise each Protocol's implementations structurally.
_PROTOCOLS = (GraphReader, MemberArcReader, GraphLookup, EngineGraphClient, GraphNodeReader)

_DUPLICATE_INPUT = ("a", "a", "b")
_DUPLICATE_EXPECTED = ("a", "b")
_DISTINCT_INPUT = ("a", "b", "c")

Case = tuple[str, Callable[[Sequence[str]], tuple[str, ...]]]


# ---------------------------------------------------------------------------
# Wire and transport helpers, shared by every real-implementation case
# ---------------------------------------------------------------------------


def _replay(body: dict) -> httpx.MockTransport:
    payload = json.dumps(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"content-type": "application/json"})

    return httpx.MockTransport(handler)


def _reader(body: dict) -> dc_graph.GraphClient:
    return dc_graph.GraphClient(
        base_url="https://graph.example.org/core/api", transport=_replay(body)
    )


def _client(body: dict) -> LiveGraphClient:
    """Return a client whose wrapped `post` returns `body` without changing other clients."""
    client = LiveGraphClient(base="http://fake")
    client._client._ours.post = lambda path, payload, *, deadline=None: body
    return client


def _client_with_queue(bodies: Sequence[dict]) -> LiveGraphClient:
    """Return a client that feeds queued responses to multi-POST methods such as `enum_members`."""
    queue = list(bodies)
    client = LiveGraphClient(base="http://fake")
    client._client._ours.post = lambda path, payload, *, deadline=None: queue.pop(0)
    return client


def _node_arcs_body(dcid: str, prop: str, values: Sequence[str]) -> dict:
    """Build a `/v2/node` response with `values` on `dcid`'s `prop` out-arc."""
    return {"data": {dcid: {"arcs": {prop: {"nodes": [{"dcid": v} for v in values]}}}}}


def _resolve_body(name: str, values: Sequence[str]) -> dict:
    return {"entities": [{"node": name, "candidates": [{"dcid": v} for v in values]}]}


def _wire_arcs(arcs: dict | None, prop: str) -> tuple[str, ...]:
    """Extract `prop`'s resolved dcids from a raw `/v2/node` arc response."""
    if arcs is None:
        return ()
    return tuple(n["dcid"] for n in arcs.get(prop, {}).get("nodes", []) if "dcid" in n)


# Arc readers and doubles, normalised to a tuple of values.

_ARC_VALUE_CASES: list[Case] = [
    (
        "GraphReader.node_arc_values/dc_graph.GraphClient",
        lambda values: _reader(_node_arcs_body("d1", "p", values)).node_arc_values(
            "d1", properties=("p",)
        )["p"],
    ),
    (
        "GraphReader.node_arc_values/FakeGraphReader",
        lambda values: FakeGraphReader(arcs={"d1": {"p": tuple(values)}}).node_arc_values(
            "d1", properties=("p",)
        )["p"],
    ),
    (
        "GraphReader.node_arc_values/FakeMintGraphReader",
        lambda values: FakeMintGraphReader(arcs={"d1": {"p": tuple(values)}}).node_arc_values(
            "d1", properties=("p",)
        )["p"],
    ),
    (
        "node_arcs_many/dc_graph.GraphClient",
        lambda values: _reader(_node_arcs_body("d1", "p", values)).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "EngineGraphClient.node_arcs_many/LiveGraphClient",
        lambda values: _client(_node_arcs_body("d1", "p", values)).node_arcs_many(
            ["d1"], properties=["p"]
        )["d1"]["p"],
    ),
    (
        "node_arcs_many/FakeGraphReader",
        lambda values: FakeGraphReader(arcs={"d1": {"p": tuple(values)}}).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "MemberArcReader.node_arcs_many/FakeMemberArcReader",
        lambda values: FakeMemberArcReader(arcs={"d1": {"p": tuple(values)}}).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "MemberArcReader.node_arcs_many/FlakyMemberArcReader",
        lambda values: FlakyMemberArcReader(arcs={"d1": {"p": tuple(values)}}).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "GraphLookup.node_arcs_many/FakeGraphLookup",
        lambda values: FakeGraphLookup(nodes={"d1": {"p": tuple(values)}}).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "GraphNodeReader.node_arcs_many/FakeGraphNodeReader",
        lambda values: FakeGraphNodeReader(nodes={"d1": {"p": tuple(values)}}).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "node_arcs_many/FakeMintGraphReader",
        lambda values: FakeMintGraphReader(arcs={"d1": {"p": tuple(values)}}).node_arcs_many(
            ["d1"], properties=("p",)
        )["d1"]["p"],
    ),
    (
        "EngineGraphClient.node_arcs_many/FakeGraph",
        lambda values: FakeGraph(
            nodes={
                "d1": {
                    "label": None,
                    "type": None,
                    "arcs": {"p": {"nodes": [{"dcid": v} for v in values]}},
                }
            }
        ).node_arcs_many(["d1"], properties=["p"])["d1"]["p"],
    ),
    (
        "EngineGraphClient.node_arcs(raw)/LiveGraphClient",
        lambda values: _wire_arcs(_client(_node_arcs_body("d1", "p", values)).node_arcs("d1"), "p"),
    ),
    (
        "EngineGraphClient.node_arcs(raw)/FakeGraph",
        lambda values: _wire_arcs(
            FakeGraph(
                nodes={
                    "d1": {
                        "label": None,
                        "type": None,
                        "arcs": {"p": {"nodes": [{"dcid": v} for v in values]}},
                    }
                }
            ).node_arcs("d1"),
            "p",
        ),
    ),
    (
        "EngineGraphClient.node_arcs_batch(raw)/LiveGraphClient",
        lambda values: _wire_arcs(
            _client(_node_arcs_body("d1", "p", values)).node_arcs_batch(["d1"])["d1"], "p"
        ),
    ),
    (
        "EngineGraphClient.node_arcs_batch(raw)/FakeGraph",
        lambda values: _wire_arcs(
            FakeGraph(
                nodes={
                    "d1": {
                        "label": None,
                        "type": None,
                        "arcs": {"p": {"nodes": [{"dcid": v} for v in values]}},
                    }
                }
            ).node_arcs_batch(["d1"])["d1"],
            "p",
        ),
    ),
    (
        "EngineGraphClient.node_observation_properties_batch/LiveGraphClient",
        lambda values: tuple(
            _client(
                _node_arcs_body("d1", "observationProperties", values)
            ).node_observation_properties_batch(["d1"])["d1"]
        ),
    ),
    (
        "EngineGraphClient.node_observation_properties_batch/FakeGraph",
        lambda values: tuple(
            FakeGraph(
                nodes={
                    "d1": {
                        "label": None,
                        "type": None,
                        "arcs": {"observationProperties": {"nodes": [{"dcid": v} for v in values]}},
                    }
                }
            ).node_observation_properties_batch(["d1"])["d1"]
        ),
    ),
    (
        "EngineGraphClient.enum_members/LiveGraphClient",
        lambda values: tuple(
            _client_with_queue(
                [
                    _node_arcs_body("prop1", "rangeIncludes", ["ClassA"]),
                    _node_arcs_body("ClassA", "typeOf", values),
                ]
            ).enum_members("prop1")
        ),
    ),
    (
        "EngineGraphClient.enum_members/FakeGraph",
        lambda values: tuple(FakeGraph(enum_members={"prop1": list(values)}).enum_members("prop1")),
    ),
]


@pytest.mark.parametrize("impl_id, run", _ARC_VALUE_CASES, ids=[c[0] for c in _ARC_VALUE_CASES])
def test_arc_value_dedups_a_value_declared_by_two_provenances(impl_id, run):
    assert run(_DUPLICATE_INPUT) == _DUPLICATE_EXPECTED, impl_id


@pytest.mark.parametrize("impl_id, run", _ARC_VALUE_CASES, ids=[c[0] for c in _ARC_VALUE_CASES])
def test_arc_value_keeps_genuinely_distinct_values(impl_id, run):
    assert run(_DISTINCT_INPUT) == _DISTINCT_INPUT, impl_id


# FakeMemberArcReader and FakeMintGraphReader key subjects by dcid, so duplicates cannot reach them.

_SUBJECT_CASES: list[Case] = [
    (
        "GraphReader.node_subjects/dc_graph.GraphClient",
        lambda values: (
            _reader(_node_arcs_body("d1", "measuredProperty", values))
            .node_subjects("d1", property="measuredProperty", max_results=100)
            .dcids
        ),
    ),
    (
        "GraphReader.node_subjects/FakeGraphReader",
        lambda values: (
            FakeGraphReader(
                subjects={
                    ("d1", "measuredProperty"): SubjectPage(dcids=tuple(values), truncated=False)
                }
            )
            .node_subjects("d1", property="measuredProperty", max_results=100)
            .dcids
        ),
    ),
]


@pytest.mark.parametrize("impl_id, run", _SUBJECT_CASES, ids=[c[0] for c in _SUBJECT_CASES])
def test_node_subjects_dedups_a_subject_declared_by_two_provenances(impl_id, run):
    assert run(_DUPLICATE_INPUT) == _DUPLICATE_EXPECTED, impl_id


@pytest.mark.parametrize("impl_id, run", _SUBJECT_CASES, ids=[c[0] for c in _SUBJECT_CASES])
def test_node_subjects_keeps_genuinely_distinct_subjects(impl_id, run):
    assert run(_DISTINCT_INPUT) == _DISTINCT_INPUT, impl_id


# FakeGraph maps each name to one candidate, so it cannot expose duplicates.

_RESOLVE_BY_ARC_CASES: list[Case] = [
    (
        "dc_graph.GraphClient.resolve_by_arc/dc_graph.GraphClient",
        lambda values: tuple(
            _reader(_resolve_body("Kenya", values)).resolve_by_arc(["Kenya"], arc="<-name->dcid")[
                "Kenya"
            ]
        ),
    ),
    (
        "EngineGraphClient.resolve_by_arc/LiveGraphClient",
        lambda values: tuple(
            _client(_resolve_body("Kenya", values)).resolve_by_arc(["Kenya"], arc="<-name->dcid")[
                "Kenya"
            ]
        ),
    ),
    (
        "GraphLookup.resolve_by_arc/FakeGraphLookup",
        lambda values: tuple(
            FakeGraphLookup(arcs={("<-name->dcid", "Kenya"): list(values)}).resolve_by_arc(
                ["Kenya"], arc="<-name->dcid"
            )["Kenya"]
        ),
    ),
    (
        "GraphNodeReader.resolve_by_arc/FakeGraphNodeReader",
        lambda values: tuple(
            FakeGraphNodeReader(arcs={("<-name->dcid", "Kenya"): list(values)}).resolve_by_arc(
                ["Kenya"], arc="<-name->dcid"
            )["Kenya"]
        ),
    ),
]


@pytest.mark.parametrize(
    "impl_id, run", _RESOLVE_BY_ARC_CASES, ids=[c[0] for c in _RESOLVE_BY_ARC_CASES]
)
def test_resolve_by_arc_dedups_a_candidate_declared_by_two_provenances(impl_id, run):
    assert run(_DUPLICATE_INPUT) == _DUPLICATE_EXPECTED, impl_id


@pytest.mark.parametrize(
    "impl_id, run", _RESOLVE_BY_ARC_CASES, ids=[c[0] for c in _RESOLVE_BY_ARC_CASES]
)
def test_resolve_by_arc_keeps_genuinely_distinct_candidates(impl_id, run):
    assert run(_DISTINCT_INPUT) == _DISTINCT_INPUT, impl_id


# `child_places_batch` cases.

_CHILD_PLACES_CASES: list[Case] = [
    (
        "dc_graph.GraphClient.child_places_batch/dc_graph.GraphClient",
        lambda values: tuple(
            _reader(_node_arcs_body("d1", "containedInPlace+", values)).child_places_batch(
                ["d1"], child_type="Country"
            )["d1"]
        ),
    ),
    (
        "EngineGraphClient.child_places_batch/LiveGraphClient",
        lambda values: tuple(
            _client(_node_arcs_body("d1", "containedInPlace+", values)).child_places_batch(
                ["d1"], "Country"
            )["d1"]
        ),
    ),
    (
        "GraphNodeReader.child_places_batch/FakeGraphNodeReader",
        lambda values: tuple(
            FakeGraphNodeReader(contained={("d1", "Country"): list(values)}).child_places_batch(
                ["d1"], child_type="Country"
            )["d1"]
        ),
    ),
    (
        "EngineGraphClient.child_places_batch/FakeGraph",
        lambda values: tuple(
            FakeGraph(children={"d1": {"Country": list(values)}}).child_places_batch(
                ["d1"], "Country"
            )["d1"]
        ),
    ),
]


@pytest.mark.parametrize(
    "impl_id, run", _CHILD_PLACES_CASES, ids=[c[0] for c in _CHILD_PLACES_CASES]
)
def test_child_places_batch_dedups_a_child_declared_by_two_provenances(impl_id, run):
    assert run(_DUPLICATE_INPUT) == _DUPLICATE_EXPECTED, impl_id


@pytest.mark.parametrize(
    "impl_id, run", _CHILD_PLACES_CASES, ids=[c[0] for c in _CHILD_PLACES_CASES]
)
def test_child_places_batch_keeps_genuinely_distinct_children(impl_id, run):
    assert run(_DISTINCT_INPUT) == _DISTINCT_INPUT, impl_id


# FakeGraph keys nodes by dcid, so only LiveGraphClient can expose duplicate children here.

_CHILD_DCIDS_CASES: list[Case] = [
    (
        "EngineGraphClient.child_dcids/LiveGraphClient",
        lambda values: tuple(
            _client(_node_arcs_body("parent1", "isPartOf", values)).child_dcids("parent1")
        ),
    ),
]


@pytest.mark.parametrize("impl_id, run", _CHILD_DCIDS_CASES, ids=[c[0] for c in _CHILD_DCIDS_CASES])
def test_child_dcids_dedups_a_child_declared_by_two_provenances(impl_id, run):
    assert run(_DUPLICATE_INPUT) == _DUPLICATE_EXPECTED, impl_id


@pytest.mark.parametrize("impl_id, run", _CHILD_DCIDS_CASES, ids=[c[0] for c in _CHILD_DCIDS_CASES])
def test_child_dcids_keeps_genuinely_distinct_children(impl_id, run):
    assert run(_DISTINCT_INPUT) == _DISTINCT_INPUT, impl_id


# Name and alternateName share a dedup. FakeGraph uses `label` for `name`, so its case seeds only
# `alternateName`.
_NAME_CASES: list[Case] = [
    (
        "GraphLookup.node_names_many/dc_graph.GraphClient",
        lambda values: _reader(_node_arcs_body("d1", "name", values)).node_names_many(["d1"])["d1"],
    ),
    (
        "EngineGraphClient.node_names_many/LiveGraphClient",
        lambda values: _client(_node_arcs_body("d1", "name", values)).node_names_many(["d1"])["d1"],
    ),
    (
        "GraphLookup.node_names_many/FakeGraphLookup",
        lambda values: FakeGraphLookup(nodes={"d1": {"name": tuple(values)}}).node_names_many(
            ["d1"]
        )["d1"],
    ),
    (
        "GraphNodeReader.node_names_many/FakeGraphNodeReader",
        lambda values: FakeGraphNodeReader(nodes={"d1": {"name": tuple(values)}}).node_names_many(
            ["d1"]
        )["d1"],
    ),
    (
        "GraphReader.node_names_many/FakeGraphReader",
        lambda values: FakeGraphReader(arcs={"d1": {"name": tuple(values)}}).node_names_many(
            ["d1"]
        )["d1"],
    ),
    (
        "MemberArcReader.node_names_many/FakeMemberArcReader",
        lambda values: FakeMemberArcReader(arcs={"d1": {"name": tuple(values)}}).node_names_many(
            ["d1"]
        )["d1"],
    ),
    (
        "MemberArcReader.node_names_many/FakeMintGraphReader",
        lambda values: FakeMintGraphReader(arcs={"d1": {"name": tuple(values)}}).node_names_many(
            ["d1"]
        )["d1"],
    ),
    (
        "EngineGraphClient.node_names_many/FakeGraph",
        lambda values: FakeGraph(
            nodes={
                "d1": {
                    "label": None,
                    "type": None,
                    "arcs": {"alternateName": {"nodes": [{"dcid": v} for v in values]}},
                }
            }
        ).node_names_many(["d1"])["d1"],
    ),
]


@pytest.mark.parametrize("impl_id, run", _NAME_CASES, ids=[c[0] for c in _NAME_CASES])
def test_node_names_many_dedups_a_name_declared_by_two_provenances(impl_id, run):
    assert run(_DUPLICATE_INPUT) == _DUPLICATE_EXPECTED, impl_id


@pytest.mark.parametrize("impl_id, run", _NAME_CASES, ids=[c[0] for c in _NAME_CASES])
def test_node_names_many_keeps_genuinely_distinct_names(impl_id, run):
    assert run(_DISTINCT_INPUT) == _DISTINCT_INPUT, impl_id
