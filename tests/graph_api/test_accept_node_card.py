"""Acceptance cases for the node routes: a code system publishes its declarations so a roll-up
of CRS purposes counts parent-coded observations once, a group id opens on the node route
(P6), and a batch reports a missing node as `{ref, status: "not_held"}` (P15).

The rows come from `accept-graph.md`."""

from __future__ import annotations

import pytest

from graph_api.doors.node import NODE_MEMBERSHIP_SLOT

from .conftest import FakeGraphNodeReader, make_client, make_resolver


@pytest.mark.parametrize(
    ("dcid", "code_system", "disjoint_from_children"),
    [
        # CRS assigns one purpose per record, so a parent's own records sit apart from its
        # children's and a roll-up adds them.
        ("CRSPurposeCodeEnum", "crsPurpose", True),
        # A DAC1 parent row is already its children's total, so the parent stands alone.
        ("DAC1AidTypeEnum", "dac1AidType", False),
    ],
)
def test_a_code_system_node_publishes_its_declarations(dcid, code_system, disjoint_from_children):
    graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": ("Class",), "name": (dcid,)}})

    with make_client(graph=graph) as client:
        response = client.get(f"/api/graph/node/{dcid}")

    assert response.status_code == 200, response.text
    node = response.json()
    assert node["ref"] == dcid
    assert node["kind"] == "code_system"
    declarations = node["code_system"]["declarations"]
    assert declarations["disjoint_from_children"] is disjoint_from_children
    assert "codes" in node["links"]


def test_a_code_node_names_its_system_and_parent_and_links_its_children():
    graph = FakeGraphNodeReader(
        nodes={"CRSPurposeCode122": {"typeOf": ("CRSPurposeCodeEnum",), "name": ("Basic health",)}}
    )

    with make_client(graph=graph) as client:
        node = client.get("/api/graph/node/CRSPurposeCode122").json()

    assert node["kind"] == "code"
    assert node["code"]["code"] == "122"
    assert node["code"]["parent"]["ref"] == "CRSPurposeCode120"
    assert "children" in node["links"]


@pytest.mark.parametrize(
    "dcid", ["groups/G20", "GroupOfSeven", "groups/UN.SIDS", "groups/Commonwealth"]
)
def test_a_group_id_opens_on_the_node_route_with_no_graph_node(dcid):
    with make_client() as client:
        response = client.get(f"/api/graph/node/{dcid}")

    assert response.status_code == 200, response.text
    node = response.json()
    assert node["ref"] == dcid
    assert node["kind"] == "group"
    assert "members" in node["links"]


def test_a_resolved_group_reopens_on_the_node_route(shared_resolver):
    """Journey 3: a group can be stored and reopened."""
    graph = FakeGraphNodeReader()
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    # An injected resolver leaves the policy to its caller; publish the shared one, as the
    # live lifespan does.
    NODE_MEMBERSHIP_SLOT.set(client.app, shared_resolver.policy)

    with client:
        (result,) = client.post("/api/graph/resolve", json={"texts": ["G20"]}).json()["results"]
        reopened = client.get(f"/api/graph/node/{result['entity']['ref']}")

    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["ref"] == result["entity"]["ref"]
    assert reopened.json()["kind"] == "group"


def test_a_batch_reports_a_missing_node_as_ref_and_status():
    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}})

    with make_client(graph=graph) as client:
        response = client.post("/api/graph/nodes", json={"dcids": ["country/KEN", "dc/base/Foo"]})

    assert response.status_code == 200, response.text
    held, missing = response.json()["nodes"]
    assert held["ref"] == "country/KEN"
    assert held["kind"] == "country"
    assert missing == {"ref": "dc/base/Foo", "status": "not_held"}
