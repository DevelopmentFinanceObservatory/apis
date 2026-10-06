"""A resubmit keeps a role's named group only when it posts a dcid whose graph members are
the role's entities, under the graph's label for the node. A group naming no graph node
drops."""

from __future__ import annotations

import logging

import pytest

from search.engine.region import resolve_spec_resubmit
from search.engine.spec_id import build_spec_identity
from search.models import BaseIdentity, IdentityEntityGroup, ResubmitIdentityPayload
from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph

_SV = "Count_Person"
_TRIANGLE = ("country/GTM", "country/HND", "country/SLV")
_GROUP = IdentityEntityGroup(dcid=None, label="Northern Triangle countries")
_NODE = "NorthernTriangle"

_NODES: dict = {
    _SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/GTM": {"label": "Guatemala", "type": "Country"},
    "country/SLV": {"label": "El Salvador", "type": "Country"},
    "country/HND": {"label": "Honduras", "type": "Country"},
    _NODE: {"label": "Northern Triangle"},
}


def _nodes(contained: tuple[str, ...]) -> dict:
    """`_NODES` with each of `contained` placed in the group's node."""
    arcs = {"containedInPlace": {"nodes": [{"dcid": _NODE}]}}
    return {
        dcid: {**node, "arcs": arcs} if dcid in contained else node for dcid, node in _NODES.items()
    }


_OBS = {
    f"{_SV}|{dcid}": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/WorldDevelopmentIndicators",
        }
    ]
    for dcid in _TRIANGLE
}


def _identity(
    dcids: tuple[str, ...], group: IdentityEntityGroup = _GROUP
) -> tuple[str, BaseIdentity]:
    return build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity(dcid) for dcid in dcids],
        entity_groups={"observationAbout": group},
        open_roles=(),
    )


def _resubmit(
    posted_dcids: tuple[str, ...],
    group: IdentityEntityGroup = _GROUP,
    contained: tuple[str, ...] = (),
):
    _spec_id, identity = _identity(posted_dcids, group)
    graph = FakeGraph(nodes=_nodes(contained), obs=_OBS)
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph),
    )


def test_a_group_naming_no_graph_node_drops(caplog):
    with caplog.at_level(logging.INFO, logger="search.engine.region.resubmit_request"):
        region = _resubmit(_TRIANGLE)

    assert region.status == "definite"
    spec = region.specs[0]
    assert spec.spec_id == _identity(_TRIANGLE)[0]
    assert {entity.ref for entity in spec.resolved.entities} == set(_TRIANGLE)
    assert spec.definition.identity.entity_groups is None
    assert spec.resolved.entity_groups is None
    assert "it names no graph node" in caplog.text


@pytest.mark.parametrize(
    ("contained", "kept", "logged"),
    [
        (_TRIANGLE, IdentityEntityGroup(dcid=_NODE, label="Northern Triangle"), None),
        (_TRIANGLE[:2], None, "dropped group NorthernTriangle: members differ"),
    ],
    ids=["graph_members_match", "graph_members_differ"],
)
def test_a_group_posting_a_dcid_is_checked_and_labelled_by_the_graph(
    contained, kept, logged, caplog
):
    """The posted label is ignored for the graph's label of the node, and a node whose graph
    members are not the role's entities drops the group, so the role reads as its members.
    A dropped group is logged with the reason."""
    with caplog.at_level(logging.INFO, logger="search.engine.region.resubmit_request"):
        region = _resubmit(
            _TRIANGLE, IdentityEntityGroup(dcid=_NODE, label="Tampered label"), contained
        )

    assert region.status == "definite"
    spec = region.specs[0]
    assert {entity.ref for entity in spec.resolved.entities} == set(_TRIANGLE)
    expected = {"observationAbout": kept} if kept is not None else None
    assert spec.definition.identity.entity_groups == expected
    assert spec.resolved.entity_groups == expected
    assert (logged in caplog.text) if logged else "dropped group" not in caplog.text
