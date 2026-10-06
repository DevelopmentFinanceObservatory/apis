"""A resubmit keeps a named group posted beside the identity only when it posts a dcid whose
graph members are the role's entities, under the graph's label for the node. A group naming
no graph node drops, and so does a group on a role the identity leaves unbound."""

from __future__ import annotations

import logging

import pytest

from search.engine.region import resolve_spec_resubmit
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    EntityGroup,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve, walk_options
from tests.fixtures import FakeGraph

_SV = "Count_Person"
_TRIANGLE = ("country/GTM", "country/HND", "country/SLV")
_GROUP = EntityGroup(dcid=None, label="Northern Triangle countries")
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


def _identity(dcids: tuple[str, ...]) -> tuple[str, BaseIdentity]:
    return build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity(dcid) for dcid in dcids],
        open_roles=(),
    )


def _resubmit(
    posted_dcids: tuple[str, ...],
    group: EntityGroup = _GROUP,
    contained: tuple[str, ...] = (),
    role: str = "observationAbout",
):
    """Resubmit the identity binding `posted_dcids` with `group` posted beside it on `role`."""
    _spec_id, identity = _identity(posted_dcids)
    graph = FakeGraph(nodes=_nodes(contained), obs=_OBS)
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity, entity_groups={role: group}),
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
    assert spec.resolved.entity_groups is None
    assert "it names no graph node" in caplog.text


@pytest.mark.parametrize(
    ("role", "contained", "kept", "logged"),
    [
        (
            "observationAbout",
            _TRIANGLE,
            EntityGroup(dcid=_NODE, label="Northern Triangle"),
            None,
        ),
        (
            "observationAbout",
            _TRIANGLE[:2],
            None,
            "dropped group NorthernTriangle: members differ",
        ),
        ("origin", _TRIANGLE, None, "dropped group NorthernTriangle on role origin"),
    ],
    ids=["graph_members_match", "graph_members_differ", "role_unbound"],
)
def test_a_group_posting_a_dcid_is_checked_and_labelled_by_the_graph(
    role, contained, kept, logged, caplog
):
    """The posted label is ignored for the graph's label of the node, and a node whose graph
    members are not the role's entities drops the group, so the role reads as its members.
    A group on a role the identity binds no entity on drops the same way and the resubmit
    still answers. A dropped group is logged with the reason."""
    with caplog.at_level(logging.INFO, logger="search.engine.region.resubmit_request"):
        region = _resubmit(
            _TRIANGLE, EntityGroup(dcid=_NODE, label="Tampered label"), contained, role
        )

    assert region.status == "definite"
    spec = region.specs[0]
    assert {entity.ref for entity in spec.resolved.entities} == set(_TRIANGLE)
    expected = {"observationAbout": kept} if kept is not None else None
    assert spec.resolved.entity_groups == expected
    assert (logged in caplog.text) if logged else "dropped group" not in caplog.text


def _identity_entry(identity: BaseIdentity, groups: dict) -> ResubmitIdentityEntry:
    return ResubmitIdentityEntry(
        variable_text="population", identity=identity, entity_groups=groups
    )


def _candidates_entry(identity: BaseIdentity, groups: dict) -> ResubmitCandidatesEntry:
    return ResubmitCandidatesEntry(
        variable_text="population",
        ordering="broadest_first",
        truncated=False,
        candidates=[ResubmitCandidate(identity=identity, entity_groups=groups)] * 2,
    )


@pytest.mark.parametrize("entry_for", [_identity_entry, _candidates_entry])
def test_a_group_posted_beside_an_identity_reaches_the_answer_on_each_resubmit_door(entry_for):
    """An identity entry and a candidate each carry their group beside the identity, and
    the part's Spec serves the graph-confirmed group on `resolved.entity_groups`."""
    spec_id, identity = _identity(_TRIANGLE)
    graph = FakeGraph(nodes=_nodes(_TRIANGLE), obs=_OBS)
    entry = entry_for(identity, {"observationAbout": EntityGroup(dcid=_NODE, label="posted")})

    response = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    (part,) = response.parts
    specs = [part.spec] if part.status == "definite" else part.candidates.specs
    assert specs
    for spec in specs:
        assert spec.spec_id == spec_id
        assert spec.resolved.entity_groups == {
            "observationAbout": EntityGroup(dcid=_NODE, label="Northern Triangle")
        }


@pytest.mark.parametrize(
    "model", [ResubmitIdentityPayload, ResubmitIdentityEntry, ResubmitCandidate]
)
@pytest.mark.parametrize(
    "groups",
    [
        {"creditor": {"label": "Paris Club"}},
        {role: {"label": "x"} for role in ("observationAbout", "origin", "destination", "a")},
    ],
    ids=["unknown_role_key", "four_roles"],
)
def test_the_request_models_refuse_a_group_on_no_entity_role(model, groups):
    """A group's key is one of the three entity role keys, so a fourth key or any other
    name fails validation on every model that carries the field."""
    from pydantic import ValidationError

    document = {
        "variable_text": "population",
        "identity": _identity(_TRIANGLE)[1].model_dump(mode="json", exclude_none=True),
        "entity_groups": groups,
    }
    fields = {name: document[name] for name in model.model_fields if name in document}

    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(fields)

    assert {error["loc"][0] for error in exc_info.value.errors()} == {"entity_groups"}
