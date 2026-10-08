"""A resubmit keeps a named group posted beside the identity only when it posts a dcid whose
graph members are the role's entities, under the graph's label for the node. A group naming
no graph node drops, and so does a group on a role the identity leaves unbound."""

from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import ValidationError

from entity_resolution import Entity, Group
from search.engine.bind import _UniversalBindOutput
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    EntityGroup,
    RequestedWindow,
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
from tests.fixtures import FakeGraph, make_resolver

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


def _identity(
    dcids: tuple[str, ...], requested_window: RequestedWindow | None = None
) -> tuple[str, BaseIdentity]:
    return build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity(dcid) for dcid in dcids],
        open_roles=(),
        requested_window=requested_window,
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


_EU_NODE = "EuropeanUnion"
_EU_MEMBERS = tuple(f"country/EU{i:02d}" for i in range(27))


class _ElectSoleShape:
    """Elects the sole shape without a recorded LLM response."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
        ), None


def _eu_graph() -> FakeGraph:
    """A group with its own typed graph node and 27 members placed directly in it, two of
    which hold data."""
    contained = {"containedInPlace": {"nodes": [{"dcid": _EU_NODE}]}}
    return FakeGraph(
        nodes={
            _SV: _NODES[_SV],
            _EU_NODE: {"label": "European Union", "type": "ContinentalUnion"},
            **{
                dcid: {"label": f"Member {dcid[-2:]}", "type": "Country", "arcs": contained}
                for dcid in _EU_MEMBERS
            },
        },
        obs={f"{_SV}|{dcid}": _OBS[f"{_SV}|country/GTM"] for dcid in _EU_MEMBERS[:2]},
        detect={"population": {"svs": [_SV], "entities": []}},
        resolve={},
    )


def test_a_first_answers_27_member_group_identity_resubmits_with_its_group():
    """A first answer binds the European Union's 27 members under its group. The identity it
    serves, posted back with that group, answers with the same `spec_id` and serves the
    same group: its dcid, the graph's label and the graph type of its node."""
    graph = _eu_graph()
    group = Group(
        members=tuple(
            Entity(dcid=dcid, name=f"Member {dcid[-2:]}", kind="country", graph_type="Country")
            for dcid in _EU_MEMBERS
        ),
        entity=Entity(
            dcid=_EU_NODE, name="European Union", kind="aggregate", graph_type="ContinentalUnion"
        ),
    )
    first = asyncio.run(
        resolve_variable(
            "population",
            entities=["European Union"],
            date_request=None,
            detect_query="population",
            role_query="population European Union",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectSoleShape(),
            base_timing={},
            pre_resolved={"European Union": group},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert first.status == "definite", first
    (served,) = first.specs
    assert served.definition.identity.entities == {"observationAbout": list(_EU_MEMBERS)}
    assert served.resolved.entity_groups == {
        "observationAbout": EntityGroup(
            dcid=_EU_NODE, label="European Union", entity_type="ContinentalUnion"
        )
    }

    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=served.definition.identity, entity_groups=served.resolved.entity_groups
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph),
    )

    assert region.status == "definite", region
    (spec,) = region.specs
    assert spec.spec_id == served.spec_id
    assert spec.resolved.entity_groups == served.resolved.entity_groups
    assert spec.resolved.coverage == served.resolved.coverage


_CONFIRMED = {"observationAbout": EntityGroup(dcid=_NODE, label="Northern Triangle")}


def _resolve_entry(entry: ResubmitIdentityEntry | ResubmitCandidatesEntry):
    """The one part a resubmit of `entry` answers, over a graph that places every triangle
    country in the group's node."""
    response = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=FakeGraph(nodes=_nodes(_TRIANGLE), obs=_OBS),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    (part,) = response.parts
    return part


def test_a_group_posted_beside_an_identity_entry_reaches_the_answer():
    """The served group carries the graph's label and type, whatever the posted group says."""
    spec_id, identity = _identity(_TRIANGLE)
    posted = EntityGroup(dcid=_NODE, label="posted", entity_type="PostedType")

    part = _resolve_entry(
        ResubmitIdentityEntry(
            variable_text="population",
            identity=identity,
            entity_groups={"observationAbout": posted},
        )
    )

    assert part.spec.spec_id == spec_id
    assert part.spec.resolved.entity_groups == _CONFIRMED


def test_a_candidate_serves_only_the_group_posted_beside_its_own_identity():
    """Both candidates bind the group's members, so the graph would confirm the group for
    either. The candidate posting it serves it on `resolved.entity_groups`, and the
    candidate posting none serves none."""
    grouped_id, grouped = _identity(_TRIANGLE)
    bare_id, bare = _identity(_TRIANGLE, "latest")

    part = _resolve_entry(
        ResubmitCandidatesEntry(
            variable_text="population",
            ordering="broadest_first",
            truncated=False,
            candidates=[
                ResubmitCandidate(
                    identity=grouped,
                    entity_groups={"observationAbout": EntityGroup(dcid=_NODE, label="posted")},
                ),
                ResubmitCandidate(identity=bare),
            ],
        )
    )

    assert [(spec.spec_id, spec.resolved.entity_groups) for spec in part.candidates.specs] == [
        (grouped_id, _CONFIRMED),
        (bare_id, None),
    ]


@pytest.mark.parametrize(
    "model", [ResubmitIdentityPayload, ResubmitIdentityEntry, ResubmitCandidate]
)
def test_the_request_models_refuse_a_group_on_no_entity_role(model):
    """A group's key is one of the three entity role keys, so any other name fails
    validation on every model that carries the field."""
    document = {
        "variable_text": "population",
        "identity": _identity(_TRIANGLE)[1].model_dump(mode="json", exclude_none=True),
        "entity_groups": {"creditor": {"label": "Paris Club"}},
    }
    fields = {name: document[name] for name in model.model_fields if name in document}

    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(fields)

    assert {error["loc"][0] for error in exc_info.value.errors()} == {"entity_groups"}
