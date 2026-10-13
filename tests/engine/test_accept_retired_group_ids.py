"""A resubmit posting an `entity_groups` entry under a group id the Graph API design retired
refuses with a pointer at that id, where today it drops the group and its total silently (P6).
An entry under a current id, the G7's and OECD's graph node ids among them, or naming no
node, still resubmits, and the G7's id keeps its group.

The rows come from `accept-graph.md`."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.entry.dataset_entries import resolve_candidates_entry
from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.engine.spec_id import build_spec_identity
from search.models import (
    EntityGroup,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph

_SV = "Count_Person"
_G7 = (
    "country/CAN",
    "country/DEU",
    "country/FRA",
    "country/GBR",
    "country/ITA",
    "country/JPN",
    "country/USA",
)
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
    **{dcid: {"label": dcid, "type": "Country"} for dcid in _G7},
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
    for dcid in _G7
}


def _identity():
    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity(dcid) for dcid in _G7],
        open_roles=(),
        requested_window=None,
    )
    return identity


def _resubmit(group: EntityGroup):
    identity = _identity()
    graph = FakeGraph(nodes=_NODES, obs=_OBS)
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity, entity_groups={"observationAbout": group}),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph),
    )


@pytest.mark.parametrize("retired", ["groups/G7", "groups/OECD"])
def test_a_posted_retired_group_id_is_refused_at_its_pointer(retired):
    with pytest.raises(EngineInputError) as refused:
        _resubmit(EntityGroup(ref=retired, label="G7"))

    assert refused.value.pointer == "/input/entity_groups/observationAbout/ref"


def test_a_retired_group_id_on_a_posted_candidate_is_refused_at_that_candidates_pointer():
    entry = ResubmitCandidatesEntry(
        variable_text="population",
        ordering="broadest_first",
        truncated=False,
        candidates=[
            ResubmitCandidate(identity=_identity(), entity_groups={}),
            ResubmitCandidate(
                identity=_identity(),
                entity_groups={"observationAbout": EntityGroup(ref="groups/G7", label="G7")},
            ),
        ],
    )
    graph = FakeGraph(nodes=_NODES, obs=_OBS)

    with pytest.raises(EngineInputError) as refused:
        asyncio.run(
            resolve_candidates_entry(
                3,
                entry,
                graph=graph,
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=walk_options(graph).materialise_graph,
                materialise_sdmx=None,
                rate_cache=None,
                population_memo=None,
                place_options_pool=None,
                deadline=None,
                full_read_deadline=None,
                sem=asyncio.Semaphore(1),
                coverage=empty_coverage_snapshot(),
            )
        )

    assert (
        refused.value.pointer == "/input/entries/3/candidates/1/entity_groups/observationAbout/ref"
    )


@pytest.mark.parametrize(
    "group",
    [
        EntityGroup(ref=None, label="G7 countries"),
        EntityGroup(ref="groups/UN.SIDS", label="small island developing states"),
        EntityGroup(ref="GroupOfSeven", label="G7"),
        EntityGroup(ref="undata-geo/G00406000", label="OECD"),
    ],
    ids=["no-node", "current-groups-id", "g7-node-id", "oecd-node-id"],
)
def test_a_posted_group_under_a_current_id_or_none_resubmits(group):
    region = _resubmit(group)

    assert region.status == "definite"
    assert {entity.ref for entity in region.specs[0].resolved.entities} == set(_G7)


def test_a_posted_g7_under_its_node_id_keeps_the_group():
    """The graph's G7 node has no member arcs, so the group is kept only once Search reads its
    members from the membership registry."""
    region = _resubmit(EntityGroup(ref="GroupOfSeven", label="G7"))

    assert region.specs[0].resolved.entity_groups["observationAbout"].ref == "GroupOfSeven"
