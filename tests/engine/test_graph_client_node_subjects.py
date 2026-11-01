"""Tests graph-client coverage mapping and FakeGraph reverse subject scans."""

import time
from typing import cast

import pytest

import dc_graph
from dfo_platform.request_deadline import bounded_deadline, request_deadline
from dfo_platform.single_flight_cache import WaitTimeout
from search.engine.clients.graph import EngineGraphClient, SubjectsCoverage, SubjectsPage
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.config import settings
from search.engine.degraded_reads import fail_open_reads, recording_degraded_reads
from search.engine.errors import GraphDeadlineFailure, GraphInfraError
from search.engine.region import resubmit_request
from search.engine.region.resubmit_request import resubmit_named_groups
from tests.engine._virtual_clock import VirtualRequests
from tests.fixtures import FakeGraph


def test_node_subjects_many_maps_every_coverage_flag_onto_subjects_page(monkeypatch) -> None:
    """Maps each graph coverage flag to the matching `SubjectsCoverage` value."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "complete/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "a"}]}}},
                "capped/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "b"}, {"dcid": "c"}]}}},
            }
        }

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    result = client.node_subjects_many(
        ["complete/LEAD", "capped/LEAD"], property="typeOf", max_results=1
    )
    assert result == {
        "complete/LEAD": SubjectsPage(dcids=("a",), coverage=SubjectsCoverage.complete),
        "capped/LEAD": SubjectsPage(dcids=("b",), coverage=SubjectsCoverage.capped),
    }


def test_node_subjects_many_repeated_token_raises_graph_infra_error(monkeypatch) -> None:
    """A repeated `nextToken` surfaces as `GraphInfraError`."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {"leaf/NODE": {"arcs": {"typeOf": {"nodes": []}}}},
            "nextToken": "tok1",
        }

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    with pytest.raises(GraphInfraError):
        client.node_subjects_many(["leaf/NODE"], property="typeOf", max_results=10)


# FakeGraph.node_subjects_many


def _enum_nodes() -> dict:
    return {
        "Enum": {"label": "Enum", "type": "TestEnum", "arcs": {}},
        "m1": {
            "label": "Member One",
            "type": "TestEnum",
            "arcs": {"typeOf": {"nodes": [{"dcid": "Enum"}]}},
        },
        "m2": {
            "label": "Member Two",
            "type": "TestEnum",
            "arcs": {"typeOf": {"nodes": [{"dcid": "Enum"}]}},
        },
        "m3": {
            "label": "Member Three",
            "type": "TestEnum",
            "arcs": {"typeOf": {"nodes": [{"dcid": "Enum"}]}},
        },
        # Reverse scans return distinct subject dcids; each fixture node has one entry.
        "leaf": {"label": "Leaf", "type": "TestEnum", "arcs": {}},
    }


def _fake_graph() -> FakeGraph:
    return FakeGraph(nodes=_enum_nodes(), obs={}, detect={}, resolve={})


def test_fake_graph_node_subjects_many_reverse_scans_typeOf() -> None:
    g = _fake_graph()
    result = g.node_subjects_many(["Enum"], property="typeOf", max_results=10)
    assert result["Enum"] == SubjectsPage(
        dcids=("m1", "m2", "m3"), coverage=SubjectsCoverage.complete
    )


def test_fake_graph_node_subjects_many_absent_lead_returns_empty_page() -> None:
    g = _fake_graph()
    result = g.node_subjects_many(["NonExistentLead"], property="typeOf", max_results=10)
    assert result["NonExistentLead"] == SubjectsPage(dcids=(), coverage=SubjectsCoverage.complete)


def test_fake_graph_node_subjects_many_leaf_has_no_subjects() -> None:
    g = _fake_graph()
    result = g.node_subjects_many(["leaf"], property="typeOf", max_results=10)
    assert result["leaf"] == SubjectsPage(dcids=(), coverage=SubjectsCoverage.complete)


def test_fake_graph_node_subjects_many_honors_max_results_and_capped() -> None:
    g = _fake_graph()
    result = g.node_subjects_many(["Enum"], property="typeOf", max_results=2)
    assert result["Enum"].coverage is SubjectsCoverage.capped
    assert result["Enum"].dcids == ("m1", "m2")


def test_fake_graph_node_subjects_many_raise_on_call() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.node_subjects_many(["Enum"], property="typeOf", max_results=10)


def _subjects_post(posts: list[dict]):
    """A `/v2/node` stand-in that records each payload.
    `complete/LEAD` has one subject, `capped/LEAD` two, and every node is named by its dcid
    and typed `Thing`."""
    subjects = {"complete/LEAD": ["a"], "capped/LEAD": ["b", "c"]}

    def fake_post(self, path, payload, *, deadline=None):
        posts.append(payload)
        if payload["property"] == "->name":
            return {
                "data": {
                    dcid: {"arcs": {"name": {"nodes": [{"value": dcid}]}}}
                    for dcid in payload["nodes"]
                }
            }
        if payload["property"] == "->typeOf":
            return {
                "data": {
                    dcid: {"arcs": {"typeOf": {"nodes": [{"dcid": "Thing"}]}}}
                    for dcid in payload["nodes"]
                }
            }
        prop = payload["property"].removeprefix("<-")
        return {
            "data": {
                dcid: {"arcs": {prop: {"nodes": [{"dcid": s} for s in subjects.get(dcid, [])]}}}
                for dcid in payload["nodes"]
            }
        }

    return fake_post


def test_node_subjects_many_reads_a_complete_page_once_until_clear(monkeypatch) -> None:
    """A complete page is served from the cache. A capped one is read again, since another
    read may complete it. `clear` empties the cache."""
    posts: list[dict] = []
    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(dc_graph.GraphClient, "post", _subjects_post(posts))

    def read(*dcids: str) -> dict[str, SubjectsPage]:
        return client.node_subjects_many(list(dcids), property="typeOf", max_results=1)

    first = read("complete/LEAD", "capped/LEAD")
    second = read("complete/LEAD", "capped/LEAD")
    client.clear()
    read("complete/LEAD")

    assert second == first
    assert [post["nodes"] for post in posts] == [
        ["complete/LEAD", "capped/LEAD"],
        ["capped/LEAD"],
        ["complete/LEAD"],
    ]


def test_a_warm_resubmit_reads_no_group_members(monkeypatch) -> None:
    """A resubmit posting a group checks its members against the graph. A second one,
    posting the same group, makes no graph request."""
    from search.engine.spec_id import build_spec_identity
    from search.models import EntityGroup, ResubmitIdentityPayload
    from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity

    group = EntityGroup(ref="complete/LEAD", label="posted label")
    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity("a")],
        open_roles=(),
    )
    payload = ResubmitIdentityPayload(identity=identity, entity_groups={"observationAbout": group})
    posts: list[dict] = []
    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(dc_graph.GraphClient, "post", _subjects_post(posts))

    cold = resubmit_named_groups(payload, graph=client, deadline=None)
    cold_posts = len(posts)
    warm = resubmit_named_groups(payload, graph=client, deadline=None)

    assert [(g.dcid, g.label, g.entity_type, g.member_dcids) for g in cold] == [
        ("complete/LEAD", "complete/LEAD", "Thing", frozenset({"a"}))
    ]
    assert warm == cold
    assert cold_posts > 0
    assert len(posts) == cold_posts, "the warm resubmit read the graph"


class _GroupGraph:
    """A types, members and labels stand-in that records each labels read, and the deadline
    each read is bound by (`bounded_deadline`), as the graph client narrows it. The node has
    the graph type `node_type` and the `<-containedInPlace` subjects `members`. `fail` names
    the read that raises `error`. A region's members come from the membership policy, so
    reading its `Country` children fails the test."""

    def __init__(
        self,
        members: tuple[str, ...],
        *,
        fail: str | None = None,
        node_type: str = "Thing",
        coverage: SubjectsCoverage = SubjectsCoverage.complete,
        error: Exception | None = None,
    ) -> None:
        self.members, self.fail, self.coverage = members, fail, coverage
        self.node_type = node_type
        self.error = error if error is not None else GraphInfraError(f"{fail} read failed")
        self.label_reads: list[list[str]] = []
        self.bounds: dict[str, float | None] = {}

    def node_types_batch(self, dcids):
        self.bounds["types"] = bounded_deadline(None)
        if self.fail == "types":
            raise self.error
        return dict.fromkeys(dcids, self.node_type)

    def node_subjects_many(self, dcids, *, property, max_results, deadline=None):
        self.bounds["members"] = bounded_deadline(deadline)
        if self.fail == "members":
            raise self.error
        return {d: SubjectsPage(dcids=self.members, coverage=self.coverage) for d in dcids}

    def child_places_batch(self, parent_dcids, child_type, *, deadline=None):
        raise AssertionError("a region's members come from the membership policy")

    def node_labels_batch(self, dcids):
        self.label_reads.append(list(dcids))
        self.bounds["labels"] = bounded_deadline(None)
        if self.fail == "labels":
            raise self.error
        return dict.fromkeys(dcids, "graph label")


_CONTINENT = "Continent"
_LOST_TO_THE_BUDGET = {
    "deadline": dc_graph.GraphDeadlineExceededError("the read ran past its deadline"),
    "timeout": TimeoutError("the read ran past its deadline"),
    "wait": WaitTimeout("the wait on a shared flight ran past its deadline"),
}
# The graph's members read of a node of each graph type. The membership policy answers a
# region's members with no read.
_MEMBERS_READ = {"Thing": ["node_subjects_many"], _CONTINENT: []}


def _group_payload():
    """A resubmit of one entity `a` posting a group on its role, with a label and a type
    the graph does not hold."""
    from search.engine.spec_id import build_spec_identity
    from search.models import EntityGroup, ResubmitIdentityPayload
    from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity

    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity("a")],
        open_roles=(),
    )
    posted = EntityGroup(ref="complete/LEAD", label="posted label", entity_type="PostedType")
    return ResubmitIdentityPayload(identity=identity, entity_groups={"observationAbout": posted})


@pytest.mark.parametrize(
    ("graph", "held", "groups", "degraded"),
    [
        (_GroupGraph(("a",)), None, [("complete/LEAD", "graph label", "Thing")], 0),
        (_GroupGraph(("a",), fail="labels"), None, [], 1),
        (_GroupGraph(("a",), fail="members"), None, [], 1),
        (_GroupGraph(("a",), fail="types"), None, [], 1),
        *[
            (_GroupGraph(("a",), fail=read, error=error), None, [], 1)
            for read in ("labels", "types")
            for error in _LOST_TO_THE_BUDGET.values()
        ],
        (_GroupGraph(("a", "b")), None, [], 0),
        (_GroupGraph(("a",), coverage=SubjectsCoverage.capped), None, [], 0),
        (
            _GroupGraph(("a", "b"), node_type=_CONTINENT),
            ("a",),
            [("complete/LEAD", "graph label", "Continent")],
            0,
        ),
        (_GroupGraph(("a",), node_type=_CONTINENT), ("a", "b"), [], 0),
        (
            _GroupGraph(("a",), node_type=_CONTINENT),
            None,
            [("complete/LEAD", "graph label", "Continent")],
            0,
        ),
        (_GroupGraph(("a",), node_type=_CONTINENT), "omitted", [], 0),
    ],
    ids=[
        "confirmed",
        "labels-fail",
        "members-fail",
        "types-fail",
        *[f"{read}-{lost}" for read in ("labels", "types") for lost in _LOST_TO_THE_BUDGET],
        "members-differ",
        "members-page-capped",
        "region-confirmed-by-the-policy",
        "region-policy-members-differ",
        "continent-the-policy-does-not-hold-reads-containment",
        "region-the-policy-omits-has-no-members",
    ],
)
def test_resubmit_groups_keep_only_groups_the_graph_confirms(
    declare_regions, graph, held, groups, degraded
) -> None:
    """A group is kept only when its members match the role's entities, and a kept group
    states the graph's label and type, whatever the posted ones say. A members page cut off
    at its cap holds only part of the node's members, so its group is dropped. A continent's
    members are the ones the membership policy holds (`held`), whatever its
    `<-containedInPlace` subjects are. A region the policy omits has no members, so its group
    drops, with no containment read and no degraded note. A node the policy neither holds nor
    omits is no region, whatever its graph type, so its members are its `<-containedInPlace`
    subjects like any other node's. A failed types, members or labels read drops the
    group, says the answer is degraded, and the resubmit still answers."""
    if held == "omitted":
        declare_regions({}, omitted=("complete/LEAD",))
    elif held is not None:
        declare_regions({"complete/LEAD": held})
    with recording_degraded_reads() as lost:
        result = resubmit_named_groups(_group_payload(), graph=graph, deadline=None)

    assert [(g.dcid, g.label, g.entity_type) for g in result] == groups
    # A failed read is lost to the answer, so no cache stores it.
    assert lost.degraded is bool(degraded)


def test_a_members_read_cut_by_a_spent_fail_open_budget_notes_nothing() -> None:
    """The members read notes its loss through `degraded_reads.note_lost_read`, so a read cut
    by an enclosing fail-open budget, with the request's deadline still ahead, drops its
    groups and leaves an answer the cache stores."""
    graph = _GroupGraph(
        ("a",), fail="members", error=GraphDeadlineFailure("the members read timed out")
    )
    with (
        request_deadline(time.monotonic() + 60),
        recording_degraded_reads() as lost,
        fail_open_reads("the enclosing read", budget_s=0.0),
    ):
        result = resubmit_named_groups(
            _group_payload(), graph=cast(EngineGraphClient, graph), deadline=None
        )

    assert result == ()
    assert lost.reads == []


def test_resubmit_group_labels_and_types_read_under_the_fail_open_budget(declare_regions) -> None:
    """The label and type reads take no deadline, and the label read runs in a worker thread,
    so the fail-open budget reaches both as the ambient request deadline. The members read
    keeps the request's: a large node's members take seconds to read when the graph is
    healthy."""
    graph = _GroupGraph(("a",))
    started = time.monotonic()

    (group,) = resubmit_named_groups(
        _group_payload(), graph=cast(EngineGraphClient, graph), deadline=None
    )

    assert group.label == "graph label"
    budget = settings().fail_open_read_budget_s
    for read in ("labels", "types"):
        bound = graph.bounds[read]
        assert bound is not None and started + budget <= bound <= time.monotonic() + budget
    assert graph.bounds["members"] is None


@pytest.mark.parametrize(
    ("graph", "depth"),
    [(_GroupGraph(("a",)), 2), (_GroupGraph(("a", "b"), node_type=_CONTINENT), 1)],
    ids=["group", "region"],
)
def test_resubmit_groups_read_in_two_serial_steps(
    monkeypatch, declare_regions, graph, depth
) -> None:
    """The posted nodes' labels are read beside their types, so a resubmit posting a group
    reads its types and labels in one step and its members in the next. The membership
    policy answers a region's members with no read, so a region takes the first step alone."""
    if graph.node_type == _CONTINENT:
        declare_regions({"complete/LEAD": ("a",)})
    requests = VirtualRequests()
    requests.install(monkeypatch, doubles=(_GroupGraph,), pools=(resubmit_request,))

    async def door():
        return resubmit_named_groups(_group_payload(), graph=graph, deadline=None)

    (group,) = requests.run(door())

    assert group.label == "graph label"
    assert sorted(name for name, *_ in requests.spans) == sorted(
        ["node_types_batch", "node_labels_batch", *_MEMBERS_READ[graph.node_type]]
    )
    assert requests.serial_depth() == depth


def test_resubmit_groups_drop_a_group_naming_no_graph_node() -> None:
    """A posted group with no dcid has nothing that ties its label to the role's entities,
    so a resubmit drops it without a read."""
    from search.engine.spec_id import build_spec_identity
    from search.models import EntityGroup, ResubmitIdentityPayload
    from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity

    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity("a")],
        open_roles=(),
    )
    payload = ResubmitIdentityPayload(
        identity=identity,
        entity_groups={"observationAbout": EntityGroup(label="posted label")},
    )
    graph = _GroupGraph(("a",))

    with recording_degraded_reads() as lost:
        assert resubmit_named_groups(payload, graph=graph, deadline=None) == ()

    assert graph.label_reads == []
    assert not lost.degraded
