"""Tests graph-client coverage mapping and FakeGraph reverse subject scans."""

import pytest

import dc_graph
from search.engine.clients.graph import SubjectsCoverage, SubjectsPage
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError
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
    `complete/LEAD` has one subject, `capped/LEAD` two, and every node is named by its dcid."""
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
    from search.engine.region.resubmit_request import resubmit_named_groups
    from search.engine.spec_id import build_spec_identity
    from search.models import IdentityEntityGroup
    from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity

    group = IdentityEntityGroup(dcid="complete/LEAD", label="posted label")
    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity("a")],
        entity_groups={"observationAbout": group},
        open_roles=(),
    )
    posts: list[dict] = []
    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(dc_graph.GraphClient, "post", _subjects_post(posts))

    cold = resubmit_named_groups(identity, graph=client, deadline=None)
    cold_posts = len(posts)
    warm = resubmit_named_groups(identity, graph=client, deadline=None)

    assert [(g.dcid, g.label, g.member_dcids) for g in cold] == [
        ("complete/LEAD", "complete/LEAD", frozenset({"a"}))
    ]
    assert warm == cold
    assert cold_posts > 0
    assert len(posts) == cold_posts, "the warm resubmit read the graph"


class _GroupGraph:
    """A members and labels stand-in that records each read. `fail` names the read that
    raises `GraphInfraError`."""

    def __init__(self, members: tuple[str, ...], *, fail: str | None = None) -> None:
        self.members, self.fail = members, fail
        self.label_reads: list[list[str]] = []

    def node_subjects_many(self, dcids, *, property, max_results, deadline=None):
        if self.fail == "members":
            raise GraphInfraError("members read failed")
        return {
            d: SubjectsPage(dcids=self.members, coverage=SubjectsCoverage.complete) for d in dcids
        }

    def node_labels_batch(self, dcids):
        self.label_reads.append(list(dcids))
        if self.fail == "labels":
            raise GraphInfraError("labels read failed")
        return dict.fromkeys(dcids, "graph label")


@pytest.mark.parametrize(
    ("members", "fail", "groups", "label_reads", "degraded"),
    [
        (("a",), None, [("complete/LEAD", "graph label")], [["complete/LEAD"]], False),
        (("a",), "labels", [], [["complete/LEAD"]], True),
        (("a",), "members", [], [], True),
        (("a", "b"), None, [], [], False),
    ],
    ids=["confirmed", "labels-fail", "members-fail", "members-differ"],
)
def test_resubmit_groups_read_labels_only_for_confirmed_members(
    members, fail, groups, label_reads, degraded
) -> None:
    """Labels are read only for groups whose graph members match the role's entities. A
    failed members or labels read drops the group, says the answer is degraded, and the
    resubmit still answers."""
    from search.engine.region.resubmit_request import resubmit_named_groups
    from search.engine.spec_id import build_spec_identity
    from search.models import IdentityEntityGroup
    from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity

    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity("a")],
        entity_groups={
            "observationAbout": IdentityEntityGroup(dcid="complete/LEAD", label="posted label")
        },
        open_roles=(),
    )
    graph = _GroupGraph(members, fail=fail)

    with recording_degraded_reads() as lost:
        result = resubmit_named_groups(identity, graph=graph, deadline=None)

    assert [(g.dcid, g.label) for g in result] == groups
    assert graph.label_reads == label_reads
    # A failed read is lost to the answer, so no cache stores it.
    assert lost.degraded is degraded


def test_resubmit_groups_drop_a_group_naming_no_graph_node() -> None:
    """A posted group with no dcid has nothing that ties its label to the role's entities,
    so a resubmit drops it without a read."""
    from search.engine.region.resubmit_request import resubmit_named_groups
    from search.engine.spec_id import build_spec_identity
    from search.models import IdentityEntityGroup
    from tests.engine._base_identity import PERSON_COUNT_SHAPE, subject_entity

    _spec_id, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [],
        entities=[subject_entity("a")],
        entity_groups={"observationAbout": IdentityEntityGroup(label="posted label")},
        open_roles=(),
    )
    graph = _GroupGraph(("a",))

    with recording_degraded_reads() as lost:
        assert resubmit_named_groups(identity, graph=graph, deadline=None) == ()

    assert graph.label_reads == []
    assert not lost.degraded
