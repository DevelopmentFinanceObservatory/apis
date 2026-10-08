"""Pins the UN table enumeration against a fake UN deployment."""

from __future__ import annotations

import time

import httpx
import pytest

from dc_graph import GraphReadError
from origin_catalogue import un_mirror
from origin_catalogue.registry import Registry, parse_registry
from origin_catalogue.schema import UnAgencyCounts
from origin_catalogue.un_mirror import UnMirror, UnMirrorBuild, UnSiteClient, UnVariable
from tests.origin_catalogue.un_fake import (
    MORT,
    MORT_BY_SEX,
    MORT_FEMALE,
    MORT_GROUP,
    MORT_MALE,
    MORT_PEERS,
    MORT_SERIES,
    MORT_TOPIC,
    ODA,
    ODA_GROUP,
    ODA_SERIES,
    ODA_TOPIC,
    TOPIC_ROOT,
    WHO,
    WHO_ROOT,
    WHO_TOPIC,
    FakeUn,
    group,
    target,
    topic,
    variable,
)

CRS = "provenance/OECDCreditorReportingSystem"


@pytest.fixture(autouse=True)
def _no_wait_before_a_repeated_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(un_mirror, "_REPEATED_PAGE_WAITS_SECONDS", (0.0, 0.0, 0.0))


def registry(*excluded: str) -> Registry:
    exclusions = ", ".join(
        f'{{ series = "{series}", reason = "superseded_by_dfo", dfo_provenance = "{CRS}" }}'
        for series in excluded
    )
    return parse_registry(
        f"""
base = []

[[un_system]]
provenance = "{WHO}"
segment = "who"
label = "World Health Organization (WHO)"
exclusions = [{exclusions}]
""".encode()
    )


def build(un: FakeUn, *excluded: str) -> UnMirrorBuild:
    return un_mirror.build_un_mirror(registry(*excluded), graph=un, group_counts=un)


def mirror(un: FakeUn, *excluded: str) -> UnMirror:
    built = build(un, *excluded)
    assert built.faults == ()
    assert built.mirror is not None
    return built.mirror


def _without_provenance(arcs: dict) -> dict:
    return {
        property: {
            "nodes": [{k: v for k, v in node.items() if k != "provenanceId"} for node in nodes]
        }
        for property, nodes in arcs.items()
    }


@pytest.mark.parametrize("batches_repeat", [False, True])
def test_a_complete_enumeration_copies_every_variable_and_its_arcs(batches_repeat):
    # One subject a page, so the group walk and the member lists both page.
    un = FakeUn(page_size=1)
    un.batches_repeat = batches_repeat

    built = build(un)

    assert built.faults == ()
    found = built.mirror
    assert found is not None
    assert set(found.variables) == {MORT, MORT_FEMALE, MORT_MALE, ODA}
    # What `node_arcs_batch` answers for the variable, less the provenance id.
    for dcid, row in found.variables.items():
        assert row.wire_arcs() == _without_provenance(un.nodes[dcid])
    assert found.variables[MORT_FEMALE].series == MORT_SERIES
    assert found.series_members() == {
        MORT_SERIES: (MORT, MORT_FEMALE, MORT_MALE),
        ODA_SERIES: (ODA,),
    }
    assert found.excluded_variables == {}
    assert found.agencies == {
        "who": UnAgencyCounts(series=2, variables=4, excluded_series=0, excluded_variables=0)
    }
    [stats] = built.agencies
    assert (stats.segment, stats.series, stats.variables, stats.counted) == ("who", 2, 4, 4)
    assert stats.requests > 0
    assert un.count_reads == [WHO_ROOT]


def test_the_topic_table_holds_what_the_topic_walk_reaches():
    found = mirror(FakeUn())

    assert found.topics == {
        TOPIC_ROOT: (MORT, ODA, MORT_FEMALE, MORT_MALE),
        WHO_TOPIC: (MORT, ODA, MORT_FEMALE, MORT_MALE),
        MORT_TOPIC: (MORT, MORT_FEMALE, MORT_MALE),
        ODA_TOPIC: (ODA,),
    }
    # A peer group is walked through and is no topic of its own.
    assert MORT_PEERS not in found.topics


def test_an_excluded_series_keeps_its_variables_out_of_every_admitted_table():
    found = mirror(FakeUn(), ODA_SERIES)

    assert found.excluded_variables == {ODA: ODA_SERIES}
    assert found.exclusions == {ODA_SERIES: CRS}
    assert found.excluded_series_names == {ODA_SERIES: f"{ODA_SERIES} name"}
    assert ODA not in found.variables
    assert ODA_TOPIC not in found.topics
    assert all(ODA not in leaves for leaves in found.topics.values())
    # The agency's count still names every member variable the UN counts.
    assert found.agencies["who"] == UnAgencyCounts(
        series=2, variables=4, excluded_series=1, excluded_variables=1
    )


def test_a_topic_leaf_outside_the_variable_table_is_dropped():
    un = FakeUn()
    un.nodes[ODA_TOPIC]["relevantVariable"].append(
        target("undata/dpo/UNREGISTERED", None, "StatisticalVariable")
    )

    assert mirror(un).topics[ODA_TOPIC] == (ODA,)


def test_a_node_whose_arcs_page_is_read_to_the_end():
    un = FakeUn()
    un.arc_targets_per_page = 3

    found = mirror(un)

    assert found.variables[MORT_FEMALE].wire_arcs() == _without_provenance(un.nodes[MORT_FEMALE])
    assert found.topics[MORT_TOPIC] == (MORT, MORT_FEMALE, MORT_MALE)


def _group_page(un: FakeUn) -> str:
    un.page_size = 1
    return MORT_BY_SEX


def _topic_page(un: FakeUn) -> str:
    un.arc_targets_per_page = 2
    return MORT_TOPIC


_PAGED_NODE = pytest.mark.parametrize(("paged", "arc"), [(_group_page, "<-"), (_topic_page, "->*")])


@_PAGED_NODE
def test_a_page_that_repeats_its_token_is_requested_again_and_the_table_is_whole(
    monkeypatch, paged, arc
):
    waits: list[float] = []
    monkeypatch.setattr(un_mirror.time, "sleep", waits.append)
    un = FakeUn()
    node = paged(un)
    whole = mirror(FakeUn())
    un.replaying = {node: 3}

    built = build(un)

    assert built.faults == ()
    assert built.mirror == whole
    assert len(waits) == 3
    assert sum(stats.page_retries for stats in built.agencies) == (3 if arc == "<-" else 0)


@_PAGED_NODE
def test_a_page_that_repeats_its_token_on_every_attempt_is_a_stall(monkeypatch, paged, arc):
    waits: list[float] = []
    monkeypatch.setattr(un_mirror.time, "sleep", waits.append)
    un = FakeUn()
    node = paged(un)
    # More replays than one read's attempts, as the group walk reads a node
    # again alone when a read naming several stops.
    un.replaying = {node: 100}

    built = build(un)

    assert built.mirror is None
    assert [(fault.kind, fault.subject) for fault in built.faults] == [
        ("un_enumeration_stalled", node)
    ]
    assert arc in built.faults[0].detail


def test_a_repeated_page_is_not_requested_again_past_the_deadline(monkeypatch):
    monkeypatch.setattr(un_mirror.time, "sleep", pytest.fail)
    un = FakeUn(page_size=1)
    un.replaying = {MORT_BY_SEX: 100}

    built = un_mirror.build_un_mirror(
        registry(), graph=un, group_counts=un, deadline=time.monotonic()
    )

    assert [fault.kind for fault in built.faults] == ["un_enumeration_stalled"]


def _stalled(un: FakeUn) -> None:
    un.page_size = 1
    un.idle = {MORT_BY_SEX}


def _repeated_token(un: FakeUn) -> None:
    un.page_size = 1
    un.repeating = {MORT_BY_SEX}


def _truncated_member_list(un: FakeUn) -> None:
    un.truncating = {MORT_BY_SEX: 1}


def _truncated_group_list(un: FakeUn) -> None:
    un.truncating = {MORT_GROUP: 0}


def _missing_count(un: FakeUn) -> None:
    un.counts = {WHO_ROOT: None}


def _unknown_child_group(un: FakeUn) -> None:
    un.counts = {"undata/g/who/UNWALKED": 3}


def _variable_without_series(un: FakeUn) -> None:
    un.nodes[ODA] = variable(name="Aid for health", series=None, member_of=ODA_GROUP)


@pytest.mark.parametrize(
    ("hazard", "kind", "subject"),
    [
        (_stalled, "un_enumeration_stalled", MORT_BY_SEX),
        (_repeated_token, "un_enumeration_stalled", MORT_BY_SEX),
        (_truncated_member_list, "un_variable_count_mismatch", WHO),
        (_truncated_group_list, "un_variable_count_mismatch", WHO),
        (_missing_count, "un_variable_count_mismatch", WHO),
        (_unknown_child_group, "un_variable_count_mismatch", WHO),
        (_variable_without_series, "un_variable_without_series", ODA),
    ],
)
def test_an_enumeration_that_is_not_complete_has_a_fault_and_no_tables(hazard, kind, subject):
    un = FakeUn()
    hazard(un)

    built = build(un)

    assert built.mirror is None
    assert [(fault.kind, fault.subject) for fault in built.faults] == [(kind, subject)]


def _unanswered_topic(un: FakeUn) -> None:
    del un.nodes[MORT_PEERS]


def _unanswered_root(un: FakeUn) -> None:
    del un.nodes[TOPIC_ROOT]


def _topic_free_tree(un: FakeUn) -> None:
    un.nodes[TOPIC_ROOT] = topic(target(WHO_TOPIC, None, "StatVarPeerGroup"), kind="Thing")
    un.nodes[WHO_TOPIC] = topic(arc="member", kind="StatVarPeerGroup")


_UNICEF_ROOT = "undata/g/unicef"


_UNICEF_ODA = "undata/unicef/ODA_IN"


def _two_agencies(*excluded: str) -> Registry:
    exclusions = ", ".join(
        f'{{ series = "{series}", reason = "superseded_by_dfo", dfo_provenance = "{CRS}" }}'
        for series in excluded
    )
    return parse_registry(
        f"""
base = []

[[un_system]]
provenance = "{WHO}"
segment = "who"
label = "World Health Organization (WHO)"
exclusions = [{exclusions}]

[[un_system]]
provenance = "undata/p/UNICEF"
segment = "unicef"
label = "UNICEF"
""".encode()
    )


def _variable_under_two_agencies(un: FakeUn) -> None:
    un.nodes[ODA] = {**un.nodes[ODA], "memberOf": [target(ODA_GROUP), target(_UNICEF_ROOT)]}
    un.nodes[_UNICEF_ROOT] = group("undata/g/Root", "UNICEF (UN Data)")


@pytest.mark.parametrize(
    ("hazard", "kind", "subject"),
    [
        (_unanswered_topic, "un_topic_unanswered", MORT_PEERS),
        (_unanswered_root, "un_topic_unanswered", TOPIC_ROOT),
        (_topic_free_tree, "un_topics_empty", TOPIC_ROOT),
    ],
)
def test_a_topic_tree_that_is_not_whole_is_an_incomplete_enumeration(hazard, kind, subject):
    un = FakeUn()
    hazard(un)

    built = build(un)

    assert built.mirror is None
    assert [(fault.kind, fault.subject) for fault in built.faults] == [(kind, subject)]
    assert built.faults[0].incomplete


def _variable_of_another_agencys_excluded_series(un: FakeUn) -> tuple[Registry, str]:
    un.nodes[_UNICEF_ROOT] = group("undata/g/Root", "UNICEF (UN Data)")
    un.nodes[_UNICEF_ODA] = variable(name="Aid", series=ODA_SERIES, member_of=_UNICEF_ROOT)
    return _two_agencies(ODA_SERIES), _UNICEF_ODA


@pytest.mark.parametrize("hazard", [_variable_of_another_agencys_excluded_series])
def test_tables_breaking_their_invariants_are_an_incomplete_enumeration(hazard):
    un = FakeUn()
    registry, subject = hazard(un)

    built = un_mirror.build_un_mirror(registry, graph=un, group_counts=un)

    assert built.mirror is None
    assert [(fault.kind, fault.subject) for fault in built.faults] == [
        ("un_tables_inconsistent", subject)
    ]
    assert built.faults[0].incomplete


def _unicef_variable_under_who(un: FakeUn) -> tuple[Registry, str]:
    foreign = "undata/unicef/MORT"
    un.nodes[foreign] = variable(name="Mortality", series=MORT_SERIES, member_of=MORT_GROUP)
    return registry(), foreign


def _who_variable_under_two_agencies(un: FakeUn) -> tuple[Registry, str]:
    _variable_under_two_agencies(un)
    return _two_agencies(), ODA


@pytest.mark.parametrize("hazard", [_unicef_variable_under_who, _who_variable_under_two_agencies])
def test_a_variable_outside_its_agencys_prefix_is_an_incomplete_enumeration(hazard):
    un = FakeUn()
    registry_, foreign = hazard(un)

    built = un_mirror.build_un_mirror(registry_, graph=un, group_counts=un)

    assert built.mirror is None
    assert [(fault.kind, fault.subject) for fault in built.faults] == [
        ("un_variable_outside_agency", foreign)
    ]
    assert built.faults[0].incomplete


def test_an_excluded_series_no_variable_names_is_a_fault():
    built = build(FakeUn(), "undata/who/SERIES-GONE")

    assert built.mirror is None
    assert [(fault.kind, fault.subject) for fault in built.faults] == [
        ("un_excluded_series_missing", "undata/who/SERIES-GONE")
    ]


def test_an_excluded_series_no_variable_names_with_a_name_is_a_fault():
    un = FakeUn()
    un.nodes[ODA]["populationType"] = [target(ODA_SERIES, None, "undata/who/SERIES")]

    built = build(un, ODA_SERIES)

    assert built.mirror is None
    assert [(fault.kind, fault.subject, fault.incomplete) for fault in built.faults] == [
        ("un_excluded_series_unnamed", ODA_SERIES, False)
    ]


def test_a_failed_un_read_raises():
    un = FakeUn()
    un.failing = True

    with pytest.raises(GraphReadError):
        build(un)


def _named_together(un: FakeUn, dcids: set[str], arc: str) -> bool:
    """Whether some read of `arc` named every dcid of `dcids`."""
    return any(
        request["property"] == arc and dcids <= set(request["nodes"]) for request in un.requests
    )


@pytest.mark.parametrize(
    ("refused", "arc"),
    [
        # Halved twice: the variable chunk's first half still names both.
        ({MORT, MORT_FEMALE}, "->*"),
        ({MORT_TOPIC, ODA_TOPIC}, "->*"),
        ({MORT_GROUP, ODA_GROUP}, "<-specializationOf"),
        ({MORT_GROUP, ODA_GROUP}, "<-memberOf"),
    ],
)
def test_a_read_the_front_door_refuses_is_read_in_halves(refused, arc):
    un = FakeUn()
    un.refused = [refused]

    built = build(un)

    assert _named_together(un, refused, arc)
    assert built.faults == ()
    assert built.mirror == mirror(FakeUn())


@pytest.mark.parametrize(("node", "arc"), [(ODA, "->*"), (MORT_BY_SEX, "<-specializationOf")])
def test_a_node_the_front_door_refuses_on_its_own_is_an_incomplete_enumeration(node, arc):
    un = FakeUn()
    un.refused = [{node}]

    built = build(un)

    assert built.mirror is None
    assert [(fault.kind, fault.subject, fault.incomplete) for fault in built.faults] == [
        ("un_read_refused", node, True)
    ]
    assert arc in built.faults[0].detail


def _batches_repeat(un: FakeUn) -> None:
    un.batches_repeat = True


def _member_list_repeats(un: FakeUn) -> None:
    un.repeating = {MORT_BY_SEX}


@pytest.mark.parametrize(
    ("hazard", "faults"),
    [
        # The half naming MORT_BY_SEX and ODA_GROUP stalls, and each reads well alone.
        (_batches_repeat, []),
        (_member_list_repeats, [("un_enumeration_stalled", MORT_BY_SEX)]),
    ],
)
def test_a_half_of_a_refused_read_that_stops_advancing_is_read_one_node_at_a_time(hazard, faults):
    un = FakeUn(page_size=1)
    refused = {MORT_GROUP, ODA_GROUP}
    un.refused = [refused]
    hazard(un)

    built = build(un)

    assert _named_together(un, refused, "<-memberOf")
    assert [(fault.kind, fault.subject) for fault in built.faults] == faults
    assert built.mirror == (None if faults else mirror(FakeUn()))


@pytest.mark.parametrize(("limited", "completes"), [(3, True), (4, False)])
def test_a_rate_limited_read_is_repeated_after_a_wait(monkeypatch, limited, completes):
    waits: list[float] = []
    monkeypatch.setattr(un_mirror.time, "sleep", waits.append)
    monkeypatch.setattr(un_mirror, "UN_MIRROR_CONCURRENCY", 1)
    un = FakeUn()
    un.rate_limited = limited

    if completes:
        assert set(mirror(un).variables) == {MORT, MORT_FEMALE, MORT_MALE, ODA}
    else:
        with pytest.raises(GraphReadError) as raised:
            build(un)
        assert raised.value.upstream_status == 429
    assert waits == [10.0, 30.0, 60.0]


def test_at_most_four_requests_are_in_flight():
    nodes = dict(FakeUn().nodes)
    for index in range(300):
        nodes[f"undata/who/MORT.AGE--{index}"] = variable(
            name=f"Mortality [Age={index}]", series=MORT_SERIES, member_of=MORT_BY_SEX
        )
    un = FakeUn(nodes, page_size=50)
    un.delay = 0.002

    found = mirror(un)

    assert len(found.variables) == 304
    assert un.max_in_flight == un_mirror.UN_MIRROR_CONCURRENCY == 4


def test_the_tables_refuse_an_admitted_variable_of_an_excluded_series():
    with pytest.raises(un_mirror.UnMirrorTableError, match="excluded series"):
        UnMirror(
            variables={ODA: UnVariable(series=ODA_SERIES, arcs=())},
            excluded_variables={},
            topics={},
            exclusions={ODA_SERIES: CRS},
            excluded_series_names={ODA_SERIES: "Aid for health"},
            agencies={
                "who": UnAgencyCounts(
                    series=1, variables=1, excluded_series=1, excluded_variables=0
                )
            },
        )


def test_the_tables_refuse_an_excluded_series_with_no_name():
    with pytest.raises(un_mirror.UnMirrorTableError, match="names held"):
        UnMirror(
            variables={},
            excluded_variables={},
            topics={},
            exclusions={ODA_SERIES: CRS},
            excluded_series_names={},
            agencies={},
        )


# --- Group counts ------------------------------------------------------------


def _site(handler) -> UnSiteClient:
    return UnSiteClient(
        site_url="https://un.example",
        timeout=5.0,
        user_agent="test",
        transport=httpx.MockTransport(handler),
    )


def test_the_site_client_reads_the_group_and_child_counts():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "descendentStatVarCount": 4,
                "childStatVarGroups": [
                    {"id": MORT_GROUP, "descendentStatVarCount": 3},
                    {"id": ODA_GROUP, "descendentStatVarCount": 1},
                ],
            },
        )

    assert _site(handler)(WHO_ROOT) == {WHO_ROOT: 4, MORT_GROUP: 3, ODA_GROUP: 1}
    [request] = seen
    assert request.url == "https://un.example/api/variable-group/info"
    assert b'"dcid":"undata/g/who"' in request.content.replace(b" ", b"")


def test_the_site_client_gives_no_count_for_a_group_the_un_does_not_know():
    site = _site(lambda request: httpx.Response(200, json={"childStatVarGroups": []}))

    assert site(WHO_ROOT) == {}


def test_the_site_client_retries_a_5xx_then_raises(monkeypatch):
    monkeypatch.setattr(un_mirror.time, "sleep", lambda seconds: None)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(GraphReadError) as raised:
        _site(handler)(WHO_ROOT)

    assert len(calls) == 3
    assert raised.value.upstream_status == 503


def test_the_site_url_is_the_api_root_without_its_suffix():
    assert un_mirror.un_site_url("https://un.example/core/api/") == "https://un.example"
    assert un_mirror.un_site_url("https://un.example/v2") is None
