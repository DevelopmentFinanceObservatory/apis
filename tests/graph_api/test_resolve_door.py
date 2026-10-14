"""Resolve-door tests use the shared resolver and an in-memory graph reader.
The same fake serves entity confirmation and the names of expanded region members.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from entity_resolution import EntityResolutionError, Matched, NotFound, Resolution, Resolved
from entity_resolution.membership import DEFAULT_PROJECTION
from graph_api.deps import GraphDeps
from graph_api.doors import resolve as resolve_door
from graph_api.doors.node import NODE_MEMBERSHIP_SLOT
from graph_api.doors.sources import new_provenance_doc_cache
from graph_api.kinds import KINDS
from graph_api.resolve_models import ResolveRequest
from materialise.graph import GraphDeadlineExceededError, GraphReadError

from .conftest import (
    FakeGraphNodeReader,
    make_client,
    make_resolver,
    send_past_a_full_pool,
)


def _post(shared_resolver, graph, body: dict, *, policy: bool = False):
    """Post `body` to the resolve door over `graph`. `policy` publishes the shared membership
    policy on the app, as the surface's lifespan does."""
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    if policy:
        NODE_MEMBERSHIP_SLOT.set(client.app, shared_resolver.policy)
    with client:
        return client.post("/api/graph/resolve", json=body)


def _result(shared_resolver, graph, text: str, *, policy: bool = False, **body) -> dict:
    resp = _post(shared_resolver, graph, {"texts": [text], **body}, policy=policy)
    assert resp.status_code == 200, resp.text
    (result,) = resp.json()["results"]
    return result


def _unused(**fields) -> dict:
    """A result's fields with every field its status leaves unused set to null."""
    empty = dict.fromkeys(
        (
            "entity",
            "matched",
            "reading",
            "projection",
            "member_count",
            "members",
            "candidates",
            "near_matches",
        )
    )
    return {**empty, **fields}


def test_kenya_resolved_and_confirmed(shared_resolver):
    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}})

    result = _result(shared_resolver, graph, "Kenya")

    assert result == {
        "input": "Kenya",
        "status": "resolved",
        **_unused(
            entity={"ref": "country/KEN", "label": "Kenya", "kind": "country"},
            matched={"text": "Kenya", "as": "name"},
            reading="entity",
        ),
    }


def test_kenya_resolved_but_not_held_reports_not_held(shared_resolver):
    result = _result(shared_resolver, FakeGraphNodeReader(nodes={}), "Kenya")

    assert result == {
        "input": "Kenya",
        "status": "not_held",
        **_unused(entity={"ref": "country/KEN", "label": "Kenya", "kind": None}),
    }


def test_below_confidence_name_refuses_as_not_found_with_near_matches(shared_resolver):
    """Cash falls below the match threshold and returns the rejected candidates as near
    matches, each a reference with its kind."""
    result = _result(shared_resolver, FakeGraphNodeReader(nodes={}), "Cash")

    assert result["status"] == "not_found"
    assert result["near_matches"], "expected near matches for 'Cash'"
    assert all(match["ref"] and match["label"] for match in result["near_matches"])


def test_nonsense_text_refuses_as_not_found_with_no_near_matches(shared_resolver):
    result = _result(shared_resolver, FakeGraphNodeReader(nodes={}), "zzqqxv")

    assert result == {"input": "zzqqxv", "status": "not_found", **_unused(near_matches=[])}


def test_congo_ambiguous_when_graph_holds_both_candidates(shared_resolver):
    graph = FakeGraphNodeReader(
        nodes={
            "country/COD": {"typeOf": ("Country",), "name": ("Congo [DRC]",)},
            "country/COG": {"typeOf": ("Country",), "name": ("Congo [Republic]",)},
        }
    )

    result = _result(shared_resolver, graph, "Congo")

    assert result["status"] == "ambiguous"
    assert result["entity"] is None
    assert {c["ref"]: c["kind"] for c in result["candidates"]} == {
        "country/COD": "country",
        "country/COG": "country",
    }


def test_congo_resolves_when_graph_holds_exactly_one_candidate(shared_resolver):
    graph = FakeGraphNodeReader(
        nodes={"country/COD": {"typeOf": ("Country",), "name": ("Congo [DRC]",)}}
    )

    result = _result(shared_resolver, graph, "Congo")

    assert result["status"] == "resolved"
    assert result["entity"]["ref"] == "country/COD"


def test_congo_not_held_when_graph_holds_neither_candidate(shared_resolver):
    resp = _post(shared_resolver, FakeGraphNodeReader(nodes={}), {"texts": ["Congo"]})

    assert resp.status_code == 200, resp.text
    (result,) = resp.json()["results"]
    assert result["status"] == "not_held"
    assert result["entity"]["ref"] == "country/COD"
    assert "country/COG" not in resp.text


@pytest.mark.parametrize(
    ("text", "nodes", "near_match"),
    [
        # One-edit spelling correction maps Lagos to Laos; the listed-name gate rejects it.
        ("Lagos", {"country/LAO": {"typeOf": ("Country",)}}, "country/LAO"),
        ("Nigria", {"country/NGA": {"typeOf": ("Country",)}}, "country/NGA"),
    ],
)
def test_a_corrected_winner_demotes_to_not_found_with_the_correction_as_near_match(
    shared_resolver, text: str, nodes: dict, near_match: str
):
    result = _result(shared_resolver, FakeGraphNodeReader(nodes=nodes), text)

    assert result["status"] == "not_found"
    assert result["entity"] is None
    assert [match["ref"] for match in result["near_matches"]] == [near_match]
    assert result["near_matches"][0]["kind"] == "country"


@pytest.mark.parametrize(
    ("text", "dcid"),
    [
        ("Laos", "country/LAO"),
        ("Kenia", "country/KEN"),
        ("Britain", "country/GBR"),
        ("Côte d'Ivoire", "country/CIV"),
        ("DRC", "country/COD"),
        ("UK", "country/GBR"),
        ("USA", "country/USA"),
    ],
)
def test_listed_alias_or_code_stays_resolved(text: str, dcid: str, shared_resolver):
    """Aliases, country codes, and canonical names resolve directly."""
    graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": ("Country",)}})

    result = _result(shared_resolver, graph, text)

    assert result["status"] == "resolved", (text, result)
    assert result["entity"]["ref"] == dcid


@pytest.mark.parametrize(
    ("text", "label", "dcid", "graph_type", "kind"),
    [
        (
            "African Development Bank",
            "African Development Bank",
            "AfricanDevelopmentBank",
            "Organization",
            "organisation",
        ),
        (
            "AfDB",
            "African Development Bank",
            "AfricanDevelopmentBank",
            "Organization",
            "organisation",
        ),
        (
            "EU Institutions",
            "European Union institutions",
            "EuropeanUnionInstitutions",
            "Organization",
            "organisation",
        ),
        (
            "Africa, regional",
            "Africa, unallocated by country (DAC)",
            "DACAfricaUnallocated",
            "DACRecipientUnallocatedByCountry",
            "aggregate",
        ),
        (
            "official donors",
            "official donors (DAC)",
            "DACOfficialDonors",
            "DACGroup",
            "aggregate",
        ),
        ("20005", "official donors (DAC)", "DACOfficialDonors", "DACGroup", "aggregate"),
        ("20001", "DAC countries", "DACCountriesTotal", "DACGroup", "aggregate"),
    ],
)
def test_organisation_or_aggregate_resolves_with_kind(
    text: str, label: str, dcid: str, graph_type: str, kind: str, shared_resolver
):
    graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": (graph_type,), "name": (label,)}})

    result = _result(shared_resolver, graph, text)

    assert result["status"] == "resolved"
    assert result["entity"] == {"ref": dcid, "label": label, "kind": kind}
    assert result["members"] is None


def test_world_bank_curated_ambiguity_names_ibrd_and_ida(shared_resolver):
    """The curated alias maps World Bank to both IBRD and IDA."""
    graph = FakeGraphNodeReader(
        nodes={
            "InternationalBankForReconstructionAndDevelopment": {"typeOf": ("Organization",)},
            "InternationalDevelopmentAssociation": {"typeOf": ("Organization",)},
        }
    )

    result = _result(shared_resolver, graph, "World Bank")

    assert result["status"] == "ambiguous"
    assert {c["ref"] for c in result["candidates"]} == {
        "InternationalBankForReconstructionAndDevelopment",
        "InternationalDevelopmentAssociation",
    }
    assert all(c["kind"] == "organisation" for c in result["candidates"])


@pytest.mark.parametrize(
    ("text", "ref", "member_count"),
    [("G7", "GroupOfSeven", 7), ("OECD", "undata-geo/G00406000", 38), ("EU", "EuropeanUnion", 27)],
)
def test_a_group_resolves_from_the_registry_when_the_graph_holds_none_of_it(
    shared_resolver, text: str, ref: str, member_count: int
):
    """The membership registry confirms a group and its members, so a graph holding no node
    still resolves the group with its count and a link, and no graph read names a member."""
    graph = FakeGraphNodeReader(nodes={})

    result = _result(shared_resolver, graph, text)

    assert result["status"] == "resolved"
    assert result["entity"]["ref"] == ref
    assert result["entity"]["kind"] == "group"
    assert result["projection"] == "countries"
    assert result["member_count"] == member_count
    assert result["members"] == (
        f"/api/graph/node/{ref}/members?relation=members&projection=countries"
    )
    assert graph.calls["node_names_many"] == 0


def test_the_eu_resolves_as_a_group_when_the_graph_holds_its_node(shared_resolver):
    """The group, not its graph node, is the entity: the node's `ContinentalUnion` type does not
    change the kind."""
    graph = FakeGraphNodeReader(
        nodes={"EuropeanUnion": {"typeOf": ("ContinentalUnion",), "name": ("European Union",)}}
    )

    result = _result(shared_resolver, graph, "EU")

    assert result["entity"]["ref"] == "EuropeanUnion"
    assert result["entity"]["kind"] == "group"
    assert result["member_count"] == 27


def test_expand_members_inlines_a_groups_registry_members(shared_resolver):
    graph = FakeGraphNodeReader(nodes={})

    result = _result(shared_resolver, graph, "G7", expand=["members"])

    refs = [member["ref"] for member in result["members"]]
    assert len(refs) == result["member_count"] == 7
    assert refs == sorted(refs)
    assert "country/JPN" in refs
    assert all(member["label"] for member in result["members"])
    assert graph.calls["node_names_many"] == 0


@pytest.mark.parametrize(
    ("text", "ref", "kind", "member_count"),
    [
        ("Africa", "africa", "continent", 54),
        ("Sub-Saharan Africa", "SubSaharanAfrica", "subregion", 48),
    ],
)
def test_a_region_takes_its_count_and_link_from_the_membership_policy(
    shared_resolver, text: str, ref: str, kind: str, member_count: int
):
    """Graph containment is not read: the fake graph's containment would give one member."""
    graph = FakeGraphNodeReader(
        nodes={ref: {"typeOf": ("Continent" if kind == "continent" else "UNGeoRegion",)}},
        contained={(ref, "Country"): ["country/AGO"]},
    )

    result = _result(shared_resolver, graph, text, policy=True)

    assert result["status"] == "resolved"
    assert result["entity"]["kind"] == kind
    assert result["projection"] == "countries"
    assert result["member_count"] == member_count
    assert result["members"] == (
        f"/api/graph/node/{ref}/members?relation=countries&projection=countries"
    )
    assert graph.calls["child_places_batch"] == 0
    assert graph.calls["node_names_many"] == 0


def test_a_region_carries_no_membership_where_no_policy_is_published(shared_resolver):
    graph = FakeGraphNodeReader(nodes={"africa": {"typeOf": ("Continent",)}})

    result = _result(shared_resolver, graph, "Africa")

    assert result["entity"]["kind"] == "continent"
    assert (result["projection"], result["member_count"], result["members"]) == (None, None, None)


def test_expand_members_labels_a_regions_members_from_its_policy(shared_resolver):
    """A batch expanding two regions labels each region and its members with the policy's
    canonical names, with no graph read, whatever the graph names them; a territory the policy
    files outside `countries` is absent."""
    graph = FakeGraphNodeReader(
        nodes={
            "africa": {"typeOf": ("Continent",), "name": ("Graph name",)},
            "MiddleAfrica": {"typeOf": ("UNGeoRegion",)},
            "country/AGO": {"typeOf": ("Country",), "name": ("Angola",)},
        },
    )
    resp = _post(
        shared_resolver,
        graph,
        {"texts": ["Africa", "Middle Africa"], "expand": ["members"]},
        policy=True,
    )

    assert resp.status_code == 200, resp.text
    africa, middle = resp.json()["results"]
    by_ref = {member["ref"]: member["label"] for member in africa["members"]}
    assert len(africa["members"]) == len(by_ref) == africa["member_count"] == 54
    # The region, like its members, takes the policy's name, as the node route labels it.
    assert africa["entity"]["label"] == shared_resolver.policy.get("africa").label
    assert (by_ref["country/AGO"], by_ref["country/KEN"]) == ("Angola", "Kenya")
    assert None not in by_ref.values()
    assert "country/ESH" not in by_ref
    assert len(middle["members"]) == middle["member_count"] == 9
    assert graph.calls["node_names_many"] == 0


def test_resolving_a_subregion_and_a_continent_leaks_no_internal_dcid_prefix(shared_resolver):
    """Region responses must not expose `m49/` or `wikidataId/` dcids, which are internal graph
    namespaces.
    """
    graph = FakeGraphNodeReader(
        nodes={
            "SubSaharanAfrica": {"typeOf": ("UNGeoRegion",)},
            "africa": {"typeOf": ("Continent",)},
        },
    )
    resp = _post(
        shared_resolver,
        graph,
        {"texts": ["Sub-Saharan Africa", "Africa"], "expand": ["members"]},
        policy=True,
    )

    assert resp.status_code == 200, resp.text
    assert "m49/" not in resp.text
    assert "wikidataId/" not in resp.text


def test_a_region_the_policy_holds_resolves_when_the_graph_holds_no_node_for_it(
    shared_resolver,
):
    """The membership policy vouches for a region as for a group: with no graph node it
    resolves with the policy's kind and count."""
    result = _result(
        shared_resolver, FakeGraphNodeReader(nodes={}), "Sub-Saharan Africa", policy=True
    )

    count = shared_resolver.policy.get("SubSaharanAfrica").member_count(DEFAULT_PROJECTION)
    assert (result["status"], result["entity"]["kind"]) == ("resolved", "subregion"), result
    assert result["entity"]["ref"] == "SubSaharanAfrica"
    assert result["member_count"] == count


def test_every_result_echoes_its_input_in_request_order(shared_resolver):
    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    texts = ["  Kenya ", "zzqqxv", "G7"]

    resp = _post(shared_resolver, graph, {"texts": texts})

    assert resp.status_code == 200, resp.text
    assert [r["input"] for r in resp.json()["results"]] == ["Kenya", "zzqqxv", "G7"]


_CODE_GRAPH = FakeGraphNodeReader(
    nodes={"CRSPurposeCode12220": {"typeOf": ("CRSPurposeCodeEnum",), "name": ("x",)}}
)


@pytest.mark.parametrize(
    ("text", "code_system", "matched"),
    [
        ("basic health care", None, {"text": "basic health care", "as": "name"}),
        ("PHC", None, {"text": "PHC", "as": "alias"}),
        ("basic health care", "CRSPurposeCodeEnum", {"text": "basic health care", "as": "name"}),
        # A code matches casefolded, with whitespace collapsed, and names the text it lists.
        ("Basic  Health Care", None, {"text": "basic health care", "as": "name"}),
        ("phc", None, {"text": "PHC", "as": "alias"}),
    ],
)
def test_a_code_resolves_by_its_label_or_synonym(
    shared_resolver, text: str, code_system: str | None, matched: dict
):
    resp = _post(
        shared_resolver,
        _CODE_GRAPH,
        {"texts": [text], "kinds": ["code"], "code_system": code_system},
    )

    assert resp.status_code == 200, resp.text
    (result,) = resp.json()["results"]
    assert result["status"] == "resolved"
    assert result["entity"] == {
        "ref": "CRSPurposeCode12220",
        "label": "basic health care",
        "kind": "code",
    }
    assert result["matched"] == matched


@pytest.mark.parametrize(
    "texts",
    [("Basic health care", "basic health care"), ("basic health care", "BASIC HEALTH CARE")],
)
def test_two_casings_of_a_code_give_one_answer_whichever_comes_first(shared_resolver, texts):
    """The resolver caches by the folded text, so the second casing reads the first's entry."""
    resolver = make_resolver(shared_resolver, graph=_CODE_GRAPH)

    async def resolve_each() -> list[Resolution]:
        return [(await resolver.resolve([text], kinds=["code"]))[0] for text in texts]

    first, second = asyncio.run(resolve_each())

    assert first == second
    assert isinstance(first, Resolved), first
    assert first.matched == Matched(text="basic health care", matched_as="name")


def test_code_system_leaves_out_the_codes_of_every_other_system(shared_resolver):
    resp = _post(
        shared_resolver,
        _CODE_GRAPH,
        {"texts": ["basic health care"], "kinds": ["code"], "code_system": "DAC1AidTypeEnum"},
    )

    assert resp.status_code == 200, resp.text
    (result,) = resp.json()["results"]
    assert result["status"] == "not_found"
    assert result["near_matches"] == []


def test_capabilities_publish_the_kind_list_and_the_kinds_resolve_accepts():
    with make_client() as client:
        capabilities = client.get("/api/graph/capabilities").json()

    assert capabilities["kinds"] == list(KINDS)
    assert set(capabilities["resolve_kinds"]) < set(KINDS)
    assert {"group", "code", "country"} <= set(capabilities["resolve_kinds"])
    assert "stat_var" not in capabilities["resolve_kinds"]


class TestGraphHoldingNothingNeverConfirms:
    """An empty reader returns `not_held` for listed entities and `not_found`
    for text with no listed candidate. Groups and regions, which the membership policy holds,
    are covered by `test_a_group_resolves_from_the_registry_when_the_graph_holds_none_of_it`
    and `test_a_region_the_policy_holds_resolves_when_the_graph_holds_no_node_for_it`.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Kenya",
            "Cash",
            "zzqqxv",
            "Congo",
            "World Bank",
        ],
    )
    def test_never_resolves(self, text: str, shared_resolver):
        status = _result(shared_resolver, FakeGraphNodeReader(nodes={}), text)["status"]

        assert status in {"not_held", "not_found"}, (text, status)


class TestValidation:
    def test_empty_texts_list_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": []})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_unknown_top_level_field_carries_pydantic_message_as_detail(self):
        with make_client() as client:
            resp = client.post(
                "/api/graph/resolve", json={"texts": ["Kenya"], "extra_field": "nope"}
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "Extra inputs are not permitted"

    def test_over_25_texts_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"] * 26})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_empty_string_text_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": [""]})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_201_char_text_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": ["a" * 201]})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    @pytest.mark.parametrize(
        ("body", "pointer"),
        [
            ({"kinds": []}, "/kinds"),
            # `stat_var` is on the shared kind list, but resolve never returns one.
            ({"kinds": ["stat_var"]}, "/kinds/0"),
            ({"expand": ["candidates"]}, "/expand/0"),
            # The default kinds leave out `code`.
            ({"code_system": "CRSPurposeCodeEnum"}, "/code_system"),
        ],
    )
    def test_a_filter_resolve_cannot_apply_refuses_at_its_member(self, body: dict, pointer: str):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"], **body})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"
        assert resp.json()["pointer"] == pointer


class _BlockingGraphNodeReader:
    """Block `node_arcs_many` until the test releases it, so a node read holds the shared
    DCP-read worker. Count the reads that reach it."""

    def __init__(self, delegate: FakeGraphNodeReader) -> None:
        self._delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()
        self.entries = 0
        self.calls = delegate.calls

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.entries += 1
        self.entered.set()
        self.release.wait(timeout=5.0)
        return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)

    def resolve_by_arc(self, names, *, arc, deadline=None):
        return self._delegate.resolve_by_arc(names, arc=arc, deadline=deadline)

    def node_names_many(self, dcids, *, deadline=None):
        return self._delegate.node_names_many(dcids, deadline=deadline)


class _DeadlineBoundedGraphNodeReader:
    """Stall selected reads until their deadline, then raise
    `GraphDeadlineExceededError`. Each stall is capped at 5 seconds. Later
    reads delegate, and `observed_deadlines` records every deadline.
    """

    _MAX_BLOCK_SECONDS = 5.0

    def __init__(self, delegate: FakeGraphNodeReader, *, stall_reads: int = 1) -> None:
        self._delegate = delegate
        self._remaining_stalls = stall_reads
        self.entered = threading.Event()
        self.entered_at: float | None = None
        self.observed_deadlines = []
        self.calls = delegate.calls

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.observed_deadlines.append(deadline)
        if self._remaining_stalls <= 0:
            return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)
        self._remaining_stalls -= 1
        self.entered_at = time.monotonic()
        self.entered.set()
        if deadline is None:
            return self._delegate.node_arcs_many(dcids, properties=properties)
        remaining = min(max(deadline - time.monotonic(), 0.0), self._MAX_BLOCK_SECONDS)
        time.sleep(remaining)
        raise GraphDeadlineExceededError("the read stopped at the door's own deadline")

    def resolve_by_arc(self, names, *, arc, deadline=None):
        return self._delegate.resolve_by_arc(names, arc=arc, deadline=deadline)

    def node_names_many(self, dcids, *, deadline=None):
        return self._delegate.node_names_many(dcids, deadline=deadline)


def test_a_resolve_while_another_doors_read_holds_the_pool_is_refused_at_once():
    """A resolve runs its confirmation read on the DCP-read executor the node door shares, so
    while a node read holds the one worker it answers retryable 503 without reaching the
    reader. A resolve after the node read returns succeeds."""
    delegate = FakeGraphNodeReader(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}}
    )
    graph = _BlockingGraphNodeReader(delegate)
    # The lifespan's own resolver confirms on the surface's DCP-read executor.
    client = make_client(graph=graph, node_pool_size=1)
    sends = iter(
        [
            lambda: client.get("/api/graph/node/country/KEN"),
            lambda: client.post("/api/graph/resolve", json={"texts": ["Kenya"]}),
        ]
    )

    with client:
        responses = send_past_a_full_pool(
            client, lambda: next(sends)(), entered=graph.entered, release=graph.release
        )
        after = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    node, refused = responses
    assert node.status_code == 200, node.text
    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert refused.json()["retryable"] is True
    assert after.status_code == 200, after.text
    assert after.json()["results"][0]["entity"]["ref"] == "country/KEN"
    assert graph.entries == 2


def test_timed_out_request_returns_the_read_at_its_deadline_and_frees_capacity(
    shared_resolver, monkeypatch: pytest.MonkeyPatch
):
    """A read stalled until its 1-second deadline returns 504, and the next request
    succeeds."""
    monkeypatch.setattr(resolve_door, "GRAPH_RESOLVE_TIMEOUT_SECONDS", 1.0)
    delegate = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    graph = _DeadlineBoundedGraphNodeReader(delegate)
    started_at = time.monotonic()
    client = make_client(
        graph=graph, resolver=make_resolver(shared_resolver, graph=graph), node_pool_size=1
    )

    with client:
        first = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})
        assert first.status_code == 504, first.text
        assert first.json()["code"] == "timeout"
        assert graph.entered.wait(timeout=5.0), "confirmation read never started"

        # The read stops at the door deadline. It is at most one timeout
        # after read entry, or earlier if resolvekit used part of the budget.
        assert graph.observed_deadlines, "the read was handed no deadline"
        door_deadline = graph.observed_deadlines[0]
        assert door_deadline is not None
        assert graph.entered_at is not None
        assert door_deadline - graph.entered_at <= resolve_door.GRAPH_RESOLVE_TIMEOUT_SECONDS
        assert time.monotonic() - started_at < resolve_door.GRAPH_RESOLVE_TIMEOUT_SECONDS + 1.0

        second = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})
        assert second.status_code == 200, second.text
        (result,) = second.json()["results"]
        assert result["status"] == "resolved"
        assert result["entity"] == {"ref": "country/KEN", "label": "Kenya", "kind": "country"}


class TestGraphReadErrorClassification:
    def test_upstream_4xx_maps_to_502_upstream_error(self):
        problem = resolve_door.classify_resolve_exception(
            GraphReadError("exemplar", upstream_status=404)
        )
        assert problem is not None
        assert problem.status == 502
        assert problem.code == "upstream_error"
        assert problem.retryable is True

    def test_no_upstream_status_maps_to_503_service_unavailable(self):
        problem = resolve_door.classify_resolve_exception(GraphReadError("exemplar"))
        assert problem is not None
        assert problem.status == 503
        assert problem.code == "service_unavailable"
        assert problem.retryable is True

    def test_upstream_5xx_maps_to_503_service_unavailable(self):
        problem = resolve_door.classify_resolve_exception(
            GraphReadError("exemplar", upstream_status=502)
        )
        assert problem is not None
        assert problem.status == 503
        assert problem.code == "service_unavailable"
        assert problem.retryable is True

    def test_deadline_exceeded_maps_to_504_timeout_before_the_general_arm(self):
        # `GraphDeadlineExceededError` subclasses `GraphReadError`, so the
        # deadline arm must precede the general arm. Combined classification
        # also uses it for node and sources door deadlines.
        problem = resolve_door.classify_resolve_exception(GraphDeadlineExceededError("exemplar"))
        assert problem is not None
        assert problem.status == 504
        assert problem.code == "timeout"
        assert problem.retryable is True


class _RaisingGraphNodeReader:
    """Raise `GraphReadError` from graph reads for end-to-end classification tests."""

    def __init__(self, exc: GraphReadError) -> None:
        self._exc = exc

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        raise self._exc

    def node_names_many(self, dcids, *, deadline=None):
        raise self._exc


def test_graph_read_error_end_to_end_maps_to_502(caplog: pytest.LogCaptureFixture, shared_resolver):
    graph = _RaisingGraphNodeReader(GraphReadError("upstream refused", upstream_status=404))
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client, caplog.at_level(logging.ERROR):
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 502, resp.text
    assert resp.json()["code"] == "upstream_error"


def test_graph_read_error_end_to_end_maps_to_503_without_upstream_status(shared_resolver):
    graph = _RaisingGraphNodeReader(GraphReadError("transport failure"))
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"


class _StubErrorResolver:
    """Stub that raises `EntityResolutionError` to test the door's 500 response."""

    async def resolve(self, texts: list[str], **_: object) -> list[Resolution]:
        raise EntityResolutionError("resolvekit reported an internal error")


def test_resolvekit_error_status_raises_and_defaults_to_500_internal_error():
    client = make_client(resolver=_StubErrorResolver())

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "internal_error"


class _StubNotFoundResolver:
    """Stub returning `NotFound` for every input."""

    async def resolve(self, texts: list[str], **_: object) -> list[Resolution]:
        return [NotFound() for _ in texts]


def test_run_resolve_refreshes_the_generation_once():
    calls: list[None] = []

    class _SpyGeneration:
        async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
            calls.append(None)

    executor = ThreadPoolExecutor(max_workers=1)
    generation = _SpyGeneration()
    deps = GraphDeps(
        graph=FakeGraphNodeReader(),
        resolver=_StubNotFoundResolver(),
        dcp_executor=executor,
        generation=generation,
        provenance_docs=new_provenance_doc_cache(executor=executor, generation=generation),
    )

    response = asyncio.run(
        resolve_door._run_resolve(deps, ResolveRequest(texts=["Kenya"]), policy=None)
    )

    assert calls == [None]
    assert response.results[0].status == "not_found"
