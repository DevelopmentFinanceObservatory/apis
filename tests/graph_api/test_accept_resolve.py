"""Acceptance cases for `POST /api/graph/resolve`: the `kinds` filter and its default, the
`code_system` narrowing, `matched.as`, `reading`, and the four statuses with groups and
regions resolving like any entity.

The rows come from `accept-graph.md`. A case for behaviour dev lacks is strict-xfail with the
slice that builds it; a case dev already answers passes now."""

from __future__ import annotations

from typing import Any

import pytest

import entity_resolution.resolver as resolver_module
from entity_resolution.membership import DEFAULT_PROJECTION
from graph_api.doors.node import NODE_MEMBERSHIP_SLOT

from .conftest import FakeGraphNodeReader, make_client, make_resolver

_DEFAULT_KINDS = [
    "country",
    "continent",
    "subregion",
    "continental_union",
    "place_other",
    "aggregate",
    "organisation",
    "group",
]
_STATUSES = {"resolved", "ambiguous", "not_found", "not_held"}
_LAST_RUNG_ARC = "<-description->dcid"

_IBRD = "InternationalBankForReconstructionAndDevelopment"
_IDA = "InternationalDevelopmentAssociation"
_LAGOS_STATE, _LAGOS_CITY = "wikidataId/Q815913", "wikidataId/Q8673"
_COMMONWEALTH_TOWN = "geoId/5516550"
_AMERICAS = "undata-geo/G00134000"


def _node(graph_type: str, *names: str) -> dict[str, tuple[str, ...]]:
    return {"typeOf": (graph_type,), "name": names or ("graph-name-unused",)}


# One graph holding every node the cases below need, with the last-rung answers dev recorded
# for Lagos and Commonwealth (live-probes.md section A).
_NODES = {
    "country/KEN": _node("Country"),
    "country/MMR": _node("Country"),
    "country/COD": _node("Country"),
    "country/COG": _node("Country"),
    "country/FSM": _node("Country"),
    "country/LAO": _node("Country"),
    "CRSPurposeCode12220": _node("CRSPurposeCodeEnum"),
    "Micronesia": _node("UNGeoRegion"),
    "MiddleAfrica": _node("UNGeoRegion"),
    "SubSaharanAfrica": _node("UNGeoRegion"),
    "EuropeanUnion": _node("ContinentalUnion"),
    "africa": _node("Continent"),
    _AMERICAS: _node("GeoRegion"),
    _IBRD: _node("Organization"),
    _IDA: _node("Organization"),
    _LAGOS_STATE: _node("AdministrativeArea1", "Lagos"),
    _LAGOS_CITY: _node("City", "Lagos"),
    _COMMONWEALTH_TOWN: _node("City", "Commonwealth"),
}
_ARCS = {
    (_LAST_RUNG_ARC, "Lagos"): [_LAGOS_STATE, _LAGOS_CITY],
    (_LAST_RUNG_ARC, "Commonwealth"): [_COMMONWEALTH_TOWN],
}


def _post(shared_resolver, body: dict[str, Any], *, nodes=_NODES):
    graph = FakeGraphNodeReader(nodes=nodes, arcs=_ARCS)
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    # An injected resolver leaves the policy to its caller; publish the shared one, as the
    # live lifespan does.
    NODE_MEMBERSHIP_SLOT.set(client.app, shared_resolver.policy)
    with client:
        return client.post("/api/graph/resolve", json=body)


def _results(shared_resolver, body: dict[str, Any], *, nodes=_NODES) -> list[dict[str, Any]]:
    response = _post(shared_resolver, body, nodes=nodes)
    assert response.status_code == 200, response.text
    return response.json()["results"]


def _refs(result: dict[str, Any]) -> set[str]:
    """The ref a resolved result names, or the refs an ambiguous one lists."""
    if result["status"] == "resolved":
        return {result["entity"]["ref"]}
    return {candidate["ref"] for candidate in result.get("candidates", [])}


# --- kinds: rows that must fire ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kinds", "code_system", "status", "refs"),
    [
        pytest.param(
            "Lagos",
            ["country"],
            None,
            "not_found",
            # The gate's rejected hint, Laos, follows the two places the filter removed.
            [
                (_LAGOS_STATE, "place_other"),
                (_LAGOS_CITY, "place_other"),
                ("country/LAO", "country"),
            ],
            id="lagos-towns-dropped",
        ),
        pytest.param(
            "Micronesia",
            ["country"],
            None,
            "resolved",
            {"country/FSM"},
            id="micronesia-one-left-resolves",
        ),
        pytest.param(
            "Americas",
            ["continent"],
            None,
            "resolved",
            {_AMERICAS},
            # The membership authority classes the Americas a continent; dev types its node a
            # `GeoRegion`.
            id="americas-is-the-authoritys-continent",
        ),
        pytest.param(
            "Commonwealth",
            ["country"],
            None,
            "not_found",
            [("groups/Commonwealth", "group")],
            id="commonwealth-group-dropped",
        ),
        pytest.param(
            "basic health care",
            ["code"],
            "CRSPurposeCodeEnum",
            "resolved",
            {"CRSPurposeCode12220"},
            id="code-label",
        ),
        pytest.param(
            "basic health care",
            ["code"],
            "DAC1AidTypeEnum",
            "not_found",
            [],
            id="code-system-narrows",
        ),
    ],
)
def test_a_kinds_filter_admits_only_the_requested_kinds(
    shared_resolver, text, kinds, code_system, status, refs
):
    body: dict[str, Any] = {"texts": [text], "kinds": kinds}
    if code_system is not None:
        body["code_system"] = code_system

    (result,) = _results(shared_resolver, body)

    assert result["status"] == status
    if status == "not_found":
        # P25: the near matches are the candidates the filter removed, with their kinds.
        assert result.get("entity") is None
        assert [(near["ref"], near["kind"]) for near in result["near_matches"]] == refs
    else:
        assert _refs(result) == refs


# --- kinds: rows that must not fire --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kinds", "status", "refs"),
    [
        ("Congo", ["country"], "ambiguous", {"country/COD", "country/COG"}),
        ("Kenya", ["country"], "resolved", {"country/KEN"}),
        ("World Bank", ["organisation"], "ambiguous", {_IBRD, _IDA}),
        ("Lagos", ["place_other"], "ambiguous", {_LAGOS_STATE, _LAGOS_CITY}),
    ],
)
def test_a_kinds_filter_keeps_every_candidate_of_a_requested_kind(
    shared_resolver, text, kinds, status, refs
):
    (result,) = _results(shared_resolver, {"texts": [text], "kinds": kinds})

    assert result["status"] == status
    assert _refs(result) == refs


@pytest.mark.parametrize(
    ("text", "status"),
    [
        # Decision 12: the default keeps subnational places, so Lagos stays ambiguous between
        # its two places.
        ("Lagos", "ambiguous"),
        # `code` is outside the default kinds.
        ("basic health care", "not_found"),
    ],
)
def test_the_default_kinds_keep_places_and_leave_out_codes(shared_resolver, text, status):
    (result,) = _results(shared_resolver, {"texts": [text]})

    assert result["status"] == status


def test_omitting_kinds_equals_naming_the_default_kinds(shared_resolver):
    texts = ["Kenya", "Congo", "Lagos", "World Bank", "basic health care", "zzqqxv"]

    omitted = _results(shared_resolver, {"texts": texts})
    named = _results(shared_resolver, {"texts": texts, "kinds": _DEFAULT_KINDS})

    assert omitted == named


def test_code_system_without_the_code_kind_is_refused_with_a_pointer(shared_resolver):
    response = _post(
        shared_resolver,
        {"texts": ["basic health care"], "kinds": ["country"], "code_system": "CRSPurposeCodeEnum"},
    )

    assert response.status_code == 400
    problem = response.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/code_system"


def test_code_system_with_the_code_kind_is_accepted(shared_resolver):
    response = _post(
        shared_resolver,
        {"texts": ["basic health care"], "kinds": ["code"], "code_system": "CRSPurposeCodeEnum"},
    )

    assert response.status_code == 200, response.text


# --- matched.as ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "body", "matched_as"),
    [
        ("Kenya", {}, "name"),
        ("KE", {}, "code"),
        ("KEN", {}, "code"),
        ("Burma", {}, "alias"),
        ("G20", {}, "alias"),
        ("basic health care", {"kinds": ["code"]}, "name"),
        ("PHC", {"kinds": ["code"]}, "alias"),
    ],
)
def test_matched_says_whether_the_text_is_a_name_an_alias_or_a_code(
    shared_resolver, text, body, matched_as
):
    (result,) = _results(shared_resolver, {"texts": [text], **body})

    assert result["status"] == "resolved"
    assert result["input"] == text
    assert result["matched"] == {"text": text, "as": matched_as}


# --- reading -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reading"),
    [
        ("EU countries", "members"),
        ("EU member states", "members"),
        ("DAC members", "members"),
        ("EU", "entity"),
        ("European Union", "entity"),
        ("G20", "entity"),
        ("Central Africa", "entity"),
        ("Kenya", "entity"),
    ],
)
def test_reading_is_members_only_for_a_text_naming_a_groups_members(shared_resolver, text, reading):
    (result,) = _results(shared_resolver, {"texts": [text]})

    assert result["status"] == "resolved"
    assert result["reading"] == reading


@pytest.mark.parametrize("text", ["G20 countries", "OECD members"])
def test_a_members_phrase_added_as_data_reads_as_members(shared_resolver, text):
    (result,) = _results(shared_resolver, {"texts": [text]})

    assert result["status"] == "resolved"
    assert result["reading"] == "members"


# --- statuses ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "nodes", "status"),
    [
        ("Kenya", _NODES, "resolved"),
        ("Kenya", {}, "not_held"),
        ("Congo", _NODES, "ambiguous"),
        ("zzqqxv", _NODES, "not_found"),
    ],
    ids=["resolved", "not_held", "ambiguous", "not_found"],
)
def test_an_entity_takes_one_of_the_four_statuses(shared_resolver, text, nodes, status):
    (result,) = _results(shared_resolver, {"texts": [text]}, nodes=nodes)

    assert result["status"] == status


def test_groups_and_regions_resolve_with_the_four_statuses_only(shared_resolver):
    texts = ["Kenya", "Congo", "zzqqxv", "G7", "Africa", "EU countries", "Sub-Saharan Africa"]

    results = _results(shared_resolver, {"texts": texts})

    assert [result["input"] for result in results] == texts
    assert {result["status"] for result in results} <= _STATUSES
    by_text = dict(zip(texts, results, strict=True))
    policy = shared_resolver.policy
    for text, ref, kind, relation in [
        ("G7", "GroupOfSeven", "group", "members"),
        ("Africa", "africa", "continent", "countries"),
        ("Sub-Saharan Africa", "SubSaharanAfrica", "subregion", "countries"),
    ]:
        result = by_text[text]
        assert result["status"] == "resolved", result
        assert (result["entity"]["ref"], result["entity"]["kind"]) == (ref, kind)
        assert result["member_count"] == policy.get(ref).member_count(DEFAULT_PROJECTION)
        assert f"relation={relation}" in result["members"]


def test_a_group_resolves_with_a_count_and_a_link_from_the_registry_alone(shared_resolver):
    """The design's result example: the graph holds no member and no group node, and G20 still
    resolves from the membership registry."""
    (result,) = _results(shared_resolver, {"texts": ["G20"]}, nodes={})

    assert result["status"] == "resolved"
    assert result["entity"]["ref"] == "groups/G20"
    assert result["entity"]["kind"] == "group"
    assert result["projection"] == "countries"
    assert result["member_count"] == 19
    assert result["members"] == (
        "/api/graph/node/groups/G20/members?relation=members&projection=countries"
    )


def test_a_group_resolved_as_a_single_node_carries_its_count_and_link(shared_resolver, monkeypatch):
    """A registry group that resolves as a plain node, as a survivor below the top one does
    (`_GateOutcome.group` is `None`), takes its count and link from the policy like a
    `Group` result."""
    monkeypatch.setattr(resolver_module, "_group_hit", lambda _shared, _candidate: None)

    (result,) = _results(shared_resolver, {"texts": ["OPEC"]}, nodes={})

    assert (result["status"], result["entity"]["kind"]) == ("resolved", "group")
    count = shared_resolver.policy.get("groups/OPEC").member_count(DEFAULT_PROJECTION)
    assert result["member_count"] == count
    assert result["members"] == (
        "/api/graph/node/groups/OPEC/members?relation=members&projection=countries"
    )


def test_expand_members_returns_a_groups_members_inline(shared_resolver):
    (result,) = _results(shared_resolver, {"texts": ["G77"], "expand": ["members"]}, nodes={})

    # Palestine falls under `other_states`, so `countries` holds 133 of G77's 134.
    refs = {member["ref"] for member in result["members"]}
    assert result["member_count"] == 133
    assert len(refs) == 133
    assert "country/PSE" not in refs
    assert all(ref.startswith("country/") for ref in refs)


# --- one kind and one hold rule on every route ---------------------------------------------

_AFDB_GROUP = "AfricanDevelopmentBankGroup"


def _routes(shared_resolver, nodes, arcs):
    graph = FakeGraphNodeReader(nodes=nodes, arcs=arcs)
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    NODE_MEMBERSHIP_SLOT.set(client.app, shared_resolver.policy)
    return client


@pytest.mark.parametrize(
    ("text", "ref", "graph_type"),
    [
        # A total party of DAC2A typed `Organization`, in the resolver's pack.
        ("African Development Bank Group", _AFDB_GROUP, "Organization"),
        # A `DACGroup` node, which only the last rung reaches.
        ("Qqqdacafrica", "DACAfrica", "DACGroup"),
    ],
)
def test_resolve_and_the_node_route_give_a_party_one_kind(shared_resolver, text, ref, graph_type):
    """dfo-apis#1281: resolve classed a node by its graph types alone, so a total party was an
    `organisation` or `other` there and an `aggregate` on the node route."""
    nodes = {ref: _node(graph_type, text)}
    arcs = {(_LAST_RUNG_ARC, text): [ref]}
    with _routes(shared_resolver, nodes, arcs) as client:
        node = client.get(f"/api/graph/node/{ref}")
        resolved = client.post("/api/graph/resolve", json={"texts": [text]})
        filtered = client.post("/api/graph/resolve", json={"texts": [text], "kinds": ["aggregate"]})

    assert node.status_code == 200, node.text
    assert node.json()["kind"] == "aggregate"
    for response in (resolved, filtered):
        (result,) = response.json()["results"]
        assert (result["status"], result["entity"]["ref"]) == ("resolved", ref), result
        assert result["entity"]["kind"] == "aggregate"


def test_a_region_the_policy_holds_without_a_node_is_held_on_every_route(shared_resolver):
    """dfo-apis#1281: the members route held a policy region with no graph node, while the node
    route answered 404 and resolve `not_held`."""
    with _routes(shared_resolver, {}, {}) as client:
        node = client.get("/api/graph/node/WesternAfrica")
        countries = client.get("/api/graph/node/WesternAfrica/members?relation=countries")
        (resolved,) = client.post("/api/graph/resolve", json={"texts": ["Western Africa"]}).json()[
            "results"
        ]

    assert node.status_code == 200, node.text
    assert node.json()["kind"] == "subregion"
    assert countries.status_code == 200, countries.text
    assert resolved["status"] == "resolved", resolved
    assert (resolved["entity"]["ref"], resolved["entity"]["kind"]) == ("WesternAfrica", "subregion")
