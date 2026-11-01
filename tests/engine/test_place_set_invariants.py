"""The rules a place set keeps at every door, checked on a small world of sets.

A place set is a continent, a subregion or a named group. It has one graph node, where the
graph holds one, and its member places. Each test asks the parsed door, the
`dataset_resubmit` door or the Datasets mint about the world in `_place_set_world`, and
compares the answer with what the world's tables imply.

1. A Spec is read at the dcids its identity binds. Its coverage and its observation series
   are the observations held at those dcids, its observation fetches name no other dcid,
   and its resubmit and its mint read the same list and state the same observations.
2. A posted identity is read at its own dcids. The resubmit door and the mint keep a
   region's dcid as its node.
3. A set binds its node only on a role whose range admits the node's type, and only when
   the series holds an observation there in the years asked for. Otherwise it binds its
   members. A rank over the axis a set fills binds its members, and so does a request
   that asks for the set's members.
4. A role carries an entity group only when the dcids it binds are that set's members.
5. A definition Search serves, posted back with its entity groups, answers with the same
   `spec_id` and keeps each group that has a dcid.
6. No door emits or accepts a Spec that binds more dcids across its roles than the bound.
   A set too large to bind answers `entity_set_too_large` with no member dropped. A role
   left open binds no dcid and is outside the bound.
7. A definition with no observation at any dcid it binds is `no_data` on Search and a 200
   manifest with `empty_result` on the mint, unless the UN front door refused a place or
   the origin registry withholds data there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest

from entity_resolution import acquire_shared, release_shared
from search.engine.errors import EngineInputError, GraphInfraError
from search.models import EntityGroup, Operation, ParsedEntity, ResolveResponse, Spec
from tests.engine import _place_set_world as world
from tests.engine._place_set_world import (
    AFRICA,
    AFRICA_ALIAS,
    AFRICAN_UNION,
    AID,
    ASIA,
    COUNTRY_YEAR,
    DEBT,
    DEU,
    EU,
    EXPANSION,
    FLOW_DATA,
    FRA,
    GHA,
    ITA,
    KEN,
    LABEL,
    NGA,
    NODE_TYPE,
    NODE_YEAR,
    PROVIDER,
    RECIPIENT,
    ROLE_RANGE,
    SHN,
    SUB_SAHARAN,
    SUB_SAHARAN_ALIAS,
    UN_SERIES,
    WDI,
    WHO,
    Doors,
    Unavailable,
    bound,
    check_first_answer,
    check_groups,
    check_minted,
    check_posted,
    check_reading,
    check_three_doors,
    departure_asks,
    destination,
    flow_reading,
    group,
    held_observations,
    members,
    members_reading,
    no_data_reason,
    no_data_terms,
    origin,
    pinned,
    places_with_data,
    subject_reading,
    the_readings,
    the_spec,
    the_specs,
    unavailable,
    warning_codes,
)

_SUBJECT = "observationAbout"


@pytest.fixture
def doors(monkeypatch: pytest.MonkeyPatch) -> Doors:
    return Doors(monkeypatch)


def _named(*names: str) -> tuple[str, ...]:
    return tuple(LABEL[dcid] for dcid in names)


@pytest.mark.parametrize(
    ("ref", "held"), [(EU, EXPANSION[EU]), ("groups/AU", AFRICAN_UNION)], ids=["eu", "au"]
)
def test_the_worlds_groups_hold_the_registrys_members(ref: str, held: tuple[str, ...]) -> None:
    """The world's groups hold the members the resolver's group registry gives, so each
    invariant checks the groups Search serves."""
    shared = acquire_shared()
    try:
        assert shared.groups.get(ref).members() == held
    finally:
        release_shared(shared)


# Invariant 1.

# Each request as (series, entities, year). The first block names each place itself or
# names a group. The rest name a region.
_REQUESTS: dict[str, tuple[str, Sequence[str | ParsedEntity], int | None]] = {
    "one-country": (WDI, ["Kenya"], None),
    "one-country-in-a-year": (WDI, ["Kenya"], COUNTRY_YEAR),
    "countries-named-one-by-one": (WDI, _named(*EXPANSION[AFRICA]), None),
    "group-with-a-node": (WDI, ["European Union"], None),
    "group-with-a-node-in-a-year-its-node-lacks": (WHO, ["European Union"], COUNTRY_YEAR),
    "group-with-no-node": (WDI, ["African Union"], None),
    "group-and-one-of-its-members": (WDI, ["European Union", "France"], None),
    "flow-between-two-countries": (AID, [origin("France"), destination("Kenya")], None),
    "flow-to-countries-named-one-by-one": (
        AID,
        [origin("France"), *(destination(name) for name in _named(GHA, KEN, NGA))],
        None,
    ),
    "flow-to-one-country": (DEBT, ["Kenya"], None),
    "region-series-at-members": (WDI, ["Africa"], None),
    "subregion-series-at-members": (WDI, ["Sub-Saharan Africa"], None),
    "region-year-only-members-hold": (WHO, ["Africa"], COUNTRY_YEAR),
    "region-with-no-node-data": (WHO, ["Asia"], None),
    "region-and-one-of-its-members": (WDI, ["Africa", "Kenya"], None),
    "region-and-an-outside-country": (WDI, ["Africa", "France"], None),
    "region-series-at-node": (WHO, ["Africa"], None),
    "region-year-only-node-holds": (WHO, ["Africa"], NODE_YEAR),
    "region-node-and-a-country": (WHO, ["Africa", "Kenya"], None),
    "two-regions-settling-apart": (WHO, ["Africa", "Asia"], None),
    "group-series-at-node": (WHO, ["European Union"], None),
    "group-year-only-node-holds": (WHO, ["European Union"], NODE_YEAR),
    "flow-to-a-region": (DEBT, ["Africa"], None),
    "flow-from-a-country-to-a-region": (AID, [origin("France"), destination("Africa")], None),
    "flow-from-a-group-to-a-region": (
        AID,
        [origin("European Union"), destination("Africa")],
        None,
    ),
    "region-asked-for-its-members": (WHO, [members("Africa")], None),
    "group-asked-for-its-members": (WHO, [members("European Union")], None),
    "two-regions-asked-for-their-members": (WHO, [members("Africa"), members("Asia")], None),
}


@pytest.mark.parametrize("name", list(_REQUESTS))
def test_every_door_reads_a_spec_at_the_dcids_it_binds(doors: Doors, name: str) -> None:
    """The first answer's coverage and series are the observations the world holds at the
    dcids its identity binds, and every observation fetch names one of them. Its resubmit
    states the same coverage and series from reads of those dcids alone, and its mint
    writes as many rows over as many places."""
    stat_var, entities, year = _REQUESTS[name]

    spec = the_spec(doors.first_answer(stat_var, *entities, year=year))

    check_three_doors(doors, stat_var, spec, year)


# Two readings of one request. `WHO` holds data at a set's node and `WDI` at its members.
_TWO_READINGS = [
    pytest.param(order, names, id=f"{elected}-elected-{label}")
    for elected, order in (("node", (WHO, WDI)), ("members", (WDI, WHO)))
    for label, names in (
        ("region", [AFRICA]),
        ("region-and-one-of-its-members", [AFRICA, KEN]),
        ("group", [EU]),
        ("two-regions", [AFRICA, ASIA]),
    )
]


@pytest.mark.parametrize(("stat_vars", "named"), _TWO_READINGS)
def test_each_offered_reading_is_read_at_the_dcids_it_binds(
    doors: Doors, stat_vars: tuple[str, str], named: list[str]
) -> None:
    """A request two series answer is offered both readings, whichever is elected. Each
    settles its sets on its own read, so one binds a set's node and the other its members,
    and each states the observations held at its own dcids from reads of those alone."""
    specs = the_readings(doors.readings(stat_vars, *_named(*named)))

    assert set(specs) == set(stat_vars)
    for stat_var, spec in specs.items():
        sets = [node for node in named if node in EXPANSION]
        places = [dcid for dcid in named if dcid not in EXPANSION]
        assert bound(spec) == {
            _SUBJECT: sorted(
                {*places, *(d for node in sets for d in subject_reading(stat_var, node))}
            )
        }
        check_reading(doors, stat_var, spec)
    assert bound(specs[WHO]) != bound(specs[WDI])


# A subject series asked with a direction on each place. The shape reads both as subjects.
_DIRECTED = [
    pytest.param(stat_var, node, place, set_first, id=f"{held}-{kind}-{order}")
    for held, stat_var in (("node", WHO), ("members", WDI))
    for kind, node, place in (("region", AFRICA, FRA), ("group", EU, KEN))
    for order, set_first in (("set-to-place", True), ("place-to-set", False))
]


@pytest.mark.parametrize(("stat_var", "node", "place", "set_first"), _DIRECTED)
def test_a_directed_place_beside_a_directed_set_binds_both_as_subjects(
    doors: Doors, stat_var: str, node: str, place: str, set_first: bool
) -> None:
    """A set named from or to a plain place, on a series with no flow role. The set's node
    is read alone on the subject role to settle it, and the Spec then binds the plain place
    beside the node, or beside the members where the node holds nothing."""
    first, second = (node, place) if set_first else (place, node)

    spec = the_spec(doors.first_answer(stat_var, origin(LABEL[first]), destination(LABEL[second])))

    assert place in bound(spec)[_SUBJECT]
    assert bound(spec) == {_SUBJECT: sorted({place, *subject_reading(stat_var, node)})}
    check_three_doors(doors, stat_var, spec)


# Invariants 2 and 7.

_POSTED = [
    pytest.param(WHO, AFRICA, None, id="region-node-held"),
    pytest.param(WDI, AFRICA, None, id="region-node-empty"),
    pytest.param(WHO, AFRICA, COUNTRY_YEAR, id="region-year-empty"),
    pytest.param(WDI, SUB_SAHARAN, None, id="subregion-node-empty"),
    pytest.param(WHO, EU, None, id="group-node-held"),
    pytest.param(WDI, KEN, None, id="country"),
]


@pytest.mark.parametrize(("stat_var", "dcid", "year"), _POSTED)
def test_a_resubmitted_identity_is_read_at_its_own_dcids(
    doors: Doors, stat_var: str, dcid: str, year: int | None
) -> None:
    """An identity binding one dcid answers with the observations the world holds at that
    dcid, from reads of that dcid alone. Where the world holds none there, the answer is
    `no_data` with `no_observations`, whatever the dcid's members hold."""
    identity = doors.identity(stat_var, {_SUBJECT: [dcid]}, year=year)

    if held_observations(stat_var, {_SUBJECT: [dcid]}, year):
        check_posted(doors, stat_var, identity, year=year)
    else:
        assert no_data_reason(doors.resubmit(stat_var, identity)) == "no_observations"
        assert all(read.dcids <= {dcid} for read in doors.reads("resubmit"))


@pytest.mark.parametrize(
    ("stat_var", "dcid", "year"),
    [
        pytest.param(WHO, AFRICA, None, id="region-node-held"),
        pytest.param(WDI, AFRICA, None, id="region-node-empty"),
        pytest.param(WHO, AFRICA, COUNTRY_YEAR, id="region-year-empty"),
        pytest.param(WHO, EU, None, id="group-node-held"),
        pytest.param(WDI, KEN, None, id="country"),
    ],
)
def test_a_minted_identity_is_read_at_its_own_dcids(
    doors: Doors, stat_var: str, dcid: str, year: int | None
) -> None:
    """A mint of an identity binding one dcid writes the observations the world holds at
    that dcid, from reads of that dcid alone. Where the world holds none there, whether in
    the years asked for or in any year, it is a 200 manifest with no rows and
    `empty_result`."""
    identity = doors.identity(stat_var, {_SUBJECT: [dcid]}, year=year)

    manifest = check_minted(doors, stat_var, identity, year)

    assert ("empty_result" in warning_codes(manifest)) == (
        held_observations(stat_var, {_SUBJECT: [dcid]}, year) == 0
    )


def test_a_place_the_un_front_door_refused_is_not_an_empty_result(doors: Doors) -> None:
    """A UN series with no observation at the place it binds refuses `upstream_error`,
    retryable, when the UN front door refused that place, since the empty read says
    nothing about it."""
    identity = doors.identity(UN_SERIES, {_SUBJECT: [KEN]})
    doors.mint_graph.un_refused = frozenset({KEN})

    refused = doors.mint(identity)

    assert refused.status_code == 502, refused.text
    assert (refused.json()["code"], refused.json()["retryable"]) == ("upstream_error", True)


def test_a_un_series_with_no_observation_and_no_refusal_is_an_empty_result(doors: Doors) -> None:
    """The same UN series at the same place, with the front door open, is the empty
    result."""
    identity = doors.identity(UN_SERIES, {_SUBJECT: [KEN]})

    manifest = check_minted(doors, UN_SERIES, identity)

    assert "empty_result" in warning_codes(manifest)


def test_data_the_registry_withholds_is_not_an_empty_result(doors: Doors) -> None:
    """A definition with no admitted observation at its place refuses
    `excluded_by_registry` when a provenance the origin registry withholds holds data
    there."""
    identity = doors.identity(WDI, {_SUBJECT: [AFRICA]})
    doors.mint_graph.withheld = ("dc/base/PlaceSetWithheld",)

    refused = doors.mint(identity)

    assert refused.status_code == 422, refused.text
    assert refused.json()["code"] == "excluded_by_registry"


# Invariant 3.

_SETTLED = [
    pytest.param(WDI, AFRICA, None, id="region-wdi"),
    pytest.param(WHO, AFRICA, None, id="region-who"),
    pytest.param(WHO, AFRICA, NODE_YEAR, id="region-who-node-year"),
    pytest.param(WHO, AFRICA, COUNTRY_YEAR, id="region-who-country-year"),
    pytest.param(WDI, SUB_SAHARAN, None, id="subregion-wdi"),
    pytest.param(WHO, SUB_SAHARAN, None, id="subregion-who"),
    pytest.param(WHO, ASIA, None, id="second-region-who"),
    pytest.param(WDI, EU, None, id="group-wdi"),
    pytest.param(WHO, EU, None, id="group-who"),
    pytest.param(WHO, EU, NODE_YEAR, id="group-who-node-year"),
    pytest.param(WHO, EU, COUNTRY_YEAR, id="group-who-country-year"),
]


@pytest.mark.parametrize(("stat_var", "node", "year"), _SETTLED)
def test_a_set_binds_its_node_only_where_the_series_holds_an_observation_there(
    doors: Doors, stat_var: str, node: str, year: int | None
) -> None:
    """As the subject of a series, a set binds its node where the series holds an
    observation at the node in the years asked for, and its members otherwise."""
    spec = the_spec(doors.first_answer(stat_var, LABEL[node], year=year))

    assert bound(spec) == {_SUBJECT: subject_reading(stat_var, node, year)}


def _debt_to_africa(doors: Doors) -> tuple[ResolveResponse, Mapping[str, list[str]]]:
    return doors.first_answer(DEBT, "Africa"), {
        "destination": flow_reading(DEBT, AFRICA, RECIPIENT)
    }


def _aid_from_france_to_africa(doors: Doors) -> tuple[ResolveResponse, Mapping[str, list[str]]]:
    return doors.first_answer(AID, origin("France"), destination("Africa")), {
        "origin": [FRA],
        "destination": flow_reading(AID, AFRICA, RECIPIENT, [FRA]),
    }


def _aid_from_the_eu_to_kenya(doors: Doors) -> tuple[ResolveResponse, Mapping[str, list[str]]]:
    return doors.first_answer(AID, origin("European Union"), destination("Kenya")), {
        "origin": flow_reading(AID, EU, PROVIDER, [KEN]),
        "destination": [KEN],
    }


def _aid_from_the_eu_to_nigeria(doors: Doors) -> tuple[ResolveResponse, Mapping[str, list[str]]]:
    return doors.first_answer(AID, origin("European Union"), destination("Nigeria")), {
        "origin": flow_reading(AID, EU, PROVIDER, [NGA]),
        "destination": [NGA],
    }


@pytest.mark.parametrize(
    "ask",
    [
        _debt_to_africa,
        _aid_from_france_to_africa,
        _aid_from_the_eu_to_kenya,
        _aid_from_the_eu_to_nigeria,
    ],
)
def test_a_set_on_a_flow_role_binds_its_node_only_where_the_role_admits_and_holds_it(
    doors: Doors, ask: Callable[[Doors], tuple[ResolveResponse, Mapping[str, list[str]]]]
) -> None:
    """On a flow role a set binds its node when the role's range admits the node's type
    and the series holds a row from or to the node across the other role's dcids. A
    continent on a role that admits only countries binds its members. The European Union
    binds its node as a provider to Kenya, where the node holds rows, and its members as a
    provider to Nigeria, where it holds none."""
    response, expected = ask(doors)

    spec = the_spec(response)

    assert bound(spec) == expected
    assert AFRICA not in bound(spec).get("destination", [])


def test_a_set_on_two_roles_takes_one_reading(doors: Doors) -> None:
    """Both flow roles admit a subregion's type, and the series holds a row from the
    `SubSaharanAfrica` node to one of its members, one from a member to the node, and none
    from the node to itself. Aid from Sub-Saharan Africa to Sub-Saharan Africa binds the
    members on both roles, since the set is read with its node on both or its members on
    both."""
    assert NODE_TYPE[SUB_SAHARAN] in ROLE_RANGE[PROVIDER]
    assert NODE_TYPE[SUB_SAHARAN] in ROLE_RANGE[RECIPIENT]
    assert flow_reading(AID, SUB_SAHARAN, PROVIDER, EXPANSION[SUB_SAHARAN]) == [SUB_SAHARAN]
    assert flow_reading(AID, SUB_SAHARAN, RECIPIENT, EXPANSION[SUB_SAHARAN]) == [SUB_SAHARAN]
    assert (SUB_SAHARAN, SUB_SAHARAN) not in FLOW_DATA[AID]

    spec = the_spec(
        doors.first_answer(AID, origin(LABEL[SUB_SAHARAN]), destination(LABEL[SUB_SAHARAN]))
    )

    members = sorted(EXPANSION[SUB_SAHARAN])
    assert bound(spec) == {"origin": members, "destination": members}
    check_three_doors(doors, AID, spec)


def test_a_subregion_named_as_the_provider_binds_its_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-Saharan Africa named as the provider of aid to Kenya, where the world's provider
    range admits a subregion and the series holds a row from the subregion's node to Kenya.
    No release lists a subregion as a provider (`role_kinds.role_takes`), and a resubmit
    refuses one there, so the first answer binds the members, whose row from Ghana holds
    data, and every door reads that binding."""
    rows = (*world.FLOW_DATA[AID], (SUB_SAHARAN, KEN))
    monkeypatch.setattr(world, "FLOW_DATA", {**world.FLOW_DATA, AID: rows})
    doors = Doors(monkeypatch)
    assert NODE_TYPE[SUB_SAHARAN] in ROLE_RANGE[PROVIDER]
    assert flow_reading(AID, SUB_SAHARAN, PROVIDER, [KEN]) == [SUB_SAHARAN]

    spec = the_spec(doors.first_answer(AID, origin(LABEL[SUB_SAHARAN]), destination("Kenya")))

    assert bound(spec) == {"origin": sorted(EXPANSION[SUB_SAHARAN]), "destination": [KEN]}
    check_three_doors(doors, AID, spec)


def test_a_set_named_by_two_texts_on_two_roles_takes_one_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-Saharan Africa as the provider under one text and the recipient under another
    is one set. With no row from a member to the node, the node holds rows as the provider
    and none as the recipient, so the set read with its node on both roles holds nothing
    and binds its members on both, as it does named by one text."""
    rows = tuple(row for row in world.FLOW_DATA[AID] if row != (KEN, SUB_SAHARAN))
    monkeypatch.setattr(world, "FLOW_DATA", {**world.FLOW_DATA, AID: rows})
    doors = Doors(monkeypatch)
    members = EXPANSION[SUB_SAHARAN]
    assert flow_reading(AID, SUB_SAHARAN, PROVIDER, members) == [SUB_SAHARAN]
    assert flow_reading(AID, SUB_SAHARAN, RECIPIENT, members) == sorted(members)

    once = the_spec(
        doors.first_answer(AID, origin(LABEL[SUB_SAHARAN]), destination(LABEL[SUB_SAHARAN]))
    )
    twice = the_spec(
        doors.first_answer(AID, origin(LABEL[SUB_SAHARAN]), destination(SUB_SAHARAN_ALIAS))
    )

    assert (
        bound(twice) == bound(once) == {"origin": sorted(members), "destination": sorted(members)}
    )
    assert twice.spec_id == once.spec_id
    assert twice.resolved.entity_groups == once.resolved.entity_groups
    check_three_doors(doors, AID, twice)


@pytest.mark.parametrize("members_end", ["origin", "destination"])
@pytest.mark.parametrize(
    "whole_text", [LABEL[SUB_SAHARAN], SUB_SAHARAN_ALIAS], ids=["one-text", "two-texts"]
)
def test_a_set_asked_for_its_members_on_one_end_reads_whole_where_the_other_role_takes_it(
    doors: Doors, members_end: str, whole_text: str
) -> None:
    """Sub-Saharan Africa asked for as its members on one end and named whole on the other,
    under one text or two, binds its members on the first end. Named whole as the recipient
    it binds its node, which holds rows with a member. Named whole as the provider it binds
    its members too, though its node holds rows there, since the provider role takes no
    subregion (`role_kinds.role_takes`). The restatement asks for the members on the first
    end alone, and posted back answers with the same Spec."""
    members = sorted(EXPANSION[SUB_SAHARAN])
    assert flow_reading(AID, SUB_SAHARAN, PROVIDER, members) == [SUB_SAHARAN]
    assert flow_reading(AID, SUB_SAHARAN, RECIPIENT, members) == [SUB_SAHARAN]
    whole_end = "destination" if members_end == "origin" else "origin"
    named = {"origin": origin, "destination": destination}
    words = {"origin": "from", "destination": "to"}

    response = doors.first_answer(
        AID,
        named[members_end](LABEL[SUB_SAHARAN], members=True),
        named[whole_end](whole_text),
    )

    spec = the_spec(response)
    whole = [SUB_SAHARAN] if whole_end == "destination" else members
    assert bound(spec) == {members_end: members, whole_end: whole}
    (part,) = response.parts
    assert part.restatement is not None
    assert [
        entity.model_dump(mode="json", exclude_none=True)
        for entity in part.restatement.entities or []
    ] == [
        {"text": LABEL[SUB_SAHARAN], "role_text": words[members_end], "members": True},
        {"text": whole_text, "role_text": words[whole_end]},
    ]
    assert the_spec(doors.post(part.restatement)).spec_id == spec.spec_id
    check_three_doors(doors, AID, spec)


@pytest.mark.parametrize("directed_first", [True, False], ids=["directed-first", "directed-last"])
@pytest.mark.parametrize("stat_var", [AID, DEBT])
@pytest.mark.parametrize(
    ("name", "alias"),
    [(LABEL[SUB_SAHARAN], SUB_SAHARAN_ALIAS), ("African Union", "african union")],
    ids=["subregion", "group-with-no-node"],
)
def test_a_set_one_of_whose_texts_is_directed_fills_that_axis(
    doors: Doors, stat_var: str, directed_first: bool, name: str, alias: str
) -> None:
    """A set named as the recipient by one text and with no direction by another is one
    directed set. The answer is the one the directed text alone gets. A group with no node
    is one set by its members, and its group takes the first text posted for it, as it has
    no graph label."""
    others = [origin("France")] if stat_var == AID else []
    texts = [destination(name), alias]
    if not directed_first:
        texts.reverse()

    once = the_spec(doors.first_answer(stat_var, *others, destination(name)))
    twice = the_spec(doors.first_answer(stat_var, *others, *texts))

    assert twice.spec_id == once.spec_id
    first = name if directed_first else alias
    assert twice.resolved.entity_groups == (
        {
            role: served if served.ref is not None else served.model_copy(update={"label": first})
            for role, served in once.resolved.entity_groups.items()
        }
        if once.resolved.entity_groups
        else None
    )
    check_three_doors(doors, stat_var, twice)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            ((EU, SUB_SAHARAN),),
            {"origin": [EU], "destination": [SUB_SAHARAN]},
            id="data-only-between-the-nodes",
        ),
        pytest.param(
            ((EU, SUB_SAHARAN), (EU, KEN)),
            {"origin": [EU], "destination": [SUB_SAHARAN]},
            id="the-node-pairing-wins-over-a-node-with-members",
        ),
    ],
)
def test_two_sets_on_two_roles_read_their_nodes_together_first(
    monkeypatch: pytest.MonkeyPatch, rows: tuple[tuple[str, str], ...], expected: dict
) -> None:
    """Aid from the European Union to Sub-Saharan Africa, each role admitting its set's
    node. The two nodes are read together first, and bind together where the series
    holds a row between them, whatever either node holds with the other set's members."""
    assert NODE_TYPE[EU] in ROLE_RANGE[PROVIDER]
    assert NODE_TYPE[SUB_SAHARAN] in ROLE_RANGE[RECIPIENT]
    monkeypatch.setattr(world, "FLOW_DATA", {**world.FLOW_DATA, AID: rows})
    doors = Doors(monkeypatch)

    spec = the_spec(doors.first_answer(AID, origin(LABEL[EU]), destination(LABEL[SUB_SAHARAN])))

    assert bound(spec) == expected
    check_three_doors(doors, AID, spec)


def _provider_admits(doors: Doors, *types: str) -> None:
    """Make the provider role's range `types` alone."""
    doors.graph._nodes[PROVIDER] = {
        **doors.graph._nodes[PROVIDER],
        "arcs": {"rangeIncludes": {"nodes": [{"dcid": t} for t in types]}},
    }


_OVER_ORIGIN = Operation(verb="rank", phrase="gave the most", over="origin")


@pytest.mark.parametrize(
    ("admitted", "names", "operations", "answer"),
    [
        pytest.param(
            (NODE_TYPE[EU],),
            [origin(LABEL[EU]), destination("Kenya")],
            [_OVER_ORIGIN],
            ("no_data", "entity_role_not_admitted", [LABEL[EU]]),
            id="members-asked-on-a-role-that-admits-only-the-node",
        ),
        pytest.param(
            (NODE_TYPE[EU],),
            [origin(LABEL[EU]), destination("Kenya")],
            [],
            ("no_data", "no_observations", []),
            id="node-without-data-on-a-role-that-admits-only-the-node",
        ),
        pytest.param(
            ("Country",),
            [origin(LABEL[EU]), destination("Kenya")],
            [],
            ("definite", {"origin": sorted(EXPANSION[EU]), "destination": [KEN]}),
            id="a-role-that-admits-only-the-members",
        ),
    ],
)
def test_a_set_takes_only_a_reading_every_role_it_fills_admits(
    monkeypatch: pytest.MonkeyPatch,
    admitted: tuple[str, ...],
    names: list[ParsedEntity],
    operations: list[Operation],
    answer: tuple,
) -> None:
    """The European Union named as the provider, its node holding no row to Kenya. On a
    provider role that admits the union's type and not a country's, the union reads only
    its node: asked for its members by a rank it is refused, and otherwise its node binds
    and holds nothing. On a role that admits countries and not the union, it reads its
    members."""
    rows = tuple(row for row in world.FLOW_DATA[AID] if row != (EU, KEN))
    monkeypatch.setattr(world, "FLOW_DATA", {**world.FLOW_DATA, AID: rows})
    doors = Doors(monkeypatch)
    _provider_admits(doors, *admitted)

    response = doors.first_answer(AID, *names, operations=operations)

    if answer[0] == "definite":
        assert bound(the_spec(response)) == answer[1]
    else:
        reason, terms = answer[1], answer[2]
        assert no_data_terms(response) == (reason, terms)


@pytest.mark.parametrize("order", [(AID, WDI), (WDI, AID)], ids=["aid-first", "wdi-first"])
@pytest.mark.parametrize("ask", ["rank", "flag"])
def test_a_reading_that_cannot_serve_a_members_ask_is_dropped_before_election(
    doors: Doors, order: tuple[str, str], ask: str
) -> None:
    """The European Union asked for its members, by a rank over the providers or by the
    members flag, where the provider role admits the union's type and not a country's.
    Aid cannot serve the ask, so it is dropped before election whichever reading ranks
    first, and health spending answers for the union's members."""
    _provider_admits(doors, NODE_TYPE[EU])
    union = origin(LABEL[EU], members=ask == "flag")

    response = doors.readings(order, union, operations=[_OVER_ORIGIN] if ask == "rank" else [])

    spec = the_spec(response)
    assert (
        spec.definition.identity.shape.population_type
        == the_spec(doors.first_answer(WDI, LABEL[EU])).definition.identity.shape.population_type
    )
    assert bound(spec) == {_SUBJECT: sorted(EXPANSION[EU])}


def test_two_sets_on_two_roles_bind_their_members_when_their_nodes_hold_no_row_together(
    doors: Doors,
) -> None:
    """The European Union's node holds rows to Sub-Saharan Africa's members, and that
    subregion's node holds rows from a member of the European Union, so each node holds
    data beside the other set's members. The series holds no row from one node to the
    other, so both sets bind their members."""
    assert flow_reading(AID, EU, PROVIDER, EXPANSION[SUB_SAHARAN]) == [EU]
    assert flow_reading(AID, SUB_SAHARAN, RECIPIENT, EXPANSION[EU]) == [SUB_SAHARAN]
    assert held_observations(AID, {"origin": [EU], "destination": [SUB_SAHARAN]}) == 0

    spec = the_spec(doors.first_answer(AID, origin(LABEL[EU]), destination(LABEL[SUB_SAHARAN])))

    assert bound(spec) == {
        "origin": sorted(EXPANSION[EU]),
        "destination": sorted(EXPANSION[SUB_SAHARAN]),
    }
    check_three_doors(doors, AID, spec)


def test_a_lost_read_of_a_sets_node_fails_the_answer(doors: Doors) -> None:
    """Africa's node holds the series and so do its members. When the read that compares
    them is lost, the answer fails as a graph read does. It does not read the node as empty
    and bind the members."""
    assert subject_reading(WHO, AFRICA) == [AFRICA]
    assert places_with_data(WHO, EXPANSION[AFRICA])
    doors.graph.lose_node_read = True

    with pytest.raises(GraphInfraError):
        doors.first_answer(WHO, "Africa")

    assert not doors.graph.lose_node_read
    assert bound(the_spec(doors.first_answer(WHO, "Africa"))) == {_SUBJECT: [AFRICA]}


_RANK = Operation(verb="rank", phrase="highest", over="place")
_TOTAL = Operation(verb="aggregate", phrase="total", over="place")


@pytest.mark.parametrize(
    ("operations", "members"),
    [
        pytest.param([_RANK], True),
        pytest.param(
            [Operation(verb="rank", phrase="received the most", over="destination")], True
        ),
        pytest.param([_RANK, _TOTAL], True),
        pytest.param([Operation(verb="rank", phrase="best year", over="year")], False),
        pytest.param([_TOTAL], False),
    ],
    ids=[
        "rank-over-place",
        "rank-over-destination",
        "rank-and-aggregate",
        "rank-over-year",
        "aggregate",
    ],
)
def test_a_rank_over_the_axis_a_set_fills_binds_its_members(
    doors: Doors, operations: list[Operation], members: bool
) -> None:
    """Africa's node holds the series. A rank over the places binds Africa's members all
    the same, since one place cannot be ranked, and so does a rank over a flow axis on a
    series with no flow roles, where every axis names the subject. A rank over years or an
    aggregate leaves the node bound."""
    spec = the_spec(doors.first_answer(WHO, "Africa", operations=operations))

    assert subject_reading(WHO, AFRICA) == [AFRICA]
    assert bound(spec) == {_SUBJECT: sorted(EXPANSION[AFRICA]) if members else [AFRICA]}


@pytest.mark.parametrize(
    "nodes",
    [[AFRICA], [EU], [AFRICA, ASIA]],
    ids=["region-whose-node-holds", "group-whose-node-holds", "two-regions"],
)
def test_a_set_asked_for_its_members_binds_them(doors: Doors, nodes: list[str]) -> None:
    """A set posted with `members` binds its members where its node holds the series, and
    two sets bind every member of either. The same names posted without the flag bind each
    set's node where the series holds an observation there."""
    names = [LABEL[node] for node in nodes]

    asked = the_spec(doors.first_answer(WHO, *(members(name) for name in names)))
    whole = the_spec(doors.first_answer(WHO, *names))

    assert bound(asked) == {_SUBJECT: members_reading(*nodes)}
    assert bound(whole) == {
        _SUBJECT: sorted(dcid for node in nodes for dcid in subject_reading(WHO, node))
    }
    assert bound(asked) != bound(whole)


@pytest.mark.parametrize(
    ("texts", "node"),
    [
        pytest.param(("EU member states", "European Union"), EU, id="wording-names-members"),
        pytest.param((members(AFRICA_ALIAS), "Africa"), AFRICA, id="one-text-states-the-flag"),
    ],
)
def test_a_set_named_by_two_texts_binds_its_members_when_either_asks(
    doors: Doors, texts: tuple[str | ParsedEntity, str], node: str
) -> None:
    """Two texts of one part that name one node are one set with one reading. One asks for
    the members, by its wording or by the flag, and the other does not, so the Spec binds
    the members though the node holds the series, in either posting order. The definition
    is the one the set's name posted once with `members` gets, served as that set's group."""
    assert subject_reading(WHO, node) == [node]
    alone = the_spec(doors.first_answer(WHO, members(LABEL[node])))

    for named in (texts, texts[::-1]):
        spec = the_spec(doors.first_answer(WHO, *named))

        assert bound(spec) == {_SUBJECT: members_reading(node)}
        assert spec.spec_id == alone.spec_id
        assert group(spec) == {
            "ref": node,
            "label": LABEL[node],
            "entity_type": NODE_TYPE[node],
            "total": None,
            "group_id": None,
        }


@pytest.mark.parametrize(
    ("node", "role", "axis", "across"),
    [
        pytest.param(EU, PROVIDER, "origin", KEN, id="group-as-origin"),
        pytest.param(SUB_SAHARAN, RECIPIENT, "destination", FRA, id="subregion-as-destination"),
    ],
)
def test_a_set_on_a_flow_role_asked_for_its_members_binds_them(
    doors: Doors, node: str, role: str, axis: str, across: str
) -> None:
    """The role admits the node's type and the series holds a row from or to the node, so
    the set posted without the flag binds its node. Posted with `members` it binds its
    members on that role, served as the set's group, and every door reads them."""
    assert flow_reading(AID, node, role, [across]) == [node]
    other_axis = "destination" if axis == "origin" else "origin"

    def ask(*, members: bool) -> Spec:
        named = {
            "origin": origin(LABEL[node], members=members),
            "destination": destination(LABEL[node], members=members),
        }[axis]
        other = {"origin": origin, "destination": destination}[other_axis](LABEL[across])
        return the_spec(doors.first_answer(AID, named, other))

    asked = ask(members=True)

    assert bound(asked) == {axis: sorted(EXPANSION[node]), other_axis: [across]}
    assert group(asked, axis) == {
        "ref": node,
        "label": LABEL[node],
        "entity_type": NODE_TYPE[node],
        "total": None,
        "group_id": None,
    }
    check_three_doors(doors, AID, asked)

    whole = ask(members=False)

    assert bound(whole) == {axis: [node], other_axis: [across]}
    assert group(whole, axis) is None


def test_a_set_named_by_two_texts_on_one_flow_role_takes_the_members_either_asks_for(
    doors: Doors,
) -> None:
    """Aid from "EU member states" and from "European Union" to Kenya names the union
    twice as the provider. The node holds a row to Kenya, and one text asks for the
    members, so the role binds the members whichever text is posted first: one definition,
    the one the union posted once with `members` gets."""
    assert flow_reading(AID, EU, PROVIDER, [KEN]) == [EU]
    alone = the_spec(doors.first_answer(AID, origin(LABEL[EU], members=True), destination("Kenya")))
    texts = (origin("EU member states"), origin(LABEL[EU]))

    for named in (texts, texts[::-1]):
        spec = the_spec(doors.first_answer(AID, *named, destination("Kenya")))

        assert bound(spec) == {"origin": sorted(EXPANSION[EU]), "destination": [KEN]}
        assert spec.spec_id == alone.spec_id


@pytest.mark.parametrize("stat_vars", [(WHO, WDI), (WDI, WHO)], ids=["node-held", "members-held"])
def test_a_set_asked_for_its_members_binds_them_on_every_offered_reading(
    doors: Doors, stat_vars: tuple[str, str]
) -> None:
    """A request that two series answer, asked for Africa's members, binds the members in
    both readings, including the one whose series holds Africa's node, whichever is elected."""
    specs = the_readings(doors.readings(stat_vars, members("Africa")))

    assert set(specs) == set(stat_vars)
    assert subject_reading(WHO, AFRICA) == [AFRICA]
    for stat_var, spec in specs.items():
        assert bound(spec) == {_SUBJECT: members_reading(AFRICA)}
        check_reading(doors, stat_var, spec)


def test_each_part_reads_a_set_as_its_own_entity_asks(doors: Doors) -> None:
    """Two parts of one request name Africa, one with `members`. Each part binds its own
    reading, in either part order."""
    asked, whole = (WHO, [members("Africa")]), (WHO, ["Africa"])

    for parts in ((asked, whole), (whole, asked)):
        specs = the_specs(doors.parts(*parts))

        assert [bound(spec) for spec in specs] == [
            {_SUBJECT: members_reading(AFRICA) if part is asked else [AFRICA]} for part in parts
        ]


def test_a_resubmitted_part_asked_for_a_sets_members_binds_them(doors: Doors) -> None:
    """A parsed entry of the `dataset_resubmit` door reads `members` as the parsed door
    does."""
    asked = the_spec(doors.resubmit_part(WHO, members("Africa")))
    whole = the_spec(doors.resubmit_part(WHO, "Africa"))

    assert bound(asked) == {_SUBJECT: members_reading(AFRICA)}
    assert bound(whole) == {_SUBJECT: [AFRICA]}


@pytest.mark.parametrize(
    ("entity", "places"),
    [
        pytest.param(ParsedEntity(text="Kenya"), [KEN], id="one-country"),
        pytest.param(ParsedEntity(text="African Union"), AFRICAN_UNION, id="group-with-no-node"),
        pytest.param(pinned(KEN), [KEN], id="pinned-country"),
    ],
)
def test_asking_for_the_members_of_a_set_with_one_reading_changes_nothing(
    doors: Doors, entity: ParsedEntity, places: Sequence[str]
) -> None:
    """A single place has no members and a group with no node has no other reading, so
    the definition served with `members` is the one served without it."""
    asked = the_spec(doors.first_answer(WDI, entity.model_copy(update={"members": True})))
    whole = the_spec(doors.first_answer(WDI, entity))

    assert bound(asked) == {_SUBJECT: sorted(places)}
    assert asked.spec_id == whole.spec_id


@pytest.mark.parametrize("node", [AFRICA, SUB_SAHARAN])
def test_a_pinned_region_asked_for_its_members_binds_them(doors: Doors, node: str) -> None:
    """A continent or UN subregion pinned by dcid expands as its name does, so `members`
    binds its members whether or not its node holds the series."""
    spec = the_spec(doors.first_answer(WHO, pinned(node, members=True)))

    assert bound(spec) == {_SUBJECT: members_reading(node)}


def test_a_pinned_group_node_is_one_place_and_refuses_members(doors: Doors) -> None:
    """The European Union's node pinned by dcid is one place, since a pin skips the name
    resolution that gives a group its members. Asked for its members it refuses
    `request_invalid` at the entity."""
    assert bound(the_spec(doors.first_answer(WHO, pinned(EU)))) == {_SUBJECT: [EU]}

    with pytest.raises(EngineInputError) as refused:
        doors.first_answer(WHO, "Kenya", pinned(EU, members=True))

    assert refused.value.code == "request_invalid"
    assert refused.value.pointer == "/input/parts/0/entities/1"


def test_a_set_named_by_two_texts_is_the_only_set_a_rank_over_place_finds(doors: Doors) -> None:
    """Aid from France to Sub-Saharan Africa, ranked over place. The subregion's node holds
    a row from France, and the rank asks for the members of the part's only set. Naming the
    subregion by a second text leaves it the only set."""
    assert flow_reading(AID, SUB_SAHARAN, RECIPIENT, [FRA]) == [SUB_SAHARAN]
    named = [origin("France"), destination(LABEL[SUB_SAHARAN])]

    once = the_spec(doors.first_answer(AID, *named, operations=[_RANK]))
    twice = the_spec(
        doors.first_answer(AID, *named, destination(SUB_SAHARAN_ALIAS), operations=[_RANK])
    )

    assert (
        bound(twice)
        == bound(once)
        == {"origin": [FRA], "destination": sorted(EXPANSION[SUB_SAHARAN])}
    )
    assert twice.spec_id == once.spec_id
    assert twice.resolved.entity_groups == once.resolved.entity_groups
    check_three_doors(doors, AID, twice)


# Invariant 4.


@pytest.mark.parametrize(
    ("stat_var", "sets", "others"),
    [
        pytest.param(WDI, [AFRICA], []),
        pytest.param(WDI, [SUB_SAHARAN], []),
        pytest.param(WDI, [AFRICA], [KEN]),
        pytest.param(WDI, [EU], []),
        pytest.param(WHO, [AFRICA], []),
        pytest.param(WHO, [EU], []),
        pytest.param(WDI, [AFRICA], [FRA]),
        pytest.param(WHO, [AFRICA, ASIA], []),
    ],
    ids=[
        "region-as-members",
        "subregion-as-members",
        "region-and-one-of-its-members",
        "group-as-members",
        "region-as-node",
        "group-as-node",
        "region-and-an-outside-country",
        "two-regions-settling-apart",
    ],
)
def test_a_role_carries_a_group_only_when_it_binds_that_sets_members(
    doors: Doors, stat_var: str, sets: list[str], others: list[str]
) -> None:
    """The subject role binds what each named set settles to and each country named beside
    it. Where those dcids are one named set's members, the role carries that set as its
    group: the set's dcid, the graph's label for it and its graph type. Where the role
    binds a set's node, or members beside a place outside the set, it carries none."""
    spec = the_spec(doors.first_answer(stat_var, *_named(*sets, *others)))

    dcids = sorted({*others, *(d for node in sets for d in subject_reading(stat_var, node))})
    carrier = next((node for node in sets if sorted(EXPANSION[node]) == dcids), None)
    assert bound(spec) == {_SUBJECT: dcids}
    check_groups(spec)
    assert group(spec) == (
        None
        if carrier is None
        else {
            "ref": carrier,
            "label": LABEL[carrier],
            "entity_type": NODE_TYPE[carrier],
            "total": None,
            "group_id": None,
        }
    )


def test_a_region_named_in_other_words_carries_the_graphs_label(doors: Doors) -> None:
    """Africa named by a text other than its label binds the same members, and the group
    states the graph's label for the node."""
    response = doors.first_answer(WDI, AFRICA_ALIAS)
    spec = the_spec(response)

    assert LABEL[AFRICA] != AFRICA_ALIAS
    assert bound(spec) == {_SUBJECT: sorted(EXPANSION[AFRICA])}
    assert len(places_with_data(WDI, EXPANSION[AFRICA])) < len(EXPANSION[AFRICA])
    assert departure_asks(response, "entity_unavailable") == [AFRICA_ALIAS]
    assert group(spec) == {
        "ref": AFRICA,
        "label": LABEL[AFRICA],
        "entity_type": NODE_TYPE[AFRICA],
        "total": None,
        "group_id": None,
    }


@pytest.mark.parametrize("stat_vars", [(WHO, WDI), (WDI, WHO)], ids=["elected", "alternate"])
def test_a_reading_labels_its_places_alike_elected_or_offered(
    doors: Doors, stat_vars: tuple[str, str]
) -> None:
    """Africa named in other words, for one series that binds its node and one that binds
    its members. The reading that binds the node labels it in the request's words whether
    it is elected or offered beside the other."""
    spec = the_readings(doors.readings(stat_vars, AFRICA_ALIAS))[WHO]

    assert bound(spec) == {_SUBJECT: [AFRICA]}
    assert [entity.label for entity in spec.resolved.entities] == [AFRICA_ALIAS]


_THE_UNION = "the union"


@pytest.mark.parametrize("pin_first", [True, False], ids=["pin-first", "name-first"])
def test_a_groups_pinned_node_beside_its_name_is_the_group(doors: Doors, pin_first: bool) -> None:
    """The European Union's node pinned by dcid under a text of the caller's, beside the
    union's name, is one set that expands as the name does, in either order. The series
    holds nothing at the node, so the Spec binds the members and carries the union's group,
    as it does for the name alone."""
    texts = [pinned(EU, _THE_UNION), LABEL[EU]]
    if not pin_first:
        texts.reverse()

    named = the_spec(doors.first_answer(WDI, LABEL[EU]))
    spec = the_spec(doors.first_answer(WDI, *texts))

    assert bound(spec) == {_SUBJECT: sorted(EXPANSION[EU])}
    assert spec.spec_id == named.spec_id
    assert group(spec) == group(named)
    check_three_doors(doors, WDI, spec)


@pytest.mark.parametrize("pin_first", [True, False], ids=["pin-first", "name-first"])
def test_a_groups_pinned_node_beside_its_name_is_admitted_as_the_group(
    monkeypatch: pytest.MonkeyPatch, pin_first: bool
) -> None:
    """External debt's recipient role admits countries only. The union's name is admitted
    by its members' type, and so is its node pinned beside the name. They are one set,
    judged once."""
    assert NODE_TYPE[EU] not in ROLE_RANGE[RECIPIENT]
    rows = (*world.FLOW_DATA[DEBT], ("", DEU))
    monkeypatch.setattr(world, "FLOW_DATA", {**world.FLOW_DATA, DEBT: rows})
    doors = Doors(monkeypatch)
    texts = [pinned(EU, _THE_UNION), LABEL[EU]]
    if not pin_first:
        texts.reverse()

    named = the_spec(doors.first_answer(DEBT, LABEL[EU]))
    spec = the_spec(doors.first_answer(DEBT, *texts))

    assert bound(spec) == {"destination": sorted(EXPANSION[EU])}
    assert spec.spec_id == named.spec_id


_SSA_TEXTS = [LABEL[SUB_SAHARAN], SUB_SAHARAN_ALIAS]


@pytest.mark.parametrize("reverse", [False, True], ids=["as-listed", "reversed"])
@pytest.mark.parametrize(
    ("texts", "others", "untold"),
    [
        pytest.param([AFRICA_ALIAS, "Africa"], [origin("France")], None, id="region-and-alias"),
        pytest.param(_SSA_TEXTS, [origin("France")], None, id="two-spellings"),
        pytest.param(
            ["African Union", "african union"],
            [origin("France")],
            None,
            id="group-with-no-node-two-spellings",
        ),
        pytest.param(
            [pinned(EU, _THE_UNION), LABEL[EU]],
            [destination("Kenya")],
            None,
            id="pinned-node-and-name",
        ),
        pytest.param(
            [destination(LABEL[SUB_SAHARAN]), SUB_SAHARAN_ALIAS],
            ["Ghana"],
            ["Ghana"],
            id="directed-and-undirected-beside-a-country",
        ),
    ],
)
def test_a_direction_refusal_names_every_text_of_each_untold_set(
    monkeypatch: pytest.MonkeyPatch,
    texts: list[str | ParsedEntity],
    others: list[str | ParsedEntity],
    untold: list[str] | None,
    reverse: bool,
) -> None:
    """Aid needs each place's direction. With no aid flow in either direction, no reading of
    an unstated direction holds data, so the answer refuses. A set no text of which is
    directed is named by every text the request posted for it, in query order, a pin's key
    included, and a set one text directs is not named. With no `untold` given, the names
    are the set's texts as posted."""
    monkeypatch.setattr(world, "FLOW_DATA", {**world.FLOW_DATA, AID: ()})
    doors = Doors(monkeypatch)
    ordered = texts[::-1] if reverse else texts

    response = doors.first_answer(AID, *ordered, *others)

    expected = untold or [text if isinstance(text, str) else text.text for text in ordered]
    assert no_data_terms(response) == ("role_not_resolved", expected)


def test_a_set_named_without_a_direction_takes_the_one_role_left_that_holds_data(
    doors: Doors,
) -> None:
    """France gives aid, and Africa, named with no direction, can only receive it: the
    providing role is France's. Aid flows from France to Africa's members, so the answer
    reads Africa as the recipient and states the direction it inferred."""
    response = doors.first_answer(AID, "Africa", origin("France"))

    spec = the_spec(response)
    assert sorted(bound(spec)) == ["destination", "origin"]
    (part,) = response.parts
    assert [(d.code, d.asked) for d in part.departures if d.code == "direction_inferred"] == [
        ("direction_inferred", "Africa")
    ]


def test_a_groups_pinned_node_alone_is_one_place(doors: Doors) -> None:
    """The European Union's node pinned by dcid with no text naming the union is one
    place. Nothing expands it. It binds the node where the series holds data there and
    answers `no_observations` where it holds none, whatever the members hold."""
    assert places_with_data(WDI, EXPANSION[EU])

    assert bound(the_spec(doors.first_answer(WHO, pinned(EU, _THE_UNION)))) == {_SUBJECT: [EU]}
    assert no_data_reason(doors.first_answer(WDI, pinned(EU, _THE_UNION))) == "no_observations"


@pytest.mark.parametrize(
    ("entities", "dropped"),
    [
        pytest.param([origin("Africa"), destination(AFRICA_ALIAS)], ["Africa", AFRICA_ALIAS]),
        pytest.param([destination(AFRICA_ALIAS), origin("Africa")], ["Africa", AFRICA_ALIAS]),
        pytest.param([AFRICA_ALIAS, destination("Africa")], ["Africa"]),
        pytest.param([destination("Africa"), AFRICA_ALIAS], ["Africa"]),
    ],
    ids=["both-directed", "both-directed-reversed", "directed-last", "directed-first"],
)
def test_a_dropped_direction_is_stated_for_each_text_that_carried_one(
    doors: Doors, entities: list[str | ParsedEntity], dropped: list[str]
) -> None:
    """Africa named by two texts on a series with no flow role is one subject. Each text
    posted with a direction has that direction dropped and says so under its own words,
    and a text posted with none says nothing."""
    response = doors.first_answer(WDI, *entities)

    assert bound(the_spec(response)) == {_SUBJECT: sorted(EXPANSION[AFRICA])}
    assert sorted(departure_asks(response, "direction_discarded")) == dropped


def test_members_without_values_count_against_the_set_they_came_from(doors: Doors) -> None:
    """Africa, the African Union and Saint Helena. Africa's node holds the series, so the
    role binds it beside the union's members and Saint Helena, which together are every
    member of Africa. The places without values are two of the union's and Saint Helena.
    The departure asks for those, and Africa, served at its node, is no part of it."""
    response = doors.first_answer(WHO, "Africa", "African Union", "Saint Helena")
    spec = the_spec(response)

    dcids = sorted({AFRICA, *AFRICAN_UNION, SHN})
    assert bound(spec) == {_SUBJECT: dcids}
    assert set(EXPANSION[AFRICA]) <= set(dcids)
    without = set(dcids) - set(places_with_data(WHO, dcids))
    assert SHN in without
    assert without & set(AFRICAN_UNION)
    assert departure_asks(response, "entity_unavailable") == ["Saint Helena, African Union"]


_SAINT_HELENA = ["Saint Helena, Africa"], [(("Saint Helena",), (("Africa", 2, 5),))]


@pytest.mark.parametrize(
    ("stat_var", "names", "asked", "counted", "group_asked"),
    [
        pytest.param(WDI, ["Africa", "Saint Helena"], *_SAINT_HELENA, ["Africa"], id="group"),
        pytest.param(WDI, ["Saint Helena", "Africa"], *_SAINT_HELENA, ["Africa"], id="group-rev"),
        pytest.param(WDI, ["Africa", "Saint Helena", "France"], *_SAINT_HELENA, [], id="no-group"),
        pytest.param(
            WDI, ["France", "Saint Helena", "Africa"], *_SAINT_HELENA, [], id="no-group-rev"
        ),
        pytest.param(
            AID,
            [origin("Africa"), origin("Nigeria"), destination("Kenya")],
            ["Nigeria, Africa"],
            [(("Nigeria",), (("Africa", 4, 5),))],
            ["Africa"],
            id="flow-origin",
        ),
        pytest.param(
            AID,
            [origin("Ghana"), destination("Africa")],
            ["Africa"],
            [((), (("Africa", 4, 5),))],
            ["Africa"],
            id="alone-on-the-other-role",
        ),
        pytest.param(
            AID,
            [destination("Africa"), origin("Ghana")],
            ["Africa"],
            [((), (("Africa", 4, 5),))],
            ["Africa"],
            id="alone-on-the-other-role-rev",
        ),
        pytest.param(
            AID,
            [origin("Ghana"), destination("Africa"), destination("Ghana")],
            ["Ghana, Africa"],
            [(("Ghana",), (("Africa", 4, 5),))],
            ["Africa"],
            id="alone-on-both-roles",
        ),
    ],
)
def test_a_member_named_on_its_own_is_named_beside_its_set(
    doors: Doors,
    stat_var: str,
    names: list[str | ParsedEntity],
    asked: list[str],
    counted: list[Unavailable],
    group_asked: list[str],
) -> None:
    """A member of Africa named on its own beside it, for a series Africa's node lacks or a
    role that does not admit a continent. Where the member holds no values on the role it
    was named on, the departure names it as the request did, and Africa's count is every
    member without values, that one included. On another role it is only one of Africa's
    members: Ghana named as the provider is counted under Africa as a recipient."""
    assert set(EXPANSION[AFRICA]) - set(places_with_data(WDI, EXPANSION[AFRICA])) == {NGA, SHN}
    assert (GHA, GHA) not in FLOW_DATA[AID]

    response = doors.first_answer(stat_var, *names)

    assert departure_asks(response, "entity_unavailable") == asked
    assert unavailable(the_spec(response)) == counted
    assert departure_asks(response, "group_read_as_members") == group_asked


def test_a_group_with_no_node_carries_its_name_and_no_dcid(doors: Doors) -> None:
    spec = the_spec(doors.first_answer(WDI, "African Union"))

    assert bound(spec) == {_SUBJECT: sorted(AFRICAN_UNION)}
    served = group(spec)
    assert served is not None
    assert served["ref"] is None


# Invariant 5.


def _named_groups(spec: Spec) -> dict[str, tuple[str, str]]:
    """The dcid and label of each group `spec` serves that has a dcid, by role."""
    return {
        role: (served.ref, served.label)
        for role, served in (spec.resolved.entity_groups or {}).items()
        if served.ref is not None
    }


_ROUND_TRIPS = [
    pytest.param(WDI, ["Kenya"], None, id="country"),
    pytest.param(WDI, ["European Union"], EU, id="group-with-a-node"),
    pytest.param(WDI, ["African Union"], None, id="group-with-no-node"),
    pytest.param(WDI, ["European Union", "France"], EU, id="group-and-one-of-its-members"),
    pytest.param(WHO, ["Africa"], None, id="region-as-node"),
    pytest.param(WDI, ["Africa"], AFRICA, id="region-as-members"),
    pytest.param(WDI, ["Africa", "Kenya"], AFRICA, id="region-and-one-of-its-members"),
    pytest.param(WDI, ["Sub-Saharan Africa"], SUB_SAHARAN, id="subregion-as-members"),
]


@pytest.mark.parametrize(("stat_var", "names", "group_dcid"), _ROUND_TRIPS)
def test_a_served_definition_resubmits_to_itself_with_its_groups(
    doors: Doors, stat_var: str, names: list[str], group_dcid: str | None
) -> None:
    """A Spec posted back with its entity groups answers definitely with the same
    `spec_id`, and serves again each group that has a dcid, under the same label."""
    spec = the_spec(doors.first_answer(stat_var, *names))
    assert {dcid for dcid, _label in _named_groups(spec).values()} == (
        {group_dcid} if group_dcid else set()
    )

    again = the_spec(
        doors.resubmit(stat_var, spec.definition.identity, spec.resolved.entity_groups)
    )

    assert again.spec_id == spec.spec_id
    assert _named_groups(again) == _named_groups(spec)


def test_a_flow_definition_naming_a_region_resubmits_with_its_group(doors: Doors) -> None:
    spec = the_spec(doors.first_answer(AID, origin("France"), destination("Africa")))
    assert _named_groups(spec) == {"destination": (AFRICA, LABEL[AFRICA])}

    again = the_spec(doors.resubmit(AID, spec.definition.identity, spec.resolved.entity_groups))

    assert again.spec_id == spec.spec_id
    assert _named_groups(again) == _named_groups(spec)


@pytest.mark.parametrize("node", [AFRICA, SUB_SAHARAN, EU])
def test_a_resubmit_states_a_kept_group_as_the_graph_does(doors: Doors, node: str) -> None:
    """A set's members posted with the set as their group, under a label of the caller's
    own: the resubmit keeps the group and states the graph's label and type for it."""
    identity = doors.identity(WDI, {_SUBJECT: EXPANSION[node]})
    posted = {_SUBJECT: EntityGroup(ref=node, label="a label of the caller's own")}

    again = check_posted(doors, WDI, identity, posted)

    assert group(again) == {
        "ref": node,
        "label": LABEL[node],
        "entity_type": NODE_TYPE[node],
        "total": None,
        "group_id": None,
    }


# Invariant 6.


def _dcids_bound(spec: Spec) -> int:
    return sum(len(set(dcids)) for dcids in bound(spec).values())


def test_a_region_with_more_members_than_the_bound_is_refused_whole(
    doors: Doors, search_settings
) -> None:
    """Africa's five members under a bound of four, for a series its node lacks. The answer
    is `no_data` with `entity_set_too_large`. No Spec binds the first four."""
    search_settings(max_child_places=len(EXPANSION[AFRICA]) - 1)

    response = doors.first_answer(WDI, "Africa")

    assert no_data_reason(response) == "entity_set_too_large"


def test_a_region_with_as_many_members_as_the_bound_binds_them_all(
    doors: Doors, search_settings
) -> None:
    search_settings(max_child_places=len(EXPANSION[AFRICA]))

    spec = the_spec(doors.first_answer(WDI, "Africa"))

    assert bound(spec) == {_SUBJECT: sorted(EXPANSION[AFRICA])}
    assert _dcids_bound(spec) == len(EXPANSION[AFRICA])


def test_a_region_over_the_bound_whose_node_holds_the_series_binds_its_node(
    doors: Doors, search_settings
) -> None:
    """The bound counts the dcids a Spec binds. Africa's node holds the series, so the Spec
    binds one dcid and reads it, however many members Africa has."""
    search_settings(max_child_places=len(EXPANSION[AFRICA]) - 1)

    spec = the_spec(doors.first_answer(WHO, "Africa"))

    assert bound(spec) == {_SUBJECT: [AFRICA]}
    check_first_answer(doors, WHO, spec)


def test_the_bound_counts_the_members_a_set_is_asked_for(doors: Doors, search_settings) -> None:
    """Africa's node holds the series. Asked for its five members under a bound of four,
    the answer is `entity_set_too_large` where the bare name binds the node. Africa and
    Asia asked for their members under a bound of their eight members bind all eight."""
    search_settings(max_child_places=len(EXPANSION[AFRICA]) - 1)
    assert no_data_reason(doors.first_answer(WHO, members("Africa"))) == "entity_set_too_large"

    both = len(EXPANSION[AFRICA]) + len(EXPANSION[ASIA])
    search_settings(max_child_places=both)
    spec = the_spec(doors.first_answer(WHO, members("Africa"), members("Asia")))
    assert bound(spec) == {_SUBJECT: members_reading(AFRICA, ASIA)}
    assert _dcids_bound(spec) == both


def test_the_bound_is_summed_over_a_specs_roles(doors: Doors, search_settings) -> None:
    """Three named origins and Africa's five members as destinations make eight dcids.
    Under a bound of seven the answer is `entity_set_too_large`, though neither role
    exceeds the bound alone. Under a bound of eight the Spec binds all eight."""
    origins = (DEU, FRA, ITA)
    entities = [*(origin(name) for name in _named(*origins)), destination("Africa")]
    total = len(origins) + len(EXPANSION[AFRICA])

    search_settings(max_child_places=total)
    spec = the_spec(doors.first_answer(AID, *entities))
    assert bound(spec) == {"origin": sorted(origins), "destination": sorted(EXPANSION[AFRICA])}

    search_settings(max_child_places=total - 1)
    assert no_data_reason(doors.first_answer(AID, *entities)) == "entity_set_too_large"


def test_an_open_role_is_outside_the_bound(doors: Doors, search_settings) -> None:
    """A flow from France with the destination left open binds France alone. The
    recipients it discovers are no part of its identity, so a bound of one admits it."""
    search_settings(max_child_places=1)

    spec = the_spec(doors.first_answer(AID, origin("France")))

    assert bound(spec) == {"origin": [FRA]}
    assert spec.resolved.scope is not None
    assert len(spec.resolved.scope.members) > 1


@pytest.mark.parametrize(
    ("stat_var", "entities", "pointer"),
    [
        (WDI, {_SUBJECT: EXPANSION[AFRICA]}, "/identity/entities/observationAbout"),
        (AID, {"origin": (DEU, FRA, ITA), "destination": (GHA, KEN, NGA)}, "/identity/entities"),
    ],
    ids=["one-role-over", "the-sum-over"],
)
def test_a_resubmit_binding_more_dcids_than_the_bound_is_refused(
    doors: Doors, search_settings, stat_var: str, entities: dict, pointer: str
) -> None:
    """A posted identity over the bound refuses `request_invalid`, pointing at the role
    that exceeds it alone, or at the entities when only their sum does."""
    identity = doors.identity(stat_var, entities)
    search_settings(max_child_places=sum(len(dcids) for dcids in entities.values()) - 1)

    with pytest.raises(EngineInputError) as refused:
        doors.resubmit(stat_var, identity)

    assert refused.value.code == "request_invalid"
    assert refused.value.pointer is not None
    assert refused.value.pointer.endswith(pointer)
