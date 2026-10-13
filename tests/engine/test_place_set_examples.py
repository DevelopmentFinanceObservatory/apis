"""Requests that name a place set, one per test, on the world in `_place_set_world`.

Each test asks one door one question and compares the answer with what the world's tables
imply: which dcids the set settles to, the observations held there, and the group the
role carries. `test_place_set_invariants` holds the rules these requests follow and the
requests a rule's own grid already asks.
"""

from __future__ import annotations

import pytest

from entity_resolution import acquire_shared, release_shared
from entity_resolution import resolver as entity_resolver
from entity_resolution.membership import NotRecorded
from search.engine.extract import ExtractedEntity, ExtractedOperation
from search.models import EntityGroup, Operation, ParsedEntity, ParsedPart, ResolveResponse
from tests.engine import _place_set_world as world
from tests.engine._place_set_world import (
    AFRICA,
    AID,
    EU,
    EXPANSION,
    FRA,
    KEN,
    LABEL,
    NO_YEAR,
    PROVIDER,
    RECIPIENT,
    WDI,
    WHO,
    Doors,
    bound,
    check_first_answer,
    check_minted,
    check_posted,
    destination,
    flow_reading,
    group,
    held_observations,
    members,
    members_reading,
    no_data_reason,
    origin,
    places_with_data,
    subject_reading,
    the_spec,
    warning_codes,
)

_SUBJECT = "observationAbout"
_AFRICAS_MEMBERS = sorted(EXPANSION[AFRICA])


def _registry_members(group_ref: str) -> list[str]:
    """The group's members under the `countries` projection, from the membership registry the
    resolver reads, sorted."""
    shared = acquire_shared()
    try:
        found = shared.groups.get(group_ref).members()
    finally:
        release_shared(shared)
    assert not isinstance(found, NotRecorded)
    return sorted(found)


@pytest.fixture
def doors(monkeypatch: pytest.MonkeyPatch) -> Doors:
    return Doors(monkeypatch)


def _restated_entities(response: ResolveResponse) -> list[dict]:
    """Each entity of the one part's restatement, as the wire carries it."""
    (part,) = response.parts
    assert part.restatement is not None
    return [
        entity.model_dump(mode="json", exclude_none=True)
        for entity in part.restatement.entities or []
    ]


def test_a_series_held_at_countries_answers_for_africas_members_by_name(doors: Doors) -> None:
    """Africa's node holds none of the series, so the Spec binds Africa's members. Its
    heading states how many and names Africa, and its mint writes the rows of the members
    that hold data and warns that the rest hold none."""
    spec = the_spec(doors.first_answer(WDI, "Africa"))

    assert bound(spec) == {_SUBJECT: _AFRICAS_MEMBERS}
    (where,) = [token.text for token in spec.resolved.sentence_plan or [] if token.axis == "where"]
    assert str(len(_AFRICAS_MEMBERS)) in where
    assert LABEL[AFRICA] in where
    manifest = check_minted(doors, WDI, spec.definition.identity)
    assert len(places_with_data(WDI, _AFRICAS_MEMBERS)) < len(_AFRICAS_MEMBERS)
    assert "coverage_below_scope" in warning_codes(manifest)


def test_a_series_held_at_africas_node_answers_from_the_node(doors: Doors) -> None:
    """The Spec binds `africa`, states the node's own observations at one place, and its
    mint writes those rows."""
    spec = the_spec(doors.first_answer(WHO, "Africa"))

    assert bound(spec) == {_SUBJECT: subject_reading(WHO, AFRICA)} == {_SUBJECT: [AFRICA]}
    check_first_answer(doors, WHO, spec)
    check_minted(doors, WHO, spec.definition.identity)


def test_a_series_held_at_africas_node_answers_for_its_members_when_asked(doors: Doors) -> None:
    """Africa posted with `members` binds Africa's members though the node holds the
    series. The Spec states the members' observations, and its mint writes their rows."""
    spec = the_spec(doors.first_answer(WHO, members("Africa")))

    assert bound(spec) == {_SUBJECT: members_reading(AFRICA)} == {_SUBJECT: _AFRICAS_MEMBERS}
    check_first_answer(doors, WHO, spec)
    check_minted(doors, WHO, spec.definition.identity)


@pytest.mark.parametrize(
    ("posted", "restated"),
    [
        pytest.param({}, {"text": "Africa"}, id="omitted"),
        pytest.param({"members": False}, {"text": "Africa"}, id="false"),
        pytest.param({"members": True}, {"text": "Africa", "members": True}, id="true"),
    ],
)
def test_a_parts_restatement_carries_members_only_when_it_is_true(
    doors: Doors, posted: dict, restated: dict
) -> None:
    """On the wire a part that does not ask for members restates its entity with no
    `members` key, and one that asks restates the flag."""
    entity = ParsedEntity.model_validate({"text": "Africa", **posted})

    assert _restated_entities(doors.first_answer(WHO, entity)) == [restated]


def test_wording_that_names_a_groups_members_binds_them_without_the_flag(doors: Doors) -> None:
    """ "EU member states" names the European Union's members, so the Spec binds the 27 the
    membership registry lists where "European Union" binds the union's node."""
    spec = the_spec(doors.first_answer(WHO, "EU member states"))
    eu_members = _registry_members(EU)

    assert subject_reading(WHO, EU) == [EU]
    assert len(eu_members) == 27
    assert bound(spec) == {_SUBJECT: eu_members}


def test_raw_text_that_asks_for_africas_members_binds_them(doors: Doors) -> None:
    """ "how much do African countries spend on their own health care?" The extractor
    states `members` on Africa, so the Spec binds Africa's members though the node holds
    the series, and the part's restatement states the reading. The same extraction
    without the flag binds the node, and its restatement carries no `members`. Two mentions
    of Africa fold to one reading, and the one that asks for the members decides it, under
    the same spelling or another."""
    query = "how much do African countries spend on their own health care?"
    as_members = ExtractedEntity(text="Africa", span="African", members=True)
    as_one_place = ExtractedEntity(text="Africa", span="African")
    respelled = ExtractedEntity(text="africa", span="African", members=True)

    asked = doors.raw_text(query, WHO, as_members)
    spec = the_spec(asked)

    assert bound(spec) == {_SUBJECT: members_reading(AFRICA)} == {_SUBJECT: _AFRICAS_MEMBERS}
    check_first_answer(doors, WHO, spec)
    assert _restated_entities(asked) == [{"text": "Africa", "members": True}]

    unasked = doors.raw_text(query, WHO, as_one_place)
    assert bound(the_spec(unasked)) == {_SUBJECT: [AFRICA]}
    assert _restated_entities(unasked) == [{"text": "Africa"}]

    for mentions in ((as_one_place, as_members), (as_members, as_one_place)):
        folded = doors.raw_text(query, WHO, *mentions)
        assert bound(the_spec(folded)) == {_SUBJECT: _AFRICAS_MEMBERS}
        assert _restated_entities(folded) == [{"text": "Africa", "members": True}]

    for mentions in ((as_one_place, respelled), (respelled, as_one_place)):
        merged = the_spec(doors.raw_text(query, WHO, *mentions))
        assert bound(merged) == {_SUBJECT: _AFRICAS_MEMBERS}
        assert merged.spec_id == spec.spec_id


@pytest.mark.parametrize("members", [True, False], ids=["members-stated", "members-unstated"])
def test_raw_text_ranking_africas_recipients_binds_its_members(doors: Doors, members: bool) -> None:
    """ "which African countries received the most aid" ranks the recipients with Africa
    as the recipient, and binds Africa's members whether or not the extractor states
    `members`."""
    africa = ExtractedEntity(
        text="Africa",
        span="African",
        flow_role="destination",
        role_span="received",
        members=members,
    )
    most = ExtractedOperation(
        verb="rank", variable=0, phrase="which received the most", over="destination"
    )

    response = doors.raw_text(
        "which African countries received the most aid", AID, africa, operations=[most]
    )

    assert bound(the_spec(response)) == {"destination": _AFRICAS_MEMBERS}


def test_a_total_over_africa_is_the_figure_its_node_holds(doors: Doors) -> None:
    """An aggregate over place leaves the node bound: where the node holds a figure, that
    figure is the total asked for."""
    total = Operation(verb="aggregate", phrase="total", over="place")

    spec = the_spec(doors.first_answer(WHO, "Africa", operations=[total]))

    assert bound(spec) == {_SUBJECT: [AFRICA]}
    check_first_answer(doors, WHO, spec)


def test_africa_and_kenya_are_read_as_the_node_and_the_country(doors: Doors) -> None:
    """Africa's node holds the series, so the Spec binds the node beside Kenya and states
    the observations of those two places."""
    spec = the_spec(doors.first_answer(WHO, "Africa", "Kenya"))

    assert bound(spec) == {_SUBJECT: sorted([*subject_reading(WHO, AFRICA), KEN])}
    check_first_answer(doors, WHO, spec)


def test_a_rank_over_recipients_cannot_place_africa_named_with_no_direction(
    doors: Doors,
) -> None:
    """The series requires a provider and a recipient. Africa named with no direction
    beside a rank over the recipients can scope the ranked recipients or name the
    providers, so the answer is `no_data` with `role_not_resolved`."""
    most = Operation(verb="rank", phrase="received the most", over="destination")

    response = doors.first_answer(AID, "Africa", operations=[most])

    assert no_data_reason(response) == "role_not_resolved"


def test_a_rank_over_recipients_binds_the_members_of_africa_named_as_the_recipient(
    doors: Doors,
) -> None:
    """Africa is named as the recipient beside a rank over the recipients. The rank runs over
    the axis Africa fills, so Africa's members bind as destinations and the provider role
    binds nothing."""
    most = Operation(verb="rank", phrase="received the most", over="destination")

    spec = the_spec(doors.first_answer(AID, destination("Africa"), operations=[most]))

    assert bound(spec) == {"destination": _AFRICAS_MEMBERS}


def test_a_flow_between_two_sets_tests_each_node_against_the_others_members(
    doors: Doors,
) -> None:
    """Aid from the European Union to Africa. The union's node holds rows to Africa's
    members, so it binds its node. `africa` holds none from the union's members, and the
    recipient role does not admit a continent, so Africa binds its members."""
    spec = the_spec(doors.first_answer(AID, origin("European Union"), destination("Africa")))

    assert bound(spec) == {
        "origin": flow_reading(AID, EU, PROVIDER, EXPANSION[AFRICA]),
        "destination": flow_reading(AID, AFRICA, RECIPIENT, EXPANSION[EU]),
    }
    assert bound(spec) == {"origin": [EU], "destination": _AFRICAS_MEMBERS}


def test_a_group_named_beside_one_of_its_members_binds_the_groups_members(doors: Doors) -> None:
    """The European Union and France, for a series the union's node lacks. The role binds
    the union's 27 registry members, France among them, and carries the union as its group."""
    spec = the_spec(doors.first_answer(WDI, "European Union", "France"))
    eu_members = _registry_members(EU)

    assert subject_reading(WDI, EU) != [EU]
    assert FRA in eu_members
    assert bound(spec) == {_SUBJECT: eu_members}
    served = group(spec)
    assert served is not None
    assert served["ref"] == EU


def test_wording_that_names_no_place_does_not_resolve(doors: Doors) -> None:
    """ "African countries" is no place's name, so it resolves to nothing."""
    assert no_data_reason(doors.first_answer(WDI, "African countries")) == "entity_not_resolved"


def test_a_year_no_place_of_a_set_holds_answers_as_its_members_do(doors: Doors) -> None:
    """Neither `africa` nor any member holds the series in the year asked for. The answer
    is the one the members named one by one get."""
    members = [LABEL[dcid] for dcid in _AFRICAS_MEMBERS]
    assert held_observations(WHO, {_SUBJECT: [AFRICA, *_AFRICAS_MEMBERS]}, NO_YEAR) == 0

    region = doors.first_answer(WHO, "Africa", year=NO_YEAR)

    assert no_data_reason(region) == no_data_reason(doors.first_answer(WHO, *members, year=NO_YEAR))


def test_a_regions_members_resubmit_with_no_group(doors: Doors) -> None:
    """An identity binding Africa's members, posted with no entity groups, answers with the
    same `spec_id` and no group."""
    identity = doors.identity(WDI, {_SUBJECT: _AFRICAS_MEMBERS})

    again = check_posted(doors, WDI, identity)

    assert again.resolved.entity_groups is None


def test_a_group_whose_members_differ_from_the_graphs_is_dropped(doors: Doors) -> None:
    """Africa's group is posted beside all but one of Africa's members. The resubmit answers
    for the dcids posted and drops the group."""
    identity = doors.identity(WDI, {_SUBJECT: _AFRICAS_MEMBERS[:-1]})
    posted = {_SUBJECT: EntityGroup(ref=AFRICA, label=LABEL[AFRICA])}

    again = check_posted(doors, WDI, identity, posted)

    assert again.resolved.entity_groups is None


def test_a_regions_node_posted_with_its_group_is_read_at_the_node(doors: Doors) -> None:
    """`africa` is posted with Africa's group beside it. The role binds one dcid, which
    differs from Africa's members, so the group is dropped and the answer is the node's observations."""
    identity = doors.identity(WHO, {_SUBJECT: [AFRICA]})
    posted = {_SUBJECT: EntityGroup(ref=AFRICA, label=LABEL[AFRICA])}

    again = check_posted(doors, WHO, identity, posted)

    assert again.resolved.entity_groups is None


# Two countries the resolver offers for "Congo". The world's graph holds them only in the
# tests that name Congo, since the resolver's African Union counts both among its members.
_CONGOS = ("country/COD", "country/COG")


def _data_query(response: ResolveResponse) -> ParsedPart:
    """The one part of an `entity_ambiguous` answer's `data_query`."""
    (part,) = response.parts
    assert part.status == "no_data", part.status
    assert part.no_data.reason == "entity_ambiguous"
    assert part.no_data.data_query is not None
    (query_part,) = part.no_data.data_query.parts
    return query_part


def _pick(part: ParsedPart, text: str, dcid: str) -> ParsedPart:
    """`part` with the entity `text` pinned to `dcid`, as a client posts the user's pick."""
    entities = [
        entity.model_copy(update={"dcid": dcid}) if entity.text == text else entity
        for entity in part.entities or ()
    ]
    return part.model_copy(update={"entities": entities})


@pytest.mark.parametrize(
    ("africa", "echoed", "places"),
    [
        pytest.param(
            members("Africa"),
            ("Africa", None, True),
            _AFRICAS_MEMBERS,
            id="flag",
        ),
        pytest.param(
            ParsedEntity(text="African nations"),
            ("African nations", None, False),
            _AFRICAS_MEMBERS,
            id="alias-row-declaring-members",
        ),
        pytest.param(
            ParsedEntity(text="Africa"),
            ("Africa", AFRICA, False),
            [AFRICA],
            id="whole",
        ),
    ],
)
def test_a_data_query_posted_back_keeps_the_reading_of_each_set(
    monkeypatch: pytest.MonkeyPatch,
    africa: ParsedEntity,
    echoed: tuple[str, str | None, bool],
    places: list[str],
) -> None:
    """Africa beside the ambiguous "Congo": the answer is `entity_ambiguous`, and its
    `data_query` states `members` where the part asked for Africa's members and leaves out
    the flag where it did not. A set read as its members, by the flag or through an alias
    row that declares its text names members, is echoed unpinned, and Africa read whole is
    pinned to its node. Posted back with Congo picked, the part binds what the same request
    with the pick in hand binds, under the same definition."""
    for dcid in _CONGOS:
        monkeypatch.setitem(world._NODES, dcid, {"label": dcid, "type": "Country", "arcs": {}})
    doors = Doors(monkeypatch)
    monkeypatch.setitem(
        entity_resolver._QUERY_ALIASES,
        "african nations",
        entity_resolver._QueryAlias(name="Africa", members=True),
    )
    assert subject_reading(WHO, AFRICA) == [AFRICA]
    picked = ParsedEntity(text="Congo", dcid=_CONGOS[0])

    query_part = _data_query(doors.first_answer(WHO, africa, "Congo"))

    entity = query_part.entities[0]
    assert (entity.text, entity.dcid, bool(entity.members)) == echoed
    again = the_spec(doors.post(_pick(query_part, "Congo", _CONGOS[0])))
    direct = the_spec(doors.first_answer(WHO, africa, picked))
    assert bound(again) == bound(direct) == {_SUBJECT: sorted([*places, _CONGOS[0]])}
    assert again.spec_id == direct.spec_id
