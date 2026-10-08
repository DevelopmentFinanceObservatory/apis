"""How one Spec settles each place set it names to its node or its members.

`settle_place_sets` is the rule, tested over assignments built by hand and a read that
answers from a table of the places holding data. `bind_place_roles` decides which sets a
request asks the members of. `UnsettledPlace.settle` runs the rule over a Spec's own read,
tested through `standard_resolve` on a small graph.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from entity_resolution import Entity
from search.engine.errors import GraphInfraError
from search.engine.extract import DateRequest
from search.engine.interpret import PinnedResolved
from search.engine.region.decisions import PlaceSetTooLargeError
from search.engine.region.place_context import PlaceContext, UnsettledPlace
from search.engine.region.place_roles import (
    PlaceAssignment,
    PlaceLists,
    bind_place_roles,
    group_request_names,
    part_sets,
    place_key,
    settle_place_sets,
)
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.region.state import ResolveState
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import FacetBatch, standard_resolve
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.models import Operation, RoleAxis, TimeWindow
from tests.fixtures import FakeGraph

_GHA, _KEN, _TZA = "country/GHA", "country/KEN", "country/TZA"
_DEU, _FRA = "country/DEU", "country/FRA"
_IDN, _IND = "country/IDN", "country/IND"

_KENYA = PlaceSet(name="Kenya", kind="place", node=PlaceNode(_KEN, "Country"))
_AFRICA = PlaceSet(
    name="Africa",
    kind="region",
    node=PlaceNode("africa", "Continent"),
    members=(_GHA, _KEN, _TZA),
    member_type="Country",
)
_ASIA = PlaceSet(
    name="Asia",
    kind="region",
    node=PlaceNode("asia", "Continent"),
    members=(_IDN, _IND),
    member_type="Country",
)
_EU = PlaceSet(
    name="European Union",
    kind="group",
    node=PlaceNode("EuropeanUnion", "ContinentalUnion"),
    members=(_DEU, _FRA),
    member_type="Country",
)
_AFRICAN_UNION = PlaceSet(
    name="African Union", kind="group", node=None, members=(_GHA, _KEN), member_type="Country"
)


def _admits_every_type(_key: str, _graph_type: str) -> bool:
    return True


def _subject_holds(*held: str) -> Callable[[PlaceLists], bool]:
    """A subject series holding data at `held`."""
    return lambda places: any(dcid in held for dcid in places["observationAbout"])


def _flow_holds(*rows: tuple[str, str]) -> Callable[[PlaceLists], bool]:
    """A flow series holding the `(origin, destination)` rows."""
    return lambda places: any(
        (origin, destination) in rows
        for origin in places["origin"]
        for destination in places["destination"]
    )


def _settle(
    assignment: PlaceAssignment,
    has_data: Callable[[PlaceLists], bool],
    *,
    admits: Callable[[str, str], bool] = _admits_every_type,
    bound: int | None = None,
) -> PlaceLists:
    return settle_place_sets(assignment, admits=admits, has_data=has_data, bound=bound)


# One set on two roles.


@pytest.mark.parametrize(
    ("rows", "refused_role", "expected"),
    [
        pytest.param([("africa", "africa")], None, ("africa",), id="node-holds-rows-on-both-roles"),
        pytest.param(
            [("africa", _KEN), (_GHA, "africa")],
            None,
            (_GHA, _KEN, _TZA),
            id="node-holds-rows-on-one-role-at-a-time",
        ),
        pytest.param(
            [("africa", "africa")],
            "destination",
            (_GHA, _KEN, _TZA),
            id="one-role-refuses-the-node",
        ),
    ],
)
def test_a_set_on_two_roles_takes_one_reading(rows, refused_role, expected) -> None:
    """A set named once that fills both flow roles binds its node only when both roles
    admit the node's type and the series holds rows with the node on both. Otherwise it
    binds its members on both."""
    assignment = PlaceAssignment(sets={"origin": (_AFRICA,), "destination": (_AFRICA,)})

    settled = _settle(
        assignment,
        _flow_holds(*rows),
        admits=lambda key, graph_type: not (key == refused_role and graph_type == "Continent"),
    )

    assert settled == {"origin": expected, "destination": expected}


def test_a_node_no_role_admits_is_not_read() -> None:
    read: list[PlaceLists] = []
    assignment = PlaceAssignment(sets={"destination": (_AFRICA,)})

    settled = _settle(
        assignment,
        lambda places: bool(read.append(places)),
        admits=lambda _key, graph_type: graph_type != "Continent",
    )

    assert settled == {"destination": (_GHA, _KEN, _TZA)}
    assert read == []


# Two sets.


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [("EuropeanUnion", _KEN)],
            {"origin": ("EuropeanUnion",), "destination": (_GHA, _KEN, _TZA)},
            id="one-node-holds-rows-to-the-others-members",
        ),
        pytest.param(
            [("EuropeanUnion", _KEN), (_FRA, "africa"), ("EuropeanUnion", "africa")],
            {"origin": ("EuropeanUnion",), "destination": ("africa",)},
            id="both-nodes-hold-rows-together",
        ),
        pytest.param(
            [("EuropeanUnion", _KEN), (_FRA, "africa")],
            {"origin": (_DEU, _FRA), "destination": (_GHA, _KEN, _TZA)},
            id="both-nodes-hold-rows-apart-and-none-together",
        ),
        pytest.param(
            [(_FRA, _KEN)],
            {"origin": (_DEU, _FRA), "destination": (_GHA, _KEN, _TZA)},
            id="neither-node-holds-rows",
        ),
    ],
)
def test_two_sets_on_two_roles_are_each_read_against_the_others_members(rows, expected) -> None:
    """Each set's node is read with the other set at its members. When both take their
    node and the series holds no row between the two nodes, both bind their members."""
    assignment = PlaceAssignment(sets={"origin": (_EU,), "destination": (_AFRICA,)})

    assert _settle(assignment, _flow_holds(*rows)) == expected


def test_two_sets_on_one_role_settle_apart() -> None:
    """Africa's node holds the series and Asia's does not, so the role binds Africa's node
    beside Asia's members and the country named with them."""
    assignment = PlaceAssignment(sets={"observationAbout": (_AFRICA, _ASIA, _KENYA)})

    settled = _settle(assignment, _subject_holds("africa", _IDN, _KEN))

    assert settled == {"observationAbout": ("africa", _IDN, _IND, _KEN)}


def test_a_node_is_read_alone_on_its_role() -> None:
    """Kenya holds the series and Africa's node does not. Africa is read at its node alone,
    so Kenya's data does not bind the node."""
    assignment = PlaceAssignment(sets={"observationAbout": (_AFRICA, _KENYA)})

    assert _settle(assignment, _subject_holds(_KEN)) == {"observationAbout": (_GHA, _KEN, _TZA)}


# Sets with one reading.


@pytest.mark.parametrize(
    ("place_set", "members_asked", "expected"),
    [
        pytest.param(_KENYA, False, (_KEN,), id="a-place-binds-itself"),
        pytest.param(_KENYA, True, (_KEN,), id="a-place-asked-for-its-members-binds-itself"),
        pytest.param(_AFRICAN_UNION, False, (_GHA, _KEN), id="a-set-with-no-node"),
        pytest.param(_AFRICA, True, (_GHA, _KEN, _TZA), id="a-set-asked-for-its-members"),
        pytest.param(
            PlaceSet(
                name="Nowhere", kind="region", node=PlaceNode("nowhere", "Continent"), members=()
            ),
            False,
            ("nowhere",),
            id="a-set-with-no-member",
        ),
    ],
)
def test_a_set_with_one_reading_settles_without_a_read(place_set, members_asked, expected) -> None:
    def unread(_places: PlaceLists) -> bool:
        raise AssertionError("no read decides this set")

    assignment = PlaceAssignment(
        sets={"observationAbout": (place_set,)},
        members_asked=frozenset({place_key(place_set)}) if members_asked else frozenset(),
    )

    assert _settle(assignment, unread) == {"observationAbout": expected}


# A failed read.


def test_a_failed_node_read_fails_the_settling_and_reads_nothing_else() -> None:
    """The node read raising is the Spec's read failing. No members are read in its place,
    and no reading is chosen."""
    read: list[PlaceLists] = []

    def failing(places: PlaceLists) -> bool:
        read.append(places)
        raise GraphInfraError("the graph did not answer")

    assignment = PlaceAssignment(sets={"observationAbout": (_AFRICA,)})

    with pytest.raises(GraphInfraError):
        _settle(assignment, failing)
    assert read == [{"observationAbout": ("africa",)}]


# The bound.


@pytest.mark.parametrize(
    ("held", "bound", "expected"),
    [
        pytest.param(["africa"], 2, ("africa",), id="oversized-set-with-data-at-its-node"),
        pytest.param([_KEN], 3, (_GHA, _KEN, _TZA), id="members-at-the-bound"),
        pytest.param([_KEN], 2, 3, id="members-over-the-bound"),
    ],
)
def test_the_bound_counts_the_reading_chosen(held, bound, expected) -> None:
    """Africa has three members. Under a bound of two it binds a node that holds the
    series. Where the node holds none, its three members are over the bound and are
    refused whole."""
    assignment = PlaceAssignment(sets={"observationAbout": (_AFRICA,)})

    if isinstance(expected, int):
        with pytest.raises(PlaceSetTooLargeError) as refusal:
            _settle(assignment, _subject_holds(*held), bound=bound)
        assert (refusal.value.count, refusal.value.bound) == (expected, bound)
    else:
        assert _settle(assignment, _subject_holds(*held), bound=bound) == {
            "observationAbout": expected
        }


def test_the_bound_sums_the_places_of_every_role() -> None:
    """Two origins and three destinations are five places. Neither role exceeds a bound of
    four alone, and their sum does."""
    assignment = PlaceAssignment(sets={"origin": (_EU,), "destination": (_AFRICA,)})

    assert sum(len(dcids) for dcids in _settle(assignment, _flow_holds(), bound=5).values()) == 5
    with pytest.raises(PlaceSetTooLargeError) as refusal:
        _settle(assignment, _flow_holds(), bound=4)
    assert refusal.value.count == 5


def _places(*dcids: str) -> tuple[PlaceSet, ...]:
    return tuple(
        PlaceSet(name=dcid, kind="place", node=PlaceNode(dcid, "Country")) for dcid in dcids
    )


_EMPTY_REGION = PlaceSet(
    name="Empty", kind="region", node=PlaceNode("empty", "Continent"), members=()
)


@pytest.mark.parametrize(
    "sets",
    [
        pytest.param({"observationAbout": _places(_KEN, _GHA, _TZA, _DEU)}, id="four-subjects"),
        pytest.param(
            {"observationAbout": (_AFRICA, *_places(_GHA, _DEU))},
            id="a-set-s-member-named-beside-it-counts-once",
        ),
        pytest.param(
            {"destination": _places(_KEN, _TZA), "origin": _places(_GHA, _DEU)},
            id="two-roles-of-two",
        ),
        pytest.param(
            {"destination": _places(_KEN, _TZA), "origin": _places(_KEN, _TZA)},
            id="self-flow-of-two",
        ),
        pytest.param(
            {"destination": (_EMPTY_REGION,), "origin": _places(_KEN, _TZA, _GHA)},
            id="memberless-region-beside-three-origins",
        ),
    ],
)
def test_places_over_the_bound_refuse_whole(sets) -> None:
    """The bound counts distinct dcids per role, summed over the roles. Four places fit a
    bound of four and refuse under a bound of three. Nothing is truncated."""
    assignment = PlaceAssignment(sets=sets)

    settled = _settle(assignment, lambda _places: False, bound=4)
    assert sum(len(dcids) for dcids in settled.values()) == 4

    with pytest.raises(PlaceSetTooLargeError) as refusal:
        _settle(assignment, lambda _places: False, bound=3)
    assert (refusal.value.count, refusal.value.bound) == (4, 3)


# Which sets a request asks the members of.

_PROVIDER_RECIPIENT = ("financingProvider", "financingRecipient")


def _shape(*roles: str) -> ShapeDraft:
    return ShapeDraft(
        shape_id="settle_shape",
        label="Settle shape",
        pop_type_dcid="SettleFixture",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={"Amount_SettleFixture": {}},
        declared_roles=roles or None,
    )


def _state(
    sets: Sequence[PlaceSet],
    *,
    axes: Mapping[str, str] | None = None,
    operations: Sequence[Operation] = (),
    pins: Mapping[str, str] | None = None,
) -> ResolveState:
    """A resolve state naming `sets`, each directed to the flow axis `axes` gives it. The
    caller pinned each text in `pins` to its set's node, which the graph labels as
    `pins` gives."""
    roles = {
        name: [DoorRole(axis=cast(RoleAxis, axis), span=name, stated=True)]
        for name, axis in (axes or {}).items()
    }
    return cast(
        ResolveState,
        SimpleNamespace(
            entities=[place_set.name for place_set in sets],
            place_sets={place_set.name: place_set for place_set in sets},
            entity_roles=roles,
            operations=tuple(operations),
            open_axes=frozenset(),
            role_bearing_recalled=False,
            role_query="",
            shape_draft=None,
            elected_place_roles=None,
            rcl=SimpleNamespace(
                resolutions={
                    place_set.name: PinnedResolved(
                        entity=Entity(
                            dcid=cast(PlaceNode, place_set.node).dcid,
                            name=pins[place_set.name],
                            kind="place",
                            graph_type=cast(PlaceNode, place_set.node).graph_type,
                        )
                    )
                    for place_set in sets
                    if place_set.name in (pins or {})
                }
            ),
        ),
    )


def _rank(over: str) -> Operation:
    return Operation.model_validate({"verb": "rank", "phrase": "top", "over": over})


_TOTAL = Operation(verb="aggregate", phrase="total", over="place")


@pytest.mark.parametrize(
    ("operations", "asked"),
    [
        pytest.param([], False, id="no-operation"),
        pytest.param([_TOTAL], False, id="aggregate-over-place"),
        pytest.param([_rank("year")], False, id="rank-over-year"),
        pytest.param([_rank("category")], False, id="rank-over-category"),
        pytest.param([_rank("place")], True, id="rank-over-place"),
        pytest.param([_rank("origin")], True, id="rank-over-origin"),
        pytest.param([_rank("destination")], True, id="rank-over-destination"),
        pytest.param([_rank("place"), _TOTAL], True, id="rank-and-aggregate-over-place"),
    ],
)
def test_a_rank_over_a_place_axis_asks_a_subjects_members(operations, asked) -> None:
    """A subject-only shape names its subject on every place axis, so a rank over any of
    them asks for the members of a set it binds. An aggregate asks for none. The settle
    rule decides, and a node holding the figure binds."""
    assignment = bind_place_roles(_state([_AFRICA], operations=operations), _shape())

    assert isinstance(assignment, PlaceAssignment)
    assert assignment.members_asked == (frozenset({place_key(_AFRICA)}) if asked else frozenset())
    assert _settle(assignment, _subject_holds("africa")) == {
        "observationAbout": (_GHA, _KEN, _TZA) if asked else ("africa",)
    }


@pytest.mark.parametrize(
    ("over", "asked"),
    [
        pytest.param("destination", {place_key(_AFRICA)}, id="the-axis-one-set-fills"),
        pytest.param("origin", {place_key(_EU)}, id="the-axis-the-other-set-fills"),
        pytest.param("place", set(), id="place-with-two-sets"),
        pytest.param("category", set(), id="category"),
    ],
)
def test_a_rank_over_a_flow_axis_asks_the_members_of_the_set_on_it(over, asked) -> None:
    state = _state(
        [_EU, _AFRICA],
        axes={"European Union": "origin", "Africa": "destination"},
        operations=[_rank(over)],
    )

    assignment = bind_place_roles(state, _shape(*_PROVIDER_RECIPIENT))

    assert isinstance(assignment, PlaceAssignment)
    assert assignment.members_asked == asked


# Africa under a second text of the request.
_AFRIQUE = dataclasses.replace(_AFRICA, name="Afrique")


@pytest.mark.parametrize(
    "destinations", [[_AFRICA], [_AFRICA, _AFRIQUE]], ids=["named-once", "named-twice"]
)
def test_a_rank_over_place_asks_the_members_of_a_parts_only_set(destinations) -> None:
    """France beside Africa on a flow shape, ranked over place: Africa is the only set of
    the part that expands, so the rank asks for its members. Africa named by two texts is
    still the only set."""
    france = PlaceSet(name="France", kind="place", node=PlaceNode(_FRA, "Country"))
    state = _state(
        [france, *destinations],
        axes={"France": "origin"} | {named.name: "destination" for named in destinations},
        operations=[_rank("place")],
    )

    assignment = bind_place_roles(state, _shape(*_PROVIDER_RECIPIENT))

    assert isinstance(assignment, PlaceAssignment)
    assert assignment.sets["destination"] == (_AFRICA,)
    assert assignment.members_asked == {place_key(_AFRICA)}


def test_a_set_named_by_two_texts_on_two_roles_takes_one_reading() -> None:
    """Africa as the origin under one text and the destination under another is one set
    on both roles. The series holds rows from the node to a member and none from the node
    to itself, so the set binds its members on both roles."""
    state = _state([_AFRICA, _AFRIQUE], axes={"Africa": "origin", "Afrique": "destination"})

    assignment = bind_place_roles(state, _shape(*_PROVIDER_RECIPIENT))

    assert isinstance(assignment, PlaceAssignment)
    assert assignment.distinct_sets == (_AFRICA,)
    members = (_GHA, _KEN, _TZA)
    assert _settle(assignment, _flow_holds(("africa", _KEN))) == {
        "origin": members,
        "destination": members,
    }


@pytest.mark.parametrize("members_first", [False, True], ids=["asked-second", "asked-first"])
@pytest.mark.parametrize("roles", [(), _PROVIDER_RECIPIENT], ids=["subject", "flow"])
def test_a_set_takes_the_members_reading_any_of_its_texts_asks_for(members_first, roles) -> None:
    """Africa named by one text for its node and by another for its members, in either
    order, is one set under the first text's name that reads its members. The series holds
    data at the node, and the set binds its members all the same."""
    texts = [_AFRICA, dataclasses.replace(_AFRIQUE, reading="members")]
    if members_first:
        texts.reverse()
    state = _state(
        texts, axes=dict.fromkeys(("Africa", "Afrique"), "destination") if roles else None
    )

    assignment = bind_place_roles(state, _shape(*roles))

    assert isinstance(assignment, PlaceAssignment)
    ((key, (placed,)),) = assignment.sets.items()
    assert (placed.name, placed.reading) == (texts[0].name, "members")
    assert assignment.members_asked == {place_key(_AFRICA)}
    assert _settle(assignment, lambda _places: True) == {key: (_GHA, _KEN, _TZA)}


@pytest.mark.parametrize("pin_first", [True, False], ids=["pin-first", "name-first"])
def test_a_set_expands_when_any_of_its_texts_does(pin_first) -> None:
    """The European Union's node pinned as one place under a text of the caller's, beside
    the union's name. The part's one set is the group, under the union's name, and a
    pin's text is only its key. The pin alone is one place under the graph's label."""
    pin = PlaceSet(name="the union", kind="place", node=_EU.node)
    texts = [pin, _EU] if pin_first else [_EU, pin]
    pins = {pin.name: "European Union (graph)"}

    (merged,) = part_sets(_state(texts, pins=pins)).values()

    assert merged == _EU
    assert part_sets(_state([pin], pins=pins)) == {
        place_key(pin): dataclasses.replace(pin, name="European Union (graph)")
    }


@pytest.mark.parametrize("other_text", [None, "Africa"], ids=["alone", "beside-a-text-equal-to-it"])
def test_a_group_is_asked_in_the_request_s_words_unless_its_label_is_a_text(other_text) -> None:
    """Africa named "Afrique" is asked as "Afrique" where an answer labels it "Africa". When
    the request posts "Africa" itself for another set, "Africa" stays the request's own
    text and is not read as "Afrique"."""
    afrique = dataclasses.replace(_AFRICA, name="Afrique", label="Africa")
    other = PlaceSet(
        name="Africa",
        kind="region",
        node=PlaceNode("SubSaharanAfrica", "UNGeoRegion"),
        members=(_GHA, _KEN),
        member_type="Country",
    )

    words = group_request_names(_state([afrique, other] if other_text else [afrique]))

    assert words == ({} if other_text else {"Africa": "Afrique"})


def test_a_set_named_for_its_members_is_asked_for_them() -> None:
    members_of_africa = PlaceSet(
        name="Africa",
        kind="region",
        node=_AFRICA.node,
        members=_AFRICA.members,
        member_type="Country",
        reading="members",
    )

    assignment = bind_place_roles(_state([members_of_africa]), _shape())

    assert isinstance(assignment, PlaceAssignment)
    assert assignment.members_asked == {place_key(_AFRICA)}


# Settling over a Spec's own read.

_SERIES = "Amount_SettleSeries"
_NODE_YEAR, _COUNTRY_YEAR = "2005", "2018"


def _facet(year: str) -> dict:
    return {
        "earliestDate": year,
        "latestDate": year,
        "obsCount": 1,
        "provenanceId": "dc/base/SettleFixture",
        "dates": [year],
        "values": [1.0],
    }


class _CountingGraph(FakeGraph):
    """Counts the facet reads and the places each names."""

    def __init__(self, obs: dict) -> None:
        super().__init__(
            nodes={
                _SERIES: {
                    "label": "Settle series",
                    "type": "StatisticalVariable",
                    "arcs": {
                        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                        "name": {"nodes": [{"value": "Settle series"}]},
                        "populationType": {"nodes": [{"dcid": "SettleFixture"}]},
                        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                        "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    },
                }
            },
            obs=obs,
            detect={},
            resolve={},
        )
        self.facet_reads: list[list[str]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs: Any):
        self.facet_reads.append(list(entities))
        return super().observation_facets_batch(stat_vars, entities, needs_dates, **kwargs)


def _settled_read(
    graph: _CountingGraph, *, year: str | None, sets: Sequence[PlaceSet] = (_AFRICA,)
) -> tuple[PlaceContext, object]:
    """Settle `sets` as the subject of the series over its own `standard_resolve`."""
    (shape,) = derive_shapes(confirmed_svs=[_SERIES], graph=graph)
    date_request = (
        None
        if year is None
        else DateRequest(window=TimeWindow(start_year=int(year), end_year=int(year)), latest=False)
    )
    unsettled = UnsettledPlace(
        shape,
        PlaceAssignment(sets={"observationAbout": tuple(sets)}),
        admitted=frozenset({("observationAbout", "Continent"), ("observationAbout", "Country")}),
    )

    def read(place: PlaceContext, facet_batch: FacetBatch | None) -> object:
        return standard_resolve(
            shape=shape,
            bindings=[],
            entities=place.roles.resolved_entities,
            graph=graph,
            date_request=date_request,
            place_dcids=place.probe_places,
            facet_batch=facet_batch,
        )

    return unsettled.settle(read)


_NODE_AND_COUNTRIES = {
    f"{_SERIES}|africa": [_facet(_NODE_YEAR)],
    f"{_SERIES}|{_KEN}": [_facet(_COUNTRY_YEAR)],
}


@pytest.mark.parametrize(
    ("obs", "year", "bound"),
    [
        pytest.param(_NODE_AND_COUNTRIES, None, ["africa"], id="no-window-reads-every-year"),
        pytest.param(_NODE_AND_COUNTRIES, _NODE_YEAR, ["africa"], id="a-year-the-node-holds"),
        pytest.param(
            _NODE_AND_COUNTRIES, _COUNTRY_YEAR, [_GHA, _KEN, _TZA], id="a-year-only-members-hold"
        ),
        pytest.param(_NODE_AND_COUNTRIES, "1990", [_GHA, _KEN, _TZA], id="a-year-no-place-holds"),
        pytest.param(
            {f"{_SERIES}|{_KEN}": [_facet(_COUNTRY_YEAR)]},
            None,
            [_GHA, _KEN, _TZA],
            id="no-observation-at-the-node",
        ),
    ],
)
def test_a_spec_settles_on_its_own_read_in_one_facet_read(obs, year, bound) -> None:
    """The node binds when the Spec's own resolve holds data there in the years asked for,
    every year when none is asked. The node and the members are read together, so the
    resolve at the members adds no request."""
    graph = _CountingGraph(obs)

    place, result = _settled_read(graph, year=year)

    assert place.roles.resolved_entities == {"observationAbout": bound}
    assert place.roles.where_dcids == bound
    assert place.probe_dcids == (None if len(bound) == 1 else tuple(bound))
    assert isinstance(result, Materialised)
    assert result.has_data is (year != "1990")
    assert graph.facet_reads == [["africa", _GHA, _KEN, _TZA]]


def test_a_spec_with_no_set_to_decide_is_read_once_at_its_places() -> None:
    graph = _CountingGraph({f"{_SERIES}|{_KEN}": [_facet(_COUNTRY_YEAR)]})

    place, _result = _settled_read(graph, year=None, sets=[_KENYA])

    assert place.roles.where_dcids == [_KEN]
    assert graph.facet_reads == [[_KEN]]


def test_members_over_the_bound_are_refused_after_one_read_of_the_node(search_settings) -> None:
    """Africa's three members are over a bound of two. Its node is read alone, and where
    the node holds no data the Spec is refused without a read of the members."""
    search_settings(max_child_places=2)
    graph = _CountingGraph({f"{_SERIES}|{_KEN}": [_facet(_COUNTRY_YEAR)]})

    with pytest.raises(PlaceSetTooLargeError) as refusal:
        _settled_read(graph, year=None)

    assert (refusal.value.count, refusal.value.bound) == (3, 2)
    assert graph.facet_reads == [["africa"]]


def test_a_failed_facet_read_fails_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one facet read failing is the Spec's read failing, as it fails a Spec that
    names a country."""
    graph = _CountingGraph(_NODE_AND_COUNTRIES)

    def failing(*_args: object, **_kwargs: object) -> None:
        raise GraphInfraError("the graph did not answer")

    monkeypatch.setattr(graph, "observation_facets_batch", failing)

    with pytest.raises(GraphInfraError):
        _settled_read(graph, year=None)


def test_the_context_before_the_read_holds_each_undecided_set_at_its_members(
    search_settings,
) -> None:
    """Before the read, a set the read decides stands at its members and no bound applies
    to it, since its node may bind. A shape with no such set is settled there, and the
    bound applies."""
    search_settings(max_child_places=2)
    shape = _shape()
    undecided = UnsettledPlace(
        shape,
        PlaceAssignment(sets={"observationAbout": (_AFRICA,)}),
        admitted=frozenset({("observationAbout", "Continent"), ("observationAbout", "Country")}),
    )
    refused_node = UnsettledPlace(
        shape,
        PlaceAssignment(sets={"observationAbout": (_AFRICA,)}),
        admitted=frozenset({("observationAbout", "Country")}),
    )

    assert not undecided.decided
    assert undecided.unread().roles.where_dcids == [_GHA, _KEN, _TZA]
    assert undecided.admission_questions == {
        ("observationAbout", "Continent"),
        ("observationAbout", "Country"),
    }
    assert refused_node.decided
    with pytest.raises(PlaceSetTooLargeError):
        refused_node.unread()
