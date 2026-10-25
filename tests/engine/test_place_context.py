"""`place_context` derives one shape's probe places, default scope and place warnings.

`DefaultPresence.scope` turns one presence read into the default scope of any StatVars it
covered, and `read_default_presence` makes that read, adding only what a known read lacks.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, wait
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dc_sdmx import SdmxCapExceededError
from graph_generation import FakeGeneration, GenerationStamp
from materialise import place_scope
from materialise.place_scope import NO_DEFAULT_SCOPE, DefaultPlaceScope, DefaultPresence
from search.engine.departures import warning_key
from search.engine.errors import DefaultScopeIndexSourcedError, GraphReadFailure
from search.engine.place import default_place_scope_message, read_default_presence
from search.engine.region import place_context as place_context_module
from search.engine.region import resolve_spec_resubmit
from search.engine.region.codes import (
    DIRECTION_DISCARDED,
    PLACE_SCOPE_DEFAULTED,
)
from search.engine.region.observations.bulk_scope import bulk_scope_for
from search.engine.region.phase_place_materialise import (
    _await_alternates_presence,
)
from search.engine.region.place_context import (
    PlaceContext,
    UnsettledPlace,
    default_scope_stat_vars,
    place_context,
    settle_place,
    stamp_offered_place_warnings,
)
from search.engine.region.place_roles import PlaceAssignment, PlaceRoles
from search.engine.retrieval.retrieve import Materialised
from search.engine.shape import ShapeDraft
from search.heading.departure_fact import DepartureFact
from search.models import Warning
from tests.engine import test_first_answer_place_scenarios as scenarios
from tests.engine import test_resubmit_place_scope_default as placeless_resubmit
from tests.engine._builders import flow_place_roles, subject_place_roles
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_TZA = "country/TZA"
_GHA = "country/GHA"
_AFRICA = "africa"
_FEMALE = "Count_Death_Female"
_MALE = "Count_Death_Male"

# Earth's countries, and each StatVar's default scope: every place with data for
# `_FEMALE`, the countries fallback for `_MALE`, Earth alone, and no place.
_PRESENCE = DefaultPresence(
    countries=(_GHA, _KEN, _TZA),
    scopes={
        _FEMALE: DefaultPlaceScope(places=(_AFRICA, _TZA), every_place=True),
        _MALE: DefaultPlaceScope(places=(_KEN,), every_place=False),
        "Count_Death_World": DefaultPlaceScope(places=("Earth",), every_place=True),
        "Count_Death_None": NO_DEFAULT_SCOPE,
    },
)


@pytest.fixture(autouse=True)
def _fresh_scope_memo() -> Iterator[None]:
    """Each test reads availability afresh: the settled-scope memo is process-wide."""
    while flights := list(place_scope._flights.values()):
        wait(flights)
    place_scope.clear_memo()
    yield
    while flights := list(place_scope._flights.values()):
        wait(flights)
    place_scope.clear_memo()


def _served_context(shape, roles, *, presence):
    """`place_context` over every StatVar `shape` holds, the members a mint serves for it."""
    return place_context(shape, roles, stat_vars=list(shape.sv_arc_facts or {}), presence=presence)


def _shape(*stat_vars: str, declared_roles: tuple[str, ...] | None = None) -> ShapeDraft:
    return ShapeDraft(
        shape_id="death_count",
        label="Deaths",
        pop_type_dcid="MortalityEvent",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={sv: {} for sv in stat_vars},
        declared_roles=declared_roles,
    )


_SUBJECT = _shape(_FEMALE, _MALE)
_FLOW = _shape("Amount_Flow", declared_roles=("financingRecipient", "financingProvider"))


class TestDefaultPresenceScope:
    def test_a_union_with_a_countries_fallback_is_not_every_place(self) -> None:
        scope = _PRESENCE.union_scope([_FEMALE, _MALE])

        assert scope == DefaultPlaceScope(places=(_AFRICA, _KEN, _TZA), every_place=False)

    def test_a_union_of_every_place_scopes_is_every_place(self) -> None:
        scope = _PRESENCE.union_scope([_FEMALE, "Count_Death_World"])

        assert scope == DefaultPlaceScope(places=(_AFRICA, _TZA, "Earth"), every_place=True)

    def test_a_statvar_the_read_did_not_cover_raises(self) -> None:
        with pytest.raises(AssertionError, match="Count_Death_Unread"):
            _PRESENCE.scope("Count_Death_Unread")


class _PresenceSdmx:
    """Availability that lists `_FEMALE`'s places and `"Count_Death_None"`'s none, and holds
    more places than the cap for `_MALE`, which so falls back to its countries. `reads`
    records each read's StatVar."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, Any] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        (stat_var,) = variables
        self.reads.append(stat_var)
        if stat_var == _MALE:
            assert max_values is not None
            raise SdmxCapExceededError(
                "too many places", cap_kind="values", cap=max_values, observed=max_values + 1
            )
        return [_TZA, _AFRICA] if stat_var == _FEMALE else []


class _PresenceGraph:
    """Knows every place, lists Earth's `countries`, and finds `_MALE`'s admitted data in
    Kenya. `calls` records each read's kind."""

    def __init__(self, countries: tuple[str, ...] = _PRESENCE.countries) -> None:
        self.calls: list[str] = []
        self._countries = countries
        self._generation = FakeGeneration(token="r1")

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def un_availability(self) -> None:
        return None

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        self.calls.append("node_arcs_many")
        return {dcid: {"typeOf": ("Place",)} for dcid in dcids}

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict:
        self.calls.append("child_places_batch")
        return {"Earth": list(self._countries)}

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        self.calls.append("entities_with_data_by_variable")
        return {sv: [_KEN] for sv in variable_dcids if sv == _MALE}


def _read(graph: _PresenceGraph, sdmx: _PresenceSdmx, stat_vars, **kwargs) -> DefaultPresence:
    return read_default_presence(
        graph=cast(Any, graph), sdmx=cast(Any, sdmx), stat_vars=stat_vars, **kwargs
    )


class TestReadDefaultPresence:
    def test_every_place_from_availability_and_countries_past_its_cap(self) -> None:
        graph, sdmx = _PresenceGraph(), _PresenceSdmx()

        presence = _read(graph, sdmx, [_FEMALE, _MALE])

        assert sorted(sdmx.reads) == [_FEMALE, _MALE]
        assert presence.scope(_FEMALE) == _PRESENCE.scope(_FEMALE)
        assert presence.scope(_MALE) == _PRESENCE.scope(_MALE)
        assert presence.unsettled == frozenset()

    def test_an_earth_listing_naming_no_country_refuses_as_a_graph_failure(self) -> None:
        """Search refuses an Earth listing that names no country as it refuses any failed
        graph read, so Earth's total is never served in place of every country."""
        graph = _PresenceGraph(countries=())

        with pytest.raises(GraphReadFailure, match="named no 'Country'"):
            _read(graph, _PresenceSdmx(), [_MALE])

    def test_a_known_read_is_topped_up_with_only_what_it_lacks(self) -> None:
        graph, sdmx = _PresenceGraph(), _PresenceSdmx()
        known = _read(graph, sdmx, [_FEMALE])
        sdmx.reads.clear()

        presence = _read(graph, sdmx, [_FEMALE, "Count_Death_None"], known=known)

        assert sdmx.reads == ["Count_Death_None"]
        assert presence.scope("Count_Death_None") == NO_DEFAULT_SCOPE
        assert presence.union_scope([_FEMALE, "Count_Death_None"]).places == (_AFRICA, _TZA)

    def test_nothing_missing_makes_no_request(self) -> None:
        graph, sdmx = _PresenceGraph(), _PresenceSdmx()
        known = _read(graph, sdmx, [_FEMALE])
        graph.calls.clear()
        sdmx.reads.clear()

        assert _read(graph, sdmx, [_FEMALE], known=known) is known
        assert _read(graph, sdmx, []).scopes == {}
        assert graph.calls == []
        assert sdmx.reads == []


class TestPlaceContext:
    def test_a_placeless_subject_shape_takes_the_default_scope_of_its_own_statvars(
        self,
    ) -> None:
        context = _served_context(
            _shape(_FEMALE),
            subject_place_roles(),
            presence=_PRESENCE,
        )

        assert context.probe_dcids == (_AFRICA, _TZA)
        assert context.mode == "every_place"
        assert context.default_scope == _PRESENCE.scope(_FEMALE)
        assert [(w.code, w.severity) for w in context.warnings] == [(PLACE_SCOPE_DEFAULTED, "warn")]
        assert context.where is None

    def test_a_countries_fallback_says_so(self) -> None:
        context = _served_context(
            _shape(_MALE),
            subject_place_roles(),
            presence=_PRESENCE,
        )

        assert context.probe_dcids == (_KEN,)
        assert context.mode == "countries_fallback"
        assert context.default_scope == _PRESENCE.scope(_MALE)
        assert [w.message for w in context.warnings] == [
            default_place_scope_message(_PRESENCE.scope(_MALE))
        ]

    def test_a_presence_read_covering_other_shapes_leaves_the_scope_to_its_own_statvars(
        self,
    ) -> None:
        """The presence read holds other shapes' StatVars with data at several places; a
        shape with no data of its own still takes an empty scope."""
        context = _served_context(
            _shape("Count_Death_None"),
            subject_place_roles(),
            presence=_PRESENCE,
        )

        assert context.probe_dcids == ()
        assert context.default_scope == NO_DEFAULT_SCOPE

    def test_a_worldwide_default_scope_says_so(self) -> None:
        context = _served_context(
            _shape("Count_Death_World"),
            subject_place_roles(),
            presence=_PRESENCE,
        )

        assert context.probe_dcids == ("Earth",)
        assert context.default_scope == _PRESENCE.scope("Count_Death_World")
        assert [w.message for w in context.warnings] == [
            "no place was named; the answer was resolved as a worldwide total."
        ]

    def test_a_default_scope_without_data_is_empty(self) -> None:
        context = _served_context(
            _shape("Count_Death_None"),
            subject_place_roles(),
            presence=_PRESENCE,
        )

        assert context.probe_dcids == ()
        assert context.default_scope == NO_DEFAULT_SCOPE
        assert [(w.code, w.severity) for w in context.warnings] == [(PLACE_SCOPE_DEFAULTED, "warn")]

    def test_a_placeless_subject_shape_without_presence_raises(self) -> None:
        with pytest.raises(AssertionError, match="no default-scope presence was read"):
            _served_context(_SUBJECT, subject_place_roles(), presence=None)

    def test_a_placeless_role_bearing_shape_takes_no_default_scope(self) -> None:
        context = _served_context(
            _FLOW,
            flow_place_roles(_FLOW, origin=[_GHA]),
            presence=_PRESENCE,
        )

        assert context.probe_dcids is None
        assert context.mode == "bound"
        assert context.default_scope is None
        assert context.warnings == ()

    @pytest.mark.parametrize("place", [_KEN, "africa"])
    def test_one_place_is_probed_at_itself(self, place: str) -> None:
        context = _served_context(_SUBJECT, subject_place_roles(place), presence=None)

        assert context.probe_dcids is None
        assert context.probe_places is None
        assert context.where is not None
        assert (context.where.kind, context.where.value_dcids) == ("value", [place])

    def test_several_places_are_probed_at_themselves(self) -> None:
        context = _served_context(
            _SUBJECT,
            subject_place_roles("africa", _GHA, "eastafrica"),
            presence=None,
        )

        assert context.probe_dcids == ("africa", _GHA, "eastafrica")
        assert context.warnings == ()
        assert context.where is not None
        assert context.where.kind == "set"

    def test_a_default_scope_is_outside_the_bound(self, search_settings) -> None:
        """A default scope binds no dcid, so its places do not count against the bound."""
        search_settings(max_child_places=1)
        shape = _shape(_FEMALE)
        context = UnsettledPlace(
            shape, PlaceAssignment(), stat_vars=tuple(shape.sv_arc_facts or {}), presence=_PRESENCE
        ).unread()

        assert context.probe_dcids == (_AFRICA, _TZA)

    def test_dropped_directions_warn_in_query_order(self) -> None:
        roles = PlaceRoles(
            **{
                **vars(subject_place_roles(_KEN, _GHA, _TZA)),
                "discarded_directions": (("Kenya", "destination"), ("Ghana", "origin")),
            }
        )

        context = _served_context(_SUBJECT, roles, presence=None)

        assert [w.code for w in context.warnings] == [DIRECTION_DISCARDED, DIRECTION_DISCARDED]
        assert context.warnings[0].message.startswith("Kenya was named as the ")
        assert context.warnings[1].message.startswith("Ghana was named as the ")
        assert {w.severity for w in context.warnings} == {"warn"}
        assert context.departure_facts == {
            warning_key(context.warnings[0]): DepartureFact(
                code="direction_discarded",
                binding_ref="where:",
                asked="Kenya",
                over="destination",
            ),
            warning_key(context.warnings[1]): DepartureFact(
                code="direction_discarded",
                binding_ref="where:",
                asked="Ghana",
                over="origin",
            ),
        }

    def test_an_inferred_direction_keeps_its_warning_and_writes_no_departure(self) -> None:
        """weo-037: the extractor read Ghana as a recipient the request never named, so
        the dropped direction is disclosed by its warning alone."""
        roles = PlaceRoles(
            **{
                **vars(subject_place_roles(_KEN, _GHA)),
                "discarded_directions": (("Kenya", "destination"), ("Ghana", "destination")),
                "inferred_directions": frozenset({("Ghana", "destination")}),
            }
        )

        context = _served_context(_SUBJECT, roles, presence=None)

        assert [w.code for w in context.warnings] == [DIRECTION_DISCARDED, DIRECTION_DISCARDED]
        assert [fact.asked for fact in context.departure_facts.values()] == ["Kenya"]


class TestSettled:
    def test_a_discovered_scope_replaces_the_probe_places_with_its_members(self) -> None:
        context = _served_context(
            _FLOW,
            flow_place_roles(_FLOW, origin=[_GHA]),
            presence=None,
        )
        # The plan serves a subset of the discovered members; the probes bind them all.
        mat = cast(
            Materialised,
            SimpleNamespace(
                discovered_scope=SimpleNamespace(member_dcids=[_KEN, _TZA]),
                scope_member_dcids=[_KEN],
                open_axes=frozenset(),
            ),
        )

        settled = context.settled(mat)

        assert settled.probe_dcids == (_KEN, _TZA)
        assert settled.mode == "discovered"
        assert settled.default_scope is None
        assert settled.roles is context.roles
        assert settled.warnings == context.warnings

    def test_without_a_discovered_or_default_scope_the_context_stands(self) -> None:
        context = _served_context(
            _SUBJECT,
            subject_place_roles(),
            presence=_PRESENCE,
        )
        mat = cast(
            Materialised,
            SimpleNamespace(
                discovered_scope=None,
                default_scope=None,
                scope_member_dcids=[_KEN],
                open_axes=frozenset(),
            ),
        )

        assert context.settled(mat) is context

    @pytest.mark.parametrize(
        ("served", "settled_open"),
        [
            pytest.param(frozenset({"origin"}), frozenset({"origin"}), id="served"),
            pytest.param(frozenset(), frozenset(), id="unserved"),
        ],
    )
    def test_the_open_axes_narrow_to_those_the_elected_members_serve(
        self, served: frozenset[str], settled_open: frozenset[str]
    ) -> None:
        roles = dataclasses.replace(
            flow_place_roles(_FLOW, destination=[_KEN]), open_axes=frozenset({"origin"})
        )
        context = _served_context(_FLOW, roles, presence=None)
        mat = cast(
            Materialised,
            SimpleNamespace(discovered_scope=None, default_scope=None, open_axes=served),
        )

        assert context.settled(mat).roles.open_axes == settled_open

    @pytest.mark.parametrize(
        "served",
        [
            pytest.param(_PRESENCE.scope(_FEMALE), id="every_place"),
            pytest.param(_PRESENCE.scope(_MALE), id="countries_fallback"),
            pytest.param(_PRESENCE.scope("Count_Death_World"), id="earth"),
        ],
    )
    def test_a_settled_default_scope_replaces_the_probe_places_and_the_warning(
        self, served: DefaultPlaceScope
    ) -> None:
        """The shape's default scope spans both members' places; the answer serves one
        member, and its scope and warning become that member's."""
        context = _served_context(
            _SUBJECT,
            subject_place_roles(),
            presence=_PRESENCE,
        )
        mat = cast(
            Materialised,
            SimpleNamespace(discovered_scope=None, default_scope=served, open_axes=frozenset()),
        )

        settled = context.settled(mat)

        assert context.probe_dcids == (_AFRICA, _KEN, _TZA)
        assert settled.probe_dcids == served.places
        assert settled.default_scope == served
        assert [(w.code, w.severity, w.message) for w in settled.warnings] == [
            (PLACE_SCOPE_DEFAULTED, "warn", default_place_scope_message(served))
        ]
        assert settled.roles is context.roles


class TestSettlePlace:
    def _context(self) -> PlaceContext:
        return _served_context(
            _SUBJECT,
            subject_place_roles(),
            presence=_PRESENCE,
        )

    def test_a_changed_scope_warning_is_replaced_where_it_stands(self) -> None:
        context = self._context()
        first = Warning(code="a", severity="info", message="a")
        last = Warning(code="z", severity="info", message="z")
        warnings = [first, *context.warnings, last]
        served = _PRESENCE.scope("Count_Death_World")
        mat = cast(
            Materialised,
            SimpleNamespace(discovered_scope=None, default_scope=served, open_axes=frozenset()),
        )

        settled = settle_place(context, mat, warnings)

        assert warnings == [first, *settled.warnings, last]
        assert warnings[1].message.endswith("as a worldwide total.")

    def test_an_unchanged_scope_warning_keeps_its_object(self) -> None:
        context = self._context()
        warnings = [*context.warnings]
        served = _PRESENCE.scope(_MALE)
        mat = cast(
            Materialised,
            SimpleNamespace(discovered_scope=None, default_scope=served, open_axes=frozenset()),
        )

        settled = settle_place(context, mat, warnings)

        assert settled.probe_dcids == (_KEN,)
        assert warnings[0] is context.scope_warning


class TestDefaultScopeStatVars:
    def test_subject_only_shapes_in_order_without_role_bearing_ones(self) -> None:
        assert default_scope_stat_vars(
            [_shape(_FEMALE), _FLOW, _shape(_MALE, _FEMALE), _shape("Count_Death_World")]
        ) == [_FEMALE, _MALE, "Count_Death_World"]

    def test_an_index_sourced_subject_only_shape_refuses(self, monkeypatch) -> None:
        """A subject-only shape whose population the code index sources is probed over
        members the presence read never covers, so choosing the read's StatVars refuses."""
        monkeypatch.setattr(
            place_context_module,
            "shape_member_facts",
            lambda shape, bound_values: ({}, shape.shape_id == "indexed"),
        )
        indexed = dataclasses.replace(_shape(_FEMALE), shape_id="indexed")

        assert default_scope_stat_vars([_shape(_MALE)]) == [_MALE]
        with pytest.raises(DefaultScopeIndexSourcedError, match="indexed"):
            default_scope_stat_vars([_shape(_MALE), indexed])

    def test_a_placeless_query_electing_an_index_sourced_subject_shape_refuses(
        self, monkeypatch
    ) -> None:
        """The refusal reaches the caller of a first answer before any presence read."""
        monkeypatch.setattr(
            place_context_module, "shape_member_facts", lambda shape, bound_values: ({}, True)
        )
        scenario = scenarios._SCENARIOS["placeless-definite-countries"]

        with pytest.raises(DefaultScopeIndexSourcedError):
            scenarios._resolve(scenario)

    def test_a_placeless_resubmit_posting_an_index_sourced_subject_shape_refuses(
        self, monkeypatch
    ) -> None:
        """The refusal reaches the caller of a resubmit before any presence read. A resubmit
        derives its shape from its elected members' arcs, so the shape is reported
        index-sourced here by patching."""
        monkeypatch.setattr(
            place_context_module, "shape_member_facts", lambda shape, bound_values: ({}, True)
        )
        graph = FakeGraph(
            nodes=placeless_resubmit._NODES, obs={}, children=placeless_resubmit._EARTH_CHILDREN
        )
        presence_reads: list[Any] = []
        monkeypatch.setattr(
            graph, "entities_with_data_by_variable", lambda *args: presence_reads.append(args)
        )

        with pytest.raises(DefaultScopeIndexSourcedError):
            resolve_spec_resubmit(
                inp=placeless_resubmit._entity_less_input(),
                graph=graph,
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=walk_options(graph),
            )
        assert presence_reads == []


class TestStampOfferedPlaceWarnings:
    def test_an_offered_spec_carries_the_scope_warning_the_shape_in_play_lacks(self) -> None:
        """The shape in play defaults to every place with data. A spec read worldwide carries
        its own scope warning stamped with its id, and one read over the same kind of scope
        adds nothing."""
        in_play = _served_context(
            _shape(_FEMALE),
            subject_place_roles(),
            presence=_PRESENCE,
        )
        worldwide = _served_context(
            _shape("Count_Death_World"),
            subject_place_roles(),
            presence=_PRESENCE,
        )
        warnings = list(in_play.warnings)

        stamp_offered_place_warnings(
            warnings, [("spec_a", in_play), ("spec_b", worldwide)], departure_facts={}
        )

        assert [(w.code, w.spec_id) for w in warnings] == [
            (PLACE_SCOPE_DEFAULTED, None),
            (PLACE_SCOPE_DEFAULTED, "spec_b"),
        ]
        assert warnings[1].message == worldwide.scope_warning.message
        assert warnings[1].message != warnings[0].message


class TestBulkScopeFor:
    @pytest.mark.parametrize(
        ("stat_var", "place", "facets_by_entity", "bulk"),
        [
            pytest.param(_FEMALE, None, {}, True, id="every_place"),
            pytest.param(_MALE, None, {}, True, id="countries_fallback"),
            pytest.param(_FEMALE, None, None, False, id="one_place_probe"),
            pytest.param("Count_Death_World", None, {}, False, id="earth"),
            pytest.param("Count_Death_None", None, {}, False, id="empty"),
            pytest.param(_MALE, _KEN, {}, False, id="bound"),
        ],
    )
    def test_a_plural_default_scope_of_either_kind_reads_in_bulk(
        self, stat_var: str, place: str | None, facets_by_entity: dict | None, bulk: bool
    ) -> None:
        context = _served_context(
            _shape(stat_var),
            subject_place_roles(*([place] if place else [])),
            presence=_PRESENCE,
        )
        sdmx = cast(Any, object())
        mat = cast(Materialised, SimpleNamespace(facets_by_entity=facets_by_entity))

        result = bulk_scope_for(context, mat, sdmx=sdmx)

        assert (result is not None) is bulk
        if result is not None:
            assert result.places == context.probe_dcids
            assert result.sdmx is sdmx


def test_the_alternates_presence_read_is_merged_into_the_requests_presence() -> None:
    """The alternates' read was made over the presence known when it was submitted. A read
    made since, a re-election's, added `_MALE`, and the alternates' read adds `_FEMALE`, so
    the request's presence covers both and keeps the countries both name."""
    countries = _PRESENCE.countries
    alternates_read: Future[DefaultPresence] = Future()
    alternates_read.set_result(
        DefaultPresence(countries=countries, scopes={_FEMALE: _PRESENCE.scope(_FEMALE)})
    )
    state = SimpleNamespace(
        default_presence=DefaultPresence(
            countries=countries, scopes={_MALE: _PRESENCE.scope(_MALE)}
        )
    )

    reason = asyncio.run(_await_alternates_presence(cast(Any, state), alternates_read))

    assert reason is None
    assert state.default_presence.countries == countries
    assert dict(state.default_presence.scopes) == {
        _MALE: _PRESENCE.scope(_MALE),
        _FEMALE: _PRESENCE.scope(_FEMALE),
    }
