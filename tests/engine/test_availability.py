"""Test STANDARD availability, confidence, member probes, and no-data matches.

The cases cover named-place data, sibling-member data, nearest-real suggestions,
and entity-less confidence grading (#305).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import ClassVar, cast

import pytest

from dfo_platform.request_deadline import bounded_deadline
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.config import settings
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphCapExceededError, GraphInfraError
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.reading_coverage import observed_years
from search.engine.region import availability, resolve_spec_resubmit, resolve_variable
from search.engine.region.availability import (
    _subject_has_data,
    has_data_where_asked,
    menu_read_places,
    read_coverage,
    read_menu_availability,
)
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import standard_resolve
from search.engine.shape import ShapeDraft
from search.models import RawTextInput, ResolveRequest, ResubmitIdentityPayload, TimeWindow
from tests.engine._base_identity import make_identity
from tests.engine._binders import elected_reply
from tests.engine._fail_open_deadline import (
    FAIL_OPEN_REQUESTS,
    assert_fail_open_deadline,
    request_deadline,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve, walk_options
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_admission import _pruning_state, _set
from tests.fixtures import FakeGraph, make_resolver

_RECIPIENT = "country/XYZ"


def _synthetic_shape(shape_id: str, *, members: list[str], score: float = 0.9) -> ShapeDraft:
    """Build a STANDARD shape whose member keys drive the probe."""
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={m: {} for m in members},
        representative_score=score,
    )


def _obs_graph(entries: dict[tuple[str, str], int]) -> FakeGraph:
    """Build one observation fixture per pair.

    A zero count means confirmed without facets.
    """
    obs = {}
    for (sv, entity), count in entries.items():
        key = f"{sv}|{entity}"
        obs[key] = (
            [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": count,
                    "provenanceId": "dc/base/AvailabilityFixture",
                }
            ]
            if count
            else []
        )
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


# (a) standard with data -> available_at_place=True, confidence set, no flagged_match


class TestAvailableStandardDefinite:
    def test_total_population_india_is_available_high_confidence(self):
        result = offline_resolve(ResolveRequest(input=RawTextInput(query="total population India")))
        inner = result.parts[0]
        assert inner.status == "definite"
        spec = inner.spec
        assert spec.resolved.available_at_place is True
        assert spec.resolved.confidence in ("high", "medium")


# (b) standard best SV exists but no obs at place -> no_data + flagged_match


class TestFlaggedUnavailableStandard:
    def test_no_member_has_observations_yields_flagged_carrier(self):
        shape = _synthetic_shape("shape_b", members=["sv_b0", "sv_b1"])
        graph = _obs_graph({("sv_b0", _RECIPIENT): 0, ("sv_b1", _RECIPIENT): 0})

        result = standard_resolve(
            shape=shape,
            bindings=[],
            entities={"observationAbout": [_RECIPIENT]},
            graph=graph,
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False
        # Falls back to the representative (first member) for ordering.
        assert result.sv_dcids == ["sv_b0"]

    def test_teacher_count_nauru_end_to_end_flagged_match(self):
        """Nauru has no observations for any Teacher/count member."""
        result = offline_resolve(ResolveRequest(input=RawTextInput(query="teacher count in Nauru")))
        inner = result.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.reason == "no_observations"
        flagged = inner.no_data.flagged_match
        assert flagged is not None
        assert flagged.resolved.available_at_place is False
        assert flagged.resolved.confidence == "low"
        assert flagged.resolved.members, "flagged_match must carry the matched-but-unavailable SV"
        assert inner.no_data.gate == NO_MEMBER_HAS_DATA_AT_PLACE
        assert inner.rendered_sentence == "No data for Nauru's teachers."
        # On this gate elected_count and flagged_match count the same
        # variable, since no_member_has_data_at_place has no separate
        # alternate to name.
        assert inner.no_data.elected_count == len(flagged.resolved.members)


# (c) A data-bearing sibling answers when the representative is empty.


class TestMemberProbeSiblingHasData:
    def test_sibling_member_wins_when_representative_has_no_observations(self):
        shape = _synthetic_shape("shape_c", members=["sv_c0", "sv_c1"])
        graph = _obs_graph({("sv_c0", _RECIPIENT): 0, ("sv_c1", _RECIPIENT): 3})

        result = standard_resolve(
            shape=shape,
            bindings=[],
            entities={"observationAbout": [_RECIPIENT]},
            graph=graph,
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.coverage.has_data is True
        # The data-bearing sibling is the single confirmed StatVar.
        assert result.sv_dcids == ["sv_c1"]


# (d) flagged_match is populated independently of nearest_real


class TestFlaggedMatchIndependentOfNearestReal:
    def test_flagged_match_and_nearest_real_are_distinct_fields(self):
        result = offline_resolve(ResolveRequest(input=RawTextInput(query="teacher count in Nauru")))
        inner = result.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.flagged_match is not None
        # All Teacher/count members lack Nauru data, so relaxing a constraint finds no match.
        assert inner.no_data.nearest_real is None


# (e) member-probe branch also runs the nearest_real relaxation probe


def _gender_arcs(gender: str | None) -> dict:
    """Build Person/count arcs, with an optional gender constraint."""
    if gender is None:
        return {"constraintProperties": {"nodes": []}}
    return {
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": gender}]},
    }


class TestMemberProbeRunsNearestReal:
    """At a place with only aggregate data, flag the gendered member and suggest the aggregate."""

    _AGGREGATE_SV = "Count_Person"
    _FEMALE_SV = "Count_Person_Female"
    _MALE_SV = "Count_Person_Male"
    _RECIPIENT = "country/ZZZ"
    _DETECT_QUERY = "population"

    class _GenderBoundLLM:
        """Elects the sole shape and binds gender directly to Female."""

        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            assert schema.__name__ == "_UniversalBindOutput"
            return elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="gender",
                        kind="value",
                        value_dcids=["Female"],
                    ),
                ],
                also_plausible=None,
            ), None

    @staticmethod
    def _population_gender_arcs(gender: str | None) -> dict:
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        arcs.update(_gender_arcs(gender))
        return arcs

    @classmethod
    def _graph(cls, *, obs: dict | None = None) -> FakeGraph:
        nodes = {
            "gender": {"label": "Gender"},
            "Female": {"label": "Female"},
            "Male": {"label": "Male"},
            cls._AGGREGATE_SV: {
                "label": "Population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs(None),
            },
            cls._FEMALE_SV: {
                "label": "Female population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs("Female"),
            },
            cls._MALE_SV: {
                "label": "Male population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs("Male"),
            },
            cls._RECIPIENT: {"label": "Zeta", "type": "Country"},
        }
        obs = (
            obs
            if obs is not None
            else {f"{cls._AGGREGATE_SV}|{cls._RECIPIENT}": [{"obsCount": 137}]}
        )
        return FakeGraph(
            nodes=nodes,
            obs=obs,
            detect={cls._DETECT_QUERY: {"svs": [cls._AGGREGATE_SV, cls._FEMALE_SV, cls._MALE_SV]}},
            resolve={"Zeta": cls._RECIPIENT},
        )

    def test_gendered_population_flags_member_and_finds_nearest_real(self):
        result = asyncio.run(
            resolve_variable(
                "female population of zeta",
                entities=["Zeta"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="female population of zeta",
                pac=True,
                graph=self._graph(),
                resolver=make_resolver(self._graph()),
                llm=self._GenderBoundLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "no_data"
        assert result.no_data_reason == "no_observations"
        # Only the member-probe branch (place availability false) sets flagged_match.
        # The classic no_observations path does not.
        assert result.flagged_match is not None
        assert len(result.flagged_match) == 1
        flagged = result.flagged_match[0]
        assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE
        assert [sv.ref.ref for sv in flagged.resolved.members] == [self._FEMALE_SV]
        assert result.nearest_real
        assert [nr.relaxation.axis for nr in result.nearest_real] == ["how"]
        assert [sv.ref.ref for sv in result.nearest_real[0].resolved.members] == [
            self._AGGREGATE_SV
        ]

    def test_usa_only_data_is_not_claimed_for_requested_place(self):
        """Nearest-real suggestions require data at Zeta."""
        graph = self._graph(
            obs={
                f"{self._AGGREGATE_SV}|country/USA": [
                    {"earliestDate": "2020", "latestDate": "2020", "obsCount": 9}
                ]
            }
        )

        result = asyncio.run(
            resolve_variable(
                "female population of zeta",
                entities=["Zeta"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="female population of zeta",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=self._GenderBoundLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(
                    observations={}, availability={}, scope_availability=graph.scope_availability
                ),
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "no_data"
        assert result.no_data_reason == "no_observations"
        assert result.flagged_match is not None
        assert result.nearest_real is None


# (f) no place named, default-scope data -> definite, confidence graded

_NO_ENTITY_POP_TYPE = "TestNoEntityIncidenceEvent"
_NO_ENTITY_MEAS_PROP = "testNoEntityIncidence"
_NO_ENTITY_STAT_TYPE = "measuredValue"
_NO_ENTITY_SV = "TEST/NoEntityIncidence_Bare"
_NO_ENTITY_COUNTRY = "country/ZZQ"


def _no_entity_sv_node(*, label: str) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": _NO_ENTITY_POP_TYPE}]},
            "measuredProperty": {"nodes": [{"dcid": _NO_ENTITY_MEAS_PROP}]},
            "statType": {"nodes": [{"dcid": _NO_ENTITY_STAT_TYPE}]},
            "constraintProperties": {"nodes": []},
        },
    }


def _no_entity_graph(*, query: str, cosine_score: float) -> FakeGraph:
    """Build an unconstrained SV with default-scope country data and a chosen score."""
    return FakeGraph(
        nodes={_NO_ENTITY_SV: _no_entity_sv_node(label="Test no-entity incidence rate")},
        detect={query: {"svs": [_NO_ENTITY_SV], "cosine_scores": [cosine_score]}},
        resolve={},
        obs={
            f"{_NO_ENTITY_SV}|{_NO_ENTITY_COUNTRY}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 5}
            ]
        },
        children={"Earth": {"Country": [_NO_ENTITY_COUNTRY]}},
    )


class _NoEntityBindLLM:
    """Elects the sole shape with no bindings, a plain, unqualified ask."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=[]), None


class TestEntityLessDefiniteConfidenceFreshPath:
    """Entity-less definite results grade confidence from the representative score (#305)."""

    _QUERY = "test no entity incidence rate"

    def test_strong_match_grades_high(self):
        graph = _no_entity_graph(query=self._QUERY, cosine_score=settings().weak_score_threshold)
        result = asyncio.run(
            resolve_variable(
                self._QUERY,
                entities=[],
                date_request=None,
                detect_query=self._QUERY,
                role_query=self._QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(
                    observations={}, availability={}, scope_availability=graph.scope_availability
                ),
                llm=_NoEntityBindLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "definite"
        spec = result.specs[0]
        assert spec.resolved.available_at_place is None
        assert spec.resolved.confidence == "high"

    def test_a_weak_match_that_is_the_only_reading_with_data_grades_high(self):
        """Without a named place, a below-threshold score on the only admissible reading
        with data grades high: the frame admitting one reading corroborates it, and scores
        reach confidence only beside a rival (status rule 5)."""
        query = "test no entity incidence rate"
        graph = _no_entity_graph(query=query, cosine_score=settings().weak_score_threshold - 0.1)
        result = asyncio.run(
            resolve_variable(
                query,
                entities=[],
                date_request=None,
                detect_query=query,
                role_query=query,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(
                    observations={}, availability={}, scope_availability=graph.scope_availability
                ),
                llm=_NoEntityBindLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "definite"
        spec = result.specs[0]
        assert spec.resolved.available_at_place is None
        assert spec.resolved.confidence == "high"


class TestEntityLessDefiniteConfidenceResubmitPath:
    """Plain entity-less resubmits also grade confidence from the representative score."""

    _SHAPE: ClassVar[dict] = {
        "kind": "statvar_schema",
        "population_type": _NO_ENTITY_POP_TYPE,
        "measured_property": _NO_ENTITY_MEAS_PROP,
        "stat_type": _NO_ENTITY_STAT_TYPE,
    }

    def test_resubmit_no_entity_definite_grades_confidence(self):
        graph = FakeGraph(
            nodes={
                _NO_ENTITY_SV: _no_entity_sv_node(label="Test resubmit no-entity incidence rate")
            },
            resolve={},
            obs={
                f"{_NO_ENTITY_SV}|{_NO_ENTITY_COUNTRY}": [
                    {"earliestDate": "2020", "latestDate": "2020", "obsCount": 5}
                ]
            },
            children={"Earth": {"Country": [_NO_ENTITY_COUNTRY]}},
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(self._SHAPE, []),
        )

        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )

        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.resolved.available_at_place is None
        assert spec.resolved.confidence is not None


# The menu availability read over a named set's places (`availability.read_menu_availability`)


_SET_MEMBERS = [f"country/S{i:02d}" for i in range(54)]
_SPAN_SV = "TESTFIXTURE/Availability/From2005"


def _span_graph(places: list[str]) -> FakeGraph:
    """One StatVar observed 2005-2020 at each of `places`."""
    facet = {"earliestDate": "2005", "latestDate": "2020", "obsCount": 16}
    return FakeGraph(nodes={}, obs={f"{_SPAN_SV}|{p}": [facet] for p in places})


class _BoundedGraph:
    """A graph whose facet read fails as a slow or oversized read does, keeping the bounds
    it was given."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.bounds: dict[str, object] = {}

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **bounds):
        self.bounds = bounds
        raise self.error


class _RecordingFacetGraph:
    """Records the deadline each facet read it serves is bound by, and answers none."""

    def __init__(self) -> None:
        self.deadlines: list[float | None] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **bounds):
        self.deadlines.append(bounded_deadline(bounds["deadline"]))
        return {sv: {entity: [] for entity in entities} for sv in stat_vars}


class TestMenuAvailabilityRead:
    @pytest.mark.parametrize(
        "error",
        [
            GraphInfraError("read past its deadline"),
            GraphCapExceededError("too big", cap_kind="bytes", cap=1, observed=2),
        ],
        ids=["slow", "oversized"],
    )
    def test_a_slow_or_oversized_read_leaves_availability_unknown(self, error):
        """The read of every set member is bounded by the request deadline and the byte
        cap; a read past either leaves D unknown, never absent, and is noted as lost. An
        absence derive found at the named country stays unknown too, since the set's other
        members went unread; it settles D where the country is the only place asked."""
        shape = dataclasses.replace(
            _synthetic_shape("from-2005", members=[_SPAN_SV]),
            place_available=False,
            member_coverage={_SPAN_SV: {}},
            availability_places=frozenset({_SET_MEMBERS[0]}),
        )
        graph = _BoundedGraph(error)
        requested = time.monotonic() + 1.0

        with recording_degraded_reads() as record:
            (read,), failed = read_menu_availability(
                [shape], graph=graph, places=_SET_MEMBERS, deadline=requested
            )

        assert graph.bounds == {"max_bytes": settings().sdmx_max_bytes, "deadline": requested}
        assert failed is True
        assert read.named_place_coverage is None
        assert _subject_has_data(read, None, asked=_SET_MEMBERS) is None
        assert _subject_has_data(read, None, asked=_SET_MEMBERS[:1]) is False
        # Data at the named country outside the requested years is no absence at the set.
        dates = DateRequest(window=TimeWindow(start_year=1990, end_year=1995), latest=False)
        facets = _span_graph(_SET_MEMBERS[:1]).observation_facets(
            stat_var=_SPAN_SV, entity=_SET_MEMBERS[0]
        )
        outside = dataclasses.replace(
            read,
            place_available=True,
            member_coverage={_SPAN_SV: {_SET_MEMBERS[0]: observed_years(facets)}},
        )
        assert _subject_has_data(outside, dates, asked=_SET_MEMBERS) is None
        assert _subject_has_data(outside, dates, asked=_SET_MEMBERS[:1]) is False
        assert record.reads == ["menu availability"]

    @FAIL_OPEN_REQUESTS
    def test_each_facet_read_runs_under_the_fail_open_budget(self, request_in_s, monkeypatch):
        """A read whose failure leaves availability unknown gets the fail-open budget, so a
        stalled graph holds the part for the budget at most. The chunks of a many-StatVar
        read share one budget, so the read as a whole holds the part no longer."""
        monkeypatch.setattr(availability, "_facet_reads", lambda *_args: [["a", "b"], ["c"]])
        graph = _RecordingFacetGraph()
        requested = request_deadline(request_in_s)
        started = time.monotonic()

        coverage = read_coverage(
            ["a", "b", "c"],
            graph=cast(EngineGraphClient, graph),
            places=_SET_MEMBERS,
            deadline=requested,
            read="test",
        )

        assert coverage is not None
        first, second = graph.deadlines
        assert_fail_open_deadline(first, requested=requested, started=started)
        assert second == first

    @pytest.mark.parametrize(
        ("start", "end", "has_data"),
        [(1990, 1995, False), (2010, 2015, True)],
        ids=["window-before-series", "window-inside-series"],
    )
    def test_a_set_place_reads_the_requested_years_as_one_country_does(self, start, end, has_data):
        """A 1990-1995 question on a series starting in 2005 has no data in range at a
        named set, as it has none at a named country."""
        dates = DateRequest(window=TimeWindow(start_year=start, end_year=end), latest=False)
        shape = _synthetic_shape("from-2005", members=[_SPAN_SV])
        (at_set,), failed = read_menu_availability(
            [shape], graph=_span_graph(_SET_MEMBERS), places=_SET_MEMBERS, deadline=None
        )
        facets = _span_graph(["country/GHA"]).observation_facets(
            stat_var=_SPAN_SV, entity="country/GHA"
        )
        at_country = dataclasses.replace(
            shape,
            place_available=True,
            member_coverage={_SPAN_SV: {"country/GHA": observed_years(facets)}},
            availability_places=frozenset({"country/GHA"}),
        )

        assert failed is False
        assert _subject_has_data(at_set, dates, asked=_SET_MEMBERS) is has_data
        assert _subject_has_data(at_country, dates, asked=["country/GHA"]) is has_data


def test_an_absence_read_at_the_named_country_alone_is_unknown_for_a_set_and_notes_nothing():
    """A part naming Kenya and Africa whose menu holds one shape makes no menu read over
    Africa's places (`phase_rank_bind`), though they are unread (`menu_read_places`). The
    shape keeps derive's absence at Kenya, which settles nothing at Africa: D is unknown,
    and since no read was made and lost, no degraded read is noted."""
    state = _pruning_state(
        {
            "Kenya": PlaceSet(
                name="Kenya", kind="place", node=PlaceNode(dcid="country/KEN", graph_type="Country")
            ),
            "Africa": _set("Africa", ("africa", "Continent"), "Country"),
        }
    )
    shape = dataclasses.replace(
        _synthetic_shape("absent-at-kenya", members=[_SPAN_SV]),
        place_available=False,
        member_coverage={_SPAN_SV: {}},
        availability_places=frozenset({"country/KEN"}),
    )

    with recording_degraded_reads() as record:
        has_data = has_data_where_asked(state, shape)

    assert menu_read_places(state.place_sets, ["country/KEN"]) != []
    assert has_data is None
    assert record.reads == []
