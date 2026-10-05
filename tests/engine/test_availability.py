"""Test STANDARD availability, confidence, member probes, and no-data matches.

The cases cover named-place data, sibling-member data, nearest-real suggestions,
and entity-less confidence grading (#305).
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.config import settings
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import standard_resolve
from search.engine.shape import ShapeDraft
from search.models import RawTextInput, ResolveRequest, ResubmitIdentityPayload
from tests.engine._base_identity import make_base, make_identity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
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
        assert flagged.resolved.stat_vars.items, (
            "flagged_match must carry the matched-but-unavailable SV"
        )
        assert inner.no_data.gate == NO_MEMBER_HAS_DATA_AT_PLACE
        assert inner.rendered_sentence == "No data for Nauru's teachers."
        # On this gate elected_stat_vars and flagged_match name the same
        # variable, since no_member_has_data_at_place has no separate
        # alternate to name.
        assert inner.no_data.elected_stat_vars is not None
        assert [ref.dcid for ref in inner.no_data.elected_stat_vars] == [
            sv.ref for sv in flagged.resolved.stat_vars.items
        ]


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
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="gender",
                        kind="value",
                        value_dcids=["Female"],
                    ),
                ],
                also_plausible=None,
                ask=None,
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
                base_steps=[],
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
        assert [sv.ref for sv in flagged.resolved.stat_vars.items] == [self._FEMALE_SV]
        assert result.nearest_real
        assert [nr.relaxation.axis for nr in result.nearest_real] == ["how"]
        assert [sv.ref for sv in result.nearest_real[0].resolved.stat_vars.items] == [
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
                base_steps=[],
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
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=[],
            ask=None,
        ), None


class TestEntityLessDefiniteConfidenceFreshPath:
    """Entity-less definite results grade confidence from the representative score (#305)."""

    _QUERY = "test no entity incidence rate strong"

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
                base_steps=[],
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "definite"
        spec = result.specs[0]
        assert spec.resolved.available_at_place is None
        assert spec.resolved.confidence == "high"

    def test_weak_match_grades_medium_not_low(self):
        """Without a named place, a below-threshold score grades medium."""
        query = "test no entity incidence rate weak"
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
                base_steps=[],
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "definite"
        spec = result.specs[0]
        assert spec.resolved.available_at_place is None
        assert spec.resolved.confidence == "medium"


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
        base_spec_id, base_identity = make_base(self._SHAPE, slots=[])
        inp = ResubmitIdentityPayload(
            identity=make_identity(self._SHAPE, []),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_NO_ENTITY_SV],
        )

        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )

        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.resolved.available_at_place is None
        assert spec.resolved.confidence is not None
