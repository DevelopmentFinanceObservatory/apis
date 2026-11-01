"""Test the default country scope for standard queries without a named place.

`resolve_variable` runs with a fixed shape election so these cases cover scope
selection and data checks, not shape ranking.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Coroutine
from typing import Any

import pytest

import dc_graph
import search.engine.region.candidate_collapse as candidate_collapse
from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from materialise import place_scope
from search.engine.bind import SlotBindingDraft
from search.engine.departures import spec_departure_facts
from search.engine.errors import GraphInfraError
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
    PLACE_SCOPE_DEFAULTED,
    RegionResult,
    resolve_variable,
)
from search.engine.region.codes import MEMBERSHIP_UNVERIFIED
from search.engine.region.state import EngineOptions
from search.engine.stated_words_read import StatedWordsOutput
from search.models import CoverageExact, TimeWindow
from search.render.plan import render_sentence_plan
from search.render.tokens import sentence_from_plan
from statvar_members import MemberEnumerator
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    materialise_double_from_obs,
    member_walk_reader,
    with_member_arcs,
)
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._scope_availability import ScopeAvailability
from tests.origin_catalogue.fake_catalogue import FakeCatalogue
from tests.origin_catalogue.support import build_arc_tables

_SV = "TEST/PlaceScopeDefault_Population"
_DETECT_QUERY = "test place scope population"
_EARTH_CHILDREN = {
    "Earth": {"Country": ["country/A1", "country/A2", "country/A3"]},
}
_NODES: dict = {
    _SV: {
        "label": "Test Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test place scope population"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV]}}


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=None), None


class _OpenSlotsBindLLM:
    """Elects the sole candidate shape and drafts each slot of `properties` unbound, as a
    reply that names no value of them does."""

    def __init__(self, *properties: str) -> None:
        self.properties = properties

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        assert schema.__name__ == "_UniversalBindOutput"
        drafts = [
            SlotBindingDraft(axis="how", property_dcid=prop, kind="unbound")
            for prop in self.properties
        ]
        return elected_reply(0, bindings=drafts, also_plausible=None), None


class _CountingFakeGraph(FakeGraph):
    """Count presence and observation probes, including requested entity IDs, and record the
    default scope's availability reads (`availability_reads`)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.presence_calls = 0
        self.probe_calls = 0
        self.presence_entity_dcids: list[list[str]] = []

    def entities_with_data_by_variable(self, variable_dcids, entity_dcids, *, deadline=None):
        self.presence_calls += 1
        self.presence_entity_dcids.append(list(entity_dcids))
        return super().entities_with_data_by_variable(
            variable_dcids, entity_dcids, deadline=deadline
        )

    @functools.cached_property
    def scope_availability(self) -> ScopeAvailability:
        return super().scope_availability

    @property
    def availability_reads(self) -> list[tuple[str, ...]]:
        return self.scope_availability.reads

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs):
        self.probe_calls += 1
        return super().observation_facets_batch(stat_vars, entities, needs_dates=needs_dates)


class _RaiseOnChildPlacesBatch(FakeGraph):
    """Simulate an Earth enumeration failure after shape binding."""

    def child_places_batch(
        self,
        parent_dcids,
        child_type,
        *,
        deadline: float | None = None,
    ):
        raise GraphInfraError("simulated Earth-enumeration transport failure")


# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_OBS_PROVENANCE = "dc/base/WorldDevelopmentIndicators"


def _resolve(
    *,
    graph: FakeGraph,
    resolver=None,
    llm=None,
    sdmx: FakeSdmxClient | None = None,
    materialise_graph=None,
    materialise_sdmx=None,
    include_observations: bool = False,
    date_request: DateRequest | None = None,
    deadline: float | None = None,
) -> Coroutine[Any, Any, RegionResult]:
    return resolve_variable(
        _DETECT_QUERY,
        entities=[],
        date_request=date_request,
        detect_query=_DETECT_QUERY,
        role_query=_DETECT_QUERY,
        pac=True,
        graph=graph,
        resolver=resolver if resolver is not None else make_resolver(graph),
        llm=llm if llm is not None else _FixedBindLLM(),
        base_timing={},
        sdmx=sdmx
        if sdmx is not None
        else FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            include_observations=include_observations,
            deadline=deadline,
        ),
    )


def _run(**kwargs) -> RegionResult:
    return asyncio.run(_resolve(**kwargs))


def _make_graph(*, obs: dict, children: dict | None = None, cls=FakeGraph) -> FakeGraph:
    return cls(
        nodes=_NODES,
        detect=_DETECT,
        resolve={},
        obs=obs,
        children=children if children is not None else _EARTH_CHILDREN,
    )


# ---------------------------------------------------------------------------
# (a) single shape, real data -> definite, CoverageExact, place_scope_defaulted
# ---------------------------------------------------------------------------


def test_single_shape_with_data_is_definite_with_real_country_count():
    """Two of three countries have data; coverage counts only those two."""
    obs = {
        f"{_SV}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
        f"{_SV}|country/A2": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
    }
    graph = _make_graph(obs=obs)
    result = _run(graph=graph)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.has_data is True
    assert spec.resolved.coverage.dimensions is not None
    places_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places_dim.count == 2  # A1 + A2 only; A3 never carried data

    defaulted = [w for w in result.warnings if w.code == PLACE_SCOPE_DEFAULTED]
    assert len(defaulted) == 1
    assert defaulted[0].message == _EVERY_PLACE_MESSAGE


# ---------------------------------------------------------------------------
# (i) no country carries data, but Earth itself does -> worldwide fallback
# ---------------------------------------------------------------------------


def test_earth_only_data_resolves_worldwide_via_per_pair_series():
    """Earth-only data resolves as a worldwide total through the per-pair path.

    Coverage has no place-count dimension, and the scope takes one availability read and no
    presence read."""
    obs = {
        f"{_SV}|Earth": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
                "observations": [{"date": "2020", "value": 42.0}],
            }
        ],
    }
    graph = _CountingFakeGraph(
        nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    sdmx = FakeSdmxClient(
        observations={}, availability={}, scope_availability=graph.scope_availability
    )
    result = _run(
        graph=graph,
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        include_observations=True,
    )

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.has_data is True
    assert spec.resolved.coverage.dimensions is not None
    assert all(d.label not in {"countries", "places"} for d in spec.resolved.coverage.dimensions)

    assert (graph.availability_reads, graph.presence_calls) == ([(_SV,)], 0)

    assert sdmx.subject_observations_calls == []
    assert spec.resolved.observations is not None
    assert len(spec.resolved.observations.series) == 1
    assert spec.resolved.observations.series[0].points

    defaulted = [w for w in result.warnings if w.code == PLACE_SCOPE_DEFAULTED]
    assert len(defaulted) == 1
    assert defaulted[0].message == (
        "no place was named; the answer was resolved as a worldwide total."
    )

    plan = render_sentence_plan(spec)
    assert "worldwide" in sentence_from_plan(plan)


# ---------------------------------------------------------------------------
# (ii) both Earth and countries carry data -> every place, Earth included
# ---------------------------------------------------------------------------


def test_earth_data_alongside_country_data_scopes_every_place_with_earth():
    """Availability lists both countries and Earth, so the scope is every place, Earth last,
    from one availability read and no presence read."""
    obs = {
        f"{_SV}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
        f"{_SV}|country/A2": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
        f"{_SV}|Earth": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 5,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
    }
    graph = _CountingFakeGraph(
        nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN
    )
    result = _run(graph=graph)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    places_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places_dim.count == 3
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == ["country/A1", "country/A2", "Earth"]

    assert (graph.availability_reads, graph.presence_calls) == ([(_SV,)], 0)

    defaulted = [w for w in result.warnings if w.code == PLACE_SCOPE_DEFAULTED]
    assert len(defaulted) == 1
    assert defaulted[0].message == _EVERY_PLACE_MESSAGE


# ---------------------------------------------------------------------------
# (b) multiple shapes -> candidates, each with real (probe-derived) has_data
# ---------------------------------------------------------------------------


class _CandidatesBindLLM:
    """Elect the first shape and mark the second as plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=[1]), None


_SV_A = "TEST/PlaceScopeDefault_Population"
_SV_B = "TEST/PlaceScopeDefault_Household"
_CANDIDATES_NODES: dict = {
    _SV_A: _NODES[_SV],
    # A genuinely distinct five-tuple (populationType differs) so this groups
    # into a SECOND ShapeDraft, not the same shape as _SV_A. Its label holds every word of
    # _SV_A's, so the query picks neither and the nomination is a rival (status rule 2(b)).
    _SV_B: {
        "label": "Test Household Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test place scope household population"}]},
            "populationType": {"nodes": [{"dcid": "Household"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}
_CANDIDATES_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.9, 0.85]}}


def test_multiple_shapes_yield_candidates_with_probe_derived_has_data():
    """Each candidate's `has_data` comes from its own observation probe.

    SV_A has data at A1; SV_B has none. Each StatVar's availability is read once, and no
    presence read runs.
    """
    obs = {
        f"{_SV_A}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
        # SV_B carries no observations anywhere in the default scope.
    }
    graph = _CountingFakeGraph(
        nodes=_CANDIDATES_NODES,
        detect=_CANDIDATES_DETECT,
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )
    result = _run(graph=graph, llm=_CandidatesBindLLM())

    assert result.status == "definite", (
        f"SV_B has no data anywhere in the scope, so it never survives "
        f"_materialise_alt_shapes' own has_data filter; got {result.status!r}"
    )
    assert sorted(graph.availability_reads) == sorted([(_SV_A,), (_SV_B,)])
    assert graph.presence_calls == 0
    spec = result.specs[0]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.has_data is True
    assert any(sv.ref.ref == _SV_A for sv in spec.resolved.members)


def test_multiple_shapes_both_with_real_data_yield_two_real_candidates():
    """Both shapes carry real data at distinct default-scope countries: both
    survive as candidates, each with its own real CoverageExact, and each StatVar's
    availability is read once."""
    obs = {
        f"{_SV_A}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
        f"{_SV_B}|country/A2": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
    }
    graph = _CountingFakeGraph(
        nodes=_CANDIDATES_NODES,
        detect=_CANDIDATES_DETECT,
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )
    result = _run(graph=graph, llm=_CandidatesBindLLM())

    assert result.status == "candidates", f"expected candidates, got {result.status!r}"
    assert len(result.specs) == 2
    assert sorted(graph.availability_reads) == sorted([(_SV_A,), (_SV_B,)])
    assert graph.presence_calls == 0
    for spec in result.specs:
        assert isinstance(spec.resolved.coverage, CoverageExact)
        assert spec.resolved.coverage.has_data is True


# ---------------------------------------------------------------------------
# (c) no country in the default scope has data -> no_observations
# ---------------------------------------------------------------------------


def test_no_country_has_data_reports_no_observations_flagged_match():
    """No place named, a confirmed shape, but no Earth-listed country carries
    data for it: honest no_data (genuine data gap), never a fabricated
    definite from SV existence."""
    graph = _make_graph(obs={})
    result = _run(graph=graph)

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None and len(result.flagged_match) == 1
    assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE
    assert any(w.code == PLACE_SCOPE_DEFAULTED for w in result.warnings)


# ---------------------------------------------------------------------------
# (d) no member asserts the bound property -> no_matching_slice, zero probes
# ---------------------------------------------------------------------------

_BOUND_SV = "TEST/PlaceScopeDefault_BareIncidence"
_BOUND_DETECT_QUERY = "test place scope bare incidence"
_BOUND_NODES: dict = {
    _BOUND_SV: {
        "label": "Test Bare Incidence",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test place scope bare incidence"}]},
            "populationType": {"nodes": [{"dcid": "TestIncidenceEvent"}]},
            "measuredProperty": {"nodes": [{"dcid": "testIncidence"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            # A bare aggregate: silent on testSex entirely (no
            # constraintProperties entry for it), so it cannot back a
            # testSex=Female claim -- fully_bound_members asks only whether
            # the property is ASSERTED at all, and this member asserts
            # nothing on that axis.
            "constraintProperties": {"nodes": []},
        },
    },
}
_BOUND_DETECT: dict = {_BOUND_DETECT_QUERY: {"svs": [_BOUND_SV]}}
_BOUND_FEMALE_SV = "TEST/PlaceScopeDefault_FemaleIncidence"
_BOUND_OLD_SV = "TEST/PlaceScopeDefault_OldIncidence"


def _bound_family_catalogue() -> FakeCatalogue:
    """Holds the bare aggregate and two unrecalled siblings, one asserting `testSex=Female`
    and one `testAge=Old`, so the shape recall builds from the aggregate has a `testSex` and
    a `testAge` slot and its family no member asserting both (`derive.held_family`)."""

    def target(dcid: str, *types: str) -> dict:
        return {"nodes": [{"dcid": dcid, "name": dcid, "types": list(types)}]}

    def arcs(name: str, *, asserts: tuple[str, str, str] | None = None) -> dict:
        constraint = {}
        if asserts is not None:
            prop, value, value_type = asserts
            constraint = {
                "constraintProperties": target(prop, "Property"),
                prop: target(value, value_type),
            }
        return {
            "typeOf": target("StatisticalVariable", "Class"),
            "name": {"nodes": [{"value": name}]},
            "populationType": target("TestIncidenceEvent", "Class"),
            "measuredProperty": target("testIncidence", "Property"),
            "statType": target("measuredValue", "Property"),
            "constraintProperties": {"nodes": []},
            **constraint,
        }

    return FakeCatalogue(
        arcs=build_arc_tables(
            dfo={},
            base={
                _BOUND_SV: arcs("Test place scope bare incidence"),
                _BOUND_FEMALE_SV: arcs(
                    "Test Female Incidence", asserts=("testSex", "Female", "TestSexEnum")
                ),
                _BOUND_OLD_SV: arcs(
                    "Test Old Incidence", asserts=("testAge", "Old", "TestAgeEnum")
                ),
            },
            schema={},
        )
    )


class _BindFemaleLLM:
    """Bind `testSex=Female`, which the bare aggregate does not assert."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        from search.engine.bind import SlotBindingDraft

        return elected_reply(
            0,
            bindings=[
                SlotBindingDraft(
                    axis="what",
                    property_dcid="testSex",
                    kind="value",
                    value_dcids=["Female"],
                ),
            ],
            also_plausible=None,
        ), None


class _BindFemaleOldLLM:
    """Bind `testSex=Female` and `testAge=Old`, which no member of the family asserts
    together."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        assert schema.__name__ == "_UniversalBindOutput"
        from search.engine.bind import SlotBindingDraft

        return elected_reply(
            0,
            bindings=[
                SlotBindingDraft(
                    axis="what", property_dcid="testSex", kind="value", value_dcids=["Female"]
                ),
                SlotBindingDraft(
                    axis="what", property_dcid="testAge", kind="value", value_dcids=["Old"]
                ),
            ],
            also_plausible=None,
        ), None


def test_no_member_asserts_bound_property_reports_no_matching_slice_zero_probes(
    bind_catalogue,
):
    """The `testSex` and `testAge` bindings, slots of the shape's family that no member
    asserts together, reject every member before any probe: the recalled aggregate asserts
    neither, and each sibling asserts one."""
    bind_catalogue(_bound_family_catalogue())
    obs = {
        f"{_BOUND_SV}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
    }
    graph = _CountingFakeGraph(
        nodes=_BOUND_NODES,
        detect=_BOUND_DETECT,
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )
    result = asyncio.run(
        resolve_variable(
            _BOUND_DETECT_QUERY,
            entities=[],
            date_request=None,
            detect_query=_BOUND_DETECT_QUERY,
            role_query=_BOUND_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindFemaleOldLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "no_matching_slice"
    assert result.flagged_match is not None and len(result.flagged_match) == 1
    assert result.gate == NO_MEMBER_ASSERTS_BOUND_PROPERTIES
    assert graph.probe_calls == 0  # the eligibility check short-circuits before any probe
    assert any(w.code == PLACE_SCOPE_DEFAULTED for w in result.warnings)


def test_a_value_bound_on_an_aggregate_without_its_slot_is_disclosed():
    """Recall builds only the bare aggregate, which has no `testSex` slot, and the bind
    model binds `testSex=Female` on it, which no word of the question names. The stray
    value is dropped and the aggregate answers, departing `measure_not_applied` under
    Female."""
    obs = {
        f"{_BOUND_SV}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            }
        ],
    }
    graph = _CountingFakeGraph(
        nodes=_BOUND_NODES, detect=_BOUND_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN
    )
    result = asyncio.run(
        resolve_variable(
            _BOUND_DETECT_QUERY,
            entities=[],
            date_request=None,
            detect_query=_BOUND_DETECT_QUERY,
            role_query=_BOUND_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindFemaleLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "definite"
    (spec,) = result.specs
    assert [m.ref.ref for m in spec.resolved.members] == [_BOUND_SV]
    assert [fact.asked for fact in result.departure_facts.values()] == ["Female"]


# ---------------------------------------------------------------------------
# (e) enumeration/presence failure fails closed
# ---------------------------------------------------------------------------


def test_scope_enumeration_failure_propagates_as_graph_infra_error():
    """An enumeration failure propagates instead of using StatVar existence as data."""
    graph = _make_graph(obs={}, cls=_RaiseOnChildPlacesBatch)

    with pytest.raises(GraphInfraError):
        _run(graph=graph)


def _dated_facet(facet_id: str, provenance: str, years: range) -> dict:
    return {
        "facetId": facet_id,
        "provenanceId": provenance,
        "unit": "USD",
        "earliestDate": str(years[0]),
        "latestDate": str(years[-1]),
        "obsCount": len(years),
        "dates": [str(y) for y in years],
        "values": [1.0] * len(years),
    }


@pytest.mark.parametrize(
    ("obs", "date_request"),
    [
        (
            {
                f"{_SV}|country/A1": [_dated_facet("fW", _OBS_PROVENANCE, range(2010, 2020))],
                f"{_SV}|country/A2": [_dated_facet("fW", _OBS_PROVENANCE, range(2015, 2025))],
            },
            DateRequest(window=TimeWindow(start_year=2021, end_year=None), latest=False),
        ),
        (
            {
                f"{_SV}|country/A1": [_dated_facet("fI", "dc/base/IMF_WEO", range(2010, 2015))],
                f"{_SV}|country/A2": [_dated_facet("fW", _OBS_PROVENANCE, range(2010, 2025))],
            },
            None,
        ),
    ],
    ids=["since_2021", "dateless_two_sources"],
)
def test_the_scope_lists_the_places_its_plan_serves(obs: dict, date_request) -> None:
    """A1's data ends in 2019 ("since 2021"), or lies on another source than the facet the
    scope read elects (dateless): either way the plan serves A2 alone, and the scope lists
    A2 alone, the place coverage counts."""
    result = _run(graph=_make_graph(obs=obs), date_request=date_request)

    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == ["country/A2"]
    places = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places.count == 1


_SINCE_2016 = DateRequest(window=TimeWindow(start_year=2016, end_year=None), latest=False)


@pytest.mark.parametrize(
    ("obs", "sentence_years"),
    [
        (
            {
                f"{_SV}|country/A1": [_dated_facet("fW", _OBS_PROVENANCE, range(2010, 2020))],
                f"{_SV}|country/A2": [
                    _dated_facet("fW", _OBS_PROVENANCE, range(2010, 2016)),
                    _dated_facet("fU", "dc/base/UN", range(2016, 2023)),
                ],
                f"{_SV}|country/A3": [_dated_facet("fU", "dc/base/UN", range(2018, 2023))],
            },
            "2016 to 2022",
        ),
        (
            {
                f"{_SV}|country/A1": [_dated_facet("fW", _OBS_PROVENANCE, range(2010, 2023))],
                f"{_SV}|country/A2": [
                    _dated_facet("fW", _OBS_PROVENANCE, range(2010, 2013)),
                    _dated_facet("fU", "dc/base/UN", range(2016, 2018)),
                ],
                f"{_SV}|country/A3": [
                    _dated_facet("fW", _OBS_PROVENANCE, range(2010, 2013)),
                    _dated_facet("fU", "dc/base/UN", range(2016, 2018)),
                ],
            },
            "2016 to 2017",
        ),
    ],
    ids=["more_points_inside", "fewer_points_inside"],
)
def test_a_range_elects_the_facet_covering_the_most_places_inside_it(
    obs: dict, sentence_years: str
) -> None:
    """Over every year `fW` covers more places than `fU`, but since 2016 it holds a point at
    A1 alone, while `fU` holds one at A2 and A3: the plan elects `fU`, and the scope,
    coverage and sentence state its two places. It does so whether `fU` holds more points
    inside the years than `fW` or fewer: places lead the election."""
    result = _run(graph=_make_graph(obs=obs), date_request=_SINCE_2016)

    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == ["country/A2", "country/A3"]
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert {d.label: d.count for d in coverage.dimensions or []}["places"] == 2
    assert f", {sentence_years}" in sentence_from_plan(render_sentence_plan(spec))


def test_a_clipped_scope_window_notes_where_the_served_series_starts() -> None:
    """Inside 2015 to 2020 `fW` and `fU` each hold a point at two places, and `fU`'s span
    covers five of the years against `fW`'s two, so the plan serves `fU` from 2016. `fW`
    publishes 2015 at A1 and A2, so the sentence and its `year_unavailable` departure state
    where the served series runs."""
    obs = {
        f"{_SV}|country/A1": [_dated_facet("fW", _OBS_PROVENANCE, range(2010, 2017))],
        f"{_SV}|country/A2": [
            _dated_facet("fW", _OBS_PROVENANCE, range(2010, 2016)),
            _dated_facet("fU", "dc/base/UN", range(2016, 2023)),
        ],
        f"{_SV}|country/A3": [_dated_facet("fU", "dc/base/UN", range(2018, 2023))],
    }
    window = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)

    result = _run(graph=_make_graph(obs=obs), date_request=window)

    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == ["country/A2", "country/A3"]
    heading = sentence_from_plan([t for t in render_sentence_plan(spec) if t.segment == "heading"])
    assert heading.endswith(", 2016 to 2020."), heading
    assert [
        fact.shown
        for fact in spec_departure_facts(spec, operations=())
        if fact.code == "year_unavailable"
    ] == ["2016 to 2020"]


# ---------------------------------------------------------------------------
# nearest-real relaxations of a placeless shape probe their own default scope
# ---------------------------------------------------------------------------

_RELAX_BOTH = "TEST/PlaceScopeDefault_FemaleAdult"
_RELAX_SEX = "TEST/PlaceScopeDefault_MaleAdult"
_RELAX_AGE = "TEST/PlaceScopeDefault_FemaleSenior"


def _relax_sv(label: str, sex: str, age: str) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": [{"dcid": "gender"}, {"dcid": "age"}]},
            "gender": {"nodes": [{"dcid": sex}]},
            "age": {"nodes": [{"dcid": age}]},
        },
    }


class _BindFemaleAdultLLM:
    """Bind `gender=Female` and `age=Years15Onwards`."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(
            0,
            bindings=[
                SlotBindingDraft(
                    axis="what", property_dcid="gender", kind="value", value_dcids=["Female"]
                ),
                SlotBindingDraft(
                    axis="what", property_dcid="age", kind="value", value_dcids=["Years15Onwards"]
                ),
            ],
            also_plausible=None,
        ), None


def test_a_placeless_relaxation_is_suggested_over_the_default_scope_of_its_own_members() -> None:
    """The bound female adult count has data nowhere, so the shape in play's default scope
    is empty. Dropping the gender binding admits the male adult count, which has data
    worldwide alone; dropping the age binding admits the female senior count, held in one
    country. Each relaxation is probed over the default scope of the members it admits, as
    a resubmit of it is, so both are suggested: the first as a worldwide total."""
    graph = FakeGraph(
        nodes={
            _RELAX_BOTH: _relax_sv(
                "Test place scope population: female adults", "Female", "Years15Onwards"
            ),
            _RELAX_SEX: _relax_sv(
                "Test place scope population: male adults", "Male", "Years15Onwards"
            ),
            _RELAX_AGE: _relax_sv(
                "Test place scope population: female seniors", "Female", "Years65Onwards"
            ),
            "Female": {"label": "Female"},
            "Male": {"label": "Male"},
            "Years15Onwards": {"label": "15 years or over"},
            "Years65Onwards": {"label": "65 years or over"},
        },
        detect={_DETECT_QUERY: {"svs": [_RELAX_BOTH, _RELAX_SEX, _RELAX_AGE]}},
        resolve={},
        obs={
            f"{_RELAX_SEX}|Earth": [_dated_facet("fW", _OBS_PROVENANCE, range(2015, 2023))],
            f"{_RELAX_AGE}|country/A1": [_dated_facet("fW", _OBS_PROVENANCE, range(2015, 2023))],
        },
        children=_EARTH_CHILDREN,
    )
    result = _run(graph=graph, llm=_BindFemaleAdultLLM())

    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    suggested = {
        s.relaxation.property_dcid: (
            [sv.ref.ref for sv in s.resolved.members],
            [d.label for d in s.resolved.coverage.dimensions or []],
        )
        for s in result.nearest_real or ()
    }
    assert suggested == {
        "gender": ([_RELAX_SEX], ["worldwide", "observations", "variables"]),
        "age": ([_RELAX_AGE], ["sources", "observations", "variables"]),
    }


def _scope_sdmx(graph: FakeGraph) -> FakeSdmxClient:
    """The SDMX client of a place-less run. Its availability names the places `graph` holds
    data at, so the default scope is every place the graph's observations name."""
    return FakeSdmxClient(
        observations={}, availability={}, scope_availability=graph.scope_availability
    )


_EVERY_PLACE_MESSAGE = (
    "no place was named; the answer covers every place with data, at every level."
)
_WORLDWIDE_MESSAGE = "no place was named; the answer was resolved as a worldwide total."
_CITY = "wikidataId/Q3870"


def _scope_messages(result: RegionResult) -> dict[str | None, str]:
    """Each `PLACE_SCOPE_DEFAULTED` message by the StatVar of the Spec it names, or by `None`
    for the answer's own."""
    specs = {
        spec.spec_id: spec
        for spec in (*result.specs, *((result.alternate_match,) if result.alternate_match else ()))
    }
    return {
        (specs[w.spec_id].resolved.members[0].ref.ref if w.spec_id else None): w.message
        for w in result.warnings
        if w.code == PLACE_SCOPE_DEFAULTED
    }


# ---------------------------------------------------------------------------
# A shape whose members a mint serves differently from the members recall holds
# ---------------------------------------------------------------------------

_SV_FEMALE = "TEST/PlaceScopeDefault_PopulationFemale"
_TOTAL_AND_PART_NODES: dict = {
    **_NODES,
    _SV_FEMALE: {
        "label": "Test Population, female",
        "type": "StatisticalVariable",
        "arcs": {
            **_NODES[_SV]["arcs"],
            "name": {"nodes": [{"value": "Test place scope population, female"}]},
            "constraintProperties": {"nodes": [{"dcid": "gender"}]},
            "gender": {"nodes": [{"dcid": "Female"}]},
        },
    },
}


def _one_point(year: str) -> list[dict]:
    return [{"earliestDate": year, "latestDate": year, "obsCount": 1}]


def test_a_shape_with_a_total_and_its_part_is_scoped_over_the_total_its_mint_serves():
    """The total and its female part share the shape, and `gender` reads `aggregate`, so the
    total dominates the part and the mint serves the total alone. The part's data in A2
    stays out of the answer's scope, which is the total's own country."""
    graph = FakeGraph(
        nodes=_TOTAL_AND_PART_NODES,
        detect={_DETECT_QUERY: {"svs": [_SV, _SV_FEMALE], "cosine_scores": [0.9, 0.89]}},
        resolve={},
        obs={
            f"{_SV}|country/A1": _one_point("2020"),
            f"{_SV_FEMALE}|country/A2": _one_point("2021"),
        },
        children=_EARTH_CHILDREN,
    )

    result = _run(graph=graph, sdmx=_scope_sdmx(graph))

    assert result.status == "definite", result.no_data_reason
    (spec,) = result.specs
    assert [sv.ref.ref for sv in spec.resolved.members] == [_SV]
    assert spec.resolved.scope is not None
    assert [entity.ref for entity in spec.resolved.scope.members] == ["country/A1"]
    assert _scope_messages(result) == {None: _EVERY_PLACE_MESSAGE}


@pytest.mark.parametrize(
    "llm",
    [_FixedBindLLM(), _OpenSlotsBindLLM("gender")],
    ids=["slot-left-out", "slot-drafted-unbound"],
)
def test_a_recalled_part_answers_as_the_total_its_mint_serves(llm):
    """Recall holds the female part alone and `gender` stays unbound. The mint resolves the
    emitted identity by its walk to the total and its part, and serves the total, so the
    answer lists the total and is scoped over the total's own country. A reply drafting
    `gender` unbound leaves a slot whose sampled taxonomy holds `Female` alone, so the
    engine claims it from the sample, and the walk, which reads the family with the claim
    open, refutes it on the total."""
    graph = FakeGraph(
        nodes=_TOTAL_AND_PART_NODES,
        detect={_DETECT_QUERY: {"svs": [_SV_FEMALE], "cosine_scores": [0.9]}},
        resolve={},
        obs={
            f"{_SV}|country/A1": _one_point("2020"),
            f"{_SV_FEMALE}|country/A2": _one_point("2021"),
        },
        children=_EARTH_CHILDREN,
    )
    member_graph = member_walk_reader(graph)

    result = _run(graph=graph, llm=llm, sdmx=_scope_sdmx(graph), materialise_graph=member_graph)

    assert result.status == "definite", result.no_data_reason
    (spec,) = result.specs
    minted = resolve_definition_members(
        spec.definition.identity.model_dump(mode="json", exclude_none=True),
        spec_id=spec.spec_id,
        graph=member_graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    )
    answered = [sv.ref.ref for sv in spec.resolved.members]
    assert answered == [shape.stat_var.removeprefix("dcid:") for shape in minted.shapes]
    assert answered == [_SV]
    assert MEMBERSHIP_UNVERIFIED not in [w.code for w in result.warnings]
    assert spec.resolved.scope is not None
    assert [entity.ref for entity in spec.resolved.scope.members] == ["country/A1"]


# ---------------------------------------------------------------------------
# A shape whose served members hold data at different places -> each member at its own
# ---------------------------------------------------------------------------

_U_FEMALE = "TEST/PlaceScopeDefault_PopulationFemaleOnly"
_U_OVER65 = "TEST/PlaceScopeDefault_PopulationOver65"


def _member_node(name: str, prop: str, value: str) -> dict:
    return {
        "label": name,
        "type": "StatisticalVariable",
        "arcs": {
            **_NODES[_SV]["arcs"],
            "name": {"nodes": [{"value": name}]},
            "constraintProperties": {"nodes": [{"dcid": prop}]},
            prop: {"nodes": [{"dcid": value}]},
        },
    }


class _ProbeLogGraph(FakeGraph):
    """Records the StatVars and places of every observation probe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probes: list[tuple[list[str], list[str]]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs):
        self.probes.append((list(stat_vars), list(entities)))
        return super().observation_facets_batch(stat_vars, entities, needs_dates=needs_dates)


@pytest.mark.parametrize(
    "answer", ["elected", "elected_slots_drafted", "alternate", "lone_alternate"]
)
def test_each_member_is_counted_and_grounded_at_its_own_scope(answer, monkeypatch):
    """The shape serves two members neither of which dominates the other: `_U_FEMALE` holds
    data in A1 and `_U_OVER65` in A2. The shape answers as the elected one, as an alternate
    beside the elected Household shape, or as the lone alternate of an elected shape with no
    data. One batch reads both members at the union of their scopes. Elected, the shape
    leaves both open and serves both, each counted at its own country; an alternate binds
    `_U_FEMALE`'s constraint, so it shows that member, counts its own country alone, and the
    lone alternate previews its series there. The elected answer's two places preview
    through a scope-wide SDMX read, which this test's SDMX double holds no rows for. A reply
    drafting `gender` and `age` unbound, or leaving both out, leaves each a slot whose
    sampled taxonomy holds one value, so the engine claims both from the sample, and the
    walk over the members' arcs, which reads the family with both claims open, refutes each
    on the member that asserts the other."""
    elected = answer in ("elected", "elected_slots_drafted")

    def facet(year: str) -> list[dict]:
        return [
            {
                "earliestDate": year,
                "latestDate": year,
                "obsCount": 1,
                "facetId": "f1",
                "observations": [{"date": year, "value": 1.0}],
            }
        ]

    obs = {f"{_U_FEMALE}|country/A1": facet("2020"), f"{_U_OVER65}|country/A2": facet("2021")}
    svs, scores = [_U_FEMALE, _U_OVER65], [0.9, 0.89]
    if not elected:
        svs, scores = [_SV_B, *svs], [0.95, *scores]
    if answer == "alternate":
        obs[f"{_SV_B}|country/A3"] = facet("2021")
    if answer == "lone_alternate":
        # As in `test_an_alternate_that_answers_states_its_own_scope`.
        monkeypatch.setattr(
            candidate_collapse,
            "_handle_sparse_survivors",
            lambda _state, specs, *_a, **_k: specs,
        )
    graph = _ProbeLogGraph(
        nodes={
            _SV_B: _CANDIDATES_NODES[_SV_B],
            _U_FEMALE: _member_node("Test place scope population, female", "gender", "Female"),
            _U_OVER65: _member_node(
                "Test place scope population, 65 and over", "age", "Years65Onwards"
            ),
        },
        detect={_DETECT_QUERY: {"svs": svs, "cosine_scores": scores}},
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    if elected:
        materialise_graph = with_member_arcs(materialise_graph, graph)

    result = _run(
        graph=graph,
        llm={
            "elected": _FixedBindLLM(),
            "elected_slots_drafted": _OpenSlotsBindLLM("gender", "age"),
        }.get(answer, _CandidatesBindLLM()),
        sdmx=_scope_sdmx(graph),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        include_observations=True,
    )

    assert result.status == ("candidates" if answer == "alternate" else "definite"), (
        result.no_data_reason
    )
    spec = next(spec for spec in result.specs if spec.resolved.members[0].ref.ref == _U_FEMALE)
    members, places = (
        ([_U_FEMALE, _U_OVER65], ["country/A1", "country/A2"])
        if elected
        else ([_U_FEMALE], ["country/A1"])
    )
    assert [sv.ref.ref for sv in spec.resolved.members] == members
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert coverage.observation_count == len(members)
    assert spec.resolved.scope is not None
    assert [entity.ref for entity in spec.resolved.scope.members] == places
    assert ([_U_FEMALE, _U_OVER65], ["country/A1", "country/A2"]) in graph.probes
    if answer == "lone_alternate":
        observations = spec.resolved.observations
        assert observations is not None and observations.series
        assert {entity.ref for series in observations.series for entity in series.entities} == {
            "country/A1"
        }


def test_each_candidate_counts_the_places_of_its_own_scope():
    """SV_A holds data in A1 and at a city, SV_B at that city and at Earth. Availability lists
    the city, so each default scope is every place its StatVar holds data at. Each candidate
    counts and states what its own mint serves: SV_A its country and the city, SV_B the city
    and Earth."""
    obs = {
        f"{_SV_A}|country/A1": _one_point("2020"),
        f"{_SV_A}|{_CITY}": _one_point("2020"),
        f"{_SV_B}|{_CITY}": _one_point("2021"),
        f"{_SV_B}|Earth": _one_point("2021"),
    }
    graph = FakeGraph(
        nodes=_CANDIDATES_NODES,
        detect=_CANDIDATES_DETECT,
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )

    result = _run(graph=graph, llm=_CandidatesBindLLM(), sdmx=_scope_sdmx(graph))

    assert result.status == "candidates", result.no_data_reason
    by_stat_var = {spec.resolved.members[0].ref.ref: spec for spec in result.specs}
    assert {sv: spec.resolved.coverage.observation_count for sv, spec in by_stat_var.items()} == {
        _SV_A: 2,
        _SV_B: 2,
    }
    assert {
        sv: [entity.ref for entity in spec.resolved.scope.members]
        for sv, spec in by_stat_var.items()
        if spec.resolved.scope is not None
    } == {_SV_A: ["country/A1", _CITY], _SV_B: [_CITY, "Earth"]}
    # Both scopes are every place, so the answer's one warning states the alternate's too.
    assert _scope_messages(result) == {None: _EVERY_PLACE_MESSAGE}


def _alternate_only_run() -> RegionResult:
    """SV_A, elected, holds data nowhere. SV_B, the alternate, holds data at a city and at
    Earth, so its scope is every place: the city and Earth."""
    obs = {f"{_SV_B}|{_CITY}": _one_point("2021"), f"{_SV_B}|Earth": _one_point("2021")}
    graph = FakeGraph(
        nodes=_CANDIDATES_NODES,
        detect=_CANDIDATES_DETECT,
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )
    return _run(graph=graph, llm=_CandidatesBindLLM(), sdmx=_scope_sdmx(graph))


def test_a_lone_alternate_match_states_its_own_scope_beside_the_elected_one():
    result = _alternate_only_run()

    assert result.status == "no_data"
    assert result.alternate_match is not None
    # Both scopes are every place, so the answer's one warning states the alternate's too.
    assert _scope_messages(result) == {None: _EVERY_PLACE_MESSAGE}


def test_an_alternate_that_answers_states_its_own_scope(monkeypatch):
    # A lone nominated alternate rides on the refusal of an elected shape with no data
    # (`candidate_collapse._handle_sparse_survivors`); this lets it answer, to read the
    # scope it states.
    monkeypatch.setattr(
        candidate_collapse,
        "_handle_sparse_survivors",
        lambda _state, specs, *_a, **_k: specs,
    )

    result = _alternate_only_run()

    assert result.status == "definite", result.no_data_reason
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_SV_B]
    assert _scope_messages(result) == {None: _EVERY_PLACE_MESSAGE}


# ---------------------------------------------------------------------------
# The scope reads run under the request's deadline
# ---------------------------------------------------------------------------


def _is_place_check(dcids) -> bool:
    """Whether a `node_arcs_many` read is the default scope's check of listed places."""
    return any(dcid.startswith("country/") or dcid == "Earth" for dcid in dcids)


class _DeadlineLogGraph(FakeGraph):
    """Records the deadline of each default-scope graph read: Earth's countries, then the
    check of the places availability lists."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope_reads: list[tuple[str, float | None]] = []

    def child_places_batch(self, parent_dcids, child_type, *, deadline=None):
        self.scope_reads.append(("countries", deadline))
        return super().child_places_batch(parent_dcids, child_type, deadline=deadline)

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if _is_place_check(dcids):
            self.scope_reads.append(("places", deadline))
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def test_the_scope_reads_run_under_the_request_deadline():
    # Sooner than the place check's own window, so the request's deadline bounds that read.
    graph = _make_graph(obs={f"{_SV}|country/A1": _one_point("2020")}, cls=_DeadlineLogGraph)
    deadline = time.monotonic() + place_scope.PLACE_CHECK_SECONDS / 2

    _run(graph=graph, sdmx=_scope_sdmx(graph), deadline=deadline)

    assert graph.scope_reads == [("countries", deadline), ("places", deadline)]


class _PlaceCheckPastDeadlineGraph(FakeGraph):
    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if _is_place_check(dcids):
            assert deadline is not None
            time.sleep(max(0.0, deadline - time.monotonic()))
            raise dc_graph.GraphDeadlineExceededError("the place check ran past the deadline")
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def test_a_scope_the_request_deadline_leaves_unread_propagates_as_a_timeout():
    # The app answers a deadline failure 504 `timeout`; no scope reads as empty.
    graph = _make_graph(
        obs={f"{_SV}|country/A1": _one_point("2020")}, cls=_PlaceCheckPastDeadlineGraph
    )

    with pytest.raises(place_scope.PlaceScopeDeadlineError):
        _run(graph=graph, sdmx=_scope_sdmx(graph), deadline=time.monotonic() + 0.5)


_SLOW_READ_S = 2.0
_BUDGET_S = 0.8


class _SlowAvailability(ScopeAvailability):
    """Holds each availability read for `_SLOW_READ_S`."""

    def __init__(self, *, data) -> None:
        super().__init__(data=data)
        self.reads_started: list[float] = []

    def availability_values(self, **kwargs) -> list[str]:
        self.reads_started.append(time.monotonic())
        time.sleep(_SLOW_READ_S)
        return super().availability_values(**kwargs)


class _SlowPresenceGraph(FakeGraph):
    """Answers the default scope's availability reads slowly (`_SlowAvailability`)."""

    @functools.cached_property
    def scope_availability(self) -> _SlowAvailability:
        return _SlowAvailability(data=super().scope_availability.pairs)


@pytest.mark.parametrize("answer", ["definite", "candidates"])
def test_a_slow_scope_read_cannot_carry_the_request_past_its_deadline(answer):
    # The buffered route bounds the pipeline with `asyncio.wait_for`, whose timer fires only
    # while the event loop runs. A scope read on the loop thread would hold the loop, and the
    # request with it, until the read returned.
    obs = {f"{_SV_A}|country/A1": _one_point("2020"), f"{_SV_B}|country/A2": _one_point("2021")}
    if answer == "definite":
        graph = _make_graph(obs={f"{_SV}|country/A1": _one_point("2020")}, cls=_SlowPresenceGraph)
        llm = _FixedBindLLM()
    else:
        graph = _SlowPresenceGraph(
            nodes=_CANDIDATES_NODES,
            detect=_CANDIDATES_DETECT,
            resolve={},
            obs=obs,
            children=_EARTH_CHILDREN,
        )
        llm = _CandidatesBindLLM()

    async def seconds_to_timeout() -> float:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                _resolve(graph=graph, llm=llm, sdmx=_scope_sdmx(graph)), timeout=_BUDGET_S
            )
        return time.monotonic() - started

    started = time.monotonic()
    elapsed = asyncio.run(seconds_to_timeout())

    reads_started = graph.scope_availability.reads_started
    assert reads_started, "the availability read never ran, so nothing was slow"
    assert reads_started[0] - started < _BUDGET_S, "the read began after the deadline"
    assert elapsed < _BUDGET_S + (_SLOW_READ_S - _BUDGET_S) / 2, (
        f"the request ended {elapsed:.2f} s in, past its {_BUDGET_S} s deadline"
    )
