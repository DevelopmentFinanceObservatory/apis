"""Pins default scope members, metadata, filtering, and batching on resolve and resubmit paths."""

from __future__ import annotations

import asyncio

from dc_sdmx import SdmxObservation
from search.engine.bind import _UniversalBindOutput
from search.engine.region import (
    PLACE_SCOPE_DEFAULTED,
    RegionResult,
    resolve_spec_resubmit,
    resolve_variable,
)
from search.engine.region.scope import build_scope
from search.engine.region.state import EngineOptions
from search.engine.spec_build import build_spec
from search.models import (
    CoverageBare,
    CoverageExact,
    IdentityBindingValue,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import make_identity
from tests.engine._builders import shape_five_tuple
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, walk_options
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import CONVERTIBLE_STAT_VAR_ARCS, conversion_arcs

_SV = "TEST/ResolvedScope_Population"
_DETECT_QUERY = "test resolved scope population"
_COUNTRIES = ["country/A1", "country/A2", "country/A3"]
_EARTH_CHILDREN = {"Earth": {"Country": _COUNTRIES}}
_NODES: dict = {
    _SV: {
        "label": "Test Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test Population"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/A1": {"label": "Testland One", "type": "Country"},
    "country/A2": {"label": "Testland Two", "type": "Country"},
    # A3 has no label or type; pure assembly tests pin those missing values.
    "Earth": {"label": "Earth", "type": "Place"},
}
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV]}}


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
        ), None


class _CandidatesBindLLM:
    """Elects capped_shapes[0], flags capped_shapes[1] also_plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


class _CountingFakeGraph(FakeGraph):
    """Count label and type batch reads across specs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_batch_calls = 0
        self.types_batch_calls = 0

    def node_labels_batch(self, dcids):
        self.labels_batch_calls += 1
        return super().node_labels_batch(dcids)

    def node_types_batch(self, dcids):
        self.types_batch_calls += 1
        return super().node_types_batch(dcids)


def _make_graph(*, obs: dict, cls=_CountingFakeGraph, resolve: dict | None = None) -> FakeGraph:
    return cls(
        nodes=_NODES,
        detect=_DETECT,
        resolve=resolve if resolve is not None else {},
        obs=obs,
        children=_EARTH_CHILDREN,
    )


def _run(
    *, graph: FakeGraph, llm=None, entities: list[str] | None = None, resolver=None
) -> RegionResult:
    return asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=entities if entities is not None else [],
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=resolver if resolver is not None else make_resolver(graph),
            llm=llm if llm is not None else _FixedBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_build_scope_sorts_members_by_ref():
    scope = build_scope(
        "observationAbout",
        ["country/C", "country/A", "country/B"],
        {"country/A": "Alpha", "country/B": "Beta", "country/C": "Gamma"},
        {"country/A": "Country", "country/B": "Country", "country/C": "Country"},
    )
    assert [m.ref for m in scope.members] == ["country/A", "country/B", "country/C"]
    assert scope.axis == "observationAbout"


def test_build_scope_missing_label_and_type_stay_none():
    scope = build_scope(
        "observationAbout",
        ["country/A", "country/B"],
        {"country/A": "Alpha"},
        {"country/B": "Country"},
    )
    by_ref = {m.ref: m for m in scope.members}
    assert by_ref["country/A"].label == "Alpha"
    assert by_ref["country/A"].entity_type is None
    assert by_ref["country/B"].label is None
    assert by_ref["country/B"].entity_type == "Country"


def test_build_scope_every_member_carries_axis_as_role():
    scope = build_scope("observationAbout", ["country/A", "country/B"], {}, {})
    assert all(m.role == "observationAbout" for m in scope.members)
    assert all(m.confidence is None for m in scope.members)


def test_build_scope_empty_members_is_empty_list():
    scope = build_scope("observationAbout", [], {}, {})
    assert scope.members == []
    assert scope.axis == "observationAbout"


def _minimal_build_spec_kwargs() -> dict:
    return {
        "shape": shape_five_tuple("Person", "count"),
        "slots": [],
        "stat_vars": [],
        "entities": [],
        "coverage": CoverageBare(has_data=False),
        "open_roles": (),
    }


def test_scope_is_never_folded_into_identity_or_spec_id():
    """Scope belongs to resolved data and does not change `spec_id` or identity."""
    kwargs = _minimal_build_spec_kwargs()
    bare = build_spec(measure_elected_properties=frozenset(), **kwargs)
    scope = build_scope(
        "observationAbout", ["country/A1", "country/A2"], {"country/A1": "Testland One"}, {}
    )
    scoped = build_spec(measure_elected_properties=frozenset(), **kwargs, scope=scope)

    assert scoped.spec_id == bare.spec_id
    assert scoped.definition.identity == bare.definition.identity
    assert bare.resolved.scope is None
    assert scoped.resolved.scope is scope
    assert scoped.resolved.entities == bare.resolved.entities == []


def test_scope_present_with_axis_and_count_invariant_and_member_content():
    obs = {
        f"{_SV}|country/A1": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}],
        f"{_SV}|country/A2": [{"earliestDate": "2021", "latestDate": "2021", "obsCount": 1}],
    }
    graph = _make_graph(obs=obs)
    result = _run(graph=graph)

    assert result.status == "definite"
    spec = result.specs[0]
    assert any(w.code == PLACE_SCOPE_DEFAULTED for w in result.warnings)

    assert spec.resolved.scope is not None
    scope = spec.resolved.scope
    assert scope.axis == "observationAbout"
    assert [m.ref for m in scope.members] == ["country/A1", "country/A2"]
    assert all(m.role == "observationAbout" for m in scope.members)
    by_ref = {m.ref: m for m in scope.members}
    assert by_ref["country/A1"].label == "Testland One"
    assert by_ref["country/A1"].entity_type == "Country"
    assert by_ref["country/A2"].label == "Testland Two"

    assert isinstance(spec.resolved.coverage, CoverageExact)
    places_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places_dim.count == len(scope.members) == 2

    # Default scope members stay separate from resolved entities.
    assert spec.resolved.entities == []

    # One label/type call pair covers the response.
    assert graph.types_batch_calls == 1


def test_scope_absent_for_a_named_place_query():
    """A named place bypasses default scope and its batch reads. The one type batch read is
    grounding's, for the entity the Spec binds."""
    obs = {f"{_SV}|country/A1": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]}
    graph = _make_graph(obs=obs, resolve={"Testland One": "country/A1"})
    result = _run(graph=graph, entities=["Testland One"])

    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.scope is None
    assert not any(w.code == PLACE_SCOPE_DEFAULTED for w in result.warnings)
    assert graph.types_batch_calls == 1


def test_earth_only_data_scope_is_the_single_earth_member():
    obs = {
        f"{_SV}|Earth": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
            }
        ],
    }
    graph = _make_graph(obs=obs)
    result = _run(graph=graph)

    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == ["Earth"]
    assert spec.resolved.scope.members[0].role == "observationAbout"
    assert spec.resolved.scope.members[0].label == "Earth"


# ---------------------------------------------------------------------------
# resolve_variable: a bound unit narrows scope in agreement with coverage
# ---------------------------------------------------------------------------


def _bulk_row(entity: str, *, facet_id: str, unit: str, value: float = 1.0) -> SdmxObservation:
    return SdmxObservation(
        variable=_SV,
        role_values={"observationAbout": entity},
        unit=unit,
        measurement_method=None,
        observation_period=None,
        provenance=None,
        time_period="2020",
        value=str(value),
        scaling_factor=None,
        facet_id=facet_id,
    )


def test_scope_narrows_to_admitted_entities_when_unit_is_bound():
    """A USD binding keeps A1 and A2, which have USD facets, and excludes EUR-only A3."""
    obs = {
        f"{_SV}|country/A1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
                "unit": "CurrencyUSD",
                "facetId": "F1",
            }
        ],
        f"{_SV}|country/A2": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
                "unit": "CurrencyUSD",
                "facetId": "F1",
            }
        ],
        f"{_SV}|country/A3": [
            {
                "earliestDate": "2022",
                "latestDate": "2022",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
                "unit": "CurrencyEUR",
                "facetId": "F1",
            }
        ],
    }
    graph = _make_graph(obs=obs)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            _SV: {
                "country/A1": [_bulk_row("country/A1", facet_id="F1", unit="USD")],
                "country/A2": [_bulk_row("country/A2", facet_id="F1", unit="USD")],
            }
        },
        scope_availability=graph.scope_availability,
    )
    # Nominal and conversion-recipe arcs let both native currencies pass conversion.
    # The test pins scope narrowing, not which facet becomes representative.
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
        **conversion_arcs(
            recipe_id="ConversionRecipeDAC",
            provenance_id="dc/base/Src",
            currencies=["CurrencyUSD", "CurrencyEUR"],
        ),
    }
    result = asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=[],
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_timing={},
            sdmx=sdmx,
            unit_text="US dollars",
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == ["country/A1", "country/A2"]

    places_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places_dim.count == len(spec.resolved.scope.members) == 2


# ---------------------------------------------------------------------------
# resolve_variable: candidates path
# ---------------------------------------------------------------------------

_SV_A = "TEST/ResolvedScope_Population"
_SV_B = "TEST/ResolvedScope_Household"
_CANDIDATES_NODES: dict = {
    **_NODES,
    _SV_B: {
        "label": "Test Household",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test Household"}]},
            "populationType": {"nodes": [{"dcid": "Household"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}
_CANDIDATES_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.9, 0.85]}}


def test_candidates_each_carry_their_own_scope():
    """Each candidate scope contains its own probed member; one label/type batch covers both."""
    obs = {
        f"{_SV_A}|country/A1": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}],
        f"{_SV_B}|country/A2": [{"earliestDate": "2021", "latestDate": "2021", "obsCount": 1}],
    }
    graph = _CountingFakeGraph(
        nodes=_CANDIDATES_NODES,
        detect=_CANDIDATES_DETECT,
        resolve={},
        obs=obs,
        children=_EARTH_CHILDREN,
    )
    result = _run(graph=graph, llm=_CandidatesBindLLM())

    assert result.status == "candidates"
    assert len(result.specs) == 2
    for spec in result.specs:
        assert spec.resolved.scope is not None
        assert spec.resolved.scope.axis == "observationAbout"
        assert len(spec.resolved.scope.members) == 1
        ref = spec.resolved.scope.members[0].ref
        if any(sv.ref.dcid == _SV_A for sv in spec.resolved.members):
            assert ref == "country/A1"
        else:
            assert ref == "country/A2"

    assert graph.types_batch_calls == 1


# ---------------------------------------------------------------------------
# resolve_spec_resubmit: scope is rediscovered fresh
# ---------------------------------------------------------------------------

_RESUBMIT_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "TestResolvedScopeResubmitPopulation",
    "measured_property": "testResolvedScopeCount",
    "stat_type": "measuredValue",
}
_RESUBMIT_SV = "TEST/ResolvedScope_Resubmit_Population"
_RESUBMIT_COUNTRIES = [f"country/RS{i}" for i in range(1, 4)]
_RESUBMIT_EARTH_CHILDREN = {"Earth": {"Country": _RESUBMIT_COUNTRIES}}
_RESUBMIT_NODES: dict = {
    _RESUBMIT_SV: {
        "label": "Test Resolved Scope Resubmit Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test Resolved Scope Resubmit Population"}]},
            "populationType": {"nodes": [{"dcid": "TestResolvedScopeResubmitPopulation"}]},
            "measuredProperty": {"nodes": [{"dcid": "testResolvedScopeCount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/RS1": {"label": "Resubmitland One", "type": "Country"},
    "country/RS2": {"label": "Resubmitland Two", "type": "Country"},
}


def _entity_less_resubmit_input() -> ResubmitIdentityPayload:
    return ResubmitIdentityPayload(
        identity=make_identity(_RESUBMIT_SHAPE, entities=[]),
    )


def test_resubmit_of_the_same_definition_carries_scope_again():
    obs = {
        f"{_RESUBMIT_SV}|{_RESUBMIT_COUNTRIES[0]}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "facetId": "F1"}
        ],
        f"{_RESUBMIT_SV}|{_RESUBMIT_COUNTRIES[1]}": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "facetId": "F1"}
        ],
    }
    graph = _CountingFakeGraph(nodes=_RESUBMIT_NODES, obs=obs, children=_RESUBMIT_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            _RESUBMIT_SV: {c: [] for c in _RESUBMIT_COUNTRIES[:2]},
        },
        scope_availability=graph.scope_availability,
    )
    region = resolve_spec_resubmit(
        inp=_entity_less_resubmit_input(),
        graph=graph,
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert any(w.code == PLACE_SCOPE_DEFAULTED for w in region.warnings)
    assert spec.resolved.scope is not None
    assert spec.resolved.scope.axis == "observationAbout"
    assert [m.ref for m in spec.resolved.scope.members] == list(_RESUBMIT_COUNTRIES[:2])
    assert graph.types_batch_calls == 1


def _entity_less_resubmit_input_with_unit_binding(ref: str) -> ResubmitIdentityPayload:
    return ResubmitIdentityPayload(
        identity=make_identity(
            _RESUBMIT_SHAPE,
            entities=[],
            explicit_facets={"unit": IdentityBindingValue(ref=ref)},
        ),
    )


def test_resubmit_scope_narrows_to_admitted_entities_when_unit_is_bound():
    """A USD binding excludes EUR-only RS3 from scope and coverage. RS1 and RS2 hold one USD
    import's facet, which the scope-wide read serves at both."""
    obs = {
        f"{_RESUBMIT_SV}|{_RESUBMIT_COUNTRIES[0]}": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/ResubmitFixture",
                "facetId": "F1",
                "unit": "CurrencyUSD",
            }
        ],
        f"{_RESUBMIT_SV}|{_RESUBMIT_COUNTRIES[1]}": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": "dc/base/ResubmitFixture",
                "facetId": "F1",
                "unit": "CurrencyUSD",
            }
        ],
        f"{_RESUBMIT_SV}|{_RESUBMIT_COUNTRIES[2]}": [
            {
                "earliestDate": "2022",
                "latestDate": "2022",
                "obsCount": 1,
                "provenanceId": "dc/base/ResubmitFixture",
                "facetId": "F3",
                "unit": "CurrencyEUR",
            }
        ],
    }
    graph = _CountingFakeGraph(nodes=_RESUBMIT_NODES, obs=obs, children=_RESUBMIT_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={}, availability={}, scope_availability=graph.scope_availability
    )
    region = resolve_spec_resubmit(
        inp=_entity_less_resubmit_input_with_unit_binding("CurrencyUSD"),
        graph=graph,
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == list(_RESUBMIT_COUNTRIES[:2])

    places_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places_dim.count == len(spec.resolved.scope.members) == 2
