"""Tests for scope-defaulted bulk observations: facet selection, dispatch, degradation, and
bound-unit behaviour at unit and pipeline level. SDMX request construction is covered by the
client tests.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx
import pytest

import search.engine.region.observations.bulk_conversion as bulk_conversion
import search.engine.region.observations.fetch as region_fetch
from dc_graph import GraphReadError
from dc_sdmx import SdmxCapExceededError, SdmxObservation
from materialise.facets import FacetSelection, UnitRequest
from materialise.rate_tables import RateTableCache
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import LiveSdmxClient
from search.engine.config import settings
from search.engine.errors import GraphInfraError
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_NO_DISPATCHABLE_FACET,
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_SCOPE_PARTIAL,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.region.facets import _resolved_facets
from search.engine.region.state import EngineOptions
from search.engine.stated_words_read import StatedWordsOutput
from search.models import IdentityUnitRequest, TimeWindow
from search.observations import (
    OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
    observation_settings,
)
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    HarnessScope,
    build_observations_planned,
    harness_reader,
    materialise_double_from_obs,
    sdmx_double,
    unit_check_planned,
)
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SV = "TEST/BulkObs_Population"
_COUNTRIES = [f"country/BLK{i}" for i in range(1, 6)]  # 5-country default scope
_NODES: dict = {
    _SV: {
        "label": "Bulk Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test bulk scope population"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    **{c: {"label": f"Country {c[-1]}"} for c in _COUNTRIES},
}

# ---------------------------------------------------------------------------
# Direct tests for merge, ranking, and dispatch.
# ---------------------------------------------------------------------------

_GRAPH = FakeGraph(nodes=_NODES, obs={})
_SCOPE_PLACES = tuple(_COUNTRIES)


def _row(
    entity: str, *, facet_id: str | None, value: float = 1.0, unit: str | None = None
) -> SdmxObservation:
    return SdmxObservation(
        variable=_SV,
        role_values={"observationAbout": entity},
        unit=unit,
        measurement_method=None,
        observation_period=None,
        provenance=None,
        time_period="2023",
        value=str(value),
        scaling_factor=None,
        facet_id=facet_id,
    )


def _run_bulk(
    facets_by_entity: dict[str, list[Facet]],
    *,
    sdmx: FakeSdmxClient,
    facet_filter=None,
):
    scope = HarnessScope(
        places=_SCOPE_PLACES, sdmx=sdmx, facets_by_sv_entity={_SV: facets_by_entity}
    )
    return build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        facet_filter=facet_filter,
        coverage=empty_coverage_snapshot(),
    )


# Shared-facet happy path.


def test_happy_path_five_countries_one_shared_facet():
    facets_by_entity = {
        c: [
            Facet(
                earliest_date="2020",
                latest_date="2023",
                obs_count=10,
                facet_id="F1",
                provenance_id="dc/base/Src",
            )
        ]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {c: [_row(c, facet_id="F1", value=float(i))] for i, c in enumerate(_COUNTRIES)}
        }
    )
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert warn is None
    assert observations is not None
    assert len(observations.series) == 5
    assert observations.truncated is True
    # `total_obs` sums the five probe counts.
    assert observations.total_obs == 50
    facet0 = observations.series[0].facet
    for series in observations.series:
        assert series.total_obs == 10  # probe-sourced, not 1
        assert len(series.points) == 1
        assert series.facet == facet0  # identical shared facet map

    resolved = _resolved_facets([s.facet for s in observations.series], {})
    assert resolved["provenance"].ref == "dc/base/Src"
    assert resolved["provenance"].selection is None  # concrete, not per_series


def test_bulk_path_never_calls_materialise_preview(monkeypatch):
    """Scope-defaulted bulk observations bypass `materialise_preview`."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("materialise_preview must not run on the bulk-scope path")

    monkeypatch.setattr(region_fetch, "materialise_preview", _fail)

    facets_by_entity = {
        c: [
            Facet(
                earliest_date="2020",
                latest_date="2023",
                obs_count=10,
                facet_id="F1",
                provenance_id="dc/base/Src",
            )
        ]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {c: [_row(c, facet_id="F1", value=float(i))] for i, c in enumerate(_COUNTRIES)}
        }
    )
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert warn is None
    assert observations is not None
    assert len(observations.series) == 5


# --- (2) shared-facet selection ------------------------------------------


def test_shared_facet_across_two_groups_wins_by_coverage():
    """Facet Z wins by covering all five countries, despite lower observation counts."""
    group_x, group_y = _COUNTRIES[:3], _COUNTRIES[3:]
    facets_by_entity = {
        c: [
            Facet(earliest_date="2020", latest_date="2020", obs_count=100, facet_id="X"),
            Facet(earliest_date="2021", latest_date="2021", obs_count=5, facet_id="Z"),
        ]
        for c in group_x
    } | {
        c: [
            Facet(earliest_date="2020", latest_date="2020", obs_count=100, facet_id="Y"),
            Facet(earliest_date="2021", latest_date="2021", obs_count=5, facet_id="Z"),
        ]
        for c in group_y
    }
    sdmx = FakeSdmxClient(
        subject_observations={_SV: {c: [_row(c, facet_id="Z")] for c in _COUNTRIES}}
    )
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert warn is None
    assert sdmx.subject_observations_calls[0]["facet_ids"] == ["Z"]
    assert len(observations.series) == 5


def test_coverage_beats_depth_amended_ranking_rule():
    """Facet P's 5/5 coverage outranks Q's 3/5 coverage and higher count."""
    facets_by_entity: dict[str, list[Facet]] = {}
    for i, c in enumerate(_COUNTRIES):
        facets = [Facet(earliest_date="2020", latest_date="2020", obs_count=1, facet_id="P")]
        if i < 3:
            facets.append(
                Facet(earliest_date="2021", latest_date="2021", obs_count=100, facet_id="Q")
            )
        facets_by_entity[c] = facets
    sdmx = FakeSdmxClient(
        subject_observations={_SV: {c: [_row(c, facet_id="P")] for c in _COUNTRIES}}
    )
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert warn is None
    assert sdmx.subject_observations_calls[0]["facet_ids"] == ["P"]
    assert len(observations.series) == 5
    assert observations.total_obs == 5  # 5 x obs_count=1, not the Q-covered 300


# --- (3) soft union -------------------------------------------------------


def test_soft_union_drops_places_not_carrying_the_selected_facet():
    covered, solo = _COUNTRIES[:4], _COUNTRIES[4]
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id="Z")]
        for c in covered
    }
    facets_by_entity[solo] = [
        Facet(earliest_date="2020", latest_date="2020", obs_count=5, facet_id="Solo")
    ]
    sdmx = FakeSdmxClient(subject_observations={_SV: {c: [_row(c, facet_id="Z")] for c in covered}})
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert observations is not None
    assert len(observations.series) == 4
    assert observations.total_obs == 40
    assert warn is not None
    assert warn.code == OBSERVATIONS_SCOPE_PARTIAL
    assert "4" in warn.message
    assert "5" in warn.message


# --- (4) request shape (engine-level dispatch args) -----------------------


def test_dispatch_request_shape():
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=1, facet_id="F1")]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={_SV: {c: [_row(c, facet_id="F1")] for c in _COUNTRIES}}
    )
    _run_bulk(facets_by_entity, sdmx=sdmx)

    assert len(sdmx.subject_observations_calls) == 1
    call = sdmx.subject_observations_calls[0]
    assert call["variables"] == [_SV]
    assert call["places"] == list(_SCOPE_PLACES)
    assert call["facet_ids"] == ["F1"]
    assert call["latest_only"] is True
    assert call["max_rows"] == settings().sdmx_max_rows
    assert call["max_bytes"] == settings().sdmx_max_bytes


# Unfiltered facet candidates produce no dispatch.


def test_all_candidates_missing_facet_id_issues_zero_sdmx_calls():
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id=None)]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(subject_observations={})
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_DISPATCHABLE_FACET
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_DISPATCHABLE_FACET
    assert sdmx.subject_observations_calls == []


# --- (6) fetch failure, no bound unit --------------------------------------


def test_fetch_failure_delivers_coverage_without_series():
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id="F1")]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={},
        raise_subject_observations=GraphInfraError("simulated SDMX failure"),
    )
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED


# Cap failure without a per-pair fallback.


def test_cap_trip_delivers_coverage_without_series():
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id="F1")]
        for c in _COUNTRIES
    }
    cap_error = SdmxCapExceededError("cap tripped", cap_kind="bytes", cap=100, observed=200)
    sdmx = FakeSdmxClient(subject_observations={}, raise_subject_observations=cap_error)
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED
    # The bulk branch returns directly and has no per-pair fallback.


# --- (8b) dispatched fetch succeeds but returns no rows at all ------------


def test_dispatched_fetch_returning_no_rows_at_all_is_no_points():
    """A successful empty fetch returns `OBSERVATIONS_NO_POINTS`."""
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id="F1")]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(subject_observations={})
    observations, warn = _run_bulk(facets_by_entity, sdmx=sdmx)

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_POINTS
    assert warn is None


# Bound-unit requests on scope-wide previews.


def test_scope_wide_unit_request_without_an_election_withholds_preview_and_calls_no_sdmx():
    """A bound unit request with no unit election to convert by withholds the scope-wide
    preview before any SDMX call."""
    facets_by_entity = {
        c: [
            Facet(
                earliest_date="2020",
                latest_date="2023",
                obs_count=10,
                facet_id="F1",
                provenance_id="dc/base/Src",
            )
        ]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {c: [_row(c, facet_id="F1", value=float(i))] for i, c in enumerate(_COUNTRIES)}
        }
    )
    unit_request = IdentityUnitRequest(currency="USD", prices="current")
    scope = HarnessScope(
        places=_SCOPE_PLACES, sdmx=sdmx, facets_by_sv_entity={_SV: facets_by_entity}
    )
    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        unit_request=unit_request,
        coverage=empty_coverage_snapshot(),
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert warn is None
    assert sdmx.subject_observations_calls == []

    # Without the unit request, the same fixture reaches SDMX.
    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert len(observations.series) == 5
    assert sdmx.subject_observations_calls  # the second call actually ran


def _facet_selection(
    *, axis: str, ref: str | None, selection_reason: str | None, request=None
) -> FacetSelection:
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason=selection_reason,
        dispatch_facet_id="F1",
        request=request,
    )


def test_scope_wide_conversion_without_materialise_clients_calls_no_sdmx():
    """A selected EUR facet needs converting to answer a USD request, which the bulk path
    cannot do without the materialise clients, so it makes no SDMX call."""
    facets_by_entity = {
        c: [
            Facet(
                earliest_date="2020",
                latest_date="2023",
                obs_count=10,
                facet_id="F1",
                provenance_id="dc/base/Src",
            )
        ]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {c: [_row(c, facet_id="F1", value=float(i))] for i, c in enumerate(_COUNTRIES)}
        }
    )
    unit_request = IdentityUnitRequest(currency="USD", prices="current")
    scope = HarnessScope(
        places=_SCOPE_PLACES, sdmx=sdmx, facets_by_sv_entity={_SV: facets_by_entity}
    )
    elected = {
        _SV: (
            {
                "provenance": _facet_selection(
                    axis="provenance", ref="dc/base/Src", selection_reason="sole_candidate"
                ),
                "unit": _facet_selection(
                    axis="unit",
                    ref="CurrencyEUR",
                    selection_reason="most_observations",
                    request=UnitRequest(currency="USD", prices="current"),
                ),
            },
        )
    }
    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        unit_request=unit_request,
        elected=elected,
        coverage=empty_coverage_snapshot(),
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert warn is None
    assert sdmx.subject_observations_calls == []


def test_scope_wide_unit_request_natively_satisfied_falls_through_to_bulk_path(monkeypatch):
    """A native EUR match narrows the scope plan to CurrencyEUR before the read, so the
    USD facet that ranks first on observation count is never dispatched."""
    facets_by_entity = {
        c: [
            Facet(
                earliest_date="2020",
                latest_date="2023",
                obs_count=obs_count,
                facet_id=facet_id,
                provenance_id="dc/base/Src",
                unit=unit,
            )
            for facet_id, unit, obs_count in (("F1", "CurrencyEUR", 10), ("F2", "CurrencyUSD", 40))
        ]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {
                c: [_row(c, facet_id=f, value=float(i)) for f in ("F1", "F2")]
                for i, c in enumerate(_COUNTRIES)
            }
        }
    )
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")
    scope = HarnessScope(
        places=_SCOPE_PLACES, sdmx=sdmx, facets_by_sv_entity={_SV: facets_by_entity}
    )
    elected = {
        _SV: (
            {
                "provenance": _facet_selection(
                    axis="provenance", ref="dc/base/Src", selection_reason="sole_candidate"
                ),
                "unit": _facet_selection(
                    axis="unit",
                    ref="CurrencyEUR",
                    selection_reason="native_currency_preferred",
                    request=UnitRequest(currency="EUR", prices="current"),
                ),
            },
        )
    }
    captured_plans: list = []
    real_bulk_scope_series = region_fetch._bulk_scope_series

    def _capture(*args, **kwargs):
        captured_plans.append(args[0])
        return real_bulk_scope_series(*args, **kwargs)

    monkeypatch.setattr(region_fetch, "_bulk_scope_series", _capture)

    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        unit_request=unit_request,
        elected=elected,
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == 5
    assert [call["facet_ids"] for call in sdmx.subject_observations_calls] == [["F1"]]
    (plan,) = captured_plans
    assert {f.unit for s in plan.served for f in s.group} == {"CurrencyEUR"}


# Bound path with no scope.


def test_bound_path_ceiling_unaffected_by_scope_none():
    """With no bulk scope, the per-pair preview cap returns `OBSERVATIONS_PREVIEW_TOO_LARGE`."""
    entities = [f"country/E{i}" for i in range(observation_settings().obs_preview_max_pairs + 1)]
    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": entities},
        graph=_GRAPH,
        materialise_graph=harness_reader(entities, stat_var=_SV),
        materialise_sdmx=sdmx_double(),
        scope=None,
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PREVIEW_TOO_LARGE
    assert warn is not None
    assert warn.code == OBSERVATIONS_PREVIEW_TOO_LARGE


# Pipeline-level bound-unit and named-region behaviour.


_DETECT_QUERY = "test bulk scope population"
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV]}}
_EARTH_CHILDREN = {"Earth": {"Country": _COUNTRIES}}


class _FixedBindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=None), None


def _run_pipeline(
    *,
    graph: FakeGraph,
    sdmx: FakeSdmxClient,
    unit_text: str | None = None,
    entities: list[str] | None = None,
    date_request: DateRequest | None = None,
    materialise_graph=None,
    materialise_sdmx=None,
):
    return asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=entities or [],
            date_request=date_request,
            detect_query=_DETECT_QUERY,
            role_query=_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_timing={},
            sdmx=sdmx,
            unit_text=unit_text,
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        )
    )


# --- (7) a failed read beside a scope plan answers from the plan -------------


def test_bound_unit_display_read_failure_answers_from_the_scope_probe():
    """The scope probe confirms the elected facet at every place, so the failed latest-point read
    is a display failure: the answer is definite with the omission and its warning, not a 503."""
    obs = {
        f"{_SV}|{c}": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 10,
                "facetId": "F1",
                "unit": "CurrencyUSD",
                # A provenance id lets the bound-unit check resolve its provenance axis.
                "provenanceId": "dc/base/BulkScopeSrc",
            }
        ]
        for c in _COUNTRIES
    }
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={},
        raise_subject_observations=GraphInfraError("simulated SDMX failure"),
        scope_availability=graph.scope_availability,
    )
    # The USD facet is native and declares the required Nominal qualifier.
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        _SV: {"measurementQualifier": ("Nominal",)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    }
    result = _run_pipeline(
        graph=graph,
        sdmx=sdmx,
        unit_text="US dollars",
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )

    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert any(w.code == OBSERVATIONS_BULK_FETCH_FAILED for w in result.warnings)


# --- (9) bound (named-region) path stays byte-identical -------------------


def test_named_region_query_never_calls_subject_observations(declare_regions):
    _region_countries = _COUNTRIES
    declare_regions({"blocland": _region_countries})
    obs = {
        f"{_SV}|{c}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 10, "facetId": "F1"}
        ]
        for c in _region_countries
    }
    places = {"Blocland": {"dcid": "blocland", "type_of": "Continent"}}
    children = {"blocland": {"Country": _region_countries}}
    graph = FakeGraph(
        nodes={**_NODES, "blocland": {"label": "Blocland", "type": "Continent"}},
        detect={"test bulk scope population in Blocland": {"svs": [_SV]}},
        resolve={},
        obs=obs,
        places=places,
        children=children,
    )
    sdmx = FakeSdmxClient(observations={}, availability={}, subject_observations={})
    result = asyncio.run(
        resolve_variable(
            "test bulk scope population in Blocland",
            entities=["Blocland"],
            date_request=None,
            detect_query="test bulk scope population in Blocland",
            role_query="test bulk scope population in Blocland",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_timing={},
            sdmx=sdmx,
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(include_observations=True),
        )
    )
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    assert sdmx.subject_observations_calls == []


# End-to-end scope-defaulted observations.


def test_end_to_end_scope_defaulted_definite_delivers_bulk_series():
    obs = {
        f"{_SV}|{c}": [
            {"earliestDate": "2020", "latestDate": "2023", "obsCount": 10, "facetId": "F1"}
        ]
        for c in _COUNTRIES
    }
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            _SV: {c: [_row(c, facet_id="F1", value=float(i))] for i, c in enumerate(_COUNTRIES)}
        },
        scope_availability=graph.scope_availability,
    )
    result = _run_pipeline(graph=graph, sdmx=sdmx)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    assert len(sdmx.subject_observations_calls) == 1
    call = sdmx.subject_observations_calls[0]
    assert call["variables"] == [_SV]
    assert call["places"] == list(_SCOPE_PLACES)
    assert call["facet_ids"]  # non-empty

    spec = result.specs[0]
    observations = spec.resolved.observations
    assert observations is not None
    assert len(observations.series) == len(_COUNTRIES)
    assert observations.truncated is True
    for series in observations.series:
        assert series.total_obs == 10  # probe-sourced obs_count, never len(points)=1
        assert len(series.points) == 1


def test_windowed_scope_defaulted_query_delivers_coverage_without_series():
    """A resolved window returns coverage without bulk points.
    Such points could misstate in-window counts.
    """
    obs = {
        f"{_SV}|{c}": [
            {"earliestDate": "2020", "latestDate": "2023", "obsCount": 10, "facetId": "F1"}
        ]
        for c in _COUNTRIES
    }
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={},
        scope_availability=graph.scope_availability,
    )
    result = _run_pipeline(
        graph=graph,
        sdmx=sdmx,
        date_request=DateRequest(window=TimeWindow(start_year=2021, end_year=2022), latest=False),
    )

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    assert spec.resolved.coverage.has_data is True
    assert spec.resolved.observations is not None
    assert spec.resolved.observations.series == []
    assert spec.resolved.observations.omitted is not None
    assert spec.resolved.observations.omitted.reason == OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED
    assert sdmx.subject_observations_calls == []
    assert not any(w.code == OBSERVATIONS_PROBE_FAILED for w in result.warnings)
    assert any(w.code == OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED for w in result.warnings)


# Candidates must use each candidate's own default-scope data for bound-unit checks.


_CAND_SV_A = "TEST/BulkObsCand_Population"
_CAND_SV_B = "TEST/BulkObsCand_Household"
# One entity more than the per-pair preview cap, so the candidate path
# exercises the ceiling-degrade branch whatever the cap's value.
_CAND_COUNTRIES = [
    f"country/CBLK{i}" for i in range(1, observation_settings().obs_preview_max_pairs + 2)
]
_CAND_NODES: dict = {
    _CAND_SV_A: {
        "label": "Cand Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Cand Population"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    # A different population type creates a second shape.
    _CAND_SV_B: {
        "label": "Cand Household",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Cand Household"}]},
            "populationType": {"nodes": [{"dcid": "Household"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    **{c: {"label": f"Country {c[-1]}"} for c in _CAND_COUNTRIES},
}
_CAND_DETECT_QUERY = "test bulk scope candidates population"
_CAND_DETECT: dict = {
    _CAND_DETECT_QUERY: {"svs": [_CAND_SV_A, _CAND_SV_B], "cosine_scores": [0.9, 0.85]}
}
_CAND_EARTH_CHILDREN = {"Earth": {"Country": _CAND_COUNTRIES}}


class _CandidatesUnitBindLLM:
    """Elect the first shape and mark the second plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=[1]), None


def test_placeless_candidates_with_bound_unit_never_hits_per_pair_ceiling():
    obs = {
        f"{sv}|{c}": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 10,
                "facetId": "F1",
                "unit": "CurrencyUSD",
                # The unit election, like `select_facet`, refuses a facet without one.
                "provenanceId": "dc/base/BulkCandSource",
            }
        ]
        for sv in (_CAND_SV_A, _CAND_SV_B)
        for c in _CAND_COUNTRIES
    }
    graph = FakeGraph(
        nodes=_CAND_NODES,
        detect=_CAND_DETECT,
        resolve={},
        obs=obs,
        children=_CAND_EARTH_CHILDREN,
    )
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            _CAND_SV_A: {c: [_row(c, facet_id="F1")] for c in _CAND_COUNTRIES},
            _CAND_SV_B: {c: [_row(c, facet_id="F1")] for c in _CAND_COUNTRIES},
        },
        scope_availability=graph.scope_availability,
    )
    # Declare USD as a currency so the per-candidate unit election can confirm it.
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    }
    result = asyncio.run(
        resolve_variable(
            _CAND_DETECT_QUERY,
            entities=[],
            date_request=None,
            detect_query=_CAND_DETECT_QUERY,
            role_query=_CAND_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidatesUnitBindLLM(),
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

    assert result.status in ("definite", "candidates"), (
        f"expected a genuine answer, got {result.status!r} "
        f"(no_data_reason={getattr(result, 'no_data_reason', None)!r})"
    )
    assert not any(w.code == OBSERVATIONS_PROBE_FAILED for w in result.warnings)


# UN System Data Commons variables read through the origin graph client.

_UN_SV = "undata/who/MORT_U5_RATE.AGE--Y0T4"
_UN_SV_B = "undata/who/MORT_U5_RATE.SEX--F"


class _DatedGraph:
    """Labels and node reads from `_GRAPH`; a dated facet read that answers `facets` and records
    its calls."""

    def __init__(self, facets: dict[str, dict[str, list[Facet]]]) -> None:
        self._facets = facets
        self.calls: list[tuple[list[str], list[str], bool]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, max_bytes=None):
        self.calls.append((list(stat_vars), list(entities), needs_dates))
        return {sv: self._facets.get(sv, {}) for sv in stat_vars}

    def node_labels_batch(self, dcids):
        return _GRAPH.node_labels_batch(dcids)

    def node_arc_targets_many(self, dcids, *, properties, deadline=None):
        return _GRAPH.node_arc_targets_many(dcids, properties=properties, deadline=deadline)


def _dated(facet_id: str, points: dict[str, float]) -> Facet:
    return Facet(
        earliest_date=min(points),
        latest_date=max(points),
        obs_count=len(points),
        facet_id=facet_id,
        provenance_id="undata/p/WHO",
        dates=tuple(points),
        values=tuple(points.values()),
    )


def _bulk_with_graph(sv_dcids, facets_by_sv_entity, *, graph, sdmx):
    scope = HarnessScope(places=_SCOPE_PLACES, sdmx=sdmx, facets_by_sv_entity=facets_by_sv_entity)
    return build_observations_planned(
        sv_dcids,
        {"observationAbout": _COUNTRIES},
        graph=graph,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        coverage=empty_coverage_snapshot(),
    )


_SV_B = "TEST/BulkObs_PopulationFemale"


def test_a_member_serves_no_row_where_its_own_probe_lacks_the_shared_facet():
    """Member A holds F1 at every country. Member B holds F1 at the first country only and F2
    elsewhere, where admission withheld F1. The scope read returns F1 rows for both members at
    every country; B serves only where its own probe admitted F1."""
    f1 = Facet(earliest_date="2023", latest_date="2023", obs_count=10, facet_id="F1")
    f2 = Facet(earliest_date="2023", latest_date="2023", obs_count=1, facet_id="F2")
    first, *rest = _COUNTRIES
    facets_by_sv_entity = {
        _SV: {c: [f1] for c in _COUNTRIES},
        _SV_B: {first: [f1], **{c: [f2] for c in rest}},
    }
    sdmx = FakeSdmxClient(
        subject_observations={
            sv: {c: [_row(c, facet_id="F1")] for c in _COUNTRIES} for sv in (_SV, _SV_B)
        }
    )

    observations, _ = _bulk_with_graph([_SV, _SV_B], facets_by_sv_entity, graph=_GRAPH, sdmx=sdmx)

    served = {(s.stat_var.ref, s.entities[0].ref) for s in observations.series}
    assert served == {(_SV, c) for c in _COUNTRIES} | {(_SV_B, first)}
    # Each row carries its own pair's probe count.
    assert observations.total_obs == 10 * len(_COUNTRIES) + 10


def test_a_row_from_another_facet_does_not_serve_under_the_chosen_facet():
    f1 = Facet(earliest_date="2023", latest_date="2023", obs_count=10, facet_id="F1")
    first, second, *_ = _COUNTRIES
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {first: [_row(first, facet_id="F0")], second: [_row(second, facet_id="F1")]}
        }
    )

    observations, _ = _bulk_with_graph(
        [_SV], {_SV: {c: [f1] for c in _COUNTRIES}}, graph=_GRAPH, sdmx=sdmx
    )

    assert [s.entities[0].ref for s in observations.series] == [second]


def test_a_row_without_a_facet_id_fails_the_bulk_fetch():
    f1 = Facet(earliest_date="2023", latest_date="2023", obs_count=10, facet_id="F1")
    first = _COUNTRIES[0]
    sdmx = FakeSdmxClient(subject_observations={_SV: {first: [_row(first, facet_id=None)]}})

    observations, warn = _bulk_with_graph(
        [_SV], {_SV: {c: [f1] for c in _COUNTRIES}}, graph=_GRAPH, sdmx=sdmx
    )

    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED


def _live_sdmx_serving(*places: str) -> LiveSdmxClient:
    """The real client and parser over a body holding one F1 row per place."""
    body = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    ) + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{_SV},{place},Person,Census,P1Y,dc/base,2023,5,,F1\r\n"
        for place in places
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return LiveSdmxClient(base="https://ours.example.org", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("place", ["NotApplicable", ""])
def test_a_row_without_a_place_fails_the_bulk_fetch(place: str):
    f1 = Facet(earliest_date="2023", latest_date="2023", obs_count=10, facet_id="F1")

    observations, warn = _bulk_with_graph(
        [_SV],
        {_SV: {c: [f1] for c in _COUNTRIES}},
        graph=_GRAPH,
        sdmx=_live_sdmx_serving(*_COUNTRIES, place),
    )

    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED


def test_the_un_branch_serves_a_member_only_where_its_own_probe_admitted_the_facet():
    """The graph read returns FU for both UN members at every country, but the second member's
    probe admitted FU only at the first country."""
    fu = Facet(earliest_date="2023", latest_date="2023", obs_count=1, facet_id="FU")
    first, *rest = _COUNTRIES
    graph = _DatedGraph(
        {sv: {c: [_dated("FU", {"2023": 1.0})] for c in _COUNTRIES} for sv in (_UN_SV, _UN_SV_B)}
    )

    observations, _ = _bulk_with_graph(
        [_UN_SV, _UN_SV_B],
        {_UN_SV: {c: [fu] for c in _COUNTRIES}, _UN_SV_B: {first: [fu], **{c: [] for c in rest}}},
        graph=graph,
        sdmx=FakeSdmxClient(subject_observations={}),
    )

    served = {(s.stat_var.ref, s.entities[0].ref) for s in observations.series}
    assert served == {(_UN_SV, c) for c in _COUNTRIES} | {(_UN_SV_B, first)}


def test_un_variable_reads_its_latest_points_through_the_graph_and_never_sdmx():
    un_facets = {
        c: [_dated("FU", {"2021": 40.0, "2023": 38.5}), _dated("FOther", {"2024": 1.0})]
        for c in _COUNTRIES
    }
    graph = _DatedGraph({_UN_SV: un_facets})
    sdmx = FakeSdmxClient(subject_observations={})

    observations, warn = _bulk_with_graph([_UN_SV], {_UN_SV: un_facets}, graph=graph, sdmx=sdmx)

    assert warn is None
    assert sdmx.subject_observations_calls == []
    assert graph.calls == [([_UN_SV], _COUNTRIES, True)]
    assert len(observations.series) == 5
    for series in observations.series:
        assert [(p.date, p.value) for p in series.points] == [("2023", 38.5)]


def test_mixed_variables_split_between_sdmx_and_the_graph():
    facets_by_entity = {
        c: [Facet(earliest_date="2023", latest_date="2023", obs_count=1, facet_id="F1")]
        for c in _COUNTRIES
    }
    graph = _DatedGraph({_UN_SV: {c: [_dated("F1", {"2022": 7.0})] for c in _COUNTRIES}})
    sdmx = FakeSdmxClient(
        subject_observations={_SV: {c: [_row(c, facet_id="F1")] for c in _COUNTRIES}}
    )

    observations, _ = _bulk_with_graph(
        [_SV, _UN_SV],
        {_SV: facets_by_entity, _UN_SV: facets_by_entity},
        graph=graph,
        sdmx=sdmx,
    )

    assert [call["variables"] for call in sdmx.subject_observations_calls] == [[_SV]]
    assert [call[0] for call in graph.calls] == [[_UN_SV]]
    assert {series.stat_var.ref for series in observations.series} == {_SV, _UN_SV}


def test_un_graph_read_failure_delivers_coverage_without_series():
    un_facets = {c: [_dated("FU", {"2023": 1.0})] for c in _COUNTRIES}

    class _FailingGraph(_DatedGraph):
        def observation_facets_batch(self, *args, **kwargs):
            raise GraphInfraError("simulated UN read failure")

    observations, warn = _bulk_with_graph(
        [_UN_SV],
        {_UN_SV: un_facets},
        graph=_FailingGraph({}),
        sdmx=FakeSdmxClient(subject_observations={}),
    )

    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED


def test_shared_facet_unit_and_method_carry_their_graph_names():
    """The unit and measurement method are labelled from the graph, so a native USD answer
    renders "US dollar" in place of the split dcid "Currency USD"."""
    facets_by_entity = {
        c: [
            Facet(
                earliest_date="2020",
                latest_date="2023",
                obs_count=10,
                facet_id="F1",
                provenance_id="dc/base/Src",
                unit="CurrencyUSD",
                measurement_method="dcAggregate/Usd",
            )
        ]
        for c in _COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={_SV: {c: [_row(c, facet_id="F1")] for c in _COUNTRIES}}
    )
    graph = FakeGraph(
        nodes={
            **_NODES,
            "CurrencyUSD": {"label": "US dollar"},
            "dcAggregate/Usd": {"label": "USD aggregate"},
        },
        obs={},
    )
    scope = HarnessScope(
        places=_SCOPE_PLACES, sdmx=sdmx, facets_by_sv_entity={_SV: facets_by_entity}
    )

    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=graph,
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations.series
    resolved = _resolved_facets([s.facet for s in observations.series], {})
    assert resolved["unit"].label == "US dollar"
    assert resolved["measurementMethod"].label == "USD aggregate"


def _eur_election(
    sv_dcids: list[str], *, converted: bool
) -> dict[str, tuple[dict[str, FacetSelection], ...]]:
    """Each StatVar elects its euro facet, natively or as the source of a dollar conversion."""
    selection = _facet_selection(
        axis="unit",
        ref="CurrencyEUR",
        selection_reason="most_observations" if converted else "native_currency_preferred",
        request=UnitRequest(currency="USD" if converted else "EUR", prices="current"),
    )
    return {sv: ({"unit": selection},) for sv in sv_dcids}


def _eur_facet(
    *, earliest: str, latest: str, dates: tuple[str, ...] = (), unit: str = "CurrencyEUR"
) -> Facet:
    return Facet(
        earliest_date=earliest,
        latest_date=latest,
        obs_count=4,
        facet_id="F1" if unit == "CurrencyEUR" else "F2",
        provenance_id="dc/base/Src",
        unit=unit,
        dates=dates,
        values=tuple(1.0 for _ in dates),
    )


def _scope_check(
    facets_by_sv_entity: dict[str, dict[str, list[Facet]]],
    *,
    window: TimeWindow | None,
    converted: bool,
    sdmx: FakeSdmxClient | None = None,
):
    """The scope read's preview and bound-unit evidence over a scope plan of
    `facets_by_sv_entity`, each StatVar electing its euro facet."""
    sv_dcids = list(facets_by_sv_entity)
    scope = HarnessScope(
        places=_SCOPE_PLACES,
        sdmx=sdmx or FakeSdmxClient(),
        facets_by_sv_entity=facets_by_sv_entity,
    )
    return unit_check_planned(
        sv_dcids,
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        window=window,
        scope=scope,
        unit_request=IdentityUnitRequest(currency="USD" if converted else "EUR", prices="current"),
        elected=_eur_election(sv_dcids, converted=converted),
        coverage=empty_coverage_snapshot(),
    )


def test_converted_scope_confirms_each_statvar_from_its_own_probe_without_a_read():
    """SV carries the elected euro facet at every place, SV_B nowhere: SV alone is confirmed,
    from the plan, and the preview, which cannot convert without the materialise clients,
    reads nothing."""
    sdmx = FakeSdmxClient()
    (observations, _warn), evidence = _scope_check(
        {
            _SV: {c: [_eur_facet(earliest="2020", latest="2023")] for c in _COUNTRIES},
            _SV_B: {c: [] for c in _COUNTRIES},
        },
        window=None,
        converted=True,
        sdmx=sdmx,
    )
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert evidence.confirmed == {_SV}
    assert sdmx.subject_observations_calls == []


def test_windowed_scope_reads_each_facet_span_against_the_window():
    """An end date inside the window proves a point; a span wholly before it disproves one."""
    window = TimeWindow(start_year=2021, end_year=2022)
    (observations, _warn), evidence = _scope_check(
        {
            _SV: {_COUNTRIES[0]: [_eur_facet(earliest="2019", latest="2021")]},
            _SV_B: {_COUNTRIES[0]: [_eur_facet(earliest="2015", latest="2018")]},
        },
        window=window,
        converted=False,
    )
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED
    assert evidence.confirmed == {_SV}


def test_windowed_scope_straddling_span_decides_by_dated_points_or_by_its_span():
    """A span straddling the window decides through its dated points when it carries them all,
    and through its span otherwise, the rule every election narrows a window with
    (`read_plan.has_points`)."""
    window = TimeWindow(start_year=2021, end_year=2022)
    _dated_preview, dated = _scope_check(
        {
            _SV: {
                _COUNTRIES[0]: [_eur_facet(earliest="2019", latest="2024", dates=("2019", "2024"))]
            }
        },
        window=window,
        converted=False,
    )
    assert dated.confirmed == frozenset()

    _undated_preview, undated = _scope_check(
        {_SV: {_COUNTRIES[0]: [_eur_facet(earliest="2019", latest="2024")]}},
        window=window,
        converted=False,
    )
    assert undated.confirmed == {_SV}


def _converted_scope_run():
    """Resolve a dollar ask over a defaulted scope whose places each hold one 2020 euro
    point, 0.9, under a provenance the DAC recipe converts at 0.9 euros per dollar. Returns
    the result and the scope read's SDMX double."""
    obs = {
        f"{_SV}|{c}": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 10,
                "facetId": "F1",
                "unit": "CurrencyEUR",
                "provenanceId": "dc/base/BulkScopeSrc",
            }
        ]
        for c in _COUNTRIES
    }
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            _SV: {
                c: [
                    replace(
                        _row(c, facet_id="F1", value=0.9, unit="CurrencyEUR"), time_period="2020"
                    )
                ]
                for c in _COUNTRIES
            }
        },
        scope_availability=graph.scope_availability,
    )
    materialise_graph, _unused_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
        **conversion_arcs(
            recipe_id="ConversionRecipeDAC",
            provenance_id="dc/base/BulkScopeSrc",
            currencies=["CurrencyUSD", "CurrencyEUR"],
        ),
    }
    server = RateTableServer(
        recipe_table_bodies(
            "ConversionRecipeDAC", rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}}
        ),
        fallback=lambda request: httpx.Response(404),
    )
    result = _run_pipeline(
        graph=graph,
        sdmx=sdmx,
        unit_text="US dollars",
        materialise_graph=materialise_graph,
        materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
    )
    return result, sdmx


def test_converted_scope_defaulted_query_previews_converted_points():
    """A converted unit over a defaulted scope converts each place's latest point: 0.9 euros
    in 2020 at 0.9 euros per dollar is one dollar, stated in dollars with its recipe."""
    result, sdmx = _converted_scope_run()

    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.omitted is None
    assert len(observations.series) == len(_COUNTRIES)
    for series in observations.series:
        assert [(p.date, p.value) for p in series.points] == [("2020", pytest.approx(1.0))]
        assert series.facet["unit"].ref == "CurrencyUSD"
        assert series.conversion is not None
        assert series.conversion.recipe_id == "ConversionRecipeDAC"
    assert [call["facet_ids"] for call in sdmx.subject_observations_calls] == [["F1"]]


class _FailingGraphReader:
    """A materialise graph whose every read raises `exc`."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __getattr__(self, name: str):
        def fail(*_args, **_kwargs):
            raise self._exc

        return fail


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        pytest.param(GraphReadError("transient"), OBSERVATIONS_BULK_FETCH_FAILED, id="graph"),
        pytest.param(TimeoutError("deadline"), OBSERVATIONS_TIMED_OUT, id="deadline"),
    ],
)
def test_a_converted_scope_whose_conversion_read_fails_omits_the_preview(
    monkeypatch, exc: Exception, code: str
):
    """The answer's gate passes, then the graph read the scope read's conversion makes
    fails, or crosses the deadline. The answer still resolves, its preview omitted under
    the failure's code with a matching warning, as a per-series read degrades."""
    monkeypatch.setattr(
        bulk_conversion,
        "conversion_graph_reader",
        lambda _graph, *, stat_vars: _FailingGraphReader(exc),
    )
    result, _sdmx = _converted_scope_run()

    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.omitted is not None
    assert observations.omitted.reason == code
    assert code in {w.code for w in result.warnings}


@pytest.mark.parametrize(
    ("convertible", "converted_places", "elected_provenance"),
    [
        pytest.param((_SV,), len(_COUNTRIES), "dc/base/Src", id="one-of-two-refuses"),
        pytest.param((), 0, "dc/base/Src", id="both-refuse"),
        pytest.param(
            (_SV,), len(_COUNTRIES), "dc/base/OtherSrc", id="elected-under-another-provenance"
        ),
    ],
)
def test_a_scope_conversion_names_the_statvars_it_leaves_out(
    convertible: tuple[str, ...], converted_places: int, elected_provenance: str
):
    """SV_B declares no Nominal qualifier, so its conversion refuses. The preview names it
    under `observations_converted_scope_unsupported`, as a warning beside SV's converted
    series or as the omission when nothing converts, and calls no place uncovered. A unit
    decision that elected the facet's unit under another provenance converts the read's
    rows under the read facet's provenance, as the gate checks them
    (`convertibility.facet_selections`)."""
    provenance = "dc/base/Src"
    sv_dcids = [_SV, _SV_B]
    sdmx = FakeSdmxClient(
        subject_observations={
            sv: {
                c: [
                    replace(
                        _row(c, facet_id="F1", value=0.9, unit="CurrencyEUR"),
                        variable=sv,
                        time_period="2020",
                    )
                ]
                for c in _COUNTRIES
            }
            for sv in sv_dcids
        }
    )
    scope = HarnessScope(
        places=_SCOPE_PLACES,
        sdmx=sdmx,
        facets_by_sv_entity={
            sv: {c: [_eur_facet(earliest="2020", latest="2020")] for c in _COUNTRIES}
            for sv in sv_dcids
        },
    )
    unit = _facet_selection(
        axis="unit",
        ref="CurrencyEUR",
        selection_reason="most_observations",
        request=UnitRequest(currency="USD", prices="current"),
    )
    source = _facet_selection(axis="provenance", ref=elected_provenance, selection_reason=None)
    server = RateTableServer(
        recipe_table_bodies(
            "ConversionRecipeDAC", rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}}
        ),
        fallback=lambda request: httpx.Response(404),
    )
    observations, warn = build_observations_planned(
        sv_dcids,
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=FakeGraphReader(
            arcs={
                **{sv: dict(CONVERTIBLE_STAT_VAR_ARCS) for sv in convertible},
                **conversion_arcs(
                    recipe_id="ConversionRecipeDAC",
                    provenance_id=provenance,
                    currencies=["CurrencyUSD", "CurrencyEUR"],
                ),
            }
        ),
        materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
        rate_cache=RateTableCache(),
        scope=scope,
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        elected={sv: ({"unit": unit, "provenance": source},) for sv in sv_dcids},
        coverage=empty_coverage_snapshot(),
    )

    assert {s.stat_var.ref for s in observations.series} == {*convertible}
    assert len(observations.series) == converted_places
    assert warn is not None
    assert warn.code == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED
    assert _SV_B in warn.message
    if converted_places:
        assert observations.omitted is None
        assert {s.facet["provenance"].ref for s in observations.series} == {provenance}
    else:
        assert observations.omitted is not None
        assert observations.omitted.reason == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED


def test_a_scope_conversion_names_a_place_whose_deflator_it_lacks():
    """At constant 2020 dollars the deflator table holds every place's 2020 value but
    BLK3's, so the conversion leaves out BLK3's one point. BLK3 holds data: the conversion's
    warning names it under `deflator_missing_omitted`, and no scope-partial note counts it as
    a place without data."""
    omitted_place = _COUNTRIES[2]
    sdmx = FakeSdmxClient(
        subject_observations={
            _SV: {
                c: [
                    replace(
                        _row(c, facet_id="F1", value=0.9, unit="CurrencyEUR"), time_period="2020"
                    )
                ]
                for c in _COUNTRIES
            }
        }
    )
    scope = HarnessScope(
        places=_SCOPE_PLACES,
        sdmx=sdmx,
        facets_by_sv_entity={
            _SV: {c: [_eur_facet(earliest="2020", latest="2020")] for c in _COUNTRIES}
        },
    )
    unit = _facet_selection(
        axis="unit",
        ref="CurrencyEUR",
        selection_reason="most_observations",
        request=UnitRequest(currency="USD", prices="constant", base_year=2020),
    )
    source = _facet_selection(axis="provenance", ref="dc/base/Src", selection_reason=None)
    server = RateTableServer(
        recipe_table_bodies(
            "ConversionRecipeDAC",
            rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}},
            deflators={c: {2019 if c == omitted_place else 2020: 100.0} for c in _COUNTRIES},
        ),
        fallback=lambda request: httpx.Response(404),
    )
    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=FakeGraphReader(
            arcs={
                _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
                **conversion_arcs(
                    recipe_id="ConversionRecipeDAC",
                    provenance_id="dc/base/Src",
                    currencies=["CurrencyUSD", "CurrencyEUR"],
                ),
            }
        ),
        materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
        rate_cache=RateTableCache(),
        scope=scope,
        unit_request=IdentityUnitRequest(currency="USD", prices="constant", base_year=2020),
        elected={_SV: ({"unit": unit, "provenance": source},)},
        coverage=empty_coverage_snapshot(),
    )

    assert observations.omitted is None
    assert {s.entities[0].ref for s in observations.series} == set(_COUNTRIES) - {omitted_place}
    assert warn is not None
    assert warn.code == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED
    assert f"{omitted_place} (deflator_missing_omitted)" in warn.message
    assert "no data" not in warn.message


def test_windowed_scope_defaulted_bound_unit_query_answers_definite_from_the_probe():
    """A dated scope preview is withheld, yet each probed facet's latest observation falls inside
    the window under the elected native unit, so the bound unit is confirmed."""
    obs = {
        f"{_SV}|{c}": [
            {
                "earliestDate": "2020",
                "latestDate": "2022",
                "obsCount": 3,
                "facetId": "F1",
                "unit": "CurrencyUSD",
                "provenanceId": "dc/base/BulkScopeSrc",
            }
        ]
        for c in _COUNTRIES
    }
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={}, obs=obs, children=_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={},
        scope_availability=graph.scope_availability,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        _SV: {"measurementQualifier": ("Nominal",)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    }
    result = _run_pipeline(
        graph=graph,
        sdmx=sdmx,
        unit_text="US dollars",
        date_request=DateRequest(window=TimeWindow(start_year=2021, end_year=2022), latest=False),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )

    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED
    assert sdmx.subject_observations_calls == []


def test_scope_unit_availability_confirms_only_the_elected_unit():
    """The StatVar's euro facet ends before the window and only its dollar facet reaches it.
    The scope plan narrowed to the elected euro unit holds no point in the window, so the
    StatVar is unconfirmed; electing dollars confirms it."""
    facets = {
        _SV: {
            c: [
                _eur_facet(earliest="2015", latest="2016"),
                _eur_facet(earliest="2015", latest="2022", unit="CurrencyUSD"),
            ]
            for c in _COUNTRIES
        }
    }
    window = TimeWindow(start_year=2021, end_year=2022)
    _preview, euro = _scope_check(facets, window=window, converted=False)
    assert euro.confirmed == frozenset()
    assert euro.serves_any([_SV]) is False

    dollar = _facet_selection(
        axis="unit",
        ref="CurrencyUSD",
        selection_reason="native_currency_preferred",
        request=UnitRequest(currency="USD", prices="current"),
    )
    _preview, usd = unit_check_planned(
        [_SV],
        {"observationAbout": _COUNTRIES},
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        window=window,
        scope=HarnessScope(places=_SCOPE_PLACES, sdmx=FakeSdmxClient(), facets_by_sv_entity=facets),
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        elected={_SV: ({"unit": dollar},)},
        coverage=empty_coverage_snapshot(),
    )
    assert usd.confirmed == {_SV}
