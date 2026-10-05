"""Checks that every warning-bearing observation omission matches its Warning.code."""

from __future__ import annotations

from search.engine.clients.graph import Facet
from search.engine.errors import GraphInfraError
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_NO_DISPATCHABLE_FACET,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_PROBE_FAILED,
)
from search.engine.region.result import FacetAxisFilter
from search.models import TimeWindow
from search.observations import (
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
    observation_settings,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    HarnessScope,
    build_observations_planned,
    bulk_scope_series_planned,
    harness_reader,
    materialise_double_from_obs,
    sdmx_double,
)
from tests.fixtures import FakeGraph


def _assert_parity(observations, warn) -> None:
    assert warn is not None, "this branch is expected to carry a Warning"
    assert observations.omitted is not None, "a Warning-carrying result must also carry omitted"
    assert observations.omitted.reason == warn.code, (
        f"omitted.reason {observations.omitted.reason!r} != warn.code {warn.code!r}"
    )


def test_preview_too_large_parity():
    entities = [f"country/PAR{i}" for i in range(observation_settings().obs_preview_max_pairs + 1)]
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": entities},
        graph=FakeGraph(),
        materialise_graph=harness_reader(entities),
        materialise_sdmx=sdmx_double(),
        coverage=empty_coverage_snapshot(),
    )
    _assert_parity(observations, warn)
    assert warn.code == OBSERVATIONS_PREVIEW_TOO_LARGE


def test_no_matching_source_parity():
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/other",
                    "observations": [{"date": "2020", "value": 1.0}],
                }
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter={
            "provenance": FacetAxisFilter(
                predicate=lambda f: f.provenance_id == "dc/base/requested",
            )
        },
        coverage=empty_coverage_snapshot(),
    )
    _assert_parity(observations, warn)
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE


def test_probe_failed_parity():
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 1.0}],
                }
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(raise_on_call=True),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    _assert_parity(observations, warn)
    assert warn.code == OBSERVATIONS_PROBE_FAILED


_BULK_SV = "TEST/ParityBulk_Population"
_BULK_COUNTRIES = [f"country/PARBLK{i}" for i in range(1, 4)]
_BULK_SCOPE_PLACES = tuple(_BULK_COUNTRIES)


def test_bulk_fetch_failed_parity():
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id="F1")]
        for c in _BULK_COUNTRIES
    }
    sdmx = FakeSdmxClient(
        subject_observations={},
        raise_subject_observations=GraphInfraError("simulated SDMX failure"),
    )
    scope = HarnessScope(
        places=_BULK_SCOPE_PLACES,
        sdmx=sdmx,
        facets_by_sv_entity={_BULK_SV: facets_by_entity},
    )
    observations, warn = bulk_scope_series_planned(
        [_BULK_SV], _BULK_COUNTRIES, scope=scope, graph=FakeGraph(), facet_filter=None
    )
    _assert_parity(observations, warn)
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED


def test_windowed_scope_unsupported_parity():
    facets_by_entity = {
        c: [Facet(earliest_date="2020", latest_date="2020", obs_count=10, facet_id="F1")]
        for c in _BULK_COUNTRIES
    }
    scope = HarnessScope(
        places=_BULK_SCOPE_PLACES,
        sdmx=FakeSdmxClient(subject_observations={}),
        facets_by_sv_entity={_BULK_SV: facets_by_entity},
    )
    observations, warn = build_observations_planned(
        [_BULK_SV],
        {"observationAbout": _BULK_COUNTRIES},
        graph=FakeGraph(),
        materialise_graph=None,
        materialise_sdmx=None,
        scope=scope,
        window=TimeWindow(start_year=2020, end_year=2021),
        coverage=empty_coverage_snapshot(),
    )
    _assert_parity(observations, warn)
    assert warn.code == OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED


def test_no_dispatchable_facet_parity():
    facets_by_entity = {c: [] for c in _BULK_COUNTRIES}
    scope = HarnessScope(
        places=_BULK_SCOPE_PLACES,
        sdmx=FakeSdmxClient(subject_observations={}),
        facets_by_sv_entity={_BULK_SV: facets_by_entity},
    )
    observations, warn = bulk_scope_series_planned(
        [_BULK_SV], _BULK_COUNTRIES, scope=scope, graph=FakeGraph(), facet_filter=None
    )
    _assert_parity(observations, warn)
    assert warn.code == OBSERVATIONS_NO_DISPATCHABLE_FACET
