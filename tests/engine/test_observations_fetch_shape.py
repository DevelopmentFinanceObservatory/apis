"""Tests for observation preview inputs, truncation, facet selection, and error
handling in `_build_observations`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import search.engine.region.observations.fetch as region_fetch
from materialise import UpstreamResponseError
from search.engine.clients.graph import Facet
from search.engine.region.codes import (
    OBSERVATIONS_NOTHING_TO_PROBE,
    OBSERVATIONS_PROBE_FAILED,
)
from search.engine.retrieval.read_plan import plan_standard
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    build_observations_planned,
    harness_reader,
    materialise_double_from_obs,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# region/observations/fetch.py: _build_observations (fetch + shape)
# ---------------------------------------------------------------------------


def test_build_observations_none_for_empty_sv_dcids():
    obs, warn = build_observations_planned(
        [],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
    )
    assert obs.series == []
    assert obs.omitted is not None
    assert obs.omitted.reason == OBSERVATIONS_NOTHING_TO_PROBE
    assert warn is None


def test_build_observations_none_for_empty_entities():
    obs, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": []},
        graph=FakeGraph(),
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
    )
    assert obs.series == []
    assert obs.omitted is not None
    assert obs.omitted.reason == OBSERVATIONS_NOTHING_TO_PROBE
    assert warn is None


def test_build_observations_no_clients_supplied_is_silent_clients_unavailable():
    """Without either materialisation client, the preview returns
    `observations_clients_unavailable` without a warning.
    """
    obs, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
    )
    assert obs.series == []
    assert obs.omitted is not None
    assert obs.omitted.reason == "observations_clients_unavailable"
    assert warn is None


def test_build_observations_no_probed_candidates_is_silent_no_data():
    """A pair with no candidate facet is no planned series: nothing dispatches, no
    warning, and the preview returns `observations_nothing_to_probe`.
    """
    materialise_graph, materialise_sdmx = materialise_double_from_obs({})
    obs, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert obs.series == []
    assert obs.omitted is not None
    assert obs.omitted.reason == OBSERVATIONS_NOTHING_TO_PROBE
    assert warn is None


def test_build_observations_candidate_with_no_facet_dispatchable_still_warns():
    """The only candidate carries no facet id, the key every read dispatches by, so the plan
    serves no group: the preview reads nothing and reports `OBSERVATIONS_PROBE_FAILED`,
    naming no dispatchable facet.
    """
    materialise_graph, materialise_sdmx = materialise_double_from_obs({})
    plan = plan_standard(
        {
            ("Count_Person", "country/IND"): [
                Facet(earliest_date="2020", latest_date="2020", obs_count=1, dates=("2020",))
            ]
        },
        members=["Count_Person"],
        years=None,
    )
    obs, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        plan=plan,
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert obs.series == []
    assert obs.omitted is not None
    assert obs.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert "no (stat_var, entity) pair had a dispatchable facet" in warn.message


def test_build_observations_only_planned_series_are_fetched():
    """Only the 2 StatVars x 2 of 5 places the plan holds a facet for reach SDMX."""
    has_data = ("country/A", "country/C")
    no_data = ("country/B", "country/D", "country/E")
    pairs = {
        (sv, (("observationAbout", entity),)): [
            ObsFacetFixture(facet_id="f1", points=(("2020", 1.0),), provenance_id="prov1")
        ]
        for sv in ("SV1", "SV2")
        for entity in has_data
    }
    materialise_graph, materialise_sdmx, requests = build_materialise_double(pairs)

    observations, warn = build_observations_planned(
        ["SV1", "SV2"],
        {"observationAbout": [*has_data, *no_data]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == 4  # 2 StatVars x 2 places with data
    assert {e.ref for s in observations.series for e in s.entities} == set(has_data)
    fetched_entities = {r.url.params.get("c[observationAbout]") for r in requests}
    assert fetched_entities == set(has_data)


def test_build_observations_one_failed_series_still_serves_the_others_and_warns():
    """An individual fetch error leaves the other planned series served and produces
    `OBSERVATIONS_PROBE_FAILED`.
    """

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        entity = next(iter(request.role_bindings["observationAbout"]))
        if entity == "country/BAD":
            raise UpstreamResponseError("role divergence")
        return stub_preview_result(sv="Count_Person", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD", "country/GOOD"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD", "country/GOOD"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations is not None
    assert {s.entities[0].ref for s in observations.series} == {"country/GOOD"}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED


def test_build_observations_complete_under_threshold():
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2022",
                    "obsCount": 3,
                    "provenanceId": "dc/base/Src",
                    "observations": [
                        {"date": "2020", "value": 100.0},
                        {"date": "2021", "value": 101.0},
                        {"date": "2022", "value": 102.0},
                    ],
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
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert observations.truncated is False
    assert observations.total_obs == 3
    assert len(observations.series) == 1
    series = observations.series[0]
    assert series.stat_var.ref == "Count_Person"
    assert [e.ref for e in series.entities] == ["country/IND"]
    assert [e.role for e in series.entities] == ["observationAbout"]
    assert [p.date for p in series.points] == ["2020", "2021", "2022"]
    assert [p.value for p in series.points] == [100.0, 101.0, 102.0]
    assert series.source_refs == []  # provenance dcid unresolved against an empty nodes fixture


def test_build_observations_truncates_to_latest_k_over_threshold(search_settings):
    search_settings(obs_inline_max=2)
    search_settings(obs_preview_points=2)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2019",
                    "latestDate": "2022",
                    "obsCount": 4,
                    "provenanceId": "dc/base/Src",
                    "observations": [
                        {"date": "2019", "value": 9.0},
                        {"date": "2020", "value": 10.0},
                        {"date": "2021", "value": 11.0},
                        {"date": "2022", "value": 12.0},
                    ],
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
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert observations.truncated is True
    # `total_obs` comes from the full `PreviewResult.total_rows` count.
    assert observations.total_obs == 4
    series = observations.series[0]
    assert [p.date for p in series.points] == ["2021", "2022"]
    assert [p.value for p in series.points] == [11.0, 12.0]


def test_build_observations_selects_one_facet_never_merges():
    """Two facets tie at 2 observations, so the later 2021-2022 facet wins.
    Each pair contributes one selected facet.
    """
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2021",
                    "latestDate": "2022",
                    "obsCount": 2,
                    "provenanceId": "dc/base/newer",
                    "observations": [
                        {"date": "2021", "value": 11.0},
                        {"date": "2022", "value": 12.0},
                    ],
                },
                {
                    "earliestDate": "2019",
                    "latestDate": "2020",
                    "obsCount": 2,
                    "provenanceId": "dc/base/older",
                    "observations": [
                        {"date": "2019", "value": 9.0},
                        {"date": "2020", "value": 10.0},
                    ],
                },
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert len(observations.series) == 1
    series = observations.series[0]
    assert [p.date for p in series.points] == ["2021", "2022"]
    assert observations.total_obs == 2


def test_build_observations_same_date_across_facets_never_double_reports():
    """Two sources report different values for 2024. Selecting one facet
    keeps its points without duplicate dates.
    """
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/ETH": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2024",
                    "obsCount": 3,
                    "provenanceId": "dc/base/WorldDevelopmentIndicators",
                    "observations": [
                        {"date": "2022", "value": 20.0},
                        {"date": "2023", "value": 21.0},
                        {"date": "2024", "value": 24.6},
                    ],
                },
                {
                    "earliestDate": "2024",
                    "latestDate": "2024",
                    "obsCount": 1,
                    "provenanceId": "dc/base/EthiopiaStatistics",
                    "observations": [
                        {"date": "2024", "value": 28.9},
                    ],
                },
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/ETH"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    series = observations.series[0]
    dates = [p.date for p in series.points]
    assert dates == sorted(dates), "points stay chronological"
    assert len(dates) == len(set(dates)), "no duplicate dates across facets"
    # The 3-obsCount commitments facet beats the 1-obsCount disbursements
    # facet on materialise's own rank_candidates (graph-declared obsCount,
    # descending).
    assert dates == ["2022", "2023", "2024"]
    assert [p.value for p in series.points] == [20.0, 21.0, 24.6]
    assert observations.total_obs == 3


def test_build_observations_fails_open_on_graph_error():
    """A Search-native graph error (the label-lookup batch) fails open even
    once every pair's own materialise fetch already succeeded."""
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
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == "observations_probe_failed"
    assert warn is not None
    assert warn.code == "observations_probe_failed"


def test_build_observations_non_infra_error_propagates():
    """GraphInfraError degrades to the observations_probe_failed warning. Any
    other exception in the label-lookup logic propagates unchanged."""
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

    class _RaisingGraph:
        def node_labels_batch(self, *a, **k):
            raise TypeError("bug in the label lookup, not a graph error")

    with pytest.raises(TypeError):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=_RaisingGraph(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            coverage=empty_coverage_snapshot(),
        )
