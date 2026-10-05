"""Tests observation window filtering, facet selection, and entity labels."""

from __future__ import annotations

from search.models import (
    TimeWindow,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    build_observations_planned,
    materialise_double_from_obs,
)
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# region/observations/fetch.py: _build_observations -- window filtering (the spec's own
# resolved requested window, e.g. Spec.resolved.coverage.window for "since 2015").
# ---------------------------------------------------------------------------


def test_build_observations_window_filters_points_before_start_year():
    """The 2015 window filters earlier points from both series and `total_obs`."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2010",
                    "latestDate": "2020",
                    "obsCount": 3,
                    "provenanceId": "dc/base/Src",
                    "observations": [
                        {"date": "2010", "value": 1.0},
                        {"date": "2015", "value": 2.0},
                        {"date": "2020", "value": 3.0},
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
        window=TimeWindow(start_year=2015),
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    series = observations.series[0]
    assert [p.date for p in series.points] == ["2015", "2020"]
    assert observations.total_obs == 2


def test_build_observations_window_years_decide_between_facets_that_both_reach_the_window():
    """The 2018 to 2020 facet wins, although the 2005 to 2015 one holds more observations.

    Both facets reach 2015+, so the window keeps both, and the one whose span covers more
    of the window's years leads: three against one. Selection ranks graph metadata before
    filtering points."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    # obsCount=3 is the highest, but its span covers one year of the
                    # 2015+ window.
                    "earliestDate": "2005",
                    "latestDate": "2015",
                    "obsCount": 3,
                    "provenanceId": "dc/base/old",
                    "observations": [
                        {"date": "2005", "value": 1.0},
                        {"date": "2010", "value": 2.0},
                        {"date": "2015", "value": 3.0},
                    ],
                },
                {
                    "earliestDate": "2018",
                    "latestDate": "2020",
                    "obsCount": 2,
                    "provenanceId": "dc/base/new",
                    "observations": [
                        {"date": "2018", "value": 4.0},
                        {"date": "2020", "value": 5.0},
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
        window=TimeWindow(start_year=2015),
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    series = observations.series[0]
    # The dispatched 2018 to 2020 facet contributes both of its points.
    assert [p.date for p in series.points] == ["2018", "2020"]
    assert observations.total_obs == 2


def _recorded_and_estimated_obs() -> dict:
    """Recorded values to 2024 with no method, and projections to 2032 in an `Estimated` facet."""
    recorded_years = range(1970, 2025)
    estimated_years = range(2025, 2033)
    return {
        "Count_Person|country/KEN": [
            {
                "earliestDate": "1970",
                "latestDate": "2024",
                "obsCount": len(recorded_years),
                "provenanceId": "dc/base/WorldBankIDS",
                "observations": [{"date": str(y), "value": float(y)} for y in recorded_years],
            },
            {
                "earliestDate": "2025",
                "latestDate": "2032",
                "obsCount": len(estimated_years),
                "provenanceId": "dc/base/WorldBankIDS",
                "measurementMethod": "Estimated",
                "observations": [{"date": str(y), "value": float(y)} for y in estimated_years],
            },
        ],
    }


def test_build_observations_window_reaching_only_projections_serves_the_estimated_facet():
    """A 2027 window reaches only the `Estimated` facet, so its 2027 point is served."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_recorded_and_estimated_obs())

    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=TimeWindow(start_year=2027, end_year=2027),
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    series = observations.series[0]
    assert [p.date for p in series.points] == ["2027"]
    assert series.facet["measurementMethod"].ref == "Estimated"


def test_build_observations_window_overlapping_recorded_years_serves_the_recorded_facet():
    """A 2020-2027 window overlaps both facets, so the recorded facet wins on count."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_recorded_and_estimated_obs())

    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=TimeWindow(start_year=2020, end_year=2027),
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    series = observations.series[0]
    assert [p.date for p in series.points] == ["2020", "2021", "2022", "2023", "2024"]
    assert "measurementMethod" not in series.facet


def test_build_observations_facet_selection_deterministic_under_full_tie():
    """Equal counts and dates choose the lower provenance id on repeated calls."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/aaa",
                    "observations": [{"date": "2020", "value": 100.0}],
                },
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/zzz",
                    "observations": [{"date": "2020", "value": 200.0}],
                },
            ],
        }
    )
    for _ in range(5):
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
        # "aaa" sorts before "zzz" lexicographically -- rank_candidates'
        # own ascending provenance_id tie-break makes it the winner every
        # time.
        assert observations.series[0].points[0].value == 100.0


# ---------------------------------------------------------------------------
# region/observations/fetch.py: _build_observations -- entity_label (label-self-contained
# payload, needed because dev-finance series are donor-keyed and the donor is
# otherwise unnamed).
# ---------------------------------------------------------------------------


def test_build_observations_populates_entity_label_when_in_scope():
    """The entity label comes from the batched `node_labels_batch` read."""
    graph = FakeGraph(nodes={"country/USA": {"label": "United States"}})
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/USA": [
                {
                    "earliestDate": "2024",
                    "latestDate": "2024",
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2024", "value": 24607887.18}],
                }
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/USA"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    series = observations.series[0]
    assert series.entities[0].ref == "country/USA"
    assert series.entities[0].label == "United States"


def test_build_observations_entity_label_none_when_not_in_scope():
    """An entity without a graph label keeps `label=None` for the frontend fallback."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 100.0}],
                }
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(nodes={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert observations.series[0].entities[0].label is None


def test_build_observations_entity_label_survives_truncation(search_settings):
    """The truncation rebuild preserves the entity label."""
    search_settings(obs_inline_max=2)
    search_settings(obs_preview_points=2)
    graph = FakeGraph(nodes={"country/IND": {"label": "India"}})
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
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert observations.truncated is True
    assert observations.series[0].entities[0].label == "India"
