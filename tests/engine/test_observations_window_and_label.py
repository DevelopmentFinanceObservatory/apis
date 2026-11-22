"""Tests observation window filtering, facet selection, and entity labels."""

from __future__ import annotations

import dataclasses

import pytest

from origin_catalogue import bound_catalogue, current_catalogue
from search.engine.departures import spec_departure_facts
from search.engine.region.codes import OBSERVATIONS_NO_POINTS
from search.engine.region.observations.served_years import (
    no_points_in_served_years,
    year_count,
)
from search.engine.retrieval.facet_policy import NO_DECLARED_POLICIES
from search.engine.retrieval.read_plan import (
    PlanSeries,
    plan_pairs,
    read_in_years,
    serves_continuation,
)
from search.engine.spec_build import build_spec
from search.models import (
    TimeWindow,
)
from search.scope.composite import CompositeMint, served_components
from tests.engine._assemble_fixtures import _make_coverage, _make_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    build_observations_planned,
    materialise_double_from_obs,
)
from tests.fixtures import FakeGraph
from tests.scope import test_provider as provider

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


def _old_and_new_obs() -> dict:
    """A 2005 to 2015 facet of eleven points and a 2016 to 2020 facet of five."""
    return {
        "Count_Person|country/IND": [
            {
                "earliestDate": "2005",
                "latestDate": "2015",
                "obsCount": 11,
                "provenanceId": "dc/base/old",
                "observations": [{"date": str(year), "value": 1.0} for year in range(2005, 2016)],
            },
            {
                "earliestDate": "2016",
                "latestDate": "2020",
                "obsCount": 5,
                "provenanceId": "dc/base/new",
                "observations": [{"date": str(year), "value": 2.0} for year in range(2016, 2021)],
            },
        ],
    }


def _dates_read(window: TimeWindow | None, years: frozenset[int] | None) -> tuple[list[str], int]:
    """The dates `_old_and_new_obs` previews under `window`, read in `years`, and its
    `total_obs`."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_old_and_new_obs())
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=window,
        coverage=empty_coverage_snapshot(),
        years=years,
    )
    assert warn is None
    (series,) = observations.series
    return [p.date for p in series.points], observations.total_obs


def test_a_read_in_given_years_keeps_the_facet_its_own_window_elects():
    """Over every date the eleven-point facet leads; over 2015 to 2017 alone the five-point
    one covers more years and leads. Read in those years, the preview keeps the facet its
    own window elects and shows its points there."""
    span = TimeWindow(start_year=2015, end_year=2017)
    assert _dates_read(span, None) == (["2016", "2017"], 2)

    assert _dates_read(None, frozenset({2015, 2016, 2017})) == (["2015"], 1)


def test_a_read_in_given_years_collapses_its_facets_as_a_mint_of_its_window_does():
    """`fA` and `fB`, one source, contest 2020. Over every date `fA`'s seven points outrank
    `fB`'s two; over 2019 and 2020 alone `fB`'s two outrank `fA`'s one. A mint collapses
    under its window and keeps the served years afterwards, so read in those years the
    preview serves `fA`'s 2020 value, as its read of every date does there."""
    obs = {
        "Count_Person|country/IND": [
            {
                "facetId": facet_id,
                "earliestDate": str(years[0]),
                "latestDate": str(years[-1]),
                "obsCount": len(years),
                "provenanceId": "dc/base/same",
                "observations": [{"date": str(year), "value": value} for year in years],
            }
            for facet_id, years, value in (
                ("fA", [2010, 2011, 2012, 2013, 2014, 2015, 2020], 1.0),
                ("fB", [2019, 2020], 2.0),
            )
        ],
    }

    def read(years: frozenset[int] | None) -> list[tuple[str, float | None]]:
        materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
        observations, _warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            coverage=empty_coverage_snapshot(),
            years=years,
        )
        (series,) = observations.series
        return [(p.date, p.value) for p in series.points]

    served = frozenset({2019, 2020})
    whole = [point for point in read(None) if int(point[0]) in served]

    assert read(served) == whole == [("2019", 2.0), ("2020", 1.0)]


@pytest.mark.parametrize(
    ("years", "max_rows", "dates"),
    [
        pytest.param(frozenset({2018, 2019, 2020}), None, ["2018", "2019", "2020"], id="tail"),
        # The read's row limit takes the latest rows of the years read, not of the window.
        pytest.param(frozenset({2016, 2017}), 2, ["2016", "2017"], id="row-limit"),
        # 2017 and 2018 lie inside the span the read takes and outside the years.
        pytest.param(frozenset({2016, 2019, 2020}), None, ["2016", "2019", "2020"], id="gap"),
    ],
)
def test_a_read_in_given_years_shows_and_counts_their_points_alone(
    search_settings, years, max_rows, dates
):
    if max_rows is not None:
        search_settings(obs_preview_max_rows=max_rows)

    assert _dates_read(TimeWindow(start_year=2016), years) == (dates, len(dates))


@pytest.mark.parametrize(
    ("window", "years", "served"),
    [
        # A plan over 2016 onwards read in 2010 alone holds no year to read.
        pytest.param(TimeWindow(start_year=2016), frozenset({2010}), None, id="outside-window"),
        # Every date holds 2022, and no series holds a point in it.
        pytest.param(None, frozenset({2022}), frozenset({2022}), id="no-series-in-years"),
        # The plan holds 2022 of the two years alone, and no series holds a point in it.
        pytest.param(
            TimeWindow(start_year=2016),
            frozenset({2010, 2022}),
            frozenset({2022}),
            id="no-series-in-the-years-the-plan-holds",
        ),
    ],
)
def test_a_read_in_years_no_series_holds_omits_with_no_points_and_no_read(window, years, served):
    """A preview whose series hold no point in the years its plan serves omits as every
    read holding none in them does (`served_years.no_points_in_served_years`), counting
    those years."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_old_and_new_obs())
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=window,
        coverage=empty_coverage_snapshot(),
        years=years,
    )
    assert warn is None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_POINTS
    assert observations.omitted.dispatched == 0
    if served is not None:
        assert observations == no_points_in_served_years(served, dispatched=0, empty=0)


def _continued_in_years(estimated_years: list[int], years: frozenset[int]):
    """A read of Kenya in `years` under a 2010 to 2027 window, where a recorded facet holds
    2010 to 2016 and 2019 and an `Estimated` facet of one source continues it in
    `estimated_years`: the observations and the warning."""
    preview, _reader = _continued_preview_and_reader(estimated_years, years)
    return preview


def _continued_preview_and_reader(estimated_years: list[int], years: frozenset[int]):
    """`_continued_in_years`' preview and the graph double it read the facets from."""

    def facet(dates: list[int], value: float, **method: str) -> dict:
        return {
            "earliestDate": str(dates[0]),
            "latestDate": str(dates[-1]),
            "obsCount": len(dates),
            "provenanceId": "dc/base/Src",
            **({"measurementMethod": method["method"]} if method else {}),
            "observations": [{"date": str(year), "value": value} for year in dates],
        }

    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/KEN": [
                facet([*range(2010, 2017), 2019], 1.0),
                facet(estimated_years, 2.0, method="Estimated"),
            ]
        }
    )
    preview = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=TimeWindow(start_year=2010, end_year=2027),
        coverage=empty_coverage_snapshot(),
        years=years,
    )
    return preview, materialise_graph


def test_a_read_in_years_states_estimates_only_where_a_served_point_is_one():
    """The estimates continue the recorded series in 2020 and from 2022. Read in 2019 and
    2021, the span holds the 2020 estimate, which 2021's absence leaves out, so the series
    serves the recorded 2019 alone and holds no estimate."""
    observations, warn = _continued_in_years([2020, *range(2022, 2030)], frozenset({2019, 2021}))

    assert warn is None
    (series,) = observations.series
    assert [(p.date, p.value) for p in series.points] == [("2019", 1.0)]
    assert series.holds_estimates is False


@pytest.mark.parametrize(
    ("years", "continued"),
    [
        pytest.param(frozenset({2019, 2021}), False, id="estimate-between-served-years"),
        pytest.param(frozenset({2019, 2020}), True, id="estimate-in-a-served-year"),
    ],
)
def test_a_plan_read_in_years_states_a_continuation_only_in_a_served_year(years, continued):
    """The estimates continue the recorded series in 2020 and from 2022, so the plan over
    its 2010 to 2027 window, its facets carrying their dates, serves them. Narrowed to the
    served years, it states the continuation only where an estimate falls in one of them,
    as the preview shows it, so a part reading the recorded 2019 alone states no modelled
    values."""
    (observations, _warn), materialise_graph = _continued_preview_and_reader(
        [2020, *range(2022, 2030)], years
    )
    candidates = materialise_graph.variable_facets_batch(
        stat_var="Count_Person", entities=["country/KEN"], needs_dates=True
    )["country/KEN"]
    plan = plan_pairs(
        [
            PlanSeries(
                stat_var="Count_Person",
                role_tuple=(("observationAbout", "country/KEN"),),
                candidates=tuple(candidates),
            )
        ],
        members=["Count_Person"],
        years=TimeWindow(start_year=2010, end_year=2027),
        declared=NO_DECLARED_POLICIES,
    )
    narrowed = read_in_years(plan, years)
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/KEN")
    previewed = build_spec(
        composite=None,
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
        plan=plan,
        preview_years=years,
    )
    modelled = [
        fact
        for fact in spec_departure_facts(previewed, operations=())
        if fact.code == "modelled_values"
    ]

    # A composite answer minted over the table identity's plan, its components serving
    # these years, states the continuation in them alone, as its component reads do.
    facts = provider._dac1_facts(provider._DAC1_TOTAL_ODA)
    identity = provider._dac1_identity(
        provider._DAC1_TOTAL_ODA, (provider._USA,), window=(2010, 2024), headline=True
    )
    with bound_catalogue(current_catalogue()):
        served = served_components(identity.model_dump(mode="json", exclude_none=True), facts=facts)
    mint = CompositeMint(
        identity=identity,
        spec_id="composite",
        served=tuple(dataclasses.replace(item, served=years) for item in served),
        facts=facts,
    )
    minted = build_spec(
        composite=mint,
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
        plan=plan,
    )

    assert narrowed is not None
    assert serves_continuation(plan) is True
    assert serves_continuation(narrowed) is continued
    assert any(series.holds_estimates for series in observations.series) is continued
    assert bool(modelled) is continued
    assert minted.resolved.continued is continued


def test_a_series_the_continuation_cuts_from_its_years_is_served_empty():
    """The estimates hold 2017 to 2029, so the series holds a point in 2018. A mint of the
    window cuts the estimates up to the recorded 2019, so read in 2018 the series serves
    nothing, and the preview counts it fetched and empty, with no warning."""
    observations, warn = _continued_in_years(list(range(2017, 2030)), frozenset({2018}))

    assert warn is None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_POINTS
    assert (observations.omitted.dispatched, observations.omitted.empty) == (1, 1)
    assert observations == no_points_in_served_years(
        frozenset({2018}), dispatched=1, empty=1, hard_failed=0, facet_rejected=0
    )


@pytest.mark.parametrize(
    ("years", "counted"), [({2018}, "1 year"), ({2018, 2020, 2021}, "3 years")]
)
def test_served_years_are_counted_with_their_noun(years, counted):
    assert year_count(frozenset(years)) == counted


def _recorded_and_estimated_obs() -> dict:
    """Recorded values to 2024 with no method, and projections from 2024 to 2032 in an
    `Estimated` facet. Each projection is its year plus 0.5, so a point shows which facet
    served it, and both facets hold 2024."""
    recorded_years = range(1970, 2025)
    estimated_years = range(2024, 2033)
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
                "earliestDate": "2024",
                "latestDate": "2032",
                "obsCount": len(estimated_years),
                "provenanceId": "dc/base/WorldBankIDS",
                "measurementMethod": "Estimated",
                "observations": [{"date": str(y), "value": y + 0.5} for y in estimated_years],
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


def test_build_observations_window_past_the_recorded_years_continues_with_the_estimates():
    """A 2020-2027 window runs past the recorded series' last year, so the recorded facet
    serves 2020 to 2024 and the `Estimated` facet continuing it serves 2025 to 2027. The
    recorded value wins 2024, which both hold, and the series carries the recorded facet."""
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
    assert [(p.date, p.value) for p in series.points] == [
        ("2020", 2020.0),
        ("2021", 2021.0),
        ("2022", 2022.0),
        ("2023", 2023.0),
        ("2024", 2024.0),
        ("2025", 2025.5),
        ("2026", 2026.5),
        ("2027", 2027.5),
    ]
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
