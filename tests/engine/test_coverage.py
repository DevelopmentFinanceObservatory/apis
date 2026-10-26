"""Tests coverage from a read plan, source counts, and coverage headlines."""

from __future__ import annotations

import pytest

from dc_graph.client import _parse_ordered_facets
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import RoleFacetCandidate
from search.engine.extract import DateRequest
from search.engine.retrieval.coverage import (
    MEMBER_VARIABLES_LABEL,
    CoverageForm,
    answer_coverage,
    citable_source_ids,
    headline_from_coverage_index,
)
from search.engine.retrieval.facet_election import merge_role_candidates, select_role_candidate
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    PlanSeries,
    plan_standard,
    plan_years,
)
from search.models import BreadthDim, CoverageBare, CoverageBreadth, CoverageExact, TimeWindow
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.engine._harness import dimensions_of
from tests.engine._role_candidates import role_candidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_PROVENANCE = "dc/base/CoverageFixture"


def _facet(
    obs_count: int = 10,
    earliest: str = "2010",
    latest: str = "2024",
    dates: list[str] | None = None,
    provenance_id: str | None = _PROVENANCE,
    facet_id: str | None = "f1",
) -> Facet:
    return Facet(
        earliest_date=earliest,
        latest_date=latest,
        obs_count=obs_count,
        dates=tuple(dates or ()),
        provenance_id=provenance_id,
        facet_id=facet_id,
    )


def _one_series(facets: list[Facet]) -> dict[tuple[str, str], list[Facet]]:
    """`facets` as the candidates of one preview series."""
    return {("sv", "country/KEN"): list(facets)}


def _series_each(facets: list[Facet]) -> dict[tuple[str, str], list[Facet]]:
    """Each of `facets` as the only candidate of its own variable's series. The variables
    elect one group, so every facet sharing the first's id and provenance is served."""
    return {(f"sv{i}", "country/KEN"): [facet] for i, facet in enumerate(facets)}


def _served_each(facets: list[Facet], **form) -> object:
    """Coverage of a plan serving each of `facets` at its own variable's series, whatever
    their sources: coverage counts what a plan serves, apart from the election that chose
    it (`plan_standard` serves one group, so one source)."""
    series = tuple(
        PlanSeries(
            stat_var=f"sv{i}",
            role_tuple=(("observationAbout", "country/KEN"),),
            candidates=(facet,),
            group=(facet,),
        )
        for i, facet in enumerate(facets)
    )
    plan = PairPlan(members=tuple(s.stat_var for s in series), years=None, series=series)
    return answer_coverage(plan, CoverageForm(**form))


def _coverage(
    series: dict[tuple[str, str], list[Facet]],
    date_request: DateRequest | None = None,
    *,
    scope: bool = False,
    **form,
):
    """Coverage of the per-series plan a resolver builds over `series`."""
    stat_vars = list(dict.fromkeys(sv for sv, _entity in series))
    plan = plan_standard(series, members=stat_vars, years=plan_years(date_request), scope=scope)
    return answer_coverage(plan, CoverageForm(**form))


def _candidate(facet_id: str = "f1", latest: str | None = "2024") -> RoleFacetCandidate:
    return role_candidate(facet_id=facet_id, role_tuples=frozenset(), latest_date=latest)


def _one_index(candidates: list[RoleFacetCandidate]) -> dict[str, list[RoleFacetCandidate]]:
    """`candidates` as one StatVar's role index."""
    return {"sv": list(candidates)}


# ---------------------------------------------------------------------------
# No-request path (full history)
# ---------------------------------------------------------------------------


def test_no_request_returns_coverage_exact() -> None:
    cov = _coverage(_one_series([_facet(obs_count=402), _facet(obs_count=18)]))
    assert isinstance(cov, CoverageExact)
    assert cov.kind == "exact"


def test_no_request_observation_count_is_sum_of_obs_count() -> None:
    facets = [_facet(obs_count=100), _facet(obs_count=200), _facet(obs_count=50)]
    cov = _coverage(_series_each(facets))
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 350


def test_no_request_window_is_none() -> None:
    cov = _coverage(_one_series([_facet(obs_count=402)]))
    assert isinstance(cov, CoverageExact)
    assert cov.window is None


def test_no_request_has_data_true_when_obs_present() -> None:
    cov = _coverage(_one_series([_facet(obs_count=402)]))
    assert cov.has_data is True


def test_no_request_dimensions_two() -> None:
    cov = _coverage(_one_series([_facet(obs_count=402)]))
    assert isinstance(cov, CoverageExact)
    assert len(dimensions_of(cov)) == 2


# ---------------------------------------------------------------------------
# Concrete window: full cover (all dates inside)
# ---------------------------------------------------------------------------


def test_concrete_window_full_cover_counts_all_dates() -> None:
    dates_in = ["2015", "2016", "2017", "2018", "2020"]
    facets = [_facet(obs_count=5, dates=dates_in)]
    req = DateRequest(window=TimeWindow(start_year=2010, end_year=2024), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 5


def test_a_range_states_the_years_it_serves() -> None:
    """ "2010 to 2024" served in 2015 and 2016 states 2015 to 2016."""
    req = DateRequest(window=TimeWindow(start_year=2010, end_year=2024), latest=False)
    cov = _coverage(_one_series([_facet(obs_count=2, dates=["2015", "2016"])]), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.window == TimeWindow(start_year=2015, end_year=2016)


def test_concrete_window_full_cover_dimensions_two() -> None:
    req = DateRequest(window=TimeWindow(start_year=2010, end_year=2024), latest=False)
    cov = _coverage(_one_series([_facet(obs_count=2, dates=["2015", "2016"])]), date_request=req)
    assert len(dimensions_of(cov)) == 2


# ---------------------------------------------------------------------------
# Partial overlap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dates", "obs_count", "window", "expected_count"),
    [
        pytest.param(
            ["2012", "2014", "2016", "2018", "2022"],
            5,
            TimeWindow(start_year=2014, end_year=2018),
            3,
            id="five-dates-three-in-window",
        ),
        pytest.param(
            ["2012", "2014", "2016"],
            3,
            TimeWindow(start_year=2014, end_year=2016),
            2,
            id="three-dates-two-in-window",
        ),
    ],
)
def test_partial_overlap(
    dates: list[str], obs_count: int, window: TimeWindow, expected_count: int
) -> None:
    req = DateRequest(window=window, latest=False)
    cov = _coverage(_one_series([_facet(obs_count=obs_count, dates=dates)]), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == expected_count
    assert cov.has_data is True
    assert len(dimensions_of(cov)) == 2


# ---------------------------------------------------------------------------
# Exclude all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dates", "obs_count"),
    [
        pytest.param(["2000", "2001", "2002"], 3, id="three-date-zero-count"),
        pytest.param(["2000"], 1, id="one-date-dimensions"),
    ],
)
def test_exclude_all_window(dates: list[str], obs_count: int) -> None:
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    cov = _coverage(_one_series([_facet(obs_count=obs_count, dates=dates)]), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 0
    assert len(dimensions_of(cov)) == 2
    # The observations dimension narrows with observation_count, not just
    # positionally present: a window that excludes every date must not
    # print a stale lifetime obs_count beside a zero observation_count.
    obs_dim = next(d for d in dimensions_of(cov) if d.label == "observations")
    assert obs_dim.count == 0


def test_exclude_all_window_has_data_false() -> None:
    """Exact windowed coverage reports no data when every observation is outside it."""
    dates = ["2000", "2001"]
    facets = [_facet(obs_count=2, dates=dates)]
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert cov.has_data is False


def test_windowed_observations_dim_agrees_with_observation_count_not_lifetime_max() -> None:
    """A facet with 33 lifetime observations but none inside the requested
    window must print an `observations` dimension of 0, matching
    observation_count, rather than the pre-window lifetime max of 33
    (the "observations: 33 beside observation_count: 0" defect)."""
    facets = [_facet(obs_count=33, dates=[str(y) for y in range(1970, 1970 + 33)])]
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 0
    obs_dim = next(d for d in dimensions_of(cov) if d.label == "observations")
    assert obs_dim.count == 0


def test_windowed_observations_dim_counts_the_richest_facet_in_window() -> None:
    """Two facets, only one of which has dates inside the window: the
    observations dimension takes the max per-facet in-window count, the
    same "richest facet" identity it applies lifetime, just narrowed to
    the window."""
    facets = [
        _facet(obs_count=10, dates=["2010", "2011"]),
        _facet(obs_count=10, dates=["2016", "2017", "2018"]),
    ]
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    cov = _coverage(_series_each(facets), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 3
    obs_dim = next(d for d in dimensions_of(cov) if d.label == "observations")
    assert obs_dim.count == 3


def test_unwindowed_observations_dim_stays_lifetime_max() -> None:
    """No date_request at all: the observations dimension is the largest series' lifetime
    count, since there is no windowed observation_count for it to disagree with."""
    facets = [_facet(obs_count=34), _facet(obs_count=402), _facet(obs_count=18)]
    cov = _coverage(_series_each(facets))
    assert isinstance(cov, CoverageExact)
    obs_dim = next(d for d in dimensions_of(cov) if d.label == "observations")
    assert obs_dim.count == 402


def test_windowed_without_dates_observations_dim_stays_lifetime() -> None:
    """The breadth (windowed-without-dates) arm keeps a lifetime observations
    dimension: there is no windowed observation_count on CoverageBreadth for
    it to disagree with, and incomplete dates leave no in-window count to
    report instead."""
    facets = [_facet(obs_count=40, earliest="2010", latest="2020", dates=[])]
    req = DateRequest(window=TimeWindow(start_year=2012, end_year=2018), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageBreadth)
    obs_dim = next(d for d in dimensions_of(cov) if d.label == "observations")
    assert obs_dim.count == 40


# ---------------------------------------------------------------------------
# latest → TimeWindow(maxYear, maxYear) and in-that-year count
# ---------------------------------------------------------------------------


def _one_provenance_facets(second_provenance: str) -> list[Facet]:
    """A 55-point facet to 2024 that the ranking elects, and a 7-point facet to 2026."""
    return [
        _facet(
            obs_count=55,
            latest="2024",
            dates=[str(y) for y in range(1970, 2025)],
            provenance_id="prov-a",
            facet_id="f-long",
        ),
        _facet(
            obs_count=7,
            earliest="2020",
            latest="2026",
            dates=[str(y) for y in range(2020, 2027)],
            provenance_id=second_provenance,
            facet_id="f-short",
        ),
    ]


def test_latest_dates_a_series_by_its_joint_dispatch_groups_latest_point() -> None:
    """Both facets share provenance, method and unit, so the preview dispatches both and a
    `latest` read keeps their latest period, 2026: coverage names 2026 and counts that one
    point, where the elected facet alone would read 2024."""
    req = DateRequest(window=None, latest=True)
    cov = _coverage(_one_series(_one_provenance_facets("prov-a")), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.window == TimeWindow(start_year=2026, end_year=2026)
    assert cov.observation_count == 1


def test_latest_leaves_out_a_facet_of_another_provenance() -> None:
    """A facet from another provenance is outside the dispatched group, so the elected
    facet's 2024 point dates the series."""
    req = DateRequest(window=None, latest=True)
    cov = _coverage(_one_series(_one_provenance_facets("prov-b")), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.window == TimeWindow(start_year=2024, end_year=2024)
    assert cov.observation_count == 1


def test_dateless_counts_the_points_the_joint_dispatch_group_collapses_to() -> None:
    """A dateless preview serves every date of the group once, since materialise collapses
    the two facets' rows to one per date: 1970 to 2026, 57 points (55 + 7 less the five
    shared years 2020 to 2024)."""
    cov = _coverage(_one_series(_one_provenance_facets("prov-a")))
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 57
    sources_dim = next(d for d in dimensions_of(cov) if d.label == "sources")
    assert sources_dim.count == 1


def test_latest_spans_each_series_own_latest_point() -> None:
    facets = [
        _facet(obs_count=3, latest="2021", dates=["2019", "2020", "2021"]),
        _facet(obs_count=2, latest="2023", dates=["2022", "2023"]),
    ]
    req = DateRequest(window=None, latest=True)
    cov = _coverage(_series_each(facets), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.window == TimeWindow(start_year=2021, end_year=2023)
    assert cov.observation_count == 2


def test_latest_dimensions_two() -> None:
    facets = [_facet(obs_count=3, latest="2022", dates=["2022", "2020", "2021"])]
    req = DateRequest(window=None, latest=True)
    cov = _coverage(_one_series(facets), date_request=req)
    assert len(dimensions_of(cov)) == 2


# ---------------------------------------------------------------------------
# No facets → CoverageBare
# ---------------------------------------------------------------------------


def test_no_facets_returns_coverage_bare() -> None:
    cov = _coverage(_one_series([]))
    assert isinstance(cov, CoverageBare)


def test_no_facets_has_data_false() -> None:
    cov = _coverage(_one_series([]))
    assert cov.has_data is False


def test_default_labels_are_sources_and_observations() -> None:
    cov = _coverage(_one_series([_facet(obs_count=10)]))
    assert isinstance(cov, CoverageExact)
    dim_labels = [d.label for d in dimensions_of(cov)]
    assert dim_labels[0] == "sources"
    assert dim_labels[1] == "observations"


def test_sources_dim_count_equals_distinct_provenance_count() -> None:
    """Three served facets citing two distinct provenances count as two sources,
    not three."""
    facets = [
        _facet(obs_count=100, provenance_id="prov-a"),
        _facet(obs_count=150, provenance_id="prov-a"),
        _facet(obs_count=200, provenance_id="prov-b"),
    ]
    cov = _served_each(facets)
    assert isinstance(cov, CoverageExact)
    sources_dim = next(d for d in dimensions_of(cov) if d.label == "sources")
    assert sources_dim.count == 2


def test_observations_dim_count_is_max_obs_count() -> None:
    facets = [_facet(obs_count=34), _facet(obs_count=402), _facet(obs_count=18)]
    cov = _coverage(_series_each(facets))
    assert isinstance(cov, CoverageExact)
    obs_dim = next(d for d in dimensions_of(cov) if d.label == "observations")
    assert obs_dim.count == 402


def test_devfinance_labels_donors_and_years() -> None:
    cov = _coverage(
        _one_series([_facet(obs_count=34)]),
        facet_label="donors",
        obs_label="years",
    )
    assert isinstance(cov, CoverageExact)
    dim_labels = {d.label for d in dimensions_of(cov)}
    assert "donors" in dim_labels
    assert "years" in dim_labels


def test_devfinance_donors_dim_count_equals_distinct_provenance_count() -> None:
    """Four served facets citing three distinct provenances count as three donors,
    not four."""
    facets = [
        _facet(obs_count=100, provenance_id="prov-a"),
        _facet(obs_count=75, provenance_id="prov-a"),
        _facet(obs_count=200, provenance_id="prov-b"),
        _facet(obs_count=50, provenance_id="prov-c"),
    ]
    cov = _served_each(facets, facet_label="donors", obs_label="years")
    assert isinstance(cov, CoverageExact)
    donors_dim = next(d for d in dimensions_of(cov) if d.label == "donors")
    assert donors_dim.count == 3


def test_devfinance_years_dim_count_is_max_obs_count() -> None:
    facets = [_facet(obs_count=34), _facet(obs_count=402), _facet(obs_count=18)]
    cov = _coverage(_series_each(facets), facet_label="donors", obs_label="years")
    assert isinstance(cov, CoverageExact)
    years_dim = next(d for d in dimensions_of(cov) if d.label == "years")
    assert years_dim.count == 402


# ---------------------------------------------------------------------------
# Sources dimension identity: distinct citable provenances, not raw facet
# count. citable_source_ids is the single rule coverage and grounding
# (region/facets.py's cited_facets) apply.
# ---------------------------------------------------------------------------


def test_three_facets_sharing_one_provenance_report_one_source() -> None:
    """Three facets citing one provenance (e.g. differing on measurement
    method) collapse to one source, not three."""
    facets = [
        _facet(obs_count=10, provenance_id="prov-a"),
        _facet(obs_count=20, provenance_id="prov-a"),
        _facet(obs_count=30, provenance_id="prov-a"),
    ]
    cov = _coverage(_one_series(facets))
    assert isinstance(cov, CoverageExact)
    sources_dim = next(d for d in dimensions_of(cov) if d.label == "sources")
    assert sources_dim.count == 1
    assert cov.observation_count == 60  # obs_count still sums across all three


def test_sources_and_observations_count_only_the_served_facet() -> None:
    """One series serves its most-observed facet, so a second source it passes over counts
    neither as a source nor for its observations."""
    facets = [
        _facet(obs_count=200, provenance_id="prov-a", facet_id="fa"),
        _facet(obs_count=3, provenance_id="prov-b", facet_id="fb"),
    ]
    cov = _coverage(_one_series(facets))
    assert isinstance(cov, CoverageExact)
    sources_dim = next(d for d in dimensions_of(cov) if d.label == "sources")
    assert sources_dim.count == 1
    assert cov.observation_count == 200


def test_a_facet_without_provenance_id_serves_and_cites_no_source() -> None:
    """A facet without a provenance_id serves its series, read by its facet id, and cites no
    source: its observations count and the sources dimension counts the other's alone."""
    facets = [_facet(obs_count=10, provenance_id=None), _facet(obs_count=5, provenance_id="prov-a")]
    cov = _served_each(facets)
    assert isinstance(cov, CoverageExact)
    assert citable_source_ids(facets) == ["prov-a"]
    sources_dim = next(d for d in dimensions_of(cov) if d.label == "sources")
    assert sources_dim.count == 1
    assert cov.observation_count == 15


def _usd_facet(facet_id: str, provenance_id: str | None, years: range) -> Facet:
    dates = [str(y) for y in years]
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance_id,
        unit="USD",
        obs_count=len(dates),
        earliest_date=dates[0],
        latest_date=dates[-1],
        dates=tuple(dates),
    )


@pytest.mark.parametrize(
    ("date_request", "served_points"),
    [
        (DateRequest(window=TimeWindow(start_year=2015, end_year=2019), latest=False), 5),
        (None, 10),
        (DateRequest(window=None, latest=True), 1),
    ],
    ids=["range", "dateless", "latest"],
)
def test_a_filter_that_removes_nothing_leaves_the_count_unchanged(
    date_request: DateRequest | None, served_points: int
) -> None:
    """Kenya's series serves `fA` (2010-2019); `fB` from another import is passed over. A USD
    filter keeps both, so the narrowed count stays the served facet's, never the pool's."""
    fa = _usd_facet("fA", "prov/A", range(2010, 2020))
    fb = _usd_facet("fB", "prov/B", range(2012, 2020))
    series = {("sv", "country/KEN"): [fa, fb]}
    plan = plan_standard(series, members=["sv"], years=plan_years(date_request))
    unfiltered = answer_coverage(plan, CoverageForm())

    narrowed = answer_coverage(plan.narrowed(lambda _sv, f: f.unit == "USD"), CoverageForm())

    assert isinstance(unfiltered, CoverageExact) and isinstance(narrowed, CoverageExact)
    assert unfiltered.observation_count == narrowed.observation_count == served_points
    assert dimensions_of(narrowed) == dimensions_of(unfiltered)


def test_a_series_whose_top_facet_has_no_provenance_serves_it() -> None:
    """The variable's most observed facet `fX` names no import and covers both places; the
    preview reads it by its facet id, so Kenya serves its 2024 point, the variable's latest
    period, which Ethiopia's series, ending in 2020, does not reach: coverage names 2024 and
    counts one point."""
    kept = _usd_facet("fA", "prov/A", range(2010, 2021))
    series = {
        ("sv", "country/KEN"): [_usd_facet("fX", None, range(2000, 2025)), kept],
        ("sv", "country/ETH"): [_usd_facet("fX", None, range(2000, 2021)), kept],
    }

    cov = _coverage(series, date_request=DateRequest(window=None, latest=True))

    assert isinstance(cov, CoverageExact)
    assert cov.window == TimeWindow(start_year=2024, end_year=2024)
    assert cov.observation_count == 1


def test_a_scope_range_elects_the_facet_inside_the_window() -> None:
    """`fA` covers three places from 1990 to 2010 and `fB` two from 2015 to 2020. A 2015 to
    2020 request ranks inside its window, so `fB` serves and its 12 points count; ranking
    over whole histories would elect `fA` and report no data."""
    fa = _usd_facet("fA", "dc/base/A", range(1990, 2011))
    fb = _usd_facet("fB", "dc/base/B", range(2015, 2021))
    series = {
        ("sv", "country/A"): [fa, fb],
        ("sv", "country/B"): [fa, fb],
        ("sv", "country/C"): [fa],
    }
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)

    cov = _coverage(series, date_request=req, facet_label="countries", scope=True)

    assert isinstance(cov, CoverageExact)
    assert (cov.has_data, cov.observation_count) == (True, 12)


# ---------------------------------------------------------------------------
# Degrade to breadth: a windowed request over facets that carry no
# per-observation dates.
# ---------------------------------------------------------------------------


def test_windowed_without_dates_keeps_family_dimensions() -> None:
    facets = [_facet(obs_count=5, dates=[]), _facet(obs_count=3, dates=[])]
    req = DateRequest(window=TimeWindow(start_year=2012, end_year=2018), latest=False)
    cov = _coverage(_one_series(facets), facet_label="donors", obs_label="years", date_request=req)
    assert isinstance(cov, CoverageBreadth)
    assert {d.label for d in cov.dimensions} == {"donors", "years"}


def test_windowed_without_dates_degrades_to_breadth() -> None:
    """A data-bearing facet with no per-observation dates cannot be counted in-window,
    so coverage emits breadth. Breadth avoids a misleading observation_count=0."""
    facets = [_facet(obs_count=40, earliest="2010", latest="2020", dates=[])]
    req = DateRequest(window=TimeWindow(start_year=2012, end_year=2018), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageBreadth)
    assert cov.has_data is True
    assert cov.window == TimeWindow(start_year=2012, end_year=2018)


def test_incomplete_dates_breadth_keeps_lifetime_has_data() -> None:
    """Breadth cannot use its known out-of-window date to rule out missing in-window data."""
    facets = [
        Facet(
            earliest_date="2000",
            latest_date="2020",
            obs_count=3,
            dates=["2000"],
            dates_complete=False,
            facet_id="f1",
            provenance_id=_PROVENANCE,
        )
    ]
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)

    cov = _coverage(_one_series(facets), date_request=req)

    assert isinstance(cov, CoverageBreadth)
    assert cov.has_data is True


def test_latest_without_dates_degrades_to_breadth() -> None:
    facets = [_facet(obs_count=40, latest="2023", dates=[])]
    req = DateRequest(window=None, latest=True)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageBreadth)


def test_partial_dates_missing_degrades_to_breadth() -> None:
    """A mixed set, where one facet has dates and another data-bearing facet lacks them,
    degrades to breadth."""
    facets = [
        _facet(obs_count=2, dates=["2015", "2016"]),
        _facet(obs_count=9, dates=[]),
    ]
    req = DateRequest(window=TimeWindow(start_year=2014, end_year=2018), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageBreadth)


def test_windowed_with_dates_stays_exact() -> None:
    """Sanity: when every data-bearing facet has dates, a window still yields exact."""
    facets = [_facet(obs_count=3, dates=["2012", "2015", "2030"])]
    req = DateRequest(window=TimeWindow(start_year=2010, end_year=2020), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 2  # 2012, 2015 inside, 2030 outside


def test_zero_obs_facets_without_dates_stay_exact() -> None:
    """A facet with obs_count == 0 and no dates keeps exactness (no data to miss)."""
    facets = [_facet(obs_count=2, dates=["2015"]), _facet(obs_count=0, dates=[])]
    req = DateRequest(window=TimeWindow(start_year=2010, end_year=2020), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.observation_count == 1


# ---------------------------------------------------------------------------
# CoverageExact.window carries the resolved concrete window
# ---------------------------------------------------------------------------


def test_window_on_exact_is_concrete_time_window() -> None:
    """CoverageExact.window is a resolved TimeWindow, the span the answer serves."""
    req = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    cov = _coverage(_one_series([_facet(obs_count=5, dates=["2016", "2018"])]), date_request=req)
    assert isinstance(cov, CoverageExact)
    assert cov.window == TimeWindow(start_year=2016, end_year=2018)


# ---------------------------------------------------------------------------
# Facet count reflects _parse_ordered_facets's facetId merge: repeated blocks
# already folded into one Facet per distinct id.
# ---------------------------------------------------------------------------


def test_sources_dim_count_after_facet_id_merge_excludes_an_id_less_block() -> None:
    """Repeated facet IDs merge; the id-less block has no provenance or import name.
    Only the two provenance-bearing facets count as sources.
    """
    facets_raw = [
        {"facetId": "A", "obsCount": 1, "earliestDate": "2020", "latestDate": "2020"},
        {"facetId": "A", "obsCount": 1, "earliestDate": "2021", "latestDate": "2021"},
        {"facetId": "B", "obsCount": 1, "earliestDate": "2020", "latestDate": "2020"},
        {"obsCount": 1, "earliestDate": "2022", "latestDate": "2022"},
    ]
    top_facets_map = {
        "A": {"provenanceId": "prov-a"},
        "B": {"provenanceId": "prov-b"},
    }
    facets = _parse_ordered_facets(facets_raw, top_facets_map, with_dates=True)
    assert len(facets) == 3  # two merged-by-id facets plus the id-less block

    cov = _served_each(facets)

    assert isinstance(cov, CoverageExact)
    sources_dim = next(d for d in dimensions_of(cov) if d.label == "sources")
    assert sources_dim.count == 2


def test_windowed_duplicate_facet_id_with_undated_block_stays_breadth() -> None:
    """Merging dated and undated blocks preserves inexact windowed coverage."""
    facets = _parse_ordered_facets(
        [
            {
                "facetId": "A",
                "obsCount": 1,
                "earliestDate": "2020",
                "latestDate": "2020",
                "observations": [{"date": "2020", "value": 1.0}],
            },
            {
                "facetId": "A",
                "obsCount": 2,
                "earliestDate": "2021",
                "latestDate": "2022",
            },
        ],
        {"A": {"provenanceId": _PROVENANCE}},
        with_dates=True,
    )
    assert len(facets) == 1
    assert facets[0].obs_count == 3
    assert facets[0].dates == ("2020",)
    assert facets[0].dates_complete is False

    req = DateRequest(window=TimeWindow(start_year=2020, end_year=2022), latest=False)
    cov = _coverage(_one_series(facets), date_request=req)

    assert isinstance(cov, CoverageBreadth)


# ---------------------------------------------------------------------------
# A bulk role plan: its elected facet across every pair, in its served years.
# ---------------------------------------------------------------------------


def _bulk_plan(years, candidates: list[RoleFacetCandidate]) -> BulkPlan:
    merged = merge_role_candidates(_one_index(candidates), ["sv"])
    best = select_role_candidate(merged, stat_vars=["sv"], time_period=None)
    return BulkPlan(
        members=("sv",),
        years=years,
        facet_id=best.facet_id if best is not None else None,
        candidates=merged,
    )


def _pairs(latest: str, counts: dict[str, int]) -> RoleFacetCandidate:
    return role_candidate(
        facet_id="f1",
        role_tuples={(("debtor", dcid),): n for dcid, n in counts.items()},
        latest_date=latest,
        provenance_id=_PROVENANCE,
    )


def test_bulk_coverage_counts_the_elected_facet_across_every_pair() -> None:
    """Three debtors with 4, 2 and 1 points: seven observations, the largest series four."""
    cov = answer_coverage(
        _bulk_plan(None, [_pairs("2024", {"A": 4, "B": 2, "C": 1})]), CoverageForm(variables=True)
    )

    assert isinstance(cov, CoverageExact)
    assert (cov.observation_count, cov.window) == (7, None)
    assert dimensions_of(cov) == [
        BreadthDim(label="sources", count=1),
        BreadthDim(label="observations", count=4),
        BreadthDim(label=MEMBER_VARIABLES_LABEL, count=1),
    ]


def test_bulk_coverage_under_latest_names_the_elected_facets_latest_year() -> None:
    cov = answer_coverage(_bulk_plan("latest", [_pairs("2022", {"A": 1})]), CoverageForm())

    assert cov.window == TimeWindow(start_year=2022, end_year=2022)


def test_bulk_coverage_carries_the_plans_range() -> None:
    window = TimeWindow(start_year=2015, end_year=2020)
    cov = answer_coverage(_bulk_plan(window, [_pairs("2024", {"A": 1})]), CoverageForm())

    assert cov.window == window


def test_bulk_coverage_without_a_facet_serves_nothing() -> None:
    cov = answer_coverage(_bulk_plan("latest", []), CoverageForm())

    assert isinstance(cov, CoverageExact)
    assert (cov.has_data, cov.observation_count, cov.window) == (False, 0, None)


def test_a_discovered_bulk_scope_counts_the_members_it_serves() -> None:
    """An open-axis form leads with the members holding the elected facet's data."""
    cov = answer_coverage(
        _bulk_plan(None, [_pairs("2024", {"A": 4, "B": 2})]),
        CoverageForm(variables=True, open_axis=("recipients", "recipient")),
    )

    assert isinstance(cov, CoverageBreadth)
    assert dimensions_of(cov) == [
        BreadthDim(label="recipients", count=2),
        BreadthDim(label="sources", count=1),
        BreadthDim(label=MEMBER_VARIABLES_LABEL, count=1),
    ]


# ---------------------------------------------------------------------------
# The `variables` dimension counts the StatVars a plan serves.
# ---------------------------------------------------------------------------


def test_the_variables_dimension_follows_the_two_count_dimensions() -> None:
    cov = _coverage(
        {("sv1", "country/KEN"): [_facet(obs_count=100)], ("sv2", "country/KEN"): [_facet()]},
        variables=True,
    )
    assert isinstance(cov, CoverageExact)
    assert [d.label for d in dimensions_of(cov)] == ["sources", "observations", "variables"]
    assert dimensions_of(cov)[-1].count == 2


def test_the_variables_dimension_counts_only_served_stat_vars() -> None:
    """`sv2`'s only facet carries no facet id, so no read dispatches it: it serves nothing
    and is not counted."""
    cov = _coverage(
        {
            ("sv1", "country/KEN"): [_facet(obs_count=1)],
            ("sv2", "country/KEN"): [_facet(obs_count=1, facet_id=None)],
        },
        variables=True,
    )
    assert dimensions_of(cov)[-1] == BreadthDim(label=MEMBER_VARIABLES_LABEL, count=1)


# ---------------------------------------------------------------------------
# headline_from_coverage_index: reads CoverageIndex.headline and takes
# year_named and named_window from date_request's own window, whether or not
# that window holds the year CoverageIndex.headline returns.
# ---------------------------------------------------------------------------

_ROLE_DCID_BY_COVERAGE_ROLE = {"donor": "financingProvider", "recipient": "financingRecipient"}

# V1's rows: two active in 2024 (donors KEN, FRA; recipient USA on both),
# one active only in 2010 -- the latest bit across all three is 2024.
_HEADLINE_ROWS = [
    ("V1", "f1", "KEN", "USA", [2020, 2024]),
    ("V1", "f2", "FRA", "USA", [2024]),
    ("V1", "f3", "DEU", "GBR", [2010]),
]
_HEADLINE_ROLE_NAMES = {"V1": RoleNames(donor="financingProvider", recipient="financingRecipient")}
_EVERY_HEADLINE_FACET = [("V1", facet_id, None, None) for facet_id in ("f1", "f2", "f3")]


def _headline_index():
    return build_coverage_index(_HEADLINE_ROWS, role_names=_HEADLINE_ROLE_NAMES)


def test_headline_from_coverage_index_explicit_year_matching_latest_is_named() -> None:
    req = DateRequest(window=TimeWindow(start_year=2024, end_year=2024), latest=False)
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=req,
    )
    assert headline is not None
    assert headline.year == 2024
    assert headline.year_named is True


def test_headline_from_coverage_index_explicit_year_before_the_latest_is_named() -> None:
    """A request naming 2020, before 2024, the latest year coverage holds, reads 2020 as
    its plan does (`answer_headline` passes the plan's years), and is named a year."""
    req = DateRequest(window=TimeWindow(start_year=2020, end_year=2020), latest=False)
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=req,
        first_year=2020,
        last_year=2020,
    )
    assert headline is not None
    assert headline.year == 2020
    assert headline.year_named is True


def test_headline_from_coverage_index_dateless_request_is_not_named() -> None:
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=None,
    )
    assert headline is not None
    assert headline.year_named is False


def test_headline_from_coverage_index_latest_request_is_not_named() -> None:
    """A `latest` ask never "names" a particular year: this deployment
    always resolves `latest` to coverage's own latest year regardless of
    what that year turns out to be."""
    req = DateRequest(window=None, latest=True)
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=req,
    )
    assert headline is not None
    assert headline.year_named is False


@pytest.mark.parametrize(
    ("window", "year", "observations"),
    [
        (TimeWindow(start_year=2010, end_year=2024), 2024, 2),
        (TimeWindow(start_year=2015, end_year=2020), 2020, 1),
        (TimeWindow(start_year=2015), 2024, 2),
        (TimeWindow(end_year=2020), 2020, 1),
        (TimeWindow(start_year=2020, end_year=2020), 2020, 1),
    ],
)
def test_headline_from_coverage_index_carries_the_years_the_request_names(
    window, year, observations
) -> None:
    """A range names years as a single year does, and the counts are those of the latest
    year with data inside it, the years its plan reads."""
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=DateRequest(window=window, latest=False),
        first_year=window.start_year,
        last_year=window.end_year,
    )
    assert headline is not None
    assert headline.year == year
    assert headline.year_named is True
    assert headline.named_window == window
    assert headline.observations == observations


@pytest.mark.parametrize("date_request", [None, DateRequest(window=None, latest=True)])
def test_headline_from_coverage_index_names_no_window_without_one(date_request) -> None:
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=date_request,
    )
    assert headline is not None
    assert headline.year_named is False
    assert headline.named_window is None


def test_headline_from_coverage_index_role_counts_and_observations_series() -> None:
    """observations/series count the two rows active in 2024 (f1, f2);
    role_counts is the distinct-dcid count per role among those active
    rows: donors {KEN, FRA} -> 2, recipients {USA} -> 1 (both rows share
    the same recipient)."""
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=None,
    )
    assert headline is not None
    assert headline.observations == 2
    assert headline.series == 2
    assert headline.role_counts == {"financingProvider": 2, "financingRecipient": 1}


def test_headline_from_coverage_index_multi_role_dcid_mapping_omits_unmapped_role() -> None:
    """A coverage role absent from role_dcid_by_coverage_role (an
    undeclared or unbound opposite axis) is omitted from role_counts
    entirely, never surfaced as a 0 entry."""
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role={"donor": "financingProvider"},
        served=_EVERY_HEADLINE_FACET,
        date_request=None,
    )
    assert headline is not None
    assert headline.role_counts == {"financingProvider": 2}


def test_headline_counts_only_the_served_facets() -> None:
    """With f2 left out, its 2024 row is not counted and the 2024 headline holds f1's row
    alone. With f1 left out as well, f3's 2010 row dates it."""
    served_f1_f3 = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=[("V1", "f1", None, None), ("V1", "f3", None, None)],
        date_request=None,
    )
    assert served_f1_f3 is not None
    assert (served_f1_f3.year, served_f1_f3.series) == (2024, 1)

    served_f3 = headline_from_coverage_index(
        _headline_index(),
        variables=["V1"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=[("V1", "f3", None, None)],
        date_request=None,
    )
    assert served_f3 is not None
    assert (served_f3.year, served_f3.series) == (2010, 1)


def test_headline_counts_a_facet_only_for_the_pair_that_serves_it() -> None:
    """Kenya's pair serves f-A and Norway's f-B, while each pair also holds the other facet in
    2024. A mask keyed by variable alone would count all four rows; keyed by pair it counts
    Kenya's f-A row and Norway's f-B row."""
    index = build_coverage_index(
        [
            ("V1", "f-A", ABSENT_ENTITY, "KEN", [2024]),
            ("V1", "f-B", ABSENT_ENTITY, "KEN", [2024]),
            ("V1", "f-A", ABSENT_ENTITY, "NOR", [2024]),
            ("V1", "f-B", ABSENT_ENTITY, "NOR", [2024]),
        ],
        role_names={"V1": RoleNames(donor=None, recipient="debtor")},
    )
    headline = headline_from_coverage_index(
        index,
        variables=["V1"],
        bound={"recipient": frozenset({"KEN", "NOR"})},
        role_dcid_by_coverage_role={"recipient": "debtor"},
        served=[("V1", "f-A", "KEN", None), ("V1", "f-B", "NOR", None)],
        date_request=None,
    )
    assert headline is not None
    assert (headline.year, headline.series, headline.observations) == (2024, 2, 2)


def test_headline_from_coverage_index_no_survivors_returns_none() -> None:
    headline = headline_from_coverage_index(
        _headline_index(),
        variables=["nonexistent-variable"],
        bound={},
        role_dcid_by_coverage_role=_ROLE_DCID_BY_COVERAGE_ROLE,
        served=_EVERY_HEADLINE_FACET,
        date_request=None,
    )
    assert headline is None
