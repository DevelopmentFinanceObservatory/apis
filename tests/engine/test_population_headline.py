"""The headline and coverage count the whole elected population, never the part a bulk
preview delivers.

A bulk role preview ships at most `role_bulk_series_max` series. Coverage and the headline
are computed before the read, over every series the elected facet covers, so a capped
preview neither shrinks their counts nor moves their year.
"""

from __future__ import annotations

from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.region.codes import OBSERVATIONS_SERIES_TRUNCATED
from search.models import CoverageBreadth, CoverageExact, TimeWindow
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine import test_role_scope_pipeline_e2e as role_scope
from tests.engine._fake_coverage import CoverageRow, FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient

_LATEST = DateRequest(window=None, latest=True)


def _yearly(years: range) -> Facet:
    return Facet(
        facet_id="f1",
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
        obs_count=len(years),
        dates=[str(y) for y in years],
        values=[float(y) for y in years],
    )


def test_a_discovered_scope_headline_counts_all_100_series_while_20_are_delivered(
    monkeypatch, search_settings
):
    """100 debtors hold data through 2024. The bulk preview delivers 20 of them; the
    headline counts 100 series in 2024 and coverage names 2024."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=20)
    debtors = [f"country/D{i:03d}" for i in range(100)]
    rows: list[CoverageRow] = [
        (role_scope._SV_DCID, "f1", role_scope._UNBOUND_DONOR, debtor, list(range(2015, 2025)))
        for debtor in debtors
    ]
    sdmx = FakeSdmxClient(
        observations={
            role_scope._SV_DCID: {
                (("creditor", role_scope._UNBOUND_DONOR), ("debtor", debtor)): [
                    _yearly(range(2015, 2025))
                ]
                for debtor in debtors
            }
        },
        availability={},
    )

    result = role_scope._run(
        entities=[],
        sv_dcid=role_scope._SV_DCID,
        sv_arcs=role_scope._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        date_request=_LATEST,
        include_observations=True,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names={role_scope._SV_DCID: RoleNames(donor="creditor", recipient="debtor")},
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    observations = spec.resolved.observations
    assert observations is not None
    assert len(observations.series) == 20
    assert isinstance(spec.resolved.coverage, CoverageBreadth)
    assert spec.resolved.coverage.window == TimeWindow(start_year=2024, end_year=2024)
    headline = spec.resolved.headline
    assert headline is not None
    assert (headline.year, headline.series, headline.observations) == (2024, 100, 100)
    # The file holds each of the 100 series' latest point.
    download = spec.resolved.download
    assert download is not None
    assert (download.rows, download.series, download.rows_exact) == (100, 100, True)
    assert download.window == TimeWindow(start_year=2024, end_year=2024)


# Five named debtors on a one-role variable. The first two in identity order end in 2023;
# the other three reach 2024.
_NAMED = {
    "Aland": ("country/AAA", range(2010, 2024)),
    "Bland": ("country/BBB", range(2010, 2024)),
    "Cland": ("country/CCC", range(2010, 2025)),
    "Dland": ("country/DDD", range(2010, 2025)),
    "Eland": ("country/EEE", range(2010, 2025)),
}


def test_a_named_bulk_latest_answer_serves_and_counts_the_facets_latest_year(
    monkeypatch, search_settings
):
    """Above the pair cap the preview serves the facet's latest year, 2024, only. The two
    delivered series are the first pairs with a 2024 point, passing over the two that end
    in 2023, and coverage and the headline both name 2024 with three series."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=2)
    index = patch_coverage(
        monkeypatch,
        [
            (role_bearing._SV_DCID, "f1", ABSENT_ENTITY, dcid, list(years))
            for dcid, years in _NAMED.values()
        ],
        role_names={role_bearing._SV_DCID: RoleNames(donor=None, recipient="debtor")},
    )
    sdmx = FakeSdmxClient(
        observations={
            role_bearing._SV_DCID: {
                (("debtor", dcid),): [_yearly(years)] for dcid, years in _NAMED.values()
            }
        },
        availability={},
    )

    result = role_bearing._run(
        entities=list(_NAMED),
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve={name: dcid for name, (dcid, _years) in _NAMED.items()},
        sv_score=0.9,
        role_query="debt flows to " + ", ".join(_NAMED),
        sdmx=sdmx,
        include_observations=True,
        date_request=_LATEST,
        coverage=FakeCoverageService(index).snapshot(),
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    observations = spec.resolved.observations
    assert observations is not None
    delivered = [[p.date for p in s.points] for s in observations.series]
    assert delivered == [["2024"], ["2024"]]
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert coverage.observation_count == 3
    assert coverage.window == TimeWindow(start_year=2024, end_year=2024)
    headline = spec.resolved.headline
    assert headline is not None
    assert (headline.year, headline.series) == (2024, 3)


def test_a_bulk_latest_preview_that_delivers_every_eligible_series_does_not_warn(
    monkeypatch, search_settings
):
    """Three of the five debtors reach 2024, the served year, and the preview delivers all
    three, so nothing eligible is cut and no truncation warning is raised."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=3)
    index = patch_coverage(
        monkeypatch,
        [
            (role_bearing._SV_DCID, "f1", ABSENT_ENTITY, dcid, list(years))
            for dcid, years in _NAMED.values()
        ],
        role_names={role_bearing._SV_DCID: RoleNames(donor=None, recipient="debtor")},
    )
    sdmx = FakeSdmxClient(
        observations={
            role_bearing._SV_DCID: {
                (("debtor", dcid),): [_yearly(years)] for dcid, years in _NAMED.values()
            }
        },
        availability={},
    )

    result = role_bearing._run(
        entities=list(_NAMED),
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve={name: dcid for name, (dcid, _years) in _NAMED.items()},
        sv_score=0.9,
        role_query="debt flows to " + ", ".join(_NAMED),
        sdmx=sdmx,
        include_observations=True,
        date_request=_LATEST,
        coverage=FakeCoverageService(index).snapshot(),
    )

    spec = result.specs[0]
    assert spec.resolved.observations is not None
    assert [[p.date for p in s.points] for s in spec.resolved.observations.series] == [
        ["2024"],
        ["2024"],
        ["2024"],
    ]
    assert OBSERVATIONS_SERIES_TRUNCATED not in [w.code for w in result.warnings]


# Three named debtors on a two-role variable, each lent to by two creditors; the origin stays
# open. Aland's creditors end in 2023, the other debtors' in 2024.
_CREDITORS = ("country/CRA", "country/CRB")
_OPEN_ORIGIN = {
    "Aland": ("country/AAA", range(2010, 2024)),
    "Cland": ("country/CCC", range(2010, 2025)),
    "Eland": ("country/EEE", range(2010, 2025)),
}


def _open_origin_answer(monkeypatch, search_settings, *, series_max: int):
    """Latest flows to the three named debtors from any creditor, read in bulk."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=series_max)
    index = patch_coverage(
        monkeypatch,
        [
            (role_bearing._SV_DCID, "f1", creditor, dcid, list(years))
            for dcid, years in _OPEN_ORIGIN.values()
            for creditor in _CREDITORS
        ],
        role_names={role_bearing._SV_DCID: RoleNames(donor="creditor", recipient="debtor")},
    )
    sdmx = FakeSdmxClient(
        observations={
            role_bearing._SV_DCID: {
                (("creditor", creditor), ("debtor", dcid)): [_yearly(years)]
                for dcid, years in _OPEN_ORIGIN.values()
                for creditor in _CREDITORS
            }
        },
        availability={},
    )
    result = role_bearing._run(
        entities=list(_OPEN_ORIGIN),
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={name: dcid for name, (dcid, _years) in _OPEN_ORIGIN.items()},
        sv_score=0.9,
        role_query="debt flows to " + ", ".join(_OPEN_ORIGIN),
        sdmx=sdmx,
        include_observations=True,
        date_request=_LATEST,
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert result.status == "definite", result
    return result


def _served(spec) -> list[tuple[str, str, list[str]]]:
    assert spec.resolved.observations is not None
    return [
        (
            next(e.ref for e in s.entities if e.role == "origin"),
            next(e.ref for e in s.entities if e.role == "destination"),
            [p.date for p in s.points],
        )
        for s in spec.resolved.observations.series
    ]


def test_an_open_origin_bulk_read_serves_each_creditors_series(monkeypatch, search_settings):
    """Issue #1018: with the origin open, each creditor's rows match the named debtor's
    dispatched pair, so the preview serves one series per creditor and debtor: the four
    with a 2024 point. Coverage and the headline count those same four series."""
    result = _open_origin_answer(monkeypatch, search_settings, series_max=10)

    spec = result.specs[0]
    assert _served(spec) == [
        (creditor, debtor, ["2024"])
        for creditor in _CREDITORS
        for debtor in ("country/CCC", "country/EEE")
    ]
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert (coverage.window, coverage.observation_count) == (
        TimeWindow(start_year=2024, end_year=2024),
        4,
    )
    assert spec.resolved.headline is not None
    assert (spec.resolved.headline.year, spec.resolved.headline.series) == (2024, 4)
    assert OBSERVATIONS_SERIES_TRUNCATED not in [w.code for w in result.warnings]


def test_an_open_origin_bulk_read_counts_series_when_it_truncates(monkeypatch, search_settings):
    """Capped at three series, the preview ships three of the four eligible creditor and
    debtor series and says so."""
    result = _open_origin_answer(monkeypatch, search_settings, series_max=3)

    assert len(_served(result.specs[0])) == 3
    (warning,) = [w for w in result.warnings if w.code == OBSERVATIONS_SERIES_TRUNCATED]
    assert "showing 3 of 4 confirmed series" in warning.message
    assert warning.params == {"shown": 3, "total": 4, "order": "ranked"}


def test_a_windowed_bulk_read_past_the_current_year_serves_projection_years(
    monkeypatch, search_settings
):
    """Three debtors hold `f1` from 2015 to 2032, projections included. A 2028 to 2030
    request reads its year list to the coverage horizon's last year, not the current
    year, so the preview serves 2028 to 2030 and coverage counts those same nine points."""
    search_settings(obs_preview_max_pairs=1)
    debtors = {"Cland": "country/CCC", "Dland": "country/DDD", "Eland": "country/EEE"}
    years = range(2015, 2033)
    index = patch_coverage(
        monkeypatch,
        [
            (role_bearing._SV_DCID, "f1", ABSENT_ENTITY, dcid, list(years))
            for dcid in debtors.values()
        ],
        role_names={role_bearing._SV_DCID: RoleNames(donor=None, recipient="debtor")},
    )
    sdmx = FakeSdmxClient(
        observations={
            role_bearing._SV_DCID: {
                (("debtor", dcid),): [_yearly(years)] for dcid in debtors.values()
            }
        },
        availability={},
    )
    window = TimeWindow(start_year=2028, end_year=2030)

    result = role_bearing._run(
        entities=list(debtors),
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve=debtors,
        sv_score=0.9,
        role_query="debt flows to " + ", ".join(debtors),
        sdmx=sdmx,
        include_observations=True,
        date_request=DateRequest(window=window, latest=False),
        coverage=FakeCoverageService(index).snapshot(),
    )

    spec = result.specs[0]
    assert spec.resolved.observations is not None
    assert [[p.date for p in s.points] for s in spec.resolved.observations.series] == [
        ["2028", "2029", "2030"]
    ] * 3
    assert [call["time_period"] for call in sdmx.series_calls] == [["2028", "2029", "2030"]]
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert (coverage.window, coverage.observation_count) == (window, 9)
