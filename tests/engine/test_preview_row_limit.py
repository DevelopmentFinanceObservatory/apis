"""A per-series preview under a row limit below its data: every series keeps its own latest
rows and its whole count, and the preview says it was cut.

Each case runs the whole answer path over the fake coverage index and SDMX doubles with
`obs_preview_max_rows` below the rows a series holds.
"""

from __future__ import annotations

import httpx
import pytest

from search.models import CoverageExact, Spec
from tests.engine.test_read_plan_invariants import (
    _AAA,
    _CRA,
    _CRB,
    _CRC,
    _answer,
    _open_origin_answer,
    _OpenOriginHoldings,
)

# One debtor and two creditors left open: one holds four observations, the other one older
# observation.
_FOUR_AND_ONE: _OpenOriginHoldings = {
    (_CRA, _AAA): [("f1", range(2019, 2023))],
    (_CRB, _AAA): [("f1", range(2015, 2016))],
}


def _points_and_totals(spec: Spec) -> dict[str, tuple[list[str], int]]:
    """Each served series' dates and its whole count, keyed by its creditor, or by its debtor
    for a series with no creditor."""
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    return {
        next(
            (e.ref for e in s.entities if e.role == "origin"),
            next(e.ref for e in s.entities if e.role == "destination"),
        ): ([p.date for p in s.points], s.total_obs)
        for s in observations.series
    }


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_an_open_role_read_limits_each_creditor_and_keeps_its_whole_count(
    monkeypatch, search_settings, discovered: bool
) -> None:
    """A row limit of two over creditors holding four and one observations: the first keeps
    its latest two, the second its one, both state their whole counts, and the preview is
    marked as cut. The read stays one request for the bound debtor."""
    search_settings(obs_preview_max_rows=2)
    requests: list[httpx.Request] = []

    spec = _open_origin_answer(
        monkeypatch,
        search_settings,
        request="dateless",
        discovered=discovered,
        holdings=_FOUR_AND_ONE,
        requests=requests,
    )

    observations = spec.resolved.observations
    assert observations is not None
    assert _points_and_totals(spec) == {_CRA: (["2021", "2022"], 4), _CRB: (["2015"], 1)}
    assert observations.total_obs == 5
    assert observations.truncated is True
    if not discovered:
        assert isinstance(spec.resolved.coverage, CoverageExact)
        assert spec.resolved.coverage.observation_count == observations.total_obs
    assert [r.url.params.get("c[creditor]") for r in requests] == [None]


def test_a_creditor_outside_coverage_leaves_each_planned_creditor_its_row_limit(
    monkeypatch, search_settings
) -> None:
    """Coverage holds two creditors and the live read returns a third. Under a row limit of
    two each planned creditor keeps its latest two rows, as it would without the third."""
    search_settings(obs_preview_max_rows=2)
    planned = {(creditor, _AAA): [("f1", range(2016, 2020))] for creditor in (_CRA, _CRB)}

    spec = _open_origin_answer(
        monkeypatch,
        search_settings,
        request="dateless",
        holdings=planned,
        live_holdings=planned | {(_CRC, _AAA): [("f1", range(2016, 2020))]},
    )

    observations = spec.resolved.observations
    assert observations is not None
    assert _points_and_totals(spec) == {
        _CRA: (["2018", "2019"], 4),
        _CRB: (["2018", "2019"], 4),
    }
    assert observations.total_obs == 8
    assert observations.truncated is True


def test_an_open_role_read_within_its_row_limit_is_whole(monkeypatch, search_settings) -> None:
    search_settings(obs_preview_max_rows=4)

    spec = _open_origin_answer(
        monkeypatch, search_settings, request="dateless", holdings=_FOUR_AND_ONE
    )

    observations = spec.resolved.observations
    assert observations is not None
    assert _points_and_totals(spec) == {
        _CRA: (["2019", "2020", "2021", "2022"], 4),
        _CRB: (["2015"], 1),
    }
    assert observations.total_obs == 5
    assert observations.truncated is False


def test_a_latest_read_serves_one_row_per_creditor_under_any_row_limit(
    monkeypatch, search_settings
) -> None:
    """CRA's series ends in 2022 and CRB's in 2015, so a `latest` read serves the variable's
    2022 period from CRA alone, one row under a one-row limit."""
    search_settings(obs_preview_max_rows=1)

    spec = _open_origin_answer(
        monkeypatch, search_settings, request="latest", holdings=_FOUR_AND_ONE
    )

    observations = spec.resolved.observations
    assert observations is not None
    assert _points_and_totals(spec) == {_CRA: (["2022"], 1)}
    assert observations.truncated is False


def test_a_fully_bound_series_cut_by_the_row_limit_says_so(monkeypatch, search_settings) -> None:
    """Three named debtors, each one series read alone, asked for 2015 to 2020 under a row
    limit of two: Kenya and Uganda hold six points there and Norway five."""
    search_settings(obs_preview_max_rows=2)

    spec = _answer(monkeypatch, search_settings, bulk=False, discovered=False, request="closed")

    observations = spec.resolved.observations
    assert observations is not None
    assert _points_and_totals(spec) == {
        "country/KEN": (["2019", "2020"], 6),
        "country/NOR": (["2018", "2019"], 5),
        "country/UGA": (["2019", "2020"], 6),
    }
    assert observations.total_obs == 17
    assert observations.truncated is True
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.observation_count == observations.total_obs
