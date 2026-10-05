"""A role answer holds no per-observation data: a coverage candidate's facet carries one date
per year with data, whatever the year's observation count."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from search.engine.extract import DateRequest
from search.engine.retrieval.read_plan import BulkPlan, PairPlan
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.retrieval.role_scope import facet_from_coverage
from search.engine.shape import ShapeDraft
from search.models import TimeWindow
from search.observations import observation_settings
from series_coverage.coverage_types import FacetCoverage
from series_coverage.index import CoverageIndex
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_SV = "TESTFIXTURE/CoverageFacetDates/Flow"
_HORIZON_SV = "TESTFIXTURE/CoverageFacetDates/Horizon"
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_YEARS = (2018, 2019, 2020)
_GAP_YEAR = 2022


class _YearCount:
    """One year's observation count that can be tested for data and cannot be used to repeat
    anything: `range`, sequence repetition and `itertools.repeat` all call `__index__`."""

    def __init__(self, count: int) -> None:
        self._count = count

    def __bool__(self) -> bool:
        return self._count > 0

    def __index__(self) -> int:
        raise AssertionError("a year's observation count was expanded into per-observation data")


@pytest.fixture
def unexpandable_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every `FacetCoverage` the index returns holds `_YearCount` values, so a reader that
    builds one entry per observation raises where it reads the count."""
    build = CoverageIndex._facet_coverages

    def _guarded(self, rows, *, role_columns):
        return [
            (
                key,
                replace(
                    facet,
                    year_counts=cast(
                        "dict[int, int]",
                        {year: _YearCount(count) for year, count in facet.year_counts.items()},
                    ),
                ),
            )
            for key, facet in build(self, rows, role_columns=role_columns)
        ]

    monkeypatch.setattr(CoverageIndex, "_facet_coverages", _guarded)


def _shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_coverage_facet_dates_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SV: {
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
            }
        },
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )


def _resolve(
    creditors: int, *, entities: dict[str, list[str]], year: int | None, discover: bool = False
) -> Materialised:
    """Resolve the flow over `creditors` creditors lending to one debtor in each of `_YEARS`.
    A second variable's row carries the horizon past `_GAP_YEAR`, a year the flow holds no
    point in."""
    rows = [(_SV, "f1", f"org/Creditor{i:04d}", "country/KEN", _YEARS) for i in range(creditors)]
    rows.append((_HORIZON_SV, "f2", "org/Creditor0000", "country/UGA", [_GAP_YEAR + 1]))
    index = build_coverage_index(rows, role_names={_SV: _ROLE_NAMES, _HORIZON_SV: _ROLE_NAMES})
    result = materialise(
        _shape(),
        [],
        entities,
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        date_request=(
            DateRequest(window=TimeWindow(start_year=year, end_year=year), latest=False)
            if year is not None
            else None
        ),
        coverage=FakeCoverageService(index).snapshot(),
        discover_scope=discover,
    )
    assert isinstance(result, Materialised)
    return result


def test_a_coverage_facet_carries_one_date_per_year_and_its_whole_count() -> None:
    facet = facet_from_coverage(
        FacetCoverage(
            facet_id="f1",
            n_obs=1_000_000,
            first_year=2019,
            last_year=2021,
            year_counts={2021: 600_000, 2019: 400_000, 2020: 0},
        ),
        None,
    )

    assert facet.dates == ("2019", "2021")
    assert facet.obs_count == 1_000_000
    assert facet.dates_complete is True


@pytest.mark.usefixtures("unexpandable_counts")
def test_named_role_confirmation_expands_no_observation_before_the_pair_limit() -> None:
    """A named debtor's candidate sums every creditor's rows, above the pair cap."""
    creditors = observation_settings().obs_preview_max_pairs + 1

    result = _resolve(creditors, entities={"debtor": ["country/KEN"]}, year=None)

    assert isinstance(result.read_plan, BulkPlan)
    assert result.facets_by_sv is not None
    [facet] = result.facets_by_sv[_SV]
    assert facet.obs_count == creditors * len(_YEARS)
    assert facet.dates == ("2018", "2019", "2020")


@pytest.mark.usefixtures("unexpandable_counts")
@pytest.mark.parametrize(
    ("entities", "discover"),
    [({"debtor": ["country/KEN"]}, False), ({}, True)],
    ids=["named_debtor", "discovered_debtors"],
)
def test_an_empty_window_expands_no_observation_across_the_family(
    entities: dict[str, list[str]], discover: bool
) -> None:
    """The carrier for data held outside the years asked keeps the family's lifetime
    candidates, and its gap is read from the plan with the window removed."""
    creditors = observation_settings().obs_preview_max_pairs + 1

    result = _resolve(creditors, entities=entities, year=_GAP_YEAR, discover=discover)

    assert result.has_data is False
    assert result.window_gap is not None
    assert (result.window_gap.first, result.window_gap.last) == ("2018", "2020")
    assert result.facets_by_sv is not None
    [facet] = result.facets_by_sv[_SV]
    assert facet.obs_count == creditors * len(_YEARS)
    assert facet.dates == ("2018", "2019", "2020")


@pytest.mark.usefixtures("unexpandable_counts")
def test_a_per_pair_plan_expands_no_observation() -> None:
    """Within the pair cap each role pair's series carries its years, and the plan counts
    one point per series and year."""
    result = _resolve(2, entities={"debtor": ["country/KEN"]}, year=None)

    plan = result.read_plan
    assert isinstance(plan, PairPlan)
    assert [f.dates for s in plan.series for f in s.candidates] == [("2018", "2019", "2020")] * 2
    assert result.coverage.observation_count == 6
