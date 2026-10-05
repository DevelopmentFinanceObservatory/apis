"""Tests window filtering, applied-window origins, and coverage shortfalls."""

from __future__ import annotations

import pytest

from dc_sdmx import SdmxObservation
from materialise import columnar
from materialise.window import (
    Coverage,
    RequestedWindow,
    compute_applied_window,
    coverage_below_scope,
    unserved_entities,
)

from ._table_support import table_from_observations

_ROLES = ("recipient",)


def _row(*, time_period: str | None, recipient: str = "country/TGO") -> SdmxObservation:
    return SdmxObservation(
        variable="dcid:MalariaGrantDisbursements",
        role_values={"recipient": recipient},
        unit="USD",
        measurement_method="GrantEquivalent",
        observation_period=None,
        provenance="OECD_DAC2A",
        time_period=time_period,
        value="1000",
        scaling_factor=None,
        facet_id="facet1",
    )


def _table(rows):
    return table_from_observations(rows, roles=_ROLES)


# --- range_window_mask / latest_rows ----------------------------------------


def test_range_window_mask_with_no_bounds_keeps_every_row():
    rows = [_row(time_period="2015"), _row(time_period="2020")]
    window = RequestedWindow(kind="range", start_year=None, end_year=None)

    mask = columnar.range_window_mask(_table(rows), window=window).to_pylist()

    assert mask == [True, True]


def test_range_window_mask_compares_yyyy_yyyymm_yyyymmdd_on_leading_year():
    rows = [
        _row(time_period="2014"),
        _row(time_period="2015-06"),
        _row(time_period="2016-06-30"),
        _row(time_period="2017"),
    ]

    mask = columnar.range_window_mask(
        _table(rows), window=RequestedWindow(kind="range", start_year=2015, end_year=2016)
    ).to_pylist()

    assert mask == [False, True, True, False]


def test_range_window_mask_unparseable_time_period_included_fails_open():
    rows = [_row(time_period="not-a-date"), _row(time_period="2010")]

    mask = columnar.range_window_mask(
        _table(rows), window=RequestedWindow(kind="range", start_year=2015)
    ).to_pylist()

    assert mask == [True, False]


def test_latest_rows_keeps_only_single_latest_period():
    rows = [_row(time_period="2019"), _row(time_period="2020"), _row(time_period="2018")]

    result = columnar.latest_rows(_table(rows))

    assert result["TIME_PERIOD"].to_pylist() == ["2020"]


# --- compute_applied_window -----------------------------------------------


def test_applied_window_zero_rows_returns_none():
    assert compute_applied_window(RequestedWindow(kind="all"), None) is None


def test_applied_window_all_origin_default():
    applied = compute_applied_window(RequestedWindow(kind="all"), (2015, 2020))

    assert applied is not None
    assert applied.origin == "default"
    assert applied.start_year == 2015
    assert applied.end_year == 2020


def test_applied_window_b7_case1_latest_single_served_year_origin_query():
    applied = compute_applied_window(RequestedWindow(kind="latest"), (2022, 2022))

    assert applied is not None
    assert applied.start_year == 2022
    assert applied.end_year == 2022
    assert applied.origin == "query"


def test_applied_window_b7_case2_half_open_range_fully_reached_no_clamp():
    requested = RequestedWindow(kind="range", start_year=2015, end_year=None)

    applied = compute_applied_window(requested, (2015, 2020))

    assert applied is not None
    assert applied.origin == "query"
    assert applied.start_year == 2015
    assert applied.end_year == 2020  # open end resolves to whatever was served


def test_applied_window_b7_case3_stated_start_narrowed_by_source_clamps():
    requested = RequestedWindow(kind="range", start_year=2010, end_year=2020)

    applied = compute_applied_window(requested, (2015, 2020))

    assert applied is not None
    assert applied.origin == "coverage_clamp"
    assert applied.start_year == 2015
    assert applied.end_year == 2020


# --- coverage / coverage_below_scope ----------------------------------------


def test_coverage_dimension_counts():
    rows = [
        _row(time_period="2015", recipient="country/TGO"),
        _row(time_period="2016", recipient="country/KEN"),
        _row(time_period="2016", recipient="country/TGO"),
    ]

    coverage = columnar.coverage(_table(rows), declared_roles=("recipient",), applied_window=None)

    assert coverage.rows == 3
    assert {"label": "recipient", "count": 2} in coverage.dimensions
    assert {"label": "years", "count": 2} in coverage.dimensions
    assert coverage.window is None


def test_coverage_below_scope_true_on_window_clamp():
    coverage = Coverage(rows=1, dimensions=(), window=None)
    applied = compute_applied_window(
        RequestedWindow(kind="range", start_year=2010, end_year=2020), (2015, 2015)
    )

    assert (
        coverage_below_scope(
            coverage,
            applied_window=applied,
            unserved=(),
        )
        is True
    )


@pytest.mark.parametrize(
    ("role_bindings", "served_entities_by_role", "expected"),
    [
        pytest.param(
            {"recipient": ["country/TGO", "country/KEN"]},
            {"recipient": {"country/TGO"}},
            True,
            id="entity_shortfall",
        ),
        pytest.param(
            {"recipient": ["country/TGO"]},
            {"recipient": {"country/TGO"}},
            False,
            id="fully_served",
        ),
        pytest.param(
            # A ContainedIn-bound role is omitted from role_bindings. Treating
            # this served-only role as a shortfall would be a false positive.
            {},
            {"recipient": {"country/TGO"}},
            False,
            id="contained_in_omitted",
        ),
    ],
)
def test_coverage_below_scope_for_entity_bindings(
    role_bindings: dict[str, list[str]],
    served_entities_by_role: dict[str, set[str]],
    expected: bool,
):
    coverage = Coverage(rows=1, dimensions=(), window=None)
    applied = compute_applied_window(RequestedWindow(kind="all"), (2015, 2015))

    result = coverage_below_scope(
        coverage,
        applied_window=applied,
        unserved=unserved_entities(role_bindings, served_entities_by_role),
    )

    assert result is expected


def test_coverage_below_scope_false_on_zero_rows_even_with_bound_unserved_entity():
    # An empty served set is trivially a strict subset of any non-empty bound
    # set -- without the rows==0 short-circuit this would also fire
    # coverage_below_scope alongside empty_result, contradicting the rule
    # that an empty result is served with the empty_result warning alone.
    coverage = Coverage(rows=0, dimensions=(), window=None)

    result = coverage_below_scope(
        coverage,
        applied_window=None,
        unserved=("country/TGO",),
    )

    assert result is False


# --- unserved_entities -------------------------------------------------------


def test_unserved_entities_returns_sorted_deduplicated_union():
    result = unserved_entities(
        {"recipient": ["country/TGO", "country/KEN"], "provider": ["country/USA", "country/KEN"]},
        {"recipient": {"country/TGO"}, "provider": set()},
    )

    assert result == ("country/KEN", "country/USA")


def test_unserved_entities_empty_when_every_bound_entity_is_served():
    result = unserved_entities(
        {"recipient": ["country/TGO"]},
        {"recipient": {"country/TGO"}},
    )

    assert result == ()


def test_unserved_entities_ignores_served_roles_absent_from_bindings():
    # A ContainedIn-bound role is omitted from role_bindings entirely; its
    # served set alone is not a shortfall.
    result = unserved_entities({}, {"recipient": {"country/TGO"}})

    assert result == ()


def test_served_entity_outside_the_bound_set_is_not_a_shortfall():
    # The shortfall is bound-minus-served, one direction only. A served
    # entity the request never bound ("B" bound and unserved, "C" served and
    # unbound) leaves the unbound one out of the tuple, and
    # coverage_below_scope reads that tuple alone -- no clamp is involved.
    assert unserved_entities({"recipient": ["A", "B"]}, {"recipient": {"A", "C"}}) == ("B",)
    assert unserved_entities({"recipient": ["A"]}, {"recipient": {"A", "C"}}) == ()

    coverage = Coverage(rows=1, dimensions=(), window=None)
    applied = compute_applied_window(RequestedWindow(kind="all"), (2015, 2015))

    assert (
        coverage_below_scope(
            coverage,
            applied_window=applied,
            unserved=unserved_entities({"recipient": ["A", "B"]}, {"recipient": {"A", "C"}}),
        )
        is True
    )
    assert (
        coverage_below_scope(
            coverage,
            applied_window=applied,
            unserved=unserved_entities({"recipient": ["A"]}, {"recipient": {"A", "C"}}),
        )
        is False
    )
