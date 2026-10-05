"""Pins coverage-based row/request estimates, sync-budget boundaries, and Excel row admission.
Fixtures use a small in-memory `CoverageIndex`."""

from __future__ import annotations

import pytest

from datasets_api.admission import (
    UNCOVERED_MEMBER_CAP,
    refuse_async_over_ceiling,
    refuse_excel_over_estimate,
    sync_estimate,
    within_sync_budget,
)
from datasets_api.config import DatasetsBuildSettings
from dc_sdmx import ContainedIn
from materialise.errors import MaterialisationError, OverCeilingError
from materialise.facets import FacetBinding
from materialise.pipeline import MaterialisationRequest
from materialise.window import RequestedWindow
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import build_coverage_index

_CRS_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_CREDITOR_ROLES = RoleNames(donor="creditor", recipient="debtor")
_ALL_WINDOW = RequestedWindow(kind="all")

_SETTINGS = DatasetsBuildSettings()

_INDEX = build_coverage_index(
    [
        # CRS-shaped variable with three DEU->KEN series across three facets,
        # years 2020-2022, and a DEU->UGA series year 2021 only.
        ("Amount_Crs_A", "facet1", "country/DEU", "country/KEN", [2020]),
        ("Amount_Crs_A", "facet2", "country/DEU", "country/KEN", [2021]),
        ("Amount_Crs_A", "facet3", "country/DEU", "country/KEN", [2022]),
        ("Amount_Crs_A", "facet1", "country/DEU", "country/UGA", [2021]),
        # A second, creditor/debtor-shaped variable sharing the same
        # DEU/KEN dcids -- estimate() must still bill it as its own group
        # (a distinct role-name pair means a distinct SDMX request).
        ("Amount_Crs_B", "facet1", "country/DEU", "country/KEN", [2020, 2021]),
    ],
    role_names={"Amount_Crs_A": _CRS_ROLES, "Amount_Crs_B": _CREDITOR_ROLES},
)


def _empty_facets() -> dict[str, FacetBinding]:
    return {}


def _request(
    *,
    stat_var: str,
    donor: str | ContainedIn | None = None,
    recipient: str | ContainedIn | None = None,
    declared_roles: tuple[str, ...] = ("financingProvider", "financingRecipient"),
    window: RequestedWindow = _ALL_WINDOW,
) -> MaterialisationRequest:
    role_bindings: dict[str, object] = {}
    if donor is not None:
        role_bindings[declared_roles[0]] = donor if isinstance(donor, ContainedIn) else [donor]
    if recipient is not None:
        role_bindings[declared_roles[1]] = (
            recipient if isinstance(recipient, ContainedIn) else [recipient]
        )
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings=role_bindings,
        declared_roles=declared_roles,
        facet_bindings=_empty_facets(),
        window=window,
    )


def test_bound_single_variable_counts_only_its_own_survivors():
    request = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    estimate = sync_estimate(
        (request,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.rows == 3
    assert estimate.requests == 1


def test_range_window_narrows_to_the_bound_years():
    request = _request(
        stat_var="Amount_Crs_A",
        donor="country/DEU",
        recipient="country/KEN",
        window=RequestedWindow(kind="range", start_year=2022, end_year=2022),
    )
    estimate = sync_estimate(
        (request,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.rows == 1
    assert estimate.requests == 1


def test_latest_window_is_estimated_the_same_as_all_a_safe_over_estimate():
    all_window = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    latest_window = _request(
        stat_var="Amount_Crs_A",
        donor="country/DEU",
        recipient="country/KEN",
        window=RequestedWindow(kind="latest"),
    )
    assert sync_estimate(
        (latest_window,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    ) == sync_estimate(
        (all_window,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )


def test_two_requests_with_different_role_name_pairs_sum_as_two_groups():
    """`Amount_Crs_A` (financingProvider/financingRecipient) and
    `Amount_Crs_B` (creditor/debtor) share the same bound DEU/KEN dcids but
    carry different declared roles, so they bill as two separate SDMX
    requests, not one combined chunk -- matching a real
    `SdmxClient.data` call, which groups by shared entity-filter keys.
    """
    request_a = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    request_b = _request(
        stat_var="Amount_Crs_B",
        donor="country/DEU",
        recipient="country/KEN",
        declared_roles=("creditor", "debtor"),
    )
    estimate = sync_estimate(
        (request_a, request_b),
        index=_INDEX,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )
    assert estimate.rows == 3 + 2  # Amount_Crs_A's 3 obs, Amount_Crs_B's 1 row x 2 years
    assert estimate.requests == 2


def test_more_than_the_lane_target_splits_into_several_requests():
    """A group of `3 * lanes + 1` variables splits into multiple requests under the shared planner.

    The bound DEU/KEN filter and shared role pair keep this below the URL budget."""
    import math

    lanes = _SETTINGS.mint_member_max_concurrency
    member_count = 3 * lanes + 1
    stat_vars = [f"Amount_Crs_Cap{i}" for i in range(member_count)]
    index = build_coverage_index(
        [(sv, "facet1", "country/DEU", "country/KEN", [2020]) for sv in stat_vars],
        role_names=dict.fromkeys(stat_vars, _CRS_ROLES),
    )
    requests = [
        _request(stat_var=sv, donor="country/DEU", recipient="country/KEN") for sv in stat_vars
    ]
    estimate = sync_estimate(
        requests, index=index, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.rows == member_count
    target = max(1, math.ceil(member_count / lanes))
    assert estimate.requests == math.ceil(member_count / target)
    assert estimate.requests > 1


def test_requests_with_different_bindings_are_never_merged_into_one_group():
    """A collection whose members carry different scope surgery (different
    recipients) must not be summed as if they shared one entities filter:
    each binding is its own group, its own request.
    """
    kenya = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    uganda = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/UGA")
    estimate = sync_estimate(
        (kenya, uganda), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.rows == 3 + 1
    assert estimate.requests == 2


def test_uncovered_variable_contributes_zero_rather_than_refusing():
    request = _request(stat_var="NotInCoverage", donor="country/DEU", recipient="country/KEN")
    estimate = sync_estimate(
        (request,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.rows == 0
    assert estimate.requests == 0
    assert estimate.uncovered_requests == 1


def test_contained_in_bound_role_is_treated_as_unbound_a_safe_over_estimate():
    bound = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    contained = _request(
        stat_var="Amount_Crs_A",
        donor="country/DEU",
        recipient=ContainedIn(parent="africa", child_type="Country"),
    )
    bound_estimate = sync_estimate(
        (bound,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    contained_estimate = sync_estimate(
        (contained,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    # The unbound recipient includes all four DEU-donor rows, 3 for Kenya and 1 for Uganda.
    assert contained_estimate.rows >= bound_estimate.rows
    assert contained_estimate.rows == 4


def test_within_sync_budget_at_and_over_the_boundary():
    from series_coverage.index import Estimate

    assert (
        within_sync_budget(
            Estimate(rows=100, requests=5), sync_budget_rows=100, sync_budget_requests=5
        )
        is True
    )
    assert (
        within_sync_budget(
            Estimate(rows=101, requests=5), sync_budget_rows=100, sync_budget_requests=5
        )
        is False
    )
    assert (
        within_sync_budget(
            Estimate(rows=100, requests=6), sync_budget_rows=100, sync_budget_requests=5
        )
        is False
    )


def test_refuse_excel_over_estimate_raises_only_for_xlsx_over_the_limit():
    """Excel counts its header toward `EXCEL_MAX_ROWS`, so `EXCEL_MAX_ROWS - 1` data rows fit."""
    from materialise.formats import EXCEL_MAX_ROWS
    from series_coverage.index import Estimate

    refuse_excel_over_estimate(Estimate(rows=EXCEL_MAX_ROWS - 1, requests=1), format="xlsx")
    refuse_excel_over_estimate(Estimate(rows=EXCEL_MAX_ROWS, requests=1), format="csv")
    refuse_excel_over_estimate(Estimate(rows=EXCEL_MAX_ROWS, requests=1), format="parquet")

    with pytest.raises(MaterialisationError) as excinfo:
        refuse_excel_over_estimate(Estimate(rows=EXCEL_MAX_ROWS, requests=1), format="xlsx")
    assert excinfo.value.code == "excel_limit"
    assert "Parquet" in excinfo.value.detail or "parquet" in excinfo.value.detail
    assert "csv" in excinfo.value.detail


def test_uncovered_variable_counts_toward_uncovered_requests_not_rows():
    """A request for an uncovered StatVar increments `uncovered_requests`, not rows or SDMX requests."""
    covered = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    uncovered = _request(stat_var="NotInCoverage", donor="country/DEU", recipient="country/KEN")
    estimate = sync_estimate(
        (covered, uncovered), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.rows == 3
    assert estimate.requests == 1
    assert estimate.uncovered_requests == 1


def test_a_subject_only_request_binding_no_entity_is_priced_as_one_naming_every_country():
    # Coverage holds no subject-only series, so neither request has a price and both run
    # inline. The materialiser's row ceiling bounds the fetch.
    countries = [f"country/C{i:03d}" for i in range(257)]
    unbound = MaterialisationRequest(
        stat_var="dcid:Count_Person",
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings=_empty_facets(),
        window=_ALL_WINDOW,
    )
    bound = MaterialisationRequest(
        stat_var="dcid:Count_Person",
        role_bindings={"observationAbout": countries},
        declared_roles=("observationAbout",),
        facet_bindings=_empty_facets(),
        window=_ALL_WINDOW,
    )

    estimates = [
        sync_estimate(
            (request,), index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
        )
        for request in (unbound, bound)
    ]

    assert estimates[0] == estimates[1]
    assert (estimates[0].rows, estimates[0].requests, estimates[0].uncovered_requests) == (0, 0, 1)
    assert within_sync_budget(
        estimates[0],
        sync_budget_rows=_SETTINGS.sync_budget_rows,
        sync_budget_requests=_SETTINGS.sync_budget_requests,
    )


def test_every_request_uncovered_still_reports_zero_rows_and_requests():
    uncovered_a = _request(stat_var="NotInCoverage", donor="country/DEU", recipient="country/KEN")
    uncovered_b = _request(
        stat_var="AlsoNotInCoverage", donor="country/DEU", recipient="country/KEN"
    )
    estimate = sync_estimate(
        (uncovered_a, uncovered_b),
        index=_INDEX,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )
    assert estimate.rows == 0
    assert estimate.requests == 0
    assert estimate.uncovered_requests == 2


def _uncovered_requests(count: int) -> tuple[MaterialisationRequest, ...]:
    return tuple(
        _request(stat_var=f"UncoveredStatVar{i}", donor="country/DEU", recipient="country/KEN")
        for i in range(count)
    )


def test_uncovered_requests_at_the_cap_does_not_refuse():
    requests = _uncovered_requests(UNCOVERED_MEMBER_CAP)
    estimate = sync_estimate(
        requests, index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )
    assert estimate.uncovered_requests == UNCOVERED_MEMBER_CAP
    assert estimate.rows == 0
    assert estimate.requests == 0


def test_uncovered_requests_one_over_the_cap_refuses_over_ceiling():
    """Crossing `UNCOVERED_MEMBER_CAP` raises the existing 422 member-cardinality refusal.

    Uncovered requests price at zero rows and requests, so they once cleared every budget.
    """
    requests = _uncovered_requests(UNCOVERED_MEMBER_CAP + 1)
    with pytest.raises(OverCeilingError) as excinfo:
        sync_estimate(
            requests, index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
        )
    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.status == 422
    assert excinfo.value.ceiling_kind == "member_cardinality"
    assert excinfo.value.count == UNCOVERED_MEMBER_CAP + 1
    assert excinfo.value.cap == UNCOVERED_MEMBER_CAP


def test_over_cap_refusal_holds_even_with_covered_members_well_within_budget():
    """The uncovered-member cap applies independently of the covered row/request budget."""
    covered = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    requests = (covered, *_uncovered_requests(UNCOVERED_MEMBER_CAP + 1))
    with pytest.raises(OverCeilingError) as excinfo:
        sync_estimate(
            requests, index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
        )
    assert excinfo.value.count == UNCOVERED_MEMBER_CAP + 1


def test_over_cap_refusal_never_computes_a_covered_estimate(monkeypatch: pytest.MonkeyPatch):
    """The uncovered cap raises before `index.estimate` prices any covered group."""
    index = build_coverage_index(
        [("Amount_Crs_A", "facet1", "country/DEU", "country/KEN", [2020])],
        role_names={"Amount_Crs_A": _CRS_ROLES},
    )
    calls: list[object] = []
    monkeypatch.setattr(index, "estimate", lambda **kwargs: calls.append(kwargs))
    covered = _request(stat_var="Amount_Crs_A", donor="country/DEU", recipient="country/KEN")
    requests = (covered, *_uncovered_requests(UNCOVERED_MEMBER_CAP + 1))

    with pytest.raises(OverCeilingError):
        sync_estimate(
            requests, index=index, member_concurrency=_SETTINGS.mint_member_max_concurrency
        )

    assert calls == []


def test_a_fully_unbound_role_bearing_member_is_priced_and_refused_over_the_async_ceiling():
    """An unbound role means every entity: the estimate prices every covered series, and the
    async ceilings refuse it with their own code."""
    request = _request(stat_var="Amount_Crs_A")

    estimate = sync_estimate(
        [request], index=_INDEX, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )

    assert (estimate.rows, estimate.requests) == (4, 1)
    with pytest.raises(OverCeilingError) as excinfo:
        refuse_async_over_ceiling(estimate, async_max_rows=3, async_max_requests=10)
    assert excinfo.value.ceiling_kind == "operation_rows"
