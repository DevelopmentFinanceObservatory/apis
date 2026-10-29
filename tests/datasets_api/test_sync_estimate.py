"""Pins coverage-based row/request estimates, sync-budget boundaries, and Excel row admission.
Fixtures use a small in-memory `CoverageIndex`."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import cast

import pytest

from code_index.types import PartyRow
from datasets_api.admission import (
    UNCOVERED_MEMBER_CAP,
    elected_provenances,
    identity_role_keys,
    refuse_async_over_ceiling,
    refuse_excel_over_estimate,
    refuse_excel_over_stored_rows,
    sync_estimate,
    within_sync_budget,
    xlsx_splits_totals,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import DefinitionDeps, plan_stored_render
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions
from dc_sdmx import ContainedIn
from materialise.errors import MaterialisationError, OverCeilingError
from materialise.facets import ElectedDataset, FacetBinding
from materialise.formats import EXCEL_MAX_ROWS
from materialise.pipeline import MaterialisationRequest
from materialise.window import RequestedWindow
from series_coverage.index import Estimate
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


@pytest.mark.parametrize(
    ("stat_vars", "donor", "second_reads"),
    [
        pytest.param(["Amount_Crs_A"], "country/DEU", 1, id="both-roles-bound"),
        pytest.param(["Amount_Crs_A"], None, 1, id="every-donor-to-one-recipient"),
        pytest.param(["Amount_Crs_A", "Amount_Uncovered"], "country/DEU", 1, id="one-uncovered"),
    ],
)
def test_latest_window_prices_all_years_and_one_more_read_of_a_member_ending_in_several_years(
    stat_vars: list[str], donor: str | None, second_reads: int
):
    """A `latest` member with a series ending after the year it elects is read once more at
    that year, so a member holding coverage rows that end in different years is priced a
    second read."""

    def _estimate(window: RequestedWindow) -> Estimate:
        return sync_estimate(
            [
                _request(stat_var=stat_var, donor=donor, recipient="country/KEN", window=window)
                for stat_var in stat_vars
            ],
            index=_INDEX,
            member_concurrency=_SETTINGS.mint_member_max_concurrency,
        )

    every_year, latest = _estimate(_ALL_WINDOW), _estimate(RequestedWindow(kind="latest"))

    assert (every_year.rows, every_year.requests) == (3, 1)
    assert latest == dataclasses.replace(every_year, requests=every_year.requests + second_reads)


_ONE_YEAR = [[2022, 2023], [2023]]
_NEWER_PARTIAL_YEAR = [[2022, 2023], [2023, 2024]]
# Nine series reach 2025 and 85 end in 2024, the year the member elects.
_NEWEST_YEAR_THIN = [*([[2024, 2025]] * 9), *([[2023, 2024]] * 85)]


def _latest_estimates(
    members: list[list[list[int]]], *, build_year: int | None = None, facets: int = 1
) -> tuple[Estimate, Estimate]:
    """The `all` and `latest` estimates of one request per entry of `members`, each a
    variable holding one series per list of years, spread over `facets` facets, from its
    own donor to Kenya."""
    stat_vars = [f"Amount_Crs_Latest{position}" for position in range(len(members))]
    index = build_coverage_index(
        [
            (stat_var, f"facet{donor % facets}", f"country/D{donor:03d}", "country/KEN", years)
            for stat_var, series in zip(stat_vars, members, strict=True)
            for donor, years in enumerate(series)
        ],
        role_names=dict.fromkeys(stat_vars, _CRS_ROLES),
        build_year=build_year,
    )
    every_year, latest = (
        sync_estimate(
            [_request(stat_var=sv, recipient="country/KEN", window=window) for sv in stat_vars],
            index=index,
            member_concurrency=_SETTINGS.mint_member_max_concurrency,
        )
        for window in (_ALL_WINDOW, RequestedWindow(kind="latest"))
    )
    return every_year, latest


@pytest.mark.parametrize(
    ("members", "build_year", "facets", "second_reads", "requests"),
    [
        pytest.param([_ONE_YEAR] * 40, 2026, 1, 0, 5, id="forty-ending-in-one-year"),
        pytest.param([_ONE_YEAR], None, 2, 0, 1, id="two-facets-ending-in-one-year"),
        pytest.param([_NEWER_PARTIAL_YEAR], 2026, 1, 1, 2, id="a-newer-partial-year"),
        pytest.param(
            [*([_ONE_YEAR] * 37), *([_NEWER_PARTIAL_YEAR] * 3)], 2026, 1, 3, 8, id="mixed-group"
        ),
        pytest.param([_NEWEST_YEAR_THIN], 2026, 1, 1, 2, id="newest-year-thin"),
        pytest.param([_ONE_YEAR], 2022, 1, 1, 2, id="one-year-after-the-build-year"),
    ],
)
def test_latest_window_prices_a_second_read_only_for_a_member_that_can_make_it(
    members: list[list[list[int]]],
    build_year: int | None,
    facets: int,
    second_reads: int,
    requests: int,
):
    """A member whose series all end in one year, at or before the build year, elects that
    year and reads once, so it is priced as an `all` window is. Any other covered member
    can hold a row after the year it elects and is priced one more read."""
    every_year, latest = _latest_estimates(members, build_year=build_year, facets=facets)

    assert latest == dataclasses.replace(every_year, requests=every_year.requests + second_reads)
    assert latest.requests == requests


def test_members_ending_in_one_year_keep_the_budgets_an_all_window_meets():
    """At the default budgets 37 `latest` members whose series end in one year run inline
    and 715 are accepted as an operation, as the same members are under an `all` window."""
    _, inline = _latest_estimates([_ONE_YEAR] * 37, build_year=2026)
    _, queued = _latest_estimates([_ONE_YEAR] * 715, build_year=2026)

    assert within_sync_budget(
        inline,
        sync_budget_rows=_SETTINGS.sync_budget_rows,
        sync_budget_requests=_SETTINGS.sync_budget_requests,
    )
    assert queued.requests > _SETTINGS.sync_budget_requests
    refuse_async_over_ceiling(
        queued,
        async_max_rows=_SETTINGS.async_max_rows,
        async_max_requests=_SETTINGS.async_max_requests,
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
    """A group of 13 variables splits into four requests of at most four members, the lane
    floor (`generation.dispatch_lanes`) applying while the group is under 32 members.

    The bound DEU/KEN filter and shared role pair keep this below the URL budget."""
    member_count = 13
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
    assert estimate.requests == 4


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


_ONE_SHEET = EXCEL_MAX_ROWS - 1
_TWO_SHEETS = 2 * (EXCEL_MAX_ROWS - 1)


@pytest.mark.parametrize(
    ("rows", "format", "splits_totals", "refused"),
    [
        (_ONE_SHEET, "xlsx", False, False),
        (_ONE_SHEET + 1, "xlsx", False, True),
        (_ONE_SHEET + 1, "csv", False, False),
        (_ONE_SHEET + 1, "parquet", False, False),
        (_ONE_SHEET + 1, "xlsx", True, False),
        (_TWO_SHEETS, "xlsx", True, False),
        (_TWO_SHEETS + 1, "xlsx", True, True),
        (_TWO_SHEETS + 1, "csv", True, False),
    ],
)
def test_refuse_excel_over_estimate_raises_only_for_xlsx_over_the_limit(
    rows: int, format: str, splits_totals: bool, refused: bool
):
    """Excel counts its header toward `EXCEL_MAX_ROWS`, so `EXCEL_MAX_ROWS - 1` data rows fit
    one sheet. A render that can keep totals on a second sheet refuses only above two sheets'
    room; the writer judges each sheet."""
    estimate = Estimate(rows=rows, requests=1)
    if not refused:
        refuse_excel_over_estimate(estimate, format=format, splits_totals=splits_totals)
        return
    with pytest.raises(MaterialisationError) as excinfo:
        refuse_excel_over_estimate(estimate, format=format, splits_totals=splits_totals)
    assert excinfo.value.code == "excel_limit"
    assert "csv or parquet" in (excinfo.value.detail or "")


@pytest.mark.parametrize(
    ("rows", "splits_totals", "refused"),
    [
        (_ONE_SHEET, False, False),
        (_ONE_SHEET + 1, False, True),
        (_ONE_SHEET + 1, True, False),
        (_TWO_SHEETS, True, False),
        (_TWO_SHEETS + 1, True, True),
    ],
)
def test_refuse_excel_over_stored_rows_follows_the_sheet_room(
    rows: int, splits_totals: bool, refused: bool
):
    """The stored row count meets the same one-sheet or two-sheet room as the estimate."""
    if not refused:
        refuse_excel_over_stored_rows(rows, splits_totals=splits_totals)
        return
    with pytest.raises(MaterialisationError) as excinfo:
        refuse_excel_over_stored_rows(rows, splits_totals=splits_totals)
    assert excinfo.value.code == "excel_limit"


_OPEN_ORIGIN = {"open_roles": ["origin"]}
_OPEN_DESTINATION = {"open_roles": ["destination"]}
_ONE_ORIGIN = {"entities": {"origin": ["country/GBR"]}}
_TWO_ORIGINS = {"entities": {"origin": ["country/GBR", "country/FRA"]}}
_ORIGIN_AND_TOTAL = {"entities": {"origin": ["country/GBR", "DACDonors"]}}
# GBR and FRA are parties on both axes; the only total, `DACDonors`, is a donor in `p1`.
_PARTIES = {
    "country/GBR": PartyRow(
        dcid="country/GBR", kind="party", datasets=("p1",), axes=("destination", "origin")
    ),
    "country/FRA": PartyRow(
        dcid="country/FRA", kind="party", datasets=("p1",), axes=("destination", "origin")
    ),
    "DACDonors": PartyRow(dcid="DACDonors", kind="total", datasets=("p1",), axes=("origin",)),
}


@pytest.mark.parametrize(
    ("options", "identity", "datasets", "expected"),
    [
        ({"format": "xlsx"}, _OPEN_ORIGIN, (), True),
        ({"format": "xlsx"}, _OPEN_ORIGIN, ("p1",), True),
        ({"format": "xlsx"}, _OPEN_ORIGIN, ("p2",), False),
        ({"format": "xlsx"}, _OPEN_DESTINATION, (), False),
        ({"format": "xlsx"}, _TWO_ORIGINS, (), False),
        ({"format": "xlsx"}, _ORIGIN_AND_TOTAL, (), True),
        ({"format": "xlsx"}, _ONE_ORIGIN, (), False),
        ({"format": "xlsx"}, {}, (), False),
        ({"format": "csv"}, _OPEN_ORIGIN, (), False),
        (
            {"format": "xlsx", "pivot": {"index": "recipient", "columns": "time_period"}},
            _OPEN_ORIGIN,
            (),
            False,
        ),
    ],
    ids=[
        "open-total-any-dataset",
        "open-total-read-dataset",
        "open-total-unread-dataset",
        "open-no-total",
        "several-parties",
        "several-with-total",
        "one-bound",
        "no-axes",
        "csv",
        "pivot",
    ],
)
def test_xlsx_splits_totals_when_the_render_keeps_totals_apart(
    options, identity, datasets, expected
):
    """The predicate agrees with the renderer's split (`movable_totals_roles`): an `xlsx`
    render with no pivot where a role bound to several entities holds an index-listed total,
    or an open role's axis has one in a dataset the read selects (any, when none is known)."""
    assert (
        xlsx_splits_totals(
            MintOptions(**options),
            identity=identity,
            role_columns=identity_role_keys(identity),
            parties=_PARTIES,
            datasets=datasets,
        )
        is expected
    )


def _elected(provenance: str | None) -> MaterialisationRequest:
    request = _request(stat_var="Amount_Crs_A")
    if provenance is None:
        return request
    return dataclasses.replace(
        request, dataset=ElectedDataset(provenance=provenance, member_provenances=(provenance,))
    )


@pytest.mark.parametrize(
    ("provenances", "expected"),
    [
        (["p1", "p2", "p1"], ("p1", "p2")),
        (["p1", None], ()),
        ([], ()),
    ],
    ids=["every-request-elects", "one-elects-none", "no-requests"],
)
def test_elected_provenances_name_the_read_only_when_every_request_elects(provenances, expected):
    """A request with no elected dataset can read any provenance, so the set is unknown."""
    assert elected_provenances([_elected(p) for p in provenances]) == expected


def _plan_xlsx_with_stored_rows(rows: int, *, identity: dict) -> None:
    manifest = SimpleNamespace(
        trace={"resolved_stat_vars": [], "facets": []},
        coverage=SimpleNamespace(rows=rows),
        definition=SimpleNamespace(identity=identity),
    )
    plan_stored_render(
        cast(Manifest, manifest),
        options=MintOptions(format="xlsx"),
        subject_id="s",
        content_version="v",
        read_version=lambda: None,
        read_locations=lambda: (),
        deps=cast(DefinitionDeps, SimpleNamespace(code_index=SimpleNamespace(parties=_PARTIES))),
    )


def test_the_stored_render_plan_admits_a_split_xlsx_between_one_and_two_sheets():
    """`plan_stored_render` passes a stored count past one sheet when the identity splits
    totals, and refuses it when it does not."""
    _plan_xlsx_with_stored_rows(_ONE_SHEET + 1, identity=_OPEN_ORIGIN)

    for identity in (_ONE_ORIGIN, _TWO_ORIGINS):
        with pytest.raises(MaterialisationError) as excinfo:
            _plan_xlsx_with_stored_rows(_ONE_SHEET + 1, identity=identity)
        assert excinfo.value.code == "excel_limit"


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
