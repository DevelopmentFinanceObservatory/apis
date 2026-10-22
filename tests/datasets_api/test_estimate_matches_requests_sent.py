"""These tests compare admission's SDMX request estimate with real dispatch across definitions
and group materialisation.

The shared planner applies lane targets and URL-byte limits. Tests cover four lanes and long dcids.
Latest-window filters count against the same URL budget.
"""

from __future__ import annotations

import httpx
import pytest

from datasets_api.admission import sync_estimate
from datasets_api.config import DatasetsBuildSettings
from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
)
from materialise.facets import FacetBinding
from materialise.window import RequestedWindow
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.materialise.conftest import FakeGraphReader

from .test_serve_bounds import (
    _BASE_URL,
    _FACET,
    _RECIPIENT,
    _aggregate_ceiling_request,
    _coverage_index,
    _csv_body,
    _graph_for,
    _sdmx_transport,
)

_LATEST_BUDGET_DONOR = "country/USA"
_LATEST_BUDGET_RECIPIENT = "country/KEN"
_LATEST_BUDGET_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_SETTINGS = DatasetsBuildSettings()


def test_estimate_480_variables_near_the_url_budget_packs_one_chunk_without_latest_two_with():
    """480 variables fit one request without `c[TIME_PERIOD]`. `LATEST` pushes them over the
    URL budget and requires two requests."""
    variables = [f"V{i:012d}" for i in range(480)]
    index = build_coverage_index(
        [
            (variable, "facet1", _LATEST_BUDGET_DONOR, _LATEST_BUDGET_RECIPIENT, [2020])
            for variable in variables
        ],
        role_names=dict.fromkeys(variables, _LATEST_BUDGET_ROLES),
    )
    bound = {
        "donor": frozenset({_LATEST_BUDGET_DONOR}),
        "recipient": frozenset({_LATEST_BUDGET_RECIPIENT}),
    }
    without_latest = index.estimate(variables=variables, bound=bound)
    with_latest = index.estimate(variables=variables, bound=bound, time_period="LATEST")
    assert without_latest.requests == 1
    assert with_latest.requests == 2


def _run(stat_vars: list[str]) -> tuple[int, int]:
    """Materialises the StatVars as one group and returns actual and estimated request
    counts."""
    bare_stat_vars = [sv.removeprefix("dcid:") for sv in stat_vars]
    requests: list[httpx.Request] = []
    bodies = {sv: _csv_body(stat_var=sv, num_rows=2) for sv in bare_stat_vars}
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    coverage = _coverage_index([(sv, _RECIPIENT) for sv in bare_stat_vars])
    member_requests = [
        _aggregate_ceiling_request(stat_var=sv, recipient=_RECIPIENT) for sv in stat_vars
    ]

    estimate = sync_estimate(
        member_requests,
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    graph = _graph_for(stat_vars)
    materialise_group(
        member_requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        ceiling=1_000_000,
        materialiser_build="test-build",
        capability_version="cap-v1",
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
        coverage=coverage,
    )
    return len(requests), estimate.requests


@pytest.mark.parametrize(
    "member_count",
    [1, 4, 11, 12, 15, 44, 61, 96],
    ids=[
        "single-member",
        "exactly-the-default-lane-count",
        "one-over-the-old-fixed-cap",
        "one-more",
        "mid-size",
        "the-original-44-member-benchmark-size",
        "past-the-default-lane-count-by-more-than-one-target",
        "bulk-scale",
    ],
)
def test_estimate_requests_equals_sdmx_requests_sent(member_count: int):
    """Short dcids keep the URL budget loose, so counts follow the lane target. The largest
    case uses 96 members."""
    stat_vars = [f"dcid:EstimateProofVar{i}" for i in range(member_count)]
    requests_sent, estimate_requests = _run(stat_vars)
    assert requests_sent == estimate_requests
    assert requests_sent > 0


def test_estimate_requests_equals_sdmx_requests_sent_when_the_byte_budget_binds_first():
    """Thirty-two 967-character dcids form four groups of eight at the default four lanes.
    Each group exceeds the 7,800-byte URL budget, so dispatch splits beyond four requests."""
    stat_vars = [f"dcid:EstimateProofVar{'X' * 950}{i}" for i in range(32)]
    requests_sent, estimate_requests = _run(stat_vars)
    assert requests_sent == estimate_requests
    assert requests_sent > 4
    assert requests_sent < 16


def _run_group(stat_vars: list[str]) -> tuple[int, int]:
    """Runs `materialise_group` for one recipient and returns actual and estimated counts."""
    bare_stat_vars = [sv.removeprefix("dcid:") for sv in stat_vars]
    requests: list[httpx.Request] = []
    bodies = {sv: _csv_body(stat_var=sv, num_rows=2) for sv in bare_stat_vars}
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    coverage = _coverage_index([(sv, _RECIPIENT) for sv in bare_stat_vars])
    graph = FakeGraphReader(facets={(sv, _RECIPIENT): [_FACET] for sv in bare_stat_vars})
    # `materialise_group` requires the `dcid:` prefix on each `stat_var`.
    member_requests = [
        _aggregate_ceiling_request(stat_var=sv, recipient=_RECIPIENT) for sv in stat_vars
    ]

    estimate = sync_estimate(
        member_requests, index=coverage, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )

    materialise_group(
        member_requests,
        sdmx=sdmx,
        graph=graph,
        ceiling=1_000_000,
        materialiser_build="test-build",
        capability_version="cap-v1",
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
        coverage=coverage,
    )

    return len(requests), estimate.requests


def test_group_dispatch_requests_equals_admitted_requests_past_the_dispatch_chunk_target():
    """Thirty-three members exceed the four-lane, eight-member chunk target. Actual dispatch
    and admission must report the same request count."""
    member_count = 4 * 8 + 1
    stat_vars = [f"dcid:LanesProofVar{i}" for i in range(member_count)]
    requests_sent, estimate_requests = _run_group(stat_vars)
    assert requests_sent == estimate_requests
    assert requests_sent > _SETTINGS.mint_member_max_concurrency


_UNBOUND_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_UNBOUND_DONORS = ("country/DEU", "country/FRA")
_UNBOUND_RECIPIENTS = ("country/ETH", "country/KEN", "country/UGA")
_UNBOUND_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId"
    "\r\n"
)


def test_a_fully_unbound_role_bearing_group_fetches_every_donor_and_matches_its_estimate():
    """Two members bind neither role, as an "every donor to every recipient" spec does. The
    probe reads coverage across every entity, each fetch lists every covered donor (the smaller
    axis) and leaves recipients open, and admission prices the rows and requests sent."""
    bare_stat_vars = ["EstimateProofUnboundGrants", "EstimateProofUnboundLoans"]
    coverage = build_coverage_index(
        [
            (stat_var, "facet1", donor, recipient, [2020])
            for stat_var in bare_stat_vars
            for donor in _UNBOUND_DONORS
            for recipient in _UNBOUND_RECIPIENTS
        ],
        role_names=dict.fromkeys(bare_stat_vars, _UNBOUND_ROLES),
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{donor},{recipient},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
            for stat_var in str(request.url.params["c[variableMeasured]"]).split(",")
            for donor in _UNBOUND_DONORS
            for recipient in _UNBOUND_RECIPIENTS
        )
        return httpx.Response(
            200,
            text=_UNBOUND_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    member_requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{stat_var}",
            role_bindings={},
            declared_roles=(_UNBOUND_ROLES.donor, _UNBOUND_ROLES.recipient),
            facet_bindings={
                "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
                "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
                "unit": FacetBinding(state="bound", ref="USD"),
            },
            window=RequestedWindow(kind="all"),
        )
        for stat_var in bare_stat_vars
    ]
    graph = FakeGraphReader()
    estimate = sync_estimate(
        member_requests,
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    result = materialise_group(
        member_requests,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="test-build",
        capability_version="cap-v1",
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
        coverage=coverage,
    )

    assert estimate.rows == 12
    assert len(sent) == estimate.requests
    for request in sent:
        assert request.url.params["c[financingProvider]"] == ",".join(_UNBOUND_DONORS)
        assert "c[financingRecipient]" not in request.url.params
    assert result.coverage.rows == estimate.rows
    assert graph.calls["observation_facets_batch_many"] == 0


def test_an_unbound_role_bearing_group_with_a_range_window_matches_its_estimate():
    """In the 2020 window one donor reaches two recipients. Across all years 250 donors reach
    251 recipients, so each fetch lists the 250 donors, which leaves room for one StatVar a
    request. Admission prices that same list: eight requests where a filter clipped to the
    window would predict four."""
    bare_stat_vars = [f"EstimateProofRangeWindow{i}{'x' * 800}" for i in range(8)]
    donors = [f"country/RANGEDONOR{i:03d}" for i in range(249)]
    recipients = [f"country/RANGERECIPIENT{i:03d}" for i in range(249)]
    coverage = build_coverage_index(
        [
            row
            for stat_var in bare_stat_vars
            for row in (
                (stat_var, "facet1", "country/DEU", "country/KEN", [2020]),
                (stat_var, "facet1", "country/DEU", "country/ETH", [2020]),
                *(
                    (stat_var, "facet1", donor, recipient, [2010])
                    for donor, recipient in zip(donors, recipients, strict=True)
                ),
            )
        ],
        role_names=dict.fromkeys(bare_stat_vars, _UNBOUND_ROLES),
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},country/DEU,{recipient},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
            for stat_var in str(request.url.params["c[variableMeasured]"]).split(",")
            for recipient in ("country/KEN", "country/ETH")
        )
        return httpx.Response(
            200,
            text=_UNBOUND_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    member_requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{stat_var}",
            role_bindings={},
            declared_roles=(_UNBOUND_ROLES.donor, _UNBOUND_ROLES.recipient),
            facet_bindings={
                "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
                "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
                "unit": FacetBinding(state="bound", ref="USD"),
            },
            window=RequestedWindow(kind="range", start_year=2020, end_year=2020),
        )
        for stat_var in bare_stat_vars
    ]
    graph = FakeGraphReader()
    estimate = sync_estimate(
        member_requests,
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    result = materialise_group(
        member_requests,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="test-build",
        capability_version="cap-v1",
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
        coverage=coverage,
    )

    all_years_donors = ",".join(sorted(["country/DEU", *donors]))
    assert estimate.rows == 16
    assert len(sent) == estimate.requests == 8
    for request in sent:
        assert request.url.params["c[financingProvider]"] == all_years_donors
        assert "c[financingRecipient]" not in request.url.params
    assert result.coverage.rows == estimate.rows


_WINDOW_FACET_METADATA = {
    "facet1": FacetMetadata(
        provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
    )
}
_WINDOW_DONORS = [f"country/WINDOWDONOR{i:03d}" for i in range(250)]
_WINDOW_RECIPIENTS = [f"country/WINDOWRECIPIENT{i:03d}" for i in range(250)]


def _run_windowed_role_bearing_group(
    bare_stat_vars: list[str],
    *,
    role_bindings: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    covered_in_window: int,
) -> tuple[list[httpx.Request], int, int, int]:
    """Runs a role-bearing group over a 2020 range window and returns the requests sent,
    the estimate's requests and rows, and the rows served.

    Every StatVar covers every pair in `pairs` in 2010. The first `covered_in_window` StatVars
    also cover the first pair in 2020, and the SDMX double serves that one row for each.
    """
    in_window = set(bare_stat_vars[:covered_in_window])
    first_donor, first_recipient = pairs[0]
    coverage = build_coverage_index(
        [
            (
                stat_var,
                "facet1",
                donor,
                recipient,
                [2010, 2020] if stat_var in in_window and position == 0 else [2010],
            )
            for stat_var in bare_stat_vars
            for position, (donor, recipient) in enumerate(pairs)
        ],
        role_names=dict.fromkeys(bare_stat_vars, _UNBOUND_ROLES),
        facet_metadata=_WINDOW_FACET_METADATA,
    )
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{first_donor},{first_recipient},USD,"
            f"GrantEquivalent,NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
            for stat_var in str(request.url.params["c[variableMeasured]"]).split(",")
            if stat_var in in_window
        )
        return httpx.Response(
            200,
            text=_UNBOUND_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    member_requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{stat_var}",
            role_bindings=role_bindings,
            declared_roles=(_UNBOUND_ROLES.donor, _UNBOUND_ROLES.recipient),
            facet_bindings={
                "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
                "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
                "unit": FacetBinding(state="bound", ref="USD"),
            },
            window=RequestedWindow(kind="range", start_year=2020, end_year=2020),
        )
        for stat_var in bare_stat_vars
    ]
    graph = FakeGraphReader()
    estimate = sync_estimate(
        member_requests,
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    result = materialise_group(
        member_requests,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="test-build",
        capability_version="cap-v1",
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
        coverage=coverage,
    )

    served = result.coverage.rows
    return sent, estimate.requests, estimate.rows, served


@pytest.mark.parametrize(
    "covered_in_window", [0, 1], ids=["no-coverage-in-window", "one-stat-var-covered-in-window"]
)
def test_an_unbound_group_prices_every_request_its_fetch_sends_outside_the_window(
    covered_in_window: int,
):
    """Ninety-six StatVars cover 250 donors in 2010 and the window asks for 2020. The fetch
    still names every StatVar beside the 250-donor list, which leaves room for three
    288-character dcids a request, so each of its twelve chunks of eight sends three. Admission
    prices all 36 while its row count follows the window."""
    bare_stat_vars = [f"EstimateProofOutsideWindow{i:02d}{'x' * 260}" for i in range(96)]
    sent, estimate_requests, estimate_rows, served = _run_windowed_role_bearing_group(
        bare_stat_vars,
        role_bindings={},
        pairs=list(zip(_WINDOW_DONORS, _WINDOW_RECIPIENTS, strict=True)),
        covered_in_window=covered_in_window,
    )
    assert estimate_rows == served == covered_in_window
    assert len(sent) == estimate_requests == 36
    for request in sent:
        assert request.url.params["c[financingProvider]"] == ",".join(_WINDOW_DONORS)


@pytest.mark.parametrize(
    "covered_in_window", [0, 1], ids=["no-coverage-in-window", "one-stat-var-covered-in-window"]
)
def test_a_bound_role_group_prices_every_request_its_fetch_sends_outside_the_window(
    covered_in_window: int,
):
    """Sixteen StatVars bind one recipient that they cover in 2010 and the window asks for
    2020. The fetch sends each of its four chunks, and admission prices all four."""
    bare_stat_vars = [f"EstimateProofBoundOutsideWindow{i:02d}" for i in range(16)]
    sent, estimate_requests, estimate_rows, served = _run_windowed_role_bearing_group(
        bare_stat_vars,
        role_bindings={_UNBOUND_ROLES.recipient: ["country/KEN"]},
        pairs=[("country/DEU", "country/KEN"), ("country/FRA", "country/KEN")],
        covered_in_window=covered_in_window,
    )
    assert estimate_rows == served == covered_in_window
    assert len(sent) == estimate_requests == 4
    for request in sent:
        assert request.url.params["c[financingRecipient]"] == "country/KEN"
