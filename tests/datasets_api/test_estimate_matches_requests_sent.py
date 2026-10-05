"""These tests compare admission's SDMX request estimate with real dispatch across definitions,
collections, and group materialisation.

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
    materialise_collection,
    materialise_group,
)
from materialise.facets import FacetBinding
from materialise.generation import _plan_collection_chunks, dispatch_lanes
from materialise.revision import RevisionSnapshot
from materialise.window import RequestedWindow
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.materialise.conftest import FakeGraphReader, FakeRevisionReader

from .test_serve_bounds import (
    _BASE_URL,
    _FACET,
    _RECIPIENT,
    _aggregate_ceiling_request,
    _aggregate_ceiling_spec_id,
    _client,
    _coverage_index,
    _coverage_service,
    _csv_body,
    _csv_body_for,
    _graph_for,
    _mint_body,
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


def _run(monkeypatch: pytest.MonkeyPatch, stat_vars: list[str]) -> tuple[int, int]:
    """Mints the StatVars and returns actual and estimated request counts."""
    bare_stat_vars = [sv.removeprefix("dcid:") for sv in stat_vars]
    requests: list[httpx.Request] = []
    bodies = {sv: _csv_body(stat_var=sv, num_rows=2) for sv in bare_stat_vars}
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    coverage = _coverage_service([(sv, _RECIPIENT) for sv in bare_stat_vars])

    estimate = sync_estimate(
        [_aggregate_ceiling_request(stat_var=sv, recipient=_RECIPIENT) for sv in bare_stat_vars],
        index=coverage.current(),
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    client = _client(
        monkeypatch=monkeypatch,
        ceiling=1_000_000,
        graph=_graph_for(stat_vars),
        sdmx=sdmx,
        coverage=coverage,
    )

    resp = client.post("/api/datasets", json=_mint_body(statvars=stat_vars))
    assert resp.status_code == 200, resp.text
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
        "bulk-collection-scale",
    ],
)
def test_estimate_requests_equals_sdmx_requests_sent(
    monkeypatch: pytest.MonkeyPatch, member_count: int
):
    """Short dcids keep the URL budget loose, so counts follow the lane target. The largest
    case uses 96 members, below the 100-member cap."""
    stat_vars = [f"dcid:EstimateProofVar{i}" for i in range(member_count)]
    requests_sent, estimate_requests = _run(monkeypatch, stat_vars)
    assert requests_sent == estimate_requests
    assert requests_sent > 0


def test_estimate_requests_equals_sdmx_requests_sent_when_the_byte_budget_binds_first(
    monkeypatch: pytest.MonkeyPatch,
):
    """Thirty-two 967-character dcids form four groups of eight at the default four lanes.
    Each group exceeds the 7,800-byte URL budget, so dispatch splits beyond four requests."""
    stat_vars = [f"dcid:EstimateProofVar{'X' * 950}{i}" for i in range(32)]
    requests_sent, estimate_requests = _run(monkeypatch, stat_vars)
    assert requests_sent == estimate_requests
    assert requests_sent > 4
    assert requests_sent < 16


def _collection_request(
    *, stat_var: str, recipient: str, window: RequestedWindow
) -> MaterialisationRequest:
    """Builds a request with an explicit window for collection tests."""
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="bound", ref="USD"),
        },
        window=window,
    )


def _run_collection(
    members: list[tuple[str, str, RequestedWindow]], *, lanes: int
) -> tuple[int, int]:
    """Runs a collection and returns actual and estimated request counts."""
    requests: list[httpx.Request] = []
    bodies = {
        stat_var.removeprefix("dcid:"): _csv_body_for(
            stat_var=stat_var, recipient=recipient, num_rows=2
        )
        for stat_var, recipient, _window in members
    }
    graph = FakeGraphReader(
        facets={
            (stat_var.removeprefix("dcid:"), recipient): [_FACET]
            for stat_var, recipient, _window in members
        }
    )
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    snapshot = RevisionSnapshot(graph_revision="rev-1", source_revisions=())
    reader = FakeRevisionReader([snapshot] * len(members))
    # Coverage rows are unique by StatVar and recipient.
    coverage = _coverage_index(
        list(dict.fromkeys((stat_var, recipient) for stat_var, recipient, _window in members))
    )

    # Position suffixes keep spec ids unique when members repeat a StatVar.
    member_requests = [
        (
            _aggregate_ceiling_spec_id(f"{stat_var}#{i}"),
            _collection_request(stat_var=stat_var, recipient=recipient, window=window),
        )
        for i, (stat_var, recipient, window) in enumerate(members)
    ]

    estimate = sync_estimate(
        [request for _spec_id, request in member_requests],
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    materialise_collection(
        member_requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="test-build",
        capability_version="cap-v1",
        member_concurrency=lanes,
        coverage=coverage,
    )

    return len(requests), estimate.requests


def test_estimate_requests_equals_sdmx_requests_sent_for_mixed_window_same_scope():
    """Four latest-window and four all-window members share a scope but require separate request
    groups because each request carries only one `c[TIME_PERIOD]` filter."""
    lanes = 4
    members = [
        (f"dcid:EstimateProofMixedWindowLatest{i}", _RECIPIENT, RequestedWindow(kind="latest"))
        for i in range(4)
    ] + [
        (f"dcid:EstimateProofMixedWindowAll{i}", _RECIPIENT, RequestedWindow(kind="all"))
        for i in range(4)
    ]
    requests_sent, estimate_requests = _run_collection(members, lanes=lanes)
    assert requests_sent == estimate_requests
    assert requests_sent > 0


def test_estimate_requests_equals_sdmx_requests_sent_for_a_repeated_bare_stat_var():
    """Two members share a StatVar but need separate requests. The third is distinct, so the
    estimate must preserve duplicates and return three requests."""
    lanes = 4
    repeated = "dcid:EstimateProofRepeatedBatchKey"
    members = [
        (repeated, _RECIPIENT, RequestedWindow(kind="all")),
        (repeated, _RECIPIENT, RequestedWindow(kind="all")),
        ("dcid:EstimateProofRepeatedBatchKeyDistinct", _RECIPIENT, RequestedWindow(kind="all")),
    ]
    requests_sent, estimate_requests = _run_collection(members, lanes=lanes)
    assert requests_sent == estimate_requests
    assert requests_sent == 3


def test_estimate_and_real_dispatch_agree_on_chunk_shape_with_an_uncovered_member():
    """Two members repeat a StatVar and one lacks `RoleNames`. The planners make three chunks.
    The uncovered member adds one to `uncovered_requests` and zero to `requests`."""
    lanes = dispatch_lanes(_SETTINGS.mint_member_max_concurrency, 3)
    repeated = "dcid:EstimateProofChunkShapeRepeated"
    uncovered = "dcid:EstimateProofChunkShapeUncovered"
    requests = [
        _collection_request(
            stat_var=repeated, recipient=_RECIPIENT, window=RequestedWindow(kind="all")
        ),
        _collection_request(
            stat_var=repeated, recipient=_RECIPIENT, window=RequestedWindow(kind="all")
        ),
        _collection_request(
            stat_var=uncovered, recipient=_RECIPIENT, window=RequestedWindow(kind="all")
        ),
    ]
    # Coverage has no row for `uncovered`.
    coverage = _coverage_index([(repeated, _RECIPIENT)])

    real_chunks = _plan_collection_chunks(
        [(f"spec_{i}", request) for i, request in enumerate(requests)],
        places=[()] * len(requests),
        lanes=lanes,
    )
    estimate = sync_estimate(
        requests, index=coverage, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )

    assert len(real_chunks) == 3
    assert estimate.requests == 2
    assert estimate.uncovered_requests == 1


def test_estimate_requests_equals_sdmx_requests_sent_for_mixed_scope():
    """Each of two recipient scopes forms its own lane-chunked request group."""
    lanes = 4
    members = [
        (f"dcid:EstimateProofMixedScopeKen{i}", "country/KEN", RequestedWindow(kind="all"))
        for i in range(3)
    ] + [
        (f"dcid:EstimateProofMixedScopeUga{i}", "country/UGA", RequestedWindow(kind="all"))
        for i in range(3)
    ]
    requests_sent, estimate_requests = _run_collection(members, lanes=lanes)
    assert requests_sent == estimate_requests
    assert requests_sent > 0


def test_estimate_requests_equals_sdmx_requests_sent_for_a_latest_window_byte_budget_case():
    """Thirty-two 963-character dcids form four groups of eight. Adding `LATEST` exceeds the
    byte budget per group, so dispatch sends eight requests."""
    members = [
        (
            f"dcid:EstimateProofLatestByteBudget{'X' * 933}{i}",
            _RECIPIENT,
            RequestedWindow(kind="latest"),
        )
        for i in range(32)
    ]
    requests_sent, estimate_requests = _run_collection(members, lanes=4)
    assert requests_sent == estimate_requests
    assert requests_sent == 8


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
    reader = FakeRevisionReader(
        [RevisionSnapshot(graph_revision="rev-1", source_revisions=())] * len(stat_vars)
    )

    estimate = sync_estimate(
        member_requests, index=coverage, member_concurrency=_SETTINGS.mint_member_max_concurrency
    )

    materialise_group(
        member_requests,
        sdmx=sdmx,
        graph=graph,
        revision_reader=reader,
        ceiling=1_000_000,
        materialiser_build="test-build",
        capability_version="cap-v1",
        enrichment=EnrichmentLookup(graph),
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


def test_a_fully_unbound_role_bearing_collection_fetches_every_donor_and_matches_its_estimate():
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
        (
            _aggregate_ceiling_spec_id(stat_var),
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
            ),
        )
        for stat_var in bare_stat_vars
    ]
    graph = FakeGraphReader()
    estimate = sync_estimate(
        [request for _spec_id, request in member_requests],
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    result = materialise_collection(
        member_requests,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader(
            [RevisionSnapshot(graph_revision="rev-1", source_revisions=())] * 2
        ),
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
    assert sum(member.coverage.rows for _spec_id, member in result.members) == estimate.rows
    assert graph.calls["observation_facets_batch_many"] == 0


def test_an_unbound_role_bearing_collection_with_a_range_window_matches_its_estimate():
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
        (
            _aggregate_ceiling_spec_id(stat_var),
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
            ),
        )
        for stat_var in bare_stat_vars
    ]
    graph = FakeGraphReader()
    estimate = sync_estimate(
        [request for _spec_id, request in member_requests],
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    result = materialise_collection(
        member_requests,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader(
            [RevisionSnapshot(graph_revision="rev-1", source_revisions=())] * 2
        ),
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
    assert sum(member.coverage.rows for _spec_id, member in result.members) == estimate.rows


_WINDOW_FACET_METADATA = {
    "facet1": FacetMetadata(
        provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
    )
}
_WINDOW_DONORS = [f"country/WINDOWDONOR{i:03d}" for i in range(250)]
_WINDOW_RECIPIENTS = [f"country/WINDOWRECIPIENT{i:03d}" for i in range(250)]


def _run_windowed_role_bearing_collection(
    bare_stat_vars: list[str],
    *,
    role_bindings: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    covered_in_window: int,
) -> tuple[list[httpx.Request], int, int, int]:
    """Runs a role-bearing collection over a 2020 range window and returns the requests sent,
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
        (
            _aggregate_ceiling_spec_id(stat_var),
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
            ),
        )
        for stat_var in bare_stat_vars
    ]
    graph = FakeGraphReader()
    estimate = sync_estimate(
        [request for _spec_id, request in member_requests],
        index=coverage,
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
    )

    result = materialise_collection(
        member_requests,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader(
            [RevisionSnapshot(graph_revision="rev-1", source_revisions=())] * 2
        ),
        materialiser_build="test-build",
        capability_version="cap-v1",
        member_concurrency=_SETTINGS.mint_member_max_concurrency,
        coverage=coverage,
    )

    served = sum(member.coverage.rows for _spec_id, member in result.members)
    return sent, estimate.requests, estimate.rows, served


@pytest.mark.parametrize(
    "covered_in_window", [0, 1], ids=["no-coverage-in-window", "one-stat-var-covered-in-window"]
)
def test_an_unbound_collection_prices_every_request_its_fetch_sends_outside_the_window(
    covered_in_window: int,
):
    """Ninety-six StatVars cover 250 donors in 2010 and the window asks for 2020. The fetch
    still names every StatVar beside the 250-donor list, which leaves room for three
    288-character dcids a request, so each of its twelve chunks of eight sends three. Admission
    prices all 36 while its row count follows the window."""
    bare_stat_vars = [f"EstimateProofOutsideWindow{i:02d}{'x' * 260}" for i in range(96)]
    sent, estimate_requests, estimate_rows, served = _run_windowed_role_bearing_collection(
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
def test_a_bound_role_collection_prices_every_request_its_fetch_sends_outside_the_window(
    covered_in_window: int,
):
    """Sixteen StatVars bind one recipient that they cover in 2010 and the window asks for
    2020. The fetch sends each of its four chunks, and admission prices all four."""
    bare_stat_vars = [f"EstimateProofBoundOutsideWindow{i:02d}" for i in range(16)]
    sent, estimate_requests, estimate_rows, served = _run_windowed_role_bearing_collection(
        bare_stat_vars,
        role_bindings={_UNBOUND_ROLES.recipient: ["country/KEN"]},
        pairs=[("country/DEU", "country/KEN"), ("country/FRA", "country/KEN")],
        covered_in_window=covered_in_window,
    )
    assert estimate_rows == served == covered_in_window
    assert len(sent) == estimate_requests == 4
    for request in sent:
        assert request.url.params["c[financingRecipient]"] == "country/KEN"
