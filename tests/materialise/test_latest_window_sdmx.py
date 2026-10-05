"""Checks `c[TIME_PERIOD]=LATEST` across single, collection, group, and preview fetch paths.

Latest and all-window members use separate chunks. A blank latest value yields no row on every
covered path.
"""

from __future__ import annotations

import httpx

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_collection,
    materialise_group,
    materialise_preview,
    materialise_spec,
)
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.revision import RevisionSnapshot, SourceRevision
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader, FakeRevisionReader, only_chunk

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR_A = "dcid:LatestWindowMemberA"
_STAT_VAR_B = "dcid:LatestWindowMemberB"
_BARE_A = "LatestWindowMemberA"
_BARE_B = "LatestWindowMemberB"
_SPEC_ID_A = "spec_0000000000000000000000000000000a"
_SPEC_ID_B = "spec_0000000000000000000000000000000b"

_SNAPSHOT = RevisionSnapshot(
    graph_revision="graph-rev-2024-01",
    source_revisions=(SourceRevision(source="OECD_DAC_CRS", revision="rev-7"),),
)

_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
    "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
    "unit": FacetBinding(state="bound", ref="USD"),
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _facet(facet_id: str) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2017",
    )


def _row(*, stat_var: str, recipient: str, year: str, value: str, facet_id: str = "facet1") -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,{facet_id}\r\n"
    )


def _request(*, stat_var: str, recipient: str, window: RequestedWindow) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=window,
    )


def _is_data_fetch(request: httpx.Request) -> bool:
    """Facet probes omit `c[facetId]`; data fetches include it."""
    return "c[facetId]" in request.url.params


def _coverage_for(entries: list[tuple[str, str]], *, facet_id: str = "facet1") -> CoverageIndex:
    """Builds coverage rows for each variable-recipient pair under `facet_id`."""
    bare_stat_vars = {bare for bare, _ in entries}
    role_names = {bare: RoleNames(donor="donor", recipient="recipient") for bare in bare_stat_vars}
    facet_metadata = {
        facet_id: FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
    }
    rows = [(bare, facet_id, "country/DONOR", recipient, [2015]) for bare, recipient in entries]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _recording_sdmx(csv_body: str) -> tuple[SdmxClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)), requests


# --- Single-member path (`materialise_spec` -> `_generate_member`) ---------


def test_single_member_latest_window_sends_time_period_latest():
    csv_body = _CSV_HEADER + _row(
        stat_var=_BARE_A, recipient="country/TGO", year="2017", value="1000"
    )
    sdmx, requests = _recording_sdmx(csv_body)
    graph = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    materialise_spec(
        _request(
            stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="latest")
        ),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert len(requests) == 1
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 1
    assert fetch_requests[0].url.params.get("c[TIME_PERIOD]") == "LATEST"


def test_single_member_non_latest_window_omits_time_period():
    csv_body = _CSV_HEADER + _row(
        stat_var=_BARE_A, recipient="country/TGO", year="2017", value="1000"
    )
    sdmx, requests = _recording_sdmx(csv_body)
    graph = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    materialise_spec(
        _request(stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="all")),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert len(requests) == 1
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 1
    assert "c[TIME_PERIOD]" not in fetch_requests[0].url.params


def test_single_member_latest_window_blank_value_at_latest_period_serves_no_row():
    # The reduction selects 2017 before value filtering; the mock returns full history.
    csv_body = (
        _CSV_HEADER
        + _row(stat_var=_BARE_A, recipient="country/TGO", year="2015", value="1000")
        + _row(stat_var=_BARE_A, recipient="country/TGO", year="2016", value="2000")
        + _row(stat_var=_BARE_A, recipient="country/TGO", year="2017", value="")
    )
    sdmx, requests = _recording_sdmx(csv_body)
    graph = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    result = materialise_spec(
        _request(
            stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="latest")
        ),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert requests[0].url.params.get("c[TIME_PERIOD]") == "LATEST"
    assert only_chunk(result).rows == ()
    assert result.coverage.rows == 0
    assert {w["code"] for w in result.warnings} == {"empty_result"}


def test_shared_chunk_both_latest_members_send_time_period_latest_in_one_request():
    combined_body = (
        _CSV_HEADER
        + _row(stat_var=_BARE_A, recipient="country/KEN", year="2017", value="1000")
        + _row(stat_var=_BARE_B, recipient="country/KEN", year="2017", value="2000")
    )
    sdmx, requests = _recording_sdmx(combined_body)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, "country/KEN"): [_facet("facet1")],
            (_BARE_B, "country/KEN"): [_facet("facet1")],
        }
    )

    coverage = _coverage_for([(_BARE_A, "country/KEN"), (_BARE_B, "country/KEN")])

    result = materialise_collection(
        [
            (
                _SPEC_ID_A,
                _request(
                    stat_var=_STAT_VAR_A,
                    recipient="country/KEN",
                    window=RequestedWindow(kind="latest"),
                ),
            ),
            (
                _SPEC_ID_B,
                _request(
                    stat_var=_STAT_VAR_B,
                    recipient="country/KEN",
                    window=RequestedWindow(kind="latest"),
                ),
            ),
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
    )

    # With concurrency 1, both latest members share one chunk; the default 4 would split them.
    assert len(requests) == 1
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 1
    assert fetch_requests[0].url.params.get("c[TIME_PERIOD]") == "LATEST"
    assert len(result.members) == 2


def test_latest_member_and_all_window_member_share_scope_but_land_in_different_chunks():
    # Latest and all-window members require separate fetch chunks.
    def handler(request: httpx.Request) -> httpx.Response:
        variables = str(request.url.params.get("c[variableMeasured]"))
        if variables == _BARE_A:
            body = _CSV_HEADER + _row(
                stat_var=_BARE_A, recipient="country/KEN", year="2017", value="1000"
            )
        elif variables == _BARE_B:
            body = _CSV_HEADER + _row(
                stat_var=_BARE_B, recipient="country/KEN", year="2016", value="2000"
            )
        else:
            raise AssertionError(f"unexpected c[variableMeasured]={variables!r}")
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(recording_handler))
    graph = FakeGraphReader(
        facets={
            (_BARE_A, "country/KEN"): [_facet("facet1")],
            (_BARE_B, "country/KEN"): [_facet("facet1")],
        }
    )
    coverage = _coverage_for([(_BARE_A, "country/KEN"), (_BARE_B, "country/KEN")])

    result = materialise_collection(
        [
            (
                _SPEC_ID_A,
                _request(
                    stat_var=_STAT_VAR_A,
                    recipient="country/KEN",
                    window=RequestedWindow(kind="latest"),
                ),
            ),
            (
                _SPEC_ID_B,
                _request(
                    stat_var=_STAT_VAR_B,
                    recipient="country/KEN",
                    window=RequestedWindow(kind="all"),
                ),
            ),
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # The two window kinds require separate `c[TIME_PERIOD]` filters.
    assert len(requests) == 2
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 2
    by_variables = {str(r.url.params.get("c[variableMeasured]")): r for r in fetch_requests}
    assert by_variables[_BARE_A].url.params.get("c[TIME_PERIOD]") == "LATEST"
    assert "c[TIME_PERIOD]" not in by_variables[_BARE_B].url.params
    assert len(result.members) == 2


def test_shared_chunk_latest_window_blank_value_at_latest_period_serves_no_row_for_that_member():
    # At concurrency 1, both KEN members share a chunk. A's blank 2017 value yields no row.
    # B's 2017 value survives.
    combined_body = (
        _CSV_HEADER
        + _row(stat_var=_BARE_A, recipient="country/KEN", year="2015", value="1000")
        + _row(stat_var=_BARE_A, recipient="country/KEN", year="2017", value="")
        + _row(stat_var=_BARE_B, recipient="country/KEN", year="2017", value="2000")
    )
    sdmx, requests = _recording_sdmx(combined_body)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, "country/KEN"): [_facet("facet1")],
            (_BARE_B, "country/KEN"): [_facet("facet1")],
        }
    )
    coverage = _coverage_for([(_BARE_A, "country/KEN"), (_BARE_B, "country/KEN")])

    result = materialise_collection(
        [
            (
                _SPEC_ID_A,
                _request(
                    stat_var=_STAT_VAR_A,
                    recipient="country/KEN",
                    window=RequestedWindow(kind="latest"),
                ),
            ),
            (
                _SPEC_ID_B,
                _request(
                    stat_var=_STAT_VAR_B,
                    recipient="country/KEN",
                    window=RequestedWindow(kind="latest"),
                ),
            ),
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
    )

    # Both members share one fetch.
    assert len(requests) == 1
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 1
    assert fetch_requests[0].url.params.get("c[TIME_PERIOD]") == "LATEST"

    members_by_spec = dict(result.members)
    assert only_chunk(members_by_spec[_SPEC_ID_A]).rows == ()
    assert members_by_spec[_SPEC_ID_A].coverage.rows == 0
    assert {w["code"] for w in members_by_spec[_SPEC_ID_A].warnings} == {"empty_result"}
    assert members_by_spec[_SPEC_ID_B].coverage.rows == 1


def test_group_member_time_period_matches_each_members_own_window_kind():
    # The group uses separate chunks so each member gets its own window filter.
    def handler(request: httpx.Request) -> httpx.Response:
        variables = str(request.url.params.get("c[variableMeasured]"))
        if variables == _BARE_A:
            body = _CSV_HEADER + _row(
                stat_var=_BARE_A, recipient="country/KEN", year="2017", value="1000"
            )
        elif variables == _BARE_B:
            body = _CSV_HEADER + _row(
                stat_var=_BARE_B, recipient="country/TGO", year="2016", value="2000"
            )
        else:
            raise AssertionError(f"unexpected c[variableMeasured]={variables!r}")
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(recording_handler))
    graph = FakeGraphReader(
        facets={
            (_BARE_A, "country/KEN"): [_facet("facet1")],
            (_BARE_B, "country/TGO"): [_facet("facet1")],
        }
    )
    coverage = _coverage_for([(_BARE_A, "country/KEN"), (_BARE_B, "country/TGO")])

    materialise_group(
        [
            _request(
                stat_var=_STAT_VAR_A, recipient="country/KEN", window=RequestedWindow(kind="latest")
            ),
            _request(
                stat_var=_STAT_VAR_B, recipient="country/TGO", window=RequestedWindow(kind="all")
            ),
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert len(requests) == 2
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 2
    by_variables = {str(r.url.params.get("c[variableMeasured]")): r for r in fetch_requests}
    assert by_variables[_BARE_A].url.params.get("c[TIME_PERIOD]") == "LATEST"
    assert "c[TIME_PERIOD]" not in by_variables[_BARE_B].url.params


def test_preview_time_period_matches_the_requests_own_window_kind():
    csv_body = _CSV_HEADER + _row(
        stat_var=_BARE_A, recipient="country/TGO", year="2017", value="1000"
    )

    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    sdmx_latest, requests_latest = _recording_sdmx(csv_body)
    graph_latest = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    materialise_preview(
        _request(
            stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="latest")
        ),
        sdmx=sdmx_latest,
        graph=graph_latest,
        coverage=coverage,
        limit=100,
    )
    fetch_latest = [r for r in requests_latest if _is_data_fetch(r)]
    assert fetch_latest[0].url.params.get("c[TIME_PERIOD]") == "LATEST"

    sdmx_all, requests_all = _recording_sdmx(csv_body)
    graph_all = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    materialise_preview(
        _request(stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="all")),
        sdmx=sdmx_all,
        graph=graph_all,
        coverage=coverage,
        limit=100,
    )
    fetch_all = [r for r in requests_all if _is_data_fetch(r)]
    assert "c[TIME_PERIOD]" not in fetch_all[0].url.params
