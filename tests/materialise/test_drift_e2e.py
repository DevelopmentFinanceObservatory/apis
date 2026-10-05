"""Snapshots A-B-C raise after two generations; A-B-B succeeds on the retry."""

from __future__ import annotations

import time

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_spec
from materialise.errors import UpstreamVersionChangedError
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.revision import RevisionSnapshot, SourceRevision
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader, FakeRevisionReader, only_chunk

_STAT_VAR = "dcid:MalariaGrantDisbursements"
# The graph wire keys on the bare dcid; `FakeGraphReader` fixtures below
# use that bare form to match what `generation.py` sends.
_BARE_STAT_VAR = "MalariaGrantDisbursements"

_CSV_BODY = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)

_A = RevisionSnapshot(
    graph_revision="graph-rev-A",
    source_revisions=(SourceRevision(source="OECD_DAC_CRS", revision="rev-1"),),
)
_B = RevisionSnapshot(
    graph_revision="graph-rev-B",
    source_revisions=(SourceRevision(source="OECD_DAC_CRS", revision="rev-2"),),
)
_C = RevisionSnapshot(
    graph_revision="graph-rev-C",
    source_revisions=(SourceRevision(source="OECD_DAC_CRS", revision="rev-3"),),
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)

_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
    "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
    "unit": FacetBinding(state="bound", ref="USD"),
}


def _request() -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/TGO"]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


def _coverage_for() -> CoverageIndex:
    """Build coverage for the bare StatVar and its bound recipient."""
    role_names = {_BARE_STAT_VAR: RoleNames(donor="donor", recipient="recipient")}
    facet_metadata = {
        "facet1": FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
    }
    rows = [(_BARE_STAT_VAR, "facet1", "country/DONOR", "country/TGO", [2015])]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _sdmx_client(*, requests: list[httpx.Request] | None = None) -> SdmxClient:
    calls = requests if requests is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def test_drifts_twice_raises_upstream_version_changed_after_exactly_two_generations():
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    reader = FakeRevisionReader([_A, _B, _C])
    requests: list[httpx.Request] = []

    with pytest.raises(UpstreamVersionChangedError) as excinfo:
        materialise_spec(
            _request(),
            sdmx=_sdmx_client(requests=requests),
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=_coverage_for(),
        )

    assert excinfo.value.code == "upstream_version_changed"
    assert excinfo.value.status == 503
    assert excinfo.value.retryable is True
    assert reader.read_count == 3
    # Exactly two full generation attempts: one data fetch per attempt.
    # Candidates come from `coverage`, an in-memory lookup, so no SDMX
    # probe call precedes either fetch.
    assert len(requests) == 2


def test_drifts_once_then_stabilises_succeeds_via_the_retry():
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    reader = FakeRevisionReader([_A, _B, _B])
    sdmx = _sdmx_client()
    deadlines: list[float | None] = []
    original_stream_data_batches = sdmx.stream_data_batches

    def recording_stream_data_batches(**kwargs):
        deadlines.append(kwargs.get("deadline"))
        return original_stream_data_batches(**kwargs)

    sdmx.stream_data_batches = recording_stream_data_batches  # type: ignore[method-assign]
    original_data = sdmx.data
    probe_calls: list[float | None] = []

    def recording_data(**kwargs):
        probe_calls.append(kwargs.get("deadline"))
        return original_data(**kwargs)

    sdmx.data = recording_data  # type: ignore[method-assign]
    deadline = time.monotonic() + 30

    result = materialise_spec(
        _request(),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        deadline=deadline,
        coverage=_coverage_for(),
    )

    assert reader.read_count == 3
    # Exactly two full generation attempts: one data fetch per attempt,
    # each carrying the same deadline. Candidates come from `coverage`, an
    # in-memory lookup, so `sdmx.data` (the removed LATEST-scoped probe) is
    # never called.
    assert probe_calls == []
    assert deadlines == [deadline, deadline]
    assert result.trace.graph_revision == "graph-rev-B"
    assert len(only_chunk(result).rows) == 1
