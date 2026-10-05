"""Tests revision drift retries and fresh facet probes for collection and group materialisation."""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_collection,
    materialise_group,
)
from materialise.errors import UpstreamVersionChangedError
from materialise.facets import CollectionFacetProbe, MemberProbe, PlaceAdmission
from materialise.graph import Facet
from materialise.revision import RevisionSnapshot, SourceRevision, read_generation
from materialise.window import RequestedWindow

from .conftest import FakeGraphReader, FakeRevisionReader

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


def _counting_generate():
    calls = {"count": 0}

    def generate() -> int:
        calls["count"] += 1
        return calls["count"]

    return generate, calls


def test_converges_immediately():
    reader = FakeRevisionReader([_A, _A])
    generate, calls = _counting_generate()

    result, snapshot = read_generation(reader, generate)

    assert result == 1
    assert calls["count"] == 1
    assert reader.read_count == 2
    assert snapshot == _A


def test_drifts_once_then_stabilises():
    reader = FakeRevisionReader([_A, _B, _B])
    generate, calls = _counting_generate()

    result, snapshot = read_generation(reader, generate)

    assert result == 2
    assert calls["count"] == 2
    assert reader.read_count == 3
    assert snapshot == _B


def test_drifts_twice_raises_upstream_version_changed():
    reader = FakeRevisionReader([_A, _B, _C])
    generate, calls = _counting_generate()

    with pytest.raises(UpstreamVersionChangedError) as excinfo:
        read_generation(reader, generate)

    assert excinfo.value.code == "upstream_version_changed"
    assert excinfo.value.status == 503
    assert excinfo.value.retryable is True
    assert calls["count"] == 2
    assert reader.read_count == 3


def test_read_more_times_than_snapshots_raises_index_error_not_endpoint_unavailable():
    reader = FakeRevisionReader([_A])
    generate, _ = _counting_generate()

    with pytest.raises(IndexError):
        read_generation(reader, generate)


def test_normalised_sorts_source_revisions_by_source():
    unsorted = RevisionSnapshot(
        graph_revision="graph-rev-A",
        source_revisions=(
            SourceRevision(source="OECD_DAC_CRS", revision="rev-1"),
            SourceRevision(source="IMF_WEO", revision="rev-9"),
        ),
    )
    sorted_snapshot = unsorted.normalised()

    assert sorted_snapshot.source_revisions == (
        SourceRevision(source="IMF_WEO", revision="rev-9"),
        SourceRevision(source="OECD_DAC_CRS", revision="rev-1"),
    )


def test_before_given_skips_the_first_read_on_a_clean_run():
    reader = FakeRevisionReader([_A])
    generate, calls = _counting_generate()

    result, snapshot = read_generation(reader, generate, before=_A)

    assert result == 1
    assert calls["count"] == 1
    assert reader.read_count == 1
    assert snapshot == _A


def test_before_given_skips_the_first_read_on_a_drift_retry():
    reader = FakeRevisionReader([_B, _B])
    generate, calls = _counting_generate()

    result, snapshot = read_generation(reader, generate, before=_A)

    assert result == 2
    assert calls["count"] == 2
    assert reader.read_count == 2
    assert snapshot == _B


# A probe revision seeds the first attempt; retries must fetch fresh candidates.

_STAT_VAR = "dcid:RevisionDriftProbeVar"
_BARE_STAT_VAR = "RevisionDriftProbeVar"
_COUNTRY = "country/KEN"


def _drift_probe(*, revision: RevisionSnapshot | None) -> CollectionFacetProbe:
    """A supplied probe holding `_drift_facet`, admitted at `_COUNTRY`."""
    return CollectionFacetProbe(
        members=(
            MemberProbe(
                candidates=(_drift_facet(),),
                admission=PlaceAdmission(places_by_facet={"facet-1": frozenset({_COUNTRY})}),
            ),
        ),
        revision=revision,
    )


def _drift_facet() -> Facet:
    return Facet(
        facet_id="facet-1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )


def _drift_request() -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"observationAbout": [_COUNTRY]},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _single_variable_sdmx_client() -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        header = (
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
            "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
            "scalingFactor,facetId\r\n"
        )
        row = (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_BARE_STAT_VAR},{_COUNTRY},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,2015,1000,,facet-1\r\n"
        )
        return httpx.Response(
            200,
            text=header + row,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def test_materialise_collection_precomputed_probe_stale_revision_retries_and_reprobes():
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, _COUNTRY): [_drift_facet()]})
    probe = _drift_probe(revision=_A)
    # The probe has revision A; the post-attempt read B triggers a retry that converges on B.
    reader = FakeRevisionReader([_B, _B])
    sdmx = _single_variable_sdmx_client()

    assert graph.calls["observation_facets_batch_many"] == 0

    result = materialise_collection(
        [(f"spec_{0:032d}", _drift_request())],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        precomputed_probe=probe,
    )

    assert graph.calls["observation_facets_batch_many"] == 1
    assert len(result.members) == 1


def test_materialise_group_single_member_precomputed_probe_stale_revision_retries_and_reprobes():
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, _COUNTRY): [_drift_facet()]})
    probe = _drift_probe(revision=_A)
    reader = FakeRevisionReader([_B, _B])
    sdmx = _single_variable_sdmx_client()

    assert graph.calls["variable_facets_batch"] == 0

    result = materialise_group(
        [_drift_request()],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        precomputed_probe=probe,
    )

    assert graph.calls["variable_facets_batch"] == 1
    assert result.coverage.rows == 1


def test_precomputed_probe_with_no_revision_reads_before_the_first_attempt_as_today():
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, _COUNTRY): [_drift_facet()]})
    probe = _drift_probe(revision=None)
    reader = FakeRevisionReader([_A, _A])
    sdmx = _single_variable_sdmx_client()

    result = materialise_group(
        [_drift_request()],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        precomputed_probe=probe,
    )

    # Without a probe revision, `read_generation` performs its initial read.
    assert reader.read_count == 2
    assert graph.calls["variable_facets_batch"] == 0
    assert result.coverage.rows == 1
