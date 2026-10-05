"""The sentinel value reaches `trace.graph_revision`; an injected drifting reader raises
`upstream_version_changed`.
"""

from __future__ import annotations

import httpx
import pytest

from datasets_api.revision import UNAVAILABLE_REVISION, SentinelRevisionReader
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_spec
from materialise.errors import UpstreamVersionChangedError
from materialise.revision import RevisionSnapshot, SourceRevision
from materialise.window import RequestedWindow
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._graph_doubles import FakeMintGraphReader, FakeRevisionReader

_STAT_VAR = "dcid:MalariaGrantDisbursements"
# The graph facet probe and fixtures use bare dcids.
_BARE_STAT_VAR = "MalariaGrantDisbursements"
_BASE_URL = "https://datacommons.one.org/core/api"
_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _sdmx_client(csv_body: str) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))


def _coverage_index():
    """A one-facet `CoverageIndex` for `_BARE_STAT_VAR`, shaped to
    reproduce each test's own `facet`: same facet_id/provenance/
    measurement_method/unit, and `n_obs`/`last_year` from the one 2015
    row each test's own `csv_row` carries.
    """
    return build_coverage_index(
        [(_BARE_STAT_VAR, "facet1", "country/USA", "country/TGO", [2015])],
        role_names={_BARE_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )


def test_sentinel_string_reaches_trace_graph_revision():
    from materialise.facets import FacetBinding
    from materialise.graph import Facet

    facet = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    csv_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    graph = FakeMintGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [facet]})
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/TGO"]},
        declared_roles=("recipient",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="bound", ref="USD"),
        },
        window=RequestedWindow(kind="all"),
    )

    result = materialise_spec(
        request,
        sdmx=_sdmx_client(_CSV_HEADER + csv_row),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=SentinelRevisionReader(),
        materialiser_build="build-1",
        capability_version="cap-1",
        coverage=_coverage_index(),
    )

    assert result.trace.graph_revision == UNAVAILABLE_REVISION
    assert result.trace.source_revisions == ()


def test_upstream_version_changed_is_reachable_via_an_injected_drifting_fake_not_the_sentinel():
    from materialise.facets import FacetBinding
    from materialise.graph import Facet

    facet = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    csv_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    graph = FakeMintGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [facet]})
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/TGO"]},
        declared_roles=("recipient",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="bound", ref="USD"),
        },
        window=RequestedWindow(kind="all"),
    )
    # Three distinct snapshots: before != after (first attempt), then the
    # retry's own before/after (after != after2) also disagree.
    drifting = FakeRevisionReader(
        [
            RevisionSnapshot(
                graph_revision="rev-1", source_revisions=(SourceRevision(source="s", revision="1"),)
            ),
            RevisionSnapshot(
                graph_revision="rev-2", source_revisions=(SourceRevision(source="s", revision="2"),)
            ),
            RevisionSnapshot(
                graph_revision="rev-3", source_revisions=(SourceRevision(source="s", revision="3"),)
            ),
        ]
    )

    with pytest.raises(UpstreamVersionChangedError):
        materialise_spec(
            request,
            sdmx=_sdmx_client(_CSV_HEADER + csv_row),
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=drifting,
            materialiser_build="build-1",
            capability_version="cap-1",
            coverage=_coverage_index(),
        )
