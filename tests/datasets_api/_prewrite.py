"""Runs `prewrite_definition` over test doubles, with the dependencies a mint door supplies."""

from __future__ import annotations

from collections.abc import Sequence

from code_index.shared import get_index
from datasets_api.config import DatasetsBuildSettings
from datasets_api.serve import DefinitionPrewrite, prewrite_definition
from dc_sdmx import SdmxClient
from materialise import CollectionFacetProbe, EnrichmentLookup, MaterialisationRequest
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.graph import GraphReader
from materialise.rate_tables import RateTableCache
from spec_registry.reader import StoredDefinition


def prewrite(
    requests: Sequence[MaterialisationRequest],
    *,
    graph: GraphReader,
    sdmx: SdmxClient,
    precomputed_probe: CollectionFacetProbe | None = None,
) -> DefinitionPrewrite:
    """`prewrite_definition` for `requests` under a fixture definition, reading `graph` and
    `sdmx`. `precomputed_probe` is the probe a mint door makes before the prewrite."""
    return prewrite_definition(
        stored=StoredDefinition(
            spec_id="spec_fixture",
            schema_version=1,
            identity={},
            identity_bytes=b"{}",
            identity_digest="sha256:fixture",
        ),
        requests=requests,
        sdmx=sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        materialiser_build="build-1",
        capability_version="cap-1",
        enrichment=EnrichmentLookup(graph, parties={}),
        code_index=get_index(),
        rate_cache=RateTableCache(),
        member_concurrency=2,
        settings=DatasetsBuildSettings(),
        precomputed_probe=precomputed_probe,
        documented=False,
    )
