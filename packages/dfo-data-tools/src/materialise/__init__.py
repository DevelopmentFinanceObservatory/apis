"""Materialises resolved specs into frames, content versions, and wire traces.
Reads observations through `dc_sdmx` and facet metadata through `materialise.graph`.
Depends on `dfo_platform`'s pure core, not `search`, `dfo_apis`, or FastAPI.
"""

from __future__ import annotations

from dc_graph import Facet, FacetDefaultPolicy, GraphReadError
from materialise.codes import MATERIALISE_CODE_CATALOGUE, MaterialisationWarning
from materialise.enrichment import EnrichmentLookup
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationError,
    MaterialisationIssue,
    MaterialisationUsageError,
    OverCeilingError,
    UpstreamResponseError,
    UpstreamVersionChangedError,
)
from materialise.facets import (
    CollectionFacetProbe,
    FacetBinding,
    facet_axis_value,
    probe_collection_candidates,
    probe_collection_facets,
    role_bearing_facets,
)
from materialise.graph import GraphReader
from materialise.pipeline import (
    DEFAULT_MEMBER_CONCURRENCY,
    ConcatenatedFrameChunks,
    MaterialisationRequest,
    MaterialisationResult,
    PreparedPreview,
    PreviewResult,
    decide_member_unit,
    finish_materialise_preview,
    materialise_group,
    materialise_preview,
    materialise_spec,
    prepare_materialise_preview,
)
from materialise.profile import (
    ColumnProfile,
    EnrichmentCoverage,
    OutputProfile,
    compute_output_profile,
)
from materialise.progress import MaterialisationCancelled, RowProgress
from materialise.reshape import ColumnUnknownError, ParsedColumns, parse_column_tokens
from materialise.revision import RevisionReader, RevisionSnapshot, SourceRevision
from materialise.trace import ManifestTrace, ResolvedStatVar
from materialise.window import Coverage, RequestedWindow

__all__ = [
    "DEFAULT_MEMBER_CONCURRENCY",
    "MATERIALISE_CODE_CATALOGUE",
    "CollectionFacetProbe",
    "ColumnProfile",
    "ColumnUnknownError",
    "ConcatenatedFrameChunks",
    "Coverage",
    "DefaultFacetUndefinedError",
    "EnrichmentCoverage",
    "EnrichmentLookup",
    "Facet",
    "FacetBinding",
    "FacetDefaultPolicy",
    "FacetNotAvailableError",
    "GraphReadError",
    "GraphReader",
    "ManifestTrace",
    "MaterialisationCancelled",
    "MaterialisationError",
    "MaterialisationIssue",
    "MaterialisationRequest",
    "MaterialisationResult",
    "MaterialisationUsageError",
    "MaterialisationWarning",
    "OutputProfile",
    "OverCeilingError",
    "ParsedColumns",
    "PreparedPreview",
    "PreviewResult",
    "RequestedWindow",
    "ResolvedStatVar",
    "RevisionReader",
    "RevisionSnapshot",
    "RowProgress",
    "SourceRevision",
    "UpstreamResponseError",
    "UpstreamVersionChangedError",
    "compute_output_profile",
    "decide_member_unit",
    "facet_axis_value",
    "finish_materialise_preview",
    "materialise_group",
    "materialise_preview",
    "materialise_spec",
    "parse_column_tokens",
    "prepare_materialise_preview",
    "probe_collection_candidates",
    "probe_collection_facets",
    "role_bearing_facets",
]
