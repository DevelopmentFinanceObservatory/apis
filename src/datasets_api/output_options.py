"""Joins a frame's `OutputProfile` with current deployment capabilities for the wire response.

Frame facts are computed at materialisation. Formats, enabled enrichments, and pivot caps are
joined at each read. Disabled enrichments and single-distinct pivot pairs are omitted.
"""

from __future__ import annotations

from datasets_api.capabilities import Capabilities
from datasets_api.manifest_models import (
    BundleOptionModel,
    ColumnOptionModel,
    EnrichmentCoverageCountsModel,
    EnrichmentOptionModel,
    FormatOptionModel,
    GroupbyOptionsModel,
    MissingEntityModel,
    OutputOptionsModel,
    PivotOptionModel,
    UnavailableFormatModel,
    VaryingColumnModel,
)
from materialise.formats import FORMAT_COLUMN_CAPS
from materialise.profile import EnrichmentCoverage, OutputProfile, PivotFeasibility

# `materialise.formats.FORMAT_COLUMN_CAPS` keys its Excel entry "excel".
# The wire uses "xlsx".
_FORMAT_COLUMN_CAP_KEYS: dict[str, str] = {"csv": "csv", "xlsx": "excel", "parquet": "parquet"}

_CAPABILITY_UNAVAILABLE_REASON = "capability_unavailable"
_ENRICHMENT_PARTIAL_REASON = "enrichment_unavailable"
_PIVOT_NOT_UNIQUE_REASON = "pivot_not_unique"
_PIVOT_NOT_HOMOGENEOUS_REASON = "pivot_not_homogeneous"
_PIVOT_OVER_COLUMN_CAP_REASON = "pivot_over_column_cap"
_PIVOT_OVER_CELL_CAP_REASON = "pivot_over_cell_cap"


def derive_output_options(
    profile: OutputProfile, *, capabilities: Capabilities
) -> OutputOptionsModel:
    """Builds `output_options` from the frame profile and capabilities.

    Summability declarations come through `capabilities`, avoiding a second deployment read.
    """
    return OutputOptionsModel(
        capability_version=capabilities.capability_version,
        columns=[
            ColumnOptionModel(
                name=column.name,
                kind=column.kind,
                droppable=column.droppable,
                distinct=column.distinct,
            )
            for column in profile.columns
        ],
        formats=[_format_option(name=fmt.name, live=fmt.live) for fmt in capabilities.formats],
        bundle=_bundle_option(capabilities.bundle_active),
        enrichments=[
            option
            for coverage in profile.enrichments
            if (option := _enrichment_option(coverage, capabilities=capabilities)) is not None
        ],
        pivots=[_pivot_option(analysis, capabilities=capabilities) for analysis in profile.pivots],
        groupby=_groupby_options(capabilities=capabilities),
    )


def _format_option(*, name: str, live: bool) -> FormatOptionModel:
    if live:
        return FormatOptionModel(name=name, state="available")
    return FormatOptionModel(name=name, state="unavailable", reason=_CAPABILITY_UNAVAILABLE_REASON)


def _bundle_option(bundle_active: bool) -> BundleOptionModel:
    if bundle_active:
        return BundleOptionModel(state="available")
    return BundleOptionModel(state="unavailable", reason=_CAPABILITY_UNAVAILABLE_REASON)


def _enrichment_option(
    coverage: EnrichmentCoverage, *, capabilities: Capabilities
) -> EnrichmentOptionModel | None:
    # Disabled fields appear in the capability document, not frame-specific options.
    if coverage.field not in capabilities.enrichment_fields:
        return None
    available = coverage.covered == coverage.entities
    return EnrichmentOptionModel(
        column=coverage.column,
        field=coverage.field,
        token=f"{coverage.column}_{coverage.field}",
        state="available" if available else "partial",
        reason=None if available else _ENRICHMENT_PARTIAL_REASON,
        coverage=EnrichmentCoverageCountsModel(
            entities=coverage.entities, covered=coverage.covered
        ),
        missing=[MissingEntityModel(ref=ref, label=label) for ref, label in coverage.missing],
        truncated=coverage.truncated,
    )


def _pivot_option(analysis: PivotFeasibility, *, capabilities: Capabilities) -> PivotOptionModel:
    """Duplicate keys precede homogeneity checks, which precede a stringification collision.

    A collision, where distinct `columns` values share an output column name, is unavailable as
    `pivot_not_unique` with no `varying`, since no duplicate key group exists to name.
    """
    if analysis.duplicate_keys:
        return PivotOptionModel(
            index=analysis.index,
            columns=analysis.columns,
            state="unavailable",
            reason=_PIVOT_NOT_UNIQUE_REASON,
            varying=[
                VaryingColumnModel(column=column, distinct=distinct)
                for column, distinct in analysis.varying
            ],
        )
    if not analysis.homogeneous:
        return PivotOptionModel(
            index=analysis.index,
            columns=analysis.columns,
            state="unavailable",
            reason=_PIVOT_NOT_HOMOGENEOUS_REASON,
        )

    if not analysis.unique:
        return PivotOptionModel(
            index=analysis.index,
            columns=analysis.columns,
            state="unavailable",
            reason=_PIVOT_NOT_UNIQUE_REASON,
        )

    unavailable_formats = [
        UnavailableFormatModel(format=format_name, reason=reason)
        for format_name in _FORMAT_COLUMN_CAP_KEYS
        if (reason := _pivot_format_refusal(analysis, format_name, capabilities=capabilities))
        is not None
    ]

    return PivotOptionModel(
        index=analysis.index,
        columns=analysis.columns,
        state="partial" if unavailable_formats else "available",
        width=analysis.width,
        cells=analysis.cells,
        unavailable_formats=unavailable_formats or None,
    )


def _pivot_format_refusal(
    analysis: PivotFeasibility, format_name: str, *, capabilities: Capabilities
) -> str | None:
    """Returns the first refusal code, checking the format column cap before the shared cell cap.

    Only the Excel column cap is finite. Returns `None` when neither cap is exceeded.
    """
    column_cap = FORMAT_COLUMN_CAPS[_FORMAT_COLUMN_CAP_KEYS[format_name]]
    if column_cap is not None and analysis.width > column_cap:
        return _PIVOT_OVER_COLUMN_CAP_REASON
    if analysis.cells > capabilities.caps.pivot_max_output_cells:
        return _PIVOT_OVER_CELL_CAP_REASON
    return None


def _groupby_options(*, capabilities: Capabilities) -> GroupbyOptionsModel:
    if capabilities.summability_declarations_live:
        axes = [{"name": axis} for axis in capabilities.summable_axes]
        return GroupbyOptionsModel(state="available", axes=axes)
    return GroupbyOptionsModel(state="unavailable", reason=_CAPABILITY_UNAVAILABLE_REASON, axes=[])
