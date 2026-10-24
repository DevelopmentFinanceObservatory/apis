"""Joins a frame's `OutputProfile` with current deployment capabilities for the wire response.

Frame facts are computed at materialisation. Formats, enabled enrichments, and pivot caps are
joined at each read. Disabled enrichments and single-distinct pivot pairs are omitted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence

from code_index.types import PartyRow
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
from dfo_platform.roles import role_key_axis
from materialise.enrichment import PARTY_FIELDS, PARTY_KIND_FIELD, party_kind_of
from materialise.formats import FORMAT_COLUMN_CAPS
from materialise.profile import EnrichmentCoverage, OutputProfile, PivotFeasibility
from materialise.projection import long_layout_default_enrichments, totals_axes, totals_roles
from materialise.reshape import enrichment_qualifies

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
    profile: OutputProfile,
    *,
    identity: Mapping[str, object],
    capabilities: Capabilities,
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> OutputOptionsModel:
    """Builds `output_options` from the frame profile, the definition's `identity` and
    capabilities.

    Summability declarations come through `capabilities`, avoiding a second deployment read.
    `long_layout_totals` names the roles whose total rows a long CSV or XLSX can move out of
    its main table (`movable_totals_roles`, the rule the render splits by), from `identity`, the
    code index's `parties` and the provenances the definition read, `datasets`.
    """
    role_columns = [column.name for column in profile.columns if column.kind == "role"]
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
            for stored in profile.enrichments
            for coverage in _party_field_coverages(stored)
            if (option := _enrichment_option(coverage, capabilities=capabilities)) is not None
        ],
        long_layout_defaults=list(long_layout_default_enrichments(role_columns)),
        long_layout_totals=list(
            movable_totals_roles(identity, role_columns, parties=parties, datasets=datasets)
        ),
        pivots=[_pivot_option(analysis, capabilities=capabilities) for analysis in profile.pivots],
        groupby=_groupby_options(capabilities=capabilities),
    )


def movable_totals_roles(
    identity: Mapping[str, object],
    role_columns: Sequence[str],
    *,
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> tuple[str, ...]:
    """The roles of `materialise.projection.totals_roles` that can hold a total, and so the
    roles whose total rows a long CSV or XLSX moves out of its main table. The options
    document's `long_layout_totals` and the render's split (`render._TotalRows`) both read this
    function, so a render drops rows only on a role the options document names.

    A role bound to several entities qualifies when one of them is a total (`party_kind_of`
    over `parties`). An open role qualifies when `parties` lists a total on its axis that one of
    `datasets` publishes (`PartyRow.axes`, `PartyRow.datasets`), or any total on its axis when
    `datasets` is empty, as for a definition whose read selected no provenance. The index builds
    those rows from the coverage that lists each provenance's parties, so a total an open role
    can hold in a covered provenance is one this rule names."""
    axes = totals_axes(identity)
    read = set(datasets)

    def movable(role: str) -> bool:
        axis = role_key_axis(role)
        bound = axes[axis] if axis is not None else ()
        if bound:
            return any(party_kind_of(entity, parties=parties) == "total" for entity in bound)
        return any(
            party.kind == "total"
            and axis in party.axes
            and (not read or not read.isdisjoint(party.datasets))
            for party in parties.values()
        )

    return tuple(role for role in totals_roles(identity, role_columns) if movable(role))


def _format_option(*, name: str, live: bool) -> FormatOptionModel:
    if live:
        return FormatOptionModel(name=name, state="available")
    return FormatOptionModel(name=name, state="unavailable", reason=_CAPABILITY_UNAVAILABLE_REASON)


def _bundle_option(bundle_active: bool) -> BundleOptionModel:
    if bundle_active:
        return BundleOptionModel(state="available")
    return BundleOptionModel(state="unavailable", reason=_CAPABILITY_UNAVAILABLE_REASON)


def _party_field_coverages(coverage: EnrichmentCoverage) -> tuple[EnrichmentCoverage, ...]:
    """`coverage`, and for a `party_kind` pair the same counts for every party field
    (`PARTY_FIELDS`). One code-index row gives a party all three cells, so a party covered for
    its kind is covered for its class and whole cell too. A profile stores the kind alone, so
    the options of a version stored before the other fields existed list them as well."""
    if coverage.field != PARTY_KIND_FIELD:
        return (coverage,)
    return tuple(dataclasses.replace(coverage, field=field) for field in PARTY_FIELDS)


def _enrichment_option(
    coverage: EnrichmentCoverage, *, capabilities: Capabilities
) -> EnrichmentOptionModel | None:
    # Disabled fields appear in the capability document, not frame-specific options. A stored
    # profile can list a subject column's kind, which the columns option refuses
    # (`enrichment_qualifies`).
    if coverage.field not in capabilities.enrichment_fields or not enrichment_qualifies(
        coverage.column, coverage.field
    ):
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
