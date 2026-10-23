"""Validates fully populated runtime manifests against the published schema.
This catches runtime fields such as `series_count` missing from `/schema`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from datasets_api import manifest_models
from datasets_api import schema as published_schema

_PREPARED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _fully_populated_coverage() -> manifest_models.CoverageModel:
    return manifest_models.CoverageModel(
        rows=2,
        dimensions=[manifest_models.CoverageDimension(label="recipient", count=1)],
        window={"kind": "range", "start_year": 2015, "end_year": 2016, "origin": "query"},
    )


def _fully_populated_citation() -> manifest_models.CitationModel:
    return manifest_models.CitationModel(
        text="OECD DAC2a",
        sources=[
            manifest_models.CitationSource(
                ref="OECD_DAC2A",
                label="OECD DAC2a",
                processing_level="minor",
                license="Creative Commons Attribution License",
                license_type="CreativeCommonsAttributionLicense",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                url="https://data-explorer.oecd.org/",
                documentation=manifest_models.CitedDocumentation(
                    publisher="OECD",
                    description="Official development assistance by recipient.",
                    source_data_url="https://example.org/dac2a.csv",
                    attribution="OECD (2026), DAC2a",
                    update_cadence="Annual",
                    last_update="2026-10-02",
                    years_covered="1960-2024",
                    granularity="Annual, by provider and recipient",
                    currencies=["USD"],
                    transformation_log=[
                        manifest_models.TransformationStepModel(
                            transformation="Download", description="The file is downloaded."
                        )
                    ],
                    version_history=[
                        manifest_models.ReleaseModel(
                            date="2026-10-02",
                            title="First release",
                            changes=[manifest_models.ChangeModel(kind="added", note="DAC2a.")],
                        )
                    ],
                ),
            )
        ],
    )


def _fully_populated_artifacts() -> list[manifest_models.ArtifactEntryModel]:
    return [
        manifest_models.ArtifactEntryModel(
            artifact_digest="sha256:aaa",
            rendering={"format": "csv", "bundle": False},
            bytes=10,
            media_type="text/csv",
            created_at="2026-01-01T00:00:00+00:00",
            tier="working",
            url="/api/datasets/spec_a/cv1/files/sha256:aaa",
        )
    ]


def _fully_populated_warnings() -> list[manifest_models.WarningModel]:
    return [manifest_models.WarningModel(code="revision_unavailable", severity="info", message="m")]


def _fully_populated_output_options() -> manifest_models.OutputOptionsModel:
    return manifest_models.OutputOptionsModel(
        capability_version="cap-v1",
        columns=[
            manifest_models.ColumnOptionModel(
                name="recipient", kind="role", droppable=False, distinct=54
            )
        ],
        formats=[manifest_models.FormatOptionModel(name="csv", state="available")],
        bundle=manifest_models.BundleOptionModel(state="available"),
        enrichments=[
            manifest_models.EnrichmentOptionModel(
                column="recipient",
                field="iso3",
                token="recipient_iso3",
                state="partial",
                reason="enrichment_unavailable",
                coverage=manifest_models.EnrichmentCoverageCountsModel(entities=54, covered=51),
                missing=[manifest_models.MissingEntityModel(ref="dcid:africa", label="Africa")],
                truncated=False,
            )
        ],
        long_layout_defaults=["recipient_label", "recipient_party_kind"],
        long_layout_totals=["recipient"],
        pivots=[
            manifest_models.PivotOptionModel(
                index="recipient",
                columns="time_period",
                state="unavailable",
                reason="pivot_not_unique",
                varying=[manifest_models.VaryingColumnModel(column="provider", distinct=30)],
            )
        ],
        groupby=manifest_models.GroupbyOptionsModel(
            state="unavailable", reason="capability_unavailable", axes=[]
        ),
    )


def test_published_manifest_admits_every_field_a_fully_populated_definition_manifest_emits():
    runtime_manifest = manifest_models.Manifest(
        definition=manifest_models.DefinitionModel(
            schema_version=1, spec_id="spec_a", identity={"identity_version": "spec1"}
        ),
        content_version="cv_aaaaaaaaaaaaaaaa",
        prepared_at=_PREPARED_AT,
        coverage=_fully_populated_coverage(),
        citation=_fully_populated_citation(),
        trace={"graph_revision": "rev1", "resolved_stat_vars": [], "facets": []},
        warnings=_fully_populated_warnings(),
        artifacts=_fully_populated_artifacts(),
        series_count=1,
        output_options=_fully_populated_output_options(),
        series_coverage_stamp="ff2ac3045785c9df092841c4ef3f166b",
        retention=manifest_models.RetentionModel(pinned=False, held_until=_PREPARED_AT),
    )

    published_schema.Manifest.model_validate(runtime_manifest.model_dump(mode="json"))
