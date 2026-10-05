"""Pins the collection writer's position-wise pairing of member signals and conversion provenance.
Two members use reverse-sorted table ids so a reordered fold would assign provenance to the wrong member."""

from __future__ import annotations

from datetime import UTC, datetime

from code_index.shared import get_index
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest import assemble_collection_manifest
from datasets_api.mint_models import MintOptions
from datasets_api.serve import ServeDeps, _write_collection_artifacts
from datasets_api.versions import MemberSignal, NullHeadWriter, SubjectHead
from materialise import CollectionMaterialisationResult, EnrichmentLookup, MaterialisationResult
from materialise.content_version import ConversionPreimage, MintPreimage
from materialise.convert import ConversionRecord
from materialise.frame_build import build_frame
from materialise.profile import compute_output_profile
from materialise.rate_tables import RateTableCache
from materialise.revision import RevisionSnapshot
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import Coverage
from spec_registry.reader import StoredCollection, StoredDefinition

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader

# Reverse-sorted table ids expose any reordering before the positional member pairing.
_PREPARED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


_TABLE_B = "provenance/TableZ"
_TABLE_A = "provenance/TableM"


# A row digest, distinct from every refresh date, so a signal that stored the digest fails.
_TABLE_DIGEST = "d" * 64


def _record(*, table_provenance_id: str, refresh_date: str) -> ConversionRecord:
    return ConversionRecord(
        recipe_id="recipe/Currency",
        exchange_rate_table_version=_TABLE_DIGEST,
        deflator_table_version=None,
        native_currency="LCU",
        native_unit_ref="LCU",
        table_provenance_id=table_provenance_id,
        table_refresh_date=refresh_date,
        fallback_count=0,
        fallback_examples=(),
        projected_count=0,
        projected_examples=(),
    )


def _statistics_file(frame):
    sink = SeriesStatisticsSink()
    sink.write(compute_series_statistics_table(frame))
    return sink.finish()


def _member(
    *,
    spec_id: str,
    stat_var_ref: str,
    content_version: str,
    conversion_record: ConversionRecord,
) -> tuple[StoredDefinition, MaterialisationResult]:
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var_ref,
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2015",
                "value": 1000.0,
            }
        ],
    )
    stored = StoredDefinition(
        spec_id=spec_id,
        schema_version=1,
        identity={"kind": "spec_id", "spec_id": spec_id},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )
    graph = FakeMintGraphReader(labels={"OECD_DAC2A": "OECD DAC2a"})
    result = MaterialisationResult(
        frame_chunks=lambda: iter((frame,)),
        content_version=content_version,
        trace=ManifestTrace(
            applied_window=None,
            graph_revision="rev1",
            source_revisions=(),
            resolved_stat_vars=(),
            facets=(
                {
                    "axis": "provenance",
                    "ref": "OECD_DAC2A",
                    "basis": "bound",
                    "default_policy": None,
                    "default_policy_version": None,
                    "selection_reason": None,
                    "dispatch_facet_id": "facet1",
                },
            ),
            normalisations=(),
            conversion=None,
            frame_grammar_version="1.0.0",
            materialiser_build="test-build",
            capability_version="dset_test",
        ),
        coverage=Coverage(rows=len(frame.rows), dimensions=(), window=None),
        warnings=(),
        series_statistics=_statistics_file(frame),
        output_profile=compute_output_profile(
            frame, role_columns=("recipient",), enrichment=EnrichmentLookup(graph)
        ),
        conversion_record=conversion_record,
        member_conversion_records={stat_var_ref: conversion_record},
        mint_preimage=MintPreimage(
            graph_revision="rev1",
            source_revisions=(),
            facets=({"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"},),
            conversion=ConversionPreimage.from_conversion_record(conversion_record),
        ),
    )
    return stored, result


class _RecordingHeadWriter(NullHeadWriter):
    """Record the signal passed to `advance_head` without a store round trip."""

    def __init__(self) -> None:
        self.written_signal: tuple[MemberSignal, ...] | None = None

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal,
        expected_content_version: str | None,
    ) -> SubjectHead:
        self.written_signal = tuple(signal)
        return super().advance_head(
            subject_id=subject_id,
            content_version=content_version,
            signal=signal,
            expected_content_version=expected_content_version,
        )


def _deps(*, head_store: _RecordingHeadWriter) -> ServeDeps:
    from datasets_api.revision import SentinelRevisionReader
    from dc_sdmx import SdmxClient

    return ServeDeps(
        sdmx=SdmxClient(base_url="https://example.invalid"),
        graph=FakeMintGraphReader(),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=2,
        enrichment=EnrichmentLookup(FakeMintGraphReader()),
        code_index=get_index(),
        rate_cache=RateTableCache(),
        head_store=head_store,
    )


def test_each_members_own_rate_provenance_lands_on_its_own_signal_entry_not_swapped():
    record_b = _record(table_provenance_id=_TABLE_B, refresh_date="2026-02-01")
    record_a = _record(table_provenance_id=_TABLE_A, refresh_date="2026-01-01")

    member_b = _member(
        spec_id="spec_b",
        stat_var_ref="dcid:GrantB",
        content_version="cv_b",
        conversion_record=record_b,
    )
    member_a = _member(
        spec_id="spec_a",
        stat_var_ref="dcid:GrantA",
        content_version="cv_a",
        conversion_record=record_a,
    )
    per_member = [member_b, member_a]

    collection_result = CollectionMaterialisationResult(
        members=(("spec_b", member_b[1]), ("spec_a", member_a[1])),
        content_version="collection_cv",
        revision_snapshot=RevisionSnapshot(graph_revision="rev1", source_revisions=()),
    )
    stored = StoredCollection(collection_id="collection_x", member_spec_ids=["spec_b", "spec_a"])

    from datasets_api.serve import CollectionPrewrite

    prewrite = CollectionPrewrite(
        stored=stored,
        per_member=per_member,
        per_member_entity_columns=[("recipient",), ("recipient",)],
        collection_result=collection_result,
        # `_write_collection_artifacts` records this manifest before it renders.
        manifest=assemble_collection_manifest(
            stored=stored,
            per_member=per_member,
            collection_result=collection_result,
            artifacts=[],
            provenance_refs=["OECD_DAC2A", "OECD_DAC2A"],
            source_citations={},
            prepared_at=_PREPARED_AT,
            member_prepared_at=(_PREPARED_AT, _PREPARED_AT),
            capability_version="dset_test",
            materialiser_build="test-build",
            code_index=get_index(),
            settings=DatasetsBuildSettings(),
        ),
        source_citations={},
        provenance_refs=["OECD_DAC2A", "OECD_DAC2A"],
        spec_dict={"collection_id": "collection_x", "members": ["spec_b", "spec_a"]},
        extra_warnings=(),
        member_origin_warnings=((), ()),
        member_prepared_at=(_PREPARED_AT, _PREPARED_AT),
        member_on_record=(False, False),
    )

    # `derive_signals` leaves rate provenance empty. The writer fills it by member position.
    signal = (
        MemberSignal("facet1", None, "2020", "2020"),
        MemberSignal("facet1", None, "2020", "2020"),
    )
    head_store = _RecordingHeadWriter()
    deps = _deps(head_store=head_store)

    _write_collection_artifacts(
        prewrite=prewrite,
        options=MintOptions(bundle=False),
        deps=deps,
        trace_id=None,
        signal=signal,
        expected_content_version=None,
    )

    assert head_store.written_signal is not None
    written_b, written_a = head_store.written_signal
    assert written_b.rate_provenance_refresh == ((_TABLE_B, "2026-02-01"),)
    assert written_a.rate_provenance_refresh == ((_TABLE_A, "2026-01-01"),)
