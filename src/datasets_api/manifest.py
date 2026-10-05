"""Assembles manifests and records their immutable version bodies.

`assemble_manifest` and `assemble_collection_manifest` combine registry records, materialisation
results, artifacts, citations, warnings, and output options. Options sit at the definition level or
per collection member. Matching `record_*_version` functions write version records.
`encode_*` and `decode_*` define their stored bodies.

Version bodies store `OutputProfile`, not artifacts or deployment-specific `output_options`.
Reads derive options from the stored profile and current capabilities. Older bodies without a
profile return `output_options=None`. A converted definition's body also stores one conversion
record per series under `series_conversions` (`materialise.series_conversion`), off the wire and
outside the content-version preimage, and `decode_series_conversions` reads them. A definition body
also stores whether the mint's default place scope was plural under `default_scope_plural`, read
by `decode_default_scope_plural`, which decides the stored version's preview mode. Definition
records may include a frame1 artifact that the response omits. Snapshot manifests list all
servable renderings.

Each serve call reads the name, url, licence, source, Data Commons and documentation of every
dispatched provenance ref once (`materialise.provenance_doc.read_source_citations`), and every
manifest's citation is built from it. Every manifest carries the `prepared_at` its caller passes,
the instant of the version's first mint, which a re-mint reads back from the stored record. The
graph has no processing-level convention, so `processing_level` is `None` and each non-empty citation adds `processing_level_undeclared`. Every new manifest's trace records the
request's origin catalogue build as `origin_catalogue_build_id`, and a stale catalogue adds
`origin_catalogue_stale` (`datasets_api.origin`). A materialisation with the
`UNAVAILABLE_REVISION` sentinel adds `revision_unavailable`. `facet_default_selected` marks an
unbound axis when `deployment_fallback` selects `most_observations_over_disagreement`, ranked by
the bound places or role tuples a facet covers, the window years it covers, observation count,
recency, and remaining tie-breaks (`docs/datasets-facet-policy.md`).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import cast

from code_index.query import CodeIndex
from datasets_api.capabilities import build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest_models import (
    ArtifactEntryModel,
    ChangeModel,
    CitationModel,
    CitationSource,
    CitedDataCommons,
    CitedDocumentation,
    CitedSourceNode,
    CollectionMemberModel,
    CollectionModel,
    CoverageDimension,
    CoverageModel,
    DefinitionModel,
    Manifest,
    OutputOptionsModel,
    ReleaseModel,
    TransformationStepModel,
    WarningModel,
)
from datasets_api.origin import catalogue_build_id, stale_warnings
from datasets_api.output_options import derive_output_options
from datasets_api.revision import UNAVAILABLE_REVISION
from datasets_api.timings import phase
from datasets_api.versions import (
    NULL_VERSION_RECORD_WRITER,
    ArtifactRecord,
    VersionRecord,
    VersionRecordWriter,
)
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dc_graph import GraphReadError
from materialise import CollectionMaterialisationResult, MaterialisationResult
from materialise.bundle import BUNDLE_CODE_CATALOGUE
from materialise.cadence import cadence_label
from materialise.finish import collection_trace
from materialise.formats import Frame
from materialise.graph import GraphReader
from materialise.profile import OutputProfile
from materialise.provenance_doc import (
    CoverageFacts,
    ProvenanceDoc,
    SourceCitation,
    coverage_facts_from_index,
    read_source_citations,
)
from materialise.series_conversion import (
    SeriesConversion,
    SeriesKey,
    decode_series_conversion_entries,
    encode_series_conversions,
)
from materialise.window import Coverage
from origin_catalogue import current_catalogue
from series_coverage.index import CoverageIndex
from spec_registry.reader import StoredCollection, StoredDefinition

logger = logging.getLogger(__name__)

_PROCESSING_LEVEL_UNDECLARED_TITLE, _PROCESSING_LEVEL_UNDECLARED_DETAIL = BUNDLE_CODE_CATALOGUE[
    "processing_level_undeclared"
]
_REVISION_UNAVAILABLE_TITLE, _REVISION_UNAVAILABLE_DETAIL, _REVISION_UNAVAILABLE_SEVERITY = (
    DATASETS_WARNING_CATALOGUE["revision_unavailable"]
)
_FACET_DEFAULT_SELECTED_TITLE, _FACET_DEFAULT_SELECTED_DETAIL, _FACET_DEFAULT_SELECTED_SEVERITY = (
    DATASETS_WARNING_CATALOGUE["facet_default_selected"]
)

# `select_facet` emits this reason when fallback ranks disagreeing candidates.
_MOST_OBSERVATIONS_OVER_DISAGREEMENT = "most_observations_over_disagreement"


def _revision_unavailable_warning_if_sentinel(graph_revision: str) -> tuple[WarningModel, ...]:
    if graph_revision != UNAVAILABLE_REVISION:
        return ()
    return (
        WarningModel(
            code="revision_unavailable",
            severity=_REVISION_UNAVAILABLE_SEVERITY,
            message=_REVISION_UNAVAILABLE_DETAIL,
        ),
    )


def _facet_default_selected_warnings(
    facets: Sequence[Mapping[str, object]],
) -> tuple[WarningModel, ...]:
    """Warns for each facet selected from disagreeing candidates by deployment fallback."""
    warnings: list[WarningModel] = []
    for facet in facets:
        if facet.get("basis") != "deployment_fallback":
            continue
        if facet.get("selection_reason") != _MOST_OBSERVATIONS_OVER_DISAGREEMENT:
            continue
        axis = cast(str, facet["axis"])
        warnings.append(
            WarningModel(
                code="facet_default_selected",
                severity=_FACET_DEFAULT_SELECTED_SEVERITY,
                message=(
                    f"facet axis {axis!r} had more than one candidate value; this "
                    f"deployment's default policy selected {facet.get('ref')!r} by its "
                    "ranking (the bound places or role tuples it covers, then the "
                    "requested years it covers, then observation count, recency and the "
                    "remaining tie-breaks)"
                ),
            )
        )
    return tuple(warnings)


def warning_model(warning: Mapping[str, object]) -> WarningModel:
    """Converts a complete `{code, severity, message}` warning mapping without reformatting it."""
    return WarningModel(
        code=cast(str, warning["code"]),
        severity=cast(str, warning["severity"]),
        message=cast(str, warning["message"]),
    )


def reshape_warning_models(warnings: Sequence[Mapping[str, object]]) -> tuple[WarningModel, ...]:
    """Formats reshape warnings from this surface's catalogue.

    `enrichment_unavailable` adds its column and empty-cell count to the message.
    `stat_vars_summed` names the contributing member count and StatVar ref in the message and
    carries them, with the family label, as the warning's structured fields.
    """
    models: list[WarningModel] = []
    for warning in warnings:
        code = cast(str, warning["code"])
        _title, detail, severity = DATASETS_WARNING_CATALOGUE[code]
        message = detail
        if code == "enrichment_unavailable":
            column = cast(str, warning["column"])
            count = cast(int, warning["count"])
            plural = "" if count == 1 else "s"
            message = f"{detail} Column {column!r} left {count} cell{plural} empty."
        elif code == "stat_vars_summed":
            member_count = cast(int, warning["contributing_member_count"])
            stat_var_ref = cast(str, warning["stat_var_ref"])
            models.append(
                WarningModel(
                    code=code,
                    severity=severity,
                    message=f"{detail} {member_count} StatVars were summed into {stat_var_ref}.",
                    contributing_member_count=member_count,
                    stat_var_ref=stat_var_ref,
                    stat_var_label=cast("str | None", warning["stat_var_label"]),
                )
            )
            continue
        models.append(WarningModel(code=code, severity=severity, message=message))
    return tuple(models)


def coverage_model(coverage: Coverage) -> CoverageModel:
    """Converts materialisation or preview coverage to the wire model."""
    return CoverageModel(
        rows=coverage.rows,
        dimensions=[
            CoverageDimension(label=cast(str, dim["label"]), count=cast(int, dim["count"]))
            for dim in coverage.dimensions
        ],
        window=coverage.window,
    )


def cited_documentation(doc: ProvenanceDoc) -> CitedDocumentation:
    """The wire snapshot of a source's documentation. `attribution` is the document's
    `citation` (`licenseAttribution`) and `update_cadence` the display label of its
    `sourceReleaseFrequency` token (`materialise.cadence.cadence_label`)."""
    return CitedDocumentation(
        publisher=doc.publisher,
        description=doc.description,
        source_data_url=doc.source_data_url,
        attribution=doc.citation,
        update_cadence=cadence_label(doc.update_cadence),
        last_update=doc.last_update,
        years_covered=doc.years_covered,
        granularity=doc.granularity,
        currencies=list(doc.currencies),
        transformation_log=[
            TransformationStepModel(transformation=step.title, description=step.description)
            for step in doc.transformation_log
        ],
        version_history=[
            ReleaseModel(
                date=release.date.isoformat(),
                title=release.title,
                changes=[
                    ChangeModel(kind=change.kind.value, note=change.note)
                    for change in release.changes
                ],
            )
            for release in doc.version_history
        ],
    )


def cited_source(
    ref: str, *, processing_level: str | None, citation: SourceCitation | None
) -> CitationSource:
    """The `CitationSource` a mint stores for `ref` at `processing_level` from its live citation
    (`read_source_citations` output), or from `ref` alone when there is none."""
    if citation is None:
        return CitationSource(ref=ref, label=ref, processing_level=processing_level)
    return CitationSource(
        ref=ref,
        label=citation.name or ref,
        processing_level=processing_level,
        license=citation.licence,
        license_type=citation.licence_type,
        license_url=citation.licence_url,
        url=citation.url,
        source=(
            CitedSourceNode(ref=citation.source.ref, label=citation.source.label)
            if citation.source is not None
            else None
        ),
        data_commons=(
            CitedDataCommons(id=citation.data_commons.id, label=citation.data_commons.label)
            if citation.data_commons is not None
            else None
        ),
        documentation=(
            cited_documentation(citation.documentation)
            if citation.documentation is not None
            else None
        ),
    )


def build_citation(
    *,
    provenance_refs: Sequence[str],
    source_citations: Mapping[str, SourceCitation],
    processing_levels: Sequence[str | None] | None = None,
) -> tuple[CitationModel, tuple[WarningModel, ...]]:
    """Builds one source per distinct `(ref, processing_level)` in first-seen order.

    Levels parallel `provenance_refs` and default to `None`. Non-empty citations add
    `processing_level_undeclared`. `source_citations` is `read_source_citations` output for at
    least `provenance_refs`. `label` falls back to the ref, and the licence fields, `url`,
    `source`, `data_commons` and `documentation` stay `None` where the citation has none.
    """
    levels = processing_levels if processing_levels is not None else [None] * len(provenance_refs)
    sources: list[CitationSource] = []
    seen: set[tuple[str, str | None]] = set()
    for ref, level in zip(provenance_refs, levels, strict=True):
        key = (ref, level)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            cited_source(ref, processing_level=level, citation=source_citations.get(ref))
        )
    warnings = (
        (
            WarningModel(
                code="processing_level_undeclared",
                severity="info",
                message=_PROCESSING_LEVEL_UNDECLARED_DETAIL,
            ),
        )
        if sources
        else ()
    )
    text = "; ".join(source.label for source in sources)
    return CitationModel(text=text, sources=sources), warnings


def coverage_facts_of(
    coverage_index: CoverageIndex | None,
) -> Callable[[tuple[str, ...]], CoverageFacts] | None:
    """The currencies-and-granularity lookup a source's documentation reads from this mint's
    coverage snapshot, `None` without one."""
    if coverage_index is None:
        return None
    return functools.partial(coverage_facts_from_index, coverage_index)


HEALED_FIELDS = tuple(
    name for name in CitationSource.model_fields if name not in {"ref", "label", "processing_level"}
)
"""Every `CitationSource` field a live read can supply. `ref`, `label` and `processing_level`
record what the mint cited and keep their stored values."""


def _needs_healing(source: CitationSource) -> bool:
    return any(getattr(source, name) is None for name in HEALED_FIELDS)


def healed_citation(
    citation: CitationModel,
    *,
    graph: GraphReader,
    deadline: float | None,
    coverage_index: CoverageIndex | None,
) -> tuple[CitationModel, bool]:
    """A stored citation as a bundle or card export cites it, and whether every read answered.

    A source with a `HEALED_FIELDS` member that is `None` (a body stored before that field
    existed, a first mint whose documentation read failed, or a source the catalogue and graph
    declare nothing for) is read live with its documentation (`read_source_citations`), within
    `deadline`, turned into a `CitationSource` by `cited_source` as a first mint does, and only
    its `None` fields are filled from that. Every set field stands, so a source with every field
    set reads nothing. The stored body is left as it is. The second value is `False` when the
    read raised `GraphReadError` or a documentation read failed. A source the graph documents
    nothing for counts as answered. `GraphCapExceededError` propagates.
    """
    refs = list(dict.fromkeys(source.ref for source in citation.sources if _needs_healing(source)))
    if not refs:
        return citation, True
    try:
        with phase("citation"):
            live = read_source_citations(
                graph,
                refs,
                catalogue=current_catalogue(),
                deadline=deadline,
                documented=True,
                coverage_facts_of=coverage_facts_of(coverage_index),
            )
    except GraphReadError:
        logger.warning(
            "Citation read failed for refs=%s; rendering the stored citation", refs, exc_info=True
        )
        return citation, False
    complete = not any(live[ref].documentation_read_failed for ref in refs if ref in live)
    sources: list[CitationSource] = []
    for stored in citation.sources:
        read = live.get(stored.ref) if _needs_healing(stored) else None
        if read is None:
            sources.append(stored)
            continue
        cited = cited_source(stored.ref, processing_level=stored.processing_level, citation=read)
        sources.append(
            stored.model_copy(
                update={
                    name: getattr(cited, name)
                    for name in HEALED_FIELDS
                    if getattr(stored, name) is None
                }
            )
        )
    return citation.model_copy(update={"sources": sources}), complete


def _code_index_stamp_entry(code_index: CodeIndex) -> dict[str, object]:
    """Returns the three code-index stamp fields carried in the trace."""
    return {
        "content_hash": code_index.stamp.content_hash,
        "exported_at": code_index.stamp.exported_at,
        "graph_endpoint": code_index.stamp.graph_endpoint,
    }


def assemble_manifest(
    *,
    stored: StoredDefinition,
    materialisation: MaterialisationResult,
    artifacts: Sequence[ArtifactEntryModel],
    provenance_refs: Sequence[str],
    source_citations: Mapping[str, SourceCitation],
    prepared_at: datetime,
    capability_version: str,
    materialiser_build: str,
    code_index: CodeIndex,
    settings: DatasetsBuildSettings,
    extra_warnings: Sequence[WarningModel] = (),
    coverage_stamp: str | None = None,
) -> Manifest:
    """Builds a definition manifest. `record_definition_version` writes it later.

    Uses the caller's `CodeIndex` for rendering and the trace stamp. `coverage_stamp` is
    admission-time lineage, excluded from `content_version`. `None` leaves it unset.
    `prepared_at` is the version's first-mint instant.
    """
    citation, citation_warnings = build_citation(
        provenance_refs=provenance_refs, source_citations=source_citations
    )
    warnings = (
        [warning_model(w) for w in materialisation.warnings]
        + list(extra_warnings)
        + list(citation_warnings)
        + list(_revision_unavailable_warning_if_sentinel(materialisation.trace.graph_revision))
        + list(_facet_default_selected_warnings(materialisation.trace.facets))
        + list(stale_warnings())
    )
    capabilities = build_capabilities(
        capability_version=capability_version,
        build=materialiser_build,
        code_index_stamp=code_index.stamp,
        declarations=code_index.declarations,
        settings=settings,
    )

    trace = replace(
        materialisation.trace,
        code_index_stamp=_code_index_stamp_entry(code_index),
        origin_catalogue_build_id=catalogue_build_id(),
    ).to_dict()

    return Manifest(
        definition=DefinitionModel(
            schema_version=stored.schema_version, spec_id=stored.spec_id, identity=stored.identity
        ),
        collection=None,
        content_version=materialisation.content_version,
        prepared_at=prepared_at,
        coverage=coverage_model(materialisation.coverage),
        citation=citation,
        trace=trace,
        warnings=warnings,
        artifacts=list(artifacts),
        # The statistics artifact is in `artifacts`. This field carries only its count.
        series_count=materialisation.series_statistics.series_count,
        output_options=derive_output_options(
            materialisation.output_profile, capabilities=capabilities
        ),
        series_coverage_stamp=coverage_stamp,
    )


_SERIES_CONVERSIONS_KEY = "series_conversions"
_DEFAULT_SCOPE_PLURAL_KEY = "default_scope_plural"


def encode_version_body(
    manifest: Manifest,
    *,
    output_profile: OutputProfile | None = None,
    series_conversions: Mapping[SeriesKey, SeriesConversion] | None = None,
    default_scope: Sequence[str] | None = None,
) -> dict[str, object]:
    """Encodes a definition version body with `output_profile` in place of artifacts and options.

    A missing profile omits the key. `series_conversions`, when non-empty, adds one entry per
    converted series (`materialise.series_conversion.encode_series_conversions`), and an
    unconverted version omits the key. `default_scope`, the places of the mint's default place scope
    (`MaterialisationResult.default_scope`), stores whether it holds more than one place, so
    the worldwide total alone and a bound place store `false`; `None` omits the key.
    `base_content_version` stays on the version record, outside the body, so older `Manifest`
    models accept it.
    """
    body = manifest.model_dump(mode="json", exclude={"artifacts", "output_options"})
    if output_profile is not None:
        body["output_profile"] = output_profile.to_json()
    if series_conversions:
        body[_SERIES_CONVERSIONS_KEY] = encode_series_conversions(series_conversions)
    if default_scope is not None:
        body[_DEFAULT_SCOPE_PLURAL_KEY] = len(default_scope) > 1
    return body


def decode_default_scope_plural(body: Mapping[str, object]) -> bool | None:
    """Whether the mint's default place scope held more than one place, or `None` for a body
    that does not say: a collapse head, or a version minted before bodies stored it."""
    plural = body.get(_DEFAULT_SCOPE_PLURAL_KEY)
    return plural if isinstance(plural, bool) else None


def decode_series_conversions(body: Mapping[str, object]) -> dict[SeriesKey, SeriesConversion]:
    """The per-series conversion records a definition version body stores, keyed by bare
    StatVar and role tuple. Empty for an unconverted version and for a converted version minted
    before bodies stored them, which a reader tells apart by `trace.conversion`."""
    entries = body.get(_SERIES_CONVERSIONS_KEY)
    if entries is None:
        return {}
    return decode_series_conversion_entries(cast(list[Mapping[str, object]], entries))


def decode_version_body(
    body: Mapping[str, object],
) -> tuple[dict[str, object], OutputProfile | None]:
    """Decodes manifest fields and the optional `OutputProfile` from a version body.

    Removes `series_conversions` and `default_scope_plural` (`decode_series_conversions` and
    `decode_default_scope_plural` read them) and legacy `base_content_version` and inline
    `statistics` keys before model validation. Legacy statistics are dropped because a read
    cannot create the missing artifact. Re-mint the version to obtain `series_count`.
    """
    fields = dict(body)
    profile_json = fields.pop("output_profile", None)
    profile = (
        OutputProfile.from_json(cast(Mapping[str, object], profile_json))
        if profile_json is not None
        else None
    )
    fields.pop(_SERIES_CONVERSIONS_KEY, None)
    fields.pop(_DEFAULT_SCOPE_PLURAL_KEY, None)
    fields.pop("base_content_version", None)
    fields.pop("statistics", None)
    return fields, profile


def persisted_manifest(
    *,
    body: Mapping[str, object],
    served: Manifest,
    call_warnings: Sequence[WarningModel] = (),
) -> Manifest:
    """The manifest a write returns when its version record already existed.

    Every field comes from `body`, the stored `immutable_body`, so a re-mint after a graph rename
    keeps the stored citation. From `served` it takes only `artifacts`, `output_options`, and
    `series_coverage_stamp`. A `call_warnings` entry is appended when no stored warning has its
    `code` and `message`; on a match the stored entry stands, so a `stat_vars_summed` warning
    keeps the structured fields and label it was recorded with. `call_warnings` holds this call's
    reshape warnings and the origin warnings its reads found (`datasets_api.origin`). Stored
    `origin_catalogue_stale` entries are dropped and replaced by the one for the catalogue bound
    to this call, as in `assemble_manifest`, so the manifest carries exactly this mint's stale
    warning.
    """
    if served.collection is not None:
        fields, _profiles = decode_collection_version_body(body)
        collection = cast(dict[str, object], fields["collection"])
        members = cast(list[dict[str, object]], collection["members"])
        for member, served_member in zip(members, served.collection.members, strict=True):
            member["output_options"] = (
                served_member.output_options.model_dump(mode="json")
                if served_member.output_options is not None
                else None
            )
    else:
        fields, _profile = decode_version_body(body)
        fields["output_options"] = (
            served.output_options.model_dump(mode="json")
            if served.output_options is not None
            else None
        )
    warnings = [
        entry
        for entry in cast(list[dict[str, object]], fields.get("warnings") or [])
        if entry.get("code") != "origin_catalogue_stale"
    ]
    keys = {(entry.get("code"), entry.get("message")) for entry in warnings}
    for warning in (*call_warnings, *stale_warnings()):
        if (warning.code, warning.message) not in keys:
            keys.add((warning.code, warning.message))
            warnings.append(warning.model_dump(mode="json"))
    return Manifest.model_validate(
        {
            **fields,
            "warnings": warnings,
            "artifacts": [artifact.model_dump(mode="json") for artifact in served.artifacts],
            "series_coverage_stamp": served.series_coverage_stamp,
        }
    )


def record_definition_version(
    *,
    manifest: Manifest,
    stored: StoredDefinition,
    materialisation: MaterialisationResult,
    artifacts: Sequence[ArtifactEntryModel],
    record_only_artifacts: Sequence[ArtifactRecord] = (),
    version_writer: VersionRecordWriter = NULL_VERSION_RECORD_WRITER,
    trace_id: str | None = None,
    base_content_version: str | None = None,
) -> bool:
    """Writes a definition version record and returns `False` if the record already existed.

    Groupby collapses store `base_content_version` outside the immutable body.
    """
    with phase("record"):
        return version_writer.record_version(
            VersionRecord(
                subject_id=stored.spec_id,
                subject_kind="spec",
                content_version=materialisation.content_version,
                schema_version=stored.schema_version,
                immutable_body=encode_version_body(
                    manifest,
                    output_profile=materialisation.output_profile,
                    series_conversions=materialisation.series_conversions,
                    default_scope=materialisation.default_scope,
                ),
                artifacts=tuple(
                    ArtifactRecord(
                        artifact_digest=artifact.artifact_digest,
                        rendering=artifact.rendering,
                        media_type=artifact.media_type,
                    )
                    for artifact in artifacts
                )
                + tuple(record_only_artifacts),
                base_content_version=base_content_version,
            ),
            trace_id=trace_id,
        )


def collapsed_head_manifest(
    *,
    manifest: Manifest,
    collapse_frame: Frame,
    artifact: ArtifactEntryModel,
    statistics_artifact: ArtifactEntryModel,
    series_count: int,
    content_version: str,
    output_options: OutputOptionsModel,
    extra_warnings: Sequence[WarningModel] = (),
) -> Manifest:
    """Patches a raw head manifest for a collapsed identity without a fresh materialisation.

    Updates content version, row count, series count, artifacts, output options, and warnings.
    Citation, trace,
    definition and `prepared_at` remain unchanged because collapse preserves provenance and scope
    and renders from the raw version's record.
    """
    return manifest.model_copy(
        update={
            "content_version": content_version,
            "coverage": manifest.coverage.model_copy(update={"rows": len(collapse_frame.rows)}),
            "series_count": series_count,
            "warnings": list(manifest.warnings) + list(extra_warnings),
            "artifacts": [artifact, statistics_artifact],
            "output_options": output_options,
        }
    )


def record_collapsed_head_version(
    *,
    manifest: Manifest,
    subject_id: str,
    schema_version: int,
    base_content_version: str,
    output_profile: OutputProfile,
    record_only_artifacts: Sequence[ArtifactRecord] = (),
    version_writer: VersionRecordWriter = NULL_VERSION_RECORD_WRITER,
    trace_id: str | None = None,
) -> bool:
    """Records a head-hit collapse without a fresh materialisation result and returns `False` if
    the record already existed.

    `manifest` is already patched. `base_content_version` names the raw materialisation, and
    `output_profile` is the collapsed frame's. `ON CONFLICT DO NOTHING` makes repeated writes
    idempotent.
    """
    with phase("record"):
        return version_writer.record_version(
            VersionRecord(
                subject_id=subject_id,
                subject_kind="spec",
                content_version=manifest.content_version,
                schema_version=schema_version,
                immutable_body=encode_version_body(manifest, output_profile=output_profile),
                artifacts=tuple(
                    ArtifactRecord(
                        artifact_digest=artifact.artifact_digest,
                        rendering=artifact.rendering,
                        media_type=artifact.media_type,
                    )
                    for artifact in manifest.artifacts
                )
                + tuple(record_only_artifacts),
                base_content_version=base_content_version,
            ),
            trace_id=trace_id,
        )


def _aggregate_window(windows: Sequence[dict[str, object] | None]) -> dict[str, object] | None:
    present = [w for w in windows if w is not None]
    if not present:
        return None
    starts = [cast(int, w["start_year"]) for w in present if w.get("start_year") is not None]
    ends = [cast(int, w["end_year"]) for w in present if w.get("end_year") is not None]
    return {
        "kind": "range",
        "start_year": min(starts) if starts else None,
        "end_year": max(ends) if ends else None,
        "origin": "query",
    }


def _aggregate_dimensions(
    per_member_coverage: Sequence[Coverage],
) -> list[CoverageDimension]:
    """Returns first-seen labels with counts from the union of each member's entity refs."""
    union: dict[str, set[str | int]] = {}
    order: list[str] = []
    for coverage in per_member_coverage:
        for label, refs in coverage.entity_refs.items():
            if label not in union:
                order.append(label)
                union[label] = set()
            union[label] |= refs
    return [CoverageDimension(label=label, count=len(union[label])) for label in order]


def assemble_collection_manifest(
    *,
    stored: StoredCollection,
    per_member: Sequence[tuple[StoredDefinition, MaterialisationResult]],
    collection_result: CollectionMaterialisationResult,
    artifacts: Sequence[ArtifactEntryModel],
    provenance_refs: Sequence[str],
    source_citations: Mapping[str, SourceCitation],
    prepared_at: datetime,
    member_prepared_at: Sequence[datetime],
    capability_version: str,
    materialiser_build: str,
    code_index: CodeIndex,
    settings: DatasetsBuildSettings,
    extra_warnings: Sequence[WarningModel] = (),
    coverage_stamp: str | None = None,
) -> Manifest:
    """Builds a collection manifest and aggregates member results.

    Rows are summed. Dimension counts union entity refs, windows form an envelope, and
    citations are deduplicated. `materialise.finish.collection_trace` merges the member traces: every
    member's `source_revisions` and `resolved_stat_vars`, and one conversion entry per recipe
    and table-version pair naming its members, with first-member facets and revision. Each member gets
    its own output options. `record_collection_version` writes the result.

    Uses the caller's `CodeIndex` for rendering and the trace stamp. `coverage_stamp` is
    admission-time lineage on the top-level manifest only. `prepared_at` is the collection
    version's first-mint instant, and `member_prepared_at`, parallel to `per_member`, each
    member version's. `provenance_refs`, also parallel to `per_member`, holds each member's
    provenance ref, which its member entry carries as `provenance_refs`.
    """
    citation, citation_warnings = build_citation(
        provenance_refs=provenance_refs, source_citations=source_citations
    )

    total_rows = sum(result.coverage.rows for _stored, result in per_member)
    dimensions = _aggregate_dimensions([result.coverage for _stored, result in per_member])
    window = _aggregate_window([result.coverage.window for _stored, result in per_member])
    coverage = CoverageModel(rows=total_rows, dimensions=dimensions, window=window)

    warnings: list[WarningModel] = list(extra_warnings) + list(citation_warnings)
    for _stored, result in per_member:
        warnings.extend(warning_model(w) for w in result.warnings)
        warnings.extend(_facet_default_selected_warnings(result.trace.facets))
    if per_member:
        warnings.extend(
            _revision_unavailable_warning_if_sentinel(per_member[0][1].trace.graph_revision)
        )
    warnings.extend(stale_warnings())

    capabilities = build_capabilities(
        capability_version=capability_version,
        build=materialiser_build,
        code_index_stamp=code_index.stamp,
        declarations=code_index.declarations,
        settings=settings,
    )
    members = [
        CollectionMemberModel(
            definition=DefinitionModel(
                schema_version=member_stored.schema_version,
                spec_id=member_stored.spec_id,
                identity=member_stored.identity,
            ),
            content_version=result.content_version,
            prepared_at=member_prepared,
            provenance_refs=[member_ref],
            coverage=coverage_model(result.coverage),
            # This member's statistics artifact is in `artifacts`. The field carries its count.
            series_count=result.series_statistics.series_count,
            output_options=derive_output_options(result.output_profile, capabilities=capabilities),
        )
        for (member_stored, result), member_prepared, member_ref in zip(
            per_member, member_prepared_at, provenance_refs, strict=True
        )
    ]
    trace = (
        replace(
            collection_trace([result for _stored, result in per_member]),
            code_index_stamp=_code_index_stamp_entry(code_index),
            origin_catalogue_build_id=catalogue_build_id(),
            # A collection's members are separate definitions, so its trace states no family.
            family=None,
        ).to_dict()
        if per_member
        else {}
    )

    return Manifest(
        definition=None,
        collection=CollectionModel(collection_id=stored.collection_id, members=members),
        content_version=collection_result.content_version,
        prepared_at=prepared_at,
        coverage=coverage,
        citation=citation,
        trace=trace,
        warnings=warnings,
        artifacts=list(artifacts),
        series_coverage_stamp=coverage_stamp,
    )


def encode_collection_version_body(
    manifest: Manifest, *, output_profiles: Sequence[OutputProfile]
) -> dict[str, object]:
    """Encodes a collection version body with `output_profile` on each member in place of
    `output_options`. Profiles follow member order. Artifacts and derived options are omitted.
    """
    assert manifest.collection is not None
    body = manifest.model_dump(
        mode="json",
        exclude={"artifacts": True, "collection": {"members": {"__all__": {"output_options"}}}},
    )
    collection = cast(dict[str, object], body["collection"])
    members = cast(list[dict[str, object]], collection["members"])
    for member, profile in zip(members, output_profiles, strict=True):
        member["output_profile"] = profile.to_json()
    return body


def decode_collection_version_body(
    body: Mapping[str, object],
) -> tuple[dict[str, object], list[OutputProfile | None]]:
    """Decodes manifest fields and positional output profiles for collection members.

    Missing profiles yield `None`. Removes legacy inline `statistics` keys before validation.
    """
    fields = dict(body)
    collection = cast(Mapping[str, object] | None, fields.get("collection"))
    profiles: list[OutputProfile | None] = []
    if collection is not None:
        new_members: list[dict[str, object]] = []
        for member in cast(list[Mapping[str, object]], collection.get("members", [])):
            member = dict(member)
            profile_json = member.pop("output_profile", None)
            profiles.append(
                OutputProfile.from_json(cast(Mapping[str, object], profile_json))
                if profile_json is not None
                else None
            )
            member.pop("statistics", None)
            new_members.append(member)
        fields["collection"] = {**collection, "members": new_members}
    return fields, profiles


def record_collection_version(
    *,
    manifest: Manifest,
    stored: StoredCollection,
    per_member: Sequence[tuple[StoredDefinition, MaterialisationResult]],
    collection_result: CollectionMaterialisationResult,
    artifacts: Sequence[ArtifactEntryModel],
    version_writer: VersionRecordWriter = NULL_VERSION_RECORD_WRITER,
    trace_id: str | None = None,
) -> bool:
    """Writes the collection's version record and returns `False` if it already existed."""
    with phase("record"):
        return version_writer.record_version(
            VersionRecord(
                subject_id=stored.collection_id,
                subject_kind="collection",
                content_version=collection_result.content_version,
                schema_version=1,
                immutable_body=encode_collection_version_body(
                    manifest,
                    output_profiles=[result.output_profile for _stored, result in per_member],
                ),
                member_versions=tuple(
                    (member_stored.spec_id, result.content_version)
                    for member_stored, result in per_member
                ),
                artifacts=tuple(
                    ArtifactRecord(
                        artifact_digest=artifact.artifact_digest,
                        rendering=artifact.rendering,
                        media_type=artifact.media_type,
                    )
                    for artifact in artifacts
                ),
            ),
            trace_id=trace_id,
        )
