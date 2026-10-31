"""Reads of a derived frame: a frame the analysis runner, or a client through the publish door,
stored under an `ops_` id and an `ocv_` content version.

`read_derived_manifest` gives the facts a client builds a Dataset from, in one read:

- `retention`: the frame's own (`retention.version_retention`), `pinned` or `held_until`;
- `recipe`: the stored recipe's hashes, execution identity, `tree` and `roots`;
- `roots`: each root dataset version once, with the rendering its frame was read with,
  `groupby` included;
- `resolved_periods`: each period the recipe states relative to the latest, with the period it
  resolved to (`analysis.resolved_periods`), and `anchor`, the recipe's rolling anchor;
- `left_out`: every series a step of the recipe dropped for a data gap, each naming its
  coordinates, its step (verb and node index), the step input it came from, the reason and its
  rows, as the run that stored the recipe recorded them (`analysis.left_out_of`), with
  `left_out_complete`. Both are null for a recipe stored without them, as one a client
  published, and `recipe` is null, with no roots, for a frame published without a recipe;
- `citation`: the sources of every root, the root versions' own citations merged, each source
  once.

A version with no record refuses `snapshot_not_found`, a revoked one `snapshot_revoked`, and a
frame past retention `artifact_expired`.

`render_derived_file` writes the frame in the format the file selector names
(`doors.files.selector_options`): `csv`, `xlsx` or `parquet`, alone or in a bundle zip beside the
Data Package documents a dataset bundle holds (`materialise.bundle.assemble_bundle`), citing the
sources the frame's `provenance` column names from its roots' citations. A derived frame's files
hold the frame's own grammar columns, the frame layout, so `layout` takes only `frame`. Bare
Parquet is the frame file the publish stored beside the canonical frame
(`render.render_and_store_frame_file`), the file a format 2 ref's `artifact.digest` names, as a
dataset version's file selector serves the file its ref names: a flat file whose `unit` is
text, a ref as its string and a converted or composed unit as its JCS text. A frame whose file
is gone, past retention or published before frames stored one, renders it again from the
canonical frame, which gives the same bytes, and links the stored file under the version, so the
next request reads it in place of rendering it again. Every other file renders from the canonical frame
and is not stored.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from dfo_frame_grammar import Frame

from artifact_store import ArtifactStore, Tier
from artifact_store.errors import ArtifactNotFoundError
from dataset_store.clock import Clock
from dataset_store.snapshot import (
    ManifestFacts,
    SnapshotReader,
    StoredArtifactLocation,
    read_servable_version,
)
from datasets_api.analysis import resolved_periods
from datasets_api.codes_snapshot import ArtifactExpired
from datasets_api.manifest_models import CitationModel, CitationSource, RetentionModel
from datasets_api.mint_models import MintOptions
from datasets_api.publish_models import Recipe
from datasets_api.recipe_store import RecipeReader, RecipeRecord
from datasets_api.render import (
    frame_file_location,
    render_and_store_frame_file,
    stored_frame_location,
)
from datasets_api.retention import version_retention
from datasets_api.store_errors import DatasetsStoreUnavailable, read_stored_frame
from datasets_api.versions import VersionRecordWriter
from dfo_platform.config import PublicSiteSettings
from materialise import EnrichmentLookup
from materialise.bundle import (
    ArtifactEntry,
    BundleContext,
    BundleLinks,
    MemberProvenance,
    assemble_bundle,
)
from materialise.bundle_columns import TableScan, column_docs, summary_line
from materialise.bundle_naming import bare_file_name, sources_slug, zip_file_name
from materialise.bundle_sources import source_provenance
from materialise.formats import (
    Rendering,
    Table,
    column_cells,
    frame_entity_columns,
    table_schema_for,
    write_csv,
    write_excel,
    write_parquet,
)

_RENDERING_KEYS = ("format", "bundle", "layout")
_ZIP_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_PLAIN = Rendering()
_WRITERS = {"csv": write_csv, "xlsx": write_excel, "parquet": write_parquet}
_TAG_LENGTH = 6
_PROVENANCE_COLUMN = "provenance"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DerivedReads:
    """The stores a derived read reads, and `version_writer`, which links a frame file a read
    stored again."""

    snapshot_reader: SnapshotReader
    version_writer: VersionRecordWriter
    artifact_store: ArtifactStore
    recipe_reader: RecipeReader
    enrichment: EnrichmentLookup
    now: Clock


def root_rendering(options: Mapping[str, Any]) -> dict[str, Any]:
    """The ref rendering of a root a recipe read with mint `options`: its format, bundle and
    layout, `without_party_totals` when true, and `groupby` when set."""
    rendering = {key: options[key] for key in _RENDERING_KEYS if key in options}
    if options.get("without_party_totals") is True:
        rendering["without_party_totals"] = True
    if options.get("groupby") is not None:
        rendering["groupby"] = options["groupby"]
    return rendering


def recipe_roots(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each root dataset version a recipe's stored `roots` `records` name, once, in their
    order, with its rendering."""
    roots: list[dict[str, Any]] = []
    for record in records:
        dataset = record["dataset"]
        options = record.get("options") or {}
        root = {
            "spec_id": dataset["spec_id"],
            "content_version": dataset["content_version"],
            "rendering": root_rendering(options),
        }
        if root not in roots:
            roots.append(root)
    return roots


def _root_citation(reader: SnapshotReader, root: Mapping[str, Any]) -> CitationModel | None:
    """The citation of `root`'s version, `None` for a root with no stored manifest record."""
    stored = reader.read_version(root["spec_id"], root["content_version"])
    if stored is None or not isinstance(stored.facts, ManifestFacts):
        return None
    return CitationModel.model_validate(stored.facts.citation)


def _merged_citation(citations: Iterable[CitationModel]) -> CitationModel:
    """`citations` merged: each text once, and each source once by
    `(ref, processing_level)`, in order."""
    texts: list[str] = []
    sources: dict[tuple[str, str | None], CitationSource] = {}
    for citation in citations:
        if citation.text not in texts:
            texts.append(citation.text)
        for source in citation.sources:
            sources.setdefault((source.ref, source.processing_level), source)
    return CitationModel(text="\n\n".join(texts), sources=list(sources.values()))


def roots_citation(reader: SnapshotReader, roots: Sequence[Mapping[str, Any]]) -> CitationModel:
    """The citations of `roots` merged, in root order. A root with no stored record cites
    nothing."""
    citations = (_root_citation(reader, root) for root in roots)
    return _merged_citation(citation for citation in citations if citation is not None)


def stored_recipe_roots(record: RecipeRecord | None) -> list[dict[str, Any]]:
    """The roots of the stored recipe `record` (`recipe_roots`), none for a frame with no
    recipe."""
    if record is None:
        return []
    return recipe_roots(cast("Sequence[Mapping[str, Any]]", record.body.get("roots") or []))


def recipe_citation(reader: SnapshotReader, record: RecipeRecord | None) -> CitationModel:
    """The citation of the derived frame whose stored recipe is `record`: its roots' citations
    merged (`roots_citation`), and no source for a frame with no recipe."""
    return roots_citation(reader, stored_recipe_roots(record))


def full_recipe_citation(
    reader: SnapshotReader, record: RecipeRecord | None
) -> CitationModel | None:
    """`recipe_citation` when the recipe has roots and each of them can be cited, else `None`,
    for a caller that must name every source of the frame."""
    roots = stored_recipe_roots(record)
    citations = [_root_citation(reader, root) for root in roots]
    cited = [citation for citation in citations if citation is not None]
    if not roots or len(cited) != len(roots):
        return None
    return _merged_citation(cited)


@dataclass(frozen=True, kw_only=True)
class DerivedManifestFacts:
    """What `read_derived_manifest` gives. See the module docstring."""

    spec_id: str
    content_version: str
    retention: RetentionModel
    recipe: Recipe | None
    roots: list[dict[str, Any]]
    citation: CitationModel


def _stored_recipe(reads: DerivedReads, *, spec_id: str, content_version: str) -> Recipe | None:
    record = reads.recipe_reader.read_recipe(spec_id=spec_id, content_version=content_version)
    return None if record is None else Recipe.model_validate(record.body)


def read_derived_manifest(
    *, spec_id: str, content_version: str, reads: DerivedReads
) -> DerivedManifestFacts:
    """The facts of the derived frame `{spec_id, content_version}`. See the module docstring.

    Raises:
        SnapshotNotFound, SnapshotRevoked: The version has no record, or is taken down.
        ArtifactExpired: No servable stored frame is left.
    """
    now = reads.now()
    stored = read_servable_version(
        reads.snapshot_reader, subject_id=spec_id, content_version=content_version, now=now
    )
    if stored_frame_location(stored.servable) is None:
        raise ArtifactExpired(id=spec_id, content_version=content_version)
    retention = version_retention(
        reads.snapshot_reader, subject_id=spec_id, content_version=content_version, now=now
    )
    recipe = _stored_recipe(reads, spec_id=spec_id, content_version=content_version)
    roots = [] if recipe is None else recipe_roots(recipe.model_dump(mode="json")["roots"])
    return DerivedManifestFacts(
        spec_id=spec_id,
        content_version=content_version,
        retention=retention,
        recipe=recipe,
        roots=roots,
        citation=roots_citation(reads.snapshot_reader, roots),
    )


def manifest_body(facts: DerivedManifestFacts) -> dict[str, Any]:
    """`facts` as the derived manifest route answers them (`doors.derived.DerivedManifest`)."""
    body: dict[str, Any] = {
        "spec_id": facts.spec_id,
        "content_version": facts.content_version,
        "retention": facts.retention.model_dump(mode="json"),
        "citation": facts.citation.model_dump(mode="json"),
        "roots": facts.roots,
        "recipe": None,
        "resolved_periods": [],
        "anchor": None,
        "left_out": None,
        "left_out_complete": None,
    }
    recipe = facts.recipe
    if recipe is None:
        return body
    anchor, left_out = recipe.anchor, recipe.left_out
    return {
        **body,
        "recipe": {
            "normal_form_hash": recipe.normal_form_hash,
            "replay_key": recipe.replay_key,
            "execution_identity": recipe.execution_identity,
            "tree": recipe.tree,
            "roots": recipe.roots,
        },
        "resolved_periods": list(
            resolved_periods(recipe.tree, rule=None if anchor is None else anchor.rule)
        ),
        "anchor": None if anchor is None else anchor.model_dump(mode="json"),
        "left_out": None
        if left_out is None
        else [series.model_dump(mode="json") for series in left_out.series],
        "left_out_complete": None if left_out is None else left_out.complete,
    }


# --- Files ---------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DerivedFile:
    """One rendered file of a derived frame, with the rows and columns of its data table."""

    data: bytes
    media_type: str
    file_name: str
    rows: int
    columns: int


@dataclass(frozen=True, kw_only=True)
class _StoredFrame:
    """The derived version's stored frame, when its bytes were first stored, and the version's
    servable locations."""

    table: Frame
    stored_on: datetime
    servable: Sequence[StoredArtifactLocation]


def _read_stored_frame(*, spec_id: str, content_version: str, reads: DerivedReads) -> _StoredFrame:
    """The stored frame of the derived version.

    Raises:
        SnapshotNotFound, SnapshotRevoked: The version has no record, or is taken down.
        ArtifactExpired: No servable stored frame is left, or its bytes are gone.
        StoredFrameUnreadable: The stored frame's bytes do not read as a frame.
    """
    stored = read_servable_version(
        reads.snapshot_reader, subject_id=spec_id, content_version=content_version, now=reads.now()
    )
    location = stored_frame_location(stored.servable)
    if location is None:
        raise ArtifactExpired(id=spec_id, content_version=content_version)
    try:
        data = reads.artifact_store.get(
            artifact_digest=location.artifact_digest, tier=Tier(location.tier)
        )
    except ArtifactNotFoundError:
        raise ArtifactExpired(id=spec_id, content_version=content_version) from None
    created = location.created_at
    # A store that drops the zone records UTC, as every write stamps it.
    stored_on = created if created.tzinfo is not None else created.replace(tzinfo=UTC)
    return _StoredFrame(
        table=read_stored_frame(
            data, media_type=location.media_type, digest=location.artifact_digest
        ),
        stored_on=stored_on,
        servable=stored.servable,
    )


def _frame_file(
    frame: _StoredFrame,
    *,
    spec_id: str,
    content_version: str,
    reads: DerivedReads,
    site: PublicSiteSettings,
) -> tuple[bytes, str]:
    """The bytes and media type of the derived version's frame file: the stored one, else one
    rendered again from `frame` (`render.render_and_store_frame_file`), the same bytes, and
    linked under the version. A link that fails to write is logged, and the bytes are served."""
    location = frame_file_location(frame.servable, site=site)
    if location is not None:
        try:
            data = reads.artifact_store.get(
                artifact_digest=location.artifact_digest, tier=Tier(location.tier)
            )
        except ArtifactNotFoundError:
            pass
        else:
            return data, location.media_type
    rendered = render_and_store_frame_file(
        frame_chunks=lambda: iter((frame.table,)),
        subject_id=spec_id,
        content_version=content_version,
        prepared_at=frame.stored_on,
        site=site,
        artifact_store=reads.artifact_store,
        enrichment=reads.enrichment,
    )
    try:
        reads.version_writer.record_artifact(
            subject_id=spec_id, content_version=content_version, artifact=rendered
        )
    except DatasetsStoreUnavailable as exc:
        logger.warning("Linking the frame file of %s %s failed: %s", spec_id, content_version, exc)
    data = reads.artifact_store.get(artifact_digest=rendered.artifact_digest, tier=Tier.working)
    return data, rendered.media_type


def _provenance_refs(table: Table) -> list[str]:
    """The distinct refs of `table`'s `provenance` column, in first-seen order."""
    if _PROVENANCE_COLUMN not in table.columns:
        return []
    refs = column_cells(table, _PROVENANCE_COLUMN)
    return [ref for ref in dict.fromkeys(refs) if isinstance(ref, str)]


def render_derived_file(
    *,
    spec_id: str,
    content_version: str,
    options: MintOptions,
    reads: DerivedReads,
    site: PublicSiteSettings,
) -> DerivedFile:
    """The derived frame's file in `options`' format, alone or in a bundle. See the module
    docstring.

    Raises:
        SnapshotNotFound, SnapshotRevoked, ArtifactExpired, StoredFrameUnreadable: As
            `_read_stored_frame`.
        ExcelLimitError, ExcelIllegalCharacterError, ParquetSizeBoundError: The writer's.
    """
    frame = _read_stored_frame(spec_id=spec_id, content_version=content_version, reads=reads)
    table, prepared_at = frame.table, frame.stored_on
    rows, columns = table.num_rows, len(table.columns)
    record = reads.recipe_reader.read_recipe(spec_id=spec_id, content_version=content_version)
    citation = recipe_citation(reads.snapshot_reader, record).sources_by_ref()
    sources = tuple(
        source_provenance(
            ref, cited.model_dump(mode="json") if (cited := citation.get(ref)) else None
        )
        for ref in _provenance_refs(table)
    )
    slug = sources_slug([source.name for source in sources])
    tag = content_version.removeprefix("ocv_")[:_TAG_LENGTH]
    if not options.bundle:
        if options.format == "parquet":
            data, media_type = _frame_file(
                frame, spec_id=spec_id, content_version=content_version, reads=reads, site=site
            )
        else:
            buffer = io.BytesIO()
            written = _WRITERS[options.format]((table,), rendering=_PLAIN, sink=buffer)
            data, media_type = buffer.getvalue(), written.media_type
        return DerivedFile(
            data=data,
            media_type=media_type,
            file_name=bare_file_name(
                slug=slug, prepared=prepared_at.date(), tag=tag, extension=options.format
            ),
            rows=rows,
            columns=columns,
        )

    scan = TableScan(role_columns=frame_entity_columns(table.columns))
    data_path = f"data/{slug}.{options.format}"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open(zipfile.ZipInfo(data_path, date_time=_ZIP_FIXED_DATE_TIME), mode="w") as entry:
            written = _WRITERS[options.format](scan.observe((table,)), rendering=_PLAIN, sink=entry)
        member = MemberProvenance(
            spec_id=spec_id,
            title=f"Computed by DFO from {', '.join(source.name for source in sources) or 'DFO data'}",
            artifact=ArtifactEntry(
                path=data_path,
                bytes=written.size_bytes,
                digest=written.digest,
                media_type=written.media_type,
                table_schema=table_schema_for(table, rendering=_PLAIN),
            ),
            sources=sources,
            columns=column_docs(scan, code_system_names={}),
        )
        origin = site.public_site_url
        documents = assemble_bundle(
            spec={"spec_id": spec_id},
            members=(member,),
            reproducing_call=None,
            context=BundleContext(
                heading=member.title,
                summary=summary_line((scan,), codes=()),
                slug=slug,
                version=content_version,
                prepared_at=prepared_at,
                site_url=origin,
                links=BundleLinks(
                    version=f"{origin}/api/datasets/derived/{spec_id}/{content_version}"
                ),
            ),
            codes=(),
        )
        for path, body in documents.items():
            zf.writestr(zipfile.ZipInfo(path, date_time=_ZIP_FIXED_DATE_TIME), body)
    return DerivedFile(
        data=out.getvalue(),
        media_type="application/zip",
        file_name=zip_file_name(
            slug=slug, prepared=prepared_at.date(), format=options.format, tag=tag
        ),
        rows=rows,
        columns=columns,
    )
