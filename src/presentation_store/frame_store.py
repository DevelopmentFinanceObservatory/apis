"""`LiveFrameStore` resolves frames from dataset snapshots and digest-verified artifacts.
It resolves best-effort labels for frames.

A published `ops_` frame's version stores only its lineage, so `load()` adds the citation
its `provenance` column names, read from the graph, as `manifest["citation"]`. The graph read
covers the first 300 distinct refs in row order; later refs are cited by their ref alone. A
dataset version's stored citation is served with its empty documentation and licence fields
filled from a live read (`datasets_api.manifest.healed_citation`), as a Datasets bundle cites it.

The label and citation graph reads stop at the earlier of two deadlines: the store's
`read_budget_seconds` from the start of `load()`, and the deadline a door sets with
`frame_reads_bounded_by` before it submits the job that calls `load()`. Each is less a grace of
one second or a tenth of its budget, whichever is smaller, so the graph reads stop before the
door's wait ends. A load whose graph label read or family definition read failed is not
cached, so the next load reads again. A quarantined family identity is a lasting state, so its
missing label is cached.

With a `recipe_reader`, `load()` reads a published `ops_` frame's recipe and attaches its
`anchor` as `ResolvedFrame.anchor`: the period "latest" names on that frame, and the series its
rule left out. A left-out series is absent from the derived frame, so its label is built here
from its coordinates as a series label is, through the same graph and family reads as the
frame's own refs, with the raw ref where they hold no name. A recipe that states a label keeps
it. Stored coordinates that name no series are labelled by their raw content, so no stored
recipe fails a load. A frame with no recipe, or a recipe with a null anchor, loads with no anchor.

`load()` reads a version's stored frame (`datasets_api.render.servable_frame_location`): its
canonical frame, Parquet, built into the parsed frame1 document the assembler reads
(`materialise.formats.canonical_frame_document`), or for a version holding only a frame1
document, that document. Distinct cells of the built document are shared objects, so a
250,000-row frame costs far less resident memory than a parsed frame1 document. A canonical
frame of more rows than `PUBLISH_MAX_ROWS` (250,000) refuses `missing_rendering`. A dataset
version, whose facts record `coverage.rows`, refuses on that count before its object is
fetched, which bounds the bytes and the document one load holds. A published `ops_` frame,
whose lineage records no count, is fetched whole and refuses on its Parquet footer's count
before its rows are decoded, which bounds the document; the publish door bounds its rows at
the write.

`load()` caches resolved frames by `(spec_id, content_version)` up to 64 MiB, a canonical frame
counted at its estimated resident size (`_document_bytes`) and a frame1 document at its byte
length. A pair keeps
its first recipe, so the anchor is cached with the frame. A derived frame cached before it had a
recipe is marked so, and each later load reads the recipe again until one exists. A load whose
citation or documentation read failed is cached too, marked incomplete, and each later load
re-runs only the citation read until one completes, so a failing read never refetches the frame
bytes. `load()` re-checks revocation and working-tier servability on every call, cache hit or
not, so a later takedown refuses the next load. `pin()` records a
presentation reference and promotes working artifacts to pinned.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from dfo_frame_grammar import STAT_VAR_COLUMN, UNIT_COLUMN
from dfo_frame_grammar.grammar import STAT_VAR_REF_PREFIX

from artifact_store import ArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from assembler.bind import role_columns, series_label
from assembler.frame_view import compute_structural_view, dimension_columns, facet_dimension_names
from assembler.ports import (
    FrameRef,
    FrameUnavailable,
    LeftOutSeries,
    ResolvedFrame,
    RollingAnchor,
)
from code_index.labels import family_label
from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureIndex
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    ArtifactNotFound,
    ArtifactReferenceWriter,
    FrameLineage,
    ManifestFacts,
    SnapshotNotFound,
    SnapshotReader,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    read_servable_version,
)
from datasets_api.codes_snapshot import SpecNotFound
from datasets_api.doors.publish import PUBLISH_MAX_ROWS
from datasets_api.manifest import build_citation, healed_citation, version_document
from datasets_api.manifest_models import CitationModel
from datasets_api.recipe_store import RecipeReader
from datasets_api.render import servable_frame_location
from datasets_api.subject_ids import served_subject_ids
from materialise.formats import (
    FRAME1_MEDIA_TYPE,
    canonical_frame_document,
    read_canonical_frame_table,
)
from materialise.graph import UNRESOLVED_NODE, GraphReader, GraphReadError
from materialise.provenance_doc import read_source_citations
from materialise.unit_cells import composed_unit_sides
from origin_catalogue import current_catalogue
from presentation_store.composed_stat_vars import (
    ComposedStatVar,
    composed_stat_var_labels,
    read_composed_stat_vars,
)
from spec_registry.errors import (
    IdentityQuarantined,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.reader import SpecReader

__all__ = [
    "FRAME1_MEDIA_TYPE",
    "PRESENTATION_PIN_REFERENCE_KIND",
    "PRESENTATION_PRINCIPAL_PREFIX",
    "LiveFrameStore",
    "frame_reads_bounded_by",
]

logger = logging.getLogger(__name__)

_FRAME_CACHE_MAX_BYTES = 64 * 1024 * 1024
"""Provisional (`limits.py`'s own naming-constant convention, mirrored here
rather than added there -- this bound is `presentation_store`'s own cache,
not an assembler-core resource maximum): total raw frame bytes this store's
per-instance cache holds before it evicts, oldest insertion first."""

_LABEL_RESOLUTION_CAP = 300
"""Distinct stat_var/entity/provenance refs `_resolve_labels` sends to
`GraphReader.node_arcs_many` in one `load()` call. A ref beyond this cap
keeps its raw-ref fallback rather than growing the batch further -- label
resolution's own contract already allows a partial map
(`assembler.ports.ResolvedFrame.labels`: "may be partial") -- so a frame
with an unusually large distinct-ref count cannot threaten this door's own
read budget (`presentation_api.config.PRESENTATION_FRAMES_TIMEOUT_SECONDS`,
default 30s)."""

_UNIT_LABEL_CAP = 100
"""Distinct unit dcid refs `_resolve_labels` adds to the same batch, after the
`_LABEL_RESOLUTION_CAP` dimension refs, so unit refs cannot crowd out stat_var, entity, or
provenance labels. A unit ref is a string `unit` cell or a string side of a composed unit. A
unit ref beyond this cap keeps its raw ref."""

_CITATION_READ_CAP = 300
"""Distinct provenance refs `_derived_citation` sends to `read_source_citations` in one
`load()` call, in row order. A ref beyond the cap is cited by its ref alone, as a ref the graph
does not hold is."""

_READ_BUDGET_SECONDS = 30.0
"""Default seconds a `load()` call may spend before its graph reads stop: the default of both
`PRESENTATION_COMPONENTS_TIMEOUT_SECONDS` and `PRESENTATION_FRAMES_TIMEOUT_SECONDS`, the doors
that call `load()`."""

PRESENTATION_PIN_REFERENCE_KIND = "presentation_draft"
"""This store's own `reference_kind` for `datasets.artifact_reference`
(migration `0007_artifact_references.sql`) -- that column is free text (its
own migration comment: "regardless of which reference_kind"), so this needs
no datasets-series migration to admit it, unlike the pin door's own `"pin"`
(`datasets_api.reference_store.PIN_REFERENCE_KIND`), a distinct kind: a
presentation component referencing a frame is a different fact from a
first-party principal pinning an artifact for its own retention."""

PRESENTATION_PRINCIPAL_PREFIX = "presentation:"
"""`pin()`'s own `datasets.artifact_reference.principal` prefix for a
`PRESENTATION_PIN_REFERENCE_KIND` reference -- the full principal is
`f"{PRESENTATION_PRINCIPAL_PREFIX}{component_id}"`. Named so a reader of
the reference table (e.g. `LivePresentationStore
.list_orphaned_presentation_references`'s own caller) can invert it back to
a `component_id` without duplicating the literal."""

# Route `spec_` family refs to `_resolve_family_labels`.
# Graph refs use other prefixes.
_FAMILY_REF_PREFIX = "spec_"

# The `provenance` column is labelled like stat_var and entity refs. measurementMethod keeps raw
# refs. Unit dcid refs are labelled under their own cap. Converted units need no label. A
# composed unit is labelled side by side, a string side as a unit ref and a StatVar side as a
# stat_var ref.
_PROVENANCE_COLUMN = "provenance"

# A published derived frame's spec id. Its version stores no citation.
_DERIVED_SPEC_PREFIX = "ops_"


_door_deadline: ContextVar[float | None] = ContextVar(
    "presentation_frame_read_deadline", default=None
)


def _graph_deadline(seconds: float) -> float:
    """The absolute monotonic time `seconds` from now, less a grace of one second or a tenth of
    `seconds`, whichever is smaller."""
    return time.monotonic() + max(0.0, seconds - min(1.0, seconds * 0.1))


@contextmanager
def frame_reads_bounded_by(*, timeout_seconds: float) -> Iterator[None]:
    """Bound the graph reads of every `LiveFrameStore.load()` run in this context, or in a copy
    of it such as `dfo_platform.executor.submit_with_context` takes, to `timeout_seconds` from
    now, less the grace. A door enters it before it submits a job it waits `timeout_seconds`
    for."""
    token = _door_deadline.set(_graph_deadline(timeout_seconds))
    try:
        yield
    finally:
        _door_deadline.reset(token)


def _revoked(*, spec_id: str, content_version: str, reason: str | None) -> FrameUnavailable:
    """Build a revoked-frame refusal using the requested IDs, including when its base was
    revoked.
    """
    detail = f"{spec_id!r} at content_version {content_version!r} was taken down"
    if reason is not None:
        detail += f": {reason}"
    return FrameUnavailable(reason="revoked", detail=detail)


def _io_failure(*, action: str, digest: str, exc: Exception) -> FrameUnavailable:
    """Build a retryable I/O refusal that carries only the exception class."""
    return FrameUnavailable(
        reason="io", detail=f"{action} artifact_digest {digest!r} raised {type(exc).__name__}"
    )


def _decode_frame(
    data: bytes, *, location: StoredArtifactLocation, max_rows: int
) -> tuple[Mapping[str, object], int]:
    """The frame1 document `data` holds and the bytes the cache counts for it: a frame1
    document parsed, at its byte length, or a canonical frame built into one, at
    `_document_bytes`. A canonical frame whose Parquet footer counts more than `max_rows` rows
    refuses `missing_rendering` before any row is decoded. Bytes that do not decode refuse
    `io`."""
    try:
        if location.media_type == FRAME1_MEDIA_TYPE:
            return cast("Mapping[str, object]", json.loads(data)), len(data)
        rows = pq.read_metadata(pa.BufferReader(data)).num_rows
        if rows > max_rows:
            raise _over_row_bound(location, rows=rows, max_rows=max_rows)
        doc = canonical_frame_document(read_canonical_frame_table(data))
    except (ValueError, pa.ArrowException) as exc:
        raise FrameUnavailable(
            reason="io",
            detail=(
                f"artifact_digest {location.artifact_digest!r} did not decode as a frame: "
                f"{type(exc).__name__}"
            ),
        ) from None
    rows = cast("Sequence[Sequence[object]]", doc["rows"])
    return doc, _document_bytes(rows=len(rows), columns=len(cast("Sequence[str]", doc["columns"])))


def _over_row_bound(
    location: StoredArtifactLocation, *, rows: int, max_rows: int
) -> FrameUnavailable:
    return FrameUnavailable(
        reason="missing_rendering",
        detail=(
            f"artifact_digest {location.artifact_digest!r} holds {rows} rows, over the "
            f"{max_rows} rows Presentation loads"
        ),
    )


def _recorded_rows(stored: StoredVersion) -> int | None:
    """The row count a dataset version's facts record as `coverage.rows`, or `None` for a
    published frame's lineage, which records none."""
    if not isinstance(stored.facts, ManifestFacts):
        return None
    rows = stored.facts.coverage.get("rows")
    return rows if isinstance(rows, int) else None


_ROW_LIST_BYTES = 56
"""CPython's size of an empty list, the per-row container of a built document."""

_VALUE_CELL_BYTES = 32
"""The float or int a row's value cell holds, with its allocator overhead."""


def _document_bytes(*, rows: int, columns: int) -> int:
    """The estimated resident size of a built canonical frame document: per row, its list,
    one pointer per column and its value cell. Strings and units are shared across rows, so
    they add little at scale."""
    return rows * (_ROW_LIST_BYTES + 8 * columns + _VALUE_CELL_BYTES)


def _capped_refs(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    composed_stat_vars: Sequence[ComposedStatVar] = (),
) -> dict[str, str]:
    """The refs to label in `rows`, each mapped to the bare dcid a graph read takes: stat_var,
    entity, and provenance refs up to `_LABEL_RESOLUTION_CAP`, then unit dcid refs up to
    `_UNIT_LABEL_CAP`, each in row order. The side refs of `composed_stat_vars`, then a
    composed unit's StatVar sides, count as stat_var refs ahead of the columns' own refs, and
    a composed unit's string sides as unit refs. A ref an entry describes is left out."""
    index_by_column = {column: position for position, column in enumerate(columns)}
    facets = facet_dimension_names()
    ref_columns = tuple(
        column
        for column in dimension_columns(columns=columns)
        if column not in facets or column == _PROVENANCE_COLUMN
    )

    unit_parts: tuple[object, ...] = ()
    if UNIT_COLUMN in index_by_column:
        unit_index = index_by_column[UNIT_COLUMN]
        unit_parts = tuple(
            part
            for row in rows
            for part in composed_unit_sides(row[unit_index]) or (row[unit_index],)
        )

    # Keep frame refs as label keys. Graph calls use bare stat_var dcids. The sides of a
    # composed StatVar and of a composed unit come first, since each labels the whole card
    # and frames hold few of them.
    composed = {entry.stat_var for entry in composed_stat_vars}
    stat_var_refs = [ref for entry in composed_stat_vars for ref in (*entry.first, *entry.second)]
    stat_var_refs += [
        cast("Mapping[str, str]", part)["stat_var"]
        for part in unit_parts
        if isinstance(part, Mapping) and "stat_var" in part
    ]
    distinct: dict[str, str] = {
        ref: ref.removeprefix(STAT_VAR_REF_PREFIX) for ref in stat_var_refs if ref not in composed
    }
    for column in ref_columns:
        index = index_by_column[column]
        strip = column == STAT_VAR_COLUMN
        for row in rows:
            ref = row[index]
            if ref is None or ref in distinct or ref in composed:
                continue
            distinct[cast(str, ref)] = (
                cast(str, ref).removeprefix(STAT_VAR_REF_PREFIX) if strip else cast(str, ref)
            )

    # Preserve row order when enforcing each cap. Later refs keep raw-ref fallbacks.
    capped = dict(tuple(distinct.items())[:_LABEL_RESOLUTION_CAP])
    unit_refs = dict.fromkeys(
        part for part in unit_parts if isinstance(part, str) and part not in capped
    )
    capped.update((ref, ref) for ref in tuple(unit_refs)[:_UNIT_LABEL_CAP])
    return capped


def _left_out_label(coordinates: Mapping[str, str], *, labels: Mapping[str, str]) -> str:
    """A left-out series' label, as `assembler.bind.series_label` builds a series label.
    Coordinates that hold neither `stat_var` nor an entity role are labelled by their raw
    content, so a stored recipe never fails its frame's load."""
    columns = tuple(coordinates)
    if STAT_VAR_COLUMN in coordinates or role_columns(columns=columns):
        return series_label(coordinates=coordinates, columns=columns, labels=labels)
    return ", ".join(f"{column}={value}" for column, value in coordinates.items())


@dataclass(frozen=True, slots=True)
class _CachedFrame:
    """A cached load: the frame bytes' size, the resolved frame, whether every citation read
    behind its manifest completed, and whether its anchor is final. The anchor of a derived
    frame is final once its recipe has been read, and that of any other frame always."""

    size_bytes: int
    resolved: ResolvedFrame
    citation_complete: bool
    anchor_final: bool


class LiveFrameStore:
    """`graph` supplies entity/provenance labels, and `None` disables them. `spec_reader`
    supplies family labels and the served id a frame ref's revocation check covers; `None`
    disables those labels and narrows the check to the ref's own id. Every other dependency is
    required.
    `recipe_reader` supplies a derived frame's rolling anchor, and `None` loads every frame
    without one.
    `code_index` and `measure_index` default to the packaged indexes. `now` supplies the clock
    for expiry checks and reference timestamps. `read_budget_seconds` bounds the graph reads of
    one `load()` from its start, as the module docstring describes, for a caller that sets no
    door deadline. `frame_max_rows` bounds the rows of a canonical frame `load()` decodes, since
    the whole frame is built in memory. A larger frame refuses `missing_rendering`, before its
    object is fetched when the version's facts record its row count.
    """

    def __init__(
        self,
        *,
        snapshots: SnapshotReader,
        artifacts: ArtifactStore,
        graph: GraphReader | None,
        reference_writer: ArtifactReferenceWriter,
        spec_reader: SpecReader | None = None,
        recipe_reader: RecipeReader | None = None,
        code_index: CodeIndex | None = None,
        measure_index: MeasureIndex | None = None,
        now: Clock = utc_now,
        cache_max_bytes: int = _FRAME_CACHE_MAX_BYTES,
        read_budget_seconds: float = _READ_BUDGET_SECONDS,
        frame_max_rows: int = PUBLISH_MAX_ROWS,
    ) -> None:
        self._snapshots = snapshots
        self._artifacts = artifacts
        self._graph = graph
        self._reference_writer = reference_writer
        self._spec_reader = spec_reader
        self._recipe_reader = recipe_reader
        self._code_index = code_index if code_index is not None else get_index()
        self._measure_index = measure_index if measure_index is not None else get_measure_index()
        self._now = now
        self._cache_max_bytes = cache_max_bytes
        self._read_budget_seconds = read_budget_seconds
        self._frame_max_rows = frame_max_rows
        self._cache_lock = threading.Lock()
        self._cache: dict[tuple[str, str], _CachedFrame] = {}
        self._cache_bytes = 0

    # -- FrameStore ---------------------------------------------------

    def load(self, *, frame_ref: FrameRef) -> ResolvedFrame:
        # Recheck revocation and servability on every load. Cache only immutable frame data.
        deadline = _graph_deadline(self._read_budget_seconds)
        door_deadline = _door_deadline.get()
        if door_deadline is not None:
            deadline = min(deadline, door_deadline)
        cache_key = (frame_ref.spec_id, frame_ref.content_version)
        stored, canonical = self._resolve_canonical_location(frame_ref)
        cached = self._cache_get(cache_key)
        if cached is not None and cached.citation_complete and cached.anchor_final:
            return cached.resolved

        document = version_document(stored)
        if cached is None:
            recorded_rows = _recorded_rows(stored)
            if (
                canonical.media_type != FRAME1_MEDIA_TYPE
                and recorded_rows is not None
                and recorded_rows > self._frame_max_rows
            ):
                raise _over_row_bound(canonical, rows=recorded_rows, max_rows=self._frame_max_rows)
            data = self._fetch_bytes(canonical)
            doc, size_bytes = _decode_frame(data, location=canonical, max_rows=self._frame_max_rows)
            labels, labels_resolved = self._resolve_labels(
                doc,
                composed_stat_vars=read_composed_stat_vars(
                    stored.facts.composed_stat_vars
                    if isinstance(stored.facts, FrameLineage)
                    else None,
                    frame_ref=frame_ref,
                ),
                deadline=deadline,
            )
            frame = ResolvedFrame(
                frame_ref=frame_ref,
                frame=doc,
                structural_view=compute_structural_view(frame=doc),
                manifest=document,
                labels=labels,
                retrieved_on=canonical.created_at.astimezone(datetime.UTC).date(),
            )
            citation_complete = anchor_final = False
        else:
            frame, size_bytes, labels_resolved = cached.resolved, cached.size_bytes, True
            citation_complete, anchor_final = cached.citation_complete, cached.anchor_final

        if not anchor_final:
            anchor, anchor_final, anchor_labelled = self._recipe_anchor(frame, deadline=deadline)
            frame = replace(frame, anchor=anchor)
            labels_resolved = labels_resolved and anchor_labelled
        if not citation_complete:
            manifest, citation_complete = self._cited_manifest(
                frame, stored_manifest=document, deadline=deadline
            )
            frame = replace(frame, manifest=manifest)
        if labels_resolved:
            self._cache_put(
                cache_key,
                _CachedFrame(
                    size_bytes=size_bytes,
                    resolved=frame,
                    citation_complete=citation_complete,
                    anchor_final=anchor_final,
                ),
            )
        return frame

    def _recipe_anchor(
        self, frame: ResolvedFrame, *, deadline: float
    ) -> tuple[RollingAnchor | None, bool, bool]:
        """The rolling anchor a derived frame's recipe records, whether that answer is final,
        and whether every label read behind it succeeded.

        A frame that is not derived, and any frame of a store with no recipe reader, has no
        anchor, finally. A derived frame with no recipe has none yet, since a recipe can be
        written after the frame is published. Each left-out series is labelled from its
        coordinates (`_left_out_label`), over the frame's labels plus one `_label_refs` read of
        the refs the frame does not label. A label the recipe states is kept."""
        spec_id = frame.frame_ref.spec_id
        if self._recipe_reader is None or not spec_id.startswith(_DERIVED_SPEC_PREFIX):
            return None, True, True
        recipe = self._recipe_reader.read_recipe(
            spec_id=spec_id, content_version=frame.frame_ref.content_version
        )
        if recipe is None:
            return None, False, True
        body = cast("Mapping[str, object] | None", recipe.body.get("anchor"))
        if body is None:
            return None, True, True

        left_out = cast("Sequence[Mapping[str, object]]", body["left_out"])
        coordinates = [cast("Mapping[str, str]", one["coordinates"]) for one in left_out]
        columns = tuple(dict.fromkeys(column for one in coordinates for column in one))
        refs = {
            ref: bare
            for ref, bare in _capped_refs(
                columns=columns,
                rows=[[one.get(column) for column in columns] for one in coordinates],
            ).items()
            if ref not in frame.labels
        }
        read, labelled = self._label_refs(refs, deadline=deadline)
        labels = {**frame.labels, **read}
        anchor = RollingAnchor(
            period=cast(str, body["period"]),
            rule=cast(str, body["rule"]),
            left_out=tuple(
                LeftOutSeries(
                    root=cast(int, one["root"]),
                    coordinates=series,
                    label=cast("str | None", one.get("label"))
                    or _left_out_label(series, labels=labels),
                )
                for one, series in zip(left_out, coordinates, strict=True)
            ),
        )
        return anchor, True, labelled

    def _cited_manifest(
        self, frame: ResolvedFrame, *, stored_manifest: Mapping[str, object], deadline: float
    ) -> tuple[Mapping[str, object], bool]:
        """The manifest with its live citation, and whether every citation read completed. A
        derived frame cites its provenance refs from the graph. A root frame's stored citation
        has its empty fields filled. With no graph, or no stored citation on a root frame, the
        stored manifest stands and is complete."""
        if frame.frame_ref.spec_id.startswith(_DERIVED_SPEC_PREFIX):
            citation, complete = self._derived_citation(
                frame.frame, frame_ref=frame.frame_ref, deadline=deadline
            )
            if citation is None:
                return stored_manifest, complete
            return {**stored_manifest, "citation": citation}, complete
        if self._graph is None or "citation" not in stored_manifest:
            return stored_manifest, True
        stored_citation = CitationModel.model_validate(stored_manifest["citation"])
        citation, complete = healed_citation(
            stored_citation, graph=self._graph, deadline=deadline, coverage_index=None
        )
        if citation is stored_citation:
            return stored_manifest, complete
        return {**stored_manifest, "citation": citation.model_dump(mode="json")}, complete

    def pin(self, *, frame_ref: FrameRef, component_id: str) -> bool:
        # The read and the write-time recheck honour revocations under the same ids.
        revoked_under = self._revoked_under(frame_ref.spec_id)
        _, canonical = self._resolve_canonical_location(frame_ref, revoked_under=revoked_under)
        if canonical.tier != Tier.pinned.value:
            try:
                self._artifacts.copy(
                    canonical.artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned
                )
            except ArtifactNotFoundError as exc:
                raise FrameUnavailable(
                    reason="expired",
                    detail=(
                        f"artifact_digest {canonical.artifact_digest!r} aged out of the working "
                        "tier before it could be pinned"
                    ),
                ) from exc
            except Exception as exc:
                # Copy-side write or inventory failure -- see `_io_failure`.
                raise _io_failure(
                    action="pinning", digest=canonical.artifact_digest, exc=exc
                ) from None

        principal = f"{PRESENTATION_PRINCIPAL_PREFIX}{component_id}"
        try:
            claim = getattr(self._reference_writer, "claim_reference_unless_revoked", None)
            if claim is None:
                self._reference_writer.record_reference_unless_revoked(
                    principal=principal,
                    artifact_digest=canonical.artifact_digest,
                    subject_id=frame_ref.spec_id,
                    content_version=frame_ref.content_version,
                    reference_kind=PRESENTATION_PIN_REFERENCE_KIND,
                    now=self._now(),
                    revoked_under=revoked_under,
                )
                return True
            write = claim(
                principal=principal,
                artifact_digest=canonical.artifact_digest,
                subject_id=frame_ref.spec_id,
                content_version=frame_ref.content_version,
                reference_kind=PRESENTATION_PIN_REFERENCE_KIND,
                now=self._now(),
                revoked_under=revoked_under,
            )
        except SnapshotRevoked as exc:
            raise FrameUnavailable(reason="revoked", detail=str(exc)) from exc
        except ArtifactNotFound as exc:
            # A re-render replaced the link after its working copy aged out during the copy.
            raise FrameUnavailable(reason="expired", detail=str(exc)) from exc
        return write.changed

    def release_pin(self, *, frame_ref: FrameRef, component_id: str) -> None:
        """Remove this component's reference after resolving the artifact digest. The pinned
        copy stays available for other references or inventory sweeps.
        """
        _, canonical = self._resolve_canonical_location(frame_ref)
        principal = f"{PRESENTATION_PRINCIPAL_PREFIX}{component_id}"
        self._reference_writer.remove_reference(
            principal=principal,
            artifact_digest=canonical.artifact_digest,
            subject_id=frame_ref.spec_id,
            content_version=frame_ref.content_version,
            reference_kind=PRESENTATION_PIN_REFERENCE_KIND,
            now=self._now(),
        )

    # -- Resolution -----------------------------------------------------

    def _revoked_under(self, spec_id: str) -> tuple[str, ...]:
        """The ids whose revocation withdraws a frame ref naming `spec_id`, as for a Datasets
        request: a ref naming a served id honours revocations under it, and a ref naming a
        posted id honours revocations under the posted id and every served id the registry
        binds to it (`datasets_api.subject_ids.served_subject_ids`). An id no definition
        registers, such as an `ops_` op-spec, has no served id, nor does any id when the store
        has no spec reader.

        The check fails closed: a quarantined id, or a quarantined served id bound to it,
        refuses `quarantined`, and a registry that cannot answer raises
        `SpecRegistryUnavailable` or `SpecRegistryStoreError`, which the doors answer as
        retryable 503."""
        if self._spec_reader is None:
            return ()
        try:
            return served_subject_ids(spec_id, spec_reader=self._spec_reader)
        except SpecNotFound:
            return ()
        except IdentityQuarantined as exc:
            raise FrameUnavailable(reason="quarantined", detail=str(exc)) from None

    def _resolve_canonical_location(
        self, frame_ref: FrameRef, *, revoked_under: Sequence[str] | None = None
    ) -> tuple[StoredVersion, StoredArtifactLocation]:
        """Resolve a frame ref to its stored version and servable stored frame location in one
        `read_servable_version` transaction, the check the snapshot door makes. A missing
        `spec_` row is `unknown`. A missing `ops_` row is `ops_unsupported` until published.
        Revocation of the requested or base version yields `revoked`, under the ref's id and,
        for a ref naming a posted id, its served id too (`_revoked_under`, read here unless the
        caller passes `revoked_under`). An unservable artifact yields `expired`. A version with
        neither a canonical frame nor a frame1 document yields `missing_rendering`.
        """
        now = self._now()
        try:
            stored = read_servable_version(
                self._snapshots,
                subject_id=frame_ref.spec_id,
                content_version=frame_ref.content_version,
                now=now,
                revoked_under=(
                    self._revoked_under(frame_ref.spec_id)
                    if revoked_under is None
                    else revoked_under
                ),
            )
        except SnapshotNotFound:
            if frame_ref.spec_id.startswith("ops_"):
                raise FrameUnavailable(
                    reason="ops_unsupported",
                    detail=(
                        f"spec_id {frame_ref.spec_id!r} content_version "
                        f"{frame_ref.content_version!r} names an Analysis Toolkit op-spec "
                        "frame that was minted but not published to the platform"
                    ),
                ) from None
            raise FrameUnavailable(
                reason="unknown",
                detail=(
                    f"no stored version for spec_id {frame_ref.spec_id!r} content_version "
                    f"{frame_ref.content_version!r}"
                ),
            ) from None
        except SnapshotRevoked as exc:
            raise _revoked(
                spec_id=frame_ref.spec_id,
                content_version=frame_ref.content_version,
                reason=exc.reason,
            ) from None

        if not stored.servable:
            raise FrameUnavailable(
                reason="expired",
                detail=(
                    f"no servable artifact location for spec_id {frame_ref.spec_id!r} "
                    f"content_version {frame_ref.content_version!r}"
                ),
            )

        canonical = servable_frame_location(stored.servable, now=now)
        if canonical is None:
            raise FrameUnavailable(
                reason="missing_rendering",
                detail=(
                    f"no servable stored frame for spec_id {frame_ref.spec_id!r} "
                    f"content_version {frame_ref.content_version!r}"
                ),
            )
        return stored.version, canonical

    def _fetch_bytes(self, canonical: StoredArtifactLocation) -> bytes:
        try:
            return self._artifacts.get(
                artifact_digest=canonical.artifact_digest, tier=Tier(canonical.tier)
            )
        except ArtifactIntegrityError as exc:
            raise FrameUnavailable(
                reason="digest_mismatch",
                detail=f"artifact_digest {canonical.artifact_digest!r} failed integrity verification",
            ) from exc
        except ArtifactNotFoundError as exc:
            raise FrameUnavailable(
                reason="expired",
                detail=f"artifact_digest {canonical.artifact_digest!r} is no longer stored",
            ) from exc
        except Exception as exc:
            # Transport/driver failure -- see `_io_failure`.
            raise _io_failure(
                action="fetching", digest=canonical.artifact_digest, exc=exc
            ) from None

    def _derived_citation(
        self, doc: Mapping[str, object], *, frame_ref: FrameRef, deadline: float
    ) -> tuple[dict[str, object] | None, bool]:
        """The citation for a derived frame's distinct `provenance` refs, in row order, shaped
        as a dataset version manifest's `citation`, each source with its documentation, and whether
        `load` may cache it. The graph read covers the first `_CITATION_READ_CAP` refs and stops
        at the absolute monotonic `deadline`. With no graph, and for refs beyond the cap, each
        source is named by its ref alone. A failed or overdue citation read is logged and
        returns `(None, False)`, and `load` then leaves the citation out and skips the cache so
        the next load reads it again. A failed documentation read leaves only that source's
        `documentation` out and also skips the cache. A source the graph documents nothing for
        is cacheable.
        """
        columns = tuple(cast("Sequence[str]", doc["columns"]))
        refs: list[str] = []
        if _PROVENANCE_COLUMN in columns:
            index = columns.index(_PROVENANCE_COLUMN)
            rows = cast("Sequence[Sequence[object]]", doc["rows"])
            refs = list(dict.fromkeys(cast(str, row[index]) for row in rows if row[index]))
        source_citations = {}
        if self._graph is not None and refs:
            try:
                source_citations = read_source_citations(
                    self._graph,
                    refs[:_CITATION_READ_CAP],
                    catalogue=current_catalogue(),
                    deadline=deadline,
                    documented=True,
                )
            except GraphReadError:
                logger.warning(
                    "Citation read failed for spec_id %s content_version %s (%d refs); "
                    "serving the frame without a citation",
                    frame_ref.spec_id,
                    frame_ref.content_version,
                    len(refs),
                    exc_info=True,
                )
                return None, False
        citation, _ = build_citation(provenance_refs=refs, source_citations=source_citations)
        cacheable = not any(cited.documentation_read_failed for cited in source_citations.values())
        return citation.model_dump(mode="json"), cacheable

    def _resolve_labels(
        self,
        doc: Mapping[str, object],
        *,
        composed_stat_vars: Sequence[ComposedStatVar],
        deadline: float,
    ) -> tuple[dict[str, str], bool]:
        """Resolve the frame's ref labels (`_resolve_ref_labels`), then word each composed
        StatVar the frame's version stores from its sides' labels
        (`composed_stat_var_labels`), under the ref its `stat_var` cells hold. A side the read
        left unlabelled reads as its bare dcid, and the returned flag is the read's own, so a
        failed read is not cached."""
        labels, resolved = self._resolve_ref_labels(
            doc, composed_stat_vars=composed_stat_vars, deadline=deadline
        )
        labels.update(composed_stat_var_labels(entries=composed_stat_vars, labels=labels))
        return labels, resolved

    def _resolve_ref_labels(
        self,
        doc: Mapping[str, object],
        *,
        composed_stat_vars: Sequence[ComposedStatVar],
        deadline: float,
    ) -> tuple[dict[str, str], bool]:
        """Resolve labels for stat_var, entity, and provenance refs, capped at 300 in row
        order, then unit dcid refs, capped at 100 in row order, and say whether the graph read
        succeeded. The side refs of `composed_stat_vars`, then a composed unit's StatVar sides,
        count as stat_var refs ahead of the columns' own refs, and a composed unit's string
        sides as unit refs. A ref an entry describes is left out of the read. `dcid:spec_...`
        family refs use stored definitions and `CodeIndex`. Other refs use one graph batch that
        stops at the absolute monotonic `deadline`. Missing readers, lookup/store failures, and
        over-cap refs remain unlabelled without failing the load. A graph failure, the deadline
        included, preserves family labels. A graph failure or a failed family definition read
        returns `False`, so `load` skips the cache.
        """
        columns = tuple(cast("Sequence[str]", doc["columns"]))
        if STAT_VAR_COLUMN not in columns:
            return {}, True
        rows = tuple(cast("Sequence[Sequence[object]]", doc["rows"]))
        refs = _capped_refs(columns=columns, rows=rows, composed_stat_vars=composed_stat_vars)
        return self._label_refs(refs, deadline=deadline)

    def _label_refs(
        self, refs: Mapping[str, str], *, deadline: float
    ) -> tuple[dict[str, str], bool]:
        """Label `refs`, a map from each ref to its bare dcid, and say whether every read
        succeeded, as `_resolve_labels` describes."""
        family_refs: dict[str, str] = {}
        graph_bare_to_refs: dict[str, list[str]] = {}
        for ref, bare in refs.items():
            if bare.startswith(_FAMILY_REF_PREFIX):
                family_refs[ref] = bare
            else:
                graph_bare_to_refs.setdefault(bare, []).append(ref)

        labels: dict[str, str] = {}
        family_read_ok = True
        if family_refs:
            family_labels, family_read_ok = self._resolve_family_labels(family_refs)
            labels.update(family_labels)

        if not graph_bare_to_refs or self._graph is None:
            return labels, family_read_ok

        try:
            arcs_by_bare = self._graph.node_arcs_many(
                tuple(graph_bare_to_refs), properties=("name",), deadline=deadline
            )
        except GraphReadError:
            return labels, False

        for bare, arcs in arcs_by_bare.items():
            if arcs is None:
                continue
            # The last truthy resolved name is the fuller roll-up label.
            values = [value for value in arcs.get("name", ()) if value and value != UNRESOLVED_NODE]
            if not values:
                continue
            label = values[-1]
            for ref in graph_bare_to_refs[bare]:
                labels[ref] = label
        return labels, family_read_ok

    def _resolve_family_labels(self, family_refs: Mapping[str, str]) -> tuple[dict[str, str], bool]:
        """Label family refs with one batched definition read and `family_label`, and say
        whether the read succeeded. Missing readers, definitions, or code-index labels omit the
        label without failing the load. A quarantined identity omits every label and counts as
        a successful read; an unavailable or failing registry returns `False`.
        """
        if self._spec_reader is None:
            return {}, True
        spec_ids = tuple(dict.fromkeys(family_refs.values()))
        try:
            stored_by_id = self._spec_reader.read_definitions(spec_ids)
        except IdentityQuarantined:
            return {}, True
        except (SpecRegistryUnavailable, SpecRegistryStoreError):
            return {}, False

        labels: dict[str, str] = {}
        for ref, spec_id in family_refs.items():
            stored = stored_by_id.get(spec_id)
            if stored is None:
                continue
            label = family_label(
                stored.identity, index=self._code_index, measure_index=self._measure_index
            )
            if label is not None:
                labels[ref] = label
        return labels, True

    # -- Cache ------------------------------------------------------------

    def _cache_get(self, cache_key: tuple[str, str]) -> _CachedFrame | None:
        with self._cache_lock:
            return self._cache.get(cache_key)

    def _cache_put(self, cache_key: tuple[str, str], entry: _CachedFrame) -> None:
        """Byte-bounded insertion-order cache keyed by `(spec_id, content_version)`. Frame refs
        identify immutable bytes, so entries need no expiry. Eviction follows insertion order
        and can drop hot entries. Keep one oversized frame. Adjust byte counts on replacement.
        """
        with self._cache_lock:
            prior = self._cache.get(cache_key)
            if prior is not None:
                self._cache_bytes -= prior.size_bytes
            self._cache[cache_key] = entry
            self._cache_bytes += entry.size_bytes
            while self._cache_bytes > self._cache_max_bytes and self._cache:
                oldest_key = next(iter(self._cache))
                if oldest_key == cache_key and len(self._cache) == 1:
                    # Keep one oversized frame so repeated reads can hit the cache.
                    break
                self._cache_bytes -= self._cache.pop(oldest_key).size_bytes
