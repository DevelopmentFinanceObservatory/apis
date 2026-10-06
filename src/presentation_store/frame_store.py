"""`LiveFrameStore` resolves frames from dataset snapshots and digest-verified artifacts.
It resolves best-effort labels for frames.

A published `ops_` frame's version body holds only its lineage, so `load()` adds the citation
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

`load()` caches resolved frame bytes by `(spec_id, content_version)` up to 64 MiB. A load whose
citation or documentation read failed is cached too, marked incomplete, and each later load
re-runs only the citation read until one completes, so a failing read never refetches the frame
bytes. `load()` re-checks revocation and working-tier servability on every call, cache hit or
not, so a later takedown refuses the next load. `pin()` records a
presentation reference and promotes working artifacts to pinned. Definition mints store
frame1 renderings.
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

from dfo_frame_grammar import STAT_VAR_COLUMN, UNIT_COLUMN

from artifact_store import ArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from assembler.frame_view import compute_structural_view, dimension_columns, facet_dimension_names
from assembler.ports import FrameRef, FrameUnavailable, ResolvedFrame
from code_index.labels import family_label
from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureIndex
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    ArtifactReferenceWriter,
    SnapshotReader,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    select_servable_locations,
    servable_location_of_media_type,
)
from datasets_api.codes_snapshot import ArtifactNotFound
from datasets_api.manifest import build_citation, healed_citation
from datasets_api.manifest_models import CitationModel
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.graph import UNRESOLVED_NODE, GraphReader, GraphReadError
from materialise.provenance_doc import read_source_citations
from origin_catalogue import current_catalogue
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
provenance labels. A unit ref beyond this cap keeps its raw ref."""

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

# Grammar refs include `dcid:`. GraphReader expects bare dcids.
_STAT_VAR_REF_PREFIX = "dcid:"

# Route `spec_` family refs to `_resolve_family_labels`.
# Graph refs use other prefixes.
_FAMILY_REF_PREFIX = "spec_"

# The `provenance` column is labelled like stat_var and entity refs. measurementMethod keeps raw
# refs. Unit dcid refs are labelled under their own cap; structured units need no label.
_PROVENANCE_COLUMN = "provenance"

# A published derived frame's spec id. Its version body carries no citation.
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


@dataclass(frozen=True, slots=True)
class _CachedFrame:
    """A cached load: the frame bytes' size, the resolved frame, and whether every citation
    read behind its manifest completed."""

    size_bytes: int
    resolved: ResolvedFrame
    citation_complete: bool


class LiveFrameStore:
    """`graph` supplies entity/provenance labels. `spec_reader` supplies family labels.
    `None` for either disables only those labels; every other dependency is required.
    `code_index` and `measure_index` default to the packaged indexes. `now` supplies the clock
    for expiry checks and reference timestamps. `read_budget_seconds` bounds the graph reads of
    one `load()` from its start, as the module docstring describes, for a caller that sets no
    door deadline.
    """

    def __init__(
        self,
        *,
        snapshots: SnapshotReader,
        artifacts: ArtifactStore,
        graph: GraphReader | None,
        reference_writer: ArtifactReferenceWriter,
        spec_reader: SpecReader | None = None,
        code_index: CodeIndex | None = None,
        measure_index: MeasureIndex | None = None,
        now: Clock = utc_now,
        cache_max_bytes: int = _FRAME_CACHE_MAX_BYTES,
        read_budget_seconds: float = _READ_BUDGET_SECONDS,
    ) -> None:
        self._snapshots = snapshots
        self._artifacts = artifacts
        self._graph = graph
        self._reference_writer = reference_writer
        self._spec_reader = spec_reader
        self._code_index = code_index if code_index is not None else get_index()
        self._measure_index = measure_index if measure_index is not None else get_measure_index()
        self._now = now
        self._cache_max_bytes = cache_max_bytes
        self._read_budget_seconds = read_budget_seconds
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
        if cached is not None and cached.citation_complete:
            return cached.resolved

        if cached is None:
            data = self._fetch_bytes(canonical)
            try:
                doc = cast("Mapping[str, object]", json.loads(data))
            except ValueError as exc:
                raise FrameUnavailable(
                    reason="io",
                    detail=(
                        f"artifact_digest {canonical.artifact_digest!r} did not decode as JSON: "
                        f"{type(exc).__name__}"
                    ),
                ) from None
            labels, labels_resolved = self._resolve_labels(doc, deadline=deadline)
            frame = ResolvedFrame(
                frame_ref=frame_ref,
                frame=doc,
                structural_view=compute_structural_view(frame=doc),
                manifest=stored.immutable_body,
                labels=labels,
                retrieved_on=canonical.created_at.astimezone(datetime.UTC).date(),
            )
            size_bytes = len(data)
        else:
            frame, size_bytes, labels_resolved = cached.resolved, cached.size_bytes, True

        manifest, citation_complete = self._cited_manifest(
            frame, stored_manifest=stored.immutable_body, deadline=deadline
        )
        resolved = replace(frame, manifest=manifest)
        if labels_resolved:
            self._cache_put(
                cache_key,
                resolved,
                size_bytes=size_bytes,
                citation_complete=citation_complete,
            )
        return resolved

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
        _, canonical = self._resolve_canonical_location(frame_ref)
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
                )
                return True
            write = claim(
                principal=principal,
                artifact_digest=canonical.artifact_digest,
                subject_id=frame_ref.spec_id,
                content_version=frame_ref.content_version,
                reference_kind=PRESENTATION_PIN_REFERENCE_KIND,
                now=self._now(),
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

    def _resolve_canonical_location(
        self, frame_ref: FrameRef
    ) -> tuple[StoredVersion, StoredArtifactLocation]:
        """Resolve a frame ref to its stored version and servable frame1 location. A missing
        `spec_` row is `unknown`. A missing `ops_` row is `ops_unsupported` until published.
        Revocation of the requested or base version yields `revoked`. An unservable artifact
        yields `expired`. A missing frame1 rendering yields `missing_rendering`. Select by media
        type because versions can have several renderings.
        """
        stored = self._snapshots.read_version(frame_ref.spec_id, frame_ref.content_version)
        if stored is None:
            if frame_ref.spec_id.startswith("ops_"):
                raise FrameUnavailable(
                    reason="ops_unsupported",
                    detail=(
                        f"spec_id {frame_ref.spec_id!r} content_version "
                        f"{frame_ref.content_version!r} names an Analysis Toolkit op-spec "
                        "frame that was minted but not published to the platform"
                    ),
                )
            raise FrameUnavailable(
                reason="unknown",
                detail=(
                    f"no stored version for spec_id {frame_ref.spec_id!r} content_version "
                    f"{frame_ref.content_version!r}"
                ),
            )

        revocation = self._snapshots.read_revocation(frame_ref.spec_id, frame_ref.content_version)
        if revocation is not None:
            raise _revoked(
                spec_id=frame_ref.spec_id,
                content_version=frame_ref.content_version,
                reason=revocation.reason,
            )

        # A full groupby collapse's own version record names its base
        # materialisation's own `content_version` on
        # `stored.base_content_version` (`datasets.version_record`'s own
        # column, migration `0010_version_record_base_content_version.sql`).
        # `version_revocation` is keyed per `(subject_id, content_version)`,
        # so a takedown of the raw version alone would otherwise leave the
        # collapsed one servable here -- matches `datasets_api.doors.snapshot
        # ._refuse_if_revoked`. `None` for every non-collapsed version.
        base_content_version = stored.base_content_version
        if base_content_version is not None:
            base_revocation = self._snapshots.read_revocation(
                frame_ref.spec_id, base_content_version
            )
            if base_revocation is not None:
                raise _revoked(
                    spec_id=frame_ref.spec_id,
                    content_version=frame_ref.content_version,
                    reason=base_revocation.reason,
                )

        locations = self._snapshots.read_artifact_locations(
            frame_ref.spec_id, frame_ref.content_version
        )
        now = self._now()
        if not select_servable_locations(locations, now=now):
            raise FrameUnavailable(
                reason="expired",
                detail=(
                    f"no servable artifact location for spec_id {frame_ref.spec_id!r} "
                    f"content_version {frame_ref.content_version!r}"
                ),
            )

        canonical = servable_location_of_media_type(
            locations, media_type=FRAME1_MEDIA_TYPE, now=now
        )
        if canonical is None:
            raise FrameUnavailable(
                reason="missing_rendering",
                detail=(
                    f"no servable rendering for spec_id {frame_ref.spec_id!r} content_version "
                    f"{frame_ref.content_version!r} carries media_type {FRAME1_MEDIA_TYPE!r}"
                ),
            )
        return stored, canonical

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
        as a dataset version body's `citation`, each source with its documentation, and whether
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
        self, doc: Mapping[str, object], *, deadline: float
    ) -> tuple[dict[str, str], bool]:
        """Resolve labels for stat_var, entity, and provenance refs, capped at 300 in row
        order, then unit dcid refs, capped at 100 in row order, and say whether the graph read
        succeeded. `dcid:spec_...` family refs use stored definitions and `CodeIndex`. Other
        refs use one graph batch that stops at the absolute monotonic `deadline`. Missing
        readers, lookup/store failures, and over-cap refs remain unlabelled without failing
        the load. A graph failure, the deadline included, preserves family labels. A graph
        failure or a failed family definition read returns `False`, so `load` skips the cache.
        """
        columns = tuple(cast("Sequence[str]", doc["columns"]))
        if STAT_VAR_COLUMN not in columns:
            return {}, True
        rows = tuple(cast("Sequence[Sequence[object]]", doc["rows"]))
        index_by_column = {column: position for position, column in enumerate(columns)}
        facets = facet_dimension_names()
        ref_columns = tuple(
            column
            for column in dimension_columns(columns=columns)
            if column not in facets or column == _PROVENANCE_COLUMN
        )

        # Keep frame refs as label keys. Graph calls use bare stat_var dcids.
        distinct: dict[str, str] = {}
        for column in ref_columns:
            index = index_by_column[column]
            strip = column == STAT_VAR_COLUMN
            for row in rows:
                ref = row[index]
                if ref is None or ref in distinct:
                    continue
                distinct[cast(str, ref)] = (
                    cast(str, ref).removeprefix(_STAT_VAR_REF_PREFIX) if strip else cast(str, ref)
                )

        # Preserve row order when enforcing each cap. Later refs keep raw-ref fallbacks.
        capped = dict(tuple(distinct.items())[:_LABEL_RESOLUTION_CAP])
        if UNIT_COLUMN in index_by_column:
            unit_index = index_by_column[UNIT_COLUMN]
            unit_refs = dict.fromkeys(
                unit
                for row in rows
                if isinstance(unit := row[unit_index], str) and unit not in capped
            )
            capped.update((ref, ref) for ref in tuple(unit_refs)[:_UNIT_LABEL_CAP])
        if not capped:
            return {}, True

        family_refs: dict[str, str] = {}
        graph_bare_to_refs: dict[str, list[str]] = {}
        for ref, bare in capped.items():
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

    def _cache_put(
        self,
        cache_key: tuple[str, str],
        resolved: ResolvedFrame,
        *,
        size_bytes: int,
        citation_complete: bool,
    ) -> None:
        """Byte-bounded insertion-order cache keyed by `(spec_id, content_version)`. Frame refs
        identify immutable bytes, so entries need no expiry. Eviction follows insertion order
        and can drop hot entries. Keep one oversized frame. Adjust byte counts on replacement.
        """
        with self._cache_lock:
            prior = self._cache.get(cache_key)
            if prior is not None:
                self._cache_bytes -= prior.size_bytes
            self._cache[cache_key] = _CachedFrame(size_bytes, resolved, citation_complete)
            self._cache_bytes += size_bytes
            while self._cache_bytes > self._cache_max_bytes and self._cache:
                oldest_key = next(iter(self._cache))
                if oldest_key == cache_key and len(self._cache) == 1:
                    # Keep one oversized frame so repeated reads can hit the cache.
                    break
                self._cache_bytes -= self._cache.pop(oldest_key).size_bytes
