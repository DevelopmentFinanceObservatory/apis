"""`LiveFrameStore` resolves frames from dataset snapshots and digest-verified artifacts.
It resolves best-effort labels for frames.

A published `ops_` frame's version stores only its lineage, so `load()` adds a citation as
`manifest["citation"]`. A frame whose recipe names roots that cite sources is cited as the
derived manifest cites it, its roots' citations merged (`datasets_api.derived.recipe_citation`),
and those are filled as a dataset version's stored citation is. A frame with no recipe, or
whose roots cite nothing, is cited with the sources its `provenance` column names, read from
the graph. That read covers the first 300 distinct refs in row order; later refs are cited by
their ref alone. A dataset version's stored citation is served with its empty documentation and licence fields
filled from a live read of the sources with an empty field
(`datasets_api.manifest.citation_healed_from`), as a Datasets bundle cites it; a stored
citation with every field set reads nothing. A frame that holds a description record reads
every cited source, since its card words what each source's document says. Either citation
read documents each source it reads, one read per load, and the frame carries what each
source's document says, its publisher, last download and next release, as
`ResolvedFrame.source_documents`.

The label and citation graph reads stop at the earlier of two deadlines: the store's
`read_budget_seconds` from the start of `load()`, and the deadline a door sets with
`frame_reads_bounded_by` before it submits the job that calls `load()`. Each is less a grace of
one second or a tenth of its budget, whichever is smaller, so the graph reads stop before the
door's wait ends. A load whose graph label read or family definition read failed is not
cached, so the next load reads again. A quarantined family identity is a lasting state, so its
missing label is cached. The load that reads the labels stamps the frame with the graph
generation the read starts under and the build of the catalogue the load is bound to, with the
bound release (`ResolvedFrame.graph_generation`,
`ResolvedFrame.catalogue_build`), and the cached frame keeps
both, so a card's vocabulary revision names the read its labels came from. A cached frame's
words, its labels, its anchor's left-out labels, its citation and its source documents, are
read again from its cached bytes once the graph generation or the bound catalogue build moves
past those stamps, or once they are older than that generation's graph cache lifetime (a day
for a known generation, an hour for an unknown one), as the graph client's own caches expire.
A load that re-reads them stamps the frame afresh. A re-read whose label, left-out label or
citation read fails keeps the cached words and their stamps and is tried again once
`_RECHECK` has passed, so no card is composed from fallbacks where good words were held; a
load in between that retries only a citation or a record keeps that wait. A card revision
already issued keeps the description it was issued with.

With a `recipe_reader`, `load()` reads a published `ops_` frame's recipe and attaches its
replay key as `ResolvedFrame.replay_key` and its `anchor` as `ResolvedFrame.anchor`: the period "latest" names on that frame, and the series its
rule left out. A left-out series is absent from the derived frame, so its label is built here
from its coordinates as a series label is, through the same graph and family reads as the
frame's own refs, with the raw ref where they hold no name. A recipe that states a label keeps
it. Stored coordinates that name no series are labelled by their raw content, so no stored
recipe fails a load. A frame with no recipe, or a recipe with a null anchor, loads with no anchor.

With a `description_reader`, `load()` reads the version's description record and attaches it as
`ResolvedFrame.description_record`. A version with no record yet loads with none, and a load
`_RECHECK` (a minute) or more later reads the record again, as it does for a
computed version's record holding `steps: null`, which a later publish fills. A failed record
read raises `DatasetsStoreUnavailable`, as a failed version read does, so no frame loads
without the record its version holds. A stored record that fails validation is logged as an
error, and its frame loads and is cached with none.

A computed version's record names the versions it was computed from. The load that reads the
record reads theirs too, in one batched read (`DescriptionReader.read_descriptions`), and
attaches them as `ResolvedFrame.input_records`, so a card names each input by its title. The
refs each input's title names (`datasets_api.description_record.title_named`) join the frame's
label read under their bare dcids, within the label read's cap. An input with no record yet
is left out, and an input record holding `steps: null` is held, and either leaves the frame's
record not final, so a later load reads the inputs again. A
failed input read raises as a failed record read does.

A version's warnings carry by ref the entities, group members, series and provenances they
count: the bound entities and members that served nothing, the entities a conversion left out,
and the series and provenances the registry withheld. A stored warning's message may name
none of them (`assembler.mint.warned_refs`, `assembler.mint.warned_provenance_refs`). The
entities, members and series hold no row, so their refs join the frame's label read ahead of
its own refs. A provenance is named as a source citation names it (`read_source_citations`):
its catalogue row, else its parent Dataset's name, else its own. A card names them all by
label.

A frame's continuation (`ResolvedFrame.continuation`, `assembler.frame_view.frame_continuation`)
comes from its stored version alone: a dataset version's `trace.continuations`, or the
`continuations` a computed version stored when it was published, inherited from the versions it
was computed from, so a derived frame over a series its source's estimates continue groups
those rows into one series, as the root does, from its first load. Description records arriving
later do not change it. A computed version published without steps stores the continuations its
lineage gives, and the publish that fills its steps replaces them with those its retrieved
versions give; its record holds `steps: null` until then, so a cached frame of it is read again
at most once per `_RECHECK`, and a load that finds the stored continuations changed computes the
structural view again. A computed version recorded before versions stored continuations holds
them as unknown (`continuations: null`, `assembler.frame_view.continuation_known`) until the
`backfill-descriptions` task fills them; it loads ungrouped meanwhile, and its cached frame is not
final, so it is read again at most once per `_RECHECK` and groups once the backfill lands.

A computed version's rows sit under variables the graph does not hold, which its record names
(`_computed_labels`). A step that writes a new variable names it in `stat_var`, and the variable
takes the name of that step's output (`description.steps_text.step_output_names`), each input
named by its title (`description.compose.version_title`) as the card's description names it. A
variable no step names takes the name `description.steps_text.composed_names` words from the
record's `computed.composed` entry, a body stored before steps named their variables. A
version with no record, before the backfill describes it or after its publish could not,
words the composed variables its stored lineage holds (`composed_stat_vars`) the same way. A
variable named by neither keeps its ref. The graph read leaves out the variables the record
names and reads the variables of composed entries' sides first, so the entries' sides are
labelled in the same read. A record that arrives or changes after the first load re-reads all
the frame's labels as the first load reads them, under a fresh generation stamp. That load
re-reads the frame's other words too, so a record arriving on a frame cached without one
widens its citation read to every cited source. When its label read fails, the new record is
cached with its names over the labels the frame held, under the old stamps, and both the
record and the words are read again once `_RECHECK` has passed. A name of two sides keeps both
sides and the word between them within `LABEL_MAX` characters
(`assembler.limits.joined_label`), and each name is capitalised and cut to `LABEL_MAX`
characters.

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
citation or documentation read failed is cached too, with the citation that read left, and a
load once `_RECHECK` has passed re-runs only the citation read, until one completes, so a
failing read costs at most one read a minute and never refetches the frame bytes. A cached
frame without a final description record is read the same way, at most once
per `_RECHECK`: a load that late reads only the record, and the labels it names,
until the record is final. One load at a time refreshes a cached frame; a load that finds it
refreshing waits for that load and serves what it cached, or, once its own deadline passes,
the words the cache holds. A load caches its result only over the entry it read, so it never
replaces a newer entry another load cached meanwhile. `load()` re-checks revocation and working-tier servability on every
call, cache hit or not, so a later takedown refuses the next load.

`load()` states the frame's rendering as `ResolvedFrame.rendering`. A dataset frame's is read
on every load from the version record and the files linked under it, the read the revocation
check makes (`datasets_api.recorded_rendering.rendering_of`, as `StoredRendering.to_record`), since
a later fetch can link a file of another rendering under the same version. A version whose
records do not say has none. A derived frame's is its recipe's `normal_form_hash`, read with
the recipe and cached with it, and a derived frame with no recipe yet, or of a store with no
recipe reader, has none.

`pin()` records a presentation reference and promotes working artifacts to pinned. A derived
frame pins with the root versions its recipe names, and theirs in turn, by the rule
`LiveFrameStore.pin` states: the frame itself must pin, and each root pins while it is held. A
derived frame with no recipe that a reference already names pins alone, and once a recipe
is written for it a pin names the roots again (`LiveFrameStore._recipe_roots`).
Each new pinned copy, the frame's or a root's, is admitted by
`datasets_api.retention_admission.copy_to_pinned` against the store's allowance ledger, and a
copy skipped because its frame is pinned already spends nothing.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
from dfo_frame_grammar import (
    STAT_VAR_COLUMN,
    UNIT_COLUMN,
)
from dfo_frame_grammar.grammar import (
    STAT_VAR_REF_PREFIX,
)
from pydantic import ValidationError
from pydantic.experimental.missing_sentinel import MISSING

from artifact_store import ArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from assembler.bind import role_columns, series_label
from assembler.errors import BindingInvalid
from assembler.frame_view import (
    compute_structural_view,
    continuation_known,
    dimension_columns,
    facet_dimension_names,
    frame_continuation,
)
from assembler.limits import LABEL_MAX, joined_label
from assembler.mint import warned_provenance_refs, warned_refs
from assembler.ports import (
    TRANSIENT_FRAME_REASONS,
    Continuation,
    FrameRef,
    FrameUnavailable,
    LeftOutSeries,
    ResolvedFrame,
    RollingAnchor,
    StructuralView,
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
    ServableVersion,
    SnapshotNotFound,
    SnapshotReader,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    read_servable_version,
)
from datasets_api.allowance import AllowanceExhausted, SharedAllowanceLedger
from datasets_api.codes_snapshot import SpecNotFound
from datasets_api.derived import recipe_citation
from datasets_api.description_record import composed_variables, title_named
from datasets_api.doors.publish import PUBLISH_MAX_ROWS
from datasets_api.manifest import (
    HEALED_FIELDS,
    build_citation,
    citation_healed_from,
    version_document,
)
from datasets_api.manifest_models import CitationModel
from datasets_api.recipe_store import RecipeReader, RecipeRecord
from datasets_api.recorded_rendering import RenderingNotRecorded, rendering_of
from datasets_api.render import servable_frame_location
from datasets_api.retention_admission import (
    CopyClaimUnsettled,
    OpenPinBudgetExhausted,
    PinBudgetExhausted,
    PinUnitHeld,
    copy_to_pinned,
)
from datasets_api.subject_ids import served_subject_ids
from dc_graph import label_or_self_name
from description.citation import figures_processed
from description.compose import version_title
from description.facts import ComposedVariable, DescriptionRecord
from description.steps import Step
from description.steps_text import bound_entities, composed_names, step_output_names
from description.vocabulary import SourceDocumentFacts, Vocabulary, build_vocabulary
from description.world_bank import WORLD_BANK_NAMESPACE
from dfo_platform.caller import Caller
from dfo_platform.executor import submit_with_context
from dfo_platform.timings import phase
from graph_generation import GenerationStamp
from materialise.formats import (
    FRAME1_MEDIA_TYPE,
    STORED_FRAME_DECODE_ERRORS,
    canonical_frame_document,
    read_canonical_frame_table,
    read_frame1,
)
from materialise.graph import GraphReader, GraphReadError
from materialise.provenance_doc import (
    SourceCitation,
    provenance_names,
    read_source_citations,
    unnamed_provenances,
)
from materialise.unit_cells import composed_unit_sides
from origin_catalogue import current_catalogue
from origin_catalogue.context import current_words_build
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
    "DescriptionReader",
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
"""Distinct source refs `_cited_manifest` sends to `read_source_citations` in one `load()` call:
a derived frame's provenance refs in row order, a root frame's stored citation sources in their
order. On a derived frame a ref beyond the cap is cited by its ref alone, as a ref the graph does
not hold is. On a root frame a source beyond the cap keeps its stored fields and documents
nothing."""

_PROVENANCE_NAME_POOL = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="frame-provenance-names"
)
"""Threads, process-wide, that read a load's provenance names beside its label read
(`_label_refs`). A name read still queued when the label read ends is cancelled and made on
the load's own thread, so a busy pool delays no load beyond a serial read."""

_RECHECK = datetime.timedelta(seconds=60)
"""How long a cached frame serves a description record that is not final, or words whose
refresh failed, before a load reads them again: at most one such read a minute for each frame,
however often it loads."""

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


class DescriptionReader(Protocol):
    """Reads versions' description records (`datasets_api.description_store`)."""

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        """Return the version's record, or `None` when it holds none.

        Raises:
            DatasetsStoreUnavailable: The store could not be read.
        """
        ...

    def read_descriptions(
        self, *, versions: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], DescriptionRecord]:
        """Return the records `versions` hold, by `(id, content_version)`, in one read,
        leaving out a version that holds none.

        Raises:
            DatasetsStoreUnavailable: The store could not be read.
        """
        ...


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


def _io_failure(
    *, action: str, digest: str, exc: Exception, retry_after_s: float | None = None
) -> FrameUnavailable:
    """Build a retryable I/O refusal that carries only the exception class."""
    return FrameUnavailable(
        reason="io",
        detail=f"{action} artifact_digest {digest!r} raised {type(exc).__name__}",
        retry_after_s=retry_after_s,
    )


# The errors Datasets refuses the same bytes by (`materialise.formats.STORED_FRAME_DECODE_ERRORS`).
_MALFORMED_CONTENT = STORED_FRAME_DECODE_ERRORS

# How long a client waits before it retries a frame this instance had no memory to decode.
_DECODE_RETRY_AFTER_S = 30


@contextmanager
def _decoding(location: StoredArtifactLocation) -> Iterator[None]:
    """Refuse what the block raises while it reads the content `location` names.

    The bytes are in memory and passed their digest check, so no failure to read them is a failed
    read. Malformed content refuses `missing_rendering`, logged with the traceback: the parsers'
    errors for bytes they cannot parse, and `FrameError` for content outside the grammar's
    forms. Any other exception is a defect, which reaches `internal_error` with a note naming the
    digest. `MemoryError`, which `pa.ArrowMemoryError` subclasses, says nothing about the bytes,
    so it refuses the retryable `io` and names a wait (`_DECODE_RETRY_AFTER_S`), since the next
    try needs memory this instance may not have."""
    try:
        yield
    except MemoryError as exc:
        raise _io_failure(
            action="decoding",
            digest=location.artifact_digest,
            exc=exc,
            retry_after_s=_DECODE_RETRY_AFTER_S,
        ) from exc
    except _MALFORMED_CONTENT as exc:
        logger.error(
            "artifact_digest %r did not decode as a frame: %s: %s",
            location.artifact_digest,
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        raise FrameUnavailable(
            reason="missing_rendering",
            detail=(
                f"artifact_digest {location.artifact_digest!r} did not decode as a frame: "
                f"{type(exc).__name__}"
            ),
        ) from exc
    except (FrameUnavailable, BindingInvalid):
        raise
    except Exception as exc:
        exc.add_note(f"while reading artifact_digest {location.artifact_digest!r}")
        raise


def _structural_view(
    frame: Mapping[str, object],
    *,
    continuation: Continuation | None,
    location: StoredArtifactLocation,
) -> StructuralView:
    """`compute_structural_view` of the frame `location` stores, a function of its stored cells."""
    with _decoding(location):
        return compute_structural_view(frame=frame, continuation=continuation)


def _decode_frame(
    data: bytes, *, location: StoredArtifactLocation, max_rows: int
) -> tuple[Mapping[str, object], int]:
    """The frame1 document `data` holds and the bytes the cache counts for it, `_document_bytes`
    of its rows and columns. Both stored forms read into a canonical frame table and build the
    document from it through `canonical_frame_document`, so their cells are shared alike: a
    canonical frame read as one (`_canonical_table`), bounded by `max_rows` before any row is
    decoded, and a frame1 document through `materialise.formats.read_frame1`, the one reader
    Datasets reads it by, which checks every cell. A frame1 document is not row-bounded: it
    predates canonical frames, and `datasets_api.maintenance.run_backfill_frame1` gives such a
    version a canonical frame. `_decoding` refuses what does not decode."""
    with _decoding(location):
        if location.media_type == FRAME1_MEDIA_TYPE:
            table = read_frame1(data).table
        else:
            table = _canonical_table(data, location=location, max_rows=max_rows)
        doc = canonical_frame_document(table)
    rows = cast("Sequence[Sequence[object]]", doc["rows"])
    return doc, _document_bytes(rows=len(rows), columns=len(cast("Sequence[str]", doc["columns"])))


def _canonical_table(data: bytes, *, location: StoredArtifactLocation, max_rows: int) -> pa.Table:
    """The table of the canonical frame `data` holds, refused past `max_rows` from its footer
    before any row is decoded, then read by `read_canonical_frame_table`, the check Datasets
    reads one by.

    Raises:
        FrameError, pa.ArrowException, OSError, ValueError: As `read_canonical_frame_table`.
        FrameUnavailable: `missing_rendering` past `max_rows`.
    """
    rows = pq.ParquetFile(pa.BufferReader(data)).metadata.num_rows
    if rows > max_rows:
        raise _over_row_bound(location, rows=rows, max_rows=max_rows)
    return read_canonical_frame_table(data)


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


def _provenance_cells(
    *, columns: Sequence[str], rows: Sequence[Sequence[object]]
) -> frozenset[str]:
    """The provenance dcids the `provenance` column of `rows` holds."""
    if _PROVENANCE_COLUMN not in columns:
        return frozenset()
    index = columns.index(_PROVENANCE_COLUMN)
    return frozenset(cast(str, row[index]) for row in rows if row[index] is not None)


def _capped_refs(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    leading_refs: Sequence[str] = (),
    worded: Sequence[str] = (),
    named: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """The refs to label in `rows`, each mapped to the bare dcid a graph read takes: stat_var,
    entity, and provenance refs up to `_LABEL_RESOLUTION_CAP`, then unit dcid refs up to
    `_UNIT_LABEL_CAP`, each in row order. `leading_refs`, then the bare dcids in `worded`,
    then a composed unit's StatVar sides, count under the first cap ahead of the columns' own
    refs, and a composed unit's string sides as unit refs. A ref in `named` is left out."""
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
    # composed variable, the refs inputs' titles and steps' names word and the sides of a
    # composed unit come first, since each labels the whole card and frames hold few of them.
    # A bare dcid is its own key and strips to itself.
    leading = [*leading_refs, *worded]
    leading += [
        cast("Mapping[str, str]", part)["stat_var"]
        for part in unit_parts
        if isinstance(part, Mapping) and "stat_var" in part
    ]
    distinct: dict[str, str] = {
        ref: ref.removeprefix(STAT_VAR_REF_PREFIX) for ref in leading if ref not in named
    }
    for column in ref_columns:
        index = index_by_column[column]
        strip = column == STAT_VAR_COLUMN
        for row in rows:
            ref = row[index]
            if ref is None or ref in distinct or ref in named:
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


def _composed(
    record: DescriptionRecord | None, *, stored: StoredVersion
) -> Sequence[ComposedVariable]:
    """The composed variables of a computed version: its record's `computed.composed` entries,
    or for a version with no record, those its stored lineage holds as `composed_stat_vars`
    (`datasets_api.description_record.composed_variables`), which a record is built from. A
    stored list that does not read as composed variables is logged and names none, so no
    stored lineage fails a load."""
    if record is not None:
        return (record.computed.composed or []) if record.computed is not None else []
    if not isinstance(stored.facts, FrameLineage):
        return []
    try:
        return composed_variables(stored.facts.composed_stat_vars) or []
    except (KeyError, TypeError, ValidationError):
        logger.warning(
            "Stored composed_stat_vars of spec_id %s content_version %s do not read as "
            "composed variables; naming them by ref",
            stored.subject_id,
            stored.content_version,
        )
        return []


def _settled(record: DescriptionRecord) -> bool:
    """Whether `record` holds every member Datasets fills after the write. A computed record
    holding `steps: null` does not, since a later publish fills its steps
    (`datasets_api.version_descriptions.PublishDescriber.settle`)."""
    return record.computed is None or record.steps is not None


def _named_refs(
    record: DescriptionRecord | None, *, composed: Sequence[ComposedVariable]
) -> Iterator[str]:
    """The `dcid:` refs of the variables a computed version names: each step's `stat_var` in
    its record and each of its `composed` entries' variable."""
    if record is not None and record.computed is not None:
        for step in record.steps or ():
            if step.stat_var is not MISSING:
                yield f"{STAT_VAR_REF_PREFIX}{step.stat_var}"
    for entry in composed:
        yield f"{STAT_VAR_REF_PREFIX}{entry.variable}"


def _step_entities(record: DescriptionRecord | None) -> Iterator[str]:
    """The bare dcids of the entities a computed version's steps record on their sides
    (`bound_entities`), which a comparison names its sides by."""
    if record is not None and record.computed is not None:
        for step in record.steps or ():
            yield from bound_entities(step)


def _composed_sides(composed: Sequence[ComposedVariable]) -> list[str]:
    """The `dcid:` refs of the variables on the sides of `composed`, in entry order."""
    return [
        f"{STAT_VAR_REF_PREFIX}{ref}" for entry in composed for ref in (*entry.first, *entry.second)
    ]


def _source_documents(live: Mapping[str, SourceCitation]) -> dict[str, SourceDocumentFacts]:
    """What each documented source's document says, by provenance dcid, as the Datasets
    description door reads it: its publisher, its last download and its next release."""
    return {
        ref: SourceDocumentFacts(
            publisher=doc.source.ref if doc.source is not None else None,
            last_downloaded=doc.last_update,
            next_release=doc.next_expected_update,
        )
        for ref, cited in live.items()
        if (doc := cited.documentation) is not None
    }


def _documented(live: Mapping[str, SourceCitation]) -> bool:
    """Whether every documentation read behind `live` answered."""
    return not any(cited.documentation_read_failed for cited in live.values())


def _joined_within_label_max(first: str, joiner: str, second: str) -> str:
    """Two sides of a variable's name joined within `LABEL_MAX` (`joined_label`), so both
    sides and the joiner stay in the name."""
    return joined_label(first, second, joiner=f" {joiner} ")


def _dataset_rendering(
    frame_ref: FrameRef, *, servable: ServableVersion
) -> Mapping[str, object] | None:
    """The rendering a dataset frame's version record and linked files say it is, or `None`
    when they do not say (`RenderingNotRecorded`)."""
    try:
        rendering = rendering_of(
            spec_id=frame_ref.spec_id,
            version=servable.version,
            renderings=[
                location.rendering for location in (*servable.servable, *servable.aged_out)
            ],
        )
    except RenderingNotRecorded:
        return None
    return rendering.to_record()


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
    """A cached load: the frame bytes' size, the resolved frame, when a load next retries a
    citation read that did not complete (`None` once one has), whether its anchor is final, when
    a load next reads its description record, and when its words were read. A citation read
    that failed, or whose documentation read failed, is retried once `_RECHECK` has passed, and
    the frame serves the citation that read left in between. The anchor of a derived frame is final
    once its recipe has been read, and that of any other frame always. The description record
    is final once a read finds it holding every member Datasets fills later
    (`_description_records`), and always for a store with no description reader. No load reads
    a final record again (`None`); a load reads one that is not final once
    `_RECHECK` has passed since its last read. The frame's words are its labels,
    its anchor's left-out labels, its citation and its source documents, all read from the
    graph and the origin catalogue under the frame's stamps (`LiveFrameStore._words_stale`).
    `words_retry_at` is when a load next tries to refresh stale words whose last refresh
    failed, `None` when none has."""

    size_bytes: int
    resolved: ResolvedFrame
    citation_retry_at: datetime.datetime | None
    anchor_final: bool
    description_recheck_at: datetime.datetime | None
    words_read_at: datetime.datetime
    words_retry_at: datetime.datetime | None = None

    def citation_due(self, now: datetime.datetime) -> bool:
        """Whether a load at `now` retries the citation read."""
        return self.citation_retry_at is not None and now >= self.citation_retry_at

    def description_due(self, now: datetime.datetime) -> bool:
        """Whether a load at `now` reads the description record again."""
        return self.description_recheck_at is not None and now >= self.description_recheck_at


class LiveFrameStore:
    """`graph` supplies entity/provenance labels, and `None` disables them. `spec_reader`
    supplies family labels and the served id a frame ref's revocation check covers; `None`
    disables those labels and narrows the check to the ref's own id. Every other dependency is
    required.
    `recipe_reader` supplies a derived frame's rolling anchor, and `None` loads every frame
    without one. `description_reader` supplies each version's description record, and `None`
    loads every frame without one. `allowances` is the ledger each new pinned copy `pin()` makes
    spends from. It has no default: `None` pins without spending, for an operator's import, and
    must be passed as such.
    `code_index` and `measure_index` are fixed when given. Left out, each read takes the
    indexes the request reads (`get_index`, `get_measure_index`), which can change between
    requests. `now` supplies the clock for expiry checks and reference timestamps.
    `read_budget_seconds` bounds the graph reads of one `load()` from its start, as the module
    docstring describes, for a caller that sets no door deadline. `frame_max_rows` bounds the
    rows of a canonical frame `load()` decodes, since the whole frame is built in memory. A
    larger frame refuses `missing_rendering`, before its object is fetched when the version's
    facts record its row count.
    """

    def __init__(
        self,
        *,
        snapshots: SnapshotReader,
        artifacts: ArtifactStore,
        graph: GraphReader | None,
        reference_writer: ArtifactReferenceWriter,
        allowances: SharedAllowanceLedger | None,
        spec_reader: SpecReader | None = None,
        recipe_reader: RecipeReader | None = None,
        description_reader: DescriptionReader | None = None,
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
        self._description_reader = description_reader
        self._allowances = allowances
        self._given_code_index = code_index
        self._given_measure_index = measure_index
        self._now = now
        self._cache_max_bytes = cache_max_bytes
        self._read_budget_seconds = read_budget_seconds
        self._frame_max_rows = frame_max_rows
        self._cache_lock = threading.Lock()
        self._cache: dict[tuple[str, str], _CachedFrame] = {}
        self._cache_bytes = 0
        self._refreshing: set[tuple[str, str]] = set()
        self._refresh_ended = threading.Condition(self._cache_lock)

    @property
    def _code_index(self) -> CodeIndex:
        return self._given_code_index if self._given_code_index is not None else get_index()

    @property
    def _measure_index(self) -> MeasureIndex:
        if self._given_measure_index is not None:
            return self._given_measure_index
        return get_measure_index()

    # -- FrameStore ---------------------------------------------------

    def load(self, *, frame_ref: FrameRef) -> ResolvedFrame:
        # Recheck revocation and servability on every load. The cache keeps the frame bytes for
        # good and re-reads the frame's words and description record as they go stale.
        deadline = _graph_deadline(self._read_budget_seconds)
        door_deadline = _door_deadline.get()
        if door_deadline is not None:
            deadline = min(deadline, door_deadline)
        servable, canonical = self._resolve_canonical_location(frame_ref)
        if frame_ref.spec_id.startswith(_DERIVED_SPEC_PREFIX):
            return self._cached_or_resolved(
                frame_ref, stored=servable.version, canonical=canonical, deadline=deadline
            )
        rendering = _dataset_rendering(frame_ref, servable=servable)
        frame = self._cached_or_resolved(
            frame_ref,
            stored=servable.version,
            canonical=canonical,
            deadline=deadline,
            rendering=rendering,
        )
        # A file of another rendering linked under the version since the frame was cached. The
        # frame loaded afresh carries `rendering`, so the usual load returns the cached frame
        # itself and copies it only after such a link.
        return frame if frame.rendering == rendering else replace(frame, rendering=rendering)

    def _cached_or_resolved(
        self,
        frame_ref: FrameRef,
        *,
        stored: StoredVersion,
        canonical: StoredArtifactLocation,
        deadline: float,
        rendering: Mapping[str, object] | None = None,
    ) -> ResolvedFrame:
        """The cached frame while it is fresh, else the frame `_resolve` loads or refreshes,
        a frame loaded afresh carrying `rendering`."""
        cache_key = (frame_ref.spec_id, frame_ref.content_version)
        cached = self._cache_get(cache_key)
        if cached is None:
            return self._resolve(
                frame_ref,
                stored=stored,
                canonical=canonical,
                cached=None,
                deadline=deadline,
                rendering=rendering,
            )
        if self._fresh(cached, now=self._now()):
            return cached.resolved
        with self._refresh_turn(cache_key, deadline=deadline) as turn:
            if not turn:
                # Another load's refresh of this frame outlasted this load's budget, so this
                # load serves the words the cache holds, as a failed refresh does.
                return cached.resolved
            # A load that waited takes the entry the refresh before it left.
            cached = self._cache_get(cache_key)
            if cached is not None and self._fresh(cached, now=self._now()):
                return cached.resolved
            return self._resolve(
                frame_ref,
                stored=stored,
                canonical=canonical,
                cached=cached,
                deadline=deadline,
                rendering=rendering,
            )

    def _fresh(self, cached: _CachedFrame, *, now: datetime.datetime) -> bool:
        """Whether a load at `now` serves `cached` as it stands, reading nothing."""
        return (
            not cached.citation_due(now)
            and cached.anchor_final
            and not cached.description_due(now)
            and not self._words_stale(cached, now=now)
        )

    @contextmanager
    def _refresh_turn(self, cache_key: tuple[str, str], *, deadline: float) -> Iterator[bool]:
        """Run one refresh of a cached frame at a time: wait until no other load refreshes the
        frame under `cache_key`, then hold the turn for the body. Yields `False`, holding no
        turn, when the wait reaches the monotonic `deadline` first."""
        with self._refresh_ended:
            turn = self._refresh_ended.wait_for(
                lambda: cache_key not in self._refreshing,
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if turn:
                self._refreshing.add(cache_key)
        if not turn:
            yield False
            return
        try:
            yield True
        finally:
            with self._refresh_ended:
                self._refreshing.discard(cache_key)
                self._refresh_ended.notify_all()

    def _resolve(
        self,
        frame_ref: FrameRef,
        *,
        stored: StoredVersion,
        canonical: StoredArtifactLocation,
        cached: _CachedFrame | None,
        deadline: float,
        rendering: Mapping[str, object] | None,
    ) -> ResolvedFrame:
        """Load the frame afresh when `cached` is `None`, or refresh what `cached` holds that is
        due, and cache the result in place of `cached` if the cache still holds it. A frame
        loaded afresh carries `rendering`, which a derived frame's recipe replaces."""
        cache_key = (frame_ref.spec_id, frame_ref.content_version)
        now = self._now()
        document = version_document(stored)
        if cached is None:
            recorded_rows = _recorded_rows(stored)
            if (
                canonical.media_type != FRAME1_MEDIA_TYPE
                and recorded_rows is not None
                and recorded_rows > self._frame_max_rows
            ):
                raise _over_row_bound(canonical, rows=recorded_rows, max_rows=self._frame_max_rows)
            record, inputs, description_final = self._description_records(frame_ref)
            data = self._fetch_bytes(canonical)
            doc, size_bytes = _decode_frame(data, location=canonical, max_rows=self._frame_max_rows)
            continuation = frame_continuation(manifest=document)
            structural_view = _structural_view(doc, continuation=continuation, location=canonical)
            graph_generation, catalogue_build = self._read_under()
            labels, descriptions, labels_resolved = self._resolve_labels(
                doc,
                record=record,
                composed=_composed(record, stored=stored),
                inputs=inputs,
                warned=warned_refs(manifest=document),
                warned_provenances=warned_provenance_refs(manifest=document),
                deadline=deadline,
            )
            frame = ResolvedFrame(
                frame_ref=frame_ref,
                frame=doc,
                structural_view=structural_view,
                continuation=continuation,
                manifest=document,
                labels=labels,
                descriptions=descriptions,
                retrieved_on=canonical.created_at.astimezone(datetime.UTC).date(),
                description_record=record,
                input_records=inputs,
                graph_generation=graph_generation,
                catalogue_build=catalogue_build,
                rendering=rendering,
            )
            citation_due, citation_retry_at, anchor_final = True, None, False
            recheck_at = (
                None
                if description_final and continuation_known(manifest=document)
                else now + _RECHECK
            )
            words_read_at = now
            refreshing = record_changed = False
        else:
            frame, size_bytes, labels_resolved = cached.resolved, cached.size_bytes, True
            citation_due, citation_retry_at = cached.citation_due(now), cached.citation_retry_at
            anchor_final = cached.anchor_final
            recheck_at, words_read_at = cached.description_recheck_at, cached.words_read_at
            record, inputs = frame.description_record, frame.input_records
            if cached.description_due(now):
                record, inputs, description_final = self._description_records(frame_ref)
                recheck_at = (
                    None
                    if description_final and continuation_known(manifest=document)
                    else now + _RECHECK
                )
            record_changed = (record, inputs) != (frame.description_record, frame.input_records)
            continuation = frame_continuation(manifest=document)
            if continuation != frame.continuation:
                # The late steps of a version published without them replace its stored
                # continuations, in the write that fills its record's steps, and the backfill
                # fills those of a version recorded before versions stored them.
                frame = replace(
                    frame,
                    structural_view=_structural_view(
                        frame.frame, continuation=continuation, location=canonical
                    ),
                    continuation=continuation,
                )
            refreshing = not record_changed and self._words_stale(cached, now=now)
            if record_changed or refreshing:
                # Re-read every word of the cached frame under one stamp: its labels, its
                # anchor's left-out labels, and its citation and source documents, which a
                # record arriving widens to every cited source. The bytes stay cached.
                citation_due, anchor_final = True, False
                words_read_at = now
                graph_generation, catalogue_build = self._read_under()
                labels, descriptions, labels_resolved = self._resolve_labels(
                    frame.frame,
                    record=record,
                    composed=_composed(record, stored=stored),
                    inputs=inputs,
                    warned=warned_refs(manifest=document),
                    warned_provenances=warned_provenance_refs(manifest=document),
                    deadline=deadline,
                )
                if record_changed and not labels_resolved:
                    # The new record's names stand, over the cached labels of the refs the
                    # failed read left unlabelled; the cached stamps keep the words due.
                    labels = {**frame.labels, **labels}
                    descriptions = {**frame.descriptions, **descriptions}
                    graph_generation, catalogue_build = (
                        frame.graph_generation,
                        frame.catalogue_build,
                    )
                    words_read_at = cached.words_read_at
                frame = replace(
                    frame,
                    description_record=record,
                    input_records=inputs,
                    labels=labels,
                    descriptions=descriptions,
                    graph_generation=graph_generation,
                    catalogue_build=catalogue_build,
                )

        if not anchor_final:
            recipe, anchor, anchor_final, anchor_labelled = self._recipe_facts(
                frame, deadline=deadline
            )
            if recipe is not None and frame.replay_key is None:
                # A recipe written after the frame was cached names the roots it cites.
                citation_due = True
            frame = replace(
                frame, replay_key=recipe.replay_key if recipe is not None else None, anchor=anchor
            )
            if recipe is not None:
                frame = replace(frame, rendering={"normal_form_hash": recipe.normal_form_hash})
            labels_resolved = labels_resolved and anchor_labelled
        if citation_due:
            manifest, documents, citation_complete = self._cited_manifest(
                frame, stored_manifest=document, deadline=deadline
            )
            frame = replace(frame, manifest=manifest, source_documents=documents)
            citation_retry_at = None if citation_complete else now + _RECHECK
        if (
            cached is not None
            and refreshing
            and not (labels_resolved and citation_retry_at is None)
        ):
            # A failed refresh of stale words keeps the cached words and their stamps, so no
            # card is composed from fallbacks where good words were held, and tries again once
            # `_RECHECK` has passed.
            self._cache_put(
                cache_key,
                replace(cached, description_recheck_at=recheck_at, words_retry_at=now + _RECHECK),
                expected=cached,
            )
            return cached.resolved
        # A load that read no words keeps a failed refresh's retry time.
        words_retry_at = (
            cached.words_retry_at
            if cached is not None and not (refreshing or record_changed)
            else None
        )
        relabel_failed = record_changed and not labels_resolved
        if relabel_failed:
            # A record that changed under a failed label read is cached with its words, which
            # are tried again, and the record read again, once `_RECHECK` has passed.
            recheck_at = words_retry_at = now + _RECHECK
        if labels_resolved or relabel_failed:
            self._cache_put(
                cache_key,
                _CachedFrame(
                    size_bytes=size_bytes,
                    resolved=frame,
                    citation_retry_at=citation_retry_at,
                    anchor_final=anchor_final,
                    description_recheck_at=recheck_at,
                    words_read_at=words_read_at,
                    words_retry_at=words_retry_at,
                ),
                expected=cached,
            )
        return frame

    def _words_stale(self, cached: _CachedFrame, *, now: datetime.datetime) -> bool:
        """Whether a load at `now` reads the cached frame's words again, as the graph client's
        caches decide for their own entries: once the graph generation or the bound catalogue
        build differs from the one they were read under (`_read_under`, read without I/O), or
        once they are older than that generation's cache lifetime
        (`graph_generation.stamp.GenerationStamp.max_age_s`: a day for a known generation, an
        hour for an unknown one). Words whose last read failed are stale from `words_retry_at`
        on, and not before. A store with no graph reads no words that can change."""
        if self._graph is None:
            return False
        if cached.words_retry_at is not None:
            return now >= cached.words_retry_at
        frame = cached.resolved
        if (frame.graph_generation, frame.catalogue_build) != self._read_under():
            return True
        lifetime = GenerationStamp(token=frame.graph_generation).max_age_s
        return now - cached.words_read_at >= datetime.timedelta(seconds=lifetime)

    def _read_under(self) -> tuple[str | None, str | None]:
        """The graph generation token a graph read starts under now, and the build of the
        catalogue this load is bound to, joined to the bound release (`current_words_build`), or
        `(None, None)` for a store with no graph."""
        if self._graph is None:
            return None, None
        return self._graph.generation_stamp().token, current_words_build()

    def _description_records(
        self, frame_ref: FrameRef
    ) -> tuple[DescriptionRecord | None, tuple[DescriptionRecord, ...], bool]:
        """The version's description record, the records of the inputs it was computed from
        (`_input_records`), and whether that answer is final. A store with no description
        reader has none, finally. A version with no record yet has none, not final, so a later
        load reads again. A record is final once it and each of its inputs' records is settled
        (`_settled`) and it found every input's record. A stored record that fails validation is logged
        as an error and counts as none, finally, since reading it again returns the same record.

        A failed read raises `DatasetsStoreUnavailable`, which the doors answer as a retryable
        503, as they answer the version read every load makes from the same store. A frame
        loaded without the record it holds would mint a card with no description and its
        computed variables named by ref, and the card keeps both.

        Raises:
            DatasetsStoreUnavailable: The record or an input record could not be read.
        """
        if self._description_reader is None:
            return None, (), True
        try:
            record = self._description_reader.read_description(
                id=frame_ref.spec_id, content_version=frame_ref.content_version
            )
        except ValidationError:
            logger.error(
                "Description record for spec_id %s content_version %s failed validation; "
                "serving the frame without one",
                frame_ref.spec_id,
                frame_ref.content_version,
                exc_info=True,
            )
            return None, (), True
        if record is None:
            return None, (), False
        inputs, inputs_final = self._input_records(record, reader=self._description_reader)
        return record, inputs, inputs_final and _settled(record)

    def _input_records(
        self, record: DescriptionRecord, *, reader: DescriptionReader
    ) -> tuple[tuple[DescriptionRecord, ...], bool]:
        """The records of the versions a computed version was computed from, in the order its
        record lists them, read in one batch, and whether that answer is final. An input
        without an id is left out, finally. An input with no record yet is left out, not final,
        since the backfill (`datasets_api.maintenance.run_backfill_descriptions`) can write one
        later. An input record that is not settled (`_settled`) is held, not final, since a
        later publish fills its steps. A stored input record that fails validation is logged as an error and leaves
        none, finally. A failed read raises `DatasetsStoreUnavailable`."""
        if record.computed is None:
            return (), True
        versions = list(
            dict.fromkeys(
                (one.id, one.content_version)
                for one in record.computed.inputs
                if one.id is not None
            )
        )
        if not versions:
            return (), True
        try:
            found = reader.read_descriptions(versions=versions)
        except ValidationError:
            logger.error(
                "An input description record of spec_id %s content_version %s failed "
                "validation; naming its inputs by id",
                record.id,
                record.content_version,
                exc_info=True,
            )
            return (), True
        held = tuple(found[version] for version in versions if version in found)
        return held, len(held) == len(versions) and all(_settled(one) for one in held)

    def _recipe_facts(
        self, frame: ResolvedFrame, *, deadline: float
    ) -> tuple[RecipeRecord | None, RollingAnchor | None, bool, bool]:
        """A derived frame's recipe and the rolling anchor it records, whether that answer is
        final, and whether every label read behind it succeeded.

        A frame that is not derived, and any frame of a store with no recipe reader, has no
        recipe and no anchor, finally. A derived frame with no recipe has none yet, since a recipe can be
        written after the frame is published. Each left-out series is labelled from its
        coordinates (`_left_out_label`), over the frame's labels plus one `_label_refs` read of
        the refs the frame does not label. A label the recipe states is kept."""
        spec_id = frame.frame_ref.spec_id
        if self._recipe_reader is None or not spec_id.startswith(_DERIVED_SPEC_PREFIX):
            return None, None, True, True
        recipe = self._recipe_reader.read_recipe(
            spec_id=spec_id, content_version=frame.frame_ref.content_version
        )
        if recipe is None:
            return None, None, False, True
        body = cast("Mapping[str, object] | None", recipe.body.get("anchor"))
        if body is None:
            return recipe, None, True, True

        left_out = cast("Sequence[Mapping[str, object]]", body["left_out"])
        coordinates = [cast("Mapping[str, str]", one["coordinates"]) for one in left_out]
        columns = tuple(dict.fromkeys(column for one in coordinates for column in one))
        rows = [[one.get(column) for column in columns] for one in coordinates]
        refs = {
            ref: bare
            for ref, bare in _capped_refs(columns=columns, rows=rows).items()
            if ref not in frame.labels
        }
        read, labelled = self._label_refs(
            refs, provenances=_provenance_cells(columns=columns, rows=rows), deadline=deadline
        )
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
        return recipe, anchor, True, labelled

    def _cited_manifest(
        self, frame: ResolvedFrame, *, stored_manifest: Mapping[str, object], deadline: float
    ) -> tuple[Mapping[str, object], dict[str, SourceDocumentFacts], bool]:
        """The manifest with its live citation, what each cited source's document says, and
        whether every citation read completed. One documented citation read
        (`_source_citations`) serves both. A derived frame whose recipe roots cite sources
        (`_roots_citation`) is cited with them, filled as a root frame's stored citation is.
        Any other derived frame cites its provenance refs from the graph. A root frame's stored
        citation has its empty fields filled
        (`datasets_api.manifest.citation_healed_from`) from a read of its distinct sources with
        an empty `HEALED_FIELDS` member, or of all its distinct sources when the frame holds a
        description record, whose card words what each source's document says. The read
        covers the first `_CITATION_READ_CAP` of those; a source beyond the cap stands as
        stored and documents nothing. A root frame with no source to read reads nothing. With no
        graph, or no stored citation on a root frame, the stored manifest stands, documents
        nothing and is complete."""
        frame_ref = frame.frame_ref
        if frame_ref.spec_id.startswith(_DERIVED_SPEC_PREFIX):
            cited = self._roots_citation(frame_ref)
            if cited is None:
                return self._derived_citation(
                    frame.frame,
                    stored_manifest=stored_manifest,
                    frame_ref=frame_ref,
                    steps=()
                    if frame.description_record is None
                    else frame.description_record.known_steps,
                    retrieved=frame.retrieved_on,
                    deadline=deadline,
                )
            stored_manifest = {**stored_manifest, "citation": cited.model_dump(mode="json")}
        if self._graph is None or "citation" not in stored_manifest:
            return stored_manifest, {}, True
        stored_citation = CitationModel.model_validate(stored_manifest["citation"])
        described = frame.description_record is not None
        refs = list(
            dict.fromkeys(
                source.ref
                for source in stored_citation.sources
                if described or any(getattr(source, name) is None for name in HEALED_FIELDS)
            )
        )
        if not refs:
            return stored_manifest, {}, True
        live = self._source_citations(
            refs[:_CITATION_READ_CAP], frame_ref=frame_ref, deadline=deadline
        )
        if live is None:
            return stored_manifest, {}, False
        citation = citation_healed_from(stored_citation, live)
        if citation != stored_citation:
            stored_manifest = {**stored_manifest, "citation": citation.model_dump(mode="json")}
        return stored_manifest, _source_documents(live), _documented(live)

    def _roots_citation(self, frame_ref: FrameRef) -> CitationModel | None:
        """A derived frame's citation as the derived manifest gives it, its recipe roots'
        citations merged (`datasets_api.derived.recipe_citation`), or `None` when the store
        reads no recipes, the frame has no recipe yet, or its roots cite no source."""
        if self._recipe_reader is None:
            return None
        record = self._recipe_reader.read_recipe(
            spec_id=frame_ref.spec_id, content_version=frame_ref.content_version
        )
        citation = recipe_citation(self._snapshots, record)
        return citation if citation.sources else None

    def pin(self, *, frame_ref: FrameRef, component_id: str, caller: Caller | None = None) -> bool:
        """Pin `frame_ref` for `component_id`, and, for a derived frame, each root version its
        recipe names, through every derived root down to the datasets. Returns whether any
        reference was created or reactivated.

        Each frame or root not pinned yet is copied through
        `retention_admission.copy_to_pinned`, spending the pin budget and, for an open `caller`,
        the open callers' budget and its `pin` allowance. A spent budget or allowance refuses
        the call at the copy it stops, and the copies made before it stand with their units.

        The frame `frame_ref` names is what the card draws, so it must pin or the call refuses.
        Its roots pin so the data it was computed from stays fetchable, and the frame is held
        without them. The recipes are read first, so a refusal over them pins nothing. A
        derived frame, or a derived root, whose recipe is not written yet refuses
        `recipe_pending`, since the recipe a publish writes after its frame, or a later recipe
        write, makes a retry pin the roots. A derived frame with no recipe that a reference
        already names pins with no roots (`_recipe_roots`). A store with no recipe reader, and
        a recipe root that names no dataset version, refuse `recipe_unreadable`, a server
        fault no retry clears (`presentation_api.refusals`). Then the frame pins,
        and a refusal there pins nothing either. Then each root pins while it is still held: a
        root that aged out of the working tier, was revoked, or cannot be read is left unpinned
        and logged. A root whose pin fails transiently (`TRANSIENT_FRAME_REASONS`) raises it, and
        a retry pins what is left.

        Raises:
            FrameUnavailable: The frame cannot be pinned, its roots cannot be named, or a
                root's pin failed transiently.
            PinBudgetExhausted: The day's pin budget is spent.
            OpenPinBudgetExhausted: The open callers' pin budget is spent.
            PinUnitHeld: A claim holds the last unit of a limit.
            CopyClaimUnsettled: The copy ran out of lease before it began.
            AllowanceExhausted: `caller`'s `pin` allowance is spent.
        """
        roots = self._root_closure(frame_ref)
        changed = self._pin_frame(frame_ref, component_id=component_id, caller=caller)
        for root in roots:
            try:
                changed = self._pin_frame(root, component_id=component_id, caller=caller) or changed
            except FrameUnavailable as exc:
                if exc.reason in TRANSIENT_FRAME_REASONS:
                    raise
                logger.warning(
                    "root %s at %s of %s at %s left unpinned for %s: %s %s",
                    root.spec_id,
                    root.content_version,
                    frame_ref.spec_id,
                    frame_ref.content_version,
                    component_id,
                    exc.reason,
                    exc.detail,
                )
        return changed

    def _root_closure(self, frame_ref: FrameRef) -> list[FrameRef]:
        """Every root version below `frame_ref`, through the recipes of its derived roots, each
        once, in the order they are first named. None for a dataset frame."""
        roots: list[FrameRef] = []
        pending, seen = [frame_ref], {frame_ref}
        while pending:
            for root in self._recipe_roots(pending.pop()):
                if root not in seen:
                    seen.add(root)
                    roots.append(root)
                    pending.append(root)
        return roots

    def _recipe_roots(self, frame_ref: FrameRef) -> list[FrameRef]:
        """The root versions a derived frame's recipe names (each root's `dataset`), none for a
        dataset frame.

        A derived frame with no recipe that a reference row already names, live or removed
        (`ArtifactReferenceWriter.holds_reference`), has none either. Every pin reads a derived
        frame's recipe before it writes a reference, and the datasets pin door names `spec_`
        ids alone, so only a pin made before recipes were stored references a derived frame
        with no recipe. A share of it draws the frame's stored rows. The recipe is read again
        after the reference is found, so a recipe written between the two reads names the
        roots, and one written later (the recipe write route, `datasets_api.doors.recipe`)
        makes the next pin name them.

        Raises:
            FrameUnavailable: `recipe_pending` for a derived frame with no recipe yet, and
                `recipe_unreadable` for a store with no recipe reader or a root naming no
                dataset version.
        """
        if not frame_ref.spec_id.startswith(_DERIVED_SPEC_PREFIX):
            return []
        if self._recipe_reader is None:
            raise FrameUnavailable(
                reason="recipe_unreadable",
                detail=(
                    f"derived frame {frame_ref.spec_id!r} cannot be pinned with its roots: "
                    "this store reads no recipes"
                ),
            )
        recipe = self._recipe_reader.read_recipe(
            spec_id=frame_ref.spec_id, content_version=frame_ref.content_version
        )
        if recipe is None and self._reference_writer.holds_reference(
            subject_id=frame_ref.spec_id, content_version=frame_ref.content_version
        ):
            # A recipe written after the first read names the roots after all.
            recipe = self._recipe_reader.read_recipe(
                spec_id=frame_ref.spec_id, content_version=frame_ref.content_version
            )
            if recipe is None:
                logger.info(
                    "derived frame %s at %s holds no recipe and a reference already names it, "
                    "so it pins with no roots",
                    frame_ref.spec_id,
                    frame_ref.content_version,
                )
                return []
        if recipe is None:
            raise FrameUnavailable(
                reason="recipe_pending",
                detail=(
                    f"derived frame {frame_ref.spec_id!r} at content_version "
                    f"{frame_ref.content_version!r} holds no recipe yet, so its roots cannot be "
                    "named; retry once its recipe is written"
                ),
            )
        roots: list[FrameRef] = []
        for root in cast("Sequence[Mapping[str, object]]", recipe.body.get("roots") or ()):
            dataset = root.get("dataset")
            pair = cast("Mapping[str, object]", dataset) if isinstance(dataset, Mapping) else {}
            spec_id, content_version = pair.get("spec_id"), pair.get("content_version")
            if not (isinstance(spec_id, str) and isinstance(content_version, str)):
                raise FrameUnavailable(
                    reason="recipe_unreadable",
                    detail=(
                        f"a root of derived frame {frame_ref.spec_id!r} at content_version "
                        f"{frame_ref.content_version!r} names no dataset version, so its roots "
                        "cannot be pinned with it"
                    ),
                )
            roots.append(FrameRef(spec_id=spec_id, content_version=content_version))
        return roots

    def _pin_frame(self, frame_ref: FrameRef, *, component_id: str, caller: Caller | None) -> bool:
        # The read and the write-time recheck honour revocations under the same ids.
        revoked_under = self._revoked_under(frame_ref.spec_id)
        _, canonical = self._resolve_canonical_location(frame_ref, revoked_under=revoked_under)
        if canonical.tier != Tier.pinned.value:
            try:
                copy_to_pinned(
                    self._artifacts,
                    canonical.artifact_digest,
                    source=(canonical.bucket, canonical.object_name),
                    allowances=self._allowances,
                    caller=caller,
                    clock=self._now,
                )
            except (
                PinBudgetExhausted,
                OpenPinBudgetExhausted,
                PinUnitHeld,
                CopyClaimUnsettled,
                AllowanceExhausted,
            ):
                raise
            except ArtifactNotFoundError as exc:
                raise FrameUnavailable(
                    reason="expired",
                    detail=(
                        f"artifact_digest {canonical.artifact_digest!r} aged out of the working "
                        "tier before it could be pinned"
                    ),
                ) from exc
            except Exception as exc:
                # Ledger, copy-side write or inventory failure -- see `_io_failure`.
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
    ) -> tuple[ServableVersion, StoredArtifactLocation]:
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
        self,
        doc: Mapping[str, object],
        *,
        stored_manifest: Mapping[str, object],
        frame_ref: FrameRef,
        steps: Sequence[Step],
        retrieved: datetime.date,
        deadline: float,
    ) -> tuple[Mapping[str, object], dict[str, SourceDocumentFacts], bool]:
        """`stored_manifest` with the citation of a derived frame's distinct `provenance` refs,
        in row order, shaped as a dataset version manifest's `citation`, each source with its
        documentation and the text written for the `retrieved` date; what each source's document
        says; and whether every read behind them completed. The line routes through the DFO as
        the steps its description record knows (`steps`) decide
        (`description.citation.figures_processed`). The read covers the first
        `_CITATION_READ_CAP` refs. With no graph, and for refs beyond the cap, each source is
        named by its ref alone. A failed or overdue citation read leaves the citation out and
        is not complete, so `load` retries it. A failed documentation read leaves only that
        source's `documentation` out and is not complete either. A source the graph documents
        nothing for is complete.
        """
        columns = tuple(cast("Sequence[str]", doc["columns"]))
        refs: list[str] = []
        if _PROVENANCE_COLUMN in columns:
            index = columns.index(_PROVENANCE_COLUMN)
            rows = cast("Sequence[Sequence[object]]", doc["rows"])
            refs = list(dict.fromkeys(cast(str, row[index]) for row in rows if row[index]))
        live: dict[str, SourceCitation] = {}
        if self._graph is not None and refs:
            read = self._source_citations(
                refs[:_CITATION_READ_CAP], frame_ref=frame_ref, deadline=deadline
            )
            if read is None:
                return stored_manifest, {}, False
            live = read
        citation, _ = build_citation(
            provenance_refs=refs,
            source_citations=live,
            retrieved=retrieved,
            processed=figures_processed(version_id=frame_ref.spec_id, steps=steps),
        )
        manifest = {**stored_manifest, "citation": citation.model_dump(mode="json")}
        return manifest, _source_documents(live), _documented(live)

    def _source_citations(
        self, refs: Sequence[str], *, frame_ref: FrameRef, deadline: float
    ) -> dict[str, SourceCitation] | None:
        """Each of `refs` cited with its documentation (`read_source_citations` with
        `documented`), read from the origin catalogue, else the graph, within the absolute
        monotonic `deadline` and timed as the call's `citation` phase, or `None` when the read
        failed, which is logged. The store has a graph."""
        assert self._graph is not None
        try:
            with phase("citation"):
                return read_source_citations(
                    self._graph,
                    refs,
                    catalogue=current_catalogue(),
                    deadline=deadline,
                    documented=True,
                )
        except GraphReadError:
            logger.warning(
                "Citation read failed for spec_id %s content_version %s (%d refs); "
                "serving the frame with its stored citation, if any",
                frame_ref.spec_id,
                frame_ref.content_version,
                len(refs),
                exc_info=True,
            )
            return None

    def _resolve_labels(
        self,
        doc: Mapping[str, object],
        *,
        record: DescriptionRecord | None,
        composed: Sequence[ComposedVariable],
        inputs: Sequence[DescriptionRecord],
        warned: Sequence[str],
        warned_provenances: Sequence[str],
        deadline: float,
    ) -> tuple[dict[str, str], dict[str, str], bool]:
        """Resolve the frame's ref labels (`_resolve_ref_labels`), with the entities, group
        members and series its version's warnings count (`warned`, `assembler.mint.warned_refs`)
        under their `dcid:` refs, since what served nothing, was left out or was withheld has no
        row, the provenances they count (`warned_provenances`), and the refs
        each of `inputs` names in its title (`datasets_api.description_record.title_named`) and the
        entities `record`'s steps record on their sides (`bound_entities`) under their bare dcids,
        then add the name of each variable `record` and `composed` name
        (`_computed_labels`), under the ref its `stat_var` cells hold, and read the description
        of the World Development Indicators variable `record` resolved to
        (`_world_bank_descriptions`). The returned flag says both reads succeeded."""
        worded = set(_step_entities(record))
        if inputs:
            indexed = self._vocabulary(labels={})
            worded = worded.union(
                *(title_named(held, index=self._code_index, indexed=indexed) for held in inputs)
            )
        labels, resolved = self._resolve_ref_labels(
            doc,
            leading_refs=[*_composed_sides(composed), *warned],
            worded=sorted(worded),
            named=frozenset(_named_refs(record, composed=composed)),
            warned_provenances=warned_provenances,
            deadline=deadline,
        )
        labels.update(
            self._computed_labels(record, composed=composed, inputs=inputs, labels=labels)
        )
        descriptions, described = self._world_bank_descriptions(record, deadline=deadline)
        return labels, descriptions, resolved and described

    def _world_bank_descriptions(
        self, record: DescriptionRecord | None, *, deadline: float
    ) -> tuple[dict[str, str], bool]:
        """The graph `description` of the one variable `record`'s definition resolved to, by
        bare dcid, when that variable is a World Development Indicators variable
        (`worldBank/`), read in one graph read that stops at the absolute monotonic
        `deadline`, and whether it succeeded. Such a description defines a card's measure
        (`description.world_bank.world_bank_definition`). A computed version, a version that
        resolved to several variables and any other variable have none read. A store with no
        graph reads nothing."""
        variables = record.variables if record is not None else None
        variable = variables.variable if variables is not None else None
        if self._graph is None or variable is None or not variable.startswith(WORLD_BANK_NAMESPACE):
            return {}, True
        try:
            arcs_by_dcid = self._graph.node_arcs_many(
                (variable,), properties=("description",), deadline=deadline
            )
        except GraphReadError:
            logger.warning(
                "World Development Indicators description read failed for %s; "
                "serving the frame without it",
                variable,
                exc_info=True,
            )
            return {}, False
        arcs = arcs_by_dcid.get(variable)
        values = arcs.get("description", ()) if arcs is not None else ()
        return ({variable: values[0]} if values else {}), True

    def _resolve_ref_labels(
        self,
        doc: Mapping[str, object],
        *,
        leading_refs: Sequence[str],
        named: frozenset[str],
        worded: Sequence[str] = (),
        warned_provenances: Sequence[str] = (),
        deadline: float,
    ) -> tuple[dict[str, str], bool]:
        """Resolve labels for stat_var, entity, and provenance refs, capped at 300 in row
        order, then unit dcid refs, capped at 100 in row order, with the `dcid:` refs of
        `warned_provenances` past both caps, and say whether the graph read succeeded. The
        provenance cells and `warned_provenances` are named as `_label_refs` names a
        provenance. `leading_refs`, then a composed unit's StatVar sides, count as stat_var
        refs ahead of the columns' own refs, and a composed unit's string sides as unit refs.
        A ref in `named` is left out of the read. Each bare dcid in `worded` counts under the
        first cap after `leading_refs`. `dcid:spec_...` family refs use stored
        definitions and `CodeIndex`. Other refs use one graph batch that stops at the absolute
        monotonic `deadline`. Missing readers, lookup/store failures, and over-cap refs remain
        unlabelled without failing the load. A graph failure, the deadline included, preserves
        family labels. A graph failure or a failed family definition read returns `False`, so
        `load` skips the cache.
        """
        columns = tuple(cast("Sequence[str]", doc["columns"]))
        if STAT_VAR_COLUMN not in columns:
            return {}, True
        rows = tuple(cast("Sequence[Sequence[object]]", doc["rows"]))
        refs = _capped_refs(
            columns=columns,
            rows=rows,
            leading_refs=leading_refs,
            worded=worded,
            named=named,
        )
        warned = {ref: ref.removeprefix(STAT_VAR_REF_PREFIX) for ref in warned_provenances}
        refs.update(warned)
        provenances = _provenance_cells(columns=columns, rows=rows) | set(warned.values())
        return self._label_refs(refs, provenances=provenances, deadline=deadline)

    def _computed_labels(
        self,
        record: DescriptionRecord | None,
        *,
        composed: Sequence[ComposedVariable],
        inputs: Sequence[DescriptionRecord],
        labels: Mapping[str, str],
    ) -> dict[str, str]:
        """The label of each variable a computed version names, by `dcid:` ref: the name of
        the output of the step of its record whose `stat_var` it is, else the name its
        `composed` entry (`_composed`) words from the sides' `labels`. The step names read a
        vocabulary of `labels` and this store's indexes (`_vocabulary`), in which each of
        `inputs` is named by its title (`description.compose.version_title`), as a card's
        description names it. A name of two sides fits them within `LABEL_MAX` characters
        (`_joined_within_label_max`), and each name is capitalised and cut to `LABEL_MAX`
        characters. A dataset version names nothing."""
        steps = record.steps if record is not None and record.computed is not None else None
        if not composed and not steps:
            return {}
        bare_labels = {
            ref.removeprefix(STAT_VAR_REF_PREFIX): label for ref, label in labels.items()
        }
        names = composed_names(composed, bare_labels, join=_joined_within_label_max)
        if steps:
            vocabulary = self._vocabulary(labels=bare_labels)
            titles = {
                held.id: title
                for held in inputs
                if (title := version_title(held, vocabulary)) is not None
            }
            if titles:
                vocabulary = self._vocabulary(labels=bare_labels | titles)
            by_step = step_output_names(steps, vocabulary, join=_joined_within_label_max)
            names.update(
                (step.stat_var, by_step[step.id])
                for step in steps
                if step.stat_var is not MISSING and step.id in by_step
            )
        return {
            f"{STAT_VAR_REF_PREFIX}{variable}": (name[:1].upper() + name[1:])[:LABEL_MAX]
            for variable, name in names.items()
        }

    def _vocabulary(self, *, labels: Mapping[str, str]) -> Vocabulary:
        """`labels` and this store's indexes as a vocabulary, whose revision nothing reads."""
        return build_vocabulary(
            index=self._code_index,
            measure_index=self._measure_index,
            labels=labels,
            sources={},
            currency_units={},
            graph_generation=None,
            catalogue_build=None,
        )

    def _label_refs(
        self, refs: Mapping[str, str], *, provenances: frozenset[str], deadline: float
    ) -> tuple[dict[str, str], bool]:
        """Label `refs`, a map from each ref to its bare dcid, and say whether every read
        succeeded, as `_resolve_labels` describes. A ref whose bare dcid is in `provenances` is
        named by `materialise.provenance_doc.provenance_names`, timed as the call's
        `provenance_names` phase. Every other graph ref is labelled by one `node_arcs_many` of
        `name`. When both read the graph, the name read runs beside the label read, so naming
        a provenance the catalogue does not list adds no serial step."""
        catalogue = current_catalogue()
        family_refs: dict[str, str] = {}
        provenance_refs: dict[str, list[str]] = {}
        graph_bare_to_refs: dict[str, list[str]] = {}
        for ref, bare in refs.items():
            if bare.startswith(_FAMILY_REF_PREFIX):
                family_refs[ref] = bare
            elif bare in provenances:
                provenance_refs.setdefault(bare, []).append(ref)
            else:
                graph_bare_to_refs.setdefault(bare, []).append(ref)

        labels: dict[str, str] = {}
        family_read_ok = True
        if family_refs:
            family_labels, family_read_ok = self._resolve_family_labels(family_refs)
            labels.update(family_labels)

        graph = self._graph
        if graph is None:
            return labels, family_read_ok

        unlisted = unnamed_provenances(list(provenance_refs), catalogue=catalogue)

        def name_provenances() -> tuple[dict[str, str], bool]:
            with phase("provenance_names"):
                return provenance_names(
                    graph,
                    list(provenance_refs),
                    catalogue=catalogue,
                    deadline=deadline,
                    unlisted=unlisted,
                )

        if graph_bare_to_refs and unlisted:
            named = submit_with_context(_PROVENANCE_NAME_POOL, name_provenances)
            graph_labels, read_ok = self._graph_labels(graph, graph_bare_to_refs, deadline)
            # A name read still queued behind other loads' runs here instead.
            names, named_ok = name_provenances() if named.cancel() else named.result()
        else:
            graph_labels, read_ok = self._graph_labels(graph, graph_bare_to_refs, deadline)
            names, named_ok = name_provenances() if provenance_refs else ({}, True)
        labels.update(graph_labels)
        labels.update((ref, name) for bare, name in names.items() for ref in provenance_refs[bare])
        return labels, family_read_ok and read_ok and named_ok

    @staticmethod
    def _graph_labels(
        graph: GraphReader, bare_to_refs: Mapping[str, list[str]], deadline: float
    ) -> tuple[dict[str, str], bool]:
        """Each ref of `bare_to_refs` labelled from one `node_arcs_many` of its bare dcid's
        `name`, and whether the read succeeded."""
        if not bare_to_refs:
            return {}, True
        try:
            arcs_by_bare = graph.node_arcs_many(
                list(bare_to_refs), properties=("name",), deadline=deadline
            )
        except GraphReadError:
            return {}, False
        labels: dict[str, str] = {}
        for bare, arcs in arcs_by_bare.items():
            label = None if arcs is None else label_or_self_name(arcs.get("name", ()), bare)
            if label is None or bare not in bare_to_refs:
                continue
            for ref in bare_to_refs[bare]:
                labels[ref] = label
        return labels, True

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
        self, cache_key: tuple[str, str], entry: _CachedFrame, *, expected: _CachedFrame | None
    ) -> None:
        """Byte-bounded insertion-order cache keyed by `(spec_id, content_version)`. Frame refs
        identify immutable bytes, so entries need no expiry; `load` refreshes an entry's words
        and record in place (`_words_stale`, `_CachedFrame.description_due`). `entry` replaces
        the entry under `cache_key` only while that entry is `expected`, the one its load read
        (`None` for a key it found absent), so a load never overwrites a newer entry another
        load cached meanwhile. Eviction follows insertion order and can drop hot entries. Keep
        one oversized frame. Adjust byte counts on replacement.
        """
        with self._cache_lock:
            prior = self._cache.get(cache_key)
            if prior is not expected:
                return
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
