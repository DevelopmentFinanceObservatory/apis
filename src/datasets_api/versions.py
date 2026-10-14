"""Version-record and subject-head interfaces used by the Datasets API.

`VersionRecordWriter` stores a version's immutable facts (`dataset_store.snapshot.VersionFacts`),
and `write_version` stores everything one
serve writes (`VersionWrite`) in one transaction. A record's `VersionDescriber` writes its
description record in the transaction that writes its row. `VersionLister` lists a
subject's stored versions. `HeadWriter` records each materialisation run
(`MaterialisationRecord`) and moves the current content version, and `HeadReader`
reads it. An operation run fences its head advance on its claim (`HeadFence`), and a write the
fence refuses raises `HeadFenceLost`. The default null writers record or advance nothing.
`datasets_surface` injects live implementations.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import JsonValue

from artifact_store.inventory import InventoryEntry
from dataset_store.snapshot import StoredVersion, VersionFacts, VersionState
from description.facts import Basis, DescriptionRecord
from description.stored import FieldPath

logger = logging.getLogger(__name__)

# Logs a description record that could not be built, with the version's id and content version.
# The version is stored without a record, which `backfill-descriptions` fills once fixed.
DESCRIPTION_BUILD_FAILED_MARKER = "description_record_build_failed"

# The entry point a materialisation ran through, recorded on its `MaterialisationRecord` and
# logged on its `datasets.timings` line.
Trigger = Literal["mint", "get", "options", "refresh", "worker"]


@dataclass(frozen=True)
class ArtifactRecord:
    """Rendering metadata for one artifact in a version record.

    `artifact_digest` joins to stored bytes and location.
    `rendering` and `media_type` preserve how the manifest entry was produced.
    """

    artifact_digest: str
    rendering: Mapping[str, object]
    media_type: str


@dataclass(frozen=True)
class VersionRecord:
    subject_id: str
    subject_kind: str  # "spec"
    content_version: str
    schema_version: int
    facts: VersionFacts
    # A manifest's download links derive from the subject, version and digest, so writers store
    # artifact metadata without them.
    artifacts: tuple[ArtifactRecord, ...] = ()
    # The raw version a full collapse was rendered from. Other records use `None`.
    base_content_version: str | None = None
    # The frame grammar version that minted `content_version`. `None` for a derived frame,
    # whose token another service minted.
    frame_grammar_version: str | None = None
    # The hash of the version's frame (`dfo_frame_grammar.frame_sha256`), which every writer
    # holds when it records the version.
    frame_sha256: str | None = None
    # Builds and settles the version's description record in the transaction that writes the
    # version row. `None` writes no record.
    describer: VersionDescriber | None = field(default=None, compare=False)


@dataclass(frozen=True)
class LateFields:
    """Members of a stored description record to fill once, and what they arrived by.

    `derived` members follow the fill: the write that fills a `values` member sets them too,
    replacing what they held, and a write that fills nothing leaves them as they are.
    `continuations` follows the fill the same way on the version row: a published frame's
    continuations (`dataset_store.snapshot.FrameLineage.continuations`) derived from the steps
    the fill brings, replacing those its publish stored; `None` leaves them.
    """

    values: Mapping[FieldPath, JsonValue]
    basis: Basis
    derived: Mapping[FieldPath, JsonValue] = field(default_factory=dict)
    continuations: Sequence[Mapping[str, object]] | None = None


class DescriptionDiverges(Exception):
    """A write's evidence states facts of a version that differ from those its description
    record holds. `pointer` names the request member that carried them."""

    def __init__(self, message: str, *, pointer: str) -> None:
        self.pointer = pointer
        super().__init__(message)


def log_description_build_failure(version: VersionRecord) -> None:
    """Log, at error level with the exception in hand, that `version`'s description record
    could not be built."""
    logger.exception(
        "%s id=%s content_version=%s",
        DESCRIPTION_BUILD_FAILED_MARKER,
        version.subject_id,
        version.content_version,
    )


class VersionDescriber(Protocol):
    """How a version write records the version's `DescriptionRecord`.

    The writer calls it inside the transaction that writes the version row. A version that
    holds no record gets the one `describe` builds from the version as stored, so a write that
    found the row already recorded describes the stored facts, never its own. A version that
    holds one gets the late fields `settle` returns. Both run with no I/O of their own. Any
    exception either raises, other than `DescriptionDiverges`, is a defect: the writer logs it
    (`log_description_build_failure`) and stores the version without touching its record.
    """

    def describe(
        self, version: VersionRecord, *, base: DescriptionRecord | None
    ) -> DescriptionRecord | None:
        """The record of `version`, whose facts and frame hash are the stored ones. `base` is
        the stored record of the version `version.base_content_version` names, `None` when it
        names none or that version holds no record. `None` when no record can be built."""
        ...

    def settle(self, stored: DescriptionRecord, *, version: VersionRecord) -> LateFields | None:
        """The late fields this write fills in `stored`, or `None` for none.

        Raises:
            DescriptionDiverges: this write's evidence contradicts `stored`.
        """
        ...


# The facts a rendered artifact carries. An artifact links to a version only when the stored
# record holds the same values, so every linked artifact dates and cites as its record.
RENDERED_FACT_FIELDS = ("prepared_at", "citation")


@dataclass(frozen=True)
class VersionWrite:
    """Everything one serve writes for a version, for `VersionRecordWriter.write_version`.

    `record` is inserted once, and its `artifacts` (the canonical frame and series statistics, which
    depend on no option and carry no fact a render reads) are linked whoever inserted it. `rendered`
    are artifacts rendered from `record`'s `RENDERED_FACT_FIELDS`, and `collapsed` the full-collapse
    version recorded from that render with its own artifacts. Both are written only when the stored
    record holds those fields. That is so when this call inserts it, or when `record` carries the
    stored facts it was read from. `inventory` holds the inventory row each written artifact owes.
    """

    record: VersionRecord
    inventory: tuple[InventoryEntry, ...] = ()
    rendered: tuple[ArtifactRecord, ...] = ()
    collapsed: VersionRecord | None = None


@dataclass(frozen=True)
class WrittenVersion:
    """What a `write_version` call stored.

    `rendered` says whether the rendered artifacts and the collapsed version were written.
    `state` is the version the response serves, read in the same transaction. It is the collapsed
    version when written, else the record. `created` says this call inserted that version, so
    `state` holds this call's body. Otherwise it holds the stored one.
    """

    created: bool
    rendered: bool
    state: VersionState


class VersionRecordWriter(Protocol):
    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        """Insert the immutable record once. Return `True` only when this call creates the
        row, including under concurrent first writes.
        """
        ...

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> str:
        """Add one rendering, idempotent on `(subject_id, content_version, rendering_key)`.
        Return the digest linked under `artifact`'s rendering key after the call. That is
        `artifact`'s, or the one a link kept."""
        ...

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        """Store `write` in one transaction. It writes the inventory rows, the record and its
        links, then the rendered links and collapsed version when the stored record holds the
        rendered body fields, and reads back the version the response serves."""
        ...


class NullVersionRecordWriter:
    """Records nothing: the injection point / interface other `VersionRecordWriter`
    implementations plug into (see `datasets_surface`'s `version_writer`
    parameter).
    """

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        # No stored row can win a concurrent insert race.
        return True

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> str:
        return artifact.artifact_digest

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        # Every write creates, and nothing is linked for a response to list.
        served = write.collapsed if write.collapsed is not None else write.record
        return WrittenVersion(created=True, rendered=True, state=unlinked_state(served))


def unlinked_state(record: VersionRecord) -> VersionState:
    """`record` as a stored version with no revocation and no linked location."""
    return VersionState(
        version=StoredVersion(
            subject_id=record.subject_id,
            subject_kind=record.subject_kind,
            content_version=record.content_version,
            schema_version=record.schema_version,
            facts=record.facts,
            base_content_version=record.base_content_version,
            frame_sha256=record.frame_sha256,
        ),
        revocation=None,
        locations=[],
    )


NULL_VERSION_RECORD_WRITER = NullVersionRecordWriter()


@dataclass(frozen=True)
class ListedVersion:
    """One stored version of a subject. `base_content_version` names the raw version a derived
    version (a full groupby collapse, a party-totals cut, or both) was rendered from, and is
    `None` on every other version. `renderings` holds the rendering of each file linked under
    the version, in the order `SnapshotReader.read_artifact_locations` returns them. A derived
    version's options are read from them (`datasets_api.recorded_rendering`).
    """

    content_version: str
    base_content_version: str | None
    renderings: tuple[Mapping[str, object], ...] = ()


class VersionLister(Protocol):
    def list_versions(self, subject_id: str) -> list[ListedVersion]:
        """Return the subject's stored versions, newest first. Empty when it holds none."""
        ...


@dataclass(frozen=True)
class MaterialisationRecord:
    """One materialisation run, stored as a `datasets.materialisation` row.

    `dependencies` is the run's fingerprint (`materialise.dependencies.Dependencies.as_json`).
    `selection` holds the run's `trace.facets` under `facets`, and `conversion` its
    `trace.conversion` entries under `entries`, which name
    the rate and deflator table row digests, or `None` for an unconverted run.
    `code_index_stamp` is the code index's content hash, and `graph_generation` the graph
    service's serving revision, `None` when unknown. The store stamps `materialised_at`.
    `materialiser_build` names the release that ran it and is never empty.
    """

    subject_id: str
    content_version: str
    trigger: Trigger
    materialiser_build: str
    frame_grammar_version: str
    code_index_stamp: str
    origin_catalogue_build_id: str
    coverage_stamp: str | None
    graph_generation: str | None
    dependencies: Mapping[str, object]
    selection: Mapping[str, object]
    conversion: Mapping[str, object] | None
    trace_id: str | None

    def __post_init__(self) -> None:
        assert self.materialiser_build, "a materialisation run names the build that ran it"


@dataclass(frozen=True)
class SubjectHead:
    """`confirmed_at` advances when a freshness check confirms the head or a materialisation
    advances it. `materialised_at` advances only when materialisation writes a version.
    Maximum-age checks use it.

    `materialisation_id` names the `datasets.materialisation` row of the run that produced
    the head's version, and `dependencies` is that run's fingerprint as stored, which
    freshness compares. Both are `None` on a head written before runs were recorded, and
    `dependencies` is `None` on a head whose run produced another version.
    """

    subject_id: str
    content_version: str
    confirmed_at: datetime
    materialised_at: datetime
    materialisation_id: int | None
    dependencies: Mapping[str, object] | None


@dataclass(frozen=True)
class HeadFence:
    """The operation claim a worker run advances the head under: the `datasets.operation` row
    `operation_id` must still be `running` at `attempt_count` `attempt`."""

    operation_id: str
    attempt: int


class HeadFenceLost(Exception):
    """A fenced head advance found its operation claimed by another attempt or no longer
    `running`, and wrote neither the run's row nor the head."""

    def __init__(self, fence: HeadFence) -> None:
        super().__init__(
            f"operation {fence.operation_id!r} is no longer running at attempt {fence.attempt}"
        )
        self.fence = fence


class HeadWriter(Protocol):
    def advance_head(
        self,
        *,
        materialisation: MaterialisationRecord,
        expected_materialisation_id: int | None,
        fence: HeadFence | None = None,
    ) -> SubjectHead | None:
        """Insert the run's row and, in the same transaction, move the subject's head to the
        run's version only if the head still names the run `expected_materialisation_id`.

        The caller passes the `materialisation_id` of the head read before materialisation,
        `None` for no head. `None` matches an empty row or a head that names no run, so of two
        writers with one expectation only the first applies, whatever versions they produce.
        An applied write sets both timestamps to the run's `materialised_at`, names the run, stores
        its dependencies, and returns the stored head. A mismatch keeps the run's row, leaves the
        head unchanged, and returns `None`. The caller rereads the head to find the winner. A
        failure writes neither.

        A `fence` makes the same transaction first lock the fence's operation row and check it
        is still `running` at the fence's attempt. When it is not, the call raises
        `HeadFenceLost` and writes neither. A claim that races the write either commits first,
        and the write refuses, or waits until the write commits. `None` checks nothing.
        """
        ...

    def confirm_head(
        self,
        *,
        subject_id: str,
        content_version: str,
    ) -> SubjectHead | None:
        """Advance `confirmed_at` only if the stored version matches. Leave the version,
        `materialised_at`, and dependencies unchanged. Return `None` on a mismatch or missing row.
        """
        ...


class HeadReader(Protocol):
    def read_head(self, subject_id: str) -> SubjectHead | None:
        """`None` when no head has ever been written for this subject."""
        ...


class HeadStore(HeadReader, HeadWriter, Protocol):
    """Before deciding a new version, writers read the head and pass it to `advance_head`.
    The definition door also reads heads. `LiveHeadStore` implements both protocols.
    """


class NullHeadWriter:
    """No-op `HeadStore` for surfaces without persistence. `advance_head` accepts any expected
    run and any fence, and returns a head stamped with the current time that names no run.
    `confirm_head` and `read_head` return `None`.
    """

    def advance_head(
        self,
        *,
        materialisation: MaterialisationRecord,
        expected_materialisation_id: int | None,
        fence: HeadFence | None = None,
    ) -> SubjectHead:
        now = datetime.now(UTC)
        return SubjectHead(
            subject_id=materialisation.subject_id,
            content_version=materialisation.content_version,
            confirmed_at=now,
            materialised_at=now,
            materialisation_id=None,
            dependencies=materialisation.dependencies,
        )

    def confirm_head(self, *, subject_id: str, content_version: str) -> SubjectHead | None:
        return None

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return None
