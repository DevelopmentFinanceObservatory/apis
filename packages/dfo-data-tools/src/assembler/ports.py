"""Protocols for frame access, revision writes, catalogue releases, and a card's words.

Live frame and revision stores live in `presentation_store`.
The baked release registry is in `assembler.catalogue`."""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Literal, Protocol

from assembler.catalogue import CatalogueRelease
from description.facts import DescriptionRecord
from description.vocabulary import SourceDocumentFacts, Vocabulary
from materialise.continued_series import ContinuedSeries

__all__ = [
    "CompositionAppend",
    "CompositionMembership",
    "DimensionView",
    "DraftNotFound",
    "EditConflict",
    "FrameRef",
    "FrameStore",
    "FrameUnavailable",
    "LeftOutSeries",
    "PriorCreate",
    "ReleaseRegistry",
    "ResolvedFrame",
    "RevisionRecord",
    "RevisionStore",
    "RollingAnchor",
    "StructuralView",
    "VocabularySource",
    "WriteContext",
    "WriteSubjectKind",
    "WriteTokenInUse",
]

type WriteSubjectKind = Literal["component", "composition"]


@dataclass(frozen=True)
class FrameRef:
    """Names a Datasets definition or Analysis Toolkit op-spec and its materialised result.

    An `ops_`/`ocv_` pair loads after analysis-mcp publishes it.
    Before publication, `FrameStore.load` raises `FrameUnavailable(reason="ops_unsupported")`."""

    spec_id: str
    content_version: str


@dataclass(frozen=True)
class DimensionView:
    """A frame dimension's capped values, full count, and plural status."""

    name: str
    values: tuple[str, ...]
    total_count: int
    truncated: bool
    plural: bool


@dataclass(frozen=True)
class StructuralView:
    # Computed once at frame load for frame views and unbound-facet disclosure.
    dimensions: tuple[DimensionView, ...]
    period_grain: str
    coverage: tuple[str, str] | None
    # The one span every row shares, or `None` for a frame of points.
    span: str | None
    series_count: int


@dataclass(frozen=True)
class LeftOutSeries:
    """A series with no row at a rolling anchor's period.

    `root` is the index of the population the series belongs to, in the order of the
    recipe's populations: one per root dataset. `coordinates` maps that dataset's columns
    to the series' values, and `label` is its display name."""

    root: int
    coordinates: Mapping[str, str]
    label: str


@dataclass(frozen=True)
class NamedSeries:
    """A series a card names without drawing it: its display `label`, and the `coordinates`
    that identify it, each dimension it is known by mapped to its value."""

    label: str
    coordinates: Mapping[str, str]


@dataclass(frozen=True)
class RollingAnchor:
    """The period "latest" names on a derived frame, as its recipe records it.

    `rule` names the anchor rule that chose `period`, `dfo_frame_grammar.rolling.ANCHOR_RULE`
    when the recipe was written. `left_out` lists the series that rule left without a row at
    `period`, in the recipe's order."""

    period: str
    rule: str
    left_out: tuple[LeftOutSeries, ...] = ()


@dataclass(frozen=True)
class Continuation:
    """How a frame's series continue past their last period with the same source's
    estimates: each entry names the continuing method and the method and source of the series
    it continues (`materialise.continued_series.ContinuedSeries`). A row an entry admits
    extends the series that holds the same cells under the method it continues."""

    series: tuple[ContinuedSeries, ...]


@dataclass(frozen=True)
class ResolvedFrame:
    """`manifest` is the stored Datasets version document the frame was minted under
    (`datasets_api.manifest.version_document`). `retrieved_on` is the UTC date the platform first
    stored this frame's canonical frame, which is when the values it holds were read from their
    sources; a minted component's `provenance.sources[]` states it as `retrieved`. `anchor` is the
    rolling anchor a derived frame's recipe records, and `None` for a dataset frame or a recipe
    without one. An intent's relative periods count from `anchor.period` when it is set, and from
    the newest period half of the component's own series hold otherwise (`assembler.rolling`).
    `description_record` is the version's description record, and `None` for a version that
    holds none yet or whose record could not be read. `replay_key` is the key a derived frame's
    recipe replays it under, and `None` for a dataset frame or a frame with no recipe yet.
    `graph_generation` is the graph generation token the store's label read started under, and
    `catalogue_build` the build id of the origin catalogue that load was bound to, with the
    bound release (`origin_catalogue.context.current_words_build`);
    both are `None` for a store that reads no graph, and `graph_generation` is `None` when the
    generation was unknown. A card's vocabulary revision hashes both
    (`assembler.mint.card_description`).
    `source_documents` holds what each cited source's document says today, by provenance dcid,
    as the store's citation read found it: the publisher, the last download and the next
    release. A source the read did not document has no entry. `input_records` holds the
    records of the versions a computed version was computed from, in the order its record lists
    them, read with `description_record`; an input without an id or a record has none, and the
    tuple is empty for a dataset version or where the read failed. `labels` then holds, by bare
    dcid, the labels of the refs each input's title names, so a card names each input by its
    title (`description.compose.version_title`). `continuation` is the continuation
    `assembler.frame_view.frame_continuation` reads from `manifest`, the
    one `structural_view` counts series under and selectors resolve under, and `None` when
    nothing continues the frame's series. `descriptions` holds the graph `description` of the
    one variable the version's definition resolved to, by bare dcid, when that variable is a
    World Development Indicators variable, and defines the version's measure
    (`description.world_bank`); it is empty otherwise.
    """

    frame_ref: FrameRef
    frame: Mapping[str, object]
    structural_view: StructuralView
    manifest: Mapping[str, object]
    labels: Mapping[str, str]
    retrieved_on: datetime.date
    anchor: RollingAnchor | None = None
    description_record: DescriptionRecord | None = None
    replay_key: str | None = None
    graph_generation: str | None = None
    catalogue_build: str | None = None
    source_documents: Mapping[str, SourceDocumentFacts] = field(default_factory=dict)
    input_records: Sequence[DescriptionRecord] = ()
    continuation: Continuation | None = None
    descriptions: Mapping[str, str] = field(default_factory=dict)


class FrameUnavailable(Exception):
    # reason: "revoked" | "expired" | "missing_rendering" | "digest_mismatch" |
    # "ops_unsupported" | "quarantined" | "io" | "unknown"
    # Publishing an `ops_`/`ocv_` pair can make `ops_unsupported` succeed on retry.
    def __init__(self, *, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


class FrameStore(Protocol):
    def load(self, *, frame_ref: FrameRef) -> ResolvedFrame: ...
    # `True` means created or reactivated. `False` means already active.
    # `None` leaves the change status unknown.
    def pin(self, *, frame_ref: FrameRef, component_id: str) -> bool | None: ...


@dataclass(frozen=True)
class RevisionRecord:
    component_id: str
    revision: int
    stored_intent: Mapping[str, object]
    resolved_payload: Mapping[str, object]
    catalog_version: str
    schema_version: int
    # The minting release's formatter profile. Reads format the fact summary from it.
    formatter_profile: Mapping[str, object]
    # One id/coordinates entry per resolved series, stored outside the validated envelope.
    # `None` on older revisions reads back as empty coordinates for each series.
    series_coordinates: Sequence[Mapping[str, object]] | None = None
    # The instant the store recorded the revision, which never changes. `None` only on a record
    # the write path builds before the store stamps it. Every read returns it set.
    created_at: datetime.datetime | None = None


@dataclass(frozen=True)
class CompositionMembership:
    """A component's composition and its current head revision.

    Membership is fixed at creation."""

    composition_id: str
    revision: int


class DraftNotFound(Exception):
    """No component or composition exists at this id. Maps to `not_found`."""

    def __init__(self, *, subject_kind: str, subject_id: str) -> None:
        self.subject_kind = subject_kind  # "component" | "composition"
        self.subject_id = subject_id
        super().__init__(f"no {subject_kind} exists at id {subject_id!r}")


class EditConflict(Exception): ...  # -> edit_conflict (vanished component/revision mid-flight)


class WriteTokenInUse(Exception):
    """A create's `write_token_sha256` already names a committed component.

    Raised by `begin_write` on a create whose digest another create stored first, including one
    that committed while this create waited on it. The write rolls back. `prior_create` then
    names the component that holds the digest."""

    def __init__(self) -> None:
        super().__init__("a component already stores this write-token digest")


@dataclass(frozen=True)
class PriorCreate:
    """The component a write-token digest names, and the request hash its create stored.

    `composition_id` is the composition the component joined at create, or `None`.
    `create_request_sha256` is `None` when the create stored no request hash."""

    component_id: str
    composition_id: str | None
    create_request_sha256: bytes | None


class WriteContext(Protocol):
    # Entry locks the stored-intent row (SELECT ... FOR UPDATE).
    # `append` returns the new composition revision, or `None` if untouched.
    @property
    def component_id(self) -> str: ...  # echoes begin_write's own component_id, create or revise

    @property
    def composition_id(
        self,
    ) -> str | None: ...  # resolved id ("new" already minted), None when untouched

    @property
    def current(self) -> RevisionRecord | None: ...  # None on create

    def append(
        self, *, record: RevisionRecord, composition_entry: CompositionAppend | None
    ) -> (
        int | None
    ): ...  # returns the minted composition revision, None when no composition touched


class RevisionStore(Protocol):
    """Holding a `cmp_` or `cps_` id grants read. The store keeps each draft's write-token
    digest and hands it back, and the pipeline compares it, so the store checks no caller."""

    def read_current(self, *, component_id: str) -> RevisionRecord | None: ...

    # An unknown `component_id` raises `DraftNotFound`.

    def read_revision(self, *, component_id: str, revision: int) -> RevisionRecord | None: ...

    # An unknown `component_id` raises `DraftNotFound`. `None` means that revision is missing.

    def allocate_component_id(self) -> str: ...

    # Called before create's transaction so frame pinning runs outside the row lock.
    # The store owns the `cmp_` + ULID format. Allocation does no I/O or locking.

    def composition_of(self, *, component_id: str) -> CompositionMembership | None: ...

    # An unknown `component_id` raises `DraftNotFound`. `None` means the component has no
    # composition.

    def write_token_digest(
        self, *, subject_kind: WriteSubjectKind, subject_id: str
    ) -> bytes | None: ...

    # The SHA-256 digest of the subject's write token, stored once at creation and never
    # changed. `None` means the subject predates write tokens. An unknown id raises
    # `DraftNotFound`.

    def prior_create(self, *, write_token_sha256: bytes) -> PriorCreate | None: ...

    # The component whose create stored this write-token digest, read in its own transaction,
    # so it sees any create committed before the call. `None` means no component stores it.

    def begin_write(
        self,
        *,
        component_id: str,
        is_create: bool,
        composition_id: str | None,
        write_token_sha256: bytes | None = None,
        composition_write_token_sha256: bytes | None = None,
        create_request_sha256: bytes | None = None,
    ) -> AbstractContextManager[WriteContext]: ...

    # `component_id` is allocated before create. `is_create` selects insert or revise.
    # `composition_id="new"` mints `cps_`. An unknown composition id, or a revise of an
    # unknown component id, raises `DraftNotFound`. A create stores `write_token_sha256` on
    # the new component, and `composition_id="new"` stores `composition_write_token_sha256`
    # on the new composition, and `create_request_sha256` on the new component. A revise
    # ignores all three. An omitted write-token digest stores NULL, which leaves the draft
    # read-only. A component write-token digest is unique: a create whose digest another
    # component already stores raises `WriteTokenInUse`, and one whose digest an uncommitted
    # create holds waits for that create to finish first.
    # Locks acquire composition before component.


@dataclass(frozen=True)
class CompositionAppend:
    composition_id: str  # resolved id (post-"new"-minting)
    actor: str  # "model" (provisional, no caller identity exists)


class ReleaseRegistry(Protocol):
    def pinned(self) -> CatalogueRelease: ...  # this deployment's mint-side release


class VocabularySource(Protocol):
    """Today's words a card's description is worded from.

    `current` gives the words no frame holds: the code index, the measure index and the
    publisher short names. The assembler adds the labels and source documents of the frame a
    card draws, and revises the vocabulary's revision with the graph generation and catalogue
    build they were read under (`assembler.mint.card_description`)."""

    def current(self) -> Vocabulary: ...
