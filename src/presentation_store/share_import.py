"""Import the website's shares from the JSON lines `scripts/export-shares.ts` writes.

Each line is one share as the website's `docs/share-export.md` defines it: `id`, `kind`, `mode`,
`owner`, `created_at`, `title`, `slots` and `payload`. `import_shares` stores each under its own
id with its kind, mode, title, owner, payload and `created_at`, at revision 1. The export's
`slots` are listed in display order and the payload's held cards name them by 0-based index;
the share's slots are numbered from 1, so the export's slot `i` is slot `i + 1`.

A slot names a card revision this service holds and the frame it shows. A slot whose `frame` is
null, a live card the website stored with no frame of its own, draws the frame its card
revision is stored on. Before storing a share the import pins what it shows, live or snapshot:
the frame each card revision draws, as a share's create does (`pin_card_frames`), and each
slot's frame, as a snapshot conversion does, so no slot's frame leaves the working tier. A pin
that fails refuses the line.

A signed-in user's share (`owner` set) gets a fresh `shw_` write token. The import hands
`{id, owner, write_token}` to `write_token`, which must make the line durable before it returns,
after the share's rows are written and before they commit (`ShareStore.import_share`), so a
stored token is always in the report. A crash between the two leaves a line naming a share that
was never stored, which a re-run imports with a later line. The website's token import keeps
the last line for each id. A guest's share (`owner` null) is stored with no token, so nobody
can change or delete it.

The import is idempotent. A share whose id already holds a share of the same kind and
`created_at` was imported by an earlier run. It is counted as unchanged, gets no new token and
writes no report line, so a re-run changes nothing. A line the import cannot place is refused
and counted, and the others carry on. The refused lines are those that are not a share as the
export defines it, those whose payload is over `MAX_SHARE_PAYLOAD_BYTES` as a create's would
be, a live conversation, a search or conversation share with slots, a slot naming a card
revision this service does not hold or one drawing no frame, a frame that cannot be pinned, and
an id that holds another share. The website's export puts no note in a slot (a note is a text card,
which has no slot), so a slot drawing no frame is not a share the export writes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

from artifact_store import ArtifactStore
from assembler.errors import FrameUnavailable, RevisionNotFound
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import DraftNotFound, EditConflict, FrameRef, FrameStore, RevisionStore
from assembler.write_token import SHARE_WRITE_TOKEN_PREFIX, mint_write_token, write_token_digest
from dataset_store import DatasetStore, LiveDatasetStore
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.reference_store import LiveArtifactReferenceWriter
from presentation_store.card_pins import pin_card_frames
from presentation_store.frame_store import LiveFrameStore
from presentation_store.ids import SHARE_ID_PATTERN
from presentation_store.provider import close_shared_pool, shared_pool
from presentation_store.shares import (
    ImportedShare,
    LiveShareStore,
    ShareCard,
    ShareIdTaken,
    ShareStore,
    payload_refusal,
)
from presentation_store.store import LivePresentationStore

__all__ = ["ImportCounts", "import_share_file", "import_shares"]

type _ContentVersion = Annotated[str, StringConstraints(min_length=1)]


class _ExportFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^(spec|ops)_[A-Za-z0-9]+$")
    content_version: _ContentVersion


class _ExportSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(pattern=r"^cmp_[A-Za-z0-9_-]{1,64}$")
    revision: int = Field(ge=1, strict=True)
    frame: _ExportFrame | None


class _ExportLine(BaseModel):
    """One exported share."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SHARE_ID_PATTERN.pattern)
    kind: Literal["card", "canvas", "search", "conversation"]
    mode: Literal["live", "snapshot"]
    owner: str | None
    created_at: AwareDatetime
    title: str | None
    slots: list[_ExportSlot]
    payload: JsonValue

    @model_validator(mode="after")
    def _placeable(self) -> Self:
        if self.kind == "conversation" and self.mode == "live":
            raise ValueError("a conversation share is a snapshot, and this one is live")
        if self.kind in ("search", "conversation") and self.slots:
            raise ValueError(f"a {self.kind} share has no slots, and this one has some")
        if refusal := payload_refusal(self.payload):
            raise ValueError(refusal)
        return self


class _Refused(Exception):
    """A line the import cannot place, with the reason it gives."""


@dataclass(frozen=True)
class ImportCounts:
    """How many lines the run imported, found imported before, and refused."""

    imported: int
    unchanged: int
    refused: int


def import_shares(
    lines: Iterable[str],
    *,
    share_store: ShareStore,
    revision_store: RevisionStore,
    frame_store: FrameStore,
    write_token: Callable[[str], None],
    refuse: Callable[[str, str], None],
) -> ImportCounts:
    """Import every share in `lines`, a JSON object per non-blank line, pinning the frames each
    shows, and give each fresh token's report line to `write_token` as JSON text before the
    share commits. Each refused line goes to `refuse` as its label, `share <id>` or `line <n>`
    when the id is unreadable, and its reason.

    Raises:
        PresentationStoreUnavailable: The store failed, or `write_token` did, which stops the
            run.
    """
    imported = unchanged = refused = 0
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        label = _label(text, number=number)
        try:
            line = _parse(text)
            share = _share(line, revision_store=revision_store)
            _pin(share.cards, revision_store=revision_store, frame_store=frame_store)
            token = (
                mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
                if line.owner is not None
                else None
            )
            report = (
                json.dumps({"id": line.id, "owner": line.owner, "write_token": token})
                if token is not None
                else None
            )
            stored = share_store.import_share(
                replace(
                    share,
                    write_token_sha256=write_token_digest(token) if token is not None else None,
                ),
                on_stored=partial(write_token, report) if report is not None else None,
            )
        except (_Refused, ShareIdTaken) as exc:
            refuse(label, str(exc))
            refused += 1
            continue
        if stored:
            imported += 1
        else:
            unchanged += 1
    return ImportCounts(imported=imported, unchanged=unchanged, refused=refused)


def import_share_file(
    *,
    in_path: Path,
    tokens: Path,
    dataset_store: DatasetStore,
    artifact_store: ArtifactStore,
    refuse: Callable[[str, str], None],
) -> ImportCounts:
    """Import the export at `in_path` into the Presentation store's runtime database, pinning
    frames over the Datasets `dataset_store` and `artifact_store`, and append each token line to
    `tokens` (created readable by its owner alone), flushed and fsynced before its share
    commits.

    Raises:
        PresentationStoreUnavailable: The store failed, or the report could not be written.
    """
    # Pinning promotes artifacts and records references. It reads no graph.
    frame_store = LiveFrameStore(
        snapshots=LiveSnapshotReader(store=dataset_store),
        artifacts=artifact_store,
        graph=None,
        reference_writer=LiveArtifactReferenceWriter(store=dataset_store),
        # An operator's import pins without spending the open callers' budgets.
        allowances=None,
        recipe_reader=LiveRecipeStore(store=dataset_store),
    )
    presentation = LiveDatasetStore(connections=shared_pool())
    descriptor = os.open(tokens, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with (
            in_path.open(encoding="utf-8") as lines,
            os.fdopen(descriptor, "a", encoding="utf-8") as report,
        ):

            def write_token(line: str) -> None:
                print(line, file=report, flush=True)
                os.fsync(report.fileno())

            return import_shares(
                lines,
                share_store=LiveShareStore(store=presentation),
                revision_store=LivePresentationStore(store=presentation),
                frame_store=frame_store,
                write_token=write_token,
                refuse=refuse,
            )
    finally:
        close_shared_pool()


def _label(text: str, *, number: int) -> str:
    """`share <id>` for a line holding a readable 8-character id, else `line <number>`."""
    try:
        share_id = json.loads(text).get("id")
    except (ValueError, AttributeError):
        share_id = None
    if isinstance(share_id, str) and SHARE_ID_PATTERN.fullmatch(share_id):
        return f"share {share_id}"
    return f"line {number}"


def _parse(text: str) -> _ExportLine:
    try:
        return _ExportLine.model_validate_json(text)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'line'}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        raise _Refused(f"not a share as the export defines it ({errors})") from None


def _share(line: _ExportLine, *, revision_store: RevisionStore) -> ImportedShare:
    """The share `line` stores, with no token yet, each slot's frame resolved and the rendering
    each card revision recorded."""
    read = [
        _card(slot, index=index, revision_store=revision_store)
        for index, slot in enumerate(line.slots)
    ]
    return ImportedShare(
        id=line.id,
        kind=line.kind,
        mode=line.mode,
        title=line.title,
        owner=line.owner,
        payload=line.payload,
        created_at=line.created_at,
        write_token_sha256=None,
        cards=tuple(card for card, _ in read),
        renderings={(card.component_id, card.revision): rendering for card, rendering in read},
    )


def _pin(
    cards: Iterable[ShareCard], *, revision_store: RevisionStore, frame_store: FrameStore
) -> None:
    """Pin the frame each distinct card revision draws and each distinct slot frame.

    Raises:
        _Refused: A frame, or a root of one, cannot be pinned.
    """
    cards = tuple(cards)
    try:
        for component_id, revision in dict.fromkeys((c.component_id, c.revision) for c in cards):
            pin_card_frames(
                component_id=component_id,
                revision=revision,
                revision_store=revision_store,
                frame_store=frame_store,
            )
        for component_id, frame in dict.fromkeys((c.component_id, c.frame) for c in cards):
            if frame is not None:
                frame_store.pin(frame_ref=frame, component_id=component_id)
    except FrameUnavailable as exc:
        raise _Refused(f"a frame it shows cannot be pinned: {exc}") from None


def _card(
    slot: _ExportSlot, *, index: int, revision_store: RevisionStore
) -> tuple[ShareCard, Mapping[str, object] | None]:
    """The card of the export's slot `index`, its frame the card revision's when the export
    names none, and the rendering the card revision recorded."""
    try:
        record = read_stored_revision(
            component_id=slot.component_id, revision=slot.revision, revision_store=revision_store
        )
    except (DraftNotFound, RevisionNotFound, EditConflict):
        raise _Refused(
            f"slot {index} names {slot.component_id} revision {slot.revision}, which this "
            "service does not hold"
        ) from None
    if slot.frame is not None:
        frame = FrameRef(spec_id=slot.frame.spec_id, content_version=slot.frame.content_version)
    else:
        stored = record.stored_intent.get("frame_ref")
        if not isinstance(stored, dict):
            raise _Refused(
                f"slot {index} names no frame, and {slot.component_id} revision "
                f"{slot.revision} draws none"
            )
        frame = FrameRef(**cast("dict[str, str]", stored))
    return (
        ShareCard(component_id=slot.component_id, revision=slot.revision, frame=frame),
        record.frame_rendering,
    )
