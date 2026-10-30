"""Import one of the website's shares under its own id, as the share import route
(`presentation_api.doors.shares`) does for the website's share migration job.

`ShareImport` is a share as the website's `docs/share-export.md` defines a line, less its `id`,
which the route's path names: `kind`, `mode`, `owner`, `created_at`, `title`, `slots` and
`payload`, plus `write_token_sha256`. `import_website_share` stores it under its id with its
kind, mode, title, owner, payload, `created_at` and digest, at revision 1. The export's `slots`
are listed in display order and the payload's held cards name them by 0-based index; the
share's slots are numbered from 1, so the export's slot `i` is slot `i + 1`.

A slot names a card revision this service holds and the frame it shows. A slot whose `frame` is
null, a live card the website stored with no frame of its own, draws the frame its card
revision is stored on. Before storing a share the import pins what it shows, live or snapshot:
the frame each card revision draws, as a share's create does (`pin_card_frames`), and each
slot's frame, as a snapshot conversion does, so no slot's frame leaves the working tier.

`write_token_sha256` is the hex SHA-256 of the `shw_` write token the website holds for a
signed-in user's share, and null for a guest's share, which nobody can then change or delete.
The token itself never reaches this service. A digest needs an `owner`.

The import is idempotent, and the first import of an id wins. A share is the same share as one
the id holds when its kind, `created_at` and digest match. It was imported before and is left
as it is, taken down since or not, even when its owner, mode, title, slots or payload now
differ, and the import reads nothing else and pins nothing. An id holding another share, or
this one under another digest, and a digest another share holds, refuse (`ShareIdTaken`)
before anything is pinned; the write checks again, for an import racing this one. A share is
refused before anything is stored when it is not a share as the export defines it, its payload is over
`MAX_SHARE_PAYLOAD_BYTES` as a create's would be, it is a live conversation or a search or
conversation share with slots, a slot names a card revision this service does not hold or one
drawing no frame, or a frame cannot be pinned. The website's export puts no note in a slot (a
note is a text card, which has no slot), so a slot drawing no frame is not a share the export
writes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, Self, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import DraftNotFound, EditConflict, FrameRef, FrameStore, RevisionStore
from presentation_store.card_pins import pin_card_frames
from presentation_store.shares import ImportedShare, ShareCard, ShareStore, payload_refusal

__all__ = ["ShareImport", "SlotDrawsNoFrame", "import_website_share"]

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


class ShareImport(BaseModel):
    """One website share as its export line states it, less the id, with the digest of the
    write token the website holds for it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["card", "canvas", "search", "conversation"]
    mode: Literal["live", "snapshot"]
    owner: str | None
    created_at: AwareDatetime
    title: str | None
    slots: list[_ExportSlot]
    payload: JsonValue
    write_token_sha256: str | None = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="the lowercase hex SHA-256 of the share's shw_ write token, null for a "
        "guest's share. A digest needs an owner",
    )

    @model_validator(mode="after")
    def _placeable(self) -> Self:
        if self.kind == "conversation" and self.mode == "live":
            raise ValueError("a conversation share is a snapshot, and this one is live")
        if self.kind in ("search", "conversation") and self.slots:
            raise ValueError(f"a {self.kind} share has no slots, and this one has some")
        if self.write_token_sha256 is not None and self.owner is None:
            raise ValueError("a guest's share has no write token, and this one has a digest")
        if refusal := payload_refusal(self.payload):
            raise ValueError(refusal)
        return self


class SlotDrawsNoFrame(Exception):
    """A slot names no frame, and its card revision draws none."""

    def __init__(self, *, index: int, component_id: str, revision: int) -> None:
        self.index = index
        super().__init__(
            f"slot {index} names no frame, and {component_id} revision {revision} draws none"
        )


def import_website_share(
    share_id: str,
    share: ShareImport,
    *,
    share_store: ShareStore,
    revision_store: RevisionStore,
    frame_store: FrameStore,
) -> bool:
    """Store `share` under `share_id`, pinning the frames it shows first. Returns `True` when
    it stored the share and `False`, reading no card and pinning nothing, when an earlier
    import had.

    Raises:
        DraftNotFound, RevisionNotFound: A slot names a card revision this service does not
            hold.
        SlotDrawsNoFrame: A slot names no frame, and its card revision draws none.
        FrameUnavailable: A frame the share shows cannot be pinned.
        ShareIdTaken: The id holds another share, or this one under another digest, or another
            share holds the digest.
        PresentationStoreUnavailable: The store failed.
    """
    digest = (
        bytes.fromhex(share.write_token_sha256) if share.write_token_sha256 is not None else None
    )
    if share_store.held_import(
        share_id=share_id, kind=share.kind, created_at=share.created_at, write_token_sha256=digest
    ):
        return False
    read = [
        _card(slot, index=index, revision_store=revision_store)
        for index, slot in enumerate(share.slots)
    ]
    cards = tuple(card for card, _ in read)
    _pin(cards, revision_store=revision_store, frame_store=frame_store)
    return share_store.import_share(
        ImportedShare(
            id=share_id,
            kind=share.kind,
            mode=share.mode,
            title=share.title,
            owner=share.owner,
            payload=share.payload,
            created_at=share.created_at,
            write_token_sha256=digest,
            cards=cards,
            renderings={(card.component_id, card.revision): rendering for card, rendering in read},
        )
    )


def _pin(
    cards: Iterable[ShareCard], *, revision_store: RevisionStore, frame_store: FrameStore
) -> None:
    """Pin the frame each distinct card revision draws and each distinct slot frame."""
    cards = tuple(cards)
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


def _card(
    slot: _ExportSlot, *, index: int, revision_store: RevisionStore
) -> tuple[ShareCard, Mapping[str, object] | None]:
    """The card of the export's slot `index`, its frame the card revision's when the export
    names none, and the rendering the card revision recorded."""
    try:
        record = read_stored_revision(
            component_id=slot.component_id, revision=slot.revision, revision_store=revision_store
        )
    except EditConflict:
        # A component with no head holds no revision to show.
        raise DraftNotFound(subject_kind="component", subject_id=slot.component_id) from None
    if slot.frame is not None:
        frame = FrameRef(spec_id=slot.frame.spec_id, content_version=slot.frame.content_version)
    else:
        stored = record.stored_intent.get("frame_ref")
        if not isinstance(stored, dict):
            raise SlotDrawsNoFrame(
                index=index, component_id=slot.component_id, revision=slot.revision
            )
        frame = FrameRef(**cast("dict[str, str]", stored))
    return (
        ShareCard(component_id=slot.component_id, revision=slot.revision, frame=frame),
        record.frame_rendering,
    )
