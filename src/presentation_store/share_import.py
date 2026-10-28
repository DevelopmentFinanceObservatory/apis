"""Import the website's shares from the JSON lines `scripts/export-shares.ts` writes.

Each line is one share as the website's `docs/share-export.md` defines it: `id`, `kind`, `mode`,
`owner`, `created_at`, `title`, `slots` and `payload`. `import_shares` stores each under its own
id with its kind, mode, title, owner, payload and `created_at`, at revision 1. The export's
`slots` are listed in display order and the payload's held cards name them by 0-based index;
the share's slots are numbered from 1, so the export's slot `i` is slot `i + 1`.

A slot names a card revision this service holds and the frame it shows. A slot whose `frame` is
null, a live card the website stored with no frame of its own, draws the frame its card
revision is stored on. A signed-in user's share (`owner` set) gets a fresh `shw_` write token,
and the import writes `{id, owner, write_token}` to the token report for the website to keep. A
guest's share (`owner` null) is stored with no token, so nobody can change or delete it.

The import is idempotent. A share whose id already holds a share of the same kind and
`created_at` was imported by an earlier run: it is counted as unchanged, gets no new token and
writes no report line, so a re-run changes nothing. A line the import cannot place is refused
and counted, and the others carry on: a line that is not a share as the export defines it, a
live conversation, a search or conversation share with slots, a slot naming a card revision
this service does not hold or one drawing no frame, and an id that holds another share.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
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

from assembler.errors import RevisionNotFound
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import DraftNotFound, EditConflict, FrameRef, RevisionStore
from assembler.write_token import SHARE_WRITE_TOKEN_PREFIX, mint_write_token, write_token_digest
from presentation_store.shares import ImportedShare, ShareCard, ShareIdTaken, ShareStore

__all__ = ["ImportCounts", "import_shares"]

_SHARE_ID = re.compile(r"[A-Za-z0-9]{8}")

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

    id: str = Field(pattern=_SHARE_ID.pattern)
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
    write_token: Callable[[str], None],
    refuse: Callable[[str, str], None],
) -> ImportCounts:
    """Import every share in `lines`, a JSON object per non-blank line, and give each fresh
    token's report line to `write_token` as JSON text. Each refused line goes to `refuse` as
    its label, `share <id>` or `line <n>` when the id is unreadable, and its reason.

    Raises:
        PresentationStoreUnavailable: The store failed, which stops the run.
    """
    imported = unchanged = refused = 0
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        label = _label(text, number=number)
        try:
            line = _parse(text)
            share = _share(line, revision_store=revision_store)
            token = (
                mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
                if line.owner is not None
                else None
            )
            stored = share_store.import_share(
                replace(
                    share,
                    write_token_sha256=write_token_digest(token) if token is not None else None,
                )
            )
        except (_Refused, ShareIdTaken) as exc:
            refuse(label, str(exc))
            refused += 1
            continue
        if not stored:
            unchanged += 1
            continue
        imported += 1
        if token is not None:
            write_token(json.dumps({"id": line.id, "owner": line.owner, "write_token": token}))
    return ImportCounts(imported=imported, unchanged=unchanged, refused=refused)


def _label(text: str, *, number: int) -> str:
    """`share <id>` for a line holding a readable 8-character id, else `line <number>`."""
    try:
        share_id = json.loads(text).get("id")
    except (ValueError, AttributeError):
        share_id = None
    if isinstance(share_id, str) and _SHARE_ID.fullmatch(share_id):
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
    """The share `line` stores, with no token yet, each slot's frame resolved."""
    return ImportedShare(
        id=line.id,
        kind=line.kind,
        mode=line.mode,
        title=line.title,
        owner=line.owner,
        payload=line.payload,
        created_at=line.created_at,
        write_token_sha256=None,
        cards=tuple(
            _card(slot, index=index, revision_store=revision_store)
            for index, slot in enumerate(line.slots)
        ),
    )


def _card(slot: _ExportSlot, *, index: int, revision_store: RevisionStore) -> ShareCard:
    """The card of the export's slot `index`, its frame the card revision's when the export
    names none."""
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
    return ShareCard(component_id=slot.component_id, revision=slot.revision, frame=frame)
