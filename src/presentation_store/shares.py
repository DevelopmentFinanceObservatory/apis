"""Share records over `presentation.share`, `share_revision` and `share_slot` (migration 0012),
the frames each slot served (`served_frame`), and each live slot's newest-data check
(`share_slot_check`, migrations 0013 and 0014).

A share publishes cards under one `/s/<id>` id. Each revision lists the share's slots, numbered
from 1, and a slot holds one card revision or none. A create writes revision 1 with one slot per
card. A replace writes the next revision with every slot the share has had, so a replacement
with fewer cards leaves the later slots empty and keeps their URLs. It locks the share row and
refuses a snapshot, or a share whose revision moved since the caller read it, so a replace
racing a snapshot conversion cannot write new cards into the snapshot. Revision rows are
append-only. A delete sets `deleted_at`, after which every read and write answers
`ShareNotFound`.

A snapshot conversion (`snapshot_share`) writes the next revision with the frame each slot
serves and sets the share's mode to `snapshot`, so its slots keep those frames for good.
`import_share` stores a website share under its own id, `created_at`, owner, payload and write
token digest, and finds a share it imported before by its id, kind, `created_at` and digest.

A live slot's data read keeps one `SlotCheck` per slot and card revision: the frame the slot
serves, its freshness, when the check ran and when the next one is due. `claim_slot_check`
claims a due check in one statement, so of two reads finding it due only one runs the check. The
claim moves only the due time, to when the claim expires, and leaves the frame, freshness and
check time of the last check for the reads that lose it. A slot's first claim inserts its row as
`pending` on the card's own frame with no check time.

`served_frame` (migrations 0012 and 0017) is a timeline of what each slot showed. An entry is
recorded only when it differs from the slot's newest entry, so a slot that goes back to an
earlier frame records it again and repeated reads record nothing new. A create, replace or
snapshot conversion records what each slot of its revision shows in the transaction that writes
the revision, dated with the revision's `created_at`: a slot with no card, or whose card draws no
frame, records no card, and a data card records the frame the slot serves, which for a live slot
whose card kept its place is its last check's frame. The revision's `clock_timestamp()` is read
while the write holds the share row lock, so entries and revisions of one share are dated in
commit order, and a write that cannot record its entries commits nothing. `record_served`
records the frame a slot read serves. It reads the slot's newest entry first and, only when it
shows something else, locks the share row, checks that the revision the read served is still the
latest, and checks the newest entry again under the lock, so concurrent reads of one slot record
one entry, a read of an unchanged slot takes no lock, and a read that served a revision a replace
or snapshot conversion has since moved past records nothing.
`served_frames` reads the timeline a page at a time, newest first by `first_served_at` and then
`id`, by keyset from a `ServedPosition` (migration 0018's index), reading one row past the page
to learn whether another follows.

`create_share` mints the id (`ids.mint_share_id`) and inserts it with `ON CONFLICT DO NOTHING`,
minting again while the id is taken, by a share made here or one imported from the website,
up to `_MINT_ATTEMPTS` times before `GeneratedIdCollision`. The store keeps
the write token digest it is given, and `read_share` returns it for the caller to compare. No
two shares hold one token digest (migration 0015). A create given a digest another share holds,
taken down or not, raises `ShareWriteTokenInUse`, and `prior_create` finds that share with the
digest of the create request that stored it, so a create keyed by a client's token can answer
its retry with the first share. Driver failures become `PresentationStoreUnavailable` with the
failure's SQLSTATE.

`MAX_SHARE_PAYLOAD_BYTES` bounds the payload a share stores, which the shares door and the
website import both apply through `payload_refusal`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, cast

from assembler.ports import EditConflict, FrameRef
from dataset_store import DatasetStore
from dataset_store.store import Transaction
from presentation_store.ids import mint_share_id
from presentation_store.store_errors import (
    GeneratedIdCollision,
    PresentationStoreUnavailable,
    is_unique_violation,
    violated_constraint,
)

__all__ = [
    "MAX_SHARE_PAYLOAD_BYTES",
    "Freshness",
    "ImportedShare",
    "LiveShareStore",
    "PriorShareCreate",
    "Renderings",
    "ServedFrame",
    "ServedPage",
    "ServedPosition",
    "ShareCard",
    "ShareIdTaken",
    "ShareKind",
    "ShareMode",
    "ShareNotFound",
    "ShareNotReplaceable",
    "ShareRecord",
    "ShareSlot",
    "ShareStore",
    "ShareWriteTokenInUse",
    "SlotCheck",
    "payload_refusal",
]

type ShareKind = Literal["card", "canvas", "search", "conversation"]
type ShareMode = Literal["live", "snapshot"]
type Freshness = Literal["current", "stale", "pending"]
type Renderings = Mapping[tuple[str, int], Mapping[str, object] | None]
"""The rendering each card revision recorded, by component id and revision. A write records
what its slots show with these; a card revision absent from it records no rendering."""

# The chance that one minted id is taken is the stored share count over 62**8, about 2.2e14, so
# five tries in a row fail only on a broken minter.
_MINT_ATTEMPTS = 5

_SHARE_COLUMNS = "id, kind, mode, title, write_token_sha256, owner, payload, created_at"

_SHARE_WRITE_TOKEN_INDEX = "share_write_token_sha256_key"

MAX_SHARE_PAYLOAD_BYTES = 10 * 1024 * 1024
"""The largest `payload` a share stores, as compact UTF-8 JSON (10 MiB, the website's limit)."""


def payload_refusal(payload: object) -> str | None:
    """Why a share cannot store `payload`, or `None` when it can. A share refuses a payload
    whose compact UTF-8 JSON is over `MAX_SHARE_PAYLOAD_BYTES`."""
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    if size <= MAX_SHARE_PAYLOAD_BYTES:
        return None
    return f"the payload is {size} bytes, over the {MAX_SHARE_PAYLOAD_BYTES}-byte bound"


class ShareNotFound(Exception):
    """No share exists at this id, or it was taken down. Maps to `not_found`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(f"no share exists at id {share_id!r}")


class ShareNotReplaceable(Exception):
    """A replace names a snapshot. Maps to 409 `share_not_replaceable`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(
            f"share {share_id!r} is a snapshot, whose cards are fixed for good; create a new "
            "share for new cards, or create it with latest=true to replace its cards later"
        )


class ShareWriteTokenInUse(Exception):
    """A create's write token digest is one another share holds."""

    def __init__(self) -> None:
        super().__init__("another share holds this write token")


class ShareIdTaken(Exception):
    """An imported share's id holds another share, either one made here or a website share of
    another kind or creation time, or holds this share with another write token digest."""

    def __init__(self, share_id: str, *, token_differs: bool = False) -> None:
        self.share_id = share_id
        self.token_differs = token_differs
        super().__init__(
            f"share id {share_id!r} holds this share under another write token digest"
            if token_differs
            else f"share id {share_id!r} already holds another share"
        )


@dataclass(frozen=True)
class ShareCard:
    """One card revision a slot shows, and the frame it draws (`None` for a note, or for an
    imported slot whose frame the export did not name)."""

    component_id: str
    revision: int
    frame: FrameRef | None


@dataclass(frozen=True)
class ShareSlot:
    """Slot `slot` (from 1) of a share revision, holding `card` or nothing."""

    slot: int
    card: ShareCard | None


@dataclass(frozen=True)
class ShareRecord:
    """A share at its latest revision. `write_token_sha256` is `None` for a share that no token
    can change."""

    id: str
    kind: ShareKind
    mode: ShareMode
    title: str | None
    write_token_sha256: bytes | None
    owner: str | None
    payload: object
    created_at: datetime
    revision: int
    revised_at: datetime
    slots: tuple[ShareSlot, ...]


@dataclass(frozen=True)
class PriorShareCreate:
    """The share a write token digest names, taken down or not, and the digest of the create
    request that stored it (`None` for a share created without one)."""

    share_id: str
    create_request_sha256: bytes | None


@dataclass(frozen=True)
class ImportedShare:
    """A website share as `import_share` stores it, at revision 1 with one slot per card, and
    the rendering each card revision recorded."""

    id: str
    kind: ShareKind
    mode: ShareMode
    title: str | None
    owner: str | None
    payload: object
    created_at: datetime
    write_token_sha256: bytes | None
    cards: tuple[ShareCard, ...]
    renderings: Renderings


@dataclass(frozen=True)
class SlotCheck:
    """A live slot's newest-data check: the frame the slot serves until the next check, its
    freshness, when the check ran (`None` while the slot's first check runs), and when the next
    one is due."""

    frame: FrameRef
    freshness: Freshness
    checked_at: datetime | None
    next_check_at: datetime


@dataclass(frozen=True)
class ServedFrame:
    """One entry of what a share's slot showed: the card revision, the frame, the rendering the
    card revision recorded (`None` when unrecorded), and when the slot began to show it. An
    entry for a slot left with no card has `component_id`, `revision`, `frame` and `rendering`
    all `None`."""

    slot: int
    component_id: str | None
    revision: int | None
    frame: FrameRef | None
    rendering: Mapping[str, object] | None
    first_served_at: datetime


@dataclass(frozen=True)
class ServedPosition:
    """Where an entry sits in a share's timeline: newest first by `first_served_at`, then by
    the entry's `id`, which orders entries recorded at one instant."""

    first_served_at: datetime
    id: int


@dataclass(frozen=True)
class ServedPage:
    """One page of a share's timeline, newest first, and the position of its last entry when
    older entries follow (`None` on the last page)."""

    entries: tuple[ServedFrame, ...]
    next: ServedPosition | None


class ShareStore(Protocol):
    def create_share(
        self,
        *,
        kind: ShareKind,
        mode: ShareMode,
        title: str | None,
        write_token_sha256: bytes | None,
        cards: Sequence[ShareCard],
        renderings: Renderings,
        payload: object = None,
        create_request_sha256: bytes | None = None,
    ) -> ShareRecord:
        """Store a new share at revision 1, one slot per card in order, under a minted id, and
        record what each slot shows. `payload` is stored as given; `None` stores none.
        `create_request_sha256` is the digest of the create request a client's write token
        keys, which `prior_create` returns.

        Raises:
            ShareWriteTokenInUse: Another share holds `write_token_sha256`.
            GeneratedIdCollision: Every minted id was taken.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def prior_create(self, *, write_token_sha256: bytes) -> PriorShareCreate | None:
        """The share holding `write_token_sha256`, taken down or not, or `None` when none does.

        Raises:
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def read_share(self, *, share_id: str) -> ShareRecord:
        """Read a share at its latest revision.

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def replace_share(
        self, *, share_id: str, revision: int, cards: Sequence[ShareCard], renderings: Renderings
    ) -> ShareRecord:
        """Write the live share's next revision after `revision`: `cards` in slots 1 to
        `len(cards)`, and every further slot the latest revision has left empty. Record what
        each slot shows.

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down.
            ShareNotReplaceable: The share is a snapshot.
            EditConflict: The share's latest revision is not `revision`.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def delete_share(self, *, share_id: str) -> None:
        """Take a share down.

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down already.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def snapshot_share(
        self,
        *,
        share_id: str,
        revision: int,
        frames: Mapping[int, FrameRef],
        renderings: Renderings,
    ) -> ShareRecord:
        """Turn a live share at `revision` into a snapshot: write its next revision with the
        same cards, each slot in `frames` drawing that frame, set its mode, and record what each
        slot shows. A snapshot is returned as it is.

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down.
            EditConflict: The share's latest revision is not `revision`.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def import_share(self, share: ImportedShare) -> bool:
        """Store a website share under its own id at revision 1 and record what each slot
        shows from its `created_at`. Returns `False`, changing nothing, when the id already
        holds a share of the same kind, `created_at` and write token digest, which an earlier
        import stored, taken down since or not.

        Raises:
            ShareIdTaken: The id holds another share, or this one under another digest.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def slot_check(self, *, share_id: str, slot: int, card: ShareCard) -> SlotCheck | None:
        """The latest newest-data check of `card` in `slot`, or `None` before the first.

        Raises:
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def claim_slot_check(
        self, *, share_id: str, slot: int, card: ShareCard, now: datetime, claim_until: datetime
    ) -> bool:
        """Claim the check of `card`, which must draw a frame, in `slot` until `claim_until`
        when it is due at `now`: move its next due time to `claim_until`, or insert it `pending`
        on `card.frame` with no check time when none is stored. Returns whether it did, so the
        caller runs the check.

        Raises:
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def record_slot_check(
        self, *, share_id: str, slot: int, card: ShareCard, check: SlotCheck
    ) -> None:
        """Store `check` as the latest check of `card` in `slot`.

        Raises:
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def record_served(
        self,
        *,
        share_id: str,
        revision: int,
        slot: int,
        card: ShareCard,
        frame: FrameRef,
        rendering: Mapping[str, object] | None,
    ) -> None:
        """Record that `slot` of the share at `revision` shows `frame` for `card` from now,
        unless the slot's newest entry already shows it or `revision` is no longer the share's
        latest.

        Raises:
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def served_frames(
        self, *, share_id: str, limit: int, before: ServedPosition | None = None
    ) -> ServedPage:
        """Up to `limit` entries of what the share's slots showed, newest first, starting after
        `before` (from the newest entry when `None`).

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down.
            PresentationStoreUnavailable: The store failed.
        """
        ...


class LiveShareStore:
    """`ShareStore` over the `presentation` database. `mint_id` defaults to
    `ids.mint_share_id`."""

    def __init__(self, *, store: DatasetStore, mint_id: Callable[[], str] = mint_share_id) -> None:
        self._store = store
        self._mint_id = mint_id

    def create_share(
        self,
        *,
        kind: ShareKind,
        mode: ShareMode,
        title: str | None,
        write_token_sha256: bytes | None,
        cards: Sequence[ShareCard],
        renderings: Renderings,
        payload: object = None,
        create_request_sha256: bytes | None = None,
    ) -> ShareRecord:
        def write(tx: Transaction) -> ShareRecord:
            share_id = self._insert_share(
                tx,
                kind=kind,
                mode=mode,
                title=title,
                write_token_sha256=write_token_sha256,
                create_request_sha256=create_request_sha256,
                payload=payload,
            )
            slots = tuple(ShareSlot(slot=n, card=card) for n, card in enumerate(cards, start=1))
            at = _insert_revision(tx, share_id=share_id, revision=1, slots=slots)
            _record_revision(
                tx, share_id=share_id, mode=mode, slots=slots, renderings=renderings, at=at
            )
            return _read(tx, share_id=share_id, for_update=False)

        return self._in_transaction("creating a share", write)

    def prior_create(self, *, write_token_sha256: bytes) -> PriorShareCreate | None:
        def read(tx: Transaction) -> PriorShareCreate | None:
            row = tx.fetch_one(
                "SELECT id, create_request_sha256 FROM presentation.share "
                "WHERE write_token_sha256 = %s",
                (write_token_sha256,),
            )
            if row is None:
                return None
            digest = row["create_request_sha256"]
            return PriorShareCreate(
                share_id=cast(str, row["id"]),
                create_request_sha256=(
                    bytes(cast("bytes | memoryview", digest)) if digest is not None else None
                ),
            )

        return self._in_transaction("reading the share by write token digest", read)

    def read_share(self, *, share_id: str) -> ShareRecord:
        return self._in_transaction(
            f"reading share {share_id!r}",
            lambda tx: _read(tx, share_id=share_id, for_update=False),
        )

    def replace_share(
        self, *, share_id: str, revision: int, cards: Sequence[ShareCard], renderings: Renderings
    ) -> ShareRecord:
        def write(tx: Transaction) -> ShareRecord:
            current = _read(tx, share_id=share_id, for_update=True)
            if current.mode == "snapshot":
                raise ShareNotReplaceable(share_id)
            if current.revision != revision:
                raise EditConflict(
                    f"share {share_id!r} moved to revision {current.revision} while it was "
                    f"replaced at revision {revision}; retry"
                )
            count = max(len(current.slots), len(cards))
            slots = tuple(
                ShareSlot(slot=n, card=cards[n - 1] if n <= len(cards) else None)
                for n in range(1, count + 1)
            )
            at = _insert_revision(tx, share_id=share_id, revision=current.revision + 1, slots=slots)
            _record_revision(
                tx, share_id=share_id, mode="live", slots=slots, renderings=renderings, at=at
            )
            return _read(tx, share_id=share_id, for_update=False)

        return self._in_transaction(f"replacing share {share_id!r}", write)

    def delete_share(self, *, share_id: str) -> None:
        def write(tx: Transaction) -> None:
            row = tx.fetch_one(
                "UPDATE presentation.share SET deleted_at = now() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (share_id,),
            )
            if row is None:
                raise ShareNotFound(share_id)

        self._in_transaction(f"deleting share {share_id!r}", write)

    def snapshot_share(
        self,
        *,
        share_id: str,
        revision: int,
        frames: Mapping[int, FrameRef],
        renderings: Renderings,
    ) -> ShareRecord:
        def write(tx: Transaction) -> ShareRecord:
            current = _read(tx, share_id=share_id, for_update=True)
            if current.mode == "snapshot":
                return current
            if current.revision != revision:
                raise EditConflict(
                    f"share {share_id!r} moved to revision {current.revision} while it was "
                    f"turned into a snapshot at revision {revision}; retry"
                )
            slots = tuple(
                ShareSlot(
                    slot=slot.slot,
                    card=(
                        ShareCard(
                            component_id=slot.card.component_id,
                            revision=slot.card.revision,
                            frame=frames.get(slot.slot, slot.card.frame),
                        )
                        if slot.card is not None
                        else None
                    ),
                )
                for slot in current.slots
            )
            at = _insert_revision(tx, share_id=share_id, revision=revision + 1, slots=slots)
            _record_revision(
                tx, share_id=share_id, mode="snapshot", slots=slots, renderings=renderings, at=at
            )
            tx.execute("UPDATE presentation.share SET mode = 'snapshot' WHERE id = %s", (share_id,))
            return _read(tx, share_id=share_id, for_update=False)

        return self._in_transaction(f"turning share {share_id!r} into a snapshot", write)

    def import_share(self, share: ImportedShare) -> bool:
        def write(tx: Transaction) -> bool:
            row = tx.fetch_one(
                "INSERT INTO presentation.share (id, kind, mode, title, write_token_sha256, "
                "owner, payload, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING RETURNING id",
                (
                    share.id,
                    share.kind,
                    share.mode,
                    share.title,
                    share.write_token_sha256,
                    share.owner,
                    _json_param(share.payload),
                    share.created_at,
                ),
            )
            if row is None:
                held = tx.fetch_one(
                    "SELECT kind, created_at, write_token_sha256 FROM presentation.share "
                    "WHERE id = %s",
                    (share.id,),
                )
                if held is None or (held["kind"], held["created_at"]) != (
                    share.kind,
                    share.created_at,
                ):
                    raise ShareIdTaken(share.id)
                if held["write_token_sha256"] != share.write_token_sha256:
                    raise ShareIdTaken(share.id, token_differs=True)
                return False
            slots = tuple(
                ShareSlot(slot=n, card=card) for n, card in enumerate(share.cards, start=1)
            )
            at = _insert_revision(
                tx, share_id=share.id, revision=1, slots=slots, created_at=share.created_at
            )
            _record_revision(
                tx,
                share_id=share.id,
                mode=share.mode,
                slots=slots,
                renderings=share.renderings,
                at=at,
            )
            return True

        return self._in_transaction(f"importing share {share.id!r}", write)

    def slot_check(self, *, share_id: str, slot: int, card: ShareCard) -> SlotCheck | None:
        return self._in_transaction(
            f"reading slot {slot} of share {share_id!r}",
            lambda tx: _slot_check(tx, share_id=share_id, slot=slot, card=card),
        )

    def claim_slot_check(
        self, *, share_id: str, slot: int, card: ShareCard, now: datetime, claim_until: datetime
    ) -> bool:
        frame = card.frame
        assert frame is not None

        def write(tx: Transaction) -> bool:
            row = tx.fetch_one(
                "INSERT INTO presentation.share_slot_check AS held (share_id, slot, component_id, "
                "component_revision, frame_spec_id, frame_content_version, freshness, "
                "checked_at, next_check_at) VALUES (%s, %s, %s, %s, %s, %s, 'pending', NULL, %s) "
                "ON CONFLICT (share_id, slot, component_id, component_revision) DO UPDATE SET "
                "next_check_at = EXCLUDED.next_check_at WHERE held.next_check_at <= %s "
                "RETURNING slot",
                (
                    share_id,
                    slot,
                    card.component_id,
                    card.revision,
                    frame.spec_id,
                    frame.content_version,
                    claim_until,
                    now,
                ),
            )
            return row is not None

        return self._in_transaction(f"claiming slot {slot} of share {share_id!r}", write)

    def record_slot_check(
        self, *, share_id: str, slot: int, card: ShareCard, check: SlotCheck
    ) -> None:
        def write(tx: Transaction) -> None:
            tx.execute(
                "INSERT INTO presentation.share_slot_check (share_id, slot, component_id, "
                "component_revision, frame_spec_id, frame_content_version, freshness, "
                "checked_at, next_check_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (share_id, slot, component_id, component_revision) DO UPDATE SET "
                "frame_spec_id = EXCLUDED.frame_spec_id, "
                "frame_content_version = EXCLUDED.frame_content_version, "
                "freshness = EXCLUDED.freshness, checked_at = EXCLUDED.checked_at, "
                "next_check_at = EXCLUDED.next_check_at",
                (
                    share_id,
                    slot,
                    card.component_id,
                    card.revision,
                    check.frame.spec_id,
                    check.frame.content_version,
                    check.freshness,
                    check.checked_at,
                    check.next_check_at,
                ),
            )

        self._in_transaction(f"recording slot {slot} of share {share_id!r}", write)

    def record_served(
        self,
        *,
        share_id: str,
        revision: int,
        slot: int,
        card: ShareCard,
        frame: FrameRef,
        rendering: Mapping[str, object] | None,
    ) -> None:
        shown = (card.component_id, card.revision, frame.spec_id, frame.content_version)

        def write(tx: Transaction) -> None:
            newest = tx.fetch_one(
                "SELECT component_id, component_revision, frame_spec_id, frame_content_version "
                "FROM presentation.served_frame WHERE share_id = %s AND slot = %s "
                "ORDER BY first_served_at DESC, id DESC LIMIT 1",
                (share_id, slot),
            )
            if newest is not None and tuple(newest[name] for name in _SHOWN_COLUMNS) == shown:
                return
            tx.execute(
                "SELECT 1 FROM presentation.share WHERE id = %s FOR NO KEY UPDATE", (share_id,)
            )
            latest = tx.fetch_one(
                "SELECT revision, created_at FROM presentation.share_revision "
                "WHERE share_id = %s ORDER BY revision DESC LIMIT 1",
                (share_id,),
            )
            if latest is None or latest["revision"] != revision:
                return
            _record_shown(
                tx, share_id=share_id, slot=slot, shown=shown, rendering=rendering, at=None
            )

        self._in_transaction(f"recording a frame share {share_id!r} served", write)

    def served_frames(
        self, *, share_id: str, limit: int, before: ServedPosition | None = None
    ) -> ServedPage:
        def read(tx: Transaction) -> ServedPage:
            _read(tx, share_id=share_id, for_update=False)
            if before is None:
                rows = tx.fetch_all(
                    f"{_SERVED_SELECT} ORDER BY first_served_at DESC, id DESC LIMIT %s",
                    (share_id, limit + 1),
                )
            else:
                rows = tx.fetch_all(
                    f"{_SERVED_SELECT} AND (first_served_at, id) < (%s, %s) "
                    "ORDER BY first_served_at DESC, id DESC LIMIT %s",
                    (share_id, before.first_served_at, before.id, limit + 1),
                )
            page = rows[:limit]
            last = page[-1] if len(rows) > limit else None
            return ServedPage(
                entries=tuple(_served(row) for row in page),
                next=(
                    ServedPosition(
                        first_served_at=cast(datetime, last["first_served_at"]),
                        id=cast(int, last["id"]),
                    )
                    if last is not None
                    else None
                ),
            )

        return self._in_transaction(f"reading what share {share_id!r} served", read)

    def _insert_share(
        self,
        tx: Transaction,
        *,
        kind: ShareKind,
        mode: ShareMode,
        title: str | None,
        write_token_sha256: bytes | None,
        create_request_sha256: bytes | None,
        payload: object,
    ) -> str:
        """Insert the share row under the first minted id no share holds. A unique violation
        on the write token index raises `ShareWriteTokenInUse`, after Postgres held the insert
        until the transaction holding the digest ended."""
        for _ in range(_MINT_ATTEMPTS):
            share_id = self._mint_id()
            in_use = False
            try:
                row = tx.fetch_one(
                    "INSERT INTO presentation.share (id, kind, mode, title, write_token_sha256, "
                    "create_request_sha256, payload) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO NOTHING RETURNING id",
                    (
                        share_id,
                        kind,
                        mode,
                        title,
                        write_token_sha256,
                        create_request_sha256,
                        _json_param(payload),
                    ),
                )
            except Exception as exc:
                if not (
                    is_unique_violation(exc)
                    and violated_constraint(exc) == _SHARE_WRITE_TOKEN_INDEX
                ):
                    raise
                in_use = True
            if in_use:
                raise ShareWriteTokenInUse
            if row is not None:
                return share_id
        raise GeneratedIdCollision(subject_kind="share")

    def _in_transaction[T](self, action: str, work: Callable[[Transaction], T]) -> T:
        """Run `work` in one transaction. `ShareNotFound`, `ShareNotReplaceable`, `ShareIdTaken`,
        `ShareWriteTokenInUse`, `EditConflict` and `GeneratedIdCollision` pass through, and any other failure becomes
        `PresentationStoreUnavailable`, raised after leaving `except` so driver context stays
        off it."""
        failure: PresentationStoreUnavailable | None = None
        result: T | None = None
        try:
            with self._store.transaction() as tx:
                result = work(tx)
        except (
            ShareNotFound,
            ShareNotReplaceable,
            ShareIdTaken,
            ShareWriteTokenInUse,
            EditConflict,
            GeneratedIdCollision,
        ):
            raise
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(f"{action} failed", exc)
        if failure is not None:
            raise failure
        return cast("T", result)


def _json_param(value: object) -> str | None:
    """`value` as JSON text for a jsonb column, since the driver binds a bare list as an array.
    `None` binds SQL NULL."""
    return json.dumps(value) if value is not None else None


def _insert_revision(
    tx: Transaction,
    *,
    share_id: str,
    revision: int,
    slots: Sequence[ShareSlot],
    created_at: datetime | None = None,
) -> datetime:
    """Insert one revision and its slots, and return the revision's `created_at`. Left out, it
    is `clock_timestamp()`, read after any share row lock the caller waited on, where the
    column default `now()` would be the transaction's start."""
    row = tx.fetch_one(
        "INSERT INTO presentation.share_revision (share_id, revision, created_at) "
        "VALUES (%s, %s, COALESCE(%s::timestamptz, clock_timestamp())) RETURNING created_at",
        (share_id, revision, created_at),
    )
    assert row is not None
    for slot in slots:
        card = slot.card
        frame = card.frame if card is not None else None
        tx.execute(
            "INSERT INTO presentation.share_slot (share_id, revision, slot, component_id, "
            "component_revision, frame_spec_id, frame_content_version) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                share_id,
                revision,
                slot.slot,
                card.component_id if card is not None else None,
                card.revision if card is not None else None,
                frame.spec_id if frame is not None else None,
                frame.content_version if frame is not None else None,
            ),
        )
    return cast(datetime, row["created_at"])


def _read(tx: Transaction, *, share_id: str, for_update: bool) -> ShareRecord:
    """Read the share, its latest revision and that revision's slots. `for_update` locks the
    share row, so concurrent replaces of one share write consecutive revisions."""
    share = tx.fetch_one(
        f"SELECT {_SHARE_COLUMNS} FROM presentation.share "
        "WHERE id = %s AND deleted_at IS NULL" + (" FOR UPDATE" if for_update else ""),
        (share_id,),
    )
    if share is None:
        raise ShareNotFound(share_id)
    revision = tx.fetch_one(
        "SELECT revision, created_at FROM presentation.share_revision "
        "WHERE share_id = %s ORDER BY revision DESC LIMIT 1",
        (share_id,),
    )
    if revision is None:
        # Every write inserts the share and its first revision in one transaction.
        raise PresentationStoreUnavailable(f"share {share_id!r} has no revision")
    number = cast(int, revision["revision"])
    slot_rows = tx.fetch_all(
        "SELECT slot, component_id, component_revision, frame_spec_id, frame_content_version "
        "FROM presentation.share_slot WHERE share_id = %s AND revision = %s ORDER BY slot",
        (share_id, number),
    )
    digest = share["write_token_sha256"]
    return ShareRecord(
        id=cast(str, share["id"]),
        kind=cast("ShareKind", share["kind"]),
        mode=cast("ShareMode", share["mode"]),
        title=cast("str | None", share["title"]),
        write_token_sha256=(
            bytes(cast("bytes | memoryview", digest)) if digest is not None else None
        ),
        owner=cast("str | None", share["owner"]),
        payload=share["payload"],
        created_at=cast(datetime, share["created_at"]),
        revision=number,
        revised_at=cast(datetime, revision["created_at"]),
        slots=tuple(_slot(row) for row in slot_rows),
    )


def _slot_check(tx: Transaction, *, share_id: str, slot: int, card: ShareCard) -> SlotCheck | None:
    row = tx.fetch_one(
        "SELECT frame_spec_id, frame_content_version, freshness, checked_at, "
        "next_check_at FROM presentation.share_slot_check WHERE share_id = %s AND slot = %s "
        "AND component_id = %s AND component_revision = %s",
        (share_id, slot, card.component_id, card.revision),
    )
    if row is None:
        return None
    return SlotCheck(
        frame=FrameRef(
            spec_id=cast(str, row["frame_spec_id"]),
            content_version=cast(str, row["frame_content_version"]),
        ),
        freshness=cast("Freshness", row["freshness"]),
        checked_at=cast("datetime | None", row["checked_at"]),
        next_check_at=cast(datetime, row["next_check_at"]),
    )


def _record_revision(
    tx: Transaction,
    *,
    share_id: str,
    mode: ShareMode,
    slots: Sequence[ShareSlot],
    renderings: Renderings,
    at: datetime,
) -> None:
    """Record what each slot of a revision written at `at` shows, for a share in `mode` after
    the write. The caller holds the share row lock, or inserted the row."""
    for slot in slots:
        card = slot.card
        if card is None or card.frame is None:
            _record_shown(
                tx, share_id=share_id, slot=slot.slot, shown=_NO_CARD, rendering=None, at=at
            )
            continue
        check = (
            _slot_check(tx, share_id=share_id, slot=slot.slot, card=card)
            if mode == "live"
            else None
        )
        frame = check.frame if check is not None else card.frame
        _record_shown(
            tx,
            share_id=share_id,
            slot=slot.slot,
            shown=(card.component_id, card.revision, frame.spec_id, frame.content_version),
            rendering=renderings.get((card.component_id, card.revision)),
            at=at,
        )


def _record_shown(
    tx: Transaction,
    *,
    share_id: str,
    slot: int,
    shown: tuple[str | None, int | None, str | None, str | None],
    rendering: Mapping[str, object] | None,
    at: datetime | None,
) -> None:
    """Record that `slot` shows `shown` (component id, component revision, frame spec id, frame
    content version, all `None` for no card) from `at`, or from `clock_timestamp()` when `at` is
    `None`, unless the slot's newest entry already shows it. The caller holds the share row
    lock, so the check and the insert are serialised across writers and readers of one share."""
    tx.execute(
        "INSERT INTO presentation.served_frame (share_id, slot, component_id, "
        "component_revision, frame_spec_id, frame_content_version, rendering, first_served_at) "
        "SELECT %s, %s, %s::text, %s::int, %s::text, %s::text, %s::jsonb, "
        "COALESCE(%s::timestamptz, clock_timestamp()) "
        "WHERE NOT EXISTS (SELECT 1 FROM (SELECT component_id, component_revision, "
        "frame_spec_id, frame_content_version FROM presentation.served_frame "
        "WHERE share_id = %s AND slot = %s ORDER BY first_served_at DESC, id DESC LIMIT 1) "
        "AS newest WHERE (newest.component_id, newest.component_revision, newest.frame_spec_id, "
        "newest.frame_content_version) IS NOT DISTINCT FROM (%s::text, %s::int, %s::text, "
        "%s::text))",
        (share_id, slot, *shown, _json_param(rendering), at, share_id, slot, *shown),
    )


_NO_CARD: tuple[None, None, None, None] = (None, None, None, None)

_SERVED_SELECT = (
    "SELECT id, slot, component_id, component_revision, frame_spec_id, frame_content_version, "
    "rendering, first_served_at FROM presentation.served_frame WHERE share_id = %s"
)

_SHOWN_COLUMNS = ("component_id", "component_revision", "frame_spec_id", "frame_content_version")


def _served(row: Mapping[str, object]) -> ServedFrame:
    spec_id = cast("str | None", row["frame_spec_id"])
    return ServedFrame(
        slot=cast(int, row["slot"]),
        component_id=cast("str | None", row["component_id"]),
        revision=cast("int | None", row["component_revision"]),
        frame=(
            FrameRef(spec_id=spec_id, content_version=cast(str, row["frame_content_version"]))
            if spec_id is not None
            else None
        ),
        rendering=cast("Mapping[str, object] | None", row["rendering"]),
        first_served_at=cast(datetime, row["first_served_at"]),
    )


def _slot(row: Mapping[str, object]) -> ShareSlot:
    component_id = cast("str | None", row["component_id"])
    if component_id is None:
        return ShareSlot(slot=cast(int, row["slot"]), card=None)
    spec_id = cast("str | None", row["frame_spec_id"])
    frame = (
        FrameRef(spec_id=spec_id, content_version=cast(str, row["frame_content_version"]))
        if spec_id is not None
        else None
    )
    return ShareSlot(
        slot=cast(int, row["slot"]),
        card=ShareCard(
            component_id=component_id, revision=cast(int, row["component_revision"]), frame=frame
        ),
    )
