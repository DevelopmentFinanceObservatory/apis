"""Share records over `presentation.share`, `share_revision` and `share_slot` (migration 0012),
the frames each slot served (`served_frame`), and each live slot's newest-data check
(`share_slot_check`, migrations 0013 and 0014).

A share publishes cards under one `/s/<id>` id. Each revision lists the share's slots, numbered
from 1, and a slot holds one card revision or none. A create writes revision 1 with one slot per
card. A replace writes the next revision with every slot the share has had, so a replacement
with fewer cards leaves the later slots empty and keeps their URLs. Revision rows are never
updated or deleted. A delete sets `deleted_at`, after which every read and write answers
`ShareNotFound`.

A snapshot conversion (`snapshot_share`) writes the next revision with the frame each slot
serves and sets the share's mode to `snapshot`, so its slots keep those frames for good.
`import_share` stores a share exported from the website under its own id, `created_at`, owner
and payload, and finds a share it imported before by its id, kind and `created_at`.

A live slot's data read keeps one `SlotCheck` per slot and card revision: the frame the slot
serves, its freshness, when the check ran and when the next one is due. `claim_slot_check`
moves a due check's times forward in one statement, so of two reads finding it due only one runs
the check. Each frame a
read serves is recorded once in `served_frame` (`record_served`), which `served_frames` reads
newest first.

`create_share` mints the id (`ids.mint_share_id`) and inserts it with `ON CONFLICT DO NOTHING`,
minting again while the id is taken, by a share made here or one imported from the website,
up to `_MINT_ATTEMPTS` times before `GeneratedIdCollision`. The store checks no write token:
it stores the digest it is given, and `read_share` returns it for the caller to compare.
Driver failures become `PresentationStoreUnavailable` with the failure's SQLSTATE.
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
from presentation_store.store_errors import GeneratedIdCollision, PresentationStoreUnavailable

__all__ = [
    "Freshness",
    "ImportedShare",
    "LiveShareStore",
    "ServedFrame",
    "ShareCard",
    "ShareIdTaken",
    "ShareKind",
    "ShareMode",
    "ShareNotFound",
    "ShareRecord",
    "ShareSlot",
    "ShareStore",
    "SlotCheck",
]

type ShareKind = Literal["card", "canvas", "search", "conversation"]
type ShareMode = Literal["live", "snapshot"]
type Freshness = Literal["current", "stale", "pending"]

# The chance that one minted id is taken is the stored share count over 62**8, about 2.2e14, so
# five tries in a row fail only on a broken minter.
_MINT_ATTEMPTS = 5

_SHARE_COLUMNS = "id, kind, mode, title, write_token_sha256, owner, payload, created_at"


class ShareNotFound(Exception):
    """No share exists at this id, or it was taken down. Maps to `not_found`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(f"no share exists at id {share_id!r}")


class ShareIdTaken(Exception):
    """An imported share's id holds another share: one made here, or a website share of
    another kind or creation time."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(f"share id {share_id!r} already holds another share")


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
class ImportedShare:
    """A website share as `import_share` stores it, at revision 1 with one slot per card."""

    id: str
    kind: ShareKind
    mode: ShareMode
    title: str | None
    owner: str | None
    payload: object
    created_at: datetime
    write_token_sha256: bytes | None
    cards: tuple[ShareCard, ...]


@dataclass(frozen=True)
class SlotCheck:
    """A live slot's newest-data check: the frame the slot serves until the next check, its
    freshness, when the check ran, and when the next one is due."""

    frame: FrameRef
    freshness: Freshness
    checked_at: datetime
    next_check_at: datetime


@dataclass(frozen=True)
class ServedFrame:
    """One frame a share's slot served: the card revision, the frame, the rendering the card
    revision recorded (`None` when unrecorded), and when a read first served it."""

    slot: int
    component_id: str
    revision: int
    frame: FrameRef
    rendering: Mapping[str, object] | None
    first_served_at: datetime


class ShareStore(Protocol):
    def create_share(
        self,
        *,
        kind: ShareKind,
        mode: ShareMode,
        title: str | None,
        write_token_sha256: bytes | None,
        cards: Sequence[ShareCard],
        payload: object = None,
    ) -> ShareRecord:
        """Store a new share at revision 1, one slot per card in order, under a minted id.
        `payload` is stored as given; `None` stores none.

        Raises:
            GeneratedIdCollision: Every minted id was taken.
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

    def replace_share(self, *, share_id: str, cards: Sequence[ShareCard]) -> ShareRecord:
        """Write the share's next revision: `cards` in slots 1 to `len(cards)`, and every
        further slot the latest revision has left empty.

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down.
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
        self, *, share_id: str, revision: int, frames: Mapping[int, FrameRef]
    ) -> ShareRecord:
        """Turn a live share at `revision` into a snapshot: write its next revision with the
        same cards, each slot in `frames` drawing that frame, and set its mode. A snapshot is
        returned as it is.

        Raises:
            ShareNotFound: No share exists at `share_id`, or it was taken down.
            EditConflict: The share's latest revision is no longer `revision`.
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def import_share(self, share: ImportedShare) -> bool:
        """Store a website share under its own id. Returns `False`, changing nothing, when the
        id already holds a share of the same kind and `created_at`, which an earlier import
        stored.

        Raises:
            ShareIdTaken: The id holds another share.
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
        """Move the check's time to `now` and its next due time to `claim_until` when it is due
        at `now`. Returns whether it did, so the caller runs the check; `False` too when no
        check is stored.

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
        slot: int,
        card: ShareCard,
        frame: FrameRef,
        rendering: Mapping[str, object] | None,
    ) -> None:
        """Record that `slot` served `frame` for `card`, once: a frame served again keeps its
        first record.

        Raises:
            PresentationStoreUnavailable: The store failed.
        """
        ...

    def served_frames(self, *, share_id: str) -> tuple[ServedFrame, ...]:
        """Every frame the share's slots served, newest first.

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
        payload: object = None,
    ) -> ShareRecord:
        def write(tx: Transaction) -> ShareRecord:
            share_id = self._insert_share(
                tx,
                kind=kind,
                mode=mode,
                title=title,
                write_token_sha256=write_token_sha256,
                payload=payload,
            )
            slots = tuple(ShareSlot(slot=n, card=card) for n, card in enumerate(cards, start=1))
            _insert_revision(tx, share_id=share_id, revision=1, slots=slots)
            return _read(tx, share_id=share_id, for_update=False)

        return self._in_transaction("creating a share", write)

    def read_share(self, *, share_id: str) -> ShareRecord:
        return self._in_transaction(
            f"reading share {share_id!r}",
            lambda tx: _read(tx, share_id=share_id, for_update=False),
        )

    def replace_share(self, *, share_id: str, cards: Sequence[ShareCard]) -> ShareRecord:
        def write(tx: Transaction) -> ShareRecord:
            current = _read(tx, share_id=share_id, for_update=True)
            count = max(len(current.slots), len(cards))
            slots = tuple(
                ShareSlot(slot=n, card=cards[n - 1] if n <= len(cards) else None)
                for n in range(1, count + 1)
            )
            _insert_revision(tx, share_id=share_id, revision=current.revision + 1, slots=slots)
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
        self, *, share_id: str, revision: int, frames: Mapping[int, FrameRef]
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
            _insert_revision(tx, share_id=share_id, revision=revision + 1, slots=slots)
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
                    "SELECT kind, created_at FROM presentation.share WHERE id = %s", (share.id,)
                )
                if held is None or (held["kind"], held["created_at"]) != (
                    share.kind,
                    share.created_at,
                ):
                    raise ShareIdTaken(share.id)
                return False
            slots = tuple(
                ShareSlot(slot=n, card=card) for n, card in enumerate(share.cards, start=1)
            )
            _insert_revision(
                tx, share_id=share.id, revision=1, slots=slots, created_at=share.created_at
            )
            return True

        return self._in_transaction(f"importing share {share.id!r}", write)

    def slot_check(self, *, share_id: str, slot: int, card: ShareCard) -> SlotCheck | None:
        def read(tx: Transaction) -> SlotCheck | None:
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
                checked_at=cast(datetime, row["checked_at"]),
                next_check_at=cast(datetime, row["next_check_at"]),
            )

        return self._in_transaction(f"reading slot {slot} of share {share_id!r}", read)

    def claim_slot_check(
        self, *, share_id: str, slot: int, card: ShareCard, now: datetime, claim_until: datetime
    ) -> bool:
        def write(tx: Transaction) -> bool:
            row = tx.fetch_one(
                "UPDATE presentation.share_slot_check SET checked_at = %s, next_check_at = %s "
                "WHERE share_id = %s AND slot = %s AND component_id = %s "
                "AND component_revision = %s AND next_check_at <= %s RETURNING slot",
                (now, claim_until, share_id, slot, card.component_id, card.revision, now),
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
        slot: int,
        card: ShareCard,
        frame: FrameRef,
        rendering: Mapping[str, object] | None,
    ) -> None:
        def write(tx: Transaction) -> None:
            tx.execute(
                "INSERT INTO presentation.served_frame (share_id, slot, component_id, "
                "component_revision, frame_spec_id, frame_content_version, rendering) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    share_id,
                    slot,
                    card.component_id,
                    card.revision,
                    frame.spec_id,
                    frame.content_version,
                    _json_param(rendering),
                ),
            )

        self._in_transaction(f"recording a frame share {share_id!r} served", write)

    def served_frames(self, *, share_id: str) -> tuple[ServedFrame, ...]:
        def read(tx: Transaction) -> tuple[ServedFrame, ...]:
            _read(tx, share_id=share_id, for_update=False)
            rows = tx.fetch_all(
                "SELECT slot, component_id, component_revision, frame_spec_id, "
                "frame_content_version, rendering, first_served_at "
                "FROM presentation.served_frame WHERE share_id = %s "
                "ORDER BY first_served_at DESC, slot",
                (share_id,),
            )
            return tuple(
                ServedFrame(
                    slot=cast(int, row["slot"]),
                    component_id=cast(str, row["component_id"]),
                    revision=cast(int, row["component_revision"]),
                    frame=FrameRef(
                        spec_id=cast(str, row["frame_spec_id"]),
                        content_version=cast(str, row["frame_content_version"]),
                    ),
                    rendering=cast("Mapping[str, object] | None", row["rendering"]),
                    first_served_at=cast(datetime, row["first_served_at"]),
                )
                for row in rows
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
        payload: object,
    ) -> str:
        """Insert the share row under the first minted id no share holds."""
        for _ in range(_MINT_ATTEMPTS):
            share_id = self._mint_id()
            row = tx.fetch_one(
                "INSERT INTO presentation.share (id, kind, mode, title, write_token_sha256, "
                "payload) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING "
                "RETURNING id",
                (share_id, kind, mode, title, write_token_sha256, _json_param(payload)),
            )
            if row is not None:
                return share_id
        raise GeneratedIdCollision(subject_kind="share")

    def _in_transaction[T](self, action: str, work: Callable[[Transaction], T]) -> T:
        """Run `work` in one transaction. `ShareNotFound`, `ShareIdTaken`, `EditConflict` and
        `GeneratedIdCollision` pass through, and any other failure becomes
        `PresentationStoreUnavailable`, raised after leaving `except` so driver context stays
        off it."""
        failure: PresentationStoreUnavailable | None = None
        result: T | None = None
        try:
            with self._store.transaction() as tx:
                result = work(tx)
        except (ShareNotFound, ShareIdTaken, EditConflict, GeneratedIdCollision):
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
) -> None:
    """Insert one revision and its slots. `created_at` left out takes the column default."""
    if created_at is None:
        tx.execute(
            "INSERT INTO presentation.share_revision (share_id, revision) VALUES (%s, %s)",
            (share_id, revision),
        )
    else:
        tx.execute(
            "INSERT INTO presentation.share_revision (share_id, revision, created_at) "
            "VALUES (%s, %s, %s)",
            (share_id, revision, created_at),
        )
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
