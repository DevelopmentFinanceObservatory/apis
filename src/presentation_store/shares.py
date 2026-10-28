"""Share records over `presentation.share`, `share_revision` and `share_slot` (migration 0012).

A share publishes cards under one `/s/<id>` id. Each revision lists the share's slots, numbered
from 1, and a slot holds one card revision or none. A create writes revision 1 with one slot per
card. A replace writes the next revision with every slot the share has had, so a replacement
with fewer cards leaves the later slots empty and keeps their URLs. Revision rows are never
updated or deleted. A delete sets `deleted_at`, after which every read and write answers
`ShareNotFound`.

`create_share` mints the id (`ids.mint_share_id`) and inserts it with `ON CONFLICT DO NOTHING`,
minting again while the id is taken, by a share made here or one imported from the website,
up to `_MINT_ATTEMPTS` times before `GeneratedIdCollision`. The store checks no write token:
it stores the digest it is given, and `read_share` returns it for the caller to compare.
Driver failures become `PresentationStoreUnavailable` with the failure's SQLSTATE.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, cast

from assembler.ports import FrameRef
from dataset_store import DatasetStore
from dataset_store.store import Transaction
from presentation_store.ids import mint_share_id
from presentation_store.store_errors import GeneratedIdCollision, PresentationStoreUnavailable

__all__ = [
    "LiveShareStore",
    "ShareCard",
    "ShareKind",
    "ShareMode",
    "ShareNotFound",
    "ShareRecord",
    "ShareSlot",
    "ShareStore",
]

type ShareKind = Literal["card", "canvas", "search", "conversation"]
type ShareMode = Literal["live", "snapshot"]

# The chance that one minted id is taken is the stored share count over 62**8, about 2.2e14, so
# five tries in a row fail only on a broken minter.
_MINT_ATTEMPTS = 5

_SHARE_COLUMNS = "id, kind, mode, title, write_token_sha256, owner, payload, created_at"


class ShareNotFound(Exception):
    """No share exists at this id, or it was taken down. Maps to `not_found`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(f"no share exists at id {share_id!r}")


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


class ShareStore(Protocol):
    def create_share(
        self,
        *,
        kind: ShareKind,
        mode: ShareMode,
        title: str | None,
        write_token_sha256: bytes | None,
        cards: Sequence[ShareCard],
    ) -> ShareRecord:
        """Store a new share at revision 1, one slot per card in order, under a minted id.

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
    ) -> ShareRecord:
        def write(tx: Transaction) -> ShareRecord:
            share_id = self._insert_share(
                tx, kind=kind, mode=mode, title=title, write_token_sha256=write_token_sha256
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

    def _insert_share(
        self,
        tx: Transaction,
        *,
        kind: ShareKind,
        mode: ShareMode,
        title: str | None,
        write_token_sha256: bytes | None,
    ) -> str:
        """Insert the share row under the first minted id no share holds."""
        for _ in range(_MINT_ATTEMPTS):
            share_id = self._mint_id()
            row = tx.fetch_one(
                "INSERT INTO presentation.share (id, kind, mode, title, write_token_sha256) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING RETURNING id",
                (share_id, kind, mode, title, write_token_sha256),
            )
            if row is not None:
                return share_id
        raise GeneratedIdCollision(subject_kind="share")

    def _in_transaction[T](self, action: str, work: Callable[[Transaction], T]) -> T:
        """Run `work` in one transaction. `ShareNotFound` and `GeneratedIdCollision` pass
        through, and any other failure becomes `PresentationStoreUnavailable`, raised after
        leaving `except` so driver context stays off it."""
        failure: PresentationStoreUnavailable | None = None
        result: T | None = None
        try:
            with self._store.transaction() as tx:
                result = work(tx)
        except (ShareNotFound, GeneratedIdCollision):
            raise
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(f"{action} failed", exc)
        if failure is not None:
            raise failure
        return cast("T", result)


def _insert_revision(
    tx: Transaction, *, share_id: str, revision: int, slots: Sequence[ShareSlot]
) -> None:
    tx.execute(
        "INSERT INTO presentation.share_revision (share_id, revision) VALUES (%s, %s)",
        (share_id, revision),
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
