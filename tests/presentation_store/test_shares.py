"""Checks `LiveShareStore`: slots by revision, id minting against stored ids, take-down, snapshot
conversion, website imports, slot checks, served frames, and failure wrapping."""

from __future__ import annotations

import dataclasses
import secrets
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest

from assembler.ports import EditConflict, FrameRef
from assembler.write_token import write_token_digest
from presentation_store.shares import (
    ImportedShare,
    LiveShareStore,
    PriorShareCreate,
    ShareCard,
    ShareIdTaken,
    ShareNotFound,
    ShareNotReplaceable,
    ShareSlot,
    ShareWriteTokenInUse,
    SlotCheck,
)
from presentation_store.store_errors import GeneratedIdCollision, PresentationStoreUnavailable

from .conftest import FakePresentationStore, SqlstateExplodingFakePresentationStore

_FRAME = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)
_NEWER = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)
_AT = datetime(2026, 9, 1, 10, 0, 0, 1, tzinfo=UTC)


def _card(name: str, *, revision: int = 1, frame: FrameRef | None = _FRAME) -> ShareCard:
    return ShareCard(component_id=f"cmp_{name}", revision=revision, frame=frame)


def _create(
    store: LiveShareStore,
    *cards: ShareCard,
    mode="live",
    digest: bytes | None | Literal["fresh"] = "fresh",
):
    """Create a card share, under a fresh token digest unless `digest` names one."""
    return store.create_share(
        kind="card",
        mode=mode,
        title="Aid",
        write_token_sha256=secrets.token_bytes(32) if digest == "fresh" else digest,
        cards=cards,
    )


def _shown(slots: tuple[ShareSlot, ...]) -> list[tuple[int, str | None]]:
    return [(slot.slot, slot.card.component_id if slot.card else None) for slot in slots]


def test_a_create_stores_revision_one_with_a_slot_per_card_in_order():
    store = LiveShareStore(store=FakePresentationStore())
    note = _card("note", frame=None)

    created = _create(store, _card("a"), note, _card("a"), digest=b"d" * 32)

    assert created.revision == 1
    assert created.mode == "live"
    assert created.title == "Aid"
    assert created.write_token_sha256 == b"d" * 32
    assert _shown(created.slots) == [(1, "cmp_a"), (2, "cmp_note"), (3, "cmp_a")]
    assert created.slots[0].card == _card("a")
    assert created.slots[1].card == note
    assert store.read_share(share_id=created.id) == created


def test_a_replace_with_fewer_cards_keeps_the_later_slots_empty_and_more_cards_fill_them():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"), _card("b"), _card("c"))

    fewer = store.replace_share(share_id=created.id, revision=1, cards=[_card("d")])
    more = store.replace_share(
        share_id=created.id, revision=2, cards=[_card("a"), _card("b"), _card("e")]
    )

    assert (fewer.revision, _shown(fewer.slots)) == (2, [(1, "cmp_d"), (2, None), (3, None)])
    assert (more.revision, _shown(more.slots)) == (
        3,
        [(1, "cmp_a"), (2, "cmp_b"), (3, "cmp_e")],
    )
    assert more.created_at == created.created_at


def test_a_replace_past_the_slots_the_share_has_adds_slots():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"))

    grown = store.replace_share(share_id=created.id, revision=1, cards=[_card("b"), _card("c")])

    assert _shown(grown.slots) == [(1, "cmp_b"), (2, "cmp_c")]


def test_the_mint_retries_an_id_another_share_holds_such_as_an_imported_one():
    fake = FakePresentationStore()
    first = LiveShareStore(store=fake, mint_id=lambda: "k3Qx8vTa")
    _create(first, _card("a"))
    minted: Iterator[str] = iter(["k3Qx8vTa", "K3qX8VtA"])
    store = LiveShareStore(store=fake, mint_id=lambda: next(minted))

    created = _create(store, _card("b"))

    # Ids are case-sensitive, so the second differs from the first.
    assert created.id == "K3qX8VtA"
    assert _shown(store.read_share(share_id="k3Qx8vTa").slots) == [(1, "cmp_a")]


def test_a_create_holding_another_shares_digest_is_refused_and_prior_create_finds_that_share():
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake)
    digest, request = write_token_digest("shw_example"), b"r" * 32
    first = store.create_share(
        kind="card",
        mode="live",
        title=None,
        write_token_sha256=digest,
        cards=[_card("a")],
        create_request_sha256=request,
    )

    with pytest.raises(ShareWriteTokenInUse):
        _create(store, _card("b"), digest=digest)
    store.delete_share(share_id=first.id)

    assert list(fake.state.shares) == [first.id]
    # A share taken down keeps its token, so a retry finds it.
    assert store.prior_create(write_token_sha256=digest) == PriorShareCreate(
        share_id=first.id, create_request_sha256=request
    )
    assert store.prior_create(write_token_sha256=b"o" * 32) is None


def test_a_mint_that_only_finds_taken_ids_refuses_a_generated_id_collision():
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake, mint_id=lambda: "k3Qx8vTa")
    _create(store, _card("a"))

    with pytest.raises(GeneratedIdCollision) as raised:
        _create(store, _card("b"))

    assert raised.value.subject_kind == "share"
    assert list(fake.state.shares) == ["k3Qx8vTa"]


def test_a_deleted_share_is_absent_to_every_read_and_write():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"))

    store.delete_share(share_id=created.id)

    with pytest.raises(ShareNotFound):
        store.read_share(share_id=created.id)
    with pytest.raises(ShareNotFound):
        store.replace_share(share_id=created.id, revision=1, cards=[_card("b")])
    with pytest.raises(ShareNotFound):
        store.delete_share(share_id=created.id)


def test_an_unknown_id_is_not_found():
    store = LiveShareStore(store=FakePresentationStore())

    with pytest.raises(ShareNotFound):
        store.read_share(share_id="Zz9Zz9Zz")


def test_a_tokenless_share_reads_back_with_no_digest():
    store = LiveShareStore(store=FakePresentationStore())

    created = _create(store, _card("a"), digest=None)

    assert store.read_share(share_id=created.id).write_token_sha256 is None


def test_the_digest_round_trips_as_bytes():
    store = LiveShareStore(store=FakePresentationStore())
    digest = write_token_digest("shw_example")

    created = _create(store, _card("a"), digest=digest)

    assert store.read_share(share_id=created.id).write_token_sha256 == digest


def test_a_driver_failure_becomes_presentation_store_unavailable_with_its_sqlstate():
    store = LiveShareStore(store=SqlstateExplodingFakePresentationStore(sqlstate="40001"))

    with pytest.raises(PresentationStoreUnavailable) as raised:
        store.read_share(share_id="Zz9Zz9Zz")

    assert raised.value.sqlstate == "40001"
    assert raised.value.__context__ is None


def test_a_create_stores_its_kind_and_payload_as_given():
    store = LiveShareStore(store=FakePresentationStore())
    payload = {"widgets": [{"kind": "chart", "slot": 0}], "pinnedIds": []}

    created = store.create_share(
        kind="canvas",
        mode="live",
        title=None,
        write_token_sha256=None,
        cards=[_card("a")],
        payload=payload,
    )

    read = store.read_share(share_id=created.id)
    assert (read.kind, read.payload) == ("canvas", payload)


def test_a_snapshot_conversion_writes_the_served_frames_and_answers_a_snapshot_as_it_is():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"), _card("b"))
    emptied = store.replace_share(share_id=created.id, revision=1, cards=[_card("a")])
    assert _shown(emptied.slots) == [(1, "cmp_a"), (2, None)]

    snapshot = store.snapshot_share(share_id=created.id, revision=2, frames={1: _NEWER})
    again = store.snapshot_share(share_id=created.id, revision=2, frames={1: _FRAME})

    assert (snapshot.mode, snapshot.revision) == ("snapshot", 3)
    assert snapshot.slots == (ShareSlot(slot=1, card=_card("a", frame=_NEWER)), ShareSlot(2, None))
    assert again == snapshot
    assert store.read_share(share_id=created.id) == snapshot


def test_a_snapshot_conversion_of_a_share_replaced_meanwhile_is_an_edit_conflict():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"))
    store.replace_share(share_id=created.id, revision=1, cards=[_card("b")])

    with pytest.raises(EditConflict):
        store.snapshot_share(share_id=created.id, revision=1, frames={})

    assert store.read_share(share_id=created.id).mode == "live"


def test_a_replace_refuses_a_share_that_became_a_snapshot_or_moved_since_it_was_read():
    store = LiveShareStore(store=FakePresentationStore())
    live = _create(store, _card("a"))
    frozen = _create(store, _card("a"))
    # Each replace read revision 1, then another write committed before its own.
    store.replace_share(share_id=live.id, revision=1, cards=[_card("b")])
    store.snapshot_share(share_id=frozen.id, revision=1, frames={})

    with pytest.raises(EditConflict):
        store.replace_share(share_id=live.id, revision=1, cards=[_card("c")])
    with pytest.raises(ShareNotReplaceable):
        store.replace_share(share_id=frozen.id, revision=1, cards=[_card("c")])

    assert _shown(store.read_share(share_id=live.id).slots) == [(1, "cmp_b")]
    assert store.read_share(share_id=frozen.id).revision == 2


_IMPORTED = ImportedShare(
    id="k3Qx8vTa",
    kind="canvas",
    mode="snapshot",
    title="Aid to Kenya",
    owner="user_1",
    payload={"widgets": []},
    created_at=_AT,
    write_token_sha256=b"t" * 32,
    cards=(_card("a"),),
)


def _imported(**changes: Any) -> ImportedShare:
    return dataclasses.replace(_IMPORTED, **changes)


def test_an_import_keeps_the_website_id_owner_and_creation_time_and_a_second_changes_nothing():
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake)

    assert store.import_share(_imported()) is True
    stored = store.read_share(share_id="k3Qx8vTa")
    assert store.import_share(_imported(write_token_sha256=b"u" * 32, title="Other")) is False

    assert store.read_share(share_id="k3Qx8vTa") == stored
    assert (stored.owner, stored.created_at, stored.revised_at) == ("user_1", _AT, _AT)
    assert (stored.title, stored.payload, stored.write_token_sha256) == (
        "Aid to Kenya",
        {"widgets": []},
        b"t" * 32,
    )
    assert _shown(stored.slots) == [(1, "cmp_a")]


@pytest.mark.parametrize(
    "held",
    [
        {"kind": "card"},
        {"created_at": _AT + timedelta(microseconds=1)},
    ],
    ids=["other-kind", "other-creation-time"],
)
def test_an_import_onto_an_id_holding_another_share_refuses(held: dict[str, Any]):
    store = LiveShareStore(store=FakePresentationStore())
    store.import_share(_imported(**held))

    with pytest.raises(ShareIdTaken):
        store.import_share(_imported())


def test_a_slot_check_is_claimed_once_when_due_and_reset_by_another_card():
    store = LiveShareStore(store=FakePresentationStore())
    share = _create(store, _card("a"))
    card = _card("a")
    due = _AT + timedelta(seconds=30)
    check = SlotCheck(frame=_NEWER, freshness="current", checked_at=_AT, next_check_at=due)
    until = due + timedelta(minutes=1)

    def claim(now):
        return store.claim_slot_check(
            share_id=share.id, slot=1, card=card, now=now, claim_until=until
        )

    store.record_slot_check(share_id=share.id, slot=1, card=card, check=check)
    early = claim(due - timedelta(seconds=1))
    first = claim(due)
    second = claim(due)
    expired = claim(until)

    assert (early, first, second, expired) == (False, True, False, True)
    # The claim moves only the due time; readers that lose it see the last check as it was.
    assert store.slot_check(share_id=share.id, slot=1, card=card) == dataclasses.replace(
        check, next_check_at=until
    )
    assert store.slot_check(share_id=share.id, slot=1, card=_card("a", revision=2)) is None


def test_a_slots_first_check_is_claimed_once_as_pending_with_no_check_time():
    store = LiveShareStore(store=FakePresentationStore())
    share = _create(store, _card("a"))
    card = _card("a")
    until = _AT + timedelta(minutes=1)

    claims = [
        store.claim_slot_check(share_id=share.id, slot=1, card=card, now=_AT, claim_until=until)
        for _ in range(2)
    ]

    assert claims == [True, False]
    assert store.slot_check(share_id=share.id, slot=1, card=card) == SlotCheck(
        frame=_FRAME, freshness="pending", checked_at=None, next_check_at=until
    )


def test_a_slot_records_what_it_shows_each_time_it_changes_read_newest_first():
    store = LiveShareStore(store=FakePresentationStore())
    share = _create(store, _card("a"))
    rendering = {"without_party_totals": False}

    for frame in (_FRAME, _FRAME, _NEWER, _FRAME):
        store.record_served(
            share_id=share.id, slot=1, card=_card("a"), frame=frame, rendering=rendering
        )
    store.record_emptied(share_id=share.id, slot=1)
    store.record_emptied(share_id=share.id, slot=1)

    served = store.served_frames(share_id=share.id, limit=10).entries
    assert [(entry.component_id, entry.frame, entry.rendering) for entry in served] == [
        (None, None, None),
        ("cmp_a", _FRAME, rendering),
        ("cmp_a", _NEWER, rendering),
        ("cmp_a", _FRAME, rendering),
    ]
    assert served[0].first_served_at >= served[1].first_served_at
    store.delete_share(share_id=share.id)
    with pytest.raises(ShareNotFound):
        store.served_frames(share_id=share.id, limit=10)


def test_a_slot_read_locks_the_share_only_when_what_the_slot_shows_changes():
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake)
    share = _create(store, _card("a"))

    def locks_taken(frame: FrameRef | None) -> int:
        fake.statement_log.clear()
        if frame is None:
            store.record_emptied(share_id=share.id, slot=1)
        else:
            store.record_served(
                share_id=share.id, slot=1, card=_card("a"), frame=frame, rendering=None
            )
        return sum("FOR NO KEY UPDATE" in sql for sql in fake.statement_log)

    assert [locks_taken(frame) for frame in (_FRAME, _FRAME, _NEWER, _NEWER, None, None)] == [
        1,
        0,
        1,
        0,
        1,
        0,
    ]
    assert len(store.served_frames(share_id=share.id, limit=10).entries) == 3
