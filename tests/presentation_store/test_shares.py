"""Checks `LiveShareStore`: slots by revision, id minting against stored ids, take-down, snapshot
conversion, website imports, slot checks, served frames, and failure wrapping."""

from __future__ import annotations

import copy
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
        renderings={},
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

    fewer = store.replace_share(share_id=created.id, revision=1, cards=[_card("d")], renderings={})
    more = store.replace_share(
        share_id=created.id, revision=2, cards=[_card("a"), _card("b"), _card("e")], renderings={}
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

    grown = store.replace_share(
        share_id=created.id, revision=1, cards=[_card("b"), _card("c")], renderings={}
    )

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
        renderings={},
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
        store.replace_share(share_id=created.id, revision=1, cards=[_card("b")], renderings={})
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
        renderings={},
    )

    read = store.read_share(share_id=created.id)
    assert (read.kind, read.payload) == ("canvas", payload)


def test_a_snapshot_conversion_writes_the_served_frames_and_answers_a_snapshot_as_it_is():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"), _card("b"))
    emptied = store.replace_share(
        share_id=created.id, revision=1, cards=[_card("a")], renderings={}
    )
    assert _shown(emptied.slots) == [(1, "cmp_a"), (2, None)]

    snapshot = store.snapshot_share(
        share_id=created.id, revision=2, frames={1: _NEWER}, renderings={}
    )
    again = store.snapshot_share(share_id=created.id, revision=2, frames={1: _FRAME}, renderings={})

    assert (snapshot.mode, snapshot.revision) == ("snapshot", 3)
    assert snapshot.slots == (ShareSlot(slot=1, card=_card("a", frame=_NEWER)), ShareSlot(2, None))
    assert again == snapshot
    assert store.read_share(share_id=created.id) == snapshot


def test_a_snapshot_conversion_of_a_share_replaced_meanwhile_is_an_edit_conflict():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"))
    store.replace_share(share_id=created.id, revision=1, cards=[_card("b")], renderings={})

    with pytest.raises(EditConflict):
        store.snapshot_share(share_id=created.id, revision=1, frames={}, renderings={})

    assert store.read_share(share_id=created.id).mode == "live"


def test_a_replace_refuses_a_share_that_became_a_snapshot_or_moved_since_it_was_read():
    store = LiveShareStore(store=FakePresentationStore())
    live = _create(store, _card("a"))
    frozen = _create(store, _card("a"))
    # Each replace read revision 1, then another write committed before its own.
    store.replace_share(share_id=live.id, revision=1, cards=[_card("b")], renderings={})
    store.snapshot_share(share_id=frozen.id, revision=1, frames={}, renderings={})

    with pytest.raises(EditConflict):
        store.replace_share(share_id=live.id, revision=1, cards=[_card("c")], renderings={})
    with pytest.raises(ShareNotReplaceable):
        store.replace_share(share_id=frozen.id, revision=1, cards=[_card("c")], renderings={})

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
    renderings={},
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


def _timeline(
    store: LiveShareStore, share_id: str
) -> list[tuple[int, str | None, FrameRef | None]]:
    """The share's entries, oldest first, as (slot, component id, frame)."""
    entries = store.served_frames(share_id=share_id, limit=100).entries
    return [(entry.slot, entry.component_id, entry.frame) for entry in reversed(entries)]


def _shown_at(store: LiveShareStore, share_id: str, *, slot: int, at: datetime) -> str | None:
    """The component `slot` showed at `at`: its newest entry from `at` or before."""
    entries = store.served_frames(share_id=share_id, limit=100).entries
    return next(e.component_id for e in entries if e.slot == slot and e.first_served_at <= at)


def test_a_create_a_replace_and_a_snapshot_each_record_what_their_slots_show_at_their_revision():
    store = LiveShareStore(store=FakePresentationStore())
    a, b, c = _card("a"), _card("b"), _card("c")
    rendering = {"without_party_totals": True}
    created = store.create_share(
        kind="canvas",
        mode="live",
        title=None,
        write_token_sha256=None,
        cards=[a, _card("note", frame=None), b],
        renderings={("cmp_a", 1): rendering},
    )
    # Slot 1's check moved to a newer frame, which no read has recorded yet.
    store.record_slot_check(
        share_id=created.id,
        slot=1,
        card=a,
        check=SlotCheck(frame=_NEWER, freshness="current", checked_at=_AT, next_check_at=_AT),
    )
    after_create = _timeline(store, created.id)

    replaced = store.replace_share(share_id=created.id, revision=1, cards=[a, c], renderings={})
    after_replace = _timeline(store, created.id)[len(after_create) :]

    snapshot = store.snapshot_share(
        share_id=created.id, revision=2, frames={1: _NEWER, 2: _NEWER}, renderings={}
    )
    after_snapshot = _timeline(store, created.id)[len(after_create) + len(after_replace) :]

    assert after_create == [(1, "cmp_a", _FRAME), (2, None, None), (3, "cmp_b", _FRAME)]
    # The kept card records its check's frame; slot 3 empties.
    assert after_replace == [(1, "cmp_a", _NEWER), (2, "cmp_c", _FRAME), (3, None, None)]
    # Slot 1 already shows what the snapshot keeps, so only slot 2 changes.
    assert after_snapshot == [(2, "cmp_c", _NEWER)]
    entries = store.served_frames(share_id=created.id, limit=100).entries
    assert [entry.first_served_at for entry in entries] == [snapshot.revised_at] + [
        replaced.revised_at
    ] * 3 + [created.revised_at] * 3
    assert [
        entry.rendering for entry in entries if entry.first_served_at == created.revised_at
    ] == [
        None,
        None,
        rendering,
    ]


def test_two_replaces_record_in_commit_order_and_as_of_between_them_selects_the_first():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"))

    first = store.replace_share(share_id=created.id, revision=1, cards=[_card("b")], renderings={})
    second = store.replace_share(share_id=created.id, revision=2, cards=[_card("c")], renderings={})

    assert [component for _, component, _ in _timeline(store, created.id)] == [
        "cmp_a",
        "cmp_b",
        "cmp_c",
    ]
    assert created.revised_at < first.revised_at < second.revised_at
    assert _shown_at(
        store, created.id, slot=1, at=second.revised_at - timedelta(microseconds=1)
    ) == ("cmp_b")


def _write(store: LiveShareStore, share_id: str, write: str) -> None:
    if write == "replace":
        store.replace_share(share_id=share_id, revision=1, cards=[_card("b")], renderings={})
    else:
        store.snapshot_share(share_id=share_id, revision=1, frames={1: _NEWER}, renderings={})


@pytest.mark.parametrize("write", ["create", "replace", "snapshot"])
def test_a_write_whose_history_insert_fails_commits_nothing(write: str):
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake)
    created = _create(store, _card("a")) if write != "create" else None
    before = copy.deepcopy(fake.state.share_revisions), list(fake.state.served_frames)
    fake.fail_before = lambda sql, _: sql.startswith("INSERT INTO presentation.served_frame")

    with pytest.raises(PresentationStoreUnavailable):
        if created is None:
            _create(store, _card("a"))
        else:
            _write(store, created.id, write)

    fake.fail_before = None
    assert (fake.state.share_revisions, fake.state.served_frames) == before
    if created is not None:
        assert store.read_share(share_id=created.id) == created


def test_a_slot_read_records_what_it_shows_each_time_it_changes_read_newest_first():
    store = LiveShareStore(store=FakePresentationStore())
    share = _create(store, _card("a"))
    rendering = {"without_party_totals": False}

    for frame in (_FRAME, _NEWER, _NEWER, _FRAME):
        store.record_served(
            share_id=share.id, revision=1, slot=1, card=_card("a"), frame=frame, rendering=rendering
        )

    served = store.served_frames(share_id=share.id, limit=10).entries
    assert [(entry.component_id, entry.frame, entry.rendering) for entry in served] == [
        ("cmp_a", _FRAME, rendering),
        ("cmp_a", _NEWER, rendering),
        ("cmp_a", _FRAME, None),
    ]
    assert served[0].first_served_at >= served[1].first_served_at
    store.delete_share(share_id=share.id)
    with pytest.raises(ShareNotFound):
        store.served_frames(share_id=share.id, limit=10)


@pytest.mark.parametrize(
    ("write", "served", "written"),
    [("replace", _NEWER, (1, "cmp_b", _FRAME)), ("snapshot", _FRAME, (1, "cmp_a", _NEWER))],
)
def test_a_read_of_a_revision_a_later_write_has_moved_past_records_nothing(
    write: str, served: FrameRef, written: tuple[int, str, FrameRef]
):
    """A snapshot conversion keeps the card in its slot, so the read is fenced on the revision
    it served, not on the card."""
    store = LiveShareStore(store=FakePresentationStore())
    share = _create(store, _card("a"))
    _write(store, share.id, write)

    # The read loaded revision 1 before the write committed, and records after it.
    store.record_served(
        share_id=share.id, revision=1, slot=1, card=_card("a"), frame=served, rendering=None
    )

    assert _timeline(store, share.id) == [(1, "cmp_a", _FRAME), written]


def test_a_slot_read_locks_the_share_only_when_what_the_slot_shows_changes():
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake)
    share = _create(store, _card("a"))

    def locks_taken(frame: FrameRef) -> int:
        fake.statement_log.clear()
        store.record_served(
            share_id=share.id, revision=1, slot=1, card=_card("a"), frame=frame, rendering=None
        )
        return sum("FOR NO KEY UPDATE" in sql for sql in fake.statement_log)

    assert [locks_taken(frame) for frame in (_FRAME, _NEWER, _NEWER, _FRAME)] == [0, 1, 0, 1]
    assert len(store.served_frames(share_id=share.id, limit=10).entries) == 3
