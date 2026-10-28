"""Checks `LiveShareStore`: slots by revision, id minting against stored ids, take-down, and
failure wrapping."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from assembler.ports import FrameRef
from assembler.write_token import write_token_digest
from presentation_store.shares import LiveShareStore, ShareCard, ShareNotFound, ShareSlot
from presentation_store.store_errors import GeneratedIdCollision, PresentationStoreUnavailable

from .conftest import FakePresentationStore, SqlstateExplodingFakePresentationStore

_FRAME = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)


def _card(name: str, *, revision: int = 1, frame: FrameRef | None = _FRAME) -> ShareCard:
    return ShareCard(component_id=f"cmp_{name}", revision=revision, frame=frame)


def _create(store: LiveShareStore, *cards: ShareCard, mode="live", digest=b"d" * 32):
    return store.create_share(
        kind="card", mode=mode, title="Aid", write_token_sha256=digest, cards=cards
    )


def _shown(slots: tuple[ShareSlot, ...]) -> list[tuple[int, str | None]]:
    return [(slot.slot, slot.card.component_id if slot.card else None) for slot in slots]


def test_a_create_stores_revision_one_with_a_slot_per_card_in_order():
    store = LiveShareStore(store=FakePresentationStore())
    note = _card("note", frame=None)

    created = _create(store, _card("a"), note, _card("a"))

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

    fewer = store.replace_share(share_id=created.id, cards=[_card("d")])
    more = store.replace_share(share_id=created.id, cards=[_card("a"), _card("b"), _card("e")])

    assert (fewer.revision, _shown(fewer.slots)) == (2, [(1, "cmp_d"), (2, None), (3, None)])
    assert (more.revision, _shown(more.slots)) == (
        3,
        [(1, "cmp_a"), (2, "cmp_b"), (3, "cmp_e")],
    )
    assert more.created_at == created.created_at


def test_a_replace_past_the_slots_the_share_has_adds_slots():
    store = LiveShareStore(store=FakePresentationStore())
    created = _create(store, _card("a"))

    grown = store.replace_share(share_id=created.id, cards=[_card("b"), _card("c")])

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
        store.replace_share(share_id=created.id, cards=[_card("b")])
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
