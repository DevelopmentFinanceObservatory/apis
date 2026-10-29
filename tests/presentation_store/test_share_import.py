"""Checks `import_website_share` over the website's export: every stored case
`docs/share-export.md` places, slot numbering, `frame: null` slots, the write token digest,
idempotence, and refused shares that store nothing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from pydantic import ValidationError

from assembler.errors import FrameUnavailable, RevisionNotFound
from assembler.ports import DraftNotFound, FrameRef, RevisionRecord
from dfo_platform.caller import Caller
from presentation_store.share_import import ShareImport, SlotDrawsNoFrame, import_website_share
from presentation_store.shares import (
    MAX_SHARE_PAYLOAD_BYTES,
    LiveShareStore,
    ShareIdTaken,
    ShareNotFound,
)

from .conftest import FakePresentationStore

_DATASET = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)
_REFRESHED = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)
_DERIVED = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "e" * 32)
_UNPINNABLE = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "f" * 32)


def _rendering(component_id: str, revision: int) -> dict[str, object]:
    return {"rendered": f"{component_id}@{revision}"}


class _Revisions:
    """The card revisions the service holds: `cmp_chart` 1 and 2 on `_DATASET`, `cmp_derived` 1
    on `_DERIVED`, each with the rendering `_rendering` names, and `cmp_note` 1, which draws no
    frame."""

    _FRAMES: ClassVar[dict[tuple[str, int], FrameRef]] = {
        ("cmp_chart", 1): _DATASET,
        ("cmp_chart", 2): _DATASET,
        ("cmp_derived", 1): _DERIVED,
    }

    def read_revision(self, *, component_id: str, revision: int) -> RevisionRecord | None:
        rendering = None
        if (component_id, revision) == ("cmp_note", 1):
            intent: dict[str, object] = {"type": "note_narration", "text": "Aid rose."}
        elif (frame := self._FRAMES.get((component_id, revision))) is not None:
            intent = {
                "type": "line_chart",
                "frame_ref": {"spec_id": frame.spec_id, "content_version": frame.content_version},
            }
            rendering = _rendering(component_id, revision)
        else:
            return None
        return RevisionRecord(
            component_id=component_id,
            revision=revision,
            stored_intent=intent,
            resolved_payload={},
            catalog_version="2026.8.0",
            schema_version=1,
            formatter_profile={},
            created_at=datetime(2026, 8, 1, tzinfo=UTC),
            frame_rendering=rendering,
        )

    def read_current(self, *, component_id: str) -> RevisionRecord | None:
        if component_id in {key[0] for key in self._FRAMES} | {"cmp_note"}:
            return self.read_revision(component_id=component_id, revision=1)
        raise DraftNotFound(subject_kind="component", subject_id=component_id)


class _Pins:
    """A frame store recording each pin, refusing `_UNPINNABLE`."""

    def __init__(self) -> None:
        self.pins: list[tuple[FrameRef, str]] = []

    def pin(self, *, frame_ref: FrameRef, component_id: str, caller: Caller | None = None) -> bool:
        if frame_ref == _UNPINNABLE:
            raise FrameUnavailable(reason="expired", detail=f"{frame_ref} is gone")
        self.pins.append((frame_ref, component_id))
        return True


class _NoPins:
    """A frame store an import must not pin through."""

    def pin(self, *, frame_ref: FrameRef, component_id: str, caller: Caller | None = None) -> bool:
        raise AssertionError(f"{component_id} pinned {frame_ref}")


def _slot(component_id: str, revision: int, frame: FrameRef | None) -> dict[str, object]:
    return {
        "component_id": component_id,
        "revision": revision,
        "frame": (
            {"spec_id": frame.spec_id, "content_version": frame.content_version}
            if frame is not None
            else None
        ),
    }


def _digest(share_id: str) -> str:
    """The hex digest of the token the website holds for `share_id`, one per share, since no
    two shares hold one digest."""
    return hashlib.sha256(f"shw_{share_id}".encode()).hexdigest()


_DIGEST = _digest("cardLive")


def _line(share_id: str, kind: str, mode: str, **members: object) -> dict[str, object]:
    line: dict[str, object] = {
        "id": share_id,
        "kind": kind,
        "mode": mode,
        "owner": "user_1",
        "created_at": "2026-09-01T10:00:00.000001Z",
        "title": f"{kind} {mode}",
        "slots": [],
        "payload": {"held": share_id},
        "write_token_sha256": _digest(share_id),
    }
    line.update(members)
    return line


# One line per stored case the export's table places, plus a guest's share.
_CASES = {
    # Search, mode live or null.
    "srchLive": _line("srchLive", "search", "live", payload={"query": "aid", "parts": []}),
    # Search, mode snapshot: each part keeps its content_version.
    "srchSnap": _line(
        "srchSnap",
        "search",
        "snapshot",
        payload={"query": "aid", "parts": [{"part_index": 0, "content_version": "cv_1"}]},
    ),
    # Canvas, live: every data card is held, one with no frame of its own.
    "canvLive": _line(
        "canvLive",
        "canvas",
        "live",
        slots=[_slot("cmp_chart", 2, None), _slot("cmp_derived", 1, _DERIVED)],
        payload={"widgets": [{"id": "w1", "slot": 0}, {"id": "w2", "slot": 1}]},
    ),
    # Canvas, snapshot or null: the slot keeps the frame the website showed.
    "canvSnap": _line("canvSnap", "canvas", "snapshot", slots=[_slot("cmp_chart", 1, _REFRESHED)]),
    # Canvas, snapshot with no card carrying a complete ref: no slot, numbers inline.
    "canvNoRf": _line("canvNoRf", "canvas", "snapshot", payload={"widgets": [{"value": 3}]}),
    # Card, live.
    "cardLive": _line("cardLive", "card", "live", slots=[_slot("cmp_chart", 1, _DATASET)]),
    # Conversation, snapshot or null.
    "convSnap": _line("convSnap", "conversation", "snapshot", payload={"messages": []}),
    # A guest's share: stored with no token.
    "guestSnp": _line(
        "guestSnp",
        "card",
        "snapshot",
        owner=None,
        write_token_sha256=None,
        slots=[_slot("cmp_chart", 1, _DATASET)],
    ),
}


def _import(
    store: LiveShareStore, line: dict[str, object], *, pins: _Pins | _NoPins | None = None
) -> bool:
    body = {key: value for key, value in line.items() if key != "id"}
    return import_website_share(
        str(line["id"]),
        ShareImport.model_validate(body),
        share_store=store,
        revision_store=_Revisions(),
        frame_store=pins or _Pins(),
    )


def test_every_exported_case_imports_under_its_id_and_a_second_run_changes_nothing():
    store = LiveShareStore(store=FakePresentationStore())

    first = [_import(store, line) for line in _CASES.values()]
    after_first = {share_id: store.read_share(share_id=share_id) for share_id in _CASES}
    # The first import wins: a re-run stating other content still finds it, and pins nothing.
    again = [
        _import(store, {**line, "title": "Changed", "payload": None}, pins=_NoPins())
        for line in _CASES.values()
    ]

    assert (first, again) == ([True] * 8, [False] * 8)
    assert {share_id: store.read_share(share_id=share_id) for share_id in _CASES} == after_first
    for share_id, line in _CASES.items():
        share = after_first[share_id]
        assert (share.kind, share.mode, share.title, share.owner, share.payload) == (
            line["kind"],
            line["mode"],
            line["title"],
            line["owner"],
            line["payload"],
        )
        assert share.created_at == datetime(2026, 9, 1, 10, 0, 0, 1, tzinfo=UTC)
        assert share.revision == 1
    assert [slot.slot for slot in after_first["canvLive"].slots] == [1, 2]
    assert after_first["srchLive"].slots == after_first["canvNoRf"].slots == ()
    # Each slot's history holds one entry from the share's creation, with its card's rendering.
    for share_id, share in after_first.items():
        entries = store.served_frames(share_id=share_id, limit=100).entries
        assert [
            (entry.slot, entry.component_id, entry.revision, entry.rendering)
            for entry in reversed(entries)
        ] == [
            (
                slot.slot,
                card.component_id,
                card.revision,
                _rendering(card.component_id, card.revision),
            )
            for slot in share.slots
            if (card := slot.card) is not None
        ]
        assert {entry.first_served_at for entry in entries} <= {share.created_at}
    assert [
        entry.frame for entry in store.served_frames(share_id="canvLive", limit=100).entries
    ] == [_DERIVED, _DATASET]


def test_a_signed_in_users_share_keeps_the_digest_sent_and_a_guests_holds_none():
    store = LiveShareStore(store=FakePresentationStore())

    _import(store, _CASES["cardLive"])
    _import(store, _CASES["guestSnp"])

    assert store.read_share(share_id="cardLive").write_token_sha256 == bytes.fromhex(_DIGEST)
    assert store.read_share(share_id="guestSnp").write_token_sha256 is None


def test_export_slot_index_i_is_slot_i_plus_one_and_a_null_frame_takes_the_revisions():
    store = LiveShareStore(store=FakePresentationStore())

    _import(store, _CASES["canvLive"])
    _import(store, _CASES["canvSnap"])

    live = store.read_share(share_id="canvLive")
    assert [(slot.slot, slot.card) for slot in live.slots] == [
        (1, live.slots[0].card),
        (2, live.slots[1].card),
    ]
    assert live.slots[0].card is not None and live.slots[1].card is not None
    assert (live.slots[0].card.component_id, live.slots[0].card.frame) == ("cmp_chart", _DATASET)
    assert (live.slots[1].card.component_id, live.slots[1].card.frame) == ("cmp_derived", _DERIVED)
    snapshot = store.read_share(share_id="canvSnap").slots[0].card
    assert snapshot is not None and snapshot.frame == _REFRESHED


@pytest.mark.parametrize(
    "line",
    [
        {**_CASES["cardLive"], "kind": "deck"},
        {**_CASES["cardLive"], "mode": None},
        {**_CASES["convSnap"], "mode": "live"},
        {**_CASES["srchLive"], "slots": [_slot("cmp_chart", 1, _DATASET)]},
        {**_CASES["cardLive"], "created_at": "2026-09-01T10:00:00"},
        {**_CASES["canvNoRf"], "payload": {"text": "x" * MAX_SHARE_PAYLOAD_BYTES}},
        {**_CASES["guestSnp"], "write_token_sha256": _DIGEST},
        {**_CASES["cardLive"], "write_token_sha256": _DIGEST.upper()},
        {**_CASES["cardLive"], "write_token_sha256": _DIGEST[:63]},
        {key: value for key, value in _CASES["cardLive"].items() if key != "write_token_sha256"},
    ],
    ids=[
        "unknown-kind",
        "no-mode",
        "live-conversation",
        "search-with-slots",
        "naive-created-at",
        "payload-over-the-bound",
        "digest-without-owner",
        "uppercase-digest",
        "short-digest",
        "no-digest",
    ],
)
def test_a_body_that_is_not_an_export_line_is_refused(line: dict[str, object]):
    body = {key: value for key, value in line.items() if key != "id"}

    with pytest.raises(ValidationError):
        ShareImport.model_validate(body)


@pytest.mark.parametrize(
    ("line", "refusal"),
    [
        ({**_CASES["cardLive"], "slots": [_slot("cmp_absent", 1, _DATASET)]}, DraftNotFound),
        ({**_CASES["cardLive"], "slots": [_slot("cmp_chart", 9, _DATASET)]}, RevisionNotFound),
        ({**_CASES["cardLive"], "slots": [_slot("cmp_note", 1, None)]}, SlotDrawsNoFrame),
        (
            {**_CASES["canvSnap"], "slots": [_slot("cmp_chart", 1, _UNPINNABLE)]},
            FrameUnavailable,
        ),
    ],
    ids=["unknown-card", "unknown-revision", "null-frame-on-a-note", "unpinnable-frame"],
)
def test_a_share_the_import_cannot_place_is_refused_and_stores_nothing(
    line: dict[str, object], refusal: type[Exception]
):
    store = LiveShareStore(store=FakePresentationStore())

    with pytest.raises(refusal):
        _import(store, line)

    with pytest.raises(ShareNotFound):
        store.read_share(share_id=str(line["id"]))


@pytest.mark.parametrize(
    ("held", "token_differs"),
    [
        ("made-here", False),
        ({**_CASES["cardLive"], "kind": "canvas"}, False),
        ({**_CASES["cardLive"], "write_token_sha256": "0" * 64}, True),
        ({**_CASES["cardLive"], "owner": None, "write_token_sha256": None}, True),
        ({**_CASES["guestSnp"], "owner": "user_1", "write_token_sha256": _DIGEST}, True),
    ],
    ids=["made-here", "other-kind", "other-digest", "no-digest", "digest-held-elsewhere"],
)
def test_an_id_holding_another_share_or_another_digest_refuses_and_pins_nothing(
    held: str | dict[str, object], token_differs: bool
):
    fake = FakePresentationStore()
    if held == "made-here":
        LiveShareStore(store=fake, mint_id=lambda: "cardLive").create_share(
            kind="card", mode="live", title=None, write_token_sha256=None, cards=[], renderings={}
        )
    else:
        assert isinstance(held, dict)
        _import(LiveShareStore(store=fake), held)
    stored = _stored(LiveShareStore(store=fake), "cardLive")

    with pytest.raises(ShareIdTaken) as refused:
        _import(LiveShareStore(store=fake), _CASES["cardLive"], pins=_NoPins())

    assert refused.value.token_differs is token_differs
    assert _stored(LiveShareStore(store=fake), "cardLive") == stored


def _stored(store: LiveShareStore, share_id: str) -> object:
    """The share at `share_id`, or `None` when none is."""
    try:
        return store.read_share(share_id=share_id)
    except ShareNotFound:
        return None


class _RacingShareStore(LiveShareStore):
    """A share store whose `held_import` found the id and digest free, as it would for an
    import that another one overtook before its write."""

    def held_import(self, **_: object) -> bool:
        return False


def test_an_import_overtaken_by_one_taking_its_digest_refuses_in_the_write():
    fake = FakePresentationStore()
    _import(
        LiveShareStore(store=fake),
        {**_CASES["guestSnp"], "owner": "user_1", "write_token_sha256": _DIGEST},
    )

    with pytest.raises(ShareIdTaken) as refused:
        _import(_RacingShareStore(store=fake), _CASES["cardLive"])

    assert refused.value.token_differs is True
    assert _stored(LiveShareStore(store=fake), "cardLive") is None


@pytest.mark.parametrize("case", ["canvLive", "canvSnap"])
def test_an_import_pins_each_card_revisions_frame_and_each_slots_frame(case: str):
    pins = _Pins()

    _import(LiveShareStore(store=FakePresentationStore()), _CASES[case], pins=pins)

    expected = {
        "canvLive": {(_DATASET, "cmp_chart"), (_DERIVED, "cmp_derived")},
        "canvSnap": {(_DATASET, "cmp_chart"), (_REFRESHED, "cmp_chart")},
    }[case]
    assert set(pins.pins) == expected
