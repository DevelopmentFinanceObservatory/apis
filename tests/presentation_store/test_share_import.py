"""Checks `import_shares` over the website's export: every stored case `docs/share-export.md`
places, slot numbering, `frame: null` slots, tokens for signed-in users alone, idempotence, and
refused lines that leave the others imported."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import ClassVar

import pytest

from assembler.ports import DraftNotFound, FrameRef, RevisionRecord
from assembler.write_token import write_token_digest
from presentation_store.share_import import ImportCounts, import_shares
from presentation_store.shares import LiveShareStore

from .conftest import FakePresentationStore

_DATASET = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)
_REFRESHED = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)
_DERIVED = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "e" * 32)


class _Revisions:
    """The card revisions the service holds: `cmp_chart` 1 and 2 on `_DATASET`, `cmp_derived` 1
    on `_DERIVED`, and `cmp_note` 1, which draws no frame."""

    _FRAMES: ClassVar[dict[tuple[str, int], FrameRef]] = {
        ("cmp_chart", 1): _DATASET,
        ("cmp_chart", 2): _DATASET,
        ("cmp_derived", 1): _DERIVED,
    }

    def read_revision(self, *, component_id: str, revision: int) -> RevisionRecord | None:
        if (component_id, revision) == ("cmp_note", 1):
            intent: dict[str, object] = {"type": "note_narration", "text": "Aid rose."}
        elif (frame := self._FRAMES.get((component_id, revision))) is not None:
            intent = {
                "type": "line_chart",
                "frame_ref": {"spec_id": frame.spec_id, "content_version": frame.content_version},
            }
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
        )

    def read_current(self, *, component_id: str) -> RevisionRecord | None:
        if component_id in {key[0] for key in self._FRAMES} | {"cmp_note"}:
            return self.read_revision(component_id=component_id, revision=1)
        raise DraftNotFound(subject_kind="component", subject_id=component_id)


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
        "guestSnp", "card", "snapshot", owner=None, slots=[_slot("cmp_chart", 1, _DATASET)]
    ),
}


def _run(store: LiveShareStore, lines: list[str]) -> tuple[ImportCounts, list[dict], list]:
    tokens: list[dict] = []
    refused: list[tuple[str, str]] = []
    counts = import_shares(
        lines,
        share_store=store,
        revision_store=_Revisions(),
        write_token=lambda text: tokens.append(json.loads(text)),
        refuse=lambda label, reason: refused.append((label, reason)),
    )
    return counts, tokens, refused


def _lines(*lines: dict[str, object]) -> list[str]:
    return [json.dumps(line) + "\n" for line in lines]


def test_every_exported_case_imports_under_its_id_and_a_second_run_changes_nothing():
    fake = FakePresentationStore()
    store = LiveShareStore(store=fake)
    lines = _lines(*_CASES.values())

    counts, _tokens, refused = _run(store, lines)
    after_first = {share_id: store.read_share(share_id=share_id) for share_id in _CASES}
    again, second_tokens, second_refused = _run(store, lines)

    assert (counts, refused) == (ImportCounts(imported=8, unchanged=0, refused=0), [])
    assert (again, second_tokens, second_refused) == (
        ImportCounts(imported=0, unchanged=8, refused=0),
        [],
        [],
    )
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


def test_a_signed_in_users_share_gets_a_token_the_report_holds_and_a_guests_gets_none():
    store = LiveShareStore(store=FakePresentationStore())

    _counts, tokens, _refused = _run(store, _lines(_CASES["cardLive"], _CASES["guestSnp"]))

    assert [(entry["id"], entry["owner"]) for entry in tokens] == [("cardLive", "user_1")]
    assert tokens[0]["write_token"].startswith("shw_")
    assert store.read_share(share_id="cardLive").write_token_sha256 == write_token_digest(
        tokens[0]["write_token"]
    )
    assert store.read_share(share_id="guestSnp").write_token_sha256 is None


def test_export_slot_index_i_is_slot_i_plus_one_and_a_null_frame_takes_the_revisions():
    store = LiveShareStore(store=FakePresentationStore())

    _run(store, _lines(_CASES["canvLive"], _CASES["canvSnap"]))

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
    ("line", "label"),
    [
        ("not json", "line 1"),
        (json.dumps({**_CASES["cardLive"], "id": "short"}), "line 1"),
        (json.dumps({**_CASES["cardLive"], "kind": "deck"}), "share cardLive"),
        (json.dumps({**_CASES["cardLive"], "mode": None}), "share cardLive"),
        (json.dumps({**_CASES["convSnap"], "mode": "live"}), "share convSnap"),
        (
            json.dumps({**_CASES["srchLive"], "slots": [_slot("cmp_chart", 1, _DATASET)]}),
            "share srchLive",
        ),
        (json.dumps({**_CASES["cardLive"], "created_at": "2026-09-01T10:00:00"}), "share cardLive"),
        (
            json.dumps({**_CASES["cardLive"], "slots": [_slot("cmp_absent", 1, _DATASET)]}),
            "share cardLive",
        ),
        (
            json.dumps({**_CASES["cardLive"], "slots": [_slot("cmp_chart", 9, _DATASET)]}),
            "share cardLive",
        ),
        (
            json.dumps({**_CASES["cardLive"], "slots": [_slot("cmp_note", 1, None)]}),
            "share cardLive",
        ),
    ],
    ids=[
        "not-json",
        "bad-id",
        "unknown-kind",
        "no-mode",
        "live-conversation",
        "search-with-slots",
        "naive-created-at",
        "unknown-card",
        "unknown-revision",
        "null-frame-on-a-note",
    ],
)
def test_a_line_the_import_cannot_place_is_refused_and_the_others_import(line: str, label: str):
    store = LiveShareStore(store=FakePresentationStore())

    counts, _tokens, refused = _run(store, [line + "\n", *_lines(_CASES["guestSnp"])])

    assert counts == ImportCounts(imported=1, unchanged=0, refused=1)
    assert [entry[0] for entry in refused] == [label]
    assert store.read_share(share_id="guestSnp").owner is None


def test_an_id_holding_a_share_made_here_refuses_the_line():
    fake = FakePresentationStore()
    LiveShareStore(store=fake, mint_id=lambda: "cardLive").create_share(
        kind="card", mode="live", title=None, write_token_sha256=None, cards=[]
    )

    counts, tokens, refused = _run(LiveShareStore(store=fake), _lines(_CASES["cardLive"]))

    assert (counts, tokens) == (ImportCounts(imported=0, unchanged=0, refused=1), [])
    assert refused[0][0] == "share cardLive"
