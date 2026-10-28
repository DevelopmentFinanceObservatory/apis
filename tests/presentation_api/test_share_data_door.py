"""The share data, history and snapshot routes: what a slot serves and how it states it, the
newest-data check window, ETags, served history with retention, and the live-to-snapshot
conversion."""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import presentation_api.share_slots as share_slots
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef, ResolvedFrame
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_rows,
)
from datasets_api.retention import RetentionUnreadable
from dfo_platform.app import create_app
from dfo_platform.config import CallerSettings
from presentation_api.app import presentation_surface
from presentation_api.doors.shares import SHARES_DEPS_SLOT
from presentation_api.newest_frames import NewestPending
from presentation_store import LiveShareStore
from tests.presentation_api.test_shares_door import _Allowances
from tests.presentation_store.conftest import FakePresentationStore

_HEADERS = {"x-forwarded-for": "203.0.113.7"}
_TOKEN = "x-dfo-write-token"
_STORED = malaria_ethiopia_frame_ref()
_MOVED = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "b" * 32)
_UNREADABLE = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "e" * 32)
_MOVED_BY = 1000.0
_START = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)


def _frame(frame_ref: FrameRef, *, moved_by: float = 0.0) -> ResolvedFrame:
    return build_resolved_frame(
        frame_ref=frame_ref,
        role_keys=("recipient",),
        rows=[
            {**row, "recipient": "country/ETH", "value": row["value"] + moved_by}
            for row in malaria_ethiopia_rows()
        ],
        labels={"country/ETH": "Ethiopia"},
        retrieved_on=date(2026, 10, 6),
    )


class _Newest:
    """`NewestFrames` answering `answer`: `None` for the frame asked about, a frame, a
    `NewestPending`, or an exception to raise. Records every frame asked about."""

    def __init__(self) -> None:
        self.answer: FrameRef | NewestPending | Exception | None = None
        self.asked: list[FrameRef] = []

    def newest(self, frame: FrameRef) -> FrameRef | NewestPending:
        self.asked.append(frame)
        if isinstance(self.answer, Exception):
            raise self.answer
        return frame if self.answer is None else self.answer


class _Clock:
    def __init__(self) -> None:
        self.at = _START

    def __call__(self) -> datetime:
        return self.at


class _Door:
    def __init__(self, client: TestClient, *, frame_store: FakeFrameStore, newest: _Newest):
        self.client = client
        self.frame_store = frame_store
        self.newest = newest
        self.clock = _Clock()
        getattr(client.app.state, SHARES_DEPS_SLOT.name).now = self.clock

    def card(self, intent: dict[str, object] | None = None) -> str:
        resp = self.client.post(
            "/api/presentation/components",
            json={
                "intent": intent
                or {
                    "type": "line_chart",
                    "frame_ref": {
                        "spec_id": _STORED.spec_id,
                        "content_version": _STORED.content_version,
                    },
                    "select": {
                        "stat_var": "dcid:MalariaGrantDisbursements",
                        "recipient": "country/ETH",
                    },
                    "title": "Malaria grants to Ethiopia",
                }
            },
        )
        assert resp.status_code == 201, resp.json()
        return resp.json()["component_id"]

    def share(self, *component_ids: str, latest: bool) -> dict:
        resp = self.client.post(
            "/api/presentation/shares",
            json={
                "cards": [{"component_id": cid, "revision": 1} for cid in component_ids],
                "latest": latest,
            },
            headers=_HEADERS,
        )
        assert resp.status_code == 201, resp.json()
        return resp.json()

    def data(self, share_id: str, slot: int = 1, **params: str):
        return self.client.get(
            f"/api/presentation/shares/{share_id}/slots/{slot}/data", params=params
        )

    def later(self, **delta: float) -> None:
        self.clock.at += timedelta(**delta)


@pytest.fixture
def door(release_registry: FakeReleaseRegistry) -> Iterator[_Door]:
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=_frame(_STORED))
    frame_store.put(resolved_frame=_frame(_MOVED, moved_by=_MOVED_BY))
    newest = _Newest()
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=LiveShareStore(store=FakePresentationStore()),
        allowances=_Allowances(limit=50),
        caller_settings=CallerSettings(caller_hmac_key=SecretStr("share-data-test-key")),
        newest_frames=newest,
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        yield _Door(client, frame_store=frame_store, newest=newest)


def _values(body: dict) -> list[float]:
    return [row["value"] for row in body["rows"]]


def _stated(resp) -> tuple[str, str, str]:
    body = resp.json()
    assert resp.headers["x-dfo-freshness"] == body["freshness"]
    return body["freshness"], body["frame"]["content_version"], body["mode"]


def test_a_snapshot_slot_serves_its_cards_frame_as_json_rows_any_origin_may_read(door: _Door):
    share = door.share(door.card(), latest=False)

    resp = door.data(share["id"])

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "x-dfo-freshness" in resp.headers["access-control-expose-headers"]
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert "x-dfo-checked-at" not in resp.headers
    body = resp.json()
    assert _stated(resp) == ("current", _STORED.content_version, "snapshot")
    assert (body["share_id"], body["slot"], body["revision"]) == (share["id"], 1, 1)
    assert body["rows"] and {"series", "value", "time_period"} <= set(body["rows"][0])
    assert door.newest.asked == []


def test_a_matching_if_none_match_answers_304_and_gzip_carries_the_etag_weak(door: _Door):
    share = door.share(door.card(), latest=False)
    url = f"/api/presentation/shares/{share['id']}/slots/1/data"
    identity = door.client.get(url, headers={"accept-encoding": "identity"})
    etag = identity.headers["etag"]

    gzipped = door.client.get(url, headers={"accept-encoding": "gzip"})
    again = door.client.get(url, headers={"if-none-match": f'"other", {etag}'})
    weak = door.client.get(url, headers={"if-none-match": f"W/{etag}"})
    other = door.client.get(url, headers={"if-none-match": '"other"'})

    assert etag.startswith('"') and etag.endswith('"')
    assert gzipped.headers["etag"] == f"W/{etag}"
    assert (again.status_code, again.content) == (304, b"")
    assert again.headers["x-dfo-freshness"] == "current"
    assert (weak.status_code, weak.headers["etag"]) == (304, f"W/{etag}")
    assert other.status_code == 200


def test_csv_and_parquet_serve_the_same_rows_as_files(door: _Door):
    share = door.share(door.card(), latest=False)
    rows = door.data(share["id"]).json()["rows"]

    csv = door.data(share["id"], format="csv")
    parquet = door.data(share["id"], format="parquet")

    assert csv.headers["content-type"].startswith("text/csv")
    assert csv.headers["x-dfo-freshness"] == "current"
    assert "attachment" in csv.headers["content-disposition"]
    assert len(csv.text.strip().splitlines()) == len(rows) + 1
    assert pq.read_table(io.BytesIO(parquet.content)).to_pylist() == rows
    assert len({csv.headers["etag"], parquet.headers["etag"]}) == 2


def test_a_live_slot_checks_for_newer_data_at_most_once_per_fifteen_minutes(door: _Door):
    share = door.share(door.card(), latest=True)

    first = door.data(share["id"])
    door.later(minutes=14, seconds=59)
    within = door.data(share["id"])
    door.later(seconds=1)
    after = door.data(share["id"])

    assert door.newest.asked == [_STORED, _STORED]
    assert _stated(first) == ("current", _STORED.content_version, "live")
    assert first.headers["cache-control"] == "no-cache"
    assert first.headers["x-dfo-checked-at"] == _START.isoformat()
    assert within.headers["x-dfo-checked-at"] == _START.isoformat()
    assert after.headers["x-dfo-checked-at"] == (_START + timedelta(minutes=15)).isoformat()
    assert first.headers["etag"] == within.headers["etag"] == after.headers["etag"]


@pytest.mark.parametrize(
    ("retry_after_s", "due_in_s"),
    [(30.0, 30), (0.0, 1), (-5.0, 1), (7200.0, 900)],
    ids=["retry-time", "floor-one-second", "negative-floor", "ceiling-the-window"],
)
def test_a_pending_slot_is_checked_again_after_its_retry_time_and_not_before(
    door: _Door, retry_after_s: float, due_in_s: int
):
    share = door.share(door.card(), latest=True)
    door.newest.answer = NewestPending(retry_after_s=retry_after_s)
    pending = door.data(share["id"])
    door.newest.answer = _MOVED

    door.later(seconds=due_in_s - 1)
    early = door.data(share["id"])
    door.later(seconds=1)
    due = door.data(share["id"])

    assert _stated(pending)[0] == _stated(early)[0] == "pending"
    assert _stated(due)[:2] == ("current", _MOVED.content_version)
    assert door.newest.asked == [_STORED, _STORED]


def test_a_current_slot_waits_the_full_window_even_after_a_pending_check_was_short(door: _Door):
    share = door.share(door.card(), latest=True)
    door.newest.answer = NewestPending(retry_after_s=30)
    door.data(share["id"])
    door.newest.answer = _MOVED
    door.later(seconds=30)
    door.data(share["id"])

    door.later(minutes=14, seconds=59)
    within = door.data(share["id"])
    door.later(seconds=1)
    door.data(share["id"])

    assert _stated(within)[0] == "current"
    assert len(door.newest.asked) == 3


def test_a_live_slot_follows_newer_data_and_states_pending_and_stale_on_its_last_frame(
    door: _Door,
):
    share = door.share(door.card(), latest=True)
    stored_values = _values(door.data(share["id"]).json())
    states = []
    for answer in (
        _MOVED,
        NewestPending(retry_after_s=30),
        RuntimeError("refresh refused"),
        _MOVED,
    ):
        door.later(minutes=15)
        door.newest.answer = answer
        resp = door.data(share["id"])
        states.append((*_stated(resp)[:2], _values(resp.json()) == stored_values))

    assert states == [
        ("current", _MOVED.content_version, False),
        ("pending", _MOVED.content_version, False),
        ("stale", _MOVED.content_version, False),
        ("current", _MOVED.content_version, False),
    ]
    assert door.newest.asked == [_STORED] * 5


def test_a_live_slot_whose_card_cannot_draw_the_newer_frame_serves_its_own_as_stale(
    door: _Door,
):
    share = door.share(door.card(), latest=True)
    door.newest.answer = _UNREADABLE

    first = door.data(share["id"])
    door.later(minutes=1)
    second = door.data(share["id"])

    assert _stated(first) == _stated(second) == ("stale", _STORED.content_version, "live")
    assert door.newest.asked == [_STORED]


def test_a_live_slot_with_no_datasets_surface_serves_its_cards_frame_as_stale(
    release_registry: FakeReleaseRegistry,
):
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=_frame(_STORED))
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=LiveShareStore(store=FakePresentationStore()),
        allowances=_Allowances(limit=50),
        caller_settings=CallerSettings(caller_hmac_key=SecretStr("share-data-test-key")),
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        door = _Door(client, frame_store=frame_store, newest=_Newest())
        share = door.share(door.card(), latest=True)

        resp = door.data(share["id"])

    assert _stated(resp) == ("stale", _STORED.content_version, "live")


def test_a_replacement_puts_a_new_card_in_the_slot_with_no_check(door: _Door):
    share = door.share(door.card(), latest=True)
    door.data(share["id"])
    replaced = door.client.put(
        f"/api/presentation/shares/{share['id']}",
        json={"cards": [{"component_id": door.card(), "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )
    assert replaced.status_code == 200, replaced.json()

    door.data(share["id"])

    assert door.newest.asked == [_STORED, _STORED]


@pytest.mark.parametrize(
    ("path", "status", "code"),
    [
        ("slots/2/data", 404, "not_found"),
        ("slots/0/data", 400, "request_invalid"),
        ("slots/1/data?format=xlsx", 400, "request_invalid"),
    ],
    ids=["no-such-slot", "slot-zero", "unknown-format"],
)
def test_a_slot_the_share_does_not_have_refuses(door: _Door, path: str, status: int, code: str):
    share = door.share(door.card(), latest=True)

    resp = door.client.get(f"/api/presentation/shares/{share['id']}/{path}")

    assert (resp.status_code, resp.json()["code"]) == (status, code)


def test_an_emptied_slot_and_a_notes_slot_answer_not_found(door: _Door):
    note = door.card({"type": "note_narration", "text": "Grants rose."})
    share = door.share(door.card(), note, latest=True)
    door.client.put(
        f"/api/presentation/shares/{share['id']}",
        json={"cards": [{"component_id": note, "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )

    note_slot, emptied = door.data(share["id"], 1), door.data(share["id"], 2)

    assert (note_slot.status_code, note_slot.json()["code"]) == (404, "not_found")
    assert (emptied.status_code, emptied.json()["code"]) == (404, "not_found")


def test_the_history_lists_each_frame_served_newest_first_with_its_retention(
    door: _Door, monkeypatch: pytest.MonkeyPatch
):
    held = (_START - timedelta(days=1)).isoformat()
    retention = {_STORED: {"pinned": True}, _MOVED: {"pinned": False, "held_until": held}}

    def stated(frame_ref: FrameRef, *, snapshot_reader) -> dict[str, object]:
        if frame_ref not in retention:
            raise RetentionUnreadable(
                subject_id=frame_ref.spec_id, content_version=frame_ref.content_version, reason=""
            )
        return retention[frame_ref]

    monkeypatch.setattr(share_slots, "frame_retention", stated)
    card = door.card()
    share = door.share(card, latest=True)
    door.data(share["id"])
    door.data(share["id"], format="csv")
    door.later(minutes=15)
    door.newest.answer = _MOVED
    door.data(share["id"])

    resp = door.client.get(f"/api/presentation/shares/{share['id']}/history")
    del retention[_STORED]
    unreadable = door.client.get(f"/api/presentation/shares/{share['id']}/history").json()

    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    served = resp.json()["served"]
    assert [
        (entry["slot"], entry["component_id"], entry["frame"]["content_version"])
        for entry in served
    ] == [(1, card, _MOVED.content_version), (1, card, _STORED.content_version)]
    assert [(entry["retention"], entry["fetchable"]) for entry in served] == [
        ({"pinned": False, "held_until": held}, False),
        ({"pinned": True}, True),
    ]
    assert served[0]["first_served_at"] >= served[1]["first_served_at"]
    assert (unreadable["served"][1]["retention"], unreadable["served"][1]["fetchable"]) == (
        None,
        False,
    )


def test_a_snapshot_conversion_keeps_and_pins_the_frame_each_slot_serves(door: _Door):
    card, other = door.card(), door.card()
    share = door.share(card, other, latest=True)
    door.newest.answer = _MOVED
    door.data(share["id"], 1)
    door.frame_store._pins.clear()
    snapshot_path = f"/api/presentation/shares/{share['id']}/snapshot"

    missing = door.client.post(snapshot_path)
    resp = door.client.post(snapshot_path, headers={_TOKEN: share["write_token"]})
    again = door.client.post(snapshot_path, headers={_TOKEN: share["write_token"]})

    assert (missing.status_code, missing.json()["code"]) == (403, "write_token_required")
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert (body["latest"], body["revision"], body["data_urls"]) == (
        False,
        2,
        share["data_urls"],
    )
    assert again.json() == body
    assert sorted(door.frame_store.pins, key=str) == sorted(
        [(_MOVED, card), (_STORED, other)], key=str
    )
    read = door.client.get(f"/api/presentation/shares/{share['id']}").json()
    assert read["mode"] == "snapshot"
    assert [slot["card"]["frame"]["content_version"] for slot in read["slots"]] == [
        _MOVED.content_version,
        _STORED.content_version,
    ]
    door.later(hours=1)
    door.newest.asked.clear()
    first = door.data(share["id"], 1)
    assert _stated(first) == ("current", _MOVED.content_version, "snapshot")
    assert door.newest.asked == []
    replaced = door.client.put(
        f"/api/presentation/shares/{share['id']}",
        json={"cards": [{"component_id": card, "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )
    assert (replaced.status_code, replaced.json()["code"]) == (409, "share_not_replaceable")
