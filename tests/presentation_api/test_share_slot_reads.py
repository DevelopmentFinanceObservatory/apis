"""The public share routes' replies across origins, and the slot routes' conditional reads:
every reply of a public route allows any origin, a refusal included, and a revalidation answers
304 from what the slot serves without drawing the body."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import presentation_api.retention as presentation_retention
from assembler.errors import FrameUnavailable
from assembler.fakes import FakeReleaseRegistry
from assembler.ports import FrameRef, ResolvedFrame
from presentation_api.newest_frames import NewestPending
from tests.presentation_api.test_share_data_door import (
    _MOVED,
    _STORED,
    _Door,
    _serve_door,
    _stated,
)

_OTHER_ORIGIN = {"origin": "https://news.example"}


@pytest.fixture
def door(release_registry: FakeReleaseRegistry) -> Iterator[_Door]:
    yield from _serve_door(release_registry, cors_origins=())


def _unloadable(door: _Door, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every frame load fail, so any read that draws a body refuses 503."""

    def unloadable(*, frame_ref: FrameRef) -> ResolvedFrame:
        raise FrameUnavailable(reason="io", detail="frame store down")

    monkeypatch.setattr(door.frame_store, "load", unloadable)


@pytest.mark.parametrize("route", ["", "/history", "/slots/1", "/slots/1/data"])
def test_a_public_route_refusing_not_found_allows_any_origin(door: _Door, route: str):
    resp = door.client.get(f"/api/presentation/shares/Zz9yXw8v{route}", headers=_OTHER_ORIGIN)

    assert (resp.status_code, resp.json()["code"]) == (404, "not_found")
    assert resp.headers["access-control-allow-origin"] == "*"


def test_a_slot_read_that_fails_allows_any_origin_and_a_write_refusal_does_not(
    door: _Door, monkeypatch: pytest.MonkeyPatch
):
    share = door.share(door.card(), latest=False)
    _unloadable(door, monkeypatch)

    failed = door.client.get(
        f"/api/presentation/shares/{share['id']}/slots/1/data", headers=_OTHER_ORIGIN
    )
    write = door.client.put(
        f"/api/presentation/shares/{share['id']}", json={"cards": []}, headers=_OTHER_ORIGIN
    )

    assert (failed.status_code, failed.json()["code"]) == (503, "service_unavailable")
    assert failed.headers["access-control-allow-origin"] == "*"
    assert (write.status_code, write.json()["code"]) == (403, "write_token_required")
    assert "access-control-allow-origin" not in write.headers


@pytest.mark.parametrize("latest", [False, True], ids=["snapshot", "live-moved"])
def test_a_revalidation_answers_304_without_drawing_the_slot(
    door: _Door, monkeypatch: pytest.MonkeyPatch, latest: bool
):
    share = door.share(door.card(), latest=latest)
    door.newest.answer = _MOVED
    data_path = f"/api/presentation/shares/{share['id']}/slots/1/data"
    card_path = f"/api/presentation/shares/{share['id']}/slots/1"
    data, card = door.client.get(data_path), door.client.get(card_path)
    assert (data.status_code, card.status_code) == (200, 200)
    _unloadable(door, monkeypatch)

    data_again = door.client.get(data_path, headers={"if-none-match": data.headers["etag"]})
    card_again = door.client.get(card_path, headers={"if-none-match": card.headers["etag"]})
    drawn = door.client.get(data_path)

    for first, again in ((data, data_again), (card, card_again)):
        assert (again.status_code, again.content) == (304, b"")
        assert again.headers["etag"] == first.headers["etag"]
        assert again.headers["x-dfo-freshness"] == first.headers["x-dfo-freshness"]
        assert again.headers["access-control-allow-origin"] == "*"
    assert drawn.status_code == 503
    assert door.newest.asked == ([_STORED] if latest else [])


def test_a_live_slots_etag_follows_its_frame_and_freshness_and_not_its_check(door: _Door):
    share = door.share(door.card(), latest=True)
    door.newest.answer = _MOVED
    path = f"/api/presentation/shares/{share['id']}/slots/1/data"
    first = door.client.get(path)
    etag = first.headers["etag"]

    door.later(minutes=15)
    rechecked = door.client.get(path, headers={"if-none-match": etag})
    door.later(minutes=15)
    door.newest.answer = NewestPending(retry_after_s=30)
    pending = door.client.get(path, headers={"if-none-match": etag})

    assert _stated(first)[:2] == ("current", _MOVED.content_version)
    # The second check found the same frame, so the body is the same and so is its ETag.
    assert (rechecked.status_code, rechecked.headers["etag"]) == (304, etag)
    assert rechecked.headers["x-dfo-checked-at"] != first.headers["x-dfo-checked-at"]
    assert pending.status_code == 200
    assert _stated(pending)[:2] == ("pending", _MOVED.content_version)
    assert pending.headers["etag"] != etag
    assert len(door.newest.asked) == 3


def test_a_slot_cards_etag_follows_its_frames_retention(
    door: _Door, monkeypatch: pytest.MonkeyPatch
):
    retention: dict[str, object] = {"pinned": False, "held_until": "2026-11-28T09:00:00+00:00"}
    monkeypatch.setattr(
        presentation_retention, "frame_retention", lambda frame_ref, *, snapshot_reader: retention
    )
    share = door.share(door.card(), latest=False)
    path = f"/api/presentation/shares/{share['id']}/slots/1"
    first = door.client.get(path)
    unchanged = door.client.get(path, headers={"if-none-match": first.headers["etag"]})

    retention = {"pinned": True}
    pinned = door.client.get(path, headers={"if-none-match": first.headers["etag"]})

    assert first.json()["retention"]["pinned"] is False
    assert unchanged.status_code == 304
    assert (pinned.status_code, pinned.json()["retention"]) == (200, {"pinned": True})
    assert pinned.headers["etag"] != first.headers["etag"]
