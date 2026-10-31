"""The share data, history and snapshot routes: what a slot serves and how it states it, the
newest-data check window, ETags, served history with retention, and the live-to-snapshot
conversion."""

from __future__ import annotations

import csv
import dataclasses
import io
import zipfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

import openpyxl
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import presentation_api.retention as presentation_retention
from assembler.errors import FrameUnavailable, ReleaseMismatch
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef, ResolvedFrame
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_rows,
)
from assembler.write_token import SHARE_WRITE_TOKEN_PREFIX, mint_write_token
from datasets_api.retention import RetentionUnreadable
from dfo_platform.app import create_app
from dfo_platform.config import CallerSettings
from presentation_api.app import presentation_surface
from presentation_api.doors.shares import SHARES_DEPS_SLOT
from presentation_api.newest_frames import NewestNotReady, NewestPending
from presentation_store import LiveShareStore, PresentationStoreUnavailable
from tests.presentation_api.test_shares_door import _Allowances
from tests.presentation_store.conftest import FakePresentationStore

_HEADERS = {"x-forwarded-for": "203.0.113.7"}
_TOKEN = "x-dfo-write-token"
_STORED = malaria_ethiopia_frame_ref()
_MOVED = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "b" * 32)
_UNREADABLE = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "e" * 32)
_MOVED_BY = 1000.0
_START = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
_SITE = "https://dfo.example"


def _rendering(frame_ref: FrameRef) -> dict[str, object]:
    """The rendering `frame_ref` resolves as, which a card made on it records."""
    return {"base_content_version": frame_ref.content_version, "minting": None}


def _frame(frame_ref: FrameRef, *, moved_by: float = 0.0) -> ResolvedFrame:
    frame = build_resolved_frame(
        frame_ref=frame_ref,
        role_keys=("recipient",),
        rows=[
            {**row, "recipient": "country/ETH", "value": row["value"] + moved_by}
            for row in malaria_ethiopia_rows()
        ],
        labels={"country/ETH": "Ethiopia"},
        retrieved_on=date(2026, 10, 6),
    )
    return dataclasses.replace(frame, rendering=_rendering(frame_ref))


class _Newest:
    """`NewestFrames` answering `answer`: `None` for the frame asked about, a frame, a
    `NewestPending`, or an exception to raise. Records every frame asked about and the
    rendering it was asked under, and calls `during` once, as another read arriving while the
    check runs."""

    def __init__(self) -> None:
        self.answer: FrameRef | NewestPending | Exception | None = None
        self.asked: list[FrameRef] = []
        self.renderings: list[Mapping[str, object] | None] = []
        self.during: Callable[[], None] | None = None

    def newest(
        self, frame: FrameRef, *, rendering: Mapping[str, object] | None
    ) -> FrameRef | NewestPending:
        self.asked.append(frame)
        self.renderings.append(rendering)
        if (during := self.during) is not None:
            self.during = None
            during()
        if isinstance(self.answer, Exception):
            raise self.answer
        return frame if self.answer is None else self.answer


class _Clock:
    def __init__(self) -> None:
        self.at = _START

    def __call__(self) -> datetime:
        return self.at


class _RefusingFrameStore(FakeFrameStore):
    """Raises `refusals[frame_ref]` for a frame it holds there, `_UNREADABLE` a frame that cannot
    be loaded."""

    def __init__(self) -> None:
        super().__init__()
        self.loads: list[FrameRef] = []
        self.refusals: dict[FrameRef, Exception] = {
            _UNREADABLE: FrameUnavailable(reason="missing_rendering", detail="exemplar")
        }

    def load(self, *, frame_ref: FrameRef) -> ResolvedFrame:
        self.loads.append(frame_ref)
        if frame_ref in self.refusals:
            raise self.refusals[frame_ref]
        return super().load(frame_ref=frame_ref)


class _Door:
    def __init__(
        self,
        client: TestClient,
        *,
        frame_store: _RefusingFrameStore,
        newest: _Newest,
        presentation: FakePresentationStore,
    ):
        self.client = client
        self.frame_store = frame_store
        self.presentation = presentation
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
            headers=_HEADERS,
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

    def slot_card(self, share_id: str, slot: int = 1, *, headers: dict[str, str] | None = None):
        return self.client.get(f"/api/presentation/shares/{share_id}/slots/{slot}", headers=headers)

    def later(self, **delta: float) -> None:
        self.clock.at += timedelta(**delta)


@pytest.fixture
def door(release_registry: FakeReleaseRegistry) -> Iterator[_Door]:
    yield from _serve_door(release_registry, cors_origins=())


@pytest.fixture
def cors_door(release_registry: FakeReleaseRegistry) -> Iterator[_Door]:
    yield from _serve_door(release_registry, cors_origins=(_SITE,))


def _serve_door(
    release_registry: FakeReleaseRegistry, *, cors_origins: tuple[str, ...]
) -> Iterator[_Door]:
    frame_store = _RefusingFrameStore()
    frame_store.put(resolved_frame=_frame(_STORED))
    frame_store.put(resolved_frame=_frame(_MOVED, moved_by=_MOVED_BY))
    newest = _Newest()
    presentation = FakePresentationStore()
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=LiveShareStore(store=presentation),
        allowances=_Allowances(limit=50),
        caller_settings=CallerSettings(caller_hmac_key=SecretStr("share-data-test-key")),
        newest_frames=newest,
    )
    app = create_app([surface], cors_origins=cors_origins)
    with TestClient(app, raise_server_exceptions=False, headers=_HEADERS) as client:
        yield _Door(client, frame_store=frame_store, newest=newest, presentation=presentation)


def _values(body: dict) -> list[float]:
    return [row["value"] for row in body["rows"]]


def _stated(resp) -> tuple[str, str, str]:
    body = resp.json()
    assert resp.headers["x-dfo-freshness"] == body["freshness"]
    return body["freshness"], body["frame"]["content_version"], body["mode"]


def _drawn_on(card: dict) -> str:
    """The content version of the one frame a card read is drawn on."""
    (frame,) = card["resolved_payload"]["provenance"]["inputs"]
    return frame["content_version"]


@pytest.mark.parametrize(
    ("answer", "freshness", "frame"),
    [
        (None, "current", _STORED),
        (_MOVED, "current", _MOVED),
        (NewestPending(retry_after_s=30), "pending", _STORED),
        (RuntimeError("refresh refused"), "stale", _STORED),
        (_UNREADABLE, "stale", _STORED),
    ],
    ids=["unchanged", "newer", "pending", "check-failed", "newer-unreadable"],
)
def test_a_live_slots_card_is_drawn_on_the_frame_its_data_serves(
    door: _Door,
    answer: FrameRef | NewestPending | Exception | None,
    freshness: str,
    frame: FrameRef,
):
    card = door.card()
    share = door.share(card, latest=True)
    door.newest.answer = answer

    resp = door.slot_card(share["id"])
    data = door.data(share["id"])

    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert (body["component_id"], body["revision"]) == (card, 1)
    assert body["freshness"] == resp.headers["x-dfo-freshness"] == freshness
    assert _stated(data)[:2] == (freshness, frame.content_version)
    assert _drawn_on(body) == frame.content_version
    assert resp.headers["access-control-allow-origin"] == "*"
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-dfo-checked-at"] == _START.isoformat()
    assert door.newest.asked == [_STORED]
    assert door.newest.renderings == [_rendering(_STORED)]


def test_a_snapshot_slots_card_is_the_cards_read_with_a_strong_etag(door: _Door):
    card = door.card()
    share = door.share(card, latest=False)
    read = door.client.get(f"/api/presentation/components/{card}").json()

    resp = door.slot_card(share["id"], headers={"accept-encoding": "identity"})
    etag = resp.headers["etag"]
    again = door.slot_card(share["id"], headers={"if-none-match": etag})

    assert resp.status_code == 200
    assert resp.json() == {**read, "freshness": "current"}
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert etag.startswith('"') and etag.endswith('"')
    assert (again.status_code, again.content) == (304, b"")
    served = door.client.get(f"/api/presentation/shares/{share['id']}/history").json()["served"]
    assert [entry["frame"]["content_version"] for entry in served] == [_STORED.content_version]
    assert door.newest.asked == []


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


@pytest.mark.parametrize("origin", [_SITE, "https://news.example"], ids=["listed", "other"])
def test_with_cors_on_any_origin_reads_a_slots_headers_and_sends_if_none_match(
    cors_door: _Door, origin: str
):
    share = cors_door.share(cors_door.card(), latest=True)
    url = f"/api/presentation/shares/{share['id']}/slots/1/data"

    preflight = cors_door.client.options(
        url,
        headers={
            "origin": origin,
            "access-control-request-method": "GET",
            "access-control-request-headers": "if-none-match",
        },
    )
    read = cors_door.client.get(url, headers={"origin": origin})
    again = cors_door.client.get(
        url, headers={"origin": origin, "if-none-match": read.headers["etag"]}
    )
    write = cors_door.client.options(
        f"/api/presentation/shares/{share['id']}",
        headers={"origin": "https://news.example", "access-control-request-method": "PUT"},
    )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert "if-none-match" in preflight.headers["access-control-allow-headers"].lower()
    assert read.headers["access-control-allow-origin"] in {origin, "*"}
    exposed = {
        name.strip().lower() for name in read.headers["access-control-expose-headers"].split(",")
    }
    assert {"etag", "x-dfo-freshness", "x-dfo-checked-at", "x-dfo-trace-id"} <= exposed
    assert again.status_code == 304
    assert "access-control-allow-origin" not in write.headers


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


@pytest.mark.parametrize(
    ("latest", "served", "cache_control"),
    [(True, _MOVED, "no-cache"), (False, _STORED, "public, max-age=300")],
    ids=["live-moved", "snapshot"],
)
def test_bundles_and_xlsx_serve_the_rows_of_the_frame_the_slot_serves(
    door: _Door, latest: bool, served: FrameRef, cache_control: str
):
    share = door.share(door.card(), latest=latest)
    door.newest.answer = _MOVED
    rows = door.data(share["id"])
    values = _values(rows.json())

    bundled = door.data(share["id"], format="csv", bundle="true")
    xlsx_bundle = door.data(share["id"], format="xlsx", bundle="true")
    xlsx = door.data(share["id"], format="xlsx")
    again = door.client.get(
        f"/api/presentation/shares/{share['id']}/slots/1/data",
        params={"bundle": "true", "format": "csv"},
        headers={"if-none-match": bundled.headers["etag"]},
    )

    archive = zipfile.ZipFile(io.BytesIO(bundled.content))
    (data_file,) = (
        name for name in archive.namelist() if name.startswith("data/") and name != "data/codes.csv"
    )
    bundled_rows = list(csv.DictReader(io.StringIO(archive.read(data_file).decode())))
    xlsx_archive = zipfile.ZipFile(io.BytesIO(xlsx_bundle.content))
    sheet = openpyxl.load_workbook(io.BytesIO(xlsx.content)).active
    assert sheet is not None
    header, *cells = sheet.iter_rows(values_only=True)
    for resp in (bundled, xlsx_bundle, xlsx):
        assert resp.status_code == 200
        assert resp.headers["x-dfo-freshness"] == "current"
        assert resp.headers["cache-control"] == cache_control
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "attachment" in resp.headers["content-disposition"]
    assert bundled.headers["content-type"] == xlsx_bundle.headers["content-type"]
    assert bundled.headers["content-type"] == "application/zip"
    assert {"datapackage.json", "CITATION.cff", "README.md"} <= set(archive.namelist())
    assert [float(row["value"]) for row in bundled_rows] == values
    assert any(name.endswith(".xlsx") for name in xlsx_archive.namelist())
    assert [row[header.index("value")] for row in cells] == values
    assert len({bundled.headers["etag"], xlsx_bundle.headers["etag"], xlsx.headers["etag"]}) == 3
    assert again.status_code == 304
    assert _stated(rows)[1] == served.content_version


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


def test_a_read_during_a_check_states_the_last_check_and_runs_none_of_its_own(door: _Door):
    share = door.share(door.card(), latest=True)
    during = []
    door.newest.during = lambda: during.append(door.data(share["id"]))
    first = door.data(share["id"])
    door.later(minutes=15)
    door.newest.answer = _MOVED
    door.newest.during = lambda: during.append(door.data(share["id"]))
    due = door.data(share["id"])

    (in_first, in_due) = during
    assert door.newest.asked == [_STORED, _STORED]
    assert _stated(in_first)[:2] == ("pending", _STORED.content_version)
    assert "x-dfo-checked-at" not in in_first.headers
    assert _stated(first)[0] == "current"
    assert _stated(in_due)[:2] == ("current", _STORED.content_version)
    assert in_due.headers["x-dfo-checked-at"] == _START.isoformat()
    assert _stated(due)[:2] == ("current", _MOVED.content_version)
    assert due.headers["x-dfo-checked-at"] == (_START + timedelta(minutes=15)).isoformat()


def test_a_claim_whose_read_never_records_its_check_expires_within_two_minutes(door: _Door):
    share = door.share(door.card(), latest=True)
    door.data(share["id"])
    door.later(minutes=15)
    deps = getattr(door.client.app.state, SHARES_DEPS_SLOT.name)
    card = deps.share_store.read_share(share_id=share["id"]).slots[0].card
    claims = []
    door.newest.during = lambda: claims.append(
        deps.share_store.slot_check(share_id=share["id"], slot=1, card=card)
    )
    door.data(share["id"])
    # Put back the claim as a read that died before recording its check would have left it.
    (claim,) = claims
    deps.share_store.record_slot_check(share_id=share["id"], slot=1, card=card, check=claim)

    door.clock.at = claim.next_check_at - timedelta(seconds=1)
    held = door.data(share["id"])
    door.later(seconds=1)
    door.newest.answer = _MOVED
    freed = door.data(share["id"])

    assert claim.next_check_at - (_START + timedelta(minutes=15)) <= timedelta(minutes=2)
    assert len(door.newest.asked) == 3
    assert held.headers["x-dfo-checked-at"] == _START.isoformat()
    assert _stated(freed)[:2] == ("current", _MOVED.content_version)


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


@pytest.mark.parametrize(
    ("raised", "wait_s"),
    [
        pytest.param(
            FrameUnavailable(reason="io", detail="exemplar", retry_after_s=7.0), 7, id="io"
        ),
        pytest.param(FrameUnavailable(reason="busy", detail="exemplar"), 30, id="busy"),
        pytest.param(PresentationStoreUnavailable("exemplar"), 30, id="store-unavailable"),
        pytest.param(
            ReleaseMismatch(catalog_version="2020.1.0", detail="exemplar"),
            30,
            id="release-mismatch",
        ),
        pytest.param(
            NewestNotReady("exemplar", retry_after_s=1), 30, id="retry-after-under-the-floor"
        ),
        pytest.param(NewestNotReady("exemplar", retry_after_s=120), 120, id="retry-after"),
        pytest.param(FrameUnavailable(reason="expired", detail="exemplar"), None, id="expired"),
        pytest.param(
            PresentationStoreUnavailable("exemplar", sqlstate="42601"),
            None,
            id="store-permanent",
        ),
    ],
)
def test_a_live_slot_serves_its_own_frame_until_a_transient_failure_s_wait_ends(
    door: _Door, monkeypatch: pytest.MonkeyPatch, raised: Exception, wait_s: int | None
):
    """A transient failure stores the slot `stale` on the card's own frame until its wait, and
    the reads inside it do not draw the failing frame. A permanent one stores it until the next
    check, past both reads here. Every read records the frame it served."""
    served: list[FrameRef] = []
    record_served = LiveShareStore.record_served

    def counting(self: LiveShareStore, **kwargs: Any) -> object:
        served.append(kwargs["frame"])
        return record_served(self, **kwargs)

    monkeypatch.setattr(LiveShareStore, "record_served", counting)
    share = door.share(door.card(), latest=True)
    door.newest.answer = _MOVED
    door.frame_store.refusals[_MOVED] = raised
    served.clear()

    first = door.data(share["id"])
    door.later(seconds=(wait_s or 60) - 1)
    inside = door.data(share["id"])
    moved_loads = door.frame_store.loads.count(_MOVED)
    del door.frame_store.refusals[_MOVED]
    door.later(seconds=2)
    after = door.data(share["id"])

    stale = ("stale", _STORED.content_version, "live")
    assert _stated(first) == _stated(inside) == stale
    assert moved_loads == 1
    if wait_s is None:
        assert _stated(after) == stale
        assert door.newest.asked == [_STORED]
        assert served == [_STORED] * 3
    else:
        assert _stated(after) == ("current", _MOVED.content_version, "live")
        assert door.newest.asked == [_STORED] * 2
        assert served == [_STORED, _STORED, _MOVED]


@pytest.mark.parametrize("last_drawable", [True, False])
def test_a_transient_failure_keeps_a_live_slot_on_its_last_good_frame(
    door: _Door, last_drawable: bool
):
    """A slot current on a newer frame serves that frame `stale` through a transient failure of
    a newer one still, and keeps it through the pending check that follows. When that frame
    cannot be drawn either, the slot serves the card's own frame in its place."""
    newer = FrameRef(spec_id=_STORED.spec_id, content_version="cv_" + "c" * 32)
    door.frame_store.refusals[newer] = FrameUnavailable(
        reason="io", detail="exemplar", retry_after_s=7.0
    )
    share = door.share(door.card(), latest=True)
    door.newest.answer = _MOVED
    first = door.data(share["id"])
    door.later(minutes=16)
    door.newest.answer = newer
    if not last_drawable:
        door.frame_store.refusals[_MOVED] = FrameUnavailable(reason="expired", detail="exemplar")
    failed = door.data(share["id"])
    door.later(seconds=6)
    inside = door.data(share["id"])
    door.later(seconds=2)
    door.newest.answer = NewestPending(retry_after_s=30)
    pending = door.data(share["id"])

    held = _MOVED if last_drawable else _STORED
    assert _stated(first) == ("current", _MOVED.content_version, "live")
    assert _stated(failed) == _stated(inside) == ("stale", held.content_version, "live")
    assert _stated(pending) == ("pending", held.content_version, "live")
    assert door.frame_store.loads.count(newer) == 1


def test_a_live_slot_with_no_datasets_surface_serves_its_cards_frame_as_stale(
    release_registry: FakeReleaseRegistry,
):
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=_frame(_STORED))
    presentation = FakePresentationStore()
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=LiveShareStore(store=presentation),
        allowances=_Allowances(limit=50),
        caller_settings=CallerSettings(caller_hmac_key=SecretStr("share-data-test-key")),
    )
    with TestClient(
        create_app([surface]), raise_server_exceptions=False, headers=_HEADERS
    ) as client:
        door = _Door(client, frame_store=frame_store, newest=_Newest(), presentation=presentation)
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
        ("slots/1/data?format=zip", 400, "request_invalid"),
        ("slots/1/data?bundle=true", 400, "request_invalid"),
    ],
    ids=["no-such-slot", "slot-zero", "unknown-format", "json-bundle"],
)
def test_a_slot_the_share_does_not_have_refuses(door: _Door, path: str, status: int, code: str):
    share = door.share(door.card(), latest=True)

    resp = door.client.get(f"/api/presentation/shares/{share['id']}/{path}")

    assert (resp.status_code, resp.json()["code"]) == (status, code)


@pytest.mark.parametrize(
    ("reason", "status", "code", "retryable"),
    [
        ("expired", 404, "frame_unavailable", True),
        ("revoked", 404, "frame_unavailable", False),
        ("ops_unsupported", 404, "frame_unavailable", True),
        ("missing_rendering", 404, "frame_unavailable", False),
        ("digest_mismatch", 404, "frame_unavailable", False),
        ("quarantined", 404, "frame_unavailable", False),
        ("recipe_pending", 404, "frame_unavailable", True),
        ("unknown", 404, "frame_unavailable", False),
        ("io", 503, "service_unavailable", True),
        ("recipe_unreadable", 500, "internal_error", False),
    ],
)
def test_a_slot_whose_frame_does_not_load_answers_by_cause(
    door: _Door,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    status: int,
    code: str,
    retryable: bool,
):
    share = door.share(door.card(), latest=False)

    def unloadable(*, frame_ref: FrameRef) -> ResolvedFrame:
        raise FrameUnavailable(reason=reason, detail="exemplar")

    monkeypatch.setattr(door.frame_store, "load", unloadable)
    resp = door.data(share["id"])

    body = resp.json()
    assert (resp.status_code, body["code"], body["retryable"]) == (status, code, retryable)
    assert body["cause"]["code"] == reason


def test_an_emptied_slot_and_a_notes_slot_answer_not_found(door: _Door):
    note = door.card({"type": "note_narration", "text": "Grants rose."})
    share = door.share(door.card(), note, latest=True)
    door.client.put(
        f"/api/presentation/shares/{share['id']}",
        json={"cards": [{"component_id": note, "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )

    note_slot, emptied = door.data(share["id"], 1), door.data(share["id"], 2)
    note_card, emptied_card = door.slot_card(share["id"], 1), door.slot_card(share["id"], 2)

    assert (note_slot.status_code, note_slot.json()["code"]) == (404, "not_found")
    assert (emptied.status_code, emptied.json()["code"]) == (404, "not_found")
    # A note draws no frame, so its card is its stored read.
    assert note_card.status_code == 200, note_card.json()
    assert (note_card.json()["freshness"], note_card.json()["retention"]) == ("current", None)
    assert (emptied_card.status_code, emptied_card.json()["code"]) == (404, "not_found")


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

    monkeypatch.setattr(presentation_retention, "frame_retention", stated)
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


def _seed_timeline(door: _Door, share_id: str, times: list[datetime]) -> None:
    """Replace the share's timeline with one emptied-slot entry per time, `id` in list order."""
    rows = door.presentation.state.served_frames
    rows[:] = [row for row in rows if row["share_id"] != share_id]
    for at in times:
        rows.append(
            {
                "id": len(rows) + 1,
                "share_id": share_id,
                "slot": 1 + len(rows) % 3,
                "component_id": None,
                "component_revision": None,
                "frame_spec_id": None,
                "frame_content_version": None,
                "rendering": None,
                "first_served_at": at,
            }
        )


def _pages(door: _Door, path: str, limit: int) -> list[dict]:
    pages = [door.client.get(path, params={"limit": limit}).json()]
    while pages[-1]["next"] is not None:
        params = {"limit": limit, "before": pages[-1]["next"]}
        pages.append(door.client.get(path, params=params).json())
    return pages


@pytest.mark.parametrize(
    "offsets_us",
    [[0, 1_000, 2_000, 3_000, 4_000], [0, 0, 1, 1, 999]],
    ids=["apart", "one-millisecond"],
)
def test_the_history_pages_every_entry_once_in_the_order_it_lists_them(
    door: _Door, offsets_us: list[int]
):
    share = door.share(door.card(), latest=True)
    path = f"/api/presentation/shares/{share['id']}/history"
    _seed_timeline(door, share["id"], [_START + timedelta(microseconds=us) for us in offsets_us])

    whole = door.client.get(path).json()
    pages = _pages(door, path, limit=2)

    assert whole["next"] is None
    assert [entry["first_served_at"] for entry in whole["served"]] == sorted(
        (entry["first_served_at"] for entry in whole["served"]), reverse=True
    )
    assert [len(page["served"]) for page in pages] == [2, 2, 1]
    assert [entry for page in pages for entry in page["served"]] == whole["served"]


def test_the_history_holds_a_hundred_entries_unless_asked_for_more(door: _Door):
    share = door.share(door.card(), latest=True)
    path = f"/api/presentation/shares/{share['id']}/history"
    _seed_timeline(door, share["id"], [_START + timedelta(seconds=n) for n in range(101)])

    default = door.client.get(path).json()
    most = door.client.get(path, params={"limit": 500}).json()
    rest = door.client.get(path, params={"before": default["next"]}).json()

    assert (len(default["served"]), len(most["served"]), most["next"]) == (100, 101, None)
    assert default["served"] == most["served"][:100]
    assert (rest["served"], rest["next"]) == (most["served"][100:], None)


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "501"},
        {"limit": "ten"},
        {"before": ""},
        {"before": "not a cursor"},
        {"before": "MTc2MTY0MjAwMDAwMDAwMC4w"},
        {"before": "MTc2MTY0MjAwMDAwMDAwMC4xMg=="},
        {"before": "OTk5OTk5OTk5OTk5OTk5OTk5OTk5OS4x"},
    ],
    ids=["zero", "over-max", "not-a-number", "empty", "not-base64", "id-zero", "padded", "far"],
)
def test_a_history_limit_out_of_range_or_a_cursor_no_page_gave_refuses_request_invalid(
    door: _Door, params: dict[str, str]
):
    share = door.share(door.card(), latest=True)

    resp = door.client.get(f"/api/presentation/shares/{share['id']}/history", params=params)

    assert (resp.status_code, resp.json()["code"]) == (400, "request_invalid")
    assert resp.headers["access-control-allow-origin"] == "*"


def test_the_history_reads_each_frames_retention_once(door: _Door, monkeypatch: pytest.MonkeyPatch):
    read: list[FrameRef] = []

    def stated(frame_ref: FrameRef, *, snapshot_reader) -> dict[str, object]:
        read.append(frame_ref)
        return {"pinned": True}

    monkeypatch.setattr(presentation_retention, "frame_retention", stated)
    share = door.share(door.card(), door.card(), door.card(), latest=True)
    read.clear()

    served = door.client.get(f"/api/presentation/shares/{share['id']}/history").json()["served"]

    assert [(entry["frame"]["content_version"], entry["fetchable"]) for entry in served] == [
        (_STORED.content_version, True)
    ] * 3
    assert read == [_STORED]


def test_a_create_and_a_replace_record_what_the_slots_show_before_any_read(door: _Door):
    first, second = door.card(), door.card()
    share = door.share(first, latest=True)
    history_path = f"/api/presentation/shares/{share['id']}/history"

    def shown() -> set[tuple[int, str, str]]:
        served = door.client.get(history_path).json()["served"]
        return {
            (entry["slot"], entry["component_id"], entry["frame"]["content_version"])
            for entry in served
        }

    created = shown()
    door.client.put(
        f"/api/presentation/shares/{share['id']}",
        json={"cards": [{"component_id": second, "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )

    assert created == {(1, first, _STORED.content_version)}
    assert shown() == {(1, first, _STORED.content_version), (1, second, _STORED.content_version)}
    assert door.newest.asked == []


def test_the_history_records_a_slot_a_replace_empties_and_refills(door: _Door):
    cards = [door.card(), door.card(), door.card()]
    share = door.share(*cards, latest=True)
    share_path = f"/api/presentation/shares/{share['id']}"

    def replace(*shown: str) -> None:
        resp = door.client.put(
            share_path,
            json={"cards": [{"component_id": card, "revision": 1} for card in shown]},
            headers={_TOKEN: share["write_token"]},
        )
        assert resp.status_code == 200, resp.text

    def timeline(slot: int) -> list[str | None]:
        served = door.client.get(f"{share_path}/history").json()["served"]
        return [entry["component_id"] for entry in reversed(served) if entry["slot"] == slot]

    replace(cards[0])
    replace(cards[0])
    emptied = door.client.get(f"{share_path}/history").json()["served"]
    replace(*cards)

    assert [timeline(slot) for slot in (1, 2, 3)] == [
        [cards[0]],
        [cards[1], None, cards[1]],
        [cards[2], None, cards[2]],
    ]
    assert [entry for entry in emptied if entry["component_id"] is None] == [
        {
            "slot": slot,
            "component_id": None,
            "revision": None,
            "frame": None,
            "rendering": None,
            "first_served_at": entry["first_served_at"],
            "retention": None,
            "fetchable": False,
        }
        for slot, entry in ((3, emptied[0]), (2, emptied[1]))
    ]


def test_a_slot_whose_card_becomes_a_note_records_an_empty_entry(door: _Door):
    card = door.card()
    note = door.card({"type": "note_narration", "text": "Grants rose."})
    share = door.share(card, latest=True)
    share_path = f"/api/presentation/shares/{share['id']}"

    resp = door.client.put(
        share_path,
        json={"cards": [{"component_id": note, "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )

    assert resp.status_code == 200, resp.text
    served = door.client.get(f"{share_path}/history").json()["served"]
    assert [(entry["component_id"], entry["frame"]) for entry in served] == [
        (None, None),
        (card, {"spec_id": _STORED.spec_id, "content_version": _STORED.content_version}),
    ]


def test_a_write_whose_history_cannot_be_recorded_commits_nothing(door: _Door):
    """A write records its history in its own transaction, so a failed record leaves no
    revision without its entries: the create stores no share and gives its unit back, the
    replace leaves the share at its revision, and a retry records both."""
    first, second = door.card(), door.card()
    deps = getattr(door.client.app.state, SHARES_DEPS_SLOT.name)
    body = {
        "cards": [{"component_id": first, "revision": 1}],
        "latest": True,
        "write_token": mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX),
    }

    def failing(fail: bool) -> None:
        door.presentation.fail_before = (
            (lambda sql, _: sql.startswith("INSERT INTO presentation.served_frame"))
            if fail
            else None
        )

    def create():
        return door.client.post("/api/presentation/shares", json=body, headers=_HEADERS)

    failing(True)
    refused_create = create()
    failing(False)
    share = create().json()
    share_path = f"/api/presentation/shares/{share['id']}"

    def replace():
        return door.client.put(
            share_path,
            json={"cards": [{"component_id": second, "revision": 1}]},
            headers={_TOKEN: share["write_token"]},
        )

    failing(True)
    refused_replace = replace()
    failing(False)
    unreplaced = door.client.get(share_path).json()["revision"]
    replaced = replace()

    assert (refused_create.status_code, refused_create.json()["code"]) == (
        503,
        "service_unavailable",
    )
    assert (refused_replace.status_code, refused_replace.json()["code"]) == (
        503,
        "service_unavailable",
    )
    assert list(door.presentation.state.shares) == [share["id"]]
    assert len(deps.allowances.spent) == 1
    assert (unreplaced, replaced.json()["revision"]) == (1, 2)
    assert [
        (entry["slot"], entry["component_id"])
        for entry in door.client.get(f"{share_path}/history").json()["served"]
    ] == [(1, second), (1, first)]


def test_a_replace_or_a_replayed_create_records_the_frame_a_kept_live_slot_serves(door: _Door):
    """A live slot whose card keeps its place serves its last check's frame, so a write that
    keeps the card records that frame and never the card's own, which viewers no longer see."""
    kept, added = door.card(), door.card()
    body = {
        "cards": [{"component_id": kept, "revision": 1}],
        "latest": True,
        "write_token": mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX),
    }
    share = door.client.post("/api/presentation/shares", json=body, headers=_HEADERS).json()
    share_path = f"/api/presentation/shares/{share['id']}"
    door.newest.answer = _MOVED
    assert _stated(door.data(share["id"]))[:2] == ("current", _MOVED.content_version)

    replaced = door.client.put(
        share_path,
        json={
            "cards": [{"component_id": kept, "revision": 1}, {"component_id": added, "revision": 1}]
        },
        headers={_TOKEN: share["write_token"]},
    )
    replayed = door.client.post("/api/presentation/shares", json=body, headers=_HEADERS)

    assert (replaced.status_code, replayed.status_code) == (200, 201)
    served = door.client.get(f"{share_path}/history").json()["served"]
    assert [
        entry["frame"]["content_version"] for entry in reversed(served) if entry["slot"] == 1
    ] == [_STORED.content_version, _MOVED.content_version]


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
    assert _drawn_on(door.slot_card(share["id"], 1).json()) == _MOVED.content_version
    assert door.newest.asked == []
    replaced = door.client.put(
        f"/api/presentation/shares/{share['id']}",
        json={"cards": [{"component_id": card, "revision": 1}]},
        headers={_TOKEN: share["write_token"]},
    )
    assert (replaced.status_code, replaced.json()["code"]) == (409, "share_not_replaceable")
