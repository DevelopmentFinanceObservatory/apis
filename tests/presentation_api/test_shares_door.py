"""The shares door: every publication rule vector over HTTP, the share allowance and its refund,
the live share's data-card limit, pinning on create and replace, ids kept from the website, the
share import and its principal, take-down, and a burst of reads over its executor. Also the card allowance the components door
spends on create and revise."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from dfo_client_contract import rule
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import SecretStr

from artifact_store import Tier
from assembler.errors import FrameUnavailable
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef
from assembler.testing import malaria_ethiopia_resolved_frame
from assembler.write_token import SHARE_WRITE_TOKEN_PREFIX, mint_write_token
from datasets_api import maintenance
from datasets_api.allowance import AllowanceExhausted, AllowanceKind, LiveAllowanceLedger
from datasets_api.principals import Grant, Principals
from datasets_api.retention_admission import (
    OpenPinBudgetExhausted,
    PinBudgetExhausted,
    PinUnitHeld,
    copy_to_pinned,
)
from dfo_platform import ProblemDetails
from dfo_platform.app import create_app
from dfo_platform.caller import Caller, address_key
from dfo_platform.codes import bound_registry_endpoint, registered_status
from dfo_platform.config import CallerSettings
from dfo_platform.problem import Refusal
from presentation_api.app import presentation_surface
from presentation_api.config import PRESENTATION_COMPONENTS_MAX_CONCURRENCY
from presentation_api.doors import shares as shares_door
from presentation_api.doors.shares import MAX_SHARE_PAYLOAD_BYTES, build_shares_door
from presentation_api.live import LiveFrames
from presentation_store import LiveShareStore, ShareCard, ShareRecord
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.presentation_store.conftest import FakePresentationStore

_SITE = "https://site.example"
_HMAC_KEY = "share-door-test-key"
_ADDRESS = "203.0.113.7"
_HEADERS = {"x-forwarded-for": _ADDRESS}
_TOKEN_HEADER = "x-dfo-write-token"
_CARD_NAMES = ("cmp_a", "cmp_b", "cmp_c", "cmp_d", "cmp_e")
_PRINCIPAL_HEADER = "x-dfo-principal"
_IMPORT_PRINCIPAL = "site-share-import"
_PRINCIPALS = Principals(
    header=_PRINCIPAL_HEADER,
    secrets=(("ai-publish", Grant.publish), (_IMPORT_PRINCIPAL, Grant.share_import)),
)


class _Allowances:
    """An `AllowanceLedger` that admits `limit` spends of each kind per caller and records each
    unit held. `spent` lists the share units, `cards` the card units."""

    def __init__(self, *, limit: int) -> None:
        self.limit = limit
        self.held: list[tuple[str, AllowanceKind]] = []

    @property
    def spent(self) -> list[tuple[str, AllowanceKind]]:
        return [unit for unit in self.held if unit[1] == AllowanceKind.share]

    @property
    def cards(self) -> list[tuple[str, AllowanceKind]]:
        return [unit for unit in self.held if unit[1] == AllowanceKind.card]

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        if self.held.count((caller.key, kind)) >= self.limit:
            raise AllowanceExhausted(kind=kind, limit=self.limit, now=now)
        self.held.append((caller.key, kind))

    def refund(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        self.held.remove((caller.key, kind))


class _Door:
    """A composed app with the shares door over fakes, and five cards made through it."""

    def __init__(self, client: TestClient, *, share_store, allowances, frame_store) -> None:
        self.client = client
        self.share_store = share_store
        self.allowances = allowances
        self.frame_store = frame_store
        self.cards = {name: self._create_card(name) for name in _CARD_NAMES}

    def _create_card(self, name: str) -> str:
        intent = {
            "type": "line_chart",
            "frame_ref": {
                "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
                "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
            },
            "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
            "window": {"start": "2015", "end": "2024"},
            "title": f"Card {name}",
        }
        resp = self.client.post(
            "/api/presentation/components", json={"intent": intent}, headers=_HEADERS
        )
        assert resp.status_code == 201, resp.json()
        return resp.json()["component_id"]

    def share(self, names, *, latest: bool, title: str | None = None):
        body: dict[str, object] = {
            "cards": [{"component_id": self.cards[name], "revision": 1} for name in names],
            "latest": latest,
        }
        if title is not None:
            body["title"] = title
        return self.client.post("/api/presentation/shares", json=body, headers=_HEADERS)

    def replace(self, share_id: str, names, *, token: str | None):
        return self.client.put(
            f"/api/presentation/shares/{share_id}",
            json={"cards": [{"component_id": self.cards[name], "revision": 1} for name in names]},
            headers=_HEADERS | ({_TOKEN_HEADER: token} if token is not None else {}),
        )

    def unshare(self, share_id: str, *, token: str | None):
        return self.client.delete(
            f"/api/presentation/shares/{share_id}",
            headers={_TOKEN_HEADER: token} if token is not None else {},
        )

    def read(self, share_id: str):
        return self.client.get(f"/api/presentation/shares/{share_id}")

    def import_share(
        self, share_id: str, body: dict[str, object], *, principal: str | None = _IMPORT_PRINCIPAL
    ):
        return self.client.put(
            f"/api/presentation/shares/{share_id}/import",
            json=body,
            headers={_PRINCIPAL_HEADER: principal} if principal is not None else {},
        )

    def shown(self, share_id: str) -> dict[str, object]:
        """The share as a publication vector states it, read back over HTTP."""
        resp = self.read(share_id)
        assert resp.status_code == 200, resp.json()
        body = resp.json()
        names = {component_id: name for name, component_id in self.cards.items()}
        return {
            "latest": body["mode"] == "live",
            "slots": [
                {
                    "slot": slot["slot"],
                    "card": names[slot["card"]["component_id"]] if slot["card"] else None,
                }
                for slot in body["slots"]
            ],
        }


@contextmanager
def _serving(
    release_registry: FakeReleaseRegistry,
    frame_store: FakeFrameStore,
    *,
    principals: Principals | None,
) -> Iterator[_Door]:
    share_store = LiveShareStore(store=FakePresentationStore())
    allowances = _Allowances(limit=50)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=share_store,
        allowances=allowances,
        caller_settings=CallerSettings(caller_hmac_key=SecretStr(_HMAC_KEY)),
        principals=principals,
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        yield _Door(client, share_store=share_store, allowances=allowances, frame_store=frame_store)


@pytest.fixture
def door(release_registry: FakeReleaseRegistry, frame_store: FakeFrameStore) -> Iterator[_Door]:
    with _serving(release_registry, frame_store, principals=_PRINCIPALS) as served:
        yield served


def _assert_refused(resp, *, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.json()
    body = resp.json()
    assert body["code"] == code
    ProblemDetails.model_validate(body)
    return body


_PUBLICATION = rule("publication")


@pytest.mark.parametrize(
    "vector", _PUBLICATION["vectors"], ids=[v["name"] for v in _PUBLICATION["vectors"]]
)
def test_publication_vector(door: _Door, vector: dict):
    shares: dict[str, tuple[str, str | None]] = {}
    for step, outcome in zip(vector["steps"], vector["outcomes"], strict=True):
        name = step["share"]
        if step["op"] == "share":
            if step.get("tokenless"):
                # A guest's share, as the website import stores it: no write token at all.
                stored = door.share_store.create_share(
                    kind="card",
                    mode="live" if step["latest"] else "snapshot",
                    title=None,
                    write_token_sha256=None,
                    cards=[
                        ShareCard(component_id=door.cards[card], revision=1, frame=None)
                        for card in step["cards"]
                    ],
                    renderings={},
                )
                shares[name] = (stored.id, None)
            else:
                resp = door.share(step["cards"], latest=step["latest"])
                assert resp.status_code == 201, resp.json()
                assert resp.json()["latest"] is step["latest"]
                shares[name] = (resp.json()["id"], resp.json()["write_token"])
            assert door.shown(shares[name][0]) == outcome["share"]
            continue

        share_id, held = shares[name]
        presented = {
            None: None,
            "held": held or mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX),
            "other": mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX),
        }[step.get("write_token")]
        if step["op"] == "replace":
            resp = door.replace(share_id, step["cards"], token=presented)
        elif step["op"] == "unshare":
            resp = door.unshare(share_id, token=presented)
        else:
            assert step["op"] == "read"
            assert door.shown(share_id) == outcome["share"]
            continue

        if "error" in outcome:
            status = 409 if outcome["error"] == "share_not_replaceable" else 403
            body = _assert_refused(resp, status=status, code=outcome["error"])
            assert body["retryable"] is False
        elif "removed" in outcome:
            assert resp.status_code == 204
            assert door.read(share_id).status_code == 404
        else:
            assert resp.status_code == 200, resp.json()
            assert door.shown(share_id) == outcome["share"]


def test_a_create_returns_the_page_embed_and_slot_data_urls_and_a_write_token(door: _Door):
    resp = door.share(["cmp_a", "cmp_b"], latest=True, title="German ODA")

    assert resp.status_code == 201
    body = resp.json()
    share_id = body["id"]
    assert body == {
        "id": share_id,
        "url": f"{_SITE}/s/{share_id}",
        "embed_url": f"{_SITE}/embed/{share_id}",
        "data_urls": [
            f"{_SITE}/api/v1/presentation/shares/{share_id}/slots/1/data",
            f"{_SITE}/api/v1/presentation/shares/{share_id}/slots/2/data",
        ],
        "write_token": body["write_token"],
        "revision": 1,
        "latest": True,
    }
    assert body["write_token"].startswith("shw_")
    assert len(share_id) == 8 and share_id.isalnum()


def _create_with_token(door: _Door, token: str, *, names=("cmp_a",), title: str | None = None):
    body: dict[str, object] = {
        "cards": [{"component_id": door.cards[name], "revision": 1} for name in names],
        "latest": True,
        "write_token": token,
    }
    if title is not None:
        body["title"] = title
    return door.client.post("/api/presentation/shares", json=body, headers=_HEADERS)


def test_a_retried_create_with_the_clients_token_answers_the_first_share(door: _Door):
    token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    first = _create_with_token(door, token)

    retried = _create_with_token(door, token)

    assert (first.status_code, retried.status_code) == (201, 201)
    assert retried.json() == first.json()
    assert first.json()["write_token"] == token
    assert len(door.allowances.spent) == 1
    # The token is the share's: it changes the share the create made.
    replaced = door.replace(first.json()["id"], ["cmp_b"], token=token)
    assert replaced.status_code == 200, replaced.json()


def test_a_create_whose_pin_fails_gives_its_unit_back_so_the_retry_spends_one(
    door: _Door, monkeypatch: pytest.MonkeyPatch
):
    token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    pin = door.frame_store.pin

    def unavailable(**_: object) -> bool:
        raise FrameUnavailable(reason="io", detail="the pin failed")

    monkeypatch.setattr(door.frame_store, "pin", unavailable)
    failed = _create_with_token(door, token)
    monkeypatch.setattr(door.frame_store, "pin", pin)
    retried = _create_with_token(door, token)

    assert (failed.status_code, retried.status_code) == (503, 201)
    assert len(door.allowances.spent) == 1


@pytest.mark.parametrize(
    ("change", "status", "code"),
    [
        ({"title": "Another"}, 400, "request_invalid"),
        ({"names": ("cmp_a", "cmp_b")}, 400, "request_invalid"),
        ({"unshared": True}, 404, "not_found"),
    ],
    ids=["other-title", "other-cards", "share-taken-down"],
)
def test_a_create_reusing_a_token_for_another_body_or_a_deleted_share_is_refused(
    door: _Door, change: dict, status: int, code: str
):
    token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    first = _create_with_token(door, token).json()
    if change.pop("unshared", False):
        assert door.unshare(first["id"], token=token).status_code == 204

    resp = _create_with_token(door, token, **change)

    body = _assert_refused(resp, status=status, code=code)
    assert body["retryable"] is False
    if code == "request_invalid":
        assert body["pointer"] == "/write_token"
    assert len(door.allowances.spent) == 1


@pytest.mark.parametrize(
    "token",
    [
        mint_write_token(prefix="cmw_"),
        SHARE_WRITE_TOKEN_PREFIX + "short",
        mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX) + "x",
    ],
    ids=["component-prefix", "too-short", "too-long"],
)
def test_a_create_with_a_malformed_token_refuses_request_invalid_and_spends_nothing(
    door: _Door, token: str
):
    body = _assert_refused(_create_with_token(door, token), status=400, code="request_invalid")
    assert body["pointer"] == "/write_token"
    assert door.allowances.spent == []


def test_a_replace_keeps_every_slot_url(door: _Door):
    created = door.share(["cmp_a", "cmp_b"], latest=True).json()

    replaced = door.replace(created["id"], ["cmp_c"], token=created["write_token"]).json()

    assert replaced["data_urls"] == created["data_urls"]
    assert (replaced["url"], replaced["embed_url"]) == (created["url"], created["embed_url"])
    assert replaced["revision"] == 2
    assert "write_token" not in replaced


def _create_of_kind(door: _Door, kind: str, names: list[str], *, latest: bool) -> dict:
    resp = door.client.post(
        "/api/presentation/shares",
        json={
            "cards": [{"component_id": door.cards[name], "revision": 1} for name in names],
            "latest": latest,
            "kind": kind,
            "payload": [],
        },
        headers=_HEADERS,
    )
    assert resp.status_code == 201, resp.json()
    return resp.json()


def test_a_replace_gives_a_live_cardless_kind_no_cards(door: _Door):
    created = _create_of_kind(door, "search", [], latest=True)

    resp = door.replace(created["id"], ["cmp_a"], token=created["write_token"])

    _assert_refused(resp, status=400, code="request_invalid")
    assert door.read(created["id"]).json()["slots"] == []


def test_a_replace_of_a_conversation_share_is_refused_as_a_snapshot(door: _Door):
    created = _create_of_kind(door, "conversation", [], latest=False)

    resp = door.replace(created["id"], ["cmp_a"], token=created["write_token"])

    _assert_refused(resp, status=409, code="share_not_replaceable")
    assert door.read(created["id"]).json()["slots"] == []


@pytest.mark.parametrize(
    ("kind", "names", "replacement"),
    [
        ("canvas", ["cmp_a"], []),
        ("canvas", [], ["cmp_a", "cmp_b"]),
        ("card", ["cmp_a"], ["cmp_b"]),
    ],
    ids=["canvas-to-none", "canvas-from-none", "card"],
)
def test_a_replace_within_the_kinds_bounds_is_accepted(
    door: _Door, kind: str, names: list[str], replacement: list[str]
):
    created = _create_of_kind(door, kind, names, latest=True)

    resp = door.replace(created["id"], replacement, token=created["write_token"])

    assert resp.status_code == 200, resp.json()


def test_a_replace_of_a_card_share_with_no_cards_is_refused(door: _Door):
    created = door.share(["cmp_a"], latest=True).json()

    resp = door.replace(created["id"], [], token=created["write_token"])

    _assert_refused(resp, status=400, code="request_invalid")


def test_a_read_carries_the_stored_record_and_never_the_token(door: _Door):
    created = door.share(["cmp_a"], latest=False, title="A figure").json()

    resp = door.read(created["id"])

    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    body = resp.json()
    assert created["write_token"] not in resp.text
    assert body["kind"] == "card"
    assert (body["mode"], body["title"], body["revision"]) == ("snapshot", "A figure", 1)
    assert body["payload"] is None
    assert body["slots"] == [
        {
            "slot": 1,
            "card": {
                "component_id": door.cards["cmp_a"],
                "revision": 1,
                "frame": {
                    "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
                    "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
                },
            },
            "data_url": created["data_urls"][0],
        }
    ]
    datetime.fromisoformat(body["created_at"])


@pytest.mark.parametrize(
    ("kind", "names", "latest", "payload"),
    [
        ("canvas", ["cmp_a", "cmp_b"], True, {"widgets": [{"id": "w1", "slot": 0}]}),
        ("canvas", [], False, {"widgets": [{"id": "w1", "value": 3}]}),
        ("search", [], True, {"query": "aid to Kenya", "parts": []}),
        ("conversation", [], False, [{"role": "user", "text": "aid"}]),
    ],
    ids=["canvas-live", "canvas-without-slots", "search", "conversation"],
)
def test_a_create_stores_its_kind_and_payload_and_reads_them_back(
    door: _Door, kind: str, names: list[str], latest: bool, payload: object
):
    resp = door.client.post(
        "/api/presentation/shares",
        json={
            "cards": [{"component_id": door.cards[name], "revision": 1} for name in names],
            "latest": latest,
            "kind": kind,
            "payload": payload,
        },
        headers=_HEADERS,
    )

    assert resp.status_code == 201, resp.json()
    body = door.read(resp.json()["id"]).json()
    assert (body["kind"], body["payload"], len(body["slots"])) == (kind, payload, len(names))


@pytest.mark.parametrize("latest", [True, False], ids=["live", "snapshot"])
def test_a_create_pins_the_frame_each_card_draws(door: _Door, frame_ref: FrameRef, latest: bool):
    resp = door.share(["cmp_a", "cmp_b", "cmp_a"], latest=latest)

    assert resp.status_code == 201
    assert set(door.frame_store.pins) == {
        (frame_ref, door.cards["cmp_a"]),
        (frame_ref, door.cards["cmp_b"]),
    }


def test_a_replace_pins_the_cards_the_share_did_not_show(door: _Door, frame_ref: FrameRef):
    created = door.share(["cmp_a"], latest=True).json()
    door.frame_store._pins.clear()

    door.replace(created["id"], ["cmp_a", "cmp_b"], token=created["write_token"])

    assert door.frame_store.pins == ((frame_ref, door.cards["cmp_b"]),)


class _CopyingFrameStore(FakeFrameStore):
    """Pins each card's frame as an artifact of its own through `copy_to_pinned`, spending from
    `ledger`, so distinct cards make distinct copies and a card pinned before makes none.
    `callers` lists the caller each pin named."""

    def __init__(self, ledger: LiveAllowanceLedger) -> None:
        super().__init__()
        self.put(resolved_frame=malaria_ethiopia_resolved_frame())
        self.ledger = ledger
        self.artifacts = FakeArtifactStore()
        self.callers: list[Caller | None] = []

    def pin(self, *, frame_ref: FrameRef, component_id: str, caller: Caller | None = None) -> bool:
        self.callers.append(caller)
        stored = self.artifacts.put(component_id.encode(), tier=Tier.working)
        if stored.digest not in self.artifacts.list(tier=Tier.pinned):
            copy_to_pinned(
                self.artifacts,
                stored.digest,
                source=(stored.bucket, stored.object_name),
                allowances=self.ledger,
                caller=caller,
                clock=lambda: datetime.now(UTC),
            )
        return super().pin(frame_ref=frame_ref, component_id=component_id)


@pytest.mark.parametrize(
    ("limit", "status", "code", "retryable"),
    [
        (AllowanceKind.pin, 429, "allowance_exhausted", False),
        (AllowanceKind.open_pin_budget, 503, "service_unavailable", True),
        (AllowanceKind.pin_budget, 503, "service_unavailable", True),
    ],
    ids=["caller-allowance", "open-budget", "daily-budget"],
)
def test_replacing_a_share_with_new_cards_spends_a_pin_unit_per_copy_until_refused(
    release_registry: FakeReleaseRegistry,
    limit: AllowanceKind,
    status: int,
    code: str,
    retryable: bool,
):
    """Each replace pins one card not pinned before. The fourth copy finds three units spent and
    is refused, and a replace back to a card pinned already spends nothing and succeeds."""
    ledger = LiveAllowanceLedger(
        store=FakeVersionRecordDatasetStore(),
        limits={
            AllowanceKind.pin_budget: 500,
            AllowanceKind.open_pin_budget: 500,
            AllowanceKind.pin: 500,
        }
        | {limit: 3},
    )
    frame_store = _CopyingFrameStore(ledger)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=LiveShareStore(store=FakePresentationStore()),
        allowances=_Allowances(limit=50),
        caller_settings=CallerSettings(caller_hmac_key=SecretStr(_HMAC_KEY)),
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        door = _Door(client, share_store=None, allowances=None, frame_store=frame_store)
        created = door.share(["cmp_a"], latest=True).json()
        token = created["write_token"]
        replaced = [door.replace(created["id"], [name], token=token) for name in ("cmp_b", "cmp_c")]
        refused = door.replace(created["id"], ["cmp_d"], token=token)
        back = door.replace(created["id"], ["cmp_a"], token=token)
        snapshot = client.post(
            f"/api/presentation/shares/{created['id']}/snapshot",
            headers=_HEADERS | {_TOKEN_HEADER: token},
        )

    assert [resp.status_code for resp in replaced] == [200, 200]
    body = _assert_refused(refused, status=status, code=code)
    assert body["retryable"] is retryable
    assert refused.headers["retry-after"].isdigit()
    assert (back.status_code, snapshot.status_code) == (200, 200)
    with ledger._store.transaction() as tx:
        counts = tx.fetch_all("SELECT kind, count FROM datasets.caller_allowance")
    assert {row["kind"]: row["count"] for row in counts}[limit.value] == 3
    caller_key = address_key(_ADDRESS, hmac_key=_HMAC_KEY)
    assert {caller.key if caller else None for caller in frame_store.callers} == {caller_key}


def test_a_create_spends_one_share_allowance_unit_of_the_forwarded_caller(door: _Door):
    door.share(["cmp_a"], latest=True)

    assert door.allowances.spent == [
        (address_key(_ADDRESS, hmac_key=_HMAC_KEY), AllowanceKind.share)
    ]


def test_a_spent_allowance_refuses_429_with_retry_after_and_stores_nothing(door: _Door):
    door.allowances.limit = 1
    door.share(["cmp_a"], latest=True)
    door.frame_store._pins.clear()

    resp = door.share(["cmp_b"], latest=True)

    body = _assert_refused(resp, status=429, code="allowance_exhausted")
    assert body["retryable"] is False
    assert int(resp.headers["retry-after"]) >= 1
    assert door.frame_store.pins == ()


def test_replace_and_delete_spend_no_allowance(door: _Door):
    created = door.share(["cmp_a"], latest=True).json()
    door.allowances.limit = 1

    assert door.replace(created["id"], ["cmp_b"], token=created["write_token"]).status_code == 200
    assert door.unshare(created["id"], token=created["write_token"]).status_code == 204
    assert len(door.allowances.spent) == 1


@pytest.mark.parametrize(
    ("count", "latest", "status"),
    [(20, True, 201), (21, True, 400), (21, False, 201)],
    ids=["live-at-the-limit", "live-past-the-limit", "snapshot-past-the-live-limit"],
)
def test_a_live_create_takes_at_most_twenty_data_cards(
    door: _Door, count: int, latest: bool, status: int
):
    resp = door.share(["cmp_a"] * count, latest=latest)

    assert resp.status_code == status, resp.json()
    if status == 400:
        _assert_refused(resp, status=400, code="request_invalid")
        assert door.allowances.spent == []


def test_a_replace_of_a_live_share_takes_at_most_twenty_data_cards(door: _Door):
    created = door.share(["cmp_a"], latest=True).json()

    past = door.replace(created["id"], ["cmp_b"] * 21, token=created["write_token"])
    at = door.replace(created["id"], ["cmp_b"] * 20, token=created["write_token"])

    _assert_refused(past, status=400, code="request_invalid")
    assert at.status_code == 200, at.json()


def _card_body(title: str) -> dict[str, object]:
    return {
        "intent": {
            "type": "line_chart",
            "frame_ref": {
                "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
                "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405",
            },
            "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
            "title": title,
        }
    }


def test_a_card_create_and_revise_each_spend_one_card_unit_of_the_forwarded_caller(door: _Door):
    caller = address_key(_ADDRESS, hmac_key=_HMAC_KEY)
    created = door.client.post(
        "/api/presentation/components", json=_card_body("Card"), headers=_HEADERS
    ).json()

    revised = door.client.patch(
        f"/api/presentation/components/{created['component_id']}",
        json={"intent": {"title": "Card, revised"}},
        headers={**_HEADERS, _TOKEN_HEADER: created["write_token"]},
    )

    assert revised.status_code == 200, revised.json()
    # The five fixture cards, this create and its revision.
    assert door.allowances.cards == [(caller, AllowanceKind.card)] * 7
    assert door.allowances.spent == []


def test_a_spent_card_allowance_refuses_create_and_revise_429_with_retry_after(door: _Door):
    created = door.client.post(
        "/api/presentation/components", json=_card_body("Card"), headers=_HEADERS
    ).json()
    door.allowances.limit = len(door.allowances.cards)

    create = door.client.post(
        "/api/presentation/components", json=_card_body("Another"), headers=_HEADERS
    )
    revise = door.client.patch(
        f"/api/presentation/components/{created['component_id']}",
        json={"intent": {"title": "Card, revised"}},
        headers={**_HEADERS, _TOKEN_HEADER: created["write_token"]},
    )

    with bound_registry_endpoint("presentation.write"):
        registered = registered_status("allowance_exhausted")
    for resp in (create, revise):
        body = _assert_refused(resp, status=registered, code="allowance_exhausted")
        assert body["retryable"] is False
        assert int(resp.headers["retry-after"]) >= 1


def test_a_card_create_retried_with_its_token_spends_one_unit(door: _Door):
    held = len(door.allowances.cards)
    body = {**_card_body("Card"), "write_token": mint_write_token(prefix="cmw_")}

    first = door.client.post("/api/presentation/components", json=body, headers=_HEADERS)
    retried = door.client.post("/api/presentation/components", json=body, headers=_HEADERS)

    assert (first.status_code, retried.status_code) == (201, 201)
    assert retried.json()["component_id"] == first.json()["component_id"]
    assert len(door.allowances.cards) == held + 1


def test_a_refused_card_write_gives_its_unit_back(door: _Door):
    held = len(door.allowances.cards)

    resp = door.client.post(
        "/api/presentation/components", json={"intent": {"type": "no_such_card"}}, headers=_HEADERS
    )

    assert resp.status_code == 400, resp.json()
    assert len(door.allowances.cards) == held


def test_a_card_write_under_the_publish_principal_spends_nothing(
    release_registry: FakeReleaseRegistry, frame_store: FakeFrameStore
):
    allowances = _Allowances(limit=50)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=LiveShareStore(store=FakePresentationStore()),
        allowances=allowances,
        caller_settings=CallerSettings(caller_hmac_key=SecretStr(_HMAC_KEY)),
        principals=Principals(
            header="x-dfo-principal",
            secrets=(("ai-publish", Grant.publish), ("site-pin", Grant.pin)),
        ),
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        published = client.post(
            "/api/presentation/components",
            json=_card_body("Card"),
            headers={"x-dfo-principal": "ai-publish"},
        )
        pinned = client.post(
            "/api/presentation/components",
            json=_card_body("Card"),
            headers={**_HEADERS, "x-dfo-principal": "site-pin"},
        )

    assert (published.status_code, pinned.status_code) == (201, 201)
    assert allowances.cards == [(address_key(_ADDRESS, hmac_key=_HMAC_KEY), AllowanceKind.card)]


@pytest.mark.parametrize(
    "body",
    [
        {"cards": [{"component_id": "cmp_x", "revision": 1}]},
        {"cards": [{"component_id": "cmp_x", "revision": 1}], "latest": "true"},
        {"cards": [], "latest": True},
        {"cards": [{"component_id": "cmp_x", "revision": 0}], "latest": True},
        {"cards": [{"component_id": "cmp_x", "revision": 1}], "latest": True, "style": "x"},
        {"cards": [{"component_id": "cmp_x", "revision": 1}], "latest": True, "kind": "search"},
        {"latest": True, "kind": "deck"},
        {"latest": True, "kind": "conversation"},
    ],
    ids=[
        "latest-missing",
        "latest-not-a-boolean",
        "no-cards",
        "revision-zero",
        "unknown-field",
        "search-with-cards",
        "unknown-kind",
        "live-conversation",
    ],
)
def test_a_malformed_create_refuses_request_invalid_and_spends_nothing(door: _Door, body: dict):
    resp = door.client.post("/api/presentation/shares", json=body, headers=_HEADERS)

    _assert_refused(resp, status=400, code="request_invalid")
    assert door.allowances.spent == []


def test_an_oversize_payload_refuses_payload_too_large_and_spends_nothing(door: _Door):
    body = {"latest": False, "kind": "search", "payload": "x" * MAX_SHARE_PAYLOAD_BYTES}
    resp = door.client.post("/api/presentation/shares", json=body, headers=_HEADERS)

    _assert_refused(resp, status=413, code="payload_too_large")
    assert door.allowances.spent == []


@pytest.mark.parametrize(
    "payload", ["NaN", "Infinity", '{"rows": [1, -Infinity]}'], ids=["nan", "infinity", "nested"]
)
def test_a_payload_holding_nan_or_infinity_refuses_request_invalid_before_any_pin(
    door: _Door, payload: str
):
    """Python's JSON parser reads NaN and Infinity, which jsonb refuses, so the create refuses
    them as invalid before it spends or pins anything."""
    door.frame_store._pins.clear()
    body = (
        f'{{"cards": [{{"component_id": "{door.cards["cmp_a"]}", "revision": 1}}], '
        f'"latest": true, "payload": {payload}}}'
    )

    resp = door.client.post(
        "/api/presentation/shares",
        content=body.encode(),
        headers={**_HEADERS, "content-type": "application/json"},
    )

    problem = _assert_refused(resp, status=400, code="request_invalid")
    assert problem["pointer"] == "/payload"
    assert door.allowances.spent == []
    assert door.frame_store.pins == ()


def test_a_card_the_store_lacks_refuses_not_found_and_spends_nothing(door: _Door):
    resp = door.client.post(
        "/api/presentation/shares",
        json={"cards": [{"component_id": "cmp_missing", "revision": 1}], "latest": True},
        headers=_HEADERS,
    )

    _assert_refused(resp, status=404, code="not_found")
    assert door.allowances.spent == []


def test_a_create_with_no_nameable_caller_refuses_and_spends_nothing(door: _Door):
    resp = door.client.post(
        "/api/presentation/shares",
        json={"cards": [{"component_id": door.cards["cmp_a"], "revision": 1}], "latest": True},
    )

    _assert_refused(resp, status=500, code="internal_error")
    assert door.allowances.spent == []


def test_a_share_kept_from_the_website_reads_and_refuses_every_token(door: _Door):
    imported = LiveShareStore(
        store=door.share_store._store, mint_id=lambda: "k3Qx8vTa"
    ).create_share(
        kind="card",
        mode="live",
        title="Imported",
        write_token_sha256=None,
        cards=[ShareCard(component_id=door.cards["cmp_a"], revision=1, frame=None)],
        renderings={},
    )

    resp = door.read(imported.id)

    assert resp.status_code == 200
    assert resp.json()["id"] == "k3Qx8vTa"
    assert resp.json()["slots"][0]["card"]["frame"] is None
    _assert_refused(
        door.unshare("k3Qx8vTa", token=mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)),
        status=403,
        code="write_token_invalid",
    )
    # Ids are case-sensitive.
    _assert_refused(door.read("K3QX8VTA"), status=404, code="not_found")


def _import_body(door: _Door, *, token: str | None, **changes: object) -> dict[str, object]:
    """A website share as its export line states it, less the id, with the digest of `token`."""
    body: dict[str, object] = {
        "kind": "card",
        "mode": "live",
        "owner": "user_1" if token is not None else None,
        "created_at": "2026-09-01T10:00:00.000001Z",
        "title": "Imported",
        "slots": [{"component_id": door.cards["cmp_a"], "revision": 1, "frame": None}],
        "payload": {"held": True},
        "write_token_sha256": (
            hashlib.sha256(token.encode()).hexdigest() if token is not None else None
        ),
    }
    return body | changes


def _stored(read) -> tuple[int, object]:
    """A share read's status and body, or its refusal's code."""
    return read.status_code, read.json() if read.status_code == 200 else read.json()["code"]


def test_an_import_answers_201_then_200_and_the_websites_token_changes_the_share(door: _Door):
    token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    body = _import_body(door, token=token)
    door.frame_store._pins.clear()

    first = door.import_share("k3Qx8vTa", body)
    pinned = door.frame_store.pins
    # The first import wins: a re-run stating another title finds it and pins nothing.
    again = door.import_share("k3Qx8vTa", body | {"title": "Changed"})

    assert (first.status_code, first.json()) == (201, {"id": "k3Qx8vTa", "imported": True})
    assert (again.status_code, again.json()) == (200, {"id": "k3Qx8vTa", "imported": False})
    assert door.frame_store.pins == pinned
    read = door.read("k3Qx8vTa").json()
    assert (read["kind"], read["mode"], read["title"], read["payload"], read["revision"]) == (
        "card",
        "live",
        "Imported",
        {"held": True},
        1,
    )
    assert read["created_at"] == "2026-09-01T10:00:00.000001+00:00"
    assert [pin[1] for pin in door.frame_store.pins] == [door.cards["cmp_a"]]
    assert door.allowances.spent == []
    assert door.replace("k3Qx8vTa", ["cmp_b"], token=token).status_code == 200


@pytest.mark.parametrize(
    ("principals", "presented", "status"),
    [
        (_PRINCIPALS, _IMPORT_PRINCIPAL, 201),
        (_PRINCIPALS, None, 403),
        (_PRINCIPALS, "wrong", 403),
        (_PRINCIPALS, "ai-publish", 403),
        (None, _IMPORT_PRINCIPAL, 403),
        (Principals(header=_PRINCIPAL_HEADER, secrets=(("", Grant.share_import),)), "", 403),
    ],
    ids=["right", "none", "wrong", "other-grant", "unset", "empty"],
)
def test_the_import_admits_its_principal_alone(
    release_registry: FakeReleaseRegistry,
    frame_store: FakeFrameStore,
    principals: Principals | None,
    presented: str | None,
    status: int,
):
    with _serving(release_registry, frame_store, principals=principals) as served:
        resp = served.import_share(
            "k3Qx8vTa", _import_body(served, token=None), principal=presented
        )
        read = served.read("k3Qx8vTa")

    if status == 201:
        assert resp.status_code == 201, resp.json()
        assert read.status_code == 200
    else:
        _assert_refused(resp, status=403, code="share_import_not_permitted")
        _assert_refused(read, status=404, code="not_found")


@pytest.mark.parametrize(
    ("held", "changes", "status", "code"),
    [
        ("made-here", {}, 409, "share_id_taken"),
        ("other-digest", {}, 409, "share_id_taken"),
        ("guest", {}, 409, "share_id_taken"),
        ("digest-elsewhere", {}, 409, "share_id_taken"),
        (None, {"owner": None}, 400, "request_invalid"),
        (None, {"write_token_sha256": "A" * 64}, 400, "request_invalid"),
        (None, {"id": "k3Qx8vTa"}, 400, "request_invalid"),
        (
            None,
            {"slots": [{"component_id": "cmp_missing", "revision": 1, "frame": None}]},
            404,
            "not_found",
        ),
        (None, "note", 400, "request_invalid"),
    ],
    ids=[
        "id-holds-a-share-made-here",
        "id-holds-it-under-another-digest",
        "id-holds-it-with-no-digest",
        "digest-held-by-another-share",
        "digest-without-owner",
        "uppercase-digest",
        "id-in-the-body",
        "unknown-card",
        "slot-drawing-no-frame",
    ],
)
def test_an_import_it_cannot_place_is_refused_and_changes_nothing(
    door: _Door, held: str | None, changes: dict[str, object] | str, status: int, code: str
):
    token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    if held == "made-here":
        LiveShareStore(store=door.share_store._store, mint_id=lambda: "k3Qx8vTa").create_share(
            kind="card", mode="live", title=None, write_token_sha256=None, cards=[], renderings={}
        )
    elif held == "other-digest":
        other = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
        assert door.import_share("k3Qx8vTa", _import_body(door, token=other)).status_code == 201
    elif held == "guest":
        assert door.import_share("k3Qx8vTa", _import_body(door, token=None)).status_code == 201
    elif held == "digest-elsewhere":
        assert door.import_share("Zz9yXw8v", _import_body(door, token=token)).status_code == 201
    before = _stored(door.read("k3Qx8vTa"))
    if changes == "note":
        note = door.client.post(
            "/api/presentation/components",
            json={"intent": {"type": "note_narration", "text": "Aid rose."}},
            headers=_HEADERS,
        ).json()["component_id"]
        changes = {"slots": [{"component_id": note, "revision": 1, "frame": None}]}
    assert isinstance(changes, dict)
    door.frame_store._pins.clear()

    resp = door.import_share("k3Qx8vTa", _import_body(door, token=token, **changes))

    problem = _assert_refused(resp, status=status, code=code)
    if held is None and changes.get("slots") and code == "request_invalid":
        assert problem["pointer"] == "/slots/0/frame"
    assert _stored(door.read("k3Qx8vTa")) == before
    assert door.frame_store.pins == ()


@pytest.mark.parametrize("share_id", ["k3Qx8vT", "k3Qx8vTa9", "k3Qx-vTa"])
def test_a_malformed_share_id_refuses_request_invalid(door: _Door, share_id: str):
    _assert_refused(door.read(share_id), status=400, code="request_invalid")


@pytest.mark.parametrize("by", ["owner", "operator"])
def test_a_deleted_share_answers_not_found_on_every_route_but_its_holders_delete(
    door: _Door, by: str
):
    """A retried delete by the token's holder answers 204 again, so a lost 204 is safe to
    retry; any other delete of the taken-down share answers as a share that never was. The
    operator's `take-down-share` leaves the share as the owner's delete does."""
    created = door.share(["cmp_a"], latest=True).json()
    share_id, token = created["id"], created["write_token"]
    other = door.share(["cmp_b"], latest=True).json()
    if by == "owner":
        assert door.unshare(share_id, token=token).status_code == 204
    else:
        maintenance.run_take_down_share(share_store=door.share_store, share_id=share_id)

    _assert_refused(door.read(share_id), status=404, code="not_found")
    _assert_refused(door.replace(share_id, ["cmp_b"], token=token), status=404, code="not_found")
    assert door.unshare(share_id, token=token).status_code == 204
    _assert_refused(
        door.unshare(share_id, token=other["write_token"]), status=404, code="not_found"
    )
    _assert_refused(door.unshare(share_id, token=None), status=404, code="not_found")
    _assert_refused(door.unshare("Zz9yXw8v", token=token), status=404, code="not_found")


def test_a_surface_over_test_doubles_without_a_share_store_mounts_no_share_route(
    release_registry: FakeReleaseRegistry, frame_store: FakeFrameStore
):
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
    )

    assert "presentation.shares" not in surface.registry_endpoints.values()
    assert not [r for r in surface.router.routes if isinstance(r, APIRoute) and "/shares" in r.path]


def test_a_frame_store_the_door_builds_pins_through_the_door_s_ledger(
    monkeypatch: pytest.MonkeyPatch, frame_store: FakeFrameStore
):
    allowances = _Allowances(limit=50)
    given: list[object] = []

    class _Graph:
        def close(self) -> None:
            pass

    def built(*, settings, allowances):
        given.append(allowances)
        return LiveFrames(
            frame_store=frame_store,
            unmetered_frame_store=frame_store,
            graph=_Graph(),
            snapshot_reader=None,
            recipe_reader=None,
        )

    monkeypatch.setattr(shares_door, "build_live_frame_store", built)
    monkeypatch.setattr(shares_door.PresentationSettings, "from_env", classmethod(lambda cls: None))
    door = build_shares_door(
        share_store=LiveShareStore(store=FakePresentationStore()),
        revision_store=FakeRevisionStore(),
        allowances=allowances,
        caller=CallerSettings(caller_hmac_key=SecretStr(_HMAC_KEY)),
    )
    assert door.lifespan is not None

    async def enter() -> None:
        async with door.lifespan(FastAPI()):
            pass

    asyncio.run(enter())

    assert given == [allowances]


def test_every_share_route_handler_is_a_module_level_function():
    routes = [r for r in build_shares_door().router.routes if isinstance(r, APIRoute)]

    assert len(routes) == 9
    assert all("<locals>" not in getattr(route.endpoint, "__qualname__", "") for route in routes)


class _GatedShareStore(LiveShareStore):
    """Holds every share read until `gate` is set, counting the reads holding a thread."""

    def __init__(self) -> None:
        super().__init__(store=FakePresentationStore())
        self.gate = threading.Event()
        self.gate.set()
        self.full = threading.Event()
        self._lock = threading.Lock()
        self._held = 0

    def read_share(self, *, share_id: str) -> ShareRecord:
        with self._lock:
            self._held += 1
            if self._held == PRESENTATION_COMPONENTS_MAX_CONCURRENCY:
                self.full.set()
        try:
            assert self.gate.wait(5)
            return super().read_share(share_id=share_id)
        finally:
            with self._lock:
                self._held -= 1


@pytest.fixture
def gated(release_registry: FakeReleaseRegistry, frame_store: FakeFrameStore) -> Iterator[_Door]:
    share_store = _GatedShareStore()
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=share_store,
        allowances=_Allowances(limit=50),
        caller_settings=CallerSettings(caller_hmac_key=SecretStr(_HMAC_KEY)),
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        yield _Door(client, share_store=share_store, allowances=None, frame_store=frame_store)


def _hold_every_thread(door: _Door, share_id: str) -> list[threading.Thread]:
    """Start one read per executor thread and return once every one holds its thread."""
    store = door.share_store
    store.gate.clear()
    held = [
        threading.Thread(target=door.read, args=(share_id,))
        for _ in range(PRESENTATION_COMPONENTS_MAX_CONCURRENCY)
    ]
    for thread in held:
        thread.start()
    assert store.full.wait(5)
    return held


def test_a_read_past_the_executor_waits_for_a_thread_and_is_served(gated: _Door):
    share_id = gated.share(["cmp_a"], latest=False).json()["id"]
    held = _hold_every_thread(gated, share_id)
    answers: list[Any] = []
    waiting = threading.Thread(target=lambda: answers.append(gated.read(share_id)))
    waiting.start()
    # Long enough for the read to reach the full executor, where it used to refuse at once.
    time.sleep(0.3)
    gated.share_store.gate.set()
    for thread in (*held, waiting):
        thread.join(5)

    assert answers[0].status_code == 200, answers[0].json()


def test_a_read_that_waits_past_the_admission_wait_refuses_503(
    gated: _Door, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(shares_door, "SHARES_ADMISSION_WAIT_SECONDS", 0.1)
    share_id = gated.share(["cmp_a"], latest=False).json()["id"]
    held = _hold_every_thread(gated, share_id)
    try:
        refused = gated.read(share_id)
    finally:
        gated.share_store.gate.set()
        for thread in held:
            thread.join(5)

    body = _assert_refused(refused, status=503, code="service_unavailable")
    assert body["retryable"] is True


_DAY = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("refusal", "detail"),
    [
        (PinUnitHeld(retry_after=10), PinUnitHeld.detail),
        (PinBudgetExhausted(limit=1, now=_DAY), PinBudgetExhausted.detail),
        (OpenPinBudgetExhausted(limit=1, now=_DAY), OpenPinBudgetExhausted.detail),
    ],
    ids=["held", "budget", "open-budget"],
)
def test_a_pin_capacity_refusal_answers_the_pin_doors_detail(refusal: Refusal, detail: str):
    spec = shares_door.classify_shares_exception(refusal)

    assert (spec.status, spec.code, spec.detail) == (503, "service_unavailable", detail)
