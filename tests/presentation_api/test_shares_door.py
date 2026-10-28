"""The shares door: every publication rule vector over HTTP, the share allowance, pinning on
create and replace, ids kept from the website, and take-down."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pytest
from dfo_client_contract import rule
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import SecretStr

from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef
from assembler.write_token import SHARE_WRITE_TOKEN_PREFIX, mint_write_token
from datasets_api.allowance import AllowanceExhausted, AllowanceKind
from dfo_platform import ProblemDetails
from dfo_platform.app import create_app
from dfo_platform.caller import Caller, address_key
from dfo_platform.config import CallerSettings
from presentation_api.app import presentation_surface
from presentation_api.doors.shares import MAX_SHARE_PAYLOAD_BYTES, build_shares_door
from presentation_store import LiveShareStore, ShareCard
from tests.presentation_store.conftest import FakePresentationStore

_SITE = "https://site.example"
_HMAC_KEY = "share-door-test-key"
_ADDRESS = "203.0.113.7"
_HEADERS = {"x-forwarded-for": _ADDRESS}
_TOKEN_HEADER = "x-dfo-write-token"
_CARD_NAMES = ("cmp_a", "cmp_b", "cmp_c", "cmp_d", "cmp_e")


class _Allowances:
    """An `AllowanceLedger` that admits `limit` spends per caller and records each."""

    def __init__(self, *, limit: int) -> None:
        self.limit = limit
        self.spent: list[tuple[str, AllowanceKind]] = []

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        if sum(1 for key, _ in self.spent if key == caller.key) >= self.limit:
            raise AllowanceExhausted(kind=kind, limit=self.limit, now=now)
        self.spent.append((caller.key, kind))


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
        resp = self.client.post("/api/presentation/components", json={"intent": intent})
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
            headers={_TOKEN_HEADER: token} if token is not None else {},
        )

    def unshare(self, share_id: str, *, token: str | None):
        return self.client.delete(
            f"/api/presentation/shares/{share_id}",
            headers={_TOKEN_HEADER: token} if token is not None else {},
        )

    def read(self, share_id: str):
        return self.client.get(f"/api/presentation/shares/{share_id}")

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


@pytest.fixture
def door(release_registry: FakeReleaseRegistry, frame_store: FakeFrameStore) -> Iterator[_Door]:
    share_store = LiveShareStore(store=FakePresentationStore())
    allowances = _Allowances(limit=50)
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
        share_store=share_store,
        allowances=allowances,
        caller_settings=CallerSettings(caller_hmac_key=SecretStr(_HMAC_KEY)),
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as client:
        yield _Door(client, share_store=share_store, allowances=allowances, frame_store=frame_store)


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


def test_a_replace_keeps_every_slot_url(door: _Door):
    created = door.share(["cmp_a", "cmp_b"], latest=True).json()

    replaced = door.replace(created["id"], ["cmp_c"], token=created["write_token"]).json()

    assert replaced["data_urls"] == created["data_urls"]
    assert (replaced["url"], replaced["embed_url"]) == (created["url"], created["embed_url"])
    assert replaced["revision"] == 2
    assert "write_token" not in replaced


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
    assert body["retryable"] is True
    assert int(resp.headers["retry-after"]) >= 1
    assert door.frame_store.pins == ()


def test_replace_and_delete_spend_no_allowance(door: _Door):
    created = door.share(["cmp_a"], latest=True).json()
    door.allowances.limit = 1

    assert door.replace(created["id"], ["cmp_b"], token=created["write_token"]).status_code == 200
    assert door.unshare(created["id"], token=created["write_token"]).status_code == 204
    assert len(door.allowances.spent) == 1


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
        {"latest": False, "kind": "search", "payload": "x" * MAX_SHARE_PAYLOAD_BYTES},
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
        "payload-over-bound",
    ],
)
def test_a_malformed_create_refuses_request_invalid_and_spends_nothing(door: _Door, body: dict):
    resp = door.client.post("/api/presentation/shares", json=body, headers=_HEADERS)

    _assert_refused(resp, status=400, code="request_invalid")
    assert door.allowances.spent == []


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


@pytest.mark.parametrize("share_id", ["k3Qx8vT", "k3Qx8vTa9", "k3Qx-vTa"])
def test_a_malformed_share_id_refuses_request_invalid(door: _Door, share_id: str):
    _assert_refused(door.read(share_id), status=400, code="request_invalid")


def test_a_deleted_share_answers_not_found_on_every_route(door: _Door):
    created = door.share(["cmp_a"], latest=True).json()
    share_id, token = created["id"], created["write_token"]
    assert door.unshare(share_id, token=token).status_code == 204

    _assert_refused(door.read(share_id), status=404, code="not_found")
    _assert_refused(door.replace(share_id, ["cmp_b"], token=token), status=404, code="not_found")
    _assert_refused(door.unshare(share_id, token=token), status=404, code="not_found")


def test_a_surface_over_test_doubles_without_a_share_store_mounts_no_share_route(
    release_registry: FakeReleaseRegistry, frame_store: FakeFrameStore
):
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=release_registry,
    )

    assert not surface.registry_endpoints
    assert not [r for r in surface.router.routes if isinstance(r, APIRoute) and "/shares" in r.path]


def test_every_share_route_handler_is_a_module_level_function():
    routes = [r for r in build_shares_door().router.routes if isinstance(r, APIRoute)]

    assert len(routes) == 7
    assert all("<locals>" not in getattr(route.endpoint, "__qualname__", "") for route in routes)
