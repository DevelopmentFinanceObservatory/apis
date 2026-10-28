"""The dashboard journey through the composed Datasets and Presentation app, as a client sees it.

Two readings are minted: `DATASET` read summed by recipient, and `GRANTS`. An analysis tree
compares them and answers a format 2 ref to a derived frame, whose file hashes to the ref's
digest. A card on that frame cites the roots' sources, and a live share of the card serves it
through its slot data and slot card routes, follows new data for `DATASET` once the freshness
window passes, keeps a history of what it served, takes a replacement with its write token and
refuses another token, turns into a snapshot that pins what each slot serves and refuses a
later replacement, and comes down. A create retried with its own `shw_` token answers the first
share. Open pins then spend the caller's allowance until it answers 429. A second test runs the
same tree while the summed reading equals the plain one, and checks the derived frame still
follows new data.

The app is `tests.live_refresh.composed_app.compose()`: both surfaces over one Datasets
dependency cell, as `dfo_apis.app` mounts them, with the live stores' SQL running against
SQLite and the presentation store fake. Only the share routes' clock is replaced, so the
freshness window can pass. Every assertion is on a response, except the pins, which no route
lists and `ComposedApp.pins` reads from the reference table.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from assembler.write_token import SHARE_WRITE_TOKEN_PREFIX, mint_write_token
from presentation_api.doors.shares import SHARES_DEPS_SLOT
from presentation_api.newest_frames import FRESHNESS_WINDOW
from tests.live_refresh.composed_app import (
    DATASET,
    GRANTS,
    GROUPBY,
    INITIAL_YEARS,
    MEMBER_B,
    RECIPIENTS,
    STARTUP_ENV,
    ComposedApp,
    compose,
    initial_value,
)

if TYPE_CHECKING:
    # `starlette.testclient` answers with httpx2's responses when httpx2 is installed.
    from httpx2 import Response

_SHARES = "/api/presentation/shares"
_TOKEN = "x-dfo-write-token"
_CALLER = {"x-forwarded-for": "203.0.113.7"}
_PIN_ALLOWANCE = 2
_START = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
_KENYA = "country/KEN"
_MOVED_LOAN = 4321.0

Pair = dict[str, str]


def _pair(body: Mapping[str, Any]) -> Pair:
    return {"spec_id": str(body["spec_id"]), "content_version": str(body["content_version"])}


def _member_pair(member: Mapping[str, Any]) -> Pair:
    spec = member["spec"]
    return {"spec_id": str(spec["spec_id"]), "content_version": str(member["content_version"])}


class _Clock:
    def __init__(self) -> None:
        self.at = _START

    def __call__(self) -> datetime:
        return self.at


@dataclass
class _Journey:
    composed: ComposedApp
    client: TestClient
    clock: _Clock = field(default_factory=_Clock)

    def mint(self, spec_id: str) -> dict:
        resp = self.client.post("/api/datasets", json=self.composed.mint_body(spec_id=spec_id))
        assert resp.status_code == 200, resp.text
        return resp.json()

    def analyse(self, tree: Mapping[str, object]) -> dict:
        resp = self.client.post("/api/datasets/analysis", json={"tree": tree})
        assert resp.status_code == 200, resp.text
        return resp.json()

    def derived(self, frame: Pair, suffix: str = "", **params: str) -> Response:
        path = f"/api/datasets/derived/{frame['spec_id']}/{frame['content_version']}{suffix}"
        resp = self.client.get(path, params=params)
        assert resp.status_code == 200, resp.text
        return resp

    def card(self, frame: Pair) -> dict:
        minimal = self.client.get("/api/presentation/release").json()["minimal_intents"]
        intent = {
            **minimal["table"],
            "frame_ref": frame,
            "select": {"stat_var": f"dcid:{frame['spec_id']}"},
        }
        resp = self.client.post("/api/presentation/components", json={"intent": intent})
        assert resp.status_code == 201, resp.text
        return resp.json()

    def share(self, component_id: str, **fields: object) -> Response:
        body = {"cards": [{"component_id": component_id, "revision": 1}], "latest": True}
        return self.client.post(_SHARES, json={**body, **fields}, headers=_CALLER)

    def data(self, share_id: str, **headers: str) -> Response:
        return self.client.get(f"{_SHARES}/{share_id}/slots/1/data", headers=headers)

    def slot_card(self, share_id: str) -> Response:
        return self.client.get(f"{_SHARES}/{share_id}/slots/1")

    def replace(self, share_id: str, component_id: str, *, token: str) -> Response:
        return self.client.put(
            f"{_SHARES}/{share_id}",
            json={"cards": [{"component_id": component_id, "revision": 1}]},
            headers={_TOKEN: token},
        )

    def pin(self, frame: Pair, digest: str) -> Response:
        return self.client.post(
            f"/api/datasets/{frame['spec_id']}/{frame['content_version']}/pin",
            json={"schema_version": 1, "artifact_digest": digest},
            headers=_CALLER,
        )

    def pinned(self) -> set[tuple[str, str]]:
        """Each `(spec_id, content_version)` a presentation reference pins."""
        return {
            (spec_id, content_version)
            for principal, spec_id, content_version in self.composed.pins()
            if principal.startswith("presentation:")
        }

    def roots(self, frame: Pair) -> set[tuple[str, str]]:
        """The dataset versions the derived frame's manifest names as its roots."""
        manifest = self.derived(frame).json()
        return {(root["spec_id"], root["content_version"]) for root in manifest["roots"]}


@pytest.fixture
def journey(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Journey]:
    for name, value in STARTUP_ENV.items():
        monkeypatch.setenv(name, value)
    # An open caller is named by its address under this key, and pins within this allowance.
    monkeypatch.setenv("DFO_CALLER_HMAC_KEY", "dashboard-journey-caller-key")
    monkeypatch.setenv("DATASETS_PIN_DAILY_ALLOWANCE", str(_PIN_ALLOWANCE))
    composed = compose()
    with TestClient(composed.app, raise_server_exceptions=False) as client:
        walk = _Journey(composed=composed, client=client)
        getattr(composed.app.state, SHARES_DEPS_SLOT.name).now = walk.clock
        yield walk


def _stated(resp: Response) -> tuple[str, str, str]:
    """A slot data reply's freshness, the content version it serves, and the share's mode."""
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert resp.headers["x-dfo-freshness"] == body["freshness"]
    return body["freshness"], body["frame"]["content_version"], body["mode"]


def _drawn_on(card: Mapping[str, Any]) -> list[Pair]:
    return [_pair(frame) for frame in card["resolved_payload"]["provenance"]["inputs"]]


def _derive(journey: _Journey) -> tuple[dict, dict]:
    """Mint both readings and compare them: `DATASET` summed by recipient minus `GRANTS`.
    Answers `DATASET`'s mint reply and the ref's one member."""
    dataset, grants = journey.mint(DATASET), journey.mint(GRANTS)
    roots = {
        "data": {
            "spec_id": DATASET,
            "content_version": dataset["content_version"],
            "rendering": {"columns": [], "groupby": GROUPBY},
        },
        "other": {
            "spec_id": GRANTS,
            "content_version": grants["content_version"],
            "rendering": {"columns": []},
        },
    }
    tree = {"verb": "compare", "arguments": {"on": ["recipient"]}, "inputs": roots}
    answer = journey.analyse(tree)
    assert answer["status"] == "ready"
    assert answer["ref"]["format_version"] == 2
    (member,) = answer["ref"]["members"]
    return dataset, member


def _values(body: bytes) -> dict[tuple[str, str], float]:
    """A derived frame file's values by recipient and period."""
    return {
        (row["recipient"], row["time_period"]): row["value"]
        for row in pq.read_table(io.BytesIO(body)).to_pylist()
    }


def _loans(*, kenya_2024: float | None = None) -> dict[tuple[str, str], float]:
    """`MEMBER_B`'s values by recipient and period, which the comparison answers."""
    values = {
        (recipient, str(year)): initial_value(stat_var=MEMBER_B, recipient=recipient, year=year)
        for recipient in RECIPIENTS
        for year in INITIAL_YEARS
    }
    if kenya_2024 is not None:
        values[(_KENYA, "2024")] = kenya_2024
    return values


def test_a_dashboard_from_analysis_to_share_to_unshare(journey: _Journey):
    # Two seeded readings: `DATASET` holds both members, so its sum by recipient is a frame of
    # its own, and `GRANTS` holds `MEMBER_A`. Comparing them answers a format 2 ref to a
    # derived frame over both.
    journey.composed.move(add_stat_var=True)
    dataset, member = _derive(journey)
    frame = _member_pair(member)
    assert frame["spec_id"].startswith("ops_")
    assert frame["content_version"].startswith("ocv_")
    assert [root["spec_id"] for root in member["roots"]] == [GRANTS, DATASET]

    # The derived frame's file hashes to the ref's digest and holds the family sum minus the
    # grants: the loans.
    body = journey.derived(frame, "/file", format="parquet", bundle="false", layout="frame").content
    assert "sha256:" + hashlib.sha256(body).hexdigest() == member["artifact"]["digest"]
    assert _values(body) == _loans()

    # A card on the derived frame cites the sources its manifest cites, in that order.
    card = journey.card(frame)
    cited = journey.derived(frame).json()["citation"]["sources"]
    assert cited
    assert [source["name"] for source in card["resolved_payload"]["provenance"]["sources"]] == [
        source["label"] for source in cited
    ]
    component_id = card["component_id"]

    # A card alone pins nothing; a live share of it pins the frame and its roots.
    assert journey.pinned() == set()
    created = journey.share(component_id)
    assert created.status_code == 201, created.text
    share = created.json()
    share_id, token = share["id"], share["write_token"]
    site = STARTUP_ENV["DFO_PUBLIC_SITE_URL"]
    assert {key: share[key] for key in ("url", "embed_url", "data_urls", "latest")} == {
        "url": f"{site}/s/{share_id}",
        "embed_url": f"{site}/embed/{share_id}",
        "data_urls": [f"{site}/api/v1/presentation/shares/{share_id}/slots/1/data"],
        "latest": True,
    }
    assert token.startswith(SHARE_WRITE_TOKEN_PREFIX)
    first_roots = journey.roots(frame)
    assert journey.pinned() == {(frame["spec_id"], frame["content_version"]), *first_roots}

    # The slot's data is current, with an ETag a matching If-None-Match answers 304 on.
    served = journey.data(share_id, **{"accept-encoding": "identity"})
    assert _stated(served) == ("current", frame["content_version"], "live")
    assert served.json()["frame"] == frame
    etag = served.headers["etag"]
    again = journey.data(share_id, **{"if-none-match": etag})
    assert (again.status_code, again.content) == (304, b"")

    # The slot's card is drawn on the frame the data route serves.
    slot_card = journey.slot_card(share_id)
    assert slot_card.status_code == 200, slot_card.text
    assert slot_card.json()["component_id"] == component_id
    assert slot_card.json()["freshness"] == "current"
    assert _drawn_on(slot_card.json()) == [frame]

    # New data for `DATASET` reaches the slot once the freshness window has passed, as a new
    # derived frame over the new `DATASET` version and the same `GRANTS` version.
    journey.composed.move(set_value=(_KENYA, 2024, _MOVED_LOAN), stat_var=MEMBER_B)
    journey.clock.at += FRESHNESS_WINDOW - timedelta(seconds=1)
    assert _stated(journey.data(share_id))[1] == frame["content_version"]
    journey.clock.at += timedelta(seconds=1)
    moved = journey.data(share_id)
    freshness, moved_version, mode = _stated(moved)
    assert (freshness, mode) == ("current", "live")
    assert moved_version != frame["content_version"]
    moved_frame = moved.json()["frame"]
    # The derived id hashes the op-spec, which names its roots' versions, so it moves too.
    assert moved_frame["spec_id"].startswith("ops_")
    moved_body = journey.derived(
        moved_frame, "/file", format="parquet", bundle="false", layout="frame"
    ).content
    assert _values(moved_body) == _loans(kenya_2024=_MOVED_LOAN)
    moved_roots = journey.roots(moved_frame)
    assert {spec_id for spec_id, _ in moved_roots - first_roots} == {DATASET}
    assert _drawn_on(journey.slot_card(share_id).json()) == [moved_frame]

    # The history lists both frames, newest first, each with its retention.
    history = journey.client.get(f"{_SHARES}/{share_id}/history")
    assert history.status_code == 200, history.text
    entries = history.json()["served"]
    assert [entry["frame"]["content_version"] for entry in entries] == [
        moved_version,
        frame["content_version"],
    ]
    assert [entry["slot"] for entry in entries] == [1, 1]
    assert all(entry["retention"] is not None for entry in entries), entries
    assert entries[1]["retention"]["pinned"] is True

    # A replacement with the share's token writes revision 2; another token refuses.
    replaced = journey.replace(share_id, component_id, token=token)
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["revision"] == 2
    wrong = journey.replace(
        share_id, component_id, token=mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    )
    assert (wrong.status_code, wrong.json()["code"]) == (403, "write_token_invalid")

    # A snapshot keeps the frame the slot serves, pins it with its roots, and refuses a later
    # replacement.
    converted = journey.client.post(f"{_SHARES}/{share_id}/snapshot", headers={_TOKEN: token})
    assert converted.status_code == 200, converted.text
    assert (converted.json()["latest"], converted.json()["data_urls"]) == (
        False,
        share["data_urls"],
    )
    assert {(moved_frame["spec_id"], moved_version), *moved_roots} <= journey.pinned()
    journey.clock.at += 2 * FRESHNESS_WINDOW
    assert _stated(journey.data(share_id)) == ("current", moved_version, "snapshot")
    assert _drawn_on(journey.slot_card(share_id).json()) == [moved_frame]
    refused = journey.replace(share_id, component_id, token=token)
    assert (refused.status_code, refused.json()["code"]) == (409, "share_not_replaceable")

    # A create retried with the client's own token answers the share the first create made.
    client_token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
    first = journey.share(component_id, write_token=client_token)
    retried = journey.share(component_id, write_token=client_token)
    assert (first.status_code, retried.status_code) == (201, 201), retried.text
    assert retried.json()["id"] == first.json()["id"] != share_id
    assert retried.json()["write_token"] == client_token

    # Unshare answers 204, and the share's reads answer 404.
    deleted = journey.client.delete(f"{_SHARES}/{share_id}", headers={_TOKEN: token})
    assert deleted.status_code == 204, deleted.text
    for path in ("", "/slots/1", "/slots/1/data", "/history"):
        gone = journey.client.get(f"{_SHARES}/{share_id}{path}")
        assert (gone.status_code, gone.json()["code"]) == (404, "not_found"), path

    # An open caller's pins copy unpinned artifacts until its allowance answers 429: the
    # statistics both first versions share, and both files of `DATASET`'s moved version.
    current = journey.mint(DATASET)
    assert current["content_version"] != dataset["content_version"]
    (statistics,) = [a for a in dataset["artifacts"] if "statistics" in a["rendering"]]
    unpinned = [
        (dataset["content_version"], statistics["artifact_digest"]),
        *(
            (current["content_version"], artifact["artifact_digest"])
            for artifact in current["artifacts"]
        ),
    ]
    assert len({digest for _, digest in unpinned}) == _PIN_ALLOWANCE + 1, unpinned
    answers = [
        journey.pin({"spec_id": DATASET, "content_version": version}, digest)
        for version, digest in unpinned
    ]
    assert [resp.status_code for resp in answers[:_PIN_ALLOWANCE]] == [200] * _PIN_ALLOWANCE, [
        resp.text for resp in answers
    ]
    exhausted = answers[_PIN_ALLOWANCE]
    assert (exhausted.status_code, exhausted.json()["code"]) == (429, "allowance_exhausted")
    assert int(exhausted.headers["retry-after"]) > 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LiveRootSource.resolve refreshes a root under `latest` by its version alone, so a root "
        "version recording both its plain and summed renderings refuses 422 "
        "rendering_not_recorded once they differ, though the recipe names the summed rendering"
    ),
)
def test_a_derived_frame_over_a_sum_equal_to_its_plain_frame_follows_new_data(journey: _Journey):
    # While `MEMBER_B` holds no observations, `DATASET` summed by recipient is its plain frame,
    # so the root's version records both renderings.
    _, member = _derive(journey)
    journey.composed.move(add_stat_var=True)

    resp = journey.client.post(
        "/api/datasets/analysis", json={"tree": _member_pair(member), "latest": True}
    )

    assert resp.status_code == 200, resp.text
    (moved,) = resp.json()["ref"]["members"]
    assert _member_pair(moved) != _member_pair(member)
