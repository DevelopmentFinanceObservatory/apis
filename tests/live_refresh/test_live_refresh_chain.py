"""A live refresh end to end on the composed Datasets and Presentation app: a card is created
on a minted or published frame, the data moves, and the frame refresh and the render routes
redraw the card on the new data. Every assertion is on a response of the service.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.live_refresh.composed_app import (
    DATASET,
    FRAME_OPTIONS,
    GROUPBY,
    LIVE,
    MEMBER_A,
    MEMBERS,
    PUBLISH,
    RECIPIENTS,
    STARTUP_ENV,
    ComposedApp,
    compose,
    derived_publish_body,
    initial_value,
)
from tests.live_refresh.wire_pairs import WIRE_PAIRS_PATH, record_wire_pairs

_GHANA, _KENYA, _TANZANIA, _UGANDA = RECIPIENTS
_NAMES = {_GHANA: "Ghana", _KENYA: "Kenya", _TANZANIA: "Tanzania", _UGANDA: "Uganda"}
_REFRESH = "/api/datasets/frames/refresh"
_COMPONENTS = "/api/presentation/components"
_LATEST = {"anchor": "latest", "offset": 0}

Frame = dict[str, str]


def _back(offset: int) -> dict[str, object]:
    return {"anchor": "latest", "offset": offset}


@dataclass
class _Rig:
    composed: ComposedApp
    client: TestClient

    def mint(self, options: dict[str, object] = FRAME_OPTIONS) -> Frame:
        resp = self.client.post("/api/datasets", json=self.composed.mint_body(options))
        assert resp.status_code == 200, resp.text
        return {"spec_id": DATASET, "content_version": resp.json()["content_version"]}

    def publish(self, body: dict[str, object]) -> Frame:
        resp = self.client.post("/api/datasets/frames/publish", json=body, headers=PUBLISH)
        assert resp.status_code == 200, resp.text
        published = resp.json()["published"]
        return {"spec_id": published["spec_id"], "content_version": published["content_version"]}

    def create(self, intent: dict[str, object]) -> str:
        # As `ai` writes cards: under its publish principal, which spends no card allowance.
        resp = self.client.post(_COMPONENTS, json={"intent": intent}, headers=PUBLISH)
        assert resp.status_code == 201, resp.text
        return resp.json()["component_id"]

    def read(self, component_id: str, **params: int) -> httpx.Response:
        return self.client.get(f"{_COMPONENTS}/{component_id}", params=params)

    def refresh(self, frame: Frame) -> httpx.Response:
        return self.client.post(_REFRESH, json=frame, headers=LIVE)

    def moved_to(self, frame: Frame) -> Frame:
        """The frame a refresh of `frame` moves to."""
        resp = self.refresh(frame)
        assert resp.status_code == 200, resp.text
        answer = resp.json()
        assert answer["status"] == "moved", answer
        assert answer["frame"]["spec_id"] == frame["spec_id"]
        assert answer["frame"]["content_version"] != frame["content_version"]
        return answer["frame"]

    def render(self, component_id: str, frame: Frame) -> httpx.Response:
        return self.client.post(
            f"{_COMPONENTS}/{component_id}/revisions/1/render", json={"frame": frame}, headers=LIVE
        )

    def redrawn(self, component_id: str, frame: Frame) -> dict:
        """The envelope the render route answers for revision 1 on `frame`."""
        resp = self.render(component_id, frame)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["component_id"], body["revision"]) == (component_id, 1)
        payload = body["resolved_payload"]
        assert payload["provenance"]["inputs"] == [frame]
        assert payload["provenance"]["refreshed_from"] == 1
        return payload

    def export(self, component_id: str, frame: Frame) -> httpx.Response:
        return self.client.get(
            f"{_COMPONENTS}/{component_id}/revisions/1/render/data",
            params={**frame, "format": "csv", "bundle": "false"},
            headers=LIVE,
        )


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Rig]:
    for name, value in STARTUP_ENV.items():
        monkeypatch.setenv(name, value)
    composed = compose()
    # The client is entered so the surfaces' lifespans run, as they do in a served app.
    with TestClient(composed.app, raise_server_exceptions=False) as client:
        yield _Rig(composed=composed, client=client)


def _intent(
    member: str,
    frame: Frame,
    *,
    recipients: Sequence[str] | str,
    stat_var: str | None = MEMBER_A,
    **fields: object,
) -> dict[str, object]:
    select: dict[str, object] = {"recipient": recipients}
    if stat_var is not None:
        select["stat_var"] = f"dcid:{stat_var}"
    return {"type": member, "frame_ref": frame, "select": select, **fields}


def _points(payload: dict) -> dict[str, dict[str, float | None]]:
    """Each drawn series' values by period, under its label. A missing point reads `None`."""
    return {
        series["label"]: {point["period"]: point.get("value") for point in series["points"]}
        for series in payload["component"]["series"]
    }


def _exported_points(resp: httpx.Response) -> dict[str, dict[str, float]]:
    """The rows of a bare CSV export, shaped as `_points` shapes a payload."""
    assert resp.status_code == 200, resp.text
    exported: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(io.StringIO(resp.text)):
        exported.setdefault(row["recipient_label"], {})[row["time_period"]] = float(row["value"])
    return exported


def _expected(
    recipients: Sequence[str], years: Sequence[int], *, members: Sequence[str] = (MEMBER_A,)
) -> dict[str, dict[str, float]]:
    """What the dataset's untouched observations draw: per recipient and year, the sum over
    `members`."""
    return {
        _NAMES[recipient]: {
            str(year): sum(
                initial_value(stat_var=member, recipient=recipient, year=year) for member in members
            )
            for year in years
        }
        for recipient in recipients
    }


def _coverage_gaps(payload: dict) -> list[str]:
    return [
        warning["detail"]
        for warning in payload["provenance"]["warnings"]
        if warning["kind"] == "coverage_gap"
    ]


def _refusal(resp: httpx.Response) -> tuple[int, str]:
    return resp.status_code, resp.json()["code"]


def test_a_dataset_card_is_redrawn_when_its_data_moves_and_only_then(rig: _Rig):
    stored = rig.mint()
    card = rig.create(_intent("line_chart", stored, recipients=[_KENYA, _GHANA]))

    # The control: unmoved data answers `current`, so nothing is redrawn.
    unmoved = rig.refresh(stored)
    assert unmoved.status_code == 200, unmoved.text
    assert unmoved.json() == {"status": "current", "frame": stored}

    rig.composed.move(add_year=2025)
    target = rig.moved_to(stored)
    assert rig.refresh(target).json() == {"status": "current", "frame": target}
    pinned_before = rig.composed.pins()

    payload = rig.redrawn(card, target)

    assert _points(payload) == _expected([_KENYA, _GHANA], [2022, 2023, 2024, 2025])
    # The render appended no revision, and the stored one still shows the data it was made on.
    head = rig.read(card).json()
    assert head["revision"] == 1
    assert head["stored_intent"]["frame_ref"] == stored
    assert _points(head["resolved_payload"]) == _expected([_KENYA, _GHANA], [2022, 2023, 2024])
    assert _refusal(rig.read(card, revision=2)) == (404, "not_found")
    # The render left the target frame in the working tier.
    assert rig.composed.pins() == pinned_before
    # The export holds the rows the payload draws.
    assert _exported_points(rig.export(card, target)) == _points(payload)

    # A corrected value moves the frame again.
    rig.composed.move(set_value=(_KENYA, 2024, 999.0))
    corrected = rig.moved_to(target)
    assert _points(rig.redrawn(card, corrected))["Kenya"]["2024"] == 999.0


def test_the_render_routes_refuse_a_foreign_frame_and_a_caller_without_the_principal(rig: _Rig):
    stored = rig.mint()
    card = rig.create(_intent("line_chart", stored, recipients=[_KENYA]))
    derived = rig.publish(
        derived_publish_body(root=stored, rows=rig.composed.observations.growth_rows([_KENYA]))
    )
    pins = rig.composed.pins()

    assert _refusal(rig.render(card, derived)) == (409, "render_target_mismatch")
    assert _refusal(rig.export(card, derived)) == (409, "render_target_mismatch")
    assert rig.composed.pins() == pins

    render_url = f"{_COMPONENTS}/{card}/revisions/1/render"
    for headers in ({}, PUBLISH):
        refused = [
            rig.client.post(_REFRESH, json=stored, headers=headers),
            rig.client.post(render_url, json={"frame": stored}, headers=headers),
            rig.client.get(f"{render_url}/data", params=stored, headers=headers),
        ]
        assert [_refusal(resp) for resp in refused] == [(403, "refresh_not_permitted")] * 3


def test_a_card_on_a_groupby_sum_is_redrawn_on_the_sum_of_the_new_data(rig: _Rig):
    rig.composed.observations.add_stat_var()
    stored = rig.mint({**FRAME_OPTIONS, "groupby": GROUPBY})
    plain_head = rig.composed.head()
    assert stored["content_version"] != plain_head
    card = rig.create(_intent("line_chart", stored, recipients=[_KENYA, _GHANA], stat_var=None))
    assert rig.refresh(stored).json() == {"status": "current", "frame": stored}

    rig.composed.move(add_year=2025)
    target = rig.moved_to(stored)

    # The answer is the sum of the new head, a version of its own.
    new_head = rig.composed.head()
    assert new_head not in (plain_head, target["content_version"])
    versions = rig.client.get(f"/api/datasets/{DATASET}/versions", headers=PUBLISH).json()
    listed = {version["content_version"]: version for version in versions["versions"]}
    assert listed[target["content_version"]] == {
        "content_version": target["content_version"],
        "base_content_version": new_head,
        "rendering": "groupby",
        "groupby": GROUPBY,
        # The sum keeps the recipient role, so its kind file carries the serving stamp.
        "enrichment_stamp": versions["enrichment_stamp"],
    }
    payload = rig.redrawn(card, target)
    assert _points(payload) == _expected(
        [_KENYA, _GHANA], [2022, 2023, 2024, 2025], members=MEMBERS
    )
    assert _exported_points(rig.export(card, target)) == _points(payload)
    # The plain frame of the same data is another rendering.
    unsummed = {"spec_id": DATASET, "content_version": new_head}
    assert _refusal(rig.render(card, unsummed)) == (409, "render_target_mismatch")


_SUM_OPTIONS = {**FRAME_OPTIONS, "groupby": GROUPBY}
# A second groupby. It keeps `stat_var`, so its sum is the plain frame whatever contributes.
_OTHER_SUM_OPTIONS = {**FRAME_OPTIONS, "groupby": {"by": ["recipient", "stat_var"], "op": "sum"}}
# A third groupby. It also keeps `provenance`, which every row shares, so its sum has the rows
# of `GROUPBY`'s and the two share one collapsed version.
_SAME_ROWS_SUM_OPTIONS = {
    **FRAME_OPTIONS,
    "groupby": {"by": ["recipient", "provenance"], "op": "sum"},
}
# How each stored frame is minted: whether a second StatVar contributes when it is, and the
# options of each mint. With one contributor the sum is the plain frame, so a sum mint links
# its file under the raw version.
_STORED_FRAMES: dict[str, tuple[bool, tuple[dict[str, object], ...]]] = {
    "plain": (False, (FRAME_OPTIONS,)),
    "collapse": (True, (_SUM_OPTIONS,)),
    "collapse-linking-two-sums": (True, (_SUM_OPTIONS, _SAME_ROWS_SUM_OPTIONS)),
    "raw-linking-one-sum-and-no-plain-file": (False, (_SUM_OPTIONS,)),
    "raw-linking-one-sum-and-a-plain-file": (False, (FRAME_OPTIONS, _SUM_OPTIONS)),
    "raw-linking-two-sums-and-no-plain-file": (False, (_SUM_OPTIONS, _OTHER_SUM_OPTIONS)),
    "raw-linking-two-sums-and-a-plain-file": (
        False,
        (FRAME_OPTIONS, _SUM_OPTIONS, _OTHER_SUM_OPTIONS),
    ),
}
# What the refresh of each stored frame answers once the data moved, by how many StatVars
# contribute to the new head: the raw `head`, the `sum` of it with a version of its own, or a
# refusal. A stored frame in `_NEVER_REFRESHED` refuses before the data moves too.
_REFRESH_ANSWERS = {
    ("plain", 1): "head",
    ("plain", 2): "head",
    ("collapse", 1): "head",
    ("collapse", 2): "sum",
    ("collapse-linking-two-sums", 1): "refused",
    ("collapse-linking-two-sums", 2): "refused",
    ("raw-linking-one-sum-and-no-plain-file", 1): "head",
    ("raw-linking-one-sum-and-no-plain-file", 2): "sum",
    ("raw-linking-one-sum-and-a-plain-file", 1): "head",
    ("raw-linking-one-sum-and-a-plain-file", 2): "refused",
    ("raw-linking-two-sums-and-no-plain-file", 1): "refused",
    ("raw-linking-two-sums-and-no-plain-file", 2): "refused",
    ("raw-linking-two-sums-and-a-plain-file", 1): "head",
    ("raw-linking-two-sums-and-a-plain-file", 2): "refused",
}
_NEVER_REFRESHED = {"collapse-linking-two-sums", "raw-linking-two-sums-and-no-plain-file"}


@pytest.mark.parametrize("advanced_by", ["refresh", "plain-mint", "sum-mint"])
@pytest.mark.parametrize(("stored_as", "contributors"), list(_REFRESH_ANSWERS))
def test_the_render_accepts_the_frame_the_refresh_answers_and_no_other_frame_of_the_head(
    rig: _Rig, stored_as: str, contributors: int, advanced_by: str
):
    """The data moves, and the head is advanced by the refresh itself or by another caller's
    mint before it. The render then accepts the stored frame and the refresh's answer, and
    refuses the head's other rendering."""
    second_contributor, mints = _STORED_FRAMES[stored_as]
    observations = rig.composed.observations
    if second_contributor:
        observations.add_stat_var()
    stored = rig.mint(mints[0])
    assert all(rig.mint(options) == stored for options in mints[1:])
    card = rig.create(
        _intent(
            "line_chart",
            stored,
            recipients=[_KENYA],
            stat_var=MEMBER_A if stored_as == "plain" else None,
        )
    )
    expected = _REFRESH_ANSWERS[(stored_as, contributors)]
    if stored_as in _NEVER_REFRESHED:
        assert _refusal(rig.refresh(stored)) == (422, "rendering_not_recorded")
    else:
        assert rig.refresh(stored).json() == {"status": "current", "frame": stored}

    if contributors == 2 and not second_contributor:
        observations.add_stat_var()
    if contributors == 1 and second_contributor:
        observations.drop_stat_var()
    rig.composed.move(add_year=2025)
    if advanced_by != "refresh":
        rig.mint(FRAME_OPTIONS if advanced_by == "plain-mint" else _SUM_OPTIONS)
    refreshed = rig.refresh(stored)
    head, summed = rig.mint(), rig.mint(_SUM_OPTIONS)

    assert (summed != head) == (contributors == 2)
    if expected == "refused":
        assert _refusal(refreshed) == (422, "rendering_not_recorded")
        answered = [stored]
    else:
        assert refreshed.status_code == 200, refreshed.text
        target = head if expected == "head" else summed
        assert refreshed.json() == {"status": "moved", "frame": target}
        answered = [stored, target]
    for frame in (stored, head, summed):
        response = rig.render(card, frame)
        if frame in answered:
            assert response.status_code == 200, (frame, response.text)
        else:
            assert _refusal(response) == (409, "render_target_mismatch"), frame


def test_a_frame_minted_plain_and_summed_refuses_once_the_two_renderings_differ(rig: _Rig):
    # With one contributing StatVar the sum is the plain frame, so both mints share a version.
    stored = rig.mint()
    assert rig.mint({**FRAME_OPTIONS, "groupby": GROUPBY}) == stored
    card = rig.create(_intent("line_chart", stored, recipients=[_KENYA], stat_var=None))
    assert rig.refresh(stored).json() == {"status": "current", "frame": stored}

    # New data with one contributor is still one frame under both renderings.
    rig.composed.move(add_year=2025)
    target = rig.moved_to(stored)
    assert _points(rig.redrawn(card, target)) == _expected([_KENYA], [2022, 2023, 2024, 2025])

    # A second contributor makes the sum a frame of its own, and the records do not say which
    # of the two the card was made on.
    rig.composed.move(add_stat_var=True)
    for frame in (stored, target):
        refused = rig.refresh(frame)
        assert _refusal(refused) == (422, "rendering_not_recorded")
        assert refused.json()["retryable"] is False


def test_a_rebuild_over_the_inline_budget_is_pending_until_the_worker_runs_it(rig: _Rig):
    stored = rig.mint()
    card = rig.create(_intent("line_chart", stored, recipients=[_KENYA]))
    rig.composed.set_inline_budget(inline=False)
    rig.composed.move(add_year=2025)

    queued = rig.refresh(stored)

    assert queued.status_code == 200, queued.text
    pending = queued.json()
    assert pending == {
        "status": "pending",
        "operation_id": pending["operation_id"],
        "retry_after_s": 1.5,
    }
    # A second call names the same operation and queues nothing.
    assert rig.refresh(stored).json() == pending
    assert rig.composed.task_client.enqueued == [(pending["operation_id"], 0)]

    assert rig.composed.run_worker() == [pending["operation_id"]]

    target = rig.moved_to(stored)
    assert _points(rig.redrawn(card, target)) == _expected([_KENYA], [2022, 2023, 2024, 2025])
    assert rig.composed.task_client.enqueued == [(pending["operation_id"], 0)]


def test_a_derived_card_is_redrawn_on_the_replay_of_its_recipe(rig: _Rig):
    """The stored frame's recipe anchors `latest` at 2024 and leaves Uganda out. The replay on
    newer data anchors at 2025 although its own rows run to 2026, and leaves Uganda out too.
    The card selects all four recipients, so it names Uganda."""
    drawn = [_GHANA, _KENYA, _TANZANIA]
    left_out = [{"root": 0, "coordinates": {"recipient": _UGANDA}}]
    root = rig.mint()
    stored_body = derived_publish_body(
        root=root,
        rows=rig.composed.observations.growth_rows(drawn),
        anchor={"period": "2024", "rule": "half1", "left_out": left_out},
    )
    stored = rig.publish(stored_body)
    card = rig.create(
        _intent(
            "line_chart",
            stored,
            recipients=list(_NAMES),
            window={"start": _back(-1), "end": _LATEST},
        )
    )
    created = rig.read(card).json()["resolved_payload"]
    assert created["component"]["x_axis"]["window"] == {"start": "2023", "end": "2024"}
    assert _coverage_gaps(created) == ["1 series has no 2024 value: Uganda"]
    assert _refusal(rig.refresh(stored)) == (422, "frame_not_refreshable")

    rig.composed.move(add_year=2025, recipients=drawn)
    rig.composed.move(add_year=2026, recipients=drawn)
    new_root = rig.moved_to(root)
    replay_body = derived_publish_body(
        root=new_root,
        rows=rig.composed.observations.growth_rows(drawn),
        anchor={"period": "2025", "rule": "half1", "left_out": left_out},
    )
    stored_recipe, replay_recipe = stored_body["recipe"], replay_body["recipe"]
    assert replay_recipe["normal_form_hash"] == stored_recipe["normal_form_hash"]
    assert replay_recipe["replay_key"] != stored_recipe["replay_key"]
    replays = f"/api/datasets/recipes/replays/{replay_recipe['replay_key']}"
    assert _refusal(rig.client.get(replays, headers=PUBLISH)) == (404, "recipe_not_found")

    replay = rig.publish(replay_body)

    found = rig.client.get(replays, headers=PUBLISH)
    assert found.status_code == 200, found.text
    assert found.json() == replay
    recipe = rig.client.get(
        f"/api/datasets/frames/{replay['spec_id']}/{replay['content_version']}/recipe",
        headers=PUBLISH,
    )
    assert recipe.json()["anchor"]["period"] == "2025"
    payload = rig.redrawn(card, found.json())
    # `latest` is the recipe's anchor, not the newest period the replayed rows hold.
    assert payload["component"]["x_axis"]["window"] == {"start": "2024", "end": "2025"}
    growth = {(row["recipient"], row["time_period"]): row["value"] for row in replay_body["rows"]}
    assert _points(payload) == {
        _NAMES[recipient]: {year: growth[(recipient, year)] for year in ("2024", "2025")}
        for recipient in drawn
    }
    assert _coverage_gaps(payload) == ["1 series has no 2025 value: Uganda"]
    assert payload["provenance"]["warnings"][0]["kind"] == "coverage_gap"
    assert (f"presentation:{card}", replay["spec_id"], replay["content_version"]) not in (
        rig.composed.pins()
    )
    assert _exported_points(rig.export(card, replay)) == _points(payload)

    # A frame of another recipe, and a dataset frame, cannot stand in for the stored one.
    other = rig.publish(
        derived_publish_body(
            root=new_root, rows=rig.composed.observations.growth_rows(drawn), verb="rank"
        )
    )
    assert _refusal(rig.render(card, other)) == (409, "render_target_mismatch")
    assert _refusal(rig.render(card, new_root)) == (409, "render_target_mismatch")


def test_a_rolling_window_on_a_dataset_frame_moves_when_half_the_series_hold_the_period(
    rig: _Rig,
):
    stored = rig.mint()
    chart = rig.create(
        _intent(
            "line_chart",
            stored,
            recipients=list(RECIPIENTS),
            window={"start": _back(-1), "end": _LATEST},
        )
    )
    # The headline is the value at the end of the basis, a year behind the newest point.
    stat = rig.create(
        _intent("stat_card", stored, recipients=_KENYA, change={"basis": [_back(-2), _back(-1)]})
    )
    made = rig.read(stat).json()["resolved_payload"]["component"]
    assert (made["period"], made["value"]) == ("2023", 210.0)
    assert made["change"]["basis"] == ["2022", "2023"]

    # One series of four holds 2025: under half, so the anchor stays at 2024.
    rig.composed.move(add_year=2025, recipients=[_GHANA])
    under_half = rig.moved_to(stored)
    payload = rig.redrawn(chart, under_half)
    assert payload["component"]["x_axis"]["window"] == {"start": "2023", "end": "2024"}
    assert _points(payload) == _expected(RECIPIENTS, [2023, 2024])
    assert _coverage_gaps(payload) == []
    held = rig.redrawn(stat, under_half)["component"]
    assert (held["period"], held["value"]) == ("2023", 210.0)

    # Two of four hold it: the anchor moves, and the card names the series left out.
    rig.composed.move(add_year=2025, recipients=[_KENYA])
    half = rig.moved_to(under_half)
    payload = rig.redrawn(chart, half)
    assert payload["component"]["x_axis"]["window"] == {"start": "2024", "end": "2025"}
    assert _points(payload) == {
        "Ghana": {"2024": 120.0, "2025": 130.0},
        "Kenya": {"2024": 220.0, "2025": 230.0},
        "Tanzania": {"2024": 320.0, "2025": None},
        "Uganda": {"2024": 420.0, "2025": None},
    }
    assert payload["provenance"]["warnings"][0] == {
        "kind": "coverage_gap",
        "detail": "2 series have no 2025 value: Tanzania, Uganda",
    }
    moved = rig.redrawn(stat, half)["component"]
    assert (moved["period"], moved["value"]) == ("2024", 220.0)
    assert moved["change"]["basis"] == ["2023", "2024"]


def test_the_recorded_wire_pairs_are_what_the_composed_app_answers(rig: _Rig):
    """`wire_pairs.json` is the contract other services' mocks are checked against. Record it
    again with `python -m tests.live_refresh.wire_pairs` when a route's wire changes."""
    assert record_wire_pairs(composed=rig.composed, client=rig.client) == json.loads(
        WIRE_PAIRS_PATH.read_text()
    )
