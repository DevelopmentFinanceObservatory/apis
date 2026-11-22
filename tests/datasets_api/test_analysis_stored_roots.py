"""`POST /api/datasets/analysis` with `latest: "stored"` over `LiveRootSource`: each root reads
the version its dataset's stored head names, from records alone, keyed as the pinned request
naming that version (the 0.6.0 client's, a cut version pinned from a head GET) is keyed.

The fixture is `test_analysis_cut_sums`'s: a flow dataset of two StatVars on one summable axis,
whose providers are France and the DAC total, minted bare and with the party-totals cut, so the
head names the raw version and the cut is a version of its own recorded from it."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx
import pytest

import datasets_api.current as current
import datasets_api.doors.analysis as analysis_door
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from spec_registry.identity import recompute_identity

from ._version_record_store import FakeVersionRecordDatasetStore
from .test_analysis_cut_sums import (
    _BARE,
    _CSV,
    _IDENTITY,
    _code_index,
    _coverage_service,
    _graph,
    _mint,
)
from .test_definition_door import _client
from .test_definition_door_currency import _PAST_INTERVAL, _FakeHeadStore

_PATH = "/api/datasets/analysis"
_CUT = {"without_party_totals": True}
# A rank by provider sums the root's two StatVars; an aggregate that groups by `stat_var`
# reads the cut root unsummed.
_SUMMED = {"verb": "rank", "arguments": {"rank_by": ["provider"], "aggregation": "sum"}}
_UNSUMMED = {
    "verb": "aggregate",
    "arguments": {"group_by": ["time_period", "stat_var"], "aggregation": "sum"},
}


class _Rig:
    """The fixture dataset minted bare and cut, and the client, stores and doubles over it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.store = FakeVersionRecordDatasetStore()
        self.heads = _FakeHeadStore()
        self.tasks = FakeTaskClient()
        self.client = _client(
            graph=_graph(),
            sdmx_transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=_CSV,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
            version_store=self.store,
            code_index=_code_index(),
            coverage=_coverage_service(),
            head_writer=self.heads,
            recipe_store=LiveRecipeStore(store=self.store),
            version_lister=LiveVersionLister(store=self.store),
            operation_store=LiveOperationStore(store=self.store),
            task_client=self.tasks,
        )
        self.spec_id = recompute_identity(_IDENTITY).spec_id
        sidecar = {"schema_version": 1, "spec_id": self.spec_id, "identity": _IDENTITY}
        self.raw = _mint(self.client, spec_id=self.spec_id, sidecar=sidecar, options=_BARE)[
            "content_version"
        ]
        self.cut = _mint(
            self.client, spec_id=self.spec_id, sidecar=sidecar, options={**_BARE, **_CUT}
        )["content_version"]
        assert self.cut != self.raw
        self.keys: list[str] = []
        prepare = analysis_door.prepare_analysis

        def keyed(*args: Any, **kwargs: Any) -> Any:
            prepared = prepare(*args, **kwargs)
            self.keys.append(prepared.key)
            return prepared

        monkeypatch.setattr(analysis_door, "prepare_analysis", keyed)
        self.renders = 0
        render = current._render_stored_rendering

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.renders += 1
            return render(*args, **kwargs)

        monkeypatch.setattr(current, "_render_stored_rendering", counted)

    def pinned(self, step: dict[str, Any]) -> httpx.Response:
        """The 0.6.0 client's post: the cut version a head GET in the cut rendering answered,
        pinned, with `latest` false."""
        root = {"spec_id": self.spec_id, "content_version": self.cut, "rendering": _CUT}
        return self.client.post(_PATH, json={"tree": {**step, "inputs": {"data": root}}})

    def stored(self, step: dict[str, Any], **members: Any) -> httpx.Response:
        root = {"spec_id": self.spec_id, "rendering": _CUT}
        return self.client.post(
            _PATH,
            json={"tree": {**step, "inputs": {"data": root}}, "latest": "stored", **members},
        )

    def age_rendered_files(self) -> None:
        """Age the frame-layout files of every version rendered from the raw version, the cut
        and any sum, past working retention, as when they are gone; their records and links
        stand."""
        with self.store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT content_version FROM datasets.version_record "
                "WHERE subject_id = %s AND content_version != %s",
                (self.spec_id, self.raw),
            )
        reader = LiveSnapshotReader(store=self.store)
        files = [
            location.artifact_digest
            for row in rows
            for location in reader.read_artifact_locations(
                self.spec_id, str(row["content_version"])
            )
            if location.rendering.get("layout") == "frame"
        ]
        assert files
        with self.store.transaction() as tx:
            for digest in files:
                tx.execute(
                    "UPDATE datasets.artifact_location SET stored_at = %s "
                    "WHERE artifact_digest = %s",
                    ("2000-01-01T00:00:00+00:00", digest),
                )

    def version_count(self) -> int:
        with self.store.transaction() as tx:
            row = tx.fetch_one("SELECT COUNT(*) AS n FROM datasets.version_record", ())
        assert row is not None
        return int(str(row["n"]))


@pytest.mark.parametrize("step", [_SUMMED, _UNSUMMED], ids=["summed", "unsummed-cut"])
def test_a_stored_request_shares_the_pinned_cut_requests_key_and_answer(
    monkeypatch: pytest.MonkeyPatch, step: dict[str, Any]
) -> None:
    rig = _Rig(monkeypatch)
    pinned = rig.pinned(step)
    versions, renders = rig.version_count(), rig.renders
    stored = rig.stored(step)

    assert pinned.status_code == 200, pinned.text
    assert stored.status_code == 200, stored.text
    (pinned_key, stored_key) = rig.keys
    assert stored_key == pinned_key
    body = stored.json()
    assert body.pop("resolved_roots") == [
        {"spec_id": rig.spec_id, "content_version": rig.raw, "freshness": "current"}
    ]
    assert body == pinned.json()
    assert (rig.version_count(), rig.renders) == (versions, renders)


@pytest.mark.parametrize("step", [_SUMMED, _UNSUMMED], ids=["summed", "unsummed-cut"])
def test_a_stored_hit_whose_rendered_files_are_gone_renders_nothing(
    monkeypatch: pytest.MonkeyPatch, step: dict[str, Any]
) -> None:
    """The files of the cut and of the sum are past retention. The stored request resolves its
    root from the records, keys it, and answers the stored answer with no rendering made."""
    rig = _Rig(monkeypatch)
    pinned = rig.pinned(step)
    rig.age_rendered_files()
    renders = rig.renders
    stored = rig.stored(step, include=[])

    assert pinned.status_code == 200, pinned.text
    assert stored.status_code == 200, stored.text
    assert rig.renders == renders
    assert rig.keys[0] == rig.keys[1]
    assert stored.json()["ref"] == pinned.json()["ref"]


def test_a_stored_miss_whose_cut_file_is_gone_renders_it_on_the_analysis_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no stored answer the root renders when it is located, and the run answers the
    frame the pinned request computes."""
    rig = _Rig(monkeypatch)
    rig.age_rendered_files()
    stored = rig.stored(_UNSUMMED)
    pinned = rig.pinned(_UNSUMMED)

    assert stored.status_code == 200, stored.text
    assert rig.renders >= 1
    assert stored.json()["ref"] == pinned.json()["ref"]
    assert rig.keys[0] == rig.keys[1]


def test_a_stale_head_resolves_pending_and_submits_one_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the recheck window, inside the serve-stale ceiling: the root reads the head as it
    stands, and the head's check is submitted once for the dataset's two roots."""
    rig = _Rig(monkeypatch)
    rig.heads.age_head(rig.spec_id, age=_PAST_INTERVAL)
    root = {"spec_id": rig.spec_id, "rendering": _CUT}
    tree = {
        "verb": "sum_of",
        "arguments": {"on": ["provider", "recipient"]},
        "inputs": {"data": root, "other": {**root, "rendering": {**_CUT, "columns": []}}},
    }
    response = rig.client.post(_PATH, json={"tree": tree, "latest": "stored"})

    assert response.status_code == 200, response.text
    assert response.json()["resolved_roots"] == [
        {"spec_id": rig.spec_id, "content_version": rig.raw, "freshness": "pending"}
    ]
    assert len(rig.tasks.enqueued) == 1


def test_a_dataset_with_no_head_answers_202_naming_its_queued_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _Rig(monkeypatch)
    rig.heads._heads.clear()
    response = rig.stored(_UNSUMMED)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert len(body["operations"]) == 1
    assert len(rig.tasks.enqueued) == 1
    assert response.headers["retry-after"]


def test_a_head_past_the_serve_stale_ceiling_answers_202(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _Rig(monkeypatch)
    rig.heads.age_head(rig.spec_id, age=timedelta(days=8))
    response = rig.stored(_UNSUMMED)

    assert response.status_code == 202, response.text
    assert len(response.json()["operations"]) == 1


def test_a_stored_read_of_a_recorded_cut_marks_its_read(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _Rig(monkeypatch)
    assert rig.pinned(_UNSUMMED).status_code == 200
    with rig.store.transaction() as tx:
        tx.execute("UPDATE datasets.version_artifact_head SET read_at = NULL")

    assert rig.stored(_UNSUMMED).status_code == 200

    with rig.store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT read_at FROM datasets.version_artifact_head "
            "WHERE subject_id = %s AND content_version = %s AND head_content_version = %s",
            (rig.spec_id, rig.cut, rig.raw),
        )
    assert rows
    assert any(row["read_at"] is not None for row in rows)
