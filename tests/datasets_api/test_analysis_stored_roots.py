"""`POST /api/datasets/analysis` with `latest: "stored"` over `LiveRootSource`: each root reads
the version its dataset's stored head names, from records alone, keyed as the pinned request
naming that version (the 0.6.0 client's, a cut version pinned from a head GET) is keyed.

The fixture is `test_analysis_cut_sums`'s: a flow dataset of two StatVars on one summable axis,
whose providers are France and the DAC total, minted bare and with the party-totals cut, so the
head names the raw version and the cut is a version of its own recorded from it."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

import datasets_api.analysis_roots as analysis_roots
import datasets_api.current as current
import datasets_api.doors.analysis as analysis_door
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.renewal import RENEW_MARKER
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.task_create_pool import TaskCreatePool
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from dfo_platform.timings import count
from spec_registry.identity import recompute_identity

from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
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
from .test_meta_doors import _operation
from .test_renew_on_read import _INSIDE_THE_MARGIN, _answer_frame, _renewals

_PATH = "/api/datasets/analysis"
_CUT = {"without_party_totals": True}
# A rank by provider sums the root's two StatVars; an aggregate that groups by `stat_var`
# reads the cut root unsummed.
_SUMMED = {"verb": "rank", "arguments": {"rank_by": ["provider"], "aggregation": "sum"}}
_UNSUMMED = {
    "verb": "aggregate",
    "arguments": {"group_by": ["time_period", "stat_var"], "aggregation": "sum"},
}


def _count_transactions(monkeypatch: pytest.MonkeyPatch, store: Any) -> None:
    """Add each transaction of `store` to the call's `db_transactions` count, as
    `LiveDatasetStore` does."""
    transaction = store.transaction

    @contextmanager
    def counted() -> Iterator[Any]:
        count("db_transactions")
        with transaction() as tx:
            yield tx

    monkeypatch.setattr(store, "transaction", counted)


class _Rig:
    """The fixture dataset minted bare and cut, and the client, stores and doubles over it.
    With `counted`, every transaction of the version and spec stores counts on the call's
    `db_transactions` (`_count_transactions`)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, counted: bool = False) -> None:
        self.store = FakeVersionRecordDatasetStore()
        definitions = FakeDatasetStore()
        if counted:
            _count_transactions(monkeypatch, self.store)
            _count_transactions(monkeypatch, definitions)
        self.heads = _FakeHeadStore()
        self.tasks = FakeTaskClient()
        # The pool the lifespan would build, held so a test waits for the head checks' task
        # creates a stored read hands off.
        self.task_creates = TaskCreatePool()
        self.operations = LiveOperationStore(store=self.store)
        self.client = _client(
            graph=_graph(),
            store=definitions,
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
            operation_store=self.operations,
            task_client=self.tasks,
            task_creates=self.task_creates,
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


def test_a_stored_miss_refuses_a_head_revoked_after_keying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head's version is taken down once the request has keyed and before its run reads
    the root: the run, on its analysis thread, reads the version as it stands then and refuses
    `snapshot_revoked`, though keying read it live."""
    rig = _Rig(monkeypatch)
    rig.age_rendered_files()
    key_job = analysis_door._key_job

    def keyed_then_revoked(**kwargs: Any) -> Any:
        keyed = key_job(**kwargs)
        assert keyed.prepared is not None
        assert keyed.answer is None
        _revoke_head(rig, monkeypatch)
        return keyed

    monkeypatch.setattr(analysis_door, "_key_job", keyed_then_revoked)
    response = rig.stored(_UNSUMMED)

    assert response.status_code == 410, response.text
    assert response.json()["code"] == "snapshot_revoked"
    assert rig.renders == 0


def test_a_stale_head_resolves_pending_and_submits_one_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the recheck window, inside the serve-stale ceiling: the dataset's two roots read
    the head as it stands from one read of it per resolution, the keying's and, for a root
    rendered at locate, the analysis thread's, and its check is submitted once."""
    rig = _Rig(monkeypatch)
    rig.heads.age_head(rig.spec_id, age=_PAST_INTERVAL)
    reads: list[str] = []
    head_version = analysis_roots.stored_head_version

    def reading(stored: Any, **kwargs: Any) -> Any:
        reads.append(threading.current_thread().name)
        return head_version(stored, **kwargs)

    monkeypatch.setattr(analysis_roots, "stored_head_version", reading)
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
    assert reads
    assert len(set(reads)) == len(reads)
    assert rig.task_creates.wait_idle(5)
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
    ceiling = DatasetsBuildSettings().head_serve_stale_max_age_seconds
    rig.heads.age_head(rig.spec_id, age=timedelta(seconds=ceiling, days=1))
    response = rig.stored(_UNSUMMED)

    assert response.status_code == 202, response.text
    assert len(response.json()["operations"]) == 1


@pytest.mark.parametrize("served", [True, False], ids=["served", "file-gone"])
def test_a_stored_read_of_a_recorded_cut_marks_its_read(
    monkeypatch: pytest.MonkeyPatch, served: bool
) -> None:
    """A stored request reading the cut marks the cut's read from the head, whether its file
    is served or gone and the hit renders nothing, so render ahead keeps rendering it."""
    rig = _Rig(monkeypatch)
    assert rig.pinned(_UNSUMMED).status_code == 200
    if not served:
        rig.age_rendered_files()
    with rig.store.transaction() as tx:
        tx.execute("UPDATE datasets.version_artifact_head SET read_at = NULL")
    renders = rig.renders

    assert rig.stored(_UNSUMMED).status_code == 200
    assert rig.renders == renders

    with rig.store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT read_at FROM datasets.version_artifact_head "
            "WHERE subject_id = %s AND content_version = %s AND head_content_version = %s",
            (rig.spec_id, rig.cut, rig.raw),
        )
    assert rows
    assert any(row["read_at"] is not None for row in rows)


def test_a_stored_root_whose_trace_counts_no_observations_reads_no_frame_while_keying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace minted before traces counted observations names its sum's StatVars only in the
    stored frame. Keying under `"stored"` reads no frame: the root resolves on the analysis
    thread, where the frame is read, under the pinned request's key and answer."""
    rig = _Rig(monkeypatch)
    monkeypatch.setattr(analysis_roots, "_counts_recorded", lambda trace: False)
    threads: list[str] = []
    read = analysis_roots.LiveRootSource._stored_stat_vars

    def reading(self: Any, *args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread().name)
        return read(self, *args, **kwargs)

    monkeypatch.setattr(analysis_roots.LiveRootSource, "_stored_stat_vars", reading)
    pinned = rig.pinned(_SUMMED)
    threads.clear()
    stored = rig.stored(_SUMMED)

    assert pinned.status_code == 200, pinned.text
    assert stored.status_code == 200, stored.text
    assert rig.keys[0] == rig.keys[-1]
    assert stored.json()["ref"] == pinned.json()["ref"]
    assert threads
    assert not [name for name in threads if name.startswith("datasets-analysis-key")], threads


def _revoke_head(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    with rig.store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, trace_id, revoked_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (rig.spec_id, rig.raw, "taken down", None, "2026-10-30"),
        )


def _fail_head_check(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """The head past its recheck window, and its check failed for good."""
    rig.heads.age_head(rig.spec_id, age=_PAST_INTERVAL)
    monkeypatch.setattr(
        current,
        "request_head_check",
        lambda **_: (
            "failed",
            _operation(id="op_check", state="failed", refusal_code="over_ceiling"),
        ),
    )


@pytest.mark.parametrize(
    ("setup", "status", "code"),
    [
        pytest.param(_revoke_head, 410, "snapshot_revoked", id="revoked"),
        pytest.param(_fail_head_check, 500, "internal_error", id="check-failed"),
    ],
)
def test_a_stored_root_refuses_a_head_the_head_get_refuses(
    monkeypatch: pytest.MonkeyPatch, setup: Any, status: int, code: str
) -> None:
    """A head whose version is taken down refuses `snapshot_revoked`, and one whose check failed
    for good refuses as a dataset root whose queued rebuild failed refuses under `latest:
    true` (`RefreshRebuildFailed`)."""
    rig = _Rig(monkeypatch)
    setup(rig, monkeypatch)
    response = rig.stored(_UNSUMMED)

    assert response.status_code == status, response.text
    assert response.json()["code"] == code


def _age_file(rig: _Rig, digest: str, *, by: timedelta) -> None:
    """Move the working location of `digest` `by` into the past."""
    with rig.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s "
            "WHERE artifact_digest = %s AND tier = 'working'",
            ((datetime.now(UTC) - by).isoformat(), digest),
        )


def _renewal_threads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The thread each stored-answer hit's renewal ran on (`renew_on_hit`)."""
    threads: list[str] = []
    renew_on_hit = analysis_door.renew_on_hit

    def recording(current_deps: Any) -> Any:
        renew = renew_on_hit(current_deps)

        def on_hit(*args: Any) -> None:
            threads.append(threading.current_thread().name)
            renew(*args)

        return on_hit

    monkeypatch.setattr(analysis_door, "renew_on_hit", recording)
    return threads


def test_a_stored_hit_on_the_key_executor_queues_one_renewal_of_its_answers_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer's frame file inside the renewal margin: the stored request's hit, answered
    on the key executor, queues one renewal of it under the answer's `ops_` pair, and answers
    the pinned request's body."""
    rig = _Rig(monkeypatch)
    pinned = rig.pinned(_UNSUMMED)
    assert pinned.status_code == 200, pinned.text
    spec_id, digest = _answer_frame(pinned.json())
    _age_file(rig, digest, by=_INSIDE_THE_MARGIN)
    threads = _renewal_threads(monkeypatch)
    queued = len(rig.tasks.enqueued)

    stored = rig.stored(_UNSUMMED)

    assert stored.status_code == 200, stored.text
    body = stored.json()
    body.pop("resolved_roots")
    assert body == pinned.json()
    assert len(threads) == 1
    assert threads[0].startswith("datasets-analysis-key")
    renewals = _renewals(rig.operations, rig.tasks)
    assert len(rig.tasks.enqueued) == queued + 1
    assert [(r["subject_id"], r[RENEW_MARKER]) for r in renewals] == [(spec_id, digest)]


def test_a_recorded_rendering_whose_file_is_gone_renews_its_served_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cut's Parquet file past retention and its stored frame inside the renewal margin:
    a stored request reads the cut from records, renders nothing, and queues one renewal of
    the stored frame a later render of the cut reads."""
    rig = _Rig(monkeypatch)
    assert rig.pinned(_UNSUMMED).status_code == 200
    reader = LiveSnapshotReader(store=rig.store)
    by_rendering = {
        tuple(sorted(location.rendering.items())): location.artifact_digest
        for location in reader.read_artifact_locations(rig.spec_id, rig.cut)
    }
    frame = by_rendering[(("format", "frame"),)]
    (rendered,) = [
        digest for rendering, digest in by_rendering.items() if ("layout", "frame") in rendering
    ]
    _age_file(rig, rendered, by=timedelta(days=3650))
    _age_file(rig, frame, by=_INSIDE_THE_MARGIN)
    renders = rig.renders

    stored = rig.stored(_UNSUMMED)

    assert stored.status_code == 200, stored.text
    assert rig.renders == renders
    renewals = [
        (r["subject_id"], r["content_version"], r[RENEW_MARKER])
        for r in _renewals(rig.operations, rig.tasks)
    ]
    assert renewals == [(rig.spec_id, rig.cut, frame)]


def _db_transactions(response: httpx.Response) -> int:
    """The call's `db_transactions` count its `Server-Timing` states."""
    (entry,) = [
        part.strip()
        for part in response.headers["server-timing"].split(",")
        if part.strip().startswith("db_transactions;")
    ]
    return int(entry.removeprefix("db_transactions;desc="))


@pytest.mark.parametrize(
    ("step", "transactions"),
    [
        # The definition, the head version's state, the answer, the answer's frame state, and
        # the answer's recipe. The roots' records, the frame's retention and the derived
        # manifest's frame state come from the states read.
        pytest.param(_UNSUMMED, 5, id="plain"),
        # Adds the recorded versions listed from the head and the recorded sum's state.
        pytest.param(_SUMMED, 7, id="groupby-implied"),
    ],
)
def test_a_stored_hit_reads_each_record_once(
    monkeypatch: pytest.MonkeyPatch, step: dict[str, Any], transactions: int
) -> None:
    """A stored hit with `include: ["manifest"]` over a root in its plain rendering reads each
    record once (`SharedVersionReads`). The head store is in memory, so its read counts
    nothing. No write is due: the requests before it marked the root's reads, its files are
    fresh, so no renewal is due, and its head was confirmed this second, so no head check is
    due; the hit writes nothing."""
    rig = _Rig(monkeypatch, counted=True)
    root = {"spec_id": rig.spec_id}
    request = {
        "tree": {**step, "inputs": {"data": root}},
        "latest": "stored",
        "include": ["manifest"],
    }
    for _ in range(2):
        assert rig.client.post(_PATH, json=request).status_code == 200
    queued, changes = len(rig.tasks.enqueued), rig.store.total_changes

    hit = rig.client.post(_PATH, json=request)

    assert hit.status_code == 200, hit.text
    body = hit.json()
    assert body["resolved_roots"][0]["freshness"] == "current"
    assert body["manifest"]["content_version"] == body["ref"]["members"][0]["content_version"]
    assert (len(rig.tasks.enqueued), rig.store.total_changes) == (queued, changes)
    assert _db_transactions(hit) == transactions
