"""Render ahead (`datasets_api.render_ahead`): a new head of a dataset renders the groupby sums and
party-totals cuts analysis roots read from an earlier head, latest read first and capped, so
the first reader of the new head renders nothing; renderings nobody read in the window are not
rendered, and a second run renders nothing."""

from __future__ import annotations

import contextlib
import dataclasses
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pytest

from datasets_api import current, maintenance
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import DefinitionDeps
from datasets_api.doors import analysis as analysis_door
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations import worker as worker_module
from datasets_api.operations.render_ahead_queue import render_ahead_identity
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker import OperationTransientFailure, run_operation
from datasets_api.render_ahead import heads_read_since, render_ahead
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveVersionLister

from ._version_record_store import FakeVersionRecordDatasetStore
from .test_analysis_summed_roots import (
    _BARE,
    _CSV_BODY,
    _aggregate,
    _count_work,
    _post,
    _rig,
    _swapping_transport,
    _two_axis_spec_id_and_sidecar,
)
from .test_serve_groupby_collapse import _mint, _spec_id_and_sidecar

# Another 2020 value of the first member: a new dataset version with another sum.
_MOVED_BODY = _CSV_BODY.replace(",2020,1000,", ",2020,600,")


def _captured_deps(monkeypatch: pytest.MonkeyPatch) -> list[DefinitionDeps]:
    """The `DefinitionDeps` each analysis request runs with, from the call on."""
    captured: list[DefinitionDeps] = []
    real = analysis_door.definition_deps

    def spy(deps: Any, **kwargs: Any) -> DefinitionDeps:
        view = real(deps, **kwargs)
        captured.append(view)
        return view

    monkeypatch.setattr(analysis_door, "definition_deps", spy)
    return captured


def _read_spec_root(
    *, two_axes: bool = False, operation_store=None, task_client=None, settings=None
):
    """A Spec root's client whose analysis read its sums from the first head, by recipient and,
    with `two_axes`, by recipient and `testAxis` after it. Returns the client, its store, the
    trees, the deps the requests ran with, and a function that stores a new head and returns
    its version."""
    body = {"text": _CSV_BODY}
    client, store, root = _rig(
        sdmx_transport=_swapping_transport(body),
        two_axes=two_axes,
        operation_store=operation_store,
        task_client=task_client,
        settings=settings,
    )
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    trees = [_aggregate(spec_root, group_by=["recipient"])]
    if two_axes:
        trees.append(_aggregate(spec_root, group_by=["recipient", "testAxis"]))
    with pytest.MonkeyPatch.context() as mp:
        captured = _captured_deps(mp)
        for tree in trees:
            assert _post(client, tree).status_code == 200
    deps = dataclasses.replace(captured[-1], deadline=None)

    def store_new_head(monkeypatch: pytest.MonkeyPatch, text: str = _MOVED_BODY) -> str:
        body["text"] = text
        with monkeypatch.context() as stale:
            stale.setattr(current, "_head_is_fresh", lambda head, *, now, deps: False)
            stale.setattr(current, "_max_age", lambda head, *, deps: timedelta(0))
            _, sidecar = _two_axis_spec_id_and_sidecar() if two_axes else _spec_id_and_sidecar()
            minted = _mint(client, spec_id=root["spec_id"], sidecar=sidecar, options=_BARE)
        assert minted["content_version"] != root["content_version"]
        return minted["content_version"]

    return client, store, root["spec_id"], trees, deps, store_new_head


def _read_at(store: FakeVersionRecordDatasetStore, *, head: str | None = None) -> list[Any]:
    """The `read_at` of each head row of a groupby sum's link, of those made from `head` when
    given."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT h.read_at, h.head_content_version, a.rendering "
            "FROM datasets.version_artifact_head h "
            "JOIN datasets.version_artifact a USING (subject_id, content_version, rendering_key)"
        )
    return [
        row["read_at"]
        for row in rows
        if "groupby" in row["rendering"] and head in (None, row["head_content_version"])
    ]


def _set_read_at(store: FakeVersionRecordDatasetStore, read_at: datetime | None) -> None:
    with store.transaction() as tx:
        tx.execute("UPDATE datasets.version_artifact_head SET read_at = %s", (read_at,))


def _ahead(deps: DefinitionDeps, spec_id: str, head: str, *, limit: int = 16, since=None):
    return render_ahead(
        spec_id,
        head_content_version=head,
        deps=deps,
        read_since=since if since is not None else deps.now() - timedelta(days=1),
        limit=limit,
    )


def test_a_new_head_renders_the_sum_a_root_read_and_its_first_reader_renders_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, store, spec_id, (tree,), deps, store_new_head = _read_spec_root()
    head = store_new_head(monkeypatch)

    ahead = _ahead(deps, spec_id, head)
    ahead_reads = _read_at(store, head=head)
    work = _count_work(monkeypatch)
    read = _post(client, tree)

    assert (ahead.due, ahead.rendered, ahead.failed) == (1, 1, 0)
    # The render ahead records no read; the reader's read of it does.
    assert ahead_reads == [None]
    assert _read_at(store, head=head) != [None]
    assert read.status_code == 200, read.text
    assert work["_render_stored_rendering"] == 0
    (derived,) = read.json()["derived_roots"]
    assert derived["base_content_version"] == head


def test_a_second_render_ahead_from_the_same_head_renders_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, spec_id, _, deps, store_new_head = _read_spec_root()
    head = store_new_head(monkeypatch)
    assert _ahead(deps, spec_id, head).rendered == 1

    work = _count_work(monkeypatch)
    again = _ahead(deps, spec_id, head)

    assert (again.due, again.rendered) == (0, 0)
    assert work["_render_stored_rendering"] == 0


@pytest.mark.parametrize("unread", ["never_read", "read_before_the_window"])
def test_a_rendering_no_root_read_in_the_window_is_not_rendered_ahead(
    monkeypatch: pytest.MonkeyPatch, unread: str
) -> None:
    _, store, spec_id, _, deps, store_new_head = _read_spec_root()
    head = store_new_head(monkeypatch)
    since = deps.now() - timedelta(days=1)
    if unread == "never_read":
        with store.transaction() as tx:
            tx.execute("UPDATE datasets.version_artifact_head SET read_at = NULL")
    else:
        since = deps.now() + timedelta(seconds=1)

    work = _count_work(monkeypatch)
    ahead = _ahead(deps, spec_id, head, since=since)

    assert (ahead.due, ahead.rendered) == (0, 0)
    assert work["_render_stored_rendering"] == 0


def test_the_cap_renders_the_latest_read_first_and_leaves_the_rest_for_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, spec_id, trees, deps, store_new_head = _read_spec_root(two_axes=True)
    head = store_new_head(monkeypatch)

    first = _ahead(deps, spec_id, head, limit=1)
    work = _count_work(monkeypatch)
    latest = _post(client, trees[-1])
    latest_renders = work["_render_stored_rendering"]
    second = _ahead(deps, spec_id, head, limit=1)
    third = _ahead(deps, spec_id, head, limit=1)

    assert (first.due, first.rendered, first.left) == (2, 1, 1)
    assert latest.status_code == 200, latest.text
    assert latest_renders == 0
    assert (second.due, second.rendered, second.left) == (1, 1, 0)
    assert (third.due, third.rendered) == (0, 0)


def test_a_root_reading_a_recorded_sum_records_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    client, store, _, (tree,), _, _ = _read_spec_root()
    with store.transaction() as tx:
        tx.execute("UPDATE datasets.version_artifact_head SET read_at = NULL")

    work = _count_work(monkeypatch)
    assert _post(client, tree).status_code == 200

    assert work["_render_stored_rendering"] == 0
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT h.read_at, a.rendering FROM datasets.version_artifact_head h "
            "JOIN datasets.version_artifact a USING (subject_id, content_version, rendering_key)"
        )
    (summed,) = [row for row in rows if "groupby" in row["rendering"]]
    assert summed["read_at"] is not None


def test_the_maintenance_run_renders_from_each_read_head_within_the_total_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store, spec_id, _, deps, store_new_head = _read_spec_root(two_axes=True)
    head = store_new_head(monkeypatch)
    lines: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(maintenance, "emit", lambda event, **fields: lines.append((event, fields)))
    # The rig's head store holds no rows here, so the table names the new head itself.
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.subject_head "
            "(subject_id, content_version, confirmed_at, materialised_at) VALUES (%s, %s, %s, %s)",
            (spec_id, head, deps.now(), deps.now()),
        )
    capped = dataclasses.replace(
        deps, settings=deps.settings.model_copy(update={"render_ahead_max": 1})
    )

    assert heads_read_since(store, read_since=deps.now() - timedelta(days=1)) == [(spec_id, head)]
    failed = maintenance.run_render_ahead(
        store=store, deps=capped, deadline=float("inf"), now=deps.now
    )
    again = maintenance.run_render_ahead(
        store=store, deps=deps, deadline=float("inf"), now=deps.now
    )
    work = _count_work(monkeypatch)
    last = maintenance.run_render_ahead(store=store, deps=deps, deadline=float("inf"), now=deps.now)

    assert failed == again == last == 0
    counts = [
        {key: fields[key] for key in ("heads", "rendered", "left")}
        for event, fields in lines
        if event == "datasets_render_ahead_run"
    ]
    assert counts == [
        {"heads": 1, "rendered": 1, "left": 1},
        {"heads": 1, "rendered": 1, "left": 0},
        {"heads": 1, "rendered": 0, "left": 0},
    ]
    assert work == Counter()


def test_heads_read_since_lists_no_dataset_nobody_read() -> None:
    assert heads_read_since(FakeVersionRecordDatasetStore(), read_since=current.utc_now()) == []


def test_an_inline_rebuild_queues_one_render_ahead_that_the_worker_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild on a request thread stores the new head and queues its render ahead as one
    operation, which renders nothing on that thread; the worker's run of it renders the root's
    sum, so the root's next read renders nothing. Storing the same head again, while the
    operation is queued or once it is done, queues nothing new."""
    stores: list[LiveOperationStore] = []

    def operation_store(store: FakeVersionRecordDatasetStore) -> LiveOperationStore:
        stores.append(LiveOperationStore(store=store, backlog_max=20))
        return stores[-1]

    tasks = FakeTaskClient()
    client, _, spec_id, (tree,), deps, store_new_head = _read_spec_root(
        operation_store=operation_store, task_client=tasks
    )
    (operations,) = stores
    assert tasks.enqueued == []

    work = _count_work(monkeypatch)
    head = store_new_head(monkeypatch)
    inline_renders = work["_render_stored_rendering"]
    ((operation_id, submission),) = tasks.enqueued
    queued = operations.get(operation_id)
    assert queued is not None
    assert queued.request_identity == render_ahead_identity(spec_id, head_content_version=head)
    assert store_new_head(monkeypatch) == head
    claimed = operations.claim(
        operation_id, submission_count=submission, lease_seconds=60, final_delivery=False
    )
    assert claimed is not None
    run_operation(operation=claimed, deps=deps, operation_store=operations)
    worker_renders = work["_render_stored_rendering"] - inline_renders
    assert store_new_head(monkeypatch) == head
    work.clear()
    read = _post(client, tree)

    assert inline_renders == 0
    assert worker_renders == 1
    done = operations.get(operation_id)
    assert done is not None
    assert (done.state, done.content_version) == ("done", head)
    assert tasks.enqueued == [(operation_id, submission)]
    assert read.status_code == 200, read.text
    assert work["_render_stored_rendering"] == 0


def test_a_rendering_linked_under_the_head_itself_is_not_rendered_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sum that leaves a head's frame as it is links its file under the head itself, with no
    head row, so the render ahead reads the head's own links too."""
    _, store, spec_id, _, deps, store_new_head = _read_spec_root()
    head = store_new_head(monkeypatch)
    with store.transaction() as tx:
        (link,) = [
            row
            for row in tx.fetch_all("SELECT * FROM datasets.version_artifact")
            if "groupby" in row["rendering"]
        ]
        tx.execute(
            "INSERT INTO datasets.version_artifact (subject_id, content_version, "
            "artifact_digest, rendering_key, rendering, media_type) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                spec_id,
                head,
                link["artifact_digest"],
                link["rendering_key"],
                link["rendering"],
                link["media_type"],
            ),
        )

    work = _count_work(monkeypatch)
    ahead = _ahead(deps, spec_id, head)

    assert (ahead.due, ahead.rendered) == (0, 0)
    assert work["_render_stored_rendering"] == 0


def test_a_render_ahead_from_a_version_that_is_no_longer_the_head_renders_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, spec_id, _, deps, store_new_head = _read_spec_root()
    moved_from = store_new_head(monkeypatch)
    store_new_head(monkeypatch, _MOVED_BODY.replace(",2020,600,", ",2020,700,"))

    work = _count_work(monkeypatch)
    ahead = _ahead(deps, spec_id, moved_from)

    assert ahead.head_moved
    assert (ahead.due, ahead.rendered) == (0, 0)
    assert work["_render_stored_rendering"] == 0


def _queued_rig() -> tuple[Any, ...]:
    stores: list[LiveOperationStore] = []

    def operation_store(store: FakeVersionRecordDatasetStore) -> LiveOperationStore:
        stores.append(LiveOperationStore(store=store, backlog_max=20))
        return stores[-1]

    tasks = FakeTaskClient()
    client, store, spec_id, trees, deps, store_new_head = _read_spec_root(
        operation_store=operation_store, task_client=tasks
    )
    return client, store, spec_id, trees, deps, store_new_head, stores[0], tasks


def test_a_store_that_lost_the_head_to_another_writer_queues_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, deps, store_new_head, _, tasks = _queued_rig()
    head_store = deps.head_store
    advance = head_store.advance_head

    def lost(**kwargs: Any) -> Any:
        advance(**kwargs)
        return None

    monkeypatch.setattr(head_store, "advance_head", lost)
    store_new_head(monkeypatch)

    assert tasks.enqueued == []


def test_a_queued_render_ahead_whose_head_moved_renders_nothing_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, deps, store_new_head, operations, tasks = _queued_rig()
    first = store_new_head(monkeypatch)
    store_new_head(monkeypatch, _MOVED_BODY.replace(",2020,600,", ",2020,700,"))
    (operation_id, submission), _ = tasks.enqueued
    claimed = operations.claim(
        operation_id, submission_count=submission, lease_seconds=60, final_delivery=False
    )
    assert claimed is not None

    work = _count_work(monkeypatch)
    run_operation(operation=claimed, deps=deps, operation_store=operations)

    assert work["_render_stored_rendering"] == 0
    done = operations.get(operation_id)
    assert done is not None
    assert (done.state, done.content_version) == ("done", first)


def test_a_later_store_of_the_head_queues_a_retryable_failed_render_ahead_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, _, store_new_head, operations, tasks = _queued_rig()
    store_new_head(monkeypatch)
    ((operation_id, submission),) = tasks.enqueued
    claimed = operations.claim(
        operation_id, submission_count=submission, lease_seconds=60, final_delivery=False
    )
    assert claimed is not None
    operations.fail(
        operation_id, attempt=claimed.attempt_count, refusal_code="internal_error", retryable=True
    )

    store_new_head(monkeypatch)

    assert [entry[0] for entry in tasks.enqueued] == [operation_id, operation_id]
    assert tasks.enqueued[1][1] > submission
    reset = operations.get(operation_id)
    assert reset is not None
    assert reset.state == "pending"


@pytest.mark.parametrize("final", [False, True], ids=["retried", "last_attempt"])
def test_a_store_failure_in_a_render_ahead_operation_is_retried_by_cloud_tasks(
    monkeypatch: pytest.MonkeyPatch, final: bool
) -> None:
    _, _, _, _, deps, store_new_head, operations, tasks = _queued_rig()
    store_new_head(monkeypatch)
    ((operation_id, submission),) = tasks.enqueued
    claimed = operations.claim(
        operation_id, submission_count=submission, lease_seconds=60, final_delivery=final
    )
    assert claimed is not None

    def unavailable(*_: Any, **__: Any) -> Any:
        raise DatasetsStoreUnavailable("store down")

    monkeypatch.setattr(worker_module, "render_ahead", unavailable)
    with contextlib.nullcontext() if final else pytest.raises(OperationTransientFailure):
        run_operation(
            operation=claimed, deps=deps, operation_store=operations, is_final_attempt=final
        )

    row = operations.get(operation_id)
    assert row is not None
    assert (row.state, row.retryable) == (("failed", True) if final else ("pending", None))


def test_a_verification_records_no_read() -> None:
    settings = DatasetsBuildSettings.from_env().model_copy(update={"verify_principal": "probe"})
    client, store, root = _rig(settings=settings)
    tree = _aggregate(root, group_by=["recipient"])
    assert _post(client, tree).status_code == 200
    assert _read_at(store) != [None]
    _set_read_at(store, None)

    verified = client.post(
        "/api/datasets/analysis",
        json={"tree": tree, "verify": True},
        headers={"x-dfo-principal": "probe"},
    )

    assert verified.status_code == 200, verified.text
    assert _read_at(store) == [None]


def test_a_store_failure_while_recording_a_read_leaves_the_read_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, store, _, (tree,), _, _ = _read_spec_root()
    _set_read_at(store, None)

    def unavailable(*_: Any, **__: Any) -> None:
        raise DatasetsStoreUnavailable("store down")

    monkeypatch.setattr(LiveVersionLister, "mark_rendering_read", unavailable)
    read = _post(client, tree)

    assert read.status_code == 200, read.text
    assert _read_at(store) == [None]


@pytest.mark.parametrize(
    ("age", "recorded_again"), [(timedelta(minutes=30), False), (timedelta(hours=2), True)]
)
def test_a_read_is_recorded_again_only_once_the_last_is_an_hour_old(
    age: timedelta, recorded_again: bool
) -> None:
    client, store, _, (tree,), deps, _ = _read_spec_root()
    earlier = deps.now() - age
    _set_read_at(store, earlier)

    assert _post(client, tree).status_code == 200

    (read_at,) = _read_at(store)
    assert (read_at != earlier) is recorded_again
