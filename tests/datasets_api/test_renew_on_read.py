"""Renewal on read (`datasets_api.operations.renewal`): a read of a working-tier file whose
retention ends within the renewal margin queues one renewal for the worker and answers as
before; the worker uploads the bytes again under a renewal name, which restarts the location's
`stored_at` and the object's age. A file further from the end of retention, a pinned one, and a
second read queue nothing, and a failing operation store never fails the read."""

from __future__ import annotations

import dataclasses
import functools
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dataset_store.snapshot import WORKING_TIER_RETENTION, StoredArtifactLocation
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import analysis as analysis_door
from datasets_api.doors import files
from datasets_api.doors.mint_errors import MintBackgroundDeclined
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations import renewal
from datasets_api.operations.renewal import RENEW_MARKER, renew_if_expiring
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker import OperationTransientFailure, run_operation
from datasets_api.store_errors import DatasetsStoreUnavailable

from ._version_record_store import FakeVersionRecordDatasetStore
from .test_analysis_summed_roots import _aggregate, _post, _rig
from .test_render_ahead import _captured_deps

_INSIDE_THE_MARGIN = WORKING_TIER_RETENTION - timedelta(days=5)
_OUTSIDE_THE_MARGIN = timedelta(days=10)


def _renewing_rig(settings: DatasetsBuildSettings | None = None) -> tuple[Any, ...]:
    """A dataset root's client whose analysis has read its sum once, its version store, the
    tree, the first answer, the deps the read ran with, its operation store and task client."""
    stores: list[LiveOperationStore] = []

    def operation_store(store: FakeVersionRecordDatasetStore) -> LiveOperationStore:
        stores.append(LiveOperationStore(store=store, backlog_max=20))
        return stores[-1]

    tasks = FakeTaskClient()
    client, store, root = _rig(
        operation_store=operation_store, task_client=tasks, settings=settings
    )
    tree = _aggregate(root, group_by=["recipient"])
    with pytest.MonkeyPatch.context() as mp:
        captured = _captured_deps(mp)
        first = _post(client, tree)
    assert first.status_code == 200
    deps = dataclasses.replace(captured[-1], deadline=None)
    return client, store, tree, first.json(), deps, stores[0], tasks


def _age(store: FakeVersionRecordDatasetStore, by: timedelta) -> None:
    """Move every working location's `stored_at` `by` into the past."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest, stored_at FROM datasets.artifact_location "
            "WHERE tier = 'working'",
            (),
        )
        for row in rows:
            tx.execute(
                "UPDATE datasets.artifact_location SET stored_at = %s "
                "WHERE artifact_digest = %s AND tier = 'working'",
                ((row["stored_at"] - by).isoformat(), row["artifact_digest"]),
            )


def _renewals(operations: LiveOperationStore, tasks: FakeTaskClient) -> list[dict[str, Any]]:
    """The `request_identity` of each renewal task created, in order."""
    identities = []
    for operation_id, _ in tasks.enqueued:
        operation = operations.get(operation_id)
        assert operation is not None
        if RENEW_MARKER in operation.request_identity:
            identities.append(dict(operation.request_identity))
    return identities


def _root_renewals(operations: LiveOperationStore, tasks: FakeTaskClient) -> list[dict[str, Any]]:
    """The renewals a root read queued: those of a dataset's version, not an answer's `ops_`
    pair."""
    return [
        renewal
        for renewal in _renewals(operations, tasks)
        if not str(renewal["subject_id"]).startswith("ops_")
    ]


def _location(store: FakeVersionRecordDatasetStore, digest: str) -> tuple[str, datetime, int]:
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT object_name, stored_at, renewal FROM datasets.artifact_location "
            "WHERE artifact_digest = %s AND tier = 'working'",
            (digest,),
        )
    assert row is not None
    return row["object_name"], row["stored_at"], row["renewal"]


def _answer_frame(answer: dict[str, Any]) -> tuple[str, str]:
    """The `ops_` spec id and frame file digest an analysis answer's ref names."""
    member = answer["ref"]["members"][0]
    return member["spec"]["spec_id"], member["artifact"]["digest"]


def _version_of(store: FakeVersionRecordDatasetStore, subject_id: str) -> str:
    """The one content version recorded for `subject_id`."""
    with store.transaction() as tx:
        (row,) = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record WHERE subject_id = %s",
            (subject_id,),
        )
    return row["content_version"]


def _claimed_renewal(operations: LiveOperationStore, tasks: FakeTaskClient, *, final: bool) -> Any:
    (operation_id, submission), *_ = tasks.enqueued
    claimed = operations.claim(
        operation_id, submission_count=submission, lease_seconds=60, final_delivery=final
    )
    assert claimed is not None
    assert RENEW_MARKER in claimed.request_identity
    return claimed


@pytest.mark.parametrize(
    ("age", "queued"),
    [(_INSIDE_THE_MARGIN, 2), (_OUTSIDE_THE_MARGIN, 0)],
    ids=["inside_the_margin", "outside_the_margin"],
)
def test_a_root_read_inside_the_margin_queues_one_renewal_per_file_and_answers_unchanged(
    age: timedelta, queued: int
) -> None:
    """The derived sum's stored frame and the file the root reads each queue one renewal; a
    second read queues nothing more."""
    client, store, tree, first, _, operations, tasks = _renewing_rig()
    assert tasks.enqueued == []
    _age(store, age)

    second = _post(client, tree)
    third = _post(client, tree)

    assert (second.status_code, third.status_code) == (200, 200)
    assert second.json() == third.json() == first
    renewals = _root_renewals(operations, tasks)
    assert len(renewals) == queued
    assert len({renewal[RENEW_MARKER] for renewal in renewals}) == queued
    # The stored answer's own frame file queues once more, under its `ops_` pair.
    assert len(tasks.enqueued) == len(_renewals(operations, tasks)) == queued + bool(queued)


def test_a_verification_read_queues_no_renewal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A root source that renders nothing (`LiveRootSource(render=False)`) is a verification."""
    client, store, tree, _, _, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)
    monkeypatch.setattr(
        analysis_door,
        "LiveRootSource",
        functools.partial(analysis_door.LiveRootSource, render=False),
    )

    assert _post(client, tree).status_code == 200

    assert _root_renewals(operations, tasks) == []


def test_a_margin_of_zero_queues_no_renewal() -> None:
    client, store, tree, _, _, _, tasks = _renewing_rig(
        DatasetsBuildSettings(renew_margin_seconds=0)
    )
    _age(store, _INSIDE_THE_MARGIN)

    assert _post(client, tree).status_code == 200

    assert tasks.enqueued == []


@pytest.mark.parametrize("failure", ["operation_store", "task_client"])
def test_a_failing_operation_store_or_task_client_never_fails_the_read(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client, store, tree, first, _, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)

    def down(*args: Any, **kwargs: Any) -> Any:
        raise DatasetsStoreUnavailable("down")

    if failure == "operation_store":
        monkeypatch.setattr(operations, "create_or_get", down)
    else:
        monkeypatch.setattr(tasks, "enqueue", down)
    answer = _post(client, tree)

    assert answer.status_code == 200
    assert answer.json() == first


def test_the_worker_renewal_restarts_the_location_and_its_object() -> None:
    """The worker uploads the bytes again under `<digest>#r1` with a fresh `stored_at`, so the
    file's `held_until` moves a retention on, and completes the row naming the version. A read
    of the renewed file queues nothing more."""
    client, store, tree, _, deps, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)
    assert _post(client, tree).status_code == 200
    claimed = _claimed_renewal(operations, tasks, final=False)
    digest = claimed.request_identity[RENEW_MARKER]
    object_name, aged, renewal = _location(store, digest)
    assert (object_name, renewal) == (digest, 0)

    before = datetime.now(UTC)
    run_operation(operation=claimed, deps=deps, operation_store=operations)

    object_name, stored_at, renewal = _location(store, digest)
    assert (object_name, renewal) == (f"{digest}#r1", 1)
    assert stored_at >= before > aged + timedelta(days=20)
    done = operations.get(claimed.id)
    assert done is not None
    assert (done.state, done.subject_id, done.content_version) == (
        "done",
        claimed.request_identity["subject_id"],
        claimed.request_identity["content_version"],
    )
    queued = len(tasks.enqueued)
    assert _post(client, tree).status_code == 200
    assert len(tasks.enqueued) == queued


def test_a_worker_whose_margin_is_smaller_than_the_readers_still_renews() -> None:
    """The worker renews the generation its read queued whatever its own margin."""
    client, store, tree, _, deps, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)
    assert _post(client, tree).status_code == 200
    claimed = _claimed_renewal(operations, tasks, final=False)
    digest = claimed.request_identity[RENEW_MARKER]
    worker = dataclasses.replace(
        deps, settings=deps.settings.model_copy(update={"renew_margin_seconds": 60})
    )

    run_operation(operation=claimed, deps=worker, operation_store=operations)

    assert _location(store, digest)[0] == f"{digest}#r1"


@pytest.mark.parametrize("failure", ["store", "permit"])
@pytest.mark.parametrize("final", [False, True], ids=["retried", "last_attempt"])
def test_a_store_failure_in_the_worker_renewal_retries_or_fails_retryable(
    monkeypatch: pytest.MonkeyPatch, final: bool, failure: str
) -> None:
    """A store failure, or no materialise permit in time (`MintBackpressure`), releases the row
    for a retry before the last attempt and fails it retryable on the last."""
    client, store, tree, _, deps, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)
    assert _post(client, tree).status_code == 200
    claimed = _claimed_renewal(operations, tasks, final=final)

    def down(*args: Any, **kwargs: Any) -> Any:
        raise DatasetsStoreUnavailable("down")

    def full(*args: Any, **kwargs: Any) -> Any:
        raise MintBackgroundDeclined()

    if failure == "store":
        monkeypatch.setattr(deps.snapshot_reader, "read_artifact_locations", down)
    else:
        monkeypatch.setattr(renewal, "bounded_materialise_slot", full)
    if final:
        run_operation(
            operation=claimed, deps=deps, operation_store=operations, is_final_attempt=True
        )
    else:
        with pytest.raises(OperationTransientFailure):
            run_operation(operation=claimed, deps=deps, operation_store=operations)

    left = operations.get(claimed.id)
    assert left is not None
    assert (left.state, left.retryable) == (("failed", True) if final else ("pending", None))


def test_a_stored_answer_hit_queues_the_renewal_of_its_frame_file() -> None:
    """The analysis door's stored-answer hit queues the renewal of the frame file its body
    names, under the `ops_` pair the body is stored under."""
    client, store, tree, first, _, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)
    spec_id, digest = _answer_frame(first)

    answer = _post(client, tree)

    assert answer.json() == first
    assert {
        "subject_id": spec_id,
        "content_version": _version_of(store, spec_id),
        RENEW_MARKER: digest,
    }.items() <= next(
        renewal for renewal in _renewals(operations, tasks) if renewal["subject_id"] == spec_id
    ).items()


def test_a_file_download_inside_the_margin_queues_its_renewal() -> None:
    """The files door's servable link (`files.resolve_link`) queues the renewal of the file it
    signs."""
    _, store, tree, _, deps, operations, tasks = _renewing_rig()
    _age(store, _INSIDE_THE_MARGIN)
    root = tree["inputs"]["data"]
    digest = deps.snapshot_reader.read_artifact_locations(root["spec_id"], root["content_version"])[
        0
    ].artifact_digest

    link = files.resolve_link(
        id=root["spec_id"], content_version=root["content_version"], digest=digest, deps=deps
    )

    assert link is not None
    assert [renewal[RENEW_MARKER] for renewal in _renewals(operations, tasks)] == [digest]


@pytest.mark.parametrize(("tier", "queued"), [("working", 1), ("pinned", 0)])
def test_a_pinned_location_never_queues_a_renewal(tier: str, queued: int) -> None:
    _, _, _, _, deps, operations, tasks = _renewing_rig()
    aged = deps.now() - _INSIDE_THE_MARGIN
    location = StoredArtifactLocation(
        artifact_digest="sha256:" + "0" * 64,
        rendering_key="rk",
        rendering={},
        media_type="text/csv",
        size_bytes=1,
        created_at=aged,
        stored_at=aged,
        tier=tier,
        bucket="bucket",
        object_name="sha256:" + "0" * 64,
    )

    renew_if_expiring(deps, location, subject_id="spec_a", content_version="cv_a")

    assert len(_renewals(operations, tasks)) == len(tasks.enqueued) == queued
