"""Pins the import key, the build lease and the conditional delete it releases through."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
from google.api_core.exceptions import NotFound, PreconditionFailed

from artifact_publication.jobs import (
    LEASE_FILE,
    ImportKey,
    acquire_lease,
    read_import_key,
    read_import_key_bq,
    release_lease,
    tracked_stamp,
    unpublished_seconds,
)
from artifact_publication.uri import delete_uri_if_unchanged, read_uri_bytes_and_generation
from tests.artifact_publication.support import (
    BUCKET,
    FakeBigQuery,
    FakeBlob,
    history_row,
    import_key_responses,
)

_CONNECTION = "proj.region.conn"
_LATER = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
_UNLOCKED = {"LockOwner": None}


def _key(*, lock_owner: str | None = None, newest: dict[str, Any] | None = None) -> ImportKey:
    success = history_row("wf-1")
    return ImportKey.from_rows(
        {"LockOwner": lock_owner}, success if newest is None else newest, success
    )


class _FakeSnapshot:
    """A Spanner snapshot's `execute_sql`: rows as sequences in column order,
    from the first marker the statement contains."""

    def __init__(self, responses: dict[str, list[Sequence[object]]]) -> None:
        self._responses = responses
        self.statements: list[str] = []

    def execute_sql(self, sql: str) -> list[Sequence[object]]:
        self.statements.append(sql)
        return next(rows for marker, rows in self._responses.items() if marker in sql)


# --- Import key --------------------------------------------------------------


@pytest.mark.parametrize(
    "newest",
    [
        history_row("wf-2", created_at=_LATER),
        history_row("wf-2", status="FAILURE", failed=True, created_at=_LATER),
        history_row("wf-2", status="RETRY", failed=True, created_at=_LATER),
        {**history_row("wf-2", status="PENDING", created_at=_LATER), "CompletionTimestamp": None},
        # A rollback retry reuses its row, so a status change on one id is new too.
        history_row("wf-1", status="FAILURE", failed=True),
    ],
    ids=["success", "failure", "retry", "pending", "same-id-new-status"],
)
def test_a_new_history_row_of_any_status_changes_the_stamp(newest):
    assert _key(newest=newest).stamp != _key().stamp


@pytest.mark.parametrize(
    ("key", "buildable"),
    [
        (_key(), True),
        (_key(lock_owner="ingestion"), False),
        (_key(newest=history_row("wf-2", status="FAILURE", failed=True, created_at=_LATER)), False),
        (
            ImportKey.from_rows(
                _UNLOCKED, history_row("wf-1", status="FAILURE", failed=True), None
            ),
            False,
        ),
    ],
    ids=["clean", "lock-held", "failure-newer-than-success", "no-success"],
)
def test_buildable_needs_a_free_lock_and_the_newest_row_to_be_the_newest_success(key, buildable):
    assert key.buildable is buildable


def test_two_reads_with_no_new_row_give_equal_stamps_and_scan_no_observation():
    bq = FakeBigQuery(import_key_responses())

    first = read_import_key_bq(bq, connection=_CONNECTION)
    locked = read_import_key_bq(
        FakeBigQuery(import_key_responses(lock_owner="ingestion")), connection=_CONNECTION
    )
    second = read_import_key_bq(bq, connection=_CONNECTION)

    assert first is not None and second is not None and locked is not None
    assert first.stamp == second.stamp == locked.stamp
    assert (first.buildable, locked.buildable) == (True, False)
    assert len(bq.queries) == 6
    assert not [sql for sql in bq.queries if "Observation" in sql or "last_update" in sql]


@pytest.mark.parametrize(
    "created_at",
    [
        DatetimeWithNanoseconds(2026, 9, 24, 10, 0, nanosecond=123, tzinfo=UTC),
        datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
        datetime(2026, 9, 24, 12, 0, tzinfo=timezone(timedelta(hours=2))),
    ],
    ids=["spanner-nanoseconds", "bigquery-datetime", "other-offset"],
)
def test_both_readers_give_one_stamp_for_one_history_row(created_at):
    """BigQuery returns microsecond `datetime`s and the Spanner client
    `DatetimeWithNanoseconds`. The key normalises both, so the readers agree."""
    bq_key = read_import_key_bq(FakeBigQuery(import_key_responses()), connection=_CONNECTION)
    spanner_row = tuple(history_row("wf-1", created_at=created_at).values())
    snapshot = _FakeSnapshot({"IngestionLock": [(None,)], "IngestionHistory": [spanner_row]})

    spanner_key = read_import_key(snapshot)

    assert bq_key is not None and spanner_key is not None
    assert spanner_key == bq_key
    assert spanner_key.stamp == bq_key.stamp
    assert len(snapshot.statements) == 3


def test_an_empty_history_gives_no_key():
    bq = FakeBigQuery({"FROM IngestionLock": [_UNLOCKED], "FROM IngestionHistory": []})
    snapshot = _FakeSnapshot({"IngestionLock": [(None,)], "IngestionHistory": []})

    assert read_import_key_bq(bq, connection=_CONNECTION) is None
    assert read_import_key(snapshot) is None


# --- Build lease -------------------------------------------------------------


def _lease(tmp_path) -> dict:
    return json.loads((tmp_path / LEASE_FILE).read_text())


def test_a_lease_is_taken_once_until_released(tmp_path):
    root = tmp_path.as_uri()

    held = acquire_lease(root, build_id="a", ttl_seconds=60, now=lambda: 1000.0)

    assert held is not None
    assert _lease(tmp_path) == {"build_id": "a", "expires_at": 1060.0}
    assert acquire_lease(root, build_id="b", ttl_seconds=60, now=lambda: 1059.0) is None
    assert release_lease(root, held) is True
    assert not (tmp_path / LEASE_FILE).exists()
    assert acquire_lease(root, build_id="b", ttl_seconds=60, now=lambda: 1059.0) is not None


def test_an_expired_lease_is_reclaimed_and_its_holder_cannot_release_it(tmp_path):
    root = tmp_path.as_uri()
    stale = acquire_lease(root, build_id="a", ttl_seconds=60, now=lambda: 1000.0)
    assert stale is not None

    fresh = acquire_lease(root, build_id="b", ttl_seconds=60, now=lambda: 1060.0)

    assert fresh is not None
    assert release_lease(root, stale) is False
    assert _lease(tmp_path)["build_id"] == "b"


def test_a_lease_that_does_not_parse_is_reclaimed(tmp_path):
    (tmp_path / LEASE_FILE).write_text("{")

    assert acquire_lease(tmp_path.as_uri(), build_id="a", ttl_seconds=60) is not None
    assert _lease(tmp_path)["build_id"] == "a"


def _conditional_delete(self, if_generation_match: int | None = None) -> None:
    stored = self._objects.get(self.name)
    if stored is None:
        raise NotFound(f"gs://{BUCKET}/{self.name} does not exist")
    if if_generation_match is not None and stored.generation != if_generation_match:
        raise PreconditionFailed(f"generation mismatch for {self.name}")
    del self._objects[self.name]


def test_gs_lease_release_deletes_only_the_generation_it_took(gs_bucket, monkeypatch):
    monkeypatch.setattr(FakeBlob, "delete", _conditional_delete)
    root = f"gs://{BUCKET}/catalogue/format-1"
    stale = acquire_lease(root, build_id="a", ttl_seconds=60, now=lambda: 1000.0)
    assert stale is not None
    fresh = acquire_lease(root, build_id="b", ttl_seconds=60, now=lambda: 2000.0)
    assert fresh is not None

    assert release_lease(root, stale) is False
    assert release_lease(root, fresh) is True
    assert read_uri_bytes_and_generation(f"{root}/{LEASE_FILE}") is None
    assert delete_uri_if_unchanged(f"{root}/{LEASE_FILE}", expected_generation="1") is False


def test_tracked_stamp_reads_the_stamp_without_restarting_its_clock(tmp_path):
    root = tmp_path.as_uri()
    assert tracked_stamp(root, state_file="state.json") is None

    assert unpublished_seconds(root, state_file="state.json", stamp="a", now=10.0) == 0.0
    assert tracked_stamp(root, state_file="state.json") == "a"
    assert unpublished_seconds(root, state_file="state.json", stamp="a", now=25.0) == 15.0

    (tmp_path / "state.json").write_text("not json")
    assert tracked_stamp(root, state_file="state.json") is None
