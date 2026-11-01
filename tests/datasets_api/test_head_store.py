"""Tests `LiveHeadStore` compare-and-set writes against the SQLite-shaped fake store.
`advance_head` updates both timestamps and applies only while the head names the run the caller
observed. `confirm_head` updates `confirmed_at` only.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveHeadStore
from materialise.trace import UNKNOWN_FAMILY

from ._store_fault_doubles import PoolTimeoutFakeDatasetStore
from ._version_record_store import (
    TEST_BASIS,
    FakeVersionRecordDatasetStore,
    materialisation_record,
)


@pytest.fixture
def version_store():
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def _insert_version_record(version_store, *, subject_id: str, content_version: str) -> None:
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            (subject_id, "spec", content_version, 1, {}, datetime(2020, 1, 1, tzinfo=UTC)),
        )


def test_read_head_returns_none_for_an_unknown_subject(version_store):
    store = LiveHeadStore(store=version_store)
    assert store.read_head("spec_a") is None


def test_read_head_translates_a_pool_timeout_into_datasets_store_unavailable():
    """A pool timeout before transaction entry becomes `DatasetsStoreUnavailable` without SQLSTATE."""
    store = LiveHeadStore(store=PoolTimeoutFakeDatasetStore())

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        store.read_head("spec_a")

    assert excinfo.value.sqlstate is None
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_advance_head_writes_content_version_confirmed_at_and_dependencies(version_store):
    """An empty-row write with `expected_materialisation_id=None` sets both timestamps equally."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    written = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )

    assert written is not None
    assert written.content_version == "cv1"
    assert written.confirmed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert written.materialised_at == datetime(2026, 1, 1, tzinfo=UTC)
    head = store.read_head("spec_a")
    assert head is not None
    assert head.subject_id == "spec_a"
    assert head.content_version == "cv1"
    assert head.confirmed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert head.materialised_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        head.dependencies
        == materialisation_record(subject_id="spec_a", content_version="cv1").dependencies
    )


def test_advance_head_moves_both_timestamps_on_a_real_advance(version_store):
    """Changing `content_version` updates both timestamps to the writer's clock."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    first_write = datetime(2026, 1, 1, tzinfo=UTC)
    second_write = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    first = LiveHeadStore(store=version_store, now=lambda: first_write).advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )
    assert first is not None

    written = LiveHeadStore(store=version_store, now=lambda: second_write).advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv2"),
        expected_materialisation_id=first.materialisation_id,
    )

    assert written is not None
    assert written.content_version == "cv2"
    assert written.confirmed_at == second_write
    assert written.materialised_at == second_write


def test_advance_head_the_slower_older_decision_loses_the_race_and_returns_none(version_store):
    """Two requests read the `cv0` head. The later decision writes `cv2` first, so the earlier
    decision reaches the store with a stale expectation and loses its compare-and-set.
    The compare-and-set follows the stored run, independent of request start time.
    """
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv0")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    observed = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv0"),
        expected_materialisation_id=None,
    )
    assert observed is not None

    # The later-started decision materialises fast and writes first.
    new_result = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv2"),
        expected_materialisation_id=observed.materialisation_id,
    )
    assert new_result is not None
    assert new_result.content_version == "cv2"

    # The earlier decision also read the cv0 head and reaches the store after the later write.
    old_result = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=observed.materialisation_id,
    )
    assert old_result is None

    head = store.read_head("spec_a")
    assert head is not None
    assert head.content_version == "cv2"


def test_advance_head_expected_none_against_an_existing_row_returns_none(version_store):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )

    result = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )

    assert result is None


def test_advance_head_conflict_leaves_the_stored_row_unchanged(version_store):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )
    before = store.read_head("spec_a")
    assert before is not None and before.materialisation_id is not None

    result = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv2"),
        expected_materialisation_id=before.materialisation_id + 1000,
    )

    assert result is None
    after = store.read_head("spec_a")
    assert after == before


def test_advance_head_on_the_same_content_version_is_a_rematerialisation(version_store):
    """Rematerialising the stored version from the observed run applies, names the new run, and
    advances both timestamps."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    first_check = datetime(2026, 1, 1, tzinfo=UTC)
    second_check = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)

    first = LiveHeadStore(store=version_store, now=lambda: first_check).advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )
    assert first is not None
    written = LiveHeadStore(store=version_store, now=lambda: second_check).advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=first.materialisation_id,
    )

    assert written is not None
    assert written.materialisation_id != first.materialisation_id
    assert written.confirmed_at == second_check
    assert written.materialised_at == second_check
    head = LiveHeadStore(store=version_store).read_head("spec_a")
    assert head is not None
    assert head.content_version == "cv1"
    assert head.confirmed_at == second_check
    assert head.materialised_at == second_check


def test_two_rebuilds_of_the_heads_own_version_from_one_observed_run_have_one_winner(
    version_store,
):
    """I5. Both rebuilds observed run `first` and produce `cv1` again. The first to commit moves
    the head to its run; the second no longer matches the observed run and loses, leaving the
    winner's run and fingerprint."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    first = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )
    assert first is not None

    winner = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=first.materialisation_id,
    )
    loser = store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=first.materialisation_id,
    )

    assert winner is not None
    assert loser is None
    head = store.read_head("spec_a")
    assert head is not None
    assert head.materialisation_id == winner.materialisation_id


def test_read_head_drops_the_fingerprint_of_a_head_whose_run_made_another_version(version_store):
    """A head advanced by a writer that leaves `materialisation_id` and `dependencies` alone
    names a run of another version. The head read reports no fingerprint, so freshness rebuilds
    it once (`schema`)."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.subject_head SET content_version = %s WHERE subject_id = %s",
            ("cv2", "spec_a"),
        )

    head = store.read_head("spec_a")

    assert head is not None
    assert head.content_version == "cv2"
    assert head.dependencies is None


def test_confirm_head_on_the_stored_content_version_moves_confirmed_at_only(version_store):
    """A freshness confirmation advances `confirmed_at` and preserves the version,
    materialisation time, and dependencies.
    """
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    materialised_at = datetime(2026, 1, 1, tzinfo=UTC)
    confirmed_at = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    LiveHeadStore(store=version_store, now=lambda: materialised_at).advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )

    confirmed = LiveHeadStore(store=version_store, now=lambda: confirmed_at).confirm_head(
        family=UNKNOWN_FAMILY, basis=TEST_BASIS, subject_id="spec_a", content_version="cv1"
    )

    assert confirmed is not None
    assert confirmed.content_version == "cv1"
    assert confirmed.confirmed_at == confirmed_at
    assert confirmed.materialised_at == materialised_at
    assert (
        confirmed.dependencies
        == materialisation_record(subject_id="spec_a", content_version="cv1").dependencies
    )
    head = LiveHeadStore(store=version_store).read_head("spec_a")
    assert head is not None
    assert head.confirmed_at == confirmed_at
    assert head.materialised_at == materialised_at


def test_confirm_head_against_a_different_content_version_returns_none_and_leaves_the_row(
    version_store,
):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    materialised_at = datetime(2026, 1, 1, tzinfo=UTC)
    store = LiveHeadStore(store=version_store, now=lambda: materialised_at)
    store.advance_head(
        family=UNKNOWN_FAMILY,
        basis=TEST_BASIS,
        materialisation=materialisation_record(subject_id="spec_a", content_version="cv1"),
        expected_materialisation_id=None,
    )
    before = store.read_head("spec_a")

    result = LiveHeadStore(
        store=version_store, now=lambda: datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    ).confirm_head(
        subject_id="spec_a", content_version="cv2", family=UNKNOWN_FAMILY, basis=TEST_BASIS
    )

    assert result is None
    after = store.read_head("spec_a")
    assert after == before


def test_confirm_head_on_an_unknown_subject_returns_none(version_store):
    store = LiveHeadStore(store=version_store)

    result = store.confirm_head(
        subject_id="spec_missing", content_version="cv1", family=UNKNOWN_FAMILY, basis=TEST_BASIS
    )

    assert result is None


def test_a_run_record_with_an_empty_build_refuses_at_construction():
    with pytest.raises(AssertionError):
        replace(
            materialisation_record(subject_id="spec_a", content_version="cv1"),
            materialiser_build="",
        )
