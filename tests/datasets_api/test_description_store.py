"""Tests for the description store, over the SQLite store double that runs its real SQL."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast, get_args

import pytest

from datasets_api.description_store import (
    DESCRIPTION_RECORD_INVALID_MARKER,
    DescriptionNotStored,
    LiveDescriptionStore,
    insert_description,
)
from description.facts import (
    SCHEMA_VERSION,
    Builds,
    ComputedFacts,
    DescriptionRecord,
    Evidence,
    InputVersion,
    LateField,
    SeriesFacts,
    StatusCounts,
)

from ._version_record_store import FakeVersionRecordDatasetStore

_OPS_ID = "ops_3c9e1f7a2b8d4e6f"
_OCV = "ocv_8d2a6f1c9e3b7a5d"
_INPUT = InputVersion(
    id="spec_91d7345eec81a029629938b3d0ffd775",
    content_version="cv_b785befb9f87471de96eb31cd7c56f5c",
)
_RECORDED = datetime(2026, 10, 11, 8, 0, tzinfo=UTC)
_ARRIVED = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)


def _evidence(title: str) -> Evidence:
    return Evidence(
        prepared="2026-10-11",
        title=title,
        sources=[],
        source_release=None,
        labels=[],
        membership=None,
        builds=Builds(origin_catalogue=None, code_index=None, coverage=None, method_status=None),
    )


def _record(*, title: str = "Gross ODA per capita", **overrides: Any) -> DescriptionRecord:
    """An `ops_` record whose `steps` and status counts are still `null`."""
    fields: dict[str, Any] = {
        "id": _OPS_ID,
        "content_version": _OCV,
        "schema_version": 1,
        "basis": "publish",
        "index_stamp": None,
        "variables": None,
        "computed": ComputedFacts(inputs=[_INPUT], composed=None),
        "series": SeriesFacts(
            series=3,
            rows=None,
            periods=None,
            span=None,
            grain=None,
            latest_period=None,
            series_reaching_latest=None,
            entities_by_role=None,
            by_source_unit=None,
            status=StatusCounts(estimated=None, provisional=None),
        ),
        "steps": None,
        "caveats": [],
        "evidence": _evidence(title),
        "late_fields": [],
    }
    return DescriptionRecord.model_validate(fields | overrides)


@pytest.fixture
def dataset_store() -> Iterator[FakeVersionRecordDatasetStore]:
    store = FakeVersionRecordDatasetStore()
    yield store
    store.close()


@pytest.fixture
def store(dataset_store: FakeVersionRecordDatasetStore) -> LiveDescriptionStore:
    return LiveDescriptionStore(store=dataset_store, now=lambda: _ARRIVED)


def _insert(dataset_store: FakeVersionRecordDatasetStore, record: DescriptionRecord) -> bool:
    with dataset_store.transaction() as tx:
        return insert_description(tx, record, recorded_at=_RECORDED)


def test_a_record_reads_back_as_written_and_an_absent_one_reads_none(
    dataset_store: FakeVersionRecordDatasetStore, store: LiveDescriptionStore
) -> None:
    record = _record()

    assert store.read_description(id=_OPS_ID, content_version=_OCV) is None
    assert _insert(dataset_store, record) is True
    assert store.read_description(id=_OPS_ID, content_version=_OCV) == record


def test_a_batched_read_returns_each_stored_record_by_version_and_leaves_out_absent_ones(
    dataset_store: FakeVersionRecordDatasetStore, store: LiveDescriptionStore
) -> None:
    first = _record()
    other = ("ops_" + "e" * 16, "ocv_" + "e" * 16)
    second = _record(id=other[0], content_version=other[1])
    absent = ("ops_" + "0" * 16, "ocv_" + "0" * 16)
    _insert(dataset_store, first)
    _insert(dataset_store, second)

    read = store.read_descriptions(versions=[other, absent, (_OPS_ID, _OCV)])

    assert read == {other: second, (_OPS_ID, _OCV): first}
    assert store.read_descriptions(versions=[]) == {}


def test_a_second_insert_leaves_the_first_record(
    dataset_store: FakeVersionRecordDatasetStore, store: LiveDescriptionStore
) -> None:
    first = _record(title="First", basis="mint")

    assert _insert(dataset_store, first) is True
    assert _insert(dataset_store, _record(title="Second", basis="backfill")) is False
    assert store.read_description(id=_OPS_ID, content_version=_OCV) == first


def test_a_row_that_fails_validation_reads_as_none_and_the_next_insert_replaces_it(
    dataset_store: FakeVersionRecordDatasetStore,
    store: LiveDescriptionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A damaged row is logged and read as no record, so a writer's record takes its place."""
    assert _insert(dataset_store, _record(title="First", basis="mint")) is True
    with dataset_store.transaction() as tx:
        tx.execute("UPDATE datasets.description_record SET record = %s", ('{"id": "damaged"}',))
    replacement = _record(title="Second", basis="backfill")

    with caplog.at_level(logging.ERROR):
        assert store.read_description(id=_OPS_ID, content_version=_OCV) is None
        assert _insert(dataset_store, replacement) is True

    assert store.read_description(id=_OPS_ID, content_version=_OCV) == replacement
    with dataset_store.transaction() as tx:
        row = tx.fetch_one("SELECT basis FROM datasets.description_record", ())
    assert row is not None
    assert row["basis"] == "backfill"
    invalid = [r for r in caplog.records if DESCRIPTION_RECORD_INVALID_MARKER in r.getMessage()]
    # The read logs it, and so does the insert that finds it in place.
    assert [r.levelno for r in invalid] == [logging.ERROR, logging.ERROR]
    assert all(f"content_version={_OCV}" in r.getMessage() for r in invalid)


def test_a_row_of_a_later_schema_reads_as_none_and_stands(
    dataset_store: FakeVersionRecordDatasetStore,
    store: LiveDescriptionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A later image's record is no record to this image, which neither logs it as damaged
    nor overwrites it, so a rollback keeps its basis and late fields."""
    later = {
        **_record(title="Later", basis="publish").model_dump(mode="json"),
        "schema_version": SCHEMA_VERSION + 1,
        "a_later_member": True,
    }
    with dataset_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.description_record "
            "(subject_id, content_version, schema_version, basis, record, recorded_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (_OPS_ID, _OCV, SCHEMA_VERSION + 1, "publish", json.dumps(later), _RECORDED),
        )

    with caplog.at_level(logging.INFO):
        assert store.read_description(id=_OPS_ID, content_version=_OCV) is None
        assert _insert(dataset_store, _record(title="Rewritten", basis="backfill")) is False

    with dataset_store.transaction() as tx:
        row = tx.fetch_one("SELECT schema_version, basis, record FROM datasets.description_record")
    assert row is not None
    assert (row["schema_version"], row["basis"], row["record"]) == (
        SCHEMA_VERSION + 1,
        "publish",
        later,
    )
    assert not [r for r in caplog.records if DESCRIPTION_RECORD_INVALID_MARKER in r.getMessage()]


# The digest of `DescriptionRecord.model_json_schema()` under each `SCHEMA_VERSION`.
_RECORD_SCHEMA_DIGESTS = {
    1: "0bb8f26a863e0a7ed2a4a5500a078dcb45e96133bf78700fae694df77d640b42",
}


def test_the_record_model_is_the_one_its_schema_version_names() -> None:
    """The store repairs only rows of this image's schema version or an earlier one, so a
    record model that changes under the same version would overwrite a later image's records
    after a rollback. A changed model either takes a new `SCHEMA_VERSION` or, while no
    released image has written the old one, a new digest here."""
    annotation = DescriptionRecord.model_fields["schema_version"].annotation
    digest = hashlib.sha256(
        json.dumps(DescriptionRecord.model_json_schema(), sort_keys=True).encode()
    ).hexdigest()

    assert get_args(annotation) == (SCHEMA_VERSION,)
    assert _RECORD_SCHEMA_DIGESTS.get(SCHEMA_VERSION) == digest


def test_insert_strips_nul_from_text_the_model_did_not_validate(
    dataset_store: FakeVersionRecordDatasetStore,
) -> None:
    # `model_copy` skips validation, so the title reaches the store with its U+0000.
    unvalidated = _evidence("x").model_copy(update={"title": "a\x00b"})
    record = _record().model_copy(update={"evidence": unvalidated})

    assert _insert(dataset_store, record) is True

    with dataset_store.transaction() as tx:
        row = tx.fetch_one("SELECT record FROM datasets.description_record", ())
    assert row is not None
    stored = cast(dict[str, Any], row["record"])
    assert stored["evidence"]["title"] == "ab"


def test_fill_sets_each_null_member_once_and_notes_its_arrival(
    dataset_store: FakeVersionRecordDatasetStore, store: LiveDescriptionStore
) -> None:
    _insert(dataset_store, _record())

    filled = store.fill_late_fields(
        id=_OPS_ID,
        content_version=_OCV,
        values={"steps": [], "series.status.estimated": 4},
        basis="publish",
    )

    assert filled == ["steps", "series.status.estimated"]
    stored = store.read_description(id=_OPS_ID, content_version=_OCV)
    assert stored is not None
    assert stored.steps == []
    assert stored.series.status.estimated == 4
    assert stored.series.status.provisional is None
    assert stored.late_fields == [
        LateField(field="steps", arrived=_ARRIVED, basis="publish"),
        LateField(field="series.status.estimated", arrived=_ARRIVED, basis="publish"),
    ]


def test_fill_never_changes_a_member_that_holds_a_value(
    dataset_store: FakeVersionRecordDatasetStore, store: LiveDescriptionStore
) -> None:
    _insert(dataset_store, _record())
    store.fill_late_fields(
        id=_OPS_ID, content_version=_OCV, values={"series.status.estimated": 4}, basis="publish"
    )
    before = store.read_description(id=_OPS_ID, content_version=_OCV)

    filled = store.fill_late_fields(
        id=_OPS_ID,
        content_version=_OCV,
        values={"series.status.estimated": 9, "evidence.title": "Other"},
        basis="backfill",
    )

    assert filled == []
    assert store.read_description(id=_OPS_ID, content_version=_OCV) == before


_ROOT = InputVersion(id="spec_" + "a7" * 16, content_version="cv_" + "b8" * 16)


@pytest.mark.parametrize(
    ("values", "inputs"),
    [
        pytest.param({"steps": []}, [_ROOT], id="set-by-the-write-that-fills-steps"),
        pytest.param(
            {"evidence.title": "Other"}, [_INPUT], id="untouched-by-a-write-filling-nothing"
        ),
    ],
)
def test_a_derived_member_follows_the_fill_it_derives_from(
    dataset_store: FakeVersionRecordDatasetStore,
    store: LiveDescriptionStore,
    values: dict[str, Any],
    inputs: list[InputVersion],
) -> None:
    _insert(dataset_store, _record())

    filled = store.fill_late_fields(
        id=_OPS_ID,
        content_version=_OCV,
        values=values,
        derived={"computed.inputs": [_ROOT.model_dump(mode="json")]},
        basis="publish",
    )

    stored = store.read_description(id=_OPS_ID, content_version=_OCV)
    assert stored is not None
    assert stored.computed is not None
    assert stored.computed.inputs == inputs
    assert filled == ([] if inputs == [_INPUT] else ["steps", "computed.inputs"])


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"steps": [], "series.nonesuch": 1}, id="unknown_member"),
        pytest.param({"steps": [], "series.status.estimated": "many"}, id="invalid_value"),
    ],
)
def test_a_fill_that_fails_writes_nothing(
    dataset_store: FakeVersionRecordDatasetStore,
    store: LiveDescriptionStore,
    values: dict[str, Any],
) -> None:
    record = _record()
    _insert(dataset_store, record)

    with pytest.raises(ValueError):
        store.fill_late_fields(id=_OPS_ID, content_version=_OCV, values=values, basis="publish")

    assert store.read_description(id=_OPS_ID, content_version=_OCV) == record


def test_fill_without_a_record_raises(store: LiveDescriptionStore) -> None:
    with pytest.raises(DescriptionNotStored):
        store.fill_late_fields(
            id=_OPS_ID, content_version=_OCV, values={"steps": []}, basis="publish"
        )
