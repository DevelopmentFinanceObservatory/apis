"""Copying run records to the eval runs bucket with create permission alone."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from dfo_evals.records import Run, write_run
from google.api_core.exceptions import PreconditionFailed

from search.eval.record_upload import upload_runs
from search.eval.run_record import record_manifest

STARTED_AT = datetime(2026, 11, 26, 4, 17, tzinfo=UTC)


class FakeBlob:
    """An object handle that records each upload, refusing names the bucket holds."""

    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self.bucket = bucket
        self.name = name

    def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
        """Create the object, or refuse when it exists, as a generation-0 precondition does."""
        assert if_generation_match == 0
        if self.name in self.bucket.objects:
            raise PreconditionFailed("object exists")
        self.bucket.objects[self.name] = Path(filename).read_text()
        self.bucket.uploads.append(self.name)


class FakeBucket:
    """A bucket in memory that hands out ``FakeBlob`` handles without reading anything."""

    def __init__(self, objects: dict[str, str] | None = None) -> None:
        self.objects = dict(objects or {})
        self.uploads: list[str] = []

    def blob(self, blob_name: str) -> FakeBlob:
        """A handle on ``blob_name``."""
        return FakeBlob(self, blob_name)


def write_record(root: Path, run_id: str) -> None:
    """Write an empty run record ``run_id`` under ``root``."""
    manifest = record_manifest(
        run_id=run_id,
        started_at=STARTED_AT,
        finished_at=STARTED_AT,
        revision="abc",
        environment={},
        dataset_digest="sha256:d",
    )
    write_run(root=root, run=Run(manifest=manifest, results=()))


def test_each_run_goes_up_results_first_then_manifest(tmp_path):
    write_record(tmp_path, "nightly-1")
    bucket = FakeBucket()

    created = upload_runs(bucket=bucket, root=tmp_path)

    assert (
        created
        == bucket.uploads
        == [
            "apis-search/nightly-1/results.jsonl",
            "apis-search/nightly-1/manifest.json",
        ]
    )
    assert '"run_id": "nightly-1"' in bucket.objects["apis-search/nightly-1/manifest.json"]


def test_an_object_the_bucket_holds_is_left_and_the_rest_go_up(tmp_path):
    write_record(tmp_path, "nightly-1")
    bucket = FakeBucket({"apis-search/nightly-1/results.jsonl": "earlier"})

    created = upload_runs(bucket=bucket, root=tmp_path)

    assert created == ["apis-search/nightly-1/manifest.json"]
    assert bucket.objects["apis-search/nightly-1/results.jsonl"] == "earlier"


def test_a_root_without_records_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        upload_runs(bucket=FakeBucket(), root=tmp_path)
