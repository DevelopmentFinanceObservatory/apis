"""Pins `gs://` reads, conditional writes, and deletion against a fake GCS client.

`google.api_core.exceptions.NotFound` once passed through `_read_gs`, while callers in `loader.py`
and `job.py` catch only `FileNotFoundError`, so a first publish with no `current.json` crashed.
Each test pairs a `gs://` condition with the assertion `test_uri.py` makes for its `file://` twin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed

from artifact_publication.uri import (
    delete_uri_prefix,
    download_uri_to_path,
    list_uri_object_generations,
    list_uri_object_times,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes,
    write_uri_bytes_if_unchanged,
)
from dc_sdmx.results import SdmxDataResult
from series_coverage import job
from series_coverage.schema import FORMAT_ROOT
from tests.series_coverage.support import code_index_for
from tests.series_coverage.test_job import (
    _CONNECTION,
    _FACET_ROW,
    _ROW_V1,
    FakeBigQuery,
    FakeSdmx,
    _obs,
    import_key_responses,
    year_stats,
)

_BUCKET = "test-coverage-bucket"


@dataclass
class _StoredObject:
    data: bytes
    generation: int
    time_created: datetime


class _FakeBlob:
    """Stands in for `google.cloud.storage.Blob`. `_objects` is the
    bucket-level dict `_FakeBucket`/`_FakeStorageClient` all share, so a
    blob handle always reads and writes the same live state a fresh
    `.blob()` call for the same name would. `generation` is what the real
    client exposes: `None` on a fresh handle, set by a download from that
    response's generation header, and set on a listed blob from the
    listing.
    """

    def __init__(
        self, objects: dict[str, _StoredObject], name: str, generation: int | None = None
    ) -> None:
        self._objects = objects
        self.name = name
        self.generation = generation

    @property
    def time_created(self) -> datetime:
        return self._objects[self.name].time_created

    def download_as_bytes(self) -> bytes:
        try:
            stored = self._objects[self.name]
        except KeyError:
            raise NotFound(f"gs://{_BUCKET}/{self.name} does not exist") from None
        self.generation = stored.generation
        return stored.data

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.download_as_bytes())

    def upload_from_string(self, data: bytes, if_generation_match: int | None = None) -> None:
        current = self._objects.get(self.name)
        if if_generation_match is not None:
            current_generation = current.generation if current is not None else 0
            if current_generation != if_generation_match:
                raise PreconditionFailed(f"generation mismatch for {self.name}")
        next_generation = (current.generation if current is not None else 0) + 1
        self._objects[self.name] = _StoredObject(
            data=data, generation=next_generation, time_created=datetime.now(UTC)
        )

    def delete(self) -> None:
        try:
            del self._objects[self.name]
        except KeyError:
            raise NotFound(f"gs://{_BUCKET}/{self.name} does not exist") from None


class _FakeBucket:
    def __init__(self, objects: dict[str, _StoredObject]) -> None:
        self._objects = objects

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._objects, name)


class _FakeStorageClient:
    def __init__(self, buckets: dict[str, dict[str, _StoredObject]]) -> None:
        self._buckets = buckets

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._buckets.setdefault(name, {}))

    def list_blobs(self, bucket_name: str, *, prefix: str = "") -> list[_FakeBlob]:
        objects = self._buckets.setdefault(bucket_name, {})
        return [
            _FakeBlob(objects, name, objects[name].generation)
            for name in objects
            if name.startswith(prefix)
        ]


@pytest.fixture
def gs_bucket(monkeypatch) -> dict[str, _StoredObject]:
    """Monkeypatches `google.cloud.storage.Client` so every `storage
    .Client()` call inside `artifact_publication.uri`'s `gs://` helpers reads and
    writes the same in-memory bucket, real `NotFound`/`PreconditionFailed`
    exceptions included. Returns `_BUCKET`'s own object store, keyed by
    object name, for a test that needs to inspect or mutate it directly
    (simulating a concurrent write or delete).
    """
    buckets: dict[str, dict[str, _StoredObject]] = {}
    monkeypatch.setattr("google.cloud.storage.Client", lambda: _FakeStorageClient(buckets))
    return buckets.setdefault(_BUCKET, {})


def _uri(*parts: str) -> str:
    joined = "/".join(part.strip("/") for part in parts)
    return f"gs://{_BUCKET}/{joined}"


# --- read_uri_bytes / write_uri_bytes ---------------------------------------


def test_read_uri_bytes_raises_file_not_found_for_a_missing_gs_object(gs_bucket):
    with pytest.raises(FileNotFoundError):
        read_uri_bytes(_uri("current.json"))


def test_read_uri_bytes_returns_what_was_written(gs_bucket):
    uri = _uri("current.json")
    write_uri_bytes(uri, b"hello")
    assert read_uri_bytes(uri) == b"hello"


# --- read_uri_bytes_and_generation / write_uri_bytes_if_unchanged ----------


def _generation(uri: str) -> str | None:
    found = read_uri_bytes_and_generation(uri)
    return None if found is None else found[1]


def test_read_uri_bytes_and_generation_is_none_for_a_missing_gs_object(gs_bucket):
    assert read_uri_bytes_and_generation(_uri("current.json")) is None


def test_read_uri_bytes_and_generation_pairs_bytes_with_their_own_generation(gs_bucket):
    uri = _uri("current.json")
    write_uri_bytes(uri, b"one")
    found = read_uri_bytes_and_generation(uri)
    assert found is not None
    data, first = found
    assert data == b"one"
    assert first == str(gs_bucket["current.json"].generation)

    write_uri_bytes(uri, b"two")
    assert read_uri_bytes_and_generation(uri) == (b"two", str(gs_bucket["current.json"].generation))
    assert _generation(uri) != first


def test_write_uri_bytes_if_unchanged_requires_absence_when_expected_generation_is_none(
    gs_bucket,
):
    uri = _uri("current.json")
    assert write_uri_bytes_if_unchanged(uri, b"first", expected_generation=None) is True
    assert read_uri_bytes(uri) == b"first"

    # A second write also claiming "nothing there yet" loses: the object
    # now exists -- the same `PreconditionFailed` GCS returns for a
    # `file://` mismatch, caught here and turned into `False`.
    assert write_uri_bytes_if_unchanged(uri, b"second", expected_generation=None) is False
    assert read_uri_bytes(uri) == b"first"


def test_write_uri_bytes_if_unchanged_refuses_on_a_stale_generation(gs_bucket):
    uri = _uri("current.json")
    write_uri_bytes(uri, b"first")
    stale_generation = _generation(uri)

    write_uri_bytes(uri, b"concurrent")  # someone else publishes in between

    refused = write_uri_bytes_if_unchanged(uri, b"mine", expected_generation=stale_generation)
    assert refused is False
    assert read_uri_bytes(uri) == b"concurrent"


def test_write_uri_bytes_if_unchanged_succeeds_when_the_generation_still_matches(gs_bucket):
    uri = _uri("current.json")
    write_uri_bytes(uri, b"first")
    current_generation = _generation(uri)

    published = write_uri_bytes_if_unchanged(uri, b"second", expected_generation=current_generation)

    assert published is True
    assert read_uri_bytes(uri) == b"second"


def _make_first_attempt_commit_then_412(monkeypatch) -> None:
    """Patches `_FakeBlob.upload_from_string` so the single call this
    module's code makes behaves the way `google.cloud.storage`'s own
    default retry does when a conditional upload's first attempt commits
    but its response is lost: the write lands, and `PreconditionFailed`
    still reaches the caller, because the library's own retry resent the
    request and met the object its first attempt just created.
    """
    real_upload = _FakeBlob.upload_from_string

    def committed_but_412(self, data: bytes, if_generation_match: int | None = None) -> None:
        real_upload(self, data, if_generation_match=if_generation_match)
        raise PreconditionFailed(f"lost response, retry met {self.name} already committed")

    monkeypatch.setattr(_FakeBlob, "upload_from_string", committed_but_412)


def test_write_uri_bytes_if_unchanged_reports_success_for_a_create_only_retry_that_already_landed(
    gs_bucket, monkeypatch
):
    uri = _uri("current.json")
    _make_first_attempt_commit_then_412(monkeypatch)

    published = write_uri_bytes_if_unchanged(uri, b"mine", expected_generation=None)

    assert published is True
    assert read_uri_bytes(uri) == b"mine"


def test_write_uri_bytes_if_unchanged_reports_success_for_a_generation_matched_retry_that_already_landed(
    gs_bucket, monkeypatch
):
    uri = _uri("current.json")
    write_uri_bytes(uri, b"first")
    current_generation = _generation(uri)
    _make_first_attempt_commit_then_412(monkeypatch)

    published = write_uri_bytes_if_unchanged(uri, b"second", expected_generation=current_generation)

    assert published is True
    assert read_uri_bytes(uri) == b"second"


def test_write_uri_bytes_if_unchanged_still_reports_failure_when_a_different_object_won_the_race(
    gs_bucket, monkeypatch
):
    """The re-read `_write_gs_if_unchanged` does on a `PreconditionFailed`
    must not turn every precondition failure into a success: an object a
    different writer landed, with different bytes, still refuses.
    """
    uri = _uri("current.json")
    write_uri_bytes(uri, b"first")
    current_generation = _generation(uri)
    write_uri_bytes(uri, b"concurrent")  # a different writer wins

    published = write_uri_bytes_if_unchanged(uri, b"mine", expected_generation=current_generation)

    assert published is False
    assert read_uri_bytes(uri) == b"concurrent"


# --- list_uri_object_times ---------------------------------------------------


def test_list_uri_object_times_is_empty_for_a_base_that_does_not_exist(gs_bucket):
    assert list_uri_object_times(_uri("format-1")) == {}
    assert list_uri_object_generations(_uri("format-1")) == {}


def test_list_uri_object_times_reads_every_object_under_the_prefix_only(gs_bucket):
    write_uri_bytes(_uri("format-1", "current.json"), b"{}")
    write_uri_bytes(_uri("format-1", "versions", "b1", "a.txt"), b"x")
    old_time = datetime.now(UTC) - timedelta(seconds=1000)
    gs_bucket["format-1/versions/b1/a.txt"].time_created = old_time
    write_uri_bytes(_uri("format-0", "current.json"), b"{}")

    times = list_uri_object_times(_uri("format-1"))

    assert set(times) == {"current.json", "versions/b1/a.txt"}
    assert abs(times["versions/b1/a.txt"] - old_time.timestamp()) < 2

    listed = list_uri_object_generations(_uri("format-1"))

    assert {name: written for name, (written, _) in listed.items()} == times
    found = read_uri_bytes_and_generation(_uri("format-1", "versions", "b1", "a.txt"))
    assert found is not None
    assert listed["versions/b1/a.txt"][1] == found[1]


def test_download_uri_to_path_streams_the_object_and_refuses_a_missing_one(gs_bucket, tmp_path):
    write_uri_bytes(_uri("parts", "a.arrow"), b"part")

    download_uri_to_path(_uri("parts", "a.arrow"), tmp_path / "local")

    assert (tmp_path / "local").read_bytes() == b"part"
    with pytest.raises(FileNotFoundError):
        download_uri_to_path(_uri("parts", "gone"), tmp_path / "local")


# --- delete_uri_prefix -------------------------------------------------------


def test_delete_uri_prefix_is_a_no_op_for_a_prefix_that_does_not_exist(gs_bucket):
    delete_uri_prefix(_uri("versions", "gone"))  # must not raise


def test_delete_uri_prefix_removes_every_object_under_the_prefix(gs_bucket):
    write_uri_bytes(_uri("versions", "s1", "a.txt"), b"x")
    write_uri_bytes(_uri("versions", "s1", "b.txt"), b"y")
    write_uri_bytes(_uri("versions", "s2", "c.txt"), b"z")

    delete_uri_prefix(_uri("versions", "s1"))

    assert sorted(gs_bucket) == ["versions/s2/c.txt"]


def test_delete_uri_prefix_tolerates_a_blob_already_gone_by_delete_time(gs_bucket, monkeypatch):
    """Two prune runs racing the same stamp: this run's own `list_blobs`
    already returned `a.txt`'s blob handle when a concurrent run deletes it
    first -- `delete_uri_prefix`'s own docstring guarantee ("two prune runs
    racing the same stamp is not an error") must hold even though `gs://`
    lists then deletes each blob individually, a wider race window than
    `file://`'s single `path.exists()` check plus `shutil.rmtree`.
    """
    write_uri_bytes(_uri("versions", "s1", "a.txt"), b"x")
    write_uri_bytes(_uri("versions", "s1", "b.txt"), b"y")

    real_list_blobs = _FakeStorageClient.list_blobs

    def racing_list_blobs(self, bucket_name, *, prefix=""):
        blobs = real_list_blobs(self, bucket_name, prefix=prefix)
        gs_bucket.pop("versions/s1/a.txt", None)  # a concurrent prune wins first
        return blobs

    monkeypatch.setattr(_FakeStorageClient, "list_blobs", racing_list_blobs)

    delete_uri_prefix(_uri("versions", "s1"))  # must not raise NotFound

    assert "versions/s1/b.txt" not in gs_bucket


# --- job.run's first-ever gs:// publish --------------------------------------


def test_job_run_publishes_a_first_ever_gs_artifact(gs_bucket, capsys):
    """The bug this file exists for, exercised end to end: `job.run`
    against a `gs://` base with nothing published yet reads the missing
    pointer as absence, builds, and publishes into its format root. A
    `NotFound` reaching the job's pointer read uncaught is what a real
    first build hit on dev (Cloud Run execution `dfo-dev-coverage-f58v5`).
    """
    responses = {
        **import_key_responses(),
        "AS max_year": [year_stats(max_year=2022)],
        "GROUP BY variable, facet_id, donor, recipient": [dict(_ROW_V1)],
        "object_id AS role": [
            {"variable": "V1", "role": "financingProvider"},
            {"variable": "V1", "role": "financingRecipient"},
        ],
        "GROUP BY facet_id, provenance, unit, measurement_method": [_FACET_ROW],
    }
    bq = FakeBigQuery(responses)
    sdmx = FakeSdmx(
        data_responder=lambda *a, **k: SdmxDataResult(
            roles=("financingProvider", "financingRecipient"), rows=(_obs("V1", {}),)
        ),
        batches_by_group={("financingProvider", "financingRecipient"): [("V1", 3)]},
    )
    settings = job.JobSettings(
        artifact_uri=f"gs://{_BUCKET}",
        bq_connection=_CONNECTION,
        bq_project="proj",
        sdmx_base_url="https://mixer/core/api",
        sdmx_token=None,
        sdmx_timeout_seconds=90.0,
        cycle_seconds=900.0,
    )

    job.run(settings, bq=bq, sdmx=sdmx, index=code_index_for({}))

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events[-2:]] == [
        "coverage_job_published",
        "coverage_index_compatible",
    ]
    build_id = events[-2]["build_id"]
    assert sorted(gs_bucket) == [
        f"{FORMAT_ROOT}/compatible/test-index.json",
        f"{FORMAT_ROOT}/coverage_state.json",
        f"{FORMAT_ROOT}/current.json",
        f"{FORMAT_ROOT}/versions/{build_id}/coverage.parquet",
        f"{FORMAT_ROOT}/versions/{build_id}/facets.json",
        f"{FORMAT_ROOT}/versions/{build_id}/manifest.json",
        f"{FORMAT_ROOT}/versions/{build_id}/role_names.json",
    ]
