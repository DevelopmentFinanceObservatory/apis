"""Pin spill-threshold uploads, staging location, listing, and abort behaviour."""

from __future__ import annotations

import hashlib

import pytest
from google.api_core.exceptions import ServiceUnavailable

from artifact_store.errors import ArtifactWriteError
from artifact_store.store import SPILL_THRESHOLD_BYTES, LiveArtifactStore, Tier

from . import conftest as conftest_module
from .conftest import DiscardingFakeGCSBlobWriter, FakeGCSClient, RecordingArtifactInventory

_BUCKETS = {
    Tier.working: "dev-fin-observatory-datasets-working",
    Tier.pinned: "dev-fin-observatory-datasets-pinned",
}


def _store(client: FakeGCSClient, inventory: RecordingArtifactInventory) -> LiveArtifactStore:
    return LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client)


def test_open_put_under_the_spill_threshold_uploads_in_one_request():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"small artifact, well under the spill threshold"

    with store.open_put(tier=Tier.working) as sink:
        sink.write(data[:10])
        sink.write(data[10:])
    result = sink.result()

    expected_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert result.digest == expected_digest
    assert result.deduplicated is False
    bucket = client.buckets[_BUCKETS[Tier.working]]
    assert bucket.upload_log == [expected_digest]
    assert bucket.objects[expected_digest] == data
    assert not any(name.startswith("staging/") for name in bucket.objects)


def test_open_put_at_the_spill_threshold_stages_then_copies_to_the_digest():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"x" * SPILL_THRESHOLD_BYTES + b"more past the boundary"

    with store.open_put(tier=Tier.working) as sink:
        sink.write(data)
    result = sink.result()

    expected_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert result.digest == expected_digest
    assert result.deduplicated is False
    assert result.size_bytes == len(data)
    bucket = client.buckets[_BUCKETS[Tier.working]]
    # Staging objects remain until the bucket lifecycle rule removes them; GCS has no delete grant.
    staging_names = [name for name in bucket.objects if name.startswith("staging/")]
    assert len(staging_names) == 1
    assert bucket.objects[staging_names[0]] == data
    assert bucket.objects[expected_digest] == data


def test_open_put_multi_chunk_write_streams_instead_of_buffering_whole_artifact(monkeypatch):
    """A 3-4x threshold write reaches the staging writer in caller-sized chunks.

    The usual fake buffers until close, so this test uses a discarding writer
    that records call sizes and peak residency like GCS `BlobWriter`.
    """
    monkeypatch.setattr(conftest_module, "FakeGCSBlobWriter", DiscardingFakeGCSBlobWriter)
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    chunk = b"0123456789abcdef" * (1024 * 1024 // 16)  # 1 MiB, deterministic content
    n_chunks = SPILL_THRESHOLD_BYTES // len(chunk) * 4  # ~4x the spill threshold
    data = chunk * n_chunks

    with store.open_put(tier=Tier.working) as sink:
        for _ in range(n_chunks):
            sink.write(chunk)
    result = sink.result()

    expected_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert result.digest == expected_digest
    bucket = client.buckets[_BUCKETS[Tier.working]]
    assert bucket.objects[expected_digest] == data

    staging_writer = bucket.last_writer
    assert isinstance(staging_writer, DiscardingFakeGCSBlobWriter)
    # The first call flushes pre-spill bytes; each later call forwards one write.
    assert len(staging_writer.call_sizes) > 1
    assert staging_writer.call_sizes[0] <= SPILL_THRESHOLD_BYTES
    assert all(size <= len(chunk) for size in staging_writer.call_sizes[1:])
    assert staging_writer.peak_resident_bytes < len(data) // 2


def test_open_put_stages_in_the_working_bucket_even_when_writing_to_pinned():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"y" * (SPILL_THRESHOLD_BYTES + 1)

    with store.open_put(tier=Tier.pinned) as sink:
        sink.write(data)
    result = sink.result()

    expected_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert result.tier == Tier.pinned
    assert result.bucket == _BUCKETS[Tier.pinned]
    working_bucket = client.buckets[_BUCKETS[Tier.working]]
    pinned_bucket = client.buckets[_BUCKETS[Tier.pinned]]
    # Staging uses the working bucket; pinned receives only the final copy.
    assert any(name.startswith("staging/") for name in working_bucket.objects)
    assert not any(name.startswith("staging/") for name in pinned_bucket.objects)
    assert pinned_bucket.objects[expected_digest] == data


def test_open_put_staged_write_dedups_on_a_concurrent_digest():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"z" * (SPILL_THRESHOLD_BYTES + 1)

    with store.open_put(tier=Tier.working) as first_sink:
        first_sink.write(data)
    first = first_sink.result()

    with store.open_put(tier=Tier.working) as second_sink:
        second_sink.write(data)
    second = second_sink.result()

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.digest == first.digest
    bucket = client.buckets[_BUCKETS[Tier.working]]
    staging_names = [name for name in bucket.objects if name.startswith("staging/")]
    assert len(staging_names) == 2
    assert bucket.upload_log.count(first.digest) == 1


def test_open_put_result_raises_before_the_block_commits():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    with store.open_put(tier=Tier.working) as sink:
        sink.write(b"not committed yet")
        with pytest.raises(RuntimeError):
            sink.result()


def test_open_put_aborts_and_commits_nothing_on_a_small_write_exception():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom), store.open_put(tier=Tier.working) as sink:
        sink.write(b"buffered but never finished")
        raise _Boom("writer blew up before finishing")

    # This write stays below the spill threshold, so abort makes no GCS calls.
    assert not any(bucket.objects for bucket in client.buckets.values())
    assert inventory.calls == []


def test_open_put_aborts_a_staged_upload_without_committing_the_staging_object():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom), store.open_put(tier=Tier.working) as sink:
        sink.write(b"w" * (SPILL_THRESHOLD_BYTES + 1))
        raise _Boom("writer blew up after crossing the spill threshold")

    bucket = client.buckets[_BUCKETS[Tier.working]]
    # Termination leaves no reachable object to clean up.
    assert bucket.objects == {}
    assert bucket.upload_log == []
    assert inventory.calls == []


def test_open_put_staged_close_failure_raises_artifact_write_error():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    bucket_name = _BUCKETS[Tier.working]
    client.bucket(bucket_name).raise_on_next_upload = ServiceUnavailable("simulated outage")

    with pytest.raises(ArtifactWriteError), store.open_put(tier=Tier.working) as sink:
        sink.write(b"v" * (SPILL_THRESHOLD_BYTES + 1))

    assert inventory.calls == []


def test_list_excludes_staging_objects_even_when_present_in_the_bucket():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    written = store.put(b"a real artifact", tier=Tier.working)
    # Simulate a staging object left behind by a prior streamed write --
    # this must never show up in a reconciliation-facing list() result.
    bucket = client.buckets[_BUCKETS[Tier.working]]
    bucket.objects["staging/leftover-upload"] = b"partial bytes"

    result = store.list(tier=Tier.working)

    assert result == frozenset({written.digest})
    assert "staging/leftover-upload" not in result
