"""Pin `LiveArtifactStore` writes, reads, copies, signed URLs, and cache behaviour over fake GCS."""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest
from google.api_core.exceptions import ServiceUnavailable

from artifact_store.errors import (
    ArtifactIntegrityError,
    ArtifactLocationDriftError,
    ArtifactNotFoundError,
    ArtifactRecordConflictError,
    ArtifactWriteError,
)
from artifact_store.signing import SIGNED_URL_REUSE_WINDOW
from artifact_store.store import LiveArtifactStore, Tier

from .conftest import FakeGCSClient, RecordingArtifactInventory

_BUCKETS = {
    Tier.working: "dev-fin-observatory-datasets-working",
    Tier.pinned: "dev-fin-observatory-datasets-pinned",
}


def _store(client: FakeGCSClient, inventory: RecordingArtifactInventory) -> LiveArtifactStore:
    return LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client)


def test_put_stores_the_object_under_its_own_digest():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    data = b"hello artifact"
    result = store.put(data, tier=Tier.working)

    expected_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert result.digest == expected_digest
    assert result.object_name == expected_digest
    assert result.bucket == "dev-fin-observatory-datasets-working"
    assert result.size_bytes == len(data)
    assert result.deduplicated is False
    assert client.buckets["dev-fin-observatory-datasets-working"].objects[expected_digest] == data


def test_digest_recomputed_from_the_exact_stored_bytes_matches():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    data = b"digest me exactly"
    result = store.put(data, tier=Tier.pinned)

    stored_bytes = client.buckets["dev-fin-observatory-datasets-pinned"].objects[result.object_name]
    recomputed = "sha256:" + hashlib.sha256(stored_bytes).hexdigest()
    assert recomputed == result.digest


def test_concurrent_duplicate_write_converges_to_one_object_deduplicated_true():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    data = b"raced by two writers"
    first = store.put(data, tier=Tier.working)
    second = store.put(data, tier=Tier.working)

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert first.digest == second.digest
    bucket = client.buckets["dev-fin-observatory-datasets-working"]
    assert bucket.upload_log == [first.object_name]  # only one upload actually landed
    assert bucket.objects[first.object_name] == data


def test_put_sets_cache_control_for_a_year_immutable():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    data = b"cache me forever"
    result = store.put(data, tier=Tier.working)

    bucket = client.buckets["dev-fin-observatory-datasets-working"]
    assert bucket.cache_control[result.object_name] == "private, max-age=31536000, immutable"


def test_deduplicated_put_leaves_the_stored_objects_cache_control_untouched():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"already stored elsewhere"
    first = store.put(data, tier=Tier.working)
    bucket = client.buckets["dev-fin-observatory-datasets-working"]
    # Simulate cache metadata written by another client.
    bucket.cache_control[first.object_name] = "public, max-age=0"

    second = store.put(data, tier=Tier.working)

    assert second.deduplicated is True
    assert bucket.cache_control[first.object_name] == "public, max-age=0"


def test_both_writers_of_a_duplicate_still_record_inventory():
    # Both writers record the same digest; the inventory upsert is idempotent.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    data = b"both writers succeed"
    store.put(data, tier=Tier.working)
    store.put(data, tier=Tier.working)

    assert len(inventory.calls) == 2
    assert inventory.calls[0]["digest"] == inventory.calls[1]["digest"]


def test_write_failure_raises_artifact_write_error_without_the_inventory_row():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    bucket_name = _BUCKETS[Tier.working]
    client.bucket(bucket_name).raise_on_next_upload = ServiceUnavailable("simulated outage")

    data = b"never lands"
    with pytest.raises(ArtifactWriteError) as excinfo:
        store.put(data, tier=Tier.working)

    expected_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert excinfo.value.object_name == expected_digest
    assert excinfo.value.digest == expected_digest
    assert inventory.calls == []  # ordering: no row for bytes that never landed


def test_write_error_never_chains_the_driver_exception():
    # Raising after the handler exits leaves `__context__` unset for middleware re-raises.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    bucket_name = _BUCKETS[Tier.working]
    client.bucket(bucket_name).raise_on_next_upload = ServiceUnavailable(
        "simulated outage naming a secret detail"
    )

    with pytest.raises(ArtifactWriteError) as excinfo:
        store.put(b"chain check", tier=Tier.working)

    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert "secret detail" not in str(excinfo.value)


def test_get_returns_the_exact_bytes_that_were_put():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"round trip me"
    written = store.put(data, tier=Tier.working)

    result = store.get(artifact_digest=written.digest, tier=Tier.working)

    assert result == data


def test_get_raises_not_found_for_a_digest_never_written():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    with pytest.raises(ArtifactNotFoundError) as excinfo:
        store.get(artifact_digest="sha256:neverwritten", tier=Tier.working)

    assert excinfo.value.digest == "sha256:neverwritten"
    assert excinfo.value.tier == Tier.working


def test_get_reads_the_tiers_configured_bucket_directly_not_the_inventory():
    # `get` reads the configured bucket directly, even for objects absent from inventory.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    bucket_name = _BUCKETS[Tier.working]
    data = b"placed without put()"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    client.bucket(bucket_name).objects[digest] = data

    result = store.get(artifact_digest=digest, tier=Tier.working)

    assert result == data
    assert inventory.calls == []


def test_get_raises_integrity_error_when_the_stored_bytes_do_not_hash_to_the_digest():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    bucket_name = _BUCKETS[Tier.working]
    # Store bytes whose hash differs from the object name.
    client.bucket(bucket_name).objects["sha256:claimeddigest"] = b"actual bytes differ"

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.get(artifact_digest="sha256:claimeddigest", tier=Tier.working)

    assert excinfo.value.digest == "sha256:claimeddigest"
    assert excinfo.value.tier == Tier.working
    assert excinfo.value.computed_digest == (
        "sha256:" + hashlib.sha256(b"actual bytes differ").hexdigest()
    )


def test_constructor_rejects_a_buckets_mapping_missing_a_tier():
    inventory = RecordingArtifactInventory()

    with pytest.raises(ValueError, match="pinned"):
        LiveArtifactStore(
            buckets={Tier.working: "dev-fin-observatory-datasets-working"},
            inventory=inventory,
            client=FakeGCSClient(),
        )


def test_signed_get_url_raises_not_found_for_a_digest_never_written():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    with pytest.raises(ArtifactNotFoundError) as excinfo:
        store.signed_get_url(
            "sha256:neverwritten",
            tier=Tier.working,
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="test.csv"',
        )

    assert excinfo.value.digest == "sha256:neverwritten"
    assert excinfo.value.tier == Tier.working


def test_signed_get_url_proceeds_to_sign_once_the_digest_is_found(monkeypatch):
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    result = store.put(b"already stored", tier=Tier.working)
    monkeypatch.setattr(
        "artifact_store.store.sign_get_url",
        lambda **kwargs: "https://storage.googleapis.com/fake-signed-url",
    )

    url = store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    assert url == "https://storage.googleapis.com/fake-signed-url"


def test_signed_get_url_signs_the_stored_bucket_and_object_name_when_they_match_config(
    monkeypatch,
):
    # Sign the matching bucket and object name recorded in inventory.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    result = store.put(b"matches config", tier=Tier.working)
    seen_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        "artifact_store.store.sign_get_url",
        lambda **kwargs: seen_kwargs.update(kwargs) or "https://storage.googleapis.com/signed",
    )

    store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    assert seen_kwargs["bucket"] == _BUCKETS[Tier.working]
    assert seen_kwargs["object_name"] == result.digest


def test_signed_get_url_refuses_a_cross_tier_stored_bucket():
    # A tier's inventory row must name its configured bucket.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    inventory.record(
        digest="sha256:drifted",
        tier=Tier.working,
        bucket=_BUCKETS[Tier.pinned],
        object_name="sha256:drifted",
        size_bytes=3,
    )

    with pytest.raises(ArtifactLocationDriftError) as excinfo:
        store.signed_get_url(
            "sha256:drifted",
            tier=Tier.working,
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="test.csv"',
        )

    assert excinfo.value.digest == "sha256:drifted"
    assert excinfo.value.tier == Tier.working
    assert excinfo.value.stored_bucket == _BUCKETS[Tier.pinned]
    assert excinfo.value.stored_object_name == "sha256:drifted"


def test_signed_get_url_refuses_a_stored_bucket_not_among_the_configured_buckets():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    inventory.record(
        digest="sha256:poisoned",
        tier=Tier.working,
        bucket="some-bucket-no-longer-configured",
        object_name="sha256:poisoned",
        size_bytes=3,
    )

    with pytest.raises(ArtifactLocationDriftError) as excinfo:
        store.signed_get_url(
            "sha256:poisoned",
            tier=Tier.working,
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="test.csv"',
        )

    assert excinfo.value.digest == "sha256:poisoned"
    assert excinfo.value.tier == Tier.working
    assert excinfo.value.stored_bucket == "some-bucket-no-longer-configured"
    assert excinfo.value.stored_object_name == "sha256:poisoned"


def test_signed_get_url_refuses_an_object_name_that_does_not_match_the_digest():
    # A signed URL for digest X must target object X.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    inventory.record(
        digest="sha256:realdigest",
        tier=Tier.working,
        bucket=_BUCKETS[Tier.working],
        object_name="sha256:someotherobject",
        size_bytes=3,
    )

    with pytest.raises(ArtifactLocationDriftError) as excinfo:
        store.signed_get_url(
            "sha256:realdigest",
            tier=Tier.working,
            ttl=timedelta(minutes=5),
            response_type="text/csv",
            response_disposition='attachment; filename="test.csv"',
        )

    assert excinfo.value.digest == "sha256:realdigest"
    assert excinfo.value.stored_bucket == _BUCKETS[Tier.working]
    assert excinfo.value.stored_object_name == "sha256:someotherobject"


def test_put_raises_artifact_record_conflict_error_after_bucket_reconfiguration():
    # A bucket rotation uploads to the new bucket before inventory rejects the conflicting location.
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store_before = _store(client, inventory)
    data = b"bytes that outlive a bucket rotation"
    first = store_before.put(data, tier=Tier.working)

    reconfigured_buckets = dict(_BUCKETS)
    reconfigured_buckets[Tier.working] = "dev-fin-observatory-datasets-working-v2"
    store_after = LiveArtifactStore(
        buckets=reconfigured_buckets, inventory=inventory, client=client
    )

    with pytest.raises(ArtifactRecordConflictError) as excinfo:
        store_after.put(data, tier=Tier.working)

    assert (
        client.buckets["dev-fin-observatory-datasets-working-v2"].objects[first.object_name] == data
    )
    assert excinfo.value.digest == first.digest
    assert excinfo.value.stored_bucket == _BUCKETS[Tier.working]
    assert excinfo.value.incoming_bucket == "dev-fin-observatory-datasets-working-v2"
    assert len(inventory.calls) == 1
    assert inventory.calls[0]["bucket"] == _BUCKETS[Tier.working]


def test_copy_moves_an_object_to_the_destination_tiers_bucket_and_records_it():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"promote me to pinned"
    written = store.put(data, tier=Tier.working)

    result = store.copy(written.digest, from_tier=Tier.working, to_tier=Tier.pinned)

    assert result.digest == written.digest
    assert result.tier == Tier.pinned
    assert result.bucket == _BUCKETS[Tier.pinned]
    assert result.object_name == written.digest
    assert result.size_bytes == len(data)
    assert result.deduplicated is False
    assert client.buckets[_BUCKETS[Tier.pinned]].objects[written.digest] == data
    # Copy retains the source object.
    assert client.buckets[_BUCKETS[Tier.working]].objects[written.digest] == data
    recorded = [c for c in inventory.calls if c["tier"] == Tier.pinned]
    assert len(recorded) == 1
    assert recorded[0]["bucket"] == _BUCKETS[Tier.pinned]
    assert recorded[0]["object_name"] == written.digest


class _FakeClock:
    """A settable `time.monotonic` stand-in: tests move `now` forward
    explicitly rather than sleeping.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _CountingInventory(RecordingArtifactInventory):
    """`RecordingArtifactInventory` plus a `get_location` call counter, so a
    test can assert a signed-URL cache hit skips the inventory read
    entirely.
    """

    def __init__(self) -> None:
        super().__init__()
        self.get_location_calls = 0

    def get_location(self, digest: str, *, tier: Tier):
        self.get_location_calls += 1
        return super().get_location(digest, tier=tier)


def _counting_sign(monkeypatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def _fake_sign(**kwargs: object) -> str:
        calls.append(kwargs)
        return f"https://storage.googleapis.com/signed/{len(calls)}"

    monkeypatch.setattr("artifact_store.store.sign_get_url", _fake_sign)
    return calls


def test_signed_get_url_within_the_reuse_window_signs_once_and_reuses_the_url(monkeypatch):
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    clock = _FakeClock()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client, clock=clock)
    result = store.put(b"cached artifact", tier=Tier.working)
    sign_calls = _counting_sign(monkeypatch)

    first = store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    clock.now += SIGNED_URL_REUSE_WINDOW.total_seconds() - 1
    second = store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    assert first == second
    assert len(sign_calls) == 1


def test_signed_get_url_past_the_reuse_window_signs_again(monkeypatch):
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    clock = _FakeClock()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client, clock=clock)
    result = store.put(b"expiring cache entry", tier=Tier.working)
    sign_calls = _counting_sign(monkeypatch)

    first = store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    clock.now += SIGNED_URL_REUSE_WINDOW.total_seconds()
    second = store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    assert first != second
    assert len(sign_calls) == 2


def test_signed_get_url_cache_key_distinguishes_tier_digest_ttl_and_response_headers(monkeypatch):
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    clock = _FakeClock()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client, clock=clock)
    working = store.put(b"tiered artifact", tier=Tier.working)
    pinned = store.copy(working.digest, from_tier=Tier.working, to_tier=Tier.pinned)
    other = store.put(b"a different artifact", tier=Tier.working)
    sign_calls = _counting_sign(monkeypatch)

    store.signed_get_url(
        working.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    store.signed_get_url(
        pinned.digest,
        tier=Tier.pinned,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    store.signed_get_url(
        other.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    store.signed_get_url(
        working.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=10),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    store.signed_get_url(
        working.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="application/zip",
        response_disposition='attachment; filename="test.csv"',
    )
    store.signed_get_url(
        working.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="different.csv"',
    )

    assert len(sign_calls) == 6


def test_signed_get_url_drops_stale_cache_entries_on_insert(monkeypatch):
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    clock = _FakeClock()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client, clock=clock)
    first_artifact = store.put(b"first artifact", tier=Tier.working)
    second_artifact = store.put(b"second artifact", tier=Tier.working)
    _counting_sign(monkeypatch)

    store.signed_get_url(
        first_artifact.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    assert len(store._url_cache) == 1

    clock.now += SIGNED_URL_REUSE_WINDOW.total_seconds()
    store.signed_get_url(
        second_artifact.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    # Expired entries are removed when a new cache entry is inserted.
    cached_keys = list(store._url_cache)
    assert len(cached_keys) == 1
    assert cached_keys[0][1] == second_artifact.digest


def test_signed_get_url_cache_hit_skips_the_inventory_lookup(monkeypatch):
    client = FakeGCSClient()
    inventory = _CountingInventory()
    clock = _FakeClock()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client, clock=clock)
    result = store.put(b"skip the inventory read on a hit", tier=Tier.working)
    _counting_sign(monkeypatch)

    store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    calls_after_first = inventory.get_location_calls
    store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    assert calls_after_first == 1
    assert inventory.get_location_calls == 1


def test_signed_get_url_cache_timestamp_predates_signing_latency(monkeypatch):
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    clock = _FakeClock()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client, clock=clock)
    signing_latency = 2.0
    calls: list[dict[str, object]] = []

    def _slow_sign(**kwargs: object) -> str:
        calls.append(kwargs)
        clock.now += signing_latency
        return f"https://storage.googleapis.com/signed/{len(calls)}"

    monkeypatch.setattr("artifact_store.store.sign_get_url", _slow_sign)

    stale_artifact = store.put(b"stale after latency", tier=Tier.working)
    pre_sign_time = clock.now
    store.signed_get_url(
        stale_artifact.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    assert len(calls) == 1

    # The window is measured from before signing; signing latency puts this entry past expiry.
    clock.now = pre_sign_time + SIGNED_URL_REUSE_WINDOW.total_seconds() + 1
    store.signed_get_url(
        stale_artifact.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    assert len(calls) == 2

    fresh_artifact = store.put(b"still fresh within the true window", tier=Tier.working)
    pre_sign_time = clock.now
    first = store.signed_get_url(
        fresh_artifact.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )
    assert len(calls) == 3

    clock.now = pre_sign_time + SIGNED_URL_REUSE_WINDOW.total_seconds() - 1
    second = store.signed_get_url(
        fresh_artifact.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="text/csv",
        response_disposition='attachment; filename="test.csv"',
    )

    assert first == second
    assert len(calls) == 3


def test_copy_raises_artifact_not_found_when_the_source_tier_has_no_such_digest():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    with pytest.raises(ArtifactNotFoundError) as excinfo:
        store.copy("sha256:never-written", from_tier=Tier.working, to_tier=Tier.pinned)

    assert excinfo.value.digest == "sha256:never-written"
    assert excinfo.value.tier == Tier.working


def test_copy_is_idempotent_when_the_destination_already_has_the_object():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"already promoted once"
    written = store.put(data, tier=Tier.working)
    first = store.copy(written.digest, from_tier=Tier.working, to_tier=Tier.pinned)

    second = store.copy(written.digest, from_tier=Tier.working, to_tier=Tier.pinned)

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.digest == first.digest
    assert second.size_bytes == first.size_bytes
    assert client.buckets[_BUCKETS[Tier.pinned]].upload_log == [written.digest]


def test_copy_raises_artifact_write_error_on_a_non_precondition_failure():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    data = b"a copy that will fail"
    written = store.put(data, tier=Tier.working)
    client.buckets[_BUCKETS[Tier.working]].raise_on_next_copy = ServiceUnavailable("gcs hiccup")

    with pytest.raises(ArtifactWriteError) as excinfo:
        store.copy(written.digest, from_tier=Tier.working, to_tier=Tier.pinned)

    assert excinfo.value.digest == written.digest
    assert excinfo.value.object_name == written.digest
    assert not [c for c in inventory.calls if c["tier"] == Tier.pinned]


def test_list_returns_every_object_name_currently_in_the_tiers_bucket():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)
    first = store.put(b"one", tier=Tier.working)
    second = store.put(b"two", tier=Tier.working)
    store.put(b"three", tier=Tier.pinned)

    result = store.list(tier=Tier.working)

    assert result == frozenset({first.digest, second.digest})


def test_list_is_empty_for_a_tier_with_nothing_written():
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = _store(client, inventory)

    assert store.list(tier=Tier.pinned) == frozenset()
