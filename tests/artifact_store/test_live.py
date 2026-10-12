"""Live staging smoke tests; skipped unless both bucket names are set.

Set `DFO_ARTIFACT_STORE_LIVE_BUCKET_WORKING` and
`DFO_ARTIFACT_STORE_LIVE_BUCKET_PINNED` to run. The runtime identity needs
`roles/iam.serviceAccountTokenCreator` on itself. Set
`DFO_ARTIFACT_STORE_LIVE_REQUIRED` to fail collection if either name is missing.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import httpx
import pytest

from artifact_store.inventory import LiveArtifactInventory
from artifact_store.store import LiveArtifactStore, Tier
from dataset_store import LiveDatasetStore, shared_pool

_WORKING_BUCKET = os.environ.get("DFO_ARTIFACT_STORE_LIVE_BUCKET_WORKING")
_PINNED_BUCKET = os.environ.get("DFO_ARTIFACT_STORE_LIVE_BUCKET_PINNED")

if not (_WORKING_BUCKET and _PINNED_BUCKET) and os.environ.get("DFO_ARTIFACT_STORE_LIVE_REQUIRED"):
    raise RuntimeError(
        "DFO_ARTIFACT_STORE_LIVE_REQUIRED is set but DFO_ARTIFACT_STORE_LIVE_BUCKET_WORKING "
        "and/or DFO_ARTIFACT_STORE_LIVE_BUCKET_PINNED is empty -- this suite would skip "
        "every test and report green over zero live calls"
    )

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (_WORKING_BUCKET and _PINNED_BUCKET),
        reason=(
            "set DFO_ARTIFACT_STORE_LIVE_BUCKET_WORKING and "
            "DFO_ARTIFACT_STORE_LIVE_BUCKET_PINNED to run this against real staging "
            "buckets; the runtime identity must also hold "
            "roles/iam.serviceAccountTokenCreator on itself for signing to succeed"
        ),
    ),
]


@pytest.fixture
def live_store():
    inventory = LiveArtifactInventory(store=LiveDatasetStore(connections=shared_pool()))
    return LiveArtifactStore(
        buckets={Tier.working: _WORKING_BUCKET, Tier.pinned: _PINNED_BUCKET},
        inventory=inventory,
    )


def test_written_artifact_is_fetchable_through_its_signed_url_and_digest_matches(live_store):
    data = b"artifact_store live smoke test payload"
    result = live_store.put(data, tier=Tier.working)

    url = live_store.signed_location_url(
        result.digest,
        tier=Tier.working,
        bucket=result.bucket,
        object_name=result.object_name,
        ttl=timedelta(minutes=5),
        response_type="application/octet-stream",
        response_disposition="attachment",
    )
    response = httpx.get(url)
    response.raise_for_status()

    assert response.content == data
    assert f"sha256:{hashlib.sha256(response.content).hexdigest()}" == result.digest


def test_concurrent_puts_of_identical_bytes_converge_to_one_stored_object(live_store):
    # GCS's `if_generation_match=0` lets one create land; both writers must
    # return the same digest and the fetched object must contain the bytes.
    data = f"artifact_store live concurrency smoke test payload {os.urandom(8).hex()}".encode()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(live_store.put, data, tier=Tier.working) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results[0].digest == results[1].digest
    assert {result.deduplicated for result in results} == {True, False}

    url = live_store.signed_location_url(
        results[0].digest,
        tier=Tier.working,
        bucket=results[0].bucket,
        object_name=results[0].object_name,
        ttl=timedelta(minutes=5),
        response_type="application/octet-stream",
        response_disposition="attachment",
    )
    response = httpx.get(url)
    response.raise_for_status()
    assert response.content == data
