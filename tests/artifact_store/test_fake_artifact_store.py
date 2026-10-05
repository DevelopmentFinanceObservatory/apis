"""Pin `FakeArtifactStore` writes, reads, deduplication, and signed URL bounds."""

from __future__ import annotations

from datetime import timedelta

import pytest

from artifact_store.errors import ArtifactNotFoundError
from artifact_store.signing import SIGNED_URL_TTL_CEILING, SIGNED_URL_TTL_FLOOR
from artifact_store.store import Tier

from .conftest import FakeArtifactStore


def test_put_is_not_deduplicated_the_first_time():
    store = FakeArtifactStore()

    result = store.put(b"first write", tier=Tier.working)

    assert result.deduplicated is False


def test_second_put_of_the_same_bytes_and_tier_is_deduplicated():
    store = FakeArtifactStore()

    first = store.put(b"same bytes", tier=Tier.working)
    second = store.put(b"same bytes", tier=Tier.working)

    assert first.digest == second.digest
    assert second.deduplicated is True


def test_second_put_of_the_same_bytes_in_a_different_tier_is_not_deduplicated():
    store = FakeArtifactStore()

    store.put(b"same bytes, other tier", tier=Tier.working)
    result = store.put(b"same bytes, other tier", tier=Tier.pinned)

    assert result.deduplicated is False


def test_get_returns_the_exact_bytes_that_were_put():
    store = FakeArtifactStore()
    written = store.put(b"round trip me", tier=Tier.working)

    assert store.get(artifact_digest=written.digest, tier=Tier.working) == b"round trip me"


@pytest.mark.parametrize(
    ("data", "stored_tier", "requested_tier", "unstored_digest"),
    [
        pytest.param(None, None, Tier.working, "sha256:nothere", id="absent-digest"),
        pytest.param(b"only in working", Tier.working, Tier.pinned, None, id="wrong-tier"),
    ],
)
def test_get_raises_not_found_when_digest_is_unavailable_in_requested_tier(
    data, stored_tier, requested_tier, unstored_digest
):
    store = FakeArtifactStore()
    digest = unstored_digest
    if data is not None:
        digest = store.put(data, tier=stored_tier).digest
    assert digest is not None

    with pytest.raises(ArtifactNotFoundError) as excinfo:
        store.get(artifact_digest=digest, tier=requested_tier)

    assert excinfo.value.digest == digest
    assert excinfo.value.tier == requested_tier


_CSV_HEADERS = {
    "response_type": "text/csv",
    "response_disposition": 'attachment; filename="test.csv"',
}


def test_signed_get_url_raises_not_found_for_a_digest_never_put():
    store = FakeArtifactStore()

    with pytest.raises(ArtifactNotFoundError) as excinfo:
        store.signed_get_url(
            "sha256:nothere", tier=Tier.working, ttl=timedelta(minutes=5), **_CSV_HEADERS
        )

    assert excinfo.value.digest == "sha256:nothere"
    assert excinfo.value.tier == Tier.working


def test_signed_get_url_below_the_ttl_floor_is_refused():
    store = FakeArtifactStore()
    result = store.put(b"floor check", tier=Tier.working)

    with pytest.raises(ValueError):
        store.signed_get_url(
            result.digest, tier=Tier.working, ttl=timedelta(milliseconds=999), **_CSV_HEADERS
        )


def test_signed_get_url_at_exactly_the_ttl_floor_is_accepted():
    store = FakeArtifactStore()
    result = store.put(b"floor boundary", tier=Tier.working)

    url = store.signed_get_url(
        result.digest, tier=Tier.working, ttl=SIGNED_URL_TTL_FLOOR, **_CSV_HEADERS
    )

    assert f"X-Goog-Expires={int(SIGNED_URL_TTL_FLOOR.total_seconds())}" in url


def test_signed_get_url_at_exactly_the_ttl_ceiling_is_accepted():
    store = FakeArtifactStore()
    result = store.put(b"ceiling boundary", tier=Tier.working)

    url = store.signed_get_url(
        result.digest, tier=Tier.working, ttl=SIGNED_URL_TTL_CEILING, **_CSV_HEADERS
    )

    assert f"X-Goog-Expires={int(SIGNED_URL_TTL_CEILING.total_seconds())}" in url


def test_signed_get_url_past_the_ttl_ceiling_is_refused():
    store = FakeArtifactStore()
    result = store.put(b"past ceiling", tier=Tier.working)

    with pytest.raises(ValueError):
        store.signed_get_url(
            result.digest,
            tier=Tier.working,
            ttl=SIGNED_URL_TTL_CEILING + timedelta(seconds=1),
            **_CSV_HEADERS,
        )


def test_signed_get_url_carries_the_x_goog_signature_marker():
    store = FakeArtifactStore()
    result = store.put(b"marker check", tier=Tier.working)

    url = store.signed_get_url(
        result.digest, tier=Tier.working, ttl=timedelta(minutes=5), **_CSV_HEADERS
    )

    assert "X-Goog-Signature=" in url


def test_signed_get_url_records_the_response_type_and_disposition():
    store = FakeArtifactStore()
    result = store.put(b"header recording", tier=Tier.working)

    store.signed_get_url(
        result.digest,
        tier=Tier.working,
        ttl=timedelta(minutes=5),
        response_type="application/zip",
        response_disposition='attachment; filename="bundle.zip"',
    )

    assert store.signed_headers[(Tier.working, result.digest)] == (
        "application/zip",
        'attachment; filename="bundle.zip"',
    )
