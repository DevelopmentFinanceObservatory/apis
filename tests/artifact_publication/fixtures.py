"""The `gs_bucket` fixture over the fake GCS client in `tests.artifact_publication.support`."""

from __future__ import annotations

import pytest

from tests.artifact_publication.support import BUCKET, FakeStorageClient, StoredObject


@pytest.fixture
def gs_bucket(monkeypatch) -> dict[str, StoredObject]:
    """Monkeypatches `google.cloud.storage.Client` so every `storage
    .Client()` call inside `artifact_publication.uri`'s `gs://` helpers reads and
    writes the same in-memory bucket, real `NotFound`/`PreconditionFailed`
    exceptions included. Returns `BUCKET`'s own object store, keyed by
    object name, for a test that needs to inspect or mutate it directly
    (simulating a concurrent write or delete).
    """
    buckets: dict[str, dict[str, StoredObject]] = {}
    monkeypatch.setattr("google.cloud.storage.Client", lambda: FakeStorageClient(buckets))
    return buckets.setdefault(BUCKET, {})
