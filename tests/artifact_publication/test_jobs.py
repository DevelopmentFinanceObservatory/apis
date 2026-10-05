"""Pins the build lease and the conditional delete it releases through."""

from __future__ import annotations

import json

from google.api_core.exceptions import NotFound, PreconditionFailed

from artifact_publication.jobs import (
    LEASE_FILE,
    acquire_lease,
    release_lease,
    tracked_stamp,
    unpublished_seconds,
)
from artifact_publication.uri import delete_uri_if_unchanged, read_uri_bytes_and_generation
from tests.series_coverage.test_uri_gs import _BUCKET, _FakeBlob, gs_bucket  # noqa: F401


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
        raise NotFound(f"gs://{_BUCKET}/{self.name} does not exist")
    if if_generation_match is not None and stored.generation != if_generation_match:
        raise PreconditionFailed(f"generation mismatch for {self.name}")
    del self._objects[self.name]


def test_gs_lease_release_deletes_only_the_generation_it_took(gs_bucket, monkeypatch):  # noqa: F811
    monkeypatch.setattr(_FakeBlob, "delete", _conditional_delete)
    root = f"gs://{_BUCKET}/catalogue/format-1"
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
