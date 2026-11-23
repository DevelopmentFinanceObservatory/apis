"""Test `SharedVersionReads` against a reader that counts its reads."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

from dataset_store.snapshot import SharedVersionReads, SnapshotReader


class _CountingReader:
    """Count reads of the inner reader. A subject named `missing` has no record."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def read_version_state(
        self, subject_id: str, content_version: str, *, revoked_under: Sequence[str] = ()
    ) -> Any:
        self.calls.append(("state", tuple(revoked_under)))
        if subject_id == "missing":
            return None
        version = SimpleNamespace(subject_id=subject_id, content_version=content_version)
        return SimpleNamespace(version=version, locations=[], revocation=None)

    def read_version(self, subject_id: str, content_version: str) -> Any:
        self.calls.append(("version", ()))
        return SimpleNamespace(subject_id=subject_id, content_version=content_version)


def _reads() -> tuple[SharedVersionReads, _CountingReader]:
    inner = _CountingReader()
    return SharedVersionReads(cast(SnapshotReader, inner)), inner


def test_distinct_revoked_under_reads_twice() -> None:
    reads, inner = _reads()
    reads.read_version_state("ds_a", "cv_1", revoked_under=["ds_x"])
    reads.read_version_state("ds_a", "cv_1", revoked_under=["ds_y"])
    assert len(inner.calls) == 2


def test_same_call_twice_reads_once() -> None:
    reads, inner = _reads()
    first = reads.read_version_state("ds_a", "cv_1", revoked_under=["ds_x"])
    second = reads.read_version_state("ds_a", "cv_1", revoked_under=["ds_x"])
    assert first is second
    assert len(inner.calls) == 1


def test_read_version_after_state_costs_no_inner_read() -> None:
    reads, inner = _reads()
    state = reads.read_version_state("ds_a", "cv_1")
    assert state is not None
    assert reads.read_version("ds_a", "cv_1") is state.version
    assert len(inner.calls) == 1


def test_none_result_is_not_held() -> None:
    reads, inner = _reads()
    assert reads.read_version_state("missing", "cv_1") is None
    assert reads.read_version_state("missing", "cv_1") is None
    assert len(inner.calls) == 2


def test_close_passes_every_read_through() -> None:
    reads, inner = _reads()
    reads.read_version_state("ds_a", "cv_1")
    reads.read_version("ds_a", "cv_1")
    inner.calls.clear()
    reads.close()
    reads.read_version_state("ds_a", "cv_1")
    reads.read_version_state("ds_a", "cv_1")
    reads.read_version("ds_a", "cv_1")
    assert len(inner.calls) == 3
