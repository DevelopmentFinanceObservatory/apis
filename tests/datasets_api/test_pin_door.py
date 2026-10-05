"""Tests `pin_artifact` decisions with in-memory fakes and the door's HTTP permission checks."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from artifact_store import Tier
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    ArtifactReferenceWriter,
    Revocation,
    SnapshotRevoked,
    servable_location_of_media_type,
)
from datasets_api import door_executor
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    ArtifactNotFound,
    SnapshotNotFound,
    SpecNotFound,
)
from datasets_api.config import DatasetsConfigError
from datasets_api.doors import pin as pin_door
from datasets_api.doors.pin import PinBudgetExhausted, build_pin_door, pin_artifact
from datasets_api.reference_store import (
    ARTIFACT_REFERENCE_WRITE_FAILED_MARKER,
    PIN_REFERENCE_KIND,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from materialise.formats import FRAME1_MEDIA_TYPE
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._executor_doubles import BrokenSubmitExecutor
from ._snapshot_doubles import (
    FakeSnapshotReader,
    FakeSpecReader,
    StoreBackedSnapshotReader,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from .conftest import FakeDatasetStore

_NOW = datetime(2026, 6, 1, tzinfo=UTC)
_PRINCIPAL = "first-party-key"
_DAILY_MAX = 500


@dataclass
class FakeArtifactReferenceWriter(ArtifactReferenceWriter):
    """Live references keep `created_at`; soft-deleted references reactivate with a new time.

    `record_reference_unless_revoked` reads `revoked` and `unlinked` at call time for
    copy-window tests. `pinned_copies` maps a digest to its pinned location's `stored_at`, and
    the count includes those whose digest holds a reference of the asked kind.
    """

    records: dict[tuple[str, str, str, str, str], datetime] = field(default_factory=dict)
    removed_at: dict[tuple[str, str, str, str, str], datetime | None] = field(default_factory=dict)
    revoked: dict[tuple[str, str], str | None] = field(default_factory=dict)
    unlinked: set[tuple[str, str, str]] = field(default_factory=set)
    pinned_copies: dict[str, datetime] = field(default_factory=dict)

    def record_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        key = (principal, artifact_digest, subject_id, content_version, reference_kind)
        if key not in self.records or self.removed_at.get(key) is not None:
            self.records[key] = now
            self.removed_at[key] = None
        return self.records[key]

    def record_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        if (subject_id, content_version) in self.revoked:
            raise SnapshotRevoked(
                id=subject_id,
                content_version=content_version,
                reason=self.revoked[(subject_id, content_version)],
            )
        if (subject_id, content_version, artifact_digest) in self.unlinked:
            raise ArtifactNotFound(
                id=subject_id, content_version=content_version, artifact_digest=artifact_digest
            )
        return self.record_reference(
            principal=principal,
            artifact_digest=artifact_digest,
            subject_id=subject_id,
            content_version=content_version,
            reference_kind=reference_kind,
            now=now,
        )

    def count_pinned_copies_since(self, *, reference_kind: str, since: datetime) -> int:
        referenced = {key[1] for key in self.records if key[4] == reference_kind}
        return sum(
            1
            for digest, stored_at in self.pinned_copies.items()
            if digest in referenced and stored_at >= since
        )

    def soft_delete(self, key: tuple[str, str, str, str, str], *, removed_at: datetime) -> None:
        self.removed_at[key] = removed_at


class AlwaysRaisingArtifactReferenceWriter(FakeArtifactReferenceWriter):
    """Raises `DatasetsStoreUnavailable` on every reference write."""

    def record_reference(self, **kwargs):
        raise DatasetsStoreUnavailable("exemplar diagnostics")


class FlakyOnceArtifactReferenceWriter(FakeArtifactReferenceWriter):
    """Fails once, then records references through the idempotent fake."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def record_reference(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise DatasetsStoreUnavailable("exemplar diagnostics")
        return super().record_reference(**kwargs)


class BlockingFakeArtifactReferenceWriter(FakeArtifactReferenceWriter):
    """Blocks reference writes on events so tests can fill the executor deterministically."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def record_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        self.entered.set()
        self.release.wait(timeout=5.0)
        return super().record_reference(
            principal=principal,
            artifact_digest=artifact_digest,
            subject_id=subject_id,
            content_version=content_version,
            reference_kind=reference_kind,
            now=now,
        )


def test_unregistered_id_refuses_spec_not_found():
    with pytest.raises(SpecNotFound):
        pin_artifact(
            id="spec_unknown",
            content_version="cv1",
            artifact_digest="sha256:aaa",
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(),
            snapshot_reader=FakeSnapshotReader(),
            artifact_store=None,  # never reached
            reference_writer=FakeArtifactReferenceWriter(),
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )


def test_unknown_content_version_refuses_snapshot_not_found():
    with pytest.raises(SnapshotNotFound):
        pin_artifact(
            id="spec_a",
            content_version="cv_unknown",
            artifact_digest="sha256:aaa",
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=FakeSnapshotReader(versions={("spec_a", "cv1"): _version()}),
            artifact_store=None,
            reference_writer=FakeArtifactReferenceWriter(),
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )


def test_a_revoked_version_refuses_snapshot_revoked():
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        revocations={("spec_a", "cv1"): Revocation(reason=None, revoked_at=_NOW)},
    )
    with pytest.raises(SnapshotRevoked):
        pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest="sha256:aaa",
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=None,
            reference_writer=FakeArtifactReferenceWriter(),
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )


def test_a_digest_never_linked_to_this_version_refuses_artifact_not_found():
    store_artifact_store = object()  # not touched: refused before any store call
    reader = FakeSnapshotReader(versions={("spec_a", "cv1"): _version()})
    with pytest.raises(ArtifactNotFound):
        pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest="sha256:never-linked",
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store_artifact_store,
            reference_writer=FakeArtifactReferenceWriter(),
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )


def test_a_digest_only_in_the_working_tier_and_expired_refuses_artifact_expired():
    from ._artifact_doubles import FakeArtifactStore

    long_ago = _NOW - WORKING_TIER_RETENTION - WORKING_TIER_RETENTION
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"aged-out", stored_at=long_ago)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    with pytest.raises(ArtifactExpired):
        pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest=location.artifact_digest,
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=FakeArtifactReferenceWriter(),
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )


def test_a_revocation_landing_during_the_copy_window_is_caught_at_reference_write_time():
    """A revocation injected during `copy` is caught by the write-time check, leaving no reference."""
    from ._artifact_doubles import FakeArtifactStore

    reference_writer = FakeArtifactReferenceWriter()

    class _RevokingDuringCopyArtifactStore(FakeArtifactStore):
        def copy(self, digest, *, from_tier, to_tier):
            reference_writer.revoked[("spec_a", "cv1")] = "revoked mid-copy"
            return super().copy(digest, from_tier=from_tier, to_tier=to_tier)

    store = _RevokingDuringCopyArtifactStore()
    location = _store_and_locate(store, b"revoked-mid-copy")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    with pytest.raises(SnapshotRevoked):
        pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest=location.artifact_digest,
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=reference_writer,
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )

    assert reference_writer.records == {}


def test_pinning_a_working_tier_artifact_copies_it_and_records_a_reference():
    from ._artifact_doubles import FakeArtifactStore

    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    reference_writer = FakeArtifactReferenceWriter()

    pinned_at = pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=reference_writer,
        daily_max=_DAILY_MAX,
        now=lambda: _NOW,
    )

    assert pinned_at == _NOW
    assert (location.artifact_digest, Tier.pinned.value) in store._objects
    key = (_PRINCIPAL, location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert reference_writer.records[key] == _NOW


def test_pinning_an_already_pinned_artifact_does_not_re_copy():
    from ._artifact_doubles import FakeArtifactStore

    class _CountingArtifactStore(FakeArtifactStore):
        def __init__(self) -> None:
            super().__init__()
            self.copy_calls = 0

        def copy(self, digest, *, from_tier, to_tier):
            self.copy_calls += 1
            return super().copy(digest, from_tier=from_tier, to_tier=to_tier)

    store = _CountingArtifactStore()
    location = _store_and_locate(store, b"already-pinned", tier=Tier.pinned)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    reference_writer = FakeArtifactReferenceWriter()

    pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=reference_writer,
        daily_max=_DAILY_MAX,
        now=lambda: _NOW,
    )

    assert store.copy_calls == 0


def _frame1_location(store: FakeArtifactStore, *, tier: Tier = Tier.working, stored_at=_NOW):
    return _store_and_locate(
        store,
        b'{"frame_version":1,"columns":[],"rows":[]}',
        tier=tier,
        stored_at=stored_at,
        rendering={"format": "frame1"},
        media_type=FRAME1_MEDIA_TYPE,
    )


def _parquet_location(store: FakeArtifactStore, *, tier: Tier = Tier.working):
    return _store_and_locate(
        store,
        b"PAR1-rendering",
        tier=tier,
        rendering={"format": "parquet", "bundle": False},
        media_type="application/vnd.apache.parquet",
    )


def _pin_parquet(
    reader: FakeSnapshotReader,
    store: FakeArtifactStore,
    writer: FakeArtifactReferenceWriter,
    digest: str,
) -> datetime:
    return pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=writer,
        daily_max=_DAILY_MAX,
        now=lambda: _NOW,
    )


def test_pinning_a_parquet_artifact_keeps_the_frame1_servable_past_working_tier_expiry():
    store = FakeArtifactStore()
    parquet = _parquet_location(store)
    frame = _frame1_location(store)
    reader = StoreBackedSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [parquet, frame]},
        store=store,
    )
    writer = FakeArtifactReferenceWriter()

    pinned_at = _pin_parquet(reader, store, writer, parquet.artifact_digest)

    assert pinned_at == _NOW
    later = _NOW + WORKING_TIER_RETENTION + timedelta(days=1)
    located = servable_location_of_media_type(
        reader.read_artifact_locations("spec_a", "cv1"), media_type=FRAME1_MEDIA_TYPE, now=later
    )
    assert located is not None
    assert (located.artifact_digest, located.tier) == (frame.artifact_digest, Tier.pinned.value)
    frame_key = (_PRINCIPAL, frame.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[frame_key] == _NOW


def test_re_pinning_an_already_pinned_artifact_pins_a_working_tier_frame1():
    """A share pinned before frame retention, or a retry after a partial failure, heals."""
    store = FakeArtifactStore()
    parquet = _parquet_location(store, tier=Tier.pinned)
    frame = _frame1_location(store)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [parquet, frame]},
    )
    writer = FakeArtifactReferenceWriter()

    _pin_parquet(reader, store, writer, parquet.artifact_digest)

    assert (frame.artifact_digest, Tier.pinned.value) in store._objects
    frame_key = (_PRINCIPAL, frame.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[frame_key] == _NOW


@pytest.mark.parametrize("frame1", ["absent", "aged_out"])
def test_a_version_without_a_servable_frame1_pins_the_named_artifact_alone(frame1: str):
    store = FakeArtifactStore()
    parquet = _parquet_location(store)
    locations = [parquet]
    if frame1 == "aged_out":
        long_ago = _NOW - WORKING_TIER_RETENTION - timedelta(days=1)
        locations.append(_frame1_location(store, stored_at=long_ago))
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): locations}
    )
    writer = FakeArtifactReferenceWriter()

    pinned_at = _pin_parquet(reader, store, writer, parquet.artifact_digest)

    assert pinned_at == _NOW
    assert list(writer.records) == [
        (_PRINCIPAL, parquet.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    ]
    assert store.list(tier=Tier.pinned) == {parquet.artifact_digest}


def _writer_with_pinned_copies(*stored_at: datetime) -> FakeArtifactReferenceWriter:
    """A writer seeing one door-pinned copy per timestamp, each of a distinct digest."""
    writer = FakeArtifactReferenceWriter()
    for i, at in enumerate(stored_at):
        writer.pinned_copies[f"sha256:seed{i}"] = at
        writer.records[(_PRINCIPAL, f"sha256:seed{i}", "spec_seed", "cv1", PIN_REFERENCE_KIND)] = at
    return writer


class _CopyForbiddenArtifactStore(FakeArtifactStore):
    def copy(self, digest, *, from_tier, to_tier):
        raise AssertionError("the budget refusal must come before any copy")


def test_pin_at_the_daily_budget_refuses_before_copying():
    store = _CopyForbiddenArtifactStore()
    location = _store_and_locate(store, b"over-budget")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    writer = _writer_with_pinned_copies(_NOW - timedelta(hours=1), _NOW - timedelta(hours=23))

    with pytest.raises(PinBudgetExhausted):
        pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest=location.artifact_digest,
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=writer,
            daily_max=2,
            now=lambda: _NOW,
        )

    assert (location.artifact_digest, Tier.pinned.value) not in store._objects
    assert len(writer.records) == 2


@pytest.mark.parametrize(
    ("tier", "seeded"),
    [
        # A repair of an already pinned artifact makes no copy, so the budget does not apply.
        (Tier.pinned, (_NOW - timedelta(hours=1), _NOW - timedelta(hours=2))),
        # Pins older than the trailing 24 hours no longer count.
        (Tier.working, (_NOW - timedelta(hours=1), _NOW - timedelta(hours=24, seconds=1))),
    ],
    ids=["repair-of-pinned-artifact-at-budget", "below-budget-in-window"],
)
def test_pin_the_budget_admits_copies_and_records_a_reference(
    tier: Tier, seeded: tuple[datetime, ...]
):
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"within-budget", tier=tier)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    writer = _writer_with_pinned_copies(*seeded)

    pinned_at = pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=writer,
        daily_max=2,
        now=lambda: _NOW,
    )

    assert pinned_at == _NOW
    assert (location.artifact_digest, Tier.pinned.value) in store._objects
    key = (_PRINCIPAL, location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[key] == _NOW


def test_pin_budget_counts_pinned_copies_not_references():
    """Two references sharing one pinned copy, a standalone version and a collection member,
    spend one unit of the budget, so a new copy still fits under a budget of two."""
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"shared-digest-budget")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    shared = "sha256:shared"
    writer = FakeArtifactReferenceWriter(pinned_copies={shared: _NOW - timedelta(hours=1)})
    for subject_id in ("spec_standalone", "col_member"):
        writer.records[(_PRINCIPAL, shared, subject_id, "cv1", PIN_REFERENCE_KIND)] = (
            _NOW - timedelta(hours=1)
        )

    pinned_at = pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=writer,
        daily_max=2,
        now=lambda: _NOW,
    )

    assert pinned_at == _NOW
    assert (location.artifact_digest, Tier.pinned.value) in store._objects


def test_a_link_replaced_during_the_copy_refuses_artifact_not_found_without_the_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A re-render that unlinks the digest while it is copied makes the reference write refuse.
    The refusal is the pin's answer, not a reference-write failure to reconcile."""
    writer = FakeArtifactReferenceWriter()

    class _RelinkingDuringCopyArtifactStore(FakeArtifactStore):
        def copy(self, digest, *, from_tier, to_tier):
            writer.unlinked.add(("spec_a", "cv1", digest))
            return super().copy(digest, from_tier=from_tier, to_tier=to_tier)

    store = _RelinkingDuringCopyArtifactStore()
    location = _store_and_locate(store, b"relinked-mid-copy")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    with caplog.at_level(logging.ERROR), pytest.raises(ArtifactNotFound):
        pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest=location.artifact_digest,
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=writer,
            daily_max=_DAILY_MAX,
            now=lambda: _NOW,
        )

    assert writer.records == {}
    assert ARTIFACT_REFERENCE_WRITE_FAILED_MARKER not in caplog.text


def test_pin_door_wire_refuses_an_exhausted_budget_503_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DATASETS_PIN_DAILY_MAX` reaches the door, and the refusal is retryable with a wait."""
    monkeypatch.setenv("DATASETS_PIN_DAILY_MAX", "0")
    store = _CopyForbiddenArtifactStore()
    # The composed surface reads the wall clock, so the working copy must be fresh.
    location = _store_and_locate(store, b"pin-me-no-budget", stored_at=datetime.now(UTC))
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=FakeArtifactReferenceWriter(),
        pin_allowed_principals=frozenset({_PRINCIPAL}),
    )

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True
    assert int(resp.headers["retry-after"]) > 0


def test_repeat_pin_by_the_same_principal_reports_the_original_pinned_at():
    from ._artifact_doubles import FakeArtifactStore

    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me-twice")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    reference_writer = FakeArtifactReferenceWriter()

    first = pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=reference_writer,
        daily_max=_DAILY_MAX,
        now=lambda: _NOW,
    )
    later = datetime(2026, 7, 1, tzinfo=UTC)
    second = pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=reference_writer,
        daily_max=_DAILY_MAX,
        now=lambda: later,
    )

    assert first == second == _NOW


def test_reference_write_failure_after_the_copy_logs_the_marker_and_an_identical_retry_converges(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reference-write failure leaves the pinned copy in place. An identical retry skips it
    and records the reference.
    """
    from datasets_api.app import classify_datasets_exception

    from ._artifact_doubles import FakeArtifactStore

    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me-flaky")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    writer = FlakyOnceArtifactReferenceWriter()

    def _pin(*, now: datetime) -> datetime:
        return pin_artifact(
            id="spec_a",
            content_version="cv1",
            artifact_digest=location.artifact_digest,
            principal=_PRINCIPAL,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=writer,
            daily_max=_DAILY_MAX,
            now=lambda: now,
        )

    with caplog.at_level(logging.ERROR), pytest.raises(DatasetsStoreUnavailable) as excinfo:
        _pin(now=_NOW)

    spec = classify_datasets_exception(excinfo.value)
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True

    assert ARTIFACT_REFERENCE_WRITE_FAILED_MARKER in caplog.text
    assert f"artifact_digest={location.artifact_digest}" in caplog.text
    assert "subject_id=spec_a" in caplog.text
    assert "content_version=cv1" in caplog.text
    assert f"principal={_PRINCIPAL}" in caplog.text
    assert "already_pinned=False" in caplog.text

    # The copy already landed before the reference write failed, and stays
    # -- pin_artifact never compensates with a delete of it.
    assert (location.artifact_digest, Tier.pinned.value) in store._objects

    pinned_at = _pin(now=_NOW)

    assert pinned_at == _NOW
    key = (_PRINCIPAL, location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[key] == _NOW


def test_pin_door_wire_maps_a_reference_writer_store_failure_to_retryable_503():
    """The composed `datasets_surface` classifier maps store failures to retryable 503."""
    store = FakeArtifactStore()
    # A pinned artifact skips the retention check, isolating the reference write.
    location = _store_and_locate(store, b"pin-me-store-unavailable", tier=Tier.pinned)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=AlwaysRaisingArtifactReferenceWriter(),
        pin_allowed_principals=frozenset({_PRINCIPAL}),
    )

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_construction_failure_returns_the_pool_borrow_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config error returns the pool borrow opened for `spec_reader`."""
    open_calls = 0
    close_calls = 0

    def _counting_shared_pool() -> object:
        nonlocal open_calls
        open_calls += 1
        return object()

    def _counting_close_shared_pool() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(pin_door, "shared_pool", _counting_shared_pool)
    monkeypatch.setattr(pin_door, "close_shared_pool", _counting_close_shared_pool)
    monkeypatch.setattr(pin_door, "LiveDatasetStore", lambda *, connections: object())

    door = build_pin_door()  # every store dependency left uninjected

    async def _enter() -> None:
        async with door.lifespan(None):
            pass  # pragma: no cover -- construction fails before this is reached

    with pytest.raises(DatasetsConfigError):
        asyncio.run(_enter())

    assert open_calls == 1
    assert close_calls == 1


@pytest.mark.parametrize(
    "uninjected_deps",
    [
        frozenset({"snapshot_reader"}),
        frozenset({"reference_writer"}),
        frozenset({"snapshot_reader", "reference_writer"}),
    ],
    ids=lambda deps: "uninjected_" + "_".join(sorted(deps)),
)
def test_build_refuses_to_start_with_only_some_dependencies_injected(
    uninjected_deps: frozenset[str],
) -> None:
    """Partial injection fails in `build_pin_door`. All-`None` dependencies defer to lifespan."""
    with pytest.raises(DatasetsConfigError) as excinfo:
        build_pin_door(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=(
                None if "snapshot_reader" in uninjected_deps else FakeSnapshotReader()
            ),
            artifact_store=FakeArtifactStore(),
            reference_writer=(
                None if "reference_writer" in uninjected_deps else FakeArtifactReferenceWriter()
            ),
            allowed_principals=frozenset({_PRINCIPAL}),
        )

    for dep in uninjected_deps:
        assert dep in str(excinfo.value)


def test_backpressure_refuses_503_retryable_when_executor_is_already_at_capacity() -> None:
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-backpressure-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    blocking_writer = BlockingFakeArtifactReferenceWriter()
    door = build_pin_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=blocking_writer,
        now=lambda: _NOW,
        allowed_principals=frozenset({_PRINCIPAL}),
        pool_size=1,
    )
    client = make_client_for_door(door)

    def _post() -> object:
        return client.post(
            "/api/datasets/spec_a/cv1/pin",
            json={"schema_version": 1, "artifact_digest": location.artifact_digest},
            headers={"x-dfo-principal": _PRINCIPAL},
        )

    first_response: list = []
    # Share one portal loop for both requests; this door's test surface has no lifespan.
    with client:
        first_thread = threading.Thread(target=lambda: first_response.append(_post()))
        first_thread.start()
        # The blocking writer holds the only executor slot until released.
        assert blocking_writer.entered.wait(timeout=5.0), (
            "first request never reached the blocking reference writer"
        )

        resp = _post()

        assert resp.status_code == 503
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True

        blocking_writer.release.set()
        first_thread.join(timeout=5.0)
        assert first_response and first_response[0].status_code == 200


def test_broken_thread_pool_surfaces_as_internal_error_not_backpressure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(door_executor, "ThreadPoolExecutor", BrokenSubmitExecutor)
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-broken-pool-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_pin_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=FakeArtifactReferenceWriter(),
        now=lambda: _NOW,
        allowed_principals=frozenset({_PRINCIPAL}),
        pool_size=1,
    )
    client = make_client_for_door(door)

    with caplog.at_level(logging.ERROR):
        resp = client.post(
            "/api/datasets/spec_a/cv1/pin",
            json={"schema_version": 1, "artifact_digest": location.artifact_digest},
            headers={"x-dfo-principal": _PRINCIPAL},
        )

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["retryable"] is False
    assert "permanently broken" in caplog.text


def _build_client(*, allowed_principals=frozenset({_PRINCIPAL}), reference_writer=None):
    from ._artifact_doubles import FakeArtifactStore

    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me-over-http")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_pin_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=(
            reference_writer if reference_writer is not None else FakeArtifactReferenceWriter()
        ),
        now=lambda: _NOW,
        allowed_principals=allowed_principals,
    )
    return make_client_for_door(door), location


def test_pin_without_a_principal_header_refuses_pin_not_permitted():
    client, location = _build_client()

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
    )

    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "pin_not_permitted"


def test_pin_with_an_unlisted_principal_refuses_pin_not_permitted():
    client, location = _build_client()

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": "some-other-caller"},
    )

    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "pin_not_permitted"


def test_pin_with_two_principal_header_values_refuses_regardless_of_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated principal headers are refused in either order.

    `Headers.get` returns only the first value, so a forged second value would pass a `get` check.
    """
    client, location = _build_client()
    body = {"schema_version": 1, "artifact_digest": location.artifact_digest}

    with caplog.at_level(logging.WARNING):
        allowed_first = client.post(
            "/api/datasets/spec_a/cv1/pin",
            json=body,
            headers=[("x-dfo-principal", _PRINCIPAL), ("x-dfo-principal", "some-other-caller")],
        )
    assert allowed_first.status_code == 403, allowed_first.text
    assert allowed_first.json()["code"] == "pin_not_permitted"
    assert "2 values" in caplog.text

    allowed_second = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json=body,
        headers=[("x-dfo-principal", "some-other-caller"), ("x-dfo-principal", _PRINCIPAL)],
    )
    assert allowed_second.status_code == 403, allowed_second.text
    assert allowed_second.json()["code"] == "pin_not_permitted"


def test_pin_with_the_first_party_principal_succeeds():
    client, location = _build_client()

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pinned"]["id"] == "spec_a"
    assert body["pinned"]["content_version"] == "cv1"
    assert body["pinned"]["artifact_digest"] == location.artifact_digest


def test_pin_reactivates_a_soft_deleted_reference_through_the_door():
    """Repinning clears `removed_at` and resets `created_at`."""
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me-reactivate")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    reference_writer = FakeArtifactReferenceWriter()
    current_time = [_NOW]
    door = build_pin_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=reference_writer,
        now=lambda: current_time[0],
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    client = make_client_for_door(door)
    body = {"schema_version": 1, "artifact_digest": location.artifact_digest}

    first = client.post(
        "/api/datasets/spec_a/cv1/pin", json=body, headers={"x-dfo-principal": _PRINCIPAL}
    )
    assert first.status_code == 200, first.text
    assert first.json()["pinned"]["pinned_at"] == _NOW.isoformat()

    key = (_PRINCIPAL, location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    later = datetime(2026, 7, 1, tzinfo=UTC)
    reference_writer.soft_delete(key, removed_at=later)
    assert reference_writer.removed_at[key] is not None

    current_time[0] = later
    second = client.post(
        "/api/datasets/spec_a/cv1/pin", json=body, headers={"x-dfo-principal": _PRINCIPAL}
    )

    assert second.status_code == 200, second.text
    assert second.json()["pinned"]["pinned_at"] == later.isoformat()
    assert reference_writer.removed_at[key] is None
    assert reference_writer.records[key] == later


def test_pin_with_no_allowed_principals_configured_refuses_everyone():
    client, location = _build_client(allowed_principals=frozenset())

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 403, resp.text


def test_pin_door_scoped_to_the_pin_principal_refuses_the_publish_principal() -> None:
    """The pin allowlist accepts its configured principal and refuses the publish principal."""
    publish_principal = "presentation-mcp"
    client, location = _build_client(allowed_principals=frozenset({_PRINCIPAL}))

    accepted = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert accepted.status_code == 200, accepted.text

    refused = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": publish_principal},
    )
    assert refused.status_code == 403, refused.text
    assert refused.json()["code"] == "pin_not_permitted"


def test_pin_body_omitting_schema_version_is_accepted():
    client, location = _build_client()

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 200, resp.text


def test_pin_with_a_malformed_digest_refuses_400_before_any_domain_lookup():
    client, _location = _build_client()

    resp = client.post(
        "/api/datasets/spec_totally_unregistered/cv1/pin",
        json={"schema_version": 1, "artifact_digest": "not-a-sha256-digest"},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    # The malformed digest fails validation before the unregistered id is looked up.
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"


def test_a_malformed_id_refuses_the_platforms_own_not_found_before_reaching_the_handler():
    """An id outside the `spec_` and `col_` namespaces gets the platform's routing 404."""
    client, _location = _build_client()

    resp = client.post(
        "/api/datasets/not-a-valid-id/cv1/pin",
        json={"schema_version": 1, "artifact_digest": "sha256:" + "a" * 64},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "not_found"


def test_a_well_formed_but_unregistered_id_still_refuses_spec_not_found():
    """A valid `spec_` id absent from the registry gets `spec_not_found`."""
    client, _location = _build_client()

    resp = client.post(
        "/api/datasets/spec_totally_unregistered/cv1/pin",
        json={"schema_version": 1, "artifact_digest": "sha256:" + "a" * 64},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "spec_not_found"
