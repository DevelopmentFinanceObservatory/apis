"""Tests `pin_artifact` decisions with in-memory fakes and the door's HTTP permission checks."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from artifact_store import Tier
from artifact_store.errors import ArtifactWriteError
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    ArtifactNotFound,
    ArtifactReferenceWriter,
    Revocation,
    SnapshotNotFound,
    SnapshotRevoked,
)
from datasets_api import door_executor
from datasets_api.allowance import BUDGET_KEY, AllowanceKind, LiveAllowanceLedger, Spend
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.pin import (
    OpenPinBudgetExhausted,
    PinBudgetExhausted,
    build_pin_door,
    pin_artifact,
)
from datasets_api.reference_store import (
    ARTIFACT_REFERENCE_WRITE_FAILED_MARKER,
    PIN_REFERENCE_KIND,
)
from datasets_api.render import servable_frame_location
from datasets_api.retention_admission import PinUnitHeld
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform.caller import Caller, address_key
from materialise.formats import FRAME1_MEDIA_TYPE
from spec_registry import LiveSpecRegistry
from spec_registry.errors import SpecRegistryStoreError

from ._app_factories import make_client, offline_allowances
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._executor_doubles import BrokenSubmitExecutor
from ._snapshot_doubles import (
    FakeSnapshotReader,
    FakeSpecReader,
    StoreBackedSnapshotReader,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore

_NOW = datetime(2026, 6, 1, tzinfo=UTC)
_PRINCIPAL = "first-party-key"
_DAILY_MAX = 500


def _ledger(
    *, pin_budget: int = _DAILY_MAX, open_pin_budget: int = _DAILY_MAX, pin: int = _DAILY_MAX
) -> LiveAllowanceLedger:
    """A ledger over a fresh store, with the door's two budgets and the caller allowance."""
    return LiveAllowanceLedger(
        store=FakeVersionRecordDatasetStore(),
        limits={
            AllowanceKind.pin_budget: pin_budget,
            AllowanceKind.open_pin_budget: open_pin_budget,
            AllowanceKind.pin: pin,
        },
    )


def _spent(ledger: LiveAllowanceLedger) -> dict[tuple[str, str], int]:
    """Each `(key, kind)` the ledger's store holds a count for, with the count."""
    with ledger._store.transaction() as tx:
        rows = tx.fetch_all("SELECT caller_key, kind, count FROM datasets.caller_allowance")
    return {
        (cast(str, row["caller_key"]), cast(str, row["kind"])): cast(int, row["count"])
        for row in rows
    }


@dataclass
class FakeArtifactReferenceWriter(ArtifactReferenceWriter):
    """Live references keep `created_at`; soft-deleted references reactivate with a new time.

    `record_reference_unless_revoked` reads `revoked` and `unlinked` at call time for
    copy-window tests.
    """

    records: dict[tuple[str, str, str, str, str], datetime] = field(default_factory=dict)
    removed_at: dict[tuple[str, str, str, str, str], datetime | None] = field(default_factory=dict)
    revoked: dict[tuple[str, str], str | None] = field(default_factory=dict)
    unlinked: set[tuple[str, str, str]] = field(default_factory=set)

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
        revoked_under: Sequence[str] = (),
        require_link: bool = True,
    ) -> datetime:
        for subject in (subject_id, *revoked_under):
            if (subject, content_version) in self.revoked:
                raise SnapshotRevoked(
                    id=subject_id,
                    content_version=content_version,
                    reason=self.revoked[(subject, content_version)],
                )
        if require_link and (subject_id, content_version, artifact_digest) in self.unlinked:
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
            allowances=_ledger(),
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
            allowances=_ledger(),
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
            allowances=_ledger(),
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
            allowances=_ledger(),
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
            allowances=_ledger(),
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
            allowances=_ledger(),
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
        allowances=_ledger(),
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
        allowances=_ledger(),
        now=lambda: _NOW,
    )

    assert store.copy_calls == 0


_FRAME_KINDS = {
    "canonical": (b"PAR1-canonical-frame", {"format": "frame"}, "application/vnd.apache.parquet"),
    "frame1": (
        b'{"frame_version":1,"columns":[],"rows":[]}',
        {"format": "frame1"},
        FRAME1_MEDIA_TYPE,
    ),
}


def _frame_location(
    store: FakeArtifactStore, *, kind: str = "canonical", tier: Tier = Tier.working, stored_at=_NOW
):
    data, rendering, media_type = _FRAME_KINDS[kind]
    return _store_and_locate(
        store, data, tier=tier, stored_at=stored_at, rendering=rendering, media_type=media_type
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
        allowances=_ledger(),
        now=lambda: _NOW,
    )


@pytest.mark.parametrize("kind", sorted(_FRAME_KINDS))
def test_pinning_a_parquet_artifact_keeps_the_stored_frame_servable_past_working_tier_expiry(
    kind: str,
):
    store = FakeArtifactStore()
    parquet = _parquet_location(store)
    frame = _frame_location(store, kind=kind)
    reader = StoreBackedSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [parquet, frame]},
        store=store,
    )
    writer = FakeArtifactReferenceWriter()

    pinned_at = _pin_parquet(reader, store, writer, parquet.artifact_digest)

    assert pinned_at == _NOW
    later = _NOW + WORKING_TIER_RETENTION + timedelta(days=1)
    located = servable_frame_location(reader.read_artifact_locations("spec_a", "cv1"), now=later)
    assert located is not None
    assert (located.artifact_digest, located.tier) == (frame.artifact_digest, Tier.pinned.value)
    frame_key = (_PRINCIPAL, frame.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[frame_key] == _NOW


def test_re_pinning_an_already_pinned_artifact_pins_a_working_tier_canonical_frame():
    """A share pinned before frame retention, or a retry after a partial failure, heals."""
    store = FakeArtifactStore()
    parquet = _parquet_location(store, tier=Tier.pinned)
    frame = _frame_location(store)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [parquet, frame]},
    )
    writer = FakeArtifactReferenceWriter()

    _pin_parquet(reader, store, writer, parquet.artifact_digest)

    assert (frame.artifact_digest, Tier.pinned.value) in store._objects
    frame_key = (_PRINCIPAL, frame.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[frame_key] == _NOW


@pytest.mark.parametrize(
    ("frame", "frame_pinned"),
    [
        pytest.param("absent", False, id="absent"),
        pytest.param("aged_out_bytes_gone", False, id="aged_out_bytes_gone"),
        pytest.param("aged_out_bytes_survive", True, id="aged_out_bytes_survive"),
    ],
)
def test_a_pin_retains_an_aged_frame_while_its_bytes_survive(
    frame: str, frame_pinned: bool, caplog: pytest.LogCaptureFixture
):
    """With no frame, or an aged one whose bytes are gone, the named artifact pins alone, and a
    frame whose bytes are gone is logged."""
    store = FakeArtifactStore()
    parquet = _parquet_location(store)
    locations = [parquet]
    if frame != "absent":
        long_ago = _NOW - WORKING_TIER_RETENTION - timedelta(days=1)
        aged = _frame_location(store, stored_at=long_ago)
        locations.append(aged)
        if frame == "aged_out_bytes_gone":
            del store._objects[(aged.artifact_digest, Tier.working.value)]
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): locations}
    )
    writer = FakeArtifactReferenceWriter()

    with caplog.at_level(logging.WARNING, logger="datasets_api.doors.pin"):
        pinned_at = _pin_parquet(reader, store, writer, parquet.artifact_digest)

    assert pinned_at == _NOW
    if frame == "aged_out_bytes_gone":
        (record,) = caplog.records
        message = record.getMessage()
        assert f"artifact_digest={aged.artifact_digest}" in message
        assert "subject_id=spec_a" in message
        assert "content_version=cv1" in message
    else:
        assert not caplog.records
    pinned = {parquet.artifact_digest, *([locations[-1].artifact_digest] if frame_pinned else [])}
    assert store.list(tier=Tier.pinned) == pinned
    assert set(writer.records) == {
        (_PRINCIPAL, digest, "spec_a", "cv1", PIN_REFERENCE_KIND) for digest in pinned
    }


def test_a_pin_whose_named_bytes_are_gone_refuses_artifact_expired_and_spends_nothing():
    """The working copy aged out after the location was read, so the copy finds no object, and
    the units spent for it go back."""
    store = FakeArtifactStore()
    parquet = _parquet_location(store)
    del store._objects[(parquet.artifact_digest, Tier.working.value)]
    ledger = _ledger()

    with pytest.raises(ArtifactExpired):
        _pin_named(store, parquet, ledger, caller=_OPEN_CALLER)

    assert _spent(ledger) == {}


class _CopyForbiddenArtifactStore(FakeArtifactStore):
    def copy(self, digest, *, from_tier, to_tier):
        raise AssertionError("the budget refusal must come before any copy")


def _pin_named(store, location, ledger: LiveAllowanceLedger, *, caller: Caller | None = None):
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    writer = FakeArtifactReferenceWriter()
    pinned_at = pin_artifact(
        id="spec_a",
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal=_PRINCIPAL if caller is None else caller.key,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=writer,
        allowances=ledger,
        now=lambda: _NOW,
        caller=caller,
    )
    return pinned_at, writer


def test_a_copy_that_landed_and_raised_ends_with_one_set_of_units():
    """The bytes can land before a copy raises, so the failed attempt keeps its units, and the
    retry, which finds them stored, gives its own set back."""

    class _LandedThenRaised(FakeArtifactStore):
        failed = False

        def copy(self, digest, *, from_tier, to_tier):
            copied = super().copy(digest, from_tier=from_tier, to_tier=to_tier)
            if not self.failed:
                self.failed = True
                raise ArtifactWriteError(object_name=digest, digest=digest)
            return copied

    store = _LandedThenRaised()
    location = _store_and_locate(store, b"landed")
    ledger = _ledger()

    with pytest.raises(ArtifactWriteError):
        _pin_named(store, location, ledger)
    _pin_named(store, location, ledger)

    assert _spent(ledger) == {(BUDGET_KEY, "pin_budget"): 1}


@pytest.mark.parametrize(
    ("landed", "bucket_read", "units"),
    [
        pytest.param(False, None, {}, id="absent-refunds"),
        pytest.param(True, None, {(BUDGET_KEY, "pin_budget"): 1}, id="present-keeps"),
        pytest.param(
            False,
            ArtifactWriteError(object_name="d", digest="d"),
            {(BUDGET_KEY, "pin_budget"): 1},
            id="failed-read-keeps",
        ),
    ],
)
def test_a_copy_that_raises_is_judged_by_the_pinned_object(
    landed: bool, bucket_read: Exception | None, units: dict
):
    class _Raises(FakeArtifactStore):
        def copy(self, digest, *, from_tier, to_tier):
            if landed:
                super().copy(digest, from_tier=from_tier, to_tier=to_tier)
            raise ArtifactWriteError(object_name=digest, digest=digest)

        def exists(self, digest, *, tier):
            if bucket_read is not None:
                raise bucket_read
            return super().exists(digest, tier=tier)

    store = _Raises()
    location = _store_and_locate(store, b"judged")
    ledger = _ledger()

    with pytest.raises(ArtifactWriteError):
        _pin_named(store, location, ledger)

    assert _spent(ledger) == units


def test_a_unit_a_running_export_holds_refuses_a_pin_retryably_and_briefly():
    """The unit can come back, so its limit reads as capacity, and not as a day that spent it."""
    store = _CopyForbiddenArtifactStore()
    location = _store_and_locate(store, b"held")
    ledger = _ledger(pin_budget=1)
    claim = ledger.claim("exp_a", now=_NOW, expires_at=_NOW + timedelta(minutes=2))
    ledger.hold_claim(claim, [Spend(BUDGET_KEY, AllowanceKind.pin_budget)], _NOW)

    with pytest.raises(PinUnitHeld) as refused:
        _pin_named(store, location, ledger)

    assert (refused.value.code, refused.value.retryable) == ("service_unavailable", True)
    assert refused.value.headers == {"Retry-After": "30"}


def test_pin_at_the_daily_budget_refuses_before_copying_until_utc_midnight():
    store = _CopyForbiddenArtifactStore()
    location = _store_and_locate(store, b"over-budget")
    ledger = _ledger(pin_budget=2)
    ledger.spend_together([Spend(BUDGET_KEY, AllowanceKind.pin_budget)] * 2, _NOW)

    with pytest.raises(PinBudgetExhausted) as refused:
        _pin_named(store, location, ledger)

    assert (location.artifact_digest, Tier.pinned.value) not in store._objects
    # `_NOW` is a UTC midnight, so the budget resets a day later.
    assert refused.value.headers == {"Retry-After": "86400"}


@pytest.mark.parametrize(
    ("tier", "spent_at"),
    [
        # A repair of an already pinned artifact makes no copy, so the budget does not apply.
        (Tier.pinned, _NOW),
        # The budget is the UTC day's, so yesterday's copies no longer count.
        (Tier.working, _NOW - timedelta(seconds=1)),
    ],
    ids=["repair-of-pinned-artifact-at-budget", "yesterdays-copies"],
)
def test_pin_the_budget_admits_copies_and_records_a_reference(tier: Tier, spent_at: datetime):
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"within-budget", tier=tier)
    ledger = _ledger(pin_budget=2)
    ledger.spend_together([Spend(BUDGET_KEY, AllowanceKind.pin_budget)] * 2, spent_at)

    pinned_at, writer = _pin_named(store, location, ledger)

    assert pinned_at == _NOW
    assert (location.artifact_digest, Tier.pinned.value) in store._objects
    key = (_PRINCIPAL, location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[key] == _NOW


_OPEN_CALLER = Caller(kind="address", key="ip:" + "0" * 32)


@pytest.mark.parametrize(
    ("seed", "refused"),
    [(_OPEN_CALLER, True), (None, False)],
    ids=["open-callers-copy-counts", "grant-holders-copy-does-not"],
)
def test_the_open_budget_counts_only_open_callers_copies(seed: Caller | None, refused: bool):
    """A copy made today fills an open budget of one only when an open caller made it, and a
    refusal by that budget, retryable after the reset, spends neither the daily budget nor
    the caller's allowance."""
    ledger = _ledger(open_pin_budget=1)
    seeded = FakeArtifactStore()
    _pin_named(seeded, _store_and_locate(seeded, b"seed"), ledger, caller=seed)
    before = _spent(ledger)
    store = _CopyForbiddenArtifactStore() if refused else FakeArtifactStore()
    location = _store_and_locate(store, b"open-ceiling")

    if refused:
        with pytest.raises(OpenPinBudgetExhausted) as exhausted:
            _pin_named(store, location, ledger, caller=_OPEN_CALLER)
        assert _spent(ledger) == before
        assert exhausted.value.retryable is True
        assert exhausted.value.headers == {"Retry-After": "86400"}
    else:
        assert _pin_named(store, location, ledger, caller=_OPEN_CALLER)[0] == _NOW
        assert _spent(ledger)[(_OPEN_CALLER.key, AllowanceKind.pin.value)] == 1


@pytest.mark.parametrize("caller", [_OPEN_CALLER, None], ids=["open-caller", "grant-holder"])
def test_a_pin_racing_another_to_one_copy_spends_nothing(caller: Caller | None):
    """Two pins read the artifact in the working tier and both spend, but the second's copy
    finds the pinned bytes the first made. It gives its units back, so one copy costs one
    unit of each budget and of one allowance."""
    ledger = _ledger()
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"raced")
    _pin_named(store, location, ledger, caller=caller)
    first = _spent(ledger)

    pinned_at, writer = _pin_named(store, location, ledger, caller=caller)

    assert _spent(ledger) == first
    assert pinned_at == _NOW
    principal = _PRINCIPAL if caller is None else caller.key
    assert (principal, location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND) in (
        writer.records
    )


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
            allowances=_ledger(),
            now=lambda: _NOW,
        )

    assert writer.records == {}
    assert ARTIFACT_REFERENCE_WRITE_FAILED_MARKER not in caplog.text


def test_pin_door_wire_refuses_an_exhausted_budget_503_retryable_after_the_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DATASETS_PIN_DAILY_MAX` reaches the door. The refusal names the reset in `Retry-After`
    and is retryable once the reset passes."""
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
        allowances=_ledger(),
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
        allowances=_ledger(),
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
            allowances=_ledger(),
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


class _StoreFaultSpecReader(FakeSpecReader):
    """Raises `SpecRegistryStoreError` on the read that names a request's served ids."""

    def read_served_spec_ids(self, spec_id: str):
        raise SpecRegistryStoreError("spec registry read failed")


def test_pin_door_wire_maps_a_spec_registry_store_failure_to_retryable_503():
    """A registry store fault on the read that names the served ids answers retryable 503
    through the composed `datasets_surface` classifier."""
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=_StoreFaultSpecReader(),
        pin_allowed_principals=frozenset({_PRINCIPAL}),
    )

    resp = client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": "sha256:aaa"},
        headers={"x-dfo-principal": _PRINCIPAL},
    )

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_backpressure_refuses_503_retryable_when_executor_is_already_at_capacity() -> None:
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-backpressure-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    blocking_writer = BlockingFakeArtifactReferenceWriter()
    door = build_pin_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"read_max_concurrency": 1}
            ),
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=blocking_writer,
            allowances=_ledger(),
            now=lambda: _NOW,
        ),
        settings=DatasetsBuildSettings.from_env().model_copy(update={"read_max_concurrency": 1}),
        allowed_principals=frozenset({_PRINCIPAL}),
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
    monkeypatch.setattr(door_executor, "ContextCarryingExecutor", BrokenSubmitExecutor)
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-broken-pool-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_pin_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"read_max_concurrency": 1}
            ),
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=FakeArtifactReferenceWriter(),
            allowances=_ledger(),
            now=lambda: _NOW,
        ),
        settings=DatasetsBuildSettings.from_env().model_copy(update={"read_max_concurrency": 1}),
        allowed_principals=frozenset({_PRINCIPAL}),
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


def _build_client(
    *,
    allowed_principals=frozenset({_PRINCIPAL}),
    reference_writer=None,
    allowances=None,
    artifact_store=None,
    tier: Tier = Tier.working,
):
    """A client over the pin door alone, with the environment's settings read now."""
    store = artifact_store if artifact_store is not None else FakeArtifactStore()
    location = _store_and_locate(store, b"pin-me-over-http", tier=tier)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    settings = DatasetsBuildSettings.from_env()
    door = build_pin_door(
        deps=fake_deps_cell(
            settings=settings,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=reference_writer
            if reference_writer is not None
            else FakeArtifactReferenceWriter(),
            allowances=allowances if allowances is not None else offline_allowances(settings),
            now=lambda: _NOW,
        ),
        settings=settings,
        allowed_principals=allowed_principals,
    )
    return make_client_for_door(door), location


_HMAC_KEY = "test-caller-key"
_CLIENT_ADDRESS = "203.0.113.7"


@pytest.fixture
def caller_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure the caller digest key, and return the key of a caller at `_CLIENT_ADDRESS`."""
    monkeypatch.setenv("DFO_CALLER_HMAC_KEY", _HMAC_KEY)
    return address_key(_CLIENT_ADDRESS, hmac_key=_HMAC_KEY)


def _pin(client, location, headers=()):
    return client.post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers=[("x-forwarded-for", f"198.51.100.1, {_CLIENT_ADDRESS}"), *headers],
    )


@pytest.mark.parametrize(
    ("allowed_principals", "headers"),
    [
        (frozenset({_PRINCIPAL}), []),
        (frozenset({_PRINCIPAL}), [("x-dfo-principal", "some-other-caller")]),
        (
            frozenset({_PRINCIPAL}),
            [("x-dfo-principal", _PRINCIPAL), ("x-dfo-principal", "some-other-caller")],
        ),
        (frozenset({_PRINCIPAL}), [("x-dfo-principal", "presentation-mcp")]),
        (frozenset(), [("x-dfo-principal", _PRINCIPAL)]),
    ],
    ids=[
        "no-principal",
        "unlisted-principal",
        "repeated-principal",
        "publish-principal",
        "no-pin-principal-configured",
    ],
)
def test_a_caller_without_the_pin_grant_pins_under_its_caller_key_and_allowance(
    caller_key: str, allowed_principals: frozenset[str], headers: list[tuple[str, str]]
):
    writer = FakeArtifactReferenceWriter()
    ledger = _ledger(pin=50)
    client, location = _build_client(
        allowed_principals=allowed_principals, reference_writer=writer, allowances=ledger
    )

    resp = _pin(client, location, headers)

    assert resp.status_code == 200, resp.text
    assert {key[0] for key in writer.records} == {caller_key}
    assert _spent(ledger) == {
        (caller_key, "pin"): 1,
        (BUDGET_KEY, "pin_budget"): 1,
        (BUDGET_KEY, "open_pin_budget"): 1,
    }


def test_an_open_pin_without_a_caller_key_configured_refuses_503_before_copying(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DFO_CALLER_HMAC_KEY", raising=False)
    client, location = _build_client(artifact_store=_CopyForbiddenArtifactStore())

    resp = _pin(client, location)

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is False


def test_an_open_caller_past_its_allowance_refuses_429_until_utc_midnight(caller_key: str):
    client, location = _build_client(
        artifact_store=_CopyForbiddenArtifactStore(),
        allowances=_ledger(pin=0),
    )

    resp = _pin(client, location)

    assert resp.status_code == 429, resp.text
    body = resp.json()
    assert body["code"] == "allowance_exhausted"
    assert body["retryable"] is False
    # `_NOW` is a UTC midnight, so the allowance resets a day later.
    assert resp.headers["retry-after"] == "86400"


def test_an_open_callers_repin_of_a_pinned_artifact_spends_no_allowance(caller_key: str):
    writer = FakeArtifactReferenceWriter()
    client, location = _build_client(
        reference_writer=writer,
        tier=Tier.pinned,
        allowances=_ledger(pin=0),
    )

    resp = _pin(client, location)

    assert resp.status_code == 200, resp.text
    assert {key[0] for key in writer.records} == {caller_key}


def test_the_pin_grant_spends_no_allowance(caller_key: str):
    writer = FakeArtifactReferenceWriter()
    client, location = _build_client(
        reference_writer=writer,
        allowances=_ledger(pin=0),
    )

    resp = _pin(client, location, [("x-dfo-principal", _PRINCIPAL)])

    assert resp.status_code == 200, resp.text
    (principal,) = {key[0] for key in writer.records}
    assert principal.startswith("secret-sha256:")


def test_a_caller_the_website_vouches_for_pins_under_the_key_it_names(
    monkeypatch: pytest.MonkeyPatch,
):
    """The website's caller digest stands in for this service's own, which needs no key."""
    monkeypatch.setenv("DFO_CALLER_PRINCIPAL", "website-secret")
    named = "ip:" + "ab" * 16
    writer = FakeArtifactReferenceWriter()
    client, location = _build_client(reference_writer=writer)

    resp = _pin(
        client,
        location,
        [("x-dfo-caller-principal", "website-secret"), ("x-dfo-caller", named)],
    )

    assert resp.status_code == 200, resp.text
    assert {key[0] for key in writer.records} == {named}


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


def test_a_pin_reference_names_its_principal_by_a_stable_label_that_hides_the_secret():
    other = "second-pin-principal"
    writer = FakeArtifactReferenceWriter()
    client, location = _build_client(
        allowed_principals=frozenset({_PRINCIPAL, other}), reference_writer=writer
    )

    for secret in (_PRINCIPAL, other, _PRINCIPAL):
        resp = client.post(
            "/api/datasets/spec_a/cv1/pin",
            json={"schema_version": 1, "artifact_digest": location.artifact_digest},
            headers={"x-dfo-principal": secret},
        )
        assert resp.status_code == 200, resp.text

    principals = {key[0] for key in writer.records}
    assert len(principals) == 2
    assert not any(
        secret in principal for principal in principals for secret in (_PRINCIPAL, other)
    )


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
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=reference_writer,
            allowances=_ledger(),
            now=lambda: current_time[0],
        ),
        settings=DatasetsBuildSettings.from_env(),
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    client = make_client_for_door(door)
    body = {"schema_version": 1, "artifact_digest": location.artifact_digest}

    first = client.post(
        "/api/datasets/spec_a/cv1/pin", json=body, headers={"x-dfo-principal": _PRINCIPAL}
    )
    assert first.status_code == 200, first.text
    assert first.json()["pinned"]["pinned_at"] == _NOW.isoformat()

    (key,) = reference_writer.records
    assert key[1:] == (location.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
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
    """An id outside the `spec_` namespace gets the platform's routing 404."""
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
