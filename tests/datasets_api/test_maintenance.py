"""`datasets-maintenance` dispatches tasks by subcommand. Its daily sweep reports reconciliation
counts per tier, `age-operations` fails abandoned operations and deletes terminal ones past
the published retention, `backfill-frame1` converts pinned frame1 documents to canonical
frames, and `settle` labels the raw pin principals migration 0026 missed and counts the version
rows holding no facts or the enrichment `kind`, each reporting structured log lines."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from artifact_store import ExistsRead, LiveArtifactInventory, LiveArtifactStore, Tier
from artifact_store.errors import ArtifactWriteError
from dataset_store.migrate import MIGRATIONS_DIR
from dataset_store.snapshot import WORKING_TIER_RETENTION, FrameLineage, LiveSnapshotReader
from datasets_api import maintenance
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceKind,
    ClaimState,
    CopyClaim,
    LiveAllowanceLedger,
    Spend,
)
from datasets_api.operation_store import LiveOperationStore
from datasets_api.reconciliation import sweep_tier
from datasets_api.render import CANONICAL_FRAME_RENDERING, render_and_store_canonical_frame
from datasets_api.retention_admission import sweep_budget, sweep_copy_claims
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from materialise.formats import (
    CANONICAL_FRAME_MEDIA_TYPE,
    FRAME1_MEDIA_TYPE,
    read_canonical_frame,
    read_frame1,
    write_frame1,
)
from materialise.frame_build import build_frame
from presentation_store import LiveShareStore, ShareNotFound, provider
from tests.artifact_store.conftest import FakeGCSClient
from tests.presentation_store.conftest import FakePresentationStore

from ._artifact_doubles import FakeArtifactStore
from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import FakeVersionRecordDatasetStore


def test_the_daily_run_logs_counts_and_capped_sorted_names_for_each_tier(capsys):
    """A working row past retention whose object is gone counts as expired, one inside retention
    counts as missing, and each non-empty leg carries its first names in sorted order."""
    now = datetime(2026, 6, 1, tzinfo=UTC)
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    pinned_missing = [f"sha256:missing-{index:02d}" for index in range(25, 0, -1)]
    working_rows = (
        ("sha256:expired-gone", now - WORKING_TIER_RETENTION - timedelta(seconds=1)),
        ("sha256:fresh-gone", now - WORKING_TIER_RETENTION),
    )
    try:
        orphan = artifact_store.put(b"orphan-bytes", tier=Tier.working)
        with store.transaction() as tx:
            for digest, tier, stored_at in (
                *((digest, "pinned", now) for digest in pinned_missing),
                *((digest, "working", stored_at) for digest, stored_at in working_rows),
            ):
                tx.execute(
                    "INSERT INTO datasets.artifact_location "
                    "(artifact_digest, tier, bucket, object_name, stored_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (digest, tier, "fake-bucket", digest, stored_at),
                )

        maintenance.run_sweep(store=store, artifact_store=artifact_store, now=lambda: now)
    finally:
        store.close()

    lines = {
        line["tier"]: line
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        if line["event"] == "datasets_reconciliation_sweep"
    }
    counts = {
        tier: (
            line["orphaned_objects"],
            line["missing_objects"],
            line["expired_working_locations"],
            line["unreferenced_pinned_digests"],
        )
        for tier, line in lines.items()
    }
    assert counts == {"working": (1, 1, 1, 0), "pinned": (0, 25, 0, 25)}
    assert lines["working"]["orphaned_objects_sample"] == [orphan.digest]
    assert lines["working"]["missing_objects_sample"] == ["sha256:fresh-gone"]
    assert "unreferenced_pinned_digests_sample" not in lines["working"]
    first_names = sorted(pinned_missing)[: maintenance.SWEEP_NAMES_PER_LEG]
    assert lines["pinned"]["missing_objects_sample"] == first_names
    assert lines["pinned"]["unreferenced_pinned_digests_sample"] == first_names
    assert "orphaned_objects_sample" not in lines["pinned"]


def _claims() -> LiveAllowanceLedger:
    return LiveAllowanceLedger(
        store=FakeVersionRecordDatasetStore(),
        limits={AllowanceKind.pin_budget: 5},
    )


_SPENT_AT = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _held(ledger: LiveAllowanceLedger, export: str = "exp_a") -> CopyClaim:
    claim = ledger.claim(export, now=_SPENT_AT, expires_at=_SPENT_AT + timedelta(minutes=2))
    return ledger.hold_claim(
        claim, [Spend(BUDGET_KEY, AllowanceKind.pin_budget)], clock=lambda: _SPENT_AT
    )


def _units(ledger: LiveAllowanceLedger) -> dict[str, int]:
    with ledger._store.transaction() as tx:
        rows = tx.fetch_all("SELECT kind, count FROM datasets.caller_allowance")
    return {row["kind"]: row["count"] for row in rows}


@pytest.mark.parametrize(
    ("digest", "pinned", "later", "units", "left"),
    [
        pytest.param(None, False, False, {}, False, id="died-mid-render"),
        pytest.param("sha256:a", False, False, {}, False, id="copy-never-landed"),
        pytest.param("sha256:a", True, False, {"pin_budget": 1}, True, id="copy-stands-same-day"),
        pytest.param("sha256:a", True, True, {"pin_budget": 1}, False, id="copy-stands-day-passed"),
    ],
)
def test_the_sweep_gives_back_the_units_of_a_claim_only_when_no_copy_stands(
    digest: str | None, pinned: bool, later: bool, units: dict[str, int], left: bool, capsys
):
    """A second run changes nothing, and a standing copy is read from the bucket, since its
    inventory row can be what failed."""
    now = _SPENT_AT + timedelta(days=1 if later else 0, minutes=5)
    ledger = _claims()
    artifact_store = FakeArtifactStore()
    held = _held(ledger)
    if digest is not None:
        ledger.begin_copy(held, digest, clock=lambda: _SPENT_AT, lease=timedelta(0))
    if pinned:
        artifact_store._objects[(digest, Tier.pinned.value)] = b"bytes"

    for _ in range(3):
        maintenance.run_copy_claim_sweep(
            store=ledger._store, artifact_store=artifact_store, now=lambda: now
        )

    assert _units(ledger) == units
    assert (ledger.read_claim("exp_a") is not None) is left


class _BrokenFor(FakeArtifactStore):
    """A bucket whose read of one digest fails."""

    def exists(self, digest: str, *, tier: Tier, read: object = None) -> bool:
        if digest == "sha256:broken":
            raise ArtifactWriteError(object_name=digest, digest=digest)
        return super().exists(digest, tier=tier)


def test_a_claim_that_fails_leaves_the_rest_settled_and_the_sweep_counts_it(capsys):
    now = _SPENT_AT + timedelta(minutes=5)
    ledger = _claims()
    ledger.begin_copy(
        _held(ledger, "exp_a"), "sha256:broken", clock=lambda: _SPENT_AT, lease=timedelta(0)
    )
    _held(ledger, "exp_b")

    failed = maintenance.run_copy_claim_sweep(
        store=ledger._store, artifact_store=_BrokenFor(), now=lambda: now
    )

    assert failed == 1
    assert ledger.read_claim("exp_a").state == ClaimState.copying
    assert ledger.read_claim("exp_b") is None
    assert _units(ledger) == {"pin_budget": 1}
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (line["settled_claims"], line["failed_claims"], line["deferred_claims"]) == (1, 1, 0)


def test_the_sweep_starts_no_claim_whose_read_could_outrun_its_budget():
    """One run ends before the next starts, whatever the bucket does: a `copying` claim
    starts only with room left for the whole long read."""
    ledger = _claims()
    _held(ledger, "exp_a")
    ledger.begin_copy(
        _held(ledger, "exp_b"), "sha256:a", clock=lambda: _SPENT_AT, lease=timedelta(0)
    )
    room = (sweep_budget(ledger.bounds) - ExistsRead.long.within).total_seconds()
    elapsed = iter([0.0, 0.0, room + 1])

    swept = sweep_copy_claims(
        allowances=ledger,
        artifacts=FakeArtifactStore(),
        now=_SPENT_AT + timedelta(minutes=5),
        monotonic=lambda: next(elapsed),
    )

    assert swept == (1, 0, 1)
    assert ledger.read_claim("exp_a") is None
    assert ledger.read_claim("exp_b").state == ClaimState.copying


def test_the_sweep_reserves_a_ledger_write_before_starting_any_claim():
    ledger = _claims()
    _held(ledger, "exp_a")
    elapsed = iter([0.0, sweep_budget(ledger.bounds).total_seconds() + 1])

    swept = sweep_copy_claims(
        allowances=ledger,
        artifacts=FakeArtifactStore(),
        now=_SPENT_AT + timedelta(minutes=5),
        monotonic=lambda: next(elapsed),
    )

    assert swept == (0, 0, 1)
    assert _units(ledger) == {"pin_budget": 1}


def test_the_sweep_counts_only_the_claims_it_moved(capsys):
    now = _SPENT_AT + timedelta(minutes=5)
    ledger = _claims()
    copying = ledger.begin_copy(
        _held(ledger), "sha256:a", clock=lambda: _SPENT_AT, lease=timedelta(0)
    )
    ledger.settle_copy(copying, landed=True, expires_at=_SPENT_AT)

    maintenance.run_copy_claim_sweep(
        store=ledger._store, artifact_store=FakeArtifactStore(), now=lambda: now
    )

    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (line["settled_claims"], line["failed_claims"]) == (0, 0)


@pytest.mark.parametrize("error", [None, RuntimeError("the ledger is down")], ids=["one", "all"])
def test_the_daily_sweep_writes_its_report_whatever_the_claims_did(error, capsys, monkeypatch):
    now = _SPENT_AT + timedelta(minutes=5)
    ledger = _claims()
    ledger.begin_copy(_held(ledger), "sha256:broken", clock=lambda: _SPENT_AT, lease=timedelta(0))
    if error is not None:

        def down(**_: object) -> int:
            raise error

        monkeypatch.setattr(maintenance, "run_copy_claim_sweep", down)

    failed = maintenance.run_sweep(
        store=ledger._store, artifact_store=_BrokenFor(), now=lambda: now
    )

    assert failed == 1
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events.count("datasets_reconciliation_sweep") == 2


def test_the_sweep_leaves_a_claim_with_a_running_lease():
    now = datetime(2026, 6, 1, 12, tzinfo=UTC)
    ledger = _claims()
    ledger.claim("exp_a", now=now, expires_at=now + timedelta(minutes=2))

    maintenance.run_copy_claim_sweep(
        store=ledger._store, artifact_store=FakeArtifactStore(), now=lambda: now
    )

    assert ledger.read_claim("exp_a") is not None


def test_a_missing_or_unknown_task_is_refused():
    for argv in ([], ["no-such-task"]):
        with pytest.raises(SystemExit) as excinfo:
            maintenance.main(argv)
        assert excinfo.value.code == 2


def test_age_operations_fails_abandoned_rows_and_deletes_terminal_ones_past_retention(capsys):
    """No poll or delivery touches these rows. The abandoned ones are failed at this run's time,
    so the same run's delete leaves them for the retention window. A running row stays unless
    its lease ended and its claim was the final delivery."""
    now = datetime(2026, 3, 1, tzinfo=UTC)
    published = timedelta(days=WORKING_TIER_RETENTION.days)
    deadline = timedelta(seconds=600)
    lease = timedelta(seconds=60)
    # Each row is (id, state, age, claim); a claim is (claimed_ago, final_delivery) under `lease`.
    rows = (
        ("op_old", "done", published + timedelta(seconds=1), None),
        ("op_fresh", "done", published, None),
        ("op_abandoned", "pending", published + timedelta(seconds=1), None),
        ("op_queued", "pending", deadline, None),
        ("op_live_lease", "running", timedelta(seconds=300), (lease / 6, True)),
        ("op_lease_ended_final", "running", timedelta(seconds=300), (lease * 2, True)),
        ("op_lease_ended_due", "running", timedelta(seconds=300), (lease * 2, False)),
    )
    fake = FakeOperationDatasetStore()
    try:
        with fake.transaction() as tx:
            for id, state, age, claim in rows:
                claimed_ago, final_delivery = claim or (None, None)
                tx.execute(
                    "INSERT INTO datasets.operation (id, request_identity, options, "
                    "coverage_stamp, state, estimate_rows, estimate_requests, created_at, "
                    "updated_at, submitted_at, claimed_at, lease_seconds, final_delivery) "
                    "VALUES (%s, '{}', '{}', 'stamp', %s, 1, 1, %s, %s, %s, %s, %s, %s)",
                    (
                        id,
                        state,
                        now - age,
                        now - age,
                        now - age,
                        None if claimed_ago is None else now - claimed_ago,
                        None if claim is None else int(lease.total_seconds()),
                        final_delivery,
                    ),
                )
        store = LiveOperationStore(store=fake, now=lambda: now)

        maintenance.run_age_operations(store=fake, deadline=deadline, now=lambda: now)

        stored = {id: store.get(id) for id, *_ in rows}
    finally:
        fake.close()

    assert {id: operation.state for id, operation in stored.items() if operation is not None} == {
        "op_fresh": "done",
        "op_abandoned": "failed",
        "op_queued": "pending",
        "op_live_lease": "running",
        "op_lease_ended_final": "failed",
        "op_lease_ended_due": "running",
    }
    for id in ("op_abandoned", "op_lease_ended_final"):
        failed = stored[id]
        assert failed is not None
        assert (failed.refusal_code, failed.retryable) == ("internal_error", True)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    aged = [line for line in lines if line["event"] == "datasets_operations_aged"]
    assert [(line["failed_abandoned_operations"], line["deleted_operations"]) for line in aged] == [
        (2, 1)
    ]


_TASKS = ["sweep", "age-operations", "backfill-frame1", "settle"]


@pytest.mark.parametrize("task", _TASKS)
def test_each_subcommand_dispatches_to_its_task(monkeypatch, task):
    ran: list[str] = []
    for name in _TASKS:
        attr = "_" + name.replace("-", "_")
        monkeypatch.setattr(maintenance, attr, lambda name=name: ran.append(name) or 0)

    assert maintenance.main([task]) == 0
    assert ran == [task]


# The environment the `infra` maintenance Job sets, and nothing else.
_JOB_ENV = {
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
    "DATASET_STORE_INSTANCE_CONNECTION_NAME": "project:region:instance",
    "DATASET_STORE_RUNTIME_DB_USER": "runtime@project.iam",
    "DATASETS_OPERATION_DEADLINE_SECONDS": "36000",
}


@pytest.fixture
def job_env(monkeypatch):
    """Only the Job's variables: every variable this process holds that the Job does not set,
    such as the `DFO_PUBLIC_SITE_URL` conftest sets, is removed."""
    for name in list(os.environ):
        if name.startswith(("DATASETS_", "DATASET_STORE_", "DFO_", "COVERAGE_", "ORIGIN_")):
            monkeypatch.delenv(name)
    for name, value in _JOB_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.mark.usefixtures("job_env")
def test_age_operations_runs_from_the_job_environment_alone(monkeypatch, capsys):
    fake = FakeOperationDatasetStore()
    closed: list[bool] = []
    monkeypatch.setattr(maintenance, "LiveDatasetStore", lambda connections: fake)
    monkeypatch.setattr(maintenance, "shared_pool", lambda: object())
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: closed.append(True))
    try:
        assert maintenance.main(["age-operations"]) == 0
    finally:
        fake.close()

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (line["failed_abandoned_operations"], line["deleted_operations"])
        for line in lines
        if line["event"] == "datasets_operations_aged"
    ] == [(0, 0)]
    assert closed == [True]


@pytest.mark.usefixtures("job_env")
def test_sweep_runs_from_the_job_environment_alone(monkeypatch, capsys):
    store = FakeVersionRecordDatasetStore()
    opened: list[dict[Tier, str]] = []

    def _artifact_store(*, buckets: dict[Tier, str], inventory):
        opened.append(buckets)
        return FakeArtifactStore()

    monkeypatch.setattr(maintenance, "LiveDatasetStore", lambda connections: store)
    monkeypatch.setattr(maintenance, "shared_pool", lambda: object())
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "LiveArtifactStore", _artifact_store)
    try:
        assert maintenance.main(["sweep"]) == 0
    finally:
        store.close()

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {line["tier"] for line in lines if line["event"] == "datasets_reconciliation_sweep"} == {
        "working",
        "pinned",
    }
    assert opened == [{Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"}]


class _RecordingStore:
    """A store that records each transaction's statements and answers every count with
    `count`."""

    def __init__(self, *, count: int = 0) -> None:
        self.transactions: list[list[str]] = []
        self._count = count

    @contextmanager
    def transaction(self):
        statements: list[str] = []
        self.transactions.append(statements)
        yield _RecordingTransaction(statements, count=self._count)


class _RecordingTransaction:
    def __init__(self, statements: list[str], *, count: int) -> None:
        self._statements = statements
        self._count = count

    def execute(self, sql: str, params: object = None) -> None:
        self._statements.append(sql)

    def fetch_one(self, sql: str, params: object = None) -> dict[str, object]:
        self._statements.append(sql)
        return {"count": self._count}


@pytest.mark.usefixtures("job_env")
@pytest.mark.parametrize(("left", "exit_code", "severity"), [(0, 0, "INFO"), (1, 1, "ERROR")])
def test_settle_runs_from_the_job_environment_alone_and_fails_on_rows_left_unsettled(
    monkeypatch, capsys, left, exit_code, severity
):
    """A version row holding no facts fails every read of it, a raw pin principal matches no
    caller, and a profile naming the enrichment `kind` loses it from the options document, so
    the deploy step running settle fails on any of them."""
    store = _RecordingStore(count=left)
    closed: list[bool] = []
    monkeypatch.setattr(maintenance, "LiveDatasetStore", lambda connections: store)
    monkeypatch.setattr(maintenance, "shared_pool", lambda: object())
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: closed.append(True))

    assert maintenance.main(["settle"]) == exit_code

    (statements,) = store.transactions
    assert statements[: len(maintenance.SETTLE_STATEMENTS)] == list(maintenance.SETTLE_STATEMENTS)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (
            line["severity"],
            line["factless_versions"],
            line["raw_pin_principals"],
            line["legacy_party_kind_profiles"],
        )
        for line in lines
        if line["event"] == "datasets_settled"
    ] == [(severity, left, left, left)]
    assert closed == [True]


def _statements(sql: str) -> list[str]:
    """`sql`'s statements with comments dropped and whitespace collapsed."""
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [" ".join(statement.split()) for statement in code.split(";") if statement.strip()]


def test_settle_repeats_the_statements_of_0026_for_raw_secrets_alone():
    """After its lock, each settle statement is a statement of migration 0026, whose window it
    closes, with the old shape narrowed from "not a label" to "neither a label nor a caller
    key", so they cannot drift apart otherwise. Version facts and the enrichment `kind` are
    counted only: the insert triggers of migrations 0030 and 0033 repair them."""
    migration = (MIGRATIONS_DIR / "0026_artifact_reference_principal_label.sql").read_text()
    expected = [
        statement.replace(
            f"NOT starts_with({column}, 'secret-sha256:')",
            f"{column} !~ '^(secret-sha256:[0-9a-f]{{16}}|ip:[0-9a-f]{{32}})$'",
        )
        for statement, column in zip(
            _statements(migration), ("raw.principal", "raw.principal", "principal"), strict=True
        )
    ]

    lock, *statements = maintenance.SETTLE_STATEMENTS
    assert lock == "LOCK TABLE datasets.artifact_reference IN SHARE ROW EXCLUSIVE MODE"
    assert [" ".join(statement.split()) for statement in statements] == expected


def _frame(value: float):
    return build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": recipient,
                "time_period": "2020",
                "value": value,
                "unit": {"currency": "EUR", "prices": "constant", "base_year": 2024},
            }
            for recipient in ("country/KEN", "country/TGO")
        ],
    )


class _Sink:
    def __init__(self) -> None:
        self.data = b""

    def write(self, data: bytes, /) -> int:
        self.data += bytes(data)
        return len(data)


def _record_frame1_version(
    store,
    artifact_store,
    *,
    content_version: str,
    value: float,
    pinned: bool,
    data: bytes | None = None,
):
    """A version of `spec_backfill` holding the frame1 document of `_frame(value)`, or `data`."""
    if data is None:
        sink = _Sink()
        write_frame1((_frame(value),), sink=sink)
        data = sink.data
    stored = artifact_store.put(data, tier=Tier.working)
    if pinned:
        artifact_store.copy(stored.digest, from_tier=Tier.working, to_tier=Tier.pinned)
    LiveVersionRecordWriter(store=store).record_version(
        VersionRecord(
            subject_id="spec_backfill",
            subject_kind="spec",
            content_version=content_version,
            schema_version=1,
            facts=FrameLineage(input_content_versions=[]),
            artifacts=(
                ArtifactRecord(
                    artifact_digest=stored.digest,
                    rendering={"format": "frame1"},
                    media_type=FRAME1_MEDIA_TYPE,
                ),
            ),
        )
    )


def _backfill_counts(capsys) -> list[tuple[int, int]]:
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return [
        (line["converted"], line["failed"])
        for line in lines
        if line["event"] == "datasets_frame1_backfilled"
    ]


def test_backfill_links_each_pinned_frame1_as_a_pinned_canonical_frame_once(capsys):
    store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=store),
            client=FakeGCSClient(),
        )
        _record_frame1_version(
            store, artifact_store, content_version="cv_pinned", value=1.0, pinned=True
        )
        _record_frame1_version(
            store, artifact_store, content_version="cv_working", value=2.0, pinned=False
        )

        maintenance.run_backfill_frame1(store=store, artifact_store=artifact_store)
        maintenance.run_backfill_frame1(store=store, artifact_store=artifact_store)

        reader = LiveSnapshotReader(store=store)
        frames = [
            location
            for location in reader.read_artifact_locations("spec_backfill", "cv_pinned")
            if location.rendering == CANONICAL_FRAME_RENDERING
        ]
        assert {(frame.media_type, frame.tier) for frame in frames} == {
            (CANONICAL_FRAME_MEDIA_TYPE, "working"),
            (CANONICAL_FRAME_MEDIA_TYPE, "pinned"),
        }
        data = artifact_store.get(artifact_digest=frames[0].artifact_digest, tier=Tier.pinned)
        assert read_canonical_frame(data) == _frame(1.0)
        assert not any(
            location.rendering == CANONICAL_FRAME_RENDERING
            for location in reader.read_artifact_locations("spec_backfill", "cv_working")
        )
    finally:
        store.close()

    assert _backfill_counts(capsys) == [(1, 0), (0, 0)]


def test_backfill_gives_each_canonical_frame_the_references_of_the_frame1_it_replaces():
    """The daily sweep counts a pinned digest referenced by its `artifact_reference` rows, so
    every reference the frame1 held, live or removed, is copied to the canonical frame."""
    store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=store),
            client=FakeGCSClient(),
        )
        _record_frame1_version(
            store, artifact_store, content_version="cv_pinned", value=1.0, pinned=True
        )
        reader = LiveSnapshotReader(store=store)
        (frame1,) = {
            location.artifact_digest
            for location in reader.read_artifact_locations("spec_backfill", "cv_pinned")
        }
        references = [
            ("secret-sha256:aaaa", "pin", "2026-01-01T00:00:00+00:00", None),
            (
                "presentation:chart",
                "presentation",
                "2026-02-01T00:00:00+00:00",
                "2026-03-01T00:00:00+00:00",
            ),
        ]
        with store.transaction() as tx:
            for principal, kind, created_at, removed_at in references:
                tx.execute(
                    "INSERT INTO datasets.artifact_reference (principal, artifact_digest, "
                    "subject_id, content_version, reference_kind, created_at, removed_at) "
                    "VALUES (%s, %s, 'spec_backfill', 'cv_pinned', %s, %s, %s)",
                    (principal, frame1, kind, created_at, removed_at),
                )

        maintenance.run_backfill_frame1(store=store, artifact_store=artifact_store)

        (frame,) = {
            location.artifact_digest
            for location in reader.read_artifact_locations("spec_backfill", "cv_pinned")
            if location.rendering == CANONICAL_FRAME_RENDERING
        }

        def _references(digest: str) -> list[tuple[object, ...]]:
            with store.transaction() as tx:
                rows = tx.fetch_all(
                    "SELECT principal, reference_kind, created_at, removed_at "
                    "FROM datasets.artifact_reference WHERE artifact_digest = %s "
                    "AND subject_id = 'spec_backfill' AND content_version = 'cv_pinned' "
                    "ORDER BY principal",
                    (digest,),
                )
            return [tuple(row.values()) for row in rows]

        assert len(_references(frame1)) == 2
        assert _references(frame) == _references(frame1)
        pinned = sweep_tier(
            store=store, artifact_store=artifact_store, tier=Tier.pinned, now=datetime.now(UTC)
        )
        assert pinned.unreferenced_pinned_digests == frozenset()
    finally:
        store.close()


def _rendering_another_frame(**kwargs: object):
    """`render_and_store_canonical_frame` storing another frame than the one it is given."""
    return render_and_store_canonical_frame(**{**kwargs, "frame_chunks": (_frame(9.0),)})


@pytest.mark.usefixtures("job_env")
def test_backfill_exits_non_zero_when_a_version_cannot_be_converted(monkeypatch, capsys):
    """A pinned frame1 whose canonical frame does not read back as its frame stays
    unconverted, and the execution fails, so nothing gated on the backfill reads it as
    complete."""
    store = FakeVersionRecordDatasetStore()
    artifact_store = LiveArtifactStore(
        buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
        inventory=LiveArtifactInventory(store=store),
        client=FakeGCSClient(),
    )
    monkeypatch.setattr(maintenance, "LiveDatasetStore", lambda connections: store)
    monkeypatch.setattr(maintenance, "shared_pool", lambda: object())
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "LiveArtifactStore", lambda **_: artifact_store)
    try:
        _record_frame1_version(
            store, artifact_store, content_version="cv_pinned", value=1.0, pinned=True
        )
        assert maintenance.main(["backfill-frame1"]) == 0

        _record_frame1_version(
            store, artifact_store, content_version="cv_broken", value=2.0, pinned=True
        )
        monkeypatch.setattr(
            maintenance, "render_and_store_canonical_frame", _rendering_another_frame
        )
        assert maintenance.main(["backfill-frame1"]) == 1
    finally:
        store.close()

    assert _backfill_counts(capsys) == [(1, 0), (0, 1)]


def test_backfill_counts_an_unreadable_frame1_as_failed_and_converts_the_rest(capsys):
    store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=store),
            client=FakeGCSClient(),
        )
        unreadable = json.loads(_frame1_bytes(_frame(2.0)))
        unreadable["rows"][0][unreadable["columns"].index("value")] = "abc"
        for content_version, data in (
            ("cv_a", None),
            ("cv_b", json.dumps(unreadable).encode()),
            ("cv_c", None),
        ):
            _record_frame1_version(
                store,
                artifact_store,
                content_version=content_version,
                value=1.0,
                pinned=True,
                data=data,
            )

        failed = maintenance.run_backfill_frame1(store=store, artifact_store=artifact_store)

        reader = LiveSnapshotReader(store=store)
        converted = [
            content_version
            for content_version in ("cv_a", "cv_b", "cv_c")
            if any(
                location.rendering == CANONICAL_FRAME_RENDERING
                for location in reader.read_artifact_locations("spec_backfill", content_version)
            )
        ]
    finally:
        store.close()

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert failed == 1
    assert converted == ["cv_a", "cv_c"]
    assert [
        line["content_version"]
        for line in lines
        if line["event"] == "datasets_frame1_backfill_failed"
    ] == ["cv_b"]


def _frame1_bytes(frame) -> bytes:
    sink = _Sink()
    write_frame1((frame,), sink=sink)
    return sink.data


def _without(column: str) -> Callable[[dict[str, Any]], None]:
    def drop(document: dict[str, Any]) -> None:
        index = document["columns"].index(column)
        del document["columns"][index]
        for row in document["rows"]:
            del row[index]

    return drop


def _first_unit_recipe(document: dict[str, Any]) -> None:
    index = document["columns"].index("unit")
    document["rows"][0][index] = {**document["rows"][0][index], "recipe": "r1"}


@pytest.mark.parametrize(
    "older",
    [
        pytest.param(_without("unit"), id="no-unit"),
        pytest.param(_without("measurementMethod"), id="no-measurement-method"),
        pytest.param(_without("status"), id="no-status"),
        pytest.param(_first_unit_recipe, id="a-unit-recipe"),
    ],
)
def test_backfill_converts_a_frame1_whose_own_frame1_differs(older, capsys):
    """A frame1 document written before a column it may lack, or holding a unit recipe, reads
    as a frame whose own frame1 differs from it, and converts to the canonical frame of that
    frame, in that frame's `unit` layout."""
    document = json.loads(_frame1_bytes(_frame(1.0)))
    older(document)
    data = json.dumps(document).encode()
    store = FakeVersionRecordDatasetStore()
    try:
        artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=store),
            client=FakeGCSClient(),
        )
        _record_frame1_version(
            store, artifact_store, content_version="cv_older", value=1.0, pinned=True, data=data
        )

        failed = maintenance.run_backfill_frame1(store=store, artifact_store=artifact_store)

        (frame,) = {
            location.artifact_digest
            for location in LiveSnapshotReader(store=store).read_artifact_locations(
                "spec_backfill", "cv_older"
            )
            if location.rendering == CANONICAL_FRAME_RENDERING
        }
        converted = read_canonical_frame(
            artifact_store.get(artifact_digest=frame, tier=Tier.pinned)
        )
    finally:
        store.close()

    assert failed == 0
    assert converted == read_frame1(data)
    assert _backfill_counts(capsys) == [(1, 0)]


def test_a_share_taken_down_twice_is_logged_once_taken_down_and_once_not_found(capsys):
    store = LiveShareStore(store=FakePresentationStore())
    share = store.create_share(
        kind="card", mode="live", title=None, write_token_sha256=None, cards=[], renderings={}
    )

    maintenance.run_take_down_share(share_store=store, share_id=share.id)
    maintenance.run_take_down_share(share_store=store, share_id=share.id)

    with pytest.raises(ShareNotFound):
        store.read_share(share_id=share.id)
    assert [
        (line["event"], line["share_id"])
        for line in map(json.loads, capsys.readouterr().out.splitlines())
    ] == [
        ("presentation_share_taken_down", share.id),
        ("presentation_share_take_down_not_found", share.id),
    ]


@pytest.mark.parametrize("share_id", ["short", "k3Qx8vT/", "k3Qx8vTa9"])
def test_take_down_share_refuses_an_id_that_is_not_a_share_id_before_opening_a_store(
    capsys, monkeypatch, share_id
):
    def no_pool():
        raise AssertionError("the store was opened")

    monkeypatch.setattr(provider, "shared_pool", no_pool)

    assert maintenance.main(["take-down-share", share_id]) == 1

    [line] = map(json.loads, capsys.readouterr().out.splitlines())
    assert (line["event"], line["severity"], line["share_id"]) == (
        "presentation_share_take_down_refused",
        "ERROR",
        share_id,
    )
