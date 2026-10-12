"""Datasets maintenance tasks, run as Cloud Run jobs under the runtime identity.

`datasets-maintenance <task>` runs one task and exits non-zero when it fails. Each task is a
subcommand registered in `main`.

`sweep` runs daily from Cloud Scheduler. It sweeps both artifact tiers with the read-only
`reconciliation.sweep` at the current time and writes one structured
`datasets_reconciliation_sweep` line per tier with its orphaned, missing, expired working and
unreferenced counts. Missing objects count only rows that should have bytes, so a working row
past `WORKING_TIER_RETENTION` counts as expired. For each non-empty orphaned, missing or
unreferenced leg the line also carries the sorted object names, digests for the unreferenced
leg, at most `SWEEP_NAMES_PER_LEG` of them, under `<leg>_sample`. It repairs nothing. It reads
`DATASETS_ARTIFACT_BUCKET_WORKING`, `DATASETS_ARTIFACT_BUCKET_PINNED` and the runtime database
settings `dataset_store.provider.shared_pool` reads.

`age-operations` runs daily. It first fails every abandoned row of `datasets.operation`
(`Operation.abandoned`) as retryable `internal_error`, so a row nobody polls leaves `pending` or
`running`. It then deletes `done` and `failed` rows last updated more than
`WORKING_TIER_RETENTION` ago, the window the capabilities document publishes as
`operation_retention_days`, and writes one `datasets_operations_aged` line with both counts. It
reads the runtime database settings and `DATASETS_OPERATION_DEADLINE_SECONDS`, whose default is
the API's own.

`backfill-frame1` runs once per environment. For every version linking a pinned frame1 document
and no pinned canonical frame, it reads the document, rebuilds its frame, checks the frame's
frame1 digest equals the document's, stores the canonical frame, copies it to the pinned tier
and, in one transaction, links it under `render.CANONICAL_FRAME_RENDERING` and copies every
`artifact_reference` row the frame1 held under the version to it, so the sweep counts the
canonical frame referenced exactly as the frame1 was. A second run finds nothing to convert. It
writes one `datasets_frame1_backfill_failed` line per version it cannot convert and one
`datasets_frame1_backfilled` line with the converted and failed counts, exits 1 when any version
failed, and reads the settings `sweep` reads.

`settle` runs on every deploy, after the services move to the new image. The previous image can
still write a pin reference holding the raw shared secret between the migration job and the
service move, so `settle` repeats the statements of migration 0026 that turn one into the
secret's label (`SETTLE_STATEMENTS`). Each touches only rows in the old shape, so a repeat
run changes nothing. It runs them in one transaction, then writes one `datasets_settled` line
counting the pin references still holding a raw secret and the version rows holding no facts.
Migration 0030's insert trigger fills the facts of a version row the previous image writes with
only `immutable_body`, so that count is above 0 only for a row whose body holds neither
`prepared_at` nor `input_content_versions`, which `settle` leaves as it is. When either count is
above 0 the line has `ERROR` severity and the task exits 1. It reads only the runtime database
settings.

Each task builds only what it uses from the environment the Job sets, so none reads a setting
of the API surfaces, such as `DFO_PUBLIC_SITE_URL`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import cast

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact_publication.jobs import emit
from artifact_store import (
    ArtifactStore,
    ArtifactStoreError,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from dataset_store import DatasetStore, Transaction
from dataset_store.provider import close_shared_pool, shared_pool
from dataset_store.snapshot import WORKING_TIER_RETENTION
from dataset_store.store import LiveDatasetStore
from datasets_api.config import DEFAULT_OPERATION_DEADLINE_SECONDS
from datasets_api.operation_store import (
    Clock,
    delete_terminal_operations,
    fail_abandoned_operations,
    utc_now,
)
from datasets_api.reconciliation import sweep
from datasets_api.render import CANONICAL_FRAME_RENDERING, render_and_store_canonical_frame
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveVersionRecordWriter, rendering_key
from datasets_api.versions import ArtifactRecord
from materialise.formats import (
    FRAME1_MEDIA_TYPE,
    Frame,
    frame_entity_columns,
    read_frame1_table,
    write_frame1,
)
from materialise.frame_build import build_frame, unit_cell_for_rebuild


class MaintenanceSettings(BaseSettings):
    """The two artifact buckets, read under the API's own `DATASETS_` names."""

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    artifact_bucket_working: str = Field(min_length=1)
    artifact_bucket_pinned: str = Field(min_length=1)


SWEEP_NAMES_PER_LEG = 20
"""The most object names, or digests, a sweep line carries for one leg."""


def run_sweep(*, store: DatasetStore, artifact_store: ArtifactStore, now: Clock = utc_now) -> None:
    """Sweep both tiers and emit one line per tier with each leg's count and, for each
    non-empty leg, its first `SWEEP_NAMES_PER_LEG` names in sorted order."""
    for report in sweep(store=store, artifact_store=artifact_store, now=now()):
        legs = {
            "orphaned_objects": report.orphaned_objects,
            "missing_objects": report.missing_objects,
            "unreferenced_pinned_digests": report.unreferenced_pinned_digests,
        }
        emit(
            "datasets_reconciliation_sweep",
            tier=report.tier.value,
            expired_working_locations=len(report.expired_working_locations),
            **{leg: len(names) for leg, names in legs.items()},
            **{
                f"{leg}_sample": sorted(names)[:SWEEP_NAMES_PER_LEG]
                for leg, names in legs.items()
                if names
            },
        )


def _sweep() -> int:
    return _run_over_artifacts("sweep", run_sweep)


def _run_over_artifacts(task: str, run: Callable[..., int | None]) -> int:
    """Run `run` over the runtime store and both artifact buckets, closing the pool after.
    Exits 1 when `run` returns a non-zero failure count."""
    try:
        settings = MaintenanceSettings()
    except ValidationError as exc:
        emit("datasets_maintenance_failed", severity="ERROR", task=task, error=str(exc))
        return 1
    store = LiveDatasetStore(connections=shared_pool())
    try:
        failures = run(
            store=store,
            artifact_store=LiveArtifactStore(
                buckets={
                    Tier.working: settings.artifact_bucket_working,
                    Tier.pinned: settings.artifact_bucket_pinned,
                },
                inventory=LiveArtifactInventory(store=store),
            ),
        )
    finally:
        close_shared_pool()
    return 1 if failures else 0


class OperationAgeingSettings(BaseSettings):
    """The operation deadline, read under the name the API and the worker read it by."""

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    operation_deadline_seconds: int = Field(default=DEFAULT_OPERATION_DEADLINE_SECONDS, ge=1)


def run_age_operations(*, store: DatasetStore, deadline: timedelta, now: Clock = utc_now) -> None:
    """Fail abandoned operations, delete terminal ones past the published retention, and emit
    both counts."""
    failed = fail_abandoned_operations(store=store, deadline=deadline, now=now)
    deleted = delete_terminal_operations(store=store, older_than=WORKING_TIER_RETENTION, now=now)
    emit(
        "datasets_operations_aged",
        failed_abandoned_operations=failed,
        deleted_operations=deleted,
    )


def _age_operations() -> int:
    try:
        settings = OperationAgeingSettings()
    except ValidationError as exc:
        emit("datasets_maintenance_failed", severity="ERROR", task="age-operations", error=str(exc))
        return 1
    try:
        run_age_operations(
            store=LiveDatasetStore(connections=shared_pool()),
            deadline=timedelta(seconds=settings.operation_deadline_seconds),
        )
    finally:
        close_shared_pool()
    return 0


SETTLE_STATEMENTS: tuple[str, ...] = (
    """UPDATE datasets.artifact_reference AS labelled
SET created_at = LEAST(labelled.created_at, raw.created_at),
    removed_at = CASE
        WHEN labelled.removed_at IS NULL OR raw.removed_at IS NULL THEN NULL
        ELSE GREATEST(labelled.removed_at, raw.removed_at)
    END
FROM datasets.artifact_reference AS raw
WHERE raw.reference_kind = 'pin'
  AND NOT starts_with(raw.principal, 'secret-sha256:')
  AND labelled.principal
      = 'secret-sha256:' || left(encode(sha256(convert_to(raw.principal, 'UTF8')), 'hex'), 16)
  AND labelled.artifact_digest = raw.artifact_digest
  AND labelled.subject_id = raw.subject_id
  AND labelled.content_version = raw.content_version
  AND labelled.reference_kind = raw.reference_kind""",
    """DELETE FROM datasets.artifact_reference AS raw
USING datasets.artifact_reference AS labelled
WHERE raw.reference_kind = 'pin'
  AND NOT starts_with(raw.principal, 'secret-sha256:')
  AND labelled.principal
      = 'secret-sha256:' || left(encode(sha256(convert_to(raw.principal, 'UTF8')), 'hex'), 16)
  AND labelled.artifact_digest = raw.artifact_digest
  AND labelled.subject_id = raw.subject_id
  AND labelled.content_version = raw.content_version
  AND labelled.reference_kind = raw.reference_kind""",
    """UPDATE datasets.artifact_reference
SET principal = 'secret-sha256:' || left(encode(sha256(convert_to(principal, 'UTF8')), 'hex'), 16)
WHERE reference_kind = 'pin'
  AND NOT starts_with(principal, 'secret-sha256:')""",
)
"""The three statements of `0026_artifact_reference_principal_label.sql`, verbatim, which
explains them."""


def run_settle(*, store: DatasetStore) -> int:
    """Run `SETTLE_STATEMENTS` in one transaction, emit the pin references holding a raw
    secret and the version rows holding no facts, at `ERROR` when either exists, and return
    their number."""
    with store.transaction() as tx:
        for statement in SETTLE_STATEMENTS:
            tx.execute(statement)
        factless = _count(
            tx,
            "SELECT count(*) AS count FROM datasets.version_record "
            "WHERE prepared_at IS NULL AND input_content_versions IS NULL",
        )
        raw = _count(
            tx,
            "SELECT count(*) AS count FROM datasets.artifact_reference "
            "WHERE reference_kind = 'pin' AND NOT starts_with(principal, 'secret-sha256:')",
        )
    unsettled = factless + raw
    emit(
        "datasets_settled",
        severity="ERROR" if unsettled else "INFO",
        factless_versions=factless,
        raw_pin_principals=raw,
    )
    return unsettled


def _count(tx: Transaction, sql: str) -> int:
    row = tx.fetch_one(sql)
    # `count(*)` without `GROUP BY` returns exactly one row.
    assert row is not None
    return int(cast(int, row["count"]))


def _settle() -> int:
    try:
        unsettled = run_settle(store=LiveDatasetStore(connections=shared_pool()))
    finally:
        close_shared_pool()
    return 1 if unsettled else 0


class _DiscardSink:
    """A writer sink that keeps no bytes, for a digest alone."""

    def write(self, data: bytes, /) -> int:
        return len(data)


def _pinned_frame1_versions(store: DatasetStore) -> list[tuple[str, str, str]]:
    """`(subject_id, content_version, frame1 digest)` of every version linking a pinned frame1
    document and no pinned canonical frame."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT va.subject_id AS subject_id, va.content_version AS content_version, "
            "va.artifact_digest AS artifact_digest "
            "FROM datasets.version_artifact va "
            "JOIN datasets.artifact_location al "
            "ON al.artifact_digest = va.artifact_digest AND al.tier = 'pinned' "
            "WHERE va.media_type = %s AND NOT EXISTS ("
            "SELECT 1 FROM datasets.version_artifact frame "
            "JOIN datasets.artifact_location pinned "
            "ON pinned.artifact_digest = frame.artifact_digest AND pinned.tier = 'pinned' "
            "WHERE frame.subject_id = va.subject_id "
            "AND frame.content_version = va.content_version "
            "AND frame.rendering_key = %s) "
            "ORDER BY va.subject_id, va.content_version",
            (FRAME1_MEDIA_TYPE, rendering_key(CANONICAL_FRAME_RENDERING)),
        )
    return [
        (str(row["subject_id"]), str(row["content_version"]), str(row["artifact_digest"]))
        for row in rows
    ]


def _frame_of_frame1(data: bytes, *, digest: str) -> Frame:
    """The frame a frame1 document holds, rebuilt through the grammar. Raises `ValueError` when
    the rebuilt frame's frame1 digest differs from `digest`, so a conversion never changes
    content."""
    table = read_frame1_table(data)
    frame = build_frame(
        role_keys=frame_entity_columns(table.columns),
        rows=[
            {
                column: unit_cell_for_rebuild(cell)
                for column, cell in zip(table.columns, row, strict=True)
                if column != "status"
            }
            for row in table.rows
        ],
    )
    rebuilt = write_frame1((frame,), sink=_DiscardSink()).digest
    if rebuilt != digest:
        raise ValueError(f"the frame rebuilt from {digest} writes frame1 digest {rebuilt}")
    return frame


def run_backfill_frame1(*, store: DatasetStore, artifact_store: ArtifactStore) -> int:
    """Link a pinned canonical frame, holding the frame1's references, under every version
    holding only a pinned frame1 document, emit the counts, and return the number of versions
    it could not convert."""
    writer = LiveVersionRecordWriter(store=store)
    converted = failed = 0
    for subject_id, content_version, digest in _pinned_frame1_versions(store):
        try:
            frame = _frame_of_frame1(
                artifact_store.get(artifact_digest=digest, tier=Tier.pinned), digest=digest
            )
            record = render_and_store_canonical_frame(
                frame_chunks=(frame,),
                subject_id=subject_id,
                content_version=content_version,
                artifact_store=artifact_store,
            )
            artifact_store.copy(record.artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
            writer.record_replacement_artifact(
                subject_id=subject_id,
                content_version=content_version,
                artifact=ArtifactRecord(
                    artifact_digest=record.artifact_digest,
                    rendering=record.rendering,
                    media_type=record.media_type,
                ),
                replaces=digest,
            )
        except (ArtifactStoreError, DatasetsStoreUnavailable, ValueError) as exc:
            failed += 1
            emit(
                "datasets_frame1_backfill_failed",
                severity="ERROR",
                subject_id=subject_id,
                content_version=content_version,
                artifact_digest=digest,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        converted += 1
    emit("datasets_frame1_backfilled", converted=converted, failed=failed)
    return failed


def _backfill_frame1() -> int:
    return _run_over_artifacts("backfill-frame1", run_backfill_frame1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datasets-maintenance")
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser(
        "sweep", help="report orphaned and missing artifact objects per tier"
    ).set_defaults(run=_sweep)
    tasks.add_parser(
        "age-operations",
        help="fail abandoned operations and delete done and failed ones past their retention",
    ).set_defaults(run=_age_operations)
    tasks.add_parser(
        "backfill-frame1", help="link a pinned canonical frame for every pinned frame1 document"
    ).set_defaults(run=_backfill_frame1)
    tasks.add_parser(
        "settle", help="label raw pin principals and count version rows holding no facts"
    ).set_defaults(run=_settle)
    args = parser.parse_args(argv)
    return args.run()


if __name__ == "__main__":
    sys.exit(main())
