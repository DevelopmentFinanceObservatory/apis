"""Postgres writers for immutable version records and mutable subject heads.

`LiveVersionRecordWriter` stores a version, collection members, and artifact
links in one transaction. Version and member inserts use `ON CONFLICT DO NOTHING`,
so records are immutable and retries are idempotent. Store errors become
`DatasetsStoreUnavailable` after the transaction exits.

`record_version` uses `RETURNING` to report whether its insert created the
row. Concurrent first writes have one winner, without a racy existence read.
`record_artifact` adds an idempotent rendering in its own transaction.

An artifact link keeps its digest while that digest is servable, and a new rendering under
the same key replaces it once it is not (`_insert_artifact`). Every link write and every pin
reference write takes `lock_version_links` first, so a pin cannot complete for a digest a
concurrent re-render is unlinking.

`now` defaults to UTC and can be injected to control `recorded_at`.
`rendering_key` hashes sorted JSON options, giving equivalent mappings the
same key. Definition retrieval uses the same function.

`subject_head` holds two timestamps. `confirmed_at` is when a probe or materialisation last
confirmed the stored `content_version` current. `materialised_at` is when a materialisation last
wrote it. `LiveHeadStore.advance_head` sets both, and updates a head only when its version matches
the caller's `expected_content_version`. Postgres applies the conflict
update under the row lock, so concurrent writers with the same expectation
have one winner. `confirm_head` updates only `confirmed_at` while the
version matches.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

from dataset_store import DatasetStore, Transaction
from dataset_store.snapshot import WORKING_TIER_RETENTION
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.versions import (
    ArtifactRecord,
    MalformedSignalError,
    MemberSignal,
    SubjectHead,
    VersionRecord,
    decode_signal,
    encode_signal,
)
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def rendering_key(rendering: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(rendering), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Two-key advisory locks live apart from the one-key space `operation_store` uses.
_VERSION_LINK_LOCK_CLASS = 0x4446_4F4C  # "DFOL" packed into an int4


def lock_version_links(tx: Transaction, *, subject_id: str, content_version: str) -> None:
    """Serialise this transaction with every other link write and pin reference write for one
    version, until it ends.

    Run it as its own statement before the statement it guards. Under READ COMMITTED the next
    statement then takes its snapshot after the lock, so it sees every row a previous holder
    committed. A link upsert sees the pinned location a completed pin wrote, and a pin sees a
    replacement a completed upsert wrote.
    """
    tx.execute(
        "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
        (_VERSION_LINK_LOCK_CLASS, json.dumps([subject_id, content_version])),
    )


def _insert_artifact(
    tx: Transaction,
    *,
    subject_id: str,
    content_version: str,
    artifact: ArtifactRecord,
    now: datetime,
) -> None:
    """Link `artifact` under its rendering key. An existing link is kept while its digest has
    a servable location, so a version lists the same bytes under one key for as long as they
    can be served. A link whose digest has none is replaced: re-rendered bytes can differ once
    the graph metadata a bundle reads has changed, and a stale link would otherwise hide them
    from snapshot reads and pins for good.

    The `WHERE` mirrors `select_servable_locations` in `dataset_store.snapshot`, which stays
    the one Python definition: a `pinned` location, or a `working` location no older than
    `WORKING_TIER_RETENTION` at `now` (`working_tier_expired`). Postgres evaluates it under
    the conflicting row's lock, so concurrent replacements of one stale link have one winner.

    `lock_version_links` runs first. A pin promotes a digest by copying it to the pinned tier
    and then writing its reference under the same lock, after checking the digest is still
    linked. The `WHERE` therefore reads either the pinned location of a pin that completed, or
    a link no pin has completed for yet, whose pin then refuses.
    """
    lock_version_links(tx, subject_id=subject_id, content_version=content_version)
    tx.execute(
        "INSERT INTO datasets.version_artifact "
        "(subject_id, content_version, artifact_digest, rendering_key, "
        "rendering, media_type) VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (subject_id, content_version, rendering_key) DO UPDATE SET "
        "artifact_digest = EXCLUDED.artifact_digest, "
        "rendering = EXCLUDED.rendering, "
        "media_type = EXCLUDED.media_type "
        "WHERE NOT EXISTS (SELECT 1 FROM datasets.artifact_location al "
        "WHERE al.artifact_digest = datasets.version_artifact.artifact_digest "
        "AND (al.tier = 'pinned' OR (al.tier = 'working' AND al.stored_at >= %s)))",
        (
            subject_id,
            content_version,
            artifact.artifact_digest,
            rendering_key(artifact.rendering),
            dict(artifact.rendering),
            artifact.media_type,
            now - WORKING_TIER_RETENTION,
        ),
    )


class LiveVersionRecordWriter:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        recorded_at = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        inserted_row: Mapping[str, object] | None = None
        try:
            with self._store.transaction() as tx:
                inserted_row = tx.fetch_one(
                    "INSERT INTO datasets.version_record "
                    "(subject_id, subject_kind, content_version, schema_version, "
                    "immutable_body, trace_id, recorded_at, base_content_version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (subject_id, content_version) DO NOTHING "
                    "RETURNING subject_id",
                    (
                        record.subject_id,
                        record.subject_kind,
                        record.content_version,
                        record.schema_version,
                        dict(record.immutable_body),
                        trace_id,
                        recorded_at,
                        record.base_content_version,
                    ),
                )
                for position, (member_spec_id, member_content_version) in enumerate(
                    record.member_versions
                ):
                    tx.execute(
                        "INSERT INTO datasets.version_member "
                        "(subject_id, content_version, position, member_spec_id, "
                        "member_content_version) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (subject_id, content_version, position) DO NOTHING",
                        (
                            record.subject_id,
                            record.content_version,
                            position,
                            member_spec_id,
                            member_content_version,
                        ),
                    )
                for artifact in record.artifacts:
                    _insert_artifact(
                        tx,
                        subject_id=record.subject_id,
                        content_version=record.content_version,
                        artifact=artifact,
                        now=recorded_at,
                    )
        except Exception as exc:
            # Save only safe diagnostics, then raise after leaving this exception handler.
            failure = (
                f"recording version subject_id {record.subject_id!r} content_version "
                f"{record.content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        return inserted_row is not None

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        now = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                _insert_artifact(
                    tx,
                    subject_id=subject_id,
                    content_version=content_version,
                    artifact=artifact,
                    now=now,
                )
        except Exception as exc:
            failure = (
                f"recording artifact subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)


class LiveHeadStore:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal: Sequence[MemberSignal],
        expected_content_version: str | None,
    ) -> SubjectHead | None:
        now = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "INSERT INTO datasets.subject_head "
                    "(subject_id, content_version, confirmed_at, materialised_at, signal) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (subject_id) DO UPDATE SET "
                    "content_version = EXCLUDED.content_version, "
                    "confirmed_at = EXCLUDED.confirmed_at, "
                    "materialised_at = EXCLUDED.materialised_at, "
                    "signal = EXCLUDED.signal "
                    "WHERE datasets.subject_head.content_version "
                    "IS NOT DISTINCT FROM %s "
                    "RETURNING content_version, confirmed_at, materialised_at, signal",
                    (
                        subject_id,
                        content_version,
                        now,
                        now,
                        encode_signal(signal),
                        expected_content_version,
                    ),
                )
        except Exception as exc:
            failure = (
                f"advancing head subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            # The expected version lost. Reread the head to find the winner.
            return None
        return SubjectHead(
            subject_id=subject_id,
            content_version=cast(str, row["content_version"]),
            confirmed_at=cast(datetime, row["confirmed_at"]),
            materialised_at=cast(datetime, row["materialised_at"]),
            signal=decode_signal(cast(Mapping[str, object], row["signal"])),
        )

    def confirm_head(self, *, subject_id: str, content_version: str) -> SubjectHead | None:
        confirmed_at = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "UPDATE datasets.subject_head SET confirmed_at = %s "
                    "WHERE subject_id = %s AND content_version = %s "
                    "RETURNING content_version, confirmed_at, materialised_at, signal",
                    (confirmed_at, subject_id, content_version),
                )
        except Exception as exc:
            failure = (
                f"confirming head subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            # No matching head exists at this version.
            return None
        return SubjectHead(
            subject_id=subject_id,
            content_version=cast(str, row["content_version"]),
            confirmed_at=cast(datetime, row["confirmed_at"]),
            materialised_at=cast(datetime, row["materialised_at"]),
            signal=decode_signal(cast(Mapping[str, object], row["signal"])),
        )

    def read_head(self, subject_id: str) -> SubjectHead | None:
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "SELECT content_version, confirmed_at, materialised_at, signal "
                    "FROM datasets.subject_head WHERE subject_id = %s",
                    (subject_id,),
                )
        except Exception as exc:
            failure = f"reading head subject_id {subject_id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            return None
        signal_failure: str | None = None
        signal: tuple[MemberSignal, ...] = ()
        try:
            signal = decode_signal(cast(Mapping[str, object], row["signal"]))
        except MalformedSignalError as exc:
            # Malformed stored data uses the same store-unavailable path as driver failures.
            signal_failure = f"reading head subject_id {subject_id!r} failed: {exc}"
        if signal_failure is not None:
            raise DatasetsStoreUnavailable(signal_failure)
        return SubjectHead(
            subject_id=subject_id,
            content_version=cast(str, row["content_version"]),
            confirmed_at=cast(datetime, row["confirmed_at"]),
            materialised_at=cast(datetime, row["materialised_at"]),
            signal=signal,
        )
