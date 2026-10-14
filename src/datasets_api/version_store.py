"""Postgres writers for immutable version records and mutable subject heads.

`LiveVersionRecordWriter` stores a version and its artifact links in one transaction. A version's
facts go in their own columns (`dataset_store.snapshot.VERSION_FACT_COLUMNS`), and `immutable_body`
holds the body those facts encode (`datasets_api.manifest.version_body`). Nothing in this image
reads the column; the writer keeps filling it for any image that decodes version bodies from it.
Version inserts use `ON CONFLICT DO NOTHING`, so records are immutable and retries are idempotent.
Two facts a version stored before versions held them are filled once, by a later write of the
same version: a published frame's `continuations` (`_write_description`), and the year members
and member nouns of a dataset version's `coverage` (`_fill_year_coverage`).
Every version insert goes through `_insert_version`, which also writes the version's description
record (`datasets.description_record`) in the same transaction when the record carries a
`VersionDescriber` (`_record_description`), so a failed record insert rolls the version row back.
Store errors become `DatasetsStoreUnavailable` after the transaction exits, except a describer's
`DescriptionDiverges`, which propagates as raised.

`record_version` uses `RETURNING` to report whether its insert created the
row. Concurrent first writes have one winner, without a racy existence read.
`record_artifact` adds an idempotent rendering in its own transaction, and
`record_replacement_artifact` does the same while copying the references of the digest it
replaces.
`write_version` stores a serve's whole write in one transaction: the inventory
rows its artifacts owe (`artifact_store.inventory.write_entry`), the record
and its links, the rendered links and collapsed version when the stored record
holds the body they were rendered from, and the served version read back.
`LiveVersionLister.list_versions` reads a subject's versions by `recorded_at`, newest first,
each with the renderings of the files linked under it. The second statement selects and orders
those files as `LiveSnapshotReader.read_artifact_locations` does, so a rendering read from the
list is the one a read of that version's locations gives.

An artifact link keeps its digest while that digest is servable, and a new rendering under
the same key replaces it once it is not (`_insert_artifact`). Every link write and every pin
reference write takes `lock_version_links` first, so a pin cannot complete for a digest a
concurrent re-render is unlinking.

`now` defaults to UTC and can be injected to control `recorded_at`.
`rendering_key` hashes sorted JSON options, giving equivalent mappings the
same key. Definition retrieval uses the same function.

`subject_head` holds two timestamps. `confirmed_at` is when a freshness check or materialisation
last confirmed the stored `content_version` current. `materialised_at` is when a materialisation
last wrote it. `LiveHeadStore.advance_head` inserts the run's `datasets.materialisation` row and, in
the same transaction, sets both timestamps, the run's id and its dependencies, updating a head only
while it names the run the caller observed (`expected_materialisation_id`, `None` for no head).
Postgres applies the conflict update under the row lock, so concurrent writers with the same
expectation have one winner, including two that produce the version the head already holds, and a
head this store writes names the run that produced its version. A losing writer keeps its run's row.
A worker run passes its claim as a `HeadFence`, and the same transaction first takes a share lock on
that `datasets.operation` row and checks it is still `running` at the claim's attempt
(`_holds_fence`). The lock conflicts with a claim's update, so a claim either commits first and the
advance raises `HeadFenceLost` having written nothing, or waits until the advance commits.
`confirm_head` updates only `confirmed_at` while the version matches. `read_head` reports no
dependencies for a head whose run made another version, as a head another writer advanced without
naming its run does. The `signal` column keeps its default on every head this store writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from artifact_store.errors import ArtifactRecordConflictError
from artifact_store.inventory import write_entry
from dataset_store import DatasetStore, Transaction
from dataset_store.snapshot import (
    VERSION_FACT_COLUMNS,
    WORKING_TIER_RETENTION,
    ManifestFacts,
    StoredArtifactLocation,
    VersionState,
    version_fact_values,
    version_state_in,
)
from datasets_api.description_store import (
    fill_late_fields,
    insert_description,
    read_description,
)
from datasets_api.manifest import version_body
from datasets_api.manifest_models import year_coverage_counted
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.versions import (
    RENDERED_FACT_FIELDS,
    ArtifactRecord,
    DescriptionDiverges,
    HeadFence,
    HeadFenceLost,
    LateFields,
    ListedVersion,
    MaterialisationRecord,
    SubjectHead,
    VersionDescriber,
    VersionRecord,
    VersionWrite,
    WrittenVersion,
    log_description_build_failure,
)
from description.facts import DescriptionRecord
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics
from materialise.formats import FRAME1_MEDIA_TYPE

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def rendering_key(rendering: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(rendering), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def recorded_frame_sha256(
    recorded: str | None, locations: Sequence[StoredArtifactLocation]
) -> str | None:
    """The frame hash (`dfo_frame_grammar.frame_sha256`) a stored version holds without reading
    its frame: `recorded`, the hash on its record (`StoredVersion.frame_sha256`), else the
    digest of the frame1 document among `locations`, its links, since a frame1 document is the
    frame's canonical bytes. `None` when it holds neither, as a version linking only a
    canonical frame recorded before frame hashes does: a Parquet digest depends on its
    encoding."""
    if recorded is not None:
        return recorded
    frame1 = next((loc for loc in locations if loc.media_type == FRAME1_MEDIA_TYPE), None)
    return None if frame1 is None else frame1.artifact_digest.removeprefix("sha256:")


# Two-key advisory locks live apart from the one-key space `operation_store` uses.
_VERSION_LINK_LOCK_CLASS = 0x4446_4F4C  # "DFOL" packed into an int4


def lock_version_links(tx: Transaction, *, subject_id: str, content_version: str) -> None:
    """Serialise this transaction with every other link write, pin reference write and
    coverage fill (`_fill_year_coverage`) for one version, until it ends.

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


_FACT_COLUMN_LIST = ", ".join(VERSION_FACT_COLUMNS)
_RECORD_PLACEHOLDERS = ", ".join(["%s"] * (10 + len(VERSION_FACT_COLUMNS)))

_INSERT_VERSION_SQL = (
    "INSERT INTO datasets.version_record "
    "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
    "trace_id, recorded_at, base_content_version, frame_grammar_version, frame_sha256, "
    f"{_FACT_COLUMN_LIST}) "
    f"VALUES ({_RECORD_PLACEHOLDERS}) "
    "ON CONFLICT (subject_id, content_version) DO NOTHING "
    "RETURNING subject_id"
)


def _insert_version(
    tx: Transaction, record: VersionRecord, *, trace_id: str | None, now: datetime
) -> bool:
    """Insert `record` and link its artifacts. `True` when this call created the row."""
    inserted_row = tx.fetch_one(
        _INSERT_VERSION_SQL,
        (
            record.subject_id,
            record.subject_kind,
            record.content_version,
            record.schema_version,
            version_body(record),
            trace_id,
            now,
            record.base_content_version,
            record.frame_grammar_version,
            record.frame_sha256,
            *version_fact_values(record.facts),
        ),
    )
    for artifact in record.artifacts:
        _insert_artifact(
            tx,
            subject_id=record.subject_id,
            content_version=record.content_version,
            artifact=artifact,
            now=now,
        )
    created = inserted_row is not None
    if not created:
        _fill_year_coverage(tx, record)
    if record.describer is not None:
        _record_description(tx, record, created=created, now=now)
    return created


def _fill_year_coverage(tx: Transaction, record: VersionRecord) -> None:
    """Store `record`'s `coverage` on its version, already on record, when the stored one was
    never counted (`year_coverage_counted`) and `record`'s was. The column and the `coverage`
    of `immutable_body` take it whole: the two describe one frame, so they differ only in
    the year members and the member nouns.

    The read and the write run under the version's link lock, so of two writers the second
    reads the first's coverage, finds it counted and writes nothing, and both read back one
    record.
    """
    facts = record.facts
    if not isinstance(facts, ManifestFacts) or not year_coverage_counted(facts.coverage):
        return
    key = (record.subject_id, record.content_version)
    lock_version_links(tx, subject_id=record.subject_id, content_version=record.content_version)
    row = tx.fetch_one(
        "SELECT coverage, immutable_body FROM datasets.version_record "
        "WHERE subject_id = %s AND content_version = %s",
        key,
    )
    stored = None if row is None else row["coverage"]
    if not isinstance(stored, Mapping) or year_coverage_counted(
        cast("Mapping[str, object]", stored)
    ):
        return
    assert row is not None
    coverage = dict(facts.coverage)
    body = row["immutable_body"]
    tx.execute(
        "UPDATE datasets.version_record SET coverage = %s, immutable_body = %s "
        "WHERE subject_id = %s AND content_version = %s",
        (
            json.dumps(coverage),
            json.dumps({**body, "coverage": coverage} if isinstance(body, Mapping) else body),
            *key,
        ),
    )


def _record_description(
    tx: Transaction, record: VersionRecord, *, created: bool, now: datetime
) -> None:
    """Write the version's description record once, and settle the one it holds.

    A version without a record gets the one `record.describer` builds from the version as
    stored, so a write that found the row already there describes the stored facts and frame,
    the frame named by its hash as `recorded_frame_sha256` reads it in this transaction, the
    rule the publish door matched it by. A version holding a record, including one a concurrent
    writer inserted first, gets the late fields the describer settles, and, when they fill any
    member, the continuations they carry on its version row (`LateFields.continuations`). A
    describer that raises anything but `DescriptionDiverges` has a
    defect: it is logged and the version is stored without touching its record, before any
    record statement runs, so the transaction stays sound. When a concurrent writer filled them first, the describer
    settles again against what it filled, so a contradiction still raises.

    A stored row that fails validation reads as no record (`read_description`), so the version
    gets the record its describer builds, which replaces the row (`insert_description`). A row
    of a later schema version, a later image's, reads as no record too, and stands. Late
    fields that would leave the stored record invalid are a describer defect too: the
    `ValueError` `fill_late_fields` raises comes before its `UPDATE`, so it is logged and the
    record is left as it was. `DescriptionDiverges` still propagates: a stored record that
    states other step facts than the posted ones refuses the write, as a diverging lineage
    does.
    """
    try:
        _write_description(tx, record, created=created, now=now)
    except ValueError:
        log_description_build_failure(record)


def _write_description(
    tx: Transaction, record: VersionRecord, *, created: bool, now: datetime
) -> None:
    describer = record.describer
    assert describer is not None
    key = {"id": record.subject_id, "content_version": record.content_version}
    stored = None if created else read_description(tx, **key)
    if created:
        version = record
    else:
        state = _state(tx, record)
        version = replace(
            record,
            facts=state.version.facts,
            frame_sha256=recorded_frame_sha256(state.version.frame_sha256, state.locations),
        )
    if stored is None:
        base = (
            None
            if record.base_content_version is None
            else read_description(
                tx, id=record.subject_id, content_version=record.base_content_version
            )
        )
        try:
            built = describer.describe(version, base=base)
        except Exception:
            log_description_build_failure(version)
            return
        if built is None or insert_description(tx, built, recorded_at=now):
            return
        # A concurrent writer recorded the version first, or a later image holds a record of
        # a later schema version, which this image neither reads nor settles.
        stored = read_description(tx, **key)
        if stored is None:
            return
    late = _settled(describer, stored, version=version)
    if late is None:
        return
    filled = fill_late_fields(
        tx, **key, values=late.values, derived=late.derived, arrived=now, basis=late.basis
    )
    if filled and late.continuations is not None:
        tx.execute(
            "UPDATE datasets.version_record SET continuations = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps(list(late.continuations)), record.subject_id, record.content_version),
        )
    if not filled:
        refilled = read_description(tx, **key)
        assert refilled is not None
        _settled(describer, refilled, version=version)


def _settled(
    describer: VersionDescriber, stored: DescriptionRecord, *, version: VersionRecord
) -> LateFields | None:
    """`describer.settle`, with a defect logged and treated as nothing to fill.
    `DescriptionDiverges` propagates."""
    try:
        return describer.settle(stored, version=version)
    except DescriptionDiverges:
        raise
    except Exception:
        log_description_build_failure(version)
        return None


def _state(tx: Transaction, record: VersionRecord) -> VersionState:
    state = version_state_in(
        tx, subject_id=record.subject_id, content_version=record.content_version
    )
    assert state is not None, "version record missing inside the transaction that wrote it"
    return state


def _holds_rendered_facts(state: VersionState, record: VersionRecord) -> bool:
    """Whether the stored version carries the facts `record`'s artifacts render."""
    stored, written = state.version.facts, record.facts
    return all(
        getattr(stored, field, None) == getattr(written, field, None)
        for field in RENDERED_FACT_FIELDS
    )


class LiveVersionRecordWriter:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        recorded_at = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        created = False
        try:
            with self._store.transaction() as tx:
                created = _insert_version(tx, record, trace_id=trace_id, now=recorded_at)
        except DescriptionDiverges:
            raise
        except Exception as exc:
            # Save only safe diagnostics, then raise after leaving this exception handler.
            failure = (
                f"recording version subject_id {record.subject_id!r} content_version "
                f"{record.content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        return created

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        now = self._now()
        record = write.record
        failure: str | None = None
        sqlstate: str | None = None
        written: WrittenVersion | None = None
        try:
            with self._store.transaction() as tx:
                for entry in write.inventory:
                    write_entry(tx, entry, now=now)
                created = _insert_version(tx, record, trace_id=trace_id, now=now)
                rendered = created or _holds_rendered_facts(_state(tx, record), record)
                served = record
                if rendered:
                    for artifact in write.rendered:
                        _insert_artifact(
                            tx,
                            subject_id=record.subject_id,
                            content_version=record.content_version,
                            artifact=artifact,
                            now=now,
                        )
                    if write.collapsed is not None:
                        served = write.collapsed
                        created = _insert_version(tx, served, trace_id=trace_id, now=now)
                state = _state(tx, served)
                written = WrittenVersion(created=created, rendered=rendered, state=state)
        except ArtifactRecordConflictError as exc:
            exc.subject_id = record.subject_id
            exc.content_version = record.content_version
            raise
        except DescriptionDiverges:
            raise
        except Exception as exc:
            failure = (
                f"writing version subject_id {record.subject_id!r} content_version "
                f"{record.content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        assert written is not None
        return written

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> str:
        now = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        linked: Mapping[str, object] | None = None
        try:
            with self._store.transaction() as tx:
                _insert_artifact(
                    tx,
                    subject_id=subject_id,
                    content_version=content_version,
                    artifact=artifact,
                    now=now,
                )
                # The upsert keeps a link whose digest is servable, so the linked digest is read.
                linked = tx.fetch_one(
                    "SELECT artifact_digest FROM datasets.version_artifact "
                    "WHERE subject_id = %s AND content_version = %s AND rendering_key = %s",
                    (subject_id, content_version, rendering_key(artifact.rendering)),
                )
        except Exception as exc:
            failure = (
                f"recording artifact subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        assert linked is not None, "artifact link missing inside the transaction that wrote it"
        return cast(str, linked["artifact_digest"])

    def record_replacement_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        replaces: str,
    ) -> None:
        """Link `artifact` and copy every `artifact_reference` row the version holds on the
        `replaces` digest, live or removed, to it, in one transaction. A converted artifact then
        counts as referenced exactly as the one it replaces, by `reconciliation.sweep` and by
        anything else that reads references per digest. A rerun changes nothing."""
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
                tx.execute(
                    "INSERT INTO datasets.artifact_reference "
                    "(principal, artifact_digest, subject_id, content_version, reference_kind, "
                    "created_at, removed_at) "
                    "SELECT principal, %s, subject_id, content_version, reference_kind, "
                    "created_at, removed_at FROM datasets.artifact_reference "
                    "WHERE artifact_digest = %s AND subject_id = %s AND content_version = %s "
                    "ON CONFLICT (principal, artifact_digest, subject_id, content_version, "
                    "reference_kind) DO NOTHING",
                    (artifact.artifact_digest, replaces, subject_id, content_version),
                )
        except Exception as exc:
            failure = (
                f"recording replacement artifact subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)


class LiveVersionLister:
    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    def list_versions(self, subject_id: str) -> list[ListedVersion]:
        failure: str | None = None
        sqlstate: str | None = None
        rows: list[Mapping[str, object]] = []
        linked: list[Mapping[str, object]] = []
        try:
            with self._store.transaction() as tx:
                rows = tx.fetch_all(
                    "SELECT content_version, base_content_version "
                    "FROM datasets.version_record WHERE subject_id = %s "
                    # `content_version` orders versions recorded at one instant.
                    "ORDER BY recorded_at DESC, content_version DESC",
                    (subject_id,),
                )
                linked = tx.fetch_all(
                    "SELECT va.content_version AS content_version, va.rendering AS rendering "
                    "FROM datasets.version_artifact va "
                    "JOIN datasets.artifact_inventory ai "
                    "ON ai.artifact_digest = va.artifact_digest "
                    "JOIN datasets.artifact_location al "
                    "ON al.artifact_digest = va.artifact_digest "
                    "WHERE va.subject_id = %s "
                    "ORDER BY COALESCE((va.rendering ->> 'statistics') = 'series', false), "
                    "va.artifact_digest, va.rendering_key, al.tier",
                    (subject_id,),
                )
        except Exception as exc:
            failure = f"listing versions subject_id {subject_id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        renderings: dict[str, list[Mapping[str, object]]] = {}
        for link in linked:
            renderings.setdefault(cast(str, link["content_version"]), []).append(
                cast("Mapping[str, object]", link["rendering"])
            )
        return [
            ListedVersion(
                content_version=cast(str, row["content_version"]),
                base_content_version=cast("str | None", row["base_content_version"]),
                renderings=tuple(renderings.get(cast(str, row["content_version"]), ())),
            )
            for row in rows
        ]


_HEAD_COLUMNS = "content_version, confirmed_at, materialised_at, materialisation_id, dependencies"

# A head's columns with its fingerprint kept only while its run produced its version.
_JUDGED_HEAD_SELECT = (
    "SELECT h.content_version, h.confirmed_at, h.materialised_at, h.materialisation_id, "
    "CASE WHEN m.content_version = h.content_version THEN h.dependencies END AS dependencies "
    "FROM datasets.subject_head h "
    "LEFT JOIN datasets.materialisation m ON m.id = h.materialisation_id "
    "WHERE h.subject_id = %s"
)


def _insert_materialisation(
    tx: Transaction, materialisation: MaterialisationRecord, *, now: datetime
) -> int:
    row = tx.fetch_one(
        "INSERT INTO datasets.materialisation "
        '(subject_id, content_version, materialised_at, "trigger", materialiser_build, '
        "frame_grammar_version, code_index_stamp, origin_catalogue_build_id, coverage_stamp, "
        "graph_generation, dependencies, selection, conversion, trace_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (
            materialisation.subject_id,
            materialisation.content_version,
            now,
            materialisation.trigger,
            materialisation.materialiser_build,
            materialisation.frame_grammar_version,
            materialisation.code_index_stamp,
            materialisation.origin_catalogue_build_id,
            materialisation.coverage_stamp,
            materialisation.graph_generation,
            dict(materialisation.dependencies),
            dict(materialisation.selection),
            dict(materialisation.conversion) if materialisation.conversion is not None else None,
            materialisation.trace_id,
        ),
    )
    assert row is not None  # an INSERT ... RETURNING without a conflict clause returns its row
    return cast(int, row["id"])


def _holds_fence(tx: Transaction, fence: HeadFence) -> bool:
    """Whether the fence's operation is still `running` at its attempt, holding a share lock on
    the row until the transaction ends. Under READ COMMITTED a lock that waited on a claim's
    update re-checks the committed row, so a claimed row reads as lost."""
    row = tx.fetch_one(
        "SELECT 1 AS held FROM datasets.operation "
        "WHERE id = %s AND state = 'running' AND attempt_count = %s FOR SHARE",
        (fence.operation_id, fence.attempt),
    )
    return row is not None


def _head(subject_id: str, row: Mapping[str, object]) -> SubjectHead:
    """The head a `_HEAD_COLUMNS` row holds."""
    return SubjectHead(
        subject_id=subject_id,
        content_version=cast(str, row["content_version"]),
        confirmed_at=cast(datetime, row["confirmed_at"]),
        materialised_at=cast(datetime, row["materialised_at"]),
        materialisation_id=cast("int | None", row["materialisation_id"]),
        dependencies=cast("Mapping[str, object] | None", row["dependencies"]),
    )


class LiveHeadStore:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def advance_head(
        self,
        *,
        materialisation: MaterialisationRecord,
        expected_materialisation_id: int | None,
        fence: HeadFence | None = None,
    ) -> SubjectHead | None:
        now = self._now()
        subject_id = materialisation.subject_id
        content_version = materialisation.content_version
        failure: str | None = None
        sqlstate: str | None = None
        fence_lost = False
        row = None
        try:
            with self._store.transaction() as tx:
                # The fence is the first statement, so the operation row is locked before any
                # write and the transaction writes nothing when the claim has moved on.
                fence_lost = fence is not None and not _holds_fence(tx, fence)
                if not fence_lost:
                    materialisation_id = _insert_materialisation(tx, materialisation, now=now)
                    row = tx.fetch_one(
                        "INSERT INTO datasets.subject_head "
                        "(subject_id, content_version, confirmed_at, materialised_at, "
                        "materialisation_id, dependencies) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (subject_id) DO UPDATE SET "
                        "content_version = EXCLUDED.content_version, "
                        "confirmed_at = EXCLUDED.confirmed_at, "
                        "materialised_at = EXCLUDED.materialised_at, "
                        "materialisation_id = EXCLUDED.materialisation_id, "
                        "dependencies = EXCLUDED.dependencies "
                        "WHERE datasets.subject_head.materialisation_id "
                        "IS NOT DISTINCT FROM %s "
                        f"RETURNING {_HEAD_COLUMNS}",
                        (
                            subject_id,
                            content_version,
                            now,
                            now,
                            materialisation_id,
                            dict(materialisation.dependencies),
                            expected_materialisation_id,
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
        if fence is not None and fence_lost:
            raise HeadFenceLost(fence)
        if row is None:
            # The head names a run other than `expected_materialisation_id`. The caller rereads it.
            return None
        return _head(subject_id, row)

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
                    f"RETURNING {_HEAD_COLUMNS}",
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
        return _head(subject_id, row)

    def read_head(self, subject_id: str) -> SubjectHead | None:
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(_JUDGED_HEAD_SELECT, (subject_id,))
        except Exception as exc:
            failure = f"reading head subject_id {subject_id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            return None
        return _head(subject_id, row)
