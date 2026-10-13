"""`LivePresentationStore` implements `RevisionStore` over the `presentation` database.
The runtime role's `search_path` omits `presentation`, so every query qualifies the schema.

A `cmp_` or `cps_` id grants read of that draft to anyone holding it. Each component and
composition row also stores the SHA-256 digest of its write token, written once at create;
`write_token_digest()` returns it for the pipeline to compare, and the store checks no caller.
An unknown id raises `DraftNotFound`. A component's write-token digest is unique, and its create
may also store the SHA-256 of the create request; `prior_create()` finds a component by digest.

`allocate_component_id()` mints an ID without I/O or locking. The pipeline can pin a frame
before `begin_write()` opens its transaction. `begin_write` holds one transaction across
composition locking, component checks, append, and commit. It locks compositions before
components, making component and composition writes atomic. A create whose write-token
digest another component stores becomes `WriteTokenInUse`; Postgres holds the insert until a
concurrent create with the same digest commits or rolls back. Other generated-ID collisions
(SQLSTATE 23505) become `GeneratedIdCollision`. Revision-key collisions (SQLSTATE 23505)
become `EditConflict`. A revised component whose stored composition row is missing also
raises `EditConflict`. Other store failures become `PresentationStoreUnavailable`, which
records whether the driver failure meant a dead connection so the presentation pool discards it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from assembler.ports import (
    CompositionAppend,
    CompositionMembership,
    DraftNotFound,
    EditConflict,
    PriorCreate,
    RevisionRecord,
    WriteSubjectKind,
    WriteTokenInUse,
)
from dataset_store import DatasetStore
from dataset_store.store import Transaction
from presentation_store.ids import mint_component_id, mint_composition_id
from presentation_store.store_errors import (
    GeneratedIdCollision,
    PresentationStoreUnavailable,
    is_unique_violation,
    violated_constraint,
)

__all__ = ["LivePresentationStore"]

_REVISION_COLUMNS = (
    "revision, stored_intent, resolved_payload, catalog_version, schema_version, "
    "formatter_profile, series_coordinates, created_at"
)

# Migration 0008's unique index on `presentation.component (write_token_sha256)`.
_COMPONENT_WRITE_TOKEN_INDEX = "component_write_token_sha256_key"


def _next_composition_blocks(
    prior_blocks: list[dict], *, component_id: str, revision: int
) -> list[dict]:
    """Update an existing component block in place, or append a new block at the end."""
    updated = False
    next_blocks: list[dict] = []
    for block in prior_blocks:
        if block["component_id"] == component_id:
            next_blocks.append({"component_id": component_id, "revision": revision})
            updated = True
        else:
            next_blocks.append(block)
    if not updated:
        next_blocks.append({"component_id": component_id, "revision": revision})
    return next_blocks


def _row_to_record(component_id: str, row: Mapping[str, object]) -> RevisionRecord:
    return RevisionRecord(
        component_id=component_id,
        revision=cast(int, row["revision"]),
        stored_intent=cast(dict, row["stored_intent"]),
        resolved_payload=cast(dict, row["resolved_payload"]),
        catalog_version=cast(str, row["catalog_version"]),
        schema_version=cast(int, row["schema_version"]),
        formatter_profile=cast(dict, row["formatter_profile"]),
        series_coordinates=cast("list[dict] | None", row["series_coordinates"]),
        created_at=cast(datetime, row["created_at"]),
    )


class LivePresentationStore:
    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    # -- RevisionStore ----------------------------------------------------

    def read_current(self, *, component_id: str) -> RevisionRecord | None:
        return self._read(component_id=component_id, revision=None)

    def read_revision(self, *, component_id: str, revision: int) -> RevisionRecord | None:
        return self._read(component_id=component_id, revision=revision)

    def _read(self, *, component_id: str, revision: int | None) -> RevisionRecord | None:
        failure: PresentationStoreUnavailable | None = None
        result: RevisionRecord | None = None
        try:
            with self._store.transaction() as tx:
                if revision is None:
                    row = tx.fetch_one(
                        f"SELECT {_REVISION_COLUMNS} FROM presentation.component_revision "
                        "WHERE component_id = %s ORDER BY revision DESC LIMIT 1",
                        (component_id,),
                    )
                else:
                    row = tx.fetch_one(
                        f"SELECT {_REVISION_COLUMNS} FROM presentation.component_revision "
                        "WHERE component_id = %s AND revision = %s",
                        (component_id, revision),
                    )
                if row is None:
                    _require_component(tx, component_id=component_id)
                result = _row_to_record(component_id, row) if row is not None else None
        except DraftNotFound:
            raise
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(
                f"reading component_id {component_id!r} revision {revision!r} failed", exc
            )
        if failure is not None:
            raise failure
        return result

    def composition_of(self, *, component_id: str) -> CompositionMembership | None:
        failure: PresentationStoreUnavailable | None = None
        result: CompositionMembership | None = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "SELECT c.composition_id, cr.revision FROM presentation.component c "
                    "JOIN presentation.composition_revision cr "
                    "ON cr.composition_id = c.composition_id "
                    "WHERE c.component_id = %s ORDER BY cr.revision DESC LIMIT 1",
                    (component_id,),
                )
                if row is None:
                    _require_component(tx, component_id=component_id)
                else:
                    result = CompositionMembership(
                        composition_id=cast(str, row["composition_id"]),
                        revision=cast(int, row["revision"]),
                    )
        except DraftNotFound:
            raise
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(
                f"reading composition membership for component_id {component_id!r} failed", exc
            )
        if failure is not None:
            raise failure
        return result

    def write_token_digest(
        self, *, subject_kind: WriteSubjectKind, subject_id: str
    ) -> bytes | None:
        sql = (
            "SELECT write_token_sha256 FROM presentation.component WHERE component_id = %s"
            if subject_kind == "component"
            else "SELECT write_token_sha256 FROM presentation.composition WHERE composition_id = %s"
        )
        failure: PresentationStoreUnavailable | None = None
        row: Mapping[str, object] | None = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(sql, (subject_id,))
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(
                f"reading the write token digest for {subject_kind} {subject_id!r} failed", exc
            )
        if failure is not None:
            raise failure
        if row is None:
            raise DraftNotFound(subject_kind=subject_kind, subject_id=subject_id)
        digest = row["write_token_sha256"]
        return bytes(cast("bytes | memoryview", digest)) if digest is not None else None

    def prior_create(self, *, write_token_sha256: bytes) -> PriorCreate | None:
        failure: PresentationStoreUnavailable | None = None
        row: Mapping[str, object] | None = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "SELECT component_id, composition_id, create_request_sha256 "
                    "FROM presentation.component WHERE write_token_sha256 = %s",
                    (write_token_sha256,),
                )
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(
                "reading the component by write token digest failed", exc
            )
        if failure is not None:
            raise failure
        if row is None:
            return None
        request_digest = row["create_request_sha256"]
        return PriorCreate(
            component_id=cast(str, row["component_id"]),
            composition_id=cast("str | None", row["composition_id"]),
            create_request_sha256=(
                bytes(cast("bytes | memoryview", request_digest))
                if request_digest is not None
                else None
            ),
        )

    def allocate_component_id(self) -> str:
        return mint_component_id()

    @contextmanager
    def begin_write(
        self,
        *,
        component_id: str,
        is_create: bool,
        composition_id: str | None,
        write_token_sha256: bytes | None = None,
        composition_write_token_sha256: bytes | None = None,
        create_request_sha256: bytes | None = None,
    ) -> Generator[_LiveWriteContext, None, None]:
        """Keep preparation, the caller's append, and commit in one transaction. Wrap failures
        opening or committing it. Caller-body exceptions propagate after rollback.
        """
        tx_manager = self._store.transaction()
        entry_failure: PresentationStoreUnavailable | None = None
        tx: Transaction | None = None
        try:
            tx = tx_manager.__enter__()
        except Exception as exc:
            entry_failure = PresentationStoreUnavailable.from_driver_failure(
                f"opening a write transaction for component_id {component_id!r} failed", exc
            )
        if entry_failure is not None:
            raise entry_failure
        assert tx is not None

        try:
            ctx = self._prepare_write(
                tx,
                component_id=component_id,
                is_create=is_create,
                composition_id=composition_id,
                write_token_sha256=write_token_sha256,
                composition_write_token_sha256=composition_write_token_sha256,
                create_request_sha256=create_request_sha256,
            )
            yield ctx
        except BaseException:
            if tx_manager.__exit__(*sys.exc_info()):
                return
            raise
        else:
            commit_failure: PresentationStoreUnavailable | None = None
            try:
                tx_manager.__exit__(None, None, None)
            except Exception as exc:
                commit_failure = PresentationStoreUnavailable.from_driver_failure(
                    f"committing the write transaction for component_id {component_id!r} failed",
                    exc,
                )
            if commit_failure is not None:
                raise commit_failure

    def _prepare_write(
        self,
        tx: Transaction,
        *,
        component_id: str,
        is_create: bool,
        composition_id: str | None,
        write_token_sha256: bytes | None,
        composition_write_token_sha256: bytes | None,
        create_request_sha256: bytes | None,
    ) -> _LiveWriteContext:
        failure: PresentationStoreUnavailable | None = None
        result: _LiveWriteContext | None = None
        try:
            result = self._do_prepare_write(
                tx,
                component_id=component_id,
                is_create=is_create,
                composition_id=composition_id,
                write_token_sha256=write_token_sha256,
                composition_write_token_sha256=composition_write_token_sha256,
                create_request_sha256=create_request_sha256,
            )
        except (DraftNotFound, EditConflict, GeneratedIdCollision, WriteTokenInUse):
            raise
        except Exception as exc:
            failure = PresentationStoreUnavailable.from_driver_failure(
                f"opening a write for component_id {component_id!r} failed", exc
            )
        if failure is not None:
            raise failure
        assert result is not None
        return result

    def _do_prepare_write(
        self,
        tx: Transaction,
        *,
        component_id: str,
        is_create: bool,
        composition_id: str | None,
        write_token_sha256: bytes | None,
        composition_write_token_sha256: bytes | None,
        create_request_sha256: bytes | None,
    ) -> _LiveWriteContext:
        # Lock composition before component.
        resolved_composition_id: str | None = None
        if composition_id == "new":
            resolved_composition_id = mint_composition_id()
            _insert_generated_subject(
                tx,
                subject_kind="composition",
                sql=(
                    "INSERT INTO presentation.composition (composition_id, write_token_sha256) "
                    "VALUES (%s, %s)"
                ),
                params=(resolved_composition_id, composition_write_token_sha256),
            )
        elif composition_id is not None:
            row = tx.fetch_one(
                "SELECT composition_id FROM presentation.composition "
                "WHERE composition_id = %s FOR UPDATE",
                (composition_id,),
            )
            if row is None:
                raise DraftNotFound(subject_kind="composition", subject_id=composition_id)
            resolved_composition_id = composition_id
        elif not is_create:
            # Membership is fixed at create. Revising a component also advances its composition.
            component_row = tx.fetch_one(
                "SELECT composition_id FROM presentation.component WHERE component_id = %s",
                (component_id,),
            )
            if component_row is None:
                raise DraftNotFound(subject_kind="component", subject_id=component_id)
            persisted_composition_id = cast("str | None", component_row["composition_id"])
            if persisted_composition_id is not None:
                row = tx.fetch_one(
                    "SELECT composition_id FROM presentation.composition "
                    "WHERE composition_id = %s FOR UPDATE",
                    (persisted_composition_id,),
                )
                if row is None:
                    raise EditConflict(f"composition {persisted_composition_id!r} no longer exists")
                resolved_composition_id = persisted_composition_id

        if is_create:
            _insert_generated_subject(
                tx,
                subject_kind="component",
                sql=(
                    "INSERT INTO presentation.component (component_id, composition_id, "
                    "write_token_sha256, create_request_sha256) VALUES (%s, %s, %s, %s)"
                ),
                params=(
                    component_id,
                    resolved_composition_id,
                    write_token_sha256,
                    create_request_sha256,
                ),
            )
            current = None
        else:
            row = tx.fetch_one(
                f"SELECT {_REVISION_COLUMNS} FROM presentation.component_revision "
                "WHERE component_id = %s ORDER BY revision DESC LIMIT 1 FOR UPDATE",
                (component_id,),
            )
            current = _row_to_record(component_id, row) if row is not None else None

        return _LiveWriteContext(
            tx=tx,
            component_id=component_id,
            composition_id=resolved_composition_id,
            current=current,
        )

    def list_orphaned_presentation_references(
        self, *, referenced_component_ids: Sequence[str]
    ) -> tuple[str, ...]:
        """Compare IDs from dataset references with this database. Return missing components for
        reconciliation.
        """
        if not referenced_component_ids:
            return ()
        with self._store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT component_id FROM presentation.component WHERE component_id = ANY(%s)",
                (list(referenced_component_ids),),
            )
        existing = frozenset(cast(str, row["component_id"]) for row in rows)
        return tuple(
            component_id
            for component_id in referenced_component_ids
            if component_id not in existing
        )


def _insert_generated_subject(
    tx: Transaction, *, subject_kind: str, sql: str, params: Sequence[object]
) -> None:
    """Raise `WriteTokenInUse` after a unique violation on the component write-token index and
    `GeneratedIdCollision` after any other unique violation. Other errors reach `_prepare_write`
    for wrapping.
    """
    constraint: str | None = None
    collision = False
    try:
        tx.execute(sql, params)
    except Exception as exc:
        if not is_unique_violation(exc):
            raise
        collision = True
        constraint = violated_constraint(exc)
    if constraint == _COMPONENT_WRITE_TOKEN_INDEX:
        raise WriteTokenInUse
    if collision:
        raise GeneratedIdCollision(subject_kind=subject_kind)


def _wrap_unavailable[T](
    *, action_detail: str, fn: Callable[[], T], passthrough: tuple[type[Exception], ...] = ()
) -> T:
    """Wrap exceptions from `fn`, except those in `passthrough`, as
    `PresentationStoreUnavailable` after leaving `except` to keep driver context out.
    """
    failure: PresentationStoreUnavailable | None = None
    result: T | None = None
    try:
        result = fn()
    except passthrough:
        raise
    except Exception as exc:
        failure = PresentationStoreUnavailable.from_driver_failure(action_detail, exc)
    if failure is not None:
        raise failure
    return cast("T", result)


def _require_component(tx: Transaction, *, component_id: str) -> None:
    """Raise `DraftNotFound` when no component exists at `component_id`."""
    row = tx.fetch_one(
        "SELECT 1 FROM presentation.component WHERE component_id = %s", (component_id,)
    )
    if row is None:
        raise DraftNotFound(subject_kind="component", subject_id=component_id)


@dataclass
class _LiveWriteContext:
    """Write context with resolved IDs. `append()` writes revisions. Forward migrations
    update stored payloads.
    """

    tx: Transaction
    component_id: str
    composition_id: str | None
    current: RevisionRecord | None

    def append(
        self, *, record: RevisionRecord, composition_entry: CompositionAppend | None
    ) -> int | None:
        return _wrap_unavailable(
            action_detail=(
                f"appending revision {record.revision} for component_id "
                f"{record.component_id!r} failed"
            ),
            fn=lambda: self._do_append(record=record, composition_entry=composition_entry),
            passthrough=(EditConflict,),
        )

    def _do_append(
        self, *, record: RevisionRecord, composition_entry: CompositionAppend | None
    ) -> int | None:
        tx = self.tx
        try:
            tx.execute(
                "INSERT INTO presentation.component_revision "
                "(component_id, revision, stored_intent, resolved_payload, catalog_version, "
                "schema_version, formatter_profile, series_coordinates) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.component_id,
                    record.revision,
                    dict(record.stored_intent),
                    dict(record.resolved_payload),
                    record.catalog_version,
                    record.schema_version,
                    dict(record.formatter_profile),
                    (
                        json.dumps([dict(entry) for entry in record.series_coordinates])
                        if record.series_coordinates is not None
                        else None
                    ),
                ),
            )
        except Exception as exc:
            if is_unique_violation(exc):
                raise EditConflict(
                    f"component_id {self.component_id!r} revision {record.revision} already exists"
                ) from None
            raise

        composition_revision: int | None = None
        if composition_entry is not None:
            prior = tx.fetch_one(
                "SELECT revision, blocks FROM presentation.composition_revision "
                "WHERE composition_id = %s ORDER BY revision DESC LIMIT 1",
                (composition_entry.composition_id,),
            )
            prior_blocks = list(cast("list[dict]", prior["blocks"])) if prior is not None else []
            prior_revision = cast(int, prior["revision"]) if prior is not None else 0
            next_blocks = _next_composition_blocks(
                prior_blocks, component_id=self.component_id, revision=record.revision
            )
            composition_revision = prior_revision + 1
            tx.execute(
                "INSERT INTO presentation.composition_revision "
                "(composition_id, revision, blocks, actor) VALUES (%s, %s, %s, %s)",
                (
                    composition_entry.composition_id,
                    composition_revision,
                    json.dumps(next_blocks),
                    composition_entry.actor,
                ),
            )

        return composition_revision
