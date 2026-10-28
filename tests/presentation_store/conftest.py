"""SQL-shaped presentation store fakes with rollback and failure controls.

The fake recognizes the statements issued by `LivePresentationStore` and snapshots state per transaction.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from dataset_store.errors import DatasetStoreError


@dataclass
class FakePresentationState:
    components: dict[str, dict] = field(default_factory=dict)
    component_revisions: dict[str, list[dict]] = field(default_factory=dict)
    compositions: set[str] = field(default_factory=set)
    composition_write_token_digests: dict[str, bytes | None] = field(default_factory=dict)
    composition_revisions: dict[str, list[dict]] = field(default_factory=dict)
    # `presentation.share` rows by id, `share_revision` rows by share id, and `share_slot` rows
    # by (share id, revision). `share_slot_check` rows by (share id, slot, component id,
    # component revision), and `served_frame` rows in insertion order.
    shares: dict[str, dict] = field(default_factory=dict)
    share_revisions: dict[str, list[dict]] = field(default_factory=dict)
    share_slots: dict[tuple[str, int], list[dict]] = field(default_factory=dict)
    slot_checks: dict[tuple[str, int, str, int], dict] = field(default_factory=dict)
    served_frames: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> FakePresentationState:
        return FakePresentationState(
            components=copy.deepcopy(self.components),
            component_revisions=copy.deepcopy(self.component_revisions),
            compositions=set(self.compositions),
            composition_write_token_digests=dict(self.composition_write_token_digests),
            composition_revisions=copy.deepcopy(self.composition_revisions),
            shares=copy.deepcopy(self.shares),
            share_revisions=copy.deepcopy(self.share_revisions),
            share_slots=copy.deepcopy(self.share_slots),
            slot_checks=copy.deepcopy(self.slot_checks),
            served_frames=copy.deepcopy(self.served_frames),
        )

    def restore_from(self, other: FakePresentationState) -> None:
        self.components = other.components
        self.component_revisions = other.component_revisions
        self.compositions = other.compositions
        self.composition_write_token_digests = other.composition_write_token_digests
        self.composition_revisions = other.composition_revisions
        self.shares = other.shares
        self.share_revisions = other.share_revisions
        self.share_slots = other.share_slots
        self.slot_checks = other.slot_checks
        self.served_frames = other.served_frames


class FakeTransaction:
    """Dispatch the store's SQL by shape and record every attempted statement."""

    def __init__(
        self,
        state: FakePresentationState,
        *,
        record: Callable[[str], None],
        fail_before: Callable[[str, Sequence[object]], bool] | None = None,
    ) -> None:
        self._state = state
        self._record = record
        self._fail_before = fail_before

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._dispatch(sql, params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self._dispatch(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        return self._dispatch(sql, params)

    def _dispatch(self, sql: str, params: Sequence[object]) -> list[dict]:
        self._record(sql)
        if self._fail_before is not None and self._fail_before(sql, params):
            raise RuntimeError("injected failure")
        s = self._state

        if sql.startswith("INSERT INTO presentation.composition ("):
            composition_id, write_token_sha256 = params
            if composition_id in s.compositions:
                raise _unique_violation("composition_pkey")
            s.compositions.add(composition_id)
            s.composition_write_token_digests[composition_id] = write_token_sha256
            s.composition_revisions.setdefault(composition_id, [])
            return []

        if sql.startswith("SELECT write_token_sha256 FROM presentation.composition WHERE"):
            (composition_id,) = params
            if composition_id not in s.compositions:
                return []
            return [{"write_token_sha256": s.composition_write_token_digests.get(composition_id)}]

        if sql.startswith(
            "SELECT component_id, composition_id, create_request_sha256 "
            "FROM presentation.component WHERE write_token_sha256 = %s"
        ):
            (write_token_sha256,) = params
            return [
                {
                    "component_id": component_id,
                    "composition_id": component["composition_id"],
                    "create_request_sha256": component["create_request_sha256"],
                }
                for component_id, component in s.components.items()
                if component["write_token_sha256"] == write_token_sha256
            ]

        if sql.startswith("SELECT write_token_sha256 FROM presentation.component WHERE"):
            (component_id,) = params
            component = s.components.get(component_id)
            if component is None:
                return []
            return [{"write_token_sha256": component["write_token_sha256"]}]

        if "FROM presentation.composition WHERE" in sql:
            (composition_id,) = params
            return [{"composition_id": composition_id}] if composition_id in s.compositions else []

        if sql.startswith("SELECT composition_id FROM presentation.component WHERE"):
            (component_id,) = params
            component = s.components.get(component_id)
            return (
                [{"composition_id": component["composition_id"]}] if component is not None else []
            )

        if sql.startswith("INSERT INTO presentation.component ("):
            component_id, composition_id, write_token_sha256, create_request_sha256 = params
            if component_id in s.components:
                raise _unique_violation("component_pkey")
            if write_token_sha256 is not None and any(
                component["write_token_sha256"] == write_token_sha256
                for component in s.components.values()
            ):
                raise _unique_violation("component_write_token_sha256_key")
            s.components[component_id] = {
                "composition_id": composition_id,
                "write_token_sha256": write_token_sha256,
                "create_request_sha256": create_request_sha256,
            }
            s.component_revisions.setdefault(component_id, [])
            return []

        if sql.startswith("SELECT 1 FROM presentation.component WHERE component_id = %s"):
            (component_id,) = params
            return [{"1": 1}] if component_id in s.components else []

        if sql.startswith("SELECT c.composition_id, cr.revision FROM presentation.component c"):
            (component_id,) = params
            component = s.components.get(component_id)
            composition_id = component.get("composition_id") if component is not None else None
            if composition_id is None:
                return []
            revisions = s.composition_revisions.get(composition_id, [])
            if not revisions:
                return []
            latest = max(revisions, key=lambda row: row["revision"])
            return [{"composition_id": composition_id, "revision": latest["revision"]}]

        if sql.startswith(
            "SELECT revision, stored_intent, resolved_payload, catalog_version, schema_version, "
            "formatter_profile, series_coordinates, created_at, frame_rendering "
            "FROM presentation.component_revision"
        ):
            revisions = s.component_revisions.get(params[0], [])
            if "AND revision = %s" in sql:
                _component_id, revision = params
                return [row for row in revisions if row["revision"] == revision]
            if not revisions:
                return []
            return [max(revisions, key=lambda row: row["revision"])]

        if sql.startswith("INSERT INTO presentation.component_revision ("):
            (
                component_id,
                revision,
                stored_intent,
                resolved_payload,
                catalog_version,
                schema_version,
                formatter_profile,
                series_coordinates,
                frame_rendering,
            ) = params
            revisions = s.component_revisions.setdefault(component_id, [])
            if any(row["revision"] == revision for row in revisions):
                raise _unique_violation("component_revision_pkey")
            revisions.append(
                {
                    "revision": revision,
                    "stored_intent": stored_intent,
                    "resolved_payload": resolved_payload,
                    "catalog_version": catalog_version,
                    "schema_version": schema_version,
                    "formatter_profile": formatter_profile,
                    "series_coordinates": (
                        json.loads(series_coordinates) if series_coordinates is not None else None
                    ),
                    # The column default, `now()` at insert.
                    "created_at": datetime.now(UTC),
                    "frame_rendering": frame_rendering,
                }
            )
            return []

        if sql.startswith("SELECT revision, blocks FROM presentation.composition_revision"):
            (composition_id,) = params
            revisions = s.composition_revisions.get(composition_id, [])
            if not revisions:
                return []
            return [max(revisions, key=lambda row: row["revision"])]

        if sql.startswith("INSERT INTO presentation.composition_revision ("):
            composition_id, revision, blocks, actor = params
            revisions = s.composition_revisions.setdefault(composition_id, [])
            revisions.append({"revision": revision, "blocks": json.loads(blocks), "actor": actor})
            return []

        if sql.startswith(
            "SELECT component_id FROM presentation.component WHERE component_id = ANY"
        ):
            (component_ids,) = params
            return [
                {"component_id": component_id}
                for component_id in component_ids
                if component_id in s.components
            ]

        if (shares := self._dispatch_share(sql, params)) is not None:
            return shares

        raise AssertionError(f"FakeTransaction has no handler for statement: {sql!r}")

    def _dispatch_share(self, sql: str, params: Sequence[object]) -> list[dict] | None:
        """The statements of `presentation_store.shares.LiveShareStore`, or `None` for others."""
        s = self._state

        if sql.startswith("INSERT INTO presentation.share (id, "):
            assert "ON CONFLICT (id) DO NOTHING RETURNING id" in sql
            if len(params) == 6:
                share_id, kind, mode, title, write_token_sha256, payload = params
                owner, created_at = None, datetime.now(UTC)
            else:
                share_id, kind, mode, title, write_token_sha256, owner, payload, created_at = params
            share_id = cast(str, share_id)
            if share_id in s.shares:
                return []
            s.shares[share_id] = {
                "id": share_id,
                "kind": kind,
                "mode": mode,
                "title": title,
                "write_token_sha256": write_token_sha256,
                "owner": owner,
                "payload": json.loads(cast(str, payload)) if payload is not None else None,
                "created_at": created_at,
                "deleted_at": None,
            }
            return [{"id": share_id}]

        if sql.startswith("SELECT kind, created_at FROM presentation.share WHERE id = %s"):
            (share_id,) = params
            row = s.shares.get(share_id)
            return [] if row is None else [{"kind": row["kind"], "created_at": row["created_at"]}]

        if sql.startswith("UPDATE presentation.share SET mode = 'snapshot' WHERE id = %s"):
            (share_id,) = params
            s.shares[share_id]["mode"] = "snapshot"
            return []

        if sql.startswith(
            "SELECT id, kind, mode, title, write_token_sha256, owner, payload, created_at "
            "FROM presentation.share WHERE id = %s AND deleted_at IS NULL"
        ):
            (share_id,) = params
            row = s.shares.get(share_id)
            if row is None or row["deleted_at"] is not None:
                return []
            return [{key: value for key, value in row.items() if key != "deleted_at"}]

        if sql.startswith("UPDATE presentation.share SET deleted_at = now() WHERE id = %s"):
            (share_id,) = params
            row = s.shares.get(share_id)
            if row is None or row["deleted_at"] is not None:
                return []
            row["deleted_at"] = datetime.now(UTC)
            return [{"id": share_id}]

        if sql.startswith("INSERT INTO presentation.share_revision (share_id, revision"):
            share_id, revision = cast(str, params[0]), cast(int, params[1])
            created_at = params[2] if len(params) == 3 else datetime.now(UTC)
            if share_id not in s.shares:
                raise _foreign_key_violation("share_revision_share_id_fkey")
            revisions = s.share_revisions.setdefault(share_id, [])
            if any(row["revision"] == revision for row in revisions):
                raise _unique_violation("share_revision_pkey")
            revisions.append({"revision": revision, "created_at": created_at})
            return []

        if sql.startswith("SELECT revision, created_at FROM presentation.share_revision"):
            (share_id,) = params
            revisions = s.share_revisions.get(share_id, [])
            return [max(revisions, key=lambda row: row["revision"])] if revisions else []

        if sql.startswith("INSERT INTO presentation.share_slot ("):
            share_id, revision, slot, component_id, component_revision, spec_id, cv = params
            share_id, revision = cast(str, share_id), cast(int, revision)
            if not any(row["revision"] == revision for row in s.share_revisions.get(share_id, [])):
                raise _foreign_key_violation("share_slot_share_id_revision_fkey")
            slots = s.share_slots.setdefault((share_id, revision), [])
            if any(row["slot"] == slot for row in slots):
                raise _unique_violation("share_slot_pkey")
            slots.append(
                {
                    "slot": slot,
                    "component_id": component_id,
                    "component_revision": component_revision,
                    "frame_spec_id": spec_id,
                    "frame_content_version": cv,
                }
            )
            return []

        if sql.startswith(
            "SELECT slot, component_id, component_revision, frame_spec_id, "
            "frame_content_version FROM presentation.share_slot"
        ):
            share_id, revision = params
            return sorted(s.share_slots.get((share_id, revision), []), key=lambda row: row["slot"])

        if sql.startswith("SELECT frame_spec_id, frame_content_version, freshness, checked_at"):
            key = cast("tuple[str, int, str, int]", tuple(params))
            row = s.slot_checks.get(key)
            return [] if row is None else [dict(row)]

        if sql.startswith("UPDATE presentation.share_slot_check SET checked_at = %s"):
            now, *key, due_before = params
            row = s.slot_checks.get(cast("tuple[str, int, str, int]", tuple(key)))
            if row is None or cast(datetime, row["checked_at"]) > cast(datetime, due_before):
                return []
            row["checked_at"] = now
            return [{"slot": key[1]}]

        if sql.startswith("INSERT INTO presentation.share_slot_check ("):
            share_id, slot, component_id, revision, spec_id, cv, freshness, checked_at = params
            if share_id not in s.shares:
                raise _foreign_key_violation("share_slot_check_share_id_fkey")
            s.slot_checks[
                cast("tuple[str, int, str, int]", (share_id, slot, component_id, revision))
            ] = {
                "frame_spec_id": spec_id,
                "frame_content_version": cv,
                "freshness": freshness,
                "checked_at": checked_at,
            }
            return []

        if sql.startswith("INSERT INTO presentation.served_frame ("):
            assert sql.endswith("ON CONFLICT DO NOTHING")
            share_id, slot, component_id, revision, spec_id, cv, rendering = params
            if share_id not in s.shares:
                raise _foreign_key_violation("served_frame_share_id_fkey")
            key = (share_id, slot, component_id, revision, spec_id, cv)
            if any(
                tuple(row[name] for name in _SERVED_FRAME_KEY) == key for row in s.served_frames
            ):
                return []
            s.served_frames.append(
                {
                    "share_id": share_id,
                    "slot": slot,
                    "component_id": component_id,
                    "component_revision": revision,
                    "frame_spec_id": spec_id,
                    "frame_content_version": cv,
                    "rendering": json.loads(cast(str, rendering))
                    if rendering is not None
                    else None,
                    "first_served_at": datetime.now(UTC),
                }
            )
            return []

        if sql.startswith(
            "SELECT slot, component_id, component_revision, frame_spec_id, "
            "frame_content_version, rendering, first_served_at FROM presentation.served_frame"
        ):
            (share_id,) = params
            rows = [row for row in s.served_frames if row["share_id"] == share_id]
            # ORDER BY first_served_at DESC, slot: a stable sort by slot first, then by time.
            rows.sort(key=lambda row: row["slot"])
            rows.sort(key=lambda row: row["first_served_at"], reverse=True)
            return [dict(row) for row in rows]

        return None


_SERVED_FRAME_KEY = (
    "share_id",
    "slot",
    "component_id",
    "component_revision",
    "frame_spec_id",
    "frame_content_version",
)


class _DriverError(Exception):
    """Driver-shaped error with PostgreSQL fields in `args[0]`."""


def _foreign_key_violation(constraint: str) -> _DriverError:
    return _DriverError(
        {"C": "23503", "M": f'violates foreign key constraint "{constraint}"', "n": constraint}
    )


def _unique_violation(constraint: str) -> _DriverError:
    return _DriverError(
        {
            "C": "23505",
            "M": f'duplicate key value violates unique constraint "{constraint}"',
            "n": constraint,
        }
    )


class FakePresentationStore:
    """A DatasetStore fake with shared state, serialized transactions, and failure injection."""

    def __init__(
        self,
        state: FakePresentationState | None = None,
        *,
        fail_before: Callable[[str, Sequence[object]], bool] | None = None,
    ) -> None:
        self.state = state if state is not None else FakePresentationState()
        self.fail_before = fail_before
        self.statement_log: list[str] = []
        # A thread-local guard lets separate stores share state like separate connections.
        self._local = threading.local()

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        if getattr(self._local, "in_transaction", False):
            raise DatasetStoreError(
                "transaction() called on this store while a transaction is already open "
                "on this thread"
            )
        self._local.in_transaction = True
        try:
            with self.state.lock:
                scratch = self.state.snapshot()
                tx = FakeTransaction(
                    scratch,
                    record=self.statement_log.append,
                    fail_before=self.fail_before,
                )
                yield tx
                self.state.restore_from(scratch)
        finally:
            self._local.in_transaction = False


class DriverExplodingFakePresentationStore:
    """A transaction that raises a driver-shaped error on its first statement."""

    def __init__(self, *, message: str) -> None:
        self._message = message

    @contextmanager
    def transaction(self) -> Generator[_ExplodingTransaction, None, None]:
        yield _ExplodingTransaction(message=self._message)


class _ExplodingTransaction:
    def __init__(self, *, message: str) -> None:
        self._message = message

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        raise RuntimeError(self._message)


class SqlstateExplodingFakePresentationStore:
    """A transaction that raises a driver error carrying the supplied SQLSTATE."""

    def __init__(self, *, sqlstate: str) -> None:
        self._sqlstate = sqlstate

    @contextmanager
    def transaction(self) -> Generator[_SqlstateExplodingTransaction, None, None]:
        yield _SqlstateExplodingTransaction(sqlstate=self._sqlstate)


class _SqlstateExplodingTransaction:
    def __init__(self, *, sqlstate: str) -> None:
        self._sqlstate = sqlstate

    def _raise(self) -> None:
        raise _DriverError({"C": self._sqlstate, "M": "syntax error"})

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._raise()

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        self._raise()

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        self._raise()


class TransactionEntryExplodingFakePresentationStore:
    """A transaction context that fails before yielding, simulating a failed borrow."""

    def __init__(self, *, message: str) -> None:
        self._message = message

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        raise RuntimeError(self._message)
        yield  # pragma: no cover -- unreachable; keeps this a generator function


class CommitExplodingFakePresentationStore:
    """A working transaction fake that fails on clean exit, optionally with a SQLSTATE."""

    def __init__(self, *, message: str, sqlstate: str | None = None) -> None:
        self._message = message
        self._sqlstate = sqlstate
        self.state = FakePresentationState()

    def _raise(self) -> None:
        if self._sqlstate is not None:
            raise _DriverError({"C": self._sqlstate, "M": self._message})
        raise RuntimeError(self._message)

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        scratch = self.state.snapshot()
        tx = FakeTransaction(scratch, record=lambda sql: None)
        yield tx
        self.state.restore_from(scratch)
        self._raise()

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        self._raise()

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        self._raise()
