"""The description store keeps each version's `DescriptionRecord` in `datasets.description_record`.

A record holds the facts and evidence a description is composed from, keyed by the version's
`(subject_id, content_version)`, where `subject_id` is the record's `id`. The row carries `schema_version` and `basis` beside the record's JSON,
and the version row's foreign key means a record only ever names a version this store holds.

`insert_description` writes a record inside the transaction the caller passes, the one that
writes the version row. It uses `ON CONFLICT DO NOTHING`, so the first writer's record stands,
and returns whether this call wrote it. Text is stripped of U+0000 before the insert, because
`jsonb` refuses it and a failed insert rolls back the version row.

A stored row whose JSON no longer validates as a `DescriptionRecord` counts as no record:
`read_description` logs it as `DESCRIPTION_RECORD_INVALID_MARKER` and returns `None`, so every
reader takes its absent-record path, and `insert_description` replaces it, so any writer that
builds the version's record, the `backfill-descriptions` task included, repairs it. A row whose
`schema_version` is above this image's (`description.facts.SCHEMA_VERSION`) was written by a
later image: it reads as no record here too, without the marker, and nothing here overwrites
it, so a rollback keeps the later image's records, their basis and late fields included.

`fill_late_fields` sets members that are `null` in a stored record, once each, and notes the
member's path, arrival time and basis in the record's `late_fields`. A member that holds a value
is never changed, except a member derived from one the same write fills, such as the computed
inputs a late `steps` replaces. It takes an advisory lock on the version first, so concurrent fillers run one
after another and each sees the previous filler's committed record.

`read_descriptions` reads the records of several versions in one query.

`LiveDescriptionStore` reads a record, or several, and fills late fields in transactions of its own. Store
errors become `DatasetsStoreUnavailable` after the transaction exits.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import cast

from pydantic import JsonValue, ValidationError

from dataset_store import DatasetStore, Transaction
from dataset_store.clock import Clock, utc_now
from datasets_api.store_errors import DatasetsStoreUnavailable
from description.facts import SCHEMA_VERSION, Basis, DescriptionRecord, LateField
from description.stored import FieldPath, canonical_json
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics

logger = logging.getLogger(__name__)

# Two-key advisory locks live apart from the classes `version_store` and `operation_store` use.
_DESCRIPTION_LOCK_CLASS = 0x4446_4452  # "DFDR" packed into an int4

# Logs a stored record that fails validation, with the version's id and content version. Reads
# treat it as absent, and the next write of the version's record replaces it.
DESCRIPTION_RECORD_INVALID_MARKER = "description_record_invalid"


class DescriptionNotStored(LookupError):
    """The version holds no description record to fill."""

    def __init__(self, *, id: str, content_version: str) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(f"id {id!r} content_version {content_version!r} holds no record")


def insert_description(
    tx: Transaction, record: DescriptionRecord, *, recorded_at: datetime
) -> bool:
    """Store `record` once. Return `True` only when this call wrote the row; a version that
    already holds a record keeps it and the call returns `False`. A held row of this image's
    schema version or an earlier one that fails validation is no record: this call replaces it,
    under the lock `fill_late_fields` takes, so of two concurrent writers only the first
    replaces it. A row of a later schema version stands, and the call returns `False`.
    """
    stored = tx.fetch_one(
        "INSERT INTO datasets.description_record "
        "(subject_id, content_version, schema_version, basis, record, recorded_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (subject_id, content_version) DO NOTHING RETURNING subject_id",
        (
            record.id,
            record.content_version,
            record.schema_version,
            record.basis,
            _stored_json(record),
            recorded_at,
        ),
    )
    if stored is not None:
        return True
    key = {"id": record.id, "content_version": record.content_version}
    _lock(tx, **key)
    if read_description(tx, **key) is not None:
        return False
    replaced = tx.fetch_one(
        "UPDATE datasets.description_record "
        "SET schema_version = %s, basis = %s, record = %s, recorded_at = %s "
        "WHERE subject_id = %s AND content_version = %s AND schema_version <= %s "
        "RETURNING subject_id",
        (
            record.schema_version,
            record.basis,
            _stored_json(record),
            recorded_at,
            record.id,
            record.content_version,
            SCHEMA_VERSION,
        ),
    )
    if replaced is None:
        return False
    logger.warning(
        "description record replaced one that failed validation id=%s content_version=%s",
        record.id,
        record.content_version,
    )
    return True


def read_description(tx: Transaction, *, id: str, content_version: str) -> DescriptionRecord | None:
    """Return the version's record, or `None` when it holds none, its row is of a later
    schema version, or its row fails validation, which is logged."""
    row = tx.fetch_one(
        "SELECT schema_version, record FROM datasets.description_record "
        "WHERE subject_id = %s AND content_version = %s",
        (id, content_version),
    )
    if row is None:
        return None
    return validated_record(
        row["record"],
        schema_version=cast(int, row["schema_version"]),
        id=id,
        content_version=content_version,
    )


def validated_record(
    stored: object, *, schema_version: int, id: str, content_version: str
) -> DescriptionRecord | None:
    """`stored`, a row's record JSON stored under `schema_version`, as a record. `None` when
    a later image's schema wrote it, and `None` after logging `DESCRIPTION_RECORD_INVALID_MARKER`
    when it fails validation."""
    if schema_version > SCHEMA_VERSION:
        logger.info(
            "description record of a later schema read as none schema_version=%s id=%s "
            "content_version=%s",
            schema_version,
            id,
            content_version,
        )
        return None
    try:
        return DescriptionRecord.model_validate(stored)
    except ValidationError:
        logger.exception(
            "%s id=%s content_version=%s",
            DESCRIPTION_RECORD_INVALID_MARKER,
            id,
            content_version,
        )
        return None


def _lock(tx: Transaction, *, id: str, content_version: str) -> None:
    tx.execute(
        "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
        (_DESCRIPTION_LOCK_CLASS, json.dumps([id, content_version])),
    )


def read_descriptions(
    tx: Transaction, *, versions: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], DescriptionRecord]:
    """The records `versions`, each an `(id, content_version)` pair, hold, in one query, by
    pair. A version that holds none, whose row is of a later schema version, or whose record
    fails validation (logged), is left out."""
    if not versions:
        return {}
    matches = " OR ".join(["(subject_id = %s AND content_version = %s)"] * len(versions))
    rows = tx.fetch_all(
        "SELECT subject_id, content_version, schema_version, record "
        f"FROM datasets.description_record WHERE {matches}",
        [part for version in versions for part in version],
    )
    records: dict[tuple[str, str], DescriptionRecord] = {}
    for row in rows:
        key = (cast(str, row["subject_id"]), cast(str, row["content_version"]))
        record = validated_record(
            row["record"],
            schema_version=cast(int, row["schema_version"]),
            id=key[0],
            content_version=key[1],
        )
        if record is not None:
            records[key] = record
    return records


def fill_late_fields(
    tx: Transaction,
    *,
    id: str,
    content_version: str,
    values: Mapping[FieldPath, JsonValue],
    arrived: datetime,
    basis: Basis,
    derived: Mapping[FieldPath, JsonValue] | None = None,
) -> list[FieldPath]:
    """Fill each member of the stored record that `values` names and that is `null`, and note
    it in `late_fields` with `arrived` and `basis`. When this call fills any, it also sets each
    `derived` member, replacing what it held, and notes it the same way. Return the paths this
    call filled or set, which leave out a member that already held a value or a value that is
    `None`.

    Raises:
        DescriptionNotStored: The version holds no record.
        ValueError: A path names no member of the record, or the record with the filled
            members fails its own validation. Nothing is written.
    """
    _lock(tx, id=id, content_version=content_version)
    row = tx.fetch_one(
        "SELECT record FROM datasets.description_record WHERE subject_id = %s AND content_version = %s",
        (id, content_version),
    )
    if row is None:
        raise DescriptionNotStored(id=id, content_version=content_version)
    record = cast(dict[str, object], row["record"])
    filled = [
        path
        for path, value in values.items()
        if value is not None and _fill_member(record, path, value)
    ]
    if not filled:
        return []
    for path, value in (derived or {}).items():
        _set_member(record, path, value)
        filled.append(path)
    late_fields = cast(list[object], record["late_fields"])
    late_fields.extend(
        LateField(field=path, arrived=arrived, basis=basis).model_dump(mode="json")
        for path in filled
    )
    updated = DescriptionRecord.model_validate(record)
    tx.execute(
        "UPDATE datasets.description_record SET record = %s WHERE subject_id = %s AND content_version = %s",
        (_stored_json(updated), id, content_version),
    )
    return filled


def _fill_member(record: dict[str, object], path: str, value: JsonValue) -> bool:
    """Set the member at the dotted `path` when it is `null`. Return whether it was set. A
    number in `path` names a list item by its position."""
    node, leaf = _parent(record, path)
    if node[leaf] is not None:
        return False
    node[leaf] = value
    return True


def _set_member(record: dict[str, object], path: str, value: JsonValue) -> None:
    """Set the member at the dotted `path`, whatever it holds."""
    node, leaf = _parent(record, path)
    node[leaf] = value


def _parent(record: dict[str, object], path: str) -> tuple[dict[str, object], str]:
    """The mapping holding the member at the dotted `path`, and the member's name."""
    *parents, leaf = path.split(".")
    node: object = record
    for name in parents:
        if isinstance(node, list) and name.isdigit() and int(name) < len(node):
            node = cast(list[object], node)[int(name)]
        elif isinstance(node, dict):
            node = cast(dict[str, object], node).get(name)
        else:
            raise ValueError(f"{path!r} names no member of the record")
    if not isinstance(node, dict):
        raise ValueError(f"{path!r} names no member of the record")
    node = cast(dict[str, object], node)
    if leaf not in node:
        raise ValueError(f"{path!r} names no member of the record")
    return node, leaf


def _stored_json(record: DescriptionRecord) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], canonical_json(record.model_dump(mode="json")))


class LiveDescriptionStore:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        """Return the version's record, or `None` when it holds none."""
        return self._run(
            lambda tx: read_description(tx, id=id, content_version=content_version),
            doing=f"reading description id {id!r} content_version {content_version!r}",
        )

    def read_descriptions(
        self, *, versions: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], DescriptionRecord]:
        """The records `versions` hold, by `(id, content_version)`, in one read; a version
        that holds none is left out."""
        return self._run(
            lambda tx: read_descriptions(tx, versions=versions),
            doing=f"reading {len(versions)} descriptions",
        )

    def fill_late_fields(
        self,
        *,
        id: str,
        content_version: str,
        values: Mapping[FieldPath, JsonValue],
        basis: Basis,
        derived: Mapping[FieldPath, JsonValue] | None = None,
    ) -> list[FieldPath]:
        """Fill the record's `null` members that `values` names, and set the `derived` ones
        with them, at the current time. See `fill_late_fields` for the result and the errors it
        raises.
        """
        arrived = self._now()
        return self._run(
            lambda tx: fill_late_fields(
                tx,
                id=id,
                content_version=content_version,
                values=values,
                arrived=arrived,
                basis=basis,
                derived=derived,
            ),
            doing=f"filling description id {id!r} content_version {content_version!r}",
        )

    def _run[T](self, work: Callable[[Transaction], T], *, doing: str) -> T:
        failure = ""
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                return work(tx)
        except (DescriptionNotStored, ValueError):
            raise
        except Exception as exc:
            # Save only safe diagnostics, then raise after leaving this exception handler.
            failure = f"{doing} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
