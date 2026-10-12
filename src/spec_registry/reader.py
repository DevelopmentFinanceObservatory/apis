"""Read-only registry interface and its stored definition records.
Missing ids return `None`. A quarantined id raises `IdentityQuarantined` even when a row
exists, and so does an id that `datasets.spec_alias` binds to a quarantined served id.
`read_served_spec_ids` reads the served ids `datasets.spec_alias` binds to a posted id.
Calls are synchronous, so async callers must use a worker thread.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from dataset_store import DatasetStore, PoolTimeoutError, Transaction
from dfo_platform.store_errors import safe_diagnostics
from spec_registry.errors import (
    IdentityQuarantined,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredDefinition:
    """Stored canonical identity projection and matching registry bytes and digest."""

    spec_id: str
    schema_version: int
    identity: Mapping[str, object]
    identity_bytes: bytes
    identity_digest: str


class SpecReader(Protocol):
    def read_definition(
        self, spec_id: str, *, schema_version: int | None = None
    ) -> StoredDefinition | None:
        """Return the requested or highest stored version, or `None` if absent.

        Raises `IdentityQuarantined` when `spec_id` or a served id bound to it is quarantined,
        for the first such id in sorted order.
        """
        ...

    def read_definitions(self, spec_ids: Sequence[str]) -> dict[str, StoredDefinition]:
        """Read ids in one query, selecting each highest version and omitting missing ids.

        Raises `IdentityQuarantined` when one of `spec_ids` or a served id bound to one is
        quarantined, for the first such id in sorted order.
        """
        ...

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        """The served ids bound to the posted id `spec_id`, newest binding first, or `None`
        for an unregistered id. A registered id with no binding returns an empty tuple.

        Raises `IdentityQuarantined` when `spec_id` or a served id bound to it is quarantined,
        for the first such id in sorted order.
        """
        ...


def read_stored_definitions(
    tx: Transaction, spec_ids: Sequence[str], *, schema_version: int | None = None
) -> tuple[set[str], dict[str, StoredDefinition]]:
    """Inside `tx`, the quarantined ids that refuse a read of `spec_ids` (`_quarantined`) and
    the stored definition of each registered one, at `schema_version` or its highest stored
    version."""
    placeholders = ", ".join(["%s"] * len(spec_ids))
    quarantined_ids = _quarantined(tx, spec_ids)
    registry_rows = tx.fetch_all(
        "SELECT spec_id, identity_bytes, identity_digest "
        f"FROM datasets.spec_registry WHERE spec_id IN ({placeholders})",
        tuple(spec_ids),
    )
    if schema_version is None:
        definition_rows = tx.fetch_all(
            "SELECT spec_id, schema_version, definition "
            f"FROM datasets.spec_definition WHERE spec_id IN ({placeholders})",
            tuple(spec_ids),
        )
    else:
        definition_rows = tx.fetch_all(
            "SELECT spec_id, schema_version, definition FROM datasets.spec_definition "
            f"WHERE spec_id IN ({placeholders}) AND schema_version = %s",
            (*spec_ids, schema_version),
        )

    registry_by_id = {cast(str, row["spec_id"]): row for row in registry_rows}
    latest_definition_by_id: dict[str, Mapping[str, object]] = {}
    for row in definition_rows:
        spec_id = cast(str, row["spec_id"])
        current = latest_definition_by_id.get(spec_id)
        if current is None or cast(int, row["schema_version"]) > cast(
            int, current["schema_version"]
        ):
            latest_definition_by_id[spec_id] = row

    result: dict[str, StoredDefinition] = {}
    for spec_id, definition_row in latest_definition_by_id.items():
        registry_row = registry_by_id.get(spec_id)
        if registry_row is None:
            continue
        definition = cast(Mapping[str, object], definition_row["definition"])
        identity = definition["identity"]
        if isinstance(identity, (bytes, str)):
            identity = json.loads(identity)
        result[spec_id] = StoredDefinition(
            spec_id=spec_id,
            schema_version=cast(int, definition_row["schema_version"]),
            identity=cast("Mapping[str, object]", identity),
            identity_bytes=cast(bytes, registry_row["identity_bytes"]),
            identity_digest=cast(str, registry_row["identity_digest"]),
        )
    return quarantined_ids, result


def _quarantined(tx: Transaction, spec_ids: Sequence[str]) -> set[str]:
    """The quarantined ids among `spec_ids` and the served ids `datasets.spec_alias` binds to
    them, read in one statement. A read under a posted id refuses when any of them is
    quarantined, so every door applies one rule."""
    placeholders = ", ".join(["%s"] * len(spec_ids))
    rows = tx.fetch_all(
        "SELECT subject_id FROM datasets.identity_quarantine "
        f"WHERE id_kind = %s AND (subject_id IN ({placeholders}) OR subject_id IN ("
        f"SELECT served_spec_id FROM datasets.spec_alias WHERE posted_spec_id IN ({placeholders})))",
        ("spec", *spec_ids, *spec_ids),
    )
    return {cast(str, row["subject_id"]) for row in rows}


class LiveSpecReader:
    """`DatasetStore` reader using one read-only transaction per call and schema-qualified SQL."""

    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    def read_definition(
        self, spec_id: str, *, schema_version: int | None = None
    ) -> StoredDefinition | None:
        return self._read_many([spec_id], schema_version=schema_version).get(spec_id)

    def read_definitions(self, spec_ids: Sequence[str]) -> dict[str, StoredDefinition]:
        if not spec_ids:
            return {}
        return self._read_many(list(dict.fromkeys(spec_ids)), schema_version=None)

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        failure: str | None = None
        unavailable = False
        registered = False
        served_ids: tuple[str, ...] = ()
        quarantined_ids: set[str] = set()
        try:
            with self._store.transaction() as tx:
                registered = (
                    tx.fetch_one(
                        "SELECT 1 FROM datasets.spec_definition WHERE spec_id = %s LIMIT 1",
                        (spec_id,),
                    )
                    is not None
                )
                served_ids = tuple(
                    cast(str, row["served_spec_id"])
                    for row in tx.fetch_all(
                        "SELECT served_spec_id FROM datasets.spec_alias "
                        "WHERE posted_spec_id = %s ORDER BY bound_at DESC, served_spec_id",
                        (spec_id,),
                    )
                )
                quarantined_ids = _quarantined(tx, [spec_id])
        except PoolTimeoutError as exc:
            failure = f"reading served ids of spec_id {spec_id!r} failed: {safe_diagnostics(exc)}"
            unavailable = True
        except Exception as exc:
            failure = f"reading served ids of spec_id {spec_id!r} failed: {safe_diagnostics(exc)}"

        if failure is not None:
            if unavailable:
                raise SpecRegistryUnavailable(failure)
            raise SpecRegistryStoreError(failure)

        if quarantined_ids:
            raise IdentityQuarantined(subject_id=sorted(quarantined_ids)[0], id_kind="spec")
        return served_ids if registered else None

    def _read_many(
        self, spec_ids: list[str], *, schema_version: int | None
    ) -> dict[str, StoredDefinition]:
        failure: str | None = None
        unavailable = False
        quarantined_ids: set[str] = set()
        result: dict[str, StoredDefinition] = {}
        try:
            with self._store.transaction() as tx:
                quarantined_ids, result = read_stored_definitions(
                    tx, spec_ids, schema_version=schema_version
                )
        except PoolTimeoutError as exc:
            failure = f"reading spec_id(s) {spec_ids!r} failed: {safe_diagnostics(exc)}"
            unavailable = True
        except Exception as exc:
            failure = f"reading spec_id(s) {spec_ids!r} failed: {safe_diagnostics(exc)}"

        if failure is not None:
            if unavailable:
                raise SpecRegistryUnavailable(failure)
            raise SpecRegistryStoreError(failure)

        if quarantined_ids:
            raise IdentityQuarantined(subject_id=sorted(quarantined_ids)[0], id_kind="spec")
        return result
