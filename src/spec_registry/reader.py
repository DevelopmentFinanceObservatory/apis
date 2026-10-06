"""Read-only registry interface and its stored definition records.
Missing ids return `None`. Quarantined ids raise `IdentityQuarantined` even when a row exists.
Calls are synchronous, so async callers must use a worker thread.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from dataset_store import DatasetStore, PoolTimeoutError
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

        Raises `IdentityQuarantined` for a quarantined id.
        """
        ...

    def read_definitions(self, spec_ids: Sequence[str]) -> dict[str, StoredDefinition]:
        """Read ids in one query, selecting each highest version and omitting missing ids.

        Raises `IdentityQuarantined` for the first quarantined id in sorted order.
        """
        ...


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

    def _read_many(
        self, spec_ids: list[str], *, schema_version: int | None
    ) -> dict[str, StoredDefinition]:
        placeholders = ", ".join(["%s"] * len(spec_ids))
        failure: str | None = None
        unavailable = False
        registry_rows: list = []
        definition_rows: list = []
        quarantined_ids: set[str] = set()
        try:
            with self._store.transaction() as tx:
                quarantine_rows = tx.fetch_all(
                    "SELECT subject_id FROM datasets.identity_quarantine "
                    f"WHERE id_kind = %s AND subject_id IN ({placeholders})",
                    ("spec", *spec_ids),
                )
                quarantined_ids = {cast(str, row["subject_id"]) for row in quarantine_rows}
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
        return result
