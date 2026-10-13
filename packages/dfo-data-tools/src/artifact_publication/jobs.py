"""Build and publication machinery shared by `series_coverage.job` and
`origin_catalogue.job`.

Both jobs publish immutable `versions/<build_id>/` directories under a format
root and commit a `current.json` pointer with a generation-conditional
write. Both read Spanner through BigQuery `EXTERNAL_QUERY`, skip a build
while `IngestionLock` is held, fingerprint the ingested data, prune builds
nothing can still be reading, and alert when publication falls behind.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic_settings import BaseSettings

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX
from artifact_publication.uri import (
    ArtifactUriError,
    delete_uri_if_unchanged,
    delete_uri_prefix,
    join_uri,
    list_uri_object_times,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes,
    write_uri_bytes_if_unchanged,
)

logger = logging.getLogger(__name__)

LEASE_FILE = "lease.json"


class AliasOnlySettings(BaseSettings):
    """Job settings that read each field's `alias` from the environment and
    nothing else.

    `populate_by_name` is off, so a field's bare Python name
    (`ARTIFACT_URI`) is never read from the environment. `__init__` maps a
    field-name keyword to its alias, so tests and CLI overrides can still
    construct one by field name.
    """

    def __init__(self, **data: Any) -> None:
        fields = type(self).model_fields
        aliased: dict[str, Any] = {
            (fields[key].alias or key) if key in fields else key: value
            for key, value in data.items()
        }
        super().__init__(**aliased)


# --- BigQuery ------------------------------------------------------------


class BigQueryReader(Protocol):
    """Runs SQL to completion and returns rows as plain dictionaries:
    `query` all at once, `stream` as the result's pages arrive, so a caller
    that keeps less than each row holds one page of rows at a time."""

    def query(self, sql: str) -> list[dict[str, Any]]: ...

    def stream(self, sql: str) -> Iterator[dict[str, Any]]: ...


class LiveBigQuery:
    """Implements `BigQueryReader` with a lazily imported BigQuery client.

    The client library is included in `dfo-data-tools[coverage]`.
    """

    def __init__(self, *, project: str) -> None:
        from google.cloud import bigquery

        self._client = bigquery.Client(project=project)

    def query(self, sql: str) -> list[dict[str, Any]]:
        return list(self.stream(sql))

    def stream(self, sql: str) -> Iterator[dict[str, Any]]:
        for row in self._client.query(sql).result():
            yield dict(row)


def lock_held(bq: BigQueryReader, *, connection: str) -> bool:
    """Returns `True` when the single `IngestionLock` row has a `LockOwner`.

    The row always exists. An unheld lock has `LockOwner IS NULL`.
    """
    sql = (
        "SELECT LockOwner\n"
        f'FROM EXTERNAL_QUERY("{connection}", "SELECT LockOwner FROM IngestionLock")'
    )
    rows = bq.query(sql)
    return any(row.get("LockOwner") for row in rows)


@dataclass(frozen=True, slots=True)
class Ingestion:
    """The latest successful `IngestionHistory` row a build read, raw. Two
    artifacts built from one ingestion record equal values, so a reader can
    tell whether a pair it loaded agree."""

    workflow_execution_id: str
    creation_timestamp: str

    def to_json(self) -> dict[str, str]:
        return {
            "workflow_execution_id": self.workflow_execution_id,
            "creation_timestamp": self.creation_timestamp,
        }

    @classmethod
    def from_json(cls, payload: Any) -> Ingestion:
        """Raises `KeyError` naming a missing field and `ValueError` for a
        non-object or a non-string value."""
        if not isinstance(payload, dict):
            raise ValueError(f"ingestion is not a JSON object: {payload!r}")
        execution, created = payload["workflow_execution_id"], payload["creation_timestamp"]
        if not isinstance(execution, str) or not isinstance(created, str):
            raise ValueError(f"ingestion fields are not strings: {payload!r}")
        return cls(workflow_execution_id=execution, creation_timestamp=created)


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """The data-version fingerprint: the latest successful `IngestionHistory`
    row plus `COUNT(*)`/`MAX(last_update_timestamp)` over the `Observation`
    rows a job reads. `stamp` is the digest a job compares and records."""

    workflow_execution_id: str
    creation_timestamp: str
    row_count: int
    max_last_update_timestamp: str

    @property
    def stamp(self) -> str:
        digest_input = "|".join(
            (
                self.workflow_execution_id,
                self.creation_timestamp,
                str(self.row_count),
                self.max_last_update_timestamp,
            )
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:32]

    @property
    def ingestion(self) -> Ingestion:
        return Ingestion(
            workflow_execution_id=self.workflow_execution_id,
            creation_timestamp=self.creation_timestamp,
        )


def observation_stats_sql(connection: str) -> str:
    """The row count and latest update over every `Observation` row.
    `read_ingestion_fingerprint` reads it as `row_count` and `max_ts`."""
    return (
        "SELECT COUNT(*) AS row_count, MAX(last_update_timestamp) AS max_ts\n"
        f'FROM EXTERNAL_QUERY("{connection}", '
        '"SELECT last_update_timestamp FROM Observation")'
    )


def read_ingestion_fingerprint(
    bq: BigQueryReader, *, connection: str, observation_sql: str
) -> Fingerprint | None:
    """Reads the latest successful ingestion and the one-row observation
    stats `observation_sql` selects as `row_count` and `max_ts`, such as
    `observation_stats_sql`. Returns `None` when `IngestionHistory` has no
    successful row.

    Sorting runs in BigQuery outside `EXTERNAL_QUERY` to preserve Spanner
    root partitioning.
    """
    history_sql = (
        "SELECT WorkflowExecutionID, CreationTimestamp\n"
        f'FROM EXTERNAL_QUERY("{connection}", '
        '"SELECT WorkflowExecutionID, CreationTimestamp FROM IngestionHistory '
        "WHERE IngestionFailure = false AND Status = 'SUCCESS'\")\n"
        "ORDER BY CreationTimestamp DESC\nLIMIT 1"
    )
    history_rows = bq.query(history_sql)
    if not history_rows:
        return None
    latest = history_rows[0]
    obs = bq.query(observation_sql)[0]
    return Fingerprint(
        workflow_execution_id=str(latest["WorkflowExecutionID"]),
        creation_timestamp=str(latest["CreationTimestamp"]),
        row_count=int(obs["row_count"]),
        max_last_update_timestamp=str(obs["max_ts"]),
    )


# --- Builds ------------------------------------------------------------------


def new_build_id() -> str:
    """A build directory name no other run chooses: the UTC time to the
    second and 8 random hex digits, e.g. `20260927T142252Z-3f9a0c1e`. The
    timestamp prefix sorts a listing by build start."""
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def emit(event: str, **fields: object) -> None:
    """One structured JSON line to stdout. Cloud Logging's agent parses a
    JSON-formatted stdout line from a Cloud Run Job into `jsonPayload`, so
    alerts can match `jsonPayload.event`."""
    payload = {"event": event, "severity": "INFO", **fields}
    print(json.dumps(payload, sort_keys=True, default=str))


def prune_versions(
    root: str,
    *,
    pointer_build_id: Callable[[bytes], str],
    grace_seconds: float,
    now: Callable[[], float] = time.time,
    pointer_prefix: str | None = None,
) -> tuple[str, ...]:
    """Deletes and returns, sorted, the build ids under `root` safe to remove.

    `pointer_build_id` parses a pointer into the build id it names. The pointers are
    `current.json` and, when `pointer_prefix` is set, every object directly under
    `root/<pointer_prefix>/`. A build is eligible when no pointer names it, every pointer is
    at least `grace_seconds` old, and the build's newest file is at least that old.
    `grace_seconds` is the job's maximum runtime: a younger build may be another run's publish
    in progress, and a younger pointer may have named another build moments ago that a reader
    is still loading. The pointers are read before the listing, so a pointer written after its
    read shows in the listing as young and stops the run. A missing `current.json` or any
    unreadable pointer deletes nothing. Only `root`'s `versions/` directory is pruned.
    """
    pointers = [POINTER_FILE]
    if pointer_prefix is not None:
        pointers.extend(
            f"{pointer_prefix}/{name}"
            for name in list_uri_object_times(join_uri(root, pointer_prefix))
            if "/" not in name
        )
    live_build_ids: set[str] = set()
    for pointer in pointers:
        uri = join_uri(root, pointer)
        try:
            live_build_ids.add(pointer_build_id(read_uri_bytes(uri)))
        except Exception:
            logger.warning(
                "prune: the pointer at %r could not be read; pruning nothing this run rather "
                "than risk deleting the build it names",
                uri,
                exc_info=True,
            )
            return ()
    current_time = now()
    times = list_uri_object_times(root)
    listed_pointers = [POINTER_FILE]
    if pointer_prefix is not None:
        listed_pointers.extend(
            relative
            for relative in times
            if relative.startswith(f"{pointer_prefix}/") and relative.count("/") == 1
        )
    if POINTER_FILE not in times or any(
        current_time - times[pointer] < grace_seconds for pointer in listed_pointers
    ):
        return ()

    newest_by_build: dict[str, float] = {}
    for relative, written in times.items():
        parts = relative.split("/")
        if len(parts) < 3 or parts[0] != VERSIONS_PREFIX:
            continue
        build_id = parts[1]
        newest_by_build[build_id] = max(written, newest_by_build.get(build_id, written))
    to_delete = tuple(
        sorted(
            build_id
            for build_id, newest in newest_by_build.items()
            if build_id not in live_build_ids and current_time - newest >= grace_seconds
        )
    )
    for build_id in to_delete:
        delete_uri_prefix(join_uri(root, VERSIONS_PREFIX, build_id))
    return to_delete


# --- Stale-publish tracking --------------------------------------------------


def _read_state(uri: str) -> tuple[str, float] | None:
    """The stamp a state file tracks and when it was first seen unpublished.
    `None` when the file is missing or malformed."""
    try:
        payload = json.loads(read_uri_bytes(uri))
        stamp, first_seen_at = payload["fingerprint_stamp"], float(payload["first_seen_at"])
    except (FileNotFoundError, ArtifactUriError, ValueError, KeyError, TypeError):
        return None
    return (stamp, first_seen_at) if isinstance(stamp, str) else None


def unpublished_seconds(root: str, *, state_file: str, stamp: str, now: float) -> float:
    """How long `stamp` has gone unpublished, tracked in `state_file` at `root`.

    The first call for a stamp records `now` and returns 0. A new stamp
    starts its own clock. Call it only while the published stamp differs
    from `stamp`.
    """
    uri = join_uri(root, state_file)
    state = _read_state(uri)
    if state is None or state[0] != stamp:
        started = {"fingerprint_stamp": stamp, "first_seen_at": now}
        write_uri_bytes(uri, json.dumps(started).encode("utf-8"))
        return 0.0
    return now - state[1]


def tracked_stamp(root: str, *, state_file: str) -> str | None:
    """The stamp `unpublished_seconds` last started a clock for in
    `state_file` at `root`, or `None`. A run that cannot compute its own
    stamp, such as one that finds `IngestionLock` held, reads it to check
    the stamp the last run saw."""
    state = _read_state(join_uri(root, state_file))
    return state[0] if state is not None else None


# --- Build lease -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lease:
    """A held build lease: the build that took it, when it lapses (Unix
    seconds), and the lease object's version token, which release needs."""

    build_id: str
    expires_at: float
    generation: str


def acquire_lease(
    root: str, *, build_id: str, ttl_seconds: float, now: Callable[[], float] = time.time
) -> Lease | None:
    """Takes the create-only lease object `lease.json` at `root`, or returns
    `None` when another build holds it.

    A lease whose `expires_at` has passed, or one that does not parse, is
    reclaimed by a write conditional on its version token, so two runs
    reclaiming the same lease cannot both win. `ttl_seconds` is the job's
    maximum runtime, after which the run holding the lease is dead.
    """
    uri = join_uri(root, LEASE_FILE)
    taken_at = now()
    found = read_uri_bytes_and_generation(uri)
    expected: str | None = None
    if found is not None:
        raw, expected = found
        try:
            if float(json.loads(raw)["expires_at"]) > taken_at:
                return None
        except (ValueError, KeyError, TypeError):
            logger.warning("lease at %r does not parse; reclaiming it", uri)
    expires_at = taken_at + ttl_seconds
    payload = json.dumps({"build_id": build_id, "expires_at": expires_at}, sort_keys=True).encode(
        "utf-8"
    )
    if not write_uri_bytes_if_unchanged(uri, payload, expected_generation=expected):
        return None
    # Release needs this write's own token, which the conditional write does not return.
    written = read_uri_bytes_and_generation(uri)
    if written is None or written[0] != payload:
        return None
    return Lease(build_id=build_id, expires_at=expires_at, generation=written[1])


def release_lease(root: str, lease: Lease) -> bool:
    """Deletes the lease object if it is still `lease`. Returns whether it did.
    `False` means another run reclaimed it after it lapsed."""
    return delete_uri_if_unchanged(join_uri(root, LEASE_FILE), expected_generation=lease.generation)


__all__ = [
    "LEASE_FILE",
    "AliasOnlySettings",
    "BigQueryReader",
    "Fingerprint",
    "Ingestion",
    "Lease",
    "LiveBigQuery",
    "acquire_lease",
    "emit",
    "lock_held",
    "new_build_id",
    "observation_stats_sql",
    "prune_versions",
    "read_ingestion_fingerprint",
    "release_lease",
    "tracked_stamp",
    "unpublished_seconds",
]
