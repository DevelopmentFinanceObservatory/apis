"""The recipe store keeps a derived frame's recipe and its replay index.

A recipe is the op-spec tree analysis-mcp ran to produce an `ops_`/`ocv_` frame. It lives in
`datasets.frame_recipe`, keyed by the frame's pair, outside the version body the publish door
compares on a republish. A frame published without a recipe gains one from a later write.

`write_recipe` stores a pair's recipe once. A repeat write succeeds when its `normal_form_hash`
and `replay_key` match the stored ones, and changes nothing: the stored body stands, because
two writers of one frame can differ in display text the hashes leave out. A different
`normal_form_hash` or `replay_key` raises `RecipeDiverges`. A pair with no version row raises
`RecipeFrameNotStored`, so a recipe and a replay key only ever name a frame this store holds.

`replay_key` hashes the recipe's normal form, its root dataset versions, and the execution
identity. Two instances replaying one card mint the same frame pair, so one key names one
frame. A write naming a key another pair holds raises `ReplayKeyHeld`.

Store errors become `DatasetsStoreUnavailable` after the transaction exits.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

from dataset_store import DatasetStore, Transaction
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import Clock, utc_now
from dfo_platform import ProblemSpec
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecipeRecord:
    """One frame's stored recipe. `body` is the whole recipe as it was written."""

    spec_id: str
    content_version: str
    normal_form_hash: str
    replay_key: str
    body: Mapping[str, object]


@dataclass(frozen=True)
class ReplayedFrame:
    """The frame stored under one replay key."""

    spec_id: str
    content_version: str


class RecipeRefusal(Exception):
    """A recipe write this store refuses whatever the retry."""


class RecipeFrameNotStored(RecipeRefusal):
    """The pair has no version row, so it cannot hold a recipe. Maps to `snapshot_not_found`
    (404).
    """

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} is not a published "
            "frame; publish the frame before writing its recipe"
        )


class RecipeDiverges(RecipeRefusal):
    """The pair holds a recipe whose `field` differs from this write's. Maps to non-retryable
    `publish_conflict` (409), with a `constraint_id` naming the field.
    """

    def __init__(self, *, spec_id: str, content_version: str, field: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        self.field = field
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} already holds a recipe "
            f"with a different {field} than this write's own; a frame's pair identifies one "
            "recipe, so the stored recipe stands and retrying with this one will refuse "
            "identically"
        )


class ReplayKeyHeld(RecipeRefusal):
    """Another pair holds this replay key. Maps to non-retryable `publish_conflict` (409)."""

    def __init__(
        self,
        *,
        replay_key: str,
        spec_id: str,
        content_version: str,
        holder_spec_id: str,
        holder_content_version: str,
    ) -> None:
        self.replay_key = replay_key
        self.spec_id = spec_id
        self.content_version = content_version
        self.holder_spec_id = holder_spec_id
        self.holder_content_version = holder_content_version
        super().__init__(
            f"replay_key {replay_key!r} is already held by spec_id {holder_spec_id!r} "
            f"content_version {holder_content_version!r}; refusing to index it for spec_id "
            f"{spec_id!r} content_version {content_version!r}. One replay key names one "
            "frame, so two pairs under it mean the key does not cover everything the frame "
            "id hashes; retrying will refuse identically"
        )


class RecipeReader(Protocol):
    def read_recipe(self, *, spec_id: str, content_version: str) -> RecipeRecord | None:
        """Return the pair's stored recipe, or `None` when it holds none."""
        ...


class RecipeStore(RecipeReader, Protocol):
    def write_recipe(self, record: RecipeRecord) -> bool:
        """Store the pair's recipe once and index its replay key. Return `True` only when this
        call creates the row.

        Raises:
            RecipeFrameNotStored: The pair has no version row.
            RecipeDiverges: The pair holds a recipe with another `normal_form_hash` or
                `replay_key`.
            ReplayKeyHeld: Another pair holds `record.replay_key`.
        """
        ...

    def read_replay(self, *, replay_key: str) -> ReplayedFrame | None:
        """Return the frame stored under `replay_key`, or `None` when no recipe carries it."""
        ...


def _write_recipe(tx: Transaction, record: RecipeRecord, *, recorded_at: datetime) -> bool:
    """Insert `record`, or compare it with the row that kept the insert out.

    `ON CONFLICT DO NOTHING` covers both the pair's primary key and the unique replay key, and
    waits for a concurrent inserter to commit. Each read after it runs as its own statement, so
    under READ COMMITTED it sees the row that won. Version rows are never deleted, so the
    existence check cannot go stale before the insert.
    """
    pair = (record.spec_id, record.content_version)
    version = tx.fetch_one(
        "SELECT 1 AS held FROM datasets.version_record "
        "WHERE subject_id = %s AND content_version = %s",
        pair,
    )
    if version is None:
        raise RecipeFrameNotStored(spec_id=record.spec_id, content_version=record.content_version)
    inserted = tx.fetch_one(
        "INSERT INTO datasets.frame_recipe "
        "(subject_id, content_version, normal_form_hash, replay_key, recipe, recorded_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING RETURNING subject_id",
        (*pair, record.normal_form_hash, record.replay_key, dict(record.body), recorded_at),
    )
    if inserted is not None:
        return True
    stored = tx.fetch_one(
        "SELECT normal_form_hash, replay_key FROM datasets.frame_recipe "
        "WHERE subject_id = %s AND content_version = %s",
        pair,
    )
    if stored is None:
        holder = tx.fetch_one(
            "SELECT subject_id, content_version FROM datasets.frame_recipe WHERE replay_key = %s",
            (record.replay_key,),
        )
        if holder is None:
            # Recipe rows are never deleted, so the conflicting row cannot have gone.
            raise AssertionError("a recipe insert conflicted with no stored recipe row")
        raise ReplayKeyHeld(
            replay_key=record.replay_key,
            spec_id=record.spec_id,
            content_version=record.content_version,
            holder_spec_id=cast(str, holder["subject_id"]),
            holder_content_version=cast(str, holder["content_version"]),
        )
    for field in ("normal_form_hash", "replay_key"):
        if stored[field] != getattr(record, field):
            raise RecipeDiverges(
                spec_id=record.spec_id, content_version=record.content_version, field=field
            )
    return False


class LiveRecipeStore:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def write_recipe(self, record: RecipeRecord) -> bool:
        recorded_at = self._now()
        failure: str | None = None
        sqlstate: str | None = None
        refusal: RecipeRefusal | None = None
        created = False
        try:
            with self._store.transaction() as tx:
                created = _write_recipe(tx, record, recorded_at=recorded_at)
        except RecipeRefusal as exc:
            refusal = exc
        except Exception as exc:
            # Save only safe diagnostics, then raise after leaving this exception handler.
            failure = (
                f"writing recipe spec_id {record.spec_id!r} content_version "
                f"{record.content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if refusal is not None:
            raise refusal
        return created

    def read_recipe(self, *, spec_id: str, content_version: str) -> RecipeRecord | None:
        row = self._fetch_one(
            "SELECT normal_form_hash, replay_key, recipe FROM datasets.frame_recipe "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, content_version),
            reading=f"recipe spec_id {spec_id!r} content_version {content_version!r}",
        )
        if row is None:
            return None
        return RecipeRecord(
            spec_id=spec_id,
            content_version=content_version,
            normal_form_hash=cast(str, row["normal_form_hash"]),
            replay_key=cast(str, row["replay_key"]),
            body=cast(Mapping[str, object], row["recipe"]),
        )

    def read_replay(self, *, replay_key: str) -> ReplayedFrame | None:
        row = self._fetch_one(
            "SELECT subject_id, content_version FROM datasets.frame_recipe WHERE replay_key = %s",
            (replay_key,),
            reading=f"replay_key {replay_key!r}",
        )
        if row is None:
            return None
        return ReplayedFrame(
            spec_id=cast(str, row["subject_id"]),
            content_version=cast(str, row["content_version"]),
        )

    def _fetch_one(
        self, sql: str, params: tuple[object, ...], *, reading: str
    ) -> Mapping[str, object] | None:
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(sql, params)
        except Exception as exc:
            failure = f"reading {reading} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        return row


_DIVERGENCE_CONSTRAINT_IDS = {
    "normal_form_hash": "recipe_requires_matching_normal_form_hash",
    "replay_key": "recipe_requires_matching_replay_key",
}


def classify_recipe_store_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, RecipeFrameNotStored):
        return ProblemSpec(status=404, code="snapshot_not_found", detail=str(exc), retryable=False)

    if isinstance(exc, RecipeDiverges):
        logger.error(
            "Datasets recipe store refused a recipe for spec_id %s content_version %s: the "
            "stored recipe has a different %s",
            exc.spec_id,
            exc.content_version,
            exc.field,
        )
        return ProblemSpec(
            status=409,
            code="publish_conflict",
            detail=str(exc),
            retryable=False,
            constraint_id=_DIVERGENCE_CONSTRAINT_IDS[exc.field],
        )

    if isinstance(exc, ReplayKeyHeld):
        logger.error(
            "Datasets recipe store refused a recipe for spec_id %s content_version %s: "
            "replay_key %s is held by spec_id %s content_version %s",
            exc.spec_id,
            exc.content_version,
            exc.replay_key,
            exc.holder_spec_id,
            exc.holder_content_version,
        )
        return ProblemSpec(
            status=409,
            code="publish_conflict",
            detail=str(exc),
            retryable=False,
            constraint_id="replay_key_requires_one_frame",
        )

    return None
