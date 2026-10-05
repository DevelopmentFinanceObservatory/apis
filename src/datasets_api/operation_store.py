"""`LiveOperationStore` persists async jobs in `datasets.operation`.

`compute_operation_id` hashes canonical JSON of `request_identity`, `coverage_stamp`,
`origin_catalogue_build_id`, and the rendering key. Identical inputs share an id. Different
renderings get separate operations, and so does a new catalogue build, whose admission may change
what the same request returns. Raw options stay on the row, while sorted-key JSON makes mapping
order and type irrelevant to the id.

Every `create_or_get` call first sweeps dead `pending` and `running` rows in a separate,
committed transaction. Claimed or released rows expire after their lease has been stale for
`lease_stale_after_seconds`. Unclaimed rows expire after their last dispatch, or submission
time if undispatched, passes `never_claimed_stale_after_seconds`. The latter window also gives
queued tasks time to arrive after `MAX_REDISPATCHES`. Expiry marks rows retryable `internal_error`.
The sweep commits even if a later backlog check rolls back. PostgreSQL rechecks the predicate
against each row's latest committed version, preserving concurrent renewals and completions.
`expire_if_dead` applies the same rule to one operation.



`create_or_get` returns an existing row unchanged unless it is a retryable failure. This
preserves its original options and estimate. A retryable failure resets in place to `pending`.
It clears `lease_until`, `refusal_code`, `refusal`, `retryable`, dispatch fields, and
`execution_coverage_stamp`, sets `rows_written` to 0, and updates `submitted_at` while
incrementing `submission_count`. Identity, options, admission coverage, and catalogue build keep
their admitted values. The last estimate stays until execution re-estimates it. Deterministic failures and
completed rows stay unchanged.

A new row or retryable reset takes `pg_advisory_xact_lock`, re-reads the id under the lock, and
counts `pending`/`running` rows after the sweep. PostgreSQL READ COMMITTED gives the re-read a
fresh snapshot, so concurrent inserts or resets are visible. At or above
`DATASETS_OPERATION_BACKLOG_MAX`, the store raises `OperationBacklogFull`. Otherwise it inserts
or resets and reads the result back. `ON CONFLICT (id) DO NOTHING` protects the insert race.
The backlog lock protects the count.



Dispatch uses one conditional update per submission. `claim_dispatch` allows the first task or an
unconfirmed task older than `REDISPATCH_GRACE_SECONDS`, up to `MAX_REDISPATCHES`. One caller wins
and creates the task. Confirmed tasks and released rows wait for Cloud Tasks. Dead-row expiry
closes confirmed tasks that miss the claim window.

`attempt_count` is a monotonic fencing token. `claim_lease` accepts pending or running rows for
the task's `submission_count` with no live lease. It returns `None` for an unknown, terminal, or
ineligible row. Worker writes require `state = 'running'` and the claim's attempt token. They
return `None` when the token is stale or the row leaves `running`.

`progress` renews the lease in the same update as `rows_written` when `lease_seconds` is set.
`release` returns a running row to pending, sets its lease to now, and preserves its attempt and
progress. The immediate lease expiry lets the next Cloud Tasks retry claim it. The non-null
lease keeps dead-row expiry on the claimed-row window and excludes redispatch.

`fail` stores the caller's retryability classification. Deterministic failures are not reset on
resubmit and carry their classified refusal in `refusal`. Transient failures and dead-row expiry
are retryable and record `refusal_code` alone. A stored refusal this image cannot parse reads as
`None` with a logged warning, so the row still names its `refusal_code`. Every statement selects
`refusal`, so the store needs migration `0016_operation_refusal.sql` applied before it runs. The
clock is injectable and defaults to UTC now.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast

from pydantic import ValidationError

from dataset_store import DatasetStore
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operations.task_client import DISPATCH_DEADLINE_SECONDS
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import rendering_key
from dfo_platform import ProblemSpec
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics
from series_coverage.index import Estimate

logger = logging.getLogger(__name__)

OperationState = Literal["pending", "running", "done", "failed"]

Clock = Callable[[], datetime]

_COLUMNS = (
    "id, request_identity, options, coverage_stamp, state, estimate_rows, "
    "estimate_requests, rows_written, subject_id, content_version, refusal_code, "
    "attempt_count, lease_until, created_at, updated_at, submission_count, retryable, "
    "dispatch_count, dispatched_at, dispatch_confirmed, execution_coverage_stamp, submitted_at, "
    "origin_catalogue_build_id, refusal"
)

# Shared by the sweep and `expire_if_dead`. Parameters are lease and unclaimed-age cutoffs.
# The latter age starts at the last dispatch, or submission time when undispatched.
_DEAD_PREDICATE = (
    "state IN ('pending', 'running') AND ("
    "(lease_until IS NOT NULL AND lease_until < %s) "
    "OR (lease_until IS NULL AND state = 'pending' "
    "AND COALESCE(dispatched_at, submitted_at) < %s))"
)

# Parameters are the redispatch cutoff and cap. The cap allows the first dispatch plus retries.
_DISPATCHABLE_PREDICATE = (
    "state = 'pending' AND lease_until IS NULL AND (dispatch_count = 0 OR ("
    "dispatch_confirmed < dispatch_count AND dispatched_at < %s AND dispatch_count <= %s))"
)

# Wait longer than one enqueue timeout before retrying an unconfirmed dispatch.
REDISPATCH_GRACE_SECONDS = 30

# Unknown enqueue outcomes allow three redispatches after the first task.
MAX_REDISPATCHES = 3

# Serialises the backlog count and insert across transactions.
_BACKLOG_LOCK_KEY = 0x4446_4F5F_4F50_424C  # "DFO_OPBL" packed into an int


def utc_now() -> datetime:
    return datetime.now(UTC)


def never_claimed_stale_after_seconds(
    *,
    backlog_max: int,
    dispatch_concurrency: int,
    dispatch_deadline_seconds: float = DISPATCH_DEADLINE_SECONDS,
    max_retry_duration_seconds: float,
) -> float:
    """Returns the expiry age for a pending row with no claimed lease.

    Measure from `dispatched_at`, or `submitted_at` if undispatched. A full queue delays a task
    by `ceil(backlog_max / dispatch_concurrency) * dispatch_deadline_seconds`. Add
    `max_retry_duration_seconds` for the retry window after its first dispatch.
    """
    return (
        math.ceil(backlog_max / dispatch_concurrency) * dispatch_deadline_seconds
        + max_retry_duration_seconds
    )


# Fixed allowance for clock skew when checking expired leases.
LEASE_STALE_MARGIN_SECONDS = 300


def lease_stale_after_seconds(*, max_retry_duration_seconds: float) -> float:
    """Returns the expired-lease window for claimed rows.

    `max_retry_duration_seconds` covers Cloud Tasks retries. The 300-second margin absorbs
    clock skew.
    """
    return max_retry_duration_seconds + LEASE_STALE_MARGIN_SECONDS


def compute_operation_id(
    *,
    request_identity: Mapping[str, object],
    coverage_stamp: str,
    origin_catalogue_build_id: str,
    options: Mapping[str, object],
) -> str:
    """Returns `op_` plus the SHA-256 hex digest of the canonical identity, coverage stamp,
    origin catalogue build id, and rendering key."""
    canonical = json.dumps(
        {
            "request_identity": dict(request_identity),
            "coverage_stamp": coverage_stamp,
            "origin_catalogue_build_id": origin_catalogue_build_id,
            "rendering": rendering_key(options),
        },
        sort_keys=True,
        default=str,
    )
    return "op_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OperationBacklogFull(Exception):
    """Raised when the pending/running count is at `DATASETS_OPERATION_BACKLOG_MAX` before a
    new row is added. A known id returns its row, including after a concurrent insert or reset.

    A transaction-scoped advisory lock serialises the exact count and insert across API
    instances, so the limit cannot be overshot. The API maps this exception to retryable 503
    `service_unavailable` with `Retry-After`.
    """

    def __init__(self, *, in_flight: int, limit: int) -> None:
        self.in_flight = in_flight
        self.limit = limit
        super().__init__(
            f"datasets.operation already holds {in_flight} pending/running row(s), at or "
            f"above the configured backlog max {limit}"
        )


def classify_operation_store_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, OperationBacklogFull):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the async operations backlog is full; retry shortly",
            retryable=True,
            headers={"Retry-After": "1"},
        )
    return None


@dataclass(frozen=True)
class Operation:
    """One durable async operation.

    `subject_id` and `content_version` are set together on completion. The poll uses them to
    rebuild a done row's manifest. Failed rows carry `refusal_code` and `retryable` to mark
    transient failures. `refusal` is the classified refusal a deterministic failure ended on,
    and `None` on every other row, including a failure recorded before the column existed and
    one whose stored refusal this image cannot parse. `submission_count` starts at 0 and
    increments on retryable resets.
    Monotonic `attempt_count` fences worker writes. Dispatch fields belong to the current
    submission. `coverage_stamp` records admission. `execution_coverage_stamp` and `estimate`
    describe execution after `record_execution`. `origin_catalogue_build_id` is the catalogue
    build the submitting request was bound to, `None` on a row submitted before the column
    existed. The build a completed operation's data was admitted under is its manifest's
    `trace.origin_catalogue_build_id`.
    """

    id: str
    request_identity: Mapping[str, object]
    options: Mapping[str, object]
    coverage_stamp: str
    state: OperationState
    estimate: Estimate
    rows_written: int
    subject_id: str | None
    content_version: str | None
    refusal_code: str | None
    attempt_count: int
    lease_until: datetime | None
    created_at: datetime
    updated_at: datetime
    submission_count: int
    retryable: bool | None
    dispatch_count: int
    dispatched_at: datetime | None
    dispatch_confirmed: int
    execution_coverage_stamp: str | None
    submitted_at: datetime
    origin_catalogue_build_id: str | None
    refusal: ProblemSpec | None = None


class OperationStore(Protocol):
    def create_or_get(
        self,
        *,
        request_identity: Mapping[str, object],
        options: Mapping[str, object],
        coverage_stamp: str,
        origin_catalogue_build_id: str,
        estimate: Estimate,
    ) -> Operation:
        """Creates or reuses the operation keyed by identity, coverage stamp, catalogue build,
        and rendering.

        Existing rows remain unchanged. Raises `OperationBacklogFull` when a new row would exceed
        the configured backlog limit.
        """
        ...

    def claim_dispatch(self, id: str) -> Operation | None:
        """Claims a due dispatch and returns `None` when none is due. Only the winner enqueues."""
        ...

    def confirm_dispatch(self, id: str, *, submission_count: int, dispatch_count: int) -> None:
        """Confirms a dispatch for its submission. Stale submission counts are ignored."""
        ...

    def claim_lease(
        self, id: str, *, submission_count: int, lease_seconds: float
    ) -> Operation | None:
        """Claims an expired or unclaimed lease for this submission and increments the fencing
        token. Returns `None` for an unknown or terminal row, a stale submission, or a live lease.
        """
        ...

    def progress(
        self, id: str, *, attempt: int, rows_written: int, lease_seconds: float | None = None
    ) -> Operation | None: ...

    def complete(
        self, id: str, *, attempt: int, subject_id: str, content_version: str
    ) -> Operation | None: ...

    def fail(
        self,
        id: str,
        *,
        attempt: int,
        refusal_code: str,
        retryable: bool,
        refusal: ProblemSpec | None = None,
    ) -> Operation | None:
        """Stores the failure code and whether a resubmit may retry the operation. `refusal`
        is the classified refusal, stored as `ProblemSpec.to_canonical_data()` for the poll to
        render."""
        ...

    def release(self, id: str, *, attempt: int) -> Operation | None: ...

    def record_execution(
        self, id: str, *, attempt: int, coverage_stamp: str, estimate: Estimate
    ) -> Operation | None:
        """Records the execution coverage stamp and estimate, replacing the admission estimate.
        Returns `None` when this attempt token does not match a running row.
        """
        ...

    def expire_if_dead(self, id: str) -> Operation | None:
        """Fails a dead row as retryable `internal_error`. Returns `None` for a live, terminal,
        or unknown id."""
        ...

    def get(self, id: str) -> Operation | None:
        """Returns any operation state, or `None` for an unknown id."""
        ...


class NullOperationStore:
    """A no-op store whose reads and updates return `None`.

    `create_or_get` raises to enforce the pollable-operation contract. Use this for tests or
    deployments where operation handling is disabled.
    """

    def create_or_get(
        self,
        *,
        request_identity: Mapping[str, object],
        options: Mapping[str, object],
        coverage_stamp: str,
        origin_catalogue_build_id: str,
        estimate: Estimate,
    ) -> Operation:
        raise NotImplementedError(
            "NullOperationStore cannot create an operation it could never find again"
        )

    def claim_dispatch(self, id: str) -> Operation | None:
        return None

    def confirm_dispatch(self, id: str, *, submission_count: int, dispatch_count: int) -> None:
        return None

    def claim_lease(
        self, id: str, *, submission_count: int, lease_seconds: float
    ) -> Operation | None:
        return None

    def progress(
        self, id: str, *, attempt: int, rows_written: int, lease_seconds: float | None = None
    ) -> Operation | None:
        return None

    def complete(
        self, id: str, *, attempt: int, subject_id: str, content_version: str
    ) -> Operation | None:
        return None

    def fail(
        self,
        id: str,
        *,
        attempt: int,
        refusal_code: str,
        retryable: bool,
        refusal: ProblemSpec | None = None,
    ) -> Operation | None:
        return None

    def release(self, id: str, *, attempt: int) -> Operation | None:
        return None

    def record_execution(
        self, id: str, *, attempt: int, coverage_stamp: str, estimate: Estimate
    ) -> Operation | None:
        return None

    def expire_if_dead(self, id: str) -> Operation | None:
        return None

    def get(self, id: str) -> Operation | None:
        return None


def _operation_from_row(row: Mapping[str, object]) -> Operation:
    return Operation(
        id=cast(str, row["id"]),
        request_identity=cast(Mapping[str, object], row["request_identity"]),
        options=cast(Mapping[str, object], row["options"]),
        coverage_stamp=cast(str, row["coverage_stamp"]),
        state=cast(OperationState, row["state"]),
        estimate=Estimate(
            rows=cast(int, row["estimate_rows"]), requests=cast(int, row["estimate_requests"])
        ),
        rows_written=cast(int, row["rows_written"]),
        subject_id=cast("str | None", row["subject_id"]),
        content_version=cast("str | None", row["content_version"]),
        refusal_code=cast("str | None", row["refusal_code"]),
        attempt_count=cast(int, row["attempt_count"]),
        lease_until=cast("datetime | None", row["lease_until"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
        submission_count=cast(int, row["submission_count"]),
        retryable=cast("bool | None", row["retryable"]),
        dispatch_count=cast(int, row["dispatch_count"]),
        dispatched_at=cast("datetime | None", row["dispatched_at"]),
        dispatch_confirmed=cast(int, row["dispatch_confirmed"]),
        execution_coverage_stamp=cast("str | None", row["execution_coverage_stamp"]),
        submitted_at=cast(datetime, row["submitted_at"]),
        origin_catalogue_build_id=cast("str | None", row["origin_catalogue_build_id"]),
        refusal=_stored_refusal(row),
    )


def _stored_refusal(row: Mapping[str, object]) -> ProblemSpec | None:
    """Parse the row's `refusal`, or return `None` when it holds none or one this image cannot
    parse: a key a newer worker image wrote, or a value of the wrong type. `refusal_code`
    still names the failure, so the poll answers with the code alone."""
    stored = cast("Mapping[str, object] | None", row["refusal"])
    if stored is None:
        return None
    try:
        return ProblemSpec.from_canonical_data(stored)
    except ValidationError as exc:
        logger.warning(
            "Datasets operation %r stores a refusal this image cannot parse; reading it as "
            "refusal_code %r alone: %s",
            row["id"],
            row["refusal_code"],
            exc,
        )
        return None


class LiveOperationStore:
    def __init__(
        self,
        *,
        store: DatasetStore,
        now: Clock = utc_now,
        backlog_max: int | None = None,
        dispatch_concurrency: int | None = None,
        settings: DatasetsBuildSettings | None = None,
    ) -> None:
        """Uses the caller's resolved settings when supplied. Otherwise reads defaults from the
        environment. Shared settings keep operation limits consistent with the doors.
        """
        _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
        self._store = store
        self._now = now
        self._backlog_max = (
            backlog_max if backlog_max is not None else _settings.operation_backlog_max
        )
        self._never_claimed_stale_after = timedelta(
            seconds=never_claimed_stale_after_seconds(
                backlog_max=backlog_max
                if backlog_max is not None
                else _settings.operation_backlog_max,
                dispatch_concurrency=dispatch_concurrency
                if dispatch_concurrency is not None
                else _settings.operation_dispatch_concurrency,
                max_retry_duration_seconds=_settings.operation_max_retry_duration_seconds,
            )
        )
        self._lease_stale_after = timedelta(
            seconds=lease_stale_after_seconds(
                max_retry_duration_seconds=_settings.operation_max_retry_duration_seconds
            )
        )

    def _dead_cutoffs(self, now: datetime) -> tuple[datetime, datetime]:
        """`_DEAD_PREDICATE`'s parameters at `now`."""
        return (now - self._lease_stale_after, now - self._never_claimed_stale_after)

    def _sweep_dead_operations(self, *, now: datetime) -> None:
        """Commits one dead-row sweep before each `create_or_get` lookup and backlog count."""
        cutoffs = self._dead_cutoffs(now)

        def _run() -> None:
            with self._store.transaction() as tx:
                tx.execute(
                    "UPDATE datasets.operation SET state = 'failed', "
                    "refusal_code = 'internal_error', retryable = TRUE, "
                    f"lease_until = NULL, updated_at = %s WHERE {_DEAD_PREDICATE}",
                    (now, *cutoffs),
                )

        self._wrap("sweeping dead", "*", _run)

    def _wrap(self, action: str, id: str, run: Callable[[], Mapping[str, object] | None]):
        failure: str | None = None
        sqlstate: str | None = None
        row: Mapping[str, object] | None = None
        try:
            row = run()
        except OperationBacklogFull:
            # Policy refusals bypass store-error wrapping.
            raise
        except Exception as exc:
            # Store a safe summary. Raise outside this handler to keep `__context__` unset.
            failure = f"{action} operation {id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        return row

    def _update_one(
        self, action: str, id: str, sql: str, params: tuple[object, ...]
    ) -> Operation | None:
        """Runs one transactional update and returns its row, if any."""

        def _run() -> Mapping[str, object] | None:
            with self._store.transaction() as tx:
                return tx.fetch_one(f"{sql} RETURNING {_COLUMNS}", params)

        row = self._wrap(action, id, _run)
        return _operation_from_row(row) if row is not None else None

    def create_or_get(
        self,
        *,
        request_identity: Mapping[str, object],
        options: Mapping[str, object],
        coverage_stamp: str,
        origin_catalogue_build_id: str,
        estimate: Estimate,
    ) -> Operation:
        id = compute_operation_id(
            request_identity=request_identity,
            coverage_stamp=coverage_stamp,
            origin_catalogue_build_id=origin_catalogue_build_id,
            options=options,
        )
        now = self._now()
        backlog_max = self._backlog_max
        self._sweep_dead_operations(now=now)

        def _run() -> Mapping[str, object] | None:
            with self._store.transaction() as tx:
                existing = tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )
                resettable = (
                    existing is not None and existing["state"] == "failed" and existing["retryable"]
                )
                if existing is not None and not resettable:
                    return existing
                # Lock before counting or inserting a new row or resetting a retryable failure.
                tx.execute("SELECT pg_advisory_xact_lock(%s)", (_BACKLOG_LOCK_KEY,))
                # Re-read under the lock. This statement's fresh snapshot controls the mutation.
                existing = tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )
                resettable = (
                    existing is not None and existing["state"] == "failed" and existing["retryable"]
                )
                if existing is not None and not resettable:
                    return existing
                # The separate sweep transaction has already committed before this count.
                backlog_row = tx.fetch_one(
                    "SELECT count(*) AS n FROM datasets.operation WHERE state IN "
                    "('pending', 'running')"
                )
                in_flight = cast(int, backlog_row["n"]) if backlog_row is not None else 0
                if in_flight >= backlog_max:
                    raise OperationBacklogFull(in_flight=in_flight, limit=backlog_max)
                if existing is None:
                    tx.execute(
                        "INSERT INTO datasets.operation "
                        "(id, request_identity, options, coverage_stamp, state, estimate_rows, "
                        "estimate_requests, rows_written, attempt_count, created_at, updated_at, "
                        "submitted_at, origin_catalogue_build_id) "
                        "VALUES (%s, %s, %s, %s, 'pending', %s, %s, 0, 0, %s, %s, %s, %s) "
                        "ON CONFLICT (id) DO NOTHING",
                        (
                            id,
                            dict(request_identity),
                            dict(options),
                            coverage_stamp,
                            estimate.rows,
                            estimate.requests,
                            now,
                            now,
                            now,
                            origin_catalogue_build_id,
                        ),
                    )
                else:
                    # Reset only the retryable failure seen under the backlog lock.
                    tx.execute(
                        "UPDATE datasets.operation SET state = 'pending', "
                        "lease_until = NULL, refusal_code = NULL, refusal = NULL, "
                        "retryable = NULL, rows_written = 0, submission_count = submission_count + 1, "
                        "dispatch_count = 0, dispatched_at = NULL, dispatch_confirmed = 0, "
                        "execution_coverage_stamp = NULL, submitted_at = %s, updated_at = %s "
                        "WHERE id = %s AND state = 'failed' AND retryable",
                        (now, now, id),
                    )
                return tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )

        row = self._wrap("creating", id, _run)
        if row is None:
            raise DatasetsStoreUnavailable(
                f"creating operation {id!r} failed: insert-then-select returned no row"
            )
        return _operation_from_row(row)

    def claim_dispatch(self, id: str) -> Operation | None:
        now = self._now()
        return self._update_one(
            "claiming a dispatch for",
            id,
            "UPDATE datasets.operation SET dispatch_count = dispatch_count + 1, "
            "dispatched_at = %s, updated_at = %s "
            f"WHERE id = %s AND {_DISPATCHABLE_PREDICATE}",
            (
                now,
                now,
                id,
                now - timedelta(seconds=REDISPATCH_GRACE_SECONDS),
                MAX_REDISPATCHES,
            ),
        )

    def confirm_dispatch(self, id: str, *, submission_count: int, dispatch_count: int) -> None:
        now = self._now()
        self._update_one(
            "confirming a dispatch for",
            id,
            "UPDATE datasets.operation SET dispatch_confirmed = %s, updated_at = %s "
            "WHERE id = %s AND submission_count = %s AND dispatch_confirmed < %s",
            (dispatch_count, now, id, submission_count, dispatch_count),
        )

    def claim_lease(
        self, id: str, *, submission_count: int, lease_seconds: float
    ) -> Operation | None:
        now = self._now()
        return self._update_one(
            "claiming a lease for",
            id,
            "UPDATE datasets.operation SET state = 'running', lease_until = %s, "
            "attempt_count = attempt_count + 1, updated_at = %s "
            "WHERE id = %s AND state IN ('pending', 'running') AND submission_count = %s "
            "AND (lease_until IS NULL OR lease_until <= %s)",
            (now + timedelta(seconds=lease_seconds), now, id, submission_count, now),
        )

    def progress(
        self, id: str, *, attempt: int, rows_written: int, lease_seconds: float | None = None
    ) -> Operation | None:
        now = self._now()
        if lease_seconds is None:
            return self._update_one(
                "reporting progress for",
                id,
                "UPDATE datasets.operation SET rows_written = %s, updated_at = %s "
                "WHERE id = %s AND state = 'running' AND attempt_count = %s",
                (rows_written, now, id, attempt),
            )
        return self._update_one(
            "reporting progress for",
            id,
            "UPDATE datasets.operation SET rows_written = %s, lease_until = %s, "
            "updated_at = %s WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (rows_written, now + timedelta(seconds=lease_seconds), now, id, attempt),
        )

    def complete(
        self, id: str, *, attempt: int, subject_id: str, content_version: str
    ) -> Operation | None:
        now = self._now()
        return self._update_one(
            "completing",
            id,
            "UPDATE datasets.operation SET state = 'done', subject_id = %s, "
            "content_version = %s, lease_until = NULL, updated_at = %s "
            "WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (subject_id, content_version, now, id, attempt),
        )

    def fail(
        self,
        id: str,
        *,
        attempt: int,
        refusal_code: str,
        retryable: bool,
        refusal: ProblemSpec | None = None,
    ) -> Operation | None:
        now = self._now()
        return self._update_one(
            "failing",
            id,
            "UPDATE datasets.operation SET state = 'failed', refusal_code = %s, "
            "refusal = %s, retryable = %s, lease_until = NULL, updated_at = %s "
            "WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (
                refusal_code,
                None if refusal is None else refusal.to_canonical_data(),
                retryable,
                now,
                id,
                attempt,
            ),
        )

    def release(self, id: str, *, attempt: int) -> Operation | None:
        now = self._now()
        return self._update_one(
            "releasing",
            id,
            "UPDATE datasets.operation SET state = 'pending', lease_until = %s, "
            "updated_at = %s WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (now, now, id, attempt),
        )

    def record_execution(
        self, id: str, *, attempt: int, coverage_stamp: str, estimate: Estimate
    ) -> Operation | None:
        now = self._now()
        return self._update_one(
            "recording execution for",
            id,
            "UPDATE datasets.operation SET execution_coverage_stamp = %s, estimate_rows = %s, "
            "estimate_requests = %s, updated_at = %s "
            "WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (coverage_stamp, estimate.rows, estimate.requests, now, id, attempt),
        )

    def expire_if_dead(self, id: str) -> Operation | None:
        now = self._now()
        return self._update_one(
            "expiring",
            id,
            "UPDATE datasets.operation SET state = 'failed', refusal_code = 'internal_error', "
            "retryable = TRUE, lease_until = NULL, updated_at = %s "
            f"WHERE id = %s AND {_DEAD_PREDICATE}",
            (now, id, *self._dead_cutoffs(now)),
        )

    def get(self, id: str) -> Operation | None:
        def _run() -> Mapping[str, object] | None:
            with self._store.transaction() as tx:
                return tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )

        row = self._wrap("reading", id, _run)
        return _operation_from_row(row) if row is not None else None
