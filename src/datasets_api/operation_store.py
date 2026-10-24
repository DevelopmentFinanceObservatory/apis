"""`LiveOperationStore` persists async jobs in `datasets.operation`.

`compute_operation_id` hashes canonical JSON of `request_identity` (the subject and its sorted
member dcids), `origin_catalogue_build_id`, and the rendering key. Identical inputs share an id.
Different renderings get separate operations, and so does a new catalogue build, whose admission
may change what the same request returns. A new graph release leaves the id unchanged, so it
never orphans a done operation. Raw options stay on the row, while sorted-key JSON makes mapping
order and type irrelevant to the id.

`create_or_get` inserts a new row `pending` with a NULL `dispatched_at` and commits. The caller
then creates the Cloud Tasks task and calls `mark_dispatched`, which sets `dispatched_at` once
for the submission. A `pending` row whose `dispatched_at` stays NULL had its create lost; the
operations poll creates the task again under the same name
(`datasets_api.operations.dispatch.dispatch_if_due`).

`attempt_count` is a monotonic fencing token. `claim` moves a `pending` row, or a `running` row
whose lease has ended, to `running`, increments the token and sets `claimed_at`. The lease is
static: a claim owns the row until `claimed_at + lease_seconds` and is never renewed. The claim
also stores `final_delivery`, whether the delivery that made it is the task's last. A
claim inside a live lease, for another submission, of a terminal row, or of a row past the
operation deadline returns `None`.
`progress`, `record_estimate`, `complete`, `fail`, and `release` require `state = 'running'` and
the claim's token, and return `None` once the token is stale or the row leaves `running`.
`release` returns a running row to `pending` and clears its lease and `final_delivery`, so the
next Cloud Tasks retry claims it at once.

A lease that has ended means its attempt has stopped owning the row. Whether another delivery
will run it is what `final_delivery` records. A row is abandoned when nothing will finish it:

- a `running` row whose lease has ended and whose `final_delivery` is true or NULL, since no
  later delivery is due or none is known;
- a `running` row whose lease has ended and whose `final_delivery` is false, once the operation
  deadline (`DATASETS_OPERATION_DEADLINE_SECONDS`) has passed since its dispatch, or its
  submission when no dispatch is recorded. Until then a later delivery is due and claims it;
- a `pending` row once the operation deadline has passed since its dispatch or else its
  submission.

A `running` row with no `claimed_at` was claimed by an image that renewed its lease by updating
`updated_at` and stored no `final_delivery`; its lease has ended once `updated_at` is
`LEASE_SECONDS` old. `abandon` fails an abandoned row as retryable `internal_error` in one
update fenced on the attempt its caller read. The operations poll calls it, and so does the
worker route for a delivery that arrives past the deadline, so the deadline ends a submission
whether or not anyone polls it. `fail_abandoned_operations` applies the same update to every
abandoned row, for the `datasets-maintenance age-operations` job.

`create_or_get` returns an existing row unchanged unless it is resettable: a retryable failure,
a `running` row whose lease has ended, whichever delivery claimed it, a `pending` row past the
operation deadline, or any `done` row. A submission reaches the store only when the subject's
head could not answer it, and the rerun serves through `serve_current`, which applies the head
rule at run time. So a `done` row resets whatever version it holds, the 202 reads `pending`, and
the run serves what the current head gives for this rendering. A reset of a row a later delivery
is still due for moves it to a new submission, and that delivery answers `superseded`. That
preserves the original options and estimate. A reset returns the row to `pending`, clears the
lease, `final_delivery`, `refusal_code`, `refusal`, `retryable` and `dispatched_at`, sets
`rows_written` to 0, and updates `submitted_at` while incrementing `submission_count`, so its
task gets a new name. Identity, options, admission coverage, and catalogue build keep their
admitted values. The attempt token carries on, so a superseded attempt's late writes stay
fenced. Deterministic failures stay unchanged.

A new row or a reset takes `pg_advisory_xact_lock`, re-reads the id under the lock, and counts
live rows other than its own: every `pending` or `running` row that is not abandoned, which
includes a row whose lease has ended while a later delivery is due. PostgreSQL READ COMMITTED
gives the re-read a fresh snapshot, so concurrent inserts or resets are visible. At or above
`DATASETS_OPERATION_BACKLOG_MAX`, the store raises `OperationBacklogFull`. Otherwise it inserts
or resets and reads the result back. `ON CONFLICT (id) DO NOTHING` protects the insert race.

`fail_unservable_done` fails a `done` row as retryable `internal_error` in one update guarded by
`state = 'done'` and the `attempt_count` the poll read. The operations poll calls it when the
snapshot lists no artifact for the row's rendering, so the next resubmit resets the row and
re-runs it. A re-run claims a higher attempt, so a poll that stalls across one leaves the newer
completion done, even when it repeats the content version.

`delete_terminal_operations` deletes `done` and `failed` rows whose `updated_at` is older than a
retention window, in bounded batches, reading only the store and the clock. It never touches a
`pending` or `running` row. The `datasets-maintenance age-operations` job calls it with the
advertised `operation_retention_days`, after `fail_abandoned_operations`.

`fail` stores the caller's retryability classification. Deterministic failures are not reset on
resubmit and carry their classified refusal in `refusal`. Transient failures and abandoned runs
are retryable and record `refusal_code` alone. A stored refusal this image cannot parse reads as
`None` with a logged warning, so the row still names its `refusal_code`. Every statement selects
the columns of `0024_operation_attempt_fencing.sql`, so the store needs that migration applied
before it runs. The clock is injectable and defaults to UTC now; every timestamp the store
compares comes from it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast

from pydantic import ValidationError

from dataset_store import DatasetStore
from datasets_api.backpressure import Backpressure
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operations.lease import LEASE_SECONDS
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
    "attempt_count, claimed_at, lease_seconds, created_at, updated_at, "
    "submission_count, retryable, dispatched_at, submitted_at, origin_catalogue_build_id, "
    "refusal, final_delivery, confirmed_at"
)

# The end of a running row's static lease.
_LEASE_END = "claimed_at + make_interval(secs => lease_seconds)"

# A running row whose lease ended, at the parameters now and now minus `LEASE_SECONDS`. A row
# with no `claimed_at` carries its lease in `updated_at`, which the image that claimed it renewed.
_LEASE_ENDED = (
    f"(state = 'running' AND ({_LEASE_END} < %s OR (claimed_at IS NULL AND updated_at < %s)))"
)

# The time the operation deadline runs from: the current submission's dispatch, or its
# submission when no dispatch is recorded.
_QUEUED_SINCE = "COALESCE(dispatched_at, submitted_at)"

# A pending row queued before the parameter, now minus the operation deadline.
_PENDING_PAST_DEADLINE = f"(state = 'pending' AND {_QUEUED_SINCE} < %s)"

# Rows nothing will finish, at the parameters `_abandoned_params` builds: now,
# the untracked-lease cutoff, and the deadline cutoff twice. A running row whose lease ended is
# abandoned at once unless its claim recorded that a later delivery is due
# (`final_delivery = FALSE`); that row and a pending row are abandoned past the deadline.
_ABANDONED = (
    f"(({_LEASE_ENDED} AND (final_delivery IS DISTINCT FROM FALSE OR {_QUEUED_SINCE} < %s)) "
    f"OR {_PENDING_PAST_DEADLINE})"
)

# Rows `claim` accepts at the parameters now, the untracked-lease cutoff and the deadline cutoff:
# unclaimed, or claimed under an ended lease, and queued inside the operation deadline. Every
# such row past the deadline is abandoned.
_CLAIMABLE = f"((state = 'pending' OR {_LEASE_ENDED}) AND {_QUEUED_SINCE} >= %s)"

# Rows `create_or_get` resets, at the parameters now, the untracked-lease cutoff and the deadline
# cutoff: a retryable failure, a running row whose lease ended, a pending row past the deadline,
# or any done row.
_RESETTABLE = (
    f"((state = 'failed' AND retryable) OR {_LEASE_ENDED} OR {_PENDING_PAST_DEADLINE} "
    "OR state = 'done')"
)

# Rows the backlog counts, at `_ABANDONED`'s parameters: everything in flight except an
# abandoned row. `IS NOT TRUE` counts a row whose lease columns are NULL.
_LIVE = f"(state IN ('pending', 'running') AND {_ABANDONED} IS NOT TRUE)"

# The update that fails a row as an abandoned run, at the parameter now.
_FAIL_ABANDONED = (
    "UPDATE datasets.operation SET state = 'failed', refusal_code = 'internal_error', "
    "retryable = TRUE, updated_at = %s"
)

# How long a running row with no `claimed_at` stays claimed after its last `updated_at`.
_UNTRACKED_LEASE = timedelta(seconds=LEASE_SECONDS)

# Serialises the backlog count and insert across transactions.
_BACKLOG_LOCK_KEY = 0x4446_4F5F_4F50_424C  # "DFO_OPBL" packed into an int


def utc_now() -> datetime:
    return datetime.now(UTC)


def compute_operation_id(
    *,
    request_identity: Mapping[str, object],
    origin_catalogue_build_id: str,
    options: Mapping[str, object],
) -> str:
    """Returns `op_` plus the SHA-256 hex digest of the canonical identity, origin catalogue
    build id, and rendering key."""
    canonical = json.dumps(
        {
            "request_identity": dict(request_identity),
            "origin_catalogue_build_id": origin_catalogue_build_id,
            "rendering": rendering_key(options),
        },
        sort_keys=True,
        default=str,
    )
    return "op_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OperationBacklogFull(Exception):
    """Raised when the live-row count is at `DATASETS_OPERATION_BACKLOG_MAX` before a new row is
    added. A known id returns its row, including after a concurrent insert or reset.

    A transaction-scoped advisory lock serialises the exact count and insert across API
    instances, so concurrent submissions cannot overshoot the limit. `claim` refuses a row
    past the operation deadline, so a late delivery cannot return to the count a `pending` row,
    or one a non-final delivery claimed, that the count has dropped. A delivery that reclaims a
    lease-ended row whose claim was the final delivery or recorded none does return that row to
    the count. The API maps this exception to retryable 503 `service_unavailable` with
    `Retry-After`.
    """

    def __init__(self, *, in_flight: int, limit: int) -> None:
        self.in_flight = in_flight
        self.limit = limit
        super().__init__(
            f"datasets.operation already holds {in_flight} live pending/running row(s), at or "
            f"above the configured backlog max {limit}"
        )


def classify_operation_store_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, OperationBacklogFull):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the async operations backlog is full; retry shortly",
            retryable=True,
            headers=Backpressure.headers,
        )
    return None


@dataclass(frozen=True)
class Operation:
    """One durable async operation.

    `subject_id`, `content_version` and `confirmed_at` are set together on completion. The poll
    uses them to rebuild a done row's manifest, `confirmed_at` being the instant the run
    confirmed the subject head, and `None` on a row an image recording no such instant
    completed. Failed rows carry `refusal_code` and `retryable` to mark
    transient failures. `refusal` is the classified refusal a deterministic failure ended on,
    and `None` on every other row, including one whose stored refusal this image cannot parse.
    `submission_count` starts at 0 and increments on each reset. Monotonic `attempt_count`
    fences worker writes. `claimed_at` and `lease_seconds` are the current claim's
    static lease, set on every `running` row. `final_delivery` is whether that claim was made
    by the task's last delivery, and `None` on a row that is not claimed or that an image
    recording no delivery claimed. `dispatched_at` is when the current submission's
    task create returned, `None` until then. `coverage_stamp` records admission, and `estimate`
    is the latest estimate, replaced by the worker's re-estimate. `origin_catalogue_build_id`
    is the catalogue build the submitting request was bound to, `None` on a row submitted
    before the column existed.
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
    claimed_at: datetime | None
    lease_seconds: int | None
    created_at: datetime
    updated_at: datetime
    submission_count: int
    retryable: bool | None
    dispatched_at: datetime | None
    submitted_at: datetime
    origin_catalogue_build_id: str | None
    refusal: ProblemSpec | None = None
    final_delivery: bool | None = None
    confirmed_at: datetime | None = None

    def lease_ended(self, now: datetime) -> bool:
        """Whether this is a `running` row whose claim has stopped owning it at `now`, the rule
        `_LEASE_ENDED` applies in SQL."""
        if self.state != "running":
            return False
        if self.claimed_at is None:
            return self.updated_at < now - _UNTRACKED_LEASE
        return (
            self.lease_seconds is not None
            and self.claimed_at + timedelta(seconds=self.lease_seconds) < now
        )

    def abandoned(self, now: datetime, *, deadline: timedelta) -> bool:
        """Whether nothing will finish this row at `now`, the rule `_ABANDONED` applies in
        SQL. A `running` row whose lease ended is abandoned unless its claim recorded that a
        later delivery is due (`final_delivery` false). That row, and a `pending` row, are
        abandoned once queued, since their dispatch or else their submission, more than
        `deadline` ago."""
        queued_since = self.dispatched_at if self.dispatched_at is not None else self.submitted_at
        past_deadline = queued_since < now - deadline
        if self.state == "pending":
            return past_deadline
        return self.lease_ended(now) and (self.final_delivery is not False or past_deadline)


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
        """Creates or reuses the operation keyed by identity, catalogue build, and rendering.

        A new row commits `pending` with no dispatch. Existing rows remain unchanged unless
        resettable; a `done` row always is. Raises `OperationBacklogFull` when a new row
        would exceed the configured backlog limit.
        """
        ...

    def mark_dispatched(self, id: str, *, submission_count: int) -> Operation | None:
        """Records that this submission's task create returned. Returns `None` when the row
        already records a dispatch or has moved to another submission."""
        ...

    def claim(
        self, id: str, *, submission_count: int, lease_seconds: int, final_delivery: bool
    ) -> Operation | None:
        """Claims a pending row, or a running row whose lease ended, for this submission and
        increments the fencing token. `final_delivery` records whether the claiming delivery is
        the task's last. Returns `None` for an unknown or terminal row, another submission, a
        live lease, or a row past the operation deadline.
        """
        ...

    def progress(self, id: str, *, attempt: int, rows_written: int) -> Operation | None: ...

    def record_estimate(self, id: str, *, attempt: int, estimate: Estimate) -> Operation | None:
        """Replaces the admission estimate with the worker's re-estimate. Returns `None` when
        this attempt token does not match a running row."""
        ...

    def complete(
        self,
        id: str,
        *,
        attempt: int,
        subject_id: str,
        content_version: str,
        confirmed_at: datetime,
    ) -> Operation | None:
        """Marks a running row `done` with the version its run served and the instant that run
        confirmed the subject head. Returns `None` when this attempt token does not match a
        running row."""
        ...

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

    def abandon(self, id: str, *, attempt: int) -> Operation | None:
        """Fails an abandoned row (`Operation.abandoned`) at that attempt as retryable
        `internal_error`. Returns `None` for any other row, including a running row whose lease
        ended while a later delivery is due, or an unknown id."""
        ...

    def fail_unservable_done(self, id: str, *, attempt: int) -> Operation | None:
        """Fails a `done` row whose rendering the snapshot lists no artifact for as retryable
        `internal_error`, so a resubmit resets and re-runs it. `attempt` is the `attempt_count`
        the poll read. Returns `None` for any row not `done` at that attempt or an unknown
        id."""
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

    def mark_dispatched(self, id: str, *, submission_count: int) -> Operation | None:
        return None

    def claim(
        self, id: str, *, submission_count: int, lease_seconds: int, final_delivery: bool
    ) -> Operation | None:
        return None

    def progress(self, id: str, *, attempt: int, rows_written: int) -> Operation | None:
        return None

    def record_estimate(self, id: str, *, attempt: int, estimate: Estimate) -> Operation | None:
        return None

    def complete(
        self,
        id: str,
        *,
        attempt: int,
        subject_id: str,
        content_version: str,
        confirmed_at: datetime,
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

    def abandon(self, id: str, *, attempt: int) -> Operation | None:
        return None

    def fail_unservable_done(self, id: str, *, attempt: int) -> Operation | None:
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
        claimed_at=cast("datetime | None", row["claimed_at"]),
        lease_seconds=cast("int | None", row["lease_seconds"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
        submission_count=cast(int, row["submission_count"]),
        retryable=cast("bool | None", row["retryable"]),
        dispatched_at=cast("datetime | None", row["dispatched_at"]),
        submitted_at=cast(datetime, row["submitted_at"]),
        origin_catalogue_build_id=cast("str | None", row["origin_catalogue_build_id"]),
        refusal=_stored_refusal(row),
        final_delivery=cast("bool | None", row["final_delivery"]),
        confirmed_at=cast("datetime | None", row["confirmed_at"]),
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


def _abandoned_params(
    now: datetime, *, deadline: timedelta
) -> tuple[datetime, datetime, datetime, datetime]:
    """The parameters of `_ABANDONED` and `_LIVE` at `now`."""
    deadline_cutoff = now - deadline
    return (now, now - _UNTRACKED_LEASE, deadline_cutoff, deadline_cutoff)


def _guarded[T](failure_prefix: str, run: Callable[[], T]) -> T:
    """Runs one store call, raising `DatasetsStoreUnavailable` with a safe summary of any
    error but `OperationBacklogFull`."""
    failure: str | None = None
    sqlstate: str | None = None
    try:
        return run()
    except OperationBacklogFull:
        # Policy refusals bypass store-error wrapping.
        raise
    except Exception as exc:
        # Store a safe summary. Raise outside this handler to keep `__context__` unset.
        failure = f"{failure_prefix} failed: {safe_diagnostics(exc)}"
        sqlstate = extract_sqlstate(exc)
    raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)


def _resettable(
    row: Mapping[str, object],
    now: datetime,
    *,
    deadline: timedelta,
) -> bool:
    """`_RESETTABLE` on a row already read."""
    operation = _operation_from_row(row)
    return (
        (operation.state == "failed" and bool(operation.retryable))
        or operation.lease_ended(now)
        or operation.abandoned(now, deadline=deadline)
        or operation.state == "done"
    )


class LiveOperationStore:
    def __init__(
        self,
        *,
        store: DatasetStore,
        now: Clock = utc_now,
        backlog_max: int | None = None,
        deadline_seconds: int | None = None,
        settings: DatasetsBuildSettings | None = None,
    ) -> None:
        """Uses `backlog_max` and `deadline_seconds`, else the caller's resolved settings,
        else the environment's."""
        if backlog_max is None or deadline_seconds is None:
            _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
            if backlog_max is None:
                backlog_max = _settings.operation_backlog_max
            if deadline_seconds is None:
                deadline_seconds = _settings.operation_deadline_seconds
        self._store = store
        self._now = now
        self._backlog_max = backlog_max
        self._deadline = timedelta(seconds=deadline_seconds)

    def _wrap(
        self, action: str, id: str, run: Callable[[], Mapping[str, object] | None]
    ) -> Mapping[str, object] | None:
        return _guarded(f"{action} operation {id!r}", run)

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
            origin_catalogue_build_id=origin_catalogue_build_id,
            options=options,
        )
        now = self._now()
        backlog_max = self._backlog_max

        def _run() -> Mapping[str, object] | None:
            with self._store.transaction() as tx:
                existing = tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )
                if existing is not None and not _resettable(existing, now, deadline=self._deadline):
                    return existing
                # Lock before counting or inserting a new row or resetting one.
                tx.execute("SELECT pg_advisory_xact_lock(%s)", (_BACKLOG_LOCK_KEY,))
                # Re-read under the lock. This statement's fresh snapshot controls the mutation.
                existing = tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )
                if existing is not None and not _resettable(existing, now, deadline=self._deadline):
                    return existing
                # The row being reset keeps its own place, so only the others count.
                backlog_row = tx.fetch_one(
                    f"SELECT count(*) AS n FROM datasets.operation WHERE id <> %s AND {_LIVE}",
                    (id, *_abandoned_params(now, deadline=self._deadline)),
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
                    # Reset only the resettable row seen under the backlog lock.
                    tx.execute(
                        "UPDATE datasets.operation SET state = 'pending', "
                        "claimed_at = NULL, lease_seconds = NULL, final_delivery = NULL, "
                        "refusal_code = NULL, refusal = NULL, retryable = NULL, rows_written = 0, "
                        "submission_count = submission_count + 1, dispatched_at = NULL, "
                        "submitted_at = %s, updated_at = %s "
                        f"WHERE id = %s AND {_RESETTABLE}",
                        (
                            now,
                            now,
                            id,
                            now,
                            now - _UNTRACKED_LEASE,
                            now - self._deadline,
                        ),
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

    def mark_dispatched(self, id: str, *, submission_count: int) -> Operation | None:
        now = self._now()
        return self._update_one(
            "recording the dispatch of",
            id,
            "UPDATE datasets.operation SET dispatched_at = %s, updated_at = %s "
            "WHERE id = %s AND submission_count = %s AND dispatched_at IS NULL",
            (now, now, id, submission_count),
        )

    def claim(
        self, id: str, *, submission_count: int, lease_seconds: int, final_delivery: bool
    ) -> Operation | None:
        now = self._now()
        return self._update_one(
            "claiming",
            id,
            "UPDATE datasets.operation SET state = 'running', "
            "attempt_count = attempt_count + 1, claimed_at = %s, lease_seconds = %s, "
            "final_delivery = %s, updated_at = %s "
            f"WHERE id = %s AND submission_count = %s AND {_CLAIMABLE}",
            (
                now,
                lease_seconds,
                final_delivery,
                now,
                id,
                submission_count,
                now,
                now - _UNTRACKED_LEASE,
                now - self._deadline,
            ),
        )

    def progress(self, id: str, *, attempt: int, rows_written: int) -> Operation | None:
        now = self._now()
        return self._update_one(
            "reporting progress for",
            id,
            "UPDATE datasets.operation SET rows_written = %s, updated_at = %s "
            "WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (rows_written, now, id, attempt),
        )

    def record_estimate(self, id: str, *, attempt: int, estimate: Estimate) -> Operation | None:
        now = self._now()
        return self._update_one(
            "recording the estimate for",
            id,
            "UPDATE datasets.operation SET estimate_rows = %s, estimate_requests = %s, "
            "updated_at = %s WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (estimate.rows, estimate.requests, now, id, attempt),
        )

    def complete(
        self,
        id: str,
        *,
        attempt: int,
        subject_id: str,
        content_version: str,
        confirmed_at: datetime,
    ) -> Operation | None:
        now = self._now()
        return self._update_one(
            "completing",
            id,
            "UPDATE datasets.operation SET state = 'done', subject_id = %s, "
            "content_version = %s, confirmed_at = %s, updated_at = %s "
            "WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (subject_id, content_version, confirmed_at, now, id, attempt),
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
            "refusal = %s, retryable = %s, updated_at = %s "
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
            "UPDATE datasets.operation SET state = 'pending', claimed_at = NULL, "
            "lease_seconds = NULL, final_delivery = NULL, updated_at = %s "
            "WHERE id = %s AND state = 'running' AND attempt_count = %s",
            (now, id, attempt),
        )

    def abandon(self, id: str, *, attempt: int) -> Operation | None:
        now = self._now()
        return self._update_one(
            "abandoning",
            id,
            f"{_FAIL_ABANDONED} WHERE id = %s AND attempt_count = %s AND {_ABANDONED}",
            (now, id, attempt, *_abandoned_params(now, deadline=self._deadline)),
        )

    def fail_unservable_done(self, id: str, *, attempt: int) -> Operation | None:
        now = self._now()
        return self._update_one(
            "failing an unservable completion of",
            id,
            "UPDATE datasets.operation SET state = 'failed', refusal_code = 'internal_error', "
            "retryable = TRUE, updated_at = %s "
            "WHERE id = %s AND state = 'done' AND attempt_count = %s",
            (now, id, attempt),
        )

    def get(self, id: str) -> Operation | None:
        def _run() -> Mapping[str, object] | None:
            with self._store.transaction() as tx:
                return tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.operation WHERE id = %s", (id,)
                )

        row = self._wrap("reading", id, _run)
        return _operation_from_row(row) if row is not None else None


def fail_abandoned_operations(
    *, store: DatasetStore, deadline: timedelta, now: Clock = utc_now
) -> int:
    """Fails every abandoned row (`Operation.abandoned`) as retryable `internal_error`, the
    update `LiveOperationStore.abandon` makes for one row.

    An abandoned row nobody polls and no delivery reaches otherwise stays `pending` or
    `running`. `deadline` is the operation deadline the API and the worker read. Returns the
    number of rows failed.
    """
    at = now()
    sql = f"{_FAIL_ABANDONED} WHERE {_ABANDONED} RETURNING id"

    def _run() -> int:
        with store.transaction() as tx:
            return len(tx.fetch_all(sql, (at, *_abandoned_params(at, deadline=deadline))))

    return _guarded("failing abandoned operations", _run)


def delete_terminal_operations(
    *,
    store: DatasetStore,
    older_than: timedelta,
    now: Clock = utc_now,
    batch_size: int = 1_000,
) -> int:
    """Deletes `done` and `failed` rows last updated more than `older_than` ago.

    It reads only `store` and the clock, so a job that runs it needs no settings. Each batch of
    at most `batch_size` rows commits in its own transaction, and the run stops at the first
    short batch. The outer predicate repeats the state and age test, so a row a concurrent
    resubmit resets to `pending` between the select and the delete stays. Returns the number of
    rows deleted.
    """
    cutoff = now() - older_than
    stale = "state IN ('done', 'failed') AND updated_at < %s"
    sql = (
        f"DELETE FROM datasets.operation WHERE {stale} AND id IN "
        f"(SELECT id FROM datasets.operation WHERE {stale} LIMIT %s) RETURNING id"
    )
    deleted = 0

    def _delete_batch() -> int:
        with store.transaction() as tx:
            return len(tx.fetch_all(sql, (cutoff, cutoff, batch_size)))

    while True:
        batch = _guarded("deleting terminal operations", _delete_batch)
        deleted += batch
        if batch < batch_size:
            return deleted
