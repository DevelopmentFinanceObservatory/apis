-- Two fixes to the async worker's own job record (`0011_operation.sql`).
--
-- `retryable` records, alongside `refusal_code`, whether a `failed`
-- operation is safe to restart under an identical resubmit:
-- `LiveOperationStore.fail` sets it from the caller's own classification
-- (`FALSE` for a deterministic refusal such as `excel_limit`, `TRUE` for a
-- transient one exhausted on its own last Cloud Tasks attempt),
-- `LiveOperationStore.expire_stale` always sets it `TRUE` -- every reason
-- that call fails an operation (an enqueue that never reached the queue, a
-- worker that stopped reporting, a task never claimed within its own
-- staleness window) is a transient condition of that one attempt, not of
-- the request itself. `create_or_get` resets a `retryable` `failed` row
-- back to `pending` for a resubmit of the identical request rather than
-- returning the poisoned row forever; a `FALSE` row still returns
-- unchanged, since the identical request would fail the identical way
-- again. Set together with `state = 'failed'`, and stays NULL otherwise --
-- the same discipline `refusal_code` already follows (0011's own comment).
--
-- `submission_count` distinguishes each such reset from the one before it:
-- `datasets_api.operations.task_client`'s task name folds it in alongside
-- `attempt_count`, so a reset operation's own re-enqueue never collides
-- with a task name a prior, now-terminal submission already used (Cloud
-- Tasks refuses a second task under a name it has recently seen, treating
-- the create as a no-op rather than genuinely dispatching a new task).
-- Starts at 0, the first submission, and increments by exactly 1 on each
-- reset.
--
-- The backlog-count query (`operation_store.py`, "SELECT count(*) ...
-- WHERE state IN ('pending', 'running')") filters on `state` alone and
-- runs on every genuinely new submission, inside the same advisory lock
-- that also now sweeps stale never-claimed rows before counting -- the
-- identical unindexed-full-table-scan shape `0008_artifact_location_tier_
-- index.sql` already fixed once for `datasets.artifact_location`'s own
-- tier-only filter. A partial index, scoped to the two states the count
-- ever filters on, covers it without also indexing the `done`/`failed`
-- rows the count never reads.

ALTER TABLE datasets.operation
    ADD COLUMN retryable boolean,
    ADD COLUMN submission_count int NOT NULL DEFAULT 0;

CREATE INDEX operation_backlog_state_idx ON datasets.operation (state)
    WHERE state IN ('pending', 'running');
