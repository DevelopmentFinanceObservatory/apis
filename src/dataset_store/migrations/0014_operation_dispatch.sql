-- Dispatch bookkeeping, the current submission's time, and the
-- execution-time coverage stamp for the async worker's job record
-- (`0011_operation.sql`).
--
-- `dispatch_count` is the dispatch intent: `LiveOperationStore.claim_dispatch`
-- bumps it in one conditional UPDATE before the caller creates the Cloud
-- Tasks task, so exactly one caller owns each dispatch of a submission.
-- `dispatched_at` records when that intent was taken. `dispatch_confirmed`
-- is the highest `dispatch_count` whose task creation returned (or reported
-- the task already exists), written by `LiveOperationStore.confirm_dispatch`
-- after the create call. A `pending`, never-claimed row whose
-- `dispatch_confirmed` trails its `dispatch_count` past a short grace had an
-- enqueue with an unknown outcome and may be dispatched again, up to a cap
-- per submission; a confirmed row belongs to Cloud Tasks and is never
-- dispatched twice. All three reset with each new submission of the row.
--
-- `submitted_at` is when the row's current submission was made: the
-- insert writes it, and each reset of a retryable `failed` row to a new
-- submission rewrites it. The never-claimed dead rule measures its window
-- from the later of `submitted_at` and `dispatched_at`, so a resubmit of
-- an old operation is judged from its own submission. Existing rows take
-- `created_at` for their first submission, and `updated_at` for a later
-- one: a reset is the last write to a row that is still `pending` and
-- unclaimed, the only rows that rule reads. `DEFAULT now()` covers an
-- insert that predates this column (an older worker image's own
-- `OperationStore.create_or_get`, mid-rollout or rolled back): it still
-- satisfies the `NOT NULL` below and gets its own insert time, rather
-- than failing the insert outright.
--
-- `execution_coverage_stamp` is the coverage stamp the worker executed
-- against, beside `coverage_stamp`, the stamp the operation was admitted
-- under (and whose value `compute_operation_id` hashes). The worker writes
-- it once it takes its execution-time coverage snapshot, and it stays NULL until then.

ALTER TABLE datasets.operation
    ADD COLUMN dispatch_count int NOT NULL DEFAULT 0,
    ADD COLUMN dispatched_at timestamptz,
    ADD COLUMN dispatch_confirmed int NOT NULL DEFAULT 0,
    ADD COLUMN execution_coverage_stamp text,
    ADD COLUMN submitted_at timestamptz DEFAULT now();

UPDATE datasets.operation
    SET submitted_at = CASE WHEN submission_count = 0 THEN created_at ELSE updated_at END;

ALTER TABLE datasets.operation ALTER COLUMN submitted_at SET NOT NULL;
