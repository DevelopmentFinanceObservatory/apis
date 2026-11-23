-- How many claims the current submission of an async operation has made.
--
-- `attempt_count` (`0024_operation_attempt_fencing.sql`) is the fencing token and carries on
-- across submissions, so it cannot say how many runs one submission has had.
-- `submission_attempts` can: `LiveOperationStore.claim` adds one, a resubmit's reset sets it to
-- 0, and `release` leaves it. The worker counts a run's place in the attempt cap
-- (`DATASETS_OPERATION_MAX_ATTEMPTS`) from it as well as from Cloud Tasks' retry count header,
-- since a transient failure is retried under a new task whose header starts again at 0
-- (`datasets_api.operations.worker_app`). A claim past the cap fails the row retryable without
-- running it.
--
-- NULL on a row written before this column, and on a row an older image created; a claim reads
-- NULL as 0. An older image's reset leaves the column as it was, so a row it resets carries the
-- earlier submission's count until this image resets it again; Cloud Tasks' own retry count
-- still bounds that submission.
--
-- Adding a nullable column with no default rewrites no row. ALTER TABLE takes ACCESS EXCLUSIVE
-- on `operation`, and `lock_timeout` fails it after five seconds so that it does not queue
-- every later write behind it. Apply this migration before deploying an API or worker image
-- that reads the column: every `LiveOperationStore` statement in such an image that returns an
-- operation row selects it. An image older than the column runs unchanged against the migrated
-- table. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.operation
    ADD COLUMN IF NOT EXISTS submission_attempts integer;
