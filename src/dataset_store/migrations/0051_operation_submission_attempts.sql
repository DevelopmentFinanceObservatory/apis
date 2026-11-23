-- How many claims the current submission of an async operation has made.
--
-- `attempt_count` (`0024_operation_attempt_fencing.sql`) is the fencing token and carries on
-- across submissions, so it cannot say how many runs one submission has had.
-- `submission_attempts` can, together with `submission_attempts_of`, the `submission_count`
-- the count belongs to. `LiveOperationStore.claim` sets both in one statement: when
-- `submission_attempts_of` equals the row's `submission_count` it adds one to
-- `submission_attempts`, and otherwise it starts the count at 1. A reset, by this image or an
-- older one, increments `submission_count`, which starts the next submission's count without
-- touching either column, and `release` leaves both. The worker counts a run's place in the
-- attempt cap (`DATASETS_OPERATION_MAX_ATTEMPTS`) from the count as well as from Cloud Tasks'
-- retry count header, since a transient failure is retried under a new task whose header starts
-- again at 0 (`datasets_api.operations.worker_app`). A claim past the cap fails the row
-- retryable without running it.
--
-- Both columns are NULL on a row written before them and on a row no claim of this image has
-- counted, and the next claim counts itself its submission's first.
--
-- Adding nullable columns with no default rewrites no row. ALTER TABLE takes ACCESS EXCLUSIVE
-- on `operation`, and `lock_timeout` fails it after five seconds so that it does not queue
-- every later write behind it. Apply this migration before deploying an API or worker image
-- that reads the columns: every `LiveOperationStore` statement in such an image that returns an
-- operation row selects them. An image older than the columns runs unchanged against the
-- migrated table. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.operation
    ADD COLUMN IF NOT EXISTS submission_attempts integer,
    ADD COLUMN IF NOT EXISTS submission_attempts_of integer;
