-- The async worker's job record (`0011_operation.sql`) fenced on its attempt
-- token, with a static lease and enqueue recovery.
--
-- `claimed_at` and `lease_seconds` are the static lease a worker delivery
-- takes in `LiveOperationStore.claim`: a `running` row belongs to the attempt
-- that claimed it until `claimed_at + lease_seconds`, and a later delivery for
-- the same submission may claim it after that. The image that ships with this
-- migration sets both on every `running` row it claims. A release back to
-- `pending` and a resubmission clear them.
--
-- `final_delivery` is whether the delivery that claimed the row is the task's
-- last (`X-CloudTasks-TaskRetryCount` at the queue's `max_attempts - 1`). The
-- claim sets it, and a release and a resubmission clear it. A `running` row
-- whose lease has ended is failed by the next poll when it is TRUE or NULL. A
-- FALSE row has a later delivery due, so it waits for that delivery until
-- the operation deadline has passed since `COALESCE(dispatched_at,
-- submitted_at)`. NULL means the row records no delivery: it is unclaimed, or
-- the image before this migration claimed it.
--
-- `dispatched_at` (from `0014_operation_dispatch.sql`) is the time a Cloud
-- Tasks create for the current submission returned, or reported the task
-- already exists, and stays NULL until then. A `pending` row with a NULL
-- `dispatched_at` past the enqueue recovery window had its create lost, and
-- the next poll creates the task again under the same name.
--
-- The migration only adds columns and maps rows, so the image before it keeps
-- working against the migrated table. The new image neither reads nor writes
-- `lease_until`, `dispatch_count`, `dispatch_confirmed` or
-- `execution_coverage_stamp`; its inserts leave them NULL or at their
-- `DEFAULT 0`. The deploy runs this migration and then moves the worker and
-- the API, so for those minutes both images serve operations:
--
-- * An old claim tests `lease_until`, which a new claim leaves NULL, so an old
--   delivery can take a row a new attempt holds. The old claim increments
--   `attempt_count`, so the new attempt's fenced writes stop, and the row
--   completes once.
-- * A row an old delivery claimed from `pending`, before or after this
--   migration, runs with a NULL `claimed_at`, and the old worker renews its
--   lease by updating `updated_at`. A row it took from a new attempt keeps
--   that attempt's `claimed_at`, `lease_seconds` and `final_delivery`, so
--   the new image reads the old worker by that static lease. The new image
--   counts a row with a NULL `claimed_at` live and leaves it to the old
--   attempt until `updated_at` is `LEASE_SECONDS` old, then treats its lease
--   as ended. Its `final_delivery` is NULL, so a dead old attempt's row is
--   reclaimed by the next delivery or failed by the next poll, whichever
--   comes first.
--
-- Rows are mapped as follows:
--
-- * A `pending` row whose last dispatch was never confirmed takes a NULL
--   `dispatched_at`, so the next poll recovers its enqueue. A confirmed one
--   keeps its `dispatched_at`: its task is on the queue.
-- * A `running` row keeps NULL lease columns and a NULL `final_delivery`,
--   since its old worker may still be renewing it through `updated_at`.
-- * `done` and `failed` rows keep NULL lease columns.
--
-- Operation ids hash the subject, member dcids, catalogue build and
-- rendering key, without the coverage stamp. Existing rows keep the ids they
-- were created under; a new submission of the same request computes the new
-- id and starts a new row.
ALTER TABLE datasets.operation
    ADD COLUMN claimed_at timestamptz,
    ADD COLUMN lease_seconds int,
    ADD COLUMN final_delivery boolean;

UPDATE datasets.operation
    SET dispatched_at = NULL
    WHERE state = 'pending' AND dispatch_confirmed < dispatch_count;
