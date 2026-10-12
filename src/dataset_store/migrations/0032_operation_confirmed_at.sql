-- The instant a completed async operation's run confirmed the subject head,
-- on the worker's job record (`0011_operation.sql`).
--
-- `LiveOperationStore.complete` writes it beside `subject_id` and
-- `content_version`: the `trace.confirmed_at` of the manifest the run served,
-- the head's `confirmed_at` (`0009_subject_head.sql`) as the run left it. The
-- poll route stamps it on a `done` operation's manifest with
-- `served_from: materialise`, the trace a synchronous mint that materialised
-- carries. The column stays fixed once written, where `updated_at` moves when
-- a late dispatch record lands after the run completed.
--
-- NULL on every row that is not `done`. The UPDATE fills each `done` row
-- written before this migration with its `updated_at`, the completion instant
-- unless a later dispatch record moved it.
--
-- Apply this migration before deploying an API or worker image that reads
-- the column. Every `LiveOperationStore` statement in such an image selects
-- `confirmed_at`, so against a table without it each one fails with an
-- undefined-column error, surfaced as `DatasetsStoreUnavailable`. An image
-- older than the column runs unchanged against the migrated table, and its
-- completions leave the column NULL, so their polls carry no `confirmed_at`.
ALTER TABLE datasets.operation
    ADD COLUMN confirmed_at timestamptz;

UPDATE datasets.operation
SET confirmed_at = updated_at
WHERE state = 'done';
