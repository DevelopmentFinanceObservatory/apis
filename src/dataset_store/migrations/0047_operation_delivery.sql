-- What the operations poll needs to log one delivery line per completion
-- (`datasets_api.delivery`, `datasets_api.doors.meta`).
--
-- `admission` is `background` for a mint sent with `X-DFO-Mint-Admission: background`, else
-- `foreground`, as the submitting request carried it (`OperationStore.create_or_get`). A resubmit
-- that resets the row records its own. The delivery line a poll logs names this value, since the
-- poll's own header says nothing about the mint.
--
-- `delivered_at` is the instant the first poll found the row `done` and logged its line
-- (`OperationStore.mark_delivered`). It is NULL until then, and a resubmit's reset clears it, so
-- each completion logs once. A row `done` before this migration reads NULL and logs on its first
-- poll after it. An image older than the columns runs unchanged against the migrated table: its
-- inserts take the `foreground` default and its resets leave both columns as they were.
ALTER TABLE datasets.operation
    ADD COLUMN admission text NOT NULL DEFAULT 'foreground'
        CHECK (admission IN ('foreground', 'background')),
    ADD COLUMN delivered_at timestamptz;
