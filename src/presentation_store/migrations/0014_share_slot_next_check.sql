-- Stores when each live share slot's newest-data check is next due (`presentation_store.shares`).
-- The check's answer sets it: a `pending` check names the retry time the refresh or the run gave,
-- and a `current` or `stale` check waits the freshness window, 15 minutes. A read re-checks a
-- slot once `next_check_at` has passed, in place of comparing `checked_at` to the window.
--
-- The UPDATE gives rows written before this migration the time they were due under the window,
-- `checked_at` plus 15 minutes. The default is the same window from the insert, for the rows
-- that a revision of the service without this column inserts while a deploy rolls out;
-- `presentation_store.shares` writes the column on every insert.

ALTER TABLE presentation.share_slot_check ADD COLUMN next_check_at timestamptz;

UPDATE presentation.share_slot_check SET next_check_at = checked_at + interval '15 minutes';

ALTER TABLE presentation.share_slot_check
    ALTER COLUMN next_check_at SET DEFAULT now() + interval '15 minutes',
    ALTER COLUMN next_check_at SET NOT NULL;
