-- Lets a live share slot's newest-data check row hold no `checked_at` (`presentation_store.shares`).
-- The first read of a slot claims its check by inserting the row as `pending` on the card's own
-- frame before any check has run, so concurrent first reads run one check, and the reads that
-- lose the claim state no check time. Every completed check writes `checked_at`. Dropping NOT
-- NULL changes only the catalogue, so it takes no lock a running revision waits on.

ALTER TABLE presentation.share_slot_check ALTER COLUMN checked_at DROP NOT NULL;
