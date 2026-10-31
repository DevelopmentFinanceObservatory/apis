-- The newest-data check of each live share slot (`presentation_store.shares`). A live slot's
-- data read asks the Datasets refresh or the analysis runner for the newest frame of its card at
-- most once per window, and this row holds the answer the reads in between serve: the frame,
-- whether it was `current`, `stale` (the check failed or the card could not draw the newer
-- frame) or `pending` (newer data is building), and when the check ran. The row is keyed by the
-- card revision as well as the slot, so a replacement that puts another card in the slot starts
-- with no check. A snapshot conversion writes the frame each slot serves into the share's next
-- revision, after which no read consults these rows. The table is new, so no running revision
-- waits on a lock it takes, and 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers it.
CREATE TABLE presentation.share_slot_check (
    share_id text NOT NULL REFERENCES presentation.share (id),
    slot int NOT NULL CHECK (slot >= 1),
    component_id text NOT NULL,
    component_revision int NOT NULL,
    frame_spec_id text NOT NULL,
    frame_content_version text NOT NULL,
    freshness text NOT NULL CHECK (freshness IN ('current', 'stale', 'pending')),
    checked_at timestamptz NOT NULL,
    PRIMARY KEY (share_id, slot, component_id, component_revision)
);
