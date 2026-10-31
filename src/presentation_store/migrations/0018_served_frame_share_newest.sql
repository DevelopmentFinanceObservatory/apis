-- Indexes a share's whole timeline newest first, so the history route reads one page of
-- `served_frame` by keyset on (first_served_at, id) (`presentation_store.shares.served_frames`).
-- `served_frame_slot_newest_idx` leads with the slot and orders only within one slot, so a
-- page across slots would read and sort every entry of the share. Building the index holds a
-- SHARE lock on `served_frame`, which slot reads that record a new entry wait on.

CREATE INDEX served_frame_share_newest_idx
    ON presentation.served_frame (share_id, first_served_at DESC, id DESC);
