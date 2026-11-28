-- Indexes each slot's entries by the card and frame they show, newest first, so a slot read
-- naming a frame finds the slot's newest entry of its card on that frame, or finds none, by one
-- index probe (`presentation_store.shares.LiveShareStore.slot_served`).
-- `served_frame_slot_newest_idx` orders only by time within a slot, so a frame the slot never
-- showed would read every entry of the slot's history. That index still finds the slot's newest
-- entry. Building the index holds a SHARE lock on `served_frame`, which slot reads that record a
-- new entry wait on. While the build waits for that lock behind a long transaction, every later
-- write to the table queues behind it, so `lock_timeout` fails it after five seconds instead.
-- The failed migration rolls back and is not recorded as applied, so the next migration run
-- applies it again.

SET LOCAL lock_timeout = '5s';

CREATE INDEX served_frame_slot_frame_idx
    ON presentation.served_frame (
        share_id,
        slot,
        component_id,
        component_revision,
        frame_spec_id,
        frame_content_version,
        first_served_at DESC,
        id DESC
    );
