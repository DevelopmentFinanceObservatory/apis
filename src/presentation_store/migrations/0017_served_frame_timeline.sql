-- Makes `served_frame` a timeline of what each share slot showed, including nothing
-- (`presentation_store.shares`). A slot records an entry whenever what it shows differs from its
-- newest entry, so a slot that goes back to a frame it served before records it again, and a
-- slot a replace leaves empty records an entry with no card: `component_id`,
-- `component_revision`, `frame_spec_id`, `frame_content_version` and `rendering` all NULL. The
-- natural key admitted each frame once and no empty entry, so it gives way to an identity `id`,
-- which also orders entries recorded at one instant. The index finds a slot's newest entry. The
-- table rewrite holds an ACCESS EXCLUSIVE lock on `served_frame` only, which slot reads and share
-- writes wait on for as long as the table takes to rewrite. 0001_bootstrap.sql's ALTER DEFAULT
-- PRIVILEGES lets the runtime role use the identity sequence.

ALTER TABLE presentation.served_frame DROP CONSTRAINT served_frame_pkey;

ALTER TABLE presentation.served_frame
    ADD COLUMN id bigint GENERATED ALWAYS AS IDENTITY,
    ALTER COLUMN component_id DROP NOT NULL,
    ALTER COLUMN component_revision DROP NOT NULL,
    ALTER COLUMN frame_spec_id DROP NOT NULL,
    ALTER COLUMN frame_content_version DROP NOT NULL,
    ADD CONSTRAINT served_frame_pkey PRIMARY KEY (id),
    ADD CONSTRAINT served_frame_empty_check CHECK (
        (component_id IS NULL) = (component_revision IS NULL)
        AND (component_id IS NULL) = (frame_spec_id IS NULL)
        AND (component_id IS NULL) = (frame_content_version IS NULL)
        AND (component_id IS NOT NULL OR rendering IS NULL)
    );

CREATE INDEX served_frame_slot_newest_idx
    ON presentation.served_frame (share_id, slot, first_served_at DESC, id DESC);
