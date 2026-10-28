-- Shares: a public `/s/<id>` record of every kind, its append-only revision log, the cards each
-- revision shows in numbered slots, and the frames each slot served (`presentation_store.shares`).
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers these tables, so this migration
-- needs no GRANT statements of its own. Every table is new, so no running revision waits on a
-- lock it takes.

-- One row per share. `id` is 8 characters from the case-sensitive `[A-Za-z0-9]` alphabet,
-- minted by `presentation_store.ids.mint_share_id` or kept from an imported website share, so
-- both live in one id space and the primary key is the collision check. `kind` is what the
-- share publishes, and `mode` whether its slots follow new data (`live`) or keep the frames
-- they were made from (`snapshot`). `write_token_sha256` is the unsalted SHA-256 of the `shw_`
-- token the create returned once; NULL is a share imported from a guest, which refuses every
-- token. `owner` is left NULL until callers can hold accounts. `payload` is website-owned JSON
-- stored as given and never read here, which carries what the website renders for the search,
-- conversation and canvas kinds beyond the slots. `deleted_at` marks a share taken down; its
-- rows stay, and every read treats it as absent.
CREATE TABLE presentation.share (
    id text PRIMARY KEY CHECK (id ~ '^[A-Za-z0-9]{8}$'),
    kind text NOT NULL CHECK (kind IN ('card', 'canvas', 'search', 'conversation')),
    mode text NOT NULL CHECK (mode IN ('live', 'snapshot')),
    title text,
    write_token_sha256 bytea,
    owner text,
    payload jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);

-- One row per (share_id, revision), append-only: a create writes revision 1 and each replace
-- the next one. Nothing updates or deletes a row here.
CREATE TABLE presentation.share_revision (
    share_id text NOT NULL REFERENCES presentation.share (id),
    revision int NOT NULL CHECK (revision >= 1),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (share_id, revision)
);

-- The slots of one share revision, numbered from 1. Each revision writes every slot the share
-- has ever had, so a slot's embed and data URLs survive a replacement with fewer cards: such a
-- slot holds no card (`component_id` and `component_revision` NULL). `frame_spec_id` and
-- `frame_content_version` name the frame the card revision draws, recorded at write; both are
-- NULL for a card that draws none and for an imported website slot whose frame the export did
-- not name.
CREATE TABLE presentation.share_slot (
    share_id text NOT NULL,
    revision int NOT NULL,
    slot int NOT NULL CHECK (slot >= 1),
    component_id text,
    component_revision int,
    frame_spec_id text,
    frame_content_version text,
    PRIMARY KEY (share_id, revision, slot),
    FOREIGN KEY (share_id, revision) REFERENCES presentation.share_revision (share_id, revision),
    CHECK ((component_id IS NULL) = (component_revision IS NULL)),
    CHECK ((frame_spec_id IS NULL) = (frame_content_version IS NULL)),
    CHECK (component_id IS NOT NULL OR frame_spec_id IS NULL)
);

-- Every frame a share's slot served, first recorded when a read served it. `rendering` is the
-- frame's rendering as `component_revision.frame_rendering` stores it, NULL when unrecorded.
CREATE TABLE presentation.served_frame (
    share_id text NOT NULL REFERENCES presentation.share (id),
    slot int NOT NULL CHECK (slot >= 1),
    component_id text NOT NULL,
    component_revision int NOT NULL,
    frame_spec_id text NOT NULL,
    frame_content_version text NOT NULL,
    rendering jsonb,
    first_served_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (
        share_id, slot, component_id, component_revision, frame_spec_id, frame_content_version
    )
);
