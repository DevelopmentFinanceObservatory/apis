-- Components and their revision log. `component` carries no session
-- column: the lease is the single session authority (`draft_lease`,
-- migration 0004), never duplicated here. `composition_id` is set once, at
-- create time, when the create is a create-in-composition -- a component
-- created standalone keeps it NULL forever; nothing in this package's own
-- write path ever moves a component between compositions.
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers both
-- tables below, so this migration needs no GRANT statements of its own.

CREATE TABLE presentation.component (
    component_id text PRIMARY KEY,
    composition_id text REFERENCES presentation.composition (composition_id),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- One row per (component_id, revision) -- store.py's `append()` never
-- updates or deletes a row here, only inserts the next revision; the PK
-- doubles as the uniqueness backstop a same-revision race is caught
-- against (store_errors.is_unique_violation -> EditConflict). Both
-- `stored_intent` (what the caller submitted) and `resolved_payload` (what
-- the pipeline minted from it) are kept, per revision, so `read_revision`
-- can serve either without re-resolving.
CREATE TABLE presentation.component_revision (
    component_id text NOT NULL REFERENCES presentation.component (component_id),
    revision int NOT NULL,
    stored_intent jsonb NOT NULL,
    resolved_payload jsonb NOT NULL,
    catalog_version text NOT NULL,
    schema_version int NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (component_id, revision)
);
