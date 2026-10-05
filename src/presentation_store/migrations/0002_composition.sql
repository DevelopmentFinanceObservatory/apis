-- Compositions and their append-only revision log. Created before
-- 0003_component.sql's `component` table so that table's own
-- `composition_id` column can carry a foreign key against this one.
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers every table
-- created later by the same migrator role, so this migration needs no
-- GRANT statements of its own.

CREATE TABLE presentation.composition (
    composition_id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- One row per (composition_id, revision), append-only -- store.py's
-- `append()` never updates or deletes a row here, only inserts the next
-- revision. `blocks` is the ordered content of the composition at this
-- revision: a JSON array of `{"component_id", "revision"}` entries, one
-- per component the composition currently references, in first-touched
-- order; touching a component already present updates its entry's
-- `revision` in place rather than appending a duplicate. `actor` is
-- hardcoded `"model"` at every call site -- no caller identity exists yet
-- -- but stored as free text rather than a fixed value, since the schema
-- itself carries no such assumption.
CREATE TABLE presentation.composition_revision (
    composition_id text NOT NULL REFERENCES presentation.composition (composition_id),
    revision int NOT NULL,
    blocks jsonb NOT NULL,
    actor text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (composition_id, revision)
);
