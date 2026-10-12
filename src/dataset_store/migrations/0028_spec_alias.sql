-- One row per pair of a posted spec id and a served spec id it has resolved
-- to. A posted identity's canonical form under the code index is its served
-- identity, which keys heads, versions, operations and references. The served
-- id of one posted id changes when a code index export changes the canonical
-- form, so a posted id keeps one row per served id it has resolved to.
--
-- `LiveSpecRegistry.register_served_definition` inserts the row in the
-- transaction that registers the served definition. A version-addressed read
-- under the posted id reads the version under each served id bound here,
-- newest `bound_at` first (`LiveSpecReader.read_served_spec_ids`).
--
-- Both ids are registered before the row is written, so both reference
-- `spec_registry`.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers the new table, so this
-- migration needs no GRANT statement of its own.

CREATE TABLE datasets.spec_alias (
    posted_spec_id text NOT NULL REFERENCES datasets.spec_registry (spec_id),
    served_spec_id text NOT NULL REFERENCES datasets.spec_registry (spec_id),
    bound_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (posted_spec_id, served_spec_id)
);
