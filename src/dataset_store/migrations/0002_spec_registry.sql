-- Spec registry: the durable store of canonical spec/collection
-- definitions the write transaction reads and writes. No grants here --
-- 0001's ALTER DEFAULT PRIVILEGES already covers tables and sequences
-- created later by the migrator role.

-- Identity rows, written once, never overwritten: the canonical identity
-- projection bytes and their digest for one spec_id. `identity_version`
-- records which identity scheme minted this row (spec1 today).
CREATE TABLE datasets.spec_registry (
    spec_id text PRIMARY KEY,
    identity_version text NOT NULL,
    identity_bytes bytea NOT NULL,
    identity_digest text NOT NULL,
    trace_id text,
    registered_at timestamptz NOT NULL DEFAULT now()
);

-- Canonical definitions, append-only, keyed by (spec_id, schema_version):
-- `definition` holds the identity projection derived from the matching
-- spec_registry row's own identity_bytes, not the caller's wire form --
-- hashing its `identity` member reproduces this row's spec_id. Nothing
-- ever overwrites identity bytes because a newer schema exists, and a
-- spec_id may carry more than one schema_version's definition over time.
CREATE TABLE datasets.spec_definition (
    spec_id text NOT NULL REFERENCES datasets.spec_registry (spec_id),
    schema_version int NOT NULL,
    definition jsonb NOT NULL,
    written_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (spec_id, schema_version)
);

-- Collection rows, written once: a collection_id names its sorted,
-- de-duplicated member spec_ids -- carried in collection_member below,
-- not duplicated onto this row.
CREATE TABLE datasets.collection_registry (
    collection_id text PRIMARY KEY,
    trace_id text,
    registered_at timestamptz NOT NULL DEFAULT now()
);

-- One row per collection member, in minted sort order: `position`
-- reconstructs the col1 projection's member array from these rows without
-- carrying the array on collection_registry itself.
CREATE TABLE datasets.collection_member (
    collection_id text NOT NULL REFERENCES datasets.collection_registry (collection_id),
    position int NOT NULL,
    spec_id text NOT NULL REFERENCES datasets.spec_registry (spec_id),
    PRIMARY KEY (collection_id, position)
);

-- Both a spec and a collection collision land here: the write transaction
-- fails closed, quarantines both the held and incoming identity bytes, and
-- alerts (Cloud Logging alert policy keyed on the
-- identity_collision_quarantined log marker is an operator act, not built
-- here). `held_bytes` is nullable only for a hypothetical quarantine entry
-- with nothing on record yet; every collision this API's write transaction
-- raises supplies both.
CREATE TABLE datasets.identity_quarantine (
    id bigserial PRIMARY KEY,
    id_kind text NOT NULL CHECK (id_kind IN ('spec', 'collection')),
    subject_id text NOT NULL,
    held_bytes bytea,
    incoming_bytes bytea NOT NULL,
    trace_id text,
    quarantined_at timestamptz NOT NULL DEFAULT now()
);
