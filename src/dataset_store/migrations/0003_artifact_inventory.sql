-- Artifact inventory: one row per stored artifact_digest, plus one location
-- row per (artifact_digest, tier) the digest has actually been written to.
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers every table
-- (and sequence) created later by the same migrator role, so this migration
-- needs no GRANT statements of its own.

CREATE TABLE datasets.artifact_inventory (
    artifact_digest text PRIMARY KEY,
    size_bytes bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE datasets.artifact_location (
    artifact_digest text REFERENCES datasets.artifact_inventory,
    tier text NOT NULL CHECK (tier IN ('working', 'pinned')),
    bucket text NOT NULL,
    object_name text NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (artifact_digest, tier),
    -- Content-addressed layout: the object's name is its own digest.
    CHECK (object_name = artifact_digest)
);
