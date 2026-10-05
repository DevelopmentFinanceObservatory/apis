-- Version records: the immutable half of a dataset manifest, written once
-- at first materialisation of a (subject_id, content_version) pair.
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers every table
-- created later by the same migrator role, so this migration needs no
-- GRANT statements of its own.

-- One row per (subject_id, content_version), written once and never
-- overwritten: `subject_id` is a spec_id or a collection_id depending on
-- `subject_kind`, and `immutable_body` holds the manifest bytes minus its
-- own `artifacts` array (artifacts carry expiring signed URLs, so they
-- live in `artifact_inventory`/`artifact_location` instead, joined through
-- `version_artifact` below).
CREATE TABLE datasets.version_record (
    subject_id text NOT NULL,
    subject_kind text NOT NULL CHECK (subject_kind IN ('spec', 'collection')),
    content_version text NOT NULL,
    schema_version int NOT NULL,
    immutable_body jsonb NOT NULL,
    trace_id text,
    recorded_at timestamptz NOT NULL,
    PRIMARY KEY (subject_id, content_version)
);

-- A collection version's own member versions, in position order:
-- reconstructs `VersionRecord.member_versions` without carrying that array
-- on `version_record` itself. Empty for a spec's own version record.
CREATE TABLE datasets.version_member (
    subject_id text NOT NULL,
    content_version text NOT NULL,
    position int NOT NULL,
    member_spec_id text NOT NULL,
    member_content_version text NOT NULL,
    PRIMARY KEY (subject_id, content_version, position),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);

-- The version-to-artifact join: every artifact a (subject_id,
-- content_version) pair actually rendered, so a later retrieval can
-- enumerate that version's servable renderings by joining this against
-- `artifact_inventory`/`artifact_location`. Primary-keyed on
-- `rendering_key`, a canonical digest of the rendering options that
-- produced a given artifact, because storage is byte-deduplicated: two
-- distinct rendering option-sets for the same version can land on the
-- same `artifact_digest` when they happen to produce identical bytes.
-- Keying on digest alone would collapse those two genuinely distinct
-- renderings into one row and silently drop one of them from a later
-- manifest read. `artifact_digest` stays an ordinary, non-PK column
-- (still FK-checked against `artifact_inventory`): pinning one rendering
-- pins the underlying bytes for every other rendering that shares them,
-- since the pin door's own `artifact_reference` table (migration
-- `0007_artifact_references.sql`) is keyed on digest, not on this
-- table's own PK.
--
-- `rendering`/`media_type` carry the two `ArtifactEntryModel` fields with
-- no other durable home: `artifact_inventory`/`artifact_location`
-- (migration 0003) carry only bytes and location; what produced them
-- lives here instead. Snapshot retrieval reads these two columns to
-- rebuild a stored artifact's manifest entry without re-deriving it --
-- `bytes`/`created_at` come from `artifact_inventory` itself, and
-- `url`/`url_expires_at` are always signed fresh rather than stored.
CREATE TABLE datasets.version_artifact (
    subject_id text NOT NULL,
    content_version text NOT NULL,
    artifact_digest text NOT NULL REFERENCES datasets.artifact_inventory (artifact_digest),
    rendering_key text NOT NULL,
    rendering jsonb NOT NULL,
    media_type text NOT NULL,
    PRIMARY KEY (subject_id, content_version, rendering_key),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
