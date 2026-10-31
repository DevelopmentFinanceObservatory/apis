-- One prepared export of a frame (`datasets_api.export_store.LiveExportStore`), keyed by its
-- `export_id`: `exp_` and the first 32 hex digits of the SHA-256 of the RFC 8785 form of
-- `{spec_id, content_version, format, bundle}` (`datasets_api.exports.export_id`). A row is
-- written once, so a repeat prepare reads it and renders nothing.
--
-- `artifact_digest`, `bucket` and `object_name` name the export file's pinned copy, which
-- `GET /api/datasets/exports/{export_id}` signs without another read. The pinned tier never
-- expires its objects. `body` holds the facts the prepare answered, with the sources and their
-- licences as they stood at prepare time.
--
-- `(subject_id, content_version)` FK-references `datasets.version_record`, so an export only
-- ever names a frame this store holds. 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers
-- this table for the runtime role, and 0012_worker_role.sql makes the worker role a member of
-- it, so this migration needs no GRANT statements of its own.
CREATE TABLE IF NOT EXISTS datasets.export (
    export_id text PRIMARY KEY,
    subject_id text NOT NULL,
    content_version text NOT NULL,
    artifact_digest text NOT NULL,
    bucket text NOT NULL,
    object_name text NOT NULL,
    body jsonb NOT NULL,
    recorded_at timestamptz NOT NULL,
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
