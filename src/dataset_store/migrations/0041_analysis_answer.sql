-- An analysis run's 200 body under its request key (`datasets_api.analysis.request_key`): the
-- sha256 of the working tree as sent, each resolved root's record, `latest`, the execution
-- identity and the answer format. A request whose key holds a row answers the stored body while
-- the frame file the body names is still served, so a retry after a timeout reads no frame
-- (`datasets_api.recipe_store.LiveRecipeStore.write_answer`, `read_answer`).
--
-- `(subject_id, content_version)` is the frame pair the body names, and FK-references
-- `datasets.version_record`, so an answer only ever names a frame this store holds. A write
-- under a held key replaces the row, since a recomputed answer stands in for one whose frame
-- file is no longer served.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers this table for the runtime role, and
-- 0012_worker_role.sql makes the worker role a member of it, so this migration needs no GRANT
-- statements of its own.
CREATE TABLE IF NOT EXISTS datasets.analysis_answer (
    request_key text PRIMARY KEY,
    subject_id text NOT NULL,
    content_version text NOT NULL,
    body jsonb NOT NULL,
    recorded_at timestamptz NOT NULL,
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
