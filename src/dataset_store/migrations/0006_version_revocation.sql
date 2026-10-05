-- Takedown record: a revoked (subject_id, content_version) refuses both
-- snapshot retrieval and pinning with `snapshot_revoked`, permanently --
-- rows here are never deleted, keeping the revocation itself as audit
-- history. No route on this API writes this table; takedown is a
-- first-party operation on the site's own surfaces, enforced here only on
-- the read path. 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already
-- covers this table.

CREATE TABLE datasets.version_revocation (
    subject_id text NOT NULL,
    content_version text NOT NULL,
    reason text,
    trace_id text,
    revoked_at timestamptz NOT NULL,
    PRIMARY KEY (subject_id, content_version),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
