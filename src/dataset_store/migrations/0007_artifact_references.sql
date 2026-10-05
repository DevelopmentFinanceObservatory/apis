-- References backing pin/promotion garbage collection: one row per
-- (principal, artifact_digest, subject_id, content_version, reference_kind).
-- A solo member's own reference and its collection membership record as
-- distinct rows here; garbage collection asks only whether any live
-- (removed_at IS NULL) row references a digest, so bytes are deleted once
-- nothing at all references them, regardless of which reference_kind.
-- Removing a reference is a soft delete (`removed_at`), never a row
-- delete, so reference history survives for audit -- the same discipline
-- `identity_quarantine` and `version_revocation` already use. No route on
-- this API sets `removed_at`; unpinning and share removal are first-party
-- operations on the site's own surfaces. 0001_bootstrap.sql's ALTER
-- DEFAULT PRIVILEGES already covers this table.

CREATE TABLE datasets.artifact_reference (
    principal text NOT NULL,
    artifact_digest text NOT NULL REFERENCES datasets.artifact_inventory (artifact_digest),
    subject_id text NOT NULL,
    content_version text NOT NULL,
    reference_kind text NOT NULL,
    created_at timestamptz NOT NULL,
    removed_at timestamptz,
    PRIMARY KEY (principal, artifact_digest, subject_id, content_version, reference_kind),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
