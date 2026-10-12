-- A working-tier location renews under a new object name. The runtime
-- identities can only create objects in the working bucket, so bytes whose
-- location has passed retention upload again as `<digest>#r<renewal>`
-- (`artifact_store.store._HashingSink`), and the location row moves to that
-- name. The object name is therefore free, and the digest stays the
-- identity: `artifact_store.store.LiveArtifactStore.get` hashes the bytes it
-- reads against the digest.
--
-- `artifact_location_check` is the name Postgres gave 0003's unnamed
-- table-level `CHECK (object_name = artifact_digest)`.
--
-- `renewal` counts the renewals that produced `object_name`; 0 is the bare
-- digest. 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers the table, so
-- this migration needs no GRANT statement of its own.

ALTER TABLE datasets.artifact_location DROP CONSTRAINT artifact_location_check;

ALTER TABLE datasets.artifact_location ADD COLUMN renewal int NOT NULL DEFAULT 0;
