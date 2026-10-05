-- `reconciliation.sweep_tier` filters datasets.artifact_location on tier
-- alone; tier is the trailing column of that table's own (artifact_digest,
-- tier) primary key, so no index on the leading column covers a tier-only
-- filter and every sweep scans the full table. Postgres has no grantable
-- privilege on an index itself (access rides on the table's own privileges,
-- already covered by 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES), so
-- this migration needs no GRANT statement of its own.

CREATE INDEX artifact_location_tier_idx ON datasets.artifact_location (tier);
