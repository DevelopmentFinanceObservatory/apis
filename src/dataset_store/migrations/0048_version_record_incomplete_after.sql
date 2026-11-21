-- The rows of a version past the last complete year of their source.
--
-- `incomplete_after` holds one entry per row `provenance` cell, source and last complete year,
-- naming the `stat_vars` whose rows reach the period the source's import holds only part of:
-- for a dataset version as its mint judged them against the origin catalogue
-- (`datasets_api.manifest.incomplete_after`), and for a published derived frame as the publish
-- door derived them from the versions it was computed from
-- (`datasets_api.lineage_incomplete.derived_incomplete_after`). `[]` means judged and nothing
-- partial. NULL means unknown: a version the mint or the publish could not judge, a row
-- recorded before this column, and a row written by the previous image, which names its
-- columns and leaves this one out. A later mint of the same dataset version replaces it.
--
-- Adding a nullable column rewrites no row. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS incomplete_after jsonb;
