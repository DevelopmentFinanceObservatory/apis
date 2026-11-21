-- The rows of a published derived frame past the last complete year of their source.
--
-- `incomplete_after` holds the entries the publish door derives from the versions a frame was
-- computed from (`datasets_api.lineage_incomplete.derived_incomplete_after`), each naming a
-- `provenance` cell, the `stat_vars` it covers, the `source` whose import holds only part of
-- the year after `last_complete_year`, and that year, and `[]` when none of the frame's rows
-- is past one. A dataset version's row leaves it NULL, since its `coverage` records its own.
-- A derived frame's row recorded before this column, or written by the previous image until
-- the services move, holds NULL, which reads as unknown.
--
-- Adding a nullable column rewrites no row. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS incomplete_after jsonb;
