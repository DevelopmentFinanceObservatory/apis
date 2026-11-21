-- How complete the rows of a version are, judged per source.
--
-- `incomplete_after` holds one entry per row `provenance` cell, source and state, naming its
-- `stat_vars` (`datasets_api.manifest_models.IncompleteAfter`): `partial` from the period the
-- source's import holds only part of, `unknown` where no catalogue documented the source, and
-- on a computed version `complete`. A dataset version's comes from its mint, judged against
-- the origin catalogue (`datasets_api.manifest.incomplete_after`), a published derived frame's
-- from the versions it was computed from
-- (`datasets_api.lineage_incomplete.derived_incomplete_after`). `[]` means judged and nothing
-- partial. NULL means unknown as a whole: a version the mint or the publish could not judge, a
-- row recorded before this column, and a row written by the previous image, which names its
-- columns and leaves this one out. A later write of the same version replaces it.
--
-- Adding a nullable column rewrites no row. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS incomplete_after jsonb;
