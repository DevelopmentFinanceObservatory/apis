-- The continuations of a published derived frame's series, and the index the publish door
-- reads its input versions through.
--
-- `continuations` holds the entries the publish door derives from the versions a frame was
-- computed from (`datasets_api.lineage_continuations`), each naming the continuing method and
-- the method and source of the series it continues
-- (`materialise.continued_series.ContinuedSeries.as_entry`), and `[]` when nothing continues
-- the frame's series. A dataset version's row leaves it NULL, since its trace records its
-- own. A derived frame's row recorded before this column, or written by the previous image
-- until the services move, holds NULL until the `backfill-descriptions` task fills it.
--
-- `version_record_content_version` serves the publish door's read of every version at the
-- content versions a frame names (`LiveSnapshotReader.read_versions_of`), which names no
-- subject.
--
-- Adding a nullable column rewrites no row. CREATE INDEX takes SHARE on `version_record`,
-- which waits for writers and blocks inserts while the index builds; `lock_timeout` fails it
-- after five seconds rather than queueing every later write behind it. Re-running this
-- migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS continuations jsonb;

CREATE INDEX IF NOT EXISTS version_record_content_version
    ON datasets.version_record (content_version);
