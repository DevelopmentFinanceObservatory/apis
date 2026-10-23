-- The index the versions door reads one subject's page through.
--
-- `version_record_recency` serves `LiveVersionLister.list_servable_versions` when the path id
-- serves under one subject id: the page is the subject's rows newest first after the cursor,
-- `(recorded_at, content_version) < (cursor)`, so the scan starts at the cursor and stops at
-- the page limit, whatever the subject's history holds.
--
-- CREATE INDEX takes SHARE on `version_record`, which waits for writers and blocks inserts
-- while the index builds; `lock_timeout` fails it after five seconds rather than queueing
-- every later write behind it. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

CREATE INDEX IF NOT EXISTS version_record_recency
    ON datasets.version_record (subject_id, recorded_at DESC, content_version DESC);
