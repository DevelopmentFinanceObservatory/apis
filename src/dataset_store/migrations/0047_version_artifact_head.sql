-- The raw head a file linked under a derived version was last made from.
--
-- `version_artifact.head_content_version` names the raw version whose stored frame last
-- rendered the link's file and made the version it is linked under. A version record keeps the
-- base it was first recorded from (`version_record.base_content_version`), and a later render
-- from another head that makes the same version links its file there, so only the link says
-- which head a rendering made the version from. `datasets_api.version_store._insert_artifact`
-- sets it on every link a derived version's write names, after the link upsert and in the same
-- transaction; the last head to write the link wins. A link under a raw version, and a link a
-- write names no head for, keeps NULL, as does every link written before this column or by the
-- previous image. The frame refresh reads links by head through `version_artifact_head`
-- (`datasets_api.version_store.LiveVersionLister.list_head_versions`) and counts a NULL link
-- only by the rule `datasets_api.doors.refresh` states.
--
-- Adding a nullable column with no default rewrites no row, and the previous image keeps
-- writing links without it. ALTER TABLE takes ACCESS EXCLUSIVE on `version_artifact` and
-- CREATE INDEX takes SHARE, each waiting for writers; `lock_timeout` fails either after five
-- seconds rather than queueing every later write behind it. Re-running this migration changes
-- nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_artifact
    ADD COLUMN IF NOT EXISTS head_content_version text;

CREATE INDEX IF NOT EXISTS version_artifact_head
    ON datasets.version_artifact (subject_id, head_content_version)
    WHERE head_content_version IS NOT NULL;
