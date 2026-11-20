-- The raw heads a file linked under a derived version was made from.
--
-- `version_artifact_head` holds one row for each raw version whose stored frame rendered a
-- link's file and made the version it is linked under. A version record keeps the base it was
-- first recorded from (`version_record.base_content_version`), and a later render from another
-- head that makes the same version links its file there, so only these rows say which heads a
-- rendering made the version from. A link made from several heads holds a row for each, so the
-- version is found from any of them. `datasets_api.version_store._insert_artifact` inserts the
-- row for every link a derived version's write names, after the link upsert and in the same
-- transaction, and a row already there stays. A link under a raw version, and every link written
-- before this table or by the previous image, has no row. The frame refresh reads the rows by
-- head through `version_artifact_head_by_head`
-- (`datasets_api.version_store.LiveVersionLister.list_head_versions`) and counts a link with no
-- row only by the rule `datasets_api.doors.refresh` states, which rests on no link ever being
-- deleted. No code path deletes a `version_artifact` row, and the foreign key refuses to delete
-- a link that holds a row here. The refresh also reads a raw version's rows by the primary key
-- to tell whether a file under it was made from another head.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers this table for the runtime role, and
-- 0012_worker_role.sql makes the worker role a member of it, so this migration needs no GRANT
-- statements of its own. The foreign key
-- takes SHARE ROW EXCLUSIVE on `version_artifact` for the moment the empty table is created;
-- `lock_timeout` fails it after five seconds rather than queueing every later write behind it.
-- The previous image never reads or writes the table. Re-running this migration changes
-- nothing.

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS datasets.version_artifact_head (
    subject_id text NOT NULL,
    content_version text NOT NULL,
    rendering_key text NOT NULL,
    head_content_version text NOT NULL,
    PRIMARY KEY (subject_id, content_version, rendering_key, head_content_version),
    FOREIGN KEY (subject_id, content_version, rendering_key)
        REFERENCES datasets.version_artifact (subject_id, content_version, rendering_key)
);

CREATE INDEX IF NOT EXISTS version_artifact_head_by_head
    ON datasets.version_artifact_head (subject_id, head_content_version);
