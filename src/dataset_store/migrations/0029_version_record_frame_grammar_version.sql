-- The frame grammar version that minted a version's `content_version`. A
-- token minted under one grammar version never moves, and a reader tells
-- tokens minted under different preimage families apart by this column.
--
-- `LiveVersionRecordWriter.record_version` writes the column for every new
-- row. `0031_version_record_facts_backfill.sql` fills it on existing rows from
-- the stored trace, and a body with no trace, as a derived `ops_`/`ocv_`
-- frame's, keeps NULL. No stored `content_version` is rewritten.
--
-- The ALTER holds an ACCESS EXCLUSIVE lock on `version_record` for an instant,
-- since adding a nullable column rewrites no row. While it waits for that lock
-- behind a long read, every later read of the table queues behind it, so
-- `lock_timeout` fails it after five seconds instead. Re-running this
-- migration changes nothing: the column is added only when absent.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS frame_grammar_version text;
