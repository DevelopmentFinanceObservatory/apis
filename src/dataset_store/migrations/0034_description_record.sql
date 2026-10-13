-- One description record per version: the facts and evidence a description is composed from
-- (`description.facts.DescriptionRecord`), fixed when the version is made.
--
-- `subject_id` holds the record's `id`, as `frame_recipe` and `version_record` name the column.
-- `record` holds the whole record as JSON. `schema_version` and `basis` repeat the record's
-- members of those names, so a reader can select by them without decoding the record. `basis`
-- names what wrote the record: a mint, a publish, or the backfill of a version made before
-- records existed. `datasets_api.description_store.insert_description` writes the row in the
-- transaction that writes the version row, with ON CONFLICT DO NOTHING, so the first writer
-- keeps the row. `fill_late_fields` is the only later write: it sets a member that is null in
-- `record` and notes the arrival in the record's `late_fields`.
--
-- The version row's columns and `immutable_body` are untouched, so the revision that still
-- serves during a deploy never meets a member it does not know; it does not read this table.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers this table, so this migration
-- needs no GRANT statement of its own.
--
-- The foreign key takes SHARE ROW EXCLUSIVE on `version_record` until this migration's
-- transaction commits. While the lock waits behind a long write, every later write of the
-- table queues behind it, so `lock_timeout` fails it after five seconds instead.

SET LOCAL lock_timeout = '5s';

CREATE TABLE datasets.description_record (
    subject_id text NOT NULL,
    content_version text NOT NULL,
    schema_version integer NOT NULL,
    basis text NOT NULL CHECK (basis IN ('mint', 'publish', 'backfill')),
    record jsonb NOT NULL,
    recorded_at timestamptz NOT NULL,
    PRIMARY KEY (subject_id, content_version),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
