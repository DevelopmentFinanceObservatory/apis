-- The facts a version stores, one column each, from which every read
-- assembles the wire manifest (`datasets_api.manifest.stored_manifest`).
--
-- A dataset version stores `prepared_at`, `definition`, `coverage`,
-- `citation`, `trace`, `warnings`, `series_count`, `series_coverage_stamp`,
-- `output_profile`, `series_conversions` and `default_scope_plural`. A
-- published derived frame stores `input_content_versions` and
-- `composed_stat_vars`. Each fact is the value of the `immutable_body` key of
-- the same name, which `0031_version_record_facts_backfill.sql` copies on the
-- rows this migration finds. `LiveVersionRecordWriter.record_version` writes
-- the facts and the body they encode (`datasets_api.manifest.version_body`).
--
-- `frame_sha256` is the hash of the version's frame
-- (`dfo_frame_grammar.frame_sha256`), the same for every encoding of the same
-- rows. `LiveVersionRecordWriter.record_version` writes it beside the facts,
-- and no body holds it: it is NULL on a row inserted with only
-- `immutable_body` and on the rows this migration finds.
--
-- The trigger `version_record_fill_facts` gives every row inserted from here
-- on its facts. The previous image inserts a row holding only
-- `immutable_body`, until the services move and from requests it still holds
-- afterwards. For a row arriving with neither `prepared_at` nor
-- `input_content_versions` the trigger sets each fact column and
-- `frame_grammar_version` from the body, with the expressions of 0031's
-- UPDATE: a JSON `null` member becomes SQL NULL and an absent member stays
-- NULL. A row arriving with facts is stored as written. The trigger and its
-- function exist only to read `immutable_body`, and the migration that drops
-- that column drops them.
--
-- The ALTER takes an ACCESS EXCLUSIVE lock on `version_record` and holds it
-- until this migration's transaction commits, an instant, since adding
-- nullable columns rewrites no row. CREATE TRIGGER takes SHARE ROW EXCLUSIVE,
-- which that lock already covers, so it waits on nothing. While the ALTER
-- waits behind a long read, every later read of the table queues behind it,
-- so `lock_timeout` fails it after five seconds instead. Re-running this migration changes nothing: the columns are
-- added only when absent, and the function and trigger are replaced by
-- themselves.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS prepared_at timestamptz,
    ADD COLUMN IF NOT EXISTS definition jsonb,
    ADD COLUMN IF NOT EXISTS coverage jsonb,
    ADD COLUMN IF NOT EXISTS citation jsonb,
    ADD COLUMN IF NOT EXISTS trace jsonb,
    ADD COLUMN IF NOT EXISTS warnings jsonb,
    ADD COLUMN IF NOT EXISTS series_count integer,
    ADD COLUMN IF NOT EXISTS series_coverage_stamp text,
    ADD COLUMN IF NOT EXISTS output_profile jsonb,
    ADD COLUMN IF NOT EXISTS series_conversions jsonb,
    ADD COLUMN IF NOT EXISTS default_scope_plural boolean,
    ADD COLUMN IF NOT EXISTS input_content_versions jsonb,
    ADD COLUMN IF NOT EXISTS composed_stat_vars jsonb,
    ADD COLUMN IF NOT EXISTS frame_sha256 text;

CREATE OR REPLACE FUNCTION datasets.version_record_fill_facts() RETURNS trigger
LANGUAGE plpgsql AS $fill_facts$
BEGIN
    IF NEW.prepared_at IS NULL AND NEW.input_content_versions IS NULL THEN
        NEW.prepared_at := (NEW.immutable_body ->> 'prepared_at')::timestamptz;
        NEW.definition := NULLIF(NEW.immutable_body -> 'definition', 'null'::jsonb);
        NEW.coverage := NULLIF(NEW.immutable_body -> 'coverage', 'null'::jsonb);
        NEW.citation := NULLIF(NEW.immutable_body -> 'citation', 'null'::jsonb);
        NEW.trace := NULLIF(NEW.immutable_body -> 'trace', 'null'::jsonb);
        NEW.warnings := NULLIF(NEW.immutable_body -> 'warnings', 'null'::jsonb);
        NEW.series_count := (NEW.immutable_body ->> 'series_count')::integer;
        NEW.series_coverage_stamp := NEW.immutable_body ->> 'series_coverage_stamp';
        NEW.output_profile := NULLIF(NEW.immutable_body -> 'output_profile', 'null'::jsonb);
        NEW.series_conversions := NULLIF(NEW.immutable_body -> 'series_conversions', 'null'::jsonb);
        NEW.default_scope_plural := (NEW.immutable_body ->> 'default_scope_plural')::boolean;
        NEW.input_content_versions := NULLIF(NEW.immutable_body -> 'input_content_versions', 'null'::jsonb);
        NEW.composed_stat_vars := NULLIF(NEW.immutable_body -> 'composed_stat_vars', 'null'::jsonb);
        NEW.frame_grammar_version := NEW.immutable_body -> 'trace' ->> 'frame_grammar_version';
    END IF;
    RETURN NEW;
END;
$fill_facts$;

CREATE OR REPLACE TRIGGER version_record_fill_facts
    BEFORE INSERT ON datasets.version_record
    FOR EACH ROW EXECUTE FUNCTION datasets.version_record_fill_facts();
