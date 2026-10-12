-- Fills the fact columns 0030 adds, and the `frame_grammar_version` 0029 adds,
-- on every version row that holds only `immutable_body`, in one statement that
-- rewrites each such row once under row locks.
--
-- Each fact is the value of the body key of the same name. A JSON `null`
-- member becomes SQL NULL, and an absent member stays NULL.
-- `frame_grammar_version` is the body trace's, NULL for a body with no trace,
-- as a derived `ops_`/`ocv_` frame's.
--
-- The rows this UPDATE fills are those inserted before 0030's trigger
-- `version_record_fill_facts` existed. The trigger fills every later insert,
-- so once this migration commits every row holds its facts. Re-running it
-- changes nothing: the UPDATE touches only rows with no facts.
--
-- A row holding neither `prepared_at` nor `input_content_versions` after the
-- UPDATE has a body with neither key, which
-- `LiveSnapshotReader.read_version` refuses with `VersionFactsMissing`. The
-- closing check counts such rows and raises when it finds one, which rolls
-- this migration back and stops the deploy before the services move.

UPDATE datasets.version_record
SET prepared_at = (immutable_body ->> 'prepared_at')::timestamptz,
    definition = NULLIF(immutable_body -> 'definition', 'null'::jsonb),
    coverage = NULLIF(immutable_body -> 'coverage', 'null'::jsonb),
    citation = NULLIF(immutable_body -> 'citation', 'null'::jsonb),
    trace = NULLIF(immutable_body -> 'trace', 'null'::jsonb),
    warnings = NULLIF(immutable_body -> 'warnings', 'null'::jsonb),
    series_count = (immutable_body ->> 'series_count')::integer,
    series_coverage_stamp = immutable_body ->> 'series_coverage_stamp',
    output_profile = NULLIF(immutable_body -> 'output_profile', 'null'::jsonb),
    series_conversions = NULLIF(immutable_body -> 'series_conversions', 'null'::jsonb),
    default_scope_plural = (immutable_body ->> 'default_scope_plural')::boolean,
    input_content_versions = NULLIF(immutable_body -> 'input_content_versions', 'null'::jsonb),
    composed_stat_vars = NULLIF(immutable_body -> 'composed_stat_vars', 'null'::jsonb),
    frame_grammar_version = immutable_body -> 'trace' ->> 'frame_grammar_version'
WHERE prepared_at IS NULL
    AND input_content_versions IS NULL;

DO $check$
DECLARE
    factless bigint;
BEGIN
    SELECT count(*) INTO factless
    FROM datasets.version_record
    WHERE prepared_at IS NULL
        AND input_content_versions IS NULL;
    IF factless > 0 THEN
        RAISE EXCEPTION
            '% datasets.version_record row(s) hold neither prepared_at nor input_content_versions after the backfill: immutable_body has neither key',
            factless;
    END IF;
END;
$check$;
