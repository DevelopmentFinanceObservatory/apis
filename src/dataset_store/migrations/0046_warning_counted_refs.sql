-- Rewrites the stored warnings naming refs in their message into the shape
-- `materialise.finish` and `datasets_api.origin` write: a message that counts,
-- and structured fields that carry the refs for a reader to label
-- (`materialise.codes.CountedRefs`, `materialise.codes.CountedEntityYears`), in
-- the `warnings` column and in the `warnings` of `immutable_body`, which
-- encodes the same facts for an image that reads bodies
-- (`manifest.version_body`). Each is rewritten from its own stored value, and a
-- row is updated only where the rewrite changes one of them, so every other row
-- keeps its stored bytes. Re-running this migration changes nothing: a
-- rewritten warning matches none of the old shapes.
--
-- The old member sentence of `coverage_below_scope`
-- (`dataset_store.warning_formats.MEMBER_SENTENCE`) is left as stored: its
-- split into off-cadence and out-of-window members needs graph reads, so the
-- deploy's `datasets-maintenance backfill-warnings` step rewrites it.
-- `datasets-maintenance settle` counts the rows still holding any old shape
-- (`dataset_store.warning_formats.OLD_FORMATS`) and fails the deploy when one
-- exists. The image serving while it applies already writes the new shapes,
-- so no row in an old shape arrives after it.
--
-- The functions live in `pg_temp`, so they belong to this migration's session
-- alone, and are dropped before the transaction commits; the migration runner
-- applies a file in one transaction on one connection. The UPDATE takes row
-- locks only, so a running revision's reads never wait on it, and
-- `lock_timeout` fails it after five seconds behind a row another writer holds.
--
-- `pg_temp.warning_counted_refs_0046` rewrites one warning, and
-- `pg_temp.warnings_counted_refs_0046` every warning of a stored array. Each old
-- sentence follows the catalogue detail, which is kept as stored:
--
--   coverage_below_scope, unserved entities:
--     ` No observations were served for KEN, UGA and 3 more.`
--     The first five sorted bare entity dcids, with a count of the rest. It
--     becomes ` No observations were served for 5 requested entities.` and
--     `unserved_entities` `{count, refs}`, each ref `dcid:` prefixed, in the
--     stored order. The text before and after the sentence stays as stored. A
--     member sentence (` No observations were served for dcid:A, dcid:B, which
--     hold none ...`) fails the pattern, because a comma and a word follow its
--     list, and stays as stored.
--   currency_unmapped_omitted:
--     ` 12 entities: A, B, ..., J (truncated)`, no closing full stop. The
--     first ten sorted bare entity dcids. It becomes ` It left out 12
--     entities.` and `omitted_entities` `{count, refs}`.
--   deflator_missing_omitted:
--     ` 2 entities: A (1990-2001, 2004), B (2010)[ (truncated)]`. The first
--     ten entities, sorted, each with every run of its omitted years. It
--     becomes ` It left out rows of 2 entities.` and `omitted_entity_years`
--     `{count, entities}`, each entity `{ref, years}`.
--   withheld_by_registry:
--     ` V served no observations for KEN, UGA and 3 more; withheld provenances
--     holding data there: dc/p1, dc/p2.` The bare variable dcid, the first
--     five places in request order with a count of the rest, and every
--     withheld provenance. It becomes ` No observations were served for 5
--     entities, where 2 withheld provenances hold data.`
--     (`datasets_api.origin.withheld_by_registry_warning`), with
--     `unserved_entities`, `withheld_members` holding the variable, and
--     `withheld_provenances` holding the first five provenances.
--   un_system_unavailable:
--     ` No UN data was read for KEN, UGA and 3 more.` It becomes ` No UN data
--     was read for 5 entities.` (`datasets_api.origin.
--     un_system_unavailable_warning`) and `unserved_entities`.
--   deflator_projected:
--     ` 12 row(s), for example: KEN@2020, ...[ (truncated)]`. It becomes ` It
--     applies to 12 rows.` (`materialise.finish.deflator_projected_warning`);
--     the trace's conversion entries hold the examples.
--
-- The functions split a list on `, `. The old lists hold bare entity dcids:
-- the bound dcids served coverage lacks (`materialise.window.unserved_entities`)
-- and the entity values of converted rows. A dcid holds no `, `, so the split
-- is exact; a list of labels such as `Korea, Rep.` would be mis-split. A list
-- holding a comma inside a name fails its pattern and stays as stored.
--
-- Each regular expression holding a backslash is an escape string (`E'...'`),
-- so it reads the same whatever `standard_conforming_strings` is. A warning in
-- none of these shapes is returned as stored, which covers every warning in
-- the new shape: each new sentence states a count where the old one lists
-- dcids.

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION pg_temp.warning_refs_0046(names text[])
RETURNS jsonb
LANGUAGE sql STABLE AS $refs$
    SELECT jsonb_agg('dcid:' || name ORDER BY position)
    FROM unnest(names) WITH ORDINALITY AS listed (name, position)
$refs$;

CREATE OR REPLACE FUNCTION pg_temp.warning_entities_0046(counted integer)
RETURNS text
LANGUAGE sql IMMUTABLE AS $entities$
    SELECT CASE WHEN counted = 1 THEN 'entity' ELSE 'entities' END
$entities$;

CREATE OR REPLACE FUNCTION pg_temp.warning_counted_refs_0046(warning jsonb)
RETURNS jsonb
LANGUAGE plpgsql STABLE AS $rewrite$
DECLARE
    message text := warning ->> 'message';
    parts text[];
    provenances text[];
    counted integer;
BEGIN
    CASE warning ->> 'code'
    WHEN 'coverage_below_scope' THEN
        parts := regexp_match(
            message,
            E'^(.*) No observations were served for ([^ ,]+(?:, [^ ,]+)*)(?: and ([0-9]+) more)?\\.( .*)?$'
        );
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := cardinality(string_to_array(parts[2], ', ')) + coalesce(parts[3]::integer, 0);
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s No observations were served for %s requested %s.%s',
                parts[1],
                counted,
                pg_temp.warning_entities_0046(counted),
                coalesce(parts[4], '')
            ),
            'unserved_entities',
            jsonb_build_object(
                'count',
                counted,
                'refs',
                pg_temp.warning_refs_0046(string_to_array(parts[2], ', '))
            )
        );
    WHEN 'currency_unmapped_omitted' THEN
        parts := regexp_match(
            message, E'^(.*) ([0-9]+) entities: ([^ ,]+(?:, [^ ,]+)*)(?: \\(truncated\\))?$'
        );
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := parts[2]::integer;
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s It left out %s %s.',
                parts[1],
                counted,
                pg_temp.warning_entities_0046(counted)
            ),
            'omitted_entities',
            jsonb_build_object(
                'count',
                counted,
                'refs',
                pg_temp.warning_refs_0046(string_to_array(parts[3], ', '))
            )
        );
    WHEN 'deflator_missing_omitted' THEN
        parts := regexp_match(
            message,
            E'^(.*) ([0-9]+) entities: ([^ ,]+ \\([^)]*\\)(?:, [^ ,]+ \\([^)]*\\))*)(?: \\(truncated\\))?$'
        );
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := parts[2]::integer;
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s It left out rows of %s %s.',
                parts[1],
                counted,
                pg_temp.warning_entities_0046(counted)
            ),
            'omitted_entity_years',
            jsonb_build_object(
                'count',
                counted,
                'entities',
                (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'ref', 'dcid:' || listed.item[1],
                            'years', to_jsonb(string_to_array(listed.item[2], ', '))
                        )
                        ORDER BY listed.position
                    )
                    FROM regexp_matches(parts[3], E'([^ ,]+) \\(([^)]*)\\)', 'g')
                        WITH ORDINALITY AS listed (item, position)
                )
            )
        );
    WHEN 'withheld_by_registry' THEN
        parts := regexp_match(
            message,
            E'^(.*) ([^ ,]+) served no observations for ([^ ,]+(?:, [^ ,]+)*)(?: and ([0-9]+) more)?; withheld provenances holding data there: ([^ ,]+(?:, [^ ,]+)*)\\.$'
        );
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := cardinality(string_to_array(parts[3], ', ')) + coalesce(parts[4]::integer, 0);
        provenances := string_to_array(parts[5], ', ');
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s No observations were served for %s %s, where %s withheld %s %s data.',
                parts[1],
                counted,
                pg_temp.warning_entities_0046(counted),
                cardinality(provenances),
                CASE WHEN cardinality(provenances) = 1 THEN 'provenance' ELSE 'provenances' END,
                CASE WHEN cardinality(provenances) = 1 THEN 'holds' ELSE 'hold' END
            ),
            'unserved_entities',
            jsonb_build_object(
                'count',
                counted,
                'refs',
                pg_temp.warning_refs_0046(string_to_array(parts[3], ', '))
            ),
            'withheld_members',
            jsonb_build_object('count', 1, 'refs', pg_temp.warning_refs_0046(ARRAY[parts[2]])),
            'withheld_provenances',
            jsonb_build_object(
                'count',
                cardinality(provenances),
                'refs',
                pg_temp.warning_refs_0046(provenances[1:5])
            )
        );
    WHEN 'un_system_unavailable' THEN
        parts := regexp_match(
            message,
            E'^(.*) No UN data was read for ([^ ,]+(?:, [^ ,]+)*)(?: and ([0-9]+) more)?\\.$'
        );
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := cardinality(string_to_array(parts[2], ', ')) + coalesce(parts[3]::integer, 0);
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s No UN data was read for %s %s.',
                parts[1],
                counted,
                pg_temp.warning_entities_0046(counted)
            ),
            'unserved_entities',
            jsonb_build_object(
                'count',
                counted,
                'refs',
                pg_temp.warning_refs_0046(string_to_array(parts[2], ', '))
            )
        );
    WHEN 'deflator_projected' THEN
        parts := regexp_match(message, E'^(.*) ([0-9]+) row\\(s\\), for example: .*$');
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := parts[2]::integer;
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s It applies to %s %s.',
                parts[1],
                counted,
                CASE WHEN counted = 1 THEN 'row' ELSE 'rows' END
            )
        );
    ELSE
        RETURN warning;
    END CASE;
END;
$rewrite$;

CREATE OR REPLACE FUNCTION pg_temp.warnings_counted_refs_0046(warnings jsonb)
RETURNS jsonb
LANGUAGE sql STABLE AS $rewrite_all$
    SELECT CASE
        WHEN jsonb_typeof(warnings) = 'array' THEN (
            SELECT coalesce(
                jsonb_agg(
                    pg_temp.warning_counted_refs_0046(warning.value) ORDER BY warning.position
                ),
                '[]'::jsonb
            )
            FROM jsonb_array_elements(warnings) WITH ORDINALITY AS warning (value, position)
        )
        ELSE warnings
    END
$rewrite_all$;

UPDATE datasets.version_record
SET
    warnings = pg_temp.warnings_counted_refs_0046(warnings),
    immutable_body = CASE
        WHEN jsonb_typeof(immutable_body -> 'warnings') = 'array' THEN jsonb_set(
            immutable_body,
            '{warnings}',
            pg_temp.warnings_counted_refs_0046(immutable_body -> 'warnings')
        )
        ELSE immutable_body
    END
WHERE pg_temp.warnings_counted_refs_0046(warnings) IS DISTINCT FROM warnings
    OR pg_temp.warnings_counted_refs_0046(immutable_body -> 'warnings')
        IS DISTINCT FROM immutable_body -> 'warnings';

DROP FUNCTION pg_temp.warnings_counted_refs_0046(jsonb);
DROP FUNCTION pg_temp.warning_counted_refs_0046(jsonb);
DROP FUNCTION pg_temp.warning_entities_0046(integer);
DROP FUNCTION pg_temp.warning_refs_0046(text[]);
