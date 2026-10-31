-- Temporary functions that rewrite the stored warnings naming entities in a
-- list into the shape `materialise.finish` writes: a message that counts, and
-- a structured field that carries the refs for a reader to label
-- (`materialise.codes.CountedRefs`, `materialise.codes.CountedEntityYears`).
-- `python -m dataset_store.warning_formats --dry-run` executes this file in a
-- transaction it rolls back. The file defines functions in `pg_temp` and
-- writes nothing else.
--
-- `pg_temp.warning_counted_refs_0046` rewrites one warning, and
-- `pg_temp.warnings_counted_refs_0046` every warning of a stored array. Each
-- old sentence follows the catalogue detail, which is kept as stored:
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
--
-- Every writer of these lists named bare entity dcids such as `country/KOR`:
-- the bound dcids served coverage lacks (`materialise.window.unserved_entities`)
-- and the entity values of converted rows. A dcid holds no comma or space, so
-- splitting a matched list on `, ` is exact. A list holding a comma inside a
-- name fails its pattern and stays as stored. A warning in none of these shapes is returned as stored, which
-- covers every warning in the new shape: a new unserved sentence counts
-- "requested entities" where the old one lists dcids, and the new conversion
-- sentences name no ref.

CREATE FUNCTION pg_temp.warning_refs_0046(names text[])
RETURNS jsonb
LANGUAGE sql STABLE AS $refs$
    SELECT jsonb_agg('dcid:' || name ORDER BY position)
    FROM unnest(names) WITH ORDINALITY AS listed (name, position)
$refs$;

CREATE FUNCTION pg_temp.warning_entities_0046(counted integer)
RETURNS text
LANGUAGE sql IMMUTABLE AS $entities$
    SELECT CASE WHEN counted = 1 THEN 'entity' ELSE 'entities' END
$entities$;

CREATE FUNCTION pg_temp.warning_counted_refs_0046(warning jsonb)
RETURNS jsonb
LANGUAGE plpgsql STABLE AS $rewrite$
DECLARE
    message text := warning ->> 'message';
    parts text[];
    counted integer;
BEGIN
    CASE warning ->> 'code'
    WHEN 'coverage_below_scope' THEN
        parts := regexp_match(
            message,
            '^(.*) No observations were served for ([^ ,]+(?:, [^ ,]+)*)'
            '(?: and ([0-9]+) more)?\.( .*)?$'
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
                'count', counted, 'refs', pg_temp.warning_refs_0046(string_to_array(parts[2], ', '))
            )
        );
    WHEN 'currency_unmapped_omitted' THEN
        parts := regexp_match(
            message, '^(.*) ([0-9]+) entities: ([^ ,]+(?:, [^ ,]+)*)(?: \(truncated\))?$'
        );
        IF parts IS NULL THEN
            RETURN warning;
        END IF;
        counted := parts[2]::integer;
        RETURN warning || jsonb_build_object(
            'message',
            format('%s It left out %s %s.', parts[1], counted, pg_temp.warning_entities_0046(counted)),
            'omitted_entities',
            jsonb_build_object(
                'count', counted, 'refs', pg_temp.warning_refs_0046(string_to_array(parts[3], ', '))
            )
        );
    WHEN 'deflator_missing_omitted' THEN
        parts := regexp_match(
            message,
            '^(.*) ([0-9]+) entities: ([^ ,]+ \([^)]*\)(?:, [^ ,]+ \([^)]*\))*)'
            '(?: \(truncated\))?$'
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
                    FROM regexp_matches(parts[3], '([^ ,]+) \(([^)]*)\)', 'g')
                        WITH ORDINALITY AS listed (item, position)
                )
            )
        );
    ELSE
        RETURN warning;
    END CASE;
END;
$rewrite$;

CREATE FUNCTION pg_temp.warnings_counted_refs_0046(warnings jsonb)
RETURNS jsonb
LANGUAGE sql STABLE AS $rewrite_all$
    SELECT CASE
        WHEN jsonb_typeof(warnings) = 'array' THEN (
            SELECT coalesce(
                jsonb_agg(pg_temp.warning_counted_refs_0046(warning.value) ORDER BY warning.position),
                '[]'::jsonb
            )
            FROM jsonb_array_elements(warnings) WITH ORDINALITY AS warning (value, position)
        )
        ELSE warnings
    END
$rewrite_all$;
