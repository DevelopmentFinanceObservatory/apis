-- Rewrites the stored warnings that name refs in their message into the shape
-- `materialise.finish` writes: a message that counts, and a structured field
-- that carries the refs for a reader to label (`materialise.codes.CountedRefs`,
-- `materialise.codes.CountedEntityYears`).
--
-- `datasets.version_record.warnings` is the column every read assembles a
-- manifest's warnings from (`datasets_api.manifest.stored_manifest`), and
-- `record_version` inserts with ON CONFLICT DO NOTHING, so a later mint of the
-- same content version never replaces it. `immutable_body` keeps its copy as
-- stored: no read decodes its warnings once the facts are filled. The trigger
-- `version_record_fill_facts` fires on INSERT only, so the UPDATE fires
-- nothing. Description records copy no text from these codes. Content
-- versions do not move: warnings are outside the version preimage.
--
-- Each old sentence follows the catalogue detail, which is kept as stored:
--
--   coverage_below_scope, unserved entities:
--     ` No observations were served for KEN, UGA and 3 more.`
--     The first five sorted bare entity dcids, with a count of the rest. It
--     becomes ` No observations were served for 5 requested entities.` and
--     `unserved_entities` `{count, refs}`, each ref `dcid:` prefixed, in the
--     stored order. A row stored while member sentences already counted can
--     hold counted member sentences after it, which stay as stored.
--   coverage_below_scope, group members that served nothing, under a window
--   that is not a range:
--     ` No observations were served for dcid:A, dcid:B, which hold none at the
--     observation period the other members serve.`
--     The first five `dcid:` StatVar refs in request order, with a count of the
--     rest. Under such a window every uncounted member is off cadence
--     (`materialise.generation._plan_group_members`), so it becomes ` 2 of 7
--     series report at a different frequency and are left out of the
--     figures.`, `off_cadence_members` `{count, refs}` and `member_count`.
--   coverage_below_scope, group members that served nothing, under a range
--   window:
--     ` No observations were served for dcid:A, dcid:B, which hold none at the
--     observation period the other members serve in the requested years.`
--     This sentence counted two kinds together: members whose facets at another
--     observation period hold the requested years, which the writer now calls
--     `off_cadence_members`, and members holding no observation in those years,
--     its `out_of_window_members`. The row records nothing that tells them
--     apart: the split was computed from each member's probe candidates and
--     never stored. Either field would claim a split the data cannot back, so
--     the sentence becomes the statement true of both, ` 2 of 7 series hold no
--     observations at the group's observation period in the requested years
--     and are left out of the figures.`, with no structured field and no
--     `member_count`. The refs stay only in `immutable_body`.
--   In both, the series total is the length of the row's
--   `trace.resolved_stat_vars`, one entry per member request
--   (`materialise.finish._finish_group`), the requests the writer counts. A
--   trace holding no such list leaves the total out: ` 2 series report ...`,
--   ` 2 series hold ...`, with no `member_count`.
--   Either can be followed by the countries-fallback sentence, kept as stored.
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
-- A warning in none of these shapes is returned as stored, which covers every
-- warning already in the new shape: a new unserved sentence counts "requested
-- entities" where the old one lists dcids, and the new member and conversion
-- sentences name no ref. A row is updated only when one of its warnings
-- changes, so a repeat run touches zero rows.
--
-- This migration must run only once no serving revision writes the old
-- shapes: the image that writes the new shapes is the previous image of the
-- deploy that carries it.

CREATE FUNCTION datasets.warning_counted_refs_0045(warning jsonb, member_count integer)
RETURNS jsonb
LANGUAGE plpgsql IMMUTABLE AS $rewrite$
DECLARE
    fallback CONSTANT text :=
        ' No place was named, so the answer covers the countries with data, '
        'because the full list of places with data was too large or unavailable.';
    truncated CONSTANT text := ' (truncated)';
    message text := warning ->> 'message';
    tail text := '';
    found text[];
    named text[];
    counted integer;
BEGIN
    IF warning ->> 'code' = 'coverage_below_scope' THEN
        IF right(message, length(fallback)) = fallback THEN
            message := left(message, -length(fallback));
            tail := fallback;
        END IF;
        found := regexp_match(
            message,
            '^(.*) No observations were served for (dcid:[^ ,]+(?:, dcid:[^ ,]+)*)'
            '(?: and ([0-9]+) more)?, which hold none at the observation period the other '
            'members serve( in the requested years)?\.$'
        );
        IF found IS NOT NULL THEN
            named := string_to_array(found[2], ', ');
            counted := cardinality(named) + coalesce(found[3]::integer, 0);
            message := found[1];
            tail := format(
                ' %s%s series %s and are left out of the figures.',
                counted,
                CASE WHEN member_count IS NULL THEN '' ELSE format(' of %s', member_count) END,
                CASE
                    WHEN found[4] IS NULL THEN 'report at a different frequency'
                    ELSE 'hold no observations at the group''s observation period in the '
                        'requested years'
                END
            ) || tail;
            IF found[4] IS NULL THEN
                warning := warning
                    || jsonb_build_object(
                        'off_cadence_members',
                        jsonb_build_object('count', counted, 'refs', to_jsonb(named))
                    )
                    || jsonb_strip_nulls(jsonb_build_object('member_count', member_count));
            END IF;
        END IF;
        found := regexp_match(
            message,
            '^(.*) No observations were served for ([^ ,]+(?:, [^ ,]+)*)(?: and ([0-9]+) more)?'
            '\.((?: [0-9]+ (?:of [0-9]+ )?series [^.]*\.)*)$'
        );
        IF found IS NOT NULL THEN
            named := string_to_array(found[2], ', ');
            counted := cardinality(named) + coalesce(found[3]::integer, 0);
            message := format(
                '%s No observations were served for %s requested %s.%s',
                found[1],
                counted,
                CASE WHEN counted = 1 THEN 'entity' ELSE 'entities' END,
                found[4]
            );
            warning := warning || jsonb_build_object(
                'unserved_entities',
                jsonb_build_object(
                    'count',
                    counted,
                    'refs',
                    (
                        SELECT jsonb_agg('dcid:' || entity ORDER BY position)
                        FROM unnest(named) WITH ORDINALITY AS listed (entity, position)
                    )
                )
            );
        END IF;
        RETURN warning || jsonb_build_object('message', message || tail);
    END IF;

    IF warning ->> 'code' NOT IN ('currency_unmapped_omitted', 'deflator_missing_omitted') THEN
        RETURN warning;
    END IF;
    IF right(message, length(truncated)) = truncated THEN
        message := left(message, -length(truncated));
    END IF;

    IF warning ->> 'code' = 'currency_unmapped_omitted' THEN
        found := regexp_match(message, '^(.*) ([0-9]+) entities: ([^ ,]+(?:, [^ ,]+)*)$');
        IF found IS NULL THEN
            RETURN warning;
        END IF;
        counted := found[2]::integer;
        RETURN warning || jsonb_build_object(
            'message',
            format(
                '%s It left out %s %s.',
                found[1],
                counted,
                CASE WHEN counted = 1 THEN 'entity' ELSE 'entities' END
            ),
            'omitted_entities',
            jsonb_build_object(
                'count',
                counted,
                'refs',
                (
                    SELECT jsonb_agg('dcid:' || entity ORDER BY position)
                    FROM unnest(string_to_array(found[3], ', '))
                        WITH ORDINALITY AS listed (entity, position)
                )
            )
        );
    END IF;

    found := regexp_match(
        message,
        '^(.*) ([0-9]+) entities: ([^ ,]+ \([^)]*\)(?:, [^ ,]+ \([^)]*\))*)$'
    );
    IF found IS NULL THEN
        RETURN warning;
    END IF;
    counted := found[2]::integer;
    RETURN warning || jsonb_build_object(
        'message',
        format(
            '%s It left out rows of %s %s.',
            found[1],
            counted,
            CASE WHEN counted = 1 THEN 'entity' ELSE 'entities' END
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
                FROM regexp_matches(found[3], '([^ ,]+) \(([^)]*)\)', 'g')
                    WITH ORDINALITY AS listed (item, position)
            )
        )
    );
END;
$rewrite$;

UPDATE datasets.version_record AS v
SET warnings = rewritten.warnings
FROM (
    SELECT
        r.subject_id,
        r.content_version,
        (
            SELECT jsonb_agg(
                datasets.warning_counted_refs_0045(
                    warning.value,
                    CASE
                        WHEN jsonb_typeof(r.trace -> 'resolved_stat_vars') = 'array'
                            THEN jsonb_array_length(r.trace -> 'resolved_stat_vars')
                    END
                )
                ORDER BY warning.position
            )
            FROM jsonb_array_elements(r.warnings) WITH ORDINALITY AS warning (value, position)
        ) AS warnings
    FROM datasets.version_record AS r
    WHERE jsonb_typeof(r.warnings) = 'array'
        AND r.warnings @? '$[*] ? (@.code == "coverage_below_scope"
            || @.code == "currency_unmapped_omitted"
            || @.code == "deflator_missing_omitted")'
) AS rewritten
WHERE v.subject_id = rewritten.subject_id
    AND v.content_version = rewritten.content_version
    AND v.warnings IS DISTINCT FROM rewritten.warnings;

DROP FUNCTION datasets.warning_counted_refs_0045(jsonb, integer);
