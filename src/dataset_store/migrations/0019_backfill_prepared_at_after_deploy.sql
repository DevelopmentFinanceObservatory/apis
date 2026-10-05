-- Fills `prepared_at` on version bodies, and `prepared_at` and
-- `provenance_refs` on collection members, for rows the previous revision
-- wrote after 0018 ran and before the services moved to the revision that
-- writes them.
--
-- The deploy applies migrations first, so the previous revision keeps minting
-- bodies without these keys until the services restart. Each statement below
-- is 0018's, and fills the same keys from the same sources: see 0018 for the
-- source of each value.
--
-- Every statement is guarded, so a row that already carries its keys, a repeat
-- run, and `ops_` subjects change nothing.

UPDATE datasets.version_record AS v
SET immutable_body = jsonb_set(
    v.immutable_body,
    '{prepared_at}',
    to_jsonb(
        to_char(
            COALESCE(base.recorded_at, own.recorded_at) AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS"Z"'
        )
    )
)
FROM datasets.version_record AS own
LEFT JOIN datasets.version_record AS base
    ON base.subject_id = own.subject_id
    AND base.content_version = own.base_content_version
WHERE own.subject_id = v.subject_id
    AND own.content_version = v.content_version
    AND NOT starts_with(v.subject_id, 'ops_')
    AND NOT v.immutable_body ? 'prepared_at';

-- Runs after the statement above, so every member record and every
-- collection body already carries its own `prepared_at`.
UPDATE datasets.version_record AS v
SET immutable_body = jsonb_set(
    v.immutable_body,
    '{collection,members}',
    (
        SELECT jsonb_agg(
            entry.member
            || CASE
                WHEN entry.member ? 'prepared_at' THEN '{}'::jsonb
                ELSE jsonb_build_object(
                    'prepared_at',
                    COALESCE(
                        member_record.immutable_body ->> 'prepared_at',
                        v.immutable_body ->> 'prepared_at'
                    )
                )
            END
            || CASE
                WHEN entry.member ? 'provenance_refs' THEN '{}'::jsonb
                ELSE jsonb_build_object(
                    'provenance_refs',
                    COALESCE(
                        (
                            SELECT jsonb_agg(cited.ref ORDER BY cited.first_position)
                            FROM (
                                SELECT source.value ->> 'ref' AS ref,
                                    min(source.position) AS first_position
                                FROM jsonb_array_elements(
                                    member_record.immutable_body -> 'citation' -> 'sources'
                                ) WITH ORDINALITY AS source (value, position)
                                GROUP BY 1
                            ) AS cited
                        ),
                        CASE
                            WHEN member_record.subject_id IS NULL
                                AND entry.position = 1
                                AND jsonb_typeof(v.immutable_body -> 'trace' -> 'facets') = 'array'
                            THEN (
                                SELECT jsonb_build_array(facet.value ->> 'ref')
                                FROM jsonb_array_elements(v.immutable_body -> 'trace' -> 'facets')
                                    AS facet (value)
                                WHERE facet.value ->> 'axis' = 'provenance'
                                LIMIT 1
                            )
                        END,
                        '[]'::jsonb
                    )
                )
            END
            ORDER BY entry.position
        )
        FROM jsonb_array_elements(v.immutable_body -> 'collection' -> 'members')
            WITH ORDINALITY AS entry (member, position)
        LEFT JOIN datasets.version_record AS member_record
            ON member_record.subject_id = entry.member -> 'definition' ->> 'spec_id'
            AND member_record.content_version = entry.member ->> 'content_version'
    )
)
WHERE v.subject_kind = 'collection'
    AND NOT starts_with(v.subject_id, 'ops_')
    AND jsonb_typeof(v.immutable_body -> 'collection' -> 'members') = 'array'
    AND EXISTS (
        SELECT 1
        FROM jsonb_array_elements(v.immutable_body -> 'collection' -> 'members') AS entry (member)
        WHERE NOT entry.member ? 'prepared_at' OR NOT entry.member ? 'provenance_refs'
    );
