-- Gives every stored manifest the `prepared_at`, and every collection member
-- the `provenance_refs`, the Datasets models now require.
--
-- `datasets.version_record.immutable_body` is written once and decoded on
-- every snapshot read, head hit, re-mint and stored rendering. `prepared_at`
-- is the instant a version was first minted, so a body that lacks it takes
-- its own record's `recorded_at`, the commit time of that first mint, in the
-- wire form `datasets_api.serve._prepared_now` writes: RFC 3339 UTC to the
-- second (`2026-10-04T09:00:00Z`). A full groupby collapse's record
-- (`base_content_version` set) takes its base record's `recorded_at`, since
-- a collapsed version is rendered and named under its base's `prepared_at`.
-- Each member entry of a collection body takes its member version's
-- `prepared_at`, and the collection's own when the member has no record of
-- its own (member records are best-effort).
-- Each member entry also takes `provenance_refs`: the `ref` of every cited
-- source in its member version's own body, first occurrence first. A member
-- with no record takes the `provenance` facet of the collection's trace when
-- it is the first member, since a collection trace carries the first
-- member's facets, and `[]` otherwise.
-- Cited sources keep no `documentation` key; the models read its absence as
-- a version minted before documentation was captured. Content versions do
-- not move: `content_version` hashes the frame, not the body, and no reader
-- digests or byte-compares a manifest body.
--
-- Artifact links stay. A bundle's `rendering` carries a rendition token
-- (`datasets_api.render.RENDITION_KEY`) that old bundles lack, so a request
-- for a bundle no longer matches their rendering keys and renders a new one,
-- while each old bundle stays listed for its version until it expires.
--
-- Published derived frames (`ops_` subjects, written by the publish door)
-- hold no manifest and are left alone.
--
-- Every statement is guarded, so a repeat run changes nothing.

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
