-- Rewrites `trace.conversion` in every stored `version_record.immutable_body`
-- that still holds the single-object shape into the list shape
-- `materialise.finish._conversion_trace` writes: one entry per recipe and
-- table-version pair, each naming its members.
--
-- `datasets.version_record.immutable_body` is the only stored copy of a
-- manifest trace. Head hits, snapshot reads and stored renderings all decode
-- it, and `record_version` inserts with ON CONFLICT DO NOTHING, so a
-- rematerialisation that mints the same content version never replaces it.
-- `version_artifact.rendering` holds rendering options, `subject_head.signal`
-- the currency signal, and `operation` the request and options; none of them
-- carries a trace.
--
-- The object shape always described one recipe and one table-version pair,
-- so it becomes a one-entry list. Its `native_facets` map
-- (`{stat_var: native_unit}`) becomes `members`, a list of
-- `{stat_var, native_unit}` sorted by `stat_var` in byte order (COLLATE "C",
-- matching Python's `sorted`). Every other key (`recipe`,
-- `rate_table_version`, `deflator_table_version` where present, `fallback`,
-- `projected`) is kept as stored. Content versions do not move: the trace is
-- outside the version preimage.
--
-- The WHERE clause selects only object-shaped conversions. A rewritten row
-- holds an array, and a row with no conversion or a JSON null one fails the
-- check, so a repeat run touches zero rows.

UPDATE datasets.version_record
SET immutable_body = jsonb_set(
    immutable_body,
    '{trace,conversion}',
    jsonb_build_array(
        ((immutable_body -> 'trace' -> 'conversion') - 'native_facets')
        || jsonb_build_object(
            'members',
            COALESCE(
                (
                    SELECT jsonb_agg(
                        jsonb_build_object('stat_var', facet.key, 'native_unit', facet.value)
                        ORDER BY facet.key COLLATE "C"
                    )
                    FROM jsonb_each(
                        immutable_body -> 'trace' -> 'conversion' -> 'native_facets'
                    ) AS facet
                ),
                '[]'::jsonb
            )
        )
    )
)
WHERE jsonb_typeof(immutable_body -> 'trace' -> 'conversion') = 'object';
