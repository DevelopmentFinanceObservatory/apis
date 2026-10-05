-- Rewrites `trace.conversion` in version bodies the previous revision wrote in
-- the single-object shape after 0017 ran and before the services moved to the
-- revision that writes the list.
--
-- The deploy applies migrations first, so the previous revision kept writing
-- the object shape until the services restarted. The statement below is
-- 0017's and writes the same list: see 0017 for the mapping.
--
-- The WHERE clause selects only object-shaped conversions, so a list-shaped,
-- null or absent conversion and a repeat run change nothing.

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
