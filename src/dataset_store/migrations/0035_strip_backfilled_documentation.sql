-- Removes the source documentation stored on version bodies prepared before
-- their mint captured any.
--
-- A stored `documentation` is a source's documentation as the version's first
-- mint read it (`datasets_api.manifest_models.CitedDocumentation`). The
-- revisions that stored it began serving on dev at 2026-10-04T09:22:14Z. A body
-- prepared earlier and holding a `documentation` holds one copied in later from
-- a live read of the graph, which describes what the pipeline had published at
-- that read and not at the version's mint. The bundle's `CITATION.cff` takes
-- `date-accessed` from `documentation.last_update`
-- (`materialise.bundle_citation`), and a description record takes its source's
-- download date and pipeline steps from it
-- (`datasets_api.description_record.source_evidence`), so such a copy cites a
-- download later than the version's data.
--
-- A body prepared in the minutes before the cut-off by a revision that stored
-- documentation loses what its mint read. `datasets_api.manifest.healed_citation`
-- fills a missing `documentation` from a live read at render and writes nowhere,
-- so no rendering loses it, and its description record states no download date
-- or pipeline steps.
--
-- For every row whose `prepared_at` precedes 2026-10-04T09:22:14Z, the UPDATE
-- drops `documentation` from each source entry of the body. Every other member
-- of the entry (licence, url, source, label, processing level) and every other
-- key of the body stay as stored.
--
-- The `citation` column holds the body's `citation` (0030). The trigger
-- `version_record_fill_facts` fires on INSERT only, so the UPDATE sets the
-- column from the rewritten body in the same statement.
--
-- Rows prepared at or after the cut-off, rows with no `prepared_at`, bodies
-- with no citation and bodies whose sources hold no `documentation` are not
-- selected, so re-running this migration changes nothing.

UPDATE datasets.version_record AS v
SET immutable_body = stripped.body,
    citation = stripped.body -> 'citation'
FROM (
    SELECT
        r.subject_id,
        r.content_version,
        jsonb_set(
            r.immutable_body,
            '{citation,sources}',
            (
                SELECT jsonb_agg(
                    CASE
                        WHEN jsonb_typeof(source.value) = 'object' THEN source.value - 'documentation'
                        ELSE source.value
                    END
                    ORDER BY source.position
                )
                FROM jsonb_array_elements(r.immutable_body -> 'citation' -> 'sources')
                    WITH ORDINALITY AS source (value, position)
            )
        ) AS body
    FROM datasets.version_record AS r
    WHERE r.prepared_at < TIMESTAMPTZ '2026-10-04T09:22:14Z'
        AND jsonb_path_exists(
            r.immutable_body,
            'strict $.citation.sources[*] ? (exists(@.documentation))',
            '{}',
            true
        )
) AS stripped
WHERE v.subject_id = stripped.subject_id
    AND v.content_version = stripped.content_version;
