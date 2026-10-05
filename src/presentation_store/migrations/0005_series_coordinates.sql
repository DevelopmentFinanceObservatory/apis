-- Adds the column a revision's resolved coordinates persist to, so a later
-- readback (`get_component`/`_fact_summary_from_payload`) can serve the
-- same `fact_summary.series[].coordinates` the create/revise reply carried
-- instead of always answering `{}`. Additive and nullable: no backfill --
-- a NULL here is a revision written before this migration shipped, and
-- reads back as `{}` the same way it always did.
--
-- Shape: an ordered JSON array of `{"id": <series id or null>,
-- "coordinates": {...}}`, one entry per resolved series in the same order
-- `candidate.series`/`bound_series` were built in -- not keyed by id alone,
-- since `table` and `stat_card` mint no per-row/per-series id
-- (`_fact_summary_from_payload` reads those back as `id: None`, so an
-- id-keyed shape would silently lose them; the read path falls back to
-- positional matching for entries with no id). Uncapped: already bounded by
-- `MAX_SERIES_PER_COMPONENT`, with `FACT_SUMMARY_SERIES_CAP` staying the
-- read-time concern it always was.

ALTER TABLE presentation.component_revision ADD COLUMN series_coordinates jsonb;
