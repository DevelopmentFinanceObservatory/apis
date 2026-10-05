-- The origin catalogue build an async operation was submitted under, on the
-- worker's job record (`0011_operation.sql`).
--
-- `compute_operation_id` hashes this build id beside the request identity,
-- the admission coverage stamp and the rendering key, so a catalogue build
-- that changes which provenances are admitted gives a resubmission a new
-- operation instead of the completed result of the old one. The worker reads
-- the column to choose the catalogue build it runs under: the build it names
-- when that is the one the worker has loaded, otherwise the published build.
-- The build the result's data was admitted under is the version record's
-- `trace.origin_catalogue_build_id`.
--
-- Rows written before this migration, and inserts from an older API image
-- mid-rollout, carry NULL: their ids hash no build, and no later
-- submission recomputes them.
ALTER TABLE datasets.operation
    ADD COLUMN origin_catalogue_build_id text;
