-- A derived frame's recipe: the op-spec tree analysis-mcp ran to produce
-- an `ops_`/`ocv_` frame, with the hashes a replay compares. One row per
-- (subject_id, content_version), written once and never overwritten
-- (`datasets_api.recipe_store.LiveRecipeStore.write_recipe`).
--
-- The recipe has a table of its own, outside
-- `datasets.version_record.immutable_body`. The publish door refuses a
-- republish whose body differs from the stored one
-- (`PublishLineageDiverge` in `datasets_api.doors.publish`), so a recipe
-- held in the body would refuse every republish of a frame published
-- without one. Here a frame gains its recipe after its version row exists.
--
-- `recipe` holds the whole recipe as received. `normal_form_hash` and
-- `replay_key` repeat two of its fields as columns: the first is the
-- value a later write for the same pair must match, the second is the
-- replay index.
--
-- `replay_key` is unique. It hashes the recipe's normal form, its root
-- dataset versions, and the execution identity, so it names one replayed
-- frame, and a second pair under a held key is refused.
--
-- `(subject_id, content_version)` FK-references `datasets.version_record`,
-- so a recipe and a replay key only ever name a frame this store holds.
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers this table for the
-- runtime role, and 0012_worker_role.sql makes the worker role a member of
-- it, so this migration needs no GRANT statements of its own.
CREATE TABLE datasets.frame_recipe (
    subject_id text NOT NULL,
    content_version text NOT NULL,
    normal_form_hash text NOT NULL,
    replay_key text NOT NULL UNIQUE,
    recipe jsonb NOT NULL,
    recorded_at timestamptz NOT NULL,
    PRIMARY KEY (subject_id, content_version),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
