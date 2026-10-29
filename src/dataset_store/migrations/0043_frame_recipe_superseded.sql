-- A recipe whose frame file is no longer served can give up its replay key to the frame a run
-- of the same computation publishes under another pair (`datasets_api.analysis._store`,
-- `datasets_api.recipe_store.LiveRecipeStore.supersede_replay`). Its row stays, read by its
-- pair, with `superseded_at` set, and the replay index holds only rows without it, so one key
-- still names one frame.
--
-- The unique index is built before the constraint it replaces is dropped, and `lock_timeout`
-- fails the migration after five seconds rather than queue every recipe write behind it.
-- Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.frame_recipe ADD COLUMN IF NOT EXISTS superseded_at timestamptz;

CREATE UNIQUE INDEX IF NOT EXISTS frame_recipe_replay_key_idx
    ON datasets.frame_recipe (replay_key) WHERE superseded_at IS NULL;

ALTER TABLE datasets.frame_recipe DROP CONSTRAINT IF EXISTS frame_recipe_replay_key_key;
