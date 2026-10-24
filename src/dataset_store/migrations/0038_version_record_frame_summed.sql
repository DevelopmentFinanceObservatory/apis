-- Whether a derived version's stored frame holds its summed rows.
--
-- `frame_summed` is set on a row naming a `base_content_version`, a derived version
-- (`datasets_api.render.DerivedVersion.summed`). It is true when the stored frame is the
-- groupby collapse and false when it is the party-totals cut frame, which a recorded groupby
-- still sums when its result is not a grammar frame. A row naming no base leaves it NULL.
-- NULL on a derived row means not recorded, as on a row recorded before this column or
-- written by the previous image until the services move. Such a version renders from its
-- base frame when it is a party-totals cut with a groupby
-- (`datasets_api.recorded_rendering.serving_frames`).
-- No CHECK ties the column to `base_content_version`, since the previous image inserts
-- derived rows without it.
--
-- Adding a nullable column rewrites no row. ALTER TABLE takes ACCESS EXCLUSIVE on
-- `version_record`, and `lock_timeout` fails it after five seconds so that it does not queue
-- every later write behind it. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_record
    ADD COLUMN IF NOT EXISTS frame_summed boolean;
