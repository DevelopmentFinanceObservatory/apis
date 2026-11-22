-- When an analysis root last read a link's file as the rendering made from that head.
--
-- `version_artifact_head.read_at` is the last time an analysis root read the link's file as
-- the minting options (a groupby sum or a party-totals cut) rendered from that raw head, set
-- when a root renders it and when a root reads it recorded
-- (`datasets_api.analysis_roots.LiveRootSource._recorded_rendering`), at most once an hour
-- (`datasets_api.render_ahead.READ_MARK_INTERVAL`). A render ahead
-- (`datasets_api.render_ahead.render_ahead`) reads it to choose which renderings to make from
-- a new head before a reader asks: those read within `DATASETS_RENDER_AHEAD_READ_WINDOW_SECONDS`,
-- newest read first. The render ahead writes the head row of what it renders and leaves its
-- `read_at` NULL, so a rendering nobody reads ages out of the window and stops being rendered.
-- No other statement writes it. NULL on a row written before this column, by the previous
-- image, or by any render other than an analysis root's, which a render ahead never picks.
--
-- Adding a nullable column with no default rewrites no row. ALTER TABLE takes ACCESS EXCLUSIVE
-- on `version_artifact_head`, and `lock_timeout` fails it after five seconds so that it does
-- not queue every later write behind it. The previous image inserts head rows without it and
-- never reads it. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_artifact_head
    ADD COLUMN IF NOT EXISTS read_at timestamptz;
