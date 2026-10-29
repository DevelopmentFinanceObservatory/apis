-- Stores the rendering of the frame each revision draws, recorded when the component is
-- created and kept by every later revision of the component, so a live draw or a share
-- reads the rendering the card was made from even after another rendering's file is linked
-- under the same version. A dataset frame's rendering is the object
-- `datasets_api.doors.refresh.StoredRendering.to_record` writes (its base version, groupby and
-- party-totals cut); a derived frame's is `{"normal_form_hash": ...}` from its recipe.
--
-- Additive and nullable, with no backfill: NULL is a note, a frame whose records did not say
-- its rendering, or a component created before this migration. A NULL rendering is read from
-- the frame's records at render time.

ALTER TABLE presentation.component_revision ADD COLUMN frame_rendering jsonb;
