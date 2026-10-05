-- Adds the write-token digest to components and compositions. A create stores the unsalted
-- SHA-256 of the token it returns once (`cmw_` for a component, `cpw_` for a composition);
-- nothing updates it afterwards. The column is nullable: a row created before this migration
-- keeps NULL, and a NULL digest refuses every write, so such a draft stays read-only.

ALTER TABLE presentation.component ADD COLUMN write_token_sha256 bytea;
ALTER TABLE presentation.composition ADD COLUMN write_token_sha256 bytea;
