-- Drops the draft lease table created by 0004_draft_lease.sql. No code reads or writes it:
-- write authority is the write token digest added in 0007_write_token.sql. Dropping the table
-- also drops its `expires_at` index.

DROP TABLE IF EXISTS presentation.draft_lease;
