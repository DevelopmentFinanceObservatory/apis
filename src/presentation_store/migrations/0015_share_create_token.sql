-- Makes a share create's write token its idempotency key, as 0008_idempotent_create.sql does
-- for components. The unique index lets two creates that present the same `shw_` token store
-- one share: the second insert waits for the first transaction and then fails with SQLSTATE
-- 23505 naming this index, which `presentation_store.shares` reports as
-- `ShareWriteTokenInUse`. A share taken down keeps its digest, so its token keys no new share.
-- NULL digests (guests' shares imported from the website) are left out of the index.
-- `create_request_sha256` holds the SHA-256 of the create request a client's token keyed, so a
-- repeat of the token with a different request can be refused; it is NULL on every share
-- created with a token the server minted, and on every imported share.

CREATE UNIQUE INDEX share_write_token_sha256_key
    ON presentation.share (write_token_sha256)
    WHERE write_token_sha256 IS NOT NULL;

ALTER TABLE presentation.share ADD COLUMN create_request_sha256 bytea;
