-- Makes a create's write token its idempotency key. The unique index lets two creates that
-- present the same component write token store one component: the second insert waits for the
-- first transaction and then fails with SQLSTATE 23505 naming this index, which store.py
-- reports as `WriteTokenInUse`. NULL digests (drafts created before migration 0007) are left
-- out of the index. `create_request_sha256` holds the SHA-256 of the create request that
-- stored the component, so a repeat of the token with a different request can be refused;
-- it is NULL on every component created without one.

CREATE UNIQUE INDEX component_write_token_sha256_key
    ON presentation.component (write_token_sha256)
    WHERE write_token_sha256 IS NOT NULL;

ALTER TABLE presentation.component ADD COLUMN create_request_sha256 bytea;
