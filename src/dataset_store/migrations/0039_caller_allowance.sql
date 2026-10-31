-- What each caller has spent of its daily allowances: one row per caller key, allowance kind and
-- UTC day (`datasets_api.allowance`).
--
-- `caller_key` is a `dfo_platform.caller.Caller` key, `ip:` and a keyed digest of the client's
-- address, so no raw address is stored. `kind` is `share` or `pin`. `count` is the writes spent
-- that day. A spend is one upsert that adds one while `count` is below the limit, so concurrent
-- spends cannot pass it. Rows of past days are left in place: nothing reads them, and a sweep
-- can delete them.
--
-- Writers connect as the datasets runtime role (`DATASET_STORE_RUNTIME_DB_USER`), which the
-- Presentation surface's datasets pool uses as well. 0001_bootstrap.sql's ALTER DEFAULT
-- PRIVILEGES already grants that role this table, so this migration needs no GRANT statement.
-- The new table takes no lock any running revision waits on.

CREATE TABLE datasets.caller_allowance (
    caller_key text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('share', 'pin')),
    day date NOT NULL,
    count integer NOT NULL CHECK (count > 0),
    PRIMARY KEY (caller_key, kind, day)
);
