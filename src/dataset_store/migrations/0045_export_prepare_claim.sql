-- One running or unfinished prepare of an export (`datasets_api.allowance.LiveAllowanceLedger`),
-- keyed by its `export_id`, so one prepare runs across instances and its units survive its
-- process. The ledger's module docstring holds the states and their transitions.
--
-- `generation` takes a new random value on every state write, and every transition names the
-- value it read, so a writer acting on an old read changes nothing, even across a deleted row. `expires_at` is the lease, which the running
-- prepare's heartbeat renews without moving `generation`. `spends` maps each `AllowanceKind` the
-- row holds a unit of to the key it is kept under, and `spent_at` names their UTC day.
-- `artifact_digest` names the file a copy began for. `refusal` is what the request's own file
-- gave, for the callers waiting on that prepare.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers this table for the runtime role, and
-- 0012_worker_role.sql makes the worker role a member of it, so this migration needs no GRANT
-- statements of its own. The new table takes no lock any running revision waits on. It holds a
-- few rows at a time, so a read of the units held scans it.
CREATE TABLE IF NOT EXISTS datasets.export_prepare_claim (
    export_id text PRIMARY KEY,
    generation text NOT NULL,
    state text NOT NULL DEFAULT 'open'
        CHECK (state IN ('open', 'held', 'copying', 'landed', 'refused')),
    expires_at timestamptz NOT NULL,
    spends jsonb NOT NULL DEFAULT '{}',
    spent_at timestamptz,
    artifact_digest text,
    refusal jsonb
);
