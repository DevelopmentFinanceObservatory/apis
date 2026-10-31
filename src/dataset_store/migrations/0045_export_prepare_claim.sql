-- One running or unfinished prepare of an export (`datasets_api.allowance.LiveAllowanceLedger`),
-- keyed by its `export_id`, so a prepare holds across instances and survives its process.
--
-- A prepare claims the row before it spends a unit, and a lease takes it over once `expires_at`
-- passes. `spends` maps each `AllowanceKind` the prepare holds a unit of to the key it is kept
-- under, and `spent_at` is when it spent them, which names their UTC day. `artifact_digest` is
-- written after the render and before the copy, so a row holding a digest may stand behind a copy
-- and one without holds none. A retry takes over such a row and inherits its units. `refusal` is
-- the refusal the request's own file gave, which every waiter answers until the lease ends. The
-- row goes once the record is written, or when the reconciliation sweep settles it.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers this table for the runtime role, and
-- 0012_worker_role.sql makes the worker role a member of it, so this migration needs no GRANT
-- statements of its own. The new table takes no lock any running revision waits on.
CREATE TABLE IF NOT EXISTS datasets.export_prepare_claim (
    export_id text PRIMARY KEY,
    expires_at timestamptz NOT NULL,
    spends jsonb NOT NULL DEFAULT '{}',
    spent_at timestamptz,
    artifact_digest text,
    refusal jsonb
);
