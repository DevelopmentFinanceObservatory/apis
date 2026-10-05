-- Bootstrap: the app schema and the runtime role's default privileges on
-- it, so every table created later inherits access without a per-table
-- grant in each future migration. Mirrors dataset_store's own
-- 0001_bootstrap.sql, scoped to the `presentation` schema instead of
-- `datasets` -- a distinct database/tenant, applied through
-- `dataset_store.migrate.apply_pending(..., bookkeeping_schema="presentation",
-- bookkeeping_table="presentation.presentation_store_schema_migrations")`.
--
-- ALTER DEFAULT PRIVILEGES only applies to objects later created by the
-- role that ran this statement (the migrator). If a future migration runs
-- as a different role, repeat this grant or use `FOR ROLE`.

CREATE SCHEMA IF NOT EXISTS presentation;

GRANT USAGE ON SCHEMA presentation TO "{{RUNTIME_DB_USER}}";

ALTER DEFAULT PRIVILEGES IN SCHEMA presentation
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{{RUNTIME_DB_USER}}";

ALTER DEFAULT PRIVILEGES IN SCHEMA presentation
    GRANT USAGE, SELECT ON SEQUENCES TO "{{RUNTIME_DB_USER}}";
