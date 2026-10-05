-- Bootstrap: the app schema and the runtime role's default privileges on
-- it, so every table created later inherits access without a per-table
-- grant in each future migration.
--
-- ALTER DEFAULT PRIVILEGES only applies to objects later created by the
-- role that ran this statement (the migrator). If a future migration runs
-- as a different role, repeat this grant or use `FOR ROLE`.

CREATE SCHEMA IF NOT EXISTS datasets;

GRANT USAGE ON SCHEMA datasets TO "{{RUNTIME_DB_USER}}";

ALTER DEFAULT PRIVILEGES IN SCHEMA datasets
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{{RUNTIME_DB_USER}}";

ALTER DEFAULT PRIVILEGES IN SCHEMA datasets
    GRANT USAGE, SELECT ON SEQUENCES TO "{{RUNTIME_DB_USER}}";
