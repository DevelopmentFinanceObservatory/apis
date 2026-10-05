-- Makes the async worker's IAM database user a member of the runtime
-- role, so it inherits every privilege already granted to
-- `{{RUNTIME_DB_USER}}`: 0001_bootstrap.sql's `USAGE` on the schema, its
-- `SELECT, INSERT, UPDATE, DELETE` on tables and `USAGE, SELECT` on
-- sequences, and anything a future migration grants the runtime role via
-- `ALTER DEFAULT PRIVILEGES` -- without this file (or any future one)
-- needing its own copy of every grant for a second substitution.
--
-- Postgres role membership is checked with `INHERIT` by default (the
-- standard `CREATE ROLE` default, which Cloud SQL IAM users get unless
-- explicitly denied), so the worker uses the inherited privileges
-- directly -- no `SET ROLE` required.

GRANT "{{RUNTIME_DB_USER}}" TO "{{WORKER_DB_USER}}";
