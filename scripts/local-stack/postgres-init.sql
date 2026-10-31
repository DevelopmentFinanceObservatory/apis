-- Runs once, when the empty Postgres container first starts. Creates the two databases and
-- the runtime and worker roles the migrations grant to. The migrator is the container's
-- superuser (POSTGRES_USER in compose.yaml). Names and passwords match local.env.
CREATE DATABASE datasets;
CREATE DATABASE presentation;
CREATE ROLE datasets_runtime LOGIN PASSWORD 'local-runtime';
CREATE ROLE datasets_worker LOGIN PASSWORD 'local-worker';
CREATE ROLE presentation_runtime LOGIN PASSWORD 'local-runtime';
