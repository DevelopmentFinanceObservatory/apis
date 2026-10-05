# dataset_store

`dataset_store` provides the PostgreSQL persistence layer for the Datasets API. It manages database connections, connection pooling, forward-only schema migrations, transaction lifecycles, and relational persistence for dataset specifications, collections, versions, and artifact references.

## Architecture and Database Connectivity

Database access uses `pg8000` combined with the Google Cloud SQL Python Connector.

```
+-------------------------------------------------------------------------+
|                              Datasets API                               |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                              dataset_store                              |
|                                                                         |
|  +--------------------+   +--------------------+   +-----------------+  |
|  |  LiveDatasetStore  |-->|   ConnectionPool   |-->|   connect()     |  |
|  +--------------------+   +--------------------+   +-----------------+  |
+-------------------------------------------------------------|-----------+
                                                              |
                          +-----------------------------------+
                          |
                          v
         +----------------------------------+
         | Google Cloud SQL Connector (PSC) |
         |       or Direct TCP Socket       |
         +----------------------------------+
                          |
                          v
         +----------------------------------+
         |    PostgreSQL (datasets schema)  |
         +----------------------------------+
```

### Connection Paths and Authentication

Connections route through two supported mechanisms:

| Mechanism                  | Configuration                                  | Details                                                                                                                                                   |
| -------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cloud SQL Python Connector | `DATASET_STORE_INSTANCE_CONNECTION_NAME` set   | Uses Private Service Connect (`IPTypes.PSC`), lazy token refreshes (`refresh_strategy="lazy"`), and IAM database authentication (`enable_iam_auth=True`). |
| Direct TCP Socket          | `DATASET_STORE_INSTANCE_CONNECTION_NAME` unset | Connects to `DATASET_STORE_HOST` and `DATASET_STORE_PORT` using standard password authentication for local development and testing.                       |

### SQL Parameterization and Schema Qualification

SQL queries use `%s` positional placeholders. Parameter substitution operates outside single-quoted string literals:

- Quoted string literals pass through unchanged (e.g. `'a%'`).
- Placeholders outside quotes bind values positionally (`%s`).
- Literal percent characters outside quotes require doubling (`%%`).

All queries explicitly qualify table names with the `datasets.` schema prefix.

```python
with store.transaction() as tx:
    row = tx.fetch_one(
        "SELECT definition FROM datasets.spec_definition "
        "WHERE spec_id = %s AND schema_version = %s",
        (spec_id, schema_version),
    )
```

## Connection Pool Management

The `ConnectionPool` class provides a bounded, thread-safe connection pool with health checks configured for Cloud Run container lifecycles.

### Pool Lifecycle and Liveness

The pool maintains a maximum capacity configured by `DATASET_STORE_POOL_SIZE` (default: 10, accommodating one `datasets_api` instance's own definition-retrieval pool plus its mint pool plus the register door's own executor, documented in `dataset_store.connection.ConnectionSettings.from_env`'s own docstring). Connections initialize lazily on demand.

Idle connections undergo two tiers of validation:

| Idle Duration                         | Action                                                                                             |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 60 to 300 seconds                     | Executes a `SELECT 1` probe before reuse. Closes and replaces unresponsive connections.            |
| Over 300 seconds (`max_idle_seconds`) | Closes and discards the connection directly to avoid stale handles across container freeze cycles. |

### Process-Wide Shared Pool

The `dataset_store.provider` module supplies a reference-counted singleton pool:

```python
from dataset_store import close_shared_pool, shared_pool

# Acquire reference to the shared pool
pool = shared_pool()

# Release reference during service shutdown
close_shared_pool()
```

Each call to `shared_pool()` increments an internal borrower counter. Calls to `close_shared_pool()` decrement the counter. When the count reaches zero, all idle connections and active Cloud SQL Connector instances close.

### Error Classification and Connection Recycling

Exceptions raised during transaction execution are classified by `default_is_broken(exc)`:

| Exception Class                                     | Classification | Pool Action                                         |
| --------------------------------------------------- | -------------- | --------------------------------------------------- |
| `pg8000.exceptions.InterfaceError`                  | Broken         | Discard connection                                  |
| `OSError`                                           | Broken         | Discard connection                                  |
| `DatasetStoreConnectionError`                       | Broken         | Discard connection                                  |
| Application exceptions (e.g. constraint violations) | Healthy        | Roll back transaction and return connection to pool |

Failed transaction rollbacks raise `DatasetStoreConnectionError`, marking the connection as broken so the pool discards it.

## Schema Migrations

The `dataset_store.migrate` module provides forward-only database migrations. Schema alterations apply through subsequent forward migrations.

### Execution Model

Migrations are SQL files in `src/dataset_store/migrations/` executed in lexical filename order:

1. Creates the `datasets` schema if missing.
2. Creates the bookkeeping table `datasets.dataset_store_schema_migrations` tracking `version`, `applied_at`, and `content_sha256`.
3. Verifies SHA-256 digests of all previously applied files. A modified, missing, or renamed migration raises `MigrationChecksumError`.
4. Renders placeholder tokens (`{{KEY}}`) using supplied substitution values validated against `^[A-Za-z0-9@._-]+$`.
5. Executes pending files within individual transactions and records their checksums.

### Running Migrations

Migrations run through the CLI using migrator credentials:

```bash
python -m dataset_store.migrate up
```

Sibling packages reuse this migration engine for their own schemas by specifying custom bookkeeping tables:

```bash
python -m dataset_store.migrate up \
    --bookkeeping-schema presentation \
    --bookkeeping-table presentation.presentation_store_schema_migrations
```

## Relational Schema and Key Tables

Tables reside in the `datasets` schema. Privileges on future tables are established in `0001_bootstrap.sql` through `ALTER DEFAULT PRIVILEGES`.

```
+----------------------------------------------------------------------------------------------------+
|                                         datasets schema                                            |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  +--------------------+         +-----------------------+         +-----------------------------+  |
|  |   spec_registry    |<--------|    spec_definition    |         |     identity_quarantine     |  |
|  +--------------------+         +-----------------------+         +-----------------------------+  |
|            ^                                                                                       |
|            |                                                                                       |
|  +--------------------+         +-----------------------+                                          |
|  | collection_member  |-------->|  collection_registry  |                                          |
|  +--------------------+         +-----------------------+                                          |
|                                                                                                    |
|  +--------------------+         +-----------------------+         +-----------------------------+  |
|  | artifact_inventory |<--------|   artifact_location   |         |     version_revocation      |  |
|  +--------------------+         +-----------------------+         +-----------------------------+  |
|            ^                                                                     |                 |
|            |                                                                     v                 |
|  +--------------------+                                           +-----------------------------+  |
|  |  version_artifact  |------------------------------------------>|       version_record        |  |
|  +--------------------+                                           +-----------------------------+  |
|            ^                                                                     ^                 |
|            |                                                                     |                 |
|  +--------------------+                                           +-----------------------------+  |
|  | artifact_reference |-------------------------------------------|       version_member        |  |
|  +--------------------+                                           +-----------------------------+  |
|                                                                   ^                                |
|                                                                   |                                |
|                                                                   +-----------------------------+  |
|                                                                   |        subject_head         |  |
|                                                                   +-----------------------------+  |
+----------------------------------------------------------------------------------------------------+
```

### Table Definitions

| Table                          | Primary Key                                                                 | Description                                                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets.spec_registry`       | `spec_id`                                                                   | Immutable canonical spec identities, storing `identity_version`, raw canonical `identity_bytes`, and `identity_digest`.                                                                        |
| `datasets.spec_definition`     | `(spec_id, schema_version)`                                                 | Append-only spec projections storing canonical `definition` JSONB derived from `identity_bytes`.                                                                                               |
| `datasets.collection_registry` | `collection_id`                                                             | Immutable collection identities and registration metadata.                                                                                                                                     |
| `datasets.collection_member`   | `(collection_id, position)`                                                 | Ordered member specs belonging to each collection, foreign-keyed to `spec_registry`.                                                                                                           |
| `datasets.identity_quarantine` | `id`                                                                        | Audit table recording hash collisions (`id_kind`, `subject_id`, `held_bytes`, `incoming_bytes`, `trace_id`).                                                                                   |
| `datasets.artifact_inventory`  | `artifact_digest`                                                           | Content-addressed artifact metadata tracking `size_bytes` and creation timestamp.                                                                                                              |
| `datasets.artifact_location`   | `(artifact_digest, tier)`                                                   | Storage location tracking per tier (`working` or `pinned`), storing `bucket`, `object_name`, and `stored_at`.                                                                                  |
| `datasets.version_record`      | `(subject_id, content_version)`                                             | Immutable manifest records storing `subject_kind` (`spec` or `collection`), `schema_version`, and `immutable_body` JSONB.                                                                      |
| `datasets.version_member`      | `(subject_id, content_version, position)`                                   | Ordered member versions for collection version records.                                                                                                                                        |
| `datasets.version_artifact`    | `(subject_id, content_version, rendering_key)`                              | Join table linking version records to artifacts with `rendering` JSONB options and `media_type`.                                                                                               |
| `datasets.version_revocation`  | `(subject_id, content_version)`                                             | Permanent takedown and revocation audit log.                                                                                                                                                   |
| `datasets.artifact_reference`  | `(principal, artifact_digest, subject_id, content_version, reference_kind)` | Reference tracking backing pin lifecycles and garbage collection sweeps. Soft-deleted via `removed_at`.                                                                                        |
| `datasets.subject_head`        | `subject_id`                                                                | Mutable pointer to the `content_version` currently served for a subject, with `confirmed_at` and a JSON currency `signal`. Written forward-only by `datasets_api.version_store.LiveHeadStore`. |

## Configuration

Environment variables configure database parameters across entry points:

| Variable                                 | Default                 | Purpose                                                        |
| ---------------------------------------- | ----------------------- | -------------------------------------------------------------- |
| `DATASET_STORE_INSTANCE_CONNECTION_NAME` | None                    | Cloud SQL instance connection name (`project:region:instance`) |
| `DATASET_STORE_HOST`                     | `127.0.0.1`             | Database server hostname for TCP connections                   |
| `DATASET_STORE_PORT`                     | `5432`                  | Database server port                                           |
| `DATASET_STORE_DATABASE`                 | `datasets`              | Database name                                                  |
| `DATASET_STORE_RUNTIME_DB_USER`          | Required for runtime    | Database user for runtime queries (IAM or standard)            |
| `DATASET_STORE_RUNTIME_DB_PASSWORD`      | None                    | Password for runtime TCP connections                           |
| `DATASET_STORE_MIGRATOR_DB_USER`         | Required for migrations | Database user for DDL migrations                               |
| `DATASET_STORE_MIGRATOR_DB_PASSWORD`     | Required for migrations | Database password for DDL migrations                           |
| `DATASET_STORE_POOL_SIZE`                | `10`                    | Maximum concurrent connections in pool                         |
| `DATASET_STORE_CONNECT_TIMEOUT`          | `30.0`                  | Connection dial timeout in seconds                             |

## Smoke Check

Verify database connectivity for the runtime role:

```bash
python -m dataset_store.smoke
```

## Error Hierarchy

The package wraps driver and infrastructure exceptions into domain types:

```
DatasetStoreError
├── DatasetStoreConfigError        # Missing or invalid environment variables
├── DatasetStoreConnectionError    # Transport, authentication, or rollback failures
├── PoolTimeoutError               # Connection wait queue timeout
├── MigrationTemplateError         # Unsupplied template token or invalid substitution value
└── MigrationChecksumError         # Migration file hash drift or missing file
```

## Testing Strategies

Testing uses test doubles for unit tests and a live PostgreSQL suite for integration verification.

### In-Memory SQLite Double

`FakeDatasetStore` in `tests/dataset_store/conftest.py` provides an offline SQLite test double for unit tests:

- Translates `%s` positional placeholders to SQLite `?` syntax.
- Manages transaction commits and rollbacks in memory.
- Executes DDL statements outside transactions.

### Recording Double

`RecordingConnection` and `RecordingCursor` capture SQL statements and parameter bindings without executing queries, allowing direct inspection of query generation.

### Live PostgreSQL Verification

`tests/dataset_store/test_live.py` executes against a live PostgreSQL instance when `DFO_DATASET_STORE_LIVE_DSN` is set:

```bash
DFO_DATASET_STORE_LIVE_DSN="postgresql://postgres:postgres@127.0.0.1:5432/datasets" \
    pytest tests/dataset_store/test_live.py
```

Live tests verify PostgreSQL transactional DDL semantics, JSONB and BYTEA data type behaviour, unique constraints, foreign key cascades, and pg8000 parameter escaping rules.
