# Postgres (`dataset_store`)

`dataset_store` reads and writes the shared `dfo-dev-platform-postgres`
instance's `datasets` database. `dfo-infra` owns the instance, the database,
and the two Postgres roles (a runtime role authenticated over IAM, and the
built-in `dfo-dev-datasets-migrator`). `dfo-infra`'s
`infra/dfo-dev/platform/README.md` and `infra/dfo-dev/dcp/dfo_datasets.tf`
are the source of truth for that provisioning. This repo owns the code that
reaches those roles and the migrations that grant against them.

## Environment variables

`ConnectionSettings.from_env()` reads these:

| Variable                                 | Default     | Used by                                                                                            |
| ---------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| `DATASET_STORE_INSTANCE_CONNECTION_NAME` | none        | routes through the Cloud SQL Connector instead of a direct host/port                               |
| `DATASET_STORE_HOST`                     | `127.0.0.1` | direct connection (local dev)                                                                      |
| `DATASET_STORE_PORT`                     | `5432`      | direct connection                                                                                  |
| `DATASET_STORE_DATABASE`                 | `datasets`  | both                                                                                               |
| `DATASET_STORE_RUNTIME_DB_USER`          | none        | runtime role (IAM auth over the connector, plain TCP for a local dry run)                          |
| `DATASET_STORE_RUNTIME_DB_PASSWORD`      | none        | runtime role's plain-TCP password (local dry run only, ignored once the connector routes the call) |
| `DATASET_STORE_POOL_SIZE`                | `10`        | `ConnectionPool` size                                                                              |
| `DATASET_STORE_CONNECT_TIMEOUT`          | `30`        | seconds a single connect attempt may take, over either path                                        |
| `DATASET_STORE_MIGRATOR_DB_USER`         | none        | migrator role                                                                                      |
| `DATASET_STORE_MIGRATOR_DB_PASSWORD`     | none        | migrator role                                                                                      |

`python -m dataset_store.migrate` reads the migrator vars, plus
`DATASET_STORE_RUNTIME_DB_USER`. Migrations grant the runtime role its
privileges, so the runtime user's name has to be available even though
migrations never authenticate as it. `python -m dataset_store.smoke` reads the
runtime vars, connecting over IAM auth once
`DATASET_STORE_INSTANCE_CONNECTION_NAME` is set, or plain TCP with
`DATASET_STORE_RUNTIME_DB_PASSWORD` otherwise. See `.env.example` for the
block to copy into `.env`.

## Mounting the Datasets HTTP surface

`DFO_DATASETS_SURFACE` (default: unset, off) is read by
`dfo_apis.app.deployed_surfaces()`. `dataset_store` does not read it directly.
Set it to `1` or `true` (case-insensitive) to mount the Datasets API's
register door (`POST /api/datasets/register`) and mint door
(`POST /api/datasets`) alongside Search on the deployed service. The register
door's lifespan builds a live registry over `dataset_store.shared_pool()`, so
turning this on requires `DATASET_STORE_RUNTIME_DB_USER` (and the migrations
under `src/dataset_store/migrations/`) to already be provisioned.

The mint door's own lifespan additionally requires `DATASETS_GRAPH_BASE_URL`,
`DATASETS_SDMX_BASE_URL`, `DATASETS_ARTIFACT_BUCKET_WORKING`, and
`DATASETS_ARTIFACT_BUCKET_PINNED` to be set. It builds its own
`GraphClient`/`SdmxClient`/`LiveArtifactStore` from them, with no safe
default for any of the four (see `datasets_api.config.DatasetsSettings`).
`DATASETS_GRAPH_TOKEN` is optional, sent as `x-graph-token` to the graph and
SDMX endpoints (same WAF rationale as Search's own `SEARCH_GRAPH_TOKEN`).

`DFO_PUBLIC_SITE_URL` is required by the Datasets surface, the Datasets worker,
and the Presentation components door: a bare `http` or `https` origin such as
`https://developmentfinance.org`, which download bundles link to. Each of the
three refuses to start without it. The Presentation frames door does not read
it. `DFO_SITE_DATASET_LINKS` (default `false`) is optional and adds the
website's dataset page to standalone dataset bundles
(see `dfo_platform.config.PublicSiteSettings`).

A deployment that hasn't provisioned every one of these must leave
`DFO_DATASETS_SURFACE` off, or the service fails to serve any request once
either door's lifespan runs.

## Connection budget

`dfo-dev-platform-postgres` runs `max_connections=50`. About 5 are reserved
for migrations and Postgres's own built-in users, leaving roughly 45 to split
across the tenants on the instance (`website`, `datasets`, `analysis`, and
`presentation`). The `dfo-dev-apis` Cloud Run service runs at most two
instances (`autoscaling.knative.dev/maxScale`). `dataset_store`'s own share of
the budget is the per-instance pool size (`DATASET_STORE_POOL_SIZE`, default
`10`) times that instance count, 20 connections at most. Raising the pool size
multiplies against `maxScale`. `maxScale` is set in dfo-infra and moves
independently of this document. Confirm the live value with:

```bash
gcloud run services describe dfo-dev-apis --region northamerica-northeast1 --project dev-fin-observatory --format=yaml
```

and reading `autoscaling.knative.dev/maxScale` from the output.

## Local development

The only working local path is a throwaway Postgres container:

```bash
docker run --rm -p 5432:5432 -e POSTGRES_PASSWORD=<x> postgres:18
```

The container starts with no `datasets` database and no runtime role. Create
both before migrating, using the same placeholder name and password you set
for `DATASET_STORE_RUNTIME_DB_USER`/`DATASET_STORE_RUNTIME_DB_PASSWORD` below.
The container enforces password auth on every role, so the runtime role needs
one to log in over plain TCP:

```bash
psql -h 127.0.0.1 -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE datasets;"
psql -h 127.0.0.1 -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE ROLE \"<runtime placeholder>\" LOGIN PASSWORD '<y>';"
```

Run `just env` first if you don't have a `.env` yet, then set:

```
DATASET_STORE_HOST=127.0.0.1
DATASET_STORE_MIGRATOR_DB_USER=postgres
DATASET_STORE_MIGRATOR_DB_PASSWORD=<x>
DATASET_STORE_RUNTIME_DB_USER=<same runtime placeholder>
DATASET_STORE_RUNTIME_DB_PASSWORD=<same y>
DATASET_STORE_WORKER_DB_USER=<same worker placeholder>
```

Then run `just migrate`, followed by `just dataset-store-smoke` to confirm
the runtime role can log in. The real dev instance is PSC-only, with no VPN,
bastion, or Cloud NAT path into the VPC, so it isn't reachable from a laptop.

## Running migrations

`just migrate` runs `python -m dataset_store.migrate up`, applying every
`*.sql` file under `src/dataset_store/migrations/` that isn't already
recorded, in filename order. Migrations are forward-only. Fix a bad one by
writing a new migration, not by rolling back. Against a local throwaway
container (see Local development above), that command is the whole
procedure.

Against the real dev instance, migrations run through two Cloud Run Jobs,
both in `northamerica-northeast1`/`dev-fin-observatory`, running as
`dfo-dev-migration-runner@dev-fin-observatory.iam.gserviceaccount.com` over
direct VPC egress (network `dfo-dev-apis-psc`, subnet
`dfo-dev-apis-psc-subnet`, mode `PRIVATE_RANGES_ONLY`), connecting over
plain TCP to `10.100.3.2:5432`:

| Job                            | Command                                   | Database       | Migrator role                                   |
| ------------------------------ | ----------------------------------------- | -------------- | ----------------------------------------------- |
| `dfo-dev-datasets-migrate`     | `python -m dataset_store.migrate up`      | `datasets`     | `BUILT_IN` user `dfo-dev-datasets-migrator`     |
| `dfo-dev-presentation-migrate` | `python -m presentation_store.migrate up` | `presentation` | `BUILT_IN` user `dfo-dev-presentation-migrator` |

Each runs the same image as the service, entrypoint overridden to its own
command above. Each job's service account holds
`roles/secretmanager.secretAccessor` on that job's own migrator password
secret only, no Cloud SQL IAM role.

The Deploy dev workflow runs both jobs on every deploy, in the order listed
above, immediately after it builds and pushes the SHA-tagged image and
before it moves the Cloud Run service onto that image: `gcloud run jobs update --image` points a job at the new image (both jobs' Terraform
definitions carry `ignore_changes` on the container image, so this doesn't
read as drift), then `gcloud run jobs execute --wait` runs it to
completion. `dataset_store.migrate` and `presentation_store.migrate` apply
only migrations not already recorded and exit 0 reporting nothing to apply
otherwise, so running both jobs on every deploy is safe whether or not the
new image carries a migration. A failed job run stops the workflow before
the service update step, leaving the service on its current image.

### Running a migration outside a deploy

Point a job at an image and execute it, the same two commands the deploy
workflow runs. Build and push the image first if it isn't already in
Artifact Registry, tagged with the full commit SHA and needing
`SPEC_KIT_TOKEN` to fetch the private `dfo-spec-kit` dependency, the same
token `just build` passes:

```bash
SPEC_KIT_TOKEN=$(gh auth token) docker buildx build \
  --platform linux/amd64 --provenance=false --sbom=false --push \
  --tag northamerica-northeast1-docker.pkg.dev/dev-fin-observatory/datacommons/dfo-apis:<full commit sha> \
  -f Dockerfile --secret id=spec_kit_token,env=SPEC_KIT_TOKEN .
```

`--provenance=false --sbom=false` are required. Without them buildx pushes
an OCI image index with an attestation manifest, and Cloud Run rejects it:
`Container manifest type 'application/vnd.oci.image.index.v1+json' must support amd64/linux`. `just build` sets the same flags for local builds but
never pushes.

```bash
gcloud run jobs update dfo-dev-datasets-migrate \
  --region=northamerica-northeast1 --project=dev-fin-observatory \
  --image=northamerica-northeast1-docker.pkg.dev/dev-fin-observatory/datacommons/dfo-apis:<full commit sha>
gcloud run jobs execute dfo-dev-datasets-migrate \
  --region=northamerica-northeast1 --project=dev-fin-observatory --wait
```

Swap `dfo-dev-datasets-migrate` for `dfo-dev-presentation-migrate` to run
the presentation job the same way.

The runner prints either `applied: <files>` or `nothing to apply`. Read the
execution's logs to see which. Point a job at a real application image every
time. A placeholder image exits 0 without touching the database, and the
execution reports success having applied nothing.

Both jobs leave their `*_INSTANCE_CONNECTION_NAME` variable unset: each
migrator authenticates with a password. Neither uses IAM, so the Cloud SQL
Python Connector that variable would route the connection through has nothing
to offer either job. And `max_retries` is `0` on both, because migrations are
forward-only and a retry would re-enter a database whose state a failed
attempt left unknown.

Confirm the datasets migrator's privileges from a session with database
access:

```sql
SELECT has_database_privilege(current_user, 'datasets', 'CREATE'),
       pg_has_role(current_user, 'cloudsqlsuperuser', 'member');
```

Both should return `true`. `dfo-dev-datasets-migrator` is a `BUILT_IN` Cloud
SQL user, auto-granted `cloudsqlsuperuser`, which is instance-wide rather
than scoped to the `datasets` database.

## Runtime IAM connection

The private managed zone `dfo-dev-apis-psc-sql` resolves the Cloud SQL
instance's PSC DNS name inside the `apis` VPC. Cloud SQL exposes that name as
`dns_name` on the instance, in the form `<uid>.<region>.sql.goog`. Read it
from there. Do not assemble it by hand, and confirm resolution with:

```bash
getent hosts <instance's PSC DNS name>
```

run from anything on the `apis` network. Both migration jobs bypass this path
and address the PSC endpoint directly over plain TCP. Neither uses the Cloud
SQL Python Connector, so a working migration run says nothing about it. The
runtime service account's IAM login through the connector has not been
exercised from the deployed service. `python -m dataset_store.smoke` against a
live deployment is the next step to confirm it.

## Smoke check

```bash
python -m dataset_store.smoke
# or
just dataset-store-smoke
```

Connects as the runtime role and runs `SELECT 1`. Requires either a local
Postgres (see above) or a working runtime IAM connection.

# Postgres (`presentation_store`)

`presentation_store` reads and writes its own `presentation` database on the
same shared `dfo-dev-platform-postgres` instance. It is a distinct tenant from
`dataset_store`'s `datasets`, never the same database under a second schema.
`dfo-infra` owns the instance, the `presentation` database, and its two
Postgres roles (an IAM-authenticated runtime role and a password-authenticated
migrator role), the same division `datasets` already has. This repo owns the
code that reaches those roles and the migrations under
`src/presentation_store/migrations/`.

## Environment variables

`PresentationStoreSettings.from_env()` reads the `PRESENTATION_STORE_*`
family, mirroring `ConnectionSettings.from_env()` above field-for-field:
`PRESENTATION_STORE_INSTANCE_CONNECTION_NAME`, `PRESENTATION_STORE_HOST`
(default `127.0.0.1`), `PRESENTATION_STORE_PORT` (default `5432`),
`PRESENTATION_STORE_DATABASE` (default `presentation`),
`PRESENTATION_STORE_RUNTIME_DB_USER`/`PRESENTATION_STORE_RUNTIME_DB_PASSWORD`,
`PRESENTATION_STORE_POOL_SIZE` (default `2`),
`PRESENTATION_STORE_CONNECT_TIMEOUT` (default `30`), and
`PRESENTATION_STORE_MIGRATOR_DB_USER`/`PRESENTATION_STORE_MIGRATOR_DB_PASSWORD`.

## Running migrations

```bash
python -m presentation_store.migrate up
# or
just migrate-presentation
```

`presentation_store.migrate.main` reuses `dataset_store.migrate.apply_pending`
directly, with no separate copy of it, parameterised with this package's own
`PresentationStoreSettings`, `src/presentation_store/migrations/`,
`bookkeeping_schema="presentation"`, and
`bookkeeping_table="presentation.presentation_store_schema_migrations"` so its
own applied-migrations ledger lives in its own schema, never in
`datasets.schema_migrations`. It is a distinct entrypoint from
`python -m dataset_store.migrate`, which connects to the `datasets` database
and applies `dataset_store/migrations/` regardless of which bookkeeping table
it's pointed at. Every migration file under
`src/presentation_store/migrations/` is forward-only, applied in filename
order, the same discipline `dataset_store`'s own migrations follow. Reads the
`PRESENTATION_STORE_MIGRATOR_DB_USER`/`PRESENTATION_STORE_MIGRATOR_DB_PASSWORD`
migrator credentials and `PRESENTATION_STORE_RUNTIME_DB_USER` (migrations
grant the runtime role its privileges, never authenticating as it), the same
division `dataset_store.migrate` draws for its own `DATASET_STORE_*` family.

Against the real dev instance, migrations run through the
`dfo-dev-presentation-migrate` Cloud Run Job, alongside
`dfo-dev-datasets-migrate`. See the table and the Deploy dev workflow
description in the `dataset_store` section above. Both jobs run the same way,
and the Deploy dev workflow drives both on every deploy. To run this job
outside a deploy, follow "Running a migration outside a deploy" above,
swapping in `dfo-dev-presentation-migrate` and
`python -m presentation_store.migrate up`.

## Connection budget

`dfo-dev-platform-postgres` runs `max_connections=50`, with roughly 45
available to split across tenants once migrations and Postgres's own built-in
users are accounted for (see the `dataset_store` section above). `presentation`
is a fourth tenant on that same budget, alongside `website`, `datasets`, and
`analysis`.

A presentation-enabled deployable holds TWO connection pools at once, not
one: `LiveFrameStore` resolves a frame by reading `datasets`' version/artifact
tables through the `dataset_store` persistence layer (`dataset_store.snapshot`), so
any process that mounts the presentation surface also opens a
`dataset_store.shared_pool()` alongside its own `presentation_store.shared_pool()`.
Per instance the dataset store pool is 10 (`DATASET_STORE_POOL_SIZE`) and the
presentation store pool is 2 (`PRESENTATION_STORE_POOL_SIZE`), 12 connections
together, multiplied by the `dfo-dev-apis` Cloud Run service's own maximum
of two instances: a presentation-enabled deployment holds at most 24
connections against the instance's shared capacity of 50. Sizing
`PRESENTATION_STORE_POOL_SIZE` without accounting for the `datasets` pool it
runs alongside undercounts a presentation-enabled deployment's real
connection footprint against that budget.

`DFO_PRESENTATION_SURFACE=1` therefore requires `DATASET_STORE_*` config to be
present even when `DFO_DATASETS_SURFACE=0`. Frame resolution's own read path
runs through `dataset_store`, independent of whether the Datasets HTTP surface
itself is mounted on that deployment.
