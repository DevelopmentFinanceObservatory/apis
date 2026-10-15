Architecture

GitHub Actions builds the `dfo-apis` container image and updates the active revision on Google Cloud Run. Service provisioning, VPC connectivity, scaling parameters, Cloud SQL instances, IAM bindings, and environment configurations are managed in the `infra` repository via Terraform.

## Overview

`apis` builds container artifacts and manages revision swaps. `infra` provisions cloud resources and infrastructure state.

```
+-------------------------------------------------------------+
|                             apis                            |
|                                                             |
|  +---------------------+        +------------------------+  |
|  |  deploy.yml         |------->| Artifact Registry      |  |
|  |  (GitHub Actions)   | Build  | dfo-apis:<commit-sha>  |  |
|  +----------+----------+        +-----------+------------+  |
+-------------|-------------------------------|---------------+
              | OIDC Auth (WIF)               |
              v                               v Revision Update
+-------------------------------------------------------------+
|                            infra                            |
|                                                             |
|  +-------------------------------------------------------+  |
|  | Cloud Run Service (SERVICE_APIS)                      |  |
|  | - Ignores container image changes in Terraform        |  |
|  | - Service accounts, scaling, env vars, secrets        |  |
|  +---------------------------+---------------------------+  |
|                              | Direct / Connector           |
|                              v                              |
|  +-------------------------------------------------------+  |
|  | Cloud SQL Postgres (dfo-dev-platform-postgres)        |  |
|  | - datasets database (managed via dataset_store)       |  |
|  | - presentation database (managed via presentation_store)|
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

Related guides:

- [deploy/postgres.md](postgres.md) covers database connectivity, IAM authentication, migrator roles, and schema migrations.
- [deploy/graph.md](graph.md) covers Graph API surface mounting, DCP authentication, and ResolveKit SQLite cache composition.
- [deploy/coverage.md](coverage.md) covers the coverage artifact: its layout, the job that builds it, and the deploy-order dependency the service has on it.
- [deploy/origin-catalogue.md](origin-catalogue.md) covers the origin catalogue: its layout, the job that builds it, its refusals, and the deploy-order dependency the services have on it.
- [deploy/search-catalogue.md](search-catalogue.md) covers the search catalogue: its layout, the job that builds and checks it, its events, and how `GET /api/search/catalogue` reads it.

## Continuous Deployment (`deploy.yml`)

`.github/workflows/deploy.yml` holds the deploy steps. Two callers choose the triggers and the GitHub Environment, so dev and production run the same steps against the names their environment's variables hold:

- `deploy-dev.yml` runs in the `dev` environment, after each successful CI run on `dev` and on `workflow_dispatch`.
- `deploy-prod.yml` runs in the `production` environment, on a push to `main` and on `workflow_dispatch`.

On a `workflow_run` trigger the workflow resolves the triggering branch's current tip as its deploy target rather than trusting the commit that triggered it. When that tip is the triggering run's own commit, this run's CI success already proves it green; otherwise the workflow queries the GitHub API for CI's newest run against the tip and proceeds only when it is `completed`/`success`, emitting a notice and deploying nothing otherwise. On a push or a dispatch the target is the commit the run is for. The workflow then reads the commit Cloud Run is currently serving with `gcloud run services describe` and diffs it against the target over `src/`, `packages/`, `pyproject.toml`, `uv.lock`, `.python-version`, `Dockerfile`, and the three deploy workflows; a target identical to the live commit, or one whose diff over those paths is empty, skips the build. A live image tag the checkout doesn't recognise as a commit deploys unconditionally, since its inputs can't be compared. On `workflow_dispatch`, the workflow still reads the live image so an unreadable service fails loudly, but skips the diff and always deploys the dispatched commit, so a dispatch can force a redeploy or a rollback regardless of what's already live. See `.github/scripts/deploy_decide.sh` for the exact decision.

Each caller has its own concurrency group (`deploy-dev`, `deploy-production`) with `queue: max`, so a run queued behind one already in progress waits rather than getting cancelled, and every queued run executes in turn: a `workflow_run` queued behind another deploy resolves `dev`'s tip for itself when its turn comes, and a queued `workflow_dispatch` still deploys the commit it was dispatched from.

### Workflow Stages

1. Resolves the deploy target: the branch's current tip on a `workflow_run` trigger, confirmed
   green through the triggering run itself or a GitHub API query; the pushed or dispatched
   commit otherwise.
1. Checks that every variable in the table below is set, before any build or job runs.
1. Authenticates to Google Cloud via Workload Identity Federation (WIF) using short-lived OIDC
   tokens.
1. Reads the live image from Cloud Run and decides whether the target differs from it over the
   image's input paths.
1. Builds the container image with BuildKit secrets and pushes it to Artifact Registry, when
   the decision is to deploy.
1. Points `JOB_DATASETS_MIGRATE` at the new image and executes it with `--wait`.
1. Points `JOB_PRESENTATION_MIGRATE` at the new image and executes it with `--wait`.
1. Points `JOB_DATASETS_MAINTENANCE` at the new image. Cloud Scheduler runs its `datasets-maintenance` tasks.
1. Points `JOB_COVERAGE` at the new image, so scheduled runs use the new code (see [deploy/coverage.md](coverage.md)).
1. Executes `JOB_COVERAGE` with its args overridden to `-m series_coverage.verify` (the command stays `python`), confirming a compatible artifact is published in the new image's coverage format root. Only when verify fails, executes the build job (`python -m series_coverage.job`) and verifies again; a second failure stops the run before either service moves onto the new image (see [deploy/coverage.md](coverage.md)'s "Deploy order" section).
1. Executes `JOB_COVERAGE` with `-m series_coverage.verify_index --record`, checking the image's code index against the published coverage build and recording its compatibility pointer. A difference stops the run before either service moves onto the new image.
1. Points `JOB_ORIGIN_CATALOGUE` at the new image, so scheduled runs use the new code (see [deploy/origin-catalogue.md](origin-catalogue.md)).
1. Executes `JOB_ORIGIN_CATALOGUE` with its args overridden to `-m origin_catalogue.verify`, confirming a compatible catalogue is published in the new image's format root. Only when verify fails, executes the build job (`python -m origin_catalogue.job`) and verifies again; a second failure stops the run before either service moves onto the new image (see [deploy/origin-catalogue.md](origin-catalogue.md)'s "Deploy order" section).
1. Points `JOB_SEARCH_CATALOGUE` at the new image without executing it. The apis service starts without a search catalogue and answers 503 on `GET /api/search/catalogue` until one loads. The job's Cloud Scheduler tick runs every 15 minutes, and its build key includes the image's commit, so the first tick after a deploy rebuilds the catalogue on the new image in about 90 seconds.
1. Updates the datasets worker service revision (`SERVICE_WORKER`, the async operations worker) using `gcloud run services update`.
1. Updates the apis service revision (`SERVICE_APIS`) using `gcloud run services update`.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=settle --wait`, repeating the data statements of the migrations over rows the previous image wrote between the migration job and the service move.
1. Executes `JOB_COVERAGE` with `-m series_coverage.record_serving_index`, recording the image's code index as the one both services run.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=backfill-descriptions --wait`, writing a description record for every version row that holds none, including those the previous image wrote before the service move. A version it cannot backfill is logged as one ERROR line and tried again on the next deploy that moves the apis service to a new image or on a `workflow_dispatch` run, as is a collapse whose base holds no record; a re-run of the workflow for a commit already live runs no step. The step fails when the run read no version and met a failure a retry can clear (an unreachable store, a timeout or an overload), so a run in which every version failed for good exits 0 and logs each one.

Every job moves to the new image before it executes, so a job still on the `gcr.io/cloudrun/hello` placeholder a fresh project starts with never runs as the placeholder. A failed job step stops the run before the service moves to the new image.

Manual deployments trigger via `workflow_dispatch` in GitHub Actions and always deploy the
dispatched commit, whether or not it's already live, since dispatch exists to force a redeploy
(to re-run the migration jobs, for instance, or to roll back). Roll back by dispatching the older
commit rather than by moving traffic in the Cloud Run console: dispatch also points the
coverage, origin catalogue and search catalogue jobs at that commit's image, so the scheduled runs keep its format roots current (see [deploy/coverage.md](coverage.md), [deploy/origin-catalogue.md](origin-catalogue.md) and [deploy/search-catalogue.md](search-catalogue.md)).

## Workload Identity Federation & Authentication

Deployments authenticate to Google Cloud using GitHub Actions OIDC tokens exchanged via a Workload Identity Pool.

### Identity Flow

1. GitHub Actions requests an OIDC token matching the configured `WIF_PROVIDER` audience.
1. Google Cloud STS validates token claims against provider attribute mappings and conditions.
1. The workflow impersonates the deployer service account (`WIF_SA`) to obtain a short-lived Google Cloud access token.
1. The access token authorizes Docker pushes to Artifact Registry and Cloud Run service updates.

### Deployment Branch Policies

The Workload Identity binding maps access to `repo:DevelopmentFinanceObservatory/apis:environment:dev` in dev and to `repo:DevelopmentFinanceObservatory/apis:environment:production` in production. GitHub restricts token minting through each environment's deployment branch policy (`dev` and `main`). Workflows from unauthorised branches fail before authentication.

## GitHub Variables & Secrets

Configuration values map to GitHub Environment secrets and variables:

| Name                       | Scope       | Type     | Description                                                                                                                                                      | Source                       |
| :------------------------- | :---------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| `WIF_PROVIDER`             | Environment | Secret   | Full path of the Workload Identity Provider resource.                                                                                                            | `just handoff apis` in infra |
| `WIF_SA`                   | Environment | Secret   | Email address of the deployer service account.                                                                                                                   | `just handoff apis` in infra |
| `PROJECT_ID`               | Environment | Variable | Google Cloud project ID (`dev-fin-observatory` in dev).                                                                                                          | `just handoff apis` in infra |
| `REGION`                   | Environment | Variable | Google Cloud region (`northamerica-northeast1` in dev, `europe-west1` in production).                                                                            | `just handoff apis` in infra |
| `AR_REPO`                  | Environment | Variable | Artifact Registry repository name (`datacommons` in dev).                                                                                                        | `just handoff apis` in infra |
| `SERVICE_APIS`             | Environment | Variable | Cloud Run apis service name.                                                                                                                                     | `just handoff apis` in infra |
| `SERVICE_WORKER`           | Environment | Variable | Cloud Run datasets worker service name (`dfo-dev-datasets-worker` in dev), the async operations worker `google_cloud_run_v2_service.datasets_worker` provisions. | `just handoff apis` in infra |
| `JOB_DATASETS_MIGRATE`     | Environment | Variable | Cloud Run Job that applies the datasets migrations (`dfo-dev-datasets-migrate` in dev).                                                                          | `just handoff apis` in infra |
| `JOB_PRESENTATION_MIGRATE` | Environment | Variable | Cloud Run Job that applies the presentation migrations (`dfo-dev-presentation-migrate` in dev).                                                                  | `just handoff apis` in infra |
| `JOB_DATASETS_MAINTENANCE` | Environment | Variable | Cloud Run Job that runs the `datasets-maintenance` tasks: the reconciliation sweep, aging terminal operations, and the deploy's settle and description backfill. | `just handoff apis` in infra |
| `JOB_COVERAGE`             | Environment | Variable | Cloud Run Job that builds and verifies the coverage artifact (`dfo-dev-coverage` in dev).                                                                        | `just handoff apis` in infra |
| `JOB_ORIGIN_CATALOGUE`     | Environment | Variable | Cloud Run Job that builds and verifies the origin catalogue (`dfo-dev-origin-catalogue` in dev).                                                                 | `just handoff apis` in infra |
| `JOB_SEARCH_CATALOGUE`     | Environment | Variable | Cloud Run Job that builds the search catalogue (`dfo-dev-search-catalogue` in dev).                                                                              | `just handoff apis` in infra |

Every environment variable lives in both the `dev` and `production` GitHub Environments under the same name. In infra, `just handoff apis` sets every environment variable and secret on `dev`, and `just ENV=dfo-prod handoff apis` sets them on `production`.

## Container Build & Image Naming

### Registry Target

Artifact Registry images use the commit SHA tag:

```text
${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/dfo-apis:<commit-sha>
```

### Dependency resolution

The application depends on the `spec-kit` repository at `DevelopmentFinanceObservatory/spec-kit`, which is public. During the multi-stage Docker build, `uv` resolves it with git over HTTPS, and the build needs no credential.

The runtime container uses `debian:bookworm-slim` with installed CA certificates and runs as an unprivileged `app` user (`UID 1000`) on port `8080`.

## Infrastructure & Persistence Boundary

`apis` and `infra` divide deployment responsibilities:

### Infrastructure (`infra`)

- Manages Cloud Run service concurrency, CPU and memory limits, ingress settings, and environment variables.
- Configures `lifecycle { ignore_changes = [template[0].spec[0].containers[0].image] }` in Terraform to preserve deployment image updates during Terraform applies.
- Provisions Cloud SQL PostgreSQL instances, private VPC peering, databases, and IAM database users.
- Grants the deployer service account `roles/run.developer`, `roles/artifactregistry.writer`, and `roles/iam.serviceAccountUser`.

### Application (`apis`)

- Builds, tests, warms ResolveKit SQLite caches, and pushes container images.
- Updates Cloud Run revisions using `gcloud run services update --image=...` after CI checks pass.
- Ships migration DDL under `src/dataset_store/migrations/` and `src/presentation_store/migrations/`; the deploy workflow drives both migration jobs on every deploy (documented in [deploy/postgres.md](postgres.md)).
