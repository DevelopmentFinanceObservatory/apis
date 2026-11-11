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
- [deploy/graph-release.md](graph-release.md) covers the graph release every serving process serves from: its layout, the builder job, its deploy check, and the rollback floor.
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
1. Points `JOB_ORIGIN_CATALOGUE` at the new image, so scheduled runs use the new code (see [deploy/origin-catalogue.md](origin-catalogue.md)).
1. Executes `JOB_ORIGIN_CATALOGUE` with its args overridden to `-m origin_catalogue.verify`, confirming a compatible catalogue is published in the new image's format root. Only when verify fails, executes the build job (`python -m origin_catalogue.job`) and verifies again; a second failure stops the run before either service moves onto the new image (see [deploy/origin-catalogue.md](origin-catalogue.md)'s "Deploy order" section).
1. Points `JOB_GRAPH_RELEASE` at the new image, so its scheduled ticks build with the new code, and executes it with its args overridden to `-m graph_release.verify`, confirming the new image opens the release the pointer names and builds its views. Only when verify fails, executes one build and verifies again; a second failure stops the run before either service moves onto the new image (see [deploy/graph-release.md](graph-release.md)).
1. Points `JOB_SEARCH_CATALOGUE` at the new image and executes it with `--wait`. Its build key includes the image's commit, so the run builds the catalogue on the new image in about 90 seconds, or exits at once when the image's format root already carries its key. Then executes it with its args overridden to `-m search.catalogue.verify`, confirming a loadable build is published at the new image's format root. A failed build or a failed verify stops the run before either service moves onto the new image (see [deploy/search-catalogue.md](search-catalogue.md)).
1. Updates the datasets worker service revision (`SERVICE_WORKER`, the async operations worker) using `gcloud run services update`.
1. Updates the apis service revision (`SERVICE_APIS`) using `gcloud run services update`.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=settle --wait`, repeating the data statements of the migrations over rows the previous image wrote between the migration job and the service move.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=backfill-descriptions --wait`, writing a description record for every version row that holds none, including those the previous image wrote before the service move. A version it cannot backfill is logged as one ERROR line and tried again on the next deploy that moves the apis service to a new image or on a `workflow_dispatch` run, as is a collapse whose base holds no record; a re-run of the workflow for a commit already live runs no step. The step fails when the run read no version and met a failure a retry can clear (an unreachable store, a timeout or an overload), so a run in which every version failed for good exits 0 and logs each one.

Every job moves to the new image before it executes, so a job still on the `gcr.io/cloudrun/hello` placeholder a fresh project starts with never runs as the placeholder. A failed job step stops the run before the service moves to the new image.

### Recipe migration after the services move

`datasets-maintenance normalise-recipes` runs by hand, once per environment, after the first deploy whose image writes a constant-price converted unit with its deflation basis (frame grammar 1.2.0, as `NGN/constant/2024/donor`). A stored recipe that pins such a unit without the basis matches no rebuilt root, so from the service move until the run its derived frame refuses `unit_not_present` on refresh. The task gives each such pin the basis its roots were deflated on and moves the cards drawing the frame to the recipe's new hash. It reads each root's basis from the conversion records the root's version stores per series, the recipe id and the series' declared roles, through `materialise.recipes.deflation_basis`, the rule the conversion applies when it labels a cell, so it reads neither the graph nor the conversion catalogue. A pin takes the basis of the roots feeding its parameter: `left_unit` and `numerator_unit` the roots under `frame_ref`, `right_unit` and `denominator_unit` those under the other side's slot, or under `frame_ref` when that slot is unset, and `unit` every root under the op-spec. It runs by hand because it exits 1 whenever it skips a recipe, and as a deploy step one recipe it can never fix would fail every later deploy. Stored card selects and series coordinates need no run: select matching and the component export read a stored constant-price unit without its basis as the frame's one cell for that currency, prices and base year (`materialise.unit_cells.stored_unit_key`), and refuse when the frame holds that unit under several bases.

Run it as an execution of the maintenance job, dry run first, with the environment's values for the three variables:

```sh
gcloud run jobs execute "$JOB_DATASETS_MAINTENANCE" --region="$REGION" --project="$PROJECT_ID" \
  --wait --args=normalise-recipes,--dry-run
gcloud run jobs execute "$JOB_DATASETS_MAINTENANCE" --region="$REGION" --project="$PROJECT_ID" \
  --wait --args=normalise-recipes
```

The dry run writes one `datasets_recipes_normalised` line and changes nothing. The line counts the `recipes` read, the card `revisions` it would move, and each recipe once under one outcome: `normalised`, `unchanged`, or a skip reason, so those counts add up to `recipes`. Each skipped recipe gets one `datasets_recipe_normalise_skipped` line naming its `spec_id`, `content_version`, `reason` and the `replay_key` its sort computes. A recipe whose sort is `unreproducible` or whose new key is `replay_key_held` is left whole under that reason. A `basis_undetermined` recipe has a pin whose roots give no basis or several, or a root whose stored version cannot tell; its line also names the pin's `parameter`, the `roots` it read as `spec_id@content_version`, and the `cause`: `root_unread`, `no_series_record`, `unknown_recipe`, `no_single_deflator_role`, `no_basis` or `several_bases`. Its pins are left as they are, and its frame needs minting again; the run still sorts its root options when they are out of order, under the `replay_key` its line names. A recipe skipped for its sort names its undetermined pin on the same line. A second run changes nothing.

Rolling back is safe before the run. After it, a migrated pin carries `deflation_basis`, which an image on a frame grammar older than 1.2.0 refuses as an unknown unit member, so every migrated recipe refuses on replay under that image. Once the task has run, roll back only to an image whose frame grammar is 1.2.0 or later.

Manual deployments trigger via `workflow_dispatch` in GitHub Actions and always deploy the
dispatched commit, whether or not it's already live, since dispatch exists to force a redeploy
(to re-run the migration jobs, for instance, or to roll back). Roll back by dispatching the older
commit rather than by moving traffic in the Cloud Run console: dispatch also points the
origin catalogue and search catalogue jobs at that commit's image, so the scheduled runs keep its format roots current (see [deploy/origin-catalogue.md](origin-catalogue.md) and [deploy/search-catalogue.md](search-catalogue.md)). An image older than `4be674b7d` cannot open a graph release; [deploy/graph-release.md](graph-release.md) gives the rollback floor and the environment older images read.

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
| `JOB_ORIGIN_CATALOGUE`     | Environment | Variable | Cloud Run Job that builds and verifies the origin catalogue (`dfo-dev-origin-catalogue` in dev).                                                                 | `just handoff apis` in infra |
| `JOB_SEARCH_CATALOGUE`     | Environment | Variable | Cloud Run Job that builds the search catalogue (`dfo-dev-search-catalogue` in dev).                                                                              | `just handoff apis` in infra |
| `JOB_GRAPH_RELEASE`        | Environment | Variable | Cloud Run Job that builds and publishes the graph release (`dfo-dev-graph-release` in dev).                                                                      | `just handoff apis` in infra |

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
