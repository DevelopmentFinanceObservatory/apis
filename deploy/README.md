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

## Dev parser replay

`scripts/replay_parser_experiment.py` runs a paired comparison through the normal `/api/search/resolve` door of a tagged, no-traffic `dfo-dev-apis` revision. Each input runs once through each of three arms, back to back, in a deterministic random order: `flash-lite` (`gemini-3.5-flash-lite`), `ministral` (`ministral-3b-2512`) and `mistral-small` (`mistral-small-2603`). Both Mistral arms use the same extraction instruction. After five consecutive arms fail without an HTTP status, such as a refused connection when the proxy dies, the runner stops sending, keeps the finished results and exits non-zero. The runner records an opaque request ID for Cloud Logging joins, the HTTP status and problem code, the accepted arm header, time to the first interpretation event, time to the `response` event, the terminal event, response status, spec IDs, and a narrow structured interpretation summary. Treat results as private data because interpretation fields can repeat words from the input. Compare arms on `response_ms`, the time to the `response` event. `done_ms` also holds the wait for the served extraction's agreement, up to `SEARCH_LLM_SECOND_SAMPLE_SETTLE_S` after its first sample was read. The application cache is bypassed for every replay request. Provider prefix caching stays on and is measured.

The replay revision needs these dev-only settings:

| Setting                         | Value                                                                        |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `SEARCH_PARSER_REPLAY_ENABLED`  | `true`                                                                       |
| `SEARCH_PARSER_REPLAY_REVISION` | The revision's name, `dfo-dev-apis-` followed by its suffix                  |
| `SEARCH_PARSER_REPLAY_KEY`      | The `dfo-dev-parser-replay-key` secret, also exported locally under the name |
| `MISTRAL_API_KEY`               | The `dfo-dev-mistral-api-key` secret                                         |

A revision's settings cannot change after it is created. An update creates a new revision with a generated name, which the gate then refuses. Create the revision and its settings in one command, from the image the current dev revision runs:

```bash
IMAGE=$(gcloud run services describe dfo-dev-apis --region northamerica-northeast1 \
  --format='value(spec.template.spec.containers[0].image)')
gcloud run deploy dfo-dev-apis --region northamerica-northeast1 \
  --image "$IMAGE" --no-traffic --tag parser-replay --revision-suffix parser-replay-01 \
  --update-env-vars SEARCH_PARSER_REPLAY_ENABLED=true,SEARCH_PARSER_REPLAY_REVISION=dfo-dev-apis-parser-replay-01 \
  --update-secrets MISTRAL_API_KEY=dfo-dev-mistral-api-key:latest,SEARCH_PARSER_REPLAY_KEY=dfo-dev-parser-replay-key:latest
```

The gate checks `K_SERVICE` and `K_REVISION`, which Cloud Run sets, against `dfo-dev-apis` and `SEARCH_PARSER_REPLAY_REVISION`. `--no-traffic` pins dev's traffic to the revision serving before the deploy, by name, so dev keeps serving the same image. A later dev revision inherits the settings but has another name, so it refuses replay requests. An infra apply that resets the service's traffic drops the tag. Re-add it with `gcloud run services update-traffic dfo-dev-apis --update-tags parser-replay=dfo-dev-apis-parser-replay-01 --region northamerica-northeast1`.

The runner accepts JSONL or a JSON array with `id`, `query`, and optional `expected` fields. Keep input files outside source control. `--write-synthetic` writes labelled synthetic cases:

```bash
uv run --extra engine --extra eval --extra datasets --extra graph \
  python scripts/replay_parser_experiment.py \
  --write-synthetic /private/tmp/dfo-parser-synthetic.jsonl
```

The runner accepts only `http` loopback URLs. Open a proxy to the tagged revision in one terminal. The proxy keeps the credentials fresh for the whole run.

```bash
gcloud run services proxy dfo-dev-apis --tag parser-replay \
  --region northamerica-northeast1 --port 8080
```

Run the replay in another terminal.

```bash
export SEARCH_PARSER_REPLAY_KEY=...
uv run --extra engine --extra eval --extra datasets --extra graph \
  python scripts/replay_parser_experiment.py \
  --url http://127.0.0.1:8080 \
  --cases /private/path/cases.jsonl \
  --output /private/path/results.jsonl \
  --concurrency 3
```

`--concurrency` bounds the pairs in flight. A pair's two arms share one slot. Use `--concurrency 1` and `--pace-ms` for a sparse pass. `--arms ministral` replays only the named arms, for a rerun that compares with earlier results for the others. The runner refuses an `--output` path that exists, so a rerun cannot overwrite earlier results. It creates the file with mode `0600` and adds one line per arm as each pair completes, so an interrupted run keeps the finished pairs. A gate refusal appears as `status_code` 403 with its `problem_code`.

Join each result's `request_id` to `parser_replay.replay_id` in the `resolve` log. `parser_replay.calls` lists the replay's extraction calls only; the tokens of every other stage are in the line's `llm_usage`. With `--stream-probe`, read `first_content_ms` from the call whose `attempt` is `first` and whose `phase` equals `parser_replay.served_phase`. The other calls are agreement samples, or parses of the whole query that the door discards once the exclusion read blanks part of it. Cost an arm from its calls' `usage`, both agreement samples and Gemini's `thought_tokens` included, and report calls with an `error` and no `usage` separately.

To remove the replay, drop the settings, then return traffic to the latest revision and drop the tag:

```bash
gcloud run services update dfo-dev-apis --region northamerica-northeast1 \
  --remove-env-vars SEARCH_PARSER_REPLAY_ENABLED,SEARCH_PARSER_REPLAY_REVISION \
  --remove-secrets MISTRAL_API_KEY,SEARCH_PARSER_REPLAY_KEY
gcloud run services update-traffic dfo-dev-apis --to-latest --remove-tags parser-replay \
  --region northamerica-northeast1
```

## Continuous Deployment (`deploy.yml`)

`.github/workflows/deploy.yml` holds the deploy steps. Two callers choose the triggers and the GitHub Environment, so dev and production run the same steps against the names their environment's variables hold:

- `deploy-dev.yml` runs in the `dev` environment, after each successful CI run on `dev` and on `workflow_dispatch`.
- `deploy-prod.yml` runs in the `production` environment, on a push to `main` and on `workflow_dispatch`.

On a `workflow_run` trigger the workflow resolves the triggering branch's current tip as its deploy target rather than trusting the commit that triggered it. When that tip is the triggering run's own commit, this run's CI success already proves it green; otherwise the workflow queries the GitHub API for CI's newest run against the tip and proceeds only when it is `completed`/`success`, emitting a notice and deploying nothing otherwise. On a push or a dispatch the target is the commit the run is for. The workflow then asks whether the revision serving the apis service runs the target's code. A revision records its image by digest, so the serving commit comes from the service template, which describes the latest created revision, and it is the serving commit only in the state a finished deploy leaves: traffic follows a ready latest revision and no revision carries the `candidate` tag. In that state the commit is diffed against the target over `src/`, `packages/`, `pyproject.toml`, `uv.lock`, `.python-version`, `Dockerfile`, and the three deploy workflows, and a target identical to it, or one whose diff over those paths is empty, skips every later step. Any other state (a deploy that stopped part-way, a revision that never started, traffic moved by hand) is unknown, and the target deploys. A serving image tag the checkout doesn't recognise as a commit deploys, since its inputs can't be compared. On `workflow_dispatch`, the workflow still reads the service so an unreadable service fails loudly, and deploys the dispatched commit whatever the diff, so a dispatch can force a redeploy or a rollback. See `.github/scripts/deploy_decide.sh` for the exact decision.

Each caller has its own concurrency group (`deploy-dev`, `deploy-production`) with `queue: max`, so a run queued behind one already in progress waits rather than getting cancelled, and every queued run executes in turn: a `workflow_run` queued behind another deploy resolves `dev`'s tip for itself when its turn comes, and a queued `workflow_dispatch` still deploys the commit it was dispatched from.

### Workflow Stages

1. Resolves the deploy target: the branch's current tip on a `workflow_run` trigger, confirmed
   green through the triggering run itself or a GitHub API query; the pushed or dispatched
   commit otherwise.
1. Checks that every variable in the table below is set, before any build or job runs.
1. Authenticates to Google Cloud via Workload Identity Federation (WIF) using short-lived OIDC
   tokens.
1. Reads the apis service and decides whether the target needs a deploy (`.github/scripts/deploy_decide.sh`). A service that a finished deploy left serving the target's code needs none, unless the scheduled `JOB_ORIGIN_CATALOGUE` runs another image than the serving revision's, which a run that stopped or failed after the traffic move leaves.
1. Points `JOB_ORIGIN_CATALOGUE` at the image of the revisions `status.traffic` gives more than 0 percent, when it runs another (`.github/scripts/deploy_reconcile.sh`), before anything else moves. A job writes only its own image's format root, so this keeps the job refreshing the root the serving revision reads, even when the run stops early. Traffic split across revisions running different images fails the run with nothing moved.
1. Builds the container image with BuildKit secrets and pushes it to Artifact Registry, when
   the decision is to deploy.
1. Reads the pushed image index's digest and its linux/amd64 manifest's, while the registry
   login is fresh, for the stage step's image check.
1. Points `JOB_DATASETS_MIGRATE` at the new image and executes it with `--wait`.
1. Points `JOB_PRESENTATION_MIGRATE` at the new image and executes it with `--wait`.
1. Points `JOB_DATASETS_MAINTENANCE` at the new image. Cloud Scheduler runs its `datasets-maintenance` tasks.
1. Points `JOB_ORIGIN_CATALOGUE_CANDIDATE`, an unscheduled copy of the origin catalogue job, at the new image. The scheduled `JOB_ORIGIN_CATALOGUE` stays on the serving image until traffic moves (see [deploy/origin-catalogue.md](origin-catalogue.md)).
1. Executes `JOB_ORIGIN_CATALOGUE_CANDIDATE` with its args overridden to `-m origin_catalogue.verify`, confirming a compatible catalogue is published in the new image's format root. Only when verify fails, executes the build job (`python -m origin_catalogue.job`) on the candidate job and verifies again; a second failure stops the run before either service or the scheduled job moves onto the new image (see [deploy/origin-catalogue.md](origin-catalogue.md)'s "Deploy order" section).
1. Points `JOB_GRAPH_RELEASE` at the new image, so its scheduled ticks build with the new code, and executes it with its args overridden to `-m graph_release.verify`, confirming the new image opens the release the pointer names and builds its views. Only when verify fails, executes one build and verifies again; a second failure stops the run before either service moves onto the new image (see [deploy/graph-release.md](graph-release.md)).
1. Points `JOB_SEARCH_CATALOGUE` at the new image and executes it with `--wait`. Its build key includes the image's commit, so the run builds the catalogue on the new image in about 90 seconds, or exits at once when the image's format root already carries its key. Then executes it with its args overridden to `-m search.catalogue.verify`, confirming a loadable build is published at the new image's format root. A failed build or a failed verify stops the run before either service moves onto the new image (see [deploy/search-catalogue.md](search-catalogue.md)).
1. Stages the new apis revision (`SERVICE_APIS`) with `gcloud run services update --no-traffic`, which pins traffic to the revision serving until then, reads the revision the update created, and fails unless that revision runs the image the build pushed (the index's digest or its linux/amd64 manifest's, since a revision records its image by digest). It then tags the revision `candidate` with `gcloud run services update-traffic --update-tags` and records its name. The revision takes no traffic and has a tagged URL of its own, and the steps after it act on that name alone. Tagging through `update-traffic` leaves the template revision unnamed. `--tag` on the update would name it, and the infra repository's Terraform would clear that name, creating a revision, on every later apply.
1. Probes the named revision at the URL of its `candidate` tag (`.github/scripts/deploy_warm.sh`). It mints an ID token for the service URL as the deployer, decodes the token's `aud` claim and fails unless it is the service URL that describe returned, and calls `/api/skills`, `/api/graph/schema` and `/api/graph/schema/terms/un_system`, whose handlers wait on the skills index, the schema document and the UN terms body that the revision builds beside serving, and `POST /api/search/stated-scope`. Cloud Run throttles the CPU of an instance with no request in flight, so without these calls the builds stall until the first readers pay for them. Each call logs its HTTP status, so a 401 or 403 after a matching `aud` shows that Cloud Run refused the token on the tagged URL or the deployer lacks `roles/run.invoker`. Each route retries a 502, 503, 504 or 429, or no answer, for up to 300 seconds of its own. A route that never answers at all within its deadline is an error everywhere and ends the probe. A 404 with `cause.code` `origin_not_served` from the UN terms route (a deployment with `ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED` off) skips that route with a log line. Any other status is an error, and so is a revision with no tagged URL. On `dev` a route still slow at its deadline, a 401 or 403, or an unreadable service URL or token warns. In `production` (`REQUIRE_PROBE`) each of those is an error, so every route must answer 200 before traffic moves.
1. Fails the deploy unless the probe status lets traffic move: `warm` everywhere, and on `dev` also `slow`, `forbidden` or `unavailable`. The previous revision still serves. The migrations and every job but the scheduled origin catalogue job are already on the new image at this point, and stay there while the previous apis revision serves, until the next run deploys again.
1. Updates the datasets worker service revision (`SERVICE_WORKER`, the async operations worker) using `gcloud run services update`. The probe calls no route that enqueues an operation or reaches the worker, so the worker moves only once the probe passes, and a failed probe leaves it on the serving image.
1. Moves all traffic to the probed revision by name with `gcloud run services update-traffic --to-revisions=<revision>=100`.
1. Points `JOB_ORIGIN_CATALOGUE` at the staged revision's image by digest, so its scheduled runs refresh the format root the apis service reads from this point on. A job writes only its own image's format root, so the job moves only once traffic has moved. The step continues on error, so a failure lets the steps after it run and fails the run at the end.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=settle --wait`, repeating the data statements of the migrations over rows the previous image wrote between the migration job and the service move.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=rewrite-warnings --wait`, rewriting the stored warnings that name refs in an old shape, in the `warnings` column and in `immutable_body`, each from its own value, touching only the rows the rewrite changes. It scans `version_record` on every run, 500 rows by key to a transaction, so a failed run keeps the pages it committed, and the next run, which scans from the start, finds only the rows still in an old shape. It leaves the member sentences and any warning the rewrite cannot parse, which stays as stored, and logs the rows it rewrote. The next step's summary counts the rows still in an old shape. The step continues on error, so a failure lets the steps after it run and fails the run at the end.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=backfill-warnings --wait`, rewriting each stored `coverage_below_scope` warning that still names members in the old member sentence, the one old shape `rewrite-warnings` leaves, since splitting its members needs graph reads. The step fails when a row failed for a cause a retry can clear, lost to a concurrent write, or was left when the task's 45-minute budget ran out; the next deploy carries on from the rows that remain. A row whose rewrite fails with a defect is logged as one ERROR line, is selected again by the next run and fails the step on every deploy until the code is fixed. A row refused for good is logged as one WARNING line on every run and sets no exit code. The summary line counts the rows still holding any old shape, `legacy_warning_refs`, which sets no exit code; a count that fails or runs past 25 s, five sixths of the store's 30-second socket bound, is logged at ERROR with a null count and sets no exit code either. The step continues on error, so a failure lets the steps after it run and fails the run at the end.
1. Executes `JOB_DATASETS_MAINTENANCE` with `--args=backfill-descriptions --wait`, writing a description record for every version row that holds none, including those the previous image wrote before the service move. A version it cannot backfill is logged as one ERROR line and tried again on the next deploy that moves the apis service to a new image or on a `workflow_dispatch` run, as is a collapse whose base holds no record; a re-run of a finished release runs no step. The step fails when the run read no version and met a failure a retry can clear (an unreachable store, a timeout or an overload), so a run in which every version failed for good exits 0 and logs each one.
1. Makes traffic follow the latest revision again and removes the `candidate` tag, in one `gcloud run services update-traffic --to-latest --remove-tags=candidate`, so a revision `infra` creates for a configuration change serves at once.
1. Fails the run unless traffic follows the latest revision at 100 percent and that revision runs the released image. An `infra` apply that updates the apis service sends the traffic and image its plan refreshed, so one that overlaps the deploy can pin traffic again or put the previous image back, and the check reports it and leaves traffic as it found it. A revision `infra` creates for a configuration change after the move runs the released image and passes.
1. Fails the run when the scheduled job's update or either warning step failed, naming each one that did. All three continue on error, so the description backfill, whose records read series statistics that leave the working tier, the follow step and the release check run first. Traffic then follows the latest revision without the `candidate` tag. After a failed warning step a rerun of the same commit deploys nothing, and the next deploy or a `workflow_dispatch` run runs the warning steps again from the rows that remain. After a failed scheduled job update the job runs another image than the serving one, so a rerun of the same commit deploys and points the job at the serving image first.
1. After a failure in settle, the description backfill or the step that makes traffic follow the latest revision again, makes traffic follow the latest revision with `gcloud run services update-traffic --to-latest`, keeping the `candidate` tag so the next run deploys. A failed scheduled job update or warning step fails the run only after the follow step succeeded, so it does not trigger this step. A cancelled run skips this step, since the cancellation can land before Cloud Run received the traffic move, and following the latest revision would then serve the candidate while the scheduled job writes the previous format root. Traffic stays pinned with the tag in place, and the next run deploys and points the scheduled job at whatever serves before anything else moves.

Every job the deploy executes moves to the new image before it executes, so none of them runs as the `gcr.io/cloudrun/hello` placeholder a fresh project starts with. The scheduled origin catalogue job, which Cloud Scheduler runs every 15 minutes, stays on the placeholder on a fresh project until a deploy passes the traffic move. A failed job step stops the run before the service moves to the new image.

A run that fails anywhere but the scheduled job's update or a warning step leaves one of three states, and the next run, a rerun included, deploys again from the build in each: a failure before the stage step leaves the previous commit in the template, a failure from the stage step to the release check leaves traffic pinned by name or the `candidate` tag in place, and a failed release check leaves another image serving. That costs a full deploy: about 9 minutes on `dev` and 12 in `production`, most of it the search catalogue run, the two service updates and the build. A failure in settle, the description backfill or the step that makes traffic follow the latest revision again does that before the run ends, and keeps the tag. A run that fails only in a warning step leaves the service released, with no tag, so a rerun deploys nothing. A run that fails in the scheduled job's update leaves the service released and the job on the previous image, so a rerun deploys and moves the job first. After a failed release check, traffic stays as the check found it, and the next run deploys. A failure before the move leaves traffic pinned to the previous revision, so an `infra` change to the apis service serves only after the next deploy succeeds.

On `dev` a `slow`, `forbidden` or `unavailable` probe still moves traffic, so only `production` guarantees that a revision that answers an error or does not answer never serves.

### Keeping a manual rollback

Every push deploys its commit while the environment's caller workflow is enabled, whatever traffic someone set by hand. The caller is `deploy-prod.yml` for `production` and `deploy-dev.yml` for `dev`. Disabling a workflow stops new runs from being triggered, but it does not cancel a run that has not completed (`pending` in the concurrency queue, `waiting` for approval, `queued` or `in_progress`), and such a run replaces the rollback when it moves traffic. To keep a rollback in `production`, disable the caller, then cancel or wait for every run the listing below prints, then roll back:

```sh
gh workflow disable deploy-prod.yml -R DevelopmentFinanceObservatory/apis
gh run list -R DevelopmentFinanceObservatory/apis --workflow deploy-prod.yml --json databaseId,status -q '.[] | select(.status != "completed")'
gh run cancel <databaseId> -R DevelopmentFinanceObservatory/apis
gcloud run services update-traffic <service> --to-revisions=<revision>=100 --region=<region> --project=<project>
```

A merge while the caller is disabled starts no run, and enabling the caller starts none either. When a fix is merged, enable the caller and dispatch it, which deploys whatever traffic serves:

```sh
gh workflow enable deploy-prod.yml -R DevelopmentFinanceObservatory/apis
gh workflow run deploy-prod.yml -R DevelopmentFinanceObservatory/apis --ref main
```

For `dev`, use `deploy-dev.yml` in the same commands, and `--ref dev` in the dispatch.

### Recipe migration after the services move

`datasets-maintenance normalise-recipes` runs by hand, once per environment, after the first deploy whose image writes a constant-price converted unit with its deflation basis (frame grammar 1.2.0, as `NGN/constant/2024/donor`). A stored recipe that pins such a unit without the basis matches no rebuilt root, so from the service move until the run its derived frame refuses `unit_not_present` on refresh. The task gives each such pin the basis its roots were deflated on and moves the cards drawing the frame to the recipe's new hash. It reads each root's basis from the conversion records the root's version stores per series, the recipe id and the series' declared roles, through `materialise.recipes.deflation_basis`, the rule the conversion applies when it labels a cell, so it reads neither the graph nor the conversion catalogue. A pin takes the basis of the roots feeding its parameter: `left_unit` and `numerator_unit` the roots under `frame_ref`, `right_unit` and `denominator_unit` those under the other side's slot, or under `frame_ref` when that slot is unset, and `unit` every root under the op-spec. It runs by hand because it exits 1 whenever it skips a recipe, and as a deploy step one recipe it can never fix would fail every later deploy. Stored card selects and series coordinates need no run. Select matching and the component export bind a constant-price unit named without its basis, alone or as a side of a composed unit, to the frame's one cell for that currency, prices and base year (`materialise.unit_cells.bind_unit_key`), and refuse when the frame holds that unit under several bases.

Run it as an execution of the maintenance job, dry run first, with the environment's values for the three variables:

```sh
gcloud run jobs execute "$JOB_DATASETS_MAINTENANCE" --region="$REGION" --project="$PROJECT_ID" \
  --wait --args=normalise-recipes,--dry-run
gcloud run jobs execute "$JOB_DATASETS_MAINTENANCE" --region="$REGION" --project="$PROJECT_ID" \
  --wait --args=normalise-recipes
```

The dry run writes one `datasets_recipes_normalised` line and changes nothing. The line counts the `recipes` read, the card `revisions` it would move, and each recipe once under one outcome: `normalised`, `unchanged`, or a skip reason, so those counts add up to `recipes`. Each skipped recipe gets one `datasets_recipe_normalise_skipped` line naming its `spec_id`, `content_version`, `reason` and the `replay_key` its sort computes. A recipe whose sort is `unreproducible` or whose new key is `replay_key_held` is left whole under that reason. A `basis_undetermined` recipe has a pin whose roots give no basis or several, or a root whose stored version cannot tell; its line also names the pin's `parameter`, the `roots` it read as `spec_id@content_version`, and the `cause`: `root_unread`, `no_series_record`, `unknown_recipe`, `no_single_deflator_role`, `no_basis` or `several_bases`. Its pins are left as they are, and its frame needs minting again; the run still sorts its root options when they are out of order, under the `replay_key` its line names. A recipe skipped for its sort names its undetermined pin on the same line. A second run changes nothing.

Rolling back is safe before the run. After it, a migrated pin carries `deflation_basis`, which an image on a frame grammar older than 1.2.0 refuses as an unknown unit member, so every migrated recipe refuses on replay under that image. Once the task has run, roll back only to an image whose frame grammar is 1.2.0 or later.

### Party-kind coverage after the services move

`datasets-maintenance backfill-party-kind-coverage` runs by hand, once per environment, after the first deploy whose image counts a party in an output profile's `party_kind` coverage where every layout fills its cells on the role column's axis: a long download, which reads each row for its own dataset, and a pivot, which reads the party once over the datasets the frame serves it from. A profile minted by an earlier image counted a party for no dataset and on any axis, so the options document can misstate a party column's kind coverage for a party that is a total in one dataset and a party in another. The task reads each version's stored frame, counts its `party_kind` coverage again under the served graph release's code index, and rewrites the profile's `party_kind` entries in the version row's `output_profile` column and in its stored body where the count differs, each entry in its place. The write takes the version's link lock, writes only over the profile it counted from, and sets the body's `output_profile` alone, so it keeps any other writer's change to the profile or the body. A new entry takes its place by the profile's own role columns, the order a mint writes them in. It reads no graph, so a party it names as missing carries the label the profile already holds for it, or none.

Run it as an execution of the maintenance job, dry run first, with the environment's values for the three variables:

```sh
gcloud run jobs execute "$JOB_DATASETS_MAINTENANCE" --region="$REGION" --project="$PROJECT_ID" \
  --wait --args=backfill-party-kind-coverage,--dry-run
gcloud run jobs execute "$JOB_DATASETS_MAINTENANCE" --region="$REGION" --project="$PROJECT_ID" \
  --wait --args=backfill-party-kind-coverage
```

Each run writes one `datasets_party_kind_backfilled` line counting the versions `refreshed` (on a dry run, those it would rewrite), `unchanged`, `frame_gone` (no tier holds the frame, so the row is left as it is), `changed_under_run` (the profile changed while it counted) and `copies_differ` (the row's `output_profile` column and its body's copy differ, which needs a hand repair), with `failed` and `dry_run`. Each row whose copies differ gets one `datasets_party_kind_backfill_copies_differ` `ERROR` line, and the run exits 1. Each version that fails gets one `datasets_party_kind_backfill_failed` line with its `subject_id`, `content_version` and traceback, and the run exits 1. A second run changes nothing. Rolling back after the run is safe: an older image reads the rewritten profile as it reads any other.

Manual deployments trigger via `workflow_dispatch` in GitHub Actions and always deploy the
dispatched commit, whether or not it's already live, since dispatch exists to force a redeploy
(to re-run the migration jobs, for instance, or to roll back). Roll back by dispatching the older
commit rather than by moving traffic in the Cloud Run console: dispatch also points the
origin catalogue and search catalogue jobs at that commit's image, so the scheduled runs keep its format roots current (see [deploy/origin-catalogue.md](origin-catalogue.md) and [deploy/search-catalogue.md](search-catalogue.md)). An image older than `4be674b7d` cannot open a graph release; [deploy/graph-release.md](graph-release.md) gives the rollback floor and the environment older images read.

### Rollback floor for account callers

An image older than the one that accepts account callers (`feat(caller): accept an account caller from the website`) breaks once the website names a caller made with an API key as `user:<digest>`, in two ways that start at different times:

- From the moment the website sends `user:` callers, its caller check answers 400 `request_invalid` to every write that names its caller with `resolve_caller`: a pin, an export prepare that mints a new export, a share create, replace or snapshot conversion, and a card create or revise. Reads and repeat exports are served as before. Analysis calls are served, but every `user:` caller counts under one shared concurrency share (`_caller_share_key` maps the refused header to `unresolved`), so a few concurrent analyses can draw 429 `caller_over_share`.
- Once a `user:` pin principal exists, its `settle`, which a dispatched rollback runs after the services move, rewrites every pin principal that is neither a `secret-sha256:` label, an `ip:` key nor a `conversation:` key into a hash label, as if it were a raw secret (an image older than `cf209a1e0` rewrites `conversation:` pins too). Each `user:` pin then stops matching its account's repeat pins, and the rewrite runs in one committed transaction with nothing to undo it from.

So no dispatched rollback may reach such an image while the website sends `user:` callers or a `user:` pin exists. If an older image is needed anyway, first set `API_KEYS_ENABLED=false` on the website and roll it onto a new revision, so it names every caller `ip:` again. Then move the services and the scheduled jobs to the older commit's image by hand, without dispatching it, so its `settle` never runs:

```sh
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/dfo-apis:$OLDER_SHA"
for job in "$JOB_SEARCH_CATALOGUE" "$JOB_GRAPH_RELEASE"; do
  gcloud run jobs update "$job" --image="$IMAGE" --region="$REGION" --project="$PROJECT_ID" --quiet
done
gcloud run services update "$SERVICE_WORKER" --image="$IMAGE" --region="$REGION" --project="$PROJECT_ID" --quiet
gcloud run services update "$SERVICE_APIS" --image="$IMAGE" --region="$REGION" --project="$PROJECT_ID" --quiet
SERVICE="$SERVICE_APIS" JOB="$JOB_ORIGIN_CATALOGUE" REGION="$REGION" PROJECT_ID="$PROJECT_ID" \
  .github/scripts/deploy_reconcile.sh
```

The last command points the scheduled origin catalogue job at the digest the older revision serves, which Cloud Run records for the service. Setting the job by tag (`dfo-apis:$OLDER_SHA`) would leave an image string that never equals the serving digest, so the next push to the branch would see the job behind, deploy, and undo the rollback. Run the script from a checkout of this repository after the traffic move, so the serving revision is the older one. This keeps the dispatch's order (the search catalogue and graph release jobs, then the worker, then the apis service, then the scheduled origin catalogue job) and its invariant that the worker and the scheduled jobs run the apis service's image. The datasets maintenance job and the two migration jobs stay on the newer image: the maintenance job's scheduled tasks run there as they do between a deploy's steps, and leaving it there is what keeps the older `settle` from ever running. The recipe skips the dispatch's catalogue and graph-release steps and relies on the older format roots still holding their last build (see [deploy/origin-catalogue.md](origin-catalogue.md) and [deploy/graph-release.md](graph-release.md)). Dispatch the current commit again before turning keys back on.

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

| Name                             | Scope       | Type     | Description                                                                                                                                                                                                         | Source                       |
| :------------------------------- | :---------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------- |
| `WIF_PROVIDER`                   | Environment | Secret   | Full path of the Workload Identity Provider resource.                                                                                                                                                               | `just handoff apis` in infra |
| `WIF_SA`                         | Environment | Secret   | Email address of the deployer service account.                                                                                                                                                                      | `just handoff apis` in infra |
| `PROJECT_ID`                     | Environment | Variable | Google Cloud project ID (`dev-fin-observatory` in dev).                                                                                                                                                             | `just handoff apis` in infra |
| `REGION`                         | Environment | Variable | Google Cloud region (`northamerica-northeast1` in dev, `europe-west1` in production).                                                                                                                               | `just handoff apis` in infra |
| `AR_REPO`                        | Environment | Variable | Artifact Registry repository name (`datacommons` in dev).                                                                                                                                                           | `just handoff apis` in infra |
| `SERVICE_APIS`                   | Environment | Variable | Cloud Run apis service name.                                                                                                                                                                                        | `just handoff apis` in infra |
| `SERVICE_WORKER`                 | Environment | Variable | Cloud Run datasets worker service name (`dfo-dev-datasets-worker` in dev), the async operations worker `google_cloud_run_v2_service.datasets_worker` provisions.                                                    | `just handoff apis` in infra |
| `JOB_DATASETS_MIGRATE`           | Environment | Variable | Cloud Run Job that applies the datasets migrations (`dfo-dev-datasets-migrate` in dev).                                                                                                                             | `just handoff apis` in infra |
| `JOB_PRESENTATION_MIGRATE`       | Environment | Variable | Cloud Run Job that applies the presentation migrations (`dfo-dev-presentation-migrate` in dev).                                                                                                                     | `just handoff apis` in infra |
| `JOB_DATASETS_MAINTENANCE`       | Environment | Variable | Cloud Run Job that runs the `datasets-maintenance` tasks: the reconciliation sweep, aging terminal operations, and the deploy's settle, warning rewrite, warning backfill and description backfill.                 | `just handoff apis` in infra |
| `JOB_ORIGIN_CATALOGUE`           | Environment | Variable | Cloud Run Job that builds and verifies the origin catalogue (`dfo-dev-origin-catalogue` in dev).                                                                                                                    | `just handoff apis` in infra |
| `JOB_ORIGIN_CATALOGUE_CANDIDATE` | Environment | Variable | Unscheduled Cloud Run Job with the origin catalogue job's template, environment, service account and task timeout, which the deploy points at the new image to verify and build the catalogue before traffic moves. | `just handoff apis` in infra |
| `JOB_SEARCH_CATALOGUE`           | Environment | Variable | Cloud Run Job that builds the search catalogue (`dfo-dev-search-catalogue` in dev).                                                                                                                                 | `just handoff apis` in infra |
| `JOB_GRAPH_RELEASE`              | Environment | Variable | Cloud Run Job that builds and publishes the graph release (`dfo-dev-graph-release` in dev).                                                                                                                         | `just handoff apis` in infra |

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
- Grants the deployer service account `roles/run.developer`, `roles/artifactregistry.writer`, and `roles/iam.serviceAccountUser`, plus `roles/run.invoker` on the apis service and `roles/iam.serviceAccountOpenIdTokenCreator` on itself for the probe step. `roles/run.developer` on the service covers the tagged deploy and the traffic update.

### Application (`apis`)

- Builds, tests, warms ResolveKit SQLite caches, and pushes container images.
- Updates Cloud Run revisions using `gcloud run services update --image=...` after CI checks pass.
- Ships migration DDL under `src/dataset_store/migrations/` and `src/presentation_store/migrations/`; the deploy workflow drives both migration jobs on every deploy (documented in [deploy/postgres.md](postgres.md)).
