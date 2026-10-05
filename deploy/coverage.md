# Coverage artifact (`series_coverage`)

The coverage artifact is a precomputed index of every role-bearing series (CRS financing flows and
two World Bank debt-stock variables), one row per `(variable, facet_id, donor, recipient)`, that
Search and Datasets both load at startup and query in memory. It replaces a graph or SDMX read on
every role-bearing scope, facet, and headline decision.

## Layout

`COVERAGE_ARTIFACT_URI` (`gs://` or `file://`) is the bucket root. Each image reads and writes only its own format root, `<COVERAGE_ARTIFACT_URI>/format-<N>/`, where `N` is that image's `series_coverage.schema.COVERAGE_FORMAT_VERSION` (currently 3). A format root holds five things:

- `current.json`, the pointer: `{"stamp": ..., "build_id": ...}`. `stamp` is the data-version fingerprint and `build_id` names the build directory that holds it. The job writes it last, so it is the commit point.
- `versions/<build_id>/`, one directory per build, holding `manifest.json` (which repeats `stamp` and `build_id`, and records `format_version`, `epoch_year` and `mask_words`), `coverage.parquet` (schema `series_coverage.schema.YearHorizon.schema`: `variable`, `facet_id`, `donor`, `recipient`, `n_obs`, `first_year`, `last_year`, and year-mask words `year_mask_0` .. `year_mask_{k-1}`, 64 years each from the manifest's `epoch_year`; the job sets `k`, the manifest's `mask_words`, from the latest year the data carries), `role_names.json` (each variable's donor/recipient role-property dcids, e.g. `financingProvider`/`financingRecipient` for CRS), and `facets.json` (each `facet_id`'s provenance/unit/measurement-method/scaling-factor). A variable that declares one role has rows with its entity in the column of that role's flow axis and an empty value (`series_coverage.schema.ABSENT_ENTITY`) in the other: an IDS total declaring `financingRecipient` fills `recipient`, and a DAC1 aggregate declaring `financingProvider` fills `donor`. Its `role_names.json` entry holds `null` for the role it lacks. Format 3 admits a one-role variable on either axis; format 2 admitted one on the recipient axis only.
- `compatible/<content_hash>.json`, one pointer per packaged code index in `current.json`'s form, naming a build that carries that index's provenances. `<content_hash>` is the index's `IndexStamp.content_hash`. The job writes it after a publish the index matches, and the deploy's `verify_index --record` step writes it for the new image (see [How the service loads it](#how-the-service-loads-it)).
- `serving/code_index.json`, `{"content_hash": ...}`: the code index both services ran when the last deploy finished. The deploy's final step writes it, and the job reads it to expire compatibility pointers.
- `coverage_state.json`, the job's record of when the current fingerprint was first seen unpublished, which drives the stale-publish alert.

A build id is the UTC start time to the second plus eight random hex digits, e.g. `20260927T142252Z-3f9a0c1e`. A build directory is written once, with create-only writes, by the run that created it, and never rewritten. Two runs of the same fingerprint (a scheduled run and a deploy run) therefore publish two separate directories, and a reader that resolved the pointer to one build reads only that build's files while another publish lands. The loader refuses a pointer and manifest that disagree on `stamp` or `build_id`, and a manifest whose `format_version` is not the image's own.

Because each format has its own root, an old and a new revision never share a pointer or a directory. A change to `COVERAGE_FORMAT_VERSION` starts a new, empty root that the new image's first job run fills, and leaves the previous format's last artifact untouched for a rollback.

## Building it

`python -m series_coverage.job` reads Spanner through a BigQuery federation connection and
publishes a new version when the underlying data has moved. A run does, in order:

1. Refuses to start while `IngestionLock` is held.
2. Reads a fingerprint (the latest successful `IngestionHistory` row plus `COUNT(*)`/`MAX(last_update_timestamp)` over role-bearing observations), and reads the format root's `current.json` with its generation in one call; no-ops (after pruning) when the published stamp equals the fingerprint's.
3. Builds the coverage table with one BigQuery job per query, each pushing a plain,
   root-partitionable projection into `EXTERNAL_QUERY`'s inner Spanner SQL and aggregating in the
   outer BigQuery statement; resolves each variable's donor/recipient role names and each
   `facet_id`'s metadata; reconciles the built table's per-variable row counts against SDMX `/data` for a sample of role-bearing StatVars (the smallest variable of each donor/recipient role group, then variables in `sha256(stamp + variable)` order while their expected rows fit `COVERAGE_RECONCILE_MAX_ROWS`), refusing to publish on any mismatch. One data version always gets the same sample.
4. Re-reads the lock and the fingerprint, and discards the build (a no-op, not a failure) if
   either moved while the build ran.
5. Publishes into a new `versions/<build_id>/` directory with create-only writes, then writes `current.json` last, conditional on its generation still matching what step 2 read (a concurrent invocation that published first makes this run's write lose cleanly, leaving an unreferenced build for a later prune); logs a structured `coverage_publish_stale` event when the published stamp has been behind the fingerprint for more than one job cycle.
6. After a publish, loads the new build with the serving loader and compares it with the code index its image packages (`series_coverage.compatibility.index_coverage_differences`). A build the index matches moves that index's `compatible/` pointer to it and logs `coverage_index_compatible`. A build it does not match logs `coverage_index_incompatible` at severity `ERROR`, naming up to twenty variables, and leaves the pointer where it was: services running that index keep serving the pointer's build until the index is re-exported against the current build.
7. Prunes build directories nothing can still be using, on every run (published or no-op alike). This is what bounds `versions/`, since the artifacts bucket carries no lifecycle rule of its own. A build is deleted only when neither `current.json` nor any `compatible/` pointer names it, every one of those pointers was written at least 1800 seconds ago (the job's Cloud Run timeout, `_MAX_RUN_SECONDS`), and the build's newest file is at least that old. The first condition protects the live build and each index's compatible build; the second protects a build a pointer named until recently, which a reader that resolved the old pointer may still be loading; the third protects a concurrent run's in-progress publish. An unreadable pointer deletes nothing, and prune never lists or deletes outside its own format root. Prune then deletes a `compatible/` pointer when its index is not the one `serving/code_index.json` names, it was last written before that record, and it was last written seven days ago or more (`_COMPATIBLE_POINTER_RETENTION_SECONDS`). The deploy writes the record only after both services moved, so no revision serving at that point runs another index, and a later deploy that moved only the worker wrote its own index's pointer after the record. The job image does not decide expiry, since the deploy moves the job before the services and a failed deploy leaves the two apart. With no record, nothing expires. The window covers a rollback in the Cloud Run console. A build only an expired pointer named goes on a later run.

Env: `COVERAGE_ARTIFACT_URI` (required, `gs://` or `file://`), `COVERAGE_BQ_CONNECTION` (required,
`project.location.connection_id`), `DATASETS_SDMX_BASE_URL` (required, the mixer's API root),
`DATASETS_GRAPH_TOKEN` (optional bearer token), `COVERAGE_BQ_PROJECT` (optional billing project,
defaults to the connection's own leading project segment), `COVERAGE_SDMX_TIMEOUT_SECONDS`
(optional, default 90), `COVERAGE_CYCLE_SECONDS` (optional, default 900, matching Cloud Scheduler's 15-minute cadence), `COVERAGE_RECONCILE_MAX_ROWS` (optional, default 250,000, the cap on expected rows the reconciliation sample streams from SDMX; each role group's smallest variable is checked even above it). `DATASETS_GRAPH_BASE_URL` is not read here: the job reads only
BigQuery and SDMX, never the graph.

Installing `dfo-data-tools[coverage]` pulls in `numpy`, `google-cloud-storage`, and
`google-cloud-bigquery`.

## Running it locally

Point `COVERAGE_ARTIFACT_URI` at a `file://` path and run the job against real BigQuery/SDMX
credentials:

```bash
export COVERAGE_ARTIFACT_URI=file:///tmp/coverage-local
export COVERAGE_BQ_CONNECTION=dev-fin-observatory.northamerica-northeast1.dfo_dev_dc_spanner_connection
export DATASETS_SDMX_BASE_URL=https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app/core/api
uv run python -m series_coverage.job
```

A live build against the full dataset publishes 1,782,460 rows over 22,509 variables; reconciliation streams at most `COVERAGE_RECONCILE_MAX_ROWS` expected rows from SDMX `/data`, plus each role group's smallest variable, and the build logs the figure as `reconciliation_rows_expected`. Point `datasets_surface`/`search_surface` at the same `file://` URI (`COVERAGE_ARTIFACT_URI`)
to serve against it locally. A service checks each build against the packaged code index, so a
local build the index does not match, or a dev-bucket build an undeployed index does not match,
refuses at startup; set `COVERAGE_INDEX_CHECK=off` to serve it unchecked.

## How the service loads it

`series_coverage.shared.get_coverage_service()` loads the artifact once, synchronously, at surface construction time, the same discipline `code_index.shared.get_index()` applies to the code index: a missing `COVERAGE_ARTIFACT_URI` or an unreadable artifact fails app startup, never a later request. After that, `COVERAGE_RECHECK_SECONDS` (default 300) bounds how often a request-triggered background reload checks `current.json`; a caller never blocks on that reload, and a failed background reload keeps serving the last-known-good artifact. The recheck only ever pays for one `current.json` read (`series_coverage.loader.read_published_pointer`): the full Parquet/role-names/facets load runs only when the pointer's `build_id` differs from the one being served, so a republish of an unchanged stamp still reloads.

Dataset election reads each StatVar's provenances from the packaged code index, and a mint reads facet provenances from the coverage build, so the service serves only a build whose provenances equal the index's (`index_coverage_differences`, the function `verify_index` runs). Every load, at startup and on refresh, runs that check, which takes 0.20 to 0.25 s over the 22,208-row index. A build that differs is rejected: the service logs an error naming the rejected build, the build it serves instead, the index's `content_hash` and `coverage_build`, and up to twenty differing variables, and then keeps the build it already serves. It moves to the build the index's `compatible/` pointer names only when that build passes too and its build id sorts after the served build's, so the served data never moves back to an older build. Build ids sort in publication order, since each starts with its build time and a publish commits only over the `current.json` it read before building. At startup it loads the pointer's build. It does not load a rejected build again while `current.json` names it. At startup with no build to fall back on, construction raises `CoverageIncompatibleError` and the surface fails readiness. The rejection is reported in the log only; `CoverageService.rejected_build` holds it in process.

`COVERAGE_INDEX_CHECK=off` turns the check off for a harness that pairs a build with the packaged index on purpose: the CI container smoke test and the eval gate, which serve the recorded test slice, and a local run against an index that was exported but never deployed, which has no `compatible/` pointer. Every load then logs a WARNING naming the index's `content_hash` and the build served unchecked. Any value other than unset or `off` refuses at startup (`CoverageIndexCheckValueError`). Terraform and `deploy-dev.yml` never set it.

## Deploy order

The coverage job runs as a Cloud Run Job (`dfo-dev-coverage` in dev). `deploy.yml` points it at the new
image after the migration jobs and before updating the Cloud Run service, so the scheduled runs
(`dfo-dev-coverage-tick`, every 15 minutes) use the new code. Search and Datasets surface
construction loads the coverage artifact at startup and fails readiness when none can be loaded, so
a revision must never roll out against an image whose coverage-dependent surfaces would start with
no artifact to load. The deploy therefore checks for a published artifact and leaves keeping it
fresh to the scheduled job. A running service reloads a newer artifact within
`COVERAGE_RECHECK_SECONDS`.

The check is `python -m series_coverage.verify`, run as an execution of the same `dfo-dev-coverage`
job with its args overridden (the command stays `python`). It inherits that job's own
`COVERAGE_ARTIFACT_URI` directly rather than a second, workflow-level copy of it, and calls
`load_coverage` against the new image's format root under it, exiting non-zero unless a compatible
artifact loads. An artifact in another format's root does not satisfy it. This is the same function
`get_coverage_service()` calls at startup, so a failure here always means the service would also
have failed readiness, just caught one step earlier with a clear message instead of a stuck Cloud
Run revision.

Verify runs first. It passes when the format root holds a compatible artifact, however old, and the
deploy goes on without a build, so a data problem in the build (a reconciliation failure, an
unrepresentable year) does not block a code release. A format bump has no artifact in its new format
root, so verify fails; the workflow then executes the build job and verifies again, and the deploy
stops if the build or the second verify fails. A format bump thus builds before traffic moves, and
any other deploy leaves freshness to the scheduled job.

A second check follows: `python -m series_coverage.verify_index --record`, run the same way. The packaged
code index carries each StatVar's provenances, joined from a coverage build when the index was
exported (`scripts/export_code_index.py`). The check compares every index row with the published
build, requires every declared dataset to hold a row in it, prints the index's `coverage_build`
beside the build it read, and exits non-zero on any difference, naming up to twenty variables. A
failure stops the deploy before traffic moves; re-export the index against the current build. On a
pass, `--record` points the new index's `compatible/` pointer at the build it read, so the new
revision has a build to start from even after a later publish its index does not match.

The deploy's final step, after both services moved, runs `python -m series_coverage.record_serving_index` the same way. It writes `serving/code_index.json` with the new index's `content_hash`, which the job's prune reads to decide which compatibility pointers no serving revision needs. A deploy that fails before this step leaves the previous record, which expires fewer pointers.

A rollback reads the artifact its own image's format root holds. Two images of the same `COVERAGE_FORMAT_VERSION` share that root, kept current by whichever image the job runs. Across a format change, roll back by redeploying the older commit through `deploy-dev.yml`: that points the coverage job at the older image and verifies its format root before either service moves, building it first when the root holds no compatible artifact. From then on the scheduled runs, now on the older image, bring that root to the current fingerprint. Moving traffic back to an older-format revision in the Cloud Run console alone leaves the job on the newer image, which writes only the newer root. The older revision then serves its format's last artifact, which nothing updates, for as long as it stays live, and the stale-publish alert does not fire for it, because the alert watches only the root the job's image writes.

## Re-recording the offline test fixture

`tests/fixtures/coverage_slice/` is a real, unmodified slice of a published artifact, kept
independent of the fixtures and corpus cases that reference it. `tests/conftest.py` points the suite at
it, and `tests/series_coverage/test_coverage_slice.py` checks it for drift against `sdmx_obs.json`.

```bash
uv run python scripts/record_coverage_slice.py \
    --source file:///path/to/a/real/coverage/artifact \
    --out tests/fixtures/coverage_slice
```

`scripts/record_coverage_slice.py` keeps every row (every donor/recipient pair) of a variable
when a `tests/fixtures/*.json`, `corpus/*.yaml` or `corpus/families/*.yaml` file holds a whole
token equal to one of the source artifact's own known role-bearing StatVar dcids.

The script refuses to write an empty slice if nothing matches. Re-run it whenever a new
role-bearing StatVar is added to a fixture, a case or a family.
