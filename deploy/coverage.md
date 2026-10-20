# Coverage artifact (`series_coverage`)

The coverage artifact is a precomputed index of every role-bearing series (every variable whose
Spanner `Edge` rows declare one or two `observationProperties` roles, as `series_coverage.job`
selects them), one row per `(variable, facet_id, donor, recipient)`. The graph release builder's
shadow diff, the `coverage` gate of `scripts/graph_release_parity.py` and the code index export
(`scripts/export_code_index.py`) read it. Serving processes read coverage from the graph
release's coverage view (`deploy/graph-release.md`).

## Layout

`COVERAGE_ARTIFACT_URI` (`gs://` or `file://`) is the bucket root. Each image reads and writes only its own format root, `<COVERAGE_ARTIFACT_URI>/format-<N>/`, where `N` is that image's `series_coverage.schema.COVERAGE_FORMAT_VERSION` (currently 3). A format root holds five things:

- `current.json`, the pointer: `{"stamp": ..., "build_id": ...}`. `stamp` is the import key's stamp, a hash of the newest `IngestionHistory` row and the newest successful one, and `build_id` names the build directory that holds it. The job writes it last, so it is the commit point.
- `versions/<build_id>/`, one directory per build, holding `manifest.json` (which repeats `stamp` and `build_id`, and records `format_version`, `epoch_year` and `mask_words`), `coverage.parquet` (schema `series_coverage.schema.YearHorizon.schema`: `variable`, `facet_id`, `donor`, `recipient`, `n_obs`, `first_year`, `last_year`, and year-mask words `year_mask_0` .. `year_mask_{k-1}`, 64 years each from the manifest's `epoch_year`; the job sets `k`, the manifest's `mask_words`, from the latest year the data carries), `role_names.json` (each variable's donor/recipient role-property dcids, e.g. `financingProvider`/`financingRecipient` for CRS), and `facets.json` (each `facet_id`'s provenance/unit/measurement-method/scaling-factor). A variable that declares one role has rows with its entity in the column of that role's flow axis and an empty value (`series_coverage.schema.ABSENT_ENTITY`) in the other: an IDS total declaring `financingRecipient` fills `recipient`, and a DAC1 aggregate declaring `financingProvider` fills `donor`. Its `role_names.json` entry holds `null` for the role it lacks.
- `compatible/<content_hash>.json`, one pointer per packaged code index in `current.json`'s form, naming a build that carries that index's provenances. `<content_hash>` is the index's `IndexStamp.content_hash`. The job writes it after a publish the index matches, and the deploy's `verify_index --record` step writes it for the new image (see [Deploy order](#deploy-order)).
- `serving/code_index.json`, `{"content_hash": ...}`: the code index both services ran when the last deploy finished. The deploy's final step writes it, and the job reads it to expire compatibility pointers.
- `coverage_state.json`, the job's record of when the current stamp was first seen unpublished, which drives the stale-publish alert.

A build id is the UTC start time to the second plus eight random hex digits, e.g. `20260927T142252Z-3f9a0c1e`. A build directory is written once, with create-only writes, by the run that created it, and never rewritten. Two runs of the same stamp (a scheduled run and a deploy run) therefore publish two separate directories, and a reader that resolved the pointer to one build reads only that build's files while another publish lands. The loader refuses a pointer and manifest that disagree on `stamp` or `build_id`, and a manifest whose `format_version` is not the image's own.

Because each format has its own root, an old and a new revision never share a pointer or a directory. A change to `COVERAGE_FORMAT_VERSION` starts a new, empty root that the new image's first job run fills, and leaves the previous format's last artifact untouched for a rollback.

## Building it

`python -m series_coverage.job` reads Spanner through a BigQuery federation connection and
publishes a new version when the underlying data has moved. A run does, in order:

1. Reads the import key (`artifact_publication.jobs.ImportKey`: the newest `IngestionHistory` row of any status, the newest successful row, and the `IngestionLock` owner) in three small statements, and refuses to start while `IngestionLock` is held.
2. Reads the format root's `current.json` with its generation in one call; no-ops (after pruning) when the published stamp equals the key's. Otherwise it skips (`coverage_job_skipped`, reason `import_not_successful`) while the newest `IngestionHistory` row is not the newest success, since the database could then hold part of an import. A failed import moves the stamp and publishes nothing, so the stale-publish alert fires a cycle later. While the newest row is a failure the job publishes nothing, so a format-bump deploy's verify-then-build step fails its second verify and stops until a later import succeeds; the stale alert reports the wait.
3. Builds the coverage table with one BigQuery job per query, each pushing a plain,
   root-partitionable projection into `EXTERNAL_QUERY`'s inner Spanner SQL and aggregating in the
   outer BigQuery statement; resolves each variable's donor/recipient role names and each
   `facet_id`'s metadata; reconciles the built table's per-variable row counts against SDMX `/data` for a sample of role-bearing StatVars (the smallest variable of each donor/recipient role group, then variables in `sha256(stamp + variable)` order while their expected rows fit `COVERAGE_RECONCILE_MAX_ROWS`), refusing to publish on any mismatch. One data version always gets the same sample.
4. Re-reads the import key, and discards the build (a no-op, not a failure) if
   the lock was taken or the stamp moved while the build ran.
5. Publishes into a new `versions/<build_id>/` directory with create-only writes, then writes `current.json` last, conditional on its generation still matching what step 2 read (a concurrent invocation that published first makes this run's write lose cleanly, leaving an unreferenced build for a later prune); logs a structured `coverage_publish_stale` event when the published stamp has been behind the key's stamp for more than one job cycle.
6. After a publish, loads the new build with the serving loader and compares it with the code index its image packages (`series_coverage.compatibility.index_coverage_differences`). A build the index matches moves that index's `compatible/` pointer to it and logs `coverage_index_compatible`. A build it does not match logs `coverage_index_incompatible` at severity `ERROR`, naming up to twenty variables, and leaves the pointer where it was until the index is re-exported against the current build.
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

A live build against the full dataset publishes 1,782,460 rows over 22,509 variables; reconciliation streams at most `COVERAGE_RECONCILE_MAX_ROWS` expected rows from SDMX `/data`, plus each role group's smallest variable, and the build logs the figure as `reconciliation_rows_expected`.

## Deploy order

The coverage job runs as a Cloud Run Job (`dfo-dev-coverage` in dev). `deploy.yml` points it at the new
image after the migration jobs and before updating the Cloud Run service, so the scheduled runs
(`dfo-dev-coverage-tick`, every 15 minutes) use the new code. The deploy checks for a published
artifact in the new image's format and leaves keeping it fresh to the scheduled job.

The check is `python -m series_coverage.verify`, run as an execution of the same `dfo-dev-coverage`
job with its args overridden (the command stays `python`). It inherits that job's own
`COVERAGE_ARTIFACT_URI` directly rather than a second, workflow-level copy of it, and calls
`load_coverage` against the new image's format root under it, exiting non-zero unless a compatible
artifact loads. An artifact in another format's root does not satisfy it.

Verify runs first. It passes when the format root holds a compatible artifact, however old, and the
deploy goes on without a build, so a data problem in the build (a reconciliation failure, an
unrepresentable year) does not block a code release. A format bump has no artifact in its new format
root, so verify fails; the workflow then executes the build job and verifies again, and the deploy
stops if the build or the second verify fails. A format bump thus builds before traffic moves, and
any other deploy leaves freshness to the scheduled job.

A second check follows: `python -m series_coverage.verify_index --record`, run the same way. The packaged
code index carries each StatVar's provenances, joined from a coverage build when the index was
exported (`scripts/export_code_index.py`). The check compares every index row with the published
build, requires every declared dataset to hold a row in it, requires every donor and recipient the build holds on a facet of a declared dataset to appear in the index's `parties` (a new total would otherwise be counted as an ordinary party), requires every StatVar the build assigns to a listed dataset to appear in the index's `listed_statvars` (a series the build adds there would otherwise belong to no answer), prints the index's `coverage_build`
beside the build it read, and exits non-zero on any difference, naming up to twenty variables, parties or listed StatVars. A
failure stops the deploy before traffic moves; re-export the index against the current build. On a
pass, `--record` (which records nothing for a build with an unlisted party or an unlisted listed StatVar) points the new index's `compatible/` pointer at the build it read.

The deploy's final step, after both services moved, runs `python -m series_coverage.record_serving_index` the same way. It writes `serving/code_index.json` with the new index's `content_hash`, which the job's prune reads to decide which compatibility pointers no serving revision needs. A deploy that fails before this step leaves the previous record, which expires fewer pointers.

A rollback reads the artifact its own image's format root holds. Two images of the same `COVERAGE_FORMAT_VERSION` share that root, kept current by whichever image the job runs. Across a format change, roll back by redeploying the older commit through `deploy-dev.yml`: that points the coverage job at the older image and verifies its format root before either service moves, building it first when the root holds no compatible artifact. From then on the scheduled runs, now on the older image, bring that root to the current stamp. Moving traffic back to an older-format revision in the Cloud Run console alone leaves the job on the newer image, which writes only the newer root. The older revision then serves its format's last artifact, which nothing updates, for as long as it stays live, and the stale-publish alert does not fire for it, because the alert watches only the root the job's image writes.

## Re-recording the offline test fixture

`tests/fixtures/coverage_slice/` is a real, unmodified slice of a published artifact, kept
independent of the fixtures and corpus cases that reference it. The session release
`tests/conftest.py` publishes carries its rows as the release's coverage, and `tests/series_coverage/test_coverage_slice.py` checks it for drift against `sdmx_obs.json`.

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
