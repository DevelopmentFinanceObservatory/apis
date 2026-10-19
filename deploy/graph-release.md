# Graph release (`graph_release`)

The graph release is one build of DFO's Data Commons graph and the DFO layer per import: the series summary, facets, edges, nodes and imports read from Spanner, plus the code-index blocks, declarations and base parties derived from them. The builder job, `python -m graph_release.job`, builds, gates and publishes a release after each import. No serving process reads it unless `GRAPH_RELEASE_URI` is set on it.

## Layout

`GRAPH_RELEASE_URI` (`gs://` or `file://`) is the release root:

- `format-1/current.json`, the pointer: `{"release_id": ..., "hold": null}`. The job moves it last, conditional on the version it read, so it is the commit point.
- `manifests/<release_id>.json`, one per release, create-only. The release id hashes the part hashes, the import execution id, the DC serving revision and the format.
- `parts/<sha256>.arrow`, `.json` and `.json.gz`, named by content and shared between releases. The `code_index_blocks` part is a code index document `code_index.loader.parse_index` loads: the blocks and a stamp holding their counts and `content_hash`, with an empty `exported_at` and a null `coverage_build`. A serving process fills those two from the manifest, the `read_timestamp` and the release id, when it opens the index (`graph_release.views.release_code_indexes`). The part cannot hold them, since its bytes feed the release id, so an unchanged index keeps its part and its `content_hash` across releases.
- `lease.json`, the build lease, and `builder_state.json`: the last built inputs and the release they built, the last failed inputs, the last unavailable inputs and when, and the shadow wait. Ticks write the state file with and without the lease, so each write is conditional on the version it read and is applied again to a newer one, up to five times.

## What a tick does

Cloud Scheduler starts the job every 2 minutes. Each tick:

1. Reads the import key (`IngestionLock` and `IngestionHistory`) in one read-only snapshot, and exits unless the lock is free and the newest row is the newest success.
2. Reads the DC service's serving revisions through the Cloud Run Admin API, and exits until every one was created after the import completed. The ingestion workflow restarts the DC service after each import, and the mixer reads Spanner as of the last ingestion it saw at start.
3. Forms the inputs: the import's execution id, the serving revision, the format, a hash of the builder's source (`graph_release.job.BUILDER_MODULES`), the declarations hash and, with `GRAPH_RELEASE_SHADOW=1`, the build ids of the published coverage build and origin catalogue. It exits when they equal the last built inputs and the pointer names the release they built, when they equal the last failed inputs, when the pointer is held, or when they ended unavailable less than 600 seconds ago (outcome `unavailable_backoff`, exit 0, logging the recorded `graph_release_unavailable` line again with `backoff: true`). With shadow on it exits `graph_release_shadow_waiting` until the published coverage build's `stamp` and the origin catalogue's `dfo_fingerprint` both equal the import's stamp, and exits unavailable, naming each comparator that is behind, once it has waited 10800 seconds.
4. Takes `lease.json` for 1800 seconds, the task timeout, or exits (outcome `lease_held`). When the inputs equal the last unavailable inputs, of any age, the lease holder is retrying them, so the tick logs the recorded `graph_release_unavailable` line again with `backoff: true` before it exits.
5. Reads every input in one snapshot at the current time, and discards the build when the import key in it differs from step 1's.
6. Builds the parts, reads the base parties from the mixer, and runs the gate: the invariants, the DFO rebuild from the staged files, the serving views, the mixer differential, the SDMX counts and, with shadow on, the diff against the coverage build and the origin catalogue. The serving views are the `CoverageIndex` and the code and measure indexes, built from the staged files through `graph_release.views`, the functions a serving process opens a release with. A view that does not build is a finding, `coverage_view_refused` or `code_index_unloadable`, so every published release opens.
7. Re-reads the serving revision and discards the build if it moved (`graph_release_discarded`, reason `serving_revision_moved`), before acting on any finding or unavailable read from step 6, so it records nothing. A dependency unavailable during step 6 ends the build and triggers the same re-read; an Admin API that cannot answer it adds its own entry to the unavailable line. A release id equal to the serving one records the inputs and publishes nothing. Otherwise it uploads the missing parts, writes the manifest and moves the pointer, then records the inputs as the last built. A failed write of that record logs `graph_release_state_write_failed` and leaves the outcome `published`; the next tick builds the same inputs, finds the release unchanged and records them. It prunes while still holding the lease, then releases it. A failed prune logs `graph_release_prune_failed` and a failed lease release a warning. Neither changes the tick's outcome.

Pruning keeps the ten newest manifests, every manifest younger than 24 hours and the pointer's release, and deletes the parts none of them names.

## Gate outcomes

A finding is deterministic: the same inputs fail the same way. The tick logs `graph_release_gate_failed` with the findings, records the inputs as the last failed build and exits 0. Later ticks skip those inputs (`graph_release_skipped`, reason `inputs_failed`) until the import, the serving revision, the builder, the declarations or, with shadow on, a comparator build changes, or until `graph_release.hold retry` clears them.

An unavailable dependency (Spanner, the Admin API, the mixer, SDMX, the bucket, a shadow comparator, or the credentials any of them needs) logs `graph_release_unavailable` and exits 0. A base party the mixer answers with no node counts as the mixer unavailable, since every base party is in the base graph. Once the inputs are formed, the tick records them with the time and the unavailable entries. For the next 600 seconds, ticks with the same inputs build nothing, exit 0 and log the recorded `graph_release_unavailable` line again with `backoff: true`, so the line recurs every tick while the outage lasts. The first tick after that retries, and the ticks that overlap the retry build end `lease_held` and log the line again too, so the 15-minute alert window stays filled until the retry ends.

Any other exception ends the tick with `graph_release_heartbeat` at outcome `error`, naming the exception type, and a traceback, and exits non-zero. It records nothing, so the next tick builds again. That exception, the task timeout and an out-of-memory kill are the only ways a tick fails its Cloud Run execution. Findings and unavailable dependencies raise their alerts through their log lines.

## Hold and rollback

```bash
uv run python -m graph_release.hold set <release_id>
uv run python -m graph_release.hold clear
uv run python -m graph_release.hold retry
```

`set` points at a published release and holds it there; the job then exits at step 3 (`graph_release_skipped`, reason `held`) and logs no `graph_release_stale`. `clear` lifts the hold. The pointer then no longer names the release the last built inputs built, so the next tick rebuilds, finds that release's manifest, and moves the pointer back to it.

`retry` removes the last failed inputs from `builder_state.json`, so the next tick builds them again. Use it when a finding came from a fault that has since cleared. It logs `graph_release_retry` with the inputs it removed, or `null`.

All three act on `GRAPH_RELEASE_URI`, or on `--local-dir`, and refuse while a build holds the lease.

## Monitoring

Every tick logs `graph_release_heartbeat` last, with its `outcome`. `graph_release_stale` comes before it while the newest import of any status completed more than 1800 seconds ago and the serving release is not built from it, less up to 10800 seconds spent shadow waiting, unless the pointer is held. A failed import never gets a release, so it raises the line until the next success.

`graph_release_published`, `graph_release_gate_failed` and a `graph_release_unavailable` from a build carry `phases`, which gives each of `reads`, `platform`, `dfo`, `gate`, `shadow` and `publish` that ran its wall time in seconds and the process's peak RSS in bytes when it ended.

The `infra` repository's `graph_release.tf` alerts on `graph_release_stale` or `graph_release_gate_failed` at once, on `graph_release_unavailable` logged for 15 minutes, on any failed execution, and on no heartbeat for 2100 seconds, the task timeout plus one tick. The failed-execution condition also covers a builder that has never logged a heartbeat, since an absence condition arms only once its metric has a sample. The heartbeat metric leaves out outcomes `error` and `lease_held`, so a tick that crashes on every run raises the no-heartbeat alert, and so does a build the task timeout or an out-of-memory kill ends on every run: it logs no heartbeat and keeps the lease, and the ticks that overlap it end `lease_held`. A healthy build lands its own heartbeat inside the 2100 seconds. `graph_release_enabled` in the dfo-infra env tfvars switches the Scheduler and both alert policies: false creates the Scheduler paused and the alerts disabled. Set it true once the deployed image carries `graph_release.job`.

## Running it locally

`--local-dir` publishes under a local directory, overriding `GRAPH_RELEASE_URI`. The run needs ADC with read access to the dev Spanner database, the Admin API's `run.services.get` and `run.revisions.get` on the DC service, and the coverage and origin buckets for the shadow diff:

```bash
export GRAPH_RELEASE_SPANNER_PROJECT=dev-fin-observatory
export GRAPH_RELEASE_SPANNER_INSTANCE=dfo-dev-dc-instance
export GRAPH_RELEASE_SPANNER_DATABASE=dfo-dev-dc-db
export DATACOMMONS_SERVICE_NAME=projects/dev-fin-observatory/locations/northamerica-northeast1/services/dfo-dev-dc-datacommons-service
export DATASETS_GRAPH_BASE_URL=https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app/core/api
export DATASETS_SDMX_BASE_URL=$DATASETS_GRAPH_BASE_URL
export GRAPH_RELEASE_SHADOW=1
export COVERAGE_ARTIFACT_URI=gs://dfo-dev-coverage-dev-fin-observatory/
export ORIGIN_CATALOGUE_URI=gs://dfo-dev-origin-catalogue-dev-fin-observatory/
uv run python -m graph_release.job --local-dir /tmp/graph-release-local
```

Point a local service's `GRAPH_RELEASE_URI` at `file:///tmp/graph-release-local` to load the release.

## Environment

- `GRAPH_RELEASE_URI`: required, `gs://` or `file://`.
- `GRAPH_RELEASE_SPANNER_PROJECT`, `GRAPH_RELEASE_SPANNER_INSTANCE`, `GRAPH_RELEASE_SPANNER_DATABASE`: required, the DC Spanner database.
- `DATACOMMONS_SERVICE_NAME`: required, the DC service as `projects/<p>/locations/<r>/services/<s>`.
- `DATASETS_GRAPH_BASE_URL`, `DATASETS_SDMX_BASE_URL`: required, our mixer's API root including `/core/api`. The dev graph serves reads anonymously, so the job sends no token.
- `GRAPH_RELEASE_SHADOW`: optional, default off. On, it needs `COVERAGE_ARTIFACT_URI` and `ORIGIN_CATALOGUE_URI`.
