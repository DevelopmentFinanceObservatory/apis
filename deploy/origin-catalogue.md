# Origin catalogue (`origin_catalogue`)

The origin catalogue lists every provenance the platform can serve, with the variables and topics each one holds and the Data Commons it comes from: DFO's own Spanner graph, the base Data Commons behind our mixer, and the UN System Data Commons. The apis service and the datasets worker load it at startup and answer origin questions from memory.

## Layout

`ORIGIN_CATALOGUE_URI` (`gs://` or `file://`) is the bucket root. Each image reads and writes only its own format root, `<ORIGIN_CATALOGUE_URI>/format-<N>/`, where `N` is `origin_catalogue.schema.CATALOGUE_FORMAT_VERSION` (currently 1, so `format-1/`). A format root holds:

- `current.json`, the pointer to the live build. The job writes it last, so it is the commit point.
- `versions/<build_id>/`, one directory per build: `provenances.json`, `dfo_variables.parquet`, `base_variables.parquet`, `provenance_places.parquet`, `withheld_variables.parquet`, `topics.parquet`, `scan.json`, `un.json` (the places the UN front door refuses to have named), `dataset_facts.json` (each source-index entry's card facts) and `manifest.json`. A directory holding a manifest is complete.
- `lease.json`, the lease that keeps two runs from building at once.
- `catalogue_state.json`, the identity last seen unpublished and when it was first seen, which drives the stale-publish log event.

## What the job builds

`python -m origin_catalogue.job` builds these tables:

- The provenance table has one row per admitted provenance. DFO rows come from Spanner, base rows from `<-source` on every registered base source through our mixer (`typeOf: Provenance`, minus DFO's provenances and each entry's exclusions), and UN rows from the registry, each read from the UN deployment together with its agency's root group `undata/g/<segment>`. A DFO or base row carries its first `isPartOf` parent typed `Dataset` as `dataset` and takes the Dataset's name; a row with no such parent is an orphan. A UN row takes the registry label and has no `dataset`.
- The dataset facts hold the card facts of each DFO and base entry of `GET /api/graph/sources` (a Dataset, or an orphan provenance): description, licence label, raw release-cadence token, years covered and source URL. `origin_catalogue.card_facts` derives them from the arcs the provenance stage already reads and one batched read of the `name` of every `licenseType` node those arcs name, over the admitted provenances naming the Dataset in `isPartOf`, with the rules the documentation route applies, so a card equals its documentation page for the graph the build read. The provenance stage runs on every build, including one that carries the base tables forward, so the facts refresh at every identity change.
- The variable tables list DFO's variables, and the base variables with the admitted base provenances that hold each and the scan places where they have data.
- The provenance places list, for each admitted base provenance holding a base variable, the scan places where it holds one. A base variable's row unions its provenances and its places, so this table is what says which publisher has data in which country; the search catalogue counts each country's publishers from it. Pass 2 fills it from the facets it already reads.
- The withheld variable set lists each variable our mixer holds data for at a scan place under a provenance that is neither an admitted base provenance nor DFO's, with no facet under an admitted base provenance at any. A DFO facet neither admits a variable nor withholds it, so a variable whose facets are all DFO's is in neither table. Search refuses withheld variables as candidates. A variable in neither table passes as base, and every facet read still admits facet by facet. The job reads facets with no domain filter to tell a withheld variable from one with no data at all, which costs about 1.1 to 1.5 times the filtered read.
- The topic table lists the variables the topic walk reaches from each topic.

The job also probes which scan-set countries the UN front door refuses to have named in an observation GET, so the origin client leaves them out of UN reads. A country counts as refused only when the front door answers a GET naming it alone with its HTTP 403 page twice.

The admitted sources come from `packages/dfo-data-tools/src/origin_catalogue/data/registry.toml`, which ships in the wheel. Adding or removing a source is a pull request to that file.

## Identity and skip

A build's identity hashes five inputs: the artifact format version, a hash of the job's modules (`origin_catalogue.job.JOB_MODULES`: the job, enumeration, the topic walk, the card-fact derivation, the artifact writer, the schema, the registry parser and the origin ids), the registry file's hash, the DFO fingerprint, and the ISO week. The DFO fingerprint is the latest successful `IngestionHistory` row plus the row count and latest `last_update_timestamp` over every `Observation` row. A registry edit, an ingestion, a new ISO week and an image that changes a job module each produce a new identity, so the base tables refresh weekly with no separate schedule and a job fix reaches the catalogue on its first run. A change to the serving modules (`catalogue`, `context`, `graph`, `service`) builds nothing.

Each run reads `current.json` and exits without building when it already carries the current identity (`origin_catalogue_job_noop`), unless that build carried its UN stage forward, in which case the run retries the UN (see UN carry-forward). It also exits without building while `IngestionLock` is held (`origin_catalogue_job_skipped`, reason `ingestion_lock_held`), because the fingerprint is unstable during a load. Before it exits, it checks that the identity the last unlocked run saw is published (see Monitoring).

## Lease

A run that needs to build takes the create-only `lease.json` at the format root. Cloud Scheduler starts an execution every 15 minutes and a full build takes longer, so most executions find the lease held and exit (`origin_catalogue_job_skipped`, reason `lease_held`). The lease lasts 7200 seconds, the job's task timeout, after which Cloud Run has killed the holder. An expired or unparseable lease is reclaimed with a write conditional on its version. The job releases the lease when it finishes.

## Carry-forward

The base variable, withheld variable and topic tables cost most of a build, since they read the mixer for every admitted base provenance. The job copies them from the published build when their key is unchanged: format version, a hash of the enumeration modules (`enumerate.py` and `topic_walk.py`), registry hash, the admitted base provenance set, DFO's provenance set, and the ISO week. DFO's variables and observations are not in the key, so an ingestion that adds no provenance publishes in minutes, new DFO variables included. A registry edit, a new base or DFO provenance, a change to the enumeration modules or a new week rebuilds the tables, within a deadline of 7200 seconds less a 900-second reserve for the topic walk and publication. The `origin_catalogue_job_published` log line reports `base_tables` as `carried` or `built`.

## UN carry-forward

The UN stage reads each registered agency's row and root group from the UN deployment, then probes the front door. When a UN read fails, the build carries the published build's UN rows and refused places forward and publishes, so a UN outage delays no DFO or base update. A UN row the checks reject (`un_agency_root_group_missing`, `provenance_without_url`) is a refusal like any other, so a bad registry edit never publishes. Carried rows keep their own segment and take the current registry's label. A carried row whose segment differs from the registry's is refused (`un_carried_segment_mismatch`), and an agency the published build does not hold stays out until the UN answers. The run logs `origin_catalogue_job_un_carried_forward` at `WARNING` with the reason and `from_build`, the build that last read the UN, and the manifest records the same as `un_carried_forward`. `origin_catalogue.verify` prints it. With no published build to carry from, a failed UN read fails the build.

A published build that carried its UN stage forward does not satisfy the identity skip. The next run with the same identity retries the UN. If the UN answers, it publishes a build that read it. If the read fails again, it logs `origin_catalogue_job_un_retry_failed` at `WARNING` and publishes nothing, since the published build already carries the same UN stage.

## Publication

A run discards its build (`origin_catalogue_job_discarded`) when the ingestion lock was taken or the fingerprint moved while it ran; the next run starts again. Otherwise it writes `versions/<build_id>/`, then writes `current.json` conditional on the generation it read at the start. A run that loses that race logs `origin_catalogue_job_lost_publish_race` and leaves an unreferenced build for pruning. Every run prunes builds that nothing can still be reading: a build is deleted only when `current.json` does not name it and both the pointer and the build are older than 7200 seconds.

## Monitoring

Every run that exits 0 logs `origin_catalogue_job_heartbeat` as its last line, with the published build's `build_id`, `built_at` and `age_seconds`. All three are `null` when nothing is published, and `age_seconds` is `null`, at `WARNING` with an `error`, when the published build's manifest is unreadable. A refused or failed run logs no heartbeat and exits 1.

`origin_catalogue_publish_stale` carries a `reason`:

- `identity_unpublished`: the current identity has gone unpublished for more than one cycle (`ORIGIN_CATALOGUE_CYCLE_SECONDS`). A run that finds `IngestionLock` held cannot compute its own identity, so it checks the identity the last unlocked run recorded in `catalogue_state.json`. A lock held past a cycle while that identity is unpublished logs the line on every run.
- `published_build_age`: the published build is older than a week plus 7200 seconds plus one cycle. The ISO week in the identity makes a healthy job publish at least weekly, and the first run of a week may wait a cycle for the lease and then build for up to 7200 seconds, so an older build means nothing has published since the week turned. The line comes just before the heartbeat, with `build_id`, `age_seconds` and `limit_seconds`.

dfo-infra's `origin_catalogue.tf` alerts on three conditions in one policy: a stale line, a failed execution, and no heartbeat for 9000 seconds. The last catches a Scheduler that stopped starting the job, which logs nothing and fails nothing. Its window exceeds the 7200-second task timeout plus one cycle.

## Refusals

The job refuses to publish a version when any check below fails. A refusal is whole-version: no partial catalogue is written, and the last good build keeps serving. The run logs `origin_catalogue_job_refused` with each refusal's `kind`, `subject` (the dcid at fault) and `detail`, then exits 1, so the Cloud Run execution fails, the failed-execution alert fires, and a deploy waiting on the job stops. The next run sees the identity still unpublished and, after one 15-minute cycle, logs `origin_catalogue_publish_stale` (reason `identity_unpublished`), which the stale-publish alert watches. A run that skips on an unchanged identity, the ingestion lock or a held lease exits 0.

| Kind | Meaning | What to do |
| --- | --- | --- |
| `dfo_provenance_under_unreviewed_source` | A DFO provenance sits under a registered base source whose entry states no `exclusions`. | Review the overlap and add `exclusions` to the entry in `registry.toml`, an empty list if none apply. |
| `dfo_provenance_without_source` | A DFO provenance has no `source` arc in the graph. | Fix the source arc in the ingestion that produced the provenance. |
| `provenance_without_url` | An admitted provenance has no `url` arc, or its URL has no hostname, and the domain filter matches facets by hostname. | Correct the `url` arc upstream, or exclude the provenance in the registry. |
| `exclusion_not_held` | A registry entry excludes a provenance its source does not hold. | Correct the dcid in the entry, or remove the exclusion after an upstream rename. |
| `base_source_without_variables` | A registered base source has admitted provenances and none holds an admitted variable. | Check the entry's `country` and `scan_type` and the source in the mixer. |
| `un_agency_root_group_missing` | A UN agency's root group `undata/g/<segment>` is absent on the UN deployment. | Check the entry's `segment`, then whether the UN deployment renamed the agency's tree. |
| `un_carried_segment_mismatch` | A UN read failed, and a carried UN row's segment differs from the registry entry's. | Wait for the UN to answer, so the new segment's root group is checked, or revert the segment edit. |

A run also exits 1 (`origin_catalogue_job_failed`) for an input the catalogue cannot represent: no successful ingestion to fingerprint, an observation whose facet has no `TimeSeries` provenance, a source whose `<-source` list exceeds 5000 subjects, a malformed registry, a base-table build that runs out of time, tables that break the catalogue's invariants, a failed read of our mixer, or a failed UN read with no published build to carry forward. The Cloud Run execution fails, and the failed-execution alert fires.

## Registry pull request before ingestion

A registry pull request precedes an ingestion that adds a DFO provenance under a registered base source. An ingestion moves the fingerprint, so the next run builds with the new provenance. If its source entry has no reviewed `exclusions`, the run refuses, and no new build publishes for any reason until the registry is fixed. Merging the registry change first, with `exclusions` stating which overlapping base provenances stay out, lets the ingestion publish on its first run.

## Stale catalogue

`CatalogueSnapshot.is_stale` reports a loaded build as stale once its `built_at` is more than 14 days old (`origin_catalogue.catalogue.STALE_AFTER`). The weekly identity change means a healthy job publishes at least every 7 days, so a stale catalogue means the job has been failing or refusing for a week or more. Read the job's logs for `origin_catalogue_job_refused` or `origin_catalogue_job_failed`.

## First deploy and format bumps

The first deploy has no published build, and a format bump reads a new, empty format root, so neither has a build to carry forward or serve. `origin_catalogue.verify` fails on such a root, so `deploy-dev.yml` executes the job with `--wait` and verifies again before either service moves (see Deploy order). With nothing to carry forward, the build is a full one and takes up to two hours. A build published from a local run shortens the wait: publish it to the bucket (below), and the deploy's first verify passes.

A format bump starts a new format root and leaves the previous root's last build in place, so a rollback to the previous image still finds its catalogue.

An image reads only its own format root. Where the bucket holds a build under another format root and none under the image's own, that image's first deploy runs the full build before either service moves.

## Running it locally

`--local-dir` publishes under a local directory in place of `ORIGIN_CATALOGUE_URI`. The run needs credentials for BigQuery and network access to the mixer and the UN deployment:

```bash
export ORIGIN_CATALOGUE_BQ_CONNECTION=dev-fin-observatory.northamerica-northeast1.dfo_dev_dc_spanner_connection
export DATASETS_GRAPH_BASE_URL=https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app/core/api
uv run python -m origin_catalogue.job --local-dir /tmp/origin-catalogue-local
```

`--local-dir` replaces `ORIGIN_CATALOGUE_URI`, which need not be set. Point a local service's `ORIGIN_CATALOGUE_URI` at `file:///tmp/origin-catalogue-local` to serve against the build.

To seed the bucket, copy the format root: `gcloud storage cp -r /tmp/origin-catalogue-local/format-<N> gs://dfo-dev-origin-catalogue-<project>/`. Copy `versions/` before `current.json`.

## Environment

- `ORIGIN_CATALOGUE_URI`: required, `gs://` or `file://`.
- `ORIGIN_CATALOGUE_BQ_CONNECTION`: required, `project.location.connection_id`.
- `DATASETS_GRAPH_BASE_URL`: required, our mixer's API root including `/core/api`.
- `UN_GRAPH_BASE_URL`: optional, the UN deployment's API root including `/core/api`, defaulting to `https://unsd-datacommons.gcp.un-icc.cloud/core/api`. The apis service and the datasets worker read the same variable with the same default, since a build records UN rows and refused places from the deployment it read.
- `DATASETS_GRAPH_TOKEN`: optional bearer token, unset on the dev graph, which serves reads anonymously.
- `ORIGIN_CATALOGUE_BQ_PROJECT`: optional billing project, defaulting to the connection's leading segment.
- `ORIGIN_CATALOGUE_GRAPH_TIMEOUT_SECONDS`: optional, default 120.
- `ORIGIN_CATALOGUE_CYCLE_SECONDS`: optional, default 900, matching Cloud Scheduler's cadence.

## Deploy order

`deploy.yml` points the origin catalogue job (`dfo-dev-origin-catalogue` in dev) at the new image after the coverage steps and before the datasets worker and the apis service move, so the scheduled runs (Cloud Scheduler, every 15 minutes) use the new code. The services load the catalogue at startup and fail readiness when none can be loaded, so the deploy checks that a compatible build is published and leaves keeping it fresh to the scheduled job. A running service reloads a newer build within `ORIGIN_CATALOGUE_RECHECK_SECONDS`.

The check is `python -m origin_catalogue.verify`, run as an execution of the same job with its args overridden (the command stays `python`), so it reads the job's own `ORIGIN_CATALOGUE_URI`. It loads the new image's format root with the loader the services call at startup and exits non-zero unless a compatible build loads. A build in another format's root does not satisfy it.

Verify runs first. It passes when the format root holds a compatible build, however old, and the deploy goes on without a build, so a refusal or a failed read in the job does not block a code release. The first deploy and a format bump have no build in the new format root, so verify fails; the workflow then executes the job with `--wait` and verifies again, and the deploy stops with both services on their current image if the build or the second verify fails. The second verify also catches a run that exits 0 without publishing, such as one that finds the lease held by a scheduled run still building.
