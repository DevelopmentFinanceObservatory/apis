# Search catalogue (`search.catalogue`)

The search catalogue tells a new visitor what DFO holds and shows them how to ask. It is one set of counts (indicators, datasets, publishers, countries, and aid donors, recipients and years) and the example sentences from `src/search/catalogue/examples.toml` that the search engine answered correctly. The apis service serves it at `GET /api/search/catalogue`. A Cloud Run job, `dfo-dev-search-catalogue`, builds it from the dfo-apis image, so the engine that checks each example is the one users reach.

## Layout

`SEARCH_CATALOGUE_URI` (`gs://` or `file://`) is the bucket root. Each image reads and writes only its own format root, `<SEARCH_CATALOGUE_URI>/format-<N>/`, where `N` is `search.catalogue.artifact.FORMAT_VERSION` (2). A format root holds:

- `current.json`, the pointer to the live build: `{catalogue_version, key}`. The job writes it last, so it is the commit point. A `catalogue_version` is one path segment of letters, digits, `.`, `_` and `-`, starting with a letter or digit; a pointer naming any other is malformed.
- `versions/<catalogue_version>/catalogue.json`, one JSON document per build: the build key and its inputs, the global counts, the examples that passed, and every counted country with its name, its counts and the ids of the examples whose StatVar has data there. A build whose own `catalogue_version` or `key` differs from the pointer's does not load.
- `lease.json`, the lease that keeps two runs from building at once.
- `refused.json`, the last refused build: `{key, refused_at, passed}`.

## What the job builds

`python -m search.catalogue.job` reads one origin catalogue build and one coverage index build and:

1. Loads `examples.toml`. Each line is a template with one `{place}` slot, the StatVar it must resolve to, a default `country/` place, and, for a StatVar the coverage index holds, the coverage role (`donor` or `recipient`) its place fills.
1. Finds the countries where each line's StatVar has data (`search.catalogue.presence`): the coverage rows on the line's role for a StatVar the coverage index holds, the origin catalogue's base variable table for any other. A line with neither source, or whose default place has no data, drops out.
1. Counts publishers, countries and aid, globally and at each country, and indicators and datasets globally (`search.catalogue.counts`). Indicators are the StatVars search can serve: the origin catalogue's base and DFO variables and the coverage variables it does not withhold. Datasets are the distinct source-index entries (`origin_catalogue.catalogue.entry_ref`) of the admitted provenances. A country's publishers are the sources of the base provenances holding a variable there (the origin catalogue's `provenance_places`) and of the coverage facets with a row naming it.
1. Runs each line through the search engine in-process, at its default place and at Tuvalu and Georgia where its StatVar has data there (`search.catalogue.check`). A run passes when the answer is a resolution of one definite part whose StatVars include the line's StatVar, whose resolved entities in the place's role are that place alone (`observationAbout` for a line with no role, `origin` for `donor`, `destination` for `recipient`), whose answer carries no departure from the sentence other than the three that describe the data, and whose observations preview has points, each series with points observing that place in that role. An answer for another place or the reverse direction fails with `scope_mismatch`, or with `preview_scope_mismatch` when only its preview strays. An answer carrying any departure in `search.catalogue.check.REFUSING_DEPARTURES` (every `DepartureCode` except `modelled_values`, `group_read_as_members` and `scope_limited`) fails with `departed`, and the failure's `detail` names the codes. A run the engine raised on, cut short at its soft deadline, or answered with a preview whose read failed is tried once more; when it fails in the engine again, the job logs `search_catalogue_job_failed` and exits 1 without publishing or recording a refusal, so the next scheduled run builds again. One run that keeps failing in the engine, at one place, is enough to fail the job. A failed read is any preview omission reason other than the seven in `search.catalogue.check.EMPTY_PREVIEW_REASONS`: `observations_probe_failed`, `observations_bulk_fetch_failed`, `observations_timed_out` and `observations_clients_unavailable`. A line with any other failing run drops out, including a run the engine refused as input (`engine_refused`), since that refusal comes back the same on every try. `SEARCH_CATALOGUE_MAX_CONCURRENCY` bounds the engine calls in flight.
1. Refuses to publish when fewer than five lines pass (`search.catalogue.job.FLOOR`), and records the build's key in `refused.json`.

A country's examples are the passing lines whose StatVar has data there. A country the graph gives no name for has counts and no examples, since its name fills the template.

## Build key and skip

A build's key hashes the format version, the origin catalogue build id, the coverage index build id, `DFO_BUILD_COMMIT` and a digest of `examples.toml`. The job first computes the key from the build ids the two artifacts' pointers name and exits without loading either artifact when `current.json` already carries it (`search_catalogue_job_noop`). Otherwise it loads both artifacts, recomputes the key from the builds it loaded, and builds. A new origin catalogue build, a new coverage build, a deploy and an edit to `examples.toml` each produce a new key.

A refused key stays refused for six hours (`search.catalogue.job.REFUSAL_RETRY_SECONDS`). A run whose key matches `refused.json` inside that window makes no engine calls and logs `search_catalogue_job_still_refused`, so a refused key costs about 16 engine calls once every six hours. A new key builds at once, and the first run after the window builds again, since an engine or graph outage can refuse a build whose inputs would pass.

## Lease and publication

A run that needs to build takes the create-only `lease.json` at the format root, and exits while another run holds an unexpired one (`search_catalogue_job_skipped`, reason `lease_held`). The lease lasts 1800 seconds; a live build takes 80 to 95 seconds. The job releases the lease when it finishes.

A run discards its build (`search_catalogue_job_discarded`, reason `coverage_moved_during_build`) when the coverage index moved while it ran; the next run starts again. Otherwise it writes `versions/<catalogue_version>/catalogue.json` create-only, then writes `current.json` conditional on the generation it read at the start. A run that loses that race logs `search_catalogue_job_lost_publish_race` and leaves an unreferenced build for pruning. Every run prunes builds that nothing can still be reading: a build is deleted only when `current.json` does not name it and both the pointer and the build are older than 1800 seconds (`search_catalogue_job_pruned`).

## Events and exit codes

Each event is one JSON line on stdout with `event` and `severity`, which Cloud Logging parses into `jsonPayload`.

| Event | Severity | Meaning |
| --- | --- | --- |
| `search_catalogue_job_noop` | `INFO` | `current.json` already carries the key; nothing to build. |
| `search_catalogue_job_skipped` | `INFO` | Another run holds the lease (`reason: lease_held`). |
| `search_catalogue_line_dropped` | `WARNING` | One example dropped out, with its `example_id`, `template`, `stat_var` and `reason`: a presence reason (`no_presence_source`, `role_required`, `role_without_coverage`, `default_place_without_data`) or `check_failed` with each failed run. |
| `search_catalogue_job_refused` | `ERROR` | Fewer than five lines passed; nothing published, and the key recorded in `refused.json`. |
| `search_catalogue_job_still_refused` | `WARNING` | `refused.json` refused this key within six hours, with its `refused_at` and the lines that passed; nothing to build. |
| `search_catalogue_job_discarded` | `INFO` | The coverage index moved during the build. |
| `search_catalogue_job_lost_publish_race` | `INFO` | Another run published first. |
| `search_catalogue_job_published` | `INFO` | A build published, with its key, inputs, example ids, counts and `build_seconds`. |
| `search_catalogue_job_pruned` | `INFO` | Builds deleted, by `catalogue_version`. |
| `search_catalogue_job_failed` | `ERROR` | An artifact, `examples.toml` or a graph read failed, or a check run failed in the engine after its retry, with each such run under `failures`. |
| `search_catalogue_job_heartbeat` | `INFO` | The last line of every run that exits 0, with the published `catalogue_version`, `built_at` and `age_seconds`. All three are `null` when nothing is published; `age_seconds` is `null`, at `WARNING` with an `error`, when the published build is unreadable. |

The job exits 1 on a refused build (`search_catalogue_job_refused`) and on a failure (`search_catalogue_job_failed`). A refused or failed run logs no heartbeat, and the last published build keeps serving. Every other run exits 0: a no-op, a still-refused key, a held lease, a discarded build, a lost race, and a build that published with some lines dropped. A still-refused run logs `search_catalogue_job_still_refused` and its heartbeat and exits 0, so the failed-execution alert fires once per refusal window: on the refusal, and again only when the first run after the six-hour window refuses. A failed run records no refusal, so during an engine outage every scheduled run builds again, fails and fires the alert.

## Environment

- `SEARCH_CATALOGUE_URI`: required, `gs://` or `file://`.
- `ORIGIN_CATALOGUE_URI`: required, the origin catalogue the job counts from and binds for its engine calls. The job loads it under the serving surfaces' `ORIGIN_CATALOGUE_*` settings.
- `COVERAGE_ARTIFACT_URI`: required, the coverage index the job counts from and the engine reads.
- `SEARCH_CATALOGUE_MAX_CONCURRENCY`: optional, default 4, the most engine calls in flight.
- The engine's own settings, as on the apis service: `GEMINI_API_KEY`, `SEARCH_GRAPH_BASE`, `SEARCH_GRAPH_TOKEN`, `SEARCH_ENGINE_MODEL`, `UN_GRAPH_BASE_URL`.

## Running it locally

Point `SEARCH_CATALOGUE_URI` at a local directory. The run needs the Gemini key, network access to the graph, and readable origin catalogue and coverage builds; `/tmp/origin-catalogue-local` is the output of a local origin catalogue run (see [deploy/origin-catalogue.md](origin-catalogue.md)):

```bash
export SEARCH_CATALOGUE_URI=file:///tmp/search-catalogue-local
export ORIGIN_CATALOGUE_URI=file:///tmp/origin-catalogue-local
export COVERAGE_ARTIFACT_URI=gs://dfo-dev-coverage-dev-fin-observatory/
export RESOLVEKIT_CACHE_DIR=/tmp/resolvekit-cache
export GEMINI_API_KEY=... SEARCH_GRAPH_TOKEN=...
uv run python -m search.catalogue.job
```

A second run with the same inputs logs `search_catalogue_job_noop`. Point a local apis service's `SEARCH_CATALOGUE_URI` at the same `file://` root to serve the build.

## How the API reads it

The search surface builds a `search.catalogue.service.SearchCatalogueService` in its lifespan. The service loads the build `current.json` names and serves it from memory. It rechecks `current.json` at most once per `SEARCH_CATALOGUE_RECHECK_SECONDS` (default 300), on a daemon thread a request starts, and reloads only when `catalogue_version` changes. A failed reload, or a pointer that disappears, keeps the last good build serving.

The search surface starts and serves resolve whether or not a catalogue is available. With `SEARCH_CATALOGUE_URI` unset, or nothing loadable under it, `GET /api/search/catalogue` answers 503 `service_unavailable` (retryable) until a build loads.

`GET /api/search/catalogue` returns `{catalogue_version, counts, examples}`, where each example is `{text, template, stat_var, place}` at its default place. `?place=<country dcid>` returns the counts at that country and the examples whose StatVar has data there, filled with its name. A `place` that is not a `country/` dcid, or a country the build does not count, answers 400 `request_invalid`. A 200 carries `Cache-Control: public, max-age=300`.

## Deploy order

`deploy-dev.yml` points `dfo-dev-search-catalogue` at the new image after the origin catalogue steps and before the datasets worker and the apis service move, so the scheduled runs (Cloud Scheduler, every 15 minutes) check the examples against the new code. The key carries `DFO_BUILD_COMMIT`, so the first scheduled run after a deploy rebuilds. The deploy does not wait for a build, since the apis service starts without one. `infra` creates the job, its bucket, its service account, the scheduler and the `SEARCH_CATALOGUE_URI` variable on the apis service.
