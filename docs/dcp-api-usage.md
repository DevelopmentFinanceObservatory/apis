# How dfo-apis uses the Data Commons Platform

Every number below is measured and carries the condition it was measured under. Each was measured in-process against the dev DC service, or against the deployed Cloud Run instance. Where the two disagree, treat the in-process number as the code's own cost and the deployed number as the code plus whatever the instance adds.

## What this service asks of the platform

Five route families carry the traffic. SDMX `GET /data` (dataflow `DC/DF_OBS/1.0.0`) returns bulk observation rows as SDMX-CSV, streamed, with no server-side row cap and a component-filter grammar (`c[<role>]=...`) that binds each of a StatVar's declared entity roles independently. `POST /v2/observation` returns a JSON body grouped by variable and entity, used here for facet discovery (which facets exist, with an observation count and an earliest and latest date each) and, for a single pair, a raw value preview. `POST /v2/node` carries both directions of the graph's arcs: out-arcs for a StatVar's declared shape and for labels, reverse-arcs for enumerating which StatVars a classification subject carries. The `/v2/agent/*` RPCs wrap these same underlying calls for an LLM-facing caller. `get_observations` answers one variable at a time with its own curated best-facet subset, and `get_variable_metadata` returns richer per-facet detail, including a source's own refresh dates where an import chose to write them, capped at ten variables by ten entities. The other two RPCs in that family, `search_indicators` and `resolve_places`, answer indicator and place search and are not called by any job in this repository. `POST /v2/internal/sdmx/data` returns the same observations as SDMX `/data` in a JSON body already grouped by series, built for mixer-to-mixer federation and not called by this repository today, kept here because it appears in the route comparison below.

## Frame materialisation

Route: SDMX `GET /data`. Members that share an entity scope, declared roles and `latest`-window status are fetched in combined requests. `materialise.generation.dispatch_lanes` sets the lane count for such a group to the larger of `DATASETS_MINT_MEMBER_MAX_CONCURRENCY` (4) and the member count over eight, rounded up. Each request carries one lane's share of the members, at most about eight, and up to `DATASETS_MINT_MEMBER_MAX_CONCURRENCY` requests run at once.

`/v2/observation` carries the same rows at a fraction of the bytes when scoped to a single entity axis, but its request shape binds one flat entity selector, with no way to bind more than one distinct role axis in the same call, so it cannot serve a role-bearing StatVar's frame at all. The agent's `get_observations` route takes one variable per call and reports only its own top-ranked facet per series. It does not report every observed one, a materially smaller row set than a stored frame is expected to carry. SDMX's own lack of a row cap, plus its component-filter grammar matching the frame's own role columns, is what a frame retrieval needs regardless of shape.

Chunking a fetch across members that share an entity scope measurably cuts wall time. Measured against a 44-member set sharing one entity scope, a request per member finished in a median 10.2 seconds. One request naming all 44 variables at once finished in 5.4 seconds. Four concurrent chunks of about eleven variables each finished in 4.1 seconds, the fastest shape tested, with the identical row set every time. A single combined request naming every variable in one URL starts failing past roughly seventy to eighty StatVars against a 254-entity scope, which chunking stays well clear of.

As the wire response streams in, `dc_sdmx.SdmxClient.stream_data_batches` parses it into Arrow record batches, and every filter between fetch and frame build runs as a columnar operation over them before `materialise.frame_build.build_frame_from_table` builds the frame.

## Facet probing

Route: `POST /v2/observation`, one batched call across a whole entity-scope group of subject-only members at once, naming every member's own StatVar crossed with every entity any member binds.

Fetching SDMX rows just to learn which facets exist would pay for the whole frame merely to inspect its facet identity. The agent's `get_variable_metadata` route carries richer per-facet fields, but caps at ten variables by ten entities, too narrow for a group carrying more members. This deployment's own graph does not populate its refresh-date fields for every source either. Of three sources checked live, two carried a `lastDataRefreshDate`, one did not.

One probe per member would sum to about 10.1 seconds of probe time across the pool for a 44-member group. The single batched call this deployment makes instead measures 0.3 to 0.6 seconds. A rebuild makes it once per entity-scope group of subject-only members (`materialise.facets.probe_collection_candidates`), and role-bearing members read their candidates from the loaded coverage index with no call. A head served inside its recheck interval, or confirmed by its dependency fingerprint, makes no facet call.

## Shape resolution

Route: `POST /v2/node`, `node_arcs_many`, batched across every pinned member of a group in at most two calls regardless of member count, plus the reverse-arc read for a described-arm member's own enumeration.

Neither SDMX nor `/v2/observation` carries a StatVar's declared shape. SDMX has no route for it, and `/v2/observation`'s facet response reports observed facet identity. It does not report declared roles or constraints. Resolving each member's shape as a separate call would run as a plain sequential loop, one call per member, none of them overlapped: 17 calls summing to 2.96 seconds at 10 members, 76 calls summing to 13.45 seconds at 44 members. The batched call this deployment makes instead totals 0.37 seconds at 10 members and 0.82 seconds at 44, independent of member count.

## Currency checking

Route: `POST /v2/node`, one `node_arcs_many` read of the `lastDataRefreshDate` of every source and rate-table provenance the stored head's dependency fingerprint names, compared source by source with the dates the last materialise recorded, beside a digest of the member dcids, the deployed recipe digests and the window the clock gives.

Re-fetching SDMX to check currency would pay the full frame cost on every request, which defeats the purpose of a currency check. The agent's `get_variable_metadata` route carries genuine provenance-refresh fields but is capped at ten variables by ten entities and those fields are not populated for every source, so it cannot serve as a general group-grain signal either.

Within the recheck interval, a request serves the stored record with no graph call at all. Past the interval, one graph read either confirms the record or triggers a rebuild, and a record with no readable fingerprint rebuilds once. Past the maximum age, the record rebuilds regardless, at materialise cost. The maximum age covers what the fingerprint cannot see: a value correction with no date move, a newly elected provenance, a catalogue build that admits or withholds a source, and sources with no refresh date. A definition whose window names no end year reads to the current year, so its record also holds the window it read, and the first request past the recheck interval whose clock ends that window at another year rebuilds it. The rebuild keeps the record's `content_version` when the years it reads hold the same rows. The head path table under Measured numbers gives the wall time of a head served inside the recheck interval.

## Latest-window fetches

Route: SDMX `GET /data` with `c[TIME_PERIOD]=LATEST`, with a client-side reduction to the single latest period kept on top of the server's own filter. A role-bearing StatVar with a series ending after the year it elected is read once more with that year as `c[TIME_PERIOD]`.

The server-side filter is a real saving. On a 30,540-row workload it returned 706 rows in 0.61 seconds and 119 KB, against 30,540 rows in 1.53 seconds and 4.8 MB unfiltered, a 43-fold row reduction. Dropping the client-side reduction on top of that saving is not safe, because the mixer's own `LATEST` is computed per entity and facet. A "latest" window means one period for a StatVar as a whole. Measured live, one variable over 20 countries returned 102 rows under the server's own filter, more than 20, confirming the reduction is computed at the series grain. The client-side reduction still runs on whatever the server already narrowed, at a fraction of what it cost against the unfiltered response.

A `latest` window serves each StatVar one period and only the series holding that period. A role-bearing StatVar serves the year the coverage index elects for it (`CoverageIndex.elected_years`): of the three calendar years ending at the newest year any of its series holds, the newest one held by at least half the series the fullest of the three holds. A series ending after that year has no row in it under `LATEST`, so the StatVar is read again at that year, and when every series ends at it the `LATEST` read is the only request. A StatVar the index elects no year for (a place-keyed one, one with a facet finer than a year, one holding only years after the index was built) serves the newest period any of its served series holds, and so does a StatVar whose elected year no row holds. A series holding no row at the served period serves nothing, so two countries whose latest GDP falls in 2022 and 2024 serve the 2024 point alone. The Datasets mint (`columnar.latest_rows`, `columnar.elected_rows`) and Search's per-series preview read and reduce through the same code (`materialise.generation._servable_rows`), and take the newest period from one definition, `materialise.window.latest_period`, which compares the leading year and then the period string.

## Search-side observation previews

Route: SDMX `GET /data`, one request per `(stat_var, entity or role-tuple)` pair, dispatched through the identical facet-selection path a dataset frame uses, preceded by one batched `/v2/observation` probe across every distinct StatVar a query leg touches so no per-pair facet lookup pays its own round trip.

A per-series `latest` preview fetches each held series with SDMX `LATEST`, and the year its StatVar elected when a series of the read ends after it. Admission and currency omissions run before Search selects each StatVar's one period across its reads: the newest in the elected year when a read holds a row there, else the newest any read holds. Conversion runs on the retained period. Coverage dates guide the reported window, and a stale coverage date cannot remove an actual latest row or require a rate for a row that the preview discards.

The agent's `get_observations` route takes one variable per call and returns its own curated best-facet subset, which could silently disagree with the facet a search leg's own selection logic already resolved for the rest of the answer. Up to eight pairs run concurrently per query leg, and thirty-two across the whole process, so a query touching many variables and entities still bounds its total concurrent upstream calls without one leg starving another.

## Role-bearing facet selection and the role_series cache

Route for facet selection: none. `series_coverage.index.CoverageIndex.filter_candidates` answers which facets exist for a role-bearing StatVar under a bound entity, from the coverage index the bound graph release serves, already in process -- a local table filter, never a network read. `search/engine/retrieval/role_scope.py`'s `_role_index_from_coverage` (behind `discover_role_scope` and `role_index_for_named_entities`) builds `role_index_by_sv` this way for both a resolve and a bulk-series read's own named-destination branch, replacing the SDMX facet-index reads `retrieval/role_resolve.py`'s `graph_confirm_resolve` used to make. Nothing on this path can cap-trip or fail on an SDMX error, and there is no cache to size, since a coverage lookup costs nothing to repeat.

Route for the SDMX traffic that remains: `GET /data` through `role_series`, which `region/observations/bulk_role_series.py`'s `_bulk_role_series` calls once per variable to fetch the real Facets and points a role-bearing bulk-series preview serves (see "Search-side observation previews" above, and "Frame materialisation" for the mint/definition path's own SDMX dispatch through `materialise`'s planned data requests -- the other consumer of this same route family). `clients/sdmx_cache.py`'s `CachingSdmxClient` wraps `role_series` alone at the point `app.py`'s `search_surface()` lifespan constructs the process's shared SDMX client, so every request sharing that client reads through the same cache. Cached per (variable, facet_id), keyed and expiring by the process-wide `graph_generation` token: `GRAPH_CACHE_MAX_AGE_SECONDS` (86,400 s / 24h by default) while the token is known, the much shorter `GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS` (3,600 s / 1h by default) while it is unknown. Bounded to `SEARCH_SDMX_FACET_CACHE_SIZE` entries (512 by default, 0 disables the cache), LRU-evicted, with single-flight: two callers requesting the same key concurrently issue one fetch, and the second awaits the first's own result or failure rather than doubling the read.

Observation reads through `materialise_preview` use a second client, `clients/materialise_bridge.py`'s `build_materialise_sdmx_client`, a `CachingMaterialiseSdmxClient` keyed on the read plan and expiring on the same generation token. `SEARCH_MATERIALISE_SDMX_CACHE_MAX_BYTES` (64 MiB, with 0 disabling the cache) bounds the Arrow bytes it keeps, and `SEARCH_MATERIALISE_SDMX_CACHE_MAX_ENTRY_BYTES` (8 MiB) bounds one read, which is served and not kept when larger.

## Concurrency

Every DCP client in this service is synchronous httpx, one `httpx.Client` per instance, HTTP/1.1, run on a thread pool behind an async door handler. Nothing in the repository issues an async HTTP call to the platform.

Each pool's width is its own setting. The Datasets doors submit to a read executor and a work executor that `datasets_api.deps` builds once per process, so a queue of materialisations cannot delay a poll or a download:

- `DATASETS_READ_MAX_CONCURRENCY` (8): the read executor's threads and its same-sized admission semaphore, for snapshot, observations, operations polls, members, pin, register, recipe and head-hit reads. A call that finds no free permit is refused at once with the retryable `service_unavailable`, 503, never queued.
- `DATASETS_WORK_MAX_CONCURRENCY` (4): the work executor's threads and its same-sized admission semaphore, for mints, definition and options rebuilds, frame refreshes and publishes, with the same refusal. A publish takes its permit before its body is read and holds it until the build ends.
- `DATASETS_MINT_MEMBER_MAX_CONCURRENCY` (4): how many members, or shape-chunks, of one request fetch SDMX at once, shared by every door that materialises. The surface's one `dc_sdmx.SdmxClient` caps the GETs in flight across all of them at its default total concurrency of 8.
- `DATASETS_MATERIALISE_MAX_CONCURRENCY` (2): the permits bounding how many frames are alive at once on an instance, independent of which door or executor admitted the call.
- `SEARCH_MAX_VARIABLE_CONCURRENCY` (8), `SEARCH_OBS_PREVIEW_CONCURRENCY` (8), `SEARCH_OBS_PREVIEW_TOTAL_CONCURRENCY` (32): search's own per-query and process-wide caps.
- `SEARCH_NODE_ARC_CALL_CONCURRENCY` (16), `SEARCH_NODE_ARC_TOTAL_CONCURRENCY` (32): the same per-call and process-wide split for the shared `dc_graph.GraphClient`'s chunked `node_arcs_batch`/`node_observation_properties_batch` reads.
- `SEARCH_SDMX_CHUNK_CONCURRENCY` (8): how many chunk GETs one chunked `SdmxClient.data()` call runs at once.
- `SEARCH_SDMX_TOTAL_CONCURRENCY` (16): how many SDMX GETs run at once across every concurrent call in the process, regardless of how many chunks any one of them submits.
- `GRAPH_NODE_POOL_SIZE` (4 by default, 16 in the deployed service): the graph door's own pool.
- `PRESENTATION_COMPONENTS_MAX_CONCURRENCY` and `PRESENTATION_FRAMES_MAX_CONCURRENCY` (4 each).

Inside one group materialise, resolve stays a single sequential pass over the batched shape read described above. The fan-out that overlaps work is the member or chunk fetch. A fresh thread pool sized to `DATASETS_MINT_MEMBER_MAX_CONCURRENCY` primes with that many chunks and refills one at a time as each resolves, waited on in submission order. The overlap is real. It is not just apparent from totals. Measured with per-call timestamps, a fetch's own SDMX calls summed 9.9 to 10.3 seconds of work across a wall span of about 4 seconds, roughly 2.4 times the span, the signature of several fetches genuinely running at once.

`DATASETS_MATERIALISE_MAX_CONCURRENCY` bounds the frames alive at once across every Datasets door, whichever executor runs it (`datasets_api.admission.MaterialisePermits`). A serve holds a permit from before its first SDMX request to its last artifact write, a head-hit render holds one while it reads and decodes a stored frame, and a graph release swap holds every permit (`MaterialisePermits.every_permit`), so the old and new release's views never sit beside a frame. The work executor has four threads, so without the permits four rebuilds could hold frames at once. At most `DATASETS_MATERIALISE_MAX_CONCURRENCY` do, and the rest read stored records or wait for a permit. The Datasets worker has a single permit per process. The apis service runs with 4 GiB of memory. A rebuild peaks at 266 MB of heap on top of a roughly 150 MB process baseline, and the default of 2 stays until a load test measures resident memory per materialisation. A request that cannot acquire a permit within `DATASETS_MATERIALISE_WAIT_SECONDS` (30) gives up and returns the retryable `service_unavailable`, 503. The request's own deadline caps that wait further, so the wait ends no later than the mint or definition door's own timeout. It does not block indefinitely.

`SEARCH_SDMX_TOTAL_CONCURRENCY` bounds GETs across every `dc_sdmx.SdmxClient` instance search builds, not one instance's own chunk pool alone: search constructs two (`LiveSdmxClient`'s own client, and `materialise_bridge.build_materialise_sdmx_client`'s), and both are handed the same `threading.BoundedSemaphore` object rather than each owning an independent one sized off the same int, since two independent N-slot semaphores would double the effective process-wide cap. Without it, one search request running several role-bearing facets concurrently, each opening its own chunked SDMX read at `SEARCH_SDMX_CHUNK_CONCURRENCY` GETs, can multiply into far more concurrent GETs than the mixer tolerates well. Measured against the dev mixer with 520 real pass-2 chunk GETs, 3 runs per level: wall time and the p95 per-GET latency both stay flat from 8 to 16 in flight (14.4s / 0.35s down to 7.75s / 0.40s), then p95 climbs about 65% between 16 and 24 in flight while wall time stops improving (7.6s at 24, 5.7s at 32), marking where the mixer starts queueing; 0 errors (429, 5xx, or timeout) were observed at any level up to 32. `SEARCH_SDMX_TOTAL_CONCURRENCY` defaults to 16, the measured point before queueing starts.

The head read path sits in front of all of this. `DATASETS_HEAD_RECHECK_SECONDS` (900, fifteen minutes) is how long a stored record serves with no graph read at all. `DATASETS_HEAD_MAX_AGE_SECONDS` (86,400, twenty-four hours), counted from the head's last materialisation, is the outer bound past which a request rebuilds unconditionally, the backstop for the changes the dependency fingerprint cannot see. A head whose StatVars and sources are held by DFO imports alone records the import key of its graph release (the import, the code index and the coverage role axes it read) and keeps past that bound while the bound release carries that import, up to `DATASETS_HEAD_IMPORTED_DATA_MAX_AGE_SECONDS` (604,800, seven days); a release carrying a new import rebuilds it. A call that a current head can answer as stored runs on the read executor (`current.serve_fresh_head`), and every other call is handed to the work executor.

## Measured numbers

### Route comparison

Same workload throughout: one StatVar, 254 entities, the full time series, 30,540 rows, measured in-process against the dev DC service.

| Route                    |     Bytes |         Time (s) | Bytes/row | Time per 100k rows, scaled (s) |
| ------------------------ | --------: | ---------------: | --------: | -----------------------------: |
| SDMX `/data`             | 5,019,123 | 1.53 (TTFB 0.68) |     164.3 |                           5.01 |
| `/v2/observation`        | 1,073,211 |             1.32 |      35.1 |                           4.32 |
| Agent `get_observations` |   345,651 |             1.55 |      32.7 |                           5.06 |
| `/v2/internal/sdmx/data` | 1,787,445 |             1.25 |      58.5 |                           4.11 |

The agent RPC's own curated subset accounts for the gap. It returned 10,586 rows against the other three routes' 30,540, and its bytes-per-row and scaled-time figures describe a smaller answer than the other three routes return.

### Door timings, first request

In-process against the dev DC service, before and after the resolve batching, facet-probe batching, and SDMX chunking above landed.

| Door           | Spec                  | Before (s) | After (s) |
| -------------- | --------------------- | ---------: | --------: |
| Definition GET | 100k rows, 10 members |      10.60 |      7.16 |
| Definition GET | 250k rows, 44 members |      30.78 |     15.11 |
| Options GET    | 100k rows, 10 members |       6.56 |      3.99 |
| Options GET    | 250k rows, 44 members |      24.81 |      8.26 |

Measured on the earlier 1 vCPU, 512 MiB configuration, the deployed Cloud Run instance reproduced the 100k numbers closely (mint 12.8 seconds, definition GET 10.2 to 11.0 seconds, options GET 9.2 to 9.6 seconds) and failed both 250k mint attempts with a plain HTTP 503 from the platform's own front end after 20 and 31 seconds. The apis service now runs with 1 vCPU, 4 GiB of memory and startup CPU boost.

Peak Python heap for one rebuild is 127 MB at 100k and 266 MB at 250k, on top of a roughly 150 MB process baseline. These are heap peaks, not resident memory, so they do not size the permit count, and `DATASETS_MATERIALISE_MAX_CONCURRENCY` stays at its default of 2 until a load test measures resident memory per materialisation. The apis service runs at 1 vCPU and 4 GiB, with one to five instances.

### The head path

100k and 250k specs, in-process against the dev DC service. Mint primes the head, so the first GET after mint is a head hit.

| Route      | Condition                   | 100k wall (s) | 250k wall (s) | Served from | Graph calls (100k / 250k) | SDMX calls (100k / 250k) |
| ---------- | --------------------------- | ------------: | ------------: | ----------- | ------------------------: | -----------------------: |
| Definition | first GET after mint        |          0.02 |          0.09 | head        |                     0 / 0 |                    0 / 0 |
| Definition | within the recheck interval |          0.02 |          0.10 | head        |                     0 / 0 |                    0 / 0 |
| Options    | within the recheck interval |          0.02 |          0.09 | head        |                     0 / 0 |                    0 / 0 |

A head past the recheck interval costs one graph read to compare the dependency fingerprint, and a head past the maximum age rebuilds. `scripts/bench_datasets.py` measures the head paths against a deployed service from the `Server-Timing` counts `sdmx_requests`, `graph_requests` and `db_transactions`.

## Limits outside this repository

- **The mixer, through the platform's own fork.** No response compression on any route measured here, SDMX or `/v2/observation` alike, even when the client offers gzip. No cache validators: no `ETag`, `Last-Modified`, or `Cache-Control`, and a conditional request carrying `If-None-Match` or `If-Modified-Since` still returns a full 200. SDMX `/data` carries no row cap of its own. The guard this repository applies against a runaway fetch is a defence added on this side. It is not a limit the platform enforces. Every request costs roughly half a second to a second regardless of size, so the number of requests matters more than their size. Two changes are worth asking for: response compression, and either a `Last-Modified` header or an `ETag`, either of which would let a currency check skip its own graph read on an unchanged resource.
- **Federation, through the platform's own configuration.** Every query the dev mixer answers also reaches the public Data Commons service, and that hop sits on the request's own critical path.
- **Service sizing, through the infrastructure project.** The DC service and this service each run at whatever CPU and memory that project allocates. The apis service runs at 1 vCPU and 4 GiB of memory with startup CPU boost, one to five instances (`infra/stacks/apis/main.tf`). The Datasets worker runs at 2 vCPU and 4 GiB, one operation per instance, up to four instances across revisions, and scales to zero unless `datasets_worker_min_instances` says otherwise (`infra/stacks/apis/datasets_worker.tf`). The Postgres budget (`infra/stacks/platform/main.tf`) is 12 connections per apis instance, a Datasets pool of 10 and a Presentation pool of 2, and 10 per worker instance. With the website and ai services, 10 for the `datasets-maintenance` job, and a reserve of 10, that sums to 165 connections, each service counted at its service-level instance cap, against `max_connections` 200 in dev and 300 in prod. The 250k failures measured above came from the earlier 512 MiB configuration.
- **Provenance refresh properties, through the ingestion pipeline and its import configuration.** A source's `lastDataRefreshDate`, `nextDataRefreshDate`, and `sourceReleaseFrequency` are graph properties an import writes. They exist only where an import chose to write them. They do not exist on every source.
- **Agent RPC caps, on the mixer itself.** `get_observations` takes one variable per call. `get_variable_metadata` caps at ten variables by ten entities.

## Measured before deciding

Two changes sit unadopted because the measurement that would justify either one has not run yet.

A shared HTTP/2 client per base URL, replacing the separate HTTP/1.1 connections each client opens today, would need a comparison run at the group fetch's own real pool width, full-size member requests, four in flight, over one shared HTTP/2 client against the same requests over separate HTTP/1.1 connections, measured on wall time and per-stream throughput. The one HTTP/2 measurement run so far split a single fetch into small entity-scoped chunks over one shared client and does not answer that question.

A member pool wider than the current four would need a sweep at four, six, and eight workers against the 250k spec, measuring wall time and CPU-busy time together. The CSV parse runs inside each pooled fetch's own thread, on the same thread that is also waiting on the network, and roughly a third of the door's own time is CPU-bound work competing for the interpreter. It is not idle time on the wire. Widening the pool without that measurement risks trading network idle time for interpreter contention. It may not shorten the door.
