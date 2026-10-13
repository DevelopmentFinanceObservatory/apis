# Graph API

The Graph surface reads DCP (Data Commons Platform) directly. It holds no persistence layer and no migrations of its own, unlike the Datasets API (see `deploy/postgres.md`).

## Mounting the Graph HTTP surface

`DFO_GRAPH_SURFACE` (default: unset, off) is read by `dfo_apis.app.deployed_surfaces()`, not by `graph_api` itself. Set it to `1` or `true` (case-insensitive) to mount the Graph API's routes: `GET /api/graph/healthz`, `GET /api/graph/capabilities`, `GET /api/graph/schema`, `POST /api/graph/resolve`, `GET /api/graph/node/{dcid}`, `POST /api/graph/nodes`, `GET /api/graph/node/{dcid}/members`, `GET /api/graph/roots`, `GET /api/graph/documentation/{dcid}`, and `GET /api/graph/sources`. They mount alongside Search (and Datasets, when enabled) on the deployed service.

The surface's own lifespan requires `GRAPH_DCP_BASE_URL` to already be set once the flag is on. There is no safe default, so startup fails fast instead of a request failing mid-flight. It builds the origin catalogue's graph client from it to read DCP, and acquires the process's shared facet profile cache over the same root for graph and SDMX availability reads (`materialise.shared_profiles`). Search and Datasets share that cache when mounted beside it and naming the same roots; a surface naming other roots gets a cache of its own. Surface construction also loads the origin catalogue, so a missing or unloadable `ORIGIN_CATALOGUE_URI` fails startup.

`GRAPH_DCP_TOKEN` is optional, sent as the `x-graph-token` header on every DCP read and every SDMX availability read. It is the same WAF-skip secret Search's `SEARCH_GRAPH_TOKEN` and Datasets' `DATASETS_GRAPH_TOKEN` send to their own graph endpoints. DCP's WAF challenges traffic from datacenter IP ranges (e.g. CI runners), which a non-browser client cannot answer, and a WAF rule keyed on this header exempts trusted automated traffic.

A deployment that hasn't provisioned `GRAPH_DCP_BASE_URL` must leave `DFO_GRAPH_SURFACE` off, or the service fails to serve any request once the surface's lifespan runs.

## Settings

The surface reads these at import or startup. Only `GRAPH_DCP_BASE_URL` has no default.

| Variable                        | Default                | Effect                                                                                                                                                 |
| :------------------------------ | :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GRAPH_DCP_BASE_URL`            | none                   | DCP API root for graph reads and, under the same root, the mixer's SDMX availability reads. An empty value fails startup.                              |
| `GRAPH_DCP_TOKEN`               | empty                  | Optional `x-graph-token` WAF-skip header on DCP and SDMX reads.                                                                                        |
| `GRAPH_NODE_POOL_SIZE`          | `4`                    | Worker count of the DCP-read executor that every graph read shares. It admits one read per worker and refuses the next with a retryable `503`.         |
| `GRAPH_RESOLVE_TIMEOUT_SECONDS` | `30`                   | Deadline of `POST /api/graph/resolve`, covering the resolvekit job and every DCP read it makes.                                                        |
| `GRAPH_NODE_TIMEOUT_SECONDS`    | `10`                   | Deadline of the node, batch, documentation and members reads. A variable card waits for its name and currency reads until just before it.             |
| `GRAPH_SDMX_TIMEOUT_SECONDS`    | `30`                   | HTTP timeout of the UN deployment's SDMX client the node reader uses.                                                                                  |
| `UN_GRAPH_BASE_URL`             | UN System Data Commons | API root that serves `undata/` dcids and their availability.                                                                                           |
| `ORIGIN_CATALOGUE_URI`          | none                   | Origin catalogue root. A missing or unloadable catalogue fails startup.                                                                                |
| `COVERAGE_ARTIFACT_URI`         | unset                  | Coverage artifact root. Unset, a missing `coverage` extra or an unreadable artifact leaves coverage out: a variable's `availability` is `null` and a source document serves no currencies or granularity. Startup continues. |

A running read keeps its executor worker until it returns, after its request has timed out, so a slow upstream fills the pool and raises `503` for later requests. The executor logs `thread pool graph-dcp saturated` when its workers fill. `GET /api/graph/capabilities` publishes the pool size and the timeouts, and its `data_builds` names the graph generation, origin catalogue, coverage index and code index the process serves.

## The resolvekit cache

`POST /api/graph/resolve` resolves entity names through resolvekit's composed SQLite store, opened by `entity_resolution.acquire_shared()`. The store is built into the image, over `entity_resolution.MODULE_IDS`: the four bundled country-level geo modules (`geo.countries`, `geo.regions`, `geo.continents`, `geo.continental_unions`) plus `dfo_graph`, a custom pack built at image-build time from this repo's own committed graph export (`packages/dfo-data-tools/src/entity_resolution/data/graph_entities.json`). Both the bundled packs and the built `dfo_graph` pack are WAL-mode, and opening a WAL database means writing `-shm`/`-wal` sidecars beside it, which Cloud Run's read-only filesystem forbids, so composing the store at startup fails there. The build stage builds the `dfo_graph` pack, composes and warms the full module set once, converts every affected SQLite file to rollback-journal mode, and bakes the result into the image at `/app/resolvekit-cache`. `RESOLVEKIT_CACHE_DIR` points there in both Docker stages. The Dockerfile sets it, and a deployment should leave it alone.

The geo modules compose into one cached SQLite file. `dfo_graph` is the "org" domain's only loaded module (the bundled `org.data_sources` pack is not loaded), so it takes resolvekit's single-pack shortcut: its own `entities.sqlite` is opened directly, with no compose step and no composed-store cache file of its own.

The composed geo store's cache key covers each source pack's own identity, file size and modification time, alongside a cache-format version resolvekit bumps when its composition logic changes. Upgrading resolvekit therefore changes the key, which is why this depends on the CI startup test staying green. If the Graph surface crash-loops on a `sqlite3.OperationalError` at an `ATTACH DATABASE`, the cause is a cache-key or cache-file problem rather than a new resolvekit bug. A cache miss cannot recover on a read-only filesystem, so every underlying cause surfaces at the same pack ATTACH. The message varies with what went wrong. `attempt to write a readonly database` and `unable to open database file` are two seen in practice, and neither names the real cause.

A cache hit still sha256s every bundled pack artifact on every startup. `dfo_graph`, a BYOD (bring-your-own-data) pack, carries no checksum in its own `metadata.json` and skips that check.
