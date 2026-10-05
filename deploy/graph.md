# Graph API

The Graph surface reads DCP (Data Commons Platform) directly. It holds no persistence layer and no migrations of its own, unlike the Datasets API (see `deploy/postgres.md`).

## Mounting the Graph HTTP surface

`DFO_GRAPH_SURFACE` (default: unset, off) is read by `dfo_apis.app.deployed_surfaces()`, not by `graph_api` itself. Set it to `1` or `true` (case-insensitive) to mount the Graph API's three doors: `GET /api/graph/healthz`, `POST /api/graph/resolve`, and `GET /api/graph/node/{dcid}`. They mount alongside Search (and Datasets, when enabled) on the deployed service.

The surface's own lifespan requires `GRAPH_DCP_BASE_URL` to already be set once the flag is on. There is no safe default, so startup fails fast instead of a request failing mid-flight. It builds a live `dc_graph.GraphClient` from it to read DCP.

`GRAPH_DCP_TOKEN` is optional, sent as the `x-graph-token` header on every DCP read. It is the same WAF-skip secret Search's `SEARCH_GRAPH_TOKEN` and Datasets' `DATASETS_GRAPH_TOKEN` send to their own graph endpoints. DCP's WAF challenges traffic from datacenter IP ranges (e.g. CI runners), which a non-browser client cannot answer, and a WAF rule keyed on this header exempts trusted automated traffic.

A deployment that hasn't provisioned `GRAPH_DCP_BASE_URL` must leave `DFO_GRAPH_SURFACE` off, or the service fails to serve any request once the surface's lifespan runs.

## The resolvekit cache

`POST /api/graph/resolve` resolves entity names through resolvekit's composed SQLite store, opened by `entity_resolution.acquire_shared()`. The store is built into the image, over `entity_resolution.MODULE_IDS`: the four bundled country-level geo modules (`geo.countries`, `geo.regions`, `geo.continents`, `geo.continental_unions`) plus `dfo_graph`, a custom pack built at image-build time from this repo's own committed graph export (`packages/dfo-data-tools/src/entity_resolution/data/graph_entities.json`). Both the bundled packs and the built `dfo_graph` pack are WAL-mode, and opening a WAL database means writing `-shm`/`-wal` sidecars beside it, which Cloud Run's read-only filesystem forbids, so composing the store at startup fails there. The build stage builds the `dfo_graph` pack, composes and warms the full module set once, converts every affected SQLite file to rollback-journal mode, and bakes the result into the image at `/app/resolvekit-cache`. `RESOLVEKIT_CACHE_DIR` points there in both Docker stages. The Dockerfile sets it, and a deployment should leave it alone.

The geo modules compose into one cached SQLite file. `dfo_graph` is the "org" domain's only loaded module (the bundled `org.data_sources` pack is not loaded), so it takes resolvekit's single-pack shortcut: its own `entities.sqlite` is opened directly, with no compose step and no composed-store cache file of its own.

The composed geo store's cache key covers each source pack's own identity, file size and modification time, alongside a cache-format version resolvekit bumps when its composition logic changes. Upgrading resolvekit therefore changes the key, which is why this depends on the CI startup test staying green. If the Graph surface crash-loops on a `sqlite3.OperationalError` at an `ATTACH DATABASE`, the cause is a cache-key or cache-file problem rather than a new resolvekit bug. A cache miss cannot recover on a read-only filesystem, so every underlying cause surfaces at the same pack ATTACH. The message varies with what went wrong. `attempt to write a readonly database` and `unable to open database file` are two seen in practice, and neither names the real cause.

A cache hit still sha256s every bundled pack artifact on every startup. `dfo_graph`, a BYOD (bring-your-own-data) pack, carries no checksum in its own `metadata.json` and skips that check.
