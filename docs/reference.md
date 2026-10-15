# API reference

Routes, platform headers, and configuration for the DFO data APIs. For a guided tour of the public API, start with the [API guide](guide/quickstart.md).

## Endpoints

Every API surface mounts onto the shared platform app. The public deployment serves each route under `/api/v1` as well as under `/api` (see [API conventions](guide/conventions.md)). The tables use the unversioned form. Application error responses use RFC 9457 problem details. `GET /codes/{code}` serves codes present in the packaged `dfo_spec_kit` registry or the merged refusal catalogue. The registry supplies classification, definition, lifecycle, and mappings. The catalogue supplies title and detail. Fields from an absent source are `null`, and `mappings` is empty when the registry has no entry for the code.

### Search API (`/api/search`)

Search queries resolve against the Data Commons graph. Observation previews and candidate coverage are computed in memory via the `materialise` bridge.

| Method | Endpoint                   | Description                                                                                                                                                                    |
| :----- | :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/api/search/resolve`      | Resolves natural-language queries into structured specifications with observation previews. Accepts `Accept: text/event-stream` for Server-Sent Events (SSE) or standard JSON. |
| `POST` | `/api/search/options`      | Computes candidate entity options on demand for a resolved entity role (`origin`, `destination`, `observationAbout`).                                                          |
| `GET`  | `/api/search/healthz`      | Liveness probe returning status and search engine build metadata.                                                                                                              |
| `GET`  | `/api/search/schema`       | JSON Schema for search request and response contracts.                                                                                                                         |
| `GET`  | `/api/search/capabilities` | Deployment capabilities, supported recipe classes, limits, and door liveness.                                                                                                  |
| `GET`  | `/api/search/catalogue`    | What DFO holds (counts) and checked example sentences, globally or at one country (`?place=`). 503 until a search catalogue build loads.                                       |

### Datasets API (`/api/datasets`)

Gated by `DFO_DATASETS_SURFACE=true` (or `1`). Requires PostgreSQL persistence (`dataset_store`) and GCS artifact storage.

| Method | Endpoint                                   | Description                                                                                                                                                                                                            |
| :----- | :----------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/api/datasets`                            | Mints a dataset and materialises observation tables. Requests over the synchronous budgets and within the asynchronous ceilings return a pollable operation. Requests beyond either ceiling return 422 `over_ceiling`. |
| `POST` | `/api/datasets/register`                   | Persists a canonical dataset definition without materialisation.                                                                                                                                                       |
| `GET`  | `/api/datasets/{id}`                       | Serves the latest dataset manifest for a canonical subject ID (`spec_...`). It rechecks the head and can materialise a new version.                                                                                    |
| `GET`  | `/api/datasets/{id}/{content_version}`     | Fetches metadata and artifact download links (CSV, Parquet, XLSX) for a versioned snapshot.                                                                                                                            |
| `POST` | `/api/datasets/{id}/{content_version}/pin` | Promotes a working-tier snapshot to pinned durability. Requires first-party principal authentication.                                                                                                                  |
| `GET`  | `/api/datasets/operations/{operation_id}`  | Polls a pending or running operation, or returns its completed manifest. Failed operations return a problem response. Unknown IDs return 404 `operation_not_found`.                                                    |
| `GET`  | `/api/datasets/healthz`                    | Liveness probe returning status and materialiser build metadata.                                                                                                                                                       |
| `GET`  | `/api/datasets/schema`                     | JSON Schema for dataset envelopes, manifests, mint requests, and snapshots.                                                                                                                                            |
| `GET`  | `/api/datasets/capabilities`               | Supported schema versions, admission budgets, asynchronous ceilings, and format support.                                                                                                                               |

### Graph API (`/api/graph`)

Gated by `DFO_GRAPH_SURFACE=true` (or `1`). Communicates directly with the Data Commons platform instance.

| Method | Endpoint                          | Description                                                                                                                                                                                |
| :----- | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/api/graph/resolve`              | Resolves free-text names to graph references. Each result is `resolved`, `ambiguous`, `not_found` or `not_held`, and a group or region carries its member count and a link to its members. |
| `GET`  | `/api/graph/node/{dcid}`          | Describes a node under one envelope, with a detail block named for its `kind` (one of 16, such as `stat_var`, `country`, `group` or `provenance`).                                         |
| `POST` | `/api/graph/nodes`                | Describes up to 50 nodes in one read, each as the GET body or `{ref, status: "not_held"}`.                                                                                                 |
| `GET`  | `/api/graph/node/{dcid}/members`  | Pages the codes, children, datasets, variables, members, countries or facets of a node under a bound cursor.                                                                               |
| `GET`  | `/api/graph/roots`                | Lists the code systems, groups, regions and sources to start listing from.                                                                                                                 |
| `GET`  | `/api/graph/documentation/{dcid}` | Retrieves source documentation for a Dataset or Provenance node.                                                                                                                           |
| `GET`  | `/api/graph/sources`              | Lists the sources this deployment admits, with their datasets and provenances.                                                                                                             |
| `GET`  | `/api/graph/schema`               | JSON Schema of every request and response body.                                                                                                                                            |
| `GET`  | `/api/graph/healthz`              | Liveness probe returning status and capability version.                                                                                                                                    |
| `GET`  | `/api/graph/capabilities`         | Door liveness, kinds, input limits, and the data builds in use.                                                                                                                            |

### Presentation API (`/api/presentation`)

Gated by `DFO_PRESENTATION_SURFACE=true` (or `1`). Reads from `presentation_store` and `dataset_store`.

| Method  | Endpoint                                               | Description                                                                       |
| :------ | :----------------------------------------------------- | :-------------------------------------------------------------------------------- |
| `GET`   | `/api/presentation/frames/{spec_id}/{content_version}` | Retrieves assembled presentation frame data for a rendered dataset specification. |
| `POST`  | `/api/presentation/components`                         | Creates a presentation component record and returns its write token once.         |
| `GET`   | `/api/presentation/components/{component_id}`          | Retrieves a stored presentation component record.                                 |
| `PATCH` | `/api/presentation/components/{component_id}`          | Modifies component layout, properties, or bindings. Needs `x-dfo-write-token`.    |
| `GET`   | `/api/presentation/release`                            | Retrieves active release manifest and intent schemas.                             |
| `GET`   | `/api/presentation/healthz`                            | Liveness probe returning status and build metadata.                               |
| `GET`   | `/api/presentation/capabilities`                       | Capability version hash, catalog_version, and build for the served release.       |

### Skills API (`/api/skills`)

Gated by `DFO_SKILLS_SURFACE=true` (or `1`). Serves the published skills catalogue from a process-wide index shared with Search, rebuilt on a timer with no graph read per request.

| Method | Endpoint                                  | Description                                                                                             |
| :----- | :---------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `GET`  | `/api/skills`                             | Every published skill's front matter, breadth figures, and warnings, read from the shared skills index. |
| `GET`  | `/api/skills/{skill_id}`                  | One skill's front matter, breadth figures, warnings, inline body, and resource listing.                 |
| `GET`  | `/api/skills/{skill_id}/resources/{path}` | One resource's inline content, matched by exact path against that skill's own listing.                  |
| `GET`  | `/api/skills/healthz`                     | Liveness probe returning status and capability version.                                                 |

### Shared platform

| Method | Endpoint        | Description                                                                             |
| :----- | :-------------- | :-------------------------------------------------------------------------------------- |
| `GET`  | `/codes/{code}` | Definition for every code present in the packaged registry or merged refusal catalogue. |

Every response carries a fixed set of platform headers:

| Header                      | Present on                                                                                                                                                                                                       |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x-dfo-trace-id`            | Every response except one marked shared-cacheable.                                                                                                                                                               |
| `x-dfo-commit`              | Every response.                                                                                                                                                                                                  |
| `Server-Timing`             | Every response that is neither an event stream nor shared-cacheable, carrying a `total` metric first, then per-surface phase names where a surface reports them (for example datasets' `render`/`store`/`sign`). |
| `Strict-Transport-Security` | Every response.                                                                                                                                                                                                  |

Every surface's `GET /healthz` body is `{status, build, capability_version}`, with `build` equal to the commit the `x-dfo-commit` header carries. Search's success bodies additionally nest `diagnostics.trace_id` (matching the header) and `diagnostics.timing_ms` (`total` plus a per-step breakdown). A streamed `POST /api/search/resolve` response carries its own total in the terminal `done` event's `elapsed_ms` rather than a `Server-Timing` header, since headers go out before the stream's first byte and the request's total duration is not yet known at that point.

## Configuration

Configuration variables load from environment variables or `.env` at boot.

### Core and Search API

| Variable                                         | Default                           | Purpose                                                                                                                                                                                                                                                        |
| :----------------------------------------------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`                                 | _None_                            | Required. API key for Gemini inference.                                                                                                                                                                                                                        |
| `SEARCH_GRAPH_BASE`                              | _None_                            | Required. Bare host of the Data Commons service the search engine reads.                                                                                                                                                                                       |
| `SEARCH_GRAPH_TOKEN`                             | `""`                              | Optional `x-graph-token` header for graph WAF exemption.                                                                                                                                                                                                       |
| `SEARCH_ENGINE_MODEL`                            | `gemini-3.5-flash-lite`           | Gemini model variant used for natural-language query resolution.                                                                                                                                                                                               |
| `SEARCH_REQUEST_TIMEOUT_S`                       | `45`                              | Request deadline in seconds for search resolution.                                                                                                                                                                                                             |
| `DFO_GRAPH_GENERATION_SERVICE`                   | `""`                              | Cloud Run Admin API v2 resource name (`projects/<p>/locations/<r>/services/<s>`) of the Data Commons service `graph_generation` reads the serving revision from. Unset degrades every graph-derived cache to the unknown generation (loud warning at startup). |
| `DFO_GRAPH_GENERATION`                           | `""`                              | Fixed generation token overriding the live Cloud Run read entirely, for a local graph mirror with no Cloud Run service of its own. Takes precedence over `DFO_GRAPH_GENERATION_SERVICE` when both are set.                                                     |
| `GRAPH_CACHE_MAX_AGE_SECONDS`                    | `86400`                           | Lifetime, in seconds, of a graph-derived cache entry captured under a known generation token (24h). At most `86400`, the UN backend's cache day; a larger value fails startup.                                                                                 |
| `GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS` | `3600`                            | Lifetime, in seconds, of a graph-derived cache entry captured while the generation is unknown (1h). Must be \<= `GRAPH_CACHE_MAX_AGE_SECONDS`.                                                                                                                 |
| `DFO_CORS_ORIGINS`                               | `""`                              | Comma-separated list of allowed CORS origins. Dev leaves it empty, which disables CORS headers. Wildcard origins are not used. The infra repository sets it per environment.                                                                                   |
| `DFO_EXECUTOR_MAX_WORKERS`                       | `32`                              | Thread pool worker count for async I/O offloading.                                                                                                                                                                                                             |
| `PORT`                                           | `8083` (dev) / `8080` (container) | Port bound by the server process.                                                                                                                                                                                                                              |
| `DFO_BUILD_COMMIT`                               | `unknown`                         | Commit sha the running image was built from, baked in by the Dockerfile from a build argument and carried by the `x-dfo-commit` header.                                                                                                                        |

### Surface mount toggles

| Variable                   | Default | Purpose                                               |
| :------------------------- | :------ | :---------------------------------------------------- |
| `DFO_DATASETS_SURFACE`     | `""`    | Mounts `/api/datasets` when set to `true` or `1`.     |
| `DFO_GRAPH_SURFACE`        | `""`    | Mounts `/api/graph` when set to `true` or `1`.        |
| `DFO_PRESENTATION_SURFACE` | `""`    | Mounts `/api/presentation` when set to `true` or `1`. |
| `DFO_SKILLS_SURFACE`       | `""`    | Mounts `/api/skills` when set to `true` or `1`.       |

### Datasets API and persistence

| Variable                             | Default           | Purpose                                                                                                                      |
| :----------------------------------- | :---------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| `DATASETS_SYNC_BUDGET_ROWS`          | `100000`          | Row budget for inline materialisation. Over-budget requests can become operations within the asynchronous ceilings.          |
| `DATASETS_SYNC_BUDGET_REQUESTS`      | `40`              | SDMX request budget for inline materialisation. Over-budget requests can become operations within the asynchronous ceilings. |
| `DATASETS_ASYNC_MAX_ROWS`            | `2000000`         | Row ceiling for an asynchronous operation.                                                                                   |
| `DATASETS_ASYNC_MAX_REQUESTS`        | `800`             | SDMX request ceiling for an asynchronous operation.                                                                          |
| `DATASETS_SDMX_TIMEOUT_SECONDS`      | `90`              | Independent httpx transport timeout for SDMX calls. The door deadline applies when lower.                                    |
| `DATASETS_GRAPH_BASE_URL`            | _None_            | Graph endpoint for dataset materialisation readers.                                                                          |
| `DATASETS_SDMX_BASE_URL`             | _None_            | SDMX endpoint for observation retrieval in datasets.                                                                         |
| `DATASETS_GRAPH_TOKEN`               | `""`              | Optional `x-graph-token` header for dataset graph and SDMX calls.                                                            |
| `DATASETS_ARTIFACT_BUCKET_WORKING`   | _None_            | GCS bucket for working-tier materialised files.                                                                              |
| `DATASETS_ARTIFACT_BUCKET_PINNED`    | _None_            | GCS bucket for pinned-tier materialised files.                                                                               |
| `DATASETS_PRINCIPAL_HEADER`          | `x-dfo-principal` | Request header containing caller identity for pin and publish operations.                                                    |
| `DATASETS_FIRST_PARTY_PRINCIPAL`     | `""`              | Authorized principal permitted to publish datasets.                                                                          |
| `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` | `""`              | Authorized principal permitted to pin datasets, independent of `DATASETS_FIRST_PARTY_PRINCIPAL`.                             |
| `LIVE_REFRESH_PRINCIPAL`             | `""`              | Authorized principal permitted to refresh a dataset frame and render a component on another frame.                           |
| `DATASET_STORE_HOST`                 | `127.0.0.1`       | PostgreSQL host for `dataset_store`.                                                                                         |
| `DATASET_STORE_PORT`                 | `5432`            | PostgreSQL port for `dataset_store`.                                                                                         |
| `DATASET_STORE_DATABASE`             | `datasets`        | PostgreSQL database name for `dataset_store`.                                                                                |
| `DATASET_STORE_RUNTIME_DB_USER`      | _None_            | Runtime IAM/DB user for dataset persistence.                                                                                 |

### Graph and Presentation APIs

| Variable                             | Default        | Purpose                                                                   |
| :----------------------------------- | :------------- | :------------------------------------------------------------------------ |
| `GRAPH_DCP_BASE_URL`                 | _None_         | Upstream Data Commons instance URL for graph reads and SDMX availability. |
| `GRAPH_DCP_TOKEN`                    | `""`           | Optional `x-graph-token` header for DCP graph and SDMX calls.             |
| `PRESENTATION_STORE_HOST`            | `127.0.0.1`    | PostgreSQL host for `presentation_store`.                                 |
| `PRESENTATION_STORE_PORT`            | `5432`         | PostgreSQL port for `presentation_store`.                                 |
| `PRESENTATION_STORE_DATABASE`        | `presentation` | PostgreSQL database name for `presentation_store`.                        |
| `PRESENTATION_STORE_RUNTIME_DB_USER` | _None_         | Runtime IAM/DB user for presentation persistence.                         |
