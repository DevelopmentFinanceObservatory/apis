# Datasets API

The Datasets API registers, mints, materialises, snapshots, and pins dataset specifications for the Development Finance Observatory (DFO). It resolves statistical variable definitions against Data Commons and SDMX endpoints, executes synchronous tabular materialisation, generates Frictionless data packages, and persists rendered artifacts to Google Cloud Storage and PostgreSQL.

The surface mounts on the shared FastAPI platform at `/api/datasets`.

---

## Architecture and Doors

The surface composes six functional doors into a unified router, error classifier, and lifespan context.

```
/api/datasets
├── /healthz                             (Meta door)
├── /schema                              (Meta door)
├── /capabilities                        (Meta door)
├── /operations/{operation_id}           (Meta door)
├── /register                            (Register door)
├── /                                    (Mint door)
├── /{id}                                (Definition door)
├── /{id}/{content_version}              (Snapshot door)
└── /{id}/{content_version}/pin          (Pin door)
```

### 1. Meta Door

Handles service liveness, capability negotiation, and schema discovery.

| Method | Endpoint                           | Description                                                          |
| :----- | :--------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/datasets/healthz`            | Returns liveness status, the running build commit, and the capability version (`{status, build, capability_version}`). |
| `GET`  | `/api/datasets/schema`             | Returns contract-stable JSON Schema definitions for dataset models.  |
| `GET`  | `/api/datasets/capabilities`       | Returns runtime capabilities, format options, and the sync budget.   |
| `GET`  | `/api/datasets/operations/{op_id}` | Polls asynchronous operation state (`operation_not_found`, 404, for an unknown id). |

Every response, healthz and capabilities included, also carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing` header; `build` in both bodies equals the commit `x-dfo-commit` carries.

### 2. Register Door

Registers canonical dataset specifications directly in the registry.

| Method | Endpoint                 | Description                                                  |
| :----- | :----------------------- | :--------------------------------------------- |
| `POST` | `/api/datasets/register` | Persists a canonical specification definition.               |

Each definition's `identity` field holds a document shaped as `search.models.identity.BaseIdentity`, published at `GET /api/search/schema`.

### 3. Mint Door

Resolves input variables, registers definitions, materialises observation data synchronously, and writes artifacts to storage.

| Method | Endpoint        | Description                                                   |
| :----- | :-------------- | :---------------------------------------------- |
| `POST` | `/api/datasets` | Mints and materialises a dataset specification.               |

Supported payload request types:

- `statvar_set`: One StatVar, in `statvars`, which holds exactly one `dcid:`-prefixed entry.
- `spec_id`: Existing canonical specification identifier.

A posted `preview` field refuses `request_invalid`. Every route that takes `{id}` accepts `spec_` ids only, and a `col_` id answers 404 `not_found`.

### 4. Definition Door

Re-runs materialisation on demand for an existing registered identifier using the latest available upstream data.

| Method | Endpoint                     | Description                                                                                    |
| :----- | :--------------------------- | :--------------------------------------------------------------------------------------------- |
| `GET`  | `/api/datasets/{id}`         | Re-materialises observation data for a `spec_` identifier.                                     |
| `GET`  | `/api/datasets/{id}/options` | Returns the `OptionsEnvelope` (output options, coverage, `content_version`) without rendering. |
| `POST` | `/api/datasets/{id}/options` | The same envelope for a spec whose canonical `definition` the body carries, registered or not. |
| `POST` | `/api/datasets/{id}/members` | The members a mint of a `spec_` id serves, with their labels and the family's count and digest, without materialising. |

Query parameters:

- `format`: Target output serialization (`csv`, `xlsx`, `parquet`).
- `bundle`: Frictionless Data Package bundling boolean (`true`, `false`). A bundle is a zip of `README.md`, `data/<slug>.<ext>`, `data/codes.csv` (when the data holds codes), `SOURCES.md`, `CITATION.cff` and `datapackage.json`, built by `materialise.bundle.assemble_bundle`. A download is named `dfo_<sources-slug>_<YYYY-MM-DD>_<format>_<cv6>.zip` (`artifact_delivery.artifact_file_name`).
- `groupby.by` / `groupby.op`: The axes a sum keeps (comma-separated) and its operation (`sum`). Every other declared-summable axis the frame decoded, and `stat_var`, is summed away. When every decoded axis the sum leaves alone holds one value, the result is a grammar frame with its own `content_version`: `stat_var` is `dcid:<spec_id>` on every row and each kept code axis is a `stat_var_<code_system>` column of labels.
- `groupby.levels`: Comma-separated `<code_system>:<depth>` pairs, such as `crsPurpose:0`. Each kept code rolls up to its ancestor at that depth before the sum (0 is the root codes), so `groupby.by=time_period,crsPurpose&groupby.op=sum&groupby.levels=crsPurpose:0` gives one row per year, donor, and recipient for each of the ten CRS sector groups. A code above the depth stays as itself, its row carrying its own label; depth 0 leaves none. `GET /api/datasets/capabilities` lists the axes it accepts as `groupby_level_axes`.
- `pivot.index` / `pivot.columns`: Dimension pivoting configuration.
- `columns`: Comma-separated column projection selection. A list that names only enrichment columns keeps every default column and adds them; a list with any table column selects. Unpivoted output keeps every decoded `stat_var_<code_system>` column unless the list names some. The long layout ignores `status` and `<role>_label` (written by default; the options document still offers `<role>_label` because pivot output writes names only on request); the frame layout keeps `status`. The long layout is fixed: `stat_var` (with `stat_var_label` when no code system decodes), each entity with its `_label` and enrichments, each code system as `<code_system>` (the code) and `<code_system>_label`, then `time_period`, `value`, `unit`, `unit_label`, `currency`, `price_basis`, `base_year`, `measurementMethod`, `measurementMethod_label`, `provenance` and `provenance_label`. A name the graph lacks or a failed name read leaves its cell empty with an `enrichment_unavailable` warning. A column that is not droppable refuses the request with `column_not_droppable`; `GET /api/datasets/{id}/options` reports each column's `droppable` flag before a caller corrects the list by trial.
- `layout`: `frame` writes a bare Parquet file in the frame layout instead of the long layout: the frame's columns as reshaped, with each decoded code system as `stat_var_<code_system>` (its label), `status` kept, a converted unit as the frame grammar's JCS text of the unit request, and no name columns or name reads (`render.FRAME_LAYOUT`). It is the layout machine readers such as data-mcp request; the long layout is the download default. The artifact's `rendering` carries `layout: "frame"` as sent and no layout version, and the snapshot lists it after the long-layout files. `layout` with `bundle=true`, a pivot, or a format other than `parquet` refuses `request_invalid` at `/options/layout`.

`POST /api/datasets/{id}/options` takes `{"schema_version": 1, "definition": {...}}`, the `definition` a search response carries for the spec, so a caller can read the formats, columns, and enrichments of a search result before anyone mints it. The identity must hash to `{id}` and `definition.spec_id` must name it, or the call refuses `spec_id_mismatch`. A registered id answers as the GET route does. An unregistered definition is admitted as a `spec_id` mint admits its sidecar (`datasets_api.kinds.spec_id.admit_sidecar`) and materialised write-free: no registry row, version record, head, or artifact. The call materialises the frame, holding the shared materialise slot as a mint does, and refuses `over_ceiling` above the inline ceiling, so a caller should send it when a visitor opens a download menu and keep default options until it answers.

`POST /api/datasets/{id}/members` takes a `spec_` id and an optional body `{"schema_version": 1, "definition": {...}}`, and answers `MembersEnvelope`: each member a mint of the spec serves, sorted by `ref` (`dcid:`-prefixed, as `trace.resolved_stat_vars` names it), with its graph `label` from one batched name read, its declared `roles` and its `constraints`; `served_count`; and `family_member_count` and `family_member_digest`, equal to the `trace` fields a mint of the spec states. A registered id needs no body. An unregistered one needs the `definition`, admitted as on the options route, and nothing is written. The route runs member resolution alone, so it costs no SDMX read, and it refuses as a mint's member resolution does: `spec_not_found`, `spec_id_mismatch`, `member_not_resolvable`, `spec_not_exportable`, `role_signature_incompatible`, `definition_not_reproducible`, `facet_not_available`, `over_ceiling` (`member_cardinality` or `candidate_window`). Query keys refuse `request_invalid`.

Members are a function of the identity, the graph and the packaged code index. No request names one. A pinned identity serves its StatVar, and the code index decides the families it holds rows for. Any other described identity walks the graph's value-filtered reverse page of 4,000 subjects (`definition_candidate_cap`). A page that truncates without a confirmed root sends the walk to the whole filtered set, up to `DATASETS_MEMBER_SUBJECT_CEILING` subjects (`definition_subject_ceiling`), each confirmed by its arcs, so a family whose root sorts past the first page still settles and states its digest. A set above that ceiling refuses `over_ceiling` with `ceiling_kind: candidate_window`, and a walk whose reads run past `DATASETS_MEMBER_READ_SECONDS` answers `timeout` (504, retryable). The mint, definition retrieval, options and members routes and the operations worker share one process-level cache of resolved families, keyed on the graph generation (`statvar_members.shared_enumerator`).

### 5. Snapshot Door

Returns immutable version metadata and fresh signed download URLs for stored artifacts.

| Method | Endpoint                               | Description                                                       |
| :----- | :------------------------------------- | :---------------------------------------------------------------- |
| `GET`  | `/api/datasets/{id}/{content_version}` | Retrieves stored artifact metadata and signed URLs for a version. |

### 6. Pin Door

Promotes rendered artifacts from the temporary working tier to the pinned retention tier and records an authenticated principal reference.

| Method | Endpoint                                   | Description                                                  |
| :----- | :----------------------------------------- | :----------------------------------------------------------- |
| `POST` | `/api/datasets/{id}/{content_version}/pin` | Promotes artifacts to permanent storage and links principal. |

---

## Lifecycle and Retention

Generated artifacts default to working-tier retention and expire 30 days after initial storage. Pinned artifacts remain persisted indefinitely.

A version's `prepared_at` is written once at its first mint and stored with the version record. Every re-mint and re-render reads the stored value, so the date in a bundle's `README.md`, `CITATION.cff` and file name stays fixed.

Signed download URLs expire 15 minutes after issuance (`url_expires_at`), the lifetime a client may rely on. The credential itself is signed for 20 minutes, five minutes longer, so a URL the artifact store re-serves from its own signing cache has the full 15 minutes left when a caller receives it.

Synchronous materialisation admits on an exact `series_coverage`-based estimate (rows, SDMX requests) computed before any network call. A request whose estimate stays at or under both `DATASETS_SYNC_BUDGET_ROWS` and `DATASETS_SYNC_BUDGET_REQUESTS` (the sync budget) runs inline; an xlsx request whose estimate already exceeds Excel's row limit refuses pre-fetch, with problem code `excel_limit`.

---

## Async operations and the worker

A mint or definition request over the sync budget is recorded as an operation (`datasets.operation`) instead of running inline: the door answers `202 Accepted` with a `Location` header and an `OperationEnvelope` (`operation_id`, `status`, `poll`, `retry_after_ms`). The operation id is derived deterministically from the request identity (the subject id and, for a definition, the members the submitting door resolved, which the worker resolves again, failing the operation `upstream_version_changed`, retryable, when they differ, so a resubmission starts a new operation for the new members), the coverage stamp, the origin catalogue build bound to the request, and the rendering key of `options` (format/groupby/pivot/columns/...), so two requests for the same subject that differ only in rendering get distinct operations, a rendering-caused deterministic failure never fails a resubmit asking for a different rendering, and a new catalogue build, which can admit different provenances, starts a new operation. The row records that build as `origin_catalogue_build_id`. An estimate over `DATASETS_ASYNC_MAX_ROWS` or `DATASETS_ASYNC_MAX_REQUESTS` refuses `over_ceiling` before any operation is recorded, and a new operation at `DATASETS_OPERATION_BACKLOG_MAX` (default 20) `pending`+`running` rows refuses `service_unavailable` (503, retryable, `Retry-After`); a resubmit of an already-recorded operation is exempt. One Cloud Tasks task is created per submission: the submit, and every later poll, claims the operation's dispatch with a conditional update, and only the winner enqueues. An enqueue that fails or times out leaves the operation `pending`, and the next submit or poll after a 30-second grace enqueues again under the same task name, up to three redispatches. `GET /api/datasets/operations/{op_id}` polls it: `pending`/`running` answers 200 with progress, `done` answers 200 with the same manifest shape a synchronous mint returns (signed fresh, never off a stored URL), listing the artifact for the rendering its request posted plus statistics artifacts, and `failed` answers `operation_failed` (500) with the operation's refusal code in `cause.code` and its stored `retryable` flag; a deterministic failure also stores the refusal the worker classified (`datasets.operation.refusal`), and the poll body carries its `detail`, `pointer`, `alternatives`, `issues` and ceiling fields. A `done` operation whose rendering the snapshot lists no artifact for (an unlinked bundle, a bundle of a superseded rendition, an expired working-tier artifact) is failed `internal_error`, retryable, by that poll and answered `operation_failed`, so a resubmit of the identical request re-runs it. The poll carries no principal check in any state: a `done` manifest is the one the public mint returns and the public snapshot route re-signs for any caller. An operation no worker will advance (a lease expired for longer than `DATASETS_OPERATION_MAX_RETRY_DURATION_SECONDS` plus a margin, or a task never claimed within the worst-case backlog wait, measured from the current submission's last dispatch or, with none, from the submission itself) is failed `internal_error`, retryable, by the next poll or submit, and a resubmit of the identical request then starts a new submission. Operations stay pollable for `operation_retention_days` (published on `/api/datasets/capabilities`).

A dedicated worker process, built from the same image with `DFO_WORKER_SURFACE=1`, mounts a single internal route (`POST /internal/datasets/operations/{operation_id}/run`) that Cloud Tasks dispatches to, verified by Cloud Run IAM plus an app-level OIDC check. It claims the operation's lease for the submission the task names; a task from an earlier submission, or naming none, answers 200 `superseded`. Each claim increments the operation's attempt counter, and every later write of the run is fenced on that attempt, so a worker whose lease lapsed and was reclaimed can no longer write. The run reads the published coverage pointer and loads a moved build before it starts, re-estimates the request against that snapshot, records its stamp and estimate on the operation, fails `over_ceiling` if the fresh estimate now exceeds the async ceilings, and sizes its fetch guard from it. The run completes the operation only when a snapshot read of the completed version, narrowed to the request's rendering, lists an artifact other than statistics. Otherwise it takes the transient-failure path below. A background heartbeat renews the lease on a timer (`DATASETS_OPERATION_LEASE_SECONDS`, 60s by default, renewed at roughly a third of that) while the materialisation runs; if a renewal is refused, the run stops at its next chunk, writes nothing more, and answers 200 `superseded`. A worker that stops renewing (crashed, OOM-killed mid-materialisation) leaves the lease to expire, and a Cloud Tasks retry reclaims it (the queue's own `retry_config.min_backoff` is set >= `DATASETS_OPERATION_LEASE_SECONDS`, so that retry never meets a still-live lease). A transient failure before the operation's last Cloud Tasks attempt (`X-CloudTasks-TaskRetryCount` against `DATASETS_OPERATION_MAX_ATTEMPTS`, 5 by default, mirroring the queue's own `retry_config.max_attempts`) releases the lease before answering non-2xx, so the next retry claims immediately; on the last attempt it marks the operation failed instead. `docs/datasets-sync-ceiling.md` covers the full admission, operation, and worker mechanics, including the env vars each side needs and the measured writer memory numbers.

---

## Currency and the Subject Head

`datasets.version_record` (`dataset_store` migration `0004_version_records.sql`) is immutable: one row per `(subject_id, content_version)` pair, written once and never updated. It has nowhere to say which of a subject's own recorded versions is the one a `GET` should currently serve.

`datasets.subject_head` (migration `0009_subject_head.sql`) is that pointer: one mutable row per `subject_id`, carrying the subject's current `content_version`, when that pointer was last `confirmed_at`, and the currency `signal` captured at that time, per member `dispatch_facet_id`, `obs_count`, `earliest_date`, and `latest_date` (the fields a facet probe already computes), so a later probe can be compared against it without re-materialising. A single-frame group's head carries exactly one member entry.

`datasets_api.version_store.LiveHeadStore.advance_head` takes the `content_version` its own caller read the head at (`expected_content_version`) and applies the write only when the stored row's own `content_version` still equals it: `INSERT ... ON CONFLICT (subject_id) DO UPDATE ... WHERE datasets.subject_head.content_version IS NOT DISTINCT FROM %s`, `IS NOT DISTINCT FROM` so a first insert with no existing row and a same-version refresh both apply. A write whose expected version has already moved returns `None` and leaves the head untouched. The caller re-reads the head and serves whichever version won instead. Every writer reads the head once before deciding what to write: `_resolve_currency` passes the version it read as `expected_content_version`, and `serve_definition` passes the version read before materialising. On `None`, the definition door re-reads the head and serves the version that won. Its own read has gone stale by then.

`ServeDeps.head_store` (type `HeadStore`), injected through `datasets_surface(head_writer=...)` the same way `version_writer` is, writes the head once a mint's version record has committed, only when the caller supplies a `signal`. `GET /api/datasets/{id}` computes that signal itself, from the currency probe below, whenever a re-check ends in a rebuild. `datasets_api.adapter.prime_head_signal` runs ahead of every mint and advances the head when it derives a signal. `ServeDeps.materialise_semaphore` (`DATASETS_MATERIALISE_MAX_CONCURRENCY`) bounds concurrent `materialise_group` calls across mint, definition retrieval, and the options route's rebuild, whichever door calls them.

`GET /api/datasets/{id}` and its write-free `/options` sibling read the head before deciding whether to serve the stored record or rebuild it. A head younger than `DATASETS_HEAD_RECHECK_SECONDS` (default `900`, fifteen minutes) is served as stored, with no probe. Past that interval, one batched facet probe is compared against the head's stored `signal`: a match refreshes `confirmed_at` and serves the stored record, a mismatch re-materialises and advances the head with the freshly captured signal. `DATASETS_HEAD_MAX_AGE_SECONDS` (default `86400`, twenty-four hours) counts from the head's last materialisation, not from a probe match: past it the frame re-materialises regardless of what the probe reports, because a probe compares counts and date ranges. It does not compare values, so it cannot see a correction to an existing observation that moves neither. The options route runs the identical decision but records no version and confirms no head. Its one head write follows a new calendar year: when the year moves the end of a window the definition left open (`datasets_api.currency.clock_moved`) and the rebuild reproduces the head's `content_version`, the route advances the head to the signal it derived, so the head is re-read once whichever route reads it first.

Both routes report the outcome on the manifest's `trace`, documented in `docs/conformance-datasets-api-v1.md`. `served_from` is `head` when the stored record was served, either as recorded or after a matching probe, and `materialise` when this request rebuilt the frame. `confirmed_at` is the UTC timestamp of when that currency was last established. On definition retrieval a probe match moves it forward. The options route writes nothing, so it reports the stored instant. `content_version` keeps its documented meaning: a digest of the canonical frame bytes, the graph and source revisions, the per-axis facet selections, and any conversion recipe. The revision half of that digest is a fixed placeholder until `RevisionReader` reads real revisions, so it does not move on a currency check by itself either way. `confirmed_at` and `served_from` are the currency-facing fields regardless of when the revision reader lands.

Neither route accepts a caller-supplied force-refresh flag. `Cache-Control: no-store` is already sent on every response, so nothing downstream caches a stale body past its own currency window. `GET /api/datasets/{id}/{content_version}` already covers a caller who wants one exact, pinned result with no currency question at all. A client-triggered rebuild would also let any caller pay, on demand, the cost `DATASETS_MATERIALISE_MAX_CONCURRENCY` exists to bound, defeating it.

---

## Origin provenance

Each request, and each worker operation, binds one origin catalogue build (`ORIGIN_CATALOGUE_URI`, see `deploy/origin-catalogue.md`). The worker runs an operation under the build its row names when that build is the one it has loaded, and otherwise reads the published build before it claims, so an operation never runs under a build older than the one it was submitted under. Every graph read goes through the origin graph client (`origin_catalogue.graph`), which drops facets whose provenance the catalogue does not admit. The manifest reports that build in three places.

- `citation.sources[]` entries carry `source` (`{ref, label}`, the Data Commons `Source` node behind the provenance, or for a UN System Data Commons series the agency provenance itself) and `data_commons` (`{id, label}`, with `id` one of `dfo`, `base`, `un_system`). `label`, `url`, `source` and `data_commons` come from the catalogue's provenance table and `license`, `license_type`, `license_url` and `documentation` from the graph (`materialise.provenance_doc.read_source_citations`). `documentation` is the source's `read_provenance_doc` content as read at first mint, `null` when the graph documents nothing for the source or that read failed, and it feeds the bundle's `README.md` and `SOURCES.md`. A bundle render of a version already on record fills every stored `null` source field except `ref`, `label` and `processing_level` (`license`, `license_type`, `license_url`, `url`, `source`, `data_commons` and `documentation`) from a live read, built as a first mint builds it (`datasets_api.manifest.cited_source`), for that render only and only when no servable bundle is linked under the request's rendering key. Every set field stands, and a source with every field set reads nothing. A mint leaves the stored body as it is. A bundle whose documentation or code label read failed is returned but not linked to the version, so the next request for it renders and reads again. A provenance the table does not list, such as a DFO provenance newer than the loaded build, is named from the graph and carries `source` and `data_commons` as `null`.
- `trace.origin_catalogue_build_id` names the build that admitted the data. A version minted before the catalogue existed keeps serving as minted, with neither the trace member nor the citation members, so its absence marks a pre-catalogue version.
- `trace.family_member_digest` and `trace.family_member_count` state every member the identity admits, before antichain reduction, by the digest Search's `resolved.shape.member_digest` uses. `trace.resolved_stat_vars` lists the members served. Both fields are null when member resolution did not read the whole family, which is a graph walk whose candidate page was truncated. They stay out of `content_version`, so an answer from a stored version carries the trace recorded when that version was first stored: the fields describe the family at that mint, and both keys are absent on a version stored before they existed. Read an absent key and a null alike as unknown.
- `warnings` carries `origin_catalogue_stale` when the build is more than 14 days old; `un_system_unavailable` naming the places the UN System Data Commons refused or failed to read; and `withheld_by_registry` when a subject entity (a bound one, or for a request that binds no entity a country under Earth) came back with no rows, has no admitted data at all, and a provenance the catalogue withholds holds data there, naming those entities and the withheld provenances. A mint that lands on a version already on record adds this call's warnings to the stored body and replaces the stored `origin_catalogue_stale` entry with this call's.

When every facet for a member's variable at its places is withheld, the facet probe finds nothing and the request refuses `excluded_by_registry` (422) naming the withheld provenances, in place of `default_facet_undefined`. A UN member with no candidate at places the UN refused or failed to read refuses `upstream_error` (502, retryable) naming those places instead, since its empty probe says nothing about data there. A UN member that binds no entity covers every country, including the few the UN front door refuses on every read, so it refuses `upstream_error` only when a UN read failed. With no data anywhere else it refuses `default_facet_undefined` (422), and the detail names the refused places its `un_system_unavailable` warning names. Both refusals name the first five places and count the rest. The facet probe runs before the mint doors open the origin record, so the probe carries what its UN reads could not serve and the materialisation adds it to that record: the refusal and the `un_system_unavailable` warning are the same at the mint doors and the worker. Both `excluded_by_registry` and `withheld_by_registry` cost one facet read with no domain filter, of the one member concerned, made at most once per mint, only when a requested series left an entity without rows, and under the request's deadline and a 16 MiB response cap (`datasets_api.origin`). A failed re-read leaves the answer without the explanation. A subject-only request that binds no entity is re-read at the countries under Earth, and for the refusals at Earth too, the places its facet probe read. A refusal reads those countries again, and a failed read refuses `upstream_error` (502, retryable). A UN member that binds none refuses on a failed UN read before that read, since its probe read only Earth and its countries. For such a request nearly every mint leaves a country without rows, so nearly every mint makes the re-read. The warning takes the countries from the materialisation result, so a mint reads containment once, in its facet probe. Role-bearing requests and `ContainedIn` scopes are not re-read, because role-bearing series are DFO's and always admitted, and a scope's members are unknown before the read.

For a variable both DFO and base Data Commons hold, facet selection ranks DFO's facet first whenever DFO has data at a requested place (`selection_reason` `dfo_precedence`), so places only the base source covers show as a coverage gap (`docs/datasets-facet-policy.md`). The ranking applies to the facets left after narrowing by the spec's bound axes and its native-currency preference. A spec that binds a provenance, unit or measurement method only a base facet carries gets that base facet, and its rows serve only at places where DFO holds no data, since admission withholds every base facet where DFO's is present.

The components envelope (`assembler.mint.provenance`) carries the same pair on each `provenance.sources[]` item as `source: {id, name}` and `data_commons: {id, name}`, left out where the citation entry has none.

---

## Trust Boundary

The upstream Data access proxy enforces rate limiting (`429`), request body size ceilings (`413`), and media type constraints (`415`).

The Datasets API assumes validated authentication upstream from the access proxy, except on the Pin door (`POST /{id}/{content_version}/pin`) and the Publish door (`POST /frames/publish`), which each independently check caller identity via the configured principal header, against `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` for pinning and `DATASETS_FIRST_PARTY_PRINCIPAL` for publishing, two independent values, since pinning has no unpin or delete route on this surface and a principal trusted to publish must not automatically gain that irreversible capability too.

---

## Configuration

Surface behavior is configured through environment variables.

### Optional Variables and Defaults

| Variable                               | Type    | Default                     | Description                                                                                                                                                                 |
| :---------------------------------------------- | :------ | :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATASETS_SYNC_BUDGET_ROWS`            | Integer | `100000`                    | Sync budget: rows above which a mint/definition request routes to an async operation instead of running inline.                                                          |
| `DATASETS_SYNC_BUDGET_REQUESTS`        | Integer | `40`                        | Sync budget: SDMX requests above which a mint/definition request routes to an async operation instead of running inline.                                                |
| `DATASETS_OPERATION_LEASE_SECONDS`     | Integer | `60`                        | How long a claimed operation's lease holds before another dispatch can reclaim it; the worker's own heartbeat renews it on a timer while a materialisation is still running. The queue's own `retry_config.min_backoff` (dfo-infra) must be >= this value, so a retry never meets a still-live lease. |
| `DATASETS_OPERATION_MAX_ATTEMPTS`      | Integer | `5`                         | How many times Cloud Tasks will ever dispatch one operation's own task, mirroring the queue's own `retry_config.max_attempts`; decides whether a transient failure marks the operation failed directly (last attempt) or releases the lease for a retry. |
| `DATASETS_OPERATION_BACKLOG_MAX`       | Integer | `20`                        | Upper bound on `pending`+`running` rows in `datasets.operation`; a new operation submitted at or above this refuses `service_unavailable`.                        |
| `DATASETS_OPERATION_MAX_RETRY_DURATION_SECONDS` | Integer | `3600`              | Mirrors the Cloud Tasks queue's own `retry_config.max_retry_duration`; an operation whose lease has been expired longer than this plus a fixed margin is failed as abandoned by the next poll or submit. |
| `DATASETS_MINT_MAX_CONCURRENCY`        | Integer | `4`                         | Dedicated worker pool size and admission semaphore for mint requests.                                                                                                       |
| `DATASETS_MINT_MEMBER_MAX_CONCURRENCY` | Integer | `4`                         | Maximum concurrent SDMX member fetch requests per request.                                                                                                                  |
| `DATASETS_MATERIALISE_MAX_CONCURRENCY` | Integer | `2`                         | Bounds concurrent `materialise_group` calls across the one shared semaphore. The mint door, definition retrieval, and the options route's rebuild all acquire it. |
| `DATASETS_MEMBER_SUBJECT_CEILING`      | Integer | `20000`                     | Subjects a described definition's member walk reads when its first reverse page truncates without a confirmed root. A larger filtered set refuses `over_ceiling` (`candidate_window`). |
| `DATASETS_MEMBER_READ_SECONDS`         | Float   | `60`                        | Seconds a member walk's graph reads may run, capped by the request deadline. A walk past it answers `timeout` (504, retryable).          |
| `DATASETS_HEAD_RECHECK_SECONDS`        | Integer | `900`                       | Age below which a stored head is served with no probe.                                                                                                                      |
| `DATASETS_HEAD_MAX_AGE_SECONDS`        | Integer | `86400`                     | Age since the head's last materialisation past which a `GET` rebuilds regardless of the probe, always greater than `DATASETS_HEAD_RECHECK_SECONDS`.                        |
| `DATASETS_MATERIALISE_WAIT_SECONDS`    | Integer | `30`                        | Upper bound on how long a request waits for a materialise slot before returning a retryable `503`. The actual wait is capped further by the request's own deadline, so it never outlives the mint or definition door's own timeout. |
| `DATASETS_METADATA_READ_SECONDS`      | Float   | `15`                        | Floor in seconds for each of a bundle render's two graph metadata reads: the live documentation read (a first mint's, or the one that fills a stored citation's empty fields), and the code label read. Each runs until the request deadline or this long from its start, whichever ends later, so a mint that used up its deadline still names its codes. With the defaults a door waits at most 150 seconds, under the apis service's 300-second Cloud Run request timeout (dfo-infra `infra/dfo-dev/apis/main.tf`, `timeout_seconds`). |
| `DATASETS_MINT_TIMEOUT_SECONDS`        | Integer | `120`                       | Request deadline in seconds for a synchronous mint. Queue waits, SDMX fetches, the citation read and a first mint's documentation reads stop at it. The door waits this long plus twice `DATASETS_METADATA_READ_SECONDS` for its executor. |
| `DATASETS_SDMX_TIMEOUT_SECONDS`        | Integer | `90`                        | Independent httpx transport timeout for SDMX calls. The mint or definition door's own request deadline applies when it expires first.                                     |
| `DATASETS_DEFINITION_MAX_CONCURRENCY`  | Integer | `4`                         | Dedicated worker pool size for definition retrieval requests.                                                                                                               |
| `DATASETS_DEFINITION_TIMEOUT_SECONDS`  | Integer | `120`                       | Request deadline in seconds for definition retrieval and the options route, bounding the same reads as `DATASETS_MINT_TIMEOUT_SECONDS`. The door waits this long plus twice `DATASETS_METADATA_READ_SECONDS` for its executor. |
| `DATASETS_SNAPSHOT_MAX_CONCURRENCY`    | Integer | `4`                         | Dedicated worker pool size for snapshot read requests.                                                                                                                      |
| `DATASETS_PIN_MAX_CONCURRENCY`         | Integer | `4`                         | Dedicated worker pool size for artifact pinning requests.                                                                                                                   |
| `DATASETS_PIN_DAILY_MAX`              | Integer | `500`                       | New pinned copies the pin door makes per trailing 24 hours, counted from pinned-tier `artifact_location` rows stored in that window whose digest holds a `pin` reference, so each of the door's promotions counts once, a second reference to a pinned digest or a repaired reference counts nothing, and presentation pins spend none of it. At the budget a new pin refuses the retryable `service_unavailable`, 503, with `Retry-After: 3600`, and copies nothing. A repeat pin of an artifact that already has a pinned copy is exempt. Pins in flight together can overshoot by up to `DATASETS_PIN_MAX_CONCURRENCY` per instance. `0` refuses every new pinned copy. |
| `DATASETS_PUBLISH_MAX_CONCURRENCY`     | Integer | `4`                         | Dedicated worker pool size and backpressure semaphore for the publish door; exhaustion refuses the retryable `service_unavailable`, 503.                                    |
| `DATASETS_PIVOT_MAX_OUTPUT_CELLS`      | Integer | `5000000`                   | Maximum output cell count permitted for pivot reshapes.                                                                                                                     |
| `DATASETS_GRAPH_TOKEN`                 | String  | Empty                       | Shared secret token for the `x-graph-token` upstream WAF header.                                                                                                            |
| `DATASETS_GRAPH_LABEL_TYPE_CACHE_SIZE` | Integer | `50000`                     | Entries in each of the graph client's label and type caches, the size Search uses. `0` turns a cache off.                                                                   |
| `DATASETS_GRAPH_ARCS_CACHE_SIZE`       | Integer | `4096`                      | Entries in the graph client's arcs cache.                                                                                                                                   |
| `DATASETS_GRAPH_OBS_PROPERTIES_CACHE_SIZE` | Integer | `5000`                      | Entries in the graph client's observation-properties cache.                                                                                                                 |
| `DATASETS_GRAPH_OBS_CACHE_SIZE`        | Integer | `512`                       | Entries in the graph client's observation cache.                                                                                                                            |
| `DATASETS_PRINCIPAL_HEADER`            | String  | `x-dfo-principal`           | Request header containing the authenticated principal name for pinning and publishing.                                                                                      |
| `DATASETS_FIRST_PARTY_PRINCIPAL`       | String  | Empty                       | Authorized principal name permitted to execute frame publishing.                                                                                                            |
| `DATASETS_PIN_FIRST_PARTY_PRINCIPAL`   | String  | Empty                       | Authorized principal name permitted to execute artifact pinning. Independent of `DATASETS_FIRST_PARTY_PRINCIPAL`.                                                          |
| `DFO_SITE_DATASET_LINKS`              | Boolean | `false`                     | Adds the website's `/data?spec=<spec_id>&v=<content_version>` page to bundles of a standalone definition with a described identity. Turn it on once the website serves that route. Pinned definitions (`kind: "stat_var"`, minted by `statvar_set`) never print it, because the website replays the identity through Search, which refuses the pinned arm. |

### Required Deployment Variables

| Variable                           | Description                                                              |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DATASETS_GRAPH_BASE_URL`          | Upstream Data Commons graph base URL for resolution and materialisation. |
| `DATASETS_SDMX_BASE_URL`           | Upstream SDMX client base URL for member queries.                        |
| `DATASETS_ARTIFACT_BUCKET_WORKING` | Google Cloud Storage bucket name for working-tier rendered artifacts.    |
| `DATASETS_ARTIFACT_BUCKET_PINNED`  | Google Cloud Storage bucket name for pinned permanent artifacts.         |
| `DFO_PUBLIC_SITE_URL`              | The public website origin, such as `https://developmentfinance.org`, which also proxies `/api/*`. Every bundle link is built on it, and the presentation components door reads it too. The datasets surface and the worker refuse to start without it, and a value with a path, query or fragment is refused. |
| `COVERAGE_ARTIFACT_URI`            | Base URI of the published coverage artifact this surface loads at startup (`gs://` or `file://`); see `deploy/coverage.md`. |
| `ORIGIN_CATALOGUE_URI`             | Base URI of the published origin catalogue this surface and the worker load at startup; each request binds the loaded build, and each worker operation the published one. |
| `DFO_WORKER_URL`                   | The worker service's own base URL; enqueued as the Cloud Tasks OIDC audience. |
| `DFO_WORKER_INVOKER_SERVICE_ACCOUNT` | Service account the worker's run route accepts as the OIDC token's `email` claim. |
| `DFO_TASKS_QUEUE`                  | Cloud Tasks' fully-qualified queue resource name this surface enqueues over-budget operations onto. |
