# The sync budget and the async worker

The Datasets API materialises a mint or definition request synchronously whenever it can, and
routes it to an async operation when it can't. The routing decision comes from an exact cost
estimate against the published `series_coverage` artifact, computed before any network call.

## Admission

`datasets_api.admission.sync_estimate(requests, coverage=...)` groups a request's
`MaterialisationRequest`s by role binding and window, and calls
`series_coverage.index.CoverageIndex.estimate` once per group, summing the results into an
`Estimate(rows, requests)`. `requests` counts SDMX `/data` GETs the same way
`dc_sdmx.SdmxClient.data`'s own chunk planner does, so it matches what a real fetch sends. The
estimate is exact for a StatVar coverage carries rows for; a safe over-estimate for a `"latest"`
window (coverage has no per-window-kind mode, so it prices the full representable year range
instead) and for a role bound through `dc_sdmx.ContainedIn` (a place-hierarchy filter this module
treats as unbound rather than resolving with a graph read); and `rows=0`/`requests=0` for a
StatVar coverage carries no row for at all, which means coverage has nothing to say about that
request's cost, not that it costs nothing.

`datasets_api.admission.within_sync_budget(estimate)` returns `True` when both `estimate.rows` and
`estimate.requests` stay at or under `DATASETS_SYNC_BUDGET_ROWS` (default 100,000) and
`DATASETS_SYNC_BUDGET_REQUESTS` (default 40), each read once at import with that default. A
request within budget runs inline.

`datasets_api.admission.refuse_excel_over_estimate(estimate, format=...)` raises `ExcelLimitError`
(problem code `excel_limit`, 422) before any fetch when the format is `xlsx` and `estimate.rows`
already exceeds `EXCEL_MAX_ROWS` (1,048,576, Excel's own row ceiling,
`materialise.formats.EXCEL_MAX_ROWS`). `write_excel`'s own post-fetch column and cell-character
checks still run for what an estimate can't predict.

## The async operation

A request over budget reaches `datasets_api.operations.submit.submit_over_budget`, called from the
mint door's own kind handlers and the definition door's re-materialise closures once
`within_sync_budget` says no. An estimate over `DATASETS_ASYNC_MAX_ROWS` (2,000,000) or
`DATASETS_ASYNC_MAX_REQUESTS` (800) refuses `over_ceiling` (`ceiling_kind` `operation_rows` or
`operation_requests`) before any row is written. Otherwise the call computes the operation id from
the request's own `subject_id` (a `spec_` or `col_` identifier a registry write or read has already
resolved), the coverage stamp the estimate ran against, the origin catalogue build bound to the
request, and the rendering key of the options, so a mint of `spec_x` and a later
`GET /api/datasets/spec_x` asking for the same rendering land on the same row.

A definition's operation also stores, and hashes into its id, `member_dcids`: the StatVars the
submitting door resolved, as sorted bare dcids. The worker passes them to member resolution as its
seed. A mint whose posted `resolved_stat_var_dcids` root proved a family complete past the
candidate window therefore runs with that root in the worker, where a first mint has no head to
recover it from. Two submissions of one spec share an operation only when their doors resolved the
same members, so a failure recorded for one member set is never returned to a submission that
resolved another. An operation that stored no members seeds from the spec's head, as
`GET /api/datasets/{spec_id}` does. A collection's operation stores the `col_` identifier alone:
its members are registered specs and take no seed.

`datasets.operation` (migrations `0011_operation.sql`, `0013_operation_retryable_and_index.sql`,
`0014_operation_dispatch.sql`, `0015_operation_origin_catalogue_build.sql` and
`0016_operation_refusal.sql`) stores the request
identity, rendering options, the admission coverage stamp, catalogue build and estimate, the
lifecycle state (`pending`, `running`, `done`, `failed`), and the dispatch and fencing counters
described below. A submit that finds an existing row returns it. Only a `failed` row marked
`retryable` is reset to `pending` as a new submission, with `submission_count` incremented. A new
row or a reset first counts `pending` and `running` rows under an advisory lock and refuses
`service_unavailable` (503, retryable) at `DATASETS_OPERATION_BACKLOG_MAX` (20). Every mint and
definition door call that reaches this point raises `OperationSubmitted`, caught by the door itself.
The response is `202 Accepted` with a `Location` header pointing at
`/api/datasets/operations/{operation_id}` and a body carrying `operation_id`, `status`, `poll`, and
`retry_after_ms` (1,500).

### Dispatch

Only a caller that wins `OperationStore.claim_dispatch` creates a Cloud Tasks task. The claim is
one conditional `UPDATE` that increments `dispatch_count` on a `pending` row no delivery has
claimed (`lease_until IS NULL`), when the submission has no dispatch yet, or when its last dispatch
is unconfirmed after `REDISPATCH_GRACE_SECONDS` (30) and the submission has redispatches left under
`MAX_REDISPATCHES` (3). The submit path and the poll route both run it through
`datasets_api.operations.dispatch.dispatch_if_due`, which enqueues after the claim has committed,
outside any transaction or lock. Concurrent submits and polls race on that one `UPDATE`, so one of
them creates the task.

A `create_task` call that returns, or reports that the task already exists, is recorded with
`confirm_dispatch`, and a confirmed dispatch belongs to Cloud Tasks from then on. It is never
repeated, however long the queue holds it. A call that raises or runs past
`ENQUEUE_TIMEOUT_SECONDS` (10) has an unknown outcome. The submit still answers 202, and the next
submit or poll after the grace dispatches again. The task name is
`{queue}/tasks/{operation_id}-{submission_count}`, the same for every dispatch of one submission,
so a redispatch whose earlier create did land answers `AlreadyExists` and the queue holds one task.
The task body carries `operation_id` and `submission_count`.

### Polling

`GET /api/datasets/operations/{operation_id}` applies the dead-row rule below to the row, reads it,
and runs the dispatch step. An unknown id answers `operation_not_found` (404). A `pending` or
`running` row answers 200 with its progress, rows written against the estimate the worker enforces.
A `done` row re-signs its manifest through the same path `GET /api/datasets/{id}/{content_version}`
uses, with no principal check. It is the manifest the public mint returns and the public snapshot
route re-signs for any caller. A `failed` row answers `operation_failed` (500) with the row's
refusal code in `cause.code`. A row the worker failed deterministically also stores the refusal
it classified (`refusal`), and the body carries that refusal's `detail`, `pointer`, `alternatives`,
`issues`, `constraint_id`, `ceiling_kind`, `count` and `cap`, and its `cause.entities`, as
`GET /api/datasets/{id}` renders them for the same failure: the worker classifies with that door's
classifier, so `member_not_resolvable` carries no `/input/spec_id` pointer. A row with no stored
refusal (a transient failure on its last attempt, a dead row, or a failure recorded before the
column existed) names the refusal code in `detail`, and so does a row whose stored refusal the
reading image cannot parse, which it logs.

Migration `0016_operation_refusal.sql` goes in before any API or worker image that reads the
column. Such an image selects `refusal` in every operation-store statement, so against a table
without the column every poll, over-budget submit and worker claim fails as
`DatasetsStoreUnavailable`; mints within the sync budget are unaffected. An older image runs
unchanged against the migrated table.
The body's `retryable` is the stored row's flag: a failed operation stays
failed on every poll, and resubmitting the mint resets a retryable row. Terminal
operations stay pollable for `operation_retention_days` (30,
`dataset_store.snapshot.WORKING_TIER_RETENTION`, the same working-tier artifact clock every other
mint/definition response is bound by), published on `GET /api/datasets/capabilities` alongside
`operation_envelope_active` (`true`), `sync_budget_rows`, and `sync_budget_requests`.

### Dead rows

One predicate in `datasets_api.operation_store` fails a `pending` or `running` row with
`refusal_code` `internal_error`, retryable, once no worker will advance it:

- a lease (set by a claim or heartbeat, or left by `release`) expired for longer than
  `DATASETS_OPERATION_MAX_RETRY_DURATION_SECONDS` (3,600) plus a 300-second margin;
- a row no delivery ever claimed, once the worst-case wait a full backlog imposes before a first
  dispatch, `ceil(backlog_max / dispatch_concurrency) * 1,800 + max_retry_duration` (12,600
  seconds by default), has passed since the current submission's last dispatch (`dispatched_at`),
  or since the submission itself (`submitted_at`) when it has none.

`submitted_at` is written by the insert and rewritten by every reset, so a resubmit of an operation
first created long ago is judged from its own submission. A submission that has used every
dispatch the cap allows gets the same window from its last dispatch, because that task may still
sit in the queue.

The poll applies it to one row, and every submit sweeps every row with it before counting the
backlog. Staleness is decided inside the `UPDATE`, so Postgres re-checks it against the row a
concurrent heartbeat, claim, or completion has just committed, and a lease renewed or a row
finished after the reader looked is left alone. A retryable failure lets a resubmit of the
identical request start a new submission.

## The worker service

Setting `DFO_WORKER_SURFACE` (truthy: `1`/`true`) swaps the whole app composition for a standalone
ASGI app carrying exactly one route, `POST /internal/datasets/operations/{operation_id}/run`
(`datasets_api.operations.worker_app.build_worker_app`), with no Search or public Datasets routes.
Cloud Run's own ingress rule keys on the `/internal/` prefix. The deployed worker runs at Cloud Run
concurrency 1, so the app builds one `DefinitionDeps` at startup and runs each dispatch's
materialisation on a worker thread.

Two authorization layers gate the run route. Cloud Run IAM (`run.invoker`, granted only to the
invoker service account, ingress internal) is the one that actually keeps an unauthenticated
caller out. `datasets_api.operations.oidc.LiveOidcVerifier` is the second, backup layer. It
verifies the bearer token's signature, expiry, and issuer through
`google.oauth2.id_token.verify_oauth2_token`, and checks the token's `email` claim against
`DFO_WORKER_INVOKER_SERVICE_ACCOUNT` and its `aud` claim against `DFO_WORKER_URL`.

The run route reads `submission_count` from the task body and claims the lease with one conditional
`UPDATE` that increments `attempt_count`, for `DATASETS_OPERATION_LEASE_SECONDS` (60s by default):
the row must be `pending` or `running`, belong to that submission, and hold no live lease. A task
with no `submission_count`, or one from an earlier submission of a since-reset row, answers 200
`{"status": "superseded"}`. A live lease answers 409. A terminal operation answers 200 with its
state, so a duplicate delivery tells Cloud Tasks not to retry. `attempt_count` is the fencing
token. Every later write of the run (progress, the execution record, `complete`, `fail`, `release`)
matches `state = 'running' AND attempt_count = <attempt>`, so an attempt whose lease lapsed and was
reclaimed can no longer write.

The run takes one coverage snapshot with `CoverageService.snapshot_current`, which reads the
published `current.json` pointer and loads a moved build before it returns, so the run never
executes against an artifact older than the pointer at run start, however long the worker instance
sat idle with its CPU throttled. A failed pointer read or load is a transient failure of the
attempt. The run then resolves the stored subject and estimates it against that snapshot with
`sync_estimate`, the same estimator admission uses. It records the snapshot's stamp
(`execution_coverage_stamp`, beside the admission `coverage_stamp`) and the fresh estimate on the
row, checks the async ceilings again (a breach fails the operation `over_ceiling`, not retryable),
and sizes the raw-fetch guard from the fresh estimate. A coverage republish between submission and
execution is therefore enforced against the data the run fetches.

A background heartbeat (`datasets_api.operations.worker.run_operation`) renews the lease at roughly
a third of its length for as long as the materialisation runs, writing `rows_written` with each
renewal. When a renewal is refused because the attempt lost its fence, the heartbeat cancels the
run's `materialise.RowProgress`. The next finished chunk raises, the run writes nothing more, and
the route answers 200 `superseded`. The queue's own `retry_config.min_backoff` (dfo-infra, 60s)
must stay at or above `DATASETS_OPERATION_LEASE_SECONDS`, so a retry after a dead worker finds the
lease already expired. A deterministic failure (`classify_definition_exception` says not retryable)
calls `fail` and answers 200. A transient failure before the operation's last Cloud Tasks attempt
(`X-CloudTasks-TaskRetryCount` against `DATASETS_OPERATION_MAX_ATTEMPTS`) releases the lease (back
to `pending` with `lease_until` set to now, so the row stays under the lease rule and out of
redispatch) and answers non-2xx. On the last attempt it fails the operation, retryable, and answers
200.

The API side needs `DFO_WORKER_URL`, `DFO_WORKER_INVOKER_SERVICE_ACCOUNT`, and `DFO_TASKS_QUEUE`
(Cloud Tasks' fully-qualified queue resource name) to enqueue, for the mint and definition doors
and for the operations poll. The worker needs `DFO_WORKER_URL` and
`DFO_WORKER_INVOKER_SERVICE_ACCOUNT` to verify, plus the same `DATASETS_GRAPH_BASE_URL`/
`DATASETS_SDMX_BASE_URL` the API doors read, since it materialises through the same
`serve_definition`/`serve_collection` entry points the synchronous doors call.

## The mid-fetch and post-fetch backstop

The worker, and every synchronous door, still passes an explicit `ceiling` to each
`materialise_group`/`materialise_collection` call, sized to the request's own estimate rather than
a fixed number. `datasets_api.serve.raw_fetch_guard(estimate, floor=...)` returns
`max(floor, ceil(estimate.rows * 1.5))`, 50% headroom over the estimate to absorb drift between
the coverage snapshot and the live SDMX read, and not a second admission decision. `floor` covers a
StatVar coverage carries no row for, and a caller with no estimate at all. It is
`datasets_api.publish_models.MAX_ROWS`. This value feeds both `generation.py`'s own mid-fetch
raw-row budget and `materialise.ceiling.check_ceiling`'s post-fetch `OverCeilingError` (problem
code `over_ceiling`, 422) as a final backstop against a response that has genuinely ballooned past
what coverage predicted. `materialise.ceiling.DEFAULT_SYNC_CEILING_ROWS` (250,000) is a separate
number, `dfo-data-tools`' own generic default for a caller of `materialise_*` outside
`datasets_api` that supplies no `ceiling` of its own. The Datasets API always supplies its own.

## Publish

`POST /frames/publish` has no coverage artifact to estimate against, so it keeps its own fixed,
non-env-tunable row ceiling, `PUBLISH_MAX_ROWS = 250_000`
(`datasets_api.doors.publish.PUBLISH_MAX_ROWS`), enforced by the same `OverCeilingError` the
mid-fetch backstop raises. The value is frame1's own ceiling, `datasets_api.render.FRAME1_MAX_ROWS`: a mint
writes the frame1 rendering presentation reads only for a frame of at most 250,000 rows, and a
larger version mints without it.

The door accepts a request body with `Content-Encoding: gzip`, `identity`, or none, and refuses any
other encoding with `request_invalid` (400) before reading the body. Every body must declare its
`Content-Length`. `MAX_PUBLISH_BODY_BYTES` (129,000,000 bytes, 512 bytes a row at the row ceiling)
bounds both the declared length of any body and the decoded size of a gzip body, since the door
holds and JSON-decodes the decoded bytes whatever the wire encoding. The 250,000-row benchmark body
is 218 bytes a row, so the bound leaves about 2.3 times that for longer DCIDs and StatVar names. The
door adds about 12 times the body size in RSS, so a body at the bound adds about 1.5 GiB.

The door admits each publish in this order: principal, permit, body read or gzip inflation, JSON
decode. It checks the principal from the headers, so an unauthorized caller gets
`publish_not_permitted` (403) without holding a permit or having its body read. The permit comes
from the door's `DATASETS_PUBLISH_MAX_CONCURRENCY` semaphore and stays held until the build ends, so
it bounds body decoding as well as the build. With no permit free the door answers `service_unavailable` (503). The door inflates a gzip
body in 1 MiB steps and refuses it with `request_invalid` as soon as the decoded size passes the
bound, so a gzip bomb costs at most the bound plus one step. Corrupt, truncated, or multi-member
gzip data also refuses with `request_invalid`. Cloud Run caps an HTTP/1 request at 32 MiB on the
wire, which an identity body reaches at about 150,000 dfo-ai rows. The same 250,000-row body is
54 MB as JSON and 4.7 MB gzipped at level 1.

## Writer memory

`materialise.formats.write_parquet`/`write_csv`/`write_excel` stream member tables through their
writers in chunks rather than building one whole-result table in memory. `scripts/measure_writer_rss.py`
drives each writer over synthetic CRS-shaped chunks (500 rows apiece) at the two row counts a
design prototype measured on the prior whole-table path, and reports the peak resident set size
inside a freshly spawned process. At 2,517,909 rows, Parquet writing peaks at 141 MB and CSV at
366 MB; xlsx refuses before writing, past `EXCEL_MAX_ROWS`. At 462,720 rows, Parquet peaks at
131 MB, CSV at 225 MB, and xlsx at 407 MB.
