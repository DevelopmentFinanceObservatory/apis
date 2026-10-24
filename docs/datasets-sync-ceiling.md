# The sync budget and the async worker

The Datasets API materialises a mint or definition request synchronously whenever it can, and
routes it to an async operation when it can't. The routing decision comes from an exact cost
estimate against the published `series_coverage` artifact, computed before any network call.

## Admission

`datasets_api.admission.sync_estimate(requests, index=..., member_concurrency=...)` groups a
request's `MaterialisationRequest`s by dispatch key (`materialise.generation.request_batch_key`) and
window years, splits each group into the StatVar chunks that materialisation dispatches
(`dispatch_lanes`, `stat_var_chunk_positions`), and calls
`series_coverage.index.CoverageIndex.estimate` once per chunk, summing the results into an
`Estimate(rows, requests, uncovered_requests)`. `requests` counts SDMX `/data` GETs the way a real
fetch sends them. The estimate is exact for a StatVar coverage carries rows for; a safe
over-estimate for a `"latest"` window (coverage has no per-window-kind mode, so it prices the full
representable year range while the request-size estimate keeps `c[TIME_PERIOD]=LATEST`) and for a
role bound through `dc_sdmx.ContainedIn` (a place-hierarchy filter this module treats as unbound
rather than resolving with a graph read); and `rows=0`/`requests=0` for a StatVar coverage carries
no row for at all, which means coverage has nothing to say about that request's cost, not that it
costs nothing. Those members count in `uncovered_requests`, and more than `UNCOVERED_MEMBER_CAP`
(700) of them refuse `over_ceiling` (`ceiling_kind` `member_cardinality`, 422) before any cost is
computed.

`datasets_api.admission.within_sync_budget(estimate, sync_budget_rows=..., sync_budget_requests=...)`
returns `True` when both `estimate.rows` and `estimate.requests` stay at or under
`DATASETS_SYNC_BUDGET_ROWS` (default 100,000) and `DATASETS_SYNC_BUDGET_REQUESTS` (default 40),
which the caller reads from `DatasetsBuildSettings`. Only covered costs enter the test. A request
within budget runs inline.

`datasets_api.admission.refuse_excel_over_estimate(estimate, format=...)` raises `ExcelLimitError`
(problem code `excel_limit`, 422) before any fetch when the format is `xlsx` and `estimate.rows`
plus the header row exceeds `EXCEL_MAX_ROWS` (1,048,576, Excel's own row ceiling,
`materialise.formats.EXCEL_MAX_ROWS`). `write_excel`'s own post-fetch column and cell-character
checks still run for what an estimate can't predict.
`datasets_api.admission.refuse_inline_over_ceiling(estimate, ceiling=...)` raises `OverCeilingError`
(`over_ceiling`, `ceiling_kind` `rows`, 422) before any fetch when the estimate exceeds the row
ceiling an inline run works under. The options route, which cannot submit an operation, calls it.

## The async operation

A request over budget reaches `datasets_api.operations.submit.submit_over_budget`, called from the
mint path in `datasets_api.kinds.spec_id` and from the definition door's admission in
`datasets_api.current` once `within_sync_budget` says no. An estimate over `DATASETS_ASYNC_MAX_ROWS`
(2,000,000) or `DATASETS_ASYNC_MAX_REQUESTS` (800) refuses `over_ceiling` (`ceiling_kind`
`operation_rows` or `operation_requests`) before any row is written. Otherwise the call computes the
operation id from the request's own `subject_id` (a `spec_` identifier a registry write or read has
already resolved), the origin catalogue build bound to the request, and the rendering key of the
options, so a mint of `spec_x` and a later `GET /api/datasets/spec_x` asking for the same rendering
land on the same row. The coverage stamp the estimate ran against is stored on the row and stays out
of the id, so a coverage build never orphans a done operation.

A definition's operation also stores, and hashes into its id, `member_dcids`: the StatVars the
submitting door resolved, as sorted bare dcids. The worker resolves members again from the
identity alone, as the door does, and compares them with `member_dcids`. When they differ, the
graph changed after submission, and the row fails `upstream_version_changed`, retryable, so a
resubmission resolves the new members under a new operation id. Two submissions of one spec share
an operation only when their doors resolved the same members, so a failure recorded for one member
set is never returned to a submission that resolved another.

`datasets.operation` (migrations `0011_operation.sql` through `0016_operation_refusal.sql`, and
`0024_operation_attempt_fencing.sql`) stores the request identity, rendering options, the
admission coverage stamp, catalogue build and estimate, the lifecycle state (`pending`, `running`,
`done`, `failed`), and the dispatch and fencing columns described below. A submit that finds an
existing row returns it. A `done` row, a `failed` row marked `retryable`, a `running` row whose
lease has ended, or a `pending` row past the operation deadline (see
[Abandoned runs](#abandoned-runs)) is reset to `pending` as a new submission, with
`submission_count` incremented. A `done` row resets because a submission reaches the store only
when the subject's head could not answer it, and the rerun serves what the head gives at run
time. A new row or a reset first
counts the other live rows (every `pending` or `running` row not abandoned) under an advisory
lock and refuses `service_unavailable` (503, retryable) at `DATASETS_OPERATION_BACKLOG_MAX` (20).
Every mint and definition door call that reaches this point raises `OperationSubmitted`, caught by
the door itself. The response is `202 Accepted` with a `Location` header pointing at
`/api/datasets/operations/{operation_id}` and a body carrying `operation_id`, `status`, `poll`, and
`retry_after_ms`. A `Retry-After` header carries the same wait in whole seconds, rounded up.
`retry_after_ms` is a quarter of the time since the operation's current submission (`submitted_at`),
held between 1,500 and 8,000, so a submission that joins a long build gets the slow value and a
resubmission that restarts a failed build starts at 1,500. Every pending or running poll carries
both. A done poll carries neither.

### Dispatch

`create_or_get` commits a new row `pending` with `dispatched_at` NULL. The submit path then
creates the Cloud Tasks task through `datasets_api.operations.dispatch.dispatch_if_due`, outside
any transaction or lock, and records `dispatched_at` with `mark_dispatched` once the create
returns or reports that the task already exists. A submit that finds a `pending` row with no
recorded dispatch creates the task too. The task name is
`{queue}/tasks/{operation_id}-{submission_count}`, the same for every create of one submission, so
a repeated create answers `AlreadyExists`, which counts as success, and the queue holds one task.
The task body carries `operation_id` and `submission_count`.

A create that raises or runs past `ENQUEUE_TIMEOUT_SECONDS` (10) has an unknown outcome, and so
does a process that dies between the commit and the create. Either leaves `dispatched_at` NULL.
The submit still answers 202, and the first poll more than `ENQUEUE_RECOVERY_SECONDS` (30) after
the submission creates the task again under the same name.

### Polling

`GET /api/datasets/operations/{operation_id}` reads the row, applies the abandoned-run rule below,
and runs enqueue recovery. An unknown id answers `operation_not_found` (404). A `pending` or
`running` row answers 200 with its progress, rows written against the estimate the worker enforces.
A `done` row reads its manifest through the same path `GET /api/datasets/{id}/{content_version}`
uses, with no principal check. It is the manifest the public mint returns and the public snapshot
route lists for any caller. Each artifact in it carries the stable link
`/api/datasets/{id}/{content_version}/files/{digest}`, which answers `302` to a URL signed for five
minutes (`DOWNLOAD_SIGNED_URL_TTL`) when it is followed, so the manifest holds no signed URL and no
expiry time. A `done` row whose rendering the snapshot lists no servable artifact for is failed
`internal_error`, retryable, by that poll (`OperationStore.fail_unservable_done`), and a resubmit
re-runs it. A `failed` row answers `operation_failed` (500) with the row's
refusal code in `cause.code`. A row the worker failed deterministically also stores the refusal
it classified (`refusal`), and the body carries that refusal's `detail`, `pointer`, `alternatives`,
`issues`, `constraint_id`, `ceiling_kind`, `count` and `cap`, and its `cause.entities`, as
`GET /api/datasets/{id}` renders them for the same failure: the worker classifies with that door's
classifier, so `member_not_resolvable` carries no `/input/spec_id` pointer. A row with no stored
refusal (a transient failure on its last attempt, an abandoned run, or a failure recorded before
the column existed) names the refusal code in `detail`, and so does a row whose stored refusal the
reading image cannot parse, which it logs.

Migration `0024_operation_attempt_fencing.sql` goes in before any API or worker image that reads its
columns. Such an image selects `refusal`, `attempt_count`, `claimed_at` and `lease_seconds` in every
operation-store statement, so against a table without them every poll, over-budget submit and worker
claim fails as `DatasetsStoreUnavailable`; mints within the sync budget are unaffected. The
migration only adds columns and maps rows, so the image before it runs unchanged against the
migrated table.
The body's `retryable` is the stored row's flag: a failed operation stays
failed on every poll, and resubmitting the mint resets a retryable row. Terminal
operations stay pollable for `operation_retention_days` (30,
`dataset_store.snapshot.WORKING_TIER_RETENTION`, the same working-tier artifact clock every other
mint/definition response is bound by), published on `GET /api/datasets/capabilities` alongside
`operation_envelope_active` (`true`), `sync_budget_rows`, and `sync_budget_requests`. The daily
`datasets-maintenance age-operations` job then deletes `done` and `failed` rows last updated longer
ago, and a poll for one answers `operation_not_found`.

### Abandoned runs

A lease that has ended (`claimed_at + lease_seconds` before now) means the attempt has stopped
owning the row. Whether another delivery will run it is a separate fact, which the claim records
as `final_delivery`: true when the claiming delivery is the task's last
(`X-CloudTasks-TaskRetryCount` at `DATASETS_OPERATION_MAX_ATTEMPTS - 1`), false when a later one
is due. A row is abandoned when nothing will finish it:

- A `running` row whose lease has ended and whose `final_delivery` is true. No delivery is left.
- A `running` row with no `claimed_at`, once `updated_at` is `LEASE_SECONDS` (1,620) old. An image
  that renewed its lease through `updated_at` claimed it and recorded no `final_delivery`, so no
  later delivery is known.
- A `running` row whose lease has ended and whose `final_delivery` is false, once
  `DATASETS_OPERATION_DEADLINE_SECONDS` have passed since its dispatch, or its submission when no
  dispatch is recorded. Until then the row polls as `running`, the poll writes nothing, the row
  keeps its place in the backlog, and the task's next delivery claims it. The queue's backoff
  (1,680 seconds) puts that delivery after the lease (1,620 seconds).
- A `pending` row once the same deadline has passed since its dispatch or else its submission.

`DATASETS_OPERATION_DEADLINE_SECONDS` is the application's own deadline for one submission. It
makes no claim about the queue: Cloud Tasks may still hold the task. The claim refuses a row
past the deadline, so a delivery that arrives after it runs nothing: the run route fails the row
through `OperationStore.abandon`, or finds it already failed, and answers 200 with the failed
state. The default lets
every delivery wait behind a full backlog and then run to its dispatch deadline, with the backoff
between deliveries and a margin: `attempts x (ceil(backlog_max / dispatch_concurrency) x
dispatch_deadline + dispatch_deadline) + (attempts - 1) x backoff + margin`, 3 x (5 x 1,800 +
1,800) + 2 x 1,680 + 240 = 36,000 seconds. `infra` computes it from the queue's own locals.

The poll fails an abandoned row with `refusal_code` `internal_error`, retryable, through
`OperationStore.abandon`, one `UPDATE` fenced on the attempt the poll read, so a delivery that
claimed the row first leaves it running. A resubmit of the identical request then resets it under
a new task name, and the backlog stops counting it. A resubmit also resets a `running` row whose
lease has ended while a later delivery is due, so a client can recover without waiting for the
backoff, and that delivery answers `superseded`. Enqueue recovery gives a polled `pending` row
with no `dispatched_at` its task long before the deadline. The daily
`datasets-maintenance age-operations` job fails every abandoned row the same way before it
deletes aged terminal rows, so an abandoned row nobody polls leaves `pending` or `running`.

## The worker service

Setting `DFO_WORKER_SURFACE` (truthy: `1`/`true`) swaps the whole app composition for a standalone
ASGI app carrying exactly one route, `POST /internal/datasets/operations/{operation_id}/run`
(`datasets_api.operations.worker_app.build_worker_app`), with no Search or public Datasets routes.
The worker's Cloud Run service takes internal-only ingress. The deployed worker runs at Cloud Run
concurrency 1 on 2 vCPU and 4 GiB, with up to four instances (the `infra` repository's
`infra/stacks/apis/datasets_worker.tf`), so the app builds one `DefinitionDeps` at startup and runs
each dispatch's materialisation on a worker thread.

Two authorization layers gate the run route. Cloud Run IAM (`run.invoker`, granted only to the
invoker service account, ingress internal) is the one that actually keeps an unauthenticated
caller out. `datasets_api.operations.oidc.LiveOidcVerifier` is the second, backup layer. It verifies
the bearer token's signature, expiry, issuer, and audience (`DFO_WORKER_URL`) through
`google.oauth2.id_token.verify_oauth2_token`, and checks that the token's `email` claim matches
`DFO_WORKER_INVOKER_SERVICE_ACCOUNT` and that `email_verified` is set.

The run route reads `submission_count` from the task body and claims the row with one conditional
`UPDATE` that increments `attempt_count` and sets `claimed_at`, `lease_seconds`
(`LEASE_SECONDS`, 1,620) and `final_delivery`. The row must belong to that submission, be `pending` or
`running` with its lease ended, and be inside the operation deadline. The lease is static and never renewed. A task from an earlier
submission of a since-reset row answers 200 `{"status": "superseded"}`, and a malformed or missing
`submission_count`, or one ahead of the row's own, answers 400. An unknown operation answers 404. A
delivery that finds another attempt inside its lease answers 409. A delivery that finds the row past
the operation deadline fails it retryable and answers 200 `{"status": "failed"}`. A terminal
operation answers 200 with its state, so a duplicate delivery tells Cloud Tasks not to retry.
`attempt_count` is the fencing token. Every later write of the run (progress, the estimate record, `complete`, `fail`,
`release`) matches `state = 'running' AND attempt_count = <attempt>`, so an attempt whose lease
ended and was reclaimed can no longer write.

After the claim the run binds one origin catalogue build and has a work deadline
(`RUN_DEADLINE_SECONDS`) 1,500 seconds after `claimed_at`, which bounds its SDMX and member reads.
The write half checks it before the version record and before each artifact render and upload, and
a run past it takes the transient path below, so no write starts after it. The lease outlasts the
work deadline by 120 seconds for the writes already under way, and stays under the task's
1,800-second dispatch deadline. Its materialisation takes one permit from the worker process's single-permit semaphore.
The run binds the graph release the worker process serves when the run starts
(`graph_release.context.bound_served_release`) and reads coverage from that release's coverage
view throughout. The process swaps to a newer published release in the background
(`deploy/graph-release.md`). The run then resolves the stored subject and estimates it against
that coverage with `sync_estimate`, the same estimator admission uses. It records the fresh estimate on the row,
checks the async ceilings again (a breach fails the operation `over_ceiling`, not retryable), and
sizes the raw-fetch guard from the fresh estimate. A release published between submission and
execution is therefore enforced against the data the run fetches once the worker serves it. The run then serves through
`current.serve_current`, so a current head that holds the request's rendering answers with no fetch,
and a rebuild advances the head.

Each finished chunk writes `rows_written` from the materialiser's own progress call
(`datasets_api.operations.worker._FencedProgress`), with no thread. A write the fence refuses
cancels the run's `materialise.RowProgress`. The next finished chunk raises, the run writes nothing
more, and the route answers 200 `superseded`. The head advance carries the claim as a
`HeadFence`: `LiveHeadStore.advance_head` locks the operation row in the head's own transaction
and, when another attempt has claimed it, raises `HeadFenceLost` with the head and the
materialisation table unchanged, which ends the run the same way. A deterministic failure
(`classify_definition_exception` says not retryable) calls `fail` and answers 200. A transient
failure before the operation's last Cloud Tasks attempt (`X-CloudTasks-TaskRetryCount` against
`DATASETS_OPERATION_MAX_ATTEMPTS`) releases the row back to `pending` with its lease cleared and
answers non-2xx, so the retry claims it at once. On the last attempt it fails the operation,
retryable, and answers 200. A worker that dies mid-run leaves its lease to end. A later delivery
reclaims it, or the next poll fails it retryable.

The API side needs `DFO_WORKER_URL`, `DFO_WORKER_INVOKER_SERVICE_ACCOUNT`, and `DFO_TASKS_QUEUE`
(Cloud Tasks' fully-qualified queue resource name) to enqueue, for the mint and definition doors
and for the operations poll. The worker needs `DFO_WORKER_URL` and
`DFO_WORKER_INVOKER_SERVICE_ACCOUNT` to verify, plus the same `DATASETS_GRAPH_BASE_URL`/
`DATASETS_SDMX_BASE_URL` the API doors read, since it materialises through the same
`current.serve_current` entry point the synchronous doors call.

## The mid-fetch and post-fetch backstop

The worker, and every synchronous door, passes an explicit `ceiling` to each
`materialise_group` call, sized to the request's own estimate rather than
a fixed number. `datasets_api.serve.raw_fetch_guard(estimate, floor=...)` returns
`max(floor, ceil(estimate.rows * 1.5))`, 50% headroom over the estimate to absorb drift between
the coverage snapshot and the live SDMX read, and not a second admission decision. `floor` covers a
StatVar coverage carries no row for, and a caller with no estimate at all. It is
`datasets_api.publish_models.MAX_ROWS` (1,000,000). This value feeds both `generation.py`'s own
mid-fetch raw-row budget and `materialise.ceiling.check_ceiling`'s post-fetch `OverCeilingError`
(problem code `over_ceiling`, 422) as a final backstop against a response that has genuinely
ballooned past what coverage predicted. `materialise.ceiling.DEFAULT_SYNC_CEILING_ROWS` (250,000) is
a separate number, `dfo-data-tools`' own generic default for a caller of `materialise_*` outside
`datasets_api` that supplies no `ceiling` of its own. The Datasets API always supplies its own.

## Publish

`POST /frames/publish` has no coverage view to estimate against, so it keeps its own fixed,
non-env-tunable row ceiling, `PUBLISH_MAX_ROWS = 250_000`
(`datasets_api.doors.publish.PUBLISH_MAX_ROWS`), enforced by the same `OverCeilingError` the
mid-fetch backstop raises. The door builds the whole frame in memory before it stores the canonical
frame, so the ceiling bounds that build. Mints store the canonical frame at every size.

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
from the work executor's semaphore (`DATASETS_WORK_MAX_CONCURRENCY`, 4) and stays held until the
build ends, so it bounds body decoding as well as the build, and a publish occupies a work thread
beside mints and rebuilds. With no permit free the door answers `service_unavailable` (503). The
door inflates a gzip body in 1 MiB steps and refuses it with `request_invalid` as soon as the
decoded size passes the bound, so a gzip bomb costs at most the bound plus one step. Corrupt,
truncated, or multi-member gzip data also refuses with `request_invalid`. Cloud Run caps an HTTP/1
request at 32 MiB on the wire, which an identity body reaches at about 150,000 `ai` rows. The same
250,000-row body is 54 MB as JSON and 4.7 MB gzipped at level 1.

## Writer memory

`materialise.formats.write_parquet`/`write_csv`/`write_excel` stream member tables through their
writers in chunks rather than building one whole-result table in memory. `scripts/measure_writer_rss.py`
drives each writer over synthetic CRS-shaped chunks (500 rows apiece) at 462,720 and 2,517,909 rows,
where whole-table builds used 636 MB to 3.28 GB, and reports the peak resident set size inside a
freshly spawned process. At 2,517,909 rows, Parquet writing peaks at 141 MB and CSV at
366 MB; xlsx refuses before writing, past `EXCEL_MAX_ROWS`. At 462,720 rows, Parquet peaks at
131 MB, CSV at 225 MB, and xlsx at 407 MB.
