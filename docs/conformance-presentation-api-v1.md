# Presentation API (V1) conformance matrix

This document tracks how the Presentation implementation (`src/presentation_api`) conforms to
its own served codes and wire shapes, clause by clause. Last verified against the code: 2026-10-03.

Status values:

- **implemented** — verified against the code; behavior matches the clause.
- **partial** — some but not all of the clause is met.
- **missing** — no implementation exists for this requirement.
- **divergent** — an implementation exists but behaves materially differently than the clause specifies.
- **n/a** — out of scope for this contract at V1 (deployment policy, or explicitly excluded).

Gap rows state what is inconsistent or missing. Presentation has no versioned external contract
text of its own inside this repo or spec-kit; every clause below states what the code itself does,
verified against `src/presentation_api` and the shared platform envelope it renders through
(`dfo_platform.problem`).

## Doors / endpoints

| Clause | Requirement                                                                                  | Status      | Gap |
| ------ | --------------------------------------------------------------------------------------------- | ----------- | --- |
| C1     | GET /api/presentation/healthz returns `{"status": "ok", "build": ...}`                        | implemented |     |
| C2     | GET /api/presentation/capabilities returns `{capability_version, catalog_version, build}`      | implemented |     |
| C3     | POST /api/presentation/components creates a component from a submitted intent                | implemented |     |
| C4     | PATCH /api/presentation/components/{component_id} revises an existing component               | implemented |     |
| C5     | GET /api/presentation/components/{component_id} reads a component's stored intent, fact summary and resolved payload back, at the head or at `?revision=N` | implemented |     |
| C6     | GET /api/presentation/frames/{spec_id}/{content_version} describes a frame (structural view, one dimension's values, or a query-filtered view); GET /api/presentation/frames/{spec_id} describes the canonical frame of a dataset's newest version the same way and names the pair it read | implemented |     |
| C7     | GET /api/presentation/release returns the pinned catalogue release's identity, retry-mode table, envelope JSON Schema, and its per-member-type component schemas | implemented |     |
| C7a    | GET /api/presentation/components/{component_id}/revisions/{revision}/data serves the frame rows behind every value that revision shows, as `format=csv`, `xlsx`, or `parquet` (default `csv`), in a zip bundle by default or as the bare file with `bundle=false`. It sends `Content-Disposition: attachment; filename="dfo_{card-title-slug}_{YYYY-MM-DD}_{format}_r{revision}.zip"` for a bundle and `filename="dfo_{card-title-slug}_{YYYY-MM-DD}_r{revision}.{csv\|xlsx\|parquet}"` for the bare file, and `Cache-Control: private, max-age=3600`. The slug is the card's title in lowercase ASCII words joined by hyphens (the component id when the card has no title), and the date is the UTC date the revision was stored, so one revision keeps one name | implemented |     |
| C7b    | POST /api/presentation/components/{component_id}/revisions/{revision}/render takes `{"frame": {spec_id, content_version}}` and answers what a component read answers for that revision, built from its stored intent on the named frame, with `resolved_payload.provenance.inputs` naming that frame and `provenance.refreshed_from` set to the revision. It appends no revision and pins nothing, so the named frame keeps its retention. It admits only the caller holding `LIVE_REFRESH_PRINCIPAL` in the `DATASETS_PRINCIPAL_HEADER` header, checked before the path or body is validated | implemented |     |
| C7c    | GET /api/presentation/components/{component_id}/revisions/{revision}/render/data?spec_id=&content_version= serves the C7a export of that revision rendered on the named frame, with the same `format` and `bundle` options, under the caller rule of C7b. It pins nothing | implemented |     |
| C7d    | C7b and C7c accept the frame the revision is stored on, a dataset frame of the same `spec_id` and rendering, or a derived frame whose recipe has the stored frame's `normal_form_hash`. Any other frame refuses `render_target_mismatch`, 409 | implemented |     |
| C7e    | On C7b and C7c every refusal answers the status the code registry maps for its code on `presentation.render`: `binding_invalid` and `constraint_violation` answer 422, and `frame_unavailable`, `release_mismatch`, and `resource_exhausted` answer 409. A store that fails to answer refuses `service_unavailable`, 503. A component with no current revision refuses `edit_conflict`, 409. "Refusal code detail" below gives the statuses create and revise serve | implemented |     |

## Request shape

| Clause | Requirement                                                                                                                                                     | Status      | Gap |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --- |
| C8     | A `cmp_` component id grants read: anyone holding it reads that component at any revision, with no header. Each id is a ULID with 80 random bits (`os.urandom`). An id that names no draft refuses `not_found`, 404, `retryable` false | implemented |     |
| C8a    | A write token grants write. A create returns a `cmw_` token as top-level `write_token`, and a create with `composition_id: "new"` also returns a `cpw_` token as `composition_write_token`, each `secrets.token_urlsafe(16)` behind its prefix. The store keeps the unsalted SHA-256 digest. No read, revise reply, stored intent, or resolved payload carries a token | implemented |     |
| C8b    | A revise needs the component's token in `x-dfo-write-token`; a composition member needs only its own. A create into an existing composition needs that composition's token in the same header. A missing or wrong token, or a draft with no stored digest, refuses `write_refused`, 403, `retryable` false, before any frame is read, with a fix telling the caller to create a new component | implemented |     |
| C9     | No door reads a session header. A request that sends `x-dfo-session`, with any value, an empty value, or the header repeated, is served exactly as one without it | implemented |     |
| C10    | `component_id`, `spec_id`, and `content_version` path parameters are format-checked before reaching a store lookup; a malformed value refuses `request_invalid`, 400, `retryable` false | implemented |     |
| C11    | The create/revise request body (`ComponentBody`) is closed (`extra="forbid"`): `intent` plus an optional `composition_id`; any other top-level field refuses through pydantic validation | implemented |     |
| C12    | A body carrying `schema_version` as a top-level field (a field `ComponentBody` does not declare) refuses `unsupported_schema_version`, 422, `retryable` false, distinct from the generic closed-body refusal | implemented |     |

## Component bodies

| Clause | Requirement | Status | Gap |
| ------ | ----------- | ------ | --- |
| C17    | The create (201), revise (200) and read (200) bodies carry `resolved_payload`, the revision's stored envelope (`presentation.component_revision.resolved_payload`), verbatim. The three bodies are equal as parsed JSON. The create and revise doors serve it in mint key order. The read door serves it in the key order Postgres returns from JSONB storage, and a number can come back reformatted. A no-op revise carries the head's stored envelope, and a `?revision=N` read carries revision N's. | implemented |     |
| C18    | `resolved_payload` validates against the `envelope_schema` of the catalogue release its `catalog_version` names, the same `CatalogueRelease.envelope_schema` `assembler.mint.stamp_envelope` validates a minted envelope against, with `envelope_schema`'s `component` field's `$ref`s resolving against that release's `component_schemas` by `$id`. GET /api/presentation/release serves only the pinned release's schemas, so a payload validates against that response only when its `catalog_version` is the pinned release. A read serves every stored revision with the `catalog_version` it was minted under and formats its fact summary with the formatter profile stored on it (`assembler.pipeline.get_component`). A write whose head carries a newer `catalog_version` than the pinned release refuses `release_mismatch` (`assembler.pipeline.write._require_release_not_behind`) | implemented |     |
| C19    | A minted component's `component` and `provenance` subtrees are measured in compact UTF-8 JSON against `MAX_ENVELOPE_BYTES` (2 MiB) before any store write; a candidate over the cap refuses `resource_exhausted`, `limit_name=MAX_ENVELOPE_BYTES`, with nothing stored | implemented |     |
| C20    | A revise whose resolved payload is otherwise unchanged but whose `catalog_version` differs from the pinned release's mints a new revision carrying the pinned release, rather than returning the head as a no-op (`assembler.pipeline.write._structurally_equal`) | implemented |     |
| C20a   | A chart's series and a table's rows follow the order `select` lists the values: by position in each list-valued dimension's list, dimensions taken in `select` key order (`assembler.resolve.resolve_selectors`). Series the lists leave unordered, such as those an unbound dimension expands into, keep frame row order. So `{"financingProvider": ["country/USA", "GlobalFund"]}` draws USA first even when the frame stores Global Fund's rows first. The fact summary's `series` and the data export's rows take the payload's order | implemented |     |

## Component data export

| Clause | Requirement | Status | Gap |
| ------ | ----------- | ------ | --- |
| C21    | The export holds one frame row per value the revision shows: each valued point of a line or bar chart series, each valued table row, and a stat card's headline period plus both change basis periods. Rows are matched by the series' stored `series_coordinates`, normalised as `resolve_selectors` groups them, and the period (`assembler.pipeline.export_component`) | implemented |     |
| C22    | When a shown value matches no frame row, two rows, a row with a non-null `status`, or a row whose `value` differs from the card's, and when a table revision's rows carry `series_id` on only some rows or carry a non-string or empty one, the route refuses `internal_error`, 500, logs the cause, and serves no file | implemented |     |
| C23    | Columns are `series` (the card's label for the row), then the frame's columns in frame order, with `stat_var_label`, `{role}_label`, and `provenance_label` after their columns, read from the frame's labels. A generated name that a frame column or an earlier generated column holds gains the first free suffix `_2`, `_3`, and so on, so every frame column keeps its own name. Files are written by the Datasets writers (`materialise.formats`), so a CSV or Excel cell starting with a formula trigger gains a leading apostrophe and Excel cells are literal strings | implemented |     |
| C24    | The bundle holds `README.md`, `data/{card-title-slug}.{format}`, `data/codes.csv` (when the rows hold codes), `SOURCES.md`, `CITATION.cff` and `datapackage.json` from `materialise.bundle.assemble_component_bundle`. Sources and licences come from the frame manifest's `citation`, one entry per provenance ref the rows carry, for `spec_` and `ops_` frames alike. A card export of a root `spec_` frame fills the citation fields the stored record left empty from a live read, and a failed read is not cached. The descriptor's `dfo` property holds `component` (`component_id`, `revision`, `frame_ref`, `select`, `window`), `prepared_at` (the revision's stored creation time, which also dates the file name), `links`, and `reproducing_call`, the export route as an absolute URL with its options and no header. The README takes the card title as its heading, says it holds the rows behind that revision, and names the frame's `spec_id` and `content_version`. The bytes match from one request to the next while the graph's code labels and the sources' documentation that a derived frame, or a root frame whose stored citation left fields empty, reads live are unchanged | implemented |     |
| C25    | An unknown component or revision, a `note_narration` or `source_card`, a revision stored without `series_coordinates`, and a table revision none of whose rows carries `series_id` refuse `not_found`, 404, the last with a detail saying the revision predates table row identities and the card must be recreated to export it. A frame the store cannot load refuses the code its reason names in "FrameUnavailable outcome by reason": `frame_unavailable`, 404, for a frame state, and `service_unavailable`, 503, for a store that fails to answer. An unknown `format` or unparseable `bundle` refuses `request_invalid`, 400. The route shares the components door's semaphore and timeout, so saturation refuses `service_unavailable` and a slow export `timeout` | implemented |     |

## Span periods

| Clause | Requirement | Status | Gap |
| ------ | ----------- | ------ | --- |
| C26    | A frame whose rows share one ISO 8601 interval in `time_period` (`2018/2022`, `2018-01/2022-12`, parsed by `dfo_frame_grammar.parse_time_period`) binds that interval as its one period (`assembler.ports.StructuralView.span`). An entity-mode bar chart states it as `category_axis.period` and each bar's period, a table as each row's `period`, and a stat card as its `period`, and the component data export carries it in each row's `time_period`. A line chart over such a frame refuses `binding_invalid` at `/type`, with `alternatives` naming `bar_chart`, `table`, and `stat_card`, before its window is read, and a bar chart whose `select` lists no values and resolves to one series refuses the same way after resolution (`assembler.pipeline.candidate`). The refusal reads the selected series. When they vary on exactly one unselected dimension other than a facet, it names that dimension, which a bar chart with the same `select` compares. When `select` resolves to one series, it names the `select` that lists two or more values of the first dimension the frame holds alongside that series' other bound values. When no dimension has two such values, `alternatives` names `table` and `stat_card` alone. A frame holding an interval beside any other period, point or interval, refuses `binding_invalid` at `/frame_ref` (`assembler.frame_view.compute_structural_view`) | implemented |     |
| C27    | A `bar_chart` whose `select` lists no values and resolves to several series that vary on exactly one dimension other than a facet, with data in one period of the window and a value in it for every series, draws one bar per value of that dimension, largest first, and its `fact_summary.defaults` names `/select/<dimension>` on create, revise and read alike (`assembler.mint.builders`, `assembler.reply`). A revise whose envelope is unchanged but whose defaults differ from the head's mints a new revision (`assembler.pipeline.write`). Series that vary on two such dimensions, or hold data in several periods, refuse `binding_invalid` at `/select` with the first dimension's values as `alternatives`. Series of which some lack a value in the one period refuse `binding_invalid` at `/select`, naming the values that lack it, with the values observed in it as `alternatives` | implemented |     |

## Response envelope

| Clause | Requirement                                                                                                                                                                                            | Status      | Gap |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --- |
| C13    | Non-2xx bodies are RFC 9457 problem details (`type`, `title`, `status`, `detail`, `code`, `pointer`, `alternatives`, `truncated`, `issues`, `constraint_id`, `trace_id`, `capability_version`, `retryable`), served through the shared platform `Envelope`; optional fields are omitted when absent | implemented |     |
| C14    | Every non-2xx response carries `Content-Type: application/problem+json` and `Cache-Control: no-store`                                                                                                     | implemented |     |
| C15    | `binding_invalid`'s case for an unbound facet that splits one series (two matched series agree on every coordinate except the unbound facets and differ on that facet, a null counting as a value) additionally renders `issues[]`, one `ProblemIssue` per frame dimension (`axis`, `alternatives`, `truncated`) | implemented |     |
| C16    | `schema_invalid`'s missing-field case renders its minimal legal intent inside `detail`; its illegal-field case renders the type's legal fields through `alternatives`                                     | implemented |     |

## Refusal code detail

Every code `presentation_api.codes` can put on the wire, its HTTP status, and its retry mode.
`CODE_RETRY_MODE` names one headline mode per code; the per-response `retryable` field on the wire
is the authoritative value for that one occurrence, and three codes (`schema_invalid`,
`frame_unavailable`, `request_invalid`) compute it per cause rather than from the headline mode
alone.

| Code                        | HTTP           | Retry mode                                     | Status      | Gap |
| ---------------------------- | -------------- | ----------------------------------------------- | ----------- | --- |
| schema_invalid                | 400            | correct_and_retry (missing field) / do_not_retry (illegal field) | implemented |     |
| binding_invalid                | 400            | correct_and_retry                               | implemented |     |
| constraint_violation            | 400            | correct_and_retry                               | implemented |     |
| edit_conflict                   | 409            | refresh_then_retry                              | implemented |     |
| not_found                       | 404            | do_not_retry                                    | implemented |     |
| write_refused                   | 403            | do_not_retry                                    | implemented |     |
| request_invalid                 | 400            | headline correct_and_retry; wire `retryable` always false | implemented |     |
| resource_exhausted               | 400            | do_not_retry                                    | implemented |     |
| frame_unavailable                | 404            | refresh_then_retry (expired, ops_unsupported, recipe_pending) / do_not_retry (revoked, missing_rendering, digest_mismatch, quarantined, unknown) | implemented |     |
| release_mismatch                 | 503            | retry_after_delay                               | implemented |     |
| service_unavailable              | 503            | retry_after_delay                               | implemented |     |
| timeout                          | 504            | retry_after_delay                               | implemented |     |
| internal_error                   | 500            | do_not_retry                                    | implemented |     |
| unsupported_schema_version         | 422            | do_not_retry                                    | implemented |     |
| refresh_not_permitted            | 403            | do_not_retry                                    | implemented |     |
| render_target_mismatch           | 409            | do_not_retry                                    | implemented |     |

### FrameUnavailable outcome by reason

`_FRAME_UNAVAILABLE_OUTCOME` (`presentation_api.refusals`) names the code and retryability of each reason in `assembler.ports.FrameUnavailableReason`, and importing the module fails if the two differ, and a bare `spec_id` whose newest version is still building refuses `service_unavailable` with `Retry-After` and no `cause`. HTTP is the status the code registry gives the code on the bound endpoint: this table shows the write, read and share routes, and on a render route `frame_unavailable` answers 409. `cause.code` carries the reason on every code. `io` is a failure to obtain a frame's bytes or the memory to read them, and it carries `Retry-After` when the memory ran out or the Datasets refusal behind it named a wait. `busy` is a bare `spec_id` refused because the Datasets surface is at capacity, with its `Retry-After`. Malformed stored content refuses `missing_rendering`, and the store logs an error naming the digest and the cause: bytes the parsers cannot parse, a table its buffers contradict, and a cell that fails its stored shape, including one that breaks the structural view, such as a `unit` struct with no variant or a null `time_period`. A raised defect while reading stored content is no refusal: it reaches `internal_error`, with a note naming the digest on its logged traceback. The platform logs each frame refusal whose code is not `frame_unavailable`, except `busy`, and `digest_mismatch`, as an error. `recipe_pending` and `recipe_unreadable` refuse a pin, so only the share routes raise them.

| Reason            | Code                | HTTP | retryable |
| ----------------- | ------------------- | ---- | --------- |
| expired           | frame_unavailable   | 404  | true      |
| revoked           | frame_unavailable   | 404  | false     |
| ops_unsupported   | frame_unavailable   | 404  | true      |
| missing_rendering | frame_unavailable   | 404  | false     |
| digest_mismatch   | frame_unavailable   | 404  | false     |
| quarantined       | frame_unavailable   | 404  | false     |
| recipe_pending    | frame_unavailable   | 404  | true      |
| unknown           | frame_unavailable   | 404  | false     |
| io                | service_unavailable | 503  | true      |
| busy              | service_unavailable | 503  | true      |
| recipe_unreadable | internal_error      | 500  | false     |

## Registry binding

`presentation_api.codes.PRESENTATION_SERVED_CODES`, the 23 codes of
`PRESENTATION_CODE_CATALOGUE`, `SHARE_CODE_CATALOGUE`, `PRESENTATION_SERVED_PLATFORM_CODES` and
`WRITE_SERVED_PLATFORM_CODES`, is exactly the set of active spec-kit registry entries whose
mappings name a `presentation.*` endpoint. `tests/platform/test_registry_binding.py`'s
`test_served_codes_match_registry` asserts this bijection and passes.
`PRESENTATION_SERVED_PLATFORM_CODES` is what carries `C12`'s `unsupported_schema_version` into
that comparison: it has no `CODE_RETRY_MODE` entry, since that table mirrors the baked catalogue
manifest's `retry_modes` byte for byte, so `CODE_RETRY_MODE`'s own key set alone would leave the
code out of the served set on both sides.

`presentation_api.routes.ROUTE_ENDPOINTS` names the registry endpoint of each route by method and
path: `presentation.write` for create and revise, `presentation.component_read` for C5,
`presentation.component_data` for the data route, `presentation.render` for C7b and C7c,
`presentation.frames` for both frames routes, and `presentation.shares` for every shares door
route. A door reads `registry_endpoints` from it, so a route with no row fails when the door is
built. The release and meta routes serve no endpoint and classify nothing.

Every classifier on this surface takes a refusal's status from the registry under the bound
endpoint: `dfo_platform.problem.refused` gives the status from the code, `request_validation_spec`
restates `validation_problem`'s, and `classify_live_refresh_exception` calls `registered_status`.
`version.py` renders its hashed exemplars at a fixed `status=400`, which reaches only the hash.
The platform's own defaults (`dfo_platform.refusal`), such as `timeout` and an unclassified
`internal_error`, keep their statuses. Each door declares `endpoint_exemplars`, an exemplar of
every refusal each endpoint's routes raise, and `presentation_surface` classifies them with the
endpoint bound (`dfo_platform.doors.check_endpoint_exemplars`), so startup fails on a pair the
registry leaves unmapped, a door whose exemplars do not name exactly the endpoints it binds, and
a declared exception type with no exemplar. Classifiers are pure: a refusal that wants a log line
names it (`ProblemSpec.log`), and the platform logs it on the request path, so the startup sweep
logs nothing. `compute_presentation_capability_version` hashes the status of each code on each
`presentation.*` endpoint, `ROUTE_ENDPOINTS`, the render routes' replay rule, and the outcomes of
frame refusals and of a bare `spec_id` still building.
`tests/presentation_api/test_refusal_registry.py` asserts that every route but the release and
meta routes is bound, that one door owns each endpoint, and that each exemplar goes out at the
one status the registry gives its code there.

The Skills surface reads its classifiers' statuses the same way and runs the same startup check.
The Search and Graph surfaces bind
their routes (`Surface.registry_endpoints`) and still state literal statuses in their classifiers,
which their refusal sweeps check against the registry.
