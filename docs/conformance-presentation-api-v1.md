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
| C6     | GET /api/presentation/frames/{spec_id}/{content_version} describes a frame (structural view, one dimension's values, or a query-filtered view) | implemented |     |
| C7     | GET /api/presentation/release returns the pinned catalogue release's identity, retry-mode table, envelope JSON Schema, and its per-member-type component schemas | implemented |     |
| C7a    | GET /api/presentation/components/{component_id}/revisions/{revision}/data serves the frame rows behind every value that revision shows, as `format=csv`, `xlsx`, or `parquet` (default `csv`), in a zip bundle by default or as the bare file with `bundle=false`. It sends `Content-Disposition: attachment; filename="dfo_{card-title-slug}_{YYYY-MM-DD}_{format}_r{revision}.zip"` for a bundle and `filename="dfo_{card-title-slug}_{YYYY-MM-DD}_r{revision}.{csv\|xlsx\|parquet}"` for the bare file, and `Cache-Control: private, max-age=3600`. The slug is the card's title in lowercase ASCII words joined by hyphens (the component id when the card has no title), and the date is the UTC date the revision was stored, so one revision keeps one name | implemented |     |

## Request shape

| Clause | Requirement                                                                                                                                                     | Status      | Gap |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --- |
| C8     | A `cmp_` component id grants read: anyone holding it reads that component at any revision, with no header. Each id is a ULID with 80 random bits (`os.urandom`). An id that names no draft refuses `not_found`, 404, `retryable` false | implemented |     |
| C8a    | A write token grants write. A create returns a `cmw_` token as top-level `write_token`, and a create with `composition_id: "new"` also returns a `cpw_` token as `composition_write_token`, each `secrets.token_urlsafe(16)` behind its prefix. The store keeps the unsalted SHA-256 digest. No read, revise reply, stored intent, or resolved payload carries a token | implemented |     |
| C8b    | A revise needs the component's token in `x-dfo-write-token`; a composition member needs only its own. A create into an existing composition needs that composition's token in the same header. A missing or wrong token, or a draft with no stored digest, refuses `write_refused`, 403, `retryable` false, before any frame is pinned, with a fix telling the caller to create a new component | implemented |     |
| C9     | No door reads a session header. A request that sends `x-dfo-session`, with any value, an empty value, or the header repeated, is served exactly as one without it | implemented |     |
| C10    | `component_id`, `spec_id`, and `content_version` path parameters are format-checked before reaching a store lookup; a malformed value refuses `request_invalid`, 400, `retryable` false | implemented |     |
| C11    | The create/revise request body (`ComponentBody`) is closed (`extra="forbid"`): `intent` plus an optional `composition_id`; any other top-level field refuses through pydantic validation | implemented |     |
| C12    | A body carrying `schema_version` as a top-level field (a field `ComponentBody` does not declare) refuses `unsupported_schema_version`, 422, `retryable` false, distinct from the generic closed-body refusal | implemented |     |

## Component bodies

| Clause | Requirement | Status | Gap |
| ------ | ----------- | ------ | --- |
| C17    | The create (201), revise (200) and read (200) bodies carry `resolved_payload`, the revision's stored envelope (`presentation.component_revision.resolved_payload`), verbatim. The three bodies are equal as parsed JSON. The create and revise doors serve it in mint key order. The read door serves it in the key order Postgres returns from JSONB storage, and a number can come back reformatted. A no-op revise carries the head's stored envelope, and a `?revision=N` read carries revision N's. | implemented |     |
| C18    | `resolved_payload` validates against the `envelope_schema` of the catalogue release its `catalog_version` names, the same `CatalogueRelease.envelope_schema` `assembler.mint.stamp_envelope` validates a minted envelope against, with `envelope_schema`'s `component` field's `$ref`s resolving against that release's `component_schemas` by `$id`. GET /api/presentation/release serves only the pinned release's schemas, so a payload validates against that response only when its `catalog_version` is the pinned release. A read serves every stored revision with the `catalog_version` it was minted under and formats its fact summary with the formatter profile stored on it (`assembler.pipeline.get_component`). A write whose head carries a newer `catalog_version` than the pinned release refuses `release_mismatch` (`assembler.pipeline.write._require_release_not_behind`) | implemented |     |
| C19    | A minted component's `component` and `provenance` subtrees are measured in compact UTF-8 JSON against `MAX_ENVELOPE_BYTES` (2 MiB) before any store write; a candidate over the cap refuses `resource_exhausted`, `limit_name=MAX_ENVELOPE_BYTES`, with nothing pinned or stored | implemented |     |
| C20    | A revise whose resolved payload is otherwise unchanged but whose `catalog_version` differs from the pinned release's mints a new revision carrying the pinned release, rather than returning the head as a no-op (`assembler.pipeline.write._structurally_equal`) | implemented |     |
| C20a   | A chart's series and a table's rows follow the order `select` lists the values: by position in each list-valued dimension's list, dimensions taken in `select` key order (`assembler.resolve.resolve_selectors`). Series the lists leave unordered, such as those an unbound dimension expands into, keep frame row order. So `{"financingProvider": ["country/USA", "GlobalFund"]}` draws USA first even when the frame stores Global Fund's rows first. The fact summary's `series` and the data export's rows take the payload's order | implemented |     |

## Component data export

| Clause | Requirement | Status | Gap |
| ------ | ----------- | ------ | --- |
| C21    | The export holds one frame row per value the revision shows: each valued point of a line or bar chart series, each valued table row, and a stat card's headline period plus both change basis periods. Rows are matched by the series' stored `series_coordinates`, normalised as `resolve_selectors` groups them, and the period (`assembler.pipeline.export_component`) | implemented |     |
| C22    | When a shown value matches no frame row, two rows, a row with a non-null `status`, or a row whose `value` differs from the card's, and when a table revision's rows carry `series_id` on only some rows or carry a non-string or empty one, the route refuses `internal_error`, 500, logs the cause, and serves no file | implemented |     |
| C23    | Columns are `series` (the card's label for the row), then the frame's columns in frame order, with `stat_var_label`, `{role}_label`, and `provenance_label` after their columns, read from the frame's labels. A generated name that a frame column or an earlier generated column holds gains the first free suffix `_2`, `_3`, and so on, so every frame column keeps its own name. Files are written by the Datasets writers (`materialise.formats`), so a CSV or Excel cell starting with a formula trigger gains a leading apostrophe and Excel cells are literal strings | implemented |     |
| C24    | The bundle holds `README.md`, `data/{card-title-slug}.{format}`, `data/codes.csv` (when the rows hold codes), `SOURCES.md`, `CITATION.cff` and `datapackage.json` from `materialise.bundle.assemble_component_bundle`. Sources and licences come from the frame manifest's `citation`, one entry per provenance ref the rows carry, for `spec_` and `ops_` frames alike. A card export of a root `spec_` frame fills the citation fields the stored record left empty from a live read, and a failed read is not cached. The descriptor's `dfo` property holds `component` (`component_id`, `revision`, `frame_ref`, `select`, `window`), `prepared_at` (the revision's stored creation time, which also dates the file name), `links`, and `reproducing_call`, the export route as an absolute URL with its options and no header. The README takes the card title as its heading, says it holds the rows behind that revision, and names the frame's `spec_id` and `content_version`. The bytes match from one request to the next while the graph's code labels and the sources' documentation that a derived frame, or a root frame whose stored citation left fields empty, reads live are unchanged | implemented |     |
| C25    | An unknown component or revision, a `note_narration` or `source_card`, a revision stored without `series_coordinates`, and a table revision none of whose rows carries `series_id` refuse `not_found`, 404, the last with a detail saying the revision predates table row identities and the card must be recreated to export it. A frame the store cannot load refuses `frame_unavailable` by its reason table. An unknown `format` or unparseable `bundle` refuses `request_invalid`, 400. The route shares the components door's semaphore and timeout, so saturation refuses `service_unavailable` and a slow export `timeout` | implemented |     |

## Span periods

| Clause | Requirement | Status | Gap |
| ------ | ----------- | ------ | --- |
| C26    | A frame whose rows share one ISO 8601 interval in `time_period` (`2018/2022`, `2018-01/2022-12`, parsed by `dfo_frame_grammar.parse_time_period`) binds that interval as its one period (`assembler.ports.StructuralView.span`). An entity-mode bar chart states it as `category_axis.period` and each bar's period, a table as each row's `period`, and a stat card as its `period`, and the component data export carries it in each row's `time_period`. A line chart or a period-mode bar chart over such a frame refuses `binding_invalid` at `/type`, with `alternatives` naming `bar_chart`, `table`, and `stat_card`, before its window is read (`assembler.pipeline.candidate`). A frame holding an interval beside any other period, point or interval, refuses `binding_invalid` at `/frame_ref` (`assembler.frame_view.compute_structural_view`) | implemented |     |

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
| frame_unavailable                | 404 or 503 per cause (see below) | refresh_then_retry (expired, ops_unsupported) / do_not_retry (revoked, missing_rendering, digest_mismatch, unknown) / retry_after_delay (io) | implemented |     |
| release_mismatch                 | 503            | retry_after_delay                               | implemented |     |
| service_unavailable              | 503            | retry_after_delay                               | implemented |     |
| timeout                          | 504            | retry_after_delay                               | implemented |     |
| internal_error                   | 500            | do_not_retry                                    | implemented |     |
| unsupported_schema_version         | 422            | do_not_retry                                    | implemented |     |

### frame_unavailable outcome by reason

| Reason            | HTTP | retryable |
| ------------------ | ---- | --------- |
| expired            | 404  | true      |
| revoked             | 404  | false     |
| ops_unsupported      | 404  | true      |
| missing_rendering     | 404  | false     |
| digest_mismatch       | 404  | false     |
| io                  | 503  | true      |
| unknown              | 404  | false     |
| any other reason      | 503  | true      |

## Registry binding

`presentation_api.codes.PRESENTATION_CODE_CATALOGUE` union
`presentation_api.codes.PRESENTATION_SERVED_PLATFORM_CODES` — the seven codes this surface mints
itself, plus the six platform codes its own handlers, its own `validation_problem` path, or the
platform default put on the wire (`not_found`, `request_invalid`, `service_unavailable`, `timeout`,
`internal_error`, `unsupported_schema_version`) — is exactly the set of active spec-kit registry
entries whose mappings name a `presentation.*` endpoint (all thirteen map onto
`presentation.write`). `tests/platform/test_registry_binding.py`'s
`test_presentation_served_codes_match_registry` asserts this bijection and passes.
`PRESENTATION_SERVED_PLATFORM_CODES` is what carries `C12`'s `unsupported_schema_version` into
that comparison: it has no `CODE_RETRY_MODE` entry, since that table mirrors the baked catalogue
manifest's `retry_modes` byte for byte, so `CODE_RETRY_MODE`'s own key set alone would leave the
code out of the served set on both sides.
