# Presentation API

The Presentation API manages interactive presentation frames, component specifications, and layout sessions for the Development Finance Observatory (DFO). It binds observation data from dataset frames into structured visual components (such as line charts, stat cards, and bar charts), tracks revisions across user sessions, and persists component layouts in PostgreSQL.

The surface mounts on the shared FastAPI platform at `/api/presentation` when enabled by environment configuration (`DFO_PRESENTATION_SURFACE=1` or `true`).

---

## Architecture and Doors

The surface exposes four doors through a single router, error classifier, and lifespan context.

```
/api/presentation
├── /healthz                              (Meta door)
├── /capabilities                         (Meta door)
├── /release                              (Release door)
├── /frames/{spec_id}/{content_version}   (Frames door)
└── /components[/{component_id}]          (Components door)
```

### 1. Meta Door

Handles liveness, build identification, and the deployment's capability version. Mounted first for static path precedence.

| Method | Endpoint                         | Description                                                                                       |
| :----- | :------------------------------- | :------------------------------------------------------------------------------------------------ |
| `GET`  | `/api/presentation/healthz`      | Returns liveness status, build commit, and capability version (`{"status": "ok", "build": ..., "capability_version": ...}`). |
| `GET`  | `/api/presentation/capabilities` | Returns `{capability_version, catalog_version, build}`, publicly cacheable for five minutes.      |

`capability_version` is a content hash over the release document as `/release` serves it, including schema wording, plus the intent grammar and the refusal surface, so it names what the deployment can do. `catalog_version` names the release that produced a given payload; inside that hash, its value is pinned to a placeholder. A catalogue republish under an otherwise unchanged grammar moves `catalog_version` without moving `capability_version`.

Every response, healthz and capabilities included, also carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing: total` header; `build` in both bodies equals the commit `x-dfo-commit` carries.

### 2. Release Door

Exposes metadata for the pinned component catalogue release (`BakedReleaseRegistry`). Clients and tool bridges (such as `presentation-mcp`) query this endpoint to discover supported component schemas, constraint definitions, retry modes, and minimal intent structures.

| Method | Endpoint                    | Description                                                                                                                                                          |
| :----- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/presentation/release` | Returns catalogue versions (`catalog_version`, `schema_version`), member types, constraint IDs, retry modes, the envelope JSON Schema and its per-member-type component schemas, tolerant intent schemas, and minimal intent structures. |

### 3. Frames Door

Inspects underlying observation frames (`get_frame_view` / `describe_frame`) across dataset specifications and content versions.

| Method | Endpoint                                               | Description                                                                   |
| :----- | :----------------------------------------------------- | :---------------------------------------------------------------------------- |
| `GET`  | `/api/presentation/frames/{spec_id}/{content_version}` | Returns structural dimension summaries, full facet lists, or filtered values. |

Query parameters:

- `dimension`: Isolates a single dimension and returns its complete list of permissible values.
- `query`: Filters dimension values across all facets by substring match.

Each dimension lists its non-null values in `values`, counted by `total_count`. `plural` also counts a null, so a `unit` dimension holding `USDollar` on some rows and null on others is plural with one listed value.

### 4. Components Door

Handles creation, revision, and readback of visual components and layout compositions via `assembler.pipeline`.

| Method  | Endpoint                                                 | Description                                                                  |
| :------ | :------------------------------------------------------- | :--------------------------------------------------------------------------- |
| `POST`  | `/api/presentation/components`                           | Creates and binds a new component within an optional composition layout.     |
| `PATCH` | `/api/presentation/components/{component_id}`            | Revises an existing component draft, validating incremental intent updates.  |
| `GET`   | `/api/presentation/components/{component_id}[?revision]` | Reads back a stored component definition by ID and optional revision number. |
| `GET`   | `/api/presentation/components/{component_id}/revisions/{revision}/data[?format&bundle]` | Downloads the frame rows behind every value one revision shows, as CSV, XLSX, or Parquet, in a zip bundle by default (`README.md`, the data file, `data/codes.csv`, `SOURCES.md`, `CITATION.cff`, `datapackage.json`). The download is named `dfo_<card-title-slug>_<YYYY-MM-DD>_<format>_r<revision>.zip`. |

---

## Quickstart

A create request carries the intent body:

```http
POST /api/presentation/components HTTP/1.1
Content-Type: application/json

{
  "intent": {
    "type": "line_chart",
    "frame_ref": {
      "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
      "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
    },
    "select": {
      "stat_var": "dcid:MalariaGrantDisbursements",
      "recipient": "country/KEN"
    },
    "window": { "start": "2015", "end": "2024" }
  }
}
```

`frame_ref` is a placeholder above. A live `spec_id`/`content_version` pair comes from the Datasets API, pinned to the frame the component is about to bind against.

The reply carries the new component's id, its fact summary, its resolved payload, an affordance naming the routes that revise or read it back, and the component's `write_token`. A create with `composition_id: "new"` also carries `composition_write_token`. The example below omits `resolved_payload`, which [Read Reply](#read-reply-get) shows in full:

```json
{
  "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
  "revision": 1,
  "catalog_version": "2026.8.0",
  "composition_id": null,
  "composition_revision": null,
  "changed": [],
  "fact_summary": {
    "changed": [],
    "series": [
      {
        "id": "series_1",
        "label": "Malaria grant disbursements, Kenya",
        "coordinates": { "recipient": "country/KEN" },
        "point_count": 9,
        "first_value": "12.4",
        "last_value": "18.7"
      }
    ],
    "series_cut": 0,
    "window": { "start": "2015", "end": "2024" },
    "unit": null,
    "change": null,
    "source_refs": null,
    "defaults": []
  },
  "affordance": "Revise this component with PATCH /api/presentation/components/cmp_01ARYZ6S41YYYYYYYYYYYYYYYY, sending only the fields to change and this reply's write_token as x-dfo-write-token, or read it back with GET /api/presentation/components/cmp_01ARYZ6S41YYYYYYYYYYYYYYYY.",
  "write_token": "cmw_Zm9vYmFyYmF6cXV4cXV1eA"
}
```

---

## Interaction Model

### Read grant and write grant

The `cmp_` component id and the `cps_` composition id grant read. Anyone holding a component id can read that component at any revision. Each id is a ULID: 48 bits of creation time and 80 bits from `os.urandom`, so an id cannot be guessed or enumerated. An id that names no draft returns HTTP `404` with problem code `not_found`.

A write token grants write. A create returns a `cmw_` token for the new component as the top-level `write_token` field, and a create with `composition_id: "new"` also returns a `cpw_` token for the new composition as `composition_write_token`. Each token is its prefix plus `secrets.token_urlsafe(16)`, 128 random bits. A create reply is the only place a token appears: the store keeps its unsalted SHA-256 digest, and no read, revise reply, stored intent, or resolved payload carries it.

- A revise sends the component's token as the `x-dfo-write-token` header. A composition member needs only its own component token.
- A create into an existing composition sends that composition's token in the same header.

A missing or wrong token, or a draft created before write tokens and so holding no digest, returns HTTP `403` with problem code `write_refused`, retry mode `do_not_retry`. The caller creates a new component instead, and may read the old one first to reuse its intent. The API checks no caller, session, or owner.

The proxy layer authenticates the client. The `/components` and `/frames` routes read no session header. A request that sends `x-dfo-session` is served as if it had not.

### Idempotent create

A create may send its own component token in the body as `write_token`: `cmw_` plus exactly 22 url-safe base64 characters, minted as `"cmw_" + secrets.token_urlsafe(16)` or from an equivalent CSPRNG. The token becomes the component's write token and keys the create, so a client that retries a timed-out or failed create sends the same token on every attempt:

```json
{ "intent": { "type": "line_chart", "...": "..." }, "write_token": "cmw_Zm9vYmFyYmF6cXV4cXV1eA" }
```

- A create whose token names a stored component, sent with the same `intent` and `composition_id`, answers `201` with that component's create reply rebuilt from revision 1, carrying the same token. It resolves and pins no frame. `composition_revision` reports the composition's head at replay time.
- The same token with a different request returns `400` `request_invalid` at pointer `/write_token`, as does a token held by a component whose create stored no request hash. Mint a new token for each create.
- Two creates racing with one token store one component: the second waits on the first's uncommitted row and then answers with the first's reply.
- A malformed token, or a token sent with `composition_id: "new"`, returns `400` `request_invalid` at `/write_token`. A revise body cannot carry `write_token`.
- An intent the request hash cannot canonicalise, such as an integer beyond 2^53, returns `400` `request_invalid` at `/intent` when sent with a token.

The request hash is the SHA-256 of the RFC 8785 canonical JSON of `{"intent", "composition_id"}` as sent, so reordered keys still match. Without a body `write_token` the server mints the token and the create is not idempotent: a retry after a timeout can store a second draft. The `x-dfo-write-token` header keeps its meaning on a create, the token of the composition the component joins.

### Request and Reply Structures

Component creation and revision accept an intent document specifying component type, data bindings, frame references, and visual encodings. Supported member types are `line_chart`, `stat_card`, `bar_chart`, `table`, `note_narration`, and `source_card`.

#### Request Payload (`POST` / `PATCH`)

```json
{
  "intent": {
    "type": "line_chart",
    "frame_ref": {
      "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
      "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
    },
    "select": {
      "stat_var": "dcid:MalariaGrantDisbursements",
      "recipient": "country/KEN"
    }
  },
  "composition_id": "cps_01ARYZ6S41ZZZZZZZZZZZZZZZZ"
}
```

`frame_ref` and the `select` values are placeholders above. A live `spec_id`/`content_version` pair comes from the Datasets API, pinned to the frame the component is about to bind against, and the `select` values come from bindings valid against that frame.

The `composition_id` field is set during initial creation (`POST`), either referencing an existing composition or `"new"` to mint one. Supplying `composition_id` on `PATCH` returns a `binding_invalid` rejection (`pointer="/composition_id"`).

#### Write Reply (`WriteReply`)

Responses to `POST` (201 Created) and `PATCH` (200 OK) return a structured write acknowledgement. A revise response, `resolved_payload` omitted:

```json
{
  "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
  "revision": 2,
  "catalog_version": "2026.8.0",
  "composition_id": "cps_01ARYZ6S41ZZZZZZZZZZZZZZZZ",
  "composition_revision": 1,
  "changed": [
    {
      "field": "title",
      "prior": "Malaria grant disbursements",
      "new": "Malaria grant disbursements, Kenya"
    }
  ],
  "fact_summary": {
    "changed": ["title"],
    "series": [
      {
        "id": "series_1",
        "label": "Malaria grant disbursements, Kenya",
        "coordinates": { "recipient": "country/KEN" },
        "point_count": 9,
        "first_value": "12.4",
        "last_value": "18.7"
      }
    ],
    "series_cut": 0,
    "window": { "start": "2015", "end": "2024" },
    "unit": null,
    "change": null,
    "source_refs": null,
    "defaults": []
  }
}
```

A revise response omits the `affordance` key. A create response carries the same shape with `changed` empty (`[]`), `revision` `1`, and an `affordance` string naming the routes that revise or read the component back. The Quickstart above shows one. In `changed`, scalar value transitions record explicit `prior` and `new` keys, while replaced or cleared objects and arrays record only `field`.

`fact_summary` always carries eight keys: `changed`, `series`, `series_cut`, `window`, `unit`, `change`, `source_refs`, and `defaults`. `series_cut` counts series dropped for exceeding the per-summary series cap of 12, plus any further series dropped once the summary crosses its token budget. `window` and `unit` are null for a member the grammar defines neither for. A component's string `unit` holds the unit's graph name, or its dcid when the graph has no name; structured units are unchanged. `change` carries a stat_card's derived comparison and is null for every other member. `source_refs` carries a source_card's provenance list and is null for every other member. `defaults` lists each choice the component made where its intent left one open, as `{pointer, detail}`, and is empty when it made none. A create's `fact_summary` equals the `fact_summary` a later `GET` of the same revision returns.

`composition_id` and `composition_revision` name the real composition and its current head revision on a create, a substantive revise, a no-op revise, and a `GET`, for a component that belongs to a composition. Both are null on every operation for a component that belongs to no composition.

`resolved_payload` is the stamped envelope stored for the reply's revision, verbatim: `{schema_version, catalog_version, component_id, revision, provenance, component}`, validated when minted against the envelope schema of the catalogue release its `catalog_version` names. [Release Door](#2-release-door) serves only the pinned release's `envelope_schema` and `component_schemas`, the per-member-type schemas `envelope_schema`'s `component` field `$ref`s by `$id`. A deployment reads every stored revision. The envelope keeps the `catalog_version` it was minted under, and the `fact_summary` is formatted with the formatter profile stored on the revision. A create or revise mints under the pinned release, and a revise under a new release mints a new revision carrying it, even when the component content is unchanged. A write whose head carries a newer `catalog_version` than the pinned release refuses `release_mismatch`, so a write cannot move a head back to an older release. Only an instance left behind by a rollback refuses this way, and its `retry_after_delay` means retrying after the newer release is deployed again. `component` carries every resolved point, so a renderer draws from it without a frame read. A no-op revise returns the head revision's stored envelope unchanged. The create, revise, and read replies for one revision carry the same stored envelope, as equal parsed JSON values. The create and revise doors serve it in mint key order. `GET` serves it in the key order Postgres returns from JSONB storage, and a number can come back reformatted (`1e2` as `100`). A consumer that compares or hashes the bytes should canonicalise first.

#### Read Reply (`GET`)

`GET /api/presentation/components/{component_id}` returns the head revision's stored intent, its fact summary with `changed` empty, and its resolved payload. After the revise above:

```json
{
  "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
  "revision": 2,
  "catalog_version": "2026.8.0",
  "composition_id": "cps_01ARYZ6S41ZZZZZZZZZZZZZZZZ",
  "composition_revision": 1,
  "stored_intent": {
    "type": "line_chart",
    "frame_ref": {
      "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
      "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
    },
    "select": {
      "stat_var": "dcid:MalariaGrantDisbursements",
      "recipient": "country/KEN"
    },
    "window": { "start": "2015", "end": "2024" },
    "title": "Malaria grant disbursements, Kenya"
  },
  "fact_summary": {
    "changed": [],
    "series": [
      {
        "id": "series_1",
        "label": "Malaria grant disbursements, Kenya",
        "coordinates": { "recipient": "country/KEN" },
        "point_count": 9,
        "first_value": "12.4",
        "last_value": "18.7"
      }
    ],
    "series_cut": 0,
    "window": { "start": "2015", "end": "2024" },
    "unit": null,
    "change": null,
    "source_refs": null,
    "defaults": []
  },
  "resolved_payload": {
    "schema_version": 1,
    "catalog_version": "2026.8.0",
    "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
    "revision": 2,
    "provenance": {
      "inputs": [
        {
          "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
          "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
        }
      ],
      "execution_ref": null,
      "sources": [
        {
          "id": "src_oecd_dac",
          "name": "OECD DAC",
          "url": "https://stats.oecd.org/",
          "licence": "CC BY 4.0",
          "retrieved": "2026-06-12"
        }
      ],
      "freshness": {
        "data_through": "2024",
        "last_refreshed": "2026-06-12",
        "next_expected": "2026-10",
        "partial_period": false
      },
      "warnings": [],
      "generated_at": "2026-07-14T10:02:00Z",
      "citation": "OECD DAC via ONE DFO, retrieved 12 June 2026",
      "refreshed_from": null
    },
    "component": {
      "type": "line_chart",
      "title": { "text": "Malaria grant disbursements, Kenya", "authorship": "model" },
      "series": [
        {
          "id": "series_1",
          "label": "Malaria grant disbursements, Kenya",
          "unit": null,
          "source_ref": "src_oecd_dac",
          "points": [
            { "period": "2015", "value": 12.4 },
            { "period": "2016", "value": 13.1 },
            { "period": "2017", "value": 14.0 },
            { "period": "2018", "status": "missing" },
            { "period": "2019", "value": 15.2 },
            { "period": "2020", "value": 15.9 },
            { "period": "2021", "value": 16.8 },
            { "period": "2022", "value": 17.3 },
            { "period": "2023", "value": 18.1 },
            { "period": "2024", "value": 18.7 }
          ]
        }
      ],
      "x_axis": { "window": { "start": "2015", "end": "2024" } },
      "y_axis": {
        "label": "dcid:MalariaGrantDisbursements",
        "scale": "linear",
        "range": { "min": null, "max": null }
      }
    }
  }
}
```

#### Intent grammar: scale, range, and change

`line_chart` and `bar_chart` accept an authored `scale`, `linear` or `log`, and an authored `range`, `{min, max}`, each a number or null. Both `range` keys are required together. Absent means `linear` and an unbounded range that renders from the data. `line_chart` resolves these onto `y_axis`. `bar_chart` resolves them onto `value_axis`, alongside `category_axis`, which names either an entity comparison (one bar per selected entity at their shared observed period) or a period comparison (one series across a window).

A `bar_chart` whose `select` lists values on one dimension draws one bar per value, in list order. A `select` that resolves to one series draws it across the window. A `select` that lists no values and resolves to several series draws one bar per value of the one dimension they vary on, other than a facet, largest first, when the window holds data in one period and every series holds a value in it; `fact_summary.defaults` states this at `/select/<dimension>`, and listing values on that dimension narrows the chart. Any other selection refuses `binding_invalid` at `/select`: series that vary on two dimensions name both with their value counts and list the values of the ones to bind to one value, series that hold data in several periods name the periods and the two calls that fit, one value to draw its periods or a list to compare at the latest shared period, and series of which some lack a value in the one period name the values that lack it and the call that lists the values to draw. `alternatives` carries the values of the first varying dimension, or for a period some series lack, the values observed in it.

`stat_card` accepts an authored `change`. `basis` names two periods, `[from, to]`, both of which must be observed points of the primary series. `derivation` selects how the comparison is computed, `difference` or `percent_change`, absent meaning `difference`. `percent_change` needs a positive `from` value; a zero or negative basis has no defined ratio and refuses `binding_invalid` at `/change`. The resolved `change` carries `derivation`, `basis`, and the computed `value`.

### Revision-scoped reads

`GET /api/presentation/components/{component_id}?revision=N` scopes the read to one stored revision: its stored intent, its fact summary, and its own resolved payload. An unknown component id, or a revision number outside the component's revision range, returns 404 `not_found`, retryable `false`. `0` and a negative number return `request_invalid`, since `revision` accepts a positive integer only. Any revise or read whose target has vanished or been rewritten between this call's own read and its write refuses `edit_conflict`. A revise that starts after another revise of the same component commits rebases onto the new head. Two overlapping revises of a standalone component leave one refusing `edit_conflict` (409, retryable `true`). Revises of composition members serialise on the composition and rebase.

### Frame Coordinates and JSONB Serialisation

Component creation and revision resolve series coordinates (`fact_summary.series[].coordinates`) from bindings against the frame.

A selector may leave the facet dimensions `unit`, `measurementMethod`, and `provenance` unbound unless one of them splits a series. Within one selector's matches, `assembler.resolve` keys each series on every coordinate except the unbound facets, and refuses `binding_invalid` at `/select/<facet>` when one key holds more than one value of a facet, a null counting as a value. The first such facet in column order is named, and `alternatives` lists the non-null values that key holds. Series that differ in other coordinates may carry different facet values, so GDP in `USDollar` beside population with a null unit resolves as two series. A line chart over them refuses `comparable_units_required`.

PostgreSQL persists these coordinate mappings in a dedicated `series_coordinates` JSONB column alongside the revision payload in `presentation.component_revision`. On `GET /api/presentation/components/{component_id}`, the response reconstructs coordinate definitions matching the original create/revise response.

List-valued parameters (such as `series_coordinates` and `blocks`) bound in SQL statements are pre-serialised to JSON text strings to match PostgreSQL JSONB types under `pg8000`.

---

## Refusals and Error Classification

Errors conform to RFC 9457 Problem Details.

The per-response `retryable` field is authoritative for every code. `retry_modes` on the release document names each code's headline mode, the mode a caller reads without seeing any individual response. `schema_invalid` and `frame_unavailable` carry more than one real mode, so `retryable` differs by the cause behind a given occurrence of either. `request_invalid` reports retryable `false` on every occurrence, even though its headline mode is `correct_and_retry`.

| Error Code                   | HTTP Status | Retry Mode           | Description / Disclosure Fields                                                                                                                                                                                                                                                                                                       |
| :--------------------------- | :---------- | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `schema_invalid`             | 400         | Contextual           | Intent fails schema. A missing required field is `correct_and_retry` and returns the type's minimal legal intent in `detail`. An illegal field for the type is `do_not_retry` and returns the type's legal fields as `alternatives`.                                                                                                  |
| `binding_invalid`            | 400         | `correct_and_retry`  | Unbound selectors or invalid values. Returns `alternatives`, `truncated`, and `issues`. Each alternative pairs a legal value with the label the frame resolved for it, falling back to the value itself when the frame names none.                                                                                                    |
| `constraint_violation`       | 400         | `correct_and_retry`  | Violates catalogue constraints. Returns `constraint_id`, `pointer`, and `detail`. `pointer` is expressed against the intent, for example `/scale`, `/range`, or `/change/basis`.                                                                                                                                                      |
| `edit_conflict`              | 409         | `refresh_then_retry` | The component, a requested revision, or on create the referenced composition vanished or was rewritten mid flight, raised by a scoped or unscoped `GET` as well as by writes.                                                                                                                                                         |
| `frame_unavailable`          | 404 / 503   | Contextual           | `unknown` (404, not retryable) names a `spec_id`/`content_version` pair that was never stored. `expired` (404, retryable) names a stored version that aged out. `ops_unsupported` (404, retryable) names an `ops_`/`ocv_` pair analysis-mcp minted but never published to the platform; publishing it is the retry that succeeds. `revoked` and the other permanent reasons (`missing_rendering`, `digest_mismatch`) return 404, not retryable. `io` returns 503, retry after delay. |
| `resource_exhausted`         | 400         | `do_not_retry`       | Selection exceeds allowed limits, including a minted component whose `component` plus `provenance` subtrees serialise past `MAX_ENVELOPE_BYTES` (2 MiB, compact UTF-8 JSON), refused before any store write. Returns `limit_name` in `detail`.                                                                                                                                                                                                                                                                   |
| `release_mismatch`           | 503         | `retry_after_delay`  | The component's head carries a newer catalogue release than this deployment writes, which happens after a rollback and clears once the newer release is deployed again, or a stored payload has a shape the release's schema does not recognise, such as a bar_chart row migration 0006 has not rewritten. |
| `request_invalid`            | 400         | `correct_and_retry`  | Retryable is always `false`. A malformed body field, an unparseable path identifier, or a non-positive `revision` query parameter. Also a create `write_token` reused with a different request.                                                                                                                                                             |
| `unsupported_schema_version` | 422         | `do_not_retry`       | A top-level `schema_version` field in the request body. A platform code outside the release document's `retry_modes`.                                                                                                                                                                                                                 |
| `not_found`                  | 404         | `do_not_retry`       | Unknown component or composition id (`DraftNotFound`), or a revision number never minted for a revision-scoped `GET`.                                                                                                                                                                                                                  |
| `write_refused`              | 403         | `do_not_retry`       | A revise without the component's write token, or a create into an existing composition without that composition's token, in `x-dfo-write-token`. Also a draft created before write tokens. `detail` ends with a fix: create a new component.                                                                                        |
| `service_unavailable`        | 503         | `retry_after_delay`  | Database failure or door thread pool backpressure exhaustion.                                                                                                                                                                                                                                                                         |
| `timeout`                    | 504         | `retry_after_delay`  | Execution exceeded door timeout threshold.                                                                                                                                                                                                                                                                                            |
| `internal_error`             | 500         | `do_not_retry`       | Permanent non-retryable store or system failure.                                                                                                                                                                                                                                                                                      |

---

## Stored revisions

Every stored `bar_chart` revision uses `category_axis` and `value_axis` keys. Revisions are append-only. A `component_revision` row changes only through a numbered migration under `src/presentation_store/migrations/`.

---

## Configuration and Deployment

The surface mounts through the platform application factory when `DFO_PRESENTATION_SURFACE` is set to `1` or `true`.

### Environment Variables

| Variable                                  | Default         | Description                                                                     |
| :---------------------------------------- | :-------------- | :------------------------------------------------------------------------------ |
| `DFO_PRESENTATION_SURFACE`                | `false`         | Enables mounting of the Presentation API surface on `/api/presentation`.        |
| `PRESENTATION_COMPONENTS_MAX_CONCURRENCY` | `4`             | Worker thread pool size and backpressure limit for component read/write routes. |
| `PRESENTATION_COMPONENTS_TIMEOUT_SECONDS` | `30`            | Maximum execution timeout for component store operations.                       |
| `PRESENTATION_FRAMES_MAX_CONCURRENCY`     | `4`             | Worker thread pool size and backpressure limit for frame inspection routes.     |
| `PRESENTATION_FRAMES_TIMEOUT_SECONDS`     | `30`            | Maximum execution timeout for frame store operations.                           |
| `PRESENTATION_STORE_RUNTIME_DB_USER`      | _(Required)_    | Database username for presentation store connection pooling.                    |
| `DATASET_STORE_RUNTIME_DB_USER`           | _(Required)_    | Database username for shared dataset and frame store connections.               |
| `DATASETS_GRAPH_BASE_URL`                 | _(Required)_    | Base URL for upstream graph reader resolution.                                  |
| `UN_GRAPH_BASE_URL`                       | UN System DC    | API root for `undata/` dcids, shared with every surface and the catalogue job.  |
| `ORIGIN_CATALOGUE_URI`                    | _(Required)_    | Origin catalogue root; a missing or unloadable catalogue fails startup.         |
| `DATASETS_SDMX_BASE_URL`                  | _(Required)_    | Base URL for SDMX client endpoints.                                             |
| `DATASETS_ARTIFACT_BUCKET_WORKING`        | _(Required)_    | GCS bucket name for unpinned/working frame artifacts.                           |
| `DATASETS_ARTIFACT_BUCKET_PINNED`         | _(Required)_    | GCS bucket name for pinned presentation frame artifacts.                        |
| `DFO_PUBLIC_SITE_URL`                     | _(Required)_    | Website origin the components door builds export bundle links on.               |

### Concurrency and Thread Pools

Component and frame operations execute in dedicated `ThreadPoolExecutor` instances with bounded semaphores (`PRESENTATION_COMPONENTS_MAX_CONCURRENCY` and `PRESENTATION_FRAMES_MAX_CONCURRENCY`). When execution permits are saturated, incoming requests return HTTP `503` with problem code `service_unavailable` immediately.
