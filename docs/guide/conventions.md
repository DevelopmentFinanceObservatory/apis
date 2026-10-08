---
description: This page lists the shared base URL, request limits, RFC 9457 error shape, error codes, schema_version, and x-dfo-trace-id header for the DFO APIs.
---

# Conventions

The Search, Datasets and Graph APIs use `schema_version` 1. Every response names the build that answered in its `x-dfo-commit` header, and `GET /api/{search,datasets,graph}/healthz` returns the same commit as `build`.

## Base URL

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

The Python examples read `DFO_API` from the environment, so run them from the shell where you exported it.

This is the development deployment. The three APIs share one host and differ by path prefix:

| API      | Prefix          |
| -------- | --------------- |
| Search   | `/api/search`   |
| Datasets | `/api/datasets` |
| Graph    | `/api/graph`    |

<!-- TODO: add the public base URL once the API access policy exists. -->

## Access

The development deployment answers every call in this guide without a key or `Authorization` header. One route needs more: `POST /api/datasets/{id}/{content_version}/pin` is limited to a first-party principal and answers `403 pin_not_permitted` to anyone else.

A bundle's `datapackage.json` and `README.md` carry the call that reproduces it as a plain URL and request body. The call needs no key.

## Limits

A proxy in front of every route enforces three limits before the request reaches an API. Handle these statuses on every call:

| Status | Meaning                                                                                                                         |
| ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `429`  | Too many requests. <!-- TODO: rate-limit thresholds; set in proxy config, not recorded in dfo-apis. -->                         |
| `413`  | Request body too large. <!-- TODO: byte ceiling; set in proxy config. -->                                                       |
| `415`  | The media type is unsupported. Send `Content-Type: application/json`. <!-- TODO: accepted media types; set in proxy config. --> |

Each API publishes its own limits at `GET /api/{search,datasets,graph}/capabilities`. The values on the development deployment:

| API      | Limit                                                                           | Value                         |
| -------- | ------------------------------------------------------------------------------- | ----------------------------- |
| Search   | Characters in a `raw_text` query (`max_query_chars`)                            | 2,000                         |
| Search   | Variables read from one query (`max_parts`)                                     | 6                             |
| Search   | Entities per variable (`max_part_entities`)                                     | 16                            |
| Search   | Candidate specs for an ambiguous variable (`max_candidates`)                    | 6                             |
| Datasets | Estimated rows per synchronous mint (`sync_budget_rows`)                        | 100,000                       |
| Datasets | Estimated source requests per synchronous mint (`sync_budget_requests`)         | 40                            |
| Datasets | Estimated rows a background operation accepts (`async_max_rows`)                | 2,000,000                     |
| Datasets | Estimated source requests a background operation accepts (`async_max_requests`) | 800                           |
| Datasets | StatVars a described definition's member walk reads, `definition_subject_ceiling` | 20,000                        |
| Datasets | Cells in a pivoted table (`pivot_max_output_cells`)                             | 5,000,000                     |
| Datasets | Days an operation stays pollable (`operation_retention_days`)                   | 30                            |
| Datasets | Lifetime of a signed artifact URL                                               | 15 minutes                    |
| Graph    | Texts per `POST /api/graph/resolve`                                             | 25, each up to 200 characters |
| Graph    | Characters in a dcid (`max_dcid_chars`)                                         | 512                           |

A mint over either sync budget returns `202 Accepted` and runs in the background. A mint over either background ceiling returns an `over_ceiling` error. [Get a dataset](get-a-dataset.md) covers both cases.

## Errors

Responses outside the 2xx range use an RFC 9457 problem-details body with `Content-Type: application/problem+json`.

```json
{
  "type": "/codes/node_not_held",
  "title": "Node Not Held",
  "status": 404,
  "detail": "graph does not hold 'country/XYZ'",
  "code": "node_not_held",
  "pointer": "/dcid",
  "trace_id": "d2c12a769c654b42818394d1327d7093",
  "capability_version": "graph_0b2015378d1ff897",
  "retryable": false
}
```

:::tabs

::tab{label="curl"}

```bash
curl -s "$DFO_API/api/graph/node/country%2FXYZ"
```

::tab{label="Python"}

```python
import os

import httpx

DFO_API = os.environ["DFO_API"]

print(httpx.get(f"{DFO_API}/api/graph/node/country%2FXYZ").text)
```

:::

The body includes optional fields only when they have values.

| Field                | Type    | Present             | Description                                                                                                            |
| -------------------- | ------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `type`               | string  | always              | Relative URI of the code's registry entry, `/codes/{code}`.                                                            |
| `title`              | string  | always              | Short summary of the problem type.                                                                                     |
| `status`             | integer | always              | The HTTP status.                                                                                                       |
| `detail`             | string  | always              | What went wrong in this request.                                                                                       |
| `code`               | string  | always              | Stable machine-readable code. Branch on this.                                                                          |
| `trace_id`           | string  | always              | This request's trace id, equal to the `x-dfo-trace-id` header.                                                         |
| `capability_version` | string  | always              | Capability version of the deployment that answered.                                                                    |
| `retryable`          | boolean | always              | Whether sending the same request again might succeed.                                                                  |
| `pointer`            | string  | optional            | JSON Pointer to the field at fault.                                                                                    |
| `alternatives`       | array   | optional            | Values that would satisfy the request, each `{ref, label}`.                                                            |
| `truncated`          | boolean | with `alternatives` | Whether the `alternatives` list is truncated.                                                                          |
| `issues`             | array   | optional            | One entry per failure when a code reports several at once. An entry holds the `pointer`, `member`, `axis`, `alternatives` and `truncated` that apply to it. On `default_facet_undefined` and `facet_not_available`, an entry also holds the `code` and `detail` of the axis failure it stands for. |
| `cause`              | object  | optional            | Why the refusal fired. On `recipe_not_declared`, `rate_coverage_gap` and `operation_failed`.                           |
| `constraint_id`      | string  | optional            | Machine-readable id of the violated constraint.                                                                        |
| `ceiling_kind`       | string  | optional            | For `over_ceiling`, and for `operation_failed` whose `cause.code` is `over_ceiling`, identifies the crossed ceiling, such as `rows`, `bytes`, `operation_rows`, `operation_requests`, `member_cardinality` or `candidate_window`. |
| `count`              | integer | optional            | For `over_ceiling`, and for `operation_failed` whose `cause.code` is `over_ceiling`, the observed count. |
| `cap`                | integer | optional            | For `over_ceiling`, and for `operation_failed` whose `cause.code` is `over_ceiling`, the ceiling crossed. |

`GET /codes/{code}` returns the registry entry for any code:

```json
{
  "code": "node_not_held",
  "classification": "refusal",
  "definition": "This deployment's graph does not hold the requested node.",
  "lifecycle": "active",
  "mappings": [{"endpoint": "graph.node", "channel": "problem_details", "doors": null, "http_status": 404}],
  "title": "Node Not Held",
  "detail": "This deployment's graph does not hold the requested node."
}
```

:::tabs

::tab{label="curl"}

```bash
curl -s "$DFO_API/codes/node_not_held"
```

::tab{label="Python"}

```python
import os

import httpx

DFO_API = os.environ["DFO_API"]

print(httpx.get(f"{DFO_API}/codes/node_not_held").text)
```

:::

### Codes every API uses

| Code                          | Status | Meaning                                                                                                    |
| ----------------------------- | ------ | ---------------------------------------------------------------------------------------------------------- |
| `request_invalid`             | 400    | The body is malformed: an unknown input kind, an unknown or extra field, or a value that fails validation. |
| `not_found`                   | 404    | The path matches zero routes.                                                                              |
| `code_not_found`              | 404    | This code is absent from the registry at `GET /codes/{code}`.                                              |
| `method_not_allowed`          | 405    | The request uses a method outside the set listed in the `Allow` header.                                    |
| `unsupported_schema_version`  | 422    | The request names a `schema_version` other than 1.                                                         |
| `facet_not_available`         | 422    | The facet binding is valid and matches zero observations.                                                  |
| `invalid_role_ref`            | 422    | The graph lacks the entity named by the role reference.                                                    |
| `role_incompatible`           | 422    | The resolved StatVar lacks the requested role.                                                             |
| `role_signature_incompatible` | 422    | The resolved StatVars carry incompatible role signatures.                                                  |
| `resolution_ambiguous`        | 422    | The definition matches multiple indistinguishable facet tuples.                                            |
| `recipe_not_declared`         | 422    | The requested unit falls outside the declared conversion recipes.                                          |
| `rate_coverage_gap`           | 422    | The conversion recipe's rates cover only part of the slice, or its deflators leave no row to convert. Rows a deflator lacks are otherwise left out, and the `200` manifest carries a `deflator_missing_omitted` warning. |
| `internal_error`              | 500    | An unclassified failure. Not retryable.                                                                    |
| `facet_binding_unsupported`   | 501    | This deployment recognises the facet axis and lacks support for it.                                        |
| `upstream_error`              | 502    | A call to an upstream data source failed.                                                                  |
| `service_unavailable`         | 503    | The service is at capacity. A retry is expected to succeed.                                                |
| `timeout`                     | 504    | The request exceeded its deadline. Search allows a `dataset_resubmit` that reads a whole StatVar family 90 seconds, and every other request 45. |

`facet_not_available`, `role_incompatible`, `role_signature_incompatible` and `rate_coverage_gap` also arrive inside a `200` Search response, as a part's `no_data.reason`. `facet_binding_unsupported` and `recipe_not_declared` arrive there as a part's `unsupported.code`, as does the Search code `scope_too_broad`.

### Search API codes

| Code                                 | Status                                         | Meaning                                                                                                       |
| ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `children_anchor_ambiguous`          | 400                                            | An options request with `scope: "children"` binds more than one anchor entity.                                |
| `children_anchor_required`           | 400                                            | An options request with `scope: "children"` has zero anchor entities to descend from.                         |
| `mixed_peers_anchor_types`           | 400                                            | An options request with `scope: "peers"` binds anchor entities of different types.                            |
| `provider_ambiguous`                 | 400                                            | An options request binds more than one provider. At most one is supported.                                    |
| `unsupported_children_anchor_type`   | 400                                            | The children-scope anchor's type lacks a children mapping.                                                    |
| `unsupported_peers_anchor_type`      | 400                                            | The peers-scope anchor's type lacks a peers mapping.                                                          |
| `query_too_long`                     | 422                                            | The `raw_text` query exceeds 2,000 characters.                                                                |
| `shape_mismatch`                     | 422                                            | A resubmit's shape does not match the five-tuple of its members.                                              |
| `option_target_invalid`              | 422                                            | The options request's `for` target names a role outside the resolved StatVar's declared roles.                |
| `directional_children_unsupported`   | 422                                            | Flow roles lack a containment hierarchy, which makes them incompatible with children scope.                   |
| `population_empty`                   | 422                                            | The shape resolves to zero StatVars.                                                                          |
| `facet_probe_ceiling`                | 422                                            | The bound facet combinations would exceed the probe cap.                                                      |
| `scope_too_broad`                    | 422 on `options`; 200, in a part, on `resolve` | The role scope, or the StatVar family of a resubmitted identity, exceeds the deployment's limits. Bind an entity, narrow the window, or bind a facet. |
| `door_not_implemented`               | 501                                            | The request uses a defined input kind outside this deployment's supported set.                                |
| `exact_binding_unsupported`          | 501                                            | `state: "exact"` is outside this deployment's accepted slot bindings.                                         |
| `no_member_asserts_bound_properties` | 200, in a part                                 | The request leaves place unbound, and zero StatVars match every bound property. The nearest match is offered. |
| `no_member_has_data_at_place`        | 200, in a part                                 | Zero StatVars in the family have data at the place. The nearest match is offered.                             |
| `only_alternate_has_data`            | 200, in a part                                 | The alternate reading is the sole reading with data at the place. It is offered as `no_data.alternate_match`. |
| `unit_request_unsupported`           | 200, in a part                                 | The unit phrase names a concept outside the request's vocabulary, such as PPP terms or per-capita.            |

The last four arrive in a `200` response. The first three are a `no_data` part's `gate`, and `unit_request_unsupported` is an `unsupported` part's `code`.

### Datasets API codes

| Code                          | Status | Meaning                                                                                           |
| ----------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `pin_not_permitted`           | 403    | The caller lacks permission to pin, or pinning is inactive on this deployment.                    |
| `publish_not_permitted`       | 403    | The caller lacks permission to publish, or publishing is inactive on this deployment.             |
| `spec_not_found`              | 404    | No definition is registered at this id and none was sent beside it.                               |
| `snapshot_not_found`          | 404    | The requested `content_version` is absent from the known id's snapshots.                          |
| `artifact_not_found`          | 404    | This version's artifact list omits the named `artifact_digest`.                                   |
| `operation_not_found`         | 404    | This operation id is unknown to the API.                                                          |
| `spec_not_exportable`         | 409    | The spec covers a region with zero enumerable series.                                             |
| `publish_conflict`            | 409    | A publish names an identity pair already held with different content.                             |
| `artifact_expired`            | 410    | Every artifact of this version aged out of storage unpinned.                                      |
| `snapshot_revoked`            | 410    | This snapshot is revoked.                                                                         |
| `spec_id_mismatch`            | 422    | The sent definition hashes to a value different from the adjacent `spec_id`.                      |
| `member_not_resolvable`       | 422    | A StatVar resolves to zero or multiple registered shapes.                                         |
| `definition_not_reproducible` | 422    | A registered definition fails to reproduce its id against the current graph.                      |
| `entities_unbound`            | 422    | Registered with no raise site. A request with no entity scope covers every entity with data.      |
| `default_facet_undefined`     | 422    | No probed facet survives narrowing by the bound axes, or the top-ranked facet carries no provenance. Places with no observations at all answer 200 with an `empty_result` warning. |
| `collection_probe_ceiling`    | 422    | The facet probe for a group of StatVars would exceed its cap.                                     |
| `over_ceiling`                | 422    | The request exceeds a size ceiling. `ceiling_kind`, `count` and `cap` say which. A StatVar walk past 20,000 subjects is `candidate_window`; one past 60 seconds answers `timeout` (504, retryable). |
| `excel_limit`                 | 422    | The table exceeds Excel's row, column or cell limits.                                             |
| `excel_illegal_character`     | 422    | A cell holds a character Excel's file format forbids.                                             |
| `parquet_size_bound`          | 422    | The Parquet file would exceed its size bound.                                                     |
| `column_unknown`              | 422    | `columns`, `pivot` or an enrichment names a column absent from the table.                         |
| `column_not_droppable`        | 422    | `columns` must retain each grain, unit and provenance column.                                     |
| `rollup_not_summable`         | 422    | The `groupby` roll-up fails summability.                                                          |
| `pivot_not_unique`            | 422    | The pivot produces duplicate cells. `issues` name the columns that vary within them.              |
| `pivot_not_homogeneous`       | 422    | The pivot would combine rows with different values in a dropped grain, unit or provenance column. |
| `pivot_over_column_cap`       | 422    | The pivot is wider than the format allows.                                                        |
| `pivot_over_cell_cap`         | 422    | The pivot exceeds 5,000,000 cells.                                                                |
| `identity_collision`          | 500    | The registry already holds different content at this id.                                          |
| `identity_quarantined`        | 500    | This id is quarantined after a collision.                                                         |
| `operation_failed`            | 500    | The background operation failed. The poll returns this body in place of a manifest, with the operation's own refusal code in `cause.code`. When the operation ended on a refusal, `detail`, `pointer`, `alternatives`, `issues`, `constraint_id`, `ceiling_kind`, `count`, `cap` and `cause.entities` are that refusal's, as `GET /api/datasets/{id}` returns them. `retryable` is the stored operation's flag: polling again returns the same failure, and resubmitting the mint resets a retryable operation. |
| `capability_unavailable`      | 501    | The requested format, reshape or conversion is disabled on this deployment.                       |
| `upstream_version_changed`    | 503    | A source refresh during the mint left the retry unsettled, or the graph changed a background operation's StatVars after submission. |

### Graph API codes

| Code            | Status | Meaning                                                    |
| --------------- | ------ | ---------------------------------------------------------- |
| `node_not_held` | 404    | The requested node is absent from this deployment's graph. |

## schema_version

`POST /api/search/resolve`, `POST /api/search/options` and `POST /api/datasets` take a `schema_version` in the request body. The only supported value is `1`, and it is the default when you leave the field out. Search and Datasets responses carry `"schema_version": 1` at the top level, as do the definitions inside them.

The API returns this problem body for any other value.

```json
{
  "type": "/codes/unsupported_schema_version",
  "title": "Unsupported Schema Version",
  "status": 422,
  "detail": "This deployment does not support the requested schema_version.",
  "code": "unsupported_schema_version",
  "pointer": "/schema_version",
  "trace_id": "8534ed88677e416cb6cd58c2985f4667",
  "capability_version": "cap_a71dd3a39765e8e5",
  "retryable": false
}
```

:::tabs

::tab{label="curl"}

```bash
curl -s -X POST "$DFO_API/api/search/resolve" \
  -H "Content-Type: application/json" \
  -d '{"schema_version": 2, "input": {"kind": "raw_text", "query": "ODA to Nigeria"}}'
```

::tab{label="Python"}

```python
import os

import httpx

DFO_API = os.environ["DFO_API"]

response = httpx.post(
    f"{DFO_API}/api/search/resolve",
    json={"schema_version": 2, "input": {"kind": "raw_text", "query": "ODA to Nigeria"}},
)
print(response.text)
```

:::

`GET /api/datasets/capabilities` lists the supported versions in `schema_versions`.

## Response headers

| Header                      | On                                                                        | Value                                                             |
| --------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `x-dfo-trace-id`            | every response except those sent with `Cache-Control: public`             | The 32-character lowercase hexadecimal trace id for this request. |
| `x-dfo-commit`              | every response                                                            | The build commit that answered.                                   |
| `Server-Timing`             | every response except event streams and `Cache-Control: public` responses | `total` first, then time per phase in milliseconds.               |
| `Strict-Transport-Security` | every response                                                            | `max-age=31536000; includeSubDomains`                             |

The server sets `x-dfo-trace-id`, and you don't send it. The same value is in every error body's `trace_id` and in a Search response's `diagnostics.trace_id`. Quote it when you report a problem.

```text
HTTP/2 422
cache-control: no-store
content-type: application/problem+json
x-dfo-trace-id: 8534ed88677e416cb6cd58c2985f4667
strict-transport-security: max-age=31536000; includeSubDomains
x-dfo-commit: a00b52ae7b072289fd6eabd3b182f6fcc424e583
server-timing: total;dur=1.5
…
```

The capabilities documents are served with `Cache-Control: public, max-age=300` and omit `x-dfo-trace-id`.

## Discovery routes

| Route                                                | Returns                                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------------------- |
| `GET /api/{search,datasets,graph}/healthz`           | `{status, build, capability_version}`                                     |
| `GET /api/{search,datasets,graph}/capabilities`      | Limits, live input kinds and formats, and policy identities for that API. |
| `GET /api/search/schema`, `GET /api/datasets/schema` | JSON Schema for the request and response bodies.                          |
| `GET /codes/{code}`                                  | The registry entry for one error or warning code.                         |
| `GET /openapi.json`                                  | The OpenAPI 3.1 document for every route. `GET /docs` renders it.         |

## Next

- [Get a dataset](get-a-dataset.md) shows `202`, `snapshot_not_found` and `operation_not_found` in a dataset flow.
- [Resolve a question](resolve-a-question.md) shows how a `200` Search response reports an unanswered question.
