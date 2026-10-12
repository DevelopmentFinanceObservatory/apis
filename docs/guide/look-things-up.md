---
description: Resolve names, inspect graph nodes, and read source documentation.
---

# How to look up names and codes

Use the Graph API to resolve names before placing identifiers in a query or dataset request.

## Before you start

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

## Resolve names

Resolve one or more names with `POST /api/graph/resolve`.

   ```bash
   curl -fsS -X POST "$DFO_API/api/graph/resolve" \
     -H "Content-Type: application/json" \
     -d '{"texts":["Nigeria","Congo","G20","Sahel"]}' \
     | jq '.results | map(
         if .status == "resolved" then
           {status, entity: {dcid: .entity.dcid, name: .entity.name}}
         elif .status == "ambiguous" then
           {status, candidates: [.candidates[] | {dcid, name}]}
         else
           {status}
         end
       )'
   ```

   The command prints selected fields. Results are aligned with the submitted `texts`. `GET /api/graph/capabilities` reports the current request limits.

## Read resolver results

   | Status | Result |
   | --- | --- |
   | `resolved` | One entity in `entity`. |
   | `ambiguous` | Two or more possible entities in `candidates`. |
   | `resolved_region` | A region and its member countries. |
   | `resolved_group` | A named group and its members. |
   | `not_found` | No accepted match, with optional `near_matches`. |
   | `not_held` | A resolved entity that this deployment does not hold. |

   These results arrive with HTTP `200`. A malformed request returns an application problem-details response.

## Inspect a node

   ```bash
   curl -fsS "$DFO_API/api/graph/node/CRSPurposeCode12220" \
     | jq '{shape, dcid, label, definition, roles, population_type}'
   ```

   The command prints selected fields. The route returns a StatVar, source, entity, or generic node shape. A StatVar's label and definition describe the measure. Its `roles` identify the entity roles carried by its observations.

   Use the bare identifier from a dataset or resolver result. `dcid:CRSPurposeCode12220` does not match this route.

## Inspect several nodes

Use `POST /api/graph/nodes` to read several node dcids in one request. `nodes` is aligned with the submitted `dcids`, and a missing node becomes a `not_held` entry.

```bash
curl -fsS -X POST "$DFO_API/api/graph/nodes" \
  -H "Content-Type: application/json" \
  -d '{"dcids":["country/NGA","CRSPurposeCode12220"]}' \
  | jq '.nodes | map({shape, dcid, label})'
```

## Inspect a source

   ```bash
   curl -fsS "$DFO_API/api/graph/sources" \
     | jq '.sources'

   curl -fsS "$DFO_API/api/graph/documentation/provenance/OECDCreditorReportingSystem" \
     | jq '{
       details: {name_of_source: .details.name_of_source, publisher: .details.publisher},
       license,
       scope_and_units,
       gaps
     }'
   ```

   These commands print selected fields. The documentation record includes source details, licence, scope and units, methodology, data notes, transformation history, and `gaps`. `gaps` names fields for which the source has no documented value.

## Verify it worked

A `resolved` result has an `entity.dcid`. `GET /api/graph/node/{dcid}` should return `200` with a readable label for a held node.

## Troubleshooting

- `404 node_not_held` means the identifier is well formed but this deployment has no node for it.
- `404 not_found` on the node route usually means that the path did not match. Remove any `dcid:` prefix.
- `400 request_invalid` with `/texts` in the pointer means the request contains an invalid name or exceeds a current request limit. Check `GET /api/graph/capabilities`.

## Next

[How to resolve a question](resolve-a-question.md) uses entities and codes in Search. [How the APIs fit together](concepts.md) explains StatVars and facets.
