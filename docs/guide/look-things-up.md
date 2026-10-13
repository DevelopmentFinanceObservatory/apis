---
description: Resolve names, inspect graph nodes, list what belongs to them, and read source documentation.
---

# How to look up names and codes

Use the Graph API to resolve names before placing identifiers in a query or dataset request.

## Before you start

```bash
export DFO_API=https://developmentfinance.org
```

## Resolve names

Resolve one or more names with `POST /api/v1/graph/resolve`.

```bash
curl -fsS -X POST "$DFO_API/api/v1/graph/resolve" \
  -H "Content-Type: application/json" \
  -d '{"texts":["Nigeria","Congo","G20","Sahel"]}' \
  | jq '.results | map(
      if .status == "resolved" then
        {status, entity, member_count}
      elif .status == "ambiguous" then
        {status, candidates: [.candidates[] | {ref, label, kind}]}
      else
        {status, entity, near_matches}
      end
    )'
```

Results are aligned with the submitted `texts`. `GET /api/v1/graph/capabilities` reports the current request limits.

A request names the kinds it admits in `kinds`. Without it the resolver admits places, aggregates, organisations, and groups. Add `code` to find a code by its label, and `code_system` (such as `CRSPurposeCodeEnum`) to search one system's codes.

## Read resolver results

| Status      | Result                                                                                                          |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| `resolved`  | One entity in `entity`, as `{ref, label, kind}`, with the text it `matched`.                                    |
| `ambiguous` | Two or more possible entities in `candidates`.                                                                  |
| `not_found` | No accepted match, with optional `near_matches` as hints for a person.                                          |
| `not_held`  | A real entity that this deployment's graph holds no node for. Its `entity.ref` does not open on the node route. |

These results arrive with HTTP `200`. A malformed request returns an application problem-details response.

A group or a region resolves like any entity. Its `entity.kind` is `group`, `continent`, or `subregion`, `projection` names the cut of members counted (`countries` by default), `member_count` gives the count, and `members` is the path that lists them. Send `"expand": ["members"]` to receive up to 256 members inline as `{ref, label}` instead of the path. `reading` is `members` when the text names a group's members, such as "EU countries".

## Inspect a node

```bash
curl -fsS "$DFO_API/api/v1/graph/node/CRSPurposeCode12220" \
  | jq '{kind, ref, label, links, code}'
```

Every node has the same envelope (`ref`, `label`, `kind`, `types`, `description`, `links`, `gaps`, `built_from`) and one detail block named for its `kind`, such as `stat_var`, `code`, `country`, `group`, or `provenance`. A `stat_var` block holds the variable's `definition` and its `availability`, which says where DFO holds observations of it. `links` names the lists the node opens on the members route. `gaps` names each value the API cannot give, and the value is then `null`.

Use the bare identifier from a dataset or resolver result. `dcid:CRSPurposeCode12220` does not match this route. A dcid is the identifier of a graph node, such as `CRSPurposeCode12220` or `country/NGA`.

## Inspect several nodes

Use `POST /api/v1/graph/nodes` to read several node dcids in one request. `nodes` is aligned with the submitted `dcids`, and a missing node becomes `{ref, status: "not_held"}`.

```bash
curl -fsS -X POST "$DFO_API/api/v1/graph/nodes" \
  -H "Content-Type: application/json" \
  -d '{"dcids":["country/NGA","CRSPurposeCode12220"]}' \
  | jq '.nodes | map({kind, ref, label, status})'
```

## List what belongs to a node

A node's `links` names its lists. `GET /api/v1/graph/node/{dcid}/members` serves them by `relation`: `codes` for a code system, `children` for a code, `datasets` for a source, `variables` for a Dataset or Provenance, `members` for a group, `countries` for a region, and `facets` for a variable.

```bash
curl -fsS "$DFO_API/api/v1/graph/node/CRSPurposeCodeEnum/members?relation=codes&limit=5" \
  | jq '{total, complete, next_cursor, items: [.items[] | {code, label}]}'
```

A page holds 200 items unless `limit` asks for up to 1000. When `next_cursor` is not `null`, send it back as `cursor` with the same other parameters to read the next page. A cursor from a different node, relation, projection, or `as_of` date is refused.

A group or region takes a `projection`. `countries` is the default and counts the countries that belong to the United Nations. `other_states`, `territories`, `former_states`, and `institutions` cut the rest. A group that dates its membership takes `as_of`. A projection that the membership source records only in part carries the gap `membership_incomplete` and `complete: false`.

```bash
curl -fsS "$DFO_API/api/v1/graph/node/africa/members?relation=countries&limit=3" \
  | jq '{total, items: [.items[] | {ref, label}]}'
```

`GET /api/v1/graph/roots` lists the code systems, groups, regions, and sources to start from.

## Inspect a source

```bash
curl -fsS "$DFO_API/api/v1/graph/sources" \
  | jq '.sources'

curl -fsS "$DFO_API/api/v1/graph/documentation/provenance/OECDCreditorReportingSystem" \
  | jq '{
    details: {name_of_source: .details.name_of_source, publisher: .details.publisher},
    license,
    scope_and_units,
    gaps
  }'
```

The documentation record includes source details, licence, scope and units, methodology, data notes, transformation history, and `gaps`. `gaps` names fields for which the source has no documented value.

## Verify it worked

A `resolved` result has an `entity.ref`. `GET /api/v1/graph/node/{ref}` should return `200` with a readable label for a held node. A `resolved` group or region has a `members` path that returns `200` on the members route.

## Troubleshooting

- `404 node_not_held` means the identifier is well formed but this deployment has no node for it. The members route answers it too, for a dcid the deployment does not hold.
- `404 not_found` on the node route usually means that the path did not match. Remove any `dcid:` prefix.
- `400 request_invalid` with `/texts` in the pointer means the request contains an invalid name or exceeds a current request limit. Check `GET /api/v1/graph/capabilities`.
- `400 request_invalid` on the members route names the fault in its pointer: `/relation` for a list a held node lacks, `/cursor` for a cursor that belongs to another list, `/as_of` for a date asked of a membership that records none.
- `503 service_unavailable` is retryable. Every graph read worker was busy, or the service was still starting.

## Next

[How to resolve a question](resolve-a-question.md) uses entities and codes in Search. [How the APIs fit together](concepts.md) explains StatVars and facets.
