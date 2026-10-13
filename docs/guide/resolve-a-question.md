---
description: Resolve a plain-English question into an interpretable DFO spec.
---

# How to resolve a question

Use `POST /api/v1/search/resolve` to turn a plain-English question into one or more candidate data specifications.

## Before you start

Set the development API URL.

```bash
export DFO_API=https://developmentfinance.org
```

## Steps

1. Send a raw-text question. `GET /api/v1/search/capabilities` reports the current query and variable limits.

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -fsS -X POST "$DFO_API/api/v1/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input":{"kind":"raw_text","query":"ODA disbursements from Germany to Nigeria for basic health care"}}' \
     > resolve.json

   jq 'if .kind == "resolution" then
         .parts[0] | {status, variable_text, spec_id: .spec.spec_id,
                      sentence: .spec.resolved.rendered_sentence, warnings}
       else
         {kind, reason, scope}
       end' resolve.json
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os
   from pathlib import Path

   import httpx

   response = httpx.post(
       f"{os.environ['DFO_API']}/api/v1/search/resolve",
       json={
           "input": {
               "kind": "raw_text",
               "query": "ODA disbursements from Germany to Nigeria for basic health care",
           }
       },
       timeout=60,
   )
   response.raise_for_status()
   document = response.json()
   Path("resolve.json").write_text(json.dumps(document))
   if document["kind"] != "resolution":
       print(json.dumps({key: document.get(key) for key in ("kind", "reason", "scope")}, indent=2))
       raise SystemExit
   part = document["parts"][0]
   summary = {
       "status": part["status"],
       "variable_text": part["variable_text"],
       "warnings": part["warnings"],
   }
   if part["status"] == "definite":
       spec = part["spec"]
       summary.update(
           {
               "spec_id": spec["spec_id"],
               "sentence": spec["resolved"]["rendered_sentence"],
           }
       )
   print(
       json.dumps(summary, indent=2)
   )
   ```

   :::

   Check `kind` before reading `parts`. A resolution has a `parts` entry for each extracted variable. Search reports `variables_clamped` when it drops variables beyond the current limit.

2. Branch on the answer's `kind`.

   | Kind | Result |
   | --- | --- |
   | `resolution` | Data answer with `parts`. |
   | `catalogue` | A request to describe what DFO holds, with an optional country `scope.place`. |
   | `declined` | A question outside the available data or one that requires analysis. |

   Read `GET /api/v1/search/catalogue` for the available data and example questions. Pass `place=<country id>`, the country's graph identifier such as `country/NGA`, to narrow it to one country.

   ```bash
   curl -fsS "$DFO_API/api/v1/search/catalogue?place=country/NGA" | jq .
   ```

3. Branch on each part's `status`.

   | Status | Meaning |
   | --- | --- |
   | `definite` | Search selected one reading. `spec` holds its definition and result summary. |
   | `candidates` | Search found more than one reading. `candidates.specs` is ordered by `candidates.ordering`. |
   | `no_data` | Search found no data for the reading. `no_data` explains why. |
   | `unsupported` | Search understood the question but this deployment cannot express it. `unsupported` gives the code. |

   These are successful request outcomes and arrive with HTTP `200`. Invalid requests use an application problem-details response.

4. Check a definite result before using it. Read `spec.resolved.rendered_sentence`, then inspect `warnings`. For this question, check that Germany fills the donor role (`origin`), Nigeria fills the recipient role (`destination`), and basic health care is `CRSPurposeCode12220`. Open dimensions can expand the result to several StatVars (measures, each with its dimensions).

5. Request observation previews when you need values in the response.

   ```bash
   curl -fsS -X POST "$DFO_API/api/v1/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{
       "input":{"kind":"raw_text","query":"ODA disbursements from Germany to Nigeria for basic health care"},
       "response_options":{"include_observations":true}
     }' \
     | jq 'if .kind == "resolution" then
              .parts[0].spec.resolved.observations
            else
              {kind, reason, scope}
            end'
   ```

   A preview is delivered only for definite results. It can include a structured omission reason when the result exceeds the deployed preview limit or cannot be retrieved in time. Each delivered series has its own StatVar, role entities, points, and facet. Use [Get a dataset](get-a-dataset.md) for a complete downloadable result.

6. Read `no_data.reason` when a part has no data. When `no_data.gate` is `only_alternate_has_data`, `no_data.alternate_match` provides another interpretation for review.

## Stream a response

Send `Accept: text/event-stream` to receive a Search response as Server-Sent Events. The stream contains a `response` event, one `observations` event for each definite part that requested observations, and a terminal `done` event with the trace identifier. An observations event can report an omission reason.

## Verify it worked

A usable result has a `definite` part whose confirmation sentence matches the requested measure, entities, date range, source, and unit. The `spec_id` is derived from the canonical spec definition. Keep the definition with the identifier when reproducibility matters.

## Troubleshooting

- `422 query_too_long` means `input.query` exceeds the current deployment limit. Split the request.
- Use [Look up names and codes](look-things-up.md) when a place name resolves incorrectly or remains ambiguous.

## Next

[How the APIs fit together](concepts.md) explains what a spec represents. [Get a dataset](get-a-dataset.md) turns a definite spec into files.
