---
description: Send a plain-English question to the Search API and read back the spec, the confirmation sentence, the observations, or the reason nothing matched.
---

# Resolve a question

`POST /api/search/resolve` turns a question such as "German aid disbursements to Nigeria for basic health care" into a spec that records its StatVars, entities and codes under a `spec_id`. Read values from the response or pass the spec to the Datasets API to mint a file.

Every example sets the base URL first. See [Conventions](conventions.md) for access and limits.

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

## Steps

1. **Send the question.** Put it in `input.query` with `kind: "raw_text"`. Queries run up to 2,000 characters.

   ```json
   {
     "status": "definite",
     "variable_text": "aid disbursements for basic health care",
     "spec_id": "spec_774d6c001a6250aa67ca676ab067895e",
     "entities": {
       "destination": ["country/NGA"],
       "origin": ["country/DEU"]
     },
     "purpose": "CRSPurposeCode12220",
     "members": 21,
     "confidence": "high",
     "warnings": []
   }
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input": {"kind": "raw_text", "query": "German aid disbursements to Nigeria for basic health care"}}' \
     | jq '.parts[] | {status, variable_text,
                       spec_id: .spec.spec_id,
                       entities: .spec.definition.identity.entities,
                       purpose: (.spec.definition.identity.slots[] | select(.property == "crsPurpose") | .binding.ref),
                       members: .spec.resolved.shape.member_count,
                       confidence: .spec.resolved.confidence,
                       warnings}'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   response = httpx.post(
       f"{DFO_API}/api/search/resolve",
       json={"input": {"kind": "raw_text", "query": "German aid disbursements to Nigeria for basic health care"}},
       timeout=60,
   )
   for part in response.raise_for_status().json()["parts"]:
       spec = part["spec"]
       identity = spec["definition"]["identity"]
       summary = {
           "status": part["status"],
           "variable_text": part["variable_text"],
           "spec_id": spec["spec_id"],
           "entities": identity["entities"],
           "purpose": next(s["binding"]["ref"] for s in identity["slots"] if s["property"] == "crsPurpose"),
           "members": spec["resolved"]["shape"]["member_count"],
           "confidence": spec["resolved"]["confidence"],
           "warnings": part["warnings"],
       }
       print(json.dumps(summary, indent=2))
   ```

   :::

   The answer is a `resolution`, with one entry in `parts` per variable the question names, up to six. This question names one. Germany took the `origin` role, Nigeria the `destination` role, and "basic health care" became CRS purpose code 12220. `members` counts the StatVars the spec covers: 21, because the question left codes such as cooperation modality open.

1. **Check the answer's `kind`.** Every answer arrives with HTTP `200` as one of three kinds, and only a `resolution` carries `parts`:

   | `kind`       | What the answer carries                                                                                                                                                                          |
   | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | `resolution` | `status` and `parts`, one part per variable. The rest of this guide reads these.                                                                                                                 |
   | `catalogue`  | `scope`, with `scope.place` when the question named one country. The question asked what the DFO holds: `GET /api/search/catalogue` returns the counts and example questions, and `?place=` with that country's dcid narrows them to it. |
   | `declined`   | `reason` and `scope`. `out_of_scope` means the question is not about data the DFO holds. `requires_analysis` means it asks for a judgement and names no data to show.                                |

   A weather question is declined:

   ```json
   {"kind":"declined","reason":"out_of_scope","scope":{}}
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input": {"kind": "raw_text", "query": "Will it rain in Accra tomorrow?"}}' \
     | jq -c 'if .kind == "resolution" then .parts[] | {status, variable_text} else {kind, reason, scope} end'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   response = httpx.post(
       f"{DFO_API}/api/search/resolve",
       json={"input": {"kind": "raw_text", "query": "Will it rain in Accra tomorrow?"}},
       timeout=60,
   )
   body = response.raise_for_status().json()
   if body["kind"] == "resolution":
       for part in body["parts"]:
           print(part["status"], part["variable_text"])
   else:
       # A catalogue or declined answer carries no parts.
       summary = {"kind": body["kind"], "reason": body.get("reason"), "scope": body["scope"]}
       print(json.dumps(summary, separators=(",", ":")))
   ```

   :::

   `scope` is empty because Accra is a city, and `scope.place` names only a country. The body carries no message, so write your own copy for a declined answer.

1. **Branch on each part's `status`.** Four values are possible:

   | `status`      | What the part carries                                                                                              |
   | ------------- | ------------------------------------------------------------------------------------------------------------------ |
   | `definite`    | One reading, in `spec`.                                                                                            |
   | `candidates`  | Two to six competing readings, in `candidates.specs`, best first.                                                  |
   | `no_data`     | Nothing to return, with the reason in `no_data`.                                                                   |
   | `unsupported` | The question was understood but asks for something this deployment cannot express, with the code in `unsupported`. |

   All four arrive with HTTP `200`. A problem-details body means the request itself failed, for example `query_too_long`.

1. **Check the reading.** `spec.resolved.rendered_sentence` states the reading in words. Show it to whoever asked, since it is the quickest way to catch a wrong reading. Then read `warnings`. A broader question shows why:

   ```json
   [{"code":"retrieval_score_weak","severity":"info","message":"the chosen reading's match to the query is weak; treat it with less confidence than a strong match."},{"code":"role_scope_defaulted","severity":"warn","message":"no provider was named; the answer was resolved across every provider with data."}]
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input": {"kind": "raw_text", "query": "CRS disbursements to Nigeria for health"}}' \
     | jq -c '.parts[0].warnings'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   response = httpx.post(
       f"{DFO_API}/api/search/resolve",
       json={"input": {"kind": "raw_text", "query": "CRS disbursements to Nigeria for health"}},
       timeout=60,
   )
   print(json.dumps(response.raise_for_status().json()["parts"][0]["warnings"], separators=(",", ":")))
   ```

   :::

   "CRS disbursements to Nigeria for health" leaves the donor open, so the spec covers every donor with data. It includes 505 StatVars. `role_scope_defaulted` reports the default. Name a donor to narrow the question.

1. **Get the values.** Set `response_options.include_observations` to `true` to include observations in `spec.resolved.observations`. Each definite spec then carries the full series when it is small and the latest points otherwise.

   ```text
   definite aid disbursements for basic health care
   spec_774d6c001a6250aa67ca676ab067895e
   Germany's gross ODA disbursements to Nigeria for basic health care, 1995 to 2024. Current US dollars · OECD Creditor Reporting System (CRS).
   Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode12220_Nominal {'date': '2009', 'value': 224204.0}
   Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityD02_CRSPurposeCode12220_Nominal {'date': '2024', 'value': 53798.0}
   ```

   ```python
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   response = httpx.post(
       f"{DFO_API}/api/search/resolve",
       json={
           "input": {"kind": "raw_text", "query": "German aid disbursements to Nigeria for basic health care"},
           "response_options": {"include_observations": True},
       },
       timeout=60,
   )
   response.raise_for_status()
   for part in response.json()["parts"]:
       print(part["status"], part["variable_text"])
       if part["status"] == "definite":
           resolved = part["spec"]["resolved"]
           print(part["spec"]["spec_id"])
           print(resolved["rendered_sentence"])
           for series in resolved["observations"]["series"][:2]:
               print(series["stat_var"]["ref"], series["points"][-1])
   ```

   Each series names its StatVar, its entities with their roles, its points, and the source and unit it came from in `facet`. Observations add latency, so leave the option off when you only need the spec. For a file with every row, mint the spec as shown in [Get a dataset](get-a-dataset.md).

1. **Handle an unmatched question.** A `no_data` part gives a `reason` and an `alternate_match` spec when the DFO finds a nearby reading with data.

   ```json
   {
     "status": "no_data",
     "variable_text": "debt",
     "reason": "no_observations",
     "gate": "only_alternate_has_data",
     "alternate": {
       "spec_id": "spec_d69cfc3a7d50fdfda6cc730efcc63cb3",
       "label": "External debt stocks, public and publicly guaranteed (PPG), by creditor"
     }
   }
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input": {"kind": "raw_text", "query": "Nigeria debt"}}' \
     | jq '.parts[] | {status, variable_text, reason: .no_data.reason, gate: .no_data.gate,
                       alternate: {spec_id: .no_data.alternate_match.spec_id,
                                   label: .no_data.alternate_match.resolved.shape.label}}'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   response = httpx.post(
       f"{DFO_API}/api/search/resolve",
       json={"input": {"kind": "raw_text", "query": "Nigeria debt"}},
       timeout=60,
   )
   for part in response.raise_for_status().json()["parts"]:
       no_data = part["no_data"]
       alternate = no_data["alternate_match"]
       summary = {
           "status": part["status"],
           "variable_text": part["variable_text"],
           "reason": no_data["reason"],
           "gate": no_data["gate"],
           "alternate": {"spec_id": alternate["spec_id"], "label": alternate["resolved"]["shape"]["label"]},
       }
       print(json.dumps(summary, indent=2))
   ```

   :::

   Show the alternate to the person who asked first. It answers a different question.

!!! tip
    Send `Accept: text/event-stream` to receive the response as Server-Sent Events. The full response arrives as one `response` event and the stream ends with a `done` event carrying the `trace_id`. With `include_observations`, the observations follow the `response` event as their own `observations` event.

## Verify it worked

You have a `definite` part with a `spec_id` starting `spec_` and a `rendered_sentence` that says what you meant. A `spec_id` is a hash of the reading, so the same reading always carries the same id and you can store it.

## Troubleshooting

- A `422 query_too_long` response means the query exceeds 2,000 characters. Split it.
- When a place resolves to the wrong entity or an unresolved result, check its name with `POST /api/graph/resolve`, as shown in [Look things up](look-things-up.md). "Congo", for example, matches two countries.

## See also

- [Get a dataset](get-a-dataset.md) mints the spec into CSV, Excel or Parquet.
- The knowledge graph, on the DFO docs hub, explains StatVars, entities and dcids. <!-- TODO: link to the hub page "The knowledge graph" when it is public. -->
