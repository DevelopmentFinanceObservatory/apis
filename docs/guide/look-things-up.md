---
description: Turn a place or donor name into a dcid, and read what the graph holds about a country, a CRS code, a StatVar or a source, with the Graph API.
---

# Look things up

The Graph API resolves names to dcids and returns the graph's record for each dcid. Use `POST /api/graph/resolve` to resolve names, `GET /api/graph/node/{dcid}` to read a node, and `POST /api/graph/nodes` to read up to 50 nodes at once. Check a place before asking Search a question, or label the dcids in a dataset.

Every example sets the base URL first. See [Conventions](conventions.md) for access and limits.

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

## Steps

1. **Resolve names to dcids.** Send up to 25 names in `texts`, each up to 200 characters. Results come back in the same order.

   ```json
   {"status":"resolved","entity":{"dcid":"country/NGA","name":"Nigeria","entity_type":"Country","kind":"country"}}
   {"status":"ambiguous","candidates":[{"dcid":"country/COD","name":"Congo [DRC]","entity_type":"Country","kind":"country"},{"dcid":"country/COG","name":"Congo [Republic]","entity_type":"Country","kind":"country"}]}
   {"status":"resolved_region","region":{"dcid":"SubSaharanAfrica","name":"Sub-Saharan Africa","entity_type":"UNGeoRegion","kind":"subregion"},"members":49}
   {"status":"resolved_group","entity":null,"members":7}
   {"status":"not_found","near_matches":["Nigeria","Niger"]}
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/graph/resolve" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Nigeria", "Congo", "Sub-Saharan Africa", "G7", "Nigera"]}' \
     | jq -c '.results[] | if .members then .members |= length else . end'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   texts = ["Nigeria", "Congo", "Sub-Saharan Africa", "G7", "Nigera"]
   response = httpx.post(f"{DFO_API}/api/graph/resolve", json={"texts": texts}, timeout=60)
   for result in response.raise_for_status().json()["results"]:
       if result.get("members") is not None:
           result["members"] = len(result["members"])
       print(json.dumps(result, separators=(",", ":")))
   ```

   :::

   The `jq` filter replaces each `members` list with its length. The full response lists every member country with its dcid and name.

   | `status`          | What you get                                                                                                                                    |
   | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
   | `resolved`        | One entity, in `entity`.                                                                                                                        |
   | `resolved_region` | A continent or UN subregion in `region`, and its member countries in `members`.                                                                 |
   | `resolved_group`  | A membership group such as the G7, G20, DAC or EU, with its members in `members`. `entity` is set when the group is itself a node in the graph. |
   | `ambiguous`       | Two or more entities the name could mean, in `candidates`.                                                                                      |
   | `not_found`       | The resolver found zero matches. `near_matches` suggests spellings.                                                                             |
   | `not_held`        | A real entity missing from this deployment's graph.                                                                                             |

   Every status arrives with HTTP `200`. `kind` sorts the entity into one of eight classes: `country`, `continent`, `subregion`, `continental_union`, `organisation`, `aggregate`, `place_other` or `other`. `aggregate` marks a DAC "unallocated by country" bucket, a non-territorial reporting category.

1. **Look up a dcid.** `GET /api/graph/node/{dcid}` returns what the graph holds, in one of four shapes chosen by the node's type.

   ```json
   {"shape":"entity","dcid":"country/NGA","label":"Nigeria","entity_type":"Country","contained_in":[{"dcid":"SubSaharanAfrica","label":"Sub-Saharan Africa"},{"dcid":"WesternAfrica","label":"Western Africa"},{"dcid":"africa","label":"Africa"}]}
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s "$DFO_API/api/graph/node/country/NGA"
   ```

   ::tab{label="Python"}

   ```python
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   print(httpx.get(f"{DFO_API}/api/graph/node/country/NGA").text)
   ```

   :::

   | `shape`    | For                                  | Main fields                                                                                    |
   | ---------- | ------------------------------------ | ---------------------------------------------------------------------------------------------- |
   | `entity`   | Countries and other places           | `label`, `entity_type`, `contained_in`                                                         |
   | `stat_var` | Statistical variables                | `label`, `definition`, `description`, `roles`, `population_type`, `source_indicator_codes`, `facets`, `sources` |
   | `source`   | Data sources                         | `label`, `license`, `url`, `publisher`, `earliest_observation_date`, `latest_observation_date` |
   | `node`     | Everything else, including CRS codes | `label`, `types`, `properties`                                                                 |

1. **Look up a CRS code.** Nodes for the four CRS classifications carry the code in the dcid: purpose `12220` is `CRSPurposeCode12220`, finance type `110` is `CRSTypeOfFinance110`, flow `10` is `CRSTypeOfFlow10`, and modality `B01` is `CRSCooperationModalityB01`. `POST /api/graph/resolve` matches places, groups and organisations, and returns `not_found` for a code name such as "basic health care".

   ```json
   {"shape":"node","dcid":"CRSPurposeCode12220","label":"basic health care (CRS Purpose)","types":["CRSPurposeCodeEnum"],"properties":{"description":["Basic and primary health care programmes; paramedical and nursing care programmes; supply of drugs, medicines, and vaccines related to basic health care; and activities aimed at achieving universal health coverage."]}}
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s "$DFO_API/api/graph/node/CRSPurposeCode12220"
   ```

   ::tab{label="Python"}

   ```python
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   print(httpx.get(f"{DFO_API}/api/graph/node/CRSPurposeCode12220").text)
   ```

   :::

   Codes and classifications, on the DFO docs hub, lists the codes in each classification. <!-- TODO: link to the hub page "Codes and classifications" when it is public. -->

1. **Label the StatVars in a dataset.** The `stat_var` column of a minted dataset carries a `dcid:` prefix. The node route takes the bare dcid, so strip the prefix first. This prints `200` and then `404`:

   ```python
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]
   client = httpx.Client(base_url=DFO_API, timeout=60)
   stat_var = "dcid:Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityB01_CRSPurposeCode12220_Nominal"
   ```

   ```python title="Right"
   # Use the bare dcid
   print(client.get(f"/api/graph/node/{stat_var.removeprefix('dcid:')}").status_code)
   ```

   ```python title="Wrong"
   # The prefixed dcid has no matching route
   print(client.get(f"/api/graph/node/{stat_var}").status_code)
   ```

   A StatVar node looks like this:

   ```json
   {
     "shape": "stat_var",
     "dcid": "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityB01_CRSPurposeCode12220_Nominal",
     "label": "Official development assistance gross disbursements for basic health care (standard grant, core support to non-governmental organisations, other private bodies, public-private partnerships, and research institutes)",
     "definition": null,
     "description": null,
     "roles": ["financingProvider", "financingRecipient"],
     "population_type": "FinancialFlow",
     "source_indicator_codes": [],
     "facets": null,
     "processing_level": null,
     "sources": null
   }
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s "$DFO_API/api/graph/node/Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityB01_CRSPurposeCode12220_Nominal" | jq .
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   dcid = "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityB01_CRSPurposeCode12220_Nominal"
   print(json.dumps(httpx.get(f"{DFO_API}/api/graph/node/{dcid}").json(), indent=2))
   ```

   :::

   `facets` and `sources` are `null` for this StatVar. The manifest's `trace.facets` from a mint names the source and unit a dataset used.

1. **Read many nodes at once.** `POST /api/graph/nodes` takes up to 50 dcids in `dcids` and reads them together. Each entry in `nodes` has the shape `GET /api/graph/node/{dcid}` returns, in request order. A dcid the graph lacks comes back as `{"shape": "not_held", "dcid": ...}`, where the single-node route answers `404`.

   ```json
   {"shape":"entity","dcid":"country/NGA","label":"Nigeria"}
   {"shape":"node","dcid":"CRSPurposeCode12220","label":"basic health care (CRS Purpose)"}
   {"shape":"not_held","dcid":"country/XYZ","label":null}
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/graph/nodes" \
     -H "Content-Type: application/json" \
     -d '{"dcids": ["country/NGA", "CRSPurposeCode12220", "country/XYZ"]}' \
     | jq -c '.nodes[] | {shape, dcid, label}'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   dcids = ["country/NGA", "CRSPurposeCode12220", "country/XYZ"]
   response = httpx.post(f"{DFO_API}/api/graph/nodes", json={"dcids": dcids}, timeout=60)
   for node in response.raise_for_status().json()["nodes"]:
       print(json.dumps({"shape": node["shape"], "dcid": node["dcid"], "label": node.get("label")}, separators=(",", ":")))
   ```

   :::

   The node routes share a pool of four graph reads, so a fifth parallel `GET /api/graph/node/{dcid}` call is refused `503`. One batch takes one place in that pool.

1. **Read about a source.** `GET /api/graph/sources` lists the sources this deployment documents, and `GET /api/graph/documentation/{dcid}` returns one source's documentation:

   ```json
   {
     "details": {
       "name_of_source": "OECD Creditor Reporting System (CRS)",
       "publisher": "Organisation for Economic Co-operation and Development (OECD)"
     },
     "license": {
       "type": "OECD Terms and Conditions",
       "url": "https://www.oecd.org/en/about/terms-conditions.html",
       "update_cadence": "Quarterly",
       "last_update": "2026-09-11",
       "next_expected_update": null
     },
     "scope_and_units": {
       "years_covered": "1973-2024"
     }
   }
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s "$DFO_API/api/graph/documentation/provenance/OECDCreditorReportingSystem" \
     | jq '{details: {name_of_source: .details.name_of_source, publisher: .details.publisher},
            license, scope_and_units: {years_covered: .scope_and_units.years_covered}}'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   doc = httpx.get(f"{DFO_API}/api/graph/documentation/provenance/OECDCreditorReportingSystem").raise_for_status().json()
   summary = {
       "details": {"name_of_source": doc["details"]["name_of_source"], "publisher": doc["details"]["publisher"]},
       "license": doc["license"],
       "scope_and_units": {"years_covered": doc["scope_and_units"]["years_covered"]},
   }
   print(json.dumps(summary, indent=2))
   ```

   :::

   The response's `gaps` list names every field the documentation leaves empty.

The same lookups in Python return these results.

```text
Nigeria -> country/NGA
Congo -> ambiguous: ['country/COD', 'country/COG']
Sahel -> not_found
node basic health care (CRS Purpose)
```

```python
import os

import httpx

DFO_API = os.environ["DFO_API"]
client = httpx.Client(base_url=DFO_API, timeout=60)

names = ["Nigeria", "Congo", "Sahel"]
results = client.post("/api/graph/resolve", json={"texts": names}).raise_for_status().json()["results"]
for name, result in zip(names, results):
    if result["status"] == "resolved":
        print(name, "->", result["entity"]["dcid"])
    elif result["status"] == "ambiguous":
        print(name, "-> ambiguous:", [c["dcid"] for c in result["candidates"]])
    else:
        print(name, "->", result["status"])

node = client.get("/api/graph/node/CRSPurposeCode12220").raise_for_status().json()
print(node["shape"], node["label"])
```

## Verify it worked

A `resolved` result carries a `dcid`, and `GET /api/graph/node/{dcid}` for that dcid answers `200` with a `label`.

## Troubleshooting

- A `404 node_not_held` response means the dcid is well formed and this deployment's graph lacks a node for it. Check the spelling or resolve the name first.
- A `404 not_found` response on the node route means the path did not match. The usual cause is a `dcid:` prefix.
- A `400 request_invalid` response with `pointer: "/texts"` means the request needs a `texts` list of 25 names or fewer. Split a longer list into batches of 25.
- A `400 request_invalid` response with `pointer: "/dcids"` means the request needs a `dcids` list of 50 dcids or fewer. Split a longer list into batches of 50.
- A `503 service_unavailable` response on a node route means every graph read slot is busy. Send the dcids as one `POST /api/graph/nodes` batch, or retry.

## See also

- [Resolve a question](resolve-a-question.md) uses these dcids in a question to Search.
- The knowledge graph, on the DFO docs hub, explains entities, StatVars and dcids. <!-- TODO: link to the hub page "The knowledge graph" when it is public. -->
