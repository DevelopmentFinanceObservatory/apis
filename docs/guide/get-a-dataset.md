---
description: Mint a spec into CSV, Excel or Parquet with the Datasets API, poll requests that run in the background, fetch a snapshot again, and see how the DFO picks a source for each series.
---

# Get a dataset

`POST /api/datasets` turns a spec into a stored CSV, Excel or Parquet table. By default, the table comes in a bundle with a README, the sources and their licences, and a citation. The manifest names the result by `spec_id` and `content_version`, and gives each file a signed download URL.

Use a spec from the Search API with its `spec_id` and `definition`. [Resolve a question](resolve-a-question.md) shows how to get one. Only a `resolution` answer carries specs, in `parts`; a `catalogue` or `declined` answer has none, so check `kind` first. Every example sets the base URL first.

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

## Steps

1. **Send the spec.** Put the `spec_id` and its `definition` in `input`, and choose a file in `options`.

   | Field              | Type    | Default | Description                                                                                                                      |
   | ------------------ | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
   | `schema_version`   | integer | `1`     | `1` is the only contract version.                                                                                                |
   | `input.kind`       | string  |         | `"spec_id"` for a spec from Search.                                                                                              |
   | `input.spec_id`    | string  |         | The `spec_…` id.                                                                                                                 |
   | `input.definition` | object  |         | The spec's definition, exactly as Search returned it. Required the first time the DFO sees the spec.                             |
   | `options.format`   | string  | `"csv"` | `"csv"`, `"xlsx"` or `"parquet"`.                                                                                                |
   | `options.bundle`   | boolean | `true`  | `true` wraps the table in a zip with its documentation, described in [The bundle](#the-bundle). `false` returns the table alone. |

   A table without a bundle comes in the long layout, with a name beside each code and the unit split into its parts. A program that parses the frame itself can send `"options": {"format": "parquet", "bundle": false, "layout": "frame"}` instead. That file holds the frame's own columns: `stat_var`, each role, each decoded code system as `stat_var_<code_system>` holding its label, `time_period`, `value`, `unit` (a converted unit as JSON text such as `{"currency":"EUR","prices":"current"}`), `measurementMethod`, `provenance` and `status`. `layout` with a bundle, a pivot or any format other than Parquet refuses `400 request_invalid` at `/options/layout`.

   A small request returns `200` with the manifest. This example has 30 rows.

   ```json
   {
     "content_version": "cv_e7f6ed9ddc5ca69902f92a8866f93ef7",
     "rows": 30,
     "artifacts": [
       {
         "rendering": {
           "format": "parquet",
           "bundle": true,
           "rendition": "v2-…"
         },
         "media_type": "application/zip",
         "bytes": 11590
       },
       {
         "rendering": { "statistics": "series" },
         "media_type": "application/vnd.apache.parquet",
         "bytes": 5978
       }
     ]
   }
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s -X POST "$DFO_API/api/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input": {"kind": "raw_text", "query": "German aid disbursements to Nigeria for basic health care"}}' \
     > resolve.json

   jq '{schema_version: 1,
        input: {kind: "spec_id", spec_id: .parts[0].spec.spec_id, definition: .parts[0].spec.definition},
        options: {format: "parquet"}}' resolve.json > mint.json

   curl -s -X POST "$DFO_API/api/datasets" \
     -H "Content-Type: application/json" \
     -d @mint.json > manifest.json

   jq '{content_version, rows: .coverage.rows, artifacts: [.artifacts[] | {rendering, media_type, bytes}]}' manifest.json
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]
   client = httpx.Client(base_url=DFO_API, timeout=120)

   question = "German aid disbursements to Nigeria for basic health care"
   resolved = client.post("/api/search/resolve", json={"input": {"kind": "raw_text", "query": question}})
   answer = resolved.raise_for_status().json()
   if answer["kind"] != "resolution":  # a catalogue or declined answer carries no parts
       raise SystemExit(f"not a data question: {answer['kind']}")
   spec = answer["parts"][0]["spec"]

   manifest = client.post(
       "/api/datasets",
       json={
           "schema_version": 1,
           "input": {"kind": "spec_id", "spec_id": spec["spec_id"], "definition": spec["definition"]},
           "options": {"format": "parquet"},
       },
   ).raise_for_status().json()

   summary = {
       "content_version": manifest["content_version"],
       "rows": manifest["coverage"]["rows"],
       "artifacts": [{k: a[k] for k in ("rendering", "media_type", "bytes")} for a in manifest["artifacts"]],
   }
   print(json.dumps(summary, indent=2))
   ```

   :::

1. **Poll if the answer is `202`.** A larger request runs in the background. The response is `202 Accepted` with a `Location` header and an operation to poll. Poll `operation.poll` until `status` is `done`, and the finished poll carries the manifest.

   A request for every donor's health disbursements to Nigeria returned 2,177 rows through background polling.

   ```text
   202 /api/datasets/operations/op_35021afdceb6fffb5d1ed1ab5e280e4722f8d7f295dbd54f5dd53671f22c8591
   {"operation": {"operation_id": "op_35021afd…", "status": "done", "poll": "/api/datasets/operations/op_35021afd…", "retry_after_ms": 1500}}
   cv_b785befb9f87471de96eb31cd7c56f5c {'rows': 2177, 'dimensions': [{'label': 'financingProvider', 'count': 63}, {'label': 'financingRecipient', 'count': 1}, {'label': 'years', 'count': 30}], 'window': {'kind': 'all', 'origin': 'default', 'end_year': 2024, 'start_year': 1995}}
   …
   (2177, 24)
   ['stat_var', 'financingProvider', 'financingProvider_label', 'financingRecipient', 'financingRecipient_label', 'crsCooperationModality', 'crsCooperationModality_label', 'crsPurpose', 'crsPurpose_label', 'crsTypeOfFinance', 'crsTypeOfFinance_label', 'crsTypeOfFlow', 'crsTypeOfFlow_label', 'time_period', 'value', 'unit', 'unit_label', 'currency', 'price_basis', 'base_year', 'measurementMethod', 'measurementMethod_label', 'provenance', 'provenance_label']
   ```

   ```python
   import io
   import json
   import os
   import time

   import httpx
   import pandas as pd

   DFO_API = os.environ["DFO_API"]
   client = httpx.Client(base_url=DFO_API, timeout=120)

   question = "CRS disbursements to Nigeria for health"
   resolved = client.post("/api/search/resolve", json={"input": {"kind": "raw_text", "query": question}})
   answer = resolved.raise_for_status().json()
   if answer["kind"] != "resolution":  # a catalogue or declined answer carries no parts
       raise SystemExit(f"not a data question: {answer['kind']}")
   spec = answer["parts"][0]["spec"]

   response = client.post(
       "/api/datasets",
       json={
           "schema_version": 1,
           "input": {"kind": "spec_id", "spec_id": spec["spec_id"], "definition": spec["definition"]},
           "options": {"format": "csv", "bundle": False},
       },
   ).raise_for_status()
   body = response.json()
   print(response.status_code, response.headers.get("location"))
   print(json.dumps(body)[:600])

   if response.status_code == 202:
       poll = body["operation"]["poll"]
       body = client.get(poll).raise_for_status().json()
       while body["operation"]["status"] != "done":
           time.sleep(body["operation"].get("retry_after_ms", 2000) / 1000)
           body = client.get(poll).raise_for_status().json()
           print(json.dumps(body["operation"]))
       body = body["manifest"]
   manifest = body

   print(manifest["content_version"], manifest["coverage"])
   print(json.dumps([{k: a[k] for k in ("rendering", "media_type", "bytes", "tier", "url_expires_at")} for a in manifest["artifacts"]], indent=1))
   csv = next(a for a in manifest["artifacts"] if a["rendering"].get("bundle") is False and a["rendering"]["format"] == "csv")
   df = pd.read_csv(io.BytesIO(httpx.get(csv["url"]).raise_for_status().content))
   print(df.shape)
   print(df.columns.tolist())
   ```

   A fresh request moves from `pending` to `running` to `done`. This one matches an operation that already finished, so its first response is already `done`. The first response includes `retry_after_ms`. Use that interval for later polls. A failed operation returns a `500 operation_failed` problem body when polled. `raise_for_status()` raises an exception for that response.

1. **Pick the file by its `rendering`.** A manifest lists every file stored for that `content_version`, including files other callers asked for. The Nigeria health manifest has four artifacts.

   ```json
   {"rendering":{"bundle":true,"format":"parquet","rendition":"v2-…"},"media_type":"application/zip","bytes":31807}
   {"rendering":{"bundle":true,"format":"csv","rendition":"v2-…"},"media_type":"application/zip","bytes":738167}
   {"rendering":{"bundle":false,"format":"csv","layout":2},"media_type":"text/csv","bytes":732347}
   {"rendering":{"statistics":"series"},"media_type":"application/vnd.apache.parquet","bytes":29579}
   ```

   :::tabs

   ::tab{label="curl"}

   ```bash
   curl -s "$DFO_API/api/datasets/spec_91d7345eec81a029629938b3d0ffd775/cv_b785befb9f87471de96eb31cd7c56f5c" \
     | jq -c '.artifacts[] | {rendering, media_type, bytes}'
   ```

   ::tab{label="Python"}

   ```python
   import json
   import os

   import httpx

   DFO_API = os.environ["DFO_API"]

   url = f"{DFO_API}/api/datasets/spec_91d7345eec81a029629938b3d0ffd775/cv_b785befb9f87471de96eb31cd7c56f5c"
   manifest = httpx.get(url, timeout=120).raise_for_status().json()
   for a in manifest["artifacts"]:
       print(json.dumps({k: a[k] for k in ("rendering", "media_type", "bytes")}, separators=(",", ":")))
   ```

   :::

   `{"statistics": "series"}` is a one-row-per-series summary with the first and latest value, minimum, maximum and change. It is always Parquet, whatever format you requested. Match files by the fields of `rendering`. A bundle's `rendering` also carries a `rendition` token that changes when the bundle's layout or the website setting behind its links changes, and a bare file's carries a `layout` number that changes when its columns change, so compare `bundle` and `format` and leave the whole object alone. A file asked for with `"layout": "frame"` carries `"layout": "frame"` and is listed after the long-layout files, so a match on `bundle` and `format` alone finds the long layout first when the version holds a long file, and the frame file when it holds only frame files; add `layout` to the match to find the frame file. A version can also hold a pinned bundle from an earlier layout. The manifest lists it after the current one, so the first match is the current layout when a current bundle exists, and the pinned bundle otherwise. Media type alone can select the summary when you need the data rows.

   ```python title="Right"
   # Match the rendering fields you asked for. The first match is the current layout when a current bundle exists, and otherwise a pinned older bundle
   data = next(a for a in manifest["artifacts"] if a["rendering"].get("bundle") is True and a["rendering"]["format"] == "parquet")
   ```

   ```python title="Wrong"
   # Matching the media type can select the per-series summary, which is also Parquet, when the rows are requested
   data = next(a for a in manifest["artifacts"] if a["media_type"] == "application/vnd.apache.parquet")
   ```

   Each `url` is signed and expires 15 minutes after the response, at `url_expires_at`.

1. **Fetch the snapshot again later.** `GET /api/datasets/{spec_id}/{content_version}` returns the same manifest from storage with fresh signed URLs. It reads from storage and bypasses the graph, so the rows match the version you minted.

   ```text
   cv_b785befb9f87471de96eb31cd7c56f5c (2177, 13)
   ```

   ```python
   import io
   import os

   import httpx
   import pandas as pd

   DFO_API = os.environ["DFO_API"]
   client = httpx.Client(base_url=DFO_API, timeout=120)
   manifest = client.get("/api/datasets/spec_91d7345eec81a029629938b3d0ffd775/cv_b785befb9f87471de96eb31cd7c56f5c").raise_for_status().json()
   csv = next(a for a in manifest["artifacts"] if a["rendering"].get("bundle") is False and a["rendering"]["format"] == "csv")
   df = pd.read_csv(io.BytesIO(httpx.get(csv["url"]).raise_for_status().content))
   print(manifest["content_version"], df.shape)
   ```

   A stored file stays downloadable for 30 days after it is made, and a download does not extend that. A pinned file stays for good. When a bundle's README prints a website link, the website pins the Parquet rows of each version a visitor downloads there, so that link keeps working and the version's rows stay available as Parquet. Once every file of a version has aged out, the route answers `410 artifact_expired` and points at the route below.

   A call to `GET /api/datasets/{spec_id}` resolves the spec against the live graph again and returns the latest data. A source revision can produce a newer `content_version`. The route takes the same options as query parameters, such as `?format=parquet&bundle=false`. <!-- TODO: verify GET /api/datasets/{spec_id} against dev; it was not run for this page because it can materialise a new version. -->

## The bundle

A bundle is a zip that explains itself to a reader who has never heard of the DFO. A one-definition bundle holds these files:

| File                | Contents                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`         | Start here: what the data is, each file and column, the licence, how to cite it, how to get it again.                                    |
| `data/<slug>.<ext>` | The table, in the format you asked for. `<slug>` comes from the source names.                                                            |
| `data/codes.csv`    | The plain name of every code in the table, such as `country/DEU` for Germany.                                                            |
| `SOURCES.md`        | For each source: its publisher, licence, attribution text, update cadence, and each step the DFO applied.                                |
| `CITATION.cff`      | The citation, in a format reference managers read.                                                                                       |
| `datapackage.json`  | A Frictionless Data Package 2.0 descriptor: typed columns, foreign keys to `codes.csv`, hashes, and the call that reproduces the bundle. |

`data/codes.csv` is absent when the table holds no codes.

The DFO is the author of the extract in `CITATION.cff`. Each source publisher appears as a reference with its own licence, link and attribution text.

Every download has a name that says what it holds:

| File                   | Name                                                             |
| ---------------------- | ---------------------------------------------------------------- |
| Bundle                 | `dfo_<sources>_<YYYY-MM-DD>_<format>_<version>.zip`              |
| Table without a bundle | `dfo_<sources>_<YYYY-MM-DD>_<version>.<ext>`                     |
| Per-series summary     | `dfo_<sources>_<YYYY-MM-DD>_series-statistics_<version>.parquet` |

`<sources>` is the slug of the source names: `oecd-creditor-reporting-system` for one source, `<a>-and-<b>` for two, and `<a>-and-<n>-more` for more. `<YYYY-MM-DD>` is the UTC date the DFO first prepared that version, and it stays fixed when the files render again. `<version>` is the first six characters of the `content_version` after `cv_`. The name arrives in the `Content-Disposition` header of the signed URL, so `curl -OJ` saves a file under it.

The bundle's `README.md` tells its reader how to get the data again without a key: an exact version through the snapshot route above, the latest data through `GET /api/datasets/{spec_id}`, and a link to the website where it offers the dataset page.

## When a mint runs in the background

The Datasets API estimates each request's row count and source requests from a published index before it fetches data. A request runs at once and returns `200` only when both estimates fit within the sync budget:

| Estimate        | Sync budget | Background ceiling |
| --------------- | ----------- | ------------------ |
| Rows            | 100,000     | 2,000,000          |
| Source requests | 40          | 800                |

A request over either sync budget returns `202` and runs in the background. A request over either background ceiling returns `422 over_ceiling` before writing data. `GET /api/datasets/capabilities` publishes the current values as `sync_budget_rows`, `sync_budget_requests`, `async_max_rows` and `async_max_requests`.

Row count alone does not predict the outcome. The Nigeria health request has 2,177 rows, far under 100,000, but covers 63 donors, and fetching them took more than 40 source requests.

The API returns an existing operation when a request matches one already running or finished for the same spec and `options`. When 20 operations are waiting or running, a new request returns retryable `503 service_unavailable`.

## How the DFO picks a source for each series

Every series in a dataset has three facet axes: `provenance` (the source), `measurementMethod` and `unit`. When one StatVar is reported by more than one source, method or unit, each combination is a separate facet. Combining rows from two facets in one series would mix, for example, two sources' figures for the same year, so each series takes one whole facet.

Bind an axis yourself or let the default facet policy, `dfo/facet-policy/most-observations` version 1, set it. The manifest records the selection in `trace.facets`.

```json
[
  {
    "axis": "provenance",
    "ref": "provenance/OECDCreditorReportingSystem",
    "basis": "deployment_fallback",
    "default_policy": "dfo/facet-policy/most-observations",
    "selection_reason": "sole_candidate"
  },
  {
    "axis": "unit",
    "ref": "CurrencyUSD",
    "basis": "deployment_fallback",
    "default_policy": "dfo/facet-policy/most-observations",
    "selection_reason": "sole_candidate"
  }
]
```

```bash
jq '[.trace.facets[] | {axis, ref, basis, default_policy, selection_reason}]' manifest.json
```

The policy keeps facets that agree with the bound axes. When the request names a year range, it then keeps the facets whose first and last dates overlap that range, or all of them if none does. It ranks the survivors and selects the first by:

1. for a variable that both DFO and base Data Commons hold, DFO's facets first, so places only the base source covers show as a coverage gap; an axis you bind to a value only a base facet carries selects that base facet instead
1. the facet with data in the requested window at the most bound places, or for a variable with a provider and a recipient at the most pairs of them, so one source serves every place you name and a place it does not cover returns no rows
1. for a year range, the most years of the range the facet's first and last dates cover
1. most observations, summed across the bound entities
1. most recent observation
1. provenance id, then measurement method, then unit, then facet id

A definition that binds no entity covers every entity with data. A variable with donor and recipient roles covers every pair the DFO holds. A variable about a place covers the countries where it holds data, and the world total alone when no country holds any. This is the scope a Search answer to the same question covers. Name a region, an income group, a state or a city to get its data, and name `Earth` to get the world total beside countries.

The order is total, so the same candidates always give the same choice, and the choice feeds `content_version`. The window step lets a request reach a facet that covers other years. A StatVar with recorded values to 2024 and projections for 2025-2032 under `measurementMethod: dcid:Estimated` serves the recorded values by default and for 2020-2027, and the projections for 2027. A choice changed by the window also changes `content_version`. A year range with no end year ends at the current year, so "since 2016" selects among the facets holding recorded data and returns rows to the current year; name the end year to reach projections. In a survey of 2,016 StatVar and entity pairs on the development graph, 95% had exactly one candidate facet, so the ranking decides the other 5%.

`selection_reason` says how each axis was set:

| `selection_reason`                    | Meaning                                                                                                                                         |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `sole_candidate`                      | One facet survived.                                                                                                                             |
| `most_observations`                   | Several survived, all agreed on this axis, and the winner ranked first in the order above.                                                      |
| `most_observations_over_disagreement` | The candidates disagreed on this axis and the top-ranked facet's value was taken. The manifest also carries a `facet_default_selected` warning. |
| `dfo_precedence`                      | The candidates disagreed on this axis and DFO's facet ranked first because DFO holds data at a requested place.                                 |
| `absent_on_selected_facet`            | The top-ranked facet leaves this axis empty.                                                                                                    |

`basis` is `deployment_fallback` for every choice this policy makes. A top-ranked facet must carry provenance for citation. The mint returns `default_facet_undefined` when the selected facet lacks provenance.

To override a choice, bind the axis in `input.scope.facets`.

```json
{
  "schema_version": 1,
  "input": {
    "kind": "spec_id",
    "spec_id": "spec_774d6c001a6250aa67ca676ab067895e",
    "definition": {…},
    "scope": {"facets": {"provenance": {"ref": "provenance/OECDCreditorReportingSystem"}}}
  },
  "options": {"format": "parquet"}
}
```

<!-- TODO: verify a scope.facets mint against dev. The example has not run. -->

Axis bindings require a concrete ref. A concrete ref fixes the choice and can select a narrower set of rows.

## Verify it worked

The manifest has a `content_version` starting `cv_`, `coverage.rows` above zero, and an artifact whose `rendering` matches the `format` and `bundle` you asked for. Minting the same spec again returns the same `content_version` while the source data is unchanged.

## Troubleshooting

- A `404 snapshot_not_found` response means the requested `content_version` is absent from the known `spec_id`'s snapshots. Check that you paired the right two ids.

  ```json
  {
    "type": "/codes/snapshot_not_found",
    "title": "Snapshot Not Found",
    "status": 404,
    "detail": "'spec_774d6c001a6250aa67ca676ab067895e' has no version at content_version 'cv_00000000000000000000000000000000'",
    "code": "snapshot_not_found",
    "trace_id": "cf1dcaa141ef451d95d3ee286097fbc4",
    "capability_version": "dset_b748e02db7e978fe",
    "retryable": false
  }
  ```

- A `404 spec_not_found` response means you sent a `spec_id` without its `definition`, and the DFO has not seen the spec before. Send the `definition` from the Search response.

- A `404 operation_not_found` response means the operation id is wrong or the operation is older than 30 days.

- A `400` with `ExpiredToken` means the signed download URL expired. Fetch the snapshot again for fresh URLs.

- A `422 excel_limit` response means the table is too large for Excel. Ask for `csv` or `parquet`.

## See also

- [Conventions](conventions.md) lists every Datasets error code and the limits in full.
- The data bundle, on the DFO docs hub, describes each file in the zip at more length. <!-- TODO: link to the hub page "The data bundle" when it is public. -->
