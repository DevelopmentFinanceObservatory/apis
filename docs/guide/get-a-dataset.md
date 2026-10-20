---
description: Create, retrieve, and download a versioned DFO dataset.
---

# How to get a dataset

Use `POST /api/v1/datasets` to create CSV, XLSX, or Parquet data from a resolved spec.

## Before you start

Start with a definite spec from [How to resolve a question](resolve-a-question.md). Set the development API URL.

```bash
export DFO_API=https://developmentfinance.org
```

This page stores the Search response in `resolve.json`, the response to the create request in `response.json`, and the completed manifest in `manifest.json`.

## Steps

1. Resolve the question and save the result.

   ```bash
   curl -fsS -X POST "$DFO_API/api/v1/search/resolve" \
     -H "Content-Type: application/json" \
     -d '{"input":{"kind":"raw_text","query":"ODA disbursements from Germany to Nigeria for basic health care"}}' \
     > resolve.json

   jq -e '.kind == "resolution" and .parts[0].status == "definite"' resolve.json > /dev/null || {
     echo "Search did not return a definite interpretation"
     exit 1
   }
   ```

2. Build the request body from the Search response's `spec_id` and `definition`, and save it as `mint.json`.

   ```bash
   jq '{
     schema_version: 1,
     input: {
       kind: "spec_id",
       spec_id: .parts[0].spec.spec_id,
       definition: .parts[0].spec.definition
     },
     options: {format: "parquet", bundle: true}
   }' resolve.json > mint.json

   curl -fsS -X POST "$DFO_API/api/v1/datasets" \
     -H "Content-Type: application/json" \
     -d @mint.json > response.json
   ```

   `schema_version` is `1`. Supported formats are `csv`, `xlsx`, and `parquet`. Check `GET /api/v1/datasets/capabilities` for the active formats. `bundle` defaults to `true` and includes the data file, `README.md`, `SOURCES.md`, `CITATION.cff`, and `datapackage.json`. `data/codes.csv` is present when the data contains codes.

   Supply `definition` when the Datasets API has not registered the `spec_id`. Send the Search definition unchanged so the API can confirm that it reproduces the identifier.

3. Check whether the request returned a manifest or an operation.

   ```bash
   jq 'if .operation then
     {operation: .operation}
   else
     {
       content_version,
       coverage,
       artifacts: [.artifacts[] | {rendering, media_type, url}]
     }
   end' response.json
   ```

   A request within the synchronous budget returns `200` and a manifest. A larger request returns `202` and an `operation` object. The initial operation response contains `operation.poll` and `retry_after_ms`.

4. Save a completed manifest from the response.

   Run this Python block after steps 1 to 3. It writes `manifest.json` for the remaining steps.

   ```python
   import json
   import os
   import time
   from pathlib import Path

   import httpx

   result = json.loads(Path("response.json").read_text())
   with httpx.Client(base_url=os.environ["DFO_API"], timeout=120) as client:
       if "operation" in result:
           poll_url = result["operation"]["poll"]
           wait_seconds = result["operation"].get("retry_after_ms", 1500) / 1000
           while True:
               time.sleep(wait_seconds)
               response = client.get(poll_url)
               response.raise_for_status()
               result = response.json()
               if result["operation"]["status"] == "done":
                   manifest = result["manifest"]
                   break
               wait_seconds = result["operation"]["retry_after_ms"] / 1000
       else:
           manifest = result
   Path("manifest.json").write_text(json.dumps(manifest))
   ```

   Operations can be `pending`, `running`, `done`, or `failed`. Poll responses for pending and running work carry `retry_after_ms`, which grows from 1,500 to 8,000 as the build runs (a resubmission that restarts a failed build starts again at 1,500), and a `Retry-After` header with the same wait in whole seconds, rounded up. Wait that long between polls. A done response omits `poll` and places the manifest beside `operation`. A failed poll returns `500 operation_failed`, so `raise_for_status()` stops the script.

5. Select the data artifact by `rendering`, then download it from its link.

   ```python
   import json
   import os
   from pathlib import Path

   import httpx

   manifest = json.loads(Path("manifest.json").read_text())

   artifact = next(
       item
       for item in manifest["artifacts"]
       if item["rendering"].get("format") == "parquet"
       and item["rendering"].get("bundle") is True
   )
   response = httpx.get(os.environ["DFO_API"] + artifact["url"], timeout=120, follow_redirects=True)
   response.raise_for_status()
   Path("bundle.zip").write_bytes(response.content)
   ```

   Each `url` is a download link relative to the API host, `/api/datasets/{spec_id}/{content_version}/files/{artifact_digest}`. The path carries no version prefix and works as returned. The address of the link is stable. Once the file behind it passes retention, the link renews it where it can and otherwise answers `410 artifact_expired` (see [Retention and capacity](#retention-and-capacity)). A request to it answers `302` with a storage URL signed for five minutes, so follow redirects (`curl -L`, or `follow_redirects=True` in httpx). Once the version is taken down, the link answers `410 snapshot_revoked`.

   A manifest can include several artifacts for one `content_version`, including a Parquet series-statistics file. Select the requested `format` and `bundle` fields inside `rendering`. Bundle renderings also carry a rendition token that can change when the archive layout changes. The statistics artifact has one row per series and is separate from the observation table.

   `datapackage.json` lists each data resource and its path inside a bundle. Read it before opening a data file.

6. Retrieve a stored snapshot to list a version's files again.

   ```bash
   spec_id="$(jq -r '.definition.spec_id' manifest.json)"
   content_version="$(jq -r '.content_version' manifest.json)"
   curl -fsS "$DFO_API/api/v1/datasets/$spec_id/$content_version" > snapshot.json
   jq '{content_version, artifacts: [.artifacts[] | {rendering, url}]}' snapshot.json
   ```

   A snapshot reads the stored version and lists the same download links while its artifacts remain servable.

   To fetch a stored version in another format, ask the file selector. `GET /api/v1/datasets/$spec_id/$content_version/file?format=csv&bundle=false` answers `302` with a signed URL for that rendering, laid over the version's own options, and renders it from the stored frame when the version holds none yet. `layout` takes `frame`, and any other query key answers `400 request_invalid`. `GET /api/v1/datasets/$spec_id/versions` lists the dataset's stored versions, newest first, with their `rendering`. It takes `limit` (1 to 500, default 100) and `cursor`, and the body carries `next_cursor` and `complete`.

## Verify it worked

A manifest has a `content_version`, coverage, and an artifact whose rendering matches the requested format. The selected link downloads a file whose SHA-256 digest equals its `artifact_digest`.

## Retention and capacity

Generated artifacts are working-tier by default and expire after the retention period published by `GET /api/v1/datasets/capabilities` (30 days), measured from `stored_at`. A pinned artifact is kept indefinitely. After a working-tier artifact expires, its download link renews the file while the file's bytes survive, renders it again from the version's stored frame when they are gone, and answers `410 artifact_expired` when the frame is gone too. Only a csv, xlsx or parquet data file in the current bundle rendition and layout renders again, so a statistics file, a superseded bundle and a bare file in an old layout answer `410 artifact_expired` with the frame intact. A snapshot can return `410 artifact_expired` when no listable artifact location is servable, or `410 snapshot_revoked` for a revoked version.

Terminal operations are guaranteed pollable for the retention period in the Datasets capabilities response. An older operation can still be present. `404 operation_not_found` means the API does not know that operation.

`GET /api/v1/datasets/capabilities` publishes the synchronous budgets and background ceilings. Requests over either synchronous budget can run in the background if they fit within both background ceilings. A request beyond a background ceiling returns `422 over_ceiling`.

```bash
curl -fsS "$DFO_API/api/v1/datasets/capabilities" \
  | jq '.caps | {sync_budget_rows, sync_budget_requests, async_max_rows, async_max_requests}'
```

## Choose a source, method, or unit

Bind a facet in `input.scope.facets` when the data requires a particular provenance, measurement method, or unit. [How the APIs fit together](concepts.md) explains facet selection and unit conversion.

Add this `scope` object to `input` in `mint.json`. Keep `kind`, `spec_id`, and `definition` alongside it.

```json
{
  "scope": {
    "facets": {
      "provenance": {"ref": "provenance/OECDCreditorReportingSystem"},
      "unit": {"ref": "CurrencyUSD"}
    }
  }
}
```

## Request a bare frame

Use `layout: "frame"` for a bare Parquet file in the frame layout. The request needs `format: "parquet"` and `bundle: false`. Frame layout cannot be combined with `options.pivot`, which reshapes the table from long to wide with one `index` column and one `columns` column.

```json
{
  "options": {"format": "parquet", "bundle": false, "layout": "frame"}
}
```

## Next

[How the APIs fit together](concepts.md) explains versions, rows, series, and facets. [API conventions](conventions.md) documents response envelopes and problem details.
