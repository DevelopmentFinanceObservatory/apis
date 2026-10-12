---
description: Build a Parquet dataset from a plain-English development-finance question.
---

# Build a dataset from a question

Turn a question about German ODA disbursements to Nigeria into a downloadable Parquet dataset.

## What you will build

The result is a ZIP archive containing a Parquet data file, citation metadata, and a data package description.

## What you will learn

- How to check how the Search API interpreted a question.
- How to create a dataset from the resulting spec.
- How to download a Parquet bundle.

## What you need

You need `curl`, `jq`, and `unzip`. The Python version also needs [uv](https://docs.astral.sh/uv/). Set the API base URL once.

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

The development deployment accepts requests without an API key.

## Step 1: resolve the question

Send a question with explicit donor and recipient roles. The confirmation sentence is the first check on the resulting data.

An example response contains these selected fields. The date range depends on the available observations.

```json
{
  "kind": "resolution",
  "status": "definite",
  "spec_id": "spec_774d6c001a6250aa67ca676ab067895e",
  "sentence": "Germany's gross ODA disbursements to Nigeria for basic health care, 1995 to 2024. Current US dollars · OECD Creditor Reporting System (CRS)."
}
```

```bash
curl -fsS -X POST "$DFO_API/api/search/resolve" \
  -H "Content-Type: application/json" \
  -d '{"input":{"kind":"raw_text","query":"ODA disbursements from Germany to Nigeria for basic health care"}}' \
  > resolve.json

jq 'if .kind == "resolution" then
      {kind} + (.parts[0] | {
        status,
        spec_id: .spec.spec_id,
        sentence: .spec.resolved.rendered_sentence
      })
    else
      {kind, reason, scope}
    end' resolve.json
```

The command prints selected fields from a resolution answer. Check `kind` before reading `parts`.

## Step 2: create the dataset

Create a Parquet bundle from the `spec_id` and definition returned by Search. A successful synchronous response is a manifest with a `content_version`, coverage, and artifacts. A background response has an `operation` object.

```bash
jq -e '.kind == "resolution" and .parts[0].status == "definite"' resolve.json > /dev/null || {
  echo "Search did not return a definite interpretation"
  exit 1
}

jq '{
  schema_version: 1,
  input: {
    kind: "spec_id",
    spec_id: .parts[0].spec.spec_id,
    definition: .parts[0].spec.definition
  },
  options: {format: "parquet"}
}' resolve.json > mint.json

if ! curl -fsS -X POST "$DFO_API/api/datasets" \
  -H "Content-Type: application/json" \
  -d @mint.json > response.json; then
  echo "Dataset request failed" >&2
  exit 1
fi

if jq -e '.operation' response.json > /dev/null; then
  poll_url="$(jq -r '.operation.poll' response.json)"
  wait_seconds="$(jq -r '.operation.retry_after_ms / 1000' response.json)"
  while true; do
    sleep "$wait_seconds"
    if ! curl -fsS "$DFO_API$poll_url" > operation.json; then
      echo "Dataset operation failed" >&2
      exit 1
    fi
    if [ "$(jq -r '.operation.status' operation.json)" = "done" ]; then
      jq '.manifest' operation.json > manifest.json
      break
    fi
  done
else
  cp response.json manifest.json
fi

jq '{
  content_version,
  coverage,
  artifacts: [.artifacts[] | {rendering, media_type, url}]
}' manifest.json
```

The command prints selected fields from the manifest. The poll URL is saved from the initial response because a completed operation no longer includes it. A failed poll stops the command before it reads the response as a completed operation.

`content_version` identifies the stored result. Keep it with the `spec_id` when you need to retrieve that same version later.

## Step 3: download the Parquet bundle

Select the requested bundle by its rendering and download it. Each artifact's `url` is a download link relative to the API host that never expires. It answers `302` with a storage URL signed for five minutes, which `-L` follows.

```bash
bundle_url="$(jq -r 'first(.artifacts[] | select(.rendering.format == "parquet" and .rendering.bundle == true)) | .url' manifest.json)"
curl -fsSL "$DFO_API$bundle_url" -o bundle.zip
data_path="$(unzip -p bundle.zip datapackage.json | jq -r '.resources[] | select(.format == "parquet") | .path')"
unzip bundle.zip "$data_path" README.md SOURCES.md CITATION.cff datapackage.json
```

The bundle contains `README.md`, `SOURCES.md`, `CITATION.cff`, and `datapackage.json`. It includes `data/codes.csv` when the data contains codes. Read `datapackage.json` to find the data resource path.

The selected resource contains the observations. `README.md` explains the data and its columns. `SOURCES.md` records the data sources, licences, and processing steps. `CITATION.cff` and `datapackage.json` provide citation and machine-readable metadata.

## Use Python instead

Use this script instead of the curl commands when you want to run the full flow in Python. Save it as `quickstart.py`.

It prints the confirmation sentence, then the first five rows of the Parquet table.

```python title="quickstart.py"
import io
import json
import os
import time
import zipfile

import httpx
import pandas as pd

DFO_API = os.environ["DFO_API"]
question = "ODA disbursements from Germany to Nigeria for basic health care"

with httpx.Client(base_url=DFO_API, timeout=120) as client:
    response = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": question}},
    )
    response.raise_for_status()
    answer = response.json()
    if answer["kind"] != "resolution":
        raise RuntimeError(f"Search returned {answer['kind']}: {answer}")
    part = answer["parts"][0]
    if part["status"] != "definite":
        raise RuntimeError(f"Search returned {part['status']}: {part}")

    spec = part["spec"]
    response = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": spec["spec_id"],
                "definition": spec["definition"],
            },
            "options": {"format": "parquet"},
        },
    )
    response.raise_for_status()
    result = response.json()

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
    else:
        manifest = result

    bundle = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["rendering"].get("format") == "parquet"
        and artifact["rendering"].get("bundle") is True
    )
    response = client.get(bundle["url"], follow_redirects=True)
    response.raise_for_status()

archive = zipfile.ZipFile(io.BytesIO(response.content))
package = json.loads(archive.read("datapackage.json"))
data_path = next(resource["path"] for resource in package["resources"] if resource["format"] == "parquet")
data = pd.read_parquet(io.BytesIO(archive.read(data_path)))
print(spec["resolved"]["rendered_sentence"])
print(data.head())
```

Run the script.

```bash
uv run --with httpx --with pandas --with pyarrow python quickstart.py
```

The script prints the confirmation sentence followed by the first five rows of the Parquet table.

## What you learned

You resolved a question, checked its interpretation, created a Parquet bundle, and read the downloaded data. You can use either the curl path or the Python path to do it.

## Next

[How the APIs fit together](concepts.md) explains specs, versions, StatVars, series, rows, and source selection. [Get a dataset](get-a-dataset.md) covers CSV, Excel, background operations, snapshots, and retention.
