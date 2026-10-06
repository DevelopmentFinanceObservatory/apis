---
description: Turn a plain-English question into a DFO dataset and read it as Parquet, first with curl and then in Python.
---

# Quickstart

Ask how much official development assistance Germany disbursed to Nigeria for basic health care. You'll turn the answer into a dataset and read it as Parquet. The result is a 30-row table covering 1995 to 2024.

You'll learn to:

- resolve a question into a spec with the Search API
- mint a dataset from that spec with the Datasets API
- download the dataset bundle and read its Parquet file
- run the whole flow from Python

You need `curl`, `jq` and `unzip` for the first three steps, and [uv](https://docs.astral.sh/uv/) for the last. It takes about ten minutes.

## Set the base URL

Set the base URL once for all calls.

```bash
export DFO_API=https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app
```

!!! note
    This is the development deployment of the DFO APIs. It takes calls without a key. <!-- TODO: replace with the public base URL and access instructions once the API access policy exists. -->

## Step 1: resolve the question

The Search API reads your question and returns a spec: the DFO's exact reading of what you asked, with a `spec_id` that names it. Nothing is downloaded yet.

You should see:

```json
{
  "kind": "resolution",
  "status": "definite",
  "spec_id": "spec_774d6c001a6250aa67ca676ab067895e",
  "sentence": "Germany's gross ODA disbursements to Nigeria for basic health care, 1995 to 2024. Current US dollars · OECD Creditor Reporting System (CRS)."
}
```

Run this to get it:

```bash
curl -s -X POST "$DFO_API/api/search/resolve" \
  -H "Content-Type: application/json" \
  -d '{"input": {"kind": "raw_text", "query": "German aid disbursements to Nigeria for basic health care"}}' \
  > resolve.json

jq '{kind} + (.parts[0] | {status, spec_id: .spec.spec_id, sentence: .spec.resolved.rendered_sentence})' resolve.json
```

`resolution` means the question asked for data, so the answer carries `parts`. A question about what the DFO holds answers `catalogue`, and an off-topic one `declined`, and neither carries `parts`; [Resolve a question](resolve-a-question.md) shows how to branch on `kind`. `definite` means the DFO found one reading of the question. The sentence is that reading in words, so check it says what you meant. Here it matched "basic health care" to CRS purpose code 12220 and read "aid" as official development assistance.

## Step 2: mint a dataset

Mint the spec into stored files. Send its `spec_id` and the `definition` returned beside it by the Search API, then request Parquet.

You should see:

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
      "rendering": {
        "statistics": "series"
      },
      "media_type": "application/vnd.apache.parquet",
      "bytes": 5978
    }
  ]
}
```

Build and send the mint request from `resolve.json`.

```bash
jq '{schema_version: 1,
     input: {kind: "spec_id", spec_id: .parts[0].spec.spec_id, definition: .parts[0].spec.definition},
     options: {format: "parquet"}}' resolve.json > mint.json

curl -s -X POST "$DFO_API/api/datasets" \
  -H "Content-Type: application/json" \
  -d @mint.json > manifest.json

jq '{content_version, rows: .coverage.rows, artifacts: [.artifacts[] | {rendering, media_type, bytes}]}' manifest.json
```

The response is a manifest. `content_version` names this exact result, so the pair `spec_774d…` and `cv_e7f6…` names these 30 rows. The first artifact is the bundle: a zip holding your Parquet file with a README, the codes behind every code column, the sources and their licences, and the citation. The second is a one-row-per-series summary, which you won't need here.

## Step 3: download the Parquet file

Each artifact carries a signed `url` that expires after 15 minutes. Select the first bundle whose `rendering` has `bundle` true and the `format` you asked for, then download and unzip it. `-OJ` saves the file under the name the download carries.

```bash
curl -s -OJ "$(jq -r 'first(.artifacts[] | select(.rendering.bundle == true and .rendering.format == "parquet")) | .url' manifest.json)"
unzip dfo_*_parquet_*.zip
```

You should see:

```text
Archive:  dfo_oecd-creditor-reporting-system_2026-09-28_parquet_e7f6ed.zip
 extracting: data/oecd-creditor-reporting-system.parquet
 extracting: CITATION.cff
 extracting: README.md
 extracting: SOURCES.md
 extracting: data/codes.csv
 extracting: datapackage.json
```

The zip is named `dfo_<sources>_<date first prepared>_<format>_<version>.zip`. The version is the first six characters of the `content_version` after `cv_`.

`data/oecd-creditor-reporting-system.parquet` is the dataset. `README.md` is the place to start. It says what the data is, explains every column, and shows how to cite it and how to get it again. `data/codes.csv` gives the plain name of each code, such as `country/DEU` for Germany. `SOURCES.md` names each source, its licence and each step the DFO applied. `CITATION.cff` is the citation in a format reference managers read, and `datapackage.json` describes every column for software.

## Step 4: do the same in Python

The script below runs steps 1 to 3 with `httpx` and loads the Parquet file into pandas. There is no DFO client package, so plain HTTP calls are the way in.

You should see:

```text
spec_774d6c001a6250aa67ca676ab067895e cv_e7f6ed9ddc5ca69902f92a8866f93ef7 (30, 13)
   time_period financingProvider financingRecipient      value         unit
27        2007       country/DEU        country/NGA  1330990.0  CurrencyUSD
28        2008       country/DEU        country/NGA   112400.0  CurrencyUSD
29        2009       country/DEU        country/NGA   224200.0  CurrencyUSD
```

Save the script as `quickstart.py`.

```python title="quickstart.py"
import io
import json
import os
import time
import zipfile

import httpx
import pandas as pd

DFO_API = os.environ["DFO_API"]
client = httpx.Client(base_url=DFO_API, timeout=120)

# 1. Resolve the question into a spec.
question = "German aid disbursements to Nigeria for basic health care"
resolved = client.post("/api/search/resolve", json={"input": {"kind": "raw_text", "query": question}})
answer = resolved.raise_for_status().json()
if answer["kind"] != "resolution":  # a catalogue or declined answer carries no parts
    raise SystemExit(f"not a data question: {answer['kind']}")
spec = answer["parts"][0]["spec"]

# 2. Mint a dataset from the spec.
body = client.post(
    "/api/datasets",
    json={
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec["spec_id"], "definition": spec["definition"]},
        "options": {"format": "parquet"},
    },
).raise_for_status().json()
if "operation" in body:  # over the sync budget: 202 Accepted, poll until done
    poll = body["operation"]["poll"]
    body = client.get(poll).raise_for_status().json()
    while body["operation"]["status"] != "done":
        time.sleep(2)
        body = client.get(poll).raise_for_status().json()
    body = body["manifest"]
manifest = body

# 3. Download the bundle and read its Parquet file.
bundle = next(a for a in manifest["artifacts"] if a["rendering"].get("bundle") is True and a["rendering"]["format"] == "parquet")
archive = zipfile.ZipFile(io.BytesIO(httpx.get(bundle["url"]).raise_for_status().content))
data_path = json.loads(archive.read("datapackage.json"))["resources"][0]["path"]
df = pd.read_parquet(io.BytesIO(archive.read(data_path)))

print(spec["spec_id"], manifest["content_version"], df.shape)
print(df[["time_period", "financingProvider", "financingRecipient", "value", "unit"]].tail(3))
```

Run it:

```bash
uv run --with httpx --with pandas --with pyarrow python quickstart.py
```

You get the same `spec_id` and `content_version` as the curl steps. Minting the same spec again returns the stored version.

This request fits the synchronous budget and returns at once. The `if "operation" in body` block polls larger requests, which return `202 Accepted` with an operation. The Get a dataset page shows how to poll it.

## What you learned

- You resolved a question into a spec with `POST /api/search/resolve`.
- You minted a dataset from the spec with `POST /api/datasets`.
- You picked the bundle from the manifest's artifacts and read its Parquet file.
- You ran the same flow from Python.

## Next

- [Get a dataset](get-a-dataset.md) covers minting with CSV and Excel, background requests, fetching a snapshot again, and source selection when several sources report the same series.
- Versions, snapshots and citations, on the DFO docs hub, explains why a `spec_id` and a `content_version` are separate names and which one to cite. <!-- TODO: link to the hub page "Versions, snapshots and citations" when it is public. -->
