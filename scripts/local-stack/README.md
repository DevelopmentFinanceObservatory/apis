# Local stack

One command runs dfo-apis on your machine with no Google Cloud: Postgres 18 and fake-gcs-server in Docker, both stores migrated, and dfo-apis serving the Datasets and Presentation surfaces on `http://127.0.0.1:58083`. A seed mints four readings through the real mint door, and a smoke script checks each one over HTTP.

```bash
just local-stack up      # containers, migrations, release and catalogue, dfo-apis
just local-stack seed    # mint the four readings
just local-stack smoke   # check them over HTTP
just local-stack down    # stop dfo-apis, remove the containers and runs/local-stack/
```

`just local-stack status` prints what is running. Each command runs `scripts/local-stack/stack.sh`, which needs Docker, `uv` and `curl`.

## Start

`up` starts the two containers of `compose.yaml` (project `dfo-local-stack`), creates the buckets `dfo-local-working` and `dfo-local-pinned` on the emulator, and runs `python -m dataset_store.migrate up` and `python -m presentation_store.migrate up`. It then publishes the graph release and origin catalogue the test suite uses under `runs/local-stack/` (`publish_inputs.py`), the catalogue's base table naming the fixture's population variable `Total population` where the test sample holds it empty and starts uvicorn on `local_app:create_app` in the background. It returns once both `/api/datasets/healthz` and `/api/presentation/healthz` answer 200. The server log is `runs/local-stack/server.log`.

| Service         | Address                  |
| --------------- | ------------------------ |
| dfo-apis        | `http://127.0.0.1:58083` |
| Postgres 18     | `127.0.0.1:55418`        |
| fake-gcs-server | `http://127.0.0.1:54443` |

The containers keep no volume, so `down` discards every row and object, and the next `up` starts empty.

## Seed

`seed` posts each reading in `fixtures/readings.json` to `POST /api/datasets` as a `statvar_set` mint with its scope, and writes the spec ids and content versions to `runs/local-stack/seeded.json`. Minting is idempotent, so a second `seed` answers the same versions and stores nothing new.

| Reading                                                                      | Spec id                                 | Content version                       |
| ---------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------- |
| ODA from Germany to Kenya, Ghana and Nigeria, 2018 to 2023                   | `spec_7f535e528a2a527aa54b316d4eccbb6e` | `cv_46e66114368dacdffa282c61ee456e0f` |
| ODA to Ethiopia from Germany, France, the UK, Japan and the US, 2018 to 2023 | `spec_cb70dde0fb3dc8f98022695a24ccbb77` | `cv_b440fa3d3b8bcaf1641e93bfb26dd79a` |
| Kenya population, 2015 to 2024                                               | `spec_429926a07022f78c65e0379e8004ae6d` | `cv_3f610869372f93965119e3c0221847f6` |
| Kenya GNI (PPP), 2015 to 2024                                                | `spec_fb3e3180bb4bac8acb968f3b6fccc6dd` | `cv_96556e3d3e312acb219d633392cd1d05` |

The ODA readings use `Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal` under `provenance/OECDDAC2A` in current US dollars. The Kenya readings use `Count_Person` and `Amount_EconomicActivity_GrossNationalIncome_PurchasingPowerParity` under the World Development Indicators. Each StatVar's arcs are the dev graph's, copied from `tests/fixtures/graph_nodes.json`. The values and facet ids are synthesised at realistic magnitudes, because no recorded fixture holds these series' points. A change to `fixtures/readings.json` changes the content versions above.

dfo-apis reads its graph, SDMX endpoint and coverage index from doubles that `upstream.py` builds over the same fixture, and admits every provenance through a fake origin catalogue. The mint and observations doors therefore read the fixture as they would read the live graph. The Presentation surface builds its own graph client over HTTP, so `local_app.py` answers its `/v2/node` reads from the same fixture at `/local-stack/graph`, and a card names the fixture's places, variables and sources. `tests/scripts/test_local_stack_seed.py` mints each reading over in-memory stores in the default suite, so a mint change the fixture no longer satisfies fails there.

## Smoke

`smoke` checks each seeded reading:

- `GET /api/datasets/{spec_id}/versions` lists its content version.
- `GET /api/datasets/{spec_id}/{content_version}` answers its manifest with artifacts.
- `GET .../observations` answers every fixture series with the fixture's points.
- Each artifact's `files/{digest}` link answers 302, and the bytes at the location hash to the digest.

It ends with `analysis_step` in `smoke.py`, which divides the German ODA reading by Kenya's population through `POST /api/datasets/analysis`, pairing the recipient with the place. It reads the derived frame's manifest and Parquet file at `/api/datasets/derived/{ops}/{ocv}` and checks the six Kenya rows against the fixture's quotients, and reads the bare derived root under `latest`, which answers the same `ops_` and `ocv_`. It then makes a table card of the frame from the release's minimal table intent, checks that the card's description names the frame's two roots as its sources, in the ref's order, makes a stat card and prints its title, series label and unit, and makes a live share of the card, checks the share's page, embed and data URLs and its `shw_` write token, replaces the share with the token, sends a wrong token for a 403 `write_token_invalid`, deletes the share, and deletes it again with the token for a second 204.

## Download links

In production the files routes answer 302 at a V4 URL that `artifact_store.signing.sign_get_url` signs through IAM `signBlob`, which needs Google credentials. With `STORAGE_EMULATOR_HOST` set, `sign_get_url` returns the object's unsigned media link instead, on `STORAGE_EMULATOR_LINK_BASE`, which `local.env` sets to the API's origin, for example `http://127.0.0.1:58083/download/storage/v1/b/dfo-local-working/o/sha256%3A...?alt=media&response-content-type=text%2Fcsv&response-content-disposition=...`. `local_app.py` serves that path: it streams the object from fake-gcs-server under the content type and disposition the link names, as a signed URL's overrides set them. A client that follows redirects only to the API's origin or `storage.googleapis.com` therefore downloads from the local stack unchanged. The rest of the route runs unchanged: the version read, revocation, retention and renewal, and the 302 with `no-store`.

## Environment

`local.env` holds every setting, passed to each `uv run` with `--env-file`. `stack.sh` also exports `GRAPH_RELEASE_URI` and `ORIGIN_CATALOGUE_URI` (absolute `file://` URIs under `runs/local-stack/`) and `PYTHONPATH` at the repository root, because the publisher, the doubles and the seed import from `tests/`.

| Setting                                                                                       | Value                                                                                                                                                    |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATASET_STORE_*`, `PRESENTATION_STORE_*`                                                     | Postgres on port 55418; migrator `dfo_local_migrator`, runtime roles `datasets_runtime`, `datasets_worker`, `presentation_runtime` (`postgres-init.sql`) |
| `STORAGE_EMULATOR_HOST`                                                                       | `http://127.0.0.1:54443`                                                                                                                                 |
| `STORAGE_EMULATOR_LINK_BASE`                                                                  | `http://127.0.0.1:58083`, the API's origin, which serves download links                                                                                 |
| `DATASETS_ARTIFACT_BUCKET_WORKING`, `_PINNED`                                                 | `dfo-local-working`, `dfo-local-pinned`                                                                                                                  |
| `DATASETS_FIRST_PARTY_PRINCIPAL` (publish)                                                    | `local-publish-principal`                                                                                                                                |
| `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` (pin)                                                    | `local-pin-principal`                                                                                                                                    |
| `LIVE_REFRESH_PRINCIPAL` (refresh)                                                            | `local-refresh-principal`                                                                                                                                |
| `DFO_CALLER_PRINCIPAL`                                                                        | `local-caller-principal`                                                                                                                                 |
| `DFO_CALLER_HMAC_KEY`                                                                         | `local-caller-hmac-key`                                                                                                                                  |
| `DATASETS_GRAPH_BASE_URL`                                                                     | `http://127.0.0.1:58083/local-stack/graph/core/api`, `local_app.py`'s `/v2/node` double over the fixture                                                 |
| `DATASETS_SDMX_BASE_URL`, `UN_GRAPH_BASE_URL`, `SEARCH_GRAPH_BASE`                            | `http://127.0.0.1:9`, which refuses connections                                                                                                          |
| `DFO_PUBLIC_SITE_URL`                                                                         | `http://127.0.0.1:3000`                                                                                                                                  |

A grant goes in the `x-dfo-principal` header. A caller without the pin grant is named from the rightmost `X-Forwarded-For` entry, which Cloud Run appends in production, so a direct local call sends one itself:

```bash
curl -X POST http://127.0.0.1:58083/api/datasets/$SPEC/$CV/pin \
  -H 'content-type: application/json' -H 'X-Forwarded-For: 127.0.0.1' \
  -d '{"artifact_digest": "sha256:..."}'
```

A caller standing in for the website sends `x-dfo-caller-principal: local-caller-principal` and its client in `x-dfo-caller`.

## What it does not cover

- Search: `local_app.py` leaves it out, because its startup checks the Gemini key with Gemini and reads the live graph. Nothing calls an LLM.
- Graph and Skills surfaces: not mounted.
- Operations: the Datasets surface takes a `NullTaskClient` and no worker runs, so a mint too large to answer in the request refuses when it would enqueue.
- Readings outside the fixture: a mint or definition read of any other StatVar or place finds nothing in the doubles.
- Signed URLs: download links are unsigned, so neither IAM signing nor URL expiry is exercised.
- Cloud SQL connector and IAM database auth: the stores connect over TCP with passwords.
