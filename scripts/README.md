# scripts/

Reference for every script in this directory: what it does, how to run it, and what runs it. Each script's module docstring has the full contract. This page is the index.

## `bench_datasets.py`

Benches a deployed Datasets API over two recorded definitions, q1 (a walked family read subject-only) and q3 (an index family read role-bearing): cold and warm mints, cold and warm definition GETs, a snapshot read, 20 concurrent warm GETs, and 3 then 6 concurrent cold mints. It writes p50 and p95 latency and the counts each response's `Server-Timing` header carries (`db_transactions`, `sign_blob_calls`, `permit_wait_ms`, `sdmx_requests`, `graph_requests`) to a JSON report.

```bash
uv run --extra datasets python scripts/bench_datasets.py --out bench-datasets.json
uv run --extra datasets python scripts/bench_datasets.py --base-url <url> --samples 5 --seed 7
```

Needs no credentials for the dev service. Every run mints new versions on the target.

Run by `.github/workflows/datasets-bench.yml` on `workflow_dispatch`, which uploads the report.

## `bundle_round_trip.R`

Round-trips a generated Data Package v2 bundle through the R `frictionless` package, checking the dual v1/v2 profile the package publishes.

```bash
Rscript scripts/bundle_round_trip.R <path-to-bundle-directory>
```

Needs the `frictionless` R package (`install.packages("frictionless")`).

Run by `.github/workflows/bundle-round-trip.yml` on a ready (non-draft) PR touching `packages/dfo-data-tools/src/materialise/**`, this script, or `generate_sample_bundle.py`. Runnable by hand against any bundle `materialise.bundle.assemble_bundle` produces.

## `eval_corpus.py`

Loads `corpus/*.yaml` with the families in `corpus/families/`, checks them, and prints each case that cannot gate with its reasons. `check` fails on a schema error, a duplicate id, an unknown family, a `why` over 200 characters, or a `known_failure` naming an unknown check. `schema` prints the JSON schema of a case file, generated from `search.eval.cases.Case`.

```bash
uv run python scripts/eval_corpus.py check     # validate the corpus
uv run python scripts/eval_corpus.py schema    # print the case JSON schema
```

No live credentials needed.

Run by `just corpus-check` / `just corpus-schema`, and by the `corpus-check` pre-commit hook on a commit touching `corpus/`.

## `eval_paired.py`

Posts each eligible case to a running Search app's `/api/search/resolve`, scores every response with the checks in `search.eval.checks`, and applies the paired gate rule from `search.eval.gate`. Paired mode (`--head-url`, `--base-url`, `--base-ref`) runs head and base apps, reads the base corpus from `git show <base-ref>:corpus/...`, and re-runs at base only the cases whose verdict needs it. Single mode (`--single-url`) scores one app and prints every check that did not pass, per case. Both modes write a JSON report (`--report`), a Markdown summary (`--summary`), and every response (`--responses`) on request.

```bash
uv run python scripts/eval_paired.py --single-url http://127.0.0.1:8100 --file crs
uv run python scripts/eval_paired.py --head-url http://127.0.0.1:8100 \
    --base-url http://127.0.0.1:8101 --base-ref <sha> --report report.json --summary summary.md
```

Scoring reads the graph, SDMX and the graph release's coverage view, so the environment needs the graph settings the apps use and `GRAPH_RELEASE_URI`. Exit code 0 means nothing blocks, 1 a blocking verdict, 2 a setup error, 3 an infrastructure failure.

Run by `just eval-paired` and `just eval-run`, and by `.github/workflows/eval-gate.yml` on pull requests. The nightly eval (`.github/workflows/eval-nightly.yml`) runs `python -m search.eval.sync --nightly`.

## `eval_role_direction.py`

Scores a flow-direction scanner against the frozen cases in `corpus/direction/role_direction_cases.yaml`.

```bash
uv run python scripts/eval_role_direction.py \
    --scanner corpus.direction.role_text_pinned:detect_directions
```

No live credentials needed.

By hand, when comparing a candidate flow-direction scanner against the frozen baseline scanner (`corpus/direction/role_text_pinned.py`).

## `export_graph_entities.py`

Regenerates `entity_resolution`'s committed graph-derived pack data: every `Organization`, `DACRecipientUnallocatedByCountry` and `DACUnspecifiedParty` node (`graph_entities.json`) and the resolvekit region-to-dcid mapping (`region_ids.json`). A record's `dac_code` is the node's numerically lowest `dacCode`; its other codes join its aliases, since a resolvekit entity holds one code per system and every code must resolve.

```bash
uv run --env-file .env python scripts/export_graph_entities.py entities \
    --out packages/dfo-data-tools/src/entity_resolution/data/graph_entities.json
uv run --env-file .env python scripts/export_graph_entities.py regions \
    --out packages/dfo-data-tools/src/entity_resolution/data/region_ids.json
```

Needs `SEARCH_GRAPH_BASE` and `GRAPH_RELEASE_URI` set to a published graph release, whose code index it reads (`code_index.shared.get_index`), with Google credentials that can read a `gs://` release.

By hand, when the graph's organisation, unallocated-recipient or unspecified-party population or the resolvekit region mapping changes.

## `generate_sample_bundle.py`

Generates a small, deterministic Data Package v2 bundle directory for `bundle_round_trip.R` to validate.

```bash
uv run python scripts/generate_sample_bundle.py <output-directory>
```

No live credentials needed.

Run by `.github/workflows/bundle-round-trip.yml`, same trigger as `bundle_round_trip.R`.

## `harvest_llm_fixtures.py`

Extends `tests/fixtures/llm_responses.json` by replaying the engine offline against the frozen graph fixtures, calling the live LLM only for prompts the offline suite cannot answer yet.

The offline suite includes `tests/eval/test_corpus_offline.py`, which runs every corpus case, so one harvest covers the corpus keys too. A test that fails for any reason other than a missing fixture stops the harvest before any LLM call. A module that fails to import stops it too, and so does a run that reports fewer tests than it collected, as when xdist shuts down after too many worker crashes.

The offline suite runs under pytest-xdist on `--workers` processes (default `auto`). It runs serially when `SEARCH_RECORD_REQUESTS`, `SEARCH_RECORD_SCENARIOS` or `SEARCH_RECORD_BASELINE_DEPTH` is 1, because those recording modes fail under xdist.

```bash
uv run python scripts/harvest_llm_fixtures.py
uv run python scripts/harvest_llm_fixtures.py --max-passes 5
uv run python scripts/harvest_llm_fixtures.py --workers 4
```

Needs `GEMINI_API_KEY`.

By hand, when the offline suite reports a FakeLLM fixture miss.

## `intake_sheet.py`

Parses a golden-submission spreadsheet CSV export into a staging YAML shaped like `corpus/*.yaml`.

```bash
uv run python scripts/intake_sheet.py parse <csv> --out corpus/staging/<name>.yaml
```

No live credentials needed.

By hand, when a new batch of goldens arrives from the intake spreadsheet, ahead of a maintainer reviewing and promoting them into `corpus/*.yaml`.

## `ladder.py`

The prompt ladder driver: `capture` runs the production pipeline live over the corpus cases and records every extract/bind call; `run` replays one rung against a completed capture as a bind, extract, or e2e cell; `report` renders a cell's scores.

```bash
uv run --env-file .env python scripts/ladder.py capture --id <capture-id>
uv run --env-file .env python scripts/ladder.py run <rung> --mode e2e
uv run python scripts/ladder.py run <rung> --mode bind --dry-run
uv run python scripts/ladder.py report <cell-id> --against <ceiling-cell-id>
```

`capture` needs `GEMINI_API_KEY` and `SEARCH_GRAPH_BASE`. `run`/`report` replay against a completed capture and need no live credentials except an e2e-mode `run` without `--dry-run`.

Run by `just ladder-capture`, `just ladder-run <rung>`, `just ladder-report`.

## `local-stack/stack.sh`

Runs dfo-apis locally with no Google Cloud: Postgres 18 and fake-gcs-server in Docker, both stores migrated, and the Datasets and Presentation surfaces on `http://127.0.0.1:58083`. `seed` mints four readings from `local-stack/fixtures/readings.json` through the mint door, and `smoke` checks each one's versions, manifest, observations and download links over HTTP.

```bash
just local-stack up
just local-stack seed
just local-stack smoke
just local-stack down
```

Needs Docker, `uv` and `curl`. `local-stack/README.md` lists the settings it uses and what it leaves out.

Run by hand. `tests/scripts/test_local_stack_seed.py` mints the same readings over in-memory stores in the default suite.

## `measure_corpus_latency.py`

Runs every case whose `why` states a timing claim through the live engine `--runs` times each and reports which ones exceed the claimed deadline.

```bash
GRAPH_RELEASE_URI=gs://dfo-dev-graph-release-dev-fin-observatory/ \
    uv run --env-file .env --extra engine --extra eval python scripts/measure_corpus_latency.py
```

Needs `GEMINI_API_KEY` and `GRAPH_RELEASE_URI` (the engine serves the code index and coverage from the bound graph release, and this script runs outside pytest, so there is no conftest to publish one), with Google credentials that can read a `gs://` release. No Langfuse credentials.

By hand, when re-checking the corpus's timing claims against a live deployment.

## `measure_writer_rss.py`

Measures `materialise.formats`' three streaming writers' peak RSS against synthetic CRS-shaped frames.

```bash
uv run python scripts/measure_writer_rss.py
uv run python scripts/measure_writer_rss.py --out /tmp/writer-rss.json
```

No live credentials needed.

By hand. `docs/datasets-sync-ceiling.md` cites this script as the method behind its writer-memory numbers; re-run it when the streaming writers change.

## `record_origin_catalogue_fixture.py`

Writes `tests/fixtures/origin_catalogue/`, the sample origin catalogue build the docker job in `ci.yml` mounts into its read-only container. The directory holds a format root's contents: `current.json` and `versions/`.

```bash
uv run python scripts/record_origin_catalogue_fixture.py
```

No credentials needed; the build is `tests.origin_catalogue.support.sample_tables()` with the UN tables of `sample_un_mirror()`.

With `--eval-from`, it replaces `tests/fixtures/origin_catalogue_eval/` with the build a local catalogue root publishes, once the loader accepts that build under the services' memory budget:

```bash
uv run python scripts/record_origin_catalogue_fixture.py --eval-from /tmp/catalogue-eval
```

By hand, when `tests/origin_catalogue/test_fixture.py` reports that a committed build stopped loading or matching, as after a change to `origin_catalogue.schema.CATALOGUE_FORMAT_VERSION`. The eval build comes from `python -m origin_catalogue.job --local-dir` (see `tests/fixtures/origin_catalogue_eval/README.md`).

## `record_per_n_units.py`

Records every unit whose ref or label reads as a value per N of something, from the three unit classes on the dev graph and the UN System Data Commons, to `tests/fixtures/per_n_units.json`. `tests/unit_words/test_per_n_units.py` holds each one to the unit registry, so a new unit needs a registry row or a named exclusion. Needs `SEARCH_GRAPH_BASE` and a `gcloud` login.

```bash
uv run --env-file .env python scripts/record_per_n_units.py
```

## `record_rate_tables.py`

Records one whole rate or deflator table for `materialise.rate_tables`, in the wire shapes `dc_sdmx.client.SdmxClient` parses: the `/availability` body listing the StatVar's `observationAbout` entities, the `/data` SDMX-CSV body for all of them, and the table provenance's `lastDataRefreshDate`.

```bash
uv run --env-file .env python scripts/record_rate_tables.py \
    --stat-var ExchangeRate_CurrencyUnit_ConversionRecipeDAC \
    --provenance-id provenance/ONEConversionTablesDAC
```

Needs `SEARCH_GRAPH_BASE` (`SEARCH_GRAPH_TOKEN` forwarded when set).

By hand, any time the source refreshes, to keep the recorded fixture current.

## `sentence_golden.py`

Trims a sentence panel run into the golden `tests/render/test_sentence_golden.py` reads: each query's parts with the Spec fields the renderer reads and the sentence plans it recorded, gzipped to `tests/fixtures/sentence_panel/<slice>.jsonl.gz`. It prints each query whose envelope sentence changed against the golden it replaces.

```bash
uv run python scripts/sentence_golden.py runs/sentence-panel
```

Needs `GRAPH_RELEASE_URI` set to a published graph release, whose measure index it reads (`get_measure_index`), with Google credentials that can read a `gs://` release. It reads the `raw-*.jsonl` files a sentence panel run wrote.

Run by `just sentence-golden <run directory>`, by hand, after a change to the renderer's output. Record a panel run from the branch, trim it, and review the printed sentences.

## `sentence_panel/run.py`

Sends the sentence panel (182 queries in `scripts/sentence_panel/queries/{crs,mls,ids,weo}.yaml`) to the deployed Search resolver and digests the sentences it returns. `scripts/sentence_panel/rubric.md` is the rubric for scoring a run.

```bash
uv run python scripts/sentence_panel/run.py                       # whole panel, about five minutes
uv run python scripts/sentence_panel/run.py crs ids               # named slices
uv run python scripts/sentence_panel/run.py --redigest            # rebuild digests from saved raw files
```

Reads the live dev deployment (`--base` overrides the URL); no credentials needed. Writes `raw-<slice>.jsonl`, `digest-<slice>.jsonl` and `digest-<slice>.md` to `--out` (default `runs/sentence-panel`). Exits non-zero when a queries file is empty or every query in a slice fails at transport level.

Run by `just sentence-panel`. Runnable by hand.

## `stage_eval_data.py`

Stages the deployed graph release and origin catalogue for the eval apps. It copies the release the deployed pointer names, with its manifest and parts. For each checkout it asks `origin_catalogue.schema.FORMAT_ROOT` for the format root, reads each distinct root's current pointer from the deployed bucket, downloads that one version into a local root, and prints the `file://` URIs.

```bash
uv run python scripts/stage_eval_data.py --checkout <head-checkout> --checkout <base-checkout> \
    --dest staged --release-uri gs://<release bucket>/ --catalogue-uri gs://<catalogue bucket>/
```

Needs read access to the deployed release and catalogue buckets. Exits 3 when a root or the release is missing from the bucket.

Run by `.github/workflows/eval-gate.yml` before the apps start.

## `stage_origin_catalogue_fixture.py`

Links a committed origin catalogue build (`tests/fixtures/origin_catalogue/` or `tests/fixtures/origin_catalogue_eval/`) into a root directory under the name `origin_catalogue.schema.FORMAT_ROOT` carries, and prints the `ORIGIN_CATALOGUE_URI` that reads it.

```bash
export ORIGIN_CATALOGUE_URI="$(uv run python scripts/stage_origin_catalogue_fixture.py \
    tests/fixtures/origin_catalogue_eval /tmp/catalogue-eval-root)"
```

No credentials needed.

By hand, to serve or score against a committed build. `tests/origin_catalogue/test_fixture.py` and `record_origin_catalogue_fixture.py` call its `stage_fixture`.

## `survey_unresolved_arcs.py`

Surveys how often a dcid-less (`ProvisionalNode`) arc target actually occurs on `StatisticalVariable` nodes in the live graph.

```bash
uv run --env-file .env python scripts/survey_unresolved_arcs.py
uv run --env-file .env python scripts/survey_unresolved_arcs.py --skip-reverse
```

Needs `SEARCH_GRAPH_BASE` (defaults to dfo-dev); no `GEMINI_API_KEY` needed.

By hand, when auditing how representative the offline fixture snapshot's unresolved-arc rate is of the live graph.

## `warm_resolvekit_cache.py`

Converts every bundled and BYOD resolvekit pack from WAL journal mode to rollback-journal mode, so the read-only Cloud Run filesystem can open them at cold start.

```bash
python scripts/warm_resolvekit_cache.py
```

No live credentials needed.

Run by `Dockerfile` (`RUN /app/.venv/bin/python scripts/warm_resolvekit_cache.py`), during every image build.
