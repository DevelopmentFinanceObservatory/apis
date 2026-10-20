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

Scoring reads the graph, SDMX and the coverage index, so the environment needs the graph settings the apps use and `COVERAGE_ARTIFACT_URI`. Exit code 0 means nothing blocks, 1 a blocking verdict, 2 a setup error, 3 an infrastructure failure.

Run by `just eval-paired` and `just eval-run`, by `.github/workflows/eval-gate.yml` on pull requests, and by `.github/workflows/eval-nightly.yml`.

## `eval_role_direction.py`

Scores a flow-direction scanner against the frozen cases in `corpus/direction/role_direction_cases.yaml`.

```bash
uv run python scripts/eval_role_direction.py \
    --scanner corpus.direction.role_text_pinned:detect_directions
```

No live credentials needed.

By hand, when comparing a candidate flow-direction scanner against the frozen baseline scanner (`corpus/direction/role_text_pinned.py`).

## `export_code_index.py`

Regenerates the committed `packages/dfo-data-tools/src/code_index/data/index.json.gz` from the live graph and the published series coverage build.

```bash
COVERAGE_ARTIFACT_URI=gs://<coverage bucket>/ uv run --env-file .env \
    python scripts/export_code_index.py \
    --out packages/dfo-data-tools/src/code_index/data/index.json.gz
```

Needs `SEARCH_GRAPH_BASE` (and `SEARCH_GRAPH_TOKEN` on a token-gated deployment), and `COVERAGE_ARTIFACT_URI`, the root the coverage job publishes to (see `deploy/coverage.md`), with Google credentials that can read a `gs://` bucket.

The graph supplies the code rows and StatVar rows. The coverage build supplies each StatVar's `provenances`: `coverage.parquet` maps a variable to its facet ids and `facets.json` maps a facet id to its provenance. The stamp records the build id (`coverage_build`) and the number of StatVars the build holds no series for (`statvars_without_coverage`); such a row carries no provenance and belongs to no dataset.

`LISTED_DATASETS` names the datasets whose StatVars declare no code system, World Bank IDS among them, so no code read reaches them. The coverage build lists their StatVars, which the export reads into a separate `listed_statvars` block with their provenances. No family, dataset election, measure axis or shape reads that block: Search and the Datasets mint elect the members of a family holding a listed row from the index's rows of its five-tuple, in place of a graph walk (`code_index.members.listed_group`). The export therefore reads each five-tuple a listed row has, unsplit, and refuses when the graph holds a StatVar there that is neither an exported nor a listed row; the largest, `FinancialFlow` amount, holds about 14,000 StatVars and reads in about 45 s. When a build assigns a listed dataset a StatVar the index does not list, `python -m series_coverage.verify_index` fails naming them, and a running service keeps the build it serves (`series_coverage.compatibility.index_coverage_differences`).

`CODE_SYSTEMS` in `code_index/declarations.py` supplies each code system's declaration, which the export writes into the index's `declarations` block and `code_index.loader` reads into `CodeSystemDeclaration`:

| Field                    | Meaning                                                                                                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hierarchical`           | Codes form a `specializationOf` tree. Derived from the graph and checked against `expected_hierarchical`.                                                                       |
| `disjoint_from_children` | A record coded to a parent is a different record from one coded to its child.                                                                                                   |
| `summable`               | Sibling values add up to their parent's total.                                                                                                                                  |
| `absent_reading`         | How a StatVar that does not carry the property reads: `not_declared` (a separate population) or `aggregate` (a total over every value).                                         |
| `plain_name`             | The singular noun a sentence uses for the axis.                                                                                                                                 |
| `default`                | A code bound onto an open slot when another slot or a query phrase triggers it, or on every open slot (`when_open`), with the `note` an answer states when it fires; or `null`. |

`declared_by_every_statvar_of` lives in `code_index/declarations.py` and lists the datasets whose every StatVar declares the property, and the export enumerates each dataset only through the systems that list it.

`DATASETS` supplies one declaration per published table, written into the `datasets` block and read into `DatasetDeclaration`:

| Field                | Meaning                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `provenance`         | The provenance the table's observations carry. A StatVar row belongs to the dataset when its `provenances` names it.                                   |
| `label`              | The name a sentence uses for the dataset.                                                                                                              |
| `opt_in`             | Whether a question answers from the table only when it names the table.                                                                                |
| `absent_observation` | Whether an observation the table holds no row for reads as `zero`, or `unknown`.                                                                       |
| `account`            | Whose account of a flow the table gives, `provider`. It names the two parties of a series and says which one leads a heading.                          |
| `rank`               | The serving order among datasets that publish one five-tuple: of those that can serve a family, the lowest rank serves it. Unique across declarations. |

ONE Imputed Multilateral Sectors is opt-in, and CRS, OECD DAC1, OECD DAC2a and OECD DAC2b are not. CRS and ONE Imputed Multilateral Sectors read an absent observation as zero, and OECD DAC1, OECD DAC2a and OECD DAC2b read it as unknown. All five give the provider's account. CRS has rank 0, ONE Imputed Multilateral Sectors rank 10, OECD DAC1 rank 20, OECD DAC2a rank 30 and OECD DAC2b rank 40. DAC1, DAC2a and DAC2b share the CRS five-tuples under their own codes, so each forms families of its own and its rank orders it against no other table. ONE Imputed Multilateral Sectors and OECD DAC2a's code 106 share the `attributedAmount` five-tuple under different key sets: the imputed rows hold a sector and a flow type beside the aid type, and DAC2a's row holds neither. A question's keys therefore decide which table answers.

The export and the loader refuse a document in which a StatVar carries a provenance no dataset declares, a declared dataset has no StatVar, or a dataset's `rank` is not an integer or repeats another's. The second case means a stale or misjoined coverage build. A StatVar may carry several provenances, and several datasets may publish one five-tuple. `code_index.members.elect_dataset` serves each family from the dataset the identity names. Without one, a dataset whose rows cover the identity's role keys serves before one whose rows do not, then one whose family declares exactly the roles those keys place, then the lowest rank.

Two gates check the committed export, each against the source it can read:

| Gate                                                                                                    | Runs                                                                                 | Checks                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `code-index-drift` in `.github/workflows/code-index-live.yml` (`tests/code_index/test_export_drift.py`) | Weekday schedule, plus ready PRs touching `code_index` paths; graph credentials only | Everything the graph and the authored config determine: declarations, datasets, codes, StatVar rows without `provenances`, measure blocks, the stamp's counts, and its content hash.                                                                                            |
| `python -m series_coverage.verify_index --record` in `.github/workflows/deploy-dev.yml`                 | Every deploy, on the coverage job, before traffic moves                              | Each StatVar row's `provenances` equals the published coverage build's for that variable, and every declared dataset has a row in the build. It prints the index's `coverage_build` beside the build it read, and on a pass records that build as the index's compatible build. |

Run the export by hand when either gate reports a difference. The coverage check also runs locally: `COVERAGE_ARTIFACT_URI=gs://<coverage bucket>/ uv run python -m series_coverage.verify_index`, with `--index <path>` for an export not yet committed.

## `export_graph_entities.py`

Regenerates `entity_resolution`'s committed graph-derived pack data: every `Organization`, `DACRecipientUnallocatedByCountry` and `DACUnspecifiedParty` node (`graph_entities.json`) and the resolvekit region-to-dcid mapping (`region_ids.json`). A record's `dac_code` is the node's numerically lowest `dacCode`; its other codes join its aliases, since a resolvekit entity holds one code per system and every code must resolve.

```bash
uv run --env-file .env python scripts/export_graph_entities.py entities \
    --out packages/dfo-data-tools/src/entity_resolution/data/graph_entities.json
uv run --env-file .env python scripts/export_graph_entities.py regions \
    --out packages/dfo-data-tools/src/entity_resolution/data/region_ids.json
```

Needs `SEARCH_GRAPH_BASE`.

By hand, when the graph's organisation, unallocated-recipient or unspecified-party population or the resolvekit region mapping changes.

## `generate_sample_bundle.py`

Generates a small, deterministic Data Package v2 bundle directory for `bundle_round_trip.R` to validate.

```bash
uv run python scripts/generate_sample_bundle.py <output-directory>
```

No live credentials needed.

Run by `.github/workflows/bundle-round-trip.yml`, same trigger as `bundle_round_trip.R`.

## `graph_release_parity.py`

Diffs what an environment's graph release serves with what the old path serves for the same import, the switch gate for `GRAPH_RELEASE_COVERAGE`. It builds the current release's views with `graph_release.views`, then compares its `CoverageIndex` with the published coverage build (every row, role names, variable provenances, facet metadata, horizon and build year) and its `code_index_blocks` document with a fresh `export_index` over the live mixer, joined with that coverage build, every block row by row.

```bash
uv run python scripts/graph_release_parity.py \
    --release-uri gs://dfo-dev-graph-release-dev-fin-observatory/ \
    --coverage-uri gs://dfo-dev-coverage-dev-fin-observatory/ \
    --graph-base https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app \
    --datacommons-service projects/dev-fin-observatory/locations/northamerica-northeast1/services/dfo-dev-dc-datacommons-service \
    --out parity-dev.json
```

It refuses with exit 2 and a `refused.code` when the coverage build and the release come from different ingestion executions (`import_mismatch`, `coverage_ingestion_unrecorded`), when the checkout's declarations hash differs from the release's (`declarations_moved`), when the DC service serves another revision than the release read (`serving_revision_moved`), or when the export refuses (`export_refused`). Otherwise the JSON report counts, per category, the items compared and the differences, with the first `--examples` of them; the exit status is 0 when nothing differs and 1 when anything does. On dev it takes about 15 minutes, nearly all of them the export's mixer reads.

Needs Google credentials that read both buckets and `run.services.get` on the DC service, and `SEARCH_GRAPH_TOKEN` on a token-gated mixer. Run by hand on each environment before its `GRAPH_RELEASE_COVERAGE` flag is switched on.

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

## `measure_corpus_latency.py`

Runs every case whose `why` states a timing claim through the live engine `--runs` times each and reports which ones exceed the claimed deadline.

```bash
COVERAGE_ARTIFACT_URI=file://$PWD/tests/fixtures/coverage_slice \
    uv run --env-file .env --extra engine --extra eval python scripts/measure_corpus_latency.py
```

Needs `GEMINI_API_KEY` and `COVERAGE_ARTIFACT_URI` (the engine loads coverage at construction time, and this script runs outside pytest, so there is no conftest to default it). No Langfuse credentials.

By hand, when re-checking the corpus's timing claims against a live deployment.

## `measure_coverage_lookup.py`

Measures `series_coverage.CoverageIndex` lookup wall-clock (p50/p95/p99) and peak RSS per query mode against a real published coverage artifact.

```bash
uv run python scripts/measure_coverage_lookup.py
uv run python scripts/measure_coverage_lookup.py --artifact-uri file:///tmp/coverage-live
```

No live credentials needed; reads a coverage artifact by URI (`file://` or `gs://`).

By hand, when checking coverage-index lookup performance against a real artifact.

## `measure_writer_rss.py`

Measures `materialise.formats`' three streaming writers' peak RSS against synthetic CRS-shaped frames.

```bash
uv run python scripts/measure_writer_rss.py
uv run python scripts/measure_writer_rss.py --out /tmp/writer-rss.json
```

No live credentials needed.

By hand. `docs/datasets-sync-ceiling.md` cites this script as the method behind its writer-memory numbers; re-run it when the streaming writers change.

## `record_coverage_slice.py`

Records `tests/fixtures/coverage_slice/`: a real, unmodified slice of a published coverage artifact, filtered to every role-bearing StatVar the offline suite's fixtures and the `corpus/*.yaml` cases and families reference.

```bash
uv run python scripts/record_coverage_slice.py \
    --source file:///path/to/real/coverage/artifact \
    --out tests/fixtures/coverage_slice
```

No live credentials needed; reads a coverage artifact by URI.

By hand, when the fixture needs refreshing after the corpus or the source coverage artifact changes.

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

No live credentials are needed. It reads the `raw-*.jsonl` files a sentence panel run wrote.

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

Stages the deployed coverage and origin catalogue artifacts for the eval apps. For each checkout it asks `series_coverage.schema.FORMAT_ROOT` and `origin_catalogue.schema.FORMAT_ROOT` for the format root, reads each distinct root's current pointer from the deployed bucket, downloads that one version into a local root, and prints the `file://` URIs.

```bash
uv run python scripts/stage_eval_data.py --checkout <head-checkout> --checkout <base-checkout>
```

Needs read access to the deployed coverage and catalogue buckets. Exits 3 when a root is missing from the bucket.

Run by `.github/workflows/eval-gate.yml` and `.github/workflows/eval-nightly.yml` before the apps start.

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
