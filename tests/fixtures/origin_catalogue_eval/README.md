# Eval origin catalogue

A published origin catalogue build from the dev graph and the UN System Data Commons, for serving or scoring without GCP credentials. `tests/origin_catalogue/test_fixture.py` checks that it loads under the services' memory budget.

This build is not a single job run. Its tables come from the job run that published build `20261007T155855Z-cf4a80b3` in format 3. On 2026-10-11 it was rewritten as format 4: each DFO and base row's licence and each entry's card facts and documentation facts (`source_documentation.json`) were derived from the dev graph with the job's own `origin_catalogue.job._citation_facts`, reading only the build's 150 DFO and base rows, and every other table was re-encoded unchanged. The card facts came out identical to the recorded ones. The next full re-record below replaces it.

The build ages. Once its `built_at` is more than 14 days old, every answer served from it carries the `origin_catalogue_stale` warning, and admissions drift from the dev graph and the UN deployment as imports change. Re-record it when either matters, and after a change to `origin_catalogue.schema.CATALOGUE_FORMAT_VERSION`.

## Re-record

Run the catalogue job against a local directory from the repository root, with the job's environment set (`ORIGIN_CATALOGUE_BQ_CONNECTION`, `DATASETS_GRAPH_BASE_URL`, `UN_GRAPH_BASE_URL` to read a UN deployment other than the default, and `DATASETS_GRAPH_TOKEN` where the graph needs it; see `origin_catalogue.job`):

```sh
uv run python -m origin_catalogue.job --local-dir /tmp/catalogue-eval
```

The run takes about 17 minutes, 8 of them reading the UN deployment, and logs `origin_catalogue_job_published` when it has written a build under the format root, `origin_catalogue.schema.FORMAT_ROOT`. This directory holds that root's contents. Copy the published build here:

```sh
uv run python scripts/record_origin_catalogue_fixture.py --eval-from /tmp/catalogue-eval
```

The script loads the build with the services' loader and memory budget, then replaces `current.json` and `versions/` with the pointer and the build directory it names, leaving the job's lease and state files behind.

## Read it

The loader reads `<ORIGIN_CATALOGUE_URI>/<FORMAT_ROOT>/`. `scripts/stage_origin_catalogue_fixture.py` links this directory into a root under that name and prints the URI to set:

```sh
export ORIGIN_CATALOGUE_URI="$(uv run python scripts/stage_origin_catalogue_fixture.py tests/fixtures/origin_catalogue_eval /tmp/catalogue-eval-root)"
```
