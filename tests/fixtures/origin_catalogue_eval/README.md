# Eval origin catalogue

A published origin catalogue build from the dev graph. The eval job in `.github/workflows/eval-gate.yml` reads it, because that job has no GCP credentials to read the build the catalogue job publishes. `tests/origin_catalogue/test_fixture.py` checks that it loads.

The build ages. Once its `built_at` is more than 14 days old, every answer the eval scores carries the `origin_catalogue_stale` warning, and admissions drift from the dev graph as imports change. Re-record it when either matters.

## Re-record

Run the catalogue job against a local directory from the repository root, with the job's environment set (`ORIGIN_CATALOGUE_BQ_CONNECTION`, `DATASETS_GRAPH_BASE_URL`, `UN_GRAPH_BASE_URL` to read a UN deployment other than the default, and `DATASETS_GRAPH_TOKEN` where the graph needs it; see `origin_catalogue.job`):

```sh
uv run python -m origin_catalogue.job --local-dir /tmp/catalogue-eval
```

The run takes about 14 minutes and logs `origin_catalogue_job_published` when it has written a build under the format root, `origin_catalogue.schema.FORMAT_ROOT`. This directory holds that root's contents. Copy only `current.json` and the build directory it names, leaving out lease and state files:

```sh
src="/tmp/catalogue-eval/$(uv run python -c 'from origin_catalogue.schema import FORMAT_ROOT; print(FORMAT_ROOT)')"
build=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["build_id"])' "$src/current.json")
rm -rf tests/fixtures/origin_catalogue_eval/current.json tests/fixtures/origin_catalogue_eval/versions
mkdir tests/fixtures/origin_catalogue_eval/versions
cp "$src/current.json" tests/fixtures/origin_catalogue_eval/
cp -R "$src/versions/$build" tests/fixtures/origin_catalogue_eval/versions/
```

## Read it

The loader reads `<ORIGIN_CATALOGUE_URI>/<FORMAT_ROOT>/`. `scripts/stage_origin_catalogue_fixture.py` links this directory into a root under that name and prints the URI to set:

```sh
export ORIGIN_CATALOGUE_URI="$(uv run python scripts/stage_origin_catalogue_fixture.py tests/fixtures/origin_catalogue_eval /tmp/catalogue-eval-root)"
```
