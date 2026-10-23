# Maintainer notes

Internal detail for the ONE Data maintainers of this repository: how the code is laid out, what CI runs, and how a release reaches production. Outside contributors need only [CONTRIBUTING.md](../CONTRIBUTING.md). Deployment wiring (Cloud Run, Workload Identity, migrations, scheduled jobs) is in [deploy/README.md](../deploy/README.md).

## Code layout

The repo is one Python service built into one container image. Each API is its own top-level package, and all of them mount through `packages/dfo-data-tools/src/dfo_platform/`: the RFC 9457 problem-details envelope, the common code registry, the per-request trace id, and the `Surface` and `create_app` composition root. `dfo_platform` depends on no API, so any API can depend on it without pulling in another. `src/dfo_apis/` composes every live surface onto the shared platform app, and its zero-argument `create_app` factory is what the container runs. The repo has no vendored code and no submodules. A sibling repository, `infra`, deploys the image to Cloud Run.

```text
dfo-apis/
├── packages/
│   └── dfo-data-tools/      # uv workspace distribution with shared libraries
│       └── src/
│           ├── assembler/         # Visual template assembly and spec-kit compilation
│           ├── dc_sdmx/           # Standalone read-only SDMX REST client
│           ├── dfo_platform/      # Shared HTTP contracts, RFC 9457 models, and Surface router
│           ├── materialise/       # Frame generator, observation fetcher, and file formatters
│           └── statvar_members/   # Statistical variable hierarchy and member resolution
└── src/
    ├── artifact_store/      # GCS and local file storage client with signed URL generation
    ├── dataset_store/       # PostgreSQL storage layer for datasets state and migrations
    ├── datasets_api/        # FastAPI surface for datasets minting, registration, and snapshots
    ├── dfo_apis/            # Root composition module and application factory (create_app)
    ├── graph_api/           # FastAPI surface for name resolution, node inspection and listing
    ├── presentation_api/    # FastAPI surface for presentation frames and component authoring
    ├── presentation_store/  # PostgreSQL storage layer for presentation components
    ├── search/              # Natural-language resolution pipeline, models, and eval harness
    ├── skills_api/          # FastAPI surface for the published skills catalogue and resources
    └── spec_registry/       # In-memory and persistent interfaces for dataset specifications
```

### Registering a new package

A package under `src/` needs its name added to the root `pyproject.toml`'s `[tool.uv.build-backend]` `module-name` list, or it does not reach the synced venv or the built image. An import that works locally under `uv run` from a source checkout can still be absent from the container. A package under `packages/*/` is a uv workspace member instead, with its own `pyproject.toml` and its own `module-name` list. Register it in the root `pyproject.toml`'s `[tool.uv.workspace]` members and add it to the root's own dependencies. A stale venv that predates a layout change fails on the next import, and `uv sync` (or `just setup`) fixes it.

### Adding an API surface

A surface's `prefix` must live under `/api/`. The platform owns every path outside that namespace for its own routes (`/codes/{code}` and FastAPI's generated documentation), and `create_app` refuses a prefix that does not. Give `Surface.codes` the API's own codes only, since a view that already includes `dfo_platform.codes.PLATFORM_CODE_CATALOGUE` makes `create_app` raise `CodeCollision` on every platform code at once. Where the API's own code needs the full registry (building its own `Envelope`, for instance), merge the platform's catalogue into the API's own view with `merge_catalogues` and keep that merged view out of `Surface.codes`. Once the surface is built, append it to the list `deployed_surfaces` returns in `src/dfo_apis/app.py`. That list is the one place that composes every live API onto the deployed service, and the platform conformance suite reads the same list, so the surface is held to the shared envelope with no further edit.

A surface's own `capability_version` is stamped only on its own responses, including routing-generated 404s and 405s under its own `prefix`, since those go through the same envelope. The platform stamps its own `PLATFORM_CAPABILITY_VERSION` on everything it owns: the `/codes/{code}` refusal for an unknown code, and any request matching no mounted surface's `prefix`. A successful `/codes/{code}` lookup returns a bare `CodeRegistryEntry` body with `code`, `classification`, `definition`, `lifecycle`, `mappings`, `title`, and `detail`, and has no `capability_version` field. It resolves codes present in either `dfo_spec_kit.registry.REGISTRY` or the merged refusal catalogue. Registry-derived fields may be null and `mappings` empty, and `title` and `detail` may be null when the merged catalogue has no entry. Only the refusal goes through the envelope. A surface never sets or reads that token itself.

## Local checks

```bash
just lint        # ruff check and format check
just format      # apply lint fixes and format
just typecheck   # ty
just test        # tests affected by your changes, offline against recorded fixtures
just test-all    # the whole suite, offline
just check       # lint, type check, and the full suite, as CI runs them
```

`just test` uses [pytest-testmon](https://testmon.org) to run only the tests that execute changed Python code. It runs the whole suite when a non-Python file in the repo or the `dfo-spec-kit` checkout has changed since the last run, since testmon does not see those. The first run takes about 50% longer than `just test-all` while it records coverage, and a run with nothing changed takes a few seconds.

### Identity vectors

`just check` and CI run the normative identity vector tests (`tests/engine/test_identity_vectors.py`) from the `spec-kit` repository. `just check` and `just test` find a clone at `../dfo-spec-kit`, or at the path in `DFO_SPEC_KIT_DIR`. A git worktree needs `DFO_SPEC_KIT_DIR` set explicitly, because the sibling fallback resolves relative to the worktree's own directory, not the main checkout's. `just check` fails when no clone resolves.

### Changing what the clients read

`tests/contract/test_client_contract.py` checks the app against the client contract spec-kit
ships as `dfo-spec-kit-client-contract`. Each recorded scenario's requests and responses must
validate against the app's OpenAPI document, every required member `strict-fields.json` lists
must stay required and non-nullable, every enum value it lists must stay admitted, and every
problem code a response carries must stay registered on its route at its status. When the test
fails for a wire change you intend, commit the change, record the contract against your branch's
local app with spec-kit's `tools/capture_client_contract.py`
(`openapi --dfo-apis <checkout>`, then `record --base-url http://127.0.0.1:8083/api`), open the
spec-kit PR, and cut a spec-kit release. Move all six spec-kit pins to that release's commit in
the same dfo-apis PR as the change: `dfo-spec-kit` and `dfo-spec-kit-client-contract` in the root
`pyproject.toml`, and the four in `packages/dfo-data-tools/pyproject.toml`. A change that keeps
the shape and alters behaviour passes this test, and the sdk's scheduled end-to-end run catches
it after merge.

### SDMX live tests

`tests/dc_sdmx/` runs offline against recorded wire fixtures. `test_live.py` calls a real Data Commons deployment and is skipped unless `DFO_SDMX_LIVE_BASE` is set.

```bash
export SEARCH_GRAPH_TOKEN=<the x-graph-token shared secret>
just sdmx-live
```

The recipe points at the same graph as `just dev` (`SEARCH_GRAPH_BASE`) and appends the Data Commons service's `/core/api` prefix, because `SdmxClient` appends its own `/sdmx/v3/...` paths to the API root. It sets `DFO_SDMX_LIVE_REQUIRED=1`, so a missing base fails the run instead of skipping it. Two tests pass when the deployment serves the SDMX routes and returns observations for `Count_Person` in `country/USA`. A failure on an empty result means the deployment authenticated but served an empty graph. An `SdmxUnexpectedResponseShapeError` means the request was challenged before it reached the mixer, so the token is missing or wrong.

`.github/workflows/sdmx-live.yml` runs the same suite on pull requests touching `packages/dfo-data-tools/src/dc_sdmx/` or `tests/dc_sdmx/` once they are out of draft, and on a weekday schedule.

### Database migrations

Apply pending forward migrations against local or Cloud SQL PostgreSQL databases (see [deploy/postgres.md](../deploy/postgres.md)):

```bash
just migrate                # dataset_store migrations (datasets database)
just migrate-presentation   # presentation_store migrations (presentation database)
just dataset-store-smoke    # smoke test dataset_store connectivity
```

## Evaluation harness

The Search API evaluation scores the engine's answers to a set of cases. Each case is a query, the door it enters by, and the answer it must get, written as one YAML entry. A check reads a running app's response for one aspect of that answer.

- `corpus/*.yaml` holds the cases, one file per source, each with the status the answer may return and what it must carry.
- `corpus/families/` holds the measure families the cases name.
- `src/search/eval/checks/` holds the checks, such as `disposition`, `measure`, `entities`, `structural`, and `groundedness`.
- `scripts/eval_paired.py` posts each case to `/api/search/resolve` over HTTP and scores the responses.

`corpus/README.md` describes the case format. Scoring needs a running app and live credentials (see `.env.example`).

```bash
just corpus-check                                  # validate cases, list those that cannot gate
just eval-app 8100                                 # serve an app with the eval profile (no response cache, no LLM memo)
just eval-run http://127.0.0.1:8100 --file crs     # score eligible cases; or --case <id>, repeatable
just eval-paired http://127.0.0.1:8100 http://127.0.0.1:8101 <base-sha>   # head against base
```

### The paired gate

A change under `src/search/` is scored by the eval gate. The gate runs every eligible case in `corpus/*.yaml` on two running apps, head (the pull request's merge commit) and base (its first parent), three samples per case, and compares them check by check. A check that passes on any head sample passes. A check that passes on no head sample blocks the merge when:

- the case is new or its expectation changed;
- the check passes on all three base samples and again in a confirm round (a regression);
- the check is `structural`.

A check that fails at base too is reported as pre-existing and does not block. A base sample that passes `disposition` or `answer` while the same sample fails `measure`, `entities` or `provenance` answered the right status on the wrong series or places. When every base sample that reached the check is such a pass, wrong on each part the head failed, a head that stops giving that answer is reported as `wrong_answer_at_base`. One base sample right in content keeps the verdict a regression. `wrong_answer_at_base` does not block, and the summary lists each one beside the base sample's content failure. The sample's own content verdicts decide, including a content check in the case's `known_failure`. A check named in a case's `known_failure` is reported and does not block, and the report marks it fixed when every head sample passes it. `known_failure` carries the issue that tracks the fault, so a case that exposes an engine fault stays in the corpus with its correct expectation. The summary lists every case the pull request deleted, made ineligible, or gave a new `known_failure`, so a reviewer sees each silenced check.

`.github/workflows/eval-gate.yml` evaluates pull requests into `dev` and `main` that touch `src/`, `packages/`, `corpus/`, the eval scripts, `uv.lock`, or `pyproject.toml`. It runs when a non-draft pull request is opened or reopened and when a draft is marked ready for review. A push does not run it. To evaluate a new head commit, or a draft, add the `run-eval` label: `.github/workflows/eval-request.yml` removes the label and dispatches the eval on the pull request's head branch, and adding the label again asks for another run. The workflow stages the deployed coverage and catalogue artifacts with `scripts/stage_eval_data.py`, serves both apps with the eval profile (`just eval-app`), runs `scripts/eval_paired.py`, uploads the report and every response as an artifact, and posts the summary. The driver exits 0 when nothing blocks, 1 on a blocking verdict, and 3 on infrastructure failure. The `eval-override` label turns a 1 or a 3 into a pass for one commit: add it and re-run the failed `Eval gate` job, and the next eval run removes it. The summary records the override. Making the job a required status check is a branch-protection setting, configured per branch in the repository settings.

The eval jobs need repository secrets, so they skip on pull requests from forks and on drafts.

A push to `dev` that touches `corpus/` syncs the cases to Langfuse, one dataset per case file. `.github/workflows/eval-nightly.yml` scores one sample per case against the deployed `dfo-dev` service and records a Langfuse run.

## CI workflows

GitHub Actions validate pull requests and pushes to `dev`. A draft pull request runs nothing until it is marked ready for review. A push to `main` runs no CI, since `main` only moves to a commit `dev` has already tested. The jobs that need repository secrets skip on pull requests from forks.

- `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, `ty check`, `pytest` with identity vectors, and verifies container builds.
- `.github/workflows/eval-gate.yml` scores the eval cases on the head and base builds of a pull request, split by case id across parallel shard jobs, and blocks on a regression, a failing new case, or a structural failure in any shard.
- `.github/workflows/sdmx-live.yml` validates `dc_sdmx` client operations against active Data Commons Cloud Run endpoints.
- `.github/workflows/migrations.yml` runs database migrations and live tests against PostgreSQL service containers.
- `.github/workflows/bundle-round-trip.yml` validates Frictionless data package export and R consumption.
- `.github/workflows/deploy-dev.yml` deploys to dev after each successful CI run on `dev`.
- `.github/workflows/deploy-prod.yml` deploys to production on a push to `main`.
- `.github/workflows/deploy.yml` holds the build, migrate, and deploy steps both callers share. See [deploy/README.md](../deploy/README.md).

## Container build

The application builds as a multi-stage Docker image with a uv-managed Python runtime.

```bash
just build   # linux/amd64 image from the justfile
just run     # run the container locally with .env configuration
```

Or build directly with Docker:

```bash
docker build -t dfo-apis .
docker run -p 8080:8080 -e GEMINI_API_KEY=<key> dfo-apis
```

## Branches and releases

Two branches map to two environments, and work flows up from `dev` to `main`.

- `dev` is the default branch. Feature branches merge here, and it deploys to `dfo-dev`.
- `main` is production. It is protected, tagged, and moved only by a pull request from `dev`. Every commit on `main` is deployable.

After a pull request merges, `dev` deploys to `dfo-dev`. Validate the change against the live graph there.

To release to production:

1. Confirm the `dev` commit you want is deployed and validated on `dfo-dev`.
1. Open a pull request from `dev` into `main`. CI runs on it, and one approval is required.
1. Merge it. The push to `main` runs `.github/workflows/deploy-prod.yml`, which deploys to the `production` environment through the same steps as dev. The environment holds the job until a member of the `release` team approves it.

## Building and deploying images by hand

```bash
just deploy                    # dev image:  dfo-dev-<sha>
just DEPLOY_ENV=prod deploy    # prod image: dfo-prod-<sha>
```

`deploy` builds an immutable `dfo-<tier>-<sha>` image. A prod cut refuses a dirty tree, so the image is always reproducible from a committed sha. `deploy` does not push, so the image stays local. `.github/workflows/deploy-dev.yml` runs after each `dev` CI run succeeds, then resolves `dev`'s current tip, checks that tip's own CI status when it differs from the triggering commit, and builds, pushes, and updates Cloud Run only when the tip's commit differs from the commit the service is already running. [deploy/README.md](../deploy/README.md) lists the variables and secrets it needs.
