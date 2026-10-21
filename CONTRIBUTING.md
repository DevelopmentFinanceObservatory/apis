# Contributing

How development works in dfo-apis. Read this before opening a branch.

## What this repo is

The ONE/DFO data APIs, as one Python service, deployed from one container image. Each API is
its own top-level package: `src/search/` is the Search API, a FastAPI app that resolves a
natural-language data request against a Data Commons graph. `src/datasets_api/` is the Datasets
API, a sibling package, not folded into `src/search/`; it registers canonical dataset
definitions durably against a Postgres-backed store (`src/dataset_store/`) and serves its own
`healthz`, `schema`, `capabilities`, and `operations/{operation_id}` routes. Both
share `packages/dfo-data-tools/src/dfo_platform/`: the RFC 9457 problem-details envelope, the
common code registry, the per-request trace id, and the `Surface`/`create_app` composition root
every API mounts through.
`dfo_platform` depends on neither API, so either can depend on it without pulling in the other.
`src/dfo_apis/` is the deployed service itself: the one place that composes every live API
surface onto the shared platform app. Its zero-argument `create_app` factory is what the
container runs. Every line here is ours. There is no upstream to track, no vendored code, and
no submodules.

The repo builds one container image. A sibling repo, `infra`, deploys it to Cloud Run.

## Getting set up

```sh
just setup    # env file, dependencies, pre-commit hooks
just check    # everything CI runs
just dev      # hot-reload server on :8083
```

`just` on its own lists every recipe.

`just check` also runs the identity vector conformance suite
(`tests/engine/test_identity_vectors.py`), which lives in the `spec-kit` repo
rather than this one. Clone it as a sibling of this checkout:

```sh
git clone https://github.com/DevelopmentFinanceObservatory/spec-kit.git ../dfo-spec-kit
```

`just check` and `just test` pick it up automatically from that path. Point
`DFO_SPEC_KIT_DIR` at a checkout elsewhere instead when the sibling layout
doesn't fit. A git worktree needs `DFO_SPEC_KIT_DIR` set explicitly: the
sibling fallback resolves relative to the worktree's own directory, not the
main checkout's.

## Branch model

Two branches, each mapped to one environment. Work flows up from `dev` to `main`.

- **`dev`.** Active development, and the default branch. Feature branches merge here.
  Deploys to `dfo-dev`.
- **`main`.** Production. Protected, tagged, and moved only by a pull request from
  `dev`. Every commit on `main` is deployable.

```
feat/* , api/*  ──►  dev  ──►  main
                 dfo-dev       dfo-prod
```

## Where your work goes

1. Branch off `dev`. Name it `api/<feature>` for work on an API surface, `feat/<feature>`
   otherwise.

1. Build the change. `just check` runs the same lint, type check, and tests CI does, so
   local green predicts CI green.

1. Open a PR into `dev`. CI has to pass before it merges.

1. On merge, deploy `dev` to `dfo-dev` and validate against the live graph.

A change under `src/search/` is scored by the eval gate. The gate runs every eligible case in
`corpus/*.yaml` on two running apps, head (the PR's merge commit) and base (its first parent),
three samples per case, and compares them check by check. A check that passes on any head
sample passes. A check that passes on no head sample blocks the merge when the case is new or
its expectation changed, when the check passes on all three base samples and again in a
confirm round (a regression), and when the check is `structural`. A check that fails at base
too is reported as pre-existing and does not block.

A check named in a case's `known_failure` is reported and does not block. The report marks it
fixed when every head sample passes it. `known_failure` carries the issue that tracks the
fault, so a case that exposes an engine fault stays in the corpus with its correct
expectation. The summary lists every case the PR deleted, made ineligible, or gave a new
`known_failure`, so a reviewer sees each silenced check.

The eval workflow (`.github/workflows/eval-gate.yml`) evaluates PRs into `dev` and `main` that
touch `src/`, `packages/`, `corpus/`, the eval scripts, `uv.lock` or `pyproject.toml`. It runs
when a non-draft PR is opened or reopened and when a draft is marked ready for review. A push
does not run it. To evaluate a new head commit, or a draft, add the `run-eval` label:
`.github/workflows/eval-request.yml` removes the label and dispatches the eval on the PR's head
branch, and adding the label again asks for another run. It stages the deployed coverage and
catalogue artifacts with `scripts/stage_eval_data.py`, serves both apps with the eval profile
(`just eval-app`), runs `scripts/eval_paired.py`, uploads the report and every response as an
artifact, and posts the summary. The driver exits 0 when nothing blocks, 1 on a blocking verdict, and 3 on
infrastructure failure. The `eval-override` label turns a 1 or a 3 into a pass for one commit:
add it and re-run the failed `Eval gate` job, and the next eval run removes it. The summary
records the override. Making the job a required status check is a branch-protection
setting, configured per branch in the repository settings.

A push to `dev` that touches `corpus/` syncs the cases to Langfuse, one dataset per case
file. `.github/workflows/eval-nightly.yml` scores one sample per case against the deployed
`dfo-dev` service and records a Langfuse run.

To score cases locally, serve an app and point the driver at it:

```sh
just eval-app 8100
just eval-run http://127.0.0.1:8100 --file crs      # or --case <id>, repeatable
just eval-paired http://127.0.0.1:8100 http://127.0.0.1:8101 <base-sha>
```

### Checking dc_sdmx against a live deployment

`tests/dc_sdmx/` runs offline against recorded wire fixtures. `test_live.py` is the exception:
it calls a real Data Commons deployment, and is skipped unless `DFO_SDMX_LIVE_BASE` is set.

```sh
export SEARCH_GRAPH_TOKEN=<the x-graph-token shared secret>
just sdmx-live
```

The recipe points at the same graph as `just dev` (`SEARCH_GRAPH_BASE`) and appends the Data
Commons service's `/core/api` prefix, because `SdmxClient` appends its own `/sdmx/v3/...` paths
to the API root. It sets `DFO_SDMX_LIVE_REQUIRED=1`, so a missing base fails the run instead of
skipping it. Two tests pass when the deployment serves the SDMX routes and returns observations
for `Count_Person` in `country/USA`. A failure on an empty result means the deployment
authenticated but served an empty graph; an `SdmxUnexpectedResponseShapeError` means the request
was challenged before it reached the mixer, so the token is missing or wrong.

`.github/workflows/sdmx-live.yml` runs the same suite on pull requests touching
`packages/dfo-data-tools/src/dc_sdmx/` or `tests/dc_sdmx/` once they are out of draft, and on a
weekday schedule.

## Releasing to production

Changes land on `dev`. Production runs from `main`.

1. Confirm the `dev` commit you want is deployed and validated on `dfo-dev`.
1. Open a pull request from `dev` into `main`. CI runs on it, and one approval is required.
1. Merging it is the release. The push to `main` runs `.github/workflows/deploy-prod.yml`,
   which deploys to the `production` environment through the same steps as dev. The
   environment holds the job until a member of the `release` team approves it.

## Building and deploying

```sh
just deploy                    # dev image:  dfo-dev-<sha>
just DEPLOY_ENV=prod deploy    # prod image: dfo-prod-<sha>
```

`deploy` builds an immutable `dfo-<tier>-<sha>` image. A prod cut refuses a dirty tree, so
the image is always reproducible from a committed sha. `deploy` never pushes, so the image
stays local. `.github/workflows/deploy-dev.yml` runs after each `dev` CI run succeeds, then
resolves `dev`'s current tip, checks that tip's own CI status when it differs from the
triggering commit, and builds, pushes, and updates Cloud Run only when the tip's commit
differs from the commit the service is already running. See `deploy/README.md` for the vars
and secrets it needs.

## Conventions

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org): `feat:`,
  `fix:`, `chore:`, and so on.
- **PRs target `dev`.** Only a pull request from `dev` writes to `main`.
- **Eval cases** live under `corpus/<source>.yaml`, one file per source, with the measure
  families they name in `corpus/families/`. See `corpus/README.md` for the case format, then run
  `just corpus-check` before you commit.
- **Delete merged branches.** Feature branches are short-lived.
- **Keep internal notes out of the repo.** Design docs, plans, and audits stay local. This
  guide and real reference docs are the exception.
- **A new top-level package must be registered.** A package under `src/` needs its name added
  to the root `pyproject.toml`'s `[tool.uv.build-backend]` `module-name` list, or it never
  reaches the synced venv or the built image. An import that works locally under `uv run` from
  a source checkout can still be silently absent from the container. A package under
  `packages/*/` is a uv workspace member instead, with its own `pyproject.toml` and its own
  `module-name` list; register it in the root `pyproject.toml`'s `[tool.uv.workspace]` members
  and add it to the root's own dependencies. A stale venv that predates a layout change like
  this fails loudly on the next import; `uv sync` (or `just setup`) fixes it.

### Adding an API surface

Your surface's `prefix` must live under `/api/`. The platform owns every path outside
that namespace for its own routes (`/codes/{code}` and FastAPI's generated documentation),
and `create_app` refuses a prefix that doesn't. Give `Surface.codes` your API's own codes
only, not a view that already includes
`dfo_platform.codes.PLATFORM_CODE_CATALOGUE`. If you do, `create_app` raises `CodeCollision` on
every platform code at once. Where your own code needs the full registry (building your own
`Envelope`, for instance), merge the platform's catalogue into your own view yourself with
`merge_catalogues`; don't fold it into what you hand to `Surface.codes`. Once your surface is
built, append it to the list `deployed_surfaces` returns in `src/dfo_apis/app.py`. That's the
one place that composes every live API onto the deployed service; the platform conformance
suite reads the same list, so your surface is held to the shared envelope with no further edit.

Your surface's own `capability_version` is stamped only on your surface's own responses,
including routing-generated 404s and 405s under your own `prefix`, since those go through the
same envelope. The platform stamps its own `PLATFORM_CAPABILITY_VERSION` on everything it owns
instead: the `/codes/{code}` refusal for an unknown code, and any request matching no mounted
surface's `prefix` at all. (A successful `/codes/{code}` lookup returns a bare
`CodeRegistryEntry` body with `code`, `classification`, `definition`, `lifecycle`, `mappings`,
`title`, and `detail`, with no `capability_version` field at all. It resolves codes present in
either `dfo_spec_kit.registry.REGISTRY` or the merged refusal catalogue. Registry-derived fields
may be null and `mappings` empty. `title` and `detail` may be null when the merged catalogue has
no entry. Only the refusal goes through the envelope.) You never set or read that token yourself.

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
