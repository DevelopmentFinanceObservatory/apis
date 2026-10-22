# Contributing

Issues and pull requests are welcome. The ONE Data team maintains this repository. For a large change, open an issue to discuss it before you write it.

Please follow the [Code of Conduct](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/CODE_OF_CONDUCT.md). Report security problems through the repository's private vulnerability reporting (the "Report a vulnerability" button under the Security tab), as described in the [security policy](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/SECURITY.md). For questions, see [SUPPORT.md](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/SUPPORT.md).

## Set up

You need Python 3.12 through [uv](https://docs.astral.sh/uv/) and the [just](https://just.systems) command runner. The tests run offline against recorded fixtures, so no cloud access is needed.

```bash
just setup    # env file, dependencies, pre-commit hooks
just check    # lint, type check, and the full test suite, as CI runs them
```

`just` on its own lists every recipe. `just setup` asks for a Gemini key and looks for one in ONE's Secret Manager. Press enter to skip it, since the tests do not use it.

`just check` also runs the identity vector suite (`tests/engine/test_identity_vectors.py`), which lives in the `spec-kit` repository. Clone it beside this checkout:

```bash
git clone https://github.com/DevelopmentFinanceObservatory/spec-kit.git ../dfo-spec-kit
```

`just check` and `just test` find it at that path. Set `DFO_SPEC_KIT_DIR` to use a checkout elsewhere. A git worktree needs `DFO_SPEC_KIT_DIR` set, because the sibling path resolves relative to the worktree.

Running the server (`just dev`) needs Gemini and ONE's Data Commons graph, so it is not part of the contributor workflow.

## Before you open a pull request

1. Branch off `dev`, the default branch. Name the branch `api/<feature>` for work on an API surface and `feat/<feature>` otherwise.
1. Run `just check`. It runs the same lint, type check, and tests CI does, so a green run locally predicts a green run in CI.
1. Run `just format` to apply lint fixes and formatting. The pre-commit hooks installed by `just setup` also check Markdown, spelling, YAML, and workflow files.
1. Open the pull request into `dev`.

A pull request needs passing CI, tests for new behaviour, and updated docs where behaviour or configuration changed. Keep a pull request to one change.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `chore:`, and so on.

## Adding code

- A new top-level package under `src/` must be registered in the root `pyproject.toml`, or it does not reach the built image. [docs/development.md](docs/development.md) has the steps.
- A new API surface mounts under `/api/` and joins the list in `src/dfo_apis/app.py`. [docs/development.md](docs/development.md) describes the rules.
- Eval cases live in `corpus/<source>.yaml`, one file per source, with the measure families they name in `corpus/families/`. `corpus/README.md` describes the case format. Run `just corpus-check` before you commit.

A change under `src/search/` is also scored by an eval gate, which compares the answers of the pull request's build with its base on the cases in `corpus/`. The gate needs repository secrets, so it does not run on pull requests from forks.

## Maintainer notes

[docs/development.md](docs/development.md) covers the package layout, the eval gate and its labels, CI workflows, releases to production, and deployment. [deploy/README.md](deploy/README.md) covers Cloud Run, migrations, and scheduled jobs.
