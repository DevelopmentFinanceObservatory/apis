# DFO data APIs

HTTP APIs that turn a plain-English question about development finance into a precise data definition, build a versioned dataset from it, and serve the result as CSV, Parquet, or Excel files. The service is a Python FastAPI application that runs as one container image.

It is part of the [Development Finance Observatory](https://github.com/DevelopmentFinanceObservatory) (DFO), an initiative of [ONE Data](https://data.one.org) built on [Google Data Commons](https://datacommons.org). The repository map lives on the organisation profile.

This repository is under active development. Interfaces, schemas, and data may change without notice.

## Use the API

The API is served from the DFO website, with every route under `/api/v1`, and needs no API key.

```bash
curl -fsS https://developmentfinance.org/api/v1/search/healthz
```

The [API guide](docs/guide/quickstart.md) walks through a full request: resolve a question, create a dataset, and download the files. [Concepts](docs/guide/concepts.md) explains specs, versions, and the confirmation sentence, and [API conventions](docs/guide/conventions.md) covers URLs, versioning, and errors.

## What's inside

The service mounts these API surfaces on one shared HTTP platform:

- Search API (`/api/search`) resolves a natural-language request with Gemini against a Data Commons graph and returns a structured data specification.
- Datasets API (`/api/datasets`) creates dataset definitions, materialises observation tables to files, and tracks versions.
- Graph API (`/api/graph`) resolves names to graph nodes and describes what the graph holds.
- Presentation API (`/api/presentation`) assembles dataset frames into component specifications for rendering.
- Skills API (`/api/skills`) serves the published skills catalogue.

The code lives in `src/`, one package per surface plus the storage and composition packages, and in `packages/dfo-data-tools`, a workspace package with the SDMX client, the materialisation pipeline, and the shared HTTP platform. Routes, response headers, and configuration variables are in the [reference](docs/reference.md). The package layout is in the [maintainer notes](docs/development.md).

## Getting started

The test suite runs offline against recorded fixtures and needs no cloud access. You need Python 3.12 through [uv](https://docs.astral.sh/uv/), the [just](https://just.systems) command runner, and a clone of [spec-kit](https://github.com/DevelopmentFinanceObservatory/spec-kit) beside this checkout (`../dfo-spec-kit`).

```bash
just setup    # env file, dependencies, pre-commit hooks
just check    # lint, type check, and the full test suite
```

Running the server needs more. The Search API calls Gemini and reads a Data Commons graph and two published artifacts (the coverage index and the origin catalogue) that live in ONE's Google Cloud project (`dev-fin-observatory`), so `just dev` works only with access to them. Without access, read the code, run the tests, and use the public API above.

With access, set `GEMINI_API_KEY` in `.env`, then start the server with hot reload on port 8083 and check it:

```bash
just dev
curl http://localhost:8083/api/search/healthz
```

## Documentation

- [API guide](docs/guide/quickstart.md): quickstart, concepts, and how-tos for the public API.
- [Reference](docs/reference.md): endpoints, platform headers, and configuration.
- [Maintainer notes](docs/development.md): package layout, evaluation harness, CI, containers, and releases.
- [Deployment](deploy/README.md): Cloud Run, migrations, and scheduled jobs.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. See [LICENSE](LICENSE).
