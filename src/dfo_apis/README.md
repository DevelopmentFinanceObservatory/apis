# Composition (`dfo_apis`)

`dfo_apis` provides the root application factory and composition entrypoint for the service. It mounts Search, Datasets, Graph, Presentation, and Skills surfaces onto the shared `dfo_platform` FastAPI application.

---

## Application Factory

The service exposes a zero-argument FastAPI factory at `dfo_apis.app:create_app`. Container entrypoints, dev runners, and test harnesses invoke this entrypoint:

```bash
uvicorn --factory dfo_apis.app:create_app --host 0.0.0.0 --port 8083
```

`create_app()`:
- Initialises shared platform middleware (proxy headers, trace ID generation and propagation via `x-dfo-trace-id`, and CORS origin handling).
- Manages the shared `SaturationTrackingExecutor` thread pool.
- Overrides OpenAPI generation to document RFC 9457 problem detail error envelopes (`application/problem+json`).
- Registers a merged `/codes/{code}` registry route across all mounted surface catalogues.
- Mounts active surfaces based on runtime configuration.

---

## Surface Composition

Surfaces are managed through two functions in `dfo_apis.app`:

- `deployed_surfaces()`: Returns surfaces enabled for the current runtime environment. Search mounts unconditionally. Datasets, Graph, Presentation, and Skills mount when their respective configuration flags evaluate to `1` or `true`.
- `all_surfaces()`: Returns every supported surface regardless of runtime configuration. Platform conformance test suites use this function to verify contract compliance across all surfaces.

Gated surface packages are imported lazily inside activation branches. Importing `dfo_apis.app` runs without pulling in database drivers or downstream optional dependencies when those surfaces remain disabled.

---

## Mounted Surfaces

| Surface | Mount Path | Gate Variable | Dependencies and Prerequisites |
| :--- | :--- | :--- | :--- |
| **Search** | `/api/search` | _Always active_ | Gemini API key (`GEMINI_API_KEY`), Data Commons graph client, SDMX client, `materialise` pipeline |
| **Datasets** | `/api/datasets` | `DFO_DATASETS_SURFACE` | PostgreSQL (`dataset_store`), `DATASET_STORE_RUNTIME_DB_USER`, GCS storage (`ARTIFACT_STORE_BUCKET`) |
| **Graph** | `/api/graph` | `DFO_GRAPH_SURFACE` | `resolvekit`, Data Commons Platform endpoint (`GRAPH_DCP_BASE_URL`) |
| **Presentation** | `/api/presentation` | `DFO_PRESENTATION_SURFACE` | PostgreSQL (`presentation_store`, `dataset_store`), `PRESENTATION_STORE_RUNTIME_DB_USER`, `DATASET_STORE_RUNTIME_DB_USER`, catalogue release |
| **Skills** | `/api/skills` | `DFO_SKILLS_SURFACE` | None of its own; acquires the process-wide shared skills index over Search's own graph client |

---

## Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DFO_DATASETS_SURFACE` | `false` | Set to `1` or `true` to mount `/api/datasets`. |
| `DFO_GRAPH_SURFACE` | `false` | Set to `1` or `true` to mount `/api/graph`. |
| `DFO_PRESENTATION_SURFACE` | `false` | Set to `1` or `true` to mount `/api/presentation`. |
| `DFO_SKILLS_SURFACE` | `false` | Set to `1` or `true` to mount `/api/skills`. |
| `DFO_CORS_ORIGINS` | `""` | Comma-separated list of allowed CORS origins applied to all mounted surfaces. Dev leaves it empty; an empty value disables CORS headers, and wildcard origins are never used. See the `infra` repository's `infra/dfo-dev/apis/README.md` for dev deployment configuration. |
| `DFO_EXECUTOR_MAX_WORKERS` | `32` | Worker thread capacity for the platform-wide asynchronous thread executor
