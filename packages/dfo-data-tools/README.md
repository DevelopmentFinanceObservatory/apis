# dfo-data-tools

`dfo-data-tools` provides dataset materialisation pipelines, Data Commons SDMX integration, Statistical Variable member enumeration, presentation assembly primitives, and shared platform wire structures.

Application services, storage persistence adapters, and HTTP route handlers reside in the repository root packages. `dfo-data-tools` contains domain logic, protocol clients, and wire models.

## Modules

| Module              | Responsibility                                                                                                                                                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `materialise`       | Builds `dfo_frame_grammar.Frame` objects from SDMX observations and graph metadata, mints content versions, reshapes frame layouts, computes summary statistics, and exports distribution formats.                                                                                              |
| `dc_sdmx`           | Typed, read-only client for Data Commons mixer SDMX REST endpoints (`/data`, `/availability`) with request builders and error hierarchies.                                                                                                                                                      |
| `dc_graph`          | Client for the Data Commons mixer's `/v2/node`, `/v2/observation`, and `/v2/resolve` endpoints (`GraphClient`), shared by `materialise`, `graph_api`, `datasets_api`, `presentation_api`, and `search.engine`. `GraphClient` satisfies `materialise.graph.GraphReader` directly.                |
| `dfo_platform`      | Shared wire protocol definitions including RFC 9457 problem details, error code catalogues, request trace propagation, and validation diagnostics. Optional ASGI application bootstrapping resides in `dfo_platform.app`.                                                                       |
| `statvar_members`   | Deterministic Statistical Variable member enumeration from structural 5-tuples and slot bindings, resolving minimal-constraint antichains against a graph reader interface.                                                                                                                     |
| `assembler`         | Visual presentation assembler that resolves component intents against frame schemas, validates catalogue constraints, performs data binding, and mints stable series identifiers.                                                                                                               |
| `code_index`        | In-memory catalogue of classification codes and the StatVars they constrain, loaded from a committed JSON export (`load_index`); consumed in production by `materialise.reshape.decode_columns`.                                                                                                |
| `graph_generation`  | Process-wide graph generation tracking shared by `search`, `graph_api`, and `entity_resolution`: reads the Data Commons Cloud Run service's serving revision and clears every registered cache on a token change.                                                                               |
| `entity_resolution` | Resolves free text to a graph-confirmed dcid and kind (`EntityResolver`): resolvekit over the bundled place packs and a pack built from the deployment's own graph, with a description-arc fallback. A membership group (G7, the EU, DAC) resolves to `Group` with its graph-confirmed members. |
| `series_coverage`   | The coverage artifact: its schema, Parquet I/O, the in-memory `CoverageIndex`, and the process-wide loader (`load_coverage`, `get_coverage_service`) that Search and Datasets surface construction load at startup.                                                                             |
| `skills_index`      | Index of `AgentSkill` nodes: `build_index` reads every skill from a `SkillsGraph`, expands its attachments down the graph hierarchy, and inverts the result into a node-to-skills map that `match_dcids` answers matches against.                                                               |

## Graph Generation

Every graph-derived cache in this repo keys its entries on a `graph_generation.GenerationStamp`, captured once before computing an entry. The stamp's `key` is the Cloud Run service's current serving revision (its `trafficStatuses`, joined with `+` when more than one carries live traffic), or the literal string `"unknown"` when that revision cannot be read. The stamp's `max_age_s` sets that entry's lifetime: `GRAPH_CACHE_MAX_AGE_SECONDS` (default and maximum 86400, 24h, the lifetime of a UN read) while the generation is known, `GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS` (default 3600, 1h) while it is unknown. A token change clears every cache registered against the shared `GraphGeneration` provider exactly once. An unreadable token degrades to the short-lived unknown generation, and the request is still served, under a tighter cache lifetime, when the Admin API is unreachable or misconfigured.

`DFO_GRAPH_GENERATION_SERVICE` names the Cloud Run service (`projects/<p>/locations/<r>/services/<s>`) read through the Cloud Run Admin API v2. `DFO_GRAPH_GENERATION` overrides it with a fixed token and makes no HTTP call at all, for a local graph mirror with no Cloud Run service of its own. Neither set makes every cache run under the unknown generation for the process's whole lifetime, logged once as a startup warning.

Every cache constructor defaults to `graph_generation.UNKNOWN_GENERATION`, so a caller that passes no `generation` behaves as "unknown, 1h". `graph_generation.FakeGeneration` is the public test double: `FakeGeneration(token="gen-a")` starts known, `advance(new_token)` fires every registered cache's `clear()` exactly the way the real provider does, and `set_unknown()` degrades the display without touching what a later `advance()` compares against.

## Frame Materialisation

`materialise.pipeline` fetches SDMX-CSV responses through `dc_sdmx.SdmxClient.stream_data_batches`, which parses the wire body with `pyarrow`'s CSV reader into `pyarrow.RecordBatch`es under one observation table contract (`dc_sdmx.results.observation_schema`). Every column is a `string`. The wire's `NotApplicable` sentinel maps to null for dimension and role columns, an empty cell maps to null for role and attribute columns, and an empty `OBS_VALUE` stays `""`.

Every filter between fetch and frame build runs as a columnar operation over those batches: window bounds, facet selection, origin admission at each row's place, and observed-value validity (`materialise.columnar`), and duplicate-facet collapse (`materialise.facet_collapse.collapse_duplicate_identity_facets_table`). `materialise.frame_build.build_frame_from_table` builds the `dfo_frame_grammar.Frame` once from the filtered table.

## Core vs. Optional Extras

The base install carries `pydantic`, `pydantic-settings`, `dfo-spec-kit-frame-grammar[arrow]` (which brings `pyarrow`), `dfo-spec-kit-components`, `cachetools`, `jsonschema`, and `rfc8785`. With no extras, `assembler`, `dfo_platform` (except `dfo_platform.app` and `dfo_platform.version`), `statvar_members`, and `code_index` import. Every other module needs an extra:

| Extra        | Dependencies                                                                                      | Enables                                                                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sdmx`       | `httpx>=0.28.1`, `pyarrow>=18`                                                                    | `dc_sdmx`, `dc_graph`, `materialise` (every submodule except `materialise.formats`, since the package imports `dc_graph` on load), and `skills_index`.                                                                                                               |
| `formats`    | `dfo-data-tools[sdmx]`, `openpyxl>=3.1.5`                                                         | `materialise.formats` (`write_csv`, `write_excel`, `write_parquet`, `write_frame1`). The module imports `openpyxl` on load, so every format writer needs this extra.                                                                                                 |
| `surface`    | `fastapi>=0.138`, `starlette`, `uvicorn`                                                          | `dfo_platform.app` (`Surface`, `create_app`) and `dfo_platform.version`.                                                                                                                                                                                             |
| `generation` | `httpx>=0.28.1`, `google-auth[requests]>=2.55`                                                    | `graph_generation`, including `graph_generation.CloudRunGenerationSource`, which reads the Cloud Run Admin API with Application Default Credentials.                                                                                                                 |
| `resolve`    | `resolvekit[parsing]==0.1.12`, `dfo-data-tools[generation]`                                       | `entity_resolution`.                                                                                                                                                                                                                                                 |
| `coverage`   | `numpy>=2.0`, `google-cloud-storage>=2.18`, `google-cloud-bigquery>=3.27`, `dfo-data-tools[sdmx]` | `series_coverage`: the in-memory index (`series_coverage.index`) needs `numpy`, `gs://` artifact reads (`artifact_publication.uri`) need `google-cloud-storage`, and the building jobs (`series_coverage.job`, `origin_catalogue.job`) need `google-cloud-bigquery`. |

```bash
# SDMX reads, graph reads, and frame materialisation
uv add "dfo-data-tools[sdmx]"

# The same, plus CSV, Excel, Parquet, and frame1 export
uv add "dfo-data-tools[formats]"

# FastAPI application surface
uv add "dfo-data-tools[surface]"

# Every extra
uv add "dfo-data-tools[sdmx,formats,surface,generation,resolve,coverage]"
```

## Import Boundaries

Modules separate network-dependent operations from offline data transformations. Consumers select entry points according to execution constraints.

### Network-Touching Operations

These operations require network connectivity or active client sessions to upstream Data Commons instances. Each names the extra it needs:

- **SDMX Ingestion** (`sdmx`): `dc_sdmx.SdmxClient` performs HTTP queries against Data Commons mixer endpoints.
- **Pipeline Execution** (`sdmx`): `materialise.pipeline` functions (`materialise_spec`, `materialise_preview`, `materialise_group`) orchestrate SDMX queries, entity graph resolution, and revision verification.
- **Live Graph Resolution** (`sdmx`): `dc_graph.GraphClient` queries remote graph endpoints, and `statvar_members.cache.MemberEnumerator` does the same when given a `GraphClient` as its reader. `statvar_members` itself imports without extras.
- **Graph Generation Read** (`generation`): `graph_generation.CloudRunGenerationSource` reads the Data Commons Cloud Run service's own serving revision through the Cloud Run Admin API v2, on a request-triggered background thread.

## Cache Freshness Against a Graph Change

`entity_resolution.EntityResolver` and `entity_resolution.Admission` (the `resolve` extra) are two of the caches keyed on a `graph_generation.GenerationStamp`. Each takes a `generation: graph_generation.GenerationProvider = graph_generation.UNKNOWN_GENERATION` keyword, captures a `GenerationStamp` once per `resolve()`/`admits()` call, and keys its own cache entries on that stamp: `EntityResolver` for confirmed resolutions, `Admission` for a role property's `rangeIncludes` set and a class's `subClassOf` parents. An uninjected instance keys and expires every entry under the "unknown" stamp.

The example below materialises one frame through `dc_sdmx`, `dc_graph`, and `materialise`, which all need the `sdmx` extra.

```python
from dc_graph import GraphClient
from dc_sdmx import SdmxClient
from materialise import (
    MaterialisationRequest,
    RequestedWindow,
    RevisionSnapshot,
    materialise_spec,
)

class StaticRevisionReader:
    def read(self) -> RevisionSnapshot:
        return RevisionSnapshot(source_revisions=(), graph_revision="rev_123")

sdmx_client = SdmxClient(base_url="https://datacommons.one.org/core/api")
graph_reader = GraphClient(base_url="https://datacommons.one.org/core/api")
revision_reader = StaticRevisionReader()

request = MaterialisationRequest(
    stat_var="dcid:Count_Person",
    role_bindings={"observationAbout": ["country/KEN", "country/NGA"]},
    declared_roles=("observationAbout",),
    facet_bindings={},
    window=RequestedWindow(kind="all"),
)

result = materialise_spec(
    request=request,
    sdmx=sdmx_client,
    graph=graph_reader,
    revision_reader=revision_reader,
    materialiser_build="v1",
    capability_version="2026-08",
)
```

### Offline Frame Manipulation

These operations execute in memory on `Frame` instances or schema structures without network or disk I/O. The `materialise` modules need the `sdmx` extra, and `materialise.formats` needs `formats`; `dfo_platform` and `assembler` need no extra:

- **Content Versioning**: `materialise.content_version.mint_version` computes deterministic content version hashes over canonical `Frame` objects.
- **Reshaping**: `materialise.reshape` provides structural transforms (`decode_columns`, `apply_reshapes`, `groupby_sum`, `select_columns`, `enrich_columns`, `pivot`).
- **Summary Statistics**: `materialise.statistics.compute_series_statistics_table` computes per-series observation metrics columnar, into the `series-statistics/1` Arrow schema; `SeriesStatisticsSink` streams the result to a Parquet build (`SeriesStatisticsFile`, `data`/`series_count`).
- **Format Serialization**: `materialise.formats` serialises frames and reshaped tables (`write_csv`, `write_excel`, `write_parquet`, `write_frame1`).
- **Wire Vocabulary**: `dfo_platform.problem`, `dfo_platform.codes`, `dfo_platform.trace`, and `dfo_platform.validation` format responses and trace state.
- **Frame Assembly**: `assembler.frame_view`, `assembler.bind`, `assembler.resolve`, and `assembler.series_id` operate on static frame structures.

```python
import io

from dfo_frame_grammar import Frame
from materialise.content_version import mint_version
from materialise.formats import Rendering, write_parquet
from materialise.reshape import PivotPlan, ReshapePlan, apply_reshapes
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table

frame = Frame.build(
    role_keys=["observationAbout"],
    rows=[
        {"stat_var": "dcid:Count_Person", "observationAbout": "country/KEN", "time_period": "2020", "value": 50000000.0},
        {"stat_var": "dcid:Count_Person", "observationAbout": "country/NGA", "time_period": "2020", "value": 200000000.0},
    ],
)

# Compute series statistics on a canonical Frame and stream them to a
# series-statistics/1 Parquet build
statistics_sink = SeriesStatisticsSink()
statistics_sink.write(compute_series_statistics_table(frame))
statistics_file = statistics_sink.finish()  # .data (bytes), .series_count (int)

# Mint content version from frame and metadata preimages
version = mint_version(
    frame,
    graph_revision="rev_123",
    source_revisions=(),
    facets=(),
)

# Apply reshape operations to convert frame into a wide table
reshape_plan = ReshapePlan(
    pivot=PivotPlan(index="observationAbout", columns="time_period", column_cap=100, cell_cap=1000)
)
table, warnings = apply_reshapes(frame, plan=reshape_plan)

# Write the table as Parquet into any sink with `write(data) -> int`
# (materialise.formats needs the formats extra)
sink = io.BytesIO()
artifact = write_parquet([table], rendering=Rendering(), sink=sink)  # .digest, .size_bytes
parquet_bytes = sink.getvalue()
```

## Installation

### Workspace Dependency (Internal Packages)

Within the `dfo-apis` workspace, the root project references `dfo-data-tools` via `[tool.uv.sources]` and selects extras per root extra. Its `datasets` extra, for example:

```toml
[project]
name = "dfo-apis"
dependencies = [
    "dfo-data-tools",
]

[project.optional-dependencies]
datasets = [
    "dfo-data-tools[formats,coverage,sdmx]",
]

[tool.uv.sources]
dfo-data-tools = { workspace = true }
```

### Git Subdirectory Dependency (External Repositories)

External packages depend on `dfo-data-tools` by referencing the git repository and subdirectory. The `sdmx` and `formats` extras cover the SDMX, graph, materialisation, and export imports documented above:

```toml
[project]
dependencies = [
    "dfo-data-tools[sdmx,formats] @ git+https://github.com/ONEcampaign/dfo-apis.git@<commit-or-tag>#subdirectory=packages/dfo-data-tools",
]
```

Transitive dependencies (`dfo-spec-kit-frame-grammar` and `dfo-spec-kit-components`) resolve via git sources declared in `[tool.uv.sources]`. Use `uv` for workspace and lockfile management:

```bash
uv lock
uv sync
```
