"""Environment-backed settings for the Search engine.

One ``SearchSettings`` object is built at import, so invalid values fail at
startup. Consumers read fields through ``settings()`` at use time, allowing
fixtures to replace the object and every caller to see the same settings.
This module makes no LLM or graph calls.
"""

import functools
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dfo_platform.flags import lenient_bool
from dfo_platform.schema_annotations import strip_schema_annotations
from materialise.recipe_bodies import RECIPE_CLASSES as _MATERIALISE_RECIPE_CLASSES
from origin_catalogue.graph import (
    ARCS_CACHE_SIZE,
    OBS_CACHE_SIZE,
    OBS_PROPERTIES_CACHE_SIZE,
    UN_BASE_URL,
    UN_BASE_URL_ENV,
)
from origin_catalogue.topic_walk import TOPIC_EXPAND_MAX_DEPTH, TOPIC_EXPAND_MAX_MEMBERS
from search.engine.extraction_store import is_extraction_store_mode
from search.observations import observation_settings
from statvar_members import DEFAULT_SUBJECT_CEILING
from unit_words.currency_codes import ISO_4217_CODES

# Accepted values for SEARCH boolean flags. Whitespace is significant.
_SEARCH_BOOL_TRUTHY = frozenset({"1", "true", "yes"})


def _load_dotenv() -> None:
    """Load unquoted, uninterpolated ``KEY=VALUE`` lines from ``.env``.

    Existing environment values take precedence. Blank and comment lines are skipped.
    """
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# ---------------------------------------------------------------------------
# Fixed module constants without environment overrides.
# ---------------------------------------------------------------------------

# User-Agent identifying DFO API traffic in the graph's request logs.
USER_AGENT: str = "dfo-apis (+https://github.com/DevelopmentFinanceObservatory/apis)"

# Fixed parsed-door caps duplicated as Pydantic ``Field(max_length=...)``
# literals in search.models, which avoids an engine import cycle and engine-extra dependency.
# Oversized parsed input raises
# Parsed inputs raise ``request_invalid``. ``max_variables`` clamps with a warning.
# Keeping these values out of the environment keeps model and capability caps aligned.
SEARCH_MAX_PARTS: int = 6
SEARCH_MAX_PART_ENTITIES: int = 16
SEARCH_MAX_VARIABLE_TEXT_CHARS: int = 200
SEARCH_MAX_ENTITY_TEXT_CHARS: int = 128
SEARCH_MAX_ROLE_TEXT_CHARS: int = 64
SEARCH_MAX_TIME_TEXT_CHARS: int = 100
SEARCH_MAX_UNIT_TEXT_CHARS: int = 100
SEARCH_MAX_SOURCE_TEXT_CHARS: int = 200

# Resolve doors in contract order. Capabilities and capability_version use this tuple.
SEARCH_RESOLVE_DOORS: tuple[str, ...] = ("raw_text", "parsed", "dataset_resubmit")

# Unserved contract doors, read by the resolve route and capabilities endpoint.
# Empty while every contract-defined door is live.
SEARCH_UNIMPLEMENTED_DOORS: frozenset[str] = frozenset()

# Capability markers for native unit refs, whose currency roster comes from a graph read.
# The hash records its source and the native ``current`` prices vocabulary. Constant
# prices and other currencies come from conversion recipes.
SEARCH_UNIT_REF_SOURCE: str = "graph:currency_units"
SEARCH_UNIT_REF_PRICES_VOCABULARY: tuple[str, ...] = ("current",)

# The conversion recipe members this deployment applies, the sorted keys of
# materialise.recipe_bodies.RECIPE_BODIES. Published as capabilities' recipe_classes.
SEARCH_RECIPE_CLASSES: tuple[str, ...] = _MATERIALISE_RECIPE_CLASSES

# Cap on NoData.unit_alternatives. The binder's largest candidate set is three, either
# the three currencies for "francs" or "kr" or the three spec roles.
SEARCH_MAX_UNIT_ALTERNATIVES: int = 10

# Distinguishes an unset derived default from an explicit value.
_UNSET_DERIVED_DEFAULT = -1


class SearchSettings(BaseSettings):
    """Engine configuration, read from `SEARCH_*` environment variables.

    Constructed once at import (`_SETTINGS` below); every consumer reads a
    field off `settings()`'s return value at the point of use. Field names
    are the `SEARCH_` env var name lowercased with that prefix stripped
    (e.g. `SEARCH_MAX_VARIABLES` -> `max_variables`). Two fields are aliased:
    `gemini_api_key` to the unprefixed `GEMINI_API_KEY` that every other
    Gemini-touching tool in this ecosystem already expects, and
    `capability_version_override` to `SEARCH_CAPABILITY_VERSION`, which sets the
    `capability_version` property.

    Cross-field invariants (a concurrency total that must dominate its own
    per-call width, a cache that must not evict what a full-cap read just
    wrote, ...) are enforced in `_validate_bounds` below, in the same order
    as the fields they check.
    """

    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="forbid")

    # -- LLM ------------------------------------------------------------
    # Concrete default ties local evaluation baselines to one model. CI pins its own value.
    engine_model: str = "gemini-3.5-flash-lite"

    # Best-effort Gemini seed. Repeated identical queries matched on the pinned model.
    # Defaults to 1. An explicit empty string selects unseeded sampling.
    llm_seed: int | None = 1

    # Overrides the model-derived thinking level where that config is supported.
    llm_thinking_level: str | None = None

    # Gemini API key for LLM calls. Reads unprefixed GEMINI_API_KEY.
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    # -- Graph ------------------------------------------------------------
    # Bare host of the Data Commons service, from SEARCH_GRAPH_BASE. Required, so
    # a process with no graph target fails at import rather than reading another
    # environment's graph.
    graph_base: str

    # Optional x-graph-token secret. The WAF challenges data-centre IPs with browser checks.
    # A matching token exempts trusted clients. Empty values omit the header.
    graph_token: str = ""

    # The UN System Data Commons API root, which serves `undata/` dcids. It takes no token.
    # Reads unprefixed UN_GRAPH_BASE_URL, like every other surface and the catalogue job.
    un_graph_base_url: str = Field(default=UN_BASE_URL, validation_alias=UN_BASE_URL_ENV)

    # -- SDMX ------------------------------------------------------------
    # Client-side /data caps. Data Commons leaves response-size limits to clients.
    # Broad ODA health reads returned 154,839 rows/43.9 MiB. Peak measured response was
    # Largest measured response reached 58.29 MiB.
    # The 200,000-row/64-MiB caps admit those reads with margin.
    # Preview reads use obs_preview_max_rows.
    sdmx_max_rows: int = 200_000
    sdmx_max_bytes: int = 64 * 1024 * 1024

    # Per-call width for chunked SDMX reads. Defaults to dc_sdmx's 8 and is bounded by the total.
    sdmx_chunk_concurrency: int = 8

    # Both SDMX clients share the process-wide GET semaphore from clients/sdmx.py.
    # Three runs of 520 GETs showed little benefit past 16. P95 rises 65% at 24.
    # No errors occurred through 32.
    # Must be at least sdmx_chunk_concurrency (8) so one call can use its full worker width.
    sdmx_total_concurrency: int = 16

    # Maximum role_series cache entries. 0 disables caching. A 26-facet, 8-role,
    # 40-point entry measured about 35 KB, or about 18 MB at the 512-entry default.
    # This worst case is small against the 150-266 MB process baseline on a 1 GiB instance.
    sdmx_facet_cache_size: int = 512

    # Arrow bytes the materialise SDMX client keeps of completed observation reads, in total and
    # per read. 0 disables caching. A read over the per-read cap is served and not kept, so one
    # large preview cannot evict the rest.
    materialise_sdmx_cache_max_bytes: int = 64 * 1024 * 1024
    materialise_sdmx_cache_max_entry_bytes: int = 8 * 1024 * 1024

    # Row cap per series of a materialise_preview read, separate from the broad-read and
    # inline-display caps. A read of several series shares the cap times its series equally.
    obs_preview_max_rows: int = 20_000

    # Open-axis discovery filters the coverage artifact without a member-count cap.
    # scope_min_year floors open-ended windows. Coverage paths also floor at the artifact's
    # epoch_year, the first year represented by its bitmask.
    # scope_max_window_years limits year lists before they become SDMX OR-filters.
    # region_max_members bounds alt-shape observation-facet prewarming.
    # The options candidate probe passes its StatVar set to the facet read.
    region_max_members: int = 700

    scope_min_year: int = 1960
    scope_max_window_years: int = 80

    # -- Timeouts ------------------------------------------------------------
    # Passed to httpx.Client(timeout=...) as one number, which sets the
    # connect, read, write, and pool timeouts alike.
    graph_timeout_s: float = 30

    # Gemini HTTP timeout in seconds. genai expects milliseconds. This bounds each call so
    # pipeline deadline handling can proceed.
    llm_timeout_s: float = 30

    # clients/llm.py limits each event loop to this many HTTP/1.1 connections.
    # Requests over the cap wait for a free connection up to llm_timeout_s.
    llm_max_concurrency: int = 32

    # Resolve's soft deadline in seconds from request start. The fan-out stops waiting for
    # unfinished variables then and answers with what it has (`diagnostics.timed_out`).
    request_timeout_s: float = 45

    # Seconds past the soft deadline before resolve's hard cap ends the request with a 504
    # (JSON) or a terminal error event (SSE). The hard cap also bounds every graph and SDMX
    # read the request makes (`dfo_platform.request_deadline`). Cancelling unfinished
    # variables and assembling the partial answer take milliseconds, so the grace covers
    # that work only. The `ai` repository's data-mcp gives up on a resolve at 50 s.
    finalise_grace_s: float = 2

    # Deadline from request start for a resubmit whose member walk reads the whole filtered
    # set (`resubmit_subject_ceiling`). The full read of a deep-root family such as
    # Person/count with gender=Female takes about 30 s on top of the capped walk, so 45 s
    # leaves no room for grounding and observations. Every other resubmit keeps
    # request_timeout_s.
    resubmit_full_read_timeout_s: float = 90

    # -- Unit completion ---------------------------------------------------
    # ISO 4217 code a unit phrase that names no currency completes to.
    unit_default_currency: str = "USD"

    # -- Directional place constraint (PAC) ------------------------------
    # Server default for place-as-constraint (PAC). When on, recipients bind directionally.
    # Every request uses this server-side default. The wire has no override.
    seam_default: bool = True

    # -- Qualifier coverage -------------------------------------------------
    # Sets what happens to a word no reader covers beside a recognised code
    # (`phase_rank_bind._coverage_records`). "refuse" records it, so the part refuses
    # qualifier_not_bound with the near match. "log" logs the record it would add and
    # answers as if it were absent.
    qualifier_coverage: Literal["log", "refuse"] = "refuse"

    # -- Security / prompt safety ------------------------------------------
    # FastAPI rejects longer queries with 422 before any LLM or graph call.
    max_query_chars: int = 2000

    # -- Shape discovery ------------------------------------------------------
    # /v2/resolve scope, passed verbatim as the request body's "target" field.
    # "base_only" searches core Data Commons only, "custom_only" searches ONE's custom
    # data only, "base_and_custom" searches both. The accepted set is exactly the
    # mixer's ResolveTarget* constants (datacommonsorg/mixer,
    # internal/server/v2/resolve/resolve.go).
    resolve_target: str = "base_and_custom"

    # Timeout in seconds for the UN System Data Commons resolve, which starts beside ours in
    # recall. It is the HTTP call's hard deadline; how long a part waits for it is
    # un_first_pass_wait_s. The UN resolve measured p50 0.73 s and p95 0.98 s.
    # SEARCH_REQUEST_TIMEOUT_S still bounds the whole request.
    un_resolve_timeout_s: float = 3.0

    # How long a part waits for the UN leg, its resolve and the derivation of its shapes,
    # from the start of derivation, in seconds. The UN derivation runs beside ours, and only
    # a part whose own pass ends no_data waits for it, or with un_on_first_menu one whose
    # best shape is weak. Past it that part keeps our answer and carries
    # un_system_unavailable. Run
    # beside ours against the live UN deployment, ten queries' UN legs finished 0.89 to
    # 1.57 s after derivation began; a 1.0 s wait missed one of the ten. The `un_derive` log
    # line gives each derivation's time. Must be in (0, un_resolve_timeout_s].
    un_first_pass_wait_s: float = 1.5

    # Whether our pass's menu also holds the UN leg's shapes, each meeting the same admission
    # and its own pool's floor (`phase_intake._merge_un_pool`). Only a part whose best shape
    # scores below `phase_intake.UN_MERGE_WEAK_SCORE` takes them, waiting for the UN leg up
    # to un_first_pass_wait_s. The UN shapes take the seats ours leave (`seating.seat_menu`).
    # A part whose merged pass ends no_data still runs the UN pass.
    un_on_first_menu: bool = False

    # Timeout in seconds for a definite answer's related-measures reads (`related`), which
    # start beside the answer's other post-election reads. Past it the answer states
    # related_measures as null. Three graph reads, two of them side by side, measured
    # 0.5-0.8 s on dfo-dev. A cold read beside a heavy answer's own took up to 2 s, and this
    # bound keeps that time from the answer.
    related_measures_timeout_s: float = 1.0

    # Number of top-scoring detect_svs candidates kept after sorting descending by
    # Client-side cap on top-scoring detect_svs candidates.
    # unknown_score_threshold gates query-level abstention.
    detect_k: int = 80

    # Resolver scores span about 0.55-0.95. Calibration: France 0.6065 (unresolved),
    # Kenya 0.7493, India 0.7628, GDP 0.9084, Spain 0.8226 (resolved), Brazil forest area
    # 0.8230 versus 0.8184 (margin 0.0046).
    # Scores below 0.65 are dropped. A query with no eligible shape returns variable_not_resolved.
    unknown_score_threshold: float = 0.65

    # The same floor for shapes the UN System Data Commons resolver recalls, on its own
    # scale. Its twins of our series score 0.69 to 0.91 against the same texts, and an
    # unrelated control tops at 0.53.
    un_unknown_score_threshold: float = 0.65

    # Minimum score for a recalled SV to be treated as confident (non-weak).
    # Only used below as the lower bound of the RETRIEVAL_SCORE_WEAK info-warning band.
    # Equal to unknown_score_threshold. Weak-match warnings start above this cutoff.
    relevance_threshold: float = 0.65

    # Scores between relevance_threshold and this value trigger RETRIEVAL_SCORE_WEAK, a
    # warning that lowers confidence and sets no status.
    weak_score_threshold: float = 0.75

    # Maximum number of candidate specs returned in a CandidatesPart: the elected reading
    # and its rivals under the status rule, in menu order.
    max_candidates: int = 6

    # Variables above this post-extraction cap are dropped with a VARIABLES_CLAMPED warning.
    max_variables: int = 6

    # Raw entity count checked before deduplication or span scanning. Overflow raises
    # request_invalid because dropping an entity would hide a place from every variable.
    # The cap 96 matches the parsed door's 6 parts x 16 entities and admits ordinary G20 asks.
    # Resolvekit costs about 0.3-0.4 ms per entity. 96 takes tens of milliseconds of the
    # 45-s request budget. The cap bounds work on the process's single shared resolver thread.
    max_entities: int = 96

    # Raw mention cap, checked alongside max_entities before scanning entity spans.
    # max_entities limits distinct names. This cap bounds repeats because span processing visits
    # every occurrence. The default is max_entities * 4. An environment value overrides it.
    max_entity_occurrences: int = _UNSET_DERIVED_DEFAULT

    # At most 20 dropped qualifiers per extraction. The densest corpus query has 2.
    # the cap bounds qualifier-to-variable scans to 120 comparisons at 6 variables.
    max_dropped_qualifiers: int = 20

    # Truncated phrases retain coded qualifiers because the matcher reads their leading text.
    # The 200-character cap matches SEARCH_MAX_VARIABLE_TEXT_CHARS.
    max_dropped_qualifier_length: int = 200

    # Maximum number of candidate shapes (across all families) presented to the universal
    # choose+bind prompt. Retrieval ranks shapes. The LLM elects the winner.
    # 4 seats lose 21 elections on 399 definite dev items (97.5% to 93.0%).
    # No acceptable shape ranks below 8: recall found an expected shape for 131 parts, the
    # menu seated one for 127, and policy refused the other 4.
    # Seating all 255 shapes fills more than 8 on 123 menus, gains nothing, adds 300 ms at
    # p95, and sends up to 60,010 tokens against a 64k request ceiling.
    bind_top_k: int = 8

    # A Topic expands when its score is within this margin of the pool top, including the
    # best non-topic candidate. This lets a specific SDG indicator Topic survive broader
    # distractor Topics and higher-scoring non-topic candidates.
    topic_expand_margin: float = 0.05

    # Drop roots above this leaf count. The per-root cap leaves other roots available.
    # Topic and StatVarPeerGroup nodes can contain 1,700-5,200+ members.
    topic_expand_max_members: int = TOPIC_EXPAND_MAX_MEMBERS

    # Maximum independent Topic roots expanded per query, bounding BFS traversals.
    topic_expand_max_roots: int = 3

    # Each BFS level costs one batched node_arcs call, independent of frontier width.
    # Four levels cover the deepest observed SDG chain: indicator Topic, series Topic,
    # StatVarPeerGroup, then leaf StatisticalVariable (3 container hops).
    topic_expand_max_depth: int = TOPIC_EXPAND_MAX_DEPTH

    # The one bound on the places a Spec binds, counted as distinct dcids per role and
    # summed over the Spec's roles. The probe list a region expands to falls under the same
    # bound. Only the subject list is sliced to the SDMX URL budget and flow-role filters
    # travel whole, so the sum across roles must fit one request. Entity filters cost 4,399
    # of the 7,800 URL bytes at 256 dcids, and 256 on each of two roles would exceed the
    # budget. At 256 the bound covers all countries under Earth and 33 resolvekit groups.
    # The UN is the largest, with 193 members. A first answer over the bound is `no_data`
    # with `entity_set_too_large`, and a resubmit over it is refused `request_invalid`.
    max_child_places: int = 256

    # Sample size per shape family across unasserted axes. One asserting StatVar confirms an axis.
    # Truncated pages may omit unconfirmed axes from slot_options. Posted keys remain valid.
    resubmit_axis_sample: int = 50

    # Subjects a resubmit's or the options door's member walk reads when its first page
    # truncates without proof. The Datasets mint shares the default, so all hit the same
    # cached groups.
    resubmit_subject_ceiling: int = DEFAULT_SUBJECT_CEILING

    # Presence-check cap for raw options candidates, separate from resolve fan-out.
    # At 500 it covers 256 Earth countries, 8 continents, and dozens of regions per country.
    max_options: int = 500

    # -- Variable pipeline concurrency ------------------------------------
    # Concurrent variable legs per request. The limit must cover max_variables.
    # Entity resolution uses one separate call for the full unique-entity set (up to 96).
    max_variable_concurrency: int = 8

    # -- Client-side caches (per-LiveGraphClient, cleared on restart) -----
    # Observation-facet LRU entries keyed by (stat_var, entity, needs_dates). Cleared on restart.
    obs_cache_size: int = OBS_CACHE_SIZE

    # detect_svs LRU entries use (generation stamp, catalogue build, UN setting, query,
    # resolve_target, detect_k) as the key.
    detect_cache_size: int = 256

    # node_arcs and node_arcs_batch share an LRU keyed by dcid. Cleared on restart.
    arcs_cache_size: int = ARCS_CACHE_SIZE

    # LRU entries for node_observation_properties_batch, separate from the full-arc cache.
    # It holds only live reads: the catalogue's arc tables answer the variables they hold.
    obs_properties_cache_size: int = OBS_PROPERTIES_CACHE_SIZE

    # -- Response cache (process-wide, shared across requests) ------------------
    # Enable raw-text response caching. The deployed model/seed pin preserves behaviour.
    response_cache_enabled: bool = True

    # Exact serialized response bytes retained before LRU eviction. The cache stores encoded bytes.
    response_cache_max_bytes: int = 64 * 1024 * 1024

    # Oversized responses are served without storage, protecting other entries from eviction.
    response_cache_max_entry_bytes: int = 2 * 1024 * 1024

    # Bytes of resubmitted identity regions the part memo holds, in total and per region, each
    # region weighed by the heap it holds (`response_cache_memo.region_bytes`). 0 disables
    # the memo, which also follows response_cache_enabled. A region over the per-region cap
    # is served and not kept, so one large part cannot evict the rest. On the dev graph a
    # subject-only part weighs 0.07 to 4.4 MB, most of it its family's arcs, so the total
    # holds about 15 to 900 parts. A role-bearing CRS part of 3,687 served members weighs
    # about 75 MB, most of it its members' arcs, specs, read plans and option facts, so
    # such a part is served and not kept.
    response_cache_memo_max_bytes: int = Field(default=64 * 1024 * 1024, ge=0)
    response_cache_memo_max_entry_bytes: int = Field(default=8 * 1024 * 1024, ge=0)

    # OptionsPopulation memo, one per search_surface(), keyed by (identity.shape, identity.slots).
    # Defaults on, like response_cache_enabled.
    options_population_cache_enabled: bool = True

    # Maximum distinct (identity.shape, identity.slots) populations held before LRU eviction.
    options_population_cache_max_entries: int = Field(default=256, ge=1)

    # Dedicated workers and non-blocking gate keep place prefetch off asyncio's shared executor.
    # Four workers cover the 2-job peers/children pair across overlapping requests.
    place_options_workers: int = 4

    # Process-wide options cache, keyed after resolve_options_target and enabled by default.
    # Cache warning-free responses only. Run validation and role/anchor refusals on each request.
    # A hit skips response building.
    options_response_cache_enabled: bool = True

    # Maximum distinct options-response keys held before LRU eviction.
    options_response_cache_max_entries: int = Field(default=512, ge=1)

    # LLM memo lifetime. Cached answers carry no graph-generation stamp.
    llm_memo_ttl_s: float = 3600

    # Exact retained JSON-byte cap before LRU eviction. 0 disables memoisation.
    llm_memo_max_bytes: int = 16 * 1024 * 1024

    # Oversized answers are served without storage, protecting other memo entries from eviction.
    llm_memo_max_entry_bytes: int = 256 * 1024

    # Where extract() keeps the model's reading of each query: `off`, `memory` (an in-process
    # LRU with no TTL) or `file:<path>` (JSON lines a later run replays). Keys hold the extract
    # prompt and schema, so an edit to either reads afresh.
    extract_store: str = "memory"

    # -- Observation values ------------------------------------------------
    # Worker cap for one preview fetch. 1 runs its pairs sequentially.
    # Each variable leg has a preview pool. obs_preview_total_concurrency caps aggregate fetches.
    obs_preview_concurrency: int = 8

    # Process-wide semaphore caps preview pair fetches.
    obs_preview_total_concurrency: int = 32

    # On dev, 139 node-arc chunks took 5.9 s at 8 threads.
    # At 16 threads they took 2.75 s, with no errors or continuation tokens.
    node_arc_call_concurrency: int = 16

    # Process-wide cap on node-arc batch POSTs, including halving retries.
    # Node label/type reads are outside this bound. Each batch POST holds one permit.
    # Must be at least node_arc_call_concurrency.
    node_arc_total_concurrency: int = 32

    # Process-wide cap for materialise facet-probe POSTs. Also limits each call's worker pool.
    facet_probe_total_concurrency: int = 8

    # -- Recall quality, acronym canonicalization, and relevance floor --------
    # Expand the code index's variable aliases before bare-variable detection. Other fields
    # keep their input.
    acronym_canonicalize: bool = True

    # Drop candidates below 0.30 before shape derivation. Topic candidates are exempt.
    # Correct fixture scores include GDP at 0.619 and life expectancy in Kenya at 0.562.
    # Unresolvable queries score well below 0.3. Query-level unknown_score_threshold is 0.65.
    recall_score_floor: float = 0.30

    # -- Eval-harness build fingerprint -------------------------------------
    # Eval fingerprint for comparing runs. Diagnostics expose the image commit separately.
    engine_build: str = "search-engine-dev"

    # SEARCH_CAPABILITY_VERSION overrides `capability_version`. Empty derives it.
    capability_version_override: str = Field(
        default="", validation_alias="SEARCH_CAPABILITY_VERSION"
    )

    @field_validator("llm_seed", mode="before")
    @classmethod
    def _seed_empty_string_opts_out(cls, value: object) -> object:
        """An explicit empty string opts back out to unseeded sampling (None)."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("llm_thinking_level", mode="before")
    @classmethod
    def _thinking_level_empty_string_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "seam_default",
        "response_cache_enabled",
        "options_population_cache_enabled",
        "options_response_cache_enabled",
        "acronym_canonicalize",
        mode="before",
    )
    @classmethod
    def _parse_lenient_bool_flags(cls, value: object) -> object:
        return lenient_bool(truthy=_SEARCH_BOOL_TRUTHY, strip=False)(value)

    @field_validator("unit_default_currency")
    @classmethod
    def _unit_default_currency_is_iso(cls, value: str) -> str:
        if value not in ISO_4217_CODES:
            raise ValueError(
                f"SEARCH_UNIT_DEFAULT_CURRENCY must be a registered ISO 4217 code; got {value!r}"
            )
        return value

    @field_validator("capability_version_override", mode="before")
    @classmethod
    def _capability_version_stripped(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _derive_defaults(self) -> "SearchSettings":
        """Set dependent field defaults before ``_validate_bounds`` checks them."""
        if self.max_entity_occurrences == _UNSET_DERIVED_DEFAULT:
            self.max_entity_occurrences = self.max_entities * 4
        return self

    @functools.cached_property
    def capability_version(self) -> str:
        """Capability token in problem details and capabilities, distinct from
        `schema_version`: `SEARCH_CAPABILITY_VERSION` where set, otherwise the ``cap_`` hash
        (`_compute_capability_version`). The hash reads the served schema and so the code
        index, so it is computed on first read, once per settings object."""
        return self.capability_version_override or self._compute_capability_version()

    def _capability_version_payload(self) -> dict:
        # Imported here: the schema module reads the models and the code index, which load
        # after this module's settings.
        from search.engine import schema

        return {
            # The served schema without its prose: a field's name, type, `enum`, `const` or
            # `default` moves the token, a reworded description does not.
            "wire_schema": strip_schema_annotations(schema.schema_payload()),
            "doors": {
                door: door not in SEARCH_UNIMPLEMENTED_DOORS for door in SEARCH_RESOLVE_DOORS
            },
            "seam_default": self.seam_default,
            "unit_default_currency": self.unit_default_currency,
            "unit_ref_source": SEARCH_UNIT_REF_SOURCE,
            "unit_ref_prices_vocabulary": list(SEARCH_UNIT_REF_PRICES_VOCABULARY),
            "recipe_classes": list(SEARCH_RECIPE_CLASSES),
            "caps": {
                "max_query_chars": self.max_query_chars,
                "max_candidates": self.max_candidates,
                "max_variables": self.max_variables,
                "obs_preview_points": observation_settings().obs_preview_points,
                "obs_inline_max": observation_settings().obs_inline_max,
                "role_bulk_rank_max_rows": observation_settings().role_bulk_rank_max_rows,
                "max_options": self.max_options,
                "max_parts": SEARCH_MAX_PARTS,
                "max_part_entities": SEARCH_MAX_PART_ENTITIES,
                "max_variable_text_chars": SEARCH_MAX_VARIABLE_TEXT_CHARS,
                "max_entity_text_chars": SEARCH_MAX_ENTITY_TEXT_CHARS,
                "max_role_text_chars": SEARCH_MAX_ROLE_TEXT_CHARS,
                "max_time_text_chars": SEARCH_MAX_TIME_TEXT_CHARS,
                "max_unit_text_chars": SEARCH_MAX_UNIT_TEXT_CHARS,
                "max_source_text_chars": SEARCH_MAX_SOURCE_TEXT_CHARS,
                "region_max_members": self.region_max_members,
                "scope_min_year": self.scope_min_year,
                "scope_max_window_years": self.scope_max_window_years,
            },
        }

    def _compute_capability_version(self) -> str:
        """Deterministic `capability_version` token: a short content hash over
        this deployment's stable capability fields (per-door liveness, the
        place-as-constraint default, and the published caps) and the schema
        `GET /api/search/schema` serves (`search.engine.schema.schema_payload`,
        without annotations), so a wire change or a config
        change mints a new token while repeated boots of the identical config
        agree on the same one. The running image's own commit sha is excluded
        from this hash, so a rebuild with no behaviour change mints no new
        token.
        """
        canonical = json.dumps(
            self._capability_version_payload(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"cap_{digest}"

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SearchSettings":
        """Startup validation: raises ValueError at import, before the port binds."""
        self._validate_retrieval_and_scope_bounds()
        self._validate_score_and_candidate_bounds()
        self._validate_bind_and_cache_bounds()
        self._validate_topic_and_place_bounds()
        self._validate_timeout_and_preview_bounds()
        self._validate_concurrency_and_deployment_bounds()
        self._validate_cache_and_memo_bounds()
        return self

    def _validate_retrieval_and_scope_bounds(self) -> None:
        """SDMX read caps, observation-preview caps, the region member cap, and the
        scope window."""
        if self.sdmx_max_rows < 1:
            raise ValueError(f"SEARCH_SDMX_MAX_ROWS must be >= 1; got {self.sdmx_max_rows!r}")
        if self.sdmx_max_bytes < 1:
            raise ValueError(f"SEARCH_SDMX_MAX_BYTES must be >= 1; got {self.sdmx_max_bytes!r}")
        if self.sdmx_chunk_concurrency < 1:
            raise ValueError(
                f"SEARCH_SDMX_CHUNK_CONCURRENCY must be >= 1; got {self.sdmx_chunk_concurrency!r}"
            )
        if self.sdmx_total_concurrency < self.sdmx_chunk_concurrency:
            raise ValueError(
                f"SEARCH_SDMX_TOTAL_CONCURRENCY ({self.sdmx_total_concurrency!r}) must be "
                f">= SEARCH_SDMX_CHUNK_CONCURRENCY ({self.sdmx_chunk_concurrency!r})"
            )
        if self.sdmx_facet_cache_size < 0:
            raise ValueError(
                f"SEARCH_SDMX_FACET_CACHE_SIZE must be >= 0; got {self.sdmx_facet_cache_size!r}"
            )
        if self.materialise_sdmx_cache_max_bytes < 0:
            raise ValueError(
                "SEARCH_MATERIALISE_SDMX_CACHE_MAX_BYTES must be >= 0; got "
                f"{self.materialise_sdmx_cache_max_bytes!r}"
            )
        if self.materialise_sdmx_cache_max_entry_bytes < 0:
            raise ValueError(
                "SEARCH_MATERIALISE_SDMX_CACHE_MAX_ENTRY_BYTES must be >= 0; got "
                f"{self.materialise_sdmx_cache_max_entry_bytes!r}"
            )
        if self.obs_preview_max_rows < 1:
            raise ValueError(
                f"SEARCH_OBS_PREVIEW_MAX_ROWS must be >= 1; got {self.obs_preview_max_rows!r}"
            )
        if self.region_max_members < 1:
            raise ValueError(
                f"SEARCH_REGION_MAX_MEMBERS must be >= 1; got {self.region_max_members!r}"
            )
        if self.scope_min_year < 1:
            raise ValueError(f"SEARCH_SCOPE_MIN_YEAR must be >= 1; got {self.scope_min_year!r}")
        if self.scope_max_window_years < 1:
            raise ValueError(
                f"SEARCH_SCOPE_MAX_WINDOW_YEARS must be >= 1; got {self.scope_max_window_years!r}"
            )

    def _validate_score_and_candidate_bounds(self) -> None:
        """Score thresholds and the resolved-StatVar/candidate/variable/entity ceilings."""
        if not (0 < self.relevance_threshold <= 1):
            raise ValueError(
                f"SEARCH_RELEVANCE_THRESHOLD must be in (0, 1]; got {self.relevance_threshold!r}"
            )
        if not (0 < self.weak_score_threshold <= 1):
            raise ValueError(
                f"SEARCH_WEAK_SCORE_THRESHOLD must be in (0, 1]; got {self.weak_score_threshold!r}"
            )
        if self.max_candidates < 1:
            raise ValueError(f"SEARCH_MAX_CANDIDATES must be >= 1; got {self.max_candidates!r}")
        if self.max_variables < 1:
            raise ValueError(f"SEARCH_MAX_VARIABLES must be >= 1; got {self.max_variables!r}")
        if self.max_entities < 1:
            raise ValueError(f"SEARCH_MAX_ENTITIES must be >= 1; got {self.max_entities!r}")
        if self.max_entity_occurrences < self.max_entities:
            raise ValueError(
                "SEARCH_MAX_ENTITY_OCCURRENCES must be >= SEARCH_MAX_ENTITIES "
                f"({self.max_entities!r}); got {self.max_entity_occurrences!r}"
            )

    def _validate_bind_and_cache_bounds(self) -> None:
        """Dropped-qualifier limits, bind/detect caps, and read-cache sizes."""
        if self.max_dropped_qualifiers < 1:
            raise ValueError(
                f"SEARCH_MAX_DROPPED_QUALIFIERS must be >= 1; got {self.max_dropped_qualifiers!r}"
            )
        if self.max_dropped_qualifier_length < 1:
            raise ValueError(
                "SEARCH_MAX_DROPPED_QUALIFIER_LENGTH must be >= 1; got "
                f"{self.max_dropped_qualifier_length!r}"
            )
        if self.bind_top_k < 1:
            raise ValueError(f"SEARCH_BIND_TOP_K must be >= 1; got {self.bind_top_k!r}")
        if self.detect_k < 1:
            raise ValueError(f"SEARCH_DETECT_K must be >= 1; got {self.detect_k!r}")
        if self.arcs_cache_size < 1:
            raise ValueError(f"SEARCH_ARCS_CACHE_SIZE must be >= 1; got {self.arcs_cache_size!r}")
        if self.obs_properties_cache_size < 1:
            raise ValueError(
                "SEARCH_OBS_PROPERTIES_CACHE_SIZE must be >= 1; got "
                f"{self.obs_properties_cache_size!r}"
            )

    def _validate_topic_and_place_bounds(self) -> None:
        """Resolve-target mode, topic-expand bounds, and place/option caps."""
        if self.resolve_target not in {"base_only", "custom_only", "base_and_custom"}:
            raise ValueError(
                "SEARCH_RESOLVE_TARGET must be one of 'base_only', 'custom_only', "
                "'base_and_custom'; "
                f"got {self.resolve_target!r}"
            )
        if not (0 < self.unknown_score_threshold <= 1):
            raise ValueError(
                "SEARCH_UNKNOWN_SCORE_THRESHOLD must be in (0, 1]; got "
                f"{self.unknown_score_threshold!r}"
            )
        if not (0 < self.un_unknown_score_threshold <= 1):
            raise ValueError(
                "SEARCH_UN_UNKNOWN_SCORE_THRESHOLD must be in (0, 1]; got "
                f"{self.un_unknown_score_threshold!r}"
            )
        if self.topic_expand_margin < 0:
            raise ValueError(
                f"SEARCH_TOPIC_EXPAND_MARGIN must be >= 0; got {self.topic_expand_margin!r}"
            )
        if self.topic_expand_max_members < 1:
            raise ValueError(
                "SEARCH_TOPIC_EXPAND_MAX_MEMBERS must be >= 1; got "
                f"{self.topic_expand_max_members!r}"
            )
        if self.topic_expand_max_roots < 1:
            raise ValueError(
                f"SEARCH_TOPIC_EXPAND_MAX_ROOTS must be >= 1; got {self.topic_expand_max_roots!r}"
            )
        if self.topic_expand_max_depth < 1:
            raise ValueError(
                f"SEARCH_TOPIC_EXPAND_MAX_DEPTH must be >= 1; got {self.topic_expand_max_depth!r}"
            )
        if self.max_child_places < 1:
            raise ValueError(f"SEARCH_MAX_CHILD_PLACES must be >= 1; got {self.max_child_places!r}")
        if self.resubmit_axis_sample < 1:
            raise ValueError(
                f"SEARCH_RESUBMIT_AXIS_SAMPLE must be >= 1; got {self.resubmit_axis_sample!r}"
            )
        if self.resubmit_subject_ceiling < 1:
            raise ValueError(
                "SEARCH_RESUBMIT_SUBJECT_CEILING must be >= 1; got "
                f"{self.resubmit_subject_ceiling!r}"
            )
        if self.max_options < 1:
            raise ValueError(f"SEARCH_MAX_OPTIONS must be >= 1; got {self.max_options!r}")

    def _validate_timeout_and_preview_bounds(self) -> None:
        """Graph/LLM/request timeouts, Gemini and observation-preview concurrency."""
        if self.graph_timeout_s <= 0:
            raise ValueError(f"SEARCH_GRAPH_TIMEOUT_S must be > 0; got {self.graph_timeout_s!r}")
        if self.llm_timeout_s <= 0:
            raise ValueError(f"SEARCH_LLM_TIMEOUT_S must be > 0; got {self.llm_timeout_s!r}")
        if self.llm_max_concurrency < 1:
            raise ValueError(
                f"SEARCH_LLM_MAX_CONCURRENCY must be >= 1; got {self.llm_max_concurrency!r}"
            )
        if self.max_query_chars < 1:
            raise ValueError(f"SEARCH_MAX_QUERY_CHARS must be >= 1; got {self.max_query_chars!r}")
        if self.request_timeout_s <= 0:
            raise ValueError(
                f"SEARCH_REQUEST_TIMEOUT_S must be > 0; got {self.request_timeout_s!r}"
            )
        if self.finalise_grace_s <= 0:
            raise ValueError(
                "SEARCH_FINALISE_GRACE_S must be > 0, so the soft deadline's partial answer "
                f"lands before the hard cap; got {self.finalise_grace_s!r}"
            )
        if self.resubmit_full_read_timeout_s < self.request_timeout_s:
            raise ValueError(
                "SEARCH_RESUBMIT_FULL_READ_TIMEOUT_S must be >= SEARCH_REQUEST_TIMEOUT_S; got "
                f"{self.resubmit_full_read_timeout_s!r} < {self.request_timeout_s!r}"
            )
        if self.un_resolve_timeout_s <= 0:
            raise ValueError(
                f"SEARCH_UN_RESOLVE_TIMEOUT_S must be > 0; got {self.un_resolve_timeout_s!r}"
            )
        if not (0 < self.un_first_pass_wait_s <= self.un_resolve_timeout_s):
            raise ValueError(
                "SEARCH_UN_FIRST_PASS_WAIT_S must be in (0, SEARCH_UN_RESOLVE_TIMEOUT_S]; got "
                f"{self.un_first_pass_wait_s!r} with {self.un_resolve_timeout_s!r}"
            )
        if self.related_measures_timeout_s <= 0:
            raise ValueError(
                "SEARCH_RELATED_MEASURES_TIMEOUT_S must be > 0; "
                f"got {self.related_measures_timeout_s!r}"
            )
        if self.max_variable_concurrency < self.max_variables:
            raise ValueError(
                f"SEARCH_MAX_VARIABLE_CONCURRENCY ({self.max_variable_concurrency!r}) must be "
                f">= SEARCH_MAX_VARIABLES ({self.max_variables!r})"
            )
        if self.obs_preview_concurrency < 1:
            raise ValueError(
                f"SEARCH_OBS_PREVIEW_CONCURRENCY must be >= 1; got {self.obs_preview_concurrency!r}"
            )
        if self.obs_preview_total_concurrency < 1:
            raise ValueError(
                "SEARCH_OBS_PREVIEW_TOTAL_CONCURRENCY must be >= 1; got "
                f"{self.obs_preview_total_concurrency!r}"
            )

    def _validate_concurrency_and_deployment_bounds(self) -> None:
        """Graph-read concurrency, the recall score floor, and deployment
        identity bounds (LLM thinking level, graph base URL scheme)."""
        if self.node_arc_call_concurrency < 1:
            raise ValueError(
                "SEARCH_NODE_ARC_CALL_CONCURRENCY must be >= 1; got "
                f"{self.node_arc_call_concurrency!r}"
            )
        if self.node_arc_total_concurrency < self.node_arc_call_concurrency:
            raise ValueError(
                f"SEARCH_NODE_ARC_TOTAL_CONCURRENCY ({self.node_arc_total_concurrency!r}) "
                f"must be >= SEARCH_NODE_ARC_CALL_CONCURRENCY "
                f"({self.node_arc_call_concurrency!r})"
            )
        if self.facet_probe_total_concurrency < 1:
            raise ValueError(
                "SEARCH_FACET_PROBE_TOTAL_CONCURRENCY must be >= 1; got "
                f"{self.facet_probe_total_concurrency!r}"
            )
        if not (0 <= self.recall_score_floor < 1):
            raise ValueError(
                f"SEARCH_RECALL_SCORE_FLOOR must be in [0, 1); got {self.recall_score_floor!r}"
            )
        if self.llm_thinking_level is not None and self.llm_thinking_level not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "SEARCH_LLM_THINKING_LEVEL must be one of 'minimal', 'low', 'medium', 'high'; "
                f"got {self.llm_thinking_level!r}"
            )
        graph_base_parsed = urlparse(self.graph_base)
        # https:// is allowed for any host. http:// is allowed only for the loopback
        # dev hosts, compared by exact hostname so spoofs like http://localhost.evil.com
        # or http://127.0.0.1.attacker are rejected.
        if not (
            graph_base_parsed.scheme == "https"
            or (
                graph_base_parsed.scheme == "http"
                and graph_base_parsed.hostname in {"localhost", "127.0.0.1"}
            )
        ):
            raise ValueError(
                f"SEARCH_GRAPH_BASE must use https:// (or http://localhost / http://127.0.0.1 "
                f"for dev); got {self.graph_base!r}"
            )

    def _validate_cache_and_memo_bounds(self) -> None:
        """Response-cache, LLM-memo and extraction-store settings."""
        if self.response_cache_max_bytes < 1:
            raise ValueError(
                "SEARCH_RESPONSE_CACHE_MAX_BYTES must be >= 1; got "
                f"{self.response_cache_max_bytes!r}"
            )
        if self.response_cache_max_entry_bytes < 1:
            raise ValueError(
                "SEARCH_RESPONSE_CACHE_MAX_ENTRY_BYTES must be >= 1; got "
                f"{self.response_cache_max_entry_bytes!r}"
            )
        if self.response_cache_max_entry_bytes > self.response_cache_max_bytes:
            raise ValueError(
                "SEARCH_RESPONSE_CACHE_MAX_ENTRY_BYTES "
                f"({self.response_cache_max_entry_bytes!r}) must be <= "
                f"SEARCH_RESPONSE_CACHE_MAX_BYTES ({self.response_cache_max_bytes!r})"
            )
        if self.place_options_workers < 1:
            raise ValueError(
                f"SEARCH_PLACE_OPTIONS_WORKERS must be >= 1; got {self.place_options_workers!r}"
            )
        if not math.isfinite(self.llm_memo_ttl_s) or self.llm_memo_ttl_s < 0:
            raise ValueError(
                f"SEARCH_LLM_MEMO_TTL_S must be a finite value >= 0; got {self.llm_memo_ttl_s!r}"
            )
        if self.llm_memo_max_bytes < 0:
            raise ValueError(
                f"SEARCH_LLM_MEMO_MAX_BYTES must be >= 0; got {self.llm_memo_max_bytes!r}"
            )
        if self.llm_memo_max_entry_bytes < 1:
            raise ValueError(
                "SEARCH_LLM_MEMO_MAX_ENTRY_BYTES must be >= 1; got "
                f"{self.llm_memo_max_entry_bytes!r}"
            )
        if self.llm_memo_max_entry_bytes > self.llm_memo_max_bytes and self.llm_memo_max_bytes > 0:
            raise ValueError(
                "SEARCH_LLM_MEMO_MAX_ENTRY_BYTES "
                f"({self.llm_memo_max_entry_bytes!r}) must be <= "
                f"SEARCH_LLM_MEMO_MAX_BYTES ({self.llm_memo_max_bytes!r})"
            )
        if not is_extraction_store_mode(self.extract_store):
            raise ValueError(
                "SEARCH_EXTRACT_STORE must be off, memory or file:<path>; got "
                f"{self.extract_store!r}"
            )


_SETTINGS: SearchSettings = SearchSettings()


def settings() -> SearchSettings:
    """Return the process-wide SearchSettings object.

    Constructed once at import (`_SETTINGS` above); a test fixture installs a
    different constructed instance for its duration via
    `monkeypatch.setattr(search.engine.config, "_SETTINGS", SearchSettings(...))`.
    """
    return _SETTINGS
