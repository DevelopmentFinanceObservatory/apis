"""Read Datasets API settings from the environment in two tiers.

`DatasetsBuildSettings` supplies safe defaults and validates supplied values at door construction.
Its one required value is the public website origin (`site`, a `PublicSiteSettings` from
`dfo_platform.config`), which every bundle render links to, fakes included.
`DatasetsSettings` requires graph, SDMX, and artifact-store values. It is validated once at app
startup when live clients are built. Tests can inject fakes without constructing that tier.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact_store.signing import SIGNED_URL_REUSE_WINDOW, SIGNED_URL_TTL_CEILING
from dfo_platform.config import PublicSiteSettings, env_validation_error_message
from materialise import DEFAULT_MEMBER_CONCURRENCY
from origin_catalogue.graph import (
    ARCS_CACHE_SIZE,
    LABEL_TYPE_CACHE_SIZE,
    OBS_CACHE_SIZE,
    OBS_PROPERTIES_CACHE_SIZE,
    UN_BASE_URL,
    UN_BASE_URL_ENV,
    GraphClientOptions,
)
from statvar_members import DEFAULT_SUBJECT_CEILING


class DatasetsConfigError(Exception):
    """Invalid `DATASETS_*` settings and door dependency-injection conflicts."""


class DatasetsBuildSettings(BaseSettings):
    """Settings with a safe default, resolved once at door-build time.
    See the module docstring for the two-tier split this is one half of.
    """

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    # Inline row/request budgets. Requests over either bound become queued operations.
    sync_budget_rows: int = Field(default=100_000, ge=1)
    sync_budget_requests: int = Field(default=40, ge=1)

    # Async row/request ceilings keep worker time below roughly 300 seconds on 2 vCPU.
    async_max_rows: int = Field(default=2_000_000, ge=1)
    async_max_requests: int = Field(default=800, ge=1)

    # Mint executor threads, independently capped by a same-sized semaphore (503 when full).
    mint_max_concurrency: int = Field(default=4, ge=1)

    # Concurrent SDMX members per request. Each door's client caps total GETs at 8, or 16
    # across both doors.
    mint_member_max_concurrency: int = Field(default=DEFAULT_MEMBER_CONCURRENCY, ge=1)

    # Client wait limit for the mint executor hop. Transport timeouts bound the worker thread.
    mint_timeout_seconds: float = Field(default=120, gt=0)

    # SDMX transport timeout. Each request also has a deadline, and the earlier one applies.
    sdmx_timeout_seconds: int = Field(default=90, ge=1)

    # Pivot cell limit. Format column caps bound width only, and CSV/Parquet have no width cap.
    pivot_max_output_cells: int = Field(default=5_000_000, ge=1)

    # Definition executor threads are independent of mint. Its semaphore returns retryable 503
    # when full.
    definition_max_concurrency: int = Field(default=4, ge=1)

    # Definition client wait limit. On a head hit it also covers on-demand rendering and storage.
    definition_timeout_seconds: float = Field(default=120, gt=0)

    # Shared mint/definition rebuild cap. Default 2 keeps memory-heavy runs (about 300 MB each)
    # within the service's 1 GiB budget.
    materialise_max_concurrency: int = Field(default=2, ge=1)

    # Slot wait limit. The semaphore uses the earlier of this limit and the request deadline (503).
    materialise_wait_seconds: float = Field(default=30, gt=0)

    # Floor for each of a bundle render's two graph metadata reads: the live documentation read
    # that fills a stored citation's empty fields, and the code label read. Each gets at least
    # this long from when it starts, however much of the request deadline is left.
    metadata_read_seconds: float = Field(default=15, gt=0)

    # Subjects a described arm's member walk reads when its first reverse page truncates
    # without a confirmed root. The walk takes the whole filtered set up to this many and
    # confirms it by the subjects' arcs. The default, shared with Search's resubmit, covers the deepest measured
    # families (Person/count by gender, about 13,800) and bounds the read near 40 s and
    # 40 MB. A larger set refuses over_ceiling (candidate_window).
    member_subject_ceiling: int = Field(default=DEFAULT_SUBJECT_CEILING, ge=1)

    # Seconds a member walk's graph reads may run, counted from the start of resolution and
    # capped by the request's own deadline. A walk past it answers timeout (504, retryable),
    # since a retry walks again.
    member_read_seconds: float = Field(default=60, gt=0)

    # Seconds before the next GET probes facet currency again. Default 900s keeps probes below
    # upstream weekly-to-monthly release cadence.
    head_recheck_seconds: int = Field(default=900, ge=1)

    # Maximum head age before rebuilding. Catches count- and date-invariant value corrections.
    head_max_age_seconds: int = Field(default=86_400, ge=1)

    # Snapshot executor threads. A same-sized semaphore returns 503 when full.
    snapshot_max_concurrency: int = Field(default=4, ge=1)

    # Pin executor threads. A same-sized semaphore returns 503 when full.
    pin_max_concurrency: int = Field(default=4, ge=1)

    # New pinned copies allowed per trailing 24 hours, counted from `pin` references. A repin of
    # an already pinned artifact is exempt. 0 refuses every new pinned copy.
    pin_daily_max: int = Field(default=500, ge=0)

    # Publish executor threads. A same-sized semaphore returns 503 when full.
    publish_max_concurrency: int = Field(default=4, ge=1)

    # Operations polling executor threads. Its semaphore returns 503 when full.
    operations_max_concurrency: int = Field(default=4, ge=1)

    # Sent as `x-graph-token` to live graph/SDMX clients to bypass the graph WAF. Empty omits the
    # header.
    graph_token: str = Field(default="")

    # Proxy header naming the authenticated caller. This API enforces pin permission.
    principal_header: str = Field(default="x-dfo-principal")

    # Sole publish principal. Empty denies all. Pin uses a separate principal because pinning is
    # irreversible on this surface.
    first_party_principal: str = Field(default="")

    # Sole pin principal, independent of publish. Empty denies all.
    pin_first_party_principal: str = Field(default="")

    # Lease expires after this many seconds. Workers renew it every ~20s. Queue min_backoff must
    # be at least this value (60s by default) to avoid retries colliding with a live lease.
    # Expiry lets retries or poll-time dead-worker detection recover abandoned operations.
    operation_lease_seconds: int = Field(default=60, ge=1)

    # Mirrors queue `max_attempts` (5 by default). The worker marks the operation failed on its
    # final transient attempt.
    operation_max_attempts: int = Field(default=5, ge=1)

    # Maximum pending/running rows. `create_or_get` refuses a new operation at the limit.
    operation_backlog_max: int = Field(default=20, ge=1)

    # Mirrors queue `max_retry_duration` (3600s by default). Polling adds a fixed margin before
    # classifying an unchanged operation as abandoned.
    operation_max_retry_duration_seconds: int = Field(default=3_600, ge=1)

    # Mirrors queue `max_concurrent_dispatches` (4 by default). Backlog size determines how many
    # full dispatch-deadline rounds a task may wait before dispatch.
    operation_dispatch_concurrency: int = Field(default=4, ge=1)

    # The website bundles link to, read from `DFO_PUBLIC_SITE_URL` (required) and
    # `DFO_SITE_DATASET_LINKS`, which the presentation surface reads too.
    site: PublicSiteSettings = Field(default_factory=PublicSiteSettings.from_env)

    @model_validator(mode="after")
    def _check_head_recheck_before_max_age(self) -> DatasetsBuildSettings:
        """Require the probe interval to be shorter than the maximum head age."""
        if self.head_recheck_seconds >= self.head_max_age_seconds:
            raise ValueError(
                "DATASETS_HEAD_RECHECK_SECONDS must be < DATASETS_HEAD_MAX_AGE_SECONDS "
                f"({self.head_max_age_seconds}), got {self.head_recheck_seconds}"
            )
        return self

    def executor_wait_seconds(self, timeout_seconds: float) -> float:
        """How long a door waits on its executor for a request whose deadline is
        `timeout_seconds` away. It adds both metadata read floors, which a bundle render can run
        past the deadline."""
        return timeout_seconds + 2 * self.metadata_read_seconds

    @classmethod
    def from_env(cls) -> DatasetsBuildSettings:
        """Construct settings and translate validation failures to `DatasetsConfigError`."""
        try:
            return cls()
        except ValidationError as exc:
            raise DatasetsConfigError(
                env_validation_error_message(exc, env_prefix="DATASETS_")
            ) from exc


class DatasetsSettings(BaseSettings):
    """Settings with no safe default, validated fail-fast, once, from the
    surface's `lifespan`: the graph and SDMX endpoints, and the two
    artifact-store bucket names, a live mint door needs to build its
    `OriginGraphClient`/`SdmxClient`/`LiveArtifactStore`. `datasets_api
    .operations.task_client.TaskClientSettings`/`datasets_api.operations
    .worker_app.WorkerSettings` are this same no-safe-default tier for the
    async worker's env, read separately rather than folded in here: the
    api app (enqueuing) needs `DFO_TASKS_QUEUE` and this deployment never
    sets it on the worker app, which never enqueues anything, so a single
    settings object required by both would force an unused var onto
    whichever one doesn't read it.
    """

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    graph_base_url: str = Field(min_length=1)
    # The UN System Data Commons API root, which serves `undata/` dcids, read from
    # `UN_GRAPH_BASE_URL` like every other surface's.
    un_graph_base_url: str = Field(
        default=UN_BASE_URL, min_length=1, validation_alias=UN_BASE_URL_ENV
    )
    sdmx_base_url: str = Field(min_length=1)
    artifact_bucket_working: str = Field(min_length=1)
    artifact_bucket_pinned: str = Field(min_length=1)
    # Graph client cache sizes, defaulting to Search's, since both surfaces read the same graph.
    # Zero turns a cache off.
    graph_label_type_cache_size: int = Field(default=LABEL_TYPE_CACHE_SIZE, ge=0)
    graph_arcs_cache_size: int = Field(default=ARCS_CACHE_SIZE, ge=0)
    graph_obs_properties_cache_size: int = Field(default=OBS_PROPERTIES_CACHE_SIZE, ge=0)
    graph_obs_cache_size: int = Field(default=OBS_CACHE_SIZE, ge=0)

    def graph_cache_options(self) -> GraphClientOptions:
        """The cache sizes every Datasets `build_graph_client` call passes."""
        return GraphClientOptions(
            label_cache_size=self.graph_label_type_cache_size,
            type_cache_size=self.graph_label_type_cache_size,
            arcs_cache_size=self.graph_arcs_cache_size,
            obs_properties_cache_size=self.graph_obs_properties_cache_size,
            obs_cache_size=self.graph_obs_cache_size,
        )

    @classmethod
    def from_env(cls) -> DatasetsSettings:
        try:
            return cls()
        except ValidationError as exc:
            raise DatasetsConfigError(
                env_validation_error_message(exc, env_prefix="DATASETS_")
            ) from exc


# Advertised lifetime for mint, definition, and snapshot URLs. This fixed policy is not
# environment-configurable.
SIGNED_URL_ADVERTISED_TTL: timedelta = timedelta(minutes=15)

# Add the cache reuse window so cached URLs retain the full advertised lifetime.
SIGNED_URL_TTL: timedelta = SIGNED_URL_ADVERTISED_TTL + SIGNED_URL_REUSE_WINDOW

# Fail at import if the signing TTL exceeds the artifact store's ceiling.
if SIGNED_URL_TTL > SIGNED_URL_TTL_CEILING:
    raise ValueError(
        f"SIGNED_URL_TTL {SIGNED_URL_TTL!r} (SIGNED_URL_ADVERTISED_TTL + "
        f"SIGNED_URL_REUSE_WINDOW) exceeds artifact_store.signing's "
        f"{SIGNED_URL_TTL_CEILING!r} ceiling"
    )
