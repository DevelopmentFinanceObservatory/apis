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

from artifact_store.signing import SIGNED_URL_REUSE_WINDOW
from dfo_platform.config import (
    CallerSettings,
    PublicSiteSettings,
    env_validation_error_message,
)
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

# `DATASETS_OPERATION_DEADLINE_SECONDS` where the environment sets none, derived at
# `DatasetsBuildSettings.operation_deadline_seconds`.
DEFAULT_OPERATION_DEADLINE_SECONDS = 36_000


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

    # Read executor threads: snapshot, observations, operations polls, members, pin, register,
    # recipe, and head hits. A same-sized semaphore refuses retryable 503 when full.
    read_max_concurrency: int = Field(default=8, ge=1)

    # Work executor threads: mints, rebuilds, publishes, and frame refreshes. Kept apart from the
    # read executor, so a queue of materialisations cannot delay a poll or a download. A
    # same-sized semaphore refuses retryable 503 when full. At most `materialise_max_concurrency`
    # of the threads that mint, rebuild or refresh hold a frame at once. A publish takes no
    # materialise permit, so every thread that runs one holds its frame as well.
    work_max_concurrency: int = Field(default=4, ge=1)

    # Concurrent SDMX members per request. The surface's one SDMX client caps total GETs at 8.
    mint_member_max_concurrency: int = Field(default=DEFAULT_MEMBER_CONCURRENCY, ge=1)

    # A mint's deadline. Every SDMX, graph and metadata read of the mint takes it, and the door
    # answers timeout (504, retryable) when it passes.
    mint_timeout_seconds: float = Field(default=120, gt=0)

    # SDMX transport timeout. Each request also has a deadline, and the earlier one applies.
    sdmx_timeout_seconds: int = Field(default=90, ge=1)

    # Pivot cell limit. Format column caps bound width only, and CSV/Parquet have no width cap.
    pivot_max_output_cells: int = Field(default=5_000_000, ge=1)

    # Frame rows a groupby that sums a canonical frame folds as Arrow. A larger frame refuses
    # over_ceiling (rows). The Arrow fold peaks at about 1 KiB a row above the frame it reads
    # (505 MiB over spec_09ff's 488,979 rows), so a fold at this ceiling takes about 2 GiB.
    groupby_fold_max_rows: int = Field(default=2_000_000, ge=1)

    # Frame rows a groupby that sums a frame1 document folds row by row. That fold peaks at
    # about 2.8 KiB a group (1.39 GiB over spec_09ff's 488,979 groups), and a frame can keep one
    # group a row, so this ceiling holds it to the Arrow fold's 2 GiB.
    groupby_row_fold_max_rows: int = Field(default=700_000, ge=1)

    # Groupby folds the process runs at once. Permits times the fold's 2 GiB at its ceiling must
    # fit the service's fold memory, so the 4 GiB service takes one. A further collapse waits.
    groupby_fold_max_concurrency: int = Field(default=1, ge=1)

    # The deadline of a definition GET, `/options` call, or frame refresh, taken as a mint's is.
    # On a head hit it also covers on-demand rendering and storage.
    definition_timeout_seconds: float = Field(default=120, gt=0)

    # Materialise permits per instance. The frames alive at once across mints, rebuilds, head-hit
    # renders, file renewals and observation reads. A serve holds one from its first SDMX request
    # to its last artifact write, and a graph release swap holds every one. A publish builds its
    # frame outside them, bounded by `work_max_concurrency`. The service has 4 GiB. The default
    # of 2 stays until a load test measures resident memory per materialisation.
    materialise_max_concurrency: int = Field(default=2, ge=1)

    # Slot wait limit. The semaphore uses the earlier of this limit and the request deadline (503).
    materialise_wait_seconds: float = Field(default=30, gt=0)

    # Analysis executor threads (`datasets_api.doors.analysis`): recipe trees the runner
    # computes. A same-sized semaphore refuses retryable 503 when full. A run reads its root
    # frames, computes and stores under one materialise permit, so at most
    # `materialise_max_concurrency` runs hold frames at once across every frame path.
    analysis_max_concurrency: int = Field(default=2, ge=1)

    # A run's deadline, from admission to its answer. A run past it answers timeout (504,
    # retryable) and its thread finishes and stores its frame and answer. A retry while it runs
    # joins it (202, `Retry-After` 5), and a retry after it ends answers the stored body.
    analysis_timeout_seconds: float = Field(default=120, gt=0)

    # The most nodes one tree holds, steps and roots, after each derived root is expanded into
    # its stored recipe.
    analysis_max_nodes: int = Field(default=64, ge=1)

    # The most rows a tree's root frames hold together. Every verb but `join` writes no more
    # rows than its `data` side holds, and a `join` writes at most `analysis_max_join_rows`, so
    # the two bound every frame a run holds.
    analysis_max_root_rows: int = Field(default=2_000_000, ge=1)
    analysis_max_join_rows: int = Field(default=250_000, ge=1)

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

    # New pinned copies the pin door and the share routes make per UTC day
    # (`datasets_api.retention_admission`), a budget spent as an allowance is
    # (`datasets_api.allowance`). A repin of an already pinned artifact and a serving-frame copy
    # spend none. 0 refuses every new pinned copy.
    pin_daily_max: int = Field(default=2_500, ge=0)

    # New pinned copies callers without the pin grant may make together per UTC day, spent with
    # `pin_daily_max` and the caller's `pin_daily_allowance` in one transaction. Their copies
    # spend `pin_daily_max` as well, so open callers leave the pin grant's holders at least
    # `pin_daily_max` minus this.
    pin_open_daily_max: int = Field(default=2_000, ge=0)

    # New pinned copies one caller without the pin grant may make per UTC day
    # (`datasets_api.allowance`), through the pin door or a share's pins
    # (`datasets_api.retention_admission`). A repin of an already pinned artifact spends none.
    pin_daily_allowance: int = Field(default=100, ge=0)

    # Shares one caller may write per UTC day (`datasets_api.allowance`).
    share_daily_allowance: int = Field(default=10, ge=0)

    # Presentation card creates and revisions one caller may write per UTC day
    # (`datasets_api.allowance`). A caller holding the publish grant spends none.
    card_daily_allowance: int = Field(default=500, ge=0)

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

    # Sole principal of the live-refresh routes: the frame refresh here and the Presentation
    # render routes. Read from `LIVE_REFRESH_PRINCIPAL`, the name the website and `ai` mount
    # the same secret under. Empty denies all. It opens no publish, recipe, or pin route.
    live_refresh_principal: str = Field(default="", validation_alias="live_refresh_principal")

    # The queue's `max_attempts`, which `infra` sets from the queue's own `max_attempts` local
    # (3). The worker marks the operation failed on its final transient attempt.
    operation_max_attempts: int = Field(default=3, ge=1)

    # Maximum live pending/running rows. `create_or_get` refuses a new operation at the limit.
    operation_backlog_max: int = Field(default=20, ge=1)

    # The application's deadline for one submission of an operation, counted from its dispatch,
    # or its submission when no dispatch is recorded. A `pending` row, or a `running` row whose
    # lease has ended, that is still unfinished after it fails retryable at the next poll, a
    # delivery that arrives later fails the row without running it, and the backlog stops
    # counting the row. It says nothing about the queue's state. Cloud Tasks may still hold the
    # task. The default lets every delivery wait behind a full backlog and then run to its
    # dispatch deadline, with the backoff between deliveries and a margin. `infra` computes it
    # from the queue's own locals as
    #   attempts x (ceil(backlog_max / dispatch_concurrency) x dispatch_deadline
    #     + dispatch_deadline) + (attempts - 1) x backoff + margin
    # = 3 x (ceil(20 / 4) x 1,800 + 1,800) + 2 x 1,680 + 240 = 36,000 s.
    operation_deadline_seconds: int = Field(default=DEFAULT_OPERATION_DEADLINE_SECONDS, ge=1)

    # The website bundles link to, read from `DFO_PUBLIC_SITE_URL` (required) and
    # `DFO_SITE_DATASET_LINKS`, which the presentation surface reads too.
    site: PublicSiteSettings = Field(default_factory=PublicSiteSettings.from_env)

    # How a request's caller is named for its allowances, read from `DFO_CALLER_PRINCIPAL`,
    # `DFO_CALLER_HMAC_KEY` and `DFO_TRUSTED_PROXY_HOPS` (`dfo_platform.config.CallerSettings`).
    caller: CallerSettings = Field(default_factory=CallerSettings.from_env)

    @model_validator(mode="after")
    def _check_head_recheck_before_max_age(self) -> DatasetsBuildSettings:
        """Require the probe interval to be shorter than the maximum head age."""
        if self.head_recheck_seconds >= self.head_max_age_seconds:
            raise ValueError(
                "DATASETS_HEAD_RECHECK_SECONDS must be < DATASETS_HEAD_MAX_AGE_SECONDS "
                f"({self.head_max_age_seconds}), got {self.head_recheck_seconds}"
            )
        return self

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


# Lifetime of the signed URL a download link redirects to. The artifact store reuses a signed URL
# for up to `SIGNED_URL_REUSE_WINDOW`, so a redirect carries at least the difference.
DOWNLOAD_SIGNED_URL_TTL: timedelta = timedelta(minutes=5)

# Fail at import if a reused signed URL could reach a client already expired.
if DOWNLOAD_SIGNED_URL_TTL <= SIGNED_URL_REUSE_WINDOW:
    raise ValueError(
        f"DOWNLOAD_SIGNED_URL_TTL {DOWNLOAD_SIGNED_URL_TTL!r} does not outlast artifact_store."
        f"signing's {SIGNED_URL_REUSE_WINDOW!r} reuse window"
    )
