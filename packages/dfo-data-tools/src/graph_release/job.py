"""The graph release builder, run as `python -m graph_release.job` every two minutes.

Each tick:

1. Reads the import key in a multi-use read-only snapshot and exits unless it is buildable.
2. Reads the DC service's serving revisions and exits unless each was created after the import
   completed, since the mixer reads Spanner as of the last ingestion it saw at start.
3. Forms the inputs (the success row's execution id, the serving revision, `FORMAT`, the
   builder code hash and the declarations hash, and with `GRAPH_RELEASE_SHADOW` on the coverage
   and origin catalogue build ids) and exits when they match the last built inputs while the
   pointer names the release they built, when they match the last failed inputs, when the
   pointer is held, or when they ended unavailable less than `UNAVAILABLE_BACKOFF_SECONDS` ago,
   in which case it logs the recorded `graph_release_unavailable` line again with `backoff`.
   With shadow on, it exits `graph_release_shadow_waiting` while the coverage build or the
   origin catalogue is behind the import, before building anything, and exits unavailable once
   that wait passes `SHADOW_WAIT_LIMIT_SECONDS`.
4. Takes the build lease, or exits. A tick that exits here with inputs that ended unavailable
   logs their recorded `graph_release_unavailable` line again with `backoff`, since the lease
   holder is retrying them.
5. Reads every input in one snapshot whose first statements re-read the import key, and
   discards the build when the key moved since step 1.
6. Builds the platform and DFO parts and gates them, then re-reads the serving revision and
   discards the build, recording nothing, when it moved. A dependency unavailable during the
   build re-reads it at once, with the same discard. Otherwise a finding records the inputs as
   the last failed build, and an unavailable dependency records them as the last unavailable
   inputs, which only the backoff reads.
7. Publishes, unless the release id is the serving one, in which case it records the inputs
   and publishes nothing. Prunes while still holding the lease, then releases it. A failed
   prune or lease release, or a failed write of the built inputs after the pointer moved, is
   logged and leaves the outcome alone.
8. After a publish, starts an execution of the search catalogue job `SEARCH_CATALOGUE_JOB_NAME`
   names, when it is set, through the Cloud Run Admin API. A failed start logs
   `graph_release_catalogue_trigger_failed` and leaves the outcome alone.

Every outcome the tick classifies exits 0, a gate finding and an unavailable dependency
included, since their log lines raise the alerts. Only an exception the tick does not classify,
or a timeout or out-of-memory kill, fails the execution. Every tick logs
`graph_release_heartbeat` last, with outcome `error` before re-raising an exception it does not
classify. The no-heartbeat alert's metric leaves out outcomes `error` and `lease_held`. A build
killed by the task timeout or out of memory logs no heartbeat and keeps the lease, and the ticks
that overlap it end `lease_held`, so builds killed on every run raise the alert.

It logs `graph_release_stale` while the newest import of any status has waited more than
`STALE_SECONDS` for a release, less the time spent waiting for the shadow comparators up to
`SHADOW_WAIT_LIMIT_SECONDS`, unless the pointer is held.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Final, Protocol, Self

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from artifact_publication.jobs import (
    AliasOnlySettings,
    ImportKey,
    Lease,
    acquire_lease,
    emit,
    new_build_id,
    read_import_key,
    release_lease,
)
from artifact_publication.uri import (
    ArtifactUriError,
    join_uri,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes_if_unchanged,
)
from dc_graph import GraphClient
from dc_graph.errors import GraphCapExceededError, GraphReadError
from dc_sdmx.client import SdmxClient
from graph_release.declarations import DECLARATIONS, Declarations
from graph_release.dfo import (
    MissingBaseParties,
    base_party_ids,
    build_dfo_parts,
    dfo_inputs,
    read_base_parties,
    stage_dfo_parts,
)
from graph_release.format import (
    FORMAT,
    LoadedRelease,
    Manifest,
    Pointer,
    ReleaseFormatError,
    StagedPart,
    manifest_uri,
    open_release,
    pointer_uri,
)
from graph_release.gate import (
    Dependency,
    GateResult,
    GateUnavailable,
    SeriesReader,
    merge,
    refused,
    run_gate,
)
from graph_release.platform import PlatformRefusal, build_platform_parts, stage_platform_parts
from graph_release.publish import PublishStatus, prune_releases, publish_release
from graph_release.reads import Snapshot, read_ingested_imports, read_inputs, snapshot_at
from graph_release.serving_revision import CloudRunAdmin, ServingRevision, ServingRevisionError
from graph_release.shadow import ComparatorBuild, shadow_builds, shadow_diff

logger = logging.getLogger(__name__)

STATE_FILE: Final[str] = "builder_state.json"
"""At the release root: the last built inputs and the release they built, the last failed and
last unavailable inputs, and the shadow wait."""

LEASE_TTL_SECONDS: Final[float] = 1800.0
"""The job's task timeout. The lease has no renewal, so it lasts as long as a run can."""

STALE_SECONDS: Final[float] = 1800.0
"""How long the newest import may wait for a release before `graph_release_stale`. It sits
above the DC restart that follows each import."""

SHADOW_WAIT_LIMIT_SECONDS: Final[float] = 10800.0
"""How long a shadow wait is excluded from stale: the origin catalogue job's 2-hour timeout plus
ticks. Past it the tick is unavailable, naming each comparator that is behind."""

STATE_WRITE_ATTEMPTS: Final[int] = 5
"""How many times a state file change is read and written before `StateConflict`. A conflict
is another tick or `graph_release.hold` writing between the read and the write."""

UNAVAILABLE_BACKOFF_SECONDS: Final[float] = 600.0
"""How long ticks skip inputs whose build last ended unavailable, so an outage does not repeat
the full Spanner read every tick. Each skipped tick logs the unavailable line again."""

BUILDER_MODULES: Final[Mapping[str, tuple[str, ...]]] = {
    "graph_release": (
        "authored.py",
        "coverage.py",
        "declarations.py",
        "dfo.py",
        "format.py",
        "gate.py",
        "job.py",
        "platform.py",
        "publish.py",
        "reads.py",
        "serving_revision.py",
        "shadow.py",
        "views.py",
    ),
    "code_index": ("export.py", "loader.py", "query.py", "types.py"),
    "dc_graph": ("visible_arcs.py",),
    "series_coverage": ("index.py", "loader.py", "schema.py"),
}
"""The modules whose source `builder_code_hash` covers: what reads, builds, gates and publishes
a release, including the loaders whose refusals `graph_release.gate.check_views` applies. A change elsewhere that changes a part still changes the release id, but builds only
at the next import or DC redeploy."""


class StateConflict(Exception):
    """The state file changed under every one of `STATE_WRITE_ATTEMPTS` writes."""


_GRAPH_TIMEOUT_SECONDS: Final[float] = 120.0
_USER_AGENT: Final[str] = "dfo-graph-release"
_SPANNER_ERRORS: Final = (GoogleAPIError, GoogleAuthError)
"""What a Spanner read raises when Spanner, or the credentials it needs, cannot answer."""
_BUCKET_ERRORS: Final = (
    OSError,
    ArtifactUriError,
    GoogleAPIError,
    GoogleAuthError,
    ReleaseFormatError,
    StateConflict,
)
"""What a read or write under the release root, or of a comparator, raises when it fails."""
_MIXER_ERRORS: Final = (GraphReadError, GraphCapExceededError, MissingBaseParties)
"""What reading the base parties raises when the mixer cannot answer them."""


@cache
def builder_code_hash() -> str:
    """SHA-256 over the source of `BUILDER_MODULES`, each keyed by package and file name."""
    digest = hashlib.sha256()
    for package, modules in sorted(BUILDER_MODULES.items()):
        files = resources.files(package)
        for name in sorted(modules):
            data = files.joinpath(name).read_bytes()
            digest.update(f"{package}/{name}\0{len(data)}\0".encode())
            digest.update(data)
    return digest.hexdigest()


class JobSettings(AliasOnlySettings):
    """Job settings from the environment, failing fast at construction. Each field's `alias` is
    the only environment variable name read; construct by field name to override one."""

    model_config = SettingsConfigDict(frozen=True)

    release_uri: str = Field(min_length=1, alias="GRAPH_RELEASE_URI")
    spanner_project: str = Field(min_length=1, alias="GRAPH_RELEASE_SPANNER_PROJECT")
    spanner_instance: str = Field(min_length=1, alias="GRAPH_RELEASE_SPANNER_INSTANCE")
    spanner_database: str = Field(min_length=1, alias="GRAPH_RELEASE_SPANNER_DATABASE")
    datacommons_service: str = Field(
        pattern=r"^projects/[^/]+/locations/[^/]+/services/[^/]+$",
        alias="DATACOMMONS_SERVICE_NAME",
    )
    graph_base_url: str = Field(min_length=1, alias="DATASETS_GRAPH_BASE_URL")
    sdmx_base_url: str = Field(min_length=1, alias="DATASETS_SDMX_BASE_URL")
    shadow: bool = Field(default=False, alias="GRAPH_RELEASE_SHADOW")
    coverage_uri: str = Field(default="", alias="COVERAGE_ARTIFACT_URI")
    origin_uri: str = Field(default="", alias="ORIGIN_CATALOGUE_URI")
    search_catalogue_job: str = Field(
        default="",
        pattern=r"^(projects/[^/]+/locations/[^/]+/jobs/[^/]+)?$",
        alias="SEARCH_CATALOGUE_JOB_NAME",
    )

    @model_validator(mode="after")
    def _shadow_needs_comparators(self) -> Self:
        if self.shadow and not (self.coverage_uri and self.origin_uri):
            raise ValueError(
                "GRAPH_RELEASE_SHADOW needs COVERAGE_ARTIFACT_URI and ORIGIN_CATALOGUE_URI"
            )
        return self


@dataclass(frozen=True)
class BuildInputs:
    """What a release is built from. Equal inputs build an equal release, and gate it against
    equal comparators (with shadow on, the coverage and origin catalogue builds the diff reads),
    so republishing a faulty comparator lets a shadow finding build again."""

    import_execution_id: str
    dc_serving_revision: str
    format: int
    builder_code_hash: str
    declarations_hash: str
    coverage_build_id: str | None = None
    origin_build_id: str | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class Database(Protocol):
    """The part of a `google.cloud.spanner` `Database` the job uses."""

    def snapshot(self, **kwargs: Any) -> AbstractContextManager[Snapshot]: ...


class AdminApi(Protocol):
    """`CloudRunAdmin`."""

    def serving(self) -> ServingRevision: ...

    def created_at(self, revision: str) -> datetime: ...

    def run_job(self, job: str) -> str: ...


class Mixer(Protocol):
    """`dc_graph.GraphClient`'s arc reads, for the base parties and the mixer differential."""

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]: ...

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, Mapping[str, object] | None]: ...


@dataclass(frozen=True)
class _Exit:
    """How a tick ended. `failed` marks a gate finding or an unavailable dependency, which an
    unreadable stale check does not replace in the heartbeat."""

    outcome: str
    failed: bool = False


class _Unavailable(Exception):
    """A dependency that could not answer. The tick records no finding, and the next tick
    retries. `phases` holds the build phases run before it, when a build ran."""

    def __init__(self, *entries: GateUnavailable) -> None:
        super().__init__("; ".join(f"{entry.dependency}: {entry.detail}" for entry in entries))
        self.entries = entries
        self.phases: Mapping[str, Mapping[str, float]] = {}


@contextmanager
def _needs(dependency: Dependency, *errors: type[BaseException]) -> Iterator[None]:
    """Turns `errors` raised inside the block into `_Unavailable` naming `dependency`."""
    try:
        yield
    except errors as exc:
        raise _Unavailable(GateUnavailable(dependency, str(exc))) from exc


class _Phases:
    """Each build phase's wall time and the process's peak RSS when it ended. The peak is a
    high-water mark, so the phase where it rises is the one that raised it."""

    def __init__(self) -> None:
        self.recorded: dict[str, dict[str, float]] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.recorded[name] = {
                "seconds": round(time.monotonic() - started, 3),
                "peak_rss_bytes": _peak_rss_bytes(),
            }


def _peak_rss_bytes() -> int:
    """`ru_maxrss` in bytes. macOS reports bytes and Linux KiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def run(
    settings: JobSettings,
    *,
    database: Database,
    admin: AdminApi,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations = DECLARATIONS,
    clock: Callable[[], float] = time.time,
) -> None:
    """One tick. Every outcome it classifies, a gate finding and an unavailable dependency
    included, returns, so the process exits 0; their log lines raise the alerts. A tick that
    published starts the search catalogue job when `settings.search_catalogue_job` names one.
    Logs `graph_release_heartbeat` last, with outcome `error` and the exception type before
    re-raising anything the tick does not classify, which fails the execution."""
    try:
        result = _run(
            settings,
            database=database,
            admin=admin,
            graph=graph,
            sdmx=sdmx,
            declarations=declarations,
            clock=clock,
        )
        if result.outcome == "published" and settings.search_catalogue_job:
            _start_search_catalogue(settings.search_catalogue_job, admin)
    except Exception as exc:
        emit("graph_release_heartbeat", severity="ERROR", outcome="error", error=type(exc).__name__)
        raise
    emit("graph_release_heartbeat", outcome=result.outcome)


def _run(
    settings: JobSettings,
    *,
    database: Database,
    admin: AdminApi,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations,
    clock: Callable[[], float],
) -> _Exit:
    key: ImportKey | None = None
    try:
        with (
            _needs(Dependency.SPANNER, *_SPANNER_ERRORS),
            database.snapshot(multi_use=True) as snap,
        ):
            key = read_import_key(snap)
        if key is None:
            result = _skipped("no_import")
        else:
            result = _tick(
                settings,
                key,
                database=database,
                admin=admin,
                graph=graph,
                sdmx=sdmx,
                declarations=declarations,
                clock=clock,
            )
    except _Unavailable as exc:
        result = _unavailable(exc)
    if key is None:
        return result
    try:
        _track_stale(settings.release_uri, key, now=clock())
    except _Unavailable as exc:
        # Logged whatever the tick's outcome, so a stale check that cannot read stays visible.
        stale_check = _unavailable(exc)
        if not result.failed:
            result = stale_check
    return result


def _tick(
    settings: JobSettings,
    key: ImportKey,
    *,
    database: Database,
    admin: AdminApi,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations,
    clock: Callable[[], float],
) -> _Exit:
    if key.lock_owner is not None:
        return _skipped("ingestion_lock_held")
    success = key.newest_success
    if not key.buildable or success is None:
        return _skipped("import_not_successful", status=key.newest.status)

    with _needs(Dependency.ADMIN_API, ServingRevisionError):
        serving = admin.serving()
        created = min(admin.created_at(revision) for revision in serving.revisions)
    completed = success.completed_at
    if completed is None or created <= datetime.fromisoformat(completed):
        return _skipped(
            "serving_revision_before_import",
            revision=serving.token,
            revision_created_at=created.isoformat(),
            import_completed_at=completed,
        )

    comparators: tuple[ComparatorBuild, ...] = ()
    if settings.shadow:
        found = shadow_builds(coverage_uri=settings.coverage_uri, origin_uri=settings.origin_uri)
        if isinstance(found, GateUnavailable):
            raise _Unavailable(found)
        comparators = found
    builds = {build.dependency: build.build_id for build in comparators}
    inputs = BuildInputs(
        import_execution_id=success.workflow_execution_id,
        dc_serving_revision=serving.token,
        format=FORMAT,
        builder_code_hash=builder_code_hash(),
        declarations_hash=declarations.declarations_hash(),
        coverage_build_id=builds.get(Dependency.COVERAGE_BUILD),
        origin_build_id=builds.get(Dependency.ORIGIN_CATALOGUE),
    )
    root = settings.release_uri
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        state = _read_state(root)
        pointer, generation = _read_pointer(root)
    # The pointer must still name what these inputs built: after a hold is cleared it names the
    # held release, and the tick republishes the newer one.
    built = state.get("last_built") or {}
    if (
        built.get("inputs") == inputs.to_json()
        and pointer is not None
        and pointer.release_id == built.get("release_id")
    ):
        return _skipped("inputs_built")
    if state.get("last_failed") == inputs.to_json():
        return _skipped("inputs_failed")
    if pointer is not None and pointer.hold is not None:
        return _skipped("held", release_id=pointer.hold)
    unavailable = state.get("last_unavailable") or {}
    retrying = unavailable.get("inputs") == inputs.to_json()
    if retrying and clock() - float(unavailable["at"]) < UNAVAILABLE_BACKOFF_SECONDS:
        _repeat_unavailable(unavailable)
        return _Exit("unavailable_backoff")

    try:
        if comparators:
            behind = [build for build in comparators if build.stamp != key.stamp]
            with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
                waited = _note_shadow(root, state, stamp=key.stamp, ready=not behind, now=clock())
            if behind:
                return _shadow_waiting(key.stamp, behind, waited=waited)

        build_id = new_build_id()
        with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
            lease = acquire_lease(root, build_id=build_id, ttl_seconds=LEASE_TTL_SECONDS, now=clock)
        if lease is None:
            if retrying:
                # The lease holder is retrying these inputs, so the outage has not yet cleared.
                _repeat_unavailable(unavailable)
            return _skipped("lease_held")
        try:
            result = _build(
                settings,
                key,
                inputs,
                pointer=pointer,
                generation=generation,
                database=database,
                admin=admin,
                graph=graph,
                sdmx=sdmx,
                declarations=declarations,
                clock=clock,
            )
            # Pruning runs under the lease, so no publish can reuse a part it is deleting.
            _prune(root, now=clock())
        finally:
            _release(root, lease)
    except _Unavailable as exc:
        _note_unavailable(root, inputs, exc, now=clock())
        raise
    return result


def _start_search_catalogue(job_name: str, admin: AdminApi) -> None:
    """Starts an execution of the search catalogue job `job_name`, which keys its build on the
    release the pointer names. A failed start logs `graph_release_catalogue_trigger_failed` and
    leaves the tick's outcome alone, since the release is published; the next publish or deploy
    starts the job again."""
    try:
        operation = admin.run_job(job_name)
    except ServingRevisionError as exc:
        emit(
            "graph_release_catalogue_trigger_failed",
            severity="WARNING",
            job=job_name,
            detail=str(exc),
        )
        return
    emit("graph_release_catalogue_started", job=job_name, operation=operation)


def _repeat_unavailable(recorded: Mapping[str, Any]) -> None:
    """Logs the recorded `graph_release_unavailable` line again with `backoff`, so the unavailable
    alert sees the outage for as long as it lasts, the retry build included."""
    emit(
        "graph_release_unavailable",
        severity="WARNING",
        unavailable=recorded.get("unavailable", []),
        backoff=True,
    )


def _shadow_waiting(stamp: str, behind: Sequence[ComparatorBuild], *, waited: float) -> _Exit:
    """`shadow_waiting`, or unavailable for each comparator behind once the wait passes
    `SHADOW_WAIT_LIMIT_SECONDS`."""
    if waited > SHADOW_WAIT_LIMIT_SECONDS:
        raise _Unavailable(
            *(
                GateUnavailable(
                    build.dependency,
                    f"build {build.build_id} holds import {build.stamp!r}, behind import "
                    f"{stamp!r} for {round(waited)} s",
                )
                for build in behind
            )
        )
    emit("graph_release_shadow_waiting", stamp=stamp)
    return _Exit("shadow_waiting")


def _prune(root: str, *, now: float) -> None:
    """Prunes old releases. A failure is logged as `graph_release_prune_failed` and leaves the
    tick's outcome alone. The next build prunes again."""
    try:
        pruned = prune_releases(root, now=now)
    except _BUCKET_ERRORS as exc:
        emit("graph_release_prune_failed", severity="WARNING", detail=str(exc))
        return
    if pruned:
        emit("graph_release_pruned", deleted=list(pruned))


def _release(root: str, lease: Lease) -> None:
    """Releases the lease. A failure is logged and leaves the tick's outcome alone. The lease
    lapses after `LEASE_TTL_SECONDS`."""
    try:
        released = release_lease(root, lease)
    except _BUCKET_ERRORS:
        logger.warning("graph_release: could not release the lease", exc_info=True)
        return
    if not released:
        logger.warning("graph_release: the lease was reclaimed before release")


def _build(
    settings: JobSettings,
    key: ImportKey,
    inputs: BuildInputs,
    *,
    pointer: Pointer | None,
    generation: str | None,
    database: Database,
    admin: AdminApi,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations,
    clock: Callable[[], float],
) -> _Exit:
    """Steps 5 to 7, holding the lease. The event that ends the build carries each phase's wall
    time and peak RSS."""
    phases = _Phases()
    try:
        return _build_phases(
            settings,
            key,
            inputs,
            phases,
            pointer=pointer,
            generation=generation,
            database=database,
            admin=admin,
            graph=graph,
            sdmx=sdmx,
            declarations=declarations,
            clock=clock,
        )
    except _Unavailable as exc:
        exc.phases = phases.recorded
        raise


def _build_phases(
    settings: JobSettings,
    key: ImportKey,
    inputs: BuildInputs,
    phases: _Phases,
    *,
    pointer: Pointer | None,
    generation: str | None,
    database: Database,
    admin: AdminApi,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations,
    clock: Callable[[], float],
) -> _Exit:
    root = settings.release_uri
    read_at = datetime.fromtimestamp(clock(), UTC)
    with (
        phases("reads"),
        _needs(Dependency.SPANNER, *_SPANNER_ERRORS),
        snapshot_at(database, read_timestamp=read_at) as snapshot,
    ):
        raw = read_inputs(snapshot)
        ingested = read_ingested_imports(snapshot, execution_id=inputs.import_execution_id)
    if raw.import_key != key:
        emit("graph_release_discarded", reason="import_moved", **inputs.to_json())
        return _Exit("import_moved")

    with tempfile.TemporaryDirectory(prefix="graph-release-") as scratch:
        staging = Path(scratch) / "staging"
        execution = inputs.import_execution_id
        try:
            try:
                with phases("platform"):
                    platform = build_platform_parts(raw, authored=declarations.authored)
                    del raw  # the platform parts hold everything the build reads from here
                    staged = stage_platform_parts(
                        platform, staging_dir=staging, execution_id=execution
                    )
                with phases("dfo"):
                    release = dfo_inputs(platform, declarations)
                    party_ids = base_party_ids(release)
                    with _needs(Dependency.MIXER, *_MIXER_ERRORS):
                        base_parties = read_base_parties(party_ids, graph=graph)
                    dfo = build_dfo_parts(release, base_parties)
                    del release
                    staged.extend(stage_dfo_parts(dfo, staging_dir=staging, execution_id=execution))
            except PlatformRefusal as refusal:
                gate = refused(refusal)
            else:
                # The gate reads the staged files; only the shadow diff reads the built parts.
                shadow_parts = platform if settings.shadow else None
                del platform
                with phases("gate"):
                    gate = run_gate(
                        staged,
                        declarations=declarations,
                        previous=_open_previous(
                            root, pointer, local_dir=Path(scratch) / "previous"
                        ),
                        graph=graph,
                        sdmx=sdmx,
                        seed=key.stamp,
                    )
                if shadow_parts is not None:
                    # With shadow on the inputs name both comparator builds (step 3).
                    assert inputs.coverage_build_id is not None
                    assert inputs.origin_build_id is not None
                    with phases("shadow"):
                        gate = merge(
                            gate,
                            shadow_diff(
                                shadow_parts,
                                import_stamp=key.stamp,
                                role_axes=declarations.role_axes,
                                code_index_blocks=dfo.code_index_blocks,
                                coverage_uri=settings.coverage_uri,
                                coverage_build_id=inputs.coverage_build_id,
                                origin_uri=settings.origin_uri,
                                origin_build_id=inputs.origin_build_id,
                            ),
                        )
        except _Unavailable as exc:
            # The read that failed may have failed because the revision moved under it.
            if (moved := _serving_moved(admin, inputs, pending=exc)) is not None:
                return moved
            raise
        # The mixer reads above may have answered from a revision the inputs do not name, so a
        # moved revision discards whatever they decided, a finding or an unavailable read alike.
        if (moved := _serving_moved(admin, inputs)) is not None:
            return moved

        if gate.findings:
            with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
                _update_state(root, last_failed=inputs.to_json())
            emit(
                "graph_release_gate_failed",
                **inputs.to_json(),
                **gate.summary(),
                phases=phases.recorded,
            )
            return _Exit("gate_failed", failed=True)
        if gate.unavailable:
            raise _Unavailable(*gate.unavailable)

        return _publish(
            root,
            inputs,
            _manifest(inputs, key, staged, ingested=ingested, read_at=read_at, gate=gate),
            staged=staged,
            pointer=pointer,
            generation=generation,
            phases=phases,
        )


def _serving_moved(
    admin: AdminApi, inputs: BuildInputs, *, pending: _Unavailable | None = None
) -> _Exit | None:
    """Re-reads the serving revision and discards the build, logging
    `graph_release_discarded`, when it no longer equals the inputs'. With an unavailable
    dependency `pending`, an Admin API that cannot answer adds its entry to `pending`'s."""
    try:
        with _needs(Dependency.ADMIN_API, ServingRevisionError):
            serving = admin.serving()
    except _Unavailable as exc:
        if pending is None:
            raise
        raise _Unavailable(*pending.entries, *exc.entries) from pending
    if serving.token == inputs.dc_serving_revision:
        return None
    emit(
        "graph_release_discarded",
        reason="serving_revision_moved",
        revision=serving.token,
        **inputs.to_json(),
    )
    return _Exit("serving_revision_moved")


def _manifest(
    inputs: BuildInputs,
    key: ImportKey,
    staged: Sequence[StagedPart],
    *,
    ingested: tuple[str, ...],
    read_at: datetime,
    gate: GateResult,
) -> Manifest:
    success = key.newest_success
    assert success is not None and success.completed_at is not None  # checked at step 2
    return Manifest(
        format=inputs.format,
        import_execution_id=inputs.import_execution_id,
        import_completed_at=success.completed_at,
        ingested_imports=ingested,
        read_timestamp=read_at.isoformat(timespec="microseconds"),
        dc_serving_revision=inputs.dc_serving_revision,
        builder_code_hash=inputs.builder_code_hash,
        declarations_hash=inputs.declarations_hash,
        parts=tuple(part.ref for part in staged),
        gates=gate.summary(),
    )


def _publish(
    root: str,
    inputs: BuildInputs,
    manifest: Manifest,
    *,
    staged: Sequence[StagedPart],
    pointer: Pointer | None,
    generation: str | None,
    phases: _Phases,
) -> _Exit:
    built = {"inputs": inputs.to_json(), "release_id": manifest.release_id}
    if pointer is not None and manifest.release_id == pointer.release_id:
        with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
            _update_state(root, last_built=built)
        emit(
            "graph_release_unchanged",
            release_id=manifest.release_id,
            **inputs.to_json(),
            phases=phases.recorded,
        )
        return _Exit("release_unchanged")

    with phases("publish"), _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        outcome = publish_release(
            root, staged=staged, manifest=manifest, expected_pointer_generation=generation
        )
    match outcome.status:
        case PublishStatus.HELD:
            return _skipped("held")
        case PublishStatus.POINTER_MOVED:
            emit("graph_release_discarded", reason="pointer_moved", release_id=outcome.release_id)
            return _Exit("pointer_moved")
    try:
        _update_state(root, last_built=built)
    except _BUCKET_ERRORS as exc:
        # The pointer has moved, so the release is published. The next tick builds the same
        # inputs again, finds the release unchanged and records them.
        emit("graph_release_state_write_failed", severity="WARNING", detail=str(exc))
    rows: dict[str, int] = {}
    for ref in manifest.parts:
        if ref.rows is not None:
            rows[ref.name] = rows.get(ref.name, 0) + ref.rows
    emit(
        "graph_release_published",
        release_id=outcome.release_id,
        rows=rows,
        uploaded=len(outcome.uploaded),
        parts=len(manifest.parts),
        **inputs.to_json(),
        phases=phases.recorded,
    )
    return _Exit("published")


def _open_previous(root: str, pointer: Pointer | None, *, local_dir: Path) -> LoadedRelease | None:
    """The serving release's `edges`, the only part the gate reads from it."""
    if pointer is None:
        return None
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        return open_release(
            root, local_dir=local_dir, release_id=pointer.release_id, parts=("edges",)
        )


def _read_pointer(root: str) -> tuple[Pointer | None, str | None]:
    """The format's pointer and its version token, both `None` when none is published."""
    found = read_uri_bytes_and_generation(pointer_uri(root))
    if found is None:
        return None, None
    return Pointer.from_json(found[0]), found[1]


# --- State file and stale tracking -------------------------------------------


def clear_failed(root: str) -> dict[str, object] | None:
    """Drops the last failed inputs from the state file at `root`, so the next tick builds them
    again. Returns the inputs it dropped, or `None` when none were recorded. Raises
    `StateConflict` as `_change_state` does."""

    def drop(state: dict[str, Any]) -> dict[str, Any] | None:
        if "last_failed" not in state:
            return None
        return {name: value for name, value in state.items() if name != "last_failed"}

    return _change_state(root, drop).get("last_failed")


def _read_state(root: str) -> dict[str, Any]:
    """The state file, empty when it is missing or does not parse."""
    try:
        return _parse_state(read_uri_bytes(join_uri(root, STATE_FILE)))
    except FileNotFoundError:
        return {}


def _parse_state(data: bytes) -> dict[str, Any]:
    try:
        state = json.loads(data)
    except ValueError:
        logger.warning("graph_release: %s does not parse; starting it afresh", STATE_FILE)
        return {}
    return state if isinstance(state, dict) else {}


def _change_state(
    root: str, change: Callable[[dict[str, Any]], dict[str, Any] | None]
) -> dict[str, Any]:
    """Applies `change` to the state file, writing only while the file holds the version
    `change` read. Ticks write the shadow wait and the unavailable inputs without the lease, so
    another writer can land between the read and the write, and the change is then applied
    again to what that writer left. `change` returns `None` to write nothing. Returns the state
    `change` applied to. Raises `StateConflict` when all `STATE_WRITE_ATTEMPTS` writes find the
    file changed."""
    uri = join_uri(root, STATE_FILE)
    for _ in range(STATE_WRITE_ATTEMPTS):
        found = read_uri_bytes_and_generation(uri)
        state, generation = ({}, None) if found is None else (_parse_state(found[0]), found[1])
        changed = change(state)
        if changed is None:
            return state
        data = json.dumps(changed, sort_keys=True).encode("utf-8")
        if write_uri_bytes_if_unchanged(uri, data, expected_generation=generation):
            return state
    raise StateConflict(f"{uri} changed under {STATE_WRITE_ATTEMPTS} writes")


def _update_state(root: str, **fields: object) -> None:
    """Rewrites `fields` in the state file, keeping the other fields. Raises `StateConflict` as
    `_change_state` does."""
    _change_state(root, lambda state: state | fields)


def _note_unavailable(root: str, inputs: BuildInputs, exc: _Unavailable, *, now: float) -> None:
    """Records the inputs whose tick ended unavailable and what was unavailable, for the backoff.
    A bucket that cannot take the write is the outage itself, so the failure is logged and the
    backoff skipped."""
    recorded = {"inputs": inputs.to_json(), "at": now, "unavailable": _entries(exc)}
    try:
        _update_state(root, last_unavailable=recorded)
    except _BUCKET_ERRORS:
        logger.warning("graph_release: could not record the unavailable inputs", exc_info=True)


def _note_shadow(
    root: str, state: Mapping[str, Any], *, stamp: str, ready: bool, now: float
) -> float:
    """Records when the comparators were first seen behind `stamp`, and when they caught up.
    Returns how long they have been behind, 0 once they caught up."""
    noted = state.get("shadow") or {}
    if noted.get("stamp") != stamp:
        if not ready:
            _update_state(root, shadow={"stamp": stamp, "waiting_since": now, "ready_at": None})
        return 0.0
    if ready:
        if noted.get("waiting_since") is not None and noted.get("ready_at") is None:
            _update_state(root, shadow={**noted, "ready_at": now})
        return 0.0
    return now - float(noted["waiting_since"])


def _shadow_wait(state: Mapping[str, Any], stamp: str, now: float) -> float:
    """Seconds spent waiting for the shadow comparators at `stamp`, up to
    `SHADOW_WAIT_LIMIT_SECONDS`."""
    noted = state.get("shadow") or {}
    if noted.get("stamp") != stamp or noted.get("waiting_since") is None:
        return 0.0
    ended = noted.get("ready_at")
    waited = (now if ended is None else float(ended)) - float(noted["waiting_since"])
    return min(waited, SHADOW_WAIT_LIMIT_SECONDS)


def _track_stale(root: str, key: ImportKey, *, now: float) -> None:
    """Logs `graph_release_stale` while the newest import, of any status, has completed more
    than `STALE_SECONDS` ago, less its shadow wait, and the serving release is not built from
    it. A failed import never gets a release, so it raises the line until the next success. A
    held pointer is a deliberate rollback and raises nothing."""
    newest = key.newest
    if newest.completed_at is None:
        return
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        pointer, _ = _read_pointer(root)
        if pointer is not None:
            if pointer.hold is not None:
                return
            serving = Manifest.from_json(read_uri_bytes(manifest_uri(root, pointer.release_id)))
            if serving.import_execution_id == newest.workflow_execution_id:
                return
        state = _read_state(root)
    completed = datetime.fromisoformat(newest.completed_at).timestamp()
    waited = now - completed - _shadow_wait(state, key.stamp, now)
    if waited > STALE_SECONDS:
        emit(
            "graph_release_stale",
            import_execution_id=newest.workflow_execution_id,
            status=newest.status,
            failed=newest.failed,
            waited_seconds=round(waited, 1),
        )


# --- Exits -------------------------------------------------------------------


def _skipped(reason: str, **fields: object) -> _Exit:
    emit("graph_release_skipped", reason=reason, **fields)
    return _Exit(reason)


def _entries(exc: _Unavailable) -> list[dict[str, str]]:
    return [{"dependency": str(e.dependency), "detail": e.detail} for e in exc.entries]


def _unavailable(exc: _Unavailable) -> _Exit:
    phases = {"phases": exc.phases} if exc.phases else {}
    emit("graph_release_unavailable", severity="WARNING", unavailable=_entries(exc), **phases)
    return _Exit("unavailable", failed=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m graph_release.job")
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="publish under this local directory in place of GRAPH_RELEASE_URI",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    overrides = {"release_uri": args.local_dir.resolve().as_uri()} if args.local_dir else {}
    settings = JobSettings(**overrides)

    from google.cloud import spanner

    database = (
        spanner.Client(project=settings.spanner_project)
        .instance(settings.spanner_instance)
        .database(settings.spanner_database)
    )
    graph = GraphClient(
        base_url=settings.graph_base_url, timeout=_GRAPH_TIMEOUT_SECONDS, user_agent=_USER_AGENT
    )
    sdmx = SdmxClient(
        base_url=settings.sdmx_base_url, timeout=_GRAPH_TIMEOUT_SECONDS, user_agent=_USER_AGENT
    )
    admin = CloudRunAdmin(service=settings.datacommons_service)
    with graph, sdmx, closing(admin):
        run(settings, database=database, admin=admin, graph=graph, sdmx=sdmx)


if __name__ == "__main__":
    main()


__all__ = [
    "BUILDER_MODULES",
    "LEASE_TTL_SECONDS",
    "SHADOW_WAIT_LIMIT_SECONDS",
    "STALE_SECONDS",
    "STATE_FILE",
    "STATE_WRITE_ATTEMPTS",
    "UNAVAILABLE_BACKOFF_SECONDS",
    "AdminApi",
    "BuildInputs",
    "Database",
    "JobSettings",
    "Mixer",
    "StateConflict",
    "builder_code_hash",
    "clear_failed",
    "main",
    "run",
]
