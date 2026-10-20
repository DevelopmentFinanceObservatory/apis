"""Process-wide coverage loading and refresh.

The service checks `current.json` at most once per `COVERAGE_RECHECK_SECONDS`
and reloads only when `build_id` changes. Refresh is request-triggered
because background loops starve when `cpu_idle=true`. `current()` and
`snapshot()` return the loaded artifact without waiting. `snapshot_current()`
waits for the published build before returning.

The service serves only a build that carries the packaged code index's
provenances (`series_coverage.compatibility.index_coverage_differences`),
since dataset election reads the index and a mint reads the build. A
published build that differs is rejected and logged with both build ids and
the differing variables. The service then keeps the build it already
serves, moving to the build the index's compatibility pointer names only when
that build was published later, so the served build never moves back to older
data. A build id starts with the UTC time of its build
(`artifact_publication.jobs.new_build_id`), and a publish commits only over
the pointer it read before building, so build ids sort in publication order.
At startup the service loads the pointer's build, and with none that
qualifies, construction raises `CoverageIncompatibleError` and the surface
fails readiness. A rejected build is not loaded again while `current.json`
still names it.

`COVERAGE_INDEX_CHECK=off` turns the check off for a harness that pairs a
coverage build with an index on purpose: the test slice in CI and the eval
gate, or a local run against an index not yet deployed. Every load then logs
a WARNING naming the index and the build it serves unchecked. Any value other
than unset or `off` refuses at startup. Terraform and the deploy never set it.

A load runs inside every registered load guard (`register_load_guard`), so a
surface can keep its memory-heavy work from overlapping one: the Datasets surface
holds every materialise permit from before the new index loads until the service
serves it and holds no reference to the old one. A `CoverageSnapshot` a caller took
earlier keeps its index resident past that point. On the Datasets surface a request
that holds a permit builds its frame beside the one index the service serves, since it
asks `serves` once it holds the permit and gives the permit back on a replaced
snapshot. A request waiting for a permit or for its turn at a shared rebuild asks
`serves` in slices of the wait and releases a replaced index within one slice of the
reload. Before it waits, a request can hold a replaced index for as long as its head
checks, its graph reads, member resolution, the estimate and the budget decision take,
beside the frames other requests build.

With `GRAPH_RELEASE_COVERAGE` on (`graph_release.context.coverage_views_enabled`),
`get_coverage_service()` returns `ReleaseCoverage` and reads none of the variables above. It
serves the coverage view of the release bound to the current request or operation
(`graph_release.context.current_views`), whose code index comes from the same release, so no
compatibility check runs. Its stamp and build id are the release id. A release swap holds every
materialise permit through the release service's swap guard (`ReleaseService.register_swap_guard`),
so `ReleaseCoverage` registers no load guard of its own.

For the 1,782,460-row artifact, a full load takes 0.43 to 0.47 s and two
concurrent loads use up to 1.72 GiB. The compatibility check over 22,208
index rows takes 0.20 to 0.25 s and runs only after a load. A pointer read
takes about 11 microseconds.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact_publication.jobs import Ingestion
from code_index.query import CodeIndex
from code_index.shared import get_index
from dfo_platform.singleton import RefCountedSingleton
from graph_release.context import coverage_views_enabled, current_release, current_views
from series_coverage.compatibility import (
    IndexCoverageDifferences,
    describe_differences,
    index_coverage_differences,
    read_compatible_pointer,
)
from series_coverage.index import CoverageIndex
from series_coverage.loader import LoadedCoverage, load_coverage, read_published_pointer
from series_coverage.schema import CoverageManifest, CoveragePointer

if TYPE_CHECKING:
    from graph_release.service import Release, ReleaseViews

logger = logging.getLogger(__name__)

_ARTIFACT_URI_ENV_VAR = "COVERAGE_ARTIFACT_URI"
_RECHECK_SECONDS_ENV_VAR = "COVERAGE_RECHECK_SECONDS"
_INDEX_CHECK_ENV_VAR = "COVERAGE_INDEX_CHECK"
DEFAULT_RECHECK_SECONDS = 300.0


class _CoverageEnv(BaseSettings):
    """Env reads for `get_coverage_service()`, constructed fresh on each
    call so a `COVERAGE_ARTIFACT_URI` set after import is still honoured.
    """

    model_config = SettingsConfigDict()

    artifact_uri: str | None = Field(default=None, alias=_ARTIFACT_URI_ENV_VAR)
    recheck_seconds: float = Field(default=DEFAULT_RECHECK_SECONDS, alias=_RECHECK_SECONDS_ENV_VAR)
    index_check: str | None = Field(default=None, alias=_INDEX_CHECK_ENV_VAR)


class CoverageArtifactUriMissingError(RuntimeError):
    """Raised at surface construction when `COVERAGE_ARTIFACT_URI` is unset.

    Unreadable artifacts raise `CoverageArtifactError`.
    """


class CoverageIndexCheckValueError(RuntimeError):
    """Raised at surface construction when `COVERAGE_INDEX_CHECK` holds a value other than
    `off`."""


class CoverageIncompatibleError(RuntimeError):
    """Raised at surface construction when the published build does not carry the code
    index's provenances and no build the index's compatibility pointer names does."""


class CoverageLoader(Protocol):
    """Loads the build `pointer` names, or the published one when it is `None`."""

    def __call__(
        self, base_uri: str, *, pointer: CoveragePointer | None = None
    ) -> LoadedCoverage: ...


# Logs the first reload failure and each tenth consecutive failure.
_FAILURE_LOG_EVERY_N = 10


@dataclass(frozen=True, slots=True)
class _Snapshot:
    loaded: LoadedCoverage
    checked_at: float


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """Index, manifest, and stamp read atomically from one loaded artifact.

    Use one snapshot when a request needs multiple fields. Separate reads can
    straddle a background swap and mix artifacts.
    """

    index: CoverageIndex
    manifest: CoverageManifest
    stamp: str


class ServedCoverage(Protocol):
    """The coverage a surface serves: `CoverageService` over a published coverage build, or
    `ReleaseCoverage` over the bound graph release's coverage view."""

    def current(self) -> CoverageIndex: ...

    @property
    def stamp(self) -> str: ...

    @property
    def manifest(self) -> CoverageManifest: ...

    def snapshot(self) -> CoverageSnapshot: ...

    def snapshot_current(self) -> CoverageSnapshot: ...

    def serves(self, snapshot: CoverageSnapshot) -> bool: ...

    def register_load_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]: ...


class CoverageService:
    """Loads one artifact compatible with `index` at construction and refreshes it in the
    background.

    `recheck_seconds <= 0` disables background refresh. Raises `CoverageIncompatibleError`
    when neither the published build nor the build `index`'s compatibility pointer names
    carries `index`'s provenances. `index_check=False` serves the published build unchecked
    and logs a WARNING on every load.
    """

    def __init__(
        self,
        *,
        artifact_uri: str,
        index: CodeIndex,
        index_check: bool = True,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        loader: CoverageLoader = load_coverage,
    ) -> None:
        self._artifact_uri = artifact_uri
        self._index = index
        self._index_check = index_check
        self._recheck_seconds = recheck_seconds
        self._clock = clock
        self._loader = loader
        self._refresh_lock = threading.Lock()
        # Serialises pointer reads and loads, avoiding duplicate full loads.
        self._load_lock = threading.Lock()
        self._in_flight: concurrent.futures.Future[None] | None = None
        self._guards_lock = threading.Lock()
        self._load_guards: tuple[Callable[[], AbstractContextManager[object]], ...] = ()
        self._consecutive_failures = 0
        # The published build last found incompatible with `index`, while `current.json`
        # names it.
        self._rejected: CoverageManifest | None = None
        # `current()` reads this reference lock-free. The GIL makes its assignment
        # and read atomic, matching `GraphGeneration`'s `_snapshot` field.
        self._snapshot = _Snapshot(loaded=self._compatible_build(served=None), checked_at=clock())

    def current(self) -> CoverageIndex:
        """Returns the last loaded index without waiting for I/O.

        A due recheck runs in a daemon thread. Use `snapshot()` when the
        request also needs the matching manifest or stamp.
        """
        self._maybe_start_refresh()
        return self._snapshot.loaded.index

    @property
    def stamp(self) -> str:
        """Data-version stamp of the currently served artifact."""
        return self._snapshot.loaded.manifest.stamp

    @property
    def manifest(self) -> CoverageManifest:
        return self._snapshot.loaded.manifest

    @property
    def rejected_build(self) -> str | None:
        """The build id `current.json` names when that build differs from the code index, or
        `None` when the service serves the published build."""
        rejected = self._rejected
        return None if rejected is None else rejected.build_id

    def snapshot(self) -> CoverageSnapshot:
        """Returns matching index, manifest, and stamp from one snapshot.

        Starts a due background recheck without waiting for I/O.
        """
        self._maybe_start_refresh()
        loaded = self._snapshot.loaded
        return CoverageSnapshot(
            index=loaded.index, manifest=loaded.manifest, stamp=loaded.manifest.stamp
        )

    def snapshot_current(self) -> CoverageSnapshot:
        """Reads `current.json` and returns a snapshot at least as recent as
        that pointer, or the compatible build served in place of a rejected
        one. Waits for an in-flight reload. Pointer and loader errors
        propagate, leaving the served artifact unchanged.
        """
        loaded = self._load_if_moved()
        return CoverageSnapshot(
            index=loaded.index, manifest=loaded.manifest, stamp=loaded.manifest.stamp
        )

    def serves(self, snapshot: CoverageSnapshot) -> bool:
        """Whether `snapshot` holds the index the service serves, without waiting for I/O.

        A load replaces the served index only inside every registered load guard, so the
        answer holds for as long as the caller excludes a guard's scope.
        """
        return self._snapshot.loaded.index is snapshot.index

    def register_load_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]:
        """Run every later load of a moved build inside a fresh `guard()`, entered once the
        pointer read finds the move and left once the new build is served and the old one
        dropped. Returns the function that unregisters it.

        A caller of `snapshot_current` that holds what a guard takes waits on itself, so none
        may.
        """
        with self._guards_lock:
            self._load_guards = (*self._load_guards, guard)

        def unregister() -> None:
            with self._guards_lock:
                self._load_guards = tuple(g for g in self._load_guards if g is not guard)

        return unregister

    def close(self) -> None:
        """Waits for an in-flight background reload to finish."""
        with self._refresh_lock:
            future = self._in_flight
        if future is not None:
            future.result()

    def _maybe_start_refresh(self) -> None:
        if self._recheck_seconds <= 0:
            return
        now = self._clock()
        if now - self._snapshot.checked_at < self._recheck_seconds:
            return
        with self._refresh_lock:
            if self._in_flight is not None:
                return
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._in_flight = future
        threading.Thread(
            target=self._run_refresh, args=(future,), daemon=True, name="coverage-recheck"
        ).start()

    def _run_refresh(self, future: concurrent.futures.Future[None]) -> None:
        try:
            self._run_load()
        finally:
            with self._refresh_lock:
                self._in_flight = None
            future.set_result(None)

    def _run_load(self) -> None:
        """Runs `_load_if_moved`. Failures retain the last good snapshot and
        reset the recheck interval.
        """
        try:
            self._load_if_moved()
        except Exception as exc:
            self._record_reload_failure(exc)
            return
        self._consecutive_failures = 0

    def _load_if_moved(self) -> LoadedCoverage:
        """Reads the pointer and reloads only when `build_id` names neither the
        served build nor the rejected one.

        The load lock serialises refreshes. A load runs inside every registered guard, which
        it leaves only after dropping its reference to the build it replaced. Errors leave
        `_snapshot` unchanged. The returned artifact is the one this call loaded or retained.
        """
        with self._load_lock:
            served = self._snapshot.loaded
            rejected = self._rejected
            published = read_published_pointer(self._artifact_uri)
            if published is not None and published.build_id in (
                served.manifest.build_id,
                None if rejected is None else rejected.build_id,
            ):
                self._snapshot = _Snapshot(loaded=served, checked_at=self._clock())
                return served
            replaced = served.manifest
            with ExitStack() as guards:
                with self._guards_lock:
                    registered = self._load_guards
                for guard in registered:
                    guards.enter_context(guard())
                loaded = self._compatible_build(served=served)
                self._snapshot = _Snapshot(loaded=loaded, checked_at=self._clock())
                changed = loaded is not served
                del served
        if changed:
            logger.info(
                "coverage: served artifact changed (stamp %r build %r -> stamp %r build %r)",
                replaced.stamp,
                replaced.build_id,
                loaded.manifest.stamp,
                loaded.manifest.build_id,
            )
        return loaded

    def _compatible_build(self, *, served: LoadedCoverage | None) -> LoadedCoverage:
        """Loads the published build and returns it when it carries the index's provenances.

        Otherwise records it as rejected and returns `served`, or the build the index's
        compatibility pointer names when that build carries them and its build id sorts after
        `served`'s (or nothing is served). Each rejection logs an error naming the builds and the
        differing variables. Raises `CoverageIncompatibleError` when no build qualifies and
        nothing is served.
        """
        published = self._loader(self._artifact_uri)
        if not self._index_check:
            logger.warning(
                "coverage: %s=off; serving build %r (stamp %r) at %r without checking it "
                "against the code index (content_hash %r)",
                _INDEX_CHECK_ENV_VAR,
                published.manifest.build_id,
                published.manifest.stamp,
                self._artifact_uri,
                self._index.stamp.content_hash,
            )
            return published
        differences = index_coverage_differences(
            self._index, published.index.variable_provenances()
        )
        if not differences:
            self._rejected = None
            return published
        rejected = published.manifest
        # Free the rejected build before a fallback load.
        del published
        fallback = served
        pointer = read_compatible_pointer(self._artifact_uri, self._index)
        if pointer is not None and (served is None or pointer.build_id > served.manifest.build_id):
            candidate = self._loader(self._artifact_uri, pointer=pointer)
            candidate_differences = index_coverage_differences(
                self._index, candidate.index.variable_provenances()
            )
            if candidate_differences:
                self._log_rejection(candidate.manifest, candidate_differences, serving=served)
            else:
                fallback = candidate
        self._log_rejection(rejected, differences, serving=fallback)
        if fallback is None:
            named = (
                "no build" if pointer is None else f"build {pointer.build_id!r}, which differs too"
            )
            raise CoverageIncompatibleError(
                f"the coverage build {rejected.build_id!r} at {self._artifact_uri!r} differs "
                f"from the code index (content_hash {self._index.stamp.content_hash!r}), and "
                f"the index's compatibility pointer names {named}; re-export the index "
                "against the current build"
            )
        self._rejected = rejected
        return fallback

    def _log_rejection(
        self,
        rejected: CoverageManifest,
        differences: IndexCoverageDifferences,
        *,
        serving: LoadedCoverage | None,
    ) -> None:
        logger.error(
            "coverage: build %r (stamp %r) at %r differs from the code index (content_hash "
            "%r, exported from build %r) and is not served; serving build %r:\n  %s",
            rejected.build_id,
            rejected.stamp,
            self._artifact_uri,
            self._index.stamp.content_hash,
            self._index.stamp.coverage_build,
            None if serving is None else serving.manifest.build_id,
            "\n  ".join(describe_differences(differences)),
        )

    def _record_reload_failure(self, exc: Exception) -> None:
        """Logs the first and every tenth consecutive failure, retaining the
        last good artifact and resetting its check time. `_load_lock` prevents
        a failed refresh from overwriting a newer snapshot.
        """
        self._consecutive_failures += 1
        if (
            self._consecutive_failures == 1
            or self._consecutive_failures % _FAILURE_LOG_EVERY_N == 0
        ):
            logger.warning(
                "coverage: background recheck of %r failed (%d consecutive failure(s)), "
                "still serving stamp %r: %s",
                self._artifact_uri,
                self._consecutive_failures,
                self.stamp,
                exc,
            )
        with self._load_lock:
            self._snapshot = _Snapshot(loaded=self._snapshot.loaded, checked_at=self._clock())


class ReleaseCoverage:
    """The coverage view of the graph release bound to the current context, read on every call,
    so a request serves the coverage of the release it bound and the next request after a swap
    serves the new one. Holds no index, so a swapped-out release dies with the last request that
    bound it.

    Every read raises `graph_release.context.ReleaseNotBound` when no release is bound and
    `ReleaseViewsNotBuilt` when the bound release carries no views.
    """

    def current(self) -> CoverageIndex:
        return self.snapshot().index

    @property
    def stamp(self) -> str:
        """The bound release's id."""
        return current_release().release_id

    @property
    def manifest(self) -> CoverageManifest:
        return self.snapshot().manifest

    def snapshot(self) -> CoverageSnapshot:
        return _release_snapshot(current_release(), current_views())

    def snapshot_current(self) -> CoverageSnapshot:
        """The bound release's snapshot, as `snapshot()`. A request reads one release from its
        start, and the release service's recheck moves the served one."""
        return self.snapshot()

    def serves(self, snapshot: CoverageSnapshot) -> bool:
        """Whether `snapshot` holds the bound release's coverage index. A request keeps its
        release across a swap, and with it the index, so a snapshot it took stays served to it.
        A replaced snapshot here would come back from every retry of the same request."""
        return snapshot.index is self.current()

    def register_load_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]:
        """Registers nothing and returns a no-op. The release service enters the surface's
        swap guard around a swap, the same `every_permit` a coverage load guard takes, and a
        swap that took it twice would wait on itself."""
        return lambda: None


def _release_snapshot(release: Release, views: ReleaseViews) -> CoverageSnapshot:
    """`release`'s coverage view, with a manifest naming the release as the build."""
    index = views.coverage
    manifest = release.manifest
    return CoverageSnapshot(
        index=index,
        manifest=CoverageManifest(
            stamp=release.release_id,
            build_id=release.release_id,
            built_at=manifest.read_timestamp,
            source="graph_release",
            row_counts={},
            files={},
            horizon=index.horizon,
            # The release records its import's completion time and no creation time, so the
            # `artifact_pair_stale` warning, which compares execution ids, names the former.
            ingestion=Ingestion(
                workflow_execution_id=manifest.import_execution_id,
                creation_timestamp=manifest.import_completed_at,
            ),
        ),
        stamp=release.release_id,
    )


def _build() -> CoverageService:
    env = _CoverageEnv()
    if env.artifact_uri is None:
        raise CoverageArtifactUriMissingError(
            f"{_ARTIFACT_URI_ENV_VAR} is not set; a coverage-serving surface requires it at startup"
        )
    if env.index_check not in (None, "off"):
        raise CoverageIndexCheckValueError(
            f"{_INDEX_CHECK_ENV_VAR} is {env.index_check!r}; leave it unset to check coverage "
            "builds against the code index, or set it to 'off' in a harness that pairs them "
            "on purpose"
        )
    return CoverageService(
        artifact_uri=env.artifact_uri,
        index=get_index(),
        index_check=env.index_check is None,
        recheck_seconds=env.recheck_seconds,
    )


_singleton: RefCountedSingleton[CoverageService] = RefCountedSingleton(
    build=_build, label="coverage service"
)


_RELEASE_COVERAGE = ReleaseCoverage()


def get_coverage_service() -> ServedCoverage:
    """Returns the process-wide service, loading the artifact on first call.

    With `GRAPH_RELEASE_COVERAGE` on, returns `ReleaseCoverage`, which reads the bound release
    and none of the variables below.

    Reads `COVERAGE_ARTIFACT_URI` (required) and `COVERAGE_RECHECK_SECONDS`
    (default 300), and checks builds against the packaged code index
    (`code_index.shared.get_index`). Surface construction calls this, so a
    missing URI, an unreadable artifact, or no build compatible with the index
    fails startup, not a later request.

    `COVERAGE_INDEX_CHECK=off` skips the check, logging a WARNING on every load.

    Raises `CoverageArtifactUriMissingError` when the required URI is unset,
    `CoverageIndexCheckValueError` when `COVERAGE_INDEX_CHECK` holds a value other than
    `off`, and `CoverageIncompatibleError` when no compatible build loads. Errors from
    `load_coverage` propagate for an unreadable artifact.
    """
    if coverage_views_enabled():
        return _RELEASE_COVERAGE
    return _singleton.acquire()


def reset_coverage_service() -> None:
    """Test-only: clears the singleton so the next access rebuilds it."""
    _singleton.force_reset()


__all__ = [
    "DEFAULT_RECHECK_SECONDS",
    "CoverageArtifactUriMissingError",
    "CoverageIncompatibleError",
    "CoverageIndexCheckValueError",
    "CoverageLoader",
    "CoverageService",
    "CoverageSnapshot",
    "ReleaseCoverage",
    "ServedCoverage",
    "get_coverage_service",
    "reset_coverage_service",
]
