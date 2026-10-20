"""The process's graph release: loaded at start, rechecked on request, swapped
under the surface's guard.

`ReleaseService` opens the release `GRAPH_RELEASE_URI`'s pointer names at
construction. `snapshot()` returns the served release without I/O and, at
most once per `recheck_seconds`, starts a daemon thread that reads the pointer.
The recheck is request-triggered because background loops starve when
`cpu_idle=true`, as in `series_coverage.shared`. When the pointer has moved,
the thread downloads and opens the new release beside the old one, then
enters every registered swap guard only to flip the served release. The
download needs no guard, since the local directory is an in-memory filesystem
on Cloud Run and the files are resident whether or not a guard is held. On the
Datasets surface and its worker the guard is `MaterialisePermits.every_permit`,
so the flip waits for the frames being built to finish. A failed recheck
keeps the served release and logs.

Parts live in one `local_dir` by content hash, so consecutive releases share
the files they have in common. A release holds the files its loader read
(`LoadedRelease.file_names`) and, once its last holder drops it
(`weakref.finalize`), unlinks those no other live release holds. A drop that
lands while a load is running defers its unlinks to the end of that load,
since the load may have found the file present and be about to map it. A load
that fails unlinks every file no live release holds, so its downloads do not
stay resident.

With `build_views`, which `get_release_service()` sets from `GRAPH_RELEASE_COVERAGE`
(`graph_release.context.coverage_views_enabled`), opening a release also builds its serving
views (`graph_release.views`): the code index, the measure index and the `CoverageIndex`. They
are built in the thread that opens the release, so at construction or in the recheck before
the flip, and live on the `Release`, so they die with it. A release whose views do not build is
refused as an unreadable one is: construction raises, and a recheck keeps the served release.
Without `build_views` a `Release` maps the parts and builds nothing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import tempfile
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from dfo_platform.singleton import RefCountedSingleton
from graph_release.context import URI_ENV_VAR, coverage_views_enabled, release_uri
from graph_release.format import LoadedRelease, Manifest, open_release, read_pointer
from graph_release.views import release_code_indexes, release_coverage_index
from series_coverage.index import CoverageIndex

logger = logging.getLogger(__name__)

DEFAULT_RECHECK_SECONDS = 60.0

# Logs the first recheck failure and each tenth consecutive failure.
_FAILURE_LOG_EVERY_N = 10


class ReleaseUriMissingError(RuntimeError):
    """Raised by `get_release_service()` when `GRAPH_RELEASE_URI` is unset or blank."""


class ReleaseLoader(Protocol):
    """Opens the release `release_id` names, or the pointer's when it is `None`."""

    def __call__(
        self, root_uri: str, *, local_dir: Path, release_id: str | None = None
    ) -> LoadedRelease: ...


@dataclass(frozen=True, eq=False)
class ReleaseViews:
    """The serving views of one release, built when it opens."""

    code_index: CodeIndex
    measure_index: MeasureIndex
    coverage: CoverageIndex


def build_release_views(loaded: LoadedRelease) -> ReleaseViews:
    """The serving views of `loaded`. Raises `graph_release.views.ViewRefused` when one does
    not build."""
    code_index, measure_index = release_code_indexes(loaded)
    return ReleaseViews(
        code_index=code_index,
        measure_index=measure_index,
        coverage=release_coverage_index(loaded),
    )


@dataclass(frozen=True, eq=False)
class Release:
    """One mapped release and, when its service builds them, its serving views. Requests and
    operations hold it for their lifetime through `graph_release.context`."""

    loaded: LoadedRelease
    views: ReleaseViews | None = None

    @property
    def manifest(self) -> Manifest:
        return self.loaded.manifest

    @property
    def release_id(self) -> str:
        return self.loaded.manifest.release_id


class _LocalFiles:
    """Counts the live releases naming each part file in `local_dir` and unlinks
    a file once none does."""

    def __init__(self, local_dir: Path) -> None:
        self._local_dir = local_dir
        # Reentrant: a cyclic collection inside a locked section can finalize a release.
        self._lock = threading.RLock()
        self._holders: Counter[str] = Counter()
        self._loading = False
        self._deferred: set[str] = set()

    def hold(self, names: Iterable[str]) -> None:
        with self._lock:
            self._holders.update(names)

    def drop(self, names: Iterable[str]) -> None:
        with self._lock:
            for name in names:
                self._holders[name] -= 1
                if self._holders[name] > 0:
                    continue
                del self._holders[name]
                if self._loading:
                    self._deferred.add(name)
                else:
                    self._unlink(name)

    @contextmanager
    def loading(self) -> Iterator[None]:
        """Defers unlinks for the body of the `with` block, then unlinks the
        deferred files no release has taken up since."""
        with self._lock:
            self._loading = True
        try:
            yield
        finally:
            with self._lock:
                self._loading = False
                deferred, self._deferred = self._deferred, set()
                for name in deferred:
                    if name not in self._holders:
                        self._unlink(name)

    def unlink_unheld(self) -> None:
        """Unlinks every file in `local_dir` no live release names, such as the
        parts a failed load downloaded."""
        with self._lock:
            if not self._local_dir.is_dir():
                return
            for path in self._local_dir.iterdir():
                if path.name not in self._holders:
                    self._unlink(path.name)

    def _unlink(self, name: str) -> None:
        # A table still mapped from the file stays readable after its unlink.
        (self._local_dir / name).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _Served:
    release: Release
    checked_at: float


class ReleaseService:
    """Opens the published release at construction and swaps to a newer one in
    the background.

    `recheck_seconds <= 0` disables the recheck. `build_views` builds each
    release's serving views as it opens. Construction raises
    `FileNotFoundError` when no release is published, propagates the
    loader's errors for one that cannot be read, and raises
    `graph_release.views.ViewRefused` for one whose views do not build.
    """

    def __init__(
        self,
        *,
        uri: str,
        local_dir: Path,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        loader: ReleaseLoader = open_release,
        build_views: bool = False,
    ) -> None:
        self._uri = uri
        self._build_views = build_views
        self._local_dir = local_dir
        self._recheck_seconds = recheck_seconds
        self._clock = clock
        self._loader = loader
        self._files = _LocalFiles(local_dir)
        self._refresh_lock = threading.Lock()
        # Serialises pointer reads and loads, so two rechecks never open one release twice.
        self._load_lock = threading.Lock()
        self._in_flight: concurrent.futures.Future[None] | None = None
        self._hooks_lock = threading.Lock()
        self._swap_guards: tuple[Callable[[], AbstractContextManager[object]], ...] = ()
        self._swap_callbacks: tuple[Callable[[Release], None], ...] = ()
        self._consecutive_failures = 0
        # `snapshot()` reads this reference lock-free. The GIL makes its assignment atomic.
        self._served = _Served(release=self._open(release_id=None), checked_at=clock())

    def snapshot(self) -> Release:
        """The served release, without waiting for I/O. Starts a due pointer
        recheck in a daemon thread."""
        self._maybe_start_recheck()
        return self._served.release

    def register_swap_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]:
        """Runs every later flip inside a fresh `guard()`, entered once the new
        release is open and left once the service serves it and holds no
        reference to the old one. Returns the function that unregisters it."""
        with self._hooks_lock:
            self._swap_guards = (*self._swap_guards, guard)

        def unregister() -> None:
            with self._hooks_lock:
                self._swap_guards = tuple(g for g in self._swap_guards if g is not guard)

        return unregister

    def on_swap(self, callback: Callable[[Release], None]) -> None:
        """Calls `callback` with each new release once it is served, outside the
        swap guards. A callback that raises is logged and the others still run."""
        with self._hooks_lock:
            self._swap_callbacks = (*self._swap_callbacks, callback)

    def close(self) -> None:
        """Waits for an in-flight recheck to finish."""
        with self._refresh_lock:
            future = self._in_flight
        if future is not None:
            future.result()

    def _open(self, *, release_id: str | None) -> Release:
        with self._files.loading():
            try:
                loaded = self._loader(self._uri, local_dir=self._local_dir, release_id=release_id)
                views = build_release_views(loaded) if self._build_views else None
            except Exception:
                self._files.unlink_unheld()
                raise
            release = Release(loaded=loaded, views=views)
            names = loaded.file_names
            self._files.hold(names)
        weakref.finalize(release, self._files.drop, names)
        return release

    def _maybe_start_recheck(self) -> None:
        if self._recheck_seconds <= 0:
            return
        if self._clock() - self._served.checked_at < self._recheck_seconds:
            return
        with self._refresh_lock:
            if self._in_flight is not None:
                return
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._in_flight = future
        threading.Thread(
            target=self._run_recheck, args=(future,), daemon=True, name="graph-release-recheck"
        ).start()

    def _run_recheck(self, future: concurrent.futures.Future[None]) -> None:
        try:
            self._swap_if_moved()
        except Exception as exc:
            self._record_recheck_failure(exc)
        else:
            self._consecutive_failures = 0
        finally:
            with self._refresh_lock:
                self._in_flight = None
            future.set_result(None)

    def _swap_if_moved(self) -> None:
        """Reads the pointer and swaps when it names another release. Errors
        leave the served release in place."""
        with self._load_lock:
            served = self._served.release
            replaced = served.release_id
            pointer = read_pointer(self._uri)
            if pointer is None:
                raise FileNotFoundError(f"no release pointer under {self._uri!r}")
            if pointer.release_id == replaced:
                self._served = _Served(release=served, checked_at=self._clock())
                return
            release = self._open(release_id=pointer.release_id)
            with ExitStack() as guards:
                with self._hooks_lock:
                    registered, callbacks = self._swap_guards, self._swap_callbacks
                for guard in registered:
                    guards.enter_context(guard())
                self._served = _Served(release=release, checked_at=self._clock())
                del served
            logger.info(
                "graph release: served release changed (%r -> %r)", replaced, release.release_id
            )
            for callback in callbacks:
                try:
                    callback(release)
                except Exception:
                    logger.exception(
                        "graph release: swap callback %r failed for release %r",
                        callback,
                        release.release_id,
                    )

    def _record_recheck_failure(self, exc: Exception) -> None:
        """Logs the first and every tenth consecutive failure and resets the
        check time, so the next recheck waits a full interval."""
        self._consecutive_failures += 1
        if (
            self._consecutive_failures == 1
            or self._consecutive_failures % _FAILURE_LOG_EVERY_N == 0
        ):
            logger.warning(
                "graph release: recheck of %r failed (%d consecutive failure(s)), "
                "still serving release %r: %s",
                self._uri,
                self._consecutive_failures,
                self._served.release.release_id,
                exc,
            )
        with self._load_lock:
            self._served = _Served(release=self._served.release, checked_at=self._clock())


def _build() -> ReleaseService:
    uri = release_uri()
    if uri is None:
        raise ReleaseUriMissingError(f"{URI_ENV_VAR} is not set")
    # On Cloud Run the temporary directory is the in-memory filesystem.
    return ReleaseService(
        uri=uri,
        local_dir=Path(tempfile.mkdtemp(prefix="graph-release-")),
        build_views=coverage_views_enabled(),
    )


_singleton: RefCountedSingleton[ReleaseService] = RefCountedSingleton(
    build=_build, label="graph release service"
)


def get_release_service() -> ReleaseService:
    """Returns the process-wide service, opening the release on first call.

    Raises `ReleaseUriMissingError` when `GRAPH_RELEASE_URI` is unset or blank,
    `FileNotFoundError` when no release is published there, and, with
    `GRAPH_RELEASE_COVERAGE` on, `graph_release.views.ViewRefused` when the
    release's views do not build, so a surface that asks for it fails at startup.
    """
    return _singleton.acquire()


def optional_release_service() -> ReleaseService | None:
    """The process-wide service when `GRAPH_RELEASE_URI` is set, `None` when it
    is unset or blank."""
    return None if release_uri() is None else get_release_service()


def reset_release_service() -> None:
    """Test-only: clears the singleton so the next access rebuilds it."""
    _singleton.force_reset()


__all__ = [
    "DEFAULT_RECHECK_SECONDS",
    "Release",
    "ReleaseLoader",
    "ReleaseService",
    "ReleaseUriMissingError",
    "ReleaseViews",
    "build_release_views",
    "get_release_service",
    "optional_release_service",
    "reset_release_service",
]
