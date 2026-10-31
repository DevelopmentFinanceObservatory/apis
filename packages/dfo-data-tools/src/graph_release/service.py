"""The process's graph release: loaded at start, rechecked on request, swapped
under the surface's guard.

`ReleaseService` opens the release `GRAPH_RELEASE_URI`'s pointer names at
construction. `snapshot()` returns the served release without I/O and, at
most once per `recheck_seconds`, starts a daemon thread that reads the pointer.
The recheck is request-triggered because background loops starve when
`cpu_idle=true`. When the pointer has moved,
the thread downloads and maps the new release beside the old one, then
enters every registered swap guard to build its views and flip the served
release. The download needs no guard, since the local directory is an
in-memory filesystem on Cloud Run and the files are resident whether or not a
guard is held. The views live on the heap, where they would otherwise sit
beside the old release's views and every frame being built. On the Datasets
surface and its worker the guard is `MaterialisePermits.every_permit`, so the
build and the flip wait for those frames to finish, and no frame starts until
the flip ends. A failed recheck keeps the served release, leaves the guards
and logs `graph_release_recheck_failed`. A release refused for what its bytes hold
(`ReleaseFormatError`, which covers a part or manifest whose hash does not match, or
`ViewRefused`) is remembered by id and logged once at ERROR (`graph_release_refused`), and
later rechecks skip it while the pointer names it, so they download nothing and enter no
guard. Any other failure, such as an unreachable bucket, is retried at the next recheck. A
swap logs `graph_release_swapped` with the time spent entering the guards, the time spent
building, and the process's memory after the build. A guard that takes longer than
`_DRAIN_WARNING_SECONDS` to enter logs `graph_release_drain_slow` at each multiple of it.

Parts live in one `local_dir` by content hash, so consecutive releases share
the files they have in common. A release holds the files its loader read
(`LoadedRelease.file_names`) and, once its last holder drops it
(`weakref.finalize`), unlinks those no other live release holds. A drop that
lands while a load is running defers its unlinks to the end of that load,
since the load may have found the file present and be about to map it. A load
that fails unlinks every file no live release holds, so its downloads do not
stay resident.

Opening a release builds its serving views (`graph_release.views`): the code index, the
measure index and the `CoverageIndex`, and its origin view (`graph_release.origin`). They are
built in the thread that opens the release, so at construction or in the recheck inside the
swap guards, and live on the `Release`, so they die with it. A release whose views do not build
is refused as an unreadable one is, and its files are unlinked. Construction raises, and a
recheck keeps the served release. After the first load and after each swap, that thread hands
the memory the load freed back to the OS (`_return_freed_memory`). The process's member
enumerator drops the member groups of each release the service replaces
(`statvar_members.MemberEnumerator.follow_swaps`).

A `Release` the service opened knows whether the service still serves it (`Release.served`),
so a request bound to a replaced release can refuse rather than build a frame beside both
releases' views (`series_coverage.shared.ReleaseCoverage.serves`).
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import logging
import os
import resource
import sys
import tempfile
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import pyarrow as pa

from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from dfo_platform.singleton import RefCountedSingleton
from graph_release.context import URI_ENV_VAR, release_uri
from graph_release.format import (
    LoadedRelease,
    Manifest,
    ReleaseFormatError,
    open_release,
    read_pointer,
)
from graph_release.origin import ReleaseOrigin, release_origin
from graph_release.views import (
    ViewRefused,
    imported_data_digest,
    release_code_indexes,
    release_coverage_index,
)
from series_coverage.index import CoverageIndex

logger = logging.getLogger(__name__)

DEFAULT_RECHECK_SECONDS = 60.0

# A permit drain longer than this already turns waiting mints into `MintBackpressure`.
_DRAIN_WARNING_SECONDS = 30.0

# Failures the same release raises on every load, so a recheck does not load it again.
_DETERMINISTIC = (ReleaseFormatError, ViewRefused)


def _resolve_malloc_trim() -> Callable[[int], int] | None:
    """glibc's `malloc_trim`, or `None` off Linux or under a libc without it."""
    if sys.platform != "linux":
        return None
    try:
        return ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):
        return None


_malloc_trim = _resolve_malloc_trim()


def _return_freed_memory() -> None:
    """Hands the memory a view build freed back to the OS."""
    # Freed view memory stays in glibc arenas and the Arrow pool: ~1.6-2.4 GB vs ~1.2-1.4 GB
    # cgroup memory across six swaps in the deployed image.
    pa.default_memory_pool().release_unused()
    if _malloc_trim is not None:
        _malloc_trim(0)


def _rss_bytes() -> int | None:
    """The process's resident memory, or `None` off Linux."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    except OSError:
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    """`ru_maxrss` in bytes. macOS reports bytes and Linux KiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _event(level: int, event: str, **fields: object) -> None:
    """One log line whose JSON entry carries `event` and `fields` as top-level fields."""
    logger.log(level, event, extra={JSON_FIELDS_ATTR: {"event": event, **fields}})


@contextmanager
def _warn_while_draining(*, served: str, pointer: str) -> Iterator[None]:
    """Logs `graph_release_drain_slow` each `_DRAIN_WARNING_SECONDS` the block runs."""
    interval = _DRAIN_WARNING_SECONDS
    started = time.monotonic()
    done = threading.Event()

    def watch() -> None:
        while not done.wait(interval):
            _event(
                logging.WARNING,
                "graph_release_drain_slow",
                served_release_id=served,
                pointer_release_id=pointer,
                drain_seconds=round(time.monotonic() - started, 3),
            )

    threading.Thread(target=watch, daemon=True, name="graph-release-drain-watch").start()
    try:
        yield
    finally:
        done.set()


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
    """One mapped release with its serving views and its origin view. Requests and operations
    hold it for their lifetime through `graph_release.context`. `service` is the service that
    opened it, `None` for a release built outside one."""

    loaded: LoadedRelease
    views: ReleaseViews
    origin: ReleaseOrigin
    service: ReleaseService | None = field(default=None, repr=False)

    @property
    def served(self) -> bool:
        """Whether the service that opened this release still serves it. A release built
        outside a service is always served."""
        return self.service is None or self.service.serves(self)

    @property
    def manifest(self) -> Manifest:
        return self.loaded.manifest

    @property
    def release_id(self) -> str:
        return self.loaded.manifest.release_id

    @cached_property
    def imported_data(self) -> str:
        """What a mint reads from this release that only an import or a serving-view change
        moves (`graph_release.views.imported_data_digest`). A head keyed on it serves across
        releases that differ only in their builder, declarations text or DC serving
        revision."""
        return imported_data_digest(self.loaded)


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


@dataclass(frozen=True, slots=True)
class _Refusal:
    """A release a recheck refused for a deterministic reason."""

    release_id: str
    exception_type: str


class ReleaseService:
    """Opens the published release at construction and swaps to a newer one in
    the background.

    `recheck_seconds <= 0` disables the recheck. Each release builds its serving views and its
    origin view as it opens. Construction raises `FileNotFoundError` when no release is
    published, propagates the loader's errors for one that cannot be read, and raises
    `graph_release.views.ViewRefused` for one whose views do not build. A recheck that refuses
    a release for a deterministic reason skips it while the pointer names it.
    """

    def __init__(
        self,
        *,
        uri: str,
        local_dir: Path,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        loader: ReleaseLoader = open_release,
    ) -> None:
        self._uri = uri
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
        # Written and read under the load lock.
        self._refused: _Refusal | None = None
        # `snapshot()` reads this reference lock-free. The GIL makes its assignment atomic.
        self._served = _Served(
            release=self._release(self._load(release_id=None)), checked_at=clock()
        )
        _return_freed_memory()

    def snapshot(self) -> Release:
        """The served release, without waiting for I/O. Starts a due pointer
        recheck in a daemon thread."""
        self._maybe_start_recheck()
        return self._served.release

    def serves(self, release: Release) -> bool:
        """Whether `release` is the served release, without starting a recheck."""
        return self._served.release is release

    def register_swap_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]:
        """Runs every later view build and flip inside a fresh `guard()`, entered
        once the new release's parts are mapped and left once the service serves
        it and holds no reference to the old one, or once its views fail to
        build. Returns the function that unregisters it."""
        with self._hooks_lock:
            self._swap_guards = (*self._swap_guards, guard)

        def unregister() -> None:
            with self._hooks_lock:
                self._swap_guards = tuple(g for g in self._swap_guards if g is not guard)

        return unregister

    def on_swap(self, callback: Callable[[Release], None]) -> Callable[[], None]:
        """Calls `callback` with each new release once it is served, outside the
        swap guards, in the recheck thread and under its load lock, so a slow callback
        delays the next recheck. A callback that raises is logged and the others still
        run. Returns the function that unregisters it."""
        with self._hooks_lock:
            self._swap_callbacks = (*self._swap_callbacks, callback)

        def unregister() -> None:
            with self._hooks_lock:
                self._swap_callbacks = tuple(c for c in self._swap_callbacks if c is not callback)

        return unregister

    def close(self) -> None:
        """Waits for an in-flight recheck to finish."""
        with self._refresh_lock:
            future = self._in_flight
        if future is not None:
            future.result()

    def _load(self, *, release_id: str | None) -> LoadedRelease:
        """Downloads and maps a release's parts and holds their files. Pass the result to
        `_release`, which drops them again when the views do not build."""
        with self._files.loading():
            try:
                loaded = self._loader(self._uri, local_dir=self._local_dir, release_id=release_id)
            except Exception:
                self._files.unlink_unheld()
                raise
            self._files.hold(loaded.file_names)
        return loaded

    def _release(self, loaded: LoadedRelease) -> Release:
        """`loaded` with its views, holding its files until the release dies."""
        names = loaded.file_names
        try:
            views = build_release_views(loaded)
            origin = release_origin(loaded)
        except Exception:
            self._files.drop(names)
            raise
        release = Release(loaded=loaded, views=views, origin=origin, service=self)
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
        finally:
            with self._refresh_lock:
                self._in_flight = None
            future.set_result(None)

    def _swap_if_moved(self) -> None:
        """Reads the pointer and swaps when it names another release that was not refused.
        Errors leave the served release in place and are logged."""
        pointer_id: str | None = None
        try:
            with self._load_lock:
                served = self._served.release
                replaced = served.release_id
                pointer = read_pointer(self._uri)
                if pointer is None:
                    raise FileNotFoundError(f"no release pointer under {self._uri!r}")
                pointer_id = pointer.release_id
                if pointer_id == replaced:
                    self._served = _Served(release=served, checked_at=self._clock())
                    return
                refused = self._refused
                if refused is not None and refused.release_id == pointer_id:
                    self._served = _Served(release=served, checked_at=self._clock())
                    self._log_recheck_failed(
                        pointer_id=pointer_id,
                        exception_type=refused.exception_type,
                        deterministic=True,
                        skipped=True,
                    )
                    return
                loaded = self._load(release_id=pointer_id)
                with ExitStack() as guards:
                    with self._hooks_lock:
                        registered, callbacks = self._swap_guards, self._swap_callbacks
                    drain_started = time.monotonic()
                    try:
                        with _warn_while_draining(served=replaced, pointer=pointer_id):
                            for guard in registered:
                                guards.enter_context(guard())
                    except Exception:
                        self._files.drop(loaded.file_names)
                        raise
                    build_started = time.monotonic()
                    # The views build inside the guards, so they are never resident beside a
                    # frame.
                    release = self._release(loaded)
                    build_seconds = time.monotonic() - build_started
                    self._served = _Served(release=release, checked_at=self._clock())
                    self._refused = None
                    del served
                _return_freed_memory()
                _event(
                    logging.INFO,
                    "graph_release_swapped",
                    from_release_id=replaced,
                    to_release_id=release.release_id,
                    drain_seconds=round(build_started - drain_started, 3),
                    build_seconds=round(build_seconds, 3),
                    rss_bytes=_rss_bytes(),
                    peak_rss_bytes=_peak_rss_bytes(),
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
        except Exception as exc:
            self._record_recheck_failure(exc, pointer_id=pointer_id)

    def _record_recheck_failure(self, exc: Exception, *, pointer_id: str | None) -> None:
        """Logs the failure, remembers a deterministic refusal of the pointer's release, and
        resets the check time, so the next recheck waits a full interval."""
        deterministic = isinstance(exc, _DETERMINISTIC)
        self._log_recheck_failed(
            pointer_id=pointer_id,
            exception_type=type(exc).__name__,
            deterministic=deterministic,
            skipped=False,
            detail=str(exc),
        )
        with self._load_lock:
            if deterministic and pointer_id is not None:
                self._refused = _Refusal(release_id=pointer_id, exception_type=type(exc).__name__)
                _event(
                    logging.ERROR,
                    "graph_release_refused",
                    release_id=pointer_id,
                    served_release_id=self._served.release.release_id,
                    exception_type=type(exc).__name__,
                    reason=str(exc),
                )
            self._served = _Served(release=self._served.release, checked_at=self._clock())

    def _log_recheck_failed(
        self,
        *,
        pointer_id: str | None,
        exception_type: str,
        deterministic: bool,
        skipped: bool,
        detail: str | None = None,
    ) -> None:
        _event(
            logging.WARNING,
            "graph_release_recheck_failed",
            uri=self._uri,
            served_release_id=self._served.release.release_id,
            pointer_release_id=pointer_id,
            exception_type=exception_type,
            deterministic=deterministic,
            skipped=skipped,
            detail=detail,
        )


def _build() -> ReleaseService:
    uri = release_uri()
    if uri is None:
        raise ReleaseUriMissingError(f"{URI_ENV_VAR} is not set")
    # On Cloud Run the temporary directory is the in-memory filesystem.
    service = ReleaseService(uri=uri, local_dir=Path(tempfile.mkdtemp(prefix="graph-release-")))
    # The process's member groups key on the release; a swap drops the replaced release's.
    from statvar_members import shared_enumerator

    shared_enumerator().follow_swaps(service)
    return service


_singleton: RefCountedSingleton[ReleaseService] = RefCountedSingleton(
    build=_build, label="graph release service"
)


def get_release_service() -> ReleaseService:
    """Returns the process-wide service, opening the release on first call.

    Raises `ReleaseUriMissingError` when `GRAPH_RELEASE_URI` is unset or blank,
    `FileNotFoundError` when no release is published there, and
    `graph_release.views.ViewRefused` when the release's views do not build, so a surface that
    asks for it fails at startup.
    """
    return _singleton.acquire()


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
    "reset_release_service",
]
