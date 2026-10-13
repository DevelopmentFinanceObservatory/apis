"""Process-wide origin catalogue loading and refresh, as `series_coverage.shared`
does for the coverage index.

The service loads the published build at construction, so a surface that
builds it fails startup when no build can be loaded. It rechecks
`current.json` at most once per `ORIGIN_CATALOGUE_RECHECK_SECONDS` and
reloads only when `build_id` changes. The recheck runs on a daemon thread
started by a request, because background loops starve under
`cpu_idle=true`. A failed reload, or a build over the memory budget, keeps
the last good snapshot serving. Each swap runs the `on_swap` callbacks.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dfo_platform.singleton import RefCountedSingleton
from origin_catalogue.artifact import load_catalogue, read_published_pointer
from origin_catalogue.catalogue import Catalogue
from origin_catalogue.context import current_build_id

logger = logging.getLogger(__name__)

_ARTIFACT_URI_ENV_VAR = "ORIGIN_CATALOGUE_URI"
DEFAULT_RECHECK_SECONDS = 300.0
DEFAULT_MAX_RESIDENT_BYTES = 384 * 1024 * 1024
"""The memory budget for one loaded build, against the size its manifest records for its
tables, its UN tables and its arc tables together. The dfo-dev apis service runs one process
in a 4 GiB container. Loading a build adds about 1.52 times its recorded size to the resident
set (a format-5 build of the dev data records 202 MiB and adds 297 to 317 MiB on macOS), and
two builds are resident during a swap. The composed app with the Search and graph surfaces
starts at about 1,170 MiB without the catalogue. A build at the budget then needs
1,170 + 2 x 1.52 x 384 = about 2,340 MiB, which leaves about 1.7 GiB for the Datasets
surface, whose start the 1,170 MiB leaves out, and a coverage index reload."""

# Logs the first reload failure and each tenth consecutive failure.
_FAILURE_LOG_EVERY_N = 10


class CatalogueSettings(BaseSettings):
    """The environment a surface loads its catalogue under, read by
    `get_catalogue_service()` and `origin_catalogue.verify`. Construct one
    per use so a variable set after import is honoured."""

    model_config = SettingsConfigDict()

    artifact_uri: str | None = Field(default=None, alias=_ARTIFACT_URI_ENV_VAR)
    recheck_seconds: float = Field(
        default=DEFAULT_RECHECK_SECONDS, alias="ORIGIN_CATALOGUE_RECHECK_SECONDS"
    )
    max_resident_bytes: int = Field(
        default=DEFAULT_MAX_RESIDENT_BYTES, alias="ORIGIN_CATALOGUE_MAX_RESIDENT_BYTES"
    )
    un_system_enabled: bool = Field(default=True, alias="ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED")
    warn_pair_stale: bool = Field(default=True, alias="ORIGIN_CATALOGUE_WARN_PAIR_STALE")
    """Whether a Search answer carries `artifact_pair_stale` when the catalogue and coverage
    builds it bound read different ingestions. `false` silences the warning; the log line
    stays."""
    warn_index_drift: bool = Field(default=True, alias="ORIGIN_CATALOGUE_WARN_INDEX_DRIFT")
    """Whether a Search answer carries `catalogue_index_drift` when the catalogue's arcs
    and the packaged code index disagree on a StatVar it serves. `false` silences the
    warning; the log line stays."""


class CatalogueUriMissingError(RuntimeError):
    """Raised at surface construction when `ORIGIN_CATALOGUE_URI` is unset."""


type CatalogueLoader = Callable[[str], Catalogue]


@dataclass(frozen=True, slots=True)
class _Loaded:
    snapshot: Catalogue
    checked_at: float


class CatalogueService:
    """Holds the loaded catalogue and refreshes it in the background.

    `recheck_seconds <= 0` disables refresh. `loader` defaults to
    `load_catalogue` under this service's memory budget and UN setting.
    """

    def __init__(
        self,
        *,
        artifact_uri: str,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        max_resident_bytes: int = DEFAULT_MAX_RESIDENT_BYTES,
        un_system_enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        loader: CatalogueLoader | None = None,
    ) -> None:
        self._artifact_uri = artifact_uri
        self._recheck_seconds = recheck_seconds
        self._clock = clock
        self._loader: CatalogueLoader = loader or (
            lambda uri: load_catalogue(
                uri, max_resident_bytes=max_resident_bytes, un_system_enabled=un_system_enabled
            )
        )
        self._refresh_lock = threading.Lock()
        # Serialises pointer reads and loads, avoiding duplicate full loads.
        self._load_lock = threading.Lock()
        self._in_flight: concurrent.futures.Future[None] | None = None
        self._consecutive_failures = 0
        self._swap_callbacks: list[Callable[[], None]] = []
        # Read lock-free, since the GIL makes this reference's assignment and read atomic.
        self._loaded = _Loaded(snapshot=self._loader(artifact_uri), checked_at=clock())

    def snapshot(self) -> Catalogue:
        """The loaded catalogue, without waiting for I/O. Starts a due
        recheck on a daemon thread. Bind the result for the whole request."""
        self._maybe_start_refresh()
        return self._loaded.snapshot

    def loaded_build_id(self) -> str:
        """The loaded catalogue's build id. Starts no recheck, so a cache
        generation can read it without driving reloads."""
        return self._loaded.snapshot.build_id

    def request_build_id(self) -> str:
        """The build id of the catalogue bound to the calling context, or the
        loaded one's where none is bound. Starts no recheck. Caches whose
        values depend on the catalogue key on it, so a request bound to one
        build never reads or writes another build's entries, a swap
        mid-request included."""
        return current_build_id(default=self.loaded_build_id)

    def on_swap(self, callback: Callable[[], None]) -> None:
        """Runs `callback` on the reloading thread after each change of the
        loaded build. A failing callback is logged and the rest still run."""
        with self._refresh_lock:
            self._swap_callbacks.append(callback)

    def snapshot_current(self) -> Catalogue:
        """Reads `current.json` and returns a snapshot at least as recent as
        the build it names, waiting for any in-flight reload. Pointer and
        loader errors propagate and leave the served snapshot unchanged."""
        return self._load_if_moved()

    def close(self) -> None:
        """Waits for an in-flight background reload to finish."""
        with self._refresh_lock:
            future = self._in_flight
        if future is not None:
            future.result()

    def _maybe_start_refresh(self) -> None:
        if self._recheck_seconds <= 0:
            return
        if self._clock() - self._loaded.checked_at < self._recheck_seconds:
            return
        with self._refresh_lock:
            if self._in_flight is not None:
                return
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._in_flight = future
        threading.Thread(
            target=self._run_refresh, args=(future,), daemon=True, name="origin-catalogue-recheck"
        ).start()

    def _run_refresh(self, future: concurrent.futures.Future[None]) -> None:
        try:
            try:
                self._load_if_moved()
            except Exception as exc:
                self._record_reload_failure(exc)
            else:
                self._consecutive_failures = 0
        finally:
            with self._refresh_lock:
                self._in_flight = None
            future.set_result(None)

    def _load_if_moved(self) -> Catalogue:
        with self._load_lock:
            old = self._loaded.snapshot
            published = read_published_pointer(self._artifact_uri)
            if published is not None and published.build_id == old.build_id:
                self._loaded = _Loaded(snapshot=old, checked_at=self._clock())
                return old
            loaded = self._loader(self._artifact_uri)
            self._loaded = _Loaded(snapshot=loaded, checked_at=self._clock())
        logger.info(
            "origin catalogue: published build changed (%r built %s -> %r built %s)",
            old.build_id,
            old.built_at.isoformat(),
            loaded.build_id,
            loaded.built_at.isoformat(),
        )
        if loaded.build_id != old.build_id:
            self._run_swap_callbacks()
        return loaded

    def _run_swap_callbacks(self) -> None:
        with self._refresh_lock:
            callbacks = list(self._swap_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("origin catalogue: a swap callback failed")

    def _record_reload_failure(self, exc: Exception) -> None:
        """Logs the first and every tenth consecutive failure and resets the
        check time, so a failing build is retried once per interval."""
        self._consecutive_failures += 1
        if (
            self._consecutive_failures == 1
            or self._consecutive_failures % _FAILURE_LOG_EVERY_N == 0
        ):
            logger.warning(
                "origin catalogue: recheck of %r failed (%d consecutive failure(s)), still "
                "serving build %r: %s",
                self._artifact_uri,
                self._consecutive_failures,
                self._loaded.snapshot.build_id,
                exc,
            )
        with self._load_lock:
            self._loaded = _Loaded(snapshot=self._loaded.snapshot, checked_at=self._clock())


def _build() -> CatalogueService:
    env = CatalogueSettings()
    if env.artifact_uri is None:
        raise CatalogueUriMissingError(
            f"{_ARTIFACT_URI_ENV_VAR} is not set; a surface that admits upstream data requires "
            "it at startup"
        )
    return CatalogueService(
        artifact_uri=env.artifact_uri,
        recheck_seconds=env.recheck_seconds,
        max_resident_bytes=env.max_resident_bytes,
        un_system_enabled=env.un_system_enabled,
    )


_singleton: RefCountedSingleton[CatalogueService] = RefCountedSingleton(
    build=_build, label="origin catalogue service"
)


def get_catalogue_service() -> CatalogueService:
    """The process-wide service, loading the catalogue on first call.

    Reads `ORIGIN_CATALOGUE_URI` (required), `ORIGIN_CATALOGUE_RECHECK_SECONDS`
    (default 300), `ORIGIN_CATALOGUE_MAX_RESIDENT_BYTES` (default 384 MiB)
    and `ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED` (default true). Raises
    `CatalogueUriMissingError` when the URI is unset. `load_catalogue`'s
    errors propagate for an unloadable build.
    """
    return _singleton.acquire()


def reset_catalogue_service() -> None:
    """Test-only: clears the singleton so the next access rebuilds it."""
    _singleton.force_reset()


__all__ = [
    "DEFAULT_MAX_RESIDENT_BYTES",
    "DEFAULT_RECHECK_SECONDS",
    "CatalogueService",
    "CatalogueSettings",
    "CatalogueUriMissingError",
    "get_catalogue_service",
    "reset_catalogue_service",
]
