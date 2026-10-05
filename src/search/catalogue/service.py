"""Loads the published search catalogue for `GET /api/search/catalogue` and
keeps it current, as `origin_catalogue.service` does for the origin
catalogue.

`search_catalogue_service()` returns `None` when `SEARCH_CATALOGUE_URI` is
unset. A service that cannot load a build at construction starts empty, so
the search surface starts without a catalogue and only the catalogue route
reports it unavailable. The service rechecks `current.json` at most once per
`SEARCH_CATALOGUE_RECHECK_SECONDS` and reloads only when
`catalogue_version` changes. The recheck runs on a daemon thread started by
a request, because background loops starve under `cpu_idle=true`. A failed
reload, or a pointer that disappears, keeps the last good build serving.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic import Field

from artifact_publication.jobs import AliasOnlySettings
from search.catalogue.artifact import CatalogueArtifact, load_published, read_pointer

logger = logging.getLogger(__name__)

DEFAULT_RECHECK_SECONDS: Final[float] = 300.0

# Logs the first reload failure and each tenth consecutive failure.
_FAILURE_LOG_EVERY_N: Final[int] = 10


class SearchCatalogueSettings(AliasOnlySettings):
    """The environment the search surface reads its catalogue under."""

    artifact_uri: str | None = Field(default=None, min_length=1, alias="SEARCH_CATALOGUE_URI")
    recheck_seconds: float = Field(
        default=DEFAULT_RECHECK_SECONDS, alias="SEARCH_CATALOGUE_RECHECK_SECONDS"
    )


type CatalogueLoader = Callable[[str], CatalogueArtifact]


@dataclass(frozen=True, slots=True)
class _Loaded:
    artifact: CatalogueArtifact | None
    checked_at: float


class SearchCatalogueService:
    """Holds the loaded catalogue build, or `None` before one loads, and
    refreshes it in the background. `recheck_seconds <= 0` disables refresh."""

    def __init__(
        self,
        *,
        artifact_uri: str,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        loader: CatalogueLoader = load_published,
    ) -> None:
        self._artifact_uri = artifact_uri
        self._recheck_seconds = recheck_seconds
        self._clock = clock
        self._loader = loader
        self._refresh_lock = threading.Lock()
        # Serialises pointer reads and loads, avoiding duplicate loads.
        self._load_lock = threading.Lock()
        self._in_flight: concurrent.futures.Future[None] | None = None
        self._consecutive_failures = 0
        # Read lock-free, since the GIL makes this reference's assignment and read atomic.
        self._loaded = _Loaded(artifact=None, checked_at=clock())
        self._refresh()

    def current(self) -> CatalogueArtifact | None:
        """The loaded build, without waiting for I/O. Starts a due recheck on
        a daemon thread."""
        self._maybe_start_refresh()
        return self._loaded.artifact

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
            target=self._run_refresh, args=(future,), daemon=True, name="search-catalogue-recheck"
        ).start()

    def _run_refresh(self, future: concurrent.futures.Future[None]) -> None:
        try:
            self._refresh()
        finally:
            with self._refresh_lock:
                self._in_flight = None
            future.set_result(None)

    def _refresh(self) -> None:
        try:
            self._load_if_moved()
        except Exception as exc:
            self._record_reload_failure(exc)
        else:
            self._consecutive_failures = 0

    def _load_if_moved(self) -> None:
        with self._load_lock:
            old = self._loaded.artifact
            pointer, _ = read_pointer(self._artifact_uri)
            if pointer is None or (
                old is not None and pointer.catalogue_version == old.catalogue_version
            ):
                if old is None:
                    logger.warning(
                        "search catalogue: nothing is published under %r", self._artifact_uri
                    )
                self._loaded = _Loaded(artifact=old, checked_at=self._clock())
                return
            loaded = self._loader(self._artifact_uri)
            self._loaded = _Loaded(artifact=loaded, checked_at=self._clock())
        logger.info(
            "search catalogue: serving build %r (was %r)",
            loaded.catalogue_version,
            None if old is None else old.catalogue_version,
        )

    def _record_reload_failure(self, exc: Exception) -> None:
        """Logs the first and every tenth consecutive failure and resets the
        check time, so a failing build is retried once per interval."""
        self._consecutive_failures += 1
        loaded = self._loaded.artifact
        if (
            self._consecutive_failures == 1
            or self._consecutive_failures % _FAILURE_LOG_EVERY_N == 0
        ):
            logger.warning(
                "search catalogue: load of %r failed (%d consecutive failure(s)), serving %s: %s",
                self._artifact_uri,
                self._consecutive_failures,
                "nothing" if loaded is None else f"build {loaded.catalogue_version!r}",
                exc,
            )
        with self._load_lock:
            self._loaded = _Loaded(artifact=self._loaded.artifact, checked_at=self._clock())


def search_catalogue_service() -> SearchCatalogueService | None:
    """A service over `SEARCH_CATALOGUE_URI`, or `None` when it is unset.
    Reads `SEARCH_CATALOGUE_RECHECK_SECONDS` (default 300)."""
    env = SearchCatalogueSettings()
    if env.artifact_uri is None:
        logger.info("search catalogue: SEARCH_CATALOGUE_URI is not set; the route serves 503")
        return None
    return SearchCatalogueService(
        artifact_uri=env.artifact_uri, recheck_seconds=env.recheck_seconds
    )


__all__ = [
    "DEFAULT_RECHECK_SECONDS",
    "SearchCatalogueService",
    "SearchCatalogueSettings",
    "search_catalogue_service",
]
