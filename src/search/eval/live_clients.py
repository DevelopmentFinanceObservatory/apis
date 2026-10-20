"""Live clients for eval processes.

``score_clients`` gives the eval runner process the graph, SDMX and coverage reads its
checks need. ``live_engine_clients``, ``bind_run_catalogue`` and ``skills_index_snapshot``
serve scripts that run the engine in process: one catalogue version, one graph release when
``GRAPH_RELEASE_URI`` is set, and one skills index build for the whole run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from graph_release.context import bind_release, bound_served_release, optional_release_service
from origin_catalogue import bind_catalogue, bound_catalogue
from origin_catalogue.service import get_catalogue_service
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.sdmx import LiveSdmxClient
from search.engine.config import settings
from series_coverage.shared import ServedCoverage, get_coverage_service
from skills_index import SkillsIndex, build_index


def bind_run_catalogue() -> None:
    """Bind the process-wide catalogue service's loaded snapshot for the rest of the process,
    and the release service's served release when `GRAPH_RELEASE_URI` is set.

    For scripts that call the engine without `live_engine_clients`. Raises
    `CatalogueUriMissingError` when `ORIGIN_CATALOGUE_URI` is unset. Asyncio tasks started
    afterwards inherit the bindings. Worker threads and child processes bind their own.
    """
    bind_catalogue(get_catalogue_service().snapshot())
    release = optional_release_service()
    if release is not None:
        bind_release(release.snapshot())


@contextmanager
def live_engine_clients() -> Iterator[tuple[LiveGraphClient, LiveSdmxClient]]:
    """Yield a (LiveGraphClient, LiveSdmxClient) pair, closing both on exit.

    Binds the process-wide catalogue service's loaded snapshot for the block, and the release
    service's served release when `GRAPH_RELEASE_URI` is set, so every query in the run reads
    one catalogue version and one release. Closes both clients in a finally block
    even when the caller's block raises.
    """
    catalogue = get_catalogue_service()
    release = optional_release_service()
    graph = LiveGraphClient.from_search_settings(settings())
    sdmx = LiveSdmxClient()
    try:
        with bound_catalogue(catalogue.snapshot()), bound_served_release(release):
            yield graph, sdmx
    finally:
        graph.close()
        sdmx.close()


@dataclass(frozen=True, slots=True)
class ScoreClients:
    """The live reads a scoring process needs: graph, SDMX and the coverage service."""

    graph: LiveGraphClient
    sdmx: LiveSdmxClient
    coverage: ServedCoverage


@contextmanager
def score_clients() -> Iterator[ScoreClients]:
    """Yield the clients the checks read, closing the graph and SDMX clients on exit.

    The coverage service is the process-wide one: it reads `COVERAGE_ARTIFACT_URI`, and
    entry raises when that is unset or unreadable.
    """
    graph = LiveGraphClient.from_search_settings(settings())
    sdmx = LiveSdmxClient()
    try:
        yield ScoreClients(graph=graph, sdmx=sdmx, coverage=get_coverage_service())
    finally:
        graph.close()
        sdmx.close()


@dataclass(frozen=True, slots=True)
class SkillsIndexSnapshot:
    """One skills index build that serves every request of a run.

    ``index`` is ``None`` when the build failed, so requests asking for skills get the
    ``skills_index_unavailable`` warning.
    """

    index: SkillsIndex | None


def skills_index_snapshot(graph: LiveGraphClient) -> SkillsIndexSnapshot:
    """Build the skills index once from ``graph``, the build the service serves.

    The snapshot is built once and kept, so every golden in a run is scored against one
    skill roster. A failed build prints a warning and yields an unset index, and every case
    whose expectation names ``skills`` then fails the ``skills`` check.
    """
    try:
        # LiveGraphClient satisfies SkillsGraph structurally, but the two
        # Protocols name separately defined SubjectsPage classes.
        index = build_index(graph=graph)  # ty: ignore[invalid-argument-type]
    except Exception as exc:
        print(
            f"WARNING: the skills index build failed ({exc!r}); cases that expect skills will fail."
        )
        return SkillsIndexSnapshot(index=None)
    return SkillsIndexSnapshot(index=index)
