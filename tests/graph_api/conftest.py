"""Shared fixtures for the `graph_api` test suite."""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable, Mapping, Sequence

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_index.shared import get_index
from dc_graph import ArcSubject, ArcSubjectPage, ArcTarget, SubjectsCoverageFlag
from dfo_platform.app import create_app
from entity_resolution import EntityResolver
from entity_resolution.shared import SharedResolver
from graph_api.app import graph_surface
from graph_api.code_lookup import CodeIndexCodes
from graph_api.reader import NodeArcsByDcid
from materialise.facet_profile import ProfileGraphReader
from materialise.graph import SubjectPage
from materialise.place_scope import AvailabilityReader


@pytest.fixture(scope="session", autouse=True)
def _register_dfo_graph_pack(_shared_resolvekit_pack: SharedResolver) -> SharedResolver:
    """Register the `dfo_graph` resolvekit pack before the app lifespan runs.

    Returns the shared resolver so tests can build an `EntityResolver` without
    acquiring the process-wide resolver again.
    """
    return _shared_resolvekit_pack


@pytest.fixture(scope="session")
def shared_resolver(_register_dfo_graph_pack: SharedResolver) -> SharedResolver:
    """Expose the session resolver under the fixture name used by tests."""
    return _register_dfo_graph_pack


class FakeGraphNodeReader:
    """Fake graph reads shared by the graph API and entity resolver tests.

    `nodes` maps dcids to arcs; absent dcids return `None` and absent properties return `()`, so
    `node_arcs_many` answers a held node holding none of the properties it names with `()` for
    each, as the served `OriginGraphClient` answers a node its catalogue tables hold. `arcs`, `contained`, and `subjects` map lookup keys to results,
    with empty results for missing keys. `arc_subjects` maps `(dcid, property)`
    to every subject `node_arc_subjects` and `node_arc_subjects_many` serve,
    capped at `max_results` with `truncated` or `"capped"` set past it. `arc_targets` maps dcids to
    `{property: targets}` for `node_arc_targets_many`, with absent dcids and
    properties as in `nodes`, and `node_all_arc_targets_many` answers every property of that
    map with a target, `None` for a dcid with none. Names combine `name` then `alternateName`,
    deduplicated. `calls` counts calls by method, and `arc_target_reads` records the
    `(individual_probe_budget, cached)` of each `node_arc_targets_many` call.
    """

    def __init__(
        self,
        *,
        nodes: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
        arcs: Mapping[tuple[str, str], list[str]] | None = None,
        contained: Mapping[tuple[str, str], list[str]] | None = None,
        subjects: Mapping[tuple[str, str], SubjectPage] | None = None,
        arc_subjects: Mapping[tuple[str, str], Sequence[ArcSubject]] | None = None,
        arc_targets: Mapping[str, Mapping[str, Sequence[ArcTarget]]] | None = None,
    ) -> None:
        self._nodes = nodes or {}
        self._arcs = arcs or {}
        self._contained = contained or {}
        self._subjects = subjects or {}
        self._arc_subjects = arc_subjects or {}
        self._arc_targets = arc_targets or {}
        self.arc_target_reads: list[tuple[int | None, bool]] = []
        self.calls: dict[str, int] = {
            "node_arcs_many": 0,
            "node_names_many": 0,
            "resolve_by_arc": 0,
            "child_places_batch": 0,
            "node_subjects": 0,
            "node_arc_subjects": 0,
            "node_arc_subjects_many": 0,
            "node_arc_targets_many": 0,
            "node_all_arc_targets_many": 0,
        }

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> NodeArcsByDcid:
        self.calls["node_arcs_many"] += 1
        result: NodeArcsByDcid = {}
        for dcid in dcids:
            entry = self._nodes.get(dcid)
            result[dcid] = (
                None
                if entry is None
                else {prop: tuple(dict.fromkeys(entry.get(prop, ()))) for prop in properties}
            )
        return result

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        self.calls["resolve_by_arc"] += 1
        return {name: list(dict.fromkeys(self._arcs.get((arc, name), []))) for name in names}

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        self.calls["node_names_many"] += 1
        return {
            dcid: tuple(
                dict.fromkeys(
                    value
                    for prop in ("name", "alternateName")
                    for value in self._nodes.get(dcid, {}).get(prop, ())
                )
            )
            for dcid in dict.fromkeys(dcids)
        }

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        self.calls["node_subjects"] += 1
        return self._subjects.get((dcid, property), SubjectPage(dcids=(), truncated=False))

    def node_arc_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> ArcSubjectPage:
        self.calls["node_arc_subjects"] += 1
        subjects = tuple(self._arc_subjects.get((dcid, property), ()))
        return ArcSubjectPage(
            subjects=subjects[:max_results], truncated=len(subjects) > max_results
        )

    def node_arc_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, tuple[tuple[ArcSubject, ...], SubjectsCoverageFlag]]:
        self.calls["node_arc_subjects_many"] += 1
        result: dict[str, tuple[tuple[ArcSubject, ...], SubjectsCoverageFlag]] = {}
        for dcid in dict.fromkeys(dcids):
            subjects = tuple(self._arc_subjects.get((dcid, property), ()))
            if len(subjects) > max_results:
                result[dcid] = (subjects[:max_results], "capped")
            else:
                result[dcid] = (subjects, "complete")
        return result

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = 0,
        cached: bool = True,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls["node_arc_targets_many"] += 1
        self.arc_target_reads.append((individual_probe_budget, cached))
        result: dict[str, dict[str, tuple[ArcTarget, ...]] | None] = {}
        for dcid in dict.fromkeys(dcids):
            entry = self._arc_targets.get(dcid)
            result[dcid] = (
                None if entry is None else {prop: tuple(entry.get(prop, ())) for prop in properties}
            )
        return result

    def node_all_arc_targets_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls["node_all_arc_targets_many"] += 1
        return {
            dcid: {
                prop: tuple(targets)
                for prop, targets in self._arc_targets.get(dcid, {}).items()
                if targets
            }
            or None
            for dcid in dict.fromkeys(dcids)
        }

    def child_places_batch(
        self,
        parent_dcids: Sequence[str],
        *,
        child_type: str,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        self.calls["child_places_batch"] += 1
        return {
            parent: list(dict.fromkeys(self._contained.get((parent, child_type), [])))
            for parent in parent_dcids
        }


def make_resolver(shared: SharedResolver, *, graph: FakeGraphNodeReader) -> EntityResolver:
    """Build an `EntityResolver` with the shared resolver and test graph."""
    return EntityResolver(
        graph=graph, shared=shared, codes=CodeIndexCodes(get_index), code_index=get_index
    )


def make_app(
    *,
    graph: FakeGraphNodeReader | None = None,
    resolver: EntityResolver | None = None,
    node_pool_size: int | None = None,
    coverage: object | None = None,
    profile_graph: ProfileGraphReader | None = None,
    availability: AvailabilityReader | None = None,
) -> FastAPI:
    if graph is None:
        graph = FakeGraphNodeReader()
    return create_app(
        [
            graph_surface(
                graph=graph,
                resolver=resolver,
                node_pool_size=node_pool_size,
                coverage=coverage,
                profile_graph=profile_graph,
                availability=availability,
            )
        ]
    )


def make_client(
    *,
    graph: FakeGraphNodeReader | None = None,
    resolver: EntityResolver | None = None,
    node_pool_size: int | None = None,
    coverage: object | None = None,
    profile_graph: ProfileGraphReader | None = None,
    availability: AvailabilityReader | None = None,
) -> TestClient:
    return TestClient(
        make_app(
            graph=graph,
            resolver=resolver,
            node_pool_size=node_pool_size,
            coverage=coverage,
            profile_graph=profile_graph,
            availability=availability,
        ),
        raise_server_exceptions=False,
    )


def queued_dcp_reads(client: TestClient) -> int:
    """Count the reads waiting for a worker in the running surface's DCP-read executor."""
    return client.app.state.graph_deps.dcp_executor._work_queue.qsize()


def wait_until(condition: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition still false after the timeout"
        time.sleep(0.01)


def send_past_a_full_pool(
    client: TestClient,
    send: Callable[[], httpx.Response],
    *,
    entered: threading.Event,
    release: threading.Event,
) -> list[httpx.Response]:
    """Send two concurrent requests through a one-worker pool whose reader blocks on `release`.

    The first request's read takes the worker. Once the second request is queued behind it,
    or has already answered, release the reader and return both responses in send order.
    """
    responses: list[httpx.Response | None] = [None, None]

    def send_into(slot: int) -> None:
        responses[slot] = send()

    first = threading.Thread(target=contextvars.copy_context().run, args=(send_into, 0))
    first.start()
    assert entered.wait(timeout=5.0), "the first request never reached the blocking read"
    second = threading.Thread(target=contextvars.copy_context().run, args=(send_into, 1))
    second.start()
    wait_until(lambda: queued_dcp_reads(client) == 1 or responses[1] is not None)
    release.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert all(response is not None for response in responses), "a request never returned"
    return [response for response in responses if response is not None]
