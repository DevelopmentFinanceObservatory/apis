"""The process's facet profile caches, one per source: holders naming one source read through
one cache, so a cold variable takes one flight; another source gets a cache of its own; the last
release of a source closes its cache; and teardown waits for a running flight before closing
the clients it reads through."""

from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlsplit

import pytest

from graph_generation import FakeGeneration
from materialise import shared_profiles
from materialise.facet_profile import FacetProfile
from materialise.shared_profiles import (
    ProfileSource,
    acquire_shared_profiles,
    release_shared_profiles,
)

from ._scope_availability import ScopeAvailability
from .test_facet_profile import (
    _EVERY_PLACE_BOTH,
    _PLACES,
    _POPULATION,
    _empty_coverage,
    _overlap,
    _ProfileGraph,
)

_SOURCE = ProfileSource(
    graph_base_url="https://mixer.example.org/core/api",
    sdmx_base_url="https://mixer.example.org/core/api/",
    un_base_url="https://un.example.org/core/api",
    token="search-token",
)


_FACETS = {(_POPULATION, place): facets for place, facets in _EVERY_PLACE_BOTH.items()}


class _Clients:
    """The clients each shared cache builds, recording reads and closes in `events`, each
    named by the host of the root it was built on. `graph` is the last graph client built."""

    def __init__(self, *, gate: threading.Event | None = None) -> None:
        self.events: list[str] = []
        self._gate = gate
        self.graph: _ClosingGraph | None = None

    def build_graph_client(self, *, base_url: str, **_options: object) -> _ProfileGraph:
        self.graph = _ClosingGraph(self.events, _host(base_url), gate=self._gate)
        return self.graph

    def sdmx_client(self, *, base_url: str, **_options: object) -> ScopeAvailability:
        return _ClosingAvailability(self.events, _host(base_url))


def _host(base_url: str) -> str:
    return urlsplit(base_url).netloc.split(".")[0]


class _ClosingGraph(_ProfileGraph):
    def __init__(self, events: list[str], host: str, *, gate: threading.Event | None) -> None:
        super().__init__(
            facets=_FACETS, child_places={("Earth", "Country"): list(_PLACES)}, gate=gate
        )
        self.events = events
        self.host = host
        self.reading = threading.Event()

    def observation_facets_batch_many(self, **kwargs):
        self.reading.set()
        found = super().observation_facets_batch_many(**kwargs)
        self.events.append(f"{self.host}: facet read ended")
        return found

    def close(self) -> None:
        self.events.append(f"{self.host}: graph closed")


class _ClosingAvailability(ScopeAvailability):
    def __init__(self, events: list[str], host: str) -> None:
        super().__init__(data=[key for key, found in _FACETS.items() if found])
        self._events = events
        self._host = host

    def close(self) -> None:
        self._events.append(f"{self._host}: availability closed")


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared_profiles, "_held", {})


def _install(monkeypatch: pytest.MonkeyPatch, clients: _Clients) -> None:
    monkeypatch.setattr(shared_profiles, "build_graph_client", clients.build_graph_client)
    monkeypatch.setattr(shared_profiles, "SdmxClient", clients.sdmx_client)


def _profile(holder: shared_profiles.SharedProfiles) -> FacetProfile:
    found = holder.cache.profiles(
        [_POPULATION], catalogue=_overlap(), coverage=_empty_coverage(), deadline=None
    )[_POPULATION]
    assert isinstance(found, FacetProfile), found
    return found


def test_two_holders_share_one_cache_and_one_flight_for_a_cold_variable(monkeypatch):
    """Graph and Search acquire over the same roots and token, a trailing slash apart. Search
    asks for a variable while Graph's cold flight for it is reading, and
    joins that flight."""
    gate = threading.Event()
    clients = _Clients(gate=gate)
    _install(monkeypatch, clients)
    generation = FakeGeneration(token="g1")
    graph_surface = acquire_shared_profiles(
        source=ProfileSource(
            graph_base_url="https://mixer.example.org/core/api/",
            sdmx_base_url="https://mixer.example.org/core/api",
            un_base_url="https://un.example.org/core/api",
            token="search-token",
        ),
        generation=generation,
    )
    search = acquire_shared_profiles(source=_SOURCE, generation=generation)
    assert search.cache is graph_surface.cache

    found: list[FacetProfile] = []
    first = threading.Thread(target=lambda: found.append(_profile(graph_surface)))
    first.start()
    assert clients.graph is not None and clients.graph.reading.wait(5)
    second = threading.Thread(target=lambda: found.append(_profile(search)))
    second.start()
    gate.set()
    first.join(5)
    second.join(5)

    assert len(found) == 2
    assert found[0] == found[1]
    assert clients.events == ["mixer: facet read ended"]

    release_shared_profiles(search)
    release_shared_profiles(graph_surface)


def test_the_last_release_closes_the_clients_and_the_next_acquire_builds_afresh(monkeypatch):
    clients = _Clients()
    _install(monkeypatch, clients)
    generation = FakeGeneration(token="g1")
    first = acquire_shared_profiles(source=_SOURCE, generation=generation)
    second = acquire_shared_profiles(source=_SOURCE, generation=generation)

    release_shared_profiles(first)
    assert clients.events == []
    _profile(second)

    release_shared_profiles(second)
    assert clients.events == [
        "mixer: facet read ended",
        "mixer: availability closed",
        "mixer: graph closed",
    ]
    with pytest.raises(RuntimeError):
        second.executor.submit(lambda: None)

    again = acquire_shared_profiles(source=_SOURCE, generation=generation)
    assert again is not second
    release_shared_profiles(again)


@pytest.mark.parametrize(
    ("source", "same_generation", "closed"),
    [
        pytest.param(
            ProfileSource(
                graph_base_url="https://other.example.org/core/api",
                sdmx_base_url=_SOURCE.sdmx_base_url,
                un_base_url=_SOURCE.un_base_url,
            ),
            True,
            ["mixer: availability closed", "other: graph closed"],
            id="other-graph-root",
        ),
        pytest.param(
            ProfileSource(
                graph_base_url=_SOURCE.graph_base_url,
                sdmx_base_url="https://sdmx.example.org/core/api",
                un_base_url=_SOURCE.un_base_url,
            ),
            True,
            ["sdmx: availability closed", "mixer: graph closed"],
            id="other-sdmx-root",
        ),
        pytest.param(
            _SOURCE,
            False,
            ["mixer: availability closed", "mixer: graph closed"],
            id="other-generation",
        ),
        pytest.param(
            ProfileSource(
                graph_base_url=_SOURCE.graph_base_url,
                sdmx_base_url=_SOURCE.sdmx_base_url,
                un_base_url=_SOURCE.un_base_url,
                token="graph-token",
            ),
            True,
            ["mixer: availability closed", "mixer: graph closed"],
            id="other-token",
        ),
    ],
)
def test_another_source_gets_its_own_cache_and_releasing_it_leaves_the_first_open(
    monkeypatch, caplog, source, same_generation, closed
):
    """A local rig pointing two surfaces at different roots, or holding different tokens,
    starts and reads through one cache per source. The log names the sources without their
    tokens. Releasing the second closes only its clients; the first still reads."""
    clients = _Clients()
    _install(monkeypatch, clients)
    generation = FakeGeneration(token="g1")
    first = acquire_shared_profiles(source=_SOURCE, generation=generation)

    with caplog.at_level(logging.INFO, logger=shared_profiles.__name__):
        second = acquire_shared_profiles(
            source=source, generation=generation if same_generation else FakeGeneration()
        )
    assert second.cache is not first.cache
    assert [record.args for record in caplog.records] == [([_SOURCE, source],)]
    assert "token" not in caplog.text

    _profile(second)
    release_shared_profiles(second)
    assert clients.events[1:] == closed

    _profile(first)
    assert clients.events[3:] == ["mixer: facet read ended"]
    release_shared_profiles(first)


def test_teardown_waits_for_a_running_flight_before_closing_its_clients(monkeypatch):
    gate = threading.Event()
    clients = _Clients(gate=gate)
    _install(monkeypatch, clients)
    holder = acquire_shared_profiles(source=_SOURCE, generation=FakeGeneration(token="g1"))
    reader = threading.Thread(target=_profile, args=(holder,))
    reader.start()
    assert clients.graph is not None and clients.graph.reading.wait(5)

    releasing = threading.Thread(target=release_shared_profiles, args=(holder,))
    releasing.start()
    # The release is waiting on the flight, so it has closed nothing.
    time.sleep(0.2)
    assert releasing.is_alive()
    assert clients.events == []
    gate.set()
    releasing.join(5)
    reader.join(5)

    assert clients.events == [
        "mixer: facet read ended",
        "mixer: availability closed",
        "mixer: graph closed",
    ]
