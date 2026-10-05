"""Every graph read on the search paths sees the catalogue bound at request entry.

The origin client reads the bound catalogue on the calling thread and raises when none is
bound. A recording proxy stands in for it here: it reads the catalogue the same way before
each call and records what it saw and on which thread. The catalogue service serves a
permissive catalogue with its own build id, so a read that saw the test thread's fake
catalogue, or none, is told apart from one that saw the request's, and recall keeps the
candidates the answer needs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import origin_catalogue.service as catalogue_service_module
from origin_catalogue import CatalogueNotBound, current_catalogue
from search.engine.app import create_app
from search.engine.extract import Extraction, ExtractionResult
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE
from tests.engine._options_fixtures import (
    _KENYA_DCID,
    _options_request,
    _place_graph,
    _place_materialise_graph,
)
from tests.engine.test_place_options_trailing import _fertility_kenya_materialise_with_population
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, fake_catalogue_service

_REQUEST_BUILD = "request-catalogue"

# Threads of the pools that request work fans out to, as opposed to the request's own
# `asyncio.to_thread` hops.
_POOL_THREAD_PREFIXES = ("obs-preview", "search-place-options", "ThreadPoolExecutor")


@dataclass(frozen=True)
class _Read:
    method: str
    build_id: str | None
    thread: str


class _CatalogueReadingGraph:
    """Forwards every call to `inner` after reading the bound catalogue on the calling thread."""

    def __init__(self, inner: object, reads: list[_Read]) -> None:
        self._inner = inner
        self._reads = reads

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def call(*args: object, **kwargs: object) -> object:
            try:
                build_id: str | None = current_catalogue().build_id
            except CatalogueNotBound:
                build_id = None
            self._reads.append(_Read(name, build_id, threading.current_thread().name))
            return attribute(*args, **kwargs)

        return call


@pytest.fixture
def request_catalogue() -> Iterator[None]:
    """Serve a permissive catalogue with its own build id from the process-wide service."""
    service = fake_catalogue_service(FakeCatalogue(build_id=_REQUEST_BUILD))
    with catalogue_service_module._singleton.poisoned_for_test(service):
        yield


async def _place_less_extract(query: str, *, llm: object) -> tuple[ExtractionResult, None]:
    extraction = Extraction(kind="resolution", variables=["fertility rate"], entities=[], dates=[])
    return ExtractionResult(extraction=extraction, qualifiers={}), None


def _assert_every_read_saw_the_request_catalogue(reads: list[_Read]) -> None:
    assert reads, "the request made no graph read, so the binding went untested"
    unbound = [read for read in reads if read.build_id != _REQUEST_BUILD]
    assert unbound == [], f"reads that missed the request's catalogue: {unbound}"


def _search_app(reads: list[_Read]) -> object:
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
    graph = FakeGraph()
    return create_app(
        graph=_CatalogueReadingGraph(graph, reads),  # ty: ignore[invalid-argument-type]
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=_CatalogueReadingGraph(materialise_graph, reads),  # ty: ignore[invalid-argument-type]
        materialise_sdmx=materialise_sdmx,
    )


@pytest.mark.usefixtures("request_catalogue")
def test_a_place_less_search_reads_the_request_catalogue() -> None:
    """A place-less search's graph reads run under the request's catalogue."""
    reads: list[_Read] = []
    with (
        TestClient(_search_app(reads), raise_server_exceptions=True) as client,  # ty: ignore[invalid-argument-type]
        patch("search.engine.entry.raw_text.extract", _place_less_extract),
    ):
        response = client.post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": "fertility rate"},
                "response_options": {"include_observations": True},
            },
        )

    assert response.status_code == 200, response.text
    warnings = [w["code"] for w in response.json()["parts"][0]["warnings"]]
    assert "place_scope_defaulted" in warnings
    _assert_every_read_saw_the_request_catalogue(reads)


@pytest.mark.usefixtures("request_catalogue")
def test_a_streamed_search_reads_the_request_catalogue_on_its_pool_threads() -> None:
    """Place options and the observation preview fan out to their own pools."""
    reads: list[_Read] = []
    with (
        TestClient(_search_app(reads), raise_server_exceptions=True) as client,  # ty: ignore[invalid-argument-type]
        patch("search.engine.extract.date") as mock_date,
    ):
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        response = client.post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": "fertility rate in Kenya"},
                "response_options": {
                    "include_observations": True,
                    "include_option_coverage": True,
                },
            },
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200, response.text
    assert "event: place_options" in response.text
    _assert_every_read_saw_the_request_catalogue(reads)
    pool_threads = {
        read.thread.rsplit("_", 1)[0]
        for read in reads
        if read.thread.startswith(_POOL_THREAD_PREFIXES)
    }
    assert {"obs-preview", "search-place-options"} <= pool_threads, sorted(
        {read.thread for read in reads}
    )


@pytest.mark.usefixtures("request_catalogue")
def test_an_options_request_reads_the_request_catalogue_on_every_path() -> None:
    reads: list[_Read] = []
    graph = _place_graph()
    app = create_app(
        graph=_CatalogueReadingGraph(graph, reads),  # ty: ignore[invalid-argument-type]
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=_CatalogueReadingGraph(_place_materialise_graph(), reads),  # ty: ignore[invalid-argument-type]
    )
    body = _options_request(
        entities={"observationAbout": [_KENYA_DCID]}, include_coverage=True
    ).model_dump(by_alias=True, mode="json")
    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.post("/api/search/options", json=body)

    assert response.status_code == 200, response.text
    _assert_every_read_saw_the_request_catalogue(reads)


def test_the_recording_proxy_reports_a_read_with_no_catalogue_bound() -> None:
    """Control: a read on a thread with no binding is recorded as unbound."""
    reads: list[_Read] = []
    proxy = _CatalogueReadingGraph(FakeGraph(), reads)
    call: Callable[[], object] = lambda: proxy.node_label("country/KEN")  # noqa: E731
    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert [read.build_id for read in reads] == [None]
