"""Tests for `GET /api/graph/schema/terms/un_system`, covering the UN-declared terms read from
the UN deployment's own client, its cache, its per-page waits and its refusal when the UN origin
is not served."""

from __future__ import annotations

import logging
import threading
import time
from datetime import date

import pytest

from dc_graph import ArcSubject, GraphReadError
from dfo_platform.static_response import FrozenJSON
from graph_api import un_terms as un_terms_module
from graph_api.doors.schema import UN_TERMS_KEY, RefusedSchema
from graph_api.un_terms import UnTermsRateLimitedError, build_un_terms
from origin_catalogue.graph import UnGeneration
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import FakeGraphNodeReader, FakeUnGraph, UnFault, make_client, wait_until

_ROUTE = "/api/graph/schema/terms/un_system"
_UN = "undata/p/UNData"
_BASE = "dc/base/Schema"
_IDENTITY = {"Accept-Encoding": "identity"}

# Shaped like the UN deployment's listings: base terms it lists under base provenances, a
# base-named Class it declares again, its own `undata/` terms and `unCode`, a UN property with
# no `undata` prefix, a term with base and UN provenances both, and one under a provenance
# outside the catalogue table and both known prefixes. The Class listing takes two pages.
_UN_LISTINGS = {
    "Class": (
        (
            ArcSubject(dcid="Dataset", names=("Dataset",), provenances=(_UN,)),
            ArcSubject(dcid="Thing", names=("Thing",), provenances=(_BASE,)),
        ),
        (
            ArcSubject(dcid="undata/AGE", names=("Age",), provenances=(_UN,)),
            ArcSubject(dcid="undata/NO_PROVENANCE", names=("Unsourced",), provenances=()),
        ),
    ),
    "Property": (
        (
            ArcSubject(dcid="name", names=("name",), provenances=(_BASE,)),
            ArcSubject(dcid="other/Sex", names=("Sex",), provenances=("dc/other/X",)),
            ArcSubject(dcid="unCode", names=("UN Data source code",), provenances=(_UN,)),
            ArcSubject(dcid="undata/AGE", names=("Age",), provenances=(_UN,)),
            ArcSubject(dcid="unit", names=("unit",), provenances=(_BASE, _UN)),
        ),
    ),
}
_UN_REFS = ["Dataset", "other/Sex", "unCode", "undata/AGE", "unit"]
_PAGES = [("Class", None), ("Class", "Class-1"), ("Property", None)]


def _rate_limited(status: int = 429, *, retry_after: float | None = None) -> GraphReadError:
    error = GraphReadError("rate limited", upstream_status=status)
    error.retry_after = retry_after
    return error


class _HeldUnGraph(FakeUnGraph):
    """`FakeUnGraph` whose requests from index `held_from` on wait for `release`."""

    def __init__(self, *args, held_from: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.held_from = held_from
        self.release = threading.Event()

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict:
        if len(self.requests) >= self.held_from:
            self.release.wait(timeout=10.0)
        return super().post(path, payload, deadline=deadline)


def test_the_route_serves_only_the_terms_the_un_origin_declares(
    serve_catalogue, caplog: pytest.LogCaptureFixture
):
    serve_catalogue(FakeCatalogue())
    caplog.set_level(logging.DEBUG, logger="graph_api.deps")
    un = FakeUnGraph(_UN_LISTINGS)

    with make_client(un_graph=un) as client:
        response = client.get(_ROUTE, headers=_IDENTITY)
        reads = list(un.requests)
        gzipped = client.get(_ROUTE, headers={"Accept-Encoding": "gzip"})
        not_modified = client.get(
            _ROUTE, headers={**_IDENTITY, "If-None-Match": response.headers["ETag"]}
        )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "generation": None,
        "terms": [
            {
                "ref": "Dataset",
                "label": None,
                "types": ["Class"],
                "provenances": [_UN],
                "declared_by": "un_system",
            },
            {
                "ref": "other/Sex",
                "label": "Sex",
                "types": ["Property"],
                "provenances": ["dc/other/X"],
                "declared_by": "un_system",
            },
            {
                "ref": "unCode",
                "label": "UN Data source code",
                "types": ["Property"],
                "provenances": [_UN],
                "declared_by": "un_system",
            },
            {
                "ref": "undata/AGE",
                "label": "Age",
                "types": ["Class", "Property"],
                "provenances": [_UN],
                "declared_by": "un_system",
            },
            {
                "ref": "unit",
                "label": None,
                "types": ["Property"],
                "provenances": [_BASE, _UN],
                "declared_by": "un_system",
            },
        ],
    }
    assert response.headers["Cache-Control"] == "public, max-age=300"
    assert gzipped.headers["Content-Encoding"] == "gzip"
    assert gzipped.json() == response.json()
    assert not_modified.status_code == 304
    assert reads == _PAGES
    assert un.requests == reads
    # The build runs on the UN terms executor, so no DCP-pool queue wait is logged for it.
    assert [record for record in caplog.records if record.name == "graph_api.deps"] == []


def test_a_new_un_generation_builds_the_body_again(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    un = FakeUnGraph(_UN_LISTINGS)
    today = [date(2026, 11, 19)]
    generation = UnGeneration(today=lambda: today[0])

    with make_client(un_graph=un, un_generation=generation) as client:
        first = client.get(_ROUTE)
        client.get(_ROUTE)
        same_day = len(un.requests)
        today[0] = date(2026, 11, 20)
        next_day = client.get(_ROUTE)

    assert first.status_code == next_day.status_code == 200
    assert same_day == len(_PAGES)
    assert len(un.requests) == 2 * len(_PAGES)


@pytest.mark.parametrize(
    ("faults", "held", "landed"),
    [
        ({}, True, FrozenJSON),
        ({3: _rate_limited(503)}, False, None),
        ({index: _rate_limited() for index in range(3, 6)}, False, RefusedSchema),
    ],
    ids=["running-build", "failed-build", "refused-build"],
)
def test_a_new_un_generation_serves_the_previous_body_until_its_build_lands(
    serve_catalogue,
    monkeypatch: pytest.MonkeyPatch,
    faults: dict[int, UnFault],
    held: bool,
    landed: type | None,
):
    """Once a body is served, a new generation's miss answers that body at once, while its
    build runs, and after it fails or its refusal is kept."""
    serve_catalogue(FakeCatalogue())
    monkeypatch.setattr(un_terms_module, "UN_TERMS_RATE_LIMIT_WAITS_SECONDS", (0.0, 0.0))
    un = _HeldUnGraph(_UN_LISTINGS, faults=faults, held_from=len(_PAGES) if held else 100)
    today = [date(2026, 11, 19)]
    generation = UnGeneration(today=lambda: today[0])

    with make_client(un_graph=un, un_generation=generation) as client:
        first = client.get(_ROUTE)
        today[0] = date(2026, 11, 20)
        started = time.monotonic()
        stale = client.get(_ROUTE)
        waited = time.monotonic() - started
        un.release.set()
        cache = client.app.state.graph_deps.un_terms.cache
        if landed is not None:
            wait_until(
                lambda: isinstance(cache.cached_many([UN_TERMS_KEY]).get(UN_TERMS_KEY), landed)
            )
        else:
            wait_until(lambda: len(un.requests) == len(_PAGES) + 1)
        after = client.get(_ROUTE)

    assert first.status_code == stale.status_code == after.status_code == 200, stale.text
    assert stale.json() == after.json() == first.json()
    assert waited < 5.0


@pytest.mark.parametrize(
    "property_listing",
    [
        (),
        ((ArcSubject(dcid="name", names=("name",), provenances=(_BASE,)),),),
    ],
    ids=["empty", "base-only"],
)
def test_a_listing_with_no_un_term_answers_503_and_is_kept(
    serve_catalogue, property_listing: tuple[tuple[ArcSubject, ...], ...]
):
    serve_catalogue(FakeCatalogue())
    un = FakeUnGraph({"Class": _UN_LISTINGS["Class"], "Property": property_listing or ((),)})

    with make_client(un_graph=un) as client:
        response = client.get(_ROUTE)
        reads = len(un.requests)
        again = client.get(_ROUTE)

    assert response.status_code == again.status_code == 503, response.text
    body = response.json()
    assert (body["retryable"], body["cause"]["code"]) == (True, "listing_incomplete")
    assert 1 <= int(again.headers["Retry-After"]) <= 60
    assert len(un.requests) == reads


@pytest.mark.parametrize(
    ("retry_after", "reads", "header"),
    [(None, 3, (1, 60)), (120.0, 1, (61, 120))],
    ids=["no-retry-after", "upstream-retry-after"],
)
def test_a_rate_limited_build_answers_503_with_retry_after_and_is_kept(
    serve_catalogue,
    monkeypatch: pytest.MonkeyPatch,
    retry_after: float | None,
    reads: int,
    header: tuple[int, int],
):
    serve_catalogue(FakeCatalogue())
    monkeypatch.setattr(un_terms_module, "UN_TERMS_RATE_LIMIT_WAITS_SECONDS", (0.0, 0.0))
    un = FakeUnGraph(
        _UN_LISTINGS,
        faults={index: _rate_limited(retry_after=retry_after) for index in range(6)},
    )

    with make_client(un_graph=un) as client:
        response = client.get(_ROUTE)
        again = client.get(_ROUTE)

    assert response.status_code == again.status_code == 503, response.text
    body = again.json()
    assert (body["retryable"], body["cause"]["code"]) == (True, "upstream_rate_limited")
    assert header[0] <= int(again.headers["Retry-After"]) <= header[1]
    assert len(un.requests) == reads


@pytest.mark.parametrize("un_graph", [True, False], ids=["un-off", "no-un-reader"])
def test_the_route_refuses_with_no_read_where_the_un_origin_is_not_served(
    serve_catalogue, un_graph: bool
):
    serve_catalogue(FakeCatalogue(un_system_enabled=not un_graph))
    un = FakeUnGraph(_UN_LISTINGS)
    graph = FakeGraphNodeReader()

    with make_client(graph=graph, un_graph=un if un_graph else None) as client:
        response = client.get(_ROUTE)

    assert response.status_code == 404, response.text
    body = response.json()
    assert (body["code"], body["retryable"], body["cause"]["code"]) == (
        "not_found",
        False,
        "origin_not_served",
    )
    assert response.headers["Cache-Control"] == "no-store"
    assert un.requests == []
    assert sum(graph.calls.values()) == 0


def test_a_truncated_listing_answers_500_and_is_kept(
    serve_catalogue, monkeypatch: pytest.MonkeyPatch
):
    serve_catalogue(FakeCatalogue())
    monkeypatch.setattr(un_terms_module, "TERM_LISTING_CAP", 3)
    un = FakeUnGraph(_UN_LISTINGS)

    with make_client(un_graph=un) as client:
        response = client.get(_ROUTE)
        reads = len(un.requests)
        again = client.get(_ROUTE)

    assert response.status_code == again.status_code == 500, response.text
    assert response.json()["retryable"] is False
    assert len(un.requests) == reads


@pytest.mark.parametrize(
    ("faults", "waits", "raised", "requests"),
    [
        ({1: _rate_limited()}, (0.0, 0.0), None, [_PAGES[0], _PAGES[1], *_PAGES[1:]]),
        ({1: "repeat"}, (0.0, 0.0), None, [_PAGES[0], _PAGES[1], *_PAGES[1:]]),
        (
            {0: _rate_limited(), 3: _rate_limited(), 4: _rate_limited()},
            (0.0, 0.0),
            429,
            [_PAGES[0], *_PAGES[:2], _PAGES[2], _PAGES[2]],
        ),
        ({1: _rate_limited()}, (120.0,), 429, _PAGES[:2]),
        ({1: _rate_limited(503)}, (0.0, 0.0), 503, _PAGES[:2]),
    ],
    ids=[
        "429-on-its-page",
        "repeated-token-on-its-page",
        "429-past-the-shared-budget",
        "429-past-deadline",
        "other-error-at-once",
    ],
)
def test_a_failed_page_is_read_again_on_its_own_within_the_waits(
    monkeypatch: pytest.MonkeyPatch,
    faults: dict[int, UnFault],
    waits: tuple[float, ...],
    raised: int | None,
    requests: list[tuple[str, str | None]],
):
    monkeypatch.setattr(un_terms_module, "UN_TERMS_RATE_LIMIT_WAITS_SECONDS", waits)
    monkeypatch.setattr(un_terms_module, "REPEATED_PAGE_WAITS_SECONDS", (0.0, 0.0, 0.0))
    reader = FakeUnGraph(_UN_LISTINGS, faults=faults)
    deadline = time.monotonic() + 60.0

    if raised is None:
        terms = build_un_terms(reader=reader, catalogue=FakeCatalogue(), deadline=deadline)
        assert [term.ref for term in terms.terms] == _UN_REFS
    else:
        with pytest.raises(GraphReadError) as error:
            build_un_terms(reader=reader, catalogue=FakeCatalogue(), deadline=deadline)
        assert error.value.upstream_status == raised
    assert reader.requests == requests


def test_a_stop_ends_a_429_wait_at_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(un_terms_module, "UN_TERMS_RATE_LIMIT_WAITS_SECONDS", (30.0,))
    stop = threading.Event()
    stop.set()
    started = time.monotonic()

    with pytest.raises(GraphReadError):
        build_un_terms(
            reader=FakeUnGraph(_UN_LISTINGS, faults={0: _rate_limited()}),
            catalogue=FakeCatalogue(),
            deadline=started + 60.0,
            stop=stop,
        )

    assert time.monotonic() - started < 5.0


def test_a_429_waits_for_the_retry_after_it_names(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(un_terms_module, "UN_TERMS_RATE_LIMIT_WAITS_SECONDS", (0.0,))
    waits: list[float] = []
    monkeypatch.setattr(un_terms_module._Pages, "_sleep", lambda _self, wait: waits.append(wait))

    build_un_terms(
        reader=FakeUnGraph(_UN_LISTINGS, faults={0: _rate_limited(retry_after=7.0)}),
        catalogue=FakeCatalogue(),
        deadline=time.monotonic() + 60.0,
    )

    assert waits == [7.0]


def test_a_429_wait_that_leaves_no_time_to_read_is_not_taken(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(un_terms_module, "UN_TERMS_RATE_LIMIT_WAITS_SECONDS", (3.0,))
    reader = FakeUnGraph(_UN_LISTINGS, faults={0: _rate_limited()})
    started = time.monotonic()

    with pytest.raises(UnTermsRateLimitedError):
        build_un_terms(reader=reader, catalogue=FakeCatalogue(), deadline=started + 6.0)

    assert time.monotonic() - started < 2.0
    assert reader.requests == _PAGES[:1]


def test_a_stop_set_before_a_request_ends_the_build_without_it():
    stop = threading.Event()
    stop.set()
    reader = FakeUnGraph(_UN_LISTINGS)

    with pytest.raises(GraphReadError):
        build_un_terms(
            reader=reader,
            catalogue=FakeCatalogue(),
            deadline=time.monotonic() + 60.0,
            stop=stop,
        )

    assert reader.requests == []


def test_pages_that_add_only_provenances_of_listed_subjects_keep_the_listing_advancing():
    """The Custom DC mixer can page a subject's entries by provenance, so a page that adds a
    provenance of a listed subject is progress."""
    provenances = [f"undata/p/Source{index}" for index in range(5)]
    pages = tuple(
        (ArcSubject(dcid="undata/AGE", names=("Age",), provenances=(provenance,)),)
        for provenance in provenances
    )
    reader = FakeUnGraph({"Class": pages, "Property": pages})

    terms = build_un_terms(
        reader=reader, catalogue=FakeCatalogue(), deadline=time.monotonic() + 60.0
    )

    assert [(term.ref, term.provenances) for term in terms.terms] == [("undata/AGE", provenances)]
