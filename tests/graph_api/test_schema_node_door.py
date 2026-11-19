"""Tests for `GET /api/graph/schema/node/{dcid}`: which deployment each dcid is read from, the
refusals, and the cached body's HTTP caching."""

from __future__ import annotations

import pytest

from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import make_client
from .test_node_view import _ARCS, _FakeGraph

_ROUTE = "/api/graph/schema/node/"
_IDENTITY = {"Accept-Encoding": "identity"}

# The mixer holds the base and DFO terms; the UN deployment holds its own nodes and `unCode`,
# a UN property with no `undata` prefix.
_MIXER_ARCS = tuple(arc for arc in _ARCS if not arc[0].startswith("undata"))
_UN_ARCS = (
    *(arc for arc in _ARCS if arc[0].startswith("undata")),
    ("Thing", "typeOf", "Class"),
    ("Intangible", "typeOf", "Class"),
    ("Intangible", "subClassOf", "Thing"),
    ("Enumeration", "typeOf", "Class"),
    ("Enumeration", "subClassOf", "Intangible"),
    ("unCode", "typeOf", "Property"),
)


def _refusal(response) -> tuple[int, str, bool, str, str]:
    body = response.json()
    return (
        response.status_code,
        body["code"],
        body["retryable"],
        body["cause"]["code"],
        response.headers["Cache-Control"],
    )


def test_a_base_class_is_read_from_the_mixer_and_cached(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "Person", headers=_IDENTITY)
        reads = len(mixer.calls)
        gzipped = client.get(_ROUTE + "Person", headers={"Accept-Encoding": "gzip"})
        not_modified = client.get(
            _ROUTE + "Person", headers={**_IDENTITY, "If-None-Match": response.headers["ETag"]}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert sorted(body) == ["capped", "centre", "generation", "nodes"]
    assert body["centre"]["ref"] == "Person"
    assert body["centre"]["declared_by"] == "base"
    assert [node["ref"] for node in body["nodes"]] == ["Class", "Student", "Thing", "age", "gender"]
    assert body["capped"] == []
    assert response.headers["Cache-Control"] == "public, max-age=300"
    assert gzipped.headers["Content-Encoding"] == "gzip"
    assert gzipped.json() == body
    assert not_modified.status_code == 304
    # Three reads build the view once; the later requests read nothing.
    assert reads == 3
    assert len(mixer.calls) == reads
    assert un.calls == []


def test_a_un_series_is_read_from_the_un_deployment(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "undata/eclac/SI_POV")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["generation"] is None
    assert body["centre"]["declared_by"] == "un_system"
    series = next(node for node in body["nodes"] if node["ref"] == "undata/eclac/SERIES")
    assert series["instances"] == {"refs": ["undata/eclac/SI_POV"], "complete": False}
    assert mixer.calls == []
    assert un.calls


@pytest.mark.parametrize("dcid", ["undata/eclac/SI_POV", "unCode"])
@pytest.mark.parametrize("un_reader", [True, False], ids=["un-off", "no-un-reader"])
def test_the_un_deployment_is_never_read_where_the_un_origin_is_not_served(
    serve_catalogue, dcid: str, un_reader: bool
):
    """A UN id is refused `origin_not_served` with no read; a non-UN id the mixer lacks is
    `node_not_held` without falling to the UN deployment."""
    serve_catalogue(FakeCatalogue(un_system_enabled=not un_reader))
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un if un_reader else None) as client:
        response = client.get(_ROUTE + dcid)

    un_id = dcid.startswith("undata/")
    assert _refusal(response) == (
        404,
        "not_found",
        False,
        "origin_not_served" if un_id else "node_not_held",
        "no-store",
    )
    assert un.calls == []
    assert (mixer.calls == []) is un_id


def test_a_node_the_mixer_lacks_is_read_from_the_un_deployment(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "unCode")
        mixer_reads, un_reads = len(mixer.calls), len(un.calls)
        again = client.get(_ROUTE + "unCode")

    assert response.status_code == again.status_code == 200, response.text
    assert response.json()["centre"]["types"] == ["Property"]
    assert response.json()["generation"] is None
    assert mixer_reads and un_reads
    # The UN view is kept; the mixer's refusal is not, so it is read again.
    assert len(un.calls) == un_reads
    assert len(mixer.calls) == 2 * mixer_reads


def test_a_node_no_deployment_holds_is_refused_and_not_kept(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    mixer, un = _FakeGraph(_MIXER_ARCS), _FakeGraph(_UN_ARCS)

    with make_client(view_graph=mixer, un_view_graph=un) as client:
        response = client.get(_ROUTE + "NoSuchNode")
        reads = len(un.calls)
        client.get(_ROUTE + "NoSuchNode")

    assert _refusal(response) == (404, "not_found", False, "node_not_held", "no-store")
    assert reads
    assert len(un.calls) == 2 * reads
