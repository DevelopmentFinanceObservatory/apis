"""`GET /api/graph/schema/terms/un_system`: the UN-declared terms read from the UN deployment's
own client, its cache, its 429 waits and its refusal when the UN origin is not served."""

from __future__ import annotations

import time

import pytest

from dc_graph import ArcSubject, GraphReadError
from graph_api import un_terms as un_terms_module
from graph_api.un_terms import build_un_terms
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import FakeGraphNodeReader, make_client

_ROUTE = "/api/graph/schema/terms/un_system"
_UN = "undata/p/UNData"
_BASE = "dc/base/Schema"
_IDENTITY = {"Accept-Encoding": "identity"}

# Shaped like the UN deployment's listings: base terms it lists under base provenances, a
# base-named Class it declares again, its own `undata/` terms and `unCode`, a UN property with
# no `undata` prefix.
_UN_LISTINGS = {
    ("Class", "typeOf"): (
        ArcSubject(dcid="Dataset", names=("Dataset",), provenances=(_UN,)),
        ArcSubject(dcid="Thing", names=("Thing",), provenances=(_BASE,)),
        ArcSubject(dcid="undata/AGE", names=("Age",), provenances=(_UN,)),
        ArcSubject(dcid="undata/NO_PROVENANCE", names=("Unsourced",), provenances=()),
    ),
    ("Property", "typeOf"): (
        ArcSubject(dcid="name", names=("name",), provenances=(_BASE,)),
        ArcSubject(dcid="unCode", names=("UN Data source code",), provenances=(_UN,)),
        ArcSubject(dcid="undata/AGE", names=("Age",), provenances=(_UN,)),
    ),
}


class _RateLimited(FakeGraphNodeReader):
    """Answers the first `refusals` listings with HTTP 429, then as the fake does."""

    def __init__(self, refusals: int, *, status: int = 429) -> None:
        super().__init__(arc_subjects=_UN_LISTINGS)
        self._refusals = refusals
        self._status = status

    def node_arc_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        if self._refusals:
            self._refusals -= 1
            self.calls["node_arc_subjects"] += 1
            raise GraphReadError("rate limited", upstream_status=self._status)
        return super().node_arc_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )


def test_the_route_serves_only_the_terms_the_un_origin_declares(serve_catalogue):
    serve_catalogue(FakeCatalogue())
    un = FakeGraphNodeReader(arc_subjects=_UN_LISTINGS)

    with make_client(un_graph=un) as client:
        response = client.get(_ROUTE, headers=_IDENTITY)
        reads = un.calls["node_arc_subjects"]
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
        ],
    }
    assert response.headers["Cache-Control"] == "public, max-age=300"
    assert gzipped.headers["Content-Encoding"] == "gzip"
    assert gzipped.json() == response.json()
    assert not_modified.status_code == 304
    assert reads == 2
    assert un.calls["node_arc_subjects"] == reads


@pytest.mark.parametrize("un_graph", [True, False], ids=["un-off", "no-un-reader"])
def test_the_route_refuses_with_no_read_where_the_un_origin_is_not_served(
    serve_catalogue, un_graph: bool
):
    serve_catalogue(FakeCatalogue(un_system_enabled=not un_graph))
    un = FakeGraphNodeReader(arc_subjects=_UN_LISTINGS)
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
    assert sum(un.calls.values()) == 0
    assert sum(graph.calls.values()) == 0


def test_a_truncated_listing_answers_500(serve_catalogue, monkeypatch: pytest.MonkeyPatch):
    serve_catalogue(FakeCatalogue())
    monkeypatch.setattr(un_terms_module, "TERM_LISTING_CAP", 3)

    with make_client(un_graph=FakeGraphNodeReader(arc_subjects=_UN_LISTINGS)) as client:
        response = client.get(_ROUTE)

    assert response.status_code == 500, response.text
    assert response.json()["retryable"] is False


@pytest.mark.parametrize(
    ("refusals", "status", "waits", "raised", "reads"),
    [
        (3, 429, (0.0, 0.0, 0.0), None, 5),
        (4, 429, (0.0, 0.0, 0.0), 429, 4),
        (1, 429, (120.0,), 429, 1),
        (1, 503, (0.0, 0.0, 0.0), 503, 1),
    ],
    ids=["429-within-waits", "429-past-waits", "429-past-deadline", "other-error-at-once"],
)
def test_a_429_is_read_again_after_each_wait_within_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
    refusals: int,
    status: int,
    waits: tuple[float, ...],
    raised: int | None,
    reads: int,
):
    monkeypatch.setattr(un_terms_module, "UN_RATE_LIMIT_WAITS_SECONDS", waits)
    reader = _RateLimited(refusals, status=status)
    deadline = time.monotonic() + 60.0

    if raised is None:
        terms = build_un_terms(reader=reader, catalogue=FakeCatalogue(), deadline=deadline)
        assert [term.ref for term in terms.terms] == ["Dataset", "unCode", "undata/AGE"]
    else:
        with pytest.raises(GraphReadError) as error:
            build_un_terms(reader=reader, catalogue=FakeCatalogue(), deadline=deadline)
        assert error.value.upstream_status == raised
    assert reader.calls["node_arc_subjects"] == reads
