"""Rate-table refusals reach the wire as their registered problems on every Search transport.

`RateTableInvalidError` and `UpstreamVersionChangedError` propagate out of the observation
fetch, which raises them from `materialise_preview` when a conversion's rate table fails
admission or moves mid-load. `classify_search_exception` maps them to `capability_unavailable`
(501) and `upstream_version_changed` (503, retryable). Each test posts a raw-text resolve
through the Search app and raises from the patched `materialise_preview`.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from materialise.errors import UpstreamVersionChangedError
from materialise.rate_tables import RateTableInvalidError
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.config import settings
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE, materialise_double_from_obs, parse_sse
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_IND_OBS = {
    "Count_Person|country/IND": [
        {
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "provenanceId": "dc/base/WorldPopulationProspects",
            "observations": [
                {"date": "2020", "value": 100.0},
                {"date": "2021", "value": 101.0},
                {"date": "2022", "value": 102.0},
            ],
        }
    ],
}

_BODY = {
    "input": {"kind": "raw_text", "query": "total population India"},
    "response_options": {"include_observations": True},
}

_TABLE_PROVENANCE = RECIPE_BODIES["ConversionRecipeDAC"].table_provenance
_INVALID_DETAIL = (
    f"provenance {_TABLE_PROVENANCE!r} carries no lastDataRefreshDate; "
    "its rate table cannot be versioned"
)
_MOVED_DETAIL = (
    f"provenance {_TABLE_PROVENANCE!r}'s lastDataRefreshDate changed while its rate "
    "table was loading, and the retry did not converge either"
)


def _post(exc: Exception, *, sse: bool):
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)

    def _raise(*_args, **_kwargs):
        raise exc

    headers = {"Accept": "text/event-stream"} if sse else {}
    with (
        patch("search.engine.extract.date") as mock_date,
        patch("search.engine.region.observations.fetch.materialise_preview", _raise),
    ):
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        return client.post("/api/search/resolve", json=_BODY, headers=headers)


_CASES = [
    pytest.param(
        RateTableInvalidError(_INVALID_DETAIL),
        501,
        "capability_unavailable",
        False,
        _INVALID_DETAIL,
        id="capability_unavailable",
    ),
    pytest.param(
        UpstreamVersionChangedError(_MOVED_DETAIL),
        503,
        "upstream_version_changed",
        True,
        _MOVED_DETAIL,
        id="upstream_version_changed",
    ),
]


@pytest.mark.parametrize(("exc", "status", "code", "retryable", "detail"), _CASES)
def test_buffered_resolve_serves_the_rate_table_problem(exc, status, code, retryable, detail):
    resp = _post(exc, sse=False)

    assert resp.status_code == status
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["type"] == f"/codes/{code}"
    assert body["code"] == code
    assert body["status"] == status
    assert body["retryable"] is retryable
    assert body["detail"] == detail
    assert body["capability_version"] == settings().capability_version


@pytest.mark.parametrize(("exc", "status", "code", "retryable", "detail"), _CASES)
def test_streamed_resolve_ends_on_the_rate_table_error_event(exc, status, code, retryable, detail):
    resp = _post(exc, sse=True)

    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    assert frames[-1][0] == "error"
    error = frames[-1][1]
    assert error["code"] == code
    assert error["status"] == status
    assert error["retryable"] is retryable
    assert error["detail"] == detail
