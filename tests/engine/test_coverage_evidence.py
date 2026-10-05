"""Tests `CoverageExact.evidence` in buffered and streaming responses (#627).
SSE response events precede observation fetches; trailing events carry the series.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from search.engine.clients.graph import Facet
from search.engine.core import resolve_async
from search.engine.extract import DateRequest
from search.engine.retrieval.coverage import CoverageForm, answer_coverage
from search.engine.retrieval.read_plan import plan_standard, plan_years
from search.models import CoverageExact, RawTextInput, ResolveOptions, ResolveRequest
from tests.engine._harness import (
    PINNED_DATE,
    materialise_double_from_obs,
    parse_sse,
    run_pinned,
    sdmx_double,
)
from tests.engine._search_app import search_app
from tests.engine.test_parts_shape import _ETHIOPIA_POP_OBS, _mixed_obs_app, _post_mixed_sse
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver


def _one_series(facets: list[Facet]) -> dict[tuple[str, str], list[Facet]]:
    """`facets` as the candidates of one preview series."""
    return {("Count_Person", "country/IND"): list(facets)}


def _coverage(facets: list[Facet], date_request: DateRequest | None = None):
    plan = plan_standard(
        _one_series(facets), members=["Count_Person"], years=plan_years(date_request)
    )
    return answer_coverage(plan, CoverageForm())


# ---------------------------------------------------------------------------
# Unit: the field's two values at the builders
# ---------------------------------------------------------------------------


def _facet(
    obs_count: int = 10,
    dates: list[str] | None = None,
    provenance_id: str | None = "dc/base/WorldPopulationProspects",
) -> Facet:
    return Facet(
        earliest_date="2010",
        latest_date="2024",
        obs_count=obs_count,
        dates=dates if dates is not None else [],
        provenance_id=provenance_id,
        facet_id="f1",
    )


def test_plan_coverage_stamps_facet_metadata() -> None:
    """A count summed from facet metadata is stamped `facet_metadata`: coverage reads the
    graph's facet index through the plan and fetches no series."""
    cov = _coverage([_facet(obs_count=402), _facet(obs_count=18)])
    assert isinstance(cov, CoverageExact) or cov.kind == "exact"
    assert cov.evidence == "facet_metadata"


def test_plan_coverage_stamps_facet_metadata_on_the_windowed_path() -> None:
    """The windowed exact count (summed from in-window dates) is facet
    metadata too, not an observed count: the dates come from the same facet
    index."""
    cov = _coverage(
        [_facet(obs_count=3, dates=["2020", "2021", "2022"])],
        DateRequest(window=None, latest=True),
    )
    assert cov.kind == "exact"
    assert cov.evidence == "facet_metadata"


def test_a_narrowed_plans_coverage_keeps_the_evidence() -> None:
    """A facet-filter recount leaves the field alone: the narrowed plan's coverage reads the
    same facet index, so the provenance does not change."""
    plan = plan_standard(
        _one_series([_facet(obs_count=402), _facet(obs_count=18, provenance_id="dc/base/Other")]),
        members=["Count_Person"],
        years=None,
    )
    narrowed = answer_coverage(
        plan.narrowed(lambda _sv, f: f.provenance_id == "dc/base/WorldPopulationProspects"),
        CoverageForm(),
    )
    assert narrowed.kind == "exact"
    assert narrowed.evidence == "facet_metadata"
    assert narrowed.observation_count == 402


def test_evidence_defaults_to_facet_metadata() -> None:
    """The default is the honest one: a CoverageExact built without naming a
    provenance claims facet metadata, never a delivered series."""
    from search.models import CoverageExact

    cov = CoverageExact(has_data=True, observation_count=5)
    assert cov.evidence == "facet_metadata"
    assert CoverageExact.model_json_schema()["properties"]["evidence"]["default"] == (
        "facet_metadata"
    )


# ---------------------------------------------------------------------------
# End-to-end buffered: a delivered preview restamps `observed`
# ---------------------------------------------------------------------------

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


def _india_request(options: ResolveOptions | None = None) -> ResolveRequest:
    return ResolveRequest(
        input=RawTextInput(query="total population India"), response_options=options
    )


def _buffered_resolve(options: ResolveOptions | None):
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    return (
        run_pinned(
            resolve_async(
                _india_request(options),
                graph=graph,
                llm=FakeLLM(),
                sdmx=sdmx_double(),
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            )
        ).parts[0],
        graph,
        materialise_graph,
        materialise_sdmx,
    )


def test_buffered_preview_delivers_series_and_stamps_observed():
    """A buffered resolve whose preview delivered a series reports the count as
    `observed`: the series is present in the same payload, so the count is
    backed by fetchable data."""
    result, *_ = _buffered_resolve(options=ResolveOptions(include_observations=True))
    assert result.status == "definite"
    spec = result.spec
    observations = spec.resolved.observations
    assert observations is not None
    assert observations.series, "the preview must have delivered a series for this assertion"
    assert observations.omitted is None
    assert spec.resolved.coverage.kind == "exact"
    assert spec.resolved.coverage.evidence == "observed"


def test_buffered_all_pairs_failure_stamps_facet_metadata_and_omits():
    """Failed preview calls keep coverage at facet metadata (#627).
    The omission names the spec with `observations_probe_failed`.
    """
    import search.engine.region.observations.fetch as region_fetch
    from materialise import UpstreamResponseError
    from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED

    def _raising_preview(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        raise UpstreamResponseError("role divergence")

    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    with patch.object(region_fetch, "materialise_preview", _raising_preview):
        result = run_pinned(
            resolve_async(
                _india_request(ResolveOptions(include_observations=True)),
                graph=graph,
                llm=FakeLLM(),
                sdmx=sdmx_double(),
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            )
        ).parts[0]

    assert result.status == "definite", f"expected a definite part, got {result.status}"
    spec = result.spec
    observations = spec.resolved.observations
    assert observations is not None
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert spec.resolved.coverage.kind == "exact"
    assert spec.resolved.coverage.evidence == "facet_metadata"


# ---------------------------------------------------------------------------
# End-to-end SSE: `response` ships before any fetch
# ---------------------------------------------------------------------------


def test_sse_response_carries_facet_metadata_while_trailing_event_has_series():
    """A streaming `response` event ships before any observation fetch, so its
    coverage stays `facet_metadata`; the trailing `observations` event carries
    the delivered series for the same spec."""
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=sdmx_double(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": "total population India"},
                "response_options": {"include_observations": True},
            },
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]

    response_idx = event_types.index("response")
    obs_idxs = [i for i, ev in enumerate(event_types) if ev == "observations"]
    assert obs_idxs, "expected a trailing observations event"

    response_payload = frames[response_idx][1]["response"]
    response_coverage = response_payload["parts"][0]["spec"]["resolved"]["coverage"]
    assert response_coverage["kind"] == "exact"
    assert response_coverage["evidence"] == "facet_metadata"

    trailing = frames[obs_idxs[0]][1]["observations"]
    assert trailing["series"], "the trailing event must carry the delivered series"
    assert trailing["total_obs"] == 3


def test_sse_response_carries_no_observed_evidence_across_a_mixed_response():
    """The response event carries facet metadata, even for a definite part whose preview has data.
    A trailing observation event confirms that the series was fetched.
    """
    default_obs = _load_json("graph_obs.json")
    graph = FakeGraph(obs={**default_obs, **_ETHIOPIA_POP_OBS})
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_ETHIOPIA_POP_OBS)
    frames = _post_mixed_sse(_mixed_obs_app(graph, materialise_graph, materialise_sdmx))
    event_types = [ev for ev, _ in frames]
    response_idx = event_types.index("response")
    parts = frames[response_idx][1]["response"]["parts"]
    statuses = [part["status"] for part in parts]
    assert "definite" in statuses and "candidates" in statuses, (
        f"the fixture must produce a mixed response, got {statuses}"
    )

    exact = [
        part["spec"]["resolved"]["coverage"]
        for part in parts
        if part.get("spec", {}).get("resolved", {}).get("coverage", {}).get("kind") == "exact"
    ]
    assert exact, "expected at least one exact coverage on the response event"
    assert all(cov["evidence"] == "facet_metadata" for cov in exact), (
        f"no SSE response-event coverage may read 'observed'; got {exact}"
    )

    trailing = [
        frames[i][1]["observations"] for i, (ev, _) in enumerate(frames) if ev == "observations"
    ]
    assert any(obs["series"] for obs in trailing), (
        "the definite part's preview must deliver a series, or the assertion above is vacuous"
    )
