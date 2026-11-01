"""End-to-end tests for the buffered inline embed vs. SSE trailing
`observations` event, chosen-facet consistency, no-op call counting, and
resubmit parity -- reusing the "total population India" golden (std-01)
with a custom obs override carrying real per-observation values."""

from __future__ import annotations

import threading
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from materialise.graph import Facet as MaterialiseFacet
from search.engine.core import resolve_async, resolve_events
from search.engine.region import materialise_step as region_materialise_step
from search.engine.region import phase_answer as region_phase_answer
from search.engine.region.observations import dispatch as preview_dispatch
from search.engine.retrieval.standard import standard_option_coverage
from search.models import (
    CoverageExact,
    DatasetResubmitInput,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
)
from tests.engine._axis_doubles import TOTAL_POPULATION_INDIA_FAMILY, axis_confirming_reader
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    PINNED_DATE,
    materialise_double_from_obs,
    parse_sse,
    run_pinned,
    with_member_arcs,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.materialise.conftest import FakeGraphReader

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


def _india_materialise_double():
    return materialise_double_from_obs(_IND_OBS)


def _india_materialise_double_with_axes_confirmed():
    """Add graph confirmations for every axis in the golden's identity."""
    graph, sdmx = _india_materialise_double()
    return axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY, inner=graph), sdmx


def _india_request(options: ResolveOptions | None = None) -> ResolveRequest:
    return ResolveRequest(
        input=RawTextInput(query="total population India"), response_options=options
    )


def test_buffered_definite_spec_gets_inline_observations_matching_coverage():
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double()
    result = run_pinned(
        resolve_async(
            _india_request(ResolveOptions(include_observations=True)),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert result.status == "definite"
    spec = result.spec
    assert spec.resolved.observations is not None
    assert spec.resolved.observations.truncated is False
    assert spec.resolved.observations.total_obs == 3
    series = spec.resolved.observations.series[0]
    # Chosen-facet consistency: the preview describes the SAME (stat_var, entity)
    # pair coverage already reports.
    assert series.stat_var.ref == spec.resolved.members[0].ref.ref
    assert [e.ref for e in series.entities] == ["country/IND"]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.observation_count == spec.resolved.observations.total_obs
    values = [p.value for p in series.points]
    assert values == [100.0, 101.0, 102.0]


def test_include_observations_defaults_off_without_extra_fetches(monkeypatch):
    # Facet options read dated facets for every answer (`_maybe_build_facet_options`), so
    # they are stubbed out to count only what observations would add.
    monkeypatch.setattr(
        region_phase_answer, "_maybe_build_facet_options", lambda *_args, **_kwargs: ([], None)
    )

    class _CountingGraph(FakeGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.needs_dates_calls = 0

        def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs):
            if needs_dates:
                self.needs_dates_calls += 1
            return super().observation_facets_batch(
                stat_vars, entities, needs_dates=needs_dates, **kwargs
            )

    graph = _CountingGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double()
    result = run_pinned(
        resolve_async(
            _india_request(),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert result.status == "definite"
    assert result.spec.resolved.observations is None
    # "total population India" has no date window, so materialise's own coverage
    # probe never requests dates either -- zero needs_dates=True calls confirms
    # The absent default adds no extra graph traffic at all.
    assert graph.needs_dates_calls == 0


def test_include_observations_true_coverage_probe_also_uses_needs_dates_true(monkeypatch):
    """With observations enabled, materialisation's coverage probe and the
    later preview fetch must both use effective `needs_dates=True`, so a real
    `LiveGraphClient` cache key matches and no duplicate POST is issued. Recording
    starts at `materialise()` to exclude the earlier existence-only probe.
    """
    from search.engine.clients.graph import _force_full_facets
    from search.engine.region import related_read
    from search.engine.related import NO_RELATED

    # The related-measures reads read summary facets of other StatVars, which the answer
    # never reads again (`related.read_related`), so they are left out.
    monkeypatch.setattr(related_read, "read_related", lambda **_kwargs: NO_RELATED)

    class _RecordingGraph(FakeGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.effective_needs_dates_seen: list[bool] = []

        def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs):
            self.effective_needs_dates_seen.append(needs_dates or _force_full_facets.get())
            return super().observation_facets_batch(
                stat_vars, entities, needs_dates=needs_dates, **kwargs
            )

    graph = _RecordingGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double()

    real_materialise = region_materialise_step.materialise

    def _materialise_from_here(*args, **kwargs):
        graph.effective_needs_dates_seen.clear()
        return real_materialise(*args, **kwargs)

    monkeypatch.setattr(region_materialise_step, "materialise", _materialise_from_here)

    result = run_pinned(
        resolve_async(
            _india_request(ResolveOptions(include_observations=True)),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert result.status == "definite"
    assert result.spec.resolved.observations is not None
    assert graph.effective_needs_dates_seen, "expected at least one observation_facets_batch call"
    assert all(graph.effective_needs_dates_seen), (
        "expected every observation_facets_batch call to have an effective "
        "needs_dates=True once include_observations is on for a dateless query; "
        f"got {graph.effective_needs_dates_seen}"
    )


def test_buffered_observations_fetch_overlaps_slot_options_coverage_probe(monkeypatch):
    """The observations fetch and slot-options probe must overlap. Both calls
    wait at a two-party barrier; sequential dispatch times out. The barrier
    proves overlap without a wall-clock threshold.
    """

    real_option_coverage = standard_option_coverage
    real_build_observations = preview_dispatch._build_observations
    barrier = threading.Barrier(2, timeout=5)

    def _barrier_option_coverage(*args, **kwargs):
        barrier.wait()
        return real_option_coverage(*args, **kwargs)

    def _barrier_build_observations(*args, **kwargs):
        barrier.wait()
        return real_build_observations(*args, **kwargs)

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _barrier_option_coverage
    )
    monkeypatch.setattr(preview_dispatch, "_build_observations", _barrier_build_observations)

    result = run_pinned(
        resolve_async(
            ResolveRequest(
                input=RawTextInput(query="fertility rate in Kenya"),
                response_options=ResolveOptions(
                    include_observations=True, include_option_coverage=True
                ),
            ),
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
        )
    )
    assert result.parts[0].status == "definite"


def test_sse_streams_trailing_observations_event_and_response_stays_null():
    """SSE emits observations between `response` and `done`, while the
    serialized response omits them. Check the wire payload because
    `model_copy(update=...)` does not validate nesting.
    """
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
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

    assert "observations" in event_types
    response_idx = event_types.index("response")
    done_idx = event_types.index("done")
    obs_idxs = [i for i, ev in enumerate(event_types) if ev == "observations"]
    assert obs_idxs, "expected at least one trailing observations event"
    assert all(response_idx < i < done_idx for i in obs_idxs), (
        f"observations events must fall strictly between response ({response_idx}) "
        f"and done ({done_idx}); got observations at {obs_idxs}"
    )

    obs_payload = frames[obs_idxs[0]][1]["observations"]
    assert obs_payload["total_obs"] == 3
    assert obs_payload["truncated"] is False

    response_payload = frames[response_idx][1]["response"]
    assert "observations" not in response_payload["parts"][0]["spec"]["resolved"]


def test_sse_observations_default_off():
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "total population India"}},
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    assert "observations" not in [event_type for event_type, _ in frames]


def test_sse_response_waits_on_the_eager_facet_fetch_not_on_the_trailing_one(monkeypatch):
    """The response waits for the eager facet fetch used by availability.
    The trailing preview fetch starts later and cannot delay the response.
    A delayed `materialise_graph.default_facet_policy`, which every preview read consults,
    isolates the eager read.
    """
    import time

    probe_delay = 0.3

    class _SlowGraphReader(FakeGraphReader):
        def default_facet_policy(self, *, stat_var, axis):
            time.sleep(probe_delay)
            return super().default_facet_policy(stat_var=stat_var, axis=axis)

    graph = FakeGraph(obs=_IND_OBS)
    _fast_graph, materialise_sdmx = _india_materialise_double()
    materialise_graph = _SlowGraphReader(
        facets={
            ("Count_Person", "country/IND"): [
                MaterialiseFacet(
                    facet_id="facet1",
                    provenance_id="dc/base/WorldPopulationProspects",
                    measurement_method=None,
                    unit=None,
                    obs_count=3,
                    earliest_date="2020",
                    latest_date="2022",
                )
            ]
        }
    )

    async def _body():
        t0 = time.monotonic()
        response_at: float | None = None
        obs_at: float | None = None
        async for ev in resolve_events(
            _india_request(ResolveOptions(include_observations=True)),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            stream_slot_options=True,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        ):
            elapsed = time.monotonic() - t0
            if ev.kind == "response" and response_at is None:
                response_at = elapsed
            if ev.kind == "observations" and obs_at is None:
                obs_at = elapsed
        assert response_at is not None
        assert obs_at is not None
        assert response_at >= probe_delay, (
            f"response arrived at {response_at:.2f}s -- expected it to wait on the "
            f"eager, {probe_delay}s facet-determining fetch"
        )
        assert obs_at >= response_at

    run_pinned(_body())


def test_resubmit_buffered_gets_inline_observations():
    """A resubmitted identity gets the same treatment via its existing trailing plumbing:
    buffered embeds inline."""
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double_with_axes_confirmed()
    base = run_pinned(
        resolve_async(
            _india_request(),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert base.status == "definite"
    spec = base.spec

    resubmit_req = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(
                    variable_text="a resubmitted identity",
                    identity=spec.definition.identity,
                )
            ]
        ),
        response_options=ResolveOptions(include_observations=True),
    )
    r = run_pinned(
        resolve_async(
            resubmit_req,
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=with_member_arcs(materialise_graph, graph),
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert r.status == "definite"
    assert r.spec.resolved.observations is not None
    assert r.spec.resolved.observations.total_obs == 3


def test_resubmit_sse_streams_trailing_observations_event():
    """A resubmitted identity over SSE defers to a trailing observations event, same as raw_text."""
    graph = FakeGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = _india_materialise_double_with_axes_confirmed()
    base = run_pinned(
        resolve_async(
            _india_request(),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
    ).parts[0]
    assert base.status == "definite"
    spec = base.spec

    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=with_member_arcs(materialise_graph, graph),
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "a resubmitted identity",
                        "identity": spec.definition.identity.model_dump(mode="json"),
                    }
                ],
            },
            "response_options": {"include_observations": True},
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert "observations" in event_types
    response_idx = event_types.index("response")
    done_idx = event_types.index("done")
    obs_idx = event_types.index("observations")
    assert response_idx < obs_idx < done_idx
    response_payload = frames[response_idx][1]["response"]
    assert "observations" not in response_payload["parts"][0]["spec"]["resolved"]
