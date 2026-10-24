"""Test place-options event launches, resubmit wiring, and task deadlines.

Fixtures resolve fertility rate in Kenya and provide data-bearing Country
peers. The raw-text and dataset-resubmit paths use separate launch sites.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from materialise.graph import SubjectPage
from search.engine.app import create_app, search_surface
from search.engine.enrichment import PLACE_OPTIONS_BUDGET_S, _stream_slot_options
from search.engine.events import PlaceOptionsEvent
from search.engine.options.options_response_cache import OptionsResponseCache
from search.engine.region.result import RegionResult
from search.models import ResubmitIdentityEntry
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._builders import value_slot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE
from tests.engine._search_app import search_app
from tests.engine.test_option_coverage import _fertility_kenya_materialise
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.materialise.conftest import FakeGraphReader

_QUERY = "fertility rate in Kenya"
_KENYA = "country/KEN"
_FERTILITY_SV = "FertilityRate_Person_Female"
# _fertility_kenya_materialise()'s own FakeGraphReader only carries
# graph_obs.json's own facets (built for the observations/coverage-probe
# tests it was written for) -- no node_subjects/node_arcs_many rows, so
# options.population.options_population's own graph branch (through
# statvar_members.enumerate_members) would find no population at all for
# this shape. Reusing that same facets mapping (grounding/coverage keep
# working unchanged) plus a self-authored arcs/subjects pair for
# FertilityRate_Person_Female -- mirroring test_place_options_event.py's own
# convention for a shape the fixture code index does not carry -- gives
# build_place_options_events' own resolve_options_target call something real
# to enumerate, without touching the query's own definite resolution at all.
_FERTILITY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _FERTILITY_SV: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("fertilityRate",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "gender": ("Female",),
        "observationProperties": (),
    },
}
_FERTILITY_SUBJECTS = {
    ("fertilityRate", "measuredProperty"): SubjectPage(dcids=(_FERTILITY_SV,), truncated=False)
}


def _fertility_kenya_materialise_with_population():
    """Add the arcs and subjects needed to derive the option population."""
    obs_graph, materialise_sdmx = _fertility_kenya_materialise()
    combined = FakeGraphReader(
        facets=obs_graph._facets,
        arcs=_FERTILITY_ARCS,
        subjects=_FERTILITY_SUBJECTS,
    )
    return combined, materialise_sdmx


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    event_type = None
    for line in text.split("\n"):
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            import json

            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


def _pinned_date_patch():
    p = patch("search.engine.extract.date")
    return p


class TestPlaceOptionsPipelineLaunch:
    def test_place_options_events_stream_between_response_and_done(self):
        """Raw-text resolution emits both scopes between `response` and `done`.

        The TestClient context starts the lifespan that creates the worker pool.
        """
        materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
        app = create_app(
            graph=FakeGraph(),
            resolver=make_resolver(FakeGraph()),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            _pinned_date_patch() as mock_date,
        ):
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
            resp = client.post(
                "/api/search/resolve",
                json={
                    "input": {"kind": "raw_text", "query": _QUERY},
                    "response_options": {"include_option_coverage": True},
                },
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        event_types = [ev for ev, _ in frames]

        assert "response" in event_types
        assert "place_options" in event_types
        response_idx = event_types.index("response")
        done_idx = event_types.index("done")
        place_options_idxs = [i for i, ev in enumerate(event_types) if ev == "place_options"]
        assert place_options_idxs, "expected at least one trailing place_options event"
        assert all(response_idx < i < done_idx for i in place_options_idxs), (
            f"place_options events must fall strictly between response ({response_idx}) "
            f"and done ({done_idx}); got place_options at {place_options_idxs}"
        )

        spec_id = frames[response_idx][1]["response"]["parts"][0]["spec"]["spec_id"]
        by_scope = {frames[i][1]["scope"]: frames[i][1] for i in place_options_idxs}
        assert set(by_scope) == {"peers", "children"}
        for payload in by_scope.values():
            assert payload["kind"] == "place_options"
            assert payload["spec_id"] == spec_id
            assert payload["role"] == "observationAbout"
            assert payload["anchor_ref"] == _KENYA

        # Peers is non-trivial: several Country-under-Earth peers carry
        # FertilityRate_Person_Female data in the shared graph_obs.json fixture.
        peer_dcids = {opt["value"]["ref"]["ref"] for opt in by_scope["peers"]["options"]}
        assert _KENYA in peer_dcids
        assert len(peer_dcids) > 1

    def test_sse_without_include_option_coverage_never_emits_place_options(self):
        """The event requires both SSE and `include_option_coverage`."""
        materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
        app = search_app(
            graph=FakeGraph(),
            resolver=make_resolver(FakeGraph()),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
        client = TestClient(app, raise_server_exceptions=False)
        with _pinned_date_patch() as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
            resp = client.post(
                "/api/search/resolve",
                json={"input": {"kind": "raw_text", "query": _QUERY}},
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200
        event_types = [ev for ev, _ in _parse_sse(resp.text)]
        assert "response" in event_types
        assert "place_options" not in event_types


_FERTILITY_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "fertilityRate",
    "stat_type": "measuredValue",
}


_GENDER_SLOT = value_slot("how", "gender", "gender", "Female", "Female")


def _fertility_resubmit_entry() -> ResubmitIdentityEntry:
    """A standard-promote resubmit of FertilityRate_Person_Female|country/KEN
    -- the same definite answer _fertility_kenya_materialise_with_population's
    own raw_text fixture above resolves, reached instead through
    dataset_resubmit's identity entry path. Re-posts the shape's own gender
    slot at the value it is already bound to (a promote, not an edit)."""
    entities = [subject_entity(_KENYA)]
    return ResubmitIdentityEntry(
        variable_text="fertility rate",
        identity=make_identity(_FERTILITY_SHAPE, [_GENDER_SLOT], entities=entities),
    )


def _resubmit_sse_body(entry: ResubmitIdentityEntry, *, include_option_coverage: bool) -> dict:
    return {
        "input": {
            "kind": "dataset_resubmit",
            "entries": [entry.model_dump(mode="json")],
        },
        "response_options": {"include_option_coverage": include_option_coverage},
    }


class TestPlaceOptionsResubmitLaunch:
    """Test the dataset-resubmit launch through its relaunch closure."""

    def test_place_options_events_stream_for_a_dataset_resubmit_binding_a_place(self):
        """The resubmitted place emits both scopes between `response` and `done`."""
        materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
        app = create_app(
            graph=FakeGraph(),
            resolver=make_resolver(FakeGraph()),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
        entry = _fertility_resubmit_entry()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/search/resolve",
                json=_resubmit_sse_body(entry, include_option_coverage=True),
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        event_types = [ev for ev, _ in frames]

        assert "response" in event_types
        assert "place_options" in event_types
        response_idx = event_types.index("response")
        done_idx = event_types.index("done")
        place_options_idxs = [i for i, ev in enumerate(event_types) if ev == "place_options"]
        assert place_options_idxs, "expected at least one trailing place_options event"
        assert all(response_idx < i < done_idx for i in place_options_idxs), (
            f"place_options events must fall strictly between response ({response_idx}) "
            f"and done ({done_idx}); got place_options at {place_options_idxs}"
        )

        response_payload = frames[response_idx][1]["response"]
        assert response_payload["status"] == "definite"
        spec_id = response_payload["parts"][0]["spec"]["spec_id"]
        by_scope = {frames[i][1]["scope"]: frames[i][1] for i in place_options_idxs}
        assert set(by_scope) == {"peers", "children"}
        for payload in by_scope.values():
            assert payload["kind"] == "place_options"
            assert payload["spec_id"] == spec_id
            assert payload["role"] == "observationAbout"
            assert payload["anchor_ref"] == _KENYA

        peer_dcids = {opt["value"]["ref"]["ref"] for opt in by_scope["peers"]["options"]}
        assert _KENYA in peer_dcids
        assert len(peer_dcids) > 1

    def test_sse_without_include_option_coverage_never_emits_place_options(self):
        """The resubmit event also requires `include_option_coverage`."""
        materialise_graph, materialise_sdmx = _fertility_kenya_materialise_with_population()
        app = search_app(
            graph=FakeGraph(),
            resolver=make_resolver(FakeGraph()),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
        entry = _fertility_resubmit_entry()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/search/resolve",
            json=_resubmit_sse_body(entry, include_option_coverage=False),
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        event_types = [ev for ev, _ in _parse_sse(resp.text)]
        assert "response" in event_types
        assert "place_options" not in event_types


class TestPlaceOptionsBudget:
    def test_a_slow_place_options_task_is_cancelled_without_blocking_done(self):
        """A task past its 0.05-second budget is cancelled without delaying `done`."""

        async def _never_finishes() -> list[PlaceOptionsEvent]:
            await asyncio.sleep(10)
            return [  # pragma: no cover -- the budget must fire first
                PlaceOptionsEvent(
                    spec_id="spec-1",
                    role="observationAbout",
                    scope="peers",
                    anchor_ref=_KENYA,
                    options=[],
                    truncated=False,
                )
            ]

        async def _body():
            task = asyncio.create_task(_never_finishes())
            region = RegionResult(
                variable_text=None,
                status="definite",
                specs=(),
                no_data_reason=None,
                warnings=(),
                timing_by_step={},
                place_options_task=task,
            )
            warnings_out: list = []
            incomplete: list[str] = []
            t0 = time.monotonic()
            events = [
                ev
                async for ev in _stream_slot_options(
                    [region],
                    deadline=time.monotonic() + 30.0,  # generous shared deadline
                    warnings_out=warnings_out,
                    late_departures_out=[],
                    incomplete_out=incomplete,
                    response=None,
                    answered=(),
                    place_options_budget_s=0.05,
                )
            ]
            elapsed = time.monotonic() - t0

            assert events == [], "a budget-expired place_options task must drop silently"
            assert warnings_out == [], "no warning for a place_options timeout (fail-open, no code)"
            assert incomplete == [], "the bundle holds no place options, so none is missing"
            assert elapsed < 1.0, (
                f"awaiting the place_options task took {elapsed:.2f}s -- "
                "the short budget failed to cut it off"
            )
            assert task.cancelled() or task.done()

            await asyncio.sleep(0)  # let the cancellation finalize
            current = asyncio.current_task()
            pending = {t for t in asyncio.all_tasks() if t is not current and not t.done()}
            assert not pending, f"orphaned tasks remain: {pending}"

        asyncio.run(_body())

    def test_a_shared_deadline_nearer_than_the_budget_ends_the_wait_at_the_shared_deadline(
        self,
    ):
        """The earlier shared deadline ends the wait before the 5-second task budget."""

        async def _never_finishes() -> list[PlaceOptionsEvent]:
            await asyncio.sleep(10)
            return [  # pragma: no cover -- the shared deadline must fire first
                PlaceOptionsEvent(
                    spec_id="spec-1",
                    role="observationAbout",
                    scope="peers",
                    anchor_ref=_KENYA,
                    options=[],
                    truncated=False,
                )
            ]

        async def _body():
            task = asyncio.create_task(_never_finishes())
            region = RegionResult(
                variable_text=None,
                status="definite",
                specs=(),
                no_data_reason=None,
                warnings=(),
                timing_by_step={},
                place_options_task=task,
            )
            warnings_out: list = []
            incomplete: list[str] = []
            t0 = time.monotonic()
            events = [
                ev
                async for ev in _stream_slot_options(
                    [region],
                    deadline=time.monotonic() + 0.05,  # near shared deadline
                    warnings_out=warnings_out,
                    late_departures_out=[],
                    incomplete_out=incomplete,
                    response=None,
                    answered=(),
                    place_options_budget_s=5.0,  # generous own budget
                )
            ]
            elapsed = time.monotonic() - t0

            assert events == [], "a deadline-expired place_options task must drop silently"
            assert warnings_out == [], "no warning for a place_options timeout (fail-open, no code)"
            assert incomplete == [], "the bundle holds no place options, so none is missing"
            assert elapsed < 1.0, (
                f"awaiting the place_options task took {elapsed:.2f}s -- "
                "the near shared deadline failed to cut it off before the own budget"
            )
            assert task.cancelled() or task.done()

            await asyncio.sleep(0)  # let the cancellation finalize
            current = asyncio.current_task()
            pending = {t for t in asyncio.all_tasks() if t is not current and not t.done()}
            assert not pending, f"orphaned tasks remain: {pending}"

        asyncio.run(_body())

    def test_default_budget_is_a_module_constant(self):
        """The task budget is positive and shorter than the request deadline."""
        assert 0 < PLACE_OPTIONS_BUDGET_S < 10


class TestPlaceOptionsPoolShutdown:
    def test_lifespan_shutdown_calls_shutdown_on_the_built_pool(self, monkeypatch):
        """The lifespan builds a default pool carrying the options route's cache, and shuts
        it down exactly once."""
        built: list[_RecordingPool] = []

        class _RecordingPoolFactory:
            @classmethod
            def build(
                cls, *, max_workers: int, options_cache: OptionsResponseCache | None = None
            ) -> _RecordingPool:
                pool = _RecordingPool(max_workers, options_cache=options_cache)
                built.append(pool)
                return pool

        monkeypatch.setattr("search.engine.app.PlaceOptionsPool", _RecordingPoolFactory)

        options_cache = OptionsResponseCache(enabled=True, max_entries=1)
        surface = search_surface(
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=FakeGraphReader(),
            options_response_cache=options_cache,
        )

        lifespan = surface.lifespan
        assert lifespan is not None  # search_surface() always sets one

        async def _body() -> None:
            async with lifespan(FastAPI()):
                assert len(built) == 1, "expected exactly one PlaceOptionsPool built at startup"
                assert built[0].options_cache is options_cache
                assert built[0].shutdown_calls == 0
            assert built[0].shutdown_calls == 1

        asyncio.run(_body())

    def test_an_injected_pool_is_never_shut_down_by_lifespan(self, monkeypatch):
        """An injected pool remains owned by its caller."""
        built: list[_RecordingPool] = []

        class _RecordingPoolFactory:
            @classmethod
            def build(
                cls, *, max_workers: int, options_cache: OptionsResponseCache | None = None
            ) -> _RecordingPool:
                pool = _RecordingPool(max_workers, options_cache=options_cache)
                built.append(pool)
                return pool

        monkeypatch.setattr("search.engine.app.PlaceOptionsPool", _RecordingPoolFactory)

        injected = _RecordingPool(max_workers=1)
        surface = search_surface(
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=FakeGraphReader(),
            place_options_pool=injected,  # ty: ignore[invalid-argument-type]
        )

        lifespan = surface.lifespan
        assert lifespan is not None  # search_surface() always sets one

        async def _body() -> None:
            async with lifespan(FastAPI()):
                assert built == [], "an injected pool must never be rebuilt"
            assert injected.shutdown_calls == 0

        asyncio.run(_body())


class _RecordingPool:
    """A pool stand-in that records shutdown calls."""

    def __init__(
        self, max_workers: int, *, options_cache: OptionsResponseCache | None = None
    ) -> None:
        self.max_workers = max_workers
        self.options_cache = options_cache
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1
