"""SSE resubmits emit the response before slow option coverage, then emit the
slot_options event. Buffered responses retain inline coverage, and no-data
responses emit only response and done events.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from search.engine.core import resolve_events
from search.engine.enrichment import _launch_trailing_tasks
from search.engine.region import option_coverage as _events_mod
from search.engine.region import resolve_spec_resubmit
from search.engine.region.facet_probe import facet_probe
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.models import (
    DatasetResubmitInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import (
    make_identity,
    subject_entity,
)
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    materialise_double_from_obs,
    member_walk_reader,
    offline_resolve,
    walk_options,
    with_member_arcs,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver

_PROBE = facet_probe(["sv"], {"observationAbout": ["country/KEN"]}, years=None)

_FERTILITY_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "fertilityRate",
    "stat_type": "measuredValue",
}
_FERTILITY_SV = "FertilityRate_Person_Female"
_GENDER_SLOT = _value_slot("how", "gender", "gender", "Female", "Female")


def _identity_entry(payload: ResubmitIdentityPayload) -> DatasetResubmitInput:
    """Wrap one resubmit identity payload into the dataset_resubmit request that
    carries it -- the single-identity shape the door answers with one part."""
    return DatasetResubmitInput(
        entries=[
            ResubmitIdentityEntry(
                variable_text="a resubmitted identity",
                identity=payload.identity,
            )
        ]
    )


@pytest.mark.parametrize(
    ("launch_slot_options", "launch_observations", "has_slot_closure", "has_observations_closure"),
    [
        (True, False, True, True),
        (False, True, True, True),
        (True, True, True, True),
        (False, False, True, True),
        (True, True, False, False),
    ],
)
def test_launch_trailing_tasks_gates_each_closure_independently(
    launch_slot_options: bool,
    launch_observations: bool,
    has_slot_closure: bool,
    has_observations_closure: bool,
):
    calls: list[tuple] = []
    spec = SimpleNamespace(spec_id="spec-id", resolved=SimpleNamespace(slot_options=["options"]))

    def launch_slot(*args):
        calls.append(("slot", *args))
        return "slot-task"

    def launch_observations(*args):
        calls.append(("observations", *args))
        return "observations-task"

    region = RegionResult(
        variable_text="fertility",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        facet_probe=_PROBE,
        slot_options_relaunch=launch_slot if has_slot_closure else None,
        observations_relaunch=launch_observations if has_observations_closure else None,
    )

    launched = _launch_trailing_tasks(
        region,
        launch_slot_options=launch_slot_options,
        launch_observations=launch_observations,
    )

    assert launched.slot_options_task == (
        "slot-task" if launch_slot_options and has_slot_closure else None
    )
    assert launched.observations_task == (
        "observations-task" if launch_observations and has_observations_closure else None
    )
    expected_calls = []
    if launch_slot_options and has_slot_closure:
        expected_calls.append(("slot", "spec-id", ["options"], _PROBE))
    if launch_observations and has_observations_closure:
        expected_calls.append(("observations", "spec-id"))
    assert calls == expected_calls


def _fertility_resubmit_payload() -> ResubmitIdentityPayload:
    """A standard-promote resubmit (FertilityRate_Person_Female|country/KEN)
    with a real confirmed StatVar, so slot_options gets populated with real
    coverage."""
    entities = [subject_entity("country/KEN")]
    return ResubmitIdentityPayload(
        identity=make_identity(_FERTILITY_SHAPE, [_GENDER_SLOT], entities=entities),
    )


def _no_data_resubmit_payload() -> ResubmitIdentityPayload:
    """A fertility identity declaring none of its members' constraints elects no member,
    a no_data short-circuit."""
    entities = [subject_entity("country/KEN")]
    return ResubmitIdentityPayload(
        identity=make_identity(_FERTILITY_SHAPE, [], entities=entities),
    )


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in body.splitlines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


# ---------------------------------------------------------------------------
# SSE: response not blocked on a slow probe; trailing event order
# ---------------------------------------------------------------------------


def test_resubmit_response_event_arrives_before_slow_probe_completes(monkeypatch):
    """The response arrives before the 1-second coverage probe completes.
    A completion flag avoids timing thresholds that vary with runner load."""
    real_option_coverage = _events_mod.standard_option_coverage
    probe_delay = 1.0
    probe_finished = threading.Event()

    def _slow_option_coverage(*args, **kwargs):
        time.sleep(probe_delay)  # option_coverage is sync; this is the real cost
        result = real_option_coverage(*args, **kwargs)
        probe_finished.set()
        return result

    monkeypatch.setattr(_events_mod, "standard_option_coverage", _slow_option_coverage)

    async def _body():
        t0 = time.monotonic()
        response_seen = False
        probe_finished_at_response: bool | None = None
        slot_options_at: float | None = None
        async for ev in resolve_events(
            ResolveRequest(
                input=_identity_entry(_fertility_resubmit_payload()),
                response_options=ResolveOptions(include_option_coverage=True),
            ),
            graph=FakeGraph(),
            llm=FakeLLM(),
            stream_slot_options=True,
            sdmx=FakeSdmxClient(),
            materialise_graph=member_walk_reader(FakeGraph()),
        ):
            if ev.kind == "response" and not response_seen:
                response_seen = True
                probe_finished_at_response = probe_finished.is_set()
            if ev.kind == "slot_options" and slot_options_at is None:
                slot_options_at = time.monotonic() - t0

        assert response_seen, "expected a response event"
        assert slot_options_at is not None, "expected the slow probe to still complete"
        assert probe_finished_at_response is False, (
            "the probe had already completed by the time `response` arrived -- "
            "it appears to be running inline before the response"
        )
        assert slot_options_at >= probe_delay, (
            f"slot_options arrived at {slot_options_at:.2f}s, before the "
            f"{probe_delay}s probe could have finished"
        )

    asyncio.run(_body())


def test_resolve_resubmit_default_stream_slot_options_false_matches_prior_behaviour():
    """Buffered callers omit `stream_slot_options` and receive inline coverage."""
    payload = _fertility_resubmit_payload()
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            f"{_FERTILITY_SV}|country/KEN": _load_json("graph_obs.json")[
                f"{_FERTILITY_SV}|country/KEN"
            ]
        }
    )
    region = resolve_spec_resubmit(
        inp=payload,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            FakeGraph(),
            EngineOptions(
                include_option_coverage=True,
                materialise_graph=with_member_arcs(materialise_graph, FakeGraph()),
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )
    assert region.status == "definite"
    assert region.slot_options_task is None
    assert region.slot_options_relaunch is None
    covered = [
        o
        for so in region.specs[0].resolved.slot_options
        for o in so.options
        if o.coverage is not None
    ]
    assert covered


# ---------------------------------------------------------------------------
# no_data short-circuit: response + done only, on both transports
# ---------------------------------------------------------------------------


def test_no_data_resubmit_sse_emits_response_and_done_only():
    app = search_app(
        graph=FakeGraph(),
        resolver=make_resolver(FakeGraph()),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=member_walk_reader(FakeGraph()),
    )
    client = TestClient(app, raise_server_exceptions=False)
    payload = _no_data_resubmit_payload()
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "a resubmitted identity",
                        "identity": payload.identity.model_dump(mode="json"),
                    }
                ],
            },
            "response_options": {"include_option_coverage": True},
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    event_types = [ev for ev, _ in _parse_sse(resp.text)]
    assert event_types == ["response", "done"]

    response_payload = next(data for ev, data in _parse_sse(resp.text) if ev == "response")
    assert response_payload["response"]["status"] == "no_data"


def test_no_data_resubmit_buffered_yields_no_data():
    req = ResolveRequest(
        input=_identity_entry(_no_data_resubmit_payload()),
        response_options=ResolveOptions(include_option_coverage=True),
    )
    r = offline_resolve(req)
    assert r.status == "no_data"


async def _collect_kinds(request: ResolveRequest, *, stream_slot_options: bool) -> list[str]:
    return [
        ev.kind
        async for ev in resolve_events(
            request,
            graph=FakeGraph(),
            llm=FakeLLM(),
            stream_slot_options=stream_slot_options,
            sdmx=FakeSdmxClient(),
            materialise_graph=member_walk_reader(FakeGraph()),
        )
    ]


def test_no_data_resubmit_events_stream_response_and_done_only():
    """With SSE and option coverage enabled, a no_data result emits no trailing
    events because it has no relaunch closure."""
    kinds = asyncio.run(
        _collect_kinds(
            ResolveRequest(
                input=_identity_entry(_no_data_resubmit_payload()),
                response_options=ResolveOptions(include_option_coverage=True),
            ),
            stream_slot_options=True,
        )
    )
    assert kinds == ["response", "done"]
