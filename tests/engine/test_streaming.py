"""Tests SSE deadlines, branch failures, terminal errors, cancellation, and trailing tasks.

Generator tests patch extraction and variable resolution; route tests use FastAPI's test client.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import json
import logging
import time
from collections.abc import Coroutine
from typing import Any, Literal

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

import dc_graph
import search.engine.app as app_module
from code_index.shared import get_index
from dfo_platform import ProblemSpec, refusal
from entity_resolution import NotFound
from search.engine.app import _ENVELOPE, classify_search_exception
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.core import resolve_async, resolve_events
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.events import (
    DoneEvent,
    InterpretationEvent,
    PlaceOptionsEvent,
    ResponseEvent,
    SentenceEvent,
    StepEvent,
)
from search.engine.extract import ExtractedEntity, Extraction, ExtractionResult
from search.engine.region import RegionResult
from search.engine.region.codes import PLACE_SCOPE_DEFAULTED, QUERY_QUALIFIER_DROPPED
from search.engine.region.sentence_events import _build_shape_event
from search.engine.region.state import EngineOptions
from search.engine.shape import STANDARD_SHAPE_LABEL, ShapeDraft, SlotKeyDraft
from search.engine.spec_build import build_spec
from search.models import (
    BindingValue,
    CoverageBare,
    DefinitePart,
    GraphRef,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
)
from search.render.tokens import sentence_from_plan
from tests.engine._base_identity import subject_entity
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX, make_request
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


async def _slow_resolver_resolve(self, texts: list[str]) -> list[NotFound]:
    """Stand-in for EntityResolver.resolve that blocks past the shared
    deadline, for the entity pre-resolution deadline tests below.
    fanout._resolve_entities_bounded calls resolver.resolve() to resolve a
    fan-out's named entities. It does not call graph.resolve_entity, so this
    patches EntityResolver.resolve itself. Matches _await_bounded's own
    docstring: dispatches its blocking work to a thread via
    asyncio.to_thread, so cancelling the outer await on timeout detaches
    from it without stopping the underlying thread."""
    await asyncio.to_thread(time.sleep, 0.3)
    return [NotFound() for _ in texts]  # pragma: no cover -- deadline fires first


def _fake_app(*, graph=None, llm=None, sdmx=None, resolver=None):
    """Build an offline app; defaults support the data-bearing health ODA query."""
    _graph = graph if graph is not None else FakeGraph()
    return search_app(
        graph=_graph,
        resolver=resolver if resolver is not None else make_resolver(_graph),
        llm=llm if llm is not None else FakeLLM(),
        sdmx=sdmx if sdmx is not None else FakeSdmxClient(),
    )


def _no_data_region(variable: str) -> RegionResult:
    return RegionResult(
        variable_text=variable,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


def _mock_extract(variables: list[str]):
    async def _inner(query: str, *, llm=None):
        return (
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution", variables=variables, entities=[], dates=[]
                ),
                qualifiers={},
            ),
            None,
        )

    return _inner


def test_dropped_qualifier_note_matches_raw_text_sentence_event(monkeypatch):
    """The raw-text stream and terminal definite response share qualifier notes."""
    query = "total population for top 10 countries by population"
    qualifier = "top 10 countries by population"

    async def _extract_with_dropped_qualifier(_query: str, *, llm=None):
        return (
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution",
                    variables=["total population"],
                    entities=[],
                    dates=[],
                    dropped_qualifiers=[qualifier],
                ),
                qualifiers={},
            ),
            None,
        )

    monkeypatch.setattr("search.engine.entry.raw_text.extract", _extract_with_dropped_qualifier)

    class _SelectFirstShapeLLM(FakeLLM):
        async def generate_structured(self, *, prompt, system, schema):
            if schema is _UniversalBindOutput:
                return (
                    _UniversalBindOutput(
                        chosen_shape_index=0,
                        bindings=[],
                        also_plausible=None,
                        ask=None,
                    ),
                    None,
                )
            return await super().generate_structured(prompt=prompt, system=system, schema=schema)

    async def _body():
        events = await _collect(
            resolve_events(
                ResolveRequest(
                    input=RawTextInput(query=query),
                    response_options=ResolveOptions(),
                ),
                graph=FakeGraph(),
                llm=_SelectFirstShapeLLM(),
                sdmx=EMPTY_SDMX,
            )
        )
        response = next(event.response for event in events if isinstance(event, ResponseEvent))
        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        interpretation = part.spec
        assert interpretation.resolved.scope is not None
        assert interpretation.resolved.scope.axis == "observationAbout"
        assert PLACE_SCOPE_DEFAULTED in {item.code for item in part.warnings}
        warning = next(
            item for item in response.diagnostics.warnings if item.code == QUERY_QUALIFIER_DROPPED
        )
        assert warning.severity == "degraded"
        assert warning.message == f'The requested qualifier "{qualifier}" was not applied.'

        sentence_events = [event for event in events if isinstance(event, SentenceEvent)]
        assert sentence_events
        assert sentence_events[-1].sentence_plan == interpretation.resolved.sentence_plan

    _run(_body())


async def _collect(gen):
    return [ev async for ev in gen]


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into [(event_type, data_dict), ...], skipping `: ping` comments."""
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


def _assert_no_orphaned_tasks(exclude: set) -> None:
    pending = {t for t in asyncio.all_tasks() if t not in exclude and not t.done()}
    assert not pending, f"Orphaned tasks remain: {pending}"


@pytest.mark.parametrize(
    ("population_type", "measured_property", "shape_label", "expected_sentence"),
    [
        (
            "MATERNAL_MORTALITY_EVENT",
            "count",
            "Maternal mortality ratio",
            "Count.",
        ),
        ("Person", "fertilityRate", STANDARD_SHAPE_LABEL, "Fertility rate."),
    ],
)
def test_shape_event_suppresses_only_unresolved_population_type_ref(
    population_type: str,
    measured_property: str,
    shape_label: str,
    expected_sentence: str,
):
    """The skeleton omits unresolved population labels but keeps a safe measured-property fallback."""
    shape_draft = ShapeDraft(
        shape_id=f"{population_type}/{measured_property}/measuredValue",
        label=shape_label,
        pop_type_dcid=population_type,
        meas_prop_dcid=measured_property,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
    )

    event = _run(_build_shape_event(shape_draft, variable_index=0, graph=FakeGraph()))

    assert sentence_from_plan(event.sentence_plan) == expected_sentence


def test_shape_event_publishes_the_classifications_roots_for_an_index_sourced_slot():
    """An unbound index-carried slot lists classification roots only."""
    shape_draft = ShapeDraft(
        shape_id="test_crs_shape",
        label="test CRS shape",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),),
        slot_taxonomy_index_sourced=True,
    )

    index = get_index()
    roots = [r.dcid for r in index.codes if r.code_system == "crsPurpose" and r.parent is None]
    labels = {r.dcid: r.label for r in index.codes if r.dcid in roots}

    event = _run(
        _build_shape_event(
            shape_draft,
            variable_index=0,
            graph=FakeGraph(nodes={d: {"label": labels[d]} for d in roots}),
        )
    )

    assert len(event.slot_options) == 1
    slot_options = event.slot_options[0]
    dcids = {o.value.ref.dcid for o in slot_options.options}
    assert dcids <= set(roots)
    assert dcids  # at least one root survived the shape-event's own label lookup
    assert all(o.relation == "root" for o in slot_options.options)
    assert all(o.member_count is not None for o in slot_options.options)


def test_shape_event_option_labels_come_from_the_code_index_not_the_raw_graph_name(
    monkeypatch,
) -> None:
    """The code-index label override strips a raw code-system suffix in shape-event options."""
    shape_draft = ShapeDraft(
        shape_id="test_crs_shape",
        label="test CRS shape",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),),
        slot_taxonomy_index_sourced=True,
    )

    index = get_index()
    clean_labels = {
        r.dcid: r.label for r in index.codes if r.code_system == "crsPurpose" and r.parent is None
    }
    assert clean_labels, "fixture assumption: crsPurpose has at least one root"

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                dcid: {"arcs": {"name": {"nodes": [{"value": f"{label} (CRS Purpose)"}]}}}
                for dcid, label in clean_labels.items()
                if dcid in payload.get("nodes", [])
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    graph = LiveGraphClient(base="http://fake")

    event = _run(_build_shape_event(shape_draft, variable_index=0, graph=graph))

    assert len(event.slot_options) == 1
    options = event.slot_options[0].options
    assert options
    for option in options:
        dcid = option.value.ref.dcid
        assert option.value.ref.label == clean_labels[dcid]
        assert "(CRS Purpose)" not in option.value.ref.label


# ---------------------------------------------------------------------------
# Soft-deadline partials
# ---------------------------------------------------------------------------


def test_soft_deadline_yields_partial_response_with_timed_out_done(monkeypatch, search_settings):
    """A slow leg yields one partial response and `done(timed_out=True)`."""
    search_settings(request_timeout_s=0.05)
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["fast", "slow"]))

    cancelled: list[str] = []

    async def _stub(variable, **kwargs):
        try:
            if variable == "slow":
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.append(variable)
            raise
        return _no_data_region(variable)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        events = await _collect(
            resolve_events(
                make_request("test soft deadline"),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=EMPTY_SDMX,
            )
        )

        responses = [e for e in events if isinstance(e, ResponseEvent)]
        dones = [e for e in events if isinstance(e, DoneEvent)]
        assert len(responses) == 1
        assert len(dones) == 1
        assert dones[0].timed_out is True

        await asyncio.sleep(0.1)  # let the cancelled "slow" leg finalize
        assert "slow" in cancelled
        _assert_no_orphaned_tasks({asyncio.current_task()})

    _run(_body())


def test_soft_deadline_single_variable_n1_path(monkeypatch, search_settings):
    """A hung single leg ends with a timed-out response and done event."""
    search_settings(request_timeout_s=0.05)
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["only"]))

    async def _stub(variable, **kwargs):
        await asyncio.sleep(10)
        return _no_data_region(variable)  # pragma: no cover — never reached

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        events = await _collect(
            resolve_events(
                make_request("test n1 soft deadline"),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=EMPTY_SDMX,
            )
        )

        dones = [e for e in events if isinstance(e, DoneEvent)]
        assert len(dones) == 1
        assert dones[0].timed_out is True
        await asyncio.sleep(0.1)
        _assert_no_orphaned_tasks({asyncio.current_task()})

    _run(_body())


def test_resolve_async_surfaces_timed_out_in_diagnostics(monkeypatch, search_settings):
    """Buffered responses expose the stream's timeout state in diagnostics."""
    search_settings(request_timeout_s=0.05)
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["fast", "slow"]))

    async def _stub(variable, **kwargs):
        await asyncio.sleep(10 if variable == "slow" else 0.01)
        return _no_data_region(variable)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        result = await resolve_async(
            make_request("test buffered soft deadline"),
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=EMPTY_SDMX,
        )
        assert result.diagnostics.timed_out is True

        await asyncio.sleep(0.1)  # let the cancelled "slow" leg finalize
        _assert_no_orphaned_tasks({asyncio.current_task()})

    _run(_body())


def test_resolve_async_timed_out_false_on_normal_completion():
    """The counterpart to the above: a request that completes within the
    deadline must report timed_out=False, not just leave it unset.
    """

    async def _body():
        result = await resolve_async(
            make_request("total population India"),
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=EMPTY_SDMX,
        )
        assert result.diagnostics.timed_out is False

    _run(_body())


# ---------------------------------------------------------------------------
# Per-branch failure isolation
# ---------------------------------------------------------------------------


def test_per_branch_non_infra_failure_degrades_without_killing_stream(monkeypatch):
    """A plain exception degrades one N>=2 leg to no data."""
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["good", "bad"]))

    async def _stub(variable, **kwargs):
        if variable == "bad":
            raise RuntimeError("simulated branch failure")
        return _no_data_region(variable)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        events = await _collect(
            resolve_events(
                make_request("test per-branch failure"),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=EMPTY_SDMX,
            )
        )

        responses = [e for e in events if isinstance(e, ResponseEvent)]
        dones = [e for e in events if isinstance(e, DoneEvent)]
        assert len(responses) == 1
        assert len(dones) == 1
        assert dones[0].timed_out is False
        # Both variable texts appear in the response after one leg fails.
        body = responses[0].response
        assert [part.variable_text for part in body.parts] == ["good", "bad"]

    _run(_body())


def test_infra_failure_fails_loud_after_all_branches_finish(monkeypatch):
    """An `EngineInfraError` propagates after every sibling leg finishes."""
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["good", "bad"]))
    good_finished = asyncio.Event()

    async def _stub(variable, **kwargs):
        if variable == "bad":
            raise GraphInfraError("simulated infra failure")
        await asyncio.sleep(0.02)
        good_finished.set()
        return _no_data_region(variable)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        with pytest.raises(GraphInfraError):
            await _collect(
                resolve_events(
                    make_request("test infra failure"),
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                )
            )
        assert good_finished.is_set(), "the infra failure must not short-circuit sibling legs"

    _run(_body())


def test_n1_any_exception_fails_loud(monkeypatch):
    """A non-infrastructure exception on the N=1 path propagates."""
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["only"]))

    async def _stub(variable, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        with pytest.raises(RuntimeError, match="simulated failure"):
            await _collect(
                resolve_events(
                    make_request("test n1 failure"),
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                )
            )

    _run(_body())


# ---------------------------------------------------------------------------
# Sanitized terminal error (route layer)
# ---------------------------------------------------------------------------


def test_sse_terminal_error_is_sanitized(monkeypatch):
    """The route emits one sanitized error event for a pipeline failure."""
    raw_message = "INTERNAL_HOST_1.2.3.4 connection refused"

    async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
        raise RuntimeError(raw_message)
        yield  # pragma: no cover — makes this a generator function

    monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)

    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "test"}},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["error"], f"Expected only an error event, got {event_types}"

    detail = frames[0][1]["detail"]
    assert detail == "Internal error"
    assert frames[0][1]["code"] == "internal_error"
    assert raw_message not in detail, "Raw exception text must NOT appear in the SSE error"


def test_sse_terminal_error_carries_engine_input_error_code(monkeypatch):
    """The SSE error event preserves an `EngineInputError` code."""

    async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
        raise EngineInputError("posted shape does not match its members", code="shape_mismatch")
        yield  # pragma: no cover — makes this a generator function

    monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)

    app = _fake_app()
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
                        "identity": {
                            "shape": {"kind": "statvar_schema", "population_type": "some_shape"},
                            "slots": [],
                            "entities": {},
                            "requested_window": {"kind": "all"},
                        },
                    }
                ],
            }
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["error"]

    payload = frames[0][1]
    # code is set, so detail resolves to the platform registry's per-code
    # default (codes.CODE_CATALOG) rather than the generic "Bad request" --
    # that fallback is reserved for a code-less (routing-failure) raise.
    assert (
        payload["detail"] == "The posted shape does not match the shape this request resolves to."
    )
    assert payload["code"] == "shape_mismatch"


def test_sse_terminal_error_survives_a_classify_that_itself_raises(monkeypatch, caplog):
    """A classification failure still produces a terminal error event."""

    async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
        raise RuntimeError("pipeline failure")
        yield  # pragma: no cover — makes this a generator function

    def _broken_classify(exc: BaseException) -> ProblemSpec | None:
        raise ValueError("classify itself is broken")

    monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)
    monkeypatch.setattr("search.engine.app.classify_search_exception", _broken_classify)

    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="search.engine.app"):
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    assert [ev for ev, _ in frames] == ["error"], frames

    payload = frames[0][1]
    assert payload["status"] == 500
    assert payload["code"] == "internal_error"
    assert "classify itself is broken" not in resp.text
    assert "classifying" in caplog.text
    assert "RuntimeError" in caplog.text


def test_sse_terminal_error_survives_a_classify_returning_an_unregistered_code(monkeypatch, caplog):
    """When classify_search_exception returns a code absent from the search
    surface's own catalogue, the terminal event must fall back to the
    platform's own internal_error rather than writing the bogus code onto
    the wire.
    """

    async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
        raise RuntimeError("pipeline failure")
        yield  # pragma: no cover — makes this a generator function

    def _bogus_classify(exc: BaseException) -> ProblemSpec | None:
        return ProblemSpec(status=409, code="not_a_real_code", retryable=False)

    monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)
    monkeypatch.setattr("search.engine.app.classify_search_exception", _bogus_classify)

    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="search.engine.app"):
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    assert [ev for ev, _ in frames] == ["error"], frames

    payload = frames[0][1]
    assert payload["status"] == 500
    assert payload["code"] == "internal_error"
    assert "not_a_real_code" not in resp.text
    assert "classifying" in caplog.text


@pytest.mark.parametrize(
    "exc",
    [
        EngineInputError("posted shape does not match its members", code="shape_mismatch"),
        EngineInputError("unknown shape_id"),
        GraphInfraError("simulated upstream failure", upstream_status=404),
        GraphInfraError("simulated transient failure"),
        TimeoutError(),
        RuntimeError("simulated unclassified failure"),
        RequestValidationError(
            errors=[
                {
                    "type": "literal_error",
                    "loc": ("body", "schema_version"),
                    "msg": "Input should be 'spec1'",
                    "input": "spec2",
                }
            ],
            body={"schema_version": "spec2"},
        ),
    ],
    ids=[
        "engine_input_error_with_code",
        "engine_input_error_without_code",
        "engine_infra_error_upstream_4xx",
        "engine_infra_error_no_upstream_status",
        "timeout",
        "unclassified",
        "request_validation_error_schema_version",
    ],
)
def test_sse_terminal_error_matches_the_json_paths_envelope_render(monkeypatch, exc):
    """The SSE terminal error event's own problem fields must equal what
    `Envelope.render` produces for the identical classification -- the
    structural counterpart to the identity test above, which only proves the
    two paths call the same function object, not that they agree on the
    wire body it produces.
    """

    async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
        raise exc
        yield  # pragma: no cover — makes this a generator function

    monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)

    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "test"}},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    assert [ev for ev, _ in frames] == ["error"]
    sse_payload = frames[0][1]

    spec = classify_search_exception(exc) or refusal.classify_default(exc)
    expected = json.loads(bytes(_ENVELOPE.render(spec).body))

    for field in (
        "status",
        "code",
        "detail",
        "pointer",
        "alternatives",
        "truncated",
        "retryable",
        "issues",
    ):
        assert sse_payload.get(field) == expected.get(field), field


def test_sse_exactly_one_terminal_event_on_success():
    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "health ODA grants from USA to Ethiopia"}},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    terminals = [ev for ev, _ in frames if ev in ("done", "error")]
    assert terminals == ["done"]


# ---------------------------------------------------------------------------
# Disconnect cancels fan-out; no orphaned tasks
# ---------------------------------------------------------------------------


def test_disconnect_cancels_fanout_no_orphaned_tasks(monkeypatch):
    """Cancelling the consumer cancels pending fan-out legs."""
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["fast", "slow"]))
    cancelled: list[str] = []

    async def _stub(variable, **kwargs):
        try:
            if variable == "slow":
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.append(variable)
            raise
        return _no_data_region(variable)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        interpretation_seen = asyncio.Event()
        collected = []

        async def _consume():
            async for ev in resolve_events(
                make_request("test disconnect"), graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX
            ):
                collected.append(ev)
                if isinstance(ev, InterpretationEvent):
                    interpretation_seen.set()

        task = asyncio.create_task(_consume())
        await asyncio.wait_for(asyncio.shield(interpretation_seen.wait()), timeout=5.0)
        await asyncio.sleep(0.02)  # let the fast leg get underway too

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.1)  # let cancelled tasks finalize

        assert "slow" in cancelled, "The slow branch should have seen CancelledError when cancelled"
        _assert_no_orphaned_tasks({asyncio.current_task()})

    _run(_body())


# ---------------------------------------------------------------------------
# Same-shape collapse: reconciled observations tasks, no orphaned probe
# ---------------------------------------------------------------------------

_WHERE_PROP = "whereProp"
_WHERE_KEY = SlotKey(
    axis="where",
    property=GraphRef(dcid=_WHERE_PROP, label=_WHERE_PROP),
    label=_WHERE_PROP,
)

_SAME_SHAPE = Shape(
    shape_id="Person/count/",
    label="Test",
    population_type=GraphRef(dcid="Person", label="Person"),
    measured_property=GraphRef(dcid="count", label="count"),
    stat_type=GraphRef(dcid="measuredValue", label="measuredValue"),
    measurement_qualifier=None,
    measurement_denominator=None,
    slot_keys=[_WHERE_KEY],
    member_count=1,
)


def _where_slot(value_dcid: str) -> Slot:
    return Slot(
        key=_WHERE_KEY,
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(dcid=value_dcid, label=value_dcid), value_kind="entity")
        ),
    )


async def _fake_obs_probe(delay: float) -> tuple[tuple, tuple]:
    if delay:
        await asyncio.sleep(delay)
    return (), ()


def _same_shape_region_with_obs_task(
    variable_text: str,
    *,
    sv_dcid: str,
    where_dcid: str,
    created: list[asyncio.Task],
    delay: float,
) -> RegionResult:
    """Build a same-shape result with a pending observations task and relaunch callback."""
    stat_var_ref = GraphRef(dcid=sv_dcid, label=sv_dcid)
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_SAME_SHAPE,
        slots=[_where_slot(where_dcid)],
        stat_vars=[
            StatVar(
                ref=stat_var_ref,
                shape_id=_SAME_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
    )
    task = asyncio.create_task(_fake_obs_probe(delay))
    created.append(task)

    def _relaunch(spec_id: str) -> asyncio.Task:
        t = asyncio.create_task(_fake_obs_probe(0.0))
        created.append(t)
        return t

    return RegionResult(
        variable_text=variable_text,
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        observations_task=task,
        observations_relaunch=_relaunch,
    )


def test_same_shape_merge_reconciles_every_observations_task(monkeypatch):
    """A same-shape merge cancels superseded tasks and drains the merged region's relaunch."""
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["var A", "var B"]))
    created_tasks: list[asyncio.Task] = []

    async def _stub(variable, **kwargs):
        if variable == "var A":
            return _same_shape_region_with_obs_task(
                "var A",
                sv_dcid="sv_a",
                where_dcid="country/KEN",
                created=created_tasks,
                delay=2.0,
            )
        return _same_shape_region_with_obs_task(
            "var B",
            sv_dcid="sv_b",
            where_dcid="country/ETH",
            created=created_tasks,
            delay=2.0,
        )

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        events = await _collect(
            resolve_events(
                make_request("var A and var B by place"),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=EMPTY_SDMX,
                stream_slot_options=True,
            )
        )

        responses = [e for e in events if isinstance(e, ResponseEvent)]
        assert len(responses) == 1
        parts = responses[0].response.parts
        assert all(isinstance(part, DefinitePart) for part in parts), (
            "the two same-shape legs must have collapsed onto a single merged definite spec"
        )
        assert parts[0].spec.spec_id == parts[1].spec.spec_id, (
            "the two same-shape legs must have collapsed onto a single merged definite spec"
        )

        # Two eagerly-launched legs + one relaunch for the merged region's new spec_id.
        assert len(created_tasks) == 3
        assert all(t.done() for t in created_tasks), (
            "every observations task tied to a same-shape merge must be done or "
            "cancelled by the time the stream is fully drained -- a still-pending "
            "task here means a merged-away region's observations fetch leaked"
        )

    _run(_body())


# ---------------------------------------------------------------------------
# Same-shape collapse: place_options relaunched under the merged spec_id only
# ---------------------------------------------------------------------------


def _fake_place_options_probe(
    events: list[PlaceOptionsEvent], *, delay: float
) -> Coroutine[Any, Any, list[PlaceOptionsEvent]]:
    async def _inner() -> list[PlaceOptionsEvent]:
        if delay:
            await asyncio.sleep(delay)
        return events

    return _inner()


def _place_options_event(
    spec_id: str, anchor: str, scope: Literal["peers", "children"]
) -> PlaceOptionsEvent:
    return PlaceOptionsEvent(
        spec_id=spec_id,
        role="observationAbout",
        scope=scope,
        anchor_ref=anchor,
        options=[],
        truncated=False,
    )


def _same_shape_region_with_place_options_task(
    variable_text: str,
    *,
    sv_dcid: str,
    where_dcid: str,
    created: list[asyncio.Task],
    relaunch_calls: list[tuple[str, str]],
) -> RegionResult:
    """Build a same-shape result with a pending place-options task and relaunch callback."""
    stat_var_ref = GraphRef(dcid=sv_dcid, label=sv_dcid)
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_SAME_SHAPE,
        slots=[_where_slot(where_dcid)],
        stat_vars=[
            StatVar(
                ref=stat_var_ref,
                shape_id=_SAME_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[subject_entity(where_dcid)],
        coverage=CoverageBare(has_data=True),
    )
    # The merge must cancel this pending task and relaunch under the merged spec id.
    task = asyncio.create_task(
        _fake_place_options_probe(
            [_place_options_event(spec.spec_id, where_dcid, "peers")], delay=10.0
        )
    )
    created.append(task)

    def _relaunch(spec_id: str) -> asyncio.Task[list[PlaceOptionsEvent]]:
        relaunch_calls.append((variable_text, spec_id))
        t = asyncio.create_task(
            _fake_place_options_probe(
                [
                    _place_options_event(spec_id, where_dcid, "peers"),
                    _place_options_event(spec_id, where_dcid, "children"),
                ],
                delay=0.0,
            )
        )
        created.append(t)
        return t

    return RegionResult(
        variable_text=variable_text,
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        place_options_task=task,
        place_options_relaunch=_relaunch,
    )


def test_same_shape_merge_relaunches_place_options_under_the_merged_spec_id_only(monkeypatch):
    """A same-shape merge cancels original tasks and emits options once under the merged spec id."""
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _mock_extract(["var A", "var B"]))
    created_tasks: list[asyncio.Task] = []
    relaunch_calls: list[tuple[str, str]] = []
    leg_specs: dict[str, str] = {}

    async def _stub(variable, **kwargs):
        where_dcid = "country/KEN" if variable == "var A" else "country/ETH"
        region = _same_shape_region_with_place_options_task(
            variable,
            sv_dcid="sv_a" if variable == "var A" else "sv_b",
            where_dcid=where_dcid,
            created=created_tasks,
            relaunch_calls=relaunch_calls,
        )
        leg_specs[variable] = region.spec.spec_id
        return region

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stub)

    async def _body():
        events = await _collect(
            resolve_events(
                make_request("var A and var B by place"),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=EMPTY_SDMX,
                stream_slot_options=True,
            )
        )

        responses = [e for e in events if isinstance(e, ResponseEvent)]
        assert len(responses) == 1
        parts = responses[0].response.parts
        assert all(isinstance(part, DefinitePart) for part in parts), (
            "the two same-shape legs must have collapsed onto a single merged definite spec"
        )
        merged_spec_id = parts[0].spec.spec_id
        assert parts[1].spec.spec_id == merged_spec_id

        pre_merge_spec_ids = set(leg_specs.values())
        assert merged_spec_id not in pre_merge_spec_ids, (
            "the merge must mint a brand-new spec_id, distinct from either leg's own"
        )

        place_options_events = [e for e in events if isinstance(e, PlaceOptionsEvent)]
        assert place_options_events, "expected the merged region's place_options relaunch to stream"
        assert {ev.spec_id for ev in place_options_events} == {merged_spec_id}, (
            "no place_options event may carry a pre-merge spec_id"
        )
        assert {ev.scope for ev in place_options_events} == {"peers", "children"}
        scope_spec_pairs = [(ev.scope, ev.spec_id) for ev in place_options_events]
        assert len(scope_spec_pairs) == len(set(scope_spec_pairs)), (
            "no duplicate (scope, spec_id) place_options pair may stream"
        )

        assert relaunch_calls == [(relaunch_calls[0][0], merged_spec_id)]
        assert len(relaunch_calls) == 1, "exactly one leg's relaunch closure must fire, once"

        # Both original per-leg place_options tasks plus the one relaunch task
        # must be done (cancelled or completed) by the time the stream is
        # fully drained -- no stale pre-merge task leaks into the background.
        assert len(created_tasks) == 3
        assert all(t.done() for t in created_tasks)
        assert sum(t.cancelled() for t in created_tasks) == 2, (
            "both original legs' own place_options_task must be cancelled outright, "
            "never allowed to complete their own (irrelevant) 10s sleep"
        )

    _run(_body())


# ---------------------------------------------------------------------------
# Trailing enrichment awaits observations and slot-option tasks. Graph errors degrade one task;
# other exceptions fail the stream.
# ---------------------------------------------------------------------------


async def _raise_after(exc: BaseException, *, delay: float = 0.0):
    if delay:
        await asyncio.sleep(delay)
    raise exc


async def _return_after(result: tuple, *, delay: float = 0.0) -> tuple:
    if delay:
        await asyncio.sleep(delay)
    return result


def test_drain_reraises_a_non_infra_error_from_a_trailing_task():
    """A programming error from a trailing task propagates to the SSE route."""
    from search.engine.enrichment import _stream_slot_options

    async def _body():
        task = asyncio.create_task(_raise_after(TypeError("bug in a trailing task")))
        region = dataclasses.replace(_no_data_region("v"), slot_options_task=task)
        with pytest.raises(TypeError):
            async for _ev in _stream_slot_options(
                [region],
                deadline=time.monotonic() + 5,
                warnings_out=[],
                late_departures_out=[],
                response=None,
                answered=(),
            ):
                pass

    _run(_body())


def test_drain_degrades_a_graph_infra_error_without_suppressing_a_sibling_task(caplog):
    """A trailing `GraphInfraError` is logged while sibling task results still stream."""
    from search.engine.enrichment import _stream_slot_options
    from search.engine.errors import GraphInfraError

    async def _body():
        failing_task = asyncio.create_task(_raise_after(GraphInfraError("boom")))
        ok_task = asyncio.create_task(_return_after((("ok-event",), ()), delay=0.05))
        region = dataclasses.replace(
            _no_data_region("v"), slot_options_task=failing_task, observations_task=ok_task
        )
        return [
            ev
            async for ev in _stream_slot_options(
                [region],
                deadline=time.monotonic() + 5,
                warnings_out=[],
                late_departures_out=[],
                response=None,
                answered=(),
            )
        ]

    with caplog.at_level(logging.WARNING):
        events = _run(_body())

    assert events == ["ok-event"], (
        "the failing slot_options_task must not suppress the sibling observations_task's own result"
    )
    assert any("trailing enrichment probe task failed" in r.message for r in caplog.records)


def test_drain_retrieves_a_sibling_tasks_exception_from_the_same_wait_batch():
    """The drain propagates one task's error and retrieves its sibling's simultaneous error."""
    from search.engine.enrichment import _stream_slot_options

    async def _body():
        loop = asyncio.get_running_loop()
        unretrieved: list[dict] = []

        def _handler(loop, context):
            if "never retrieved" in context.get("message", ""):
                unretrieved.append(context)

        loop.set_exception_handler(_handler)

        failing_a = asyncio.create_task(_raise_after(ValueError("a")))
        failing_b = asyncio.create_task(_raise_after(ValueError("b")))
        region = dataclasses.replace(
            _no_data_region("v"), slot_options_task=failing_a, observations_task=failing_b
        )

        with pytest.raises(ValueError):
            async for _ev in _stream_slot_options(
                [region],
                deadline=time.monotonic() + 5,
                warnings_out=[],
                late_departures_out=[],
                response=None,
                answered=(),
            ):
                pass

        del failing_a, failing_b, region
        gc.collect()
        assert unretrieved == [], f"a sibling task's exception was never retrieved: {unretrieved}"

    _run(_body())


def test_deadline_synthesizes_observations_timed_out_event_for_a_pending_task():
    """A pending observations task yields a timeout omission and warning."""
    from search.engine.enrichment import _stream_slot_options
    from search.engine.events import ObservationsEvent
    from search.engine.region.codes import OBSERVATIONS_TIMED_OUT
    from search.models import Spec
    from tests.conftest import minimal_spec

    spec = Spec.model_validate(minimal_spec("s-timeout"))

    async def _body():
        never_finishes = asyncio.create_task(_return_after(((), ()), delay=10))
        region = dataclasses.replace(
            _no_data_region("v"),
            status="definite",
            specs=(spec,),
            no_data_reason=None,
            observations_task=never_finishes,
        )
        warnings_out: list = []
        events = [
            ev
            async for ev in _stream_slot_options(
                [region],
                deadline=time.monotonic() + 0.05,
                warnings_out=warnings_out,
                late_departures_out=[],
                response=None,
                answered=(),
            )
        ]
        return events, warnings_out

    events, warnings_out = _run(_body())
    assert len(events) == 1
    (ev,) = events
    assert isinstance(ev, ObservationsEvent)
    assert ev.spec_id == "s-timeout"
    assert ev.observations.series == []
    assert ev.observations.omitted is not None
    assert ev.observations.omitted.reason == OBSERVATIONS_TIMED_OUT
    assert any(w.code == OBSERVATIONS_TIMED_OUT for w in warnings_out)


@pytest.mark.parametrize(
    ("projected_count", "late"),
    [
        pytest.param(3, True, id="projected_points_are_estimates"),
        pytest.param(0, False, id="observed_points_add_nothing"),
    ],
)
def test_done_carries_the_departures_a_streamed_read_adds(projected_count, late):
    """A streamed part ships before its observations are read. A read whose conversion
    used projected rates adds `modelled_values` to the part, and the `done` frame carries the
    part's complete departures as `late_departures: [{part_index, spec_id, departures}]`.
    A read that changes nothing adds no entry."""
    import json

    from search.engine.conjoin import assemble_region
    from search.engine.enrichment import _stream_slot_options
    from search.engine.events import DoneEvent, DoneTelemetry, ObservationsEvent, serialize_sse
    from search.models import (
        ObservationSeries,
        ObservationSeriesConversion,
        SeriesEntityRef,
        SeriesFacetRef,
        SeriesStatVarRef,
        SpecObservations,
    )
    from tests.engine._builders import conjoin_spec, shape_five_tuple

    spec = conjoin_spec(shape_five_tuple("Person", "count"), "x", sv_dcid="Test_SV")
    conversion = ObservationSeriesConversion(
        recipe_id="recipe/Test",
        table_provenance=SeriesFacetRef(ref="provenance/TestRateTables"),
        exchange_rate_table_version="2026-01-01T00:00:00Z",
        fallback_count=0,
        projected_count=projected_count,
    )
    observations = SpecObservations(
        series=[
            ObservationSeries(
                stat_var=SeriesStatVarRef(ref="Test_SV"),
                entities=[SeriesEntityRef(role="observationAbout", ref="country/KEN")],
                points=[],
                total_obs=0,
                conversion=conversion,
            )
        ],
        total_obs=0,
        truncated=False,
    )

    async def _body():
        read = asyncio.create_task(
            _return_after(
                ((ObservationsEvent(spec_id=spec.spec_id, observations=observations),), ())
            )
        )
        region = dataclasses.replace(
            _no_data_region("v"),
            status="definite",
            specs=(spec,),
            no_data_reason=None,
            observations_task=read,
        )
        response = assemble_region(
            region,
            extra_warnings=[],
            start_ms=0,
            engine_build="test",
            max_candidates=5,
        )
        late_departures: list = []
        async for _ev in _stream_slot_options(
            [region],
            deadline=time.monotonic() + 5,
            warnings_out=[],
            late_departures_out=late_departures,
            response=response,
            answered=[region],
        ):
            pass
        return late_departures

    done = DoneEvent(
        elapsed_ms=1,
        trace_id="t",
        telemetry=DoneTelemetry(),
        late_departures=_run(_body()),
    )

    frame = json.loads(serialize_sse(done).split("data: ", 1)[1])
    expected = [
        {
            "part_index": 0,
            "spec_id": spec.spec_id,
            "departures": [
                {"code": "modelled_values", "text": "Some of these values are modelled estimates."}
            ],
        }
    ]
    assert frame["late_departures"] == (expected if late else [])


# ---------------------------------------------------------------------------
# Extract and resubmit stages enforce the shared deadline before fan-out.
# ---------------------------------------------------------------------------


def test_extract_deadline_exceeded_yields_sse_terminal_error(monkeypatch, search_settings):
    """A slow extract call ends the SSE request with a classified timeout."""
    search_settings(request_timeout_s=0.02)

    async def _slow_extract(query, *, llm=None):
        await asyncio.sleep(10)
        return (
            ExtractionResult(
                extraction=Extraction(kind="resolution", variables=["x"], entities=[], dates=[]),
                qualifiers={},
            ),
            None,
        )  # pragma: no cover

    monkeypatch.setattr("search.engine.entry.raw_text.extract", _slow_extract)

    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "test extract deadline"}},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["error"], f"Expected only an error event, got {event_types}"
    assert frames[0][1]["detail"] == "Request timed out"


def test_resubmit_deadline_exceeded_yields_sse_terminal_error(monkeypatch, search_settings):
    """A resubmit call slower than both its deadlines ends the SSE request with a classified
    timeout."""
    search_settings(request_timeout_s=0.02, resubmit_full_read_timeout_s=0.02)

    def _slow_resolve_spec_resubmit(
        *,
        inp,
        graph,
        sdmx,
        coverage=None,
        options: EngineOptions | None = None,
        full_read_deadline: float | None = None,
    ):
        time.sleep(0.3)
        return _no_data_region("test var")  # pragma: no cover -- deadline fires first

    monkeypatch.setattr(
        "search.engine.entry.dataset_resubmit.resolve_spec_resubmit", _slow_resolve_spec_resubmit
    )

    app = _fake_app()
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
                        "identity": {
                            "shape": {"kind": "statvar_schema", "population_type": "test_shape"},
                            "slots": [],
                            "entities": {},
                            "requested_window": {"kind": "all"},
                        },
                    }
                ],
            }
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["error"], f"Expected only an error event, got {event_types}"
    assert frames[0][1]["detail"] == "Request timed out"


def test_extract_deadline_exceeded_returns_504_on_buffered_path(monkeypatch, search_settings):
    """A slow extract call returns HTTP 504 on the buffered path."""
    search_settings(request_timeout_s=0.02)

    async def _slow_extract(query, *, llm=None):
        await asyncio.sleep(10)
        return (
            ExtractionResult(
                extraction=Extraction(kind="resolution", variables=["x"], entities=[], dates=[]),
                qualifiers={},
            ),
            None,
        )  # pragma: no cover

    monkeypatch.setattr("search.engine.entry.raw_text.extract", _slow_extract)

    app = _fake_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "test extract deadline buffered"}},
    )
    assert resp.status_code == 504
    body = resp.json()
    assert body["detail"] == "Request timed out"
    assert body["code"] == "timeout"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"kind": "raw_text", "query": "a slow question"}, 504),
        (
            {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "placeholder",
                        "variable_text": "a slow part",
                        "status": "no_data",
                        "reason": "no_observations",
                    }
                ],
            },
            200,
        ),
    ],
    ids=["raw_text", "dataset_resubmit"],
)
def test_the_buffered_hard_bound_allows_a_resubmit_its_full_read(
    monkeypatch, search_settings, body, status
):
    """The buffered path's hard bound is `request_timeout_s`, and `resubmit_full_read_timeout_s`
    for a dataset_resubmit, whose member walk may run its full read past the first."""
    search_settings(request_timeout_s=0.05, resubmit_full_read_timeout_s=5)
    real = app_module.resolve_async

    async def _slow_resolve_async(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await real(*args, **kwargs)

    monkeypatch.setattr(app_module, "resolve_async", _slow_resolve_async)
    client = TestClient(_fake_app(), raise_server_exceptions=False)

    resp = client.post("/api/search/resolve", json={"input": body})

    assert resp.status_code == status, resp.text


def _slow_two_part_parsed_request() -> ResolveRequest:
    """Build a parsed request that runs entity pre-resolution."""
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text="v0", entities=[ParsedEntity(text="E0")]),
                ParsedPart(variable_text="v1", entities=[ParsedEntity(text="E1")]),
            ]
        )
    )


def _mock_extract_with_entities(variables: list[str], entities: list[str]):
    """Build extraction results with named entities for pre-resolution."""

    async def _inner(query: str, *, llm=None):
        return (
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution",
                    variables=variables,
                    entities=[ExtractedEntity(text=e) for e in entities],
                    dates=[],
                ),
                qualifiers={},
            ),
            None,
        )

    return _inner


def test_parsed_door_entity_pre_resolution_deadline_exceeded_yields_sse_terminal_error(
    monkeypatch, search_settings
):
    """A parsed-door pre-resolution timeout follows its `interpretation` event."""
    search_settings(request_timeout_s=0.02)

    fake_graph = FakeGraph()
    monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _slow_resolver_resolve)

    app = _fake_app(graph=fake_graph)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [
                    {"variable_text": "v0", "entities": [{"text": "E0"}]},
                    {"variable_text": "v1", "entities": [{"text": "E1"}]},
                ],
            }
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["interpretation", "error"], event_types
    assert frames[-1][1]["detail"] == "Request timed out"


def test_parsed_door_entity_pre_resolution_deadline_exceeded_returns_504_on_buffered_path(
    monkeypatch, search_settings
):
    """A parsed-door pre-resolution timeout returns HTTP 504 when buffered."""
    search_settings(request_timeout_s=0.02)

    fake_graph = FakeGraph()
    monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _slow_resolver_resolve)

    app = _fake_app(graph=fake_graph)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [
                    {"variable_text": "v0", "entities": [{"text": "E0"}]},
                    {"variable_text": "v1", "entities": [{"text": "E1"}]},
                ],
            }
        },
    )
    assert resp.status_code == 504
    body = resp.json()
    assert body["detail"] == "Request timed out"
    assert body["code"] == "timeout"


def test_parsed_door_entity_pre_resolution_deadline_leaves_no_orphaned_tasks(
    monkeypatch, search_settings
):
    """Deadline cancellation awaits every asyncio task created for pre-resolution."""
    search_settings(request_timeout_s=0.02)

    fake_graph = FakeGraph()
    monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _slow_resolver_resolve)

    request = _slow_two_part_parsed_request()

    async def _body():
        with pytest.raises(asyncio.TimeoutError):
            await _collect(
                resolve_events(request, graph=fake_graph, llm=FakeLLM(), sdmx=EMPTY_SDMX)
            )
        await asyncio.sleep(0.4)  # let the background resolve_entity thread return
        _assert_no_orphaned_tasks({asyncio.current_task()})

    _run(_body())


def test_raw_text_entity_pre_resolution_deadline_exceeded_yields_sse_terminal_error(
    monkeypatch, search_settings
):
    """A raw-text pre-resolution timeout follows its extract and interpretation events."""
    search_settings(request_timeout_s=0.02)
    monkeypatch.setattr(
        "search.engine.entry.raw_text.extract",
        _mock_extract_with_entities(["v0", "v1"], ["E0", "E1"]),
    )

    fake_graph = FakeGraph()
    monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _slow_resolver_resolve)

    app = _fake_app(graph=fake_graph)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "test raw_text entity deadline"}},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["step", "interpretation", "error"], event_types
    assert frames[-1][1]["detail"] == "Request timed out"


def test_raw_text_entity_pre_resolution_deadline_exceeded_returns_504_on_buffered_path(
    monkeypatch, search_settings
):
    """A raw-text pre-resolution timeout returns HTTP 504 when buffered."""
    search_settings(request_timeout_s=0.02)
    monkeypatch.setattr(
        "search.engine.entry.raw_text.extract",
        _mock_extract_with_entities(["v0", "v1"], ["E0", "E1"]),
    )

    fake_graph = FakeGraph()
    monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _slow_resolver_resolve)

    app = _fake_app(graph=fake_graph)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "test raw_text entity deadline buffered"}},
    )
    assert resp.status_code == 504
    body = resp.json()
    assert body["detail"] == "Request timed out"
    assert body["code"] == "timeout"


def test_raw_text_entity_pre_resolution_deadline_leaves_no_orphaned_tasks(
    monkeypatch, search_settings
):
    """Raw-text deadline cancellation awaits every asyncio pre-resolution task."""
    search_settings(request_timeout_s=0.02)
    monkeypatch.setattr(
        "search.engine.entry.raw_text.extract",
        _mock_extract_with_entities(["v0", "v1"], ["E0", "E1"]),
    )

    fake_graph = FakeGraph()
    monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _slow_resolver_resolve)

    request = make_request("test raw_text entity deadline orphaned tasks")

    async def _body():
        with pytest.raises(asyncio.TimeoutError):
            await _collect(
                resolve_events(request, graph=fake_graph, llm=FakeLLM(), sdmx=EMPTY_SDMX)
            )
        await asyncio.sleep(0.4)  # let the background resolve_entity thread return
        _assert_no_orphaned_tasks({asyncio.current_task()})

    _run(_body())


# ---------------------------------------------------------------------------
# Request-level step events (extract, ahead of the per-variable fan-out)
# ---------------------------------------------------------------------------


def test_extract_step_event_precedes_first_per_variable_event():
    """The extract step has no variable index and precedes per-variable events."""

    async def _body():
        # Canned observations exercise the real data-bearing variable path.
        events = await _collect(
            resolve_events(
                make_request("health ODA grants from USA to Ethiopia"),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
            )
        )

        extract_steps = [e for e in events if isinstance(e, StepEvent) and e.step == "extract"]
        assert len(extract_steps) == 1
        assert extract_steps[0].variable_index is None
        assert extract_steps[0].ran is True

        per_variable_indices = [
            i for i, e in enumerate(events) if getattr(e, "variable_index", None) is not None
        ]
        assert per_variable_indices, "expected at least one per-variable event"
        assert events.index(extract_steps[0]) < min(per_variable_indices)

    _run(_body())
