"""The gap fill: search answers from our candidates as dev does, and runs a second pass over the
UN System Data Commons' candidates only when the first pass ends `no_data`
(`region.pipeline._fill_gap`)."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dc_graph import FacetDefaultPolicy
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import DetectedSV, Detection
from search.engine.errors import (
    DeclaredDefaultUnsupportedError,
    DefaultScopeIndexSourcedError,
    GraphCapExceededError,
    GraphInfraError,
)
from search.engine.events import Event, EventSink, SentenceEvent, ShapeEvent, StepEvent
from search.engine.region import pipeline, place_context
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.models import DefinitePart, ParsedEntity, ParsedInput, ParsedPart, ResolveRequest
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_KEN = "country/KEN"
_OURS = "Percent_Person_Literate"
_ENROLMENT = "Percent_Person_EnrolledInSchool"
_UN = "undata/unicef/SERIES-ED_LR"
_UN_EMPTY = "undata/unesco/SERIES-LR_AG15T99"
_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_UN_GDP_PER_CAPITA = "undata/wb/SERIES-NY_GDP_PCAP_CD"
_QUERY = "literacy rate"
_ENROLMENT_QUERY = "school enrolment"
_PER_CAPITA_QUERY = "gdp per capita"


def _sv_node(dcid: str, *, meas_prop: str) -> dict:
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": dcid}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": []},
        },
    }


_NODES = {
    _KEN: {"label": "Kenya", "type": "Country"},
    _OURS: _sv_node(_OURS, meas_prop="literacyRate"),
    _ENROLMENT: _sv_node(_ENROLMENT, meas_prop="enrollmentRate"),
    _UN: _sv_node(_UN, meas_prop="literacyRate"),
    _UN_EMPTY: _sv_node(_UN_EMPTY, meas_prop="literacyRate"),
    _GDP: _sv_node(_GDP, meas_prop="amount"),
    _UN_GDP_PER_CAPITA: _sv_node(_UN_GDP_PER_CAPITA, meas_prop="amount"),
}

_FACET = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}
_UN_OBS = {f"{_UN}|{_KEN}": [_FACET]}


def _leg(*scored: tuple[str, float]) -> dict:
    return {"svs": [dcid for dcid, _ in scored], "cosine_scores": [score for _, score in scored]}


def _graph_kwargs(
    *,
    un: tuple[tuple[str, float], ...] | None = ((_UN, 0.85),),
    ours_have_data: bool = False,
) -> dict:
    """Our resolve holds `_OURS` and the UN's `un` (`None` fails it); only `_UN` has data at
    Kenya unless `ours_have_data`."""
    obs = {**_UN_OBS, f"{_OURS}|{_KEN}": [_FACET]} if ours_have_data else _UN_OBS
    return {
        "nodes": _NODES,
        "obs": obs,
        "detect": {_QUERY: _leg((_OURS, 0.9))},
        "un_detect": {_QUERY: None if un is None else _leg(*un)},
        "resolve": {"Kenya": _KEN},
    }


def _graph(**kwargs) -> FakeGraph:
    return FakeGraph(**_graph_kwargs(**kwargs))


class _FailingUnReads(FakeGraph):
    """Raises `error` from every facet read that names a UN variable; the default is a UN
    deployment outage."""

    def __init__(self, *, error: Exception | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.error = error or GraphInfraError("UN deployment unavailable")

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, max_bytes=None):
        if any(sv.startswith("undata/") for sv in stat_vars):
            raise self.error
        return super().observation_facets_batch(
            stat_vars, entities, needs_dates, max_bytes=max_bytes
        )


class _ElectFirstLLM:
    """Elects the first shape on the menu with no bindings, and records each menu it saw.
    From the second call on, each call first awaits `second_bind`, when set."""

    def __init__(self, *, second_bind: asyncio.Event | None = None) -> None:
        self.prompts: list[str] = []
        self.second_bind = second_bind

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        if self.second_bind is not None and len(self.prompts) > 1:
            await self.second_bind.wait()
        usage = {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[], ask=None), usage


def _resolve(
    graph: FakeGraph,
    *,
    query: str = _QUERY,
    entities: list[str] | None = None,
    llm: _ElectFirstLLM | None = None,
    deadline: float | None = None,
    emit: EventSink | None = None,
) -> RegionResult:
    return asyncio.run(
        _resolving(graph, query=query, entities=entities, llm=llm, deadline=deadline, emit=emit)
    )


async def _resolving(
    graph: FakeGraph,
    *,
    query: str = _QUERY,
    entities: list[str] | None = None,
    llm: _ElectFirstLLM | None = None,
    deadline: float | None = None,
    emit: EventSink | None = None,
) -> RegionResult:
    return await resolve_variable(
        query,
        entities=["Kenya"] if entities is None else entities,
        date_request=None,
        detect_query=query,
        role_query=query,
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        llm=llm or _ElectFirstLLM(),
        base_steps=[],
        base_timing={},
        emit=emit,
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(deadline=deadline),
    )


def _elected(result: RegionResult) -> list[str]:
    return [sv.ref for spec in result.specs for sv in spec.resolved.stat_vars.items]


def _codes(result: RegionResult) -> list[str]:
    return [w.code for w in result.warnings]


def _gap_fill_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap_fill ")]


def _menu_named(llm: _ElectFirstLLM, dcid: str) -> bool:
    return any(dcid in prompt for prompt in llm.prompts)


def test_a_first_pass_answer_stands_without_a_second_pass(caplog) -> None:
    llm = _ElectFirstLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(ours_have_data=True), llm=llm)

    assert result.status == "definite"
    assert _elected(result) == [_OURS]
    assert not _menu_named(llm, _UN)
    assert "un_system_unavailable" not in _codes(result)
    assert _gap_fill_lines(caplog) == [
        "gap_fill part=0 pass1=definite/None pass1_measure_refused=0 un_candidates=1 "
        "pass2=skipped:answered pass2_status=- pass2_ms=0 pass2_below_floor=0 "
        "pass2_measure_refused=0 pass2_score=-"
    ]


def test_a_streamed_filled_answer_ends_on_the_second_passs_sentence() -> None:
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    result = _resolve(_graph(), emit=collect)

    assert result.status == "definite"
    (spec,) = result.specs
    sentences = [event for event in events if isinstance(event, SentenceEvent)]
    assert sentences, "the filled answer streamed no sentence"
    assert sentences[-1].spec_id == spec.spec_id
    shapes = [event for event in events if isinstance(event, ShapeEvent)]
    assert [event.label for event in shapes] == [_OURS]
    steps = [event.step for event in events if isinstance(event, StepEvent)]
    assert steps.count("bind") == 1


def test_a_second_pass_that_does_not_answer_leaves_the_stream_unchanged() -> None:
    def stream(graph: FakeGraph) -> tuple[RegionResult, list[Event]]:
        events: list[Event] = []

        async def collect(event: Event) -> None:
            events.append(event)

        return _resolve(graph, emit=collect), events

    filled, filled_events = stream(_graph(un=((_UN_EMPTY, 0.85),)))
    first_only, first_only_events = stream(_graph(un=()))

    assert filled.status == first_only.status == "no_data"
    assert [e for e in filled_events if not isinstance(e, StepEvent)] == [
        e for e in first_only_events if not isinstance(e, StepEvent)
    ]


def test_a_first_pass_without_observations_is_filled_from_the_un_shape(caplog) -> None:
    llm = _ElectFirstLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(), llm=llm)

    assert result.status == "definite"
    assert _elected(result) == [_UN]
    (spec,) = result.specs
    assert [origin.id for origin in spec.resolved.shape.data_commons] == ["un_system"]
    first_menu, second_menu = llm.prompts
    assert _OURS in first_menu and _UN not in first_menu
    assert _UN in second_menu and _OURS not in second_menu
    (line,) = _gap_fill_lines(caplog)
    assert "pass1=no_data/no_observations pass1_measure_refused=0 un_candidates=1 pass2=ran" in line
    assert "pass2_status=definite/None" in line
    assert line.endswith("pass2_score=0.8500")


def test_the_filled_answer_carries_both_passes_usage_and_the_second_passs_trace() -> None:
    result = _resolve(_graph())

    assert result.llm_usage == {"input_tokens": 2, "output_tokens": 2, "cached_tokens": 0}
    steps = [step.step for step in result.specs[0].resolved.trace.pipeline_trace]
    assert steps.count("recall") == steps.count("shape") == steps.count("bind") == 1


def test_a_failed_second_pass_keeps_the_first_with_a_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_FailingUnReads(**_graph_kwargs()))

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert "un_system_unavailable" in _codes(result)
    (line,) = _gap_fill_lines(caplog)
    assert "pass2=ran pass2_status=error:GraphInfraError" in line


def test_a_failed_second_pass_logs_the_refusals_its_prune_made(monkeypatch, caplog) -> None:
    """A second pass whose prune refused shapes before a later phase failed upstream still
    reports its counts, and the first pass's, on the `gap_fill` line."""

    async def failing_after_the_prune(state) -> RegionResult:
        state.n_below_score_floor = 1
        state.n_measure_class_refused = 2
        raise GraphInfraError("UN deployment unavailable")

    async def settled() -> None:
        return None

    monkeypatch.setattr(pipeline, "_run_phases", failing_after_the_prune)
    first = RegionResult(
        variable_text=_QUERY,
        status="no_data",
        specs=(),
        no_data_reason="no_observations",
        warnings=(),
        timing_by_step={},
    )
    rcl = SimpleNamespace(
        settle_un=settled,
        un_unavailable=False,
        un_candidates=[object()],
        un_leg=SimpleNamespace(done=lambda: True),
    )
    state = SimpleNamespace(rcl=None, n_below_score_floor=0, n_measure_class_refused=0)
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        asyncio.run(
            pipeline._fill_gap(
                first,
                rcl=cast(Any, rcl),
                deadline=None,
                variable_index=0,
                emit=None,
                first_sentences=[],
                first_measure_class_refused=3,
                second_state=lambda emit, bound: cast(Any, state),
            )
        )

    (line,) = _gap_fill_lines(caplog)
    assert "pass1_measure_refused=3" in line
    assert "pass2_status=error:GraphInfraError" in line
    assert "pass2_below_floor=1 pass2_measure_refused=2" in line


def test_a_second_pass_over_an_upstream_cap_keeps_the_first_with_a_warning(caplog) -> None:
    cap = GraphCapExceededError("over the byte cap", cap_kind="bytes", cap=1, observed=2)
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_FailingUnReads(error=cap, **_graph_kwargs()))

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert [sv.dcid for sv in result.elected_stat_vars or []] == [_OURS]
    assert "un_system_unavailable" in _codes(result)
    (line,) = _gap_fill_lines(caplog)
    assert "pass2=ran pass2_status=error:GraphCapExceededError" in line


def test_a_defect_in_the_second_pass_keeps_the_first_unwarned_and_logs_it(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_FailingUnReads(error=RuntimeError("a bug"), **_graph_kwargs()))

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert [sv.dcid for sv in result.elected_stat_vars or []] == [_OURS]
    assert "un_system_unavailable" not in _codes(result)
    (error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error.exc_info is not None and isinstance(error.exc_info[1], RuntimeError)


def test_a_declared_default_in_the_second_pass_fails_the_request() -> None:
    """A failed second pass keeps the first pass, except on a declared-default refusal: the
    UN variable is found with data, so a `no_data` first pass would misstate the answer."""
    policy = FacetDefaultPolicy(
        resolved_ref="Percent",
        policy_node="dc/policy/literacy-unit",
        policy_version="1",
        selection_reason="declared",
    )
    graph = FakeGraph(**_graph_kwargs(), facet_policies={(f"dcid:{_UN}", "unit"): policy})

    with pytest.raises(DeclaredDefaultUnsupportedError, match=_UN):
        _resolve(graph)


def test_an_index_sourced_default_scope_in_the_second_pass_fails_the_request(
    monkeypatch,
) -> None:
    """A failed second pass keeps the first pass, except when a subject-only shape whose
    population the code index sources would take a default scope: the request refuses
    rather than answer from a scope its presence read never covered."""
    real = place_context.shape_member_facts
    monkeypatch.setattr(
        place_context,
        "shape_member_facts",
        lambda shape, bound_values: (
            ({}, True) if _UN in (shape.sv_arc_facts or {}) else real(shape, bound_values)
        ),
    )

    with pytest.raises(DefaultScopeIndexSourcedError, match="index-sourced"):
        _resolve(_graph(), entities=[])


def test_a_second_pass_that_outruns_its_bound_keeps_the_first_before_the_deadline(
    caplog, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "_SECOND_PASS_RESERVE_S", 1.0)
    llm = _ElectFirstLLM(second_bind=asyncio.Event())
    deadline = time.monotonic() + 2.0
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = asyncio.run(
            asyncio.wait_for(_resolving(_graph(), llm=llm, deadline=deadline), timeout=5)
        )

    assert time.monotonic() < deadline - 0.5, "the second pass ran into the reserve"
    assert _menu_named(llm, _UN), "the second pass never reached bind"
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert [sv.dcid for sv in result.elected_stat_vars or []] == [_OURS]
    assert result.flagged_match
    assert "un_system_unavailable" in _codes(result)
    (line,) = _gap_fill_lines(caplog)
    assert "pass2=ran pass2_status=error:TimeoutError" in line


def test_cancelling_the_request_during_the_second_pass_propagates() -> None:
    llm = _ElectFirstLLM(second_bind=asyncio.Event())

    async def cancel_in_second_bind() -> None:
        task = asyncio.create_task(_resolving(_graph(), llm=llm))
        while len(llm.prompts) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(cancel_in_second_bind(), timeout=10))


def test_a_second_pass_that_also_ends_no_data_keeps_the_first(caplog) -> None:
    llm = _ElectFirstLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(un=((_UN_EMPTY, 0.85),)), llm=llm)

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert [sv.dcid for sv in result.elected_stat_vars or []] == [_OURS]
    assert _menu_named(llm, _UN_EMPTY), "the second pass never reached bind"
    (line,) = _gap_fill_lines(caplog)
    assert "pass1=no_data/no_observations" in line
    assert "pass2=ran pass2_status=no_data/" in line


def test_an_unresolved_entity_runs_no_second_pass(caplog) -> None:
    llm = _ElectFirstLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(), entities=["Atlantis"], llm=llm)

    assert result.status == "no_data"
    assert result.no_data_reason == "entity_not_resolved"
    assert not _menu_named(llm, _UN)
    (line,) = _gap_fill_lines(caplog)
    assert "pass2=skipped:entity" in line


def test_a_failed_un_resolve_warns_only_on_a_no_data_answer() -> None:
    no_data = _resolve(_graph(un=None))
    answered = _resolve(_graph(un=None, ours_have_data=True))

    assert no_data.status == "no_data"
    assert no_data.no_data_reason == "no_observations"
    assert "un_system_unavailable" in _codes(no_data)
    assert answered.status == "definite"
    assert "un_system_unavailable" not in _codes(answered)


def test_a_deadline_spent_before_the_second_pass_keeps_the_first_with_a_warning(caplog) -> None:
    llm = _ElectFirstLLM()
    deadline = time.monotonic() + pipeline._SECOND_PASS_RESERVE_S - 0.5
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(), llm=llm, deadline=deadline)

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert "un_system_unavailable" in _codes(result)
    assert not _menu_named(llm, _UN)
    (line,) = _gap_fill_lines(caplog)
    assert "pass2=skipped:deadline" in line


def _first_pass(result: RegionResult) -> tuple:
    """What a first-pass answer says, without step latencies and model usage."""
    return (
        result.status,
        result.no_data_reason,
        result.gate,
        [spec.spec_id for spec in result.specs],
        result.elected_stat_vars,
        [spec.spec_id for spec in result.flagged_match or ()],
        result.warnings,
    )


@pytest.mark.parametrize("ours_have_data", [True, False], ids=["answered", "no_data"])
def test_the_first_pass_is_the_same_with_the_un_origin_on_and_off(
    bind_catalogue, ours_have_data: bool
) -> None:
    """The UN's candidate has no data, so a first pass that ends no_data stands either way."""

    def run(*, un_system_enabled: bool) -> tuple[RegionResult, _ElectFirstLLM]:
        bind_catalogue(FakeCatalogue(un_system_enabled=un_system_enabled))
        llm = _ElectFirstLLM()
        graph = _graph(un=((_UN_EMPTY, 0.85),), ours_have_data=ours_have_data)
        return _resolve(graph, llm=llm), llm

    on, on_llm = run(un_system_enabled=True)
    off, off_llm = run(un_system_enabled=False)

    assert _first_pass(on) == _first_pass(off)
    assert on.status == ("definite" if ours_have_data else "no_data")
    assert off_llm.prompts == on_llm.prompts[:1]
    assert len(on_llm.prompts) == (1 if ours_have_data else 2)


def test_a_two_part_request_fills_only_the_part_without_data() -> None:
    kwargs = _graph_kwargs()
    kwargs["obs"] = {**kwargs["obs"], f"{_ENROLMENT}|{_KEN}": [_FACET]}
    kwargs["detect"][_ENROLMENT_QUERY] = _leg((_ENROLMENT, 0.9))
    kwargs["un_detect"][_ENROLMENT_QUERY] = _leg((_UN, 0.8))
    kenya = [ParsedEntity(text="Kenya")]
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text=_QUERY, entities=kenya),
                ParsedPart(variable_text=_ENROLMENT_QUERY, entities=kenya),
            ]
        )
    )

    response = offline_resolve(request, graph=FakeGraph(**kwargs), llm=_ElectFirstLLM())

    parts = response.parts
    assert all(isinstance(part, DefinitePart) for part in parts)
    elected = [[sv.ref for sv in part.spec.resolved.stat_vars.items] for part in parts]
    assert elected == [[_UN], [_ENROLMENT]]


def _per_capita_graph(*, un: tuple[tuple[str, float], ...]) -> FakeGraph:
    """Our GDP variable and the UN's per-capita series have data at Kenya, and neither has a
    denominator."""
    return FakeGraph(
        nodes=_NODES,
        obs={f"{_GDP}|{_KEN}": [_FACET], f"{_UN_GDP_PER_CAPITA}|{_KEN}": [_FACET]},
        detect={_PER_CAPITA_QUERY: _leg((_GDP, 0.9))},
        un_detect={_PER_CAPITA_QUERY: _leg(*un)},
        resolve={"Kenya": _KEN},
    )


def test_a_per_capita_query_refuses_a_un_shape_without_a_denominator(caplog) -> None:
    llm = _ElectFirstLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(
            _per_capita_graph(un=((_UN_GDP_PER_CAPITA, 0.85),)), query=_PER_CAPITA_QUERY, llm=llm
        )

    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    assert _menu_named(llm, _UN_GDP_PER_CAPITA), "the second pass never reached bind"
    (line,) = _gap_fill_lines(caplog)
    assert "pass1=no_data/denominator_not_available" in line
    assert "pass2_status=no_data/denominator_not_available" in line


def test_a_per_capita_query_on_a_base_shape_without_a_denominator_is_refused() -> None:
    result = _resolve(_per_capita_graph(un=()), query=_PER_CAPITA_QUERY)

    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"


# -- the UN resolve runs beside the first pass --


class _Leg(Future):
    """A running UN leg that the test settles; counts the waiters it gains."""

    def __init__(self) -> None:
        super().__init__()
        self.set_running_or_notify_cancel()
        self.waiters = 0

    def add_done_callback(self, fn: Callable[[Future], object]) -> None:
        self.waiters += 1
        super().add_done_callback(fn)

    def settle_after(self, seconds: float, un: tuple[DetectedSV, ...] | None) -> None:
        timer = threading.Timer(seconds, self.set_result, args=(un,))
        timer.daemon = True
        timer.start()


class _PendingUnGraph(FakeGraph):
    """Hands out `leg` as every detection's UN leg, as the live client does while the UN
    resolve is still running."""

    def __init__(self, *, leg: _Leg, **kwargs) -> None:
        super().__init__(**kwargs)
        self.leg = leg

    def detect_svs(self, query: str) -> Detection:
        return dataclasses.replace(super().detect_svs(query), un=self.leg)


def _un_leg(dcid: str) -> tuple[DetectedSV, ...]:
    return (DetectedSV(dcid=dcid, score=0.85, type_of="StatisticalVariable", sentence=dcid),)


def test_a_first_pass_answer_does_not_wait_for_a_running_un_resolve(caplog) -> None:
    leg = _Leg()
    graph = _PendingUnGraph(leg=leg, **_graph_kwargs(ours_have_data=True))

    async def answer() -> tuple[RegionResult, set[asyncio.Task]]:
        async with asyncio.timeout(5):
            result = await _resolving(graph)
        return result, asyncio.all_tasks() - {asyncio.current_task()}

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result, leftover = asyncio.run(answer())

    assert result.status == "definite"
    assert "un_system_unavailable" not in _codes(result)
    assert not leg.done() and leg.waiters == 0
    assert leftover == set()
    assert _gap_fill_lines(caplog) == [
        "gap_fill part=0 pass1=definite/None pass1_measure_refused=0 un_candidates=pending "
        "pass2=skipped:answered pass2_status=- pass2_ms=0 pass2_below_floor=0 "
        "pass2_measure_refused=0 pass2_score=-"
    ]


def test_a_first_pass_without_data_waits_for_the_un_resolve_and_is_filled(caplog) -> None:
    leg = _Leg()
    leg.settle_after(0.2, _un_leg(_UN))
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_PendingUnGraph(leg=leg, **_graph_kwargs()))

    assert result.status == "definite"
    assert _elected(result) == [_UN]
    assert leg.waiters == 1
    (line,) = _gap_fill_lines(caplog)
    assert "pass1=no_data/no_observations pass1_measure_refused=0 un_candidates=1 pass2=ran" in line


@pytest.mark.parametrize("ours_have_data", [True, False], ids=["answered", "no_data"])
def test_a_un_resolve_that_times_out_warns_only_a_part_without_data(ours_have_data: bool) -> None:
    leg = _Leg()
    leg.settle_after(0.2, None)
    result = _resolve(_PendingUnGraph(leg=leg, **_graph_kwargs(ours_have_data=ours_have_data)))

    if ours_have_data:
        assert result.status == "definite"
        assert "un_system_unavailable" not in _codes(result)
        assert leg.waiters == 0
    else:
        assert result.status == "no_data"
        assert "un_system_unavailable" in _codes(result)
        assert leg.waiters == 1


def test_a_un_resolve_still_running_at_the_second_pass_bound_is_left_behind(
    caplog, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "_SECOND_PASS_RESERVE_S", 1.0)
    leg = _Leg()
    deadline = time.monotonic() + 1.5
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = asyncio.run(
            asyncio.wait_for(
                _resolving(_PendingUnGraph(leg=leg, **_graph_kwargs()), deadline=deadline),
                timeout=5,
            )
        )

    assert time.monotonic() < deadline, "the wait for the UN resolve ran past its bound"
    assert result.status == "no_data"
    assert "un_system_unavailable" in _codes(result)
    assert not leg.done() and not leg.cancelled()
    (line,) = _gap_fill_lines(caplog)
    assert "un_candidates=pending pass2=skipped:deadline" in line


# -- a first pass whose only data-bearing reading is a nominated alternate --

_ETH = "country/ETH"
_BIRTH_RATE_QUERY = "birth rate"
_BIRTHS_PER_CAPITA = "Count_BirthEvent_AsAFractionOfCount_Person"
_CRUDE_BIRTH_RATE = "Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person"
_FERTILITY_RATE = "FertilityRate_Person_Female"
_UN_TOTAL_FERTILITY_RATE = "undata/unfpa/FERT_RATE_TOTAL"


def _arc(*dcids: str) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids]}


def _constrained_sv_node(
    dcid: str,
    *,
    population_type: str,
    meas_prop: str,
    constraints: dict[str, str],
    denominator: str | None = None,
) -> dict:
    node = _sv_node(dcid, meas_prop=meas_prop)
    node["arcs"]["populationType"] = _arc(population_type)
    node["arcs"]["constraintProperties"] = _arc(*constraints)
    node["arcs"].update({prop: _arc(value) for prop, value in constraints.items()})
    if denominator is not None:
        node["arcs"]["measurementDenominator"] = _arc(denominator)
    return node


def _birth_rate_graph(*, un_score: float, live_births_have_data: bool = True) -> FakeGraph:
    """The reads behind "birth rate Ethiopia" on the dev graph. The births-per-capita family's
    total has no data at Ethiopia and dominates the crude birth rate, which has unless
    `live_births_have_data` is false; the fertility rate the model nominates as an alternate
    has data, and so does the UN's total fertility rate, recalled at `un_score`."""
    births = {"population_type": "BirthEvent", "meas_prop": "count", "denominator": "Count_Person"}
    with_data = (_FERTILITY_RATE, _UN_TOTAL_FERTILITY_RATE) + (
        (_CRUDE_BIRTH_RATE,) if live_births_have_data else ()
    )
    return FakeGraph(
        nodes={
            _ETH: {"label": "Ethiopia", "type": "Country"},
            _BIRTHS_PER_CAPITA: _constrained_sv_node(_BIRTHS_PER_CAPITA, constraints={}, **births),
            _CRUDE_BIRTH_RATE: _constrained_sv_node(
                _CRUDE_BIRTH_RATE, constraints={"medicalStatus": "LiveBirth"}, **births
            ),
            _FERTILITY_RATE: _constrained_sv_node(
                _FERTILITY_RATE,
                population_type="Person",
                meas_prop="fertilityRate",
                constraints={"gender": "Female"},
            ),
            _UN_TOTAL_FERTILITY_RATE: _constrained_sv_node(
                _UN_TOTAL_FERTILITY_RATE,
                population_type="undata/unfpa/SERIES-FERT_RATE_TOTAL",
                meas_prop="value",
                constraints={},
            ),
        },
        obs={f"{sv}|{_ETH}": [_FACET] for sv in with_data},
        detect={
            _BIRTH_RATE_QUERY: _leg(
                (_BIRTHS_PER_CAPITA, 1.0), (_CRUDE_BIRTH_RATE, 1.0), (_FERTILITY_RATE, 0.865)
            )
        },
        un_detect={_BIRTH_RATE_QUERY: _leg((_UN_TOTAL_FERTILITY_RATE, un_score))},
        resolve={"Ethiopia": _ETH},
    )


class _RecordedBirthRateLLM(_ElectFirstLLM):
    """Answers the first bind as the live model did for "birth rate Ethiopia": the births shape,
    medicalStatus unbound, and the fertility rate as a plausible alternate. Later binds elect
    the first shape."""

    async def generate_structured(self, *, prompt, system, schema):
        output, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if len(self.prompts) > 1:
            return output, usage
        recorded = _UniversalBindOutput.model_validate_json(
            '{"chosen_shape_index":0,"bindings":[{"axis":"how","property_dcid":"medicalStatus",'
            '"kind":"unbound","value_dcids":[],"evidence":"mentioned",'
            '"requested_all_phrase":null}],"also_plausible":[1],"ask":null,"unplaced":[]}'
        )
        return recorded, usage


def _resolve_birth_rate(*, un_score: float, llm: _ElectFirstLLM) -> RegionResult:
    """Resolves where no births-per-capita member has data, so the first pass ends no_data
    holding the fertility rate as its alternate."""
    return _resolve(
        _birth_rate_graph(un_score=un_score, live_births_have_data=False),
        query=_BIRTH_RATE_QUERY,
        entities=["Ethiopia"],
        llm=llm,
    )


def test_a_weak_second_pass_answer_leaves_the_first_passs_alternate_standing(caplog) -> None:
    llm = _RecordedBirthRateLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve_birth_rate(un_score=0.73, llm=llm)

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.gate == "only_alternate_has_data"
    assert result.alternate_match is not None
    assert [sv.ref for sv in result.alternate_match.resolved.stat_vars.items] == [_FERTILITY_RATE]
    assert [sv.dcid for sv in result.elected_stat_vars or []] == [_CRUDE_BIRTH_RATE]
    assert _menu_named(llm, _UN_TOTAL_FERTILITY_RATE), "the second pass never reached bind"
    assert "un_system_unavailable" not in _codes(result)
    (line,) = _gap_fill_lines(caplog)
    assert "pass1=no_data/no_observations pass1_measure_refused=0 un_candidates=1 pass2=ran" in line
    assert "pass2_status=definite/None" in line
    assert line.endswith("pass2_score=0.7300")


def test_a_second_pass_answer_above_the_weak_band_displaces_the_first_passs_alternate() -> None:
    result = _resolve_birth_rate(un_score=0.85, llm=_RecordedBirthRateLLM())

    assert result.status == "definite"
    assert _elected(result) == [_UN_TOTAL_FERTILITY_RATE]
