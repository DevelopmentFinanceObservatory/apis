"""A fresh answer's scope is decided once, before its preview launches
(`phase_answer._decide_answer_scope`, `reading.decide_read_scope`). An answer the binders
settle as a composite then reads no preview of the one table it resolved, streamed or
buffered. Any other answer reads its preview of that table."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import pytest

from code_index.composites import CompositeNotServable
from materialise.rate_tables import RateTableCache
from search.engine import spec_build
from search.engine.concurrency import scoped_concurrency
from search.engine.errors import GraphInfraError
from search.engine.events import DoneEvent, ResponseEvent, SlotOptionsEvent
from search.engine.region import (
    candidate_collapse,
    candidate_options,
    phase_answer,
    reading,
    scope_decision,
)
from search.engine.region.candidates import resolve_candidates
from search.engine.region.composite_answer import CompositeRelaunch
from search.engine.region.observations.dispatch import PreviewRead, PreviewReader
from search.engine.region.reading import ReadScope
from search.engine.region.result import RegionResult
from search.engine.region.scope_decision import ScopeDecision
from search.engine.region.state import ResolveState
from search.models import (
    DefinitePart,
    NoDataPart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolvedQualifier,
    ResolveOptions,
    ResolveRequest,
    Spec,
)
from tests.engine._harness import offline_events, offline_resolve
from tests.engine.test_composite_answer import (
    _BOTH,
    _QUESTION,
    _answered,
    _clients,
    _components,
    _dates,
    _observed,
    _stream,
)
from tests.engine.test_dac2a_candidate_readings import (
    india_clients,
    india_request,
    patch_india_coverage,
    refuse_the_recipient_reading_s_mint,
)
from tests.engine.test_dac2a_destination_populations import _BILATERAL as _DAC2A_BILATERAL
from tests.engine.test_observations_rate_cache_dispatch import _resolve_candidates_in_pass

# The provider rule settles this total as CRS's bilateral flows, a single-table answer.
_BILATERAL = "total bilateral health aid from the United States to Kenya, 2015 to 2020"


def _fresh(question: str) -> ResolveRequest:
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="health aid",
                    entities=[
                        ParsedEntity(text="United States", role_text="from"),
                        ParsedEntity(text="Kenya", role_text="to"),
                    ],
                    time_text="2015-2020",
                )
            ],
            user_message=question,
        ),
        response_options=ResolveOptions(include_observations=True),
    )


def _buffered(question: str) -> Spec:
    response = offline_resolve(_fresh(question), **_clients())
    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    return part.spec


def _decisions(monkeypatch: pytest.MonkeyPatch) -> list[ScopeDecision]:
    """Every scope decided from here on, by the answer phase (`reading.decide_read_scope`)
    or by `_with_scope_choices` (`scope_decision.decide_scope`)."""
    decided: list[ScopeDecision] = []
    read_scope, decide = phase_answer.decide_read_scope, scope_decision.decide_scope

    def record_read(*args: Any, **kwargs: Any) -> ReadScope:
        scope = read_scope(*args, **kwargs)
        assert scope.decision is not None
        decided.append(scope.decision)
        return scope

    def record(*args: Any, **kwargs: Any) -> ScopeDecision:
        decision = decide(*args, **kwargs)
        decided.append(decision)
        return decision

    monkeypatch.setattr(phase_answer, "decide_read_scope", record_read)
    monkeypatch.setattr(scope_decision, "decide_scope", record)
    return decided


def _identities(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The `spec_id` of every answer identity minted from here on (`answer_identity`), by
    the answer phase or by `build_spec`."""
    minted: list[str] = []
    real = spec_build.answer_identity

    def record(*args: Any, **kwargs: Any) -> spec_build.AnswerIdentity:
        answer = real(*args, **kwargs)
        minted.append(answer.spec_id)
        return answer

    monkeypatch.setattr(phase_answer, "answer_identity", record)
    monkeypatch.setattr(spec_build, "answer_identity", record)
    return minted


def _table_reads(monkeypatch: pytest.MonkeyPatch, *, refuse: bool) -> list[str]:
    """How the answer phase reads the resolved table's preview from here on: `observations`
    for the buffered or eager read, `relaunch` for the streamed one. With `refuse`, any such
    read fails the request."""
    reads: list[str] = []
    real = phase_answer._preview_reader

    class Reader:
        def __init__(self, reader: PreviewReader) -> None:
            self.reader = reader

        def observations(self) -> Any:
            reads.append("observations")
            if refuse:
                raise AssertionError("the composite read its table identity's preview")
            return self.reader.observations()

        def relaunch(self, labels: Any) -> Any:
            reads.append("relaunch")
            if refuse:
                raise AssertionError("the composite launched its table identity's preview")
            return self.reader.relaunch(labels)

    def reader(state: ResolveState, read: PreviewRead) -> Reader:
        return Reader(real(state, read))

    monkeypatch.setattr(phase_answer, "_preview_reader", reader)
    return reads


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_a_composite_answer_reads_no_preview_of_its_table_identity(monkeypatch, stream):
    """The scope is decided once, before the preview launches, and the composite previews
    its components alone."""
    first, _ = _components(monkeypatch)
    decided = _decisions(monkeypatch)
    _table_reads(monkeypatch, refuse=True)

    if stream:
        events = _stream(_fresh(_QUESTION))
        spec = _answered(events)
        observations = _observed(events).observations
    else:
        spec = _buffered(_QUESTION)
        observations = spec.resolved.observations

    assert spec.spec_id == first.spec_id
    assert _dates(observations) == _BOTH
    assert [(d.spec_id, d.composite) for d in decided] == [(decided[0].spec_id, True)]
    assert decided[0].scope is not None
    assert decided[0].scope.compiled.spec_id == first.spec_id


@pytest.mark.parametrize(
    ("stream", "expected"),
    [(False, ["observations"]), (True, ["observations", "relaunch"])],
    ids=["buffered", "streamed"],
)
def test_a_single_table_answer_reads_its_preview_as_before(monkeypatch, stream, expected):
    """A scope that settles no composite leaves the answer's preview reads as they were,
    and is decided once, for the answer it states, over the one identity the Spec is minted
    from."""
    _components(monkeypatch)
    decided = _decisions(monkeypatch)
    minted = _identities(monkeypatch)
    reads = _table_reads(monkeypatch, refuse=False)

    spec = _answered(_stream(_fresh(_BILATERAL))) if stream else _buffered(_BILATERAL)

    assert reads == expected
    assert minted == [spec.spec_id]
    assert [(d.spec_id, d.composite) for d in decided] == [(spec.spec_id, False)]
    assert spec.resolved.scope_choices is not None


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize(
    ("question", "early_composite", "table_reads"),
    [(_BILATERAL, True, 1), (_QUESTION, False, None)],
    ids=["early-composite", "early-single-table"],
)
def test_the_preview_follows_the_scope_decided_for_the_finished_answer(
    monkeypatch, stream, question, early_composite, table_reads
):
    """An early decision made for another `spec_id`, calling the answer the opposite of what
    it is, is decided again from the finished answer, and the preview follows that one: a
    single-table answer reads its table once, a composite its components."""
    _components(monkeypatch)
    reads = _table_reads(monkeypatch, refuse=False)
    real = phase_answer._decide_answer_scope

    def mismatched(*args: Any) -> ReadScope:
        scope = real(*args)
        assert scope.decision.composite is not early_composite
        decision = dataclasses.replace(scope.decision, spec_id="other", composite=early_composite)
        return dataclasses.replace(scope, decision=decision)

    monkeypatch.setattr(phase_answer, "_decide_answer_scope", mismatched)

    if stream:
        events = _stream(_fresh(question))
        spec = _answered(events)
        observations = _observed(events).observations
    else:
        spec = _buffered(question)
        observations = spec.resolved.observations

    assert observations is not None
    assert spec.resolved.scope_choices is not None
    if table_reads is None:
        assert _dates(observations) == _BOTH
    else:
        assert len(reads) == table_reads


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_a_raising_decision_for_the_finished_answer_keeps_it_as_a_single_table(
    monkeypatch, caplog, stream
):
    """An early decision made for another `spec_id` is decided again from the finished
    answer, guarded as the early one is: a decision that raises there leaves the answer as
    one table that states no scope, and the error is logged."""
    _components(monkeypatch)
    real = phase_answer._decide_answer_scope

    def mismatched(*args: Any) -> ReadScope:
        scope = real(*args)
        return dataclasses.replace(
            scope, decision=dataclasses.replace(scope.decision, spec_id="other")
        )

    def raises(spec: Spec, **kwargs: Any) -> ScopeDecision:
        raise _not_servable()

    monkeypatch.setattr(phase_answer, "_decide_answer_scope", mismatched)
    monkeypatch.setattr(scope_decision, "decide_spec_scope", raises)

    with caplog.at_level(logging.WARNING, logger=scope_decision.logger.name):
        spec = _answered(_stream(_fresh(_BILATERAL))) if stream else _buffered(_BILATERAL)

    assert spec.resolved.scope_choices is None
    _assert_decision_error_logged(caplog)


def test_a_raising_decision_for_an_undecided_answer_keeps_it_as_a_single_table(monkeypatch, caplog):
    """An answer judged with no decision (`phase_answer.finished_scope`) is decided there,
    guarded as every decision is: a decision that raises leaves the answer as one table that
    states no scope, and the error is logged."""
    _components(monkeypatch)
    real = phase_answer.finished_scope

    async def undecided(state: ResolveState, spec: Spec, **kwargs: Any) -> ScopeDecision | None:
        return await real(state, spec, **{**kwargs, "decision": None})

    def raises(*args: Any, **kwargs: Any) -> ScopeDecision:
        raise _not_servable()

    monkeypatch.setattr(phase_answer, "finished_scope", undecided)
    # The answer phase decides through `reading.build_part_scope`, so only the late decision
    # raises.
    monkeypatch.setattr(scope_decision, "decide_scope", raises)

    with caplog.at_level(logging.WARNING, logger=scope_decision.logger.name):
        spec = _buffered(_BILATERAL)

    assert spec.resolved.scope_choices is None
    _assert_decision_error_logged(caplog)


def _not_servable() -> Exception:
    return CompositeNotServable("no component", ref="dfo-composite", pointer="/facets")


def _raising_decisions(
    monkeypatch: pytest.MonkeyPatch, error: Callable[[], Exception] = _not_servable
) -> list[str]:
    """Make every scope decision from here on raise `error()`, `CompositeNotServable` by
    default, and list the `spec_id` of each one asked for. Those are the answer phase's and
    the candidates path's (`reading.build_part_scope`, called by `reading.decide_read_scope`)
    and a late one (`scope_decision.decide_scope`). The build made by a reading election
    (`reading.elect_reading`) passes through to the real one."""
    asked: list[str] = []
    build = reading.build_part_scope

    def raises(spec_id: str, *args: Any, **kwargs: Any) -> ScopeDecision:
        asked.append(spec_id)
        raise error()

    def raising_build(identity: Any, choice: Any, *, spec_id: str, facts: Any) -> ScopeDecision:
        if spec_id == reading._ELECTION:
            return build(identity, choice, spec_id=spec_id, facts=facts)
        return raises(spec_id)

    monkeypatch.setattr(reading, "build_part_scope", raising_build)
    monkeypatch.setattr(scope_decision, "decide_scope", raises)
    return asked


def _assert_decision_error_logged(
    caplog: pytest.LogCaptureFixture, error: type[Exception] = CompositeNotServable
) -> None:
    (record,) = [r for r in caplog.records if r.exc_info is not None]
    assert record.exc_info is not None
    assert record.exc_info[0] is error


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize(
    ("error", "degraded"),
    [
        pytest.param(_not_servable, False, id="not-servable"),
        pytest.param(lambda: GraphInfraError("down"), False, id="lost-read"),
        pytest.param(lambda: ValueError("a fault"), True, id="fault"),
    ],
)
def test_a_raising_scope_decision_keeps_the_answer_as_a_single_table(
    monkeypatch, caplog, stream, error, degraded
):
    """A decision that raises is logged. The answer stands as one table that states no
    scope, and reads its preview as a single-table answer does. A composite its components
    cannot serve or a lost read leaves the answer storable; any other error is a fault,
    noted as a degraded read, so the answer is served and no cache stores it."""
    _components(monkeypatch)
    asked = _raising_decisions(monkeypatch, error)
    reads = _table_reads(monkeypatch, refuse=False)

    with caplog.at_level(logging.WARNING, logger=scope_decision.logger.name):
        events = (
            _stream(_fresh(_QUESTION))
            if stream
            else offline_events(_fresh(_QUESTION), **_clients())
        )

    spec = _answered(events)
    assert spec.resolved.scope_choices is None
    assert stream or spec.resolved.observations is not None
    assert reads == (["observations", "relaunch"] if stream else ["observations"])
    assert asked == [spec.spec_id]
    _assert_decision_error_logged(caplog, type(error()))
    (done,) = [e for e in events if isinstance(e, DoneEvent)]
    assert done.degraded_leg is degraded


def test_a_raising_scope_decision_keeps_a_candidates_survivor_as_a_single_table(
    monkeypatch, caplog
):
    """The candidates path guards its decision as the answer phase does: the survivor stands
    as one table that states no scope, and its table preview is read."""
    reads: list[str] = []
    real = PreviewReader.observations

    def record(self: PreviewReader) -> Any:
        reads.append("observations")
        return real(self)

    monkeypatch.setattr(PreviewReader, "observations", record)
    asked = _raising_decisions(monkeypatch)
    outcome: list[tuple[RegionResult | None, ScopeDecision | None]] = []

    async def body(state: ResolveState) -> None:
        state.include_observations = True
        async with scoped_concurrency():
            result = await resolve_candidates(state, rate_cache=RateTableCache())
            if result is not None and result.observations_task is not None:
                result.observations_task.cancel()
        outcome.append((result, None if result is None else result.scope_decision))

    with caplog.at_level(logging.WARNING, logger=scope_decision.logger.name):
        _resolve_candidates_in_pass(monkeypatch, body)

    ((result, decision),) = outcome
    assert result is not None
    assert result.status == "definite"
    assert decision is not None
    assert (decision.spec_id, decision.scope, decision.composite) == (
        result.spec.spec_id,
        None,
        False,
    )
    assert reads == ["observations"]
    # Each grounded candidate's decision raises, and the unit filter keeps one of them.
    assert result.spec.spec_id in asked
    errors = [r.exc_info[0] for r in caplog.records if r.exc_info is not None]
    assert errors and set(errors) == {CompositeNotServable}


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_a_raising_scope_decision_leaves_a_carve_out_refusal_standing(monkeypatch, stream):
    """The decision runs before the carve-outs, and an answer they refuse is refused whether
    or not it raised."""
    _components(monkeypatch)
    asked = _raising_decisions(monkeypatch)
    unbound = ResolvedQualifier(text="grants", named=[], slot=None, disposition="unbound")
    monkeypatch.setattr(phase_answer, "settle_answer", lambda *args, **kwargs: (unbound,))

    if stream:
        (response,) = [e for e in _stream(_fresh(_QUESTION)) if isinstance(e, ResponseEvent)]
        resolution = response.response
    else:
        resolution = offline_resolve(_fresh(_QUESTION), **_clients())

    assert isinstance(resolution, Resolution)
    (part,) = resolution.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    assert len(asked) == 1


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_a_candidates_survivor_minted_as_a_composite_reads_no_table_preview(monkeypatch, stream):
    """The candidates path decides each candidate's scope when its Spec is minted
    (`candidate_grounding._candidate_answer`), before the survivor's preview launches
    (`candidate_collapse._assemble_definite`). A survivor minted as a composite reads no
    preview of its table, buffered or streamed, and its preview reads the composite's
    components (`CompositeRelaunch`).

    The set is India's DAC2a readings with every recipient open, left with the donor reading
    over the bilateral composite's population once the recipient reading's mint refuses
    (`test_dac2a_candidate_readings`)."""
    patch_india_coverage(monkeypatch)
    refuse_the_recipient_reading_s_mint(monkeypatch)
    # The survivor's table preview is read by the candidates path or by the answer phase. The
    # composite's components read theirs through their own resubmits.
    reads = _table_reads(monkeypatch, refuse=False)

    class Recording(PreviewReader):
        def observations(self) -> Any:
            reads.append("observations")
            return super().observations()

        def relaunch(self, labels: Any) -> Any:
            reads.append("relaunch")
            return super().relaunch(labels)

    monkeypatch.setattr(candidate_collapse, "PreviewReader", Recording)
    survivors: list[RegionResult] = []
    assemble = candidate_collapse._assemble_definite

    async def capture(*args: Any, **kwargs: Any) -> RegionResult:
        result = await assemble(*args, **kwargs)
        survivors.append(result)
        return result

    monkeypatch.setattr(candidate_collapse, "_assemble_definite", capture)
    request = india_request(open_role="recipient", observations=True)

    events = offline_events(request, stream_slot_options=stream, **india_clients())

    spec = _answered(events)
    (result,) = survivors
    assert result.status == "definite"
    assert result.spec.spec_id == spec.spec_id
    pin = (spec.definition.identity.facets or {}).get("provenance")
    assert pin is not None and pin.ref == _DAC2A_BILATERAL
    assert result.observations_task is None
    assert isinstance(result.observations_relaunch, CompositeRelaunch)
    assert reads == []


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize("door", ["definite", "candidates"])
def test_a_composite_answer_probes_its_slot_options_on_every_door(monkeypatch, door, stream):
    """An answer minted as a composite with option coverage asked for probes its uncounted
    slot menus once (`_maybe_enrich_slot_options`) when buffered, and streams no option
    event of the table it read, whether the answer phase mints it (`phase_answer._finalize_definite_result`) or
    a candidate set's survivor is minted as it (`candidate_grounding._finish_candidate`).
    The set is the one `test_a_candidates_survivor_minted_as_a_composite_reads_no_table_preview`
    reads."""
    probes: list[str] = []
    for module in (phase_answer, candidate_options):
        enrich = module._maybe_enrich_slot_options

        def recording(*args: Any, _enrich: Any = enrich, **kwargs: Any) -> Any:
            probes.append("slot_options")
            return _enrich(*args, **kwargs)

        monkeypatch.setattr(module, "_maybe_enrich_slot_options", recording)
    coverage = ResolveOptions(include_option_coverage=True)
    if door == "definite":
        _components(monkeypatch)
        request = _fresh(_QUESTION).model_copy(update={"response_options": coverage})
        clients = _clients()
        provider = None
    else:
        patch_india_coverage(monkeypatch)
        refuse_the_recipient_reading_s_mint(monkeypatch)
        request = india_request(open_role="recipient").model_copy(
            update={"response_options": coverage}
        )
        clients = india_clients()
        provider = _DAC2A_BILATERAL

    events = offline_events(request, stream_slot_options=stream, **clients)

    spec = _answered(events)
    assert spec.resolved.scope_choices is not None
    assert spec.resolved.scope_choices.compiled.spec_id == spec.spec_id
    if provider is not None:
        pin = (spec.definition.identity.facets or {}).get("provenance")
        assert pin is not None and pin.ref == provider
    # A streamed request reads its option coverage from the trailing events alone.
    assert probes == ([] if stream else ["slot_options"])
    assert not [event for event in events if isinstance(event, SlotOptionsEvent)]
