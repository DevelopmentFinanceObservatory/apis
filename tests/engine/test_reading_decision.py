"""A part's provider reading elected before its read (`reading.elect_reading`), applied to
the identity the answer phase mints after the read (`scope_decision.build_part_scope`),
states the scope decided over that identity itself (`scope_decision.decide_scope`), which
the answer phase checks it against. Where the range the answer settles on would read a
composite the years read withheld, the answer says so. A place set reads with the election
of the place it settles on, and a composite that refuses its table read answers no_data.

The fixture is `test_dac2a_destination_populations`': France's 206 to Ethiopia, to the
European Commission and to `DACAllRecipients`, and its imputed 106 to Ethiopia.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from code_index import members
from code_index.composites import destination_population
from code_index.declarations import DAC2A_INCLUSIVE
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.errors import EngineInputError
from search.engine.identity_edit import TableRead
from search.engine.region import (
    phase_answer,
    phase_place_materialise,
    pipeline,
    reading,
    resubmit,
    scope_decision,
)
from search.engine.region.candidates import _Candidate
from search.engine.region.place_context import empty_place_context
from search.engine.region.reading import ElectionFacts, ReadingDecision, elect_reading
from search.engine.region.scope_decision import (
    READ_WINDOW_NOT_COVERED,
    ProviderChoice,
    ScopeDecision,
    build_part_scope,
)
from search.engine.unit.conversion import ConversionPeriod
from search.models import (
    BaseIdentity,
    DefinitePart,
    IdentityBindingValue,
    NoDataPart,
    Resolution,
    ResolveRequest,
)
from search.scope.state import ScopeState
from search.scope.tables import ScopeFacts
from tests.engine._builders import slow_related_reads
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_dac2a_destination_populations import (
    _AGENCY,
    _BILATERAL,
    _COMPOSITE_AHEAD,
    _COMPOSITE_BEHIND,
    _DAC2A,
    _DEVELOPING,
    _ETH,
    _FRA,
    _FRANCE,
    _HELD,
    _SCENARIOS,
    _TOTAL_YEARS,
    _ended_state,
    _fresh_request,
    _graph_with_parties,
    _identity,
    _part,
    _patched_coverage,
    _ragged,
    _sdmx,
    _trailing,
    _withheld,
    _years_short,
)
from tests.engine.test_whole_total_bind_e2e import _VARIABLE


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_coverage(monkeypatch, _HELD)


@dataclasses.dataclass(frozen=True)
class _Decided:
    """The scope decided over an identity the answer phase minted: what it read and what
    it decided."""

    spec_id: str
    identity: BaseIdentity
    facts: ScopeFacts
    decision: ScopeDecision


def _elect_and_decide(
    request: ResolveRequest, sdmx: FakeSdmxClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[ReadingDecision | None], list[_Decided]]:
    """The reading each table read of `request` elects from the part's state before it
    reads, and the scope decided over each identity the answer phase mints after it
    (`scope_decision.decide_scope`), which the answer phase checks its applied choice
    against (`reading._checked`)."""
    elected: list[ReadingDecision | None] = []
    decided: list[_Decided] = []
    task, checked = phase_place_materialise._materialise_task, reading._checked

    def electing_task(state: Any) -> Any:
        unsettled = state.unsettled_place
        assert unsettled is None or unsettled.decided, "the fixture settles its place first"
        elected.append(
            elect_reading(
                state.shape_draft,
                list(state.bindings),
                state.place,
                date_request=state.date_request,
                selection=copy.deepcopy(state.scope_selection),
                facts=ElectionFacts(coverage=state.coverage, operations=state.operations),
            )
        )
        return task(state)

    def recording_check(
        applied: ScopeDecision,
        identity: BaseIdentity,
        elected: ReadingDecision,
        *,
        selection: ScopeState,
        facts: ScopeFacts,
    ) -> ScopeDecision:
        spec_id = applied.spec_id
        decision = scope_decision.decide_scope(
            spec_id, identity, selection=copy.deepcopy(selection), facts=facts
        )
        decided.append(_Decided(spec_id, identity, facts, decision))
        return checked(applied, identity, elected, selection=selection, facts=facts)

    monkeypatch.setattr(phase_place_materialise, "_materialise_task", electing_task)
    monkeypatch.setattr(reading, "_checked", recording_check)
    offline_resolve(request, graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=sdmx)
    return elected, decided


def _stated(decision: ScopeDecision) -> dict[str, Any]:
    """What a scope decision states of its answer."""
    return {
        "scope": None if decision.scope is None else decision.scope.model_dump(mode="json"),
        "composite": decision.composite,
        "warning": None if decision.warning is None else decision.warning.model_dump(mode="json"),
    }


def _assert_elected_scope_is_decided(
    elected: list[ReadingDecision | None], decided: list[_Decided]
) -> ReadingDecision:
    """The one reading elected before the read, applied to the identity the answer phase
    minted, states the scope it decided over that identity."""
    (reading,) = elected
    (answer,) = decided
    assert reading is not None
    applied = build_part_scope(
        answer.identity, reading.choice, spec_id=answer.spec_id, facts=answer.facts
    )
    assert _stated(applied) == _stated(answer.decision)
    return reading


def _composite(decision: ScopeDecision) -> str | None:
    """The provenance the decision's scope compiles to, where it is a composite."""
    if not decision.composite:
        return None
    assert decision.scope is not None
    pin = (decision.scope.compiled.identity.facets or {}).get("provenance")
    return pin.ref if isinstance(pin, IdentityBindingValue) else None


_TO_AGENCY = lambda: _fresh_request(  # noqa: E731
    [_FRANCE, ("to", "European Commission")],
    message="official development assistance from France to the European Commission",
    open_role=None,
)


@pytest.mark.parametrize(
    ("request_of", "source", "population"),
    [
        pytest.param(_SCENARIOS["H1"], _BILATERAL, True, id="H1"),
        pytest.param(_SCENARIOS["H2"], DAC2A_INCLUSIVE, True, id="H2"),
        # All recipients spans both sides, so the default reads total ODA there.
        pytest.param(_SCENARIOS["H5"], None, False, id="H5"),
        # A ranking over every donor has no provider axis and still reads the composite.
        pytest.param(_SCENARIOS["H6"], _BILATERAL, True, id="H6"),
        # An agency is a multilateral destination, which the composite holds no row for.
        pytest.param(_TO_AGENCY, None, False, id="agency"),
    ],
)
def test_the_reading_elected_before_the_read_states_the_scope_decided_after_it(
    request_of: Any, source: str | None, population: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    elected, decided = _elect_and_decide(request_of(), _sdmx(), monkeypatch)

    reading = _assert_elected_scope_is_decided(elected, decided)
    (answer,) = decided
    assert _composite(answer.decision) == source
    assert (reading.table is not None) == (source is not None)
    assert bool(reading.table and reading.table.populations) == population
    if request_of is _TO_AGENCY:
        assert answer.identity.entities.get("destination") == [_AGENCY]


@pytest.mark.parametrize(
    ("years", "source"),
    [
        # The bilateral destinations lack 2021, the newest year of the settled range.
        pytest.param(_COMPOSITE_BEHIND, None, id="composite-behind"),
        pytest.param(_COMPOSITE_AHEAD, _BILATERAL, id="composite-ahead"),
    ],
)
def test_a_reading_elected_over_the_read_window_states_the_settled_range_s_scope(
    years: dict[tuple[str, str, str], tuple[int, ...]],
    source: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "The last five years" reads a window wider than the range the answer settles on
    after the read (`extract.read_window`), and the reading elected over the wider window
    states the settled range's scope."""
    sdmx = _ragged(monkeypatch, years)

    elected, decided = _elect_and_decide(_trailing("H1"), sdmx, monkeypatch)

    _assert_elected_scope_is_decided(elected, decided)
    (answer,) = decided
    settled = answer.identity.requested_window
    assert settled.kind == "range"
    assert (settled.start_year, settled.end_year) == (2017, 2021)
    assert _composite(answer.decision) == source


def _pin(part: DefinitePart) -> str | None:
    """The provenance `part`'s definition pins."""
    pin = (part.spec.definition.identity.facets or {}).get("provenance")
    return pin.ref if isinstance(pin, IdentityBindingValue) else None


def _faults(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The ERROR records the reading decision logged, a fault in the reading election."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR and record.name == reading.__name__
    ]


# The bilateral destinations' rows start a year after the rest of the table's, so "the last
# five years" settles on 2017 to 2021, which the composite covers, but reads 2016 too.
_WINDOW_GAP = {
    key: _TOTAL_YEARS if key[2] == _ETH else (_TOTAL_YEARS[0] - 1, *_TOTAL_YEARS) for key in _HELD
}


def _unwithheld(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every election from here on withholds nothing, as an election whose document lost the
    bilateral default would. That diverges the pre-read identity from the minted one
    (`reading._pre_read_identity`)."""
    elect = reading.elect_reading

    def unwithheld(*args: Any, **kwargs: Any) -> ReadingDecision | None:
        elected = elect(*args, **kwargs)
        if elected is None:
            return None
        return dataclasses.replace(
            elected, choice=dataclasses.replace(elected.choice, withheld=None)
        )

    monkeypatch.setattr(reading, "elect_reading", unwithheld)


def _degraded(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[str]:
    """The subject of each degraded read `module` notes from here on."""
    noted: list[str] = []
    monkeypatch.setattr(module, "note_degraded_read", noted.append)
    return noted


@pytest.mark.parametrize(
    ("scenario", "withheld", "stated", "faulted"),
    [
        pytest.param(
            "H6", True, [{"why": READ_WINDOW_NOT_COVERED}], False, id="option-not-covered"
        ),
        # Withheld for no reason, the table election is no window gap, so it is a fault.
        pytest.param("H6", False, [], True, id="withheld-nothing"),
        # A part with a provider axis withholds nothing: its bilateral option states the
        # composite's years, so the window gap reads the table with no warning and no fault.
        pytest.param("H1", True, [], False, id="provider-axis"),
    ],
)
def test_a_reading_the_read_window_withholds_and_the_settled_range_allows_says_so(
    scenario: str,
    withheld: bool,
    stated: list[dict[str, str]],
    faulted: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reading is elected over the years the read reads, so a composite missing a year
    the settled range leaves out is withheld, and the answer reads total ODA saying the
    bilateral destinations do not cover every year read, with no fault logged. An election
    of the table that withheld nothing is a different case. It is logged at ERROR and
    noted as a degraded read."""
    sdmx = _ragged(monkeypatch, _WINDOW_GAP)
    if not withheld:
        _unwithheld(monkeypatch)
    degraded = _degraded(monkeypatch, reading)

    with caplog.at_level(logging.WARNING, logger=reading.__name__):
        part = _part(
            offline_resolve(
                _trailing(scenario),
                graph=_graph_with_parties(),
                llm=_ScriptedLLM(_VARIABLE),
                sdmx=sdmx,
            )
        )

    window = part.spec.definition.identity.requested_window
    assert (window.kind, window.start_year, window.end_year) == ("range", 2017, 2021)
    assert _pin(part) is None
    assert [params for _spec_id, params in _withheld(part)] == stated
    assert bool(_faults(caplog)) is faulted
    assert degraded == (["reading election"] if faulted else [])


@pytest.mark.parametrize(
    "step",
    [
        pytest.param("_elected_dataset", id="electing-the-dataset"),
        # Before the document exists, so the bilateral default's reach is read after the read.
        pytest.param("_pre_read_identity", id="building-the-document"),
    ],
)
def test_an_election_that_raises_building_its_document_reads_total_oda_saying_so(
    step: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An election that raises before its choice, building its document or electing the
    dataset that document names, is logged at ERROR and noted as a degraded read, and the
    part reads the table as total ODA saying its scope could not be decided."""

    def raising(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("index fault")

    monkeypatch.setattr(reading, step, raising)
    degraded = _degraded(monkeypatch, scope_decision)

    with caplog.at_level(logging.ERROR, logger=scope_decision.logger.name):
        part = _part(
            offline_resolve(
                _SCENARIOS["H1"](),
                graph=_graph_with_parties(),
                llm=_ScriptedLLM(_VARIABLE),
                sdmx=_sdmx(),
            )
        )

    assert _pin(part) is None
    assert [params for _spec_id, params in _withheld(part)] == [{"why": "scope_undecided"}]
    assert any(
        record.exc_info is not None and record.exc_info[0] is RuntimeError
        for record in caplog.records
    )
    assert "scope decision" in degraded


def test_a_composite_elected_without_its_read_within_reads_as_the_table(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A reading whose choice names the bilateral composite but whose read was not made within
    the composite's population (`ReadingDecision.table` None) read the whole table, so the
    answer is the table's Spec, never minted as the composite. It says it reads total ODA,
    and the election's fault is logged at ERROR and noted as a degraded read."""
    elect = reading.elect_reading

    def unread(*args: Any, **kwargs: Any) -> ReadingDecision | None:
        elected = elect(*args, **kwargs)
        assert elected is not None and elected.table is not None
        return dataclasses.replace(elected, table=None, totals={})

    monkeypatch.setattr(reading, "elect_reading", unread)
    degraded = _degraded(monkeypatch, reading)

    with caplog.at_level(logging.ERROR, logger=reading.__name__):
        part = _part(
            offline_resolve(
                _SCENARIOS["H1"](),
                graph=_graph_with_parties(),
                llm=_ScriptedLLM(_VARIABLE),
                sdmx=_sdmx(),
            )
        )

    assert _pin(part) is None
    assert part.spec.resolved.scope_choices is None
    assert [params for _spec_id, params in _withheld(part)] == [{"why": "scope_undecided"}]
    assert _faults(caplog)
    assert degraded == ["reading election"]


def _decided_as_the_table(monkeypatch: pytest.MonkeyPatch, *, spec_id: str | None) -> None:
    """The answer phase's scope from here on compiles no composite and mints none, so the
    finished answer is the table's Spec. The decision is made for another answer, or, with
    `spec_id` None, for the answer itself, the table's Spec (`ReadScope.answer`)."""
    decide = phase_answer._decide_answer_scope

    def as_table(*args: Any) -> Any:
        scope = decide(*args)
        assert scope.decision is not None
        decision = dataclasses.replace(
            scope.decision,
            spec_id=scope.answer.spec_id if spec_id is None else spec_id,
            composite=False,
            scope=None,
        )
        return dataclasses.replace(scope, decision=decision, minted=None)

    monkeypatch.setattr(phase_answer, "_decide_answer_scope", as_table)


@pytest.mark.parametrize(
    ("within", "decided_for"),
    [
        pytest.param(True, "other", id="read-within"),
        # The table's decision states the answer's spec_id, but its rows were read within the
        # population, so no table label stands over them.
        pytest.param(True, None, id="read-within-as-table"),
        pytest.param(False, "other", id="table-read"),
    ],
)
def test_a_finished_answer_decided_again_chooses_no_composite(
    within: bool,
    decided_for: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A finished answer no decision made before its read states is decided again from its
    Spec, which here, the table of H1, chooses the bilateral composite. That decision chooses
    nothing. It is logged at ERROR and noted as a degraded read, and the answer fails
    closed. Rows read within the composite's population and not minted as it answer no_data,
    `facet_not_available`, before anything is decided, whatever the decision made before the
    read states, and before the answer's related read is awaited, which is cancelled. Rows
    read from the whole table read total ODA, saying its scope could not be decided, and
    state their related read."""
    if not within:
        # The read elects no reading, so it reads the whole table.
        monkeypatch.setattr(reading, "elect_reading", lambda *args, **kwargs: None)
    _decided_as_the_table(monkeypatch, spec_id=decided_for)
    degraded = _degraded(monkeypatch, scope_decision)
    decided_again: list[str] = []
    decide_spec_scope = scope_decision.decide_spec_scope

    def deciding(spec: Any, **kwargs: Any) -> Any:
        decided_again.append(spec.spec_id)
        return decide_spec_scope(spec, **kwargs)

    monkeypatch.setattr(scope_decision, "decide_spec_scope", deciding)
    related = slow_related_reads(monkeypatch)
    awaited: list[Any] = []
    with_related = phase_answer.with_related

    async def awaiting(spec: Any, read: Any) -> Any:
        awaited.append(read)
        return await with_related(spec, read)

    monkeypatch.setattr(phase_answer, "with_related", awaiting)

    with caplog.at_level(logging.ERROR, logger=scope_decision.logger.name):
        response = offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    if within:
        assert isinstance(part, NoDataPart), part
        assert part.no_data.reason == "facet_not_available"
    else:
        assert isinstance(part, DefinitePart), part
        assert _pin(part) is None
        assert part.spec.resolved.scope_choices is None
        assert [params for _spec_id, params in _withheld(part)] == [{"why": "scope_undecided"}]
    # Rows read within the population fail closed before anything is decided again.
    assert decided_again == ([] if within else [part.spec.spec_id])
    # Each read but the answer's last is replaced as its years settle, and cancelled.
    *replaced, read = related
    assert all(earlier.task.cancelled() for earlier in replaced)
    assert awaited == ([] if within else [read])
    assert read.task.cancelled() is within
    logged = [r.levelno for r in caplog.records if r.name == scope_decision.logger.name]
    assert logged == [logging.ERROR]
    assert degraded == ["scope decision"]


def test_a_finished_answer_that_fails_closed_raises_and_judges_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer that fails closed at its check contributes no warning, departure fact or
    judgement. Over the trailing years, grounding departs `date_coverage_partial` for the
    stale years it settles on (`phase_answer._ground_elected_answer`), the Spec minted as
    the composite rewords `role_scope_defaulted` to count the composite's parties
    (`CompositeMint.role_scope_warnings`), and the carve-outs judge the answer
    (`requirements.settle_answer`). Refused at its check, the part holds what it had raised
    and judged before grounding (`ResolveState.mark_answer`, `phase_answer.failed_closed`)."""

    async def refused(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(phase_answer, "finished_scope", refused)
    held: dict[str, tuple[Any, ...]] = {}

    def holding(state: Any) -> tuple[Any, ...]:
        return (tuple(state.warnings), dict(state.departure_facts), state.judgement)

    ground = phase_answer._ground_elected_answer

    async def grounding(state: Any) -> Any:
        held["grounding"] = holding(state)
        return await ground(state)

    monkeypatch.setattr(phase_answer, "_ground_elected_answer", grounding)
    assemble = phase_answer._assemble_definite_answer

    async def assembling(state: Any, *args: Any) -> Any:
        held["assembly"] = holding(state)
        return await assemble(state, *args)

    monkeypatch.setattr(phase_answer, "_assemble_definite_answer", assembling)
    closed: list[Any] = []
    close = phase_answer.failed_closed

    def closing(state: Any, *args: Any, **kwargs: Any) -> Any:
        held["check"] = holding(state)
        result = close(state, *args, **kwargs)
        closed.append((result, state.judgement))
        return result

    monkeypatch.setattr(phase_answer, "failed_closed", closing)

    offline_resolve(
        _trailing("H1"), graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=_sdmx()
    )

    ((result, judgement),) = closed
    warnings, facts, before = held["grounding"]
    assert (result.status, result.no_data_reason) == ("no_data", "facet_not_available")
    # Grounding departed for the stale years before assembly started.
    assert [w.code for w in held["assembly"][0]] == [
        *(w.code for w in warnings),
        "date_coverage_partial",
    ]
    checked, _facts, judged = held["check"]
    assert [w for w in checked if w.code == "role_scope_defaulted"] != [
        w for w in warnings if w.code == "role_scope_defaulted"
    ]
    assert judged is not None
    assert result.warnings == warnings
    assert result.departure_facts == facts
    assert judgement is before


def test_a_finished_answer_is_judged_by_the_reading_its_own_rows_were_read_under(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished answer is judged by the reading its own rows were read under (the
    `reading` its producer passes `phase_answer.finished_scope`), which on the candidates
    path is the survivor's, not the elected shape's (`ResolveState.reading`). Here the
    elected shape read the whole table and the answer's rows were read within the bilateral
    population. Not minted as the composite and stated by no decision, it fails closed,
    noted as a degraded read."""
    captured: list[tuple[Any, Any]] = []
    composite_preview = pipeline._with_composite_preview

    async def capture(result: Any, state: Any) -> Any:
        captured.append((result, state))
        return await composite_preview(result, state)

    monkeypatch.setattr(pipeline, "_with_composite_preview", capture)
    readings: list[Any] = []
    finished = phase_answer.finished_scope

    async def judged(*args: Any, reading: Any, **kwargs: Any) -> Any:
        readings.append(reading)
        return await finished(*args, reading=reading, **kwargs)

    monkeypatch.setattr(phase_answer, "finished_scope", judged)
    offline_resolve(
        _SCENARIOS["H1"](), graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=_sdmx()
    )
    monkeypatch.setattr(phase_answer, "finished_scope", finished)
    ((result, state),) = captured
    (answer_reading,) = readings
    assert answer_reading is not None and answer_reading.table is not None
    state.reading = None
    degraded = _degraded(monkeypatch, scope_decision)

    decision = asyncio.run(
        phase_answer.finished_scope(
            state, result.spec, decision=None, minted=None, reading=answer_reading
        )
    )

    assert decision is None
    assert degraded == ["scope decision"]


def test_a_definite_answer_carries_the_scope_decision_it_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The definite answer's producer carries the decision its check returned on the result
    (`RegionResult.scope_decision`), and the pass states that decision's scope, whatever the
    pass's state holds by then. A definite result carrying none raises."""
    captured: list[Any] = []
    scope_choices = pipeline._with_scope_choices

    def capture(result: Any) -> Any:
        captured.append(result)
        return scope_choices(result)

    monkeypatch.setattr(pipeline, "_with_scope_choices", capture)
    response = offline_resolve(
        _SCENARIOS["H1"](), graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=_sdmx()
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    (result,) = captured
    decision = result.scope_decision
    assert decision is not None and decision.composite
    assert decision.spec_id == part.spec.spec_id
    assert part.spec.resolved.scope_choices == decision.scope
    with pytest.raises(TypeError):
        scope_choices(dataclasses.replace(result, scope_decision=None))


def _narrowed(start: int, end: int) -> Any:
    """`phase_answer._over_conversion_period` narrowing every answer to `start` to `end`, as
    a conversion whose tables cover those years alone narrows it (`_narrowed_to_period`)."""

    def over_period(
        state: Any, mat_result: Any, filtered: Any, admission: Any, *, settled: Any
    ) -> Any:
        period = ConversionPeriod(
            served=(_TOTAL_YEARS[0], _TOTAL_YEARS[-1]),
            years=(start, end),
            limited_by="exchange_rates",
        )
        mat_result, filtered, _shown = phase_answer._narrowed_to_period(
            state, mat_result, admission, period, requested=None
        )
        return mat_result, filtered

    return over_period


@pytest.mark.parametrize(
    ("years", "source", "withheld"),
    [
        pytest.param(dict.fromkeys(_HELD, _TOTAL_YEARS), _BILATERAL, [], id="composite-every-year"),
        # The bilateral destinations lack 2021, the latest year, but cover 2018 to 2020.
        pytest.param(
            _COMPOSITE_BEHIND,
            None,
            [{"why": READ_WINDOW_NOT_COVERED}],
            id="composite-behind",
        ),
    ],
)
def test_a_latest_request_narrowed_to_a_range_keeps_the_reading_it_read(
    years: dict[tuple[str, str, str], tuple[int, ...]],
    source: str | None,
    withheld: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `latest` request elects its reading over the latest year, and a conversion's period
    narrows its answer to a range after the read (`_narrowed_to_period`). The answer keeps
    the reading it read. Where the range would have read the composite the latest year
    withheld, it says so, and no fault is logged."""
    sdmx = _ragged(monkeypatch, years)
    monkeypatch.setattr(phase_answer, "_over_conversion_period", _narrowed(2018, 2020))
    (parsed,) = _SCENARIOS["H6"]().input.parts
    request = _SCENARIOS["H6"]().model_copy(deep=True)
    request.input.parts[0] = parsed.model_copy(update={"time_text": "latest"})

    with caplog.at_level(logging.WARNING, logger=reading.__name__):
        part = _part(
            offline_resolve(
                request, graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=sdmx
            )
        )

    window = part.spec.definition.identity.requested_window
    assert (window.kind, window.start_year, window.end_year) == ("range", 2018, 2020)
    assert _pin(part) == source
    assert [params for _spec_id, params in _withheld(part)] == withheld
    assert _faults(caplog) == []


def test_a_place_set_settling_on_its_members_reads_with_their_election(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A place set read at its node and then at its members elects at each, and the read
    the set settles on, the members', carries the members' election and reads within the
    population that election names, where the node's read elected none."""
    population = destination_population(get_index(), _DAC2A)
    document = _identity(_DAC2A).model_dump(mode="json", exclude_none=True)
    document.pop("facets")
    document.pop("open_roles")
    document["entities"] = {**document["entities"], "destination": [population.total]}
    table = TableRead(
        identity=BaseIdentity.model_validate(document), populations={"destination": population}
    )
    choice = ProviderChoice(
        states_scope=True,
        bilateral=_BILATERAL,
        withheld=None,
        selection=ScopeState(),
        provider=None,
    )
    node, members = empty_place_context(), empty_place_context()
    elected = {id(node): None, id(members): ReadingDecision(choice=choice, table=table)}
    read_within: list[Any] = []

    def settle(_place: Any, read: Any) -> Any:
        read(node, None)
        return members, read(members, None)

    def materialise(*_args: Any, within: Any, **_kwargs: Any) -> Any:
        read_within.append(within)
        return within

    monkeypatch.setattr(phase_place_materialise, "settle_sets", settle)
    monkeypatch.setattr(reading, "elect_reading", lambda *a, **k: elected[id(a[2])])
    monkeypatch.setattr(phase_place_materialise, "run_materialise", materialise)
    state = SimpleNamespace(
        shape_draft=None,
        bindings=(),
        default_presence=None,
        unsettled_place=None,
        place=node,
        graph=None,
        sdmx=None,
        date_request=None,
        deadline=None,
        coverage=None,
        include_observations=False,
        stream_observations=False,
        scope_selection=ScopeState(),
        operations=(),
    )

    async def read() -> Any:
        return await phase_place_materialise._materialise_task(state)

    settled = asyncio.run(read())

    assert settled.place is members
    assert settled.reading is elected[id(members)]
    assert read_within == [None, {"destination": frozenset(population.members)}]
    assert settled.outcome == read_within[-1]


@pytest.mark.parametrize(
    ("elections", "identities"),
    [
        pytest.param(1, 1, id="read-once"),
        pytest.param(2, 1, id="read-twice"),
        # The members of a place set, or other windows or shapes, each elect the composite
        # under an identity of their own.
        pytest.param(2, 2, id="read-twice-under-two-identities"),
    ],
)
def test_a_composite_that_refuses_its_table_read_answers_facet_not_available(
    elections: int,
    identities: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A composite the part's scope compiles to that refuses its table read is logged at
    ERROR and noted as a degraded read, so no cache stores the answer, and the part reads
    nothing and answers no_data, `facet_not_available`. A part whose reads elect the refused
    composite twice logs and notes it once (`reading.noting_refusals_once`), also where
    each read elects it under its own identity."""

    def refusing(*_args: object, **_kwargs: object) -> object:
        raise EngineInputError("refused", code="requested_option_unavailable", status=422)

    monkeypatch.setattr(reading, "composite_table_identity", refusing)
    if identities > 1:
        minted = iter(range(elections))
        monkeypatch.setattr(reading, "identity_spec_id", lambda _identity: f"spec_{next(minted)}")
    elect = reading.elect_reading

    def electing(*args: Any, **kwargs: Any) -> Any:
        for _ in range(elections - 1):
            elect(*args, **kwargs)
        return elect(*args, **kwargs)

    monkeypatch.setattr(reading, "elect_reading", electing)
    degraded = _degraded(monkeypatch, reading)

    with caplog.at_level(logging.ERROR, logger=reading.__name__):
        response = offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "facet_not_available"
    assert [r.levelno for r in caplog.records if r.name == reading.__name__] == [logging.ERROR]
    assert degraded == ["reading election"]


@pytest.mark.parametrize(
    ("code", "raises"),
    [
        # The answer's mint refuses a reversed window, so no election is made for it.
        pytest.param("request_invalid", False, id="reversed-window"),
        pytest.param("requested_option_unavailable", True, id="other-refusal"),
    ],
)
def test_a_reversed_window_elects_no_reading(
    code: str, raises: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refusing(*_args: object, **_kwargs: object) -> object:
        raise EngineInputError("refused", code=code)

    monkeypatch.setattr(reading, "elect_reading", refusing)

    def elect() -> ReadingDecision | None:
        return reading.elect_at_read(
            cast(Any, None),
            (),
            empty_place_context(),
            date_request=None,
            selection=ScopeState(),
            facts=ElectionFacts(coverage=cast(Any, None), operations=()),
        )

    if raises:
        with pytest.raises(EngineInputError):
            elect()
    else:
        assert elect() is None


def test_a_stand_in_is_adopted_with_its_own_read_and_election(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-observation stand-in answers through its own read, which elected its provider
    reading at its place as the elected shape's read does (`candidates._materialise_alt_shapes`),
    so it is adopted with that read and that election and is read no second time."""
    choice = ProviderChoice(
        states_scope=True,
        bilateral=_BILATERAL,
        withheld=None,
        selection=ScopeState(),
        provider=None,
    )
    elected = ReadingDecision(choice=choice, table=None)

    def reread(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a stand-in is read once")

    monkeypatch.setattr(phase_place_materialise, "run_materialise", reread)
    place = empty_place_context()
    stand_in = _Candidate(
        shape=cast(Any, None),
        bindings=[_AID_TYPE_206, _DISBURSEMENT],
        mat=cast(Any, "its own read"),
        place=place,
        walked=None,
        date_request=None,
        reading=elected,
    )

    adopted = phase_place_materialise._adopted_read(stand_in)

    assert (adopted.outcome, adopted.reading, adopted.place) == ("its own read", elected, place)


_AID_TYPE_206 = SlotBindingDraft(
    axis="what", property_dcid="dac2aAidType", kind="value", value_dcids=["DAC2aAidType206"]
)
_DISBURSEMENT = SlotBindingDraft(
    axis="how", property_dcid="financialFlowType", kind="value", value_dcids=["Disbursement"]
)


def test_a_composite_that_cannot_be_minted_answers_no_data_not_the_table(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A part read within the bilateral composite's population whose composite cannot be
    minted answers no_data, `facet_not_available`, and logs at ERROR, since the table's
    label would state the population's rows as total ODA."""

    def raising(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("mint failed")

    monkeypatch.setattr(resubmit, "composite_mint", raising)

    with caplog.at_level(logging.ERROR, logger=reading.__name__):
        response = offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "facet_not_available"
    assert _faults(caplog)


def test_a_carve_out_flagging_a_reading_that_reads_total_oda_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over every donor with bilateral destinations covering fewer years than the answer,
    the part reads total ODA. A carve-out refusal that flags that answer
    (`phase_answer._assemble_carve_outs`), here no member holding data at the place, carries
    the decision's `bilateral_reading_unavailable` keyed to the flagged match, as a definite
    answer of it does."""
    _years_short(monkeypatch)
    carve_outs = phase_answer._assemble_carve_outs

    async def unavailable(state: Any, mat_result: Any, *args: Any, **kwargs: Any) -> Any:
        unheld = dataclasses.replace(mat_result, available_at_place=False)
        return await carve_outs(state, unheld, *args, **kwargs)

    async def no_suggestion(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(phase_answer, "_assemble_carve_outs", unavailable)
    monkeypatch.setattr(phase_answer, "_nearest_real_for", no_suggestion)

    response = offline_resolve(
        _SCENARIOS["H6"](), graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=_sdmx()
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "no_observations"
    flagged = part.no_data.flagged_match
    assert flagged is not None and not isinstance(flagged, list)
    withheld = [
        (w.spec_id, w.params) for w in part.warnings if w.code == "bilateral_reading_unavailable"
    ]
    assert withheld == [(flagged.spec_id, {"why": "option_not_covered"})]


def test_a_part_whose_dataset_fixes_the_aid_type_elects_the_bilateral_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A part whose draft leaves DAC2a's aid type unbound while the dataset it names fixes
    206 as an invariant mints an identity binding 206 (`code_index.members.canonical_identity`),
    which the bilateral default reaches after the read. The election reads that identity
    before the read by the same rule (`provider.counted_source`), so it elects the bilateral
    composite and reads within its population, as the part's answer is minted."""
    state = _ended_state(monkeypatch, "H1")
    elected_invariants = members.elected_invariants

    def fixing_206(index: Any, query: Any) -> dict[str, str]:
        if query.source == _DAC2A:
            return {"dac2aAidType": _AID_TYPE_206.value_dcids[0]}
        return elected_invariants(index, query)

    monkeypatch.setattr(members, "elected_invariants", fixing_206)
    bindings = [
        binding.model_copy(update={"kind": "unbound", "value_dcids": []})
        if binding.property_dcid == "dac2aAidType"
        else binding
        for binding in state.bindings
    ]
    assert not any(b.property_dcid == "dac2aAidType" and b.value_dcids for b in bindings)

    elected = elect_reading(
        dataclasses.replace(state.shape_draft, named_dataset=_DAC2A),
        bindings,
        state.place,
        date_request=state.date_request,
        selection=ScopeState(),
        facts=ElectionFacts(coverage=state.coverage, operations=state.operations),
    )

    assert elected is not None and elected.table is not None
    assert elected.choice.bilateral == _BILATERAL


def test_the_elections_of_one_part_s_reads_share_one_set_of_scope_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each read of a part elects at the place it tries, a set's node and then its members,
    and each nearest-real relaxation elects at its own read. All of them read one
    `ScopeFacts` per dataset (`reading.ElectionFacts`), so its family scans and coverage
    reads run once for the part."""
    state = _ended_state(monkeypatch, "H1")
    read: list[ScopeFacts] = []
    choose = reading.choose_provider

    def choosing(identity: Any, selection: Any, facts: ScopeFacts) -> Any:
        read.append(facts)
        return choose(identity, selection, facts)

    monkeypatch.setattr(reading, "choose_provider", choosing)
    facts = ElectionFacts(coverage=state.coverage, operations=state.operations)
    node, members = state.place, dataclasses.replace(state.place)

    for bindings in (list(state.bindings), list(state.bindings)):
        reads = reading.ElectedReads(
            state.shape_draft,
            bindings,
            date_request=state.date_request,
            selection=ScopeState(),
            facts=facts,
            read=lambda *_args: "read",
        )
        assert [reads(place, None) for place in (node, members)] == ["read", "read"]
        assert reads.settled(members) is not None

    assert len(read) == 4
    assert all(facts is read[0] for facts in read)


@pytest.mark.parametrize(
    ("roles", "listed"),
    [
        pytest.param(
            [(_FRA, "origin")],
            [(_DEVELOPING, "destination"), (_FRA, "origin")],
            id="resubmit-order",
        ),
        pytest.param(
            [(_ETH, "observationAbout"), (_FRA, "origin")],
            [(_ETH, "observationAbout"), (_DEVELOPING, "destination"), (_FRA, "origin")],
            id="subject-first",
        ),
        pytest.param(
            [(_FRA, "origin"), (_ETH, "observationAbout")],
            [(_DEVELOPING, "destination"), (_FRA, "origin"), (_ETH, "observationAbout")],
            id="subject-last",
        ),
        pytest.param(
            [(_ETH, "observationAbout")],
            [(_ETH, "observationAbout"), (_DEVELOPING, "destination")],
            id="no-flow-role",
        ),
    ],
)
def test_the_roles_a_read_lists_keep_every_context_role_and_add_its_totals(
    roles: list[tuple[str, Any]], listed: list[tuple[str, str]]
) -> None:
    """A read that binds a total on an axis the part leaves out lists the context's roles,
    every key in order, with the total before the first origin, as a resubmit of the
    composite lists destinations first (`reading.roles_read`)."""
    elected = ReadingDecision(
        choice=cast(Any, None),
        table=cast(Any, SimpleNamespace()),
        totals={"destination": (_DEVELOPING,)},
    )

    assert reading.roles_read(roles, elected) == listed
