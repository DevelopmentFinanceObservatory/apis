"""A fresh candidate set whose DAC2a readings settle on population composites elects each
reading at its own read, reads it once, within the population where the reading leaves its
destination open, and mints its Spec as the composite, so each candidate states what a
resubmit of its definition states.

India holds DAC2a's total ODA on both sides here, so "official development assistance in
India" offers India as the donor and as the recipient. Each reading settles on the
bilateral composite, the donor reading at the bilateral population's total and the
recipient reading at India, a bilateral recipient."""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from code_index.composites import destination_population
from code_index.shared import get_index
from origin_catalogue import current_catalogue, scoped_reads
from search.engine.events import ResponseEvent
from search.engine.region import (
    RegionResult,
    candidate_collapse,
    candidate_grounding,
    phase_place_materialise,
    pipeline,
    reading,
    resubmit,
)
from search.engine.region.grounding import carrier_flag_warnings
from search.engine.region.scope_decision import _withheld_warning
from search.engine.region.state import ResolveState
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    CandidatesPart,
    DefinitePart,
    Entry,
    NearestRealSpec,
    NoDataPart,
    Relaxation,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    Warning,
)
from search.scope.provider import Withheld
from series_coverage.schema import FacetMetadata
from tests.engine._builders import raise_answer_warnings, slow_related_reads
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index, patch_coverage
from tests.engine._harness import offline_events, offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_dac2a_destination_populations import (
    _AGENCY,
    _BILATERAL,
    _NAMES,
    _YEARS,
    _counted,
    _fresh_request,
    _identity,
    _offer_the_bilateral_composite,
    _patched_coverage,
    _resolve_resubmit,
    _sdmx,
    _shared_fields,
    _table_reads,
)
from tests.engine.test_dac2a_destination_populations import _HELD as _FRANCE_HELD
from tests.engine.test_whole_total_bind_e2e import (
    _ALL_RECIPIENTS,
    _DAC2A,
    _DAC2A_ATTRIBUTED,
    _DAC2A_MEMBER,
    _ETH,
    _FRA,
    _IND,
    _OFFICIAL,
    _VARIABLE,
    _graph,
)
from tests.scope._axes import carried

_DEVELOPING = "DACDevelopingCountries"
# India's receipts from France and the official donors, and its own ODA to Ethiopia, to an
# agency and to the bilateral population's total.
_HELD = {
    (_DAC2A_MEMBER, _FRA, _IND): 100,
    (_DAC2A_MEMBER, _OFFICIAL, _IND): 300,
    (_DAC2A_MEMBER, _IND, _ETH): 40,
    (_DAC2A_MEMBER, _IND, _AGENCY): 20,
    (_DAC2A_MEMBER, _IND, _DEVELOPING): 45,
    (_DAC2A_ATTRIBUTED, _IND, _ETH): 3,
}


_ROLE_NAMES = dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), _NAMES)
_FACETS = {"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")}


def _rows(held: dict[tuple[str, str, str], int]) -> list[tuple[str, str, str, str, list[int]]]:
    return [(sv, "dac2a", donor, recipient, list(_YEARS)) for (sv, donor, recipient) in held]


def patch_india_coverage(
    monkeypatch: pytest.MonkeyPatch, held: dict[tuple[str, str, str], int] = _HELD
) -> None:
    """The coverage snapshot holding the series `held` in every year of the window."""
    patch_coverage(monkeypatch, _rows(held), role_names=_ROLE_NAMES, facet_metadata=_FACETS)


def read_the_answer_against(
    monkeypatch: pytest.MonkeyPatch, held: dict[tuple[str, str, str], int]
) -> None:
    """The answer's read reads a coverage snapshot holding the series `held` alone, while
    the election and the role resolution before it read the one in force, as a table the
    snapshot disagrees with would."""
    snapshot = FakeCoverageService(
        build_coverage_index(_rows(held), role_names=_ROLE_NAMES, facet_metadata=_FACETS)
    ).snapshot()
    run = phase_place_materialise.run_materialise

    def reading_held(*args: Any, **kwargs: Any) -> Any:
        return run(*args, **{**kwargs, "coverage": snapshot})

    monkeypatch.setattr(phase_place_materialise, "run_materialise", reading_held)


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_india_coverage(monkeypatch)


def india_request(
    *,
    open_role: str | None = None,
    carried_entries: tuple[Entry, ...] = (),
    observations: bool = False,
) -> ResolveRequest:
    """The request for official development assistance in India, India's role left to the
    readings, with `open_role` open, the conversation's `carried_entries`, and the preview
    asked for with `observations`."""
    request = _fresh_request(
        [(None, "India")], message="official development assistance in India", open_role=open_role
    )
    return request.model_copy(
        update={
            "input": request.input.model_copy(
                update={"carried_entries": list(carried_entries) or None}
            ),
            "response_options": ResolveOptions(include_observations=True) if observations else None,
        }
    )


def india_clients(held: dict[tuple[str, str, str], int] = _HELD) -> dict[str, Any]:
    """The graph, binder and SDMX doubles a request about India reads, over `held`."""
    graph = _graph(variable=_VARIABLE, members=(_DAC2A_MEMBER, _DAC2A_ATTRIBUTED))
    population = destination_population(get_index(), _DAC2A)
    for dcid in (*population.members, population.total, _AGENCY, _ALL_RECIPIENTS):
        graph._nodes.setdefault(dcid, {"label": dcid, "type": "Organization"})
    return {"graph": graph, "llm": _ScriptedLLM(_VARIABLE), "sdmx": _sdmx(held)}


def _india(request: ResolveRequest, held: dict[tuple[str, str, str], int] = _HELD) -> Any:
    response = offline_resolve(request, **india_clients(held))
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    return part


def _candidates(request: ResolveRequest | None = None) -> CandidatesPart:
    part = _india(india_request() if request is None else request)
    assert isinstance(part, CandidatesPart), part
    return part


def _source(identity: BaseIdentity) -> str | None:
    pin = (identity.facets or {}).get("provenance")
    return None if pin is None else pin.ref


def _own(part: Any, spec: Any) -> dict[str, Any]:
    """`_shared_fields` of `spec` in `part`, over the warnings keyed to it alone."""
    return _shared_fields(
        SimpleNamespace(
            spec=spec,
            warnings=[warning for warning in part.warnings if warning.spec_id == spec.spec_id],
            departures=[],
        )
    )


@pytest.mark.parametrize(
    ("entries", "source"),
    [
        pytest.param((), _BILATERAL, id="no-entry"),
        # The conversation settled total ODA, contributions included, so each reading keeps it.
        pytest.param((carried("provider:contributions"),), None, id="carried-total"),
    ],
)
def test_each_dac2a_candidate_is_read_within_its_population_and_minted_as_its_composite(
    entries: tuple[Entry, ...], source: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each reading elects its provider reading at its own read, under the part's provider
    entry, and is read once. The donor reading reads at the population's total and the
    recipient reading at India. Each states what a resubmit of its definition states."""
    built: list[str] = []
    build = candidate_grounding.build_spec

    def recorded(**kwargs: Any) -> Any:
        spec = build(**kwargs)
        built.append(spec.spec_id)
        return spec

    monkeypatch.setattr(candidate_grounding, "build_spec", recorded)
    reads = _table_reads(monkeypatch)

    part = _candidates(india_request(carried_entries=entries))

    assert _counted(reads) == [("answer", False), ("candidate", False)]
    specs = part.candidates.specs
    assert [_source(spec.definition.identity) for spec in specs] == [source, source]
    assert all(spec.spec_id in built for spec in specs)
    for spec in specs:
        again = _resolve_resubmit(spec.definition.identity, held=_HELD)
        assert _own(again, again.spec) == _own(part, spec)


@pytest.mark.parametrize(
    ("entries", "within"),
    [
        pytest.param((), True, id="population-composite"),
        # The conversation settled total ODA, so the donor reading reads the plain table.
        pytest.param((carried("provider:contributions"),), False, id="plain-table"),
    ],
)
def test_a_set_whose_elected_reading_s_read_holds_no_row_answers_no_data(
    entries: tuple[Entry, ...], within: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """India's own ODA goes to an agency alone, so India reads as the donor and as the
    recipient, and the donor reading is elected first. Its read returns no row. Settled on
    the bilateral composite, it reads within the population, which holds none. Settled on
    the plain table, it reads a table holding no row India gives. Either way the part
    answers no_data `no_observations` from that one read and reads no other reading. The
    recipient reading is not offered in its place, since a reading without data offers only
    rule 1's stand-ins and the riders (`phase_place_materialise._offered`). The coverage
    the election reads offers the composite only where it serves a year, and holds the
    agency row the plain read finds, so the election is made to offer the composite and
    the read to read no row India gives, as a table the snapshot disagrees with would.

    The no_data answer's search for nearest-real suggestions reads relaxed queries of its
    own (`grounding._relaxed_binding_suggestion`), which read no reading of the set and are
    left out of the count."""
    held = {key: value for key, value in _HELD.items() if key[1] != _IND or key[2] == _AGENCY}
    patch_india_coverage(monkeypatch, held)
    _offer_the_bilateral_composite(monkeypatch)
    read_the_answer_against(
        monkeypatch, {key: value for key, value in held.items() if key[1] != _IND}
    )
    reads = _table_reads(monkeypatch)

    part = _india(india_request(open_role="recipient", carried_entries=entries), held)

    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "no_observations"
    readings = [read for read in _counted(reads) if read[0] != "nearest-real"]
    assert readings == [("answer", within)]


def refuse_the_recipient_reading_s_mint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mint of the composite at India as the recipient raises, so that reading is dropped
    (`candidate_grounding.ground_candidates`) and the donor reading is left alone."""
    mint = resubmit.composite_mint

    def refusing(identity: BaseIdentity, **kwargs: Any) -> Any:
        if identity.entities.get("destination") == [_IND]:
            raise RuntimeError("mint failed")
        return mint(identity, **kwargs)

    monkeypatch.setattr(resubmit, "composite_mint", refusing)


def test_a_set_whose_elected_reading_cannot_be_minted_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The donor reading, elected first, is read within the bilateral population, and its
    composite cannot be minted, so its rows have no label to stand under. The part fails
    closed as the definite answer does (`phase_answer.failed_closed`), describing the
    donor reading, and the recipient reading does not answer in its place."""
    mint = resubmit.composite_mint

    def refusing(identity: BaseIdentity, **kwargs: Any) -> Any:
        if identity.entities.get("destination") != [_IND]:
            raise RuntimeError("mint failed")
        return mint(identity, **kwargs)

    monkeypatch.setattr(resubmit, "composite_mint", refusing)
    closed: list[Any] = []
    close = candidate_collapse.failed_closed

    def closing(state: Any, mark: Any, *, candidate: Any = None) -> Any:
        closed.append(candidate)
        return close(state, mark, candidate=candidate)

    monkeypatch.setattr(candidate_collapse, "failed_closed", closing)

    part = _india(india_request(open_role="recipient"))

    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "facet_not_available"
    (candidate,) = closed
    assert candidate.reading is not None and candidate.reading.table is not None


def test_a_set_left_with_one_composite_reading_answers_it_as_its_resubmit_does(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With every recipient open, the donor reading reads India's ODA within the bilateral
    population and the recipient reading reads India's receipts. The recipient reading's
    composite cannot be minted, so it is dropped with an ERROR, and the set left with the
    donor reading answers it definite, as a resubmit of its composite does, with its spec_id
    and its scope compiled to the composite and the component reading the population."""
    refuse_the_recipient_reading_s_mint(monkeypatch)
    reads = _table_reads(monkeypatch)
    survivors: list[Any] = []
    finished = candidate_collapse.finished_scope

    async def judged(*args: Any, reading: Any, **kwargs: Any) -> Any:
        decision = await finished(*args, reading=reading, **kwargs)
        survivors.append((reading, decision))
        return decision

    monkeypatch.setattr(candidate_collapse, "finished_scope", judged)
    results: list[Any] = []
    assemble = candidate_collapse._assemble_definite

    async def assembled(*args: Any, **kwargs: Any) -> Any:
        result = await assemble(*args, **kwargs)
        results.append(result.scope_decision)
        return result

    monkeypatch.setattr(candidate_collapse, "_assemble_definite", assembled)

    with caplog.at_level(logging.ERROR, logger=reading.__name__):
        part = _india(india_request(open_role="recipient"))

    assert isinstance(part, DefinitePart), part
    # The finished answer is judged by the survivor's own reading
    # (`phase_answer.finished_scope`), read within the population.
    ((survivor, decision),) = survivors
    assert survivor is not None and survivor.table is not None
    # The result carries the decision its check returned, which the part states.
    assert results == [decision]
    assert decision is not None and decision.spec_id == part.spec.spec_id
    assert _counted(reads) == [("answer", True), ("candidate", False)]
    assert [r.levelno for r in caplog.records if r.name == reading.__name__] == [logging.ERROR]
    assert _source(part.spec.definition.identity) == _BILATERAL
    again = _resolve_resubmit(part.spec.definition.identity, held=_HELD)
    assert again.spec.spec_id == part.spec.spec_id
    assert _shared_fields(again) == _shared_fields(part)
    choices = part.spec.resolved.scope_choices
    assert choices is not None
    assert choices.compiled.spec_id == part.spec.spec_id
    (component,) = choices.compiled.components or ()
    population = destination_population(get_index(), _DAC2A)
    assert component.identity.entities["destination"] == list(population.members)


def test_a_reading_dropped_unminted_raises_no_warning_of_its_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recipient reading's composite cannot be minted, so it is dropped and the donor
    reading answers. Each reading's reads are made to raise an observation warning and
    every carrier flag (`candidate_grounding._carrier_flags`); the answer states the donor
    reading's observation warning and none of the dropped reading's, and no carrier flag,
    since the elected donor reading's flags warned at binding."""
    refuse_the_recipient_reading_s_mint(monkeypatch)
    observed = candidate_grounding._candidate_observation_entities

    def warning_of(cand: Any) -> Any:
        entities, _warning = observed(cand)
        role = "recipient" if cand.place.roles.obs_destination_dcid == _IND else "donor"
        return entities, Warning(code=f"observed_{role}", severity="info", message=role)

    monkeypatch.setattr(candidate_grounding, "_candidate_observation_entities", warning_of)
    flags = candidate_grounding._CarrierFlags(True, True, True)
    monkeypatch.setattr(
        candidate_grounding,
        "_carrier_flags",
        lambda _cand, *, elected: (
            candidate_grounding._CarrierFlags(False, False, False) if elected else flags
        ),
    )

    part = _india(india_request(open_role="recipient"))

    assert isinstance(part, DefinitePart), part
    codes = [warning.code for warning in part.warnings]
    assert "observed_donor" in codes
    assert "observed_recipient" not in codes
    flagged = carrier_flag_warnings(
        role_declaration_unverified=True, open_origin_defaulted=True, membership_unverified=True
    )
    stated = {(warning.code, str(warning.params)) for warning in part.warnings}
    assert stated.isdisjoint((warning.code, str(warning.params)) for warning in flagged)


def test_a_survivor_no_scope_states_fails_closed_launching_no_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A survivor whose rows were read within the bilateral population but whose Spec is
    minted as the table, under a decision made for the composite, is stated by no scope. It
    is judged once, before anything of it launches (`candidate_collapse._assemble_definite`,
    `phase_answer.finished_scope`), so it fails closed with no warning about it and no
    related read, option or preview of its own, and the related read started for the
    elected shape before grounding is cancelled: no_data `facet_not_available` stands in its
    place, describing the survivor (`ResolveState.no_data(candidate=)`), and the pass states
    its departures, warnings and timing on it."""
    refuse_the_recipient_reading_s_mint(monkeypatch)
    decide = candidate_grounding.decide_read_scope

    def as_the_table(*args: Any, **kwargs: Any) -> Any:
        scoped = decide(*args, **kwargs)
        if scoped.decision is None:
            return scoped
        return dataclasses.replace(scoped, minted=None)

    monkeypatch.setattr(candidate_grounding, "decide_read_scope", as_the_table)
    launched: list[str] = []

    def recording(name: str, launch: Any) -> Any:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            launched.append(name)
            return launch(*args, **kwargs)

        return recorded

    related = slow_related_reads(monkeypatch)
    for name in (
        "start_answer_related_read",
        "note_denominator_unverified",
        "_unit_omitted_warning",
        "related_read_for",
        "_attach_candidate_slot_options",
        "_make_slot_options_task",
        "_relaunch_place_options",
        "CompositeRelaunch",
    ):
        monkeypatch.setattr(
            candidate_collapse, name, recording(name, getattr(candidate_collapse, name))
        )
    for name in ("relaunch", "observations"):
        monkeypatch.setattr(
            candidate_collapse.PreviewReader,
            name,
            recording(name, getattr(candidate_collapse.PreviewReader, name)),
        )
    assembled: list[tuple[Any, RegionResult]] = []
    assemble = candidate_collapse._assemble_definite

    async def capture(
        state: Any, specs: list[Any], grounded: Any, rate_cache: Any, **kwargs: Any
    ) -> RegionResult:
        result = await assemble(state, specs, grounded, rate_cache, **kwargs)
        assembled.append((grounded[specs[0].spec_id].candidate, result))
        return result

    monkeypatch.setattr(candidate_collapse, "_assemble_definite", capture)
    decorated: list[tuple[str, str | None]] = []
    departures = pipeline.with_requirement_departures

    def decorating(result: RegionResult, *args: Any) -> RegionResult:
        decorated.append((result.status, result.no_data_reason))
        return departures(result, *args)

    monkeypatch.setattr(pipeline, "with_requirement_departures", decorating)

    events = offline_events(
        india_request(open_role="recipient", observations=True),
        stream_slot_options=True,
        **india_clients(),
    )

    (response,) = [event.response for event in events if isinstance(event, ResponseEvent)]
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "facet_not_available"
    ((survivor, result),) = assembled
    assert survivor.reading is not None and survivor.reading.table is not None
    assert result.refused_answer == (survivor.shape, tuple(survivor.bindings), survivor.uncovered)
    # Only the elected shape's related read started, before grounding, and it was cancelled.
    assert launched == ["start_answer_related_read"]
    assert [read.task.cancelled() for read in related] == [True]
    assert decorated == [("no_data", "facet_not_available")]


def test_a_survivor_that_fails_closed_raises_and_judges_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A survivor that fails closed at its check contributes no warning, departure fact or
    judgement. Grounding the candidates restates `role_scope_defaulted` to count the
    survivor's composite's parties (`candidate_grounding.restate_scope_warning`), and the
    survivor's answer raises an omitted currency and an option coverage warning, each with a
    departure fact, and is judged (`requirements.settle_answer`). Refused at its check, the
    part holds what it had raised and judged before the candidates were grounded
    (`ResolveState.mark_answer`, `phase_answer.failed_closed`)."""
    refuse_the_recipient_reading_s_mint(monkeypatch)

    async def refused(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(candidate_collapse, "finished_scope", refused)
    settle = candidate_collapse.settle_answer

    def raising(state: Any, *args: Any, **kwargs: Any) -> Any:
        raise_answer_warnings(state)
        return settle(state, *args, **kwargs)

    monkeypatch.setattr(candidate_collapse, "settle_answer", raising)
    held: dict[str, tuple[Any, ...]] = {}

    def holding(state: Any) -> tuple[Any, ...]:
        return (tuple(state.warnings), dict(state.departure_facts), state.judgement)

    ground = candidate_collapse.ground_candidates

    def grounding(state: Any, candidates: Any) -> Any:
        held["grounding"] = holding(state)
        return ground(state, candidates)

    monkeypatch.setattr(candidate_collapse, "ground_candidates", grounding)
    assemble = candidate_collapse._assemble_definite

    async def assembling(state: Any, *args: Any, **kwargs: Any) -> Any:
        held["assembly"] = holding(state)
        return await assemble(state, *args, **kwargs)

    monkeypatch.setattr(candidate_collapse, "_assemble_definite", assembling)
    closed: list[Any] = []
    close = candidate_collapse.failed_closed

    def closing(state: Any, *args: Any, **kwargs: Any) -> Any:
        held["check"] = holding(state)
        result = close(state, *args, **kwargs)
        closed.append((result, state.judgement))
        return result

    monkeypatch.setattr(candidate_collapse, "failed_closed", closing)

    offline_events(
        india_request(open_role="recipient", observations=True),
        stream_slot_options=True,
        **india_clients(),
    )

    ((result, judgement),) = closed
    warnings, facts, before = held["grounding"]
    assert (result.status, result.no_data_reason) == ("no_data", "facet_not_available")
    # Grounding restated the scope warning before the survivor's assembly started.
    scoped = [w for w in warnings if w.code == "role_scope_defaulted"]
    assert scoped
    assert [w for w in held["assembly"][0] if w.code == "role_scope_defaulted"] != scoped
    _checked, raised, judged = held["check"]
    assert set(raised) - set(facts)
    assert judged is not None
    assert result.warnings == warnings
    assert result.departure_facts == facts
    assert judgement is before


def test_a_survivor_stated_as_no_composite_reads_as_its_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A survivor minted as a composite whose check returns a decision compiling none reads
    as the table that decision states, as the answer phase's does
    (`phase_answer._finalize_definite_result`): its preview streams from the table's read,
    and nothing reads it per component."""
    refuse_the_recipient_reading_s_mint(monkeypatch)
    finished = candidate_collapse.finished_scope

    async def as_no_composite(*args: Any, **kwargs: Any) -> Any:
        decision = await finished(*args, **kwargs)
        assert decision is not None and decision.composite
        return dataclasses.replace(decision, composite=False, scope=None)

    monkeypatch.setattr(candidate_collapse, "finished_scope", as_no_composite)
    results: list[RegionResult] = []
    assemble = candidate_collapse._assemble_definite

    async def assembling(*args: Any, **kwargs: Any) -> Any:
        result = await assemble(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(candidate_collapse, "_assemble_definite", assembling)

    offline_events(
        india_request(open_role="recipient", observations=True),
        stream_slot_options=True,
        **india_clients(),
    )

    (result,) = results
    assert result.status == "definite"
    assert not isinstance(result.observations_relaunch, candidate_collapse.CompositeRelaunch)
    assert result.observations_task is not None


def test_each_candidate_is_minted_under_the_request_s_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate's composite is minted under the request's catalogue
    (`reading.decide_read_scope`), as the definite answer's is, so a source scope its own
    reads ran under narrows none of its figures."""
    scopes: list[frozenset[str] | None] = []
    mint = resubmit.composite_mint

    def recording(identity: BaseIdentity, **kwargs: Any) -> Any:
        scopes.append(current_catalogue().scope)
        return mint(identity, **kwargs)

    monkeypatch.setattr(resubmit, "composite_mint", recording)
    decide = candidate_grounding.decide_read_scope

    def narrowed(*args: Any, **kwargs: Any) -> Any:
        with scoped_reads(frozenset({_DAC2A})):
            return decide(*args, **kwargs)

    monkeypatch.setattr(candidate_grounding, "decide_read_scope", narrowed)
    states: list[ResolveState] = []
    assemble = candidate_collapse._assemble_candidate_set

    async def capture(state: Any, specs: list[Any], grounded: Any, truncated: bool) -> Any:
        states.append(state)
        return await assemble(state, specs, grounded, truncated)

    monkeypatch.setattr(candidate_collapse, "_assemble_candidate_set", capture)

    _candidates()

    (state,) = states
    assert state.request_catalogue is not None
    assert scopes == [state.request_catalogue.scope] * 2


def test_a_candidate_repeating_an_earlier_one_mints_no_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate whose identity an earlier one holds is skipped before its scope is
    decided (`candidate_grounding.ground_candidates`), so its composite is not minted again
    and the set keeps one Spec per reading."""
    minted: list[str] = []
    mint = resubmit.composite_mint

    def recording(identity: BaseIdentity, **kwargs: Any) -> Any:
        minted.append(identity_spec_id(identity))
        return mint(identity, **kwargs)

    monkeypatch.setattr(resubmit, "composite_mint", recording)
    ground = candidate_collapse.ground_candidates

    def repeated(state: Any, candidates: list[Any]) -> Any:
        return ground(state, [*candidates, candidates[-1]])

    monkeypatch.setattr(candidate_collapse, "ground_candidates", repeated)

    part = _candidates()

    assert len(part.candidates.specs) == 2
    assert len(minted) == len(set(minted)) == 2


def test_a_refusal_showing_a_composite_candidate_carries_its_mint_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate a refusal shows as its flagged match hands the composite it is minted as to
    the refusal (`candidate_collapse._shown`), whose whole-total warnings read the identity
    its rows were read for, and leaves the part's own answer's composite
    (`ResolveState.minted`) unchanged."""
    seen: list[tuple[Any, Any, list[Any]]] = []
    assemble = candidate_collapse._assemble_candidate_set

    async def capture(state: Any, specs: list[Any], grounded: Any, truncated: bool) -> Any:
        seen.append((state, grounded, specs))
        return await assemble(state, specs, grounded, truncated)

    monkeypatch.setattr(candidate_collapse, "_assemble_candidate_set", capture)
    _candidates()
    ((state, grounded, specs),) = seen
    before = state.minted
    spec = specs[0]
    minted = grounded[spec.spec_id].minted
    assert minted is not None

    matched = candidate_collapse._shown(state, grounded, spec)

    assert state.minted is before
    assert matched == (minted,)
    assert state.minted_as(spec, matched) is minted


def test_a_refusal_offering_a_composite_alternate_carries_its_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal whose first nominated reading rides as its alternate match hands the
    composite that reading is minted as to the refusal (`ResolveState.no_data` `matched`),
    as a flagged match does, so the alternate's whole-total warnings read the identity its
    rows were read for."""
    seen: list[tuple[Any, Any, list[Any]]] = []
    assemble = candidate_collapse._assemble_candidate_set

    async def capture(state: Any, specs: list[Any], grounded: Any, truncated: bool) -> Any:
        seen.append((state, grounded, specs))
        return await assemble(state, specs, grounded, truncated)

    monkeypatch.setattr(candidate_collapse, "_assemble_candidate_set", capture)
    _candidates()
    ((state, grounded, specs),) = seen
    minted = grounded[specs[0].spec_id].minted
    assert minted is not None
    # The donor and recipient readings stand in for no elected shape and break no family down.
    assert {grounded[spec.spec_id].candidate.basis for spec in specs}.isdisjoint(
        {"structural", "breakdown"}
    )
    refusals: list[dict[str, Any]] = []
    no_data = ResolveState.no_data

    def recorded(self: ResolveState, reason: Any, **kwargs: Any) -> RegionResult:
        refusals.append(kwargs)
        return no_data(self, reason, **kwargs)

    monkeypatch.setattr(ResolveState, "no_data", recorded)
    # The elected shape holds no data and no stand-in answers for it.
    state.elected_has_data = False

    refused = candidate_collapse._handle_sparse_survivors(state, list(specs), grounded, None)

    assert isinstance(refused, RegionResult)
    assert refused.alternate_match is specs[0]
    (kwargs,) = refusals
    assert kwargs.get("matched") == (minted,)


def test_each_candidate_carries_its_own_withheld_reading_warning() -> None:
    """Two candidates whose readings the bilateral default cannot reach for the same reason
    each carry the warning keyed to their own spec_id, though the messages match."""
    specs = [SimpleNamespace(spec_id=spec_id) for spec_id in ("spec_a", "spec_b")]
    grounded = {
        spec.spec_id: SimpleNamespace(
            decision=SimpleNamespace(
                warning=_withheld_warning(spec.spec_id, Withheld(why="option_not_covered"))
            )
        )
        for spec in specs
    }

    found = candidate_collapse._decision_warnings((), cast(Any, grounded), cast(Any, specs))

    assert [(w.spec_id, w.message) for w in found] == [
        ("spec_a", found[0].message),
        ("spec_b", found[0].message),
    ]


def test_a_refusal_s_composite_readings_leave_the_pipeline_as_minted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal whose flagged match and nearest real suggestion were minted as the bilateral
    composite at their own builds leaves the part's scope step unchanged. Each reading was
    elected at its read, so none is read again."""
    _patched_coverage(monkeypatch, _FRANCE_HELD)
    composite = _resolve_resubmit(_identity(_BILATERAL)).spec
    near = NearestRealSpec.model_validate(
        {
            **composite.model_dump(),
            "relaxation": Relaxation(
                axis="what", property_dcid="measurementDenominator", dropped_value_dcids=["x"]
            ),
        }
    )
    refused = RegionResult(
        variable_text=_VARIABLE,
        status="no_data",
        specs=(),
        no_data_reason="no_observations",
        warnings=(),
        timing_by_step={},
        flagged_match=(composite,),
        nearest_real=(near,),
    )

    stated = pipeline._with_scope_choices(refused)

    assert stated is refused
