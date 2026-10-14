"""Every corpus case resolves offline without raising, and serves what its question pins.

Each case in `corpus/` runs once through `offline_resolve` with the suite's fake graph, the
recorded FakeLLM bank, fixture-backed SDMX and empty materialise doubles. A refusal
(`EngineInputError`) is an answer and passes. Any other exception fails, and the suite's miss
guard fails a case that missed a recorded LLM fixture. The test scores nothing. The eval gate
scores answers.

Every definite part of an answer then holds the pinned-part invariant
(`pinned_part_violations`): it departs no code the question names on a property its Spec's
slots hold, its elected shape's class honours every form its frame names, and its resolved
denominator matches the one the question names. Every candidate of a candidates part holds
the form and denominator clauses, so a candidate set never offers a reading the gate
refuses. A part is matched to the region pass that produced it by `spec_id`, through a spy
on `pipeline._run_phases`.

Each case's stream keeps the sentence contract: at most one sentence event per part, after
every shape event of that part, keyed to the spec its response part states
(`sentence_events.response_sentence_events`).

The module keeps every corpus case runnable offline, and it is how
`scripts/harvest_llm_fixtures.py` reaches the corpus's LLM fixture keys. The harvest's pytest
pass collects it with the rest of the suite and prunes the bank to the keys that pass reached.
It therefore carries no skip or deselecting marker, so a run of the suite always runs every case.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from materialise.rate_tables import RateTableCache
from search.engine.errors import EngineInputError
from search.engine.events import Event, ResponseEvent, SentenceEvent, ShapeEvent
from search.engine.quantity import (
    NamedDenominator,
    QuantityRequirement,
    check_denominator,
    serves_every_form,
)
from search.engine.region import pipeline
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from search.eval.cases import Case, load_cases
from search.eval.runner import build_request
from search.models import (
    CandidatesPart,
    DefinitePart,
    Departure,
    GraphRef,
    NoDataPart,
    Part,
    Resolution,
    ResolveRequest,
    Spec,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import build_materialise_double, offline_events

_CASES = load_cases()

type Pass = tuple[ResolveState, RegionResult]


def _answer_spec_id(part: Part) -> str | None:
    """The spec_id of the Spec a part answers with: a definite part's Spec, a candidates
    part's first candidate, or a no-data part's flagged match."""
    if isinstance(part, DefinitePart):
        return part.spec.spec_id
    if isinstance(part, CandidatesPart):
        return part.candidates.specs[0].spec_id
    if isinstance(part, NoDataPart) and part.no_data.flagged_match is not None:
        return part.no_data.flagged_match.spec_id
    return None


def _sentence_violations(events: list[Event]) -> list[str]:
    """Where the stream breaks the sentence contract."""
    (response,) = [event.response for event in events if isinstance(event, ResponseEvent)]
    if not isinstance(response, Resolution):
        return []
    violations: list[str] = []
    sentences: dict[int, int] = {}
    answered = {i for i, part in enumerate(response.parts) if _answer_spec_id(part) is not None}
    for position, event in enumerate(events):
        if isinstance(event, SentenceEvent):
            index = event.variable_index
            if index in sentences:
                violations.append(f"part {index}: a second sentence")
            sentences[index] = position
            if index >= len(response.parts):
                violations.append(f"part {index}: a sentence for no part")
            elif event.spec_id != _answer_spec_id(response.parts[index]):
                violations.append(f"part {index}: sentence spec_id {event.spec_id}")
        elif isinstance(event, ShapeEvent) and event.variable_index in sentences:
            violations.append(f"part {event.variable_index}: a shape after its sentence")
    violations.extend(f"part {i}: no sentence" for i in sorted(answered - sentences.keys()))
    return violations


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
def test_corpus_case_resolves_offline(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = _resolve(case, monkeypatch)
    if resolved is None:
        return
    response, passes, events = resolved
    assert _sentence_violations(events) == []
    violations = pinned_part_violations(
        response, passes, posted_identity=case.door == "dataset_resubmit"
    )
    assert not violations, "\n".join(violations)


def _break_form(part: DefinitePart, state: ResolveState) -> DefinitePart:
    state.quantity_requirement = QuantityRequirement(named=frozenset({"per_area"}))
    return part


def _break_denominator(part: DefinitePart, state: ResolveState) -> DefinitePart:
    state.named_denominator = NamedDenominator(
        GraphRef(ref="Amount_EconomicActivity_GrossNationalIncome", label="GNI"),
        base_words="GNI",
    )
    return part


def _break_pinned_code(part: DefinitePart, state: ResolveState) -> DefinitePart:
    dropped = Departure(
        code="measure_not_applied",
        binding_ref="how:transferType",
        asked="personal transfers",
        text="The answer does not apply personal transfers.",
    )
    return part.model_copy(update={"departures": [*part.departures, dropped]})


@pytest.mark.parametrize(
    ("breach", "violation"),
    [
        (_break_form, "does not honour every form of ['per_area']"),
        (_break_denominator, "the question names Amount_EconomicActivity_GrossNationalIncome"),
        (_break_pinned_code, "departs pinned code 'personal transfers' on how:transferType"),
    ],
    ids=["form", "denominator", "pinned_code"],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the recorded bind answer for 'remittances as percentage of GDP' nominates the "
        "SDG remittances share of "
        "GDP, a duplicate of the elected series under another name, and the question picks "
        "neither by its words, so the part answers candidates; the parent's answer, "
        "nominating no electable shape, answers definite on this engine"
    ),
)
def test_the_invariant_fails_a_definite_part_that_breaks_a_pinned_part(
    breach: Callable[[DefinitePart, ResolveState], DefinitePart],
    violation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """remittances-pct-gdp-nigeria answers definite with a share of GDP over a transferType
    slot, and holds the invariant. The same answer breaks it once its frame names a form a
    share of GDP does not honour, once the question names GNI as the denominator, and once
    it departs a code on its transferType slot."""
    (case,) = [case for case in _CASES if case.id == "remittances-pct-gdp-nigeria"]
    resolved = _resolve(case, monkeypatch)
    assert resolved is not None
    response, passes, _events = resolved
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert pinned_part_violations(response, passes) == []
    (state,) = [state for state, result in passes if result.status == "definite"]

    broken = response.model_copy(update={"parts": [breach(part, state)]})

    (message,) = pinned_part_violations(broken, passes)
    assert violation in message


def test_the_invariant_fails_a_candidate_that_breaks_a_pinned_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sdg-06 answers with two candidates and names no form, so it holds the invariant.
    Once its frame names a per-area form neither candidate honours, each candidate breaks
    it."""
    (case,) = [case for case in _CASES if case.id == "sdg-06"]
    resolved = _resolve(case, monkeypatch)
    assert resolved is not None
    response, passes, _events = resolved
    (part,) = response.parts
    assert isinstance(part, CandidatesPart)
    assert pinned_part_violations(response, passes) == []
    (state,) = [state for state, result in passes if result.status == "candidates"]

    state.quantity_requirement = QuantityRequirement(named=frozenset({"per_area"}))

    messages = pinned_part_violations(response, passes)
    assert len(messages) == len(part.candidates.specs)
    assert all("does not honour every form of ['per_area']" in m for m in messages)


def _resolve(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> tuple[Resolution, list[Pass], list[Event]] | None:
    """Resolve `case` offline, with its region passes recorded (`_record_passes`) and the
    events it streams. `None` when the answer is a refusal (`EngineInputError`) or answers
    no resolution."""
    request = ResolveRequest.model_validate(build_request(case))
    materialise_graph, materialise_sdmx, _requests = build_materialise_double({})
    passes = _record_passes(monkeypatch)
    try:
        events = offline_events(
            request,
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
    except EngineInputError:
        # A refusal is an answer.
        return None
    (response,) = [event.response for event in events if isinstance(event, ResponseEvent)]
    return (response, passes, events) if isinstance(response, Resolution) else None


def _record_passes(monkeypatch: pytest.MonkeyPatch) -> list[Pass]:
    """Record each region pass's final state and result for the rest of the test."""
    passes: list[Pass] = []
    run_phases = pipeline._run_phases

    async def _recording(state: ResolveState) -> RegionResult:
        result = await run_phases(state)
        passes.append((state, result))
        return result

    monkeypatch.setattr(pipeline, "_run_phases", _recording)
    return passes


def pinned_part_violations(
    response: Resolution, passes: list[Pass], *, posted_identity: bool = False
) -> list[str]:
    """One message per pinned part a definite part, or a candidate of a candidates part,
    of `response` does not serve.

    A definite part departs a code the question names (`measure_not_applied` on
    `how:<property>`) on a property its Spec's slots hold. Or the pass that produced a
    definite Spec or a candidate (`passes`, by `spec_id`) names a form its shape's class
    does not honour (`quantity.serves_every_form`), or a denominator the shape's resolved
    denominator does not match (`quantity.check_denominator`). A text-door Spec no pass
    produced is reported. A part of a `posted_identity` request that no pass produced
    answers the identity alone, which names no form or denominator, so only its departures
    are read."""
    messages: list[str] = []
    for index, part in enumerate(response.parts):
        if isinstance(part, DefinitePart):
            spec = part.spec
            held = {
                slot.key.property.ref
                for slot in spec.resolved.slots
                if slot.key.property is not None
            }
            messages.extend(
                f"part {index}: departs pinned code {departure.asked!r} on {departure.binding_ref}"
                for departure in part.departures
                if departure.code == "measure_not_applied"
                and (departure.binding_ref or "").removeprefix("how:") in held
            )
            messages.extend(_quantity_violations(index, spec, "definite", passes, posted_identity))
        elif isinstance(part, CandidatesPart):
            for spec in part.candidates.specs:
                messages.extend(
                    _quantity_violations(index, spec, "candidates", passes, posted_identity)
                )
    return messages


def _quantity_violations(
    index: int, spec: Spec, status: str, passes: list[Pass], posted_identity: bool
) -> list[str]:
    """The form and denominator clauses of `pinned_part_violations` for `spec`, judged in
    each pass that produced it with `status`."""
    messages: list[str] = []
    produced = [
        state
        for state, result in passes
        if result.status == status and spec.spec_id in {s.spec_id for s in result.specs}
    ]
    if not produced and not posted_identity:
        messages.append(f"part {index}: no region pass produced {spec.spec_id}")
    for state in produced:
        shape = _elected_shape(state, spec)
        if shape is None:
            messages.append(f"part {index}: no shape of its pass reads as {spec.spec_id}")
            continue
        requirement = state.quantity_requirement
        if not serves_every_form(requirement, shape, state.denominator_kinds):
            messages.append(
                f"part {index}: {shape.shape_id} does not honour every form of "
                f"{sorted(requirement.named)}"
            )
        named = state.named_denominator
        if named is not None and check_denominator(shape, named) == "refused":
            asked = named.ref.ref if named.ref is not None else "none of these"
            messages.append(
                f"part {index}: {shape.shape_id} divides by {shape.meas_denom_dcid}, "
                f"the question names {asked}"
            )
    return messages


def _elected_shape(state: ResolveState, spec: Spec) -> ShapeDraft | None:
    """The shape of the pass whose five-tuple `spec` reads: the elected shape first, then
    the shapes the pass capped and admitted, one of which a candidate survivor is."""
    shape = spec.resolved.shape

    def dcid(ref: GraphRef | None) -> str:
        return ref.ref if ref is not None else ""

    wanted = (
        dcid(shape.population_type),
        dcid(shape.measured_property),
        dcid(shape.stat_type),
        dcid(shape.measurement_qualifier),
        dcid(shape.measurement_denominator),
        shape.measurement_denominator_unresolved,
    )
    for draft in (state.shape_draft, *state.capped_shapes, *state.eligible_shapes):
        if draft is not None and wanted == (
            draft.pop_type_dcid,
            draft.meas_prop_dcid,
            draft.stat_type_dcid,
            draft.meas_qual_dcid or "",
            draft.meas_denom_dcid or "",
            draft.meas_denom_unresolved,
        ):
            return draft
    return None
