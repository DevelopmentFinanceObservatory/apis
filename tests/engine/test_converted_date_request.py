"""A conversion that omits rows settles the window's opt-in again (`facets.plan_answer`), and
every door takes the settled request back, so the identity states the opt-in of the answer
it serves: a fresh request (`phase_answer._install_filtered_answer`), a resubmit
(`promote_ground._convert_narrowed_answer`) and each candidate
(`candidate_grounding._prepare_candidate_context`)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from search.engine.extract import DateRequest
from search.engine.region import candidate_grounding, phase_answer, promote_ground
from search.engine.region.candidate_grounding import _CandidateConversion
from search.engine.region.candidates import _Candidate
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.facets import FilteredAnswer
from search.engine.region.promote_state import _PromoteState
from search.engine.retrieval.read_plan import ReadPlan
from search.engine.retrieval.retrieve import Materialised
from search.models import CoverageBare, ResubmitIdentityPayload
from search.models.grammar import TimeWindow
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine.test_currency_unmapped_omitted_gate import _fresh_state

_HELD = frozenset({"partial"})
_OPTED_IN = DateRequest(
    window=TimeWindow(start_year=2020, end_year=2025, include=["partial"]),
    latest=False,
    reads_held=_HELD,
)
# The converted answer's plan holds no partial run, so its window settles none.
_SETTLED = DateRequest(
    window=TimeWindow(start_year=2020, end_year=2025), latest=False, reads_held=_HELD
)


def _mat() -> Materialised:
    return Materialised(sv_dcids=["sv"], has_data=True, coverage=CoverageBare(has_data=True))


def _answer(date_request: DateRequest, plan: ReadPlan | None = None) -> FilteredAnswer:
    return FilteredAnswer(CoverageBare(has_data=True), None, plan, date_request)


def test_a_fresh_request_states_the_converted_answers_request() -> None:
    state = _fresh_state()
    state.date_request = _OPTED_IN

    phase_answer._install_filtered_answer(state, _mat(), _answer(_SETTLED))

    assert state.date_request == _SETTLED


def test_a_resubmit_states_the_converted_answers_request(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _PromoteState(
        inp=cast(ResubmitIdentityPayload, None),
        graph=cast(Any, None),
        sdmx=cast(Any, None),
        deadline=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        materialise_graph=cast(Any, object()),
        materialise_sdmx=cast(Any, object()),
        coverage=empty_coverage_snapshot(),
    )
    state.date_request = _OPTED_IN
    state.narrowed_coverage = CoverageBare(has_data=True)
    state.narrowed_plan = cast(ReadPlan, object())
    converted = _answer(_SETTLED, plan=cast(ReadPlan, object()))
    monkeypatch.setattr(
        promote_ground, "convertible_answer", lambda *_a, **_k: (converted, UnitCoverage())
    )

    promote_ground._convert_narrowed_answer(state, _mat())

    assert state.date_request == _SETTLED
    assert state.narrowed_plan is converted.plan


def test_each_candidate_states_its_converted_answers_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _fresh_state()
    state.date_request = _OPTED_IN
    state.bound_unit_outcome = cast(Any, object())
    monkeypatch.setattr(
        candidate_grounding,
        "_candidate_conversions",
        lambda _state, candidates, *_rest: [
            _CandidateConversion(_answer(_SETTLED), UnitCoverage(), None) for _ in candidates
        ],
    )
    monkeypatch.setattr(candidate_grounding, "fetch_scope_members", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(
        candidate_grounding, "_candidate_observation_entities", lambda _cand: ({}, None)
    )
    candidate = _Candidate(
        cast(Any, None), [], _mat(), cast(Any, None), None, _OPTED_IN, reading=None
    )

    prep = candidate_grounding._prepare_candidate_context(state, [candidate])

    assert [cand.date_request for cand in prep.candidates] == [_SETTLED]
