"""The candidates `answer` timer starts after unit election and scope-member fetch."""

from __future__ import annotations

import search.engine.region.candidate_collapse as candidate_collapse_module
import search.engine.region.candidate_grounding as candidate_grounding_module
from tests.engine.test_candidates_unit_election_e2e import _run

# Advance the shared fake clock only after candidate preparation returns.
_PREP_DURATION_MS = 300


def test_answer_step_excludes_prepare_candidate_context_time(monkeypatch):
    """The 300 ms fake-clock jump occurs during preparation, so the answer timer must read 0."""
    clock = [0]

    def _fake_now_ms() -> int:
        return clock[0]

    real_prepare = candidate_grounding_module._prepare_candidate_context

    def _prepare_and_advance_clock(state, candidate_triples):
        result = real_prepare(state, candidate_triples)
        clock[0] += _PREP_DURATION_MS
        return result

    monkeypatch.setattr(candidate_grounding_module, "now_ms", _fake_now_ms)
    monkeypatch.setattr(candidate_collapse_module, "now_ms", _fake_now_ms)
    monkeypatch.setattr(
        candidate_grounding_module, "_prepare_candidate_context", _prepare_and_advance_clock
    )

    result = _run()

    assert result.status == "candidates", result
    assert result.timing_by_step["answer"] == 0, result.timing_by_step
