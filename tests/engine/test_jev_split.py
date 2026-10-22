"""`jev_split.split_questions`: a request over the budget splits into chunks that each fit
with the state and together hold every question once, in the original order."""

from __future__ import annotations

import pytest

from search.engine.binders.jev_split import (
    SPLIT_BUDGET,
    RequestTooLarge,
    estimate_tokens,
    split_questions,
)

_STATE = {"query": "life expectancy at birth in Kenya since 2000", "menu": ["a"] * 200}


def _questions(n: int, chars: int) -> dict[str, dict[str, str]]:
    return {f"m|{i}": {"type": "noul", "instructions": f"{i} " + "x" * chars} for i in range(n)}


def _estimate(questions: dict[str, dict[str, str]]) -> int:
    return estimate_tokens({"state": _STATE, "questions": questions})


def test_request_within_the_budget_stays_whole() -> None:
    questions = _questions(5, 100)
    assert split_questions(_STATE, questions) == [questions]


def test_sixty_thousand_tokens_split_into_chunks_under_the_budget() -> None:
    # Uneven sizes, so balancing by size and the restored order both matter.
    questions = {
        f"m|{i}": {"type": "noul", "instructions": "x" * (1_500 + 3_000 * (i % 3))}
        for i in range(48)
    }
    assert _estimate(questions) >= 60_000

    chunks = split_questions(_STATE, questions)

    assert len(chunks) > 1
    assert all(_estimate(chunk) <= SPLIT_BUDGET for chunk in chunks)
    held = [qid for chunk in chunks for qid in chunk]
    assert sorted(held) == sorted(questions)
    assert len(held) == len(set(held))
    order = list(questions)
    assert all(list(chunk) == [q for q in order if q in chunk] for chunk in chunks)


@pytest.mark.parametrize(
    ("state", "questions"),
    [
        pytest.param({"query": "x" * 90_000}, _questions(1, 10), id="state-over-budget"),
        pytest.param(
            _STATE, {"big": {"type": "noul", "instructions": "x" * 90_000}}, id="question"
        ),
    ],
)
def test_oversized_state_or_question_raises(state: dict, questions: dict) -> None:
    with pytest.raises(RequestTooLarge, match="request budget"):
        split_questions(state, questions)
