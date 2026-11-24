"""Tests for how candidate answers read a variable word that restates a bound party.

`phase_election.judge_answer_coverage` judges each candidate the part offers on its own
read. A word one candidate leaves uncovered can be one the elected answer covered through a
code span, so the elected answer's own restatement read never asked about it. The
candidate read asks the resolver about the union of the candidates' uncovered words in one
call. The coverage rule's readers are replaced here by one that leaves "US" uncovered on
the candidates named `leaves`, unless the answer reads it as an entity.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from entity_resolution import Entity, NotFound, Resolved
from search.engine.region import phase_election
from search.engine.region.phase_rank_bind import _CoverageReading
from search.models.trace import ResolvedQualifier

_USA = Entity(dcid="country/USA", name="United States", kind="country", graph_types=("Country",))


class _Resolver:
    """Resolve "US" to the United States and every other text to nothing, keeping each
    call's texts."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def resolve(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [Resolved(entity=_USA) if text == "US" else NotFound() for text in texts]


class _State:
    """The part's state as the candidate coverage read uses it."""

    def __init__(self, resolver: _Resolver) -> None:
        self.rcl = type("Recall", (), {"resolutions": {"United States": Resolved(entity=_USA)}})
        self.resolver = resolver
        self.deadline = None
        self.restated_entity_words: tuple[str, ...] = ()
        self.shape_draft = "elected"


def _uncovered(word: str) -> ResolvedQualifier:
    return ResolvedQualifier(text=word, named=[], disposition="unbound", evidence="uncovered")


@pytest.fixture
def coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    leaves = {"candidate-a", "candidate-b"}

    def reading(view, terms) -> _CoverageReading:
        unread = view.shape_draft in leaves and "US" not in view.restated_entity_words
        return _CoverageReading(records=[_uncovered("US")] if unread else [], verdicts=())

    async def unrated(state, view) -> dict[str, float]:
        return {}

    monkeypatch.setattr(phase_election, "_coverage_reading", reading)
    monkeypatch.setattr(
        phase_election, "_coverage_records", lambda view, terms: reading(view, terms).records
    )
    monkeypatch.setattr(phase_election, "_query_records", lambda view: ())
    monkeypatch.setattr(phase_election, "_rated_words", unrated)
    monkeypatch.setattr(phase_election, "record_coverage", lambda state, records: None)


def _judge(state: _State, shapes: tuple[str, ...]) -> tuple:
    """`judge_answer_coverage` on a candidate named by each of `shapes`, the stand-in state
    and shape names typed as the part's."""
    answers = [(cast(Any, shape), [], cast(Any, None)) for shape in shapes]
    return asyncio.run(phase_election.judge_answer_coverage(cast(Any, state), answers))


def test_a_word_only_candidates_leave_uncovered_restates_the_party_on_each(coverage: None):
    """The elected answer covered "US", so it restated nothing. Two candidates leave it
    uncovered: one resolver call reads it as the bound United States, and no candidate
    holds a record for it."""
    resolver = _Resolver()
    state = _State(resolver)

    records = _judge(state, ("candidate-a", "candidate-b", "covers-it"))

    assert records == ((), (), ())
    assert resolver.calls == [["US"]]
    assert state.restated_entity_words == ("US",)


def test_candidates_leaving_nothing_uncovered_ask_the_resolver_nothing(coverage: None):
    resolver = _Resolver()

    assert _judge(_State(resolver), ("covers-it",)) == ((),)
    assert resolver.calls == []
