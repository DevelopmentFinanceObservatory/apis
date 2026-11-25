"""Tests for the coverage rule's read of variable words that restate a bound party.

`phase_election.read_restated_entity_words` asks the resolver whether a word the coverage
rule would leave uncovered restates an entity the part binds, and records the answer for the
part (`ResolveState.restatement_reads`). The elected answer's judgement (`judge_coverage`)
and the candidates' (`judge_answer_coverage`) both read through that record, so each text
is asked once and has one answer for every judgement of the part. The coverage rule's
readers are replaced here by one that leaves a word uncovered on the answers named in
`leaves`, unless the part reads it as an entity.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from entity_resolution import Entity, EntityResolutionError, NotFound, Resolved
from search.engine.region import phase_election
from search.engine.region.phase_rank_bind import _CoverageReading
from search.models.trace import ResolvedQualifier

_USA = Entity(dcid="country/USA", name="United States", kind="country", graph_types=("Country",))
_GBR = Entity(dcid="country/GBR", name="United Kingdom", kind="country", graph_types=("Country",))


class _Resolver:
    """Resolve "US" to the United States, "UK" to the United Kingdom and every other text to
    nothing, failing the first `fails` calls, and keep each call's texts."""

    def __init__(self, fails: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.fails = fails

    async def resolve(self, texts, **kwargs):
        self.calls.append(list(texts))
        if len(self.calls) <= self.fails:
            raise EntityResolutionError("resolver down")
        named = {"US": _USA, "UK": _GBR}
        return [Resolved(entity=named[t]) if t in named else NotFound() for t in texts]


@dataclass
class _State:
    """The part's state as the coverage read uses it, binding the United States."""

    resolver: _Resolver
    shape_draft: str = "elected"
    deadline: float | None = None
    restatement_reads: dict[str, str] = field(default_factory=dict)
    rcl: Any = field(
        default_factory=lambda: type(
            "Recall", (), {"resolutions": {"United States": Resolved(entity=_USA)}}
        )
    )

    @property
    def restated_entity_words(self) -> tuple[str, ...]:
        return tuple(text for text, read in self.restatement_reads.items() if read == "restated")


@dataclass
class _Coverage:
    """The word the stand-in coverage rule leaves uncovered, and the answers it leaves it on."""

    word: str = "US"
    leaves: set[str] = field(default_factory=lambda: {"candidate-a", "candidate-b"})
    lost: list[str] = field(default_factory=list)

    def records(self, view: _State) -> list[ResolvedQualifier]:
        if view.shape_draft not in self.leaves or self.word in view.restated_entity_words:
            return []
        return [
            ResolvedQualifier(text=self.word, named=[], disposition="unbound", evidence="uncovered")
        ]


@pytest.fixture
def coverage(monkeypatch: pytest.MonkeyPatch) -> _Coverage:
    world = _Coverage()

    async def unrated(state, view) -> dict[str, float]:
        return {}

    monkeypatch.setattr(
        phase_election,
        "_coverage_reading",
        lambda view, terms: _CoverageReading(records=world.records(view), verdicts=()),
    )
    monkeypatch.setattr(phase_election, "_coverage_records", lambda view, t: world.records(view))
    monkeypatch.setattr(phase_election, "_query_records", lambda view: ())
    monkeypatch.setattr(phase_election, "_rated_words", unrated)
    monkeypatch.setattr(phase_election, "record_coverage", lambda state, records: None)
    monkeypatch.setattr(phase_election, "note_lost_read", lambda read, exc: world.lost.append(read))
    monkeypatch.setattr(phase_election, "answer_view", _view)
    return world


def _view(state: _State, shape: str, bindings: object, read: object) -> _State:
    """A candidate's view of `state`, sharing its record as `phase_election.answer_view` does."""
    view = copy.copy(state)
    view.shape_draft = shape
    return view


def _judge(state: _State, shapes: tuple[str, ...]) -> tuple:
    """`judge_answer_coverage` on a candidate named by each of `shapes`."""
    answers = [(cast(Any, shape), [], cast(Any, None)) for shape in shapes]
    return asyncio.run(phase_election.judge_answer_coverage(cast(Any, state), answers))


def _read_elected(state: _State) -> None:
    """The elected answer's read, as `judge_coverage` makes it."""
    asyncio.run(
        phase_election.read_restated_entity_words(cast(Any, state), [(cast(Any, state), ())])
    )


def test_a_word_only_candidates_leave_uncovered_restates_the_party_on_each(coverage: _Coverage):
    """The elected answer covered "US", so its read asked nothing. Two candidates leave it
    uncovered: one resolver call reads it as the bound United States, and no candidate
    holds a record for it."""
    resolver = _Resolver()
    state = _State(resolver)

    _read_elected(state)
    records = _judge(state, ("candidate-a", "candidate-b", "covers-it"))

    assert records == ((), (), ())
    assert resolver.calls == [["US"]]
    assert state.restatement_reads == {"US": "restated"}


def test_a_failed_read_holds_for_every_judgement_of_the_part(coverage: _Coverage):
    """The elected answer's read of "US" fails and a later one would succeed. The failure is
    the text's one answer: the candidates do not ask again, and the word stays uncovered on
    each of them as on the elected answer."""
    coverage.leaves = {"elected", "candidate-a"}
    resolver = _Resolver(fails=1)
    state = _State(resolver)

    _read_elected(state)
    records = _judge(state, ("candidate-a",))

    assert resolver.calls == [["US"]]
    assert state.restatement_reads == {"US": "failed"}
    assert [[record.text for record in held] for held in records] == [["US"]]
    assert coverage.lost == ["restated entity read"]


def test_a_text_naming_no_bound_party_is_asked_once(coverage: _Coverage):
    """ "UK" names the United Kingdom, which the part does not bind: the elected answer's
    read records it unbound, and the candidates leaving it uncovered do not ask again."""
    coverage.word = "UK"
    coverage.leaves = {"elected", "candidate-a", "candidate-b"}
    resolver = _Resolver()
    state = _State(resolver)

    _read_elected(state)
    records = _judge(state, ("candidate-a", "candidate-b"))

    assert resolver.calls == [["UK"]]
    assert state.restatement_reads == {"UK": "unbound"}
    assert [[record.text for record in held] for held in records] == [["UK"], ["UK"]]


def test_a_read_past_the_parts_deadline_raises(coverage: _Coverage):
    """A read once the part's deadline has passed raises, as the part's other reads do, and
    records and refuses nothing."""
    coverage.leaves = {"elected"}
    state = _State(_Resolver(), deadline=time.monotonic() - 1)

    with pytest.raises(TimeoutError):
        _read_elected(state)

    assert state.restatement_reads == {}
    assert coverage.lost == []


def test_candidates_leaving_nothing_uncovered_ask_the_resolver_nothing(coverage: _Coverage):
    resolver = _Resolver()

    assert _judge(_State(resolver), ("covers-it",)) == ((),)
    assert resolver.calls == []
