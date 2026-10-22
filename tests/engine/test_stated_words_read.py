"""The stated-words read: its prompt, its fixed schema, its answers as ratings, Gemini's call
of it, and a part that refuses or answers on its answer."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.binders import GeminiBinder, StatedWordsRequest
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import LLMInfraError
from search.engine.stated_words_read import (
    STATED_WORDS_SYSTEM_PROMPT,
    StatedWord,
    StatedWordsOutput,
    build_stated_words_prompt,
    word_ratings,
)
from tests.engine._harness import make_request, offline_resolve
from tests.engine.test_bind import _shape
from tests.engine.test_qualifier_guard import _READER_PERSONS, _run, _state
from tests.fixtures import FakeLLM

_PERSONS = replace(
    _shape("Total population"),
    pop_type_dcid="Person",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
)


def test_the_prompt_states_the_series_then_numbers_the_words():
    """`StatedWord.index` indexes the WORDS list, in the order the request holds them."""
    assert build_stated_words_prompt(
        "stateless refugee population", _PERSONS, ("stateless", "refugee")
    ) == (
        "Variable phrase: stateless refugee population\n"
        "\n"
        "SERIES:\n"
        "  label: Total population\n"
        "  family: standard Data Commons\n"
        "  five_tuple: populationType=Person, measuredProperty=count, statType=measuredValue\n"
        "\n"
        "WORDS (words of the variable phrase):\n"
        "  0: stateless\n"
        "  1: refugee\n"
    )


def test_the_schema_holds_no_menu_and_the_bind_schema_no_rating():
    """The answer is a list of numbered booleans whatever the menu and the word count, so a
    recorded answer is keyed by the prompt alone, and bind's schema keeps its own fields."""
    assert set(StatedWordsOutput.model_fields) == {"words"}
    assert set(StatedWord.model_fields) == {"index", "stated"}
    assert not set(StatedWordsOutput.model_fields) & set(_UniversalBindOutput.model_fields)


@pytest.mark.parametrize(
    ("answers", "ratings"),
    [
        ([(0, False), (1, True)], {"stateless": 0.0, "refugee": 1.0}),
        ([(1, False)], {"refugee": 0.0}),
        ([], {}),
        ([(0, True), (2, False), (-1, False)], {"stateless": 1.0}),
        ([(1, False), (1, True), (0, True)], {"stateless": 1.0}),
        ([(1, False), (1, False)], {"refugee": 0.0}),
    ],
    ids=["both", "missing-index", "none", "index-out-of-range", "answered-both-ways", "repeated"],
)
def test_each_answer_rates_the_word_its_index_names(answers, ratings):
    """Stated reads 1.0 and not stated 0.0. A word with no answer, or answered both ways,
    stays unrated, and an index naming no word rates nothing."""
    output = StatedWordsOutput(words=[StatedWord(index=i, stated=s) for i, s in answers])
    assert word_ratings(output, ("stateless", "refugee")) == ratings


_USAGE = {"input_tokens": 3, "output_tokens": 2}


class _StatedLLM:
    """Answers the stated-words read with `answers`, or raises `fail`, recording each call."""

    def __init__(
        self, answers: list[tuple[int, bool]] | None = None, fail: Exception | None = None
    ) -> None:
        self.answers = answers or []
        self.fail = fail
        self.calls: list[tuple[str, str, type]] = []

    async def generate_structured(self, *, prompt, system, schema):
        self.calls.append((prompt, system, schema))
        if self.fail is not None:
            raise self.fail
        words = [StatedWord(index=i, stated=s) for i, s in self.answers]
        return StatedWordsOutput(words=words), _USAGE


def _request(words: tuple[str, ...]) -> StatedWordsRequest:
    return StatedWordsRequest(
        variable="stateless refugee population",
        shape=_PERSONS,
        shape_index=0,
        words=words,
        decision=None,
    )


def test_gemini_rates_the_words_in_one_call_of_its_own():
    llm = _StatedLLM([(0, True), (1, False)])
    ratings, usage = asyncio.run(
        GeminiBinder().rate_words_stated(_request(("stateless", "refugee")), llm=llm)
    )

    assert ratings == {"stateless": 1.0, "refugee": 0.0}
    assert usage == _USAGE
    assert llm.calls == [
        (
            build_stated_words_prompt(
                "stateless refugee population", _PERSONS, ("stateless", "refugee")
            ),
            STATED_WORDS_SYSTEM_PROMPT,
            StatedWordsOutput,
        )
    ]


@pytest.mark.parametrize(
    ("answers", "fail", "records", "ratings", "usage", "degraded"),
    [
        ([(0, False)], None, ["refugee"], {"refugee": 0.0}, _USAGE, []),
        ([(0, True)], None, [], {"refugee": 1.0}, _USAGE, []),
        ([], None, [], {}, _USAGE, []),
        ([], LLMInfraError("read failed"), [], {}, None, ["stated words read"]),
    ],
    ids=["not-stated", "stated", "unanswered", "failed-read"],
)
def test_binding_repair_reads_gemini_s_rating(
    search_settings, answers, fail, records, ratings, usage, degraded
):
    """Binding repair asks Gemini once for "refugee" on the count of all persons and keeps
    its rating: not stated is a record, stated or unanswered is read. The read's usage
    joins the part's. A failed read raises out of the binder, so it rates nothing and
    `phase_election._rate_unread_words` notes it as lost once."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable="refugee population", shape_draft=_READER_PERSONS, bindings=[], unplaced=[]
    )
    state.capped_shapes = [_READER_PERSONS]
    state.binder = GeminiBinder()
    state.llm = llm = _StatedLLM(answers, fail)
    with recording_degraded_reads() as lost:
        qualifiers = _run(state)

    assert len(llm.calls) == 1
    assert llm.calls[0][0].endswith("  0: refugee\n")
    assert [q.text for q in qualifiers if q.evidence == "uncovered"] == records
    assert state.stated_word_ratings == ratings
    assert state.var_usage == usage
    assert lost.reads == degraded


class _RatedOutLLM(FakeLLM):
    """The recorded bank, with every stated-words read answering each word `stated`."""

    def __init__(self, *, stated: bool) -> None:
        super().__init__()
        self.stated = stated
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            self.prompts.append(prompt)
            count = sum(line.startswith("  ") for line in prompt.split("WORDS")[-1].splitlines())
            words = [StatedWord(index=i, stated=self.stated) for i in range(count)]
            return StatedWordsOutput(words=words), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


@pytest.mark.parametrize(
    ("stated", "status", "unbound"),
    [(False, "no_data", ["participation"]), (True, "definite", None)],
    ids=["rated-not-stated", "rated-stated"],
)
def test_a_gemini_part_refuses_on_a_word_its_series_does_not_state(stated, status, unbound):
    """The part "female labor force participation rate in Bangladesh" elects the percent of
    female working-age population in the labour force, whose members' names leave
    "participation" unread. Gemini's read rating it not stated refuses the part
    ``qualifier_not_bound`` on it, and rating it stated answers."""
    llm = _RatedOutLLM(stated=stated)
    response = offline_resolve(
        make_request("female labor force participation rate in Bangladesh"), llm=llm
    )

    assert len(llm.prompts) == 1
    assert llm.prompts[0].endswith("WORDS (words of the variable phrase):\n  0: participation\n")
    part = response.parts[0]
    assert part.status == status
    if unbound is not None:
        assert part.no_data.reason == "qualifier_not_bound"
        assert [q.text for q in part.no_data.unbound_qualifiers] == unbound
