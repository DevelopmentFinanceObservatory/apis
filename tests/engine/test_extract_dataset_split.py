"""Tests for what extraction does with a request that names a dataset.

`extract` reads each mention the gate finds (`dataset_mention.read_mentions`) and
`_apply_mention_readings` turns the readings into variables, the mention each variable
binds, and dropped qualifiers. A request naming no dataset makes one model call and is
left as extracted.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from code_index.shared import get_index
from search.engine.dataset_mention import (
    MentionReading,
    MentionReadings,
    ReadMention,
    gate_mentions,
)
from search.engine.errors import LLMInfraError
from search.engine.extract import (
    ExtractedOperation,
    Extraction,
    _apply_mention_readings,
    _system_instruction,
    extract,
)
from search.eval.cases import load_cases
from tests import fixtures
from tests.engine.test_aid_dataset_bind_e2e import _ACCEPTANCE, _PLUS
from tests.fixtures import FakeLLM
from tests.fixtures.keys import fixture_key

_INDEX = get_index()
_E = "education aid"
_IM = "imputed multilateral"
_IM_E = "imputed multilateral education aid"


def _rank(variable: int) -> ExtractedOperation:
    return ExtractedOperation(verb="rank", variable=variable, phrase="top 5 donors", over="origin")


def _apply(
    variables: list[str],
    relations: list[tuple[str, int | None]],
    *,
    dropped: tuple[str, ...] = (),
    operations: tuple[ExtractedOperation, ...] = (),
    question: str = "",
) -> tuple[Extraction, dict[str, str]]:
    """Apply one relation to each gated mention of the extraction, in gate order."""
    gated = gate_mentions(question, variables, dropped, _INDEX)
    assert len(gated) == len(relations), [g.mention.text for g in gated]
    readings = [
        ReadMention(g, relation, variable)  # ty: ignore[invalid-argument-type]
        for g, (relation, variable) in zip(gated, relations, strict=True)
    ]
    return _apply_mention_readings(
        Extraction(
            kind="resolution",
            variables=variables,
            dropped_qualifiers=list(dropped),
            operations=list(operations),
        ),
        readings,
    )


# ---------------------------------------------------------------------------
# What the readings write into the extraction
# ---------------------------------------------------------------------------


def test_the_bound_mention_is_handed_to_its_variable_as_a_phrase():
    extraction, phrases = _apply([_E, _IM_E], [("only", 1)])

    assert extraction.variables == [_E, _IM_E]
    assert phrases == {_IM_E: _IM}
    assert extraction.dropped_qualifiers == []


def test_the_added_variable_carries_the_operations_on_its_measure():
    extraction, phrases = _apply(
        [_E], [("also", 0)], dropped=(f"including {_IM}",), operations=(_rank(0),)
    )

    assert extraction.variables == [_E, _IM_E]
    assert phrases == {_IM_E: _IM}
    assert extraction.dropped_qualifiers == []
    assert extraction.operations == [_rank(0), _rank(1)]


@pytest.mark.parametrize(
    "variables",
    [[_E, f"{_IM} aid"], [_E, _IM_E]],
    ids=["name-alone", "own-variable"],
)
def test_the_datasets_own_variable_takes_the_operations_the_model_left_on_the_measure(variables):
    extraction, _phrases = _apply(variables, [("also", 0)], operations=(_rank(0),))

    assert extraction.variables == [_E, _IM_E]
    assert extraction.operations == [_rank(0), _rank(1)]


def test_an_operation_the_model_put_on_both_variables_is_not_repeated():
    extraction, _phrases = _apply(
        [_E, f"{_IM} aid"], [("also", 0)], operations=(_rank(0), _rank(1))
    )

    assert extraction.operations == [_rank(0), _rank(1)]


def test_the_measure_added_beside_a_collapsed_variable_carries_its_operations():
    extraction, phrases = _apply([_IM_E], [("also", 0)], operations=(_rank(0),))

    assert extraction.variables == [_IM_E, _E]
    assert phrases == {_IM_E: _IM}
    assert extraction.operations == [_rank(0), _rank(1)]


def test_a_removed_variable_takes_its_operations_with_it_and_later_ones_follow_their_variable():
    extraction, _phrases = _apply(
        [f"{_IM} aid", _E], [("excluded", None)], operations=(_rank(0), _rank(1))
    )

    assert extraction.variables == [_E]
    assert extraction.operations == [_rank(0)]


def test_a_dropped_phrase_goes_when_every_mention_in_it_took_effect():
    excluded, _phrases = _apply([_E], [("excluded", None)], dropped=(f"excluding {_IM}",))
    included, _phrases = _apply([_E], [("also", 0)], dropped=(f"including {_IM}", "audited"))

    assert excluded.dropped_qualifiers == []
    assert included.dropped_qualifiers == ["audited"]


@pytest.mark.parametrize(
    ("dropped", "relations", "left"),
    [
        (
            f"including {_IM} aid, limited to standard grants",
            [("also", 0)],
            ["limited to standard grants"],
        ),
        (f"including {_IM} aid", [("also", 0)], []),
        (f"excluding {_IM}", [("excluded", None)], []),
        (f"audited, including {_IM} aid", [("also", 0)], ["audited"]),
        (
            f"audited; including {_IM}; limited to standard grants",
            [("also", 0)],
            ["audited, limited to standard grants"],
        ),
        (f"and {_IM}", [("also", 0)], []),
        (f"plus {_IM}", [("also", 0)], []),
        (f"from {_IM} sources", [("only", 0)], []),
        (f"don't include {_IM} aid in", [("excluded", None)], []),
        (f"counting {_IM}", [("unrelated", None)], [f"counting {_IM}"]),
        (
            f"excluding CRS and {_IM}",
            [("excluded", None), ("excluded", None)],
            [f"excluding CRS and {_IM}"],
        ),
        (f"excluding CRS, including {_IM}", [("excluded", None), ("also", 0)], ["excluding CRS"]),
        (f"audited including {_IM} aid", [("also", 0)], ["audited"]),
        (
            f"including {_IM} aid and limited to standard grants",
            [("also", 0)],
            ["and limited to standard grants"],
        ),
        (f"including {_IM} grants", [("also", 0)], ["grants"]),
        (f"from {_IM} sources for girls only", [("only", 0)], ["sources for girls only"]),
        (f"not from the {_IM} dataset for girls", [("excluded", None)], ["dataset for girls"]),
        (f"excluding {_IM}", [("also", 0)], ["excluding"]),
        (
            f"including {_IM} aid with verified figures only",
            [("also", 0)],
            ["with verified figures only"],
        ),
        (
            f"including {_IM} aid, with verified figures only",
            [("also", 0)],
            ["with verified figures only"],
        ),
        (f"including {_IM} data quality", [("also", 0)], ["data quality"]),
        (f"excluding {_IM} in rural areas", [("excluded", None)], ["in rural areas"]),
    ],
)
def test_a_dropped_phrase_keeps_every_word_its_applied_mentions_do_not_account_for(
    dropped, relations, left
):
    """An applied mention accounts for its span and its relation's words beside it. A linking
    word ("and", "data", "in") counts only where the words around it are accounted for."""
    extraction, _phrases = _apply([_E], relations, dropped=(dropped,))

    assert extraction.dropped_qualifiers == left


@pytest.mark.parametrize(
    ("dropped", "question", "reported"),
    [
        ((f"counting {_IM}",), "", [f"counting {_IM}"]),
        ((), f"{_E} to Kenya, {_IM}", [_IM]),
        (("audited",), f"{_E} to Kenya, {_IM}", ["audited", _IM]),
    ],
    ids=["own-phrase", "left-out", "another-phrase"],
)
def test_a_reading_that_took_no_effect_is_reported_once(dropped, question, reported):
    extraction, phrases = _apply([_E], [("unrelated", None)], dropped=dropped, question=question)

    assert (extraction.variables, phrases) == ([_E], {})
    assert extraction.dropped_qualifiers == reported


def test_a_dropped_phrase_with_one_mention_that_took_no_effect_stays():
    extraction, _phrases = _apply(
        [_E],
        [("unrelated", None), ("excluded", None)],
        dropped=(f"excluding CRS and {_IM}",),
    )

    assert extraction.dropped_qualifiers == [f"excluding CRS and {_IM}"]


def test_a_mention_inside_a_variable_that_took_no_effect_is_reported_by_its_text():
    extraction, phrases = _apply([_E, f"{_IM} health aid"], [("only", 0)])

    assert phrases == {}
    assert extraction.dropped_qualifiers == [_IM]


# ---------------------------------------------------------------------------
# The calls extract makes
# ---------------------------------------------------------------------------


class _Model:
    """Return a fixed extraction and fixed readings, or raise in the reading, and count
    the calls per schema."""

    def __init__(
        self, extraction: Extraction, readings: list[tuple[str, int | None]] | Exception = ()
    ) -> None:
        self._extraction = extraction
        self._readings = readings
        self.calls: Counter[str] = Counter()

    async def generate_structured(self, *, prompt, system, schema):
        self.calls[schema.__name__] += 1
        if schema is Extraction:
            return self._extraction, {"input_tokens": 100, "output_tokens": 10}
        if isinstance(self._readings, Exception):
            raise self._readings
        return MentionReadings(
            readings=[
                MentionReading(mention=number, relation=relation, variable=variable)
                for number, (relation, variable) in enumerate(self._readings)
            ]
        ), {"input_tokens": 40, "output_tokens": 4}


def _extraction(variables: list[str], dropped: tuple[str, ...] = ()) -> Extraction:
    return Extraction(kind="resolution", variables=variables, dropped_qualifiers=list(dropped))


@pytest.mark.parametrize(
    "query",
    ["education aid to Kenya", "population including refugees in Kenya", "aid to 106 countries"],
)
def test_a_query_naming_no_dataset_makes_one_model_call(query: str):
    model = _Model(_extraction(["education aid"]))

    result, usage = asyncio.run(extract(query, llm=model))

    assert model.calls == {"Extraction": 1}
    assert result.extraction.variables == ["education aid"]
    assert result.qualifiers == {}
    assert result.mention_reading_failed is False
    assert usage == {"input_tokens": 100, "output_tokens": 10}


@pytest.mark.parametrize(
    ("query", "variables", "dropped"),
    [
        (f"{_E} to Kenya including {_IM}", [_E], (f"including {_IM}",)),
        (f"{_E} to Kenya including {_IM}", [_E], ()),
        (f"{_IM_E} to Kenya", [_IM_E], ()),
        ("ayuda a la educación a Kenia", [f"{_E} {_IM}"], ()),
    ],
    ids=["query-and-dropped", "query-only", "query-and-variable", "variable-only"],
)
def test_a_mention_anywhere_in_the_request_makes_one_reading_call(query, variables, dropped):
    model = _Model(_extraction(variables, dropped), [("unrelated", None)])

    _result, usage = asyncio.run(extract(query, llm=model))

    assert model.calls == {"Extraction": 1, "MentionReadings": 1}
    assert usage == {"input_tokens": 140, "output_tokens": 14}


def test_the_bound_mention_rides_in_its_variables_sidecar_as_its_name():
    model = _Model(_extraction([_E], (f"including {_IM} aid",)), [("also", 0)])

    result, _usage = asyncio.run(extract(f"{_E} to Kenya including {_IM} aid", llm=model))

    assert result.extraction.variables == [_E, _IM_E]
    assert {
        variable: [coded.phrase for coded in sidecar]
        for variable, sidecar in result.qualifiers.items()
    } == {_IM_E: [_IM]}


@pytest.mark.parametrize(
    "readings",
    [
        LLMInfraError("the model is unavailable"),
        TimeoutError("the model timed out"),
        ValueError("the output did not validate"),
        [],
        [("only", 3)],
    ],
    ids=["infra", "timeout", "invalid", "missing", "bad-variable"],
)
def test_a_failed_reading_binds_nothing_and_reports_the_mention(readings):
    model = _Model(_extraction([_IM_E]), readings)

    result, _usage = asyncio.run(extract(f"{_IM_E} to Kenya", llm=model))

    assert result.extraction.variables == [_IM_E]
    assert result.qualifiers == {}
    assert result.extraction.dropped_qualifiers == [_IM]
    assert result.mention_reading_failed is isinstance(readings, Exception)


def test_the_layer2_key_inputs_differ_for_each_relation():
    """The cache keys on the variables, the sidecar and the dropped qualifiers, so no two
    readings of one question share an answer."""

    def inputs(reading: tuple[str, int | None]) -> tuple:
        model = _Model(_extraction([_E], (f"including {_IM}",)), [reading])
        result, _usage = asyncio.run(extract(f"{_E} to Kenya including {_IM}", llm=model))
        return (
            tuple(result.extraction.variables),
            tuple(result.extraction.dropped_qualifiers),
            tuple((v, tuple(c.phrase for c in s)) for v, s in result.qualifiers.items()),
        )

    read = [inputs(r) for r in [("only", 0), ("also", 0), ("excluded", None), ("unrelated", None)]]

    assert len(set(read)) == 4


# ---------------------------------------------------------------------------
# Recorded responses
# ---------------------------------------------------------------------------


def _recorded() -> dict[str, dict]:
    return json.loads((Path(fixtures.__file__).parent / "llm_responses.json").read_text())


def _recorded_extractions() -> dict[str, Extraction]:
    return {
        key: Extraction.model_validate(value)
        for key, value in _recorded().items()
        if key.startswith("Extraction@")
    }


_DATASET_QUERIES = [
    *(row[0] for row in _ACCEPTANCE),
    _PLUS,
    f"top 5 donors of {_E} to Kenya including {_IM}",
    "population and CRS in Kenya",
]


def test_the_gate_fires_for_no_recorded_extraction_but_those_of_the_dataset_questions():
    system = _system_instruction()
    dataset_keys = {fixture_key(Extraction, system, query) for query in _DATASET_QUERIES}
    recorded = _recorded_extractions()
    fired = {
        key
        for key, extraction in recorded.items()
        if gate_mentions("", extraction.variables, extraction.dropped_qualifiers, _INDEX)
    }

    assert fired
    assert fired <= dataset_keys
    assert len(recorded) - len(fired) >= 244


# The cases written to name a dataset.
_DATASET_CASES = frozenset({"imputed-aid-kenya", "education-aid-kenya-excluding-imputed"})


def test_no_recorded_corpus_query_but_a_dataset_case_fires_the_gate_under_its_own_text():
    recorded = _recorded_extractions()
    system = _system_instruction()
    replayed = {
        case.id: (case.query, recorded[key])
        for case in load_cases()
        if (key := fixture_key(Extraction, system, case.query)) in recorded
    }

    assert len(replayed) >= 100
    fired = {
        case_id
        for case_id, (query, extraction) in replayed.items()
        if gate_mentions(query, extraction.variables, extraction.dropped_qualifiers, _INDEX)
    }
    assert fired <= _DATASET_CASES


class _CountingFakeLLM:
    """Replay the recorded responses and count the calls per schema."""

    def __init__(self) -> None:
        self._llm = FakeLLM()
        self.calls: Counter[str] = Counter()

    async def generate_structured(self, *, prompt, system, schema):
        self.calls[schema.__name__] += 1
        return await self._llm.generate_structured(prompt=prompt, system=system, schema=schema)


@pytest.mark.parametrize(
    ("query", "variables"),
    [
        ("population including refugees in Kenya", ["population including refugees"]),
        ("GDP including the informal economy in Kenya", ["GDP including the informal economy"]),
        (
            "health spending including private spending in Kenya",
            ["health spending including private spending"],
        ),
        ("electricity access in Rwanda and Ghana", ["electricity access"]),
        ("population and GDP in Brazil", ["population", "GDP"]),
    ],
)
def test_a_recorded_query_naming_no_dataset_is_left_as_extracted_with_one_call(
    query: str, variables: list[str]
):
    llm = _CountingFakeLLM()

    result, _usage = asyncio.run(extract(query, llm=llm))

    assert llm.calls == {"Extraction": 1}
    assert result.extraction.variables == variables
    assert result.qualifiers == {}


def test_population_and_crs_answers_population_alone_and_reports_the_dataset_name():
    """The model extracts "CRS" as a variable of its own. A dataset's name with no measure
    is no part, so population answers alone and "CRS" is reported, whatever the reading."""
    result, _usage = asyncio.run(extract("population and CRS in Kenya", llm=FakeLLM()))

    assert result.extraction.variables == ["population"]
    assert result.qualifiers == {}
    assert result.extraction.dropped_qualifiers == ["CRS"]
