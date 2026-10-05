"""Tests for the gate that finds dataset mentions and for the reading of what each asks."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from code_index.shared import get_index
from search.engine.dataset_mention import (
    _MENTION_READING_SYSTEM_PROMPT,
    MentionReading,
    MentionReadings,
    ReadMention,
    _reading_prompt,
    apply_readings,
    dataset_mentions,
    gate_mentions,
    is_opt_in,
    read_mentions,
    readable_phrase,
    whole_mention,
)
from search.engine.departures import warning_key
from search.engine.entry.parsed import interpret_parts
from search.engine.errors import LLMInfraError
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.heading.departure_fact import DepartureFact
from search.models import ParsedPart
from tests.code_index.aid_identity import (
    CRS_PROVENANCE,
    IMPUTED_PROVENANCE,
)
from tests.engine._edited_index import packaged_index_with

_INDEX = get_index()


def _located(text: str, index=_INDEX) -> list[tuple[str, str]]:
    return [(mention.text, mention.provenance) for mention in dataset_mentions(text, index)]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CRS education aid", [("CRS", CRS_PROVENANCE)]),
        ("non-CRS education aid", [("CRS", CRS_PROVENANCE)]),
        ("the CRS's education aid", [("CRS", CRS_PROVENANCE)]),
        (
            "aid in the creditor reporting system",
            [("creditor reporting system", CRS_PROVENANCE)],
        ),
        (
            "imputed multilateral education aid",
            [("imputed multilateral", IMPUTED_PROVENANCE)],
        ),
        (
            "non-imputed multilateral aid",
            [("imputed multilateral aid", IMPUTED_PROVENANCE)],
        ),
        (
            "imputed multilateral ODA for education",
            [("imputed multilateral ODA", IMPUTED_PROVENANCE)],
        ),
        (
            "ONE Imputed Multilateral Sectors education aid",
            [("ONE Imputed Multilateral Sectors", IMPUTED_PROVENANCE)],
        ),
        (
            "CRS and imputed multilateral education aid",
            [
                ("CRS", CRS_PROVENANCE),
                ("imputed multilateral", IMPUTED_PROVENANCE),
            ],
        ),
        ("multilateral aid to Kenya", []),
        ("imputed, multilateral aid", []),
        ("education aid to Kenya", []),
        ("aid to 106 countries", []),
        ("aid in the crs purpose 110 to Kenya", []),
        ("crs type of finance 110", []),
        ("ayuda multilateral imputada", []),
    ],
)
def test_a_text_names_each_dataset_once_and_span_exact(text: str, expected: list):
    assert _located(text) == expected


@pytest.mark.parametrize(
    ("located", "name", "measure_words"),
    [
        ("imputed multilateral aid", "imputed multilateral", ("aid",)),
        ("imputed ODA", "imputed", ("oda",)),
    ],
)
def test_a_mention_carries_its_offsets_its_name_and_the_measure_words_ending_it(
    located: str, name: str, measure_words: tuple[str, ...]
):
    """A synonym that ends in a word another name of the dataset adds as its measure ends
    in that measure word too."""
    text = f"aid, the {located}"

    (mention,) = dataset_mentions(text, _INDEX)

    assert text[mention.start : mention.end] == mention.text == located
    assert (mention.name, mention.measure_words) == (name, measure_words)
    assert mention.provenance == IMPUTED_PROVENANCE
    assert mention.label(_INDEX) == _INDEX.datasets[IMPUTED_PROVENANCE].label
    assert is_opt_in(mention, _INDEX)


def test_a_dataset_not_declared_opt_in_is_not_opt_in():
    (mention,) = dataset_mentions("CRS aid", _INDEX)

    assert mention.provenance == CRS_PROVENANCE
    assert not is_opt_in(mention, _INDEX)


@pytest.mark.parametrize(
    ("text", "dataset"),
    [
        ("ONE Imputed Multilateral Sectors", IMPUTED_PROVENANCE),
        ("imputed multilateral", IMPUTED_PROVENANCE),
        ("CRS", CRS_PROVENANCE),
    ],
)
def test_every_name_of_a_dataset_names_the_one_dataset(text: str, dataset: str):
    (mention,) = dataset_mentions(text, _INDEX)

    assert mention.provenance == dataset


def test_a_name_two_datasets_declare_names_neither():
    def share_a_synonym(document: dict) -> None:
        for dataset in document["datasets"]:
            dataset["synonyms"] = [*dataset["synonyms"], "aid tables"]

    assert _located("aid tables for Kenya", packaged_index_with(share_a_synonym)) == []


def _gated(question: str, variables: list[str], dropped: tuple[str, ...] = ()) -> list[tuple]:
    return [
        (g.mention.text, g.mention.provenance, g.variable, g.phrase)
        for g in gate_mentions(question, variables, dropped, _INDEX)
    ]


def test_the_gate_hands_every_occurrence_in_a_variable_and_in_a_dropped_phrase():
    gated = _gated(
        "imputed multilateral aid and imputed multilateral education aid, including CRS",
        ["imputed multilateral aid", "imputed multilateral education aid"],
        ("including CRS",),
    )

    assert gated == [
        ("imputed multilateral aid", IMPUTED_PROVENANCE, 0, None),
        ("imputed multilateral", IMPUTED_PROVENANCE, 1, None),
        ("CRS", CRS_PROVENANCE, None, 0),
    ]


def test_the_gate_hands_a_question_occurrence_only_for_a_dataset_no_other_text_names():
    question = "education aid to Kenya, imputed multilateral, per the CRS"

    assert _gated(question, ["education aid"]) == [
        ("imputed multilateral", IMPUTED_PROVENANCE, None, None),
        ("CRS", CRS_PROVENANCE, None, None),
    ]
    assert _gated(question, ["education aid"], ("imputed multilateral",)) == [
        ("imputed multilateral", IMPUTED_PROVENANCE, None, 0),
        ("CRS", CRS_PROVENANCE, None, None),
    ]
    assert _gated("education aid to Kenya", ["education aid"]) == []


def test_the_gate_hands_no_question_occurrence_of_a_dataset_a_variable_names_another_way():
    gated = _gated(
        "ONE Imputed Multilateral Sectors education aid to Kenya",
        ["imputed multilateral education aid"],
    )

    assert gated == [("imputed multilateral", IMPUTED_PROVENANCE, 0, None)]


@pytest.mark.parametrize(
    ("text", "located"),
    [
        ("Imputed Multilateral education aid", "Imputed Multilateral"),
        ("imputéd multilatéral education aid", "imputéd multilatéral"),
        ("imputed-multilateral education aid", "imputed-multilateral"),
        ("imputed multilaterals education aid", "imputed multilaterals"),
        ("non-imputed multilateral aid", "imputed multilateral aid"),
    ],
)
def test_case_accents_a_joining_hyphen_and_a_plural_fold_into_a_declared_name(text, located):
    (mention,) = dataset_mentions(text, _INDEX)

    assert (mention.text, mention.provenance) == (located, IMPUTED_PROVENANCE)
    assert readable_phrase(mention, _INDEX) is not None


@pytest.mark.parametrize(
    "text",
    [
        "OECD Creditor Reporting System (CRS) education aid",
        "Creditor Reporting System (CRS) education aid",
        "education aid in the creditor reporting system [CRS]",
    ],
)
def test_a_dataset_name_followed_by_its_bracketed_short_name_is_one_mention(text):
    (mention,) = dataset_mentions(text, _INDEX)

    assert (mention.provenance, mention.measure_words) == (CRS_PROVENANCE, ())
    assert mention.text.casefold() in (
        "creditor reporting system (crs)",
        "creditor reporting system [crs]",
    )
    assert readable_phrase(mention, _INDEX) is not None


def test_two_datasets_side_by_side_stay_two_mentions():
    mentions = dataset_mentions("CRS (imputed multilateral) education aid", _INDEX)

    assert [mention.provenance for mention in mentions] == [CRS_PROVENANCE, IMPUTED_PROVENANCE]


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("imputed multilateral", IMPUTED_PROVENANCE),
        ("Imputed Multilateral aid", IMPUTED_PROVENANCE),
        ("CRS", CRS_PROVENANCE),
    ],
)
def test_a_text_that_is_a_mention_alone_is_read_back_whole(text: str, key: str):
    mention = whole_mention(text, _INDEX)

    assert mention is not None
    assert mention.provenance == key


@pytest.mark.parametrize(
    "text", ["imputed multilateral education aid", "the CRS", "education aid", "CRS and CRS"]
)
def test_a_text_holding_anything_else_is_no_whole_mention(text: str):
    assert whole_mention(text, _INDEX) is None


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------

_VARIABLES = ["CRS education aid", "imputed multilateral health aid"]
_GATED = gate_mentions("the question", _VARIABLES, (), _INDEX)


class _Model:
    """Return the given readings, or raise the given error, and count the calls."""

    def __init__(self, answer: list[tuple[int, str, int | None]] | Exception) -> None:
        self._answer = answer
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        self.prompts.append(prompt)
        assert system == _MENTION_READING_SYSTEM_PROMPT
        assert schema is MentionReadings
        if isinstance(self._answer, Exception):
            raise self._answer
        return MentionReadings(
            readings=[
                MentionReading(mention=mention, relation=relation, variable=variable)
                for mention, relation, variable in self._answer
            ]
        ), {"input_tokens": 7, "output_tokens": 3}


def _read(answer, variables=_VARIABLES):
    model = _Model(answer)
    result = asyncio.run(read_mentions("the question", variables, (), index=_INDEX, llm=model))
    return [(r.mention.provenance, r.relation, r.variable) for r in result.readings], result, model


def test_each_handed_mention_takes_the_relation_and_variable_read_for_it():
    read, result, model = _read([(1, "also", 1), (0, "only", 0)])

    assert read == [(CRS_PROVENANCE, "only", 0), (IMPUTED_PROVENANCE, "also", 1)]
    assert result.failed is False
    assert result.usage == {"input_tokens": 7, "output_tokens": 3}
    assert len(model.prompts) == 1


def test_an_excluded_mention_keeps_a_listed_variable_and_needs_none():
    read, _result, _model = _read([(0, "excluded", 0), (1, "excluded", None)])
    out_of_range, _result, _model = _read([(0, "excluded", 5), (1, "excluded", -1)])

    assert read == [(CRS_PROVENANCE, "excluded", 0), (IMPUTED_PROVENANCE, "excluded", None)]
    assert out_of_range == [
        (CRS_PROVENANCE, "excluded", None),
        (IMPUTED_PROVENANCE, "excluded", None),
    ]


@pytest.mark.parametrize(
    "answer",
    [
        [(0, "only", 0)],
        [(0, "only", 0), (1, "unrelated", None)],
        [(0, "only", 0), (1, "only", None)],
        [(0, "only", 0), (1, "also", 2)],
        [(0, "only", 0), (1, "also", -1)],
        [(0, "only", 0), (1, "only", 0), (1, "also", 1)],
        [(0, "only", 0), (5, "only", 1)],
    ],
    ids=["missing", "unrelated", "no-variable", "past-range", "negative", "twice", "unknown"],
)
def test_a_mention_with_no_usable_entry_is_unrelated(answer):
    read, result, _model = _read(answer)

    assert read == [(CRS_PROVENANCE, "only", 0), (IMPUTED_PROVENANCE, "unrelated", None)]
    assert result.failed is False


@pytest.mark.parametrize(
    "error",
    [
        LLMInfraError("the model is unavailable"),
        TimeoutError("the model timed out"),
        ValueError("the output is not JSON"),
    ],
)
def test_a_failed_call_reads_every_mention_unrelated_and_says_it_failed(error: Exception):
    read, result, _model = _read(error)

    assert read == [
        (CRS_PROVENANCE, "unrelated", None),
        (IMPUTED_PROVENANCE, "unrelated", None),
    ]
    assert result.failed is True


def test_an_output_outside_the_closed_relations_does_not_validate():
    with pytest.raises(ValidationError):
        MentionReading(mention=0, relation="maybe", variable=0)

    class _Invalid:
        async def generate_structured(self, *, prompt, system, schema):
            return schema.model_validate({"readings": [{"mention": 0, "relation": "maybe"}]}), None

    result = asyncio.run(read_mentions("q", _VARIABLES, (), index=_INDEX, llm=_Invalid()))

    assert {r.relation for r in result.readings} == {"unrelated"}
    assert result.failed is True


def test_no_mention_makes_no_call():
    read, result, model = _read([(0, "only", 0)], variables=["education aid"])

    assert read == []
    assert result.failed is False
    assert model.prompts == []


def test_the_prompt_hands_the_question_the_variables_and_each_mention_where_it_stands():
    dropped = ("including imputed multilateral",)
    variables = ["CRS education aid"]
    gated = gate_mentions("the question", variables, dropped, _INDEX)

    prompt = _reading_prompt("the question", variables, dropped, gated, _INDEX)

    assert prompt.splitlines() == [
        "Question: the question",
        "",
        "Measures:",
        "  0: CRS education aid",
        "",
        "Set-aside phrases:",
        "  including imputed multilateral",
        "",
        "Mentions:",
        '  0: "CRS" in measure 0, can name the data source OECD Creditor Reporting System (CRS)',
        '  1: "imputed multilateral" in the set-aside phrase "including imputed multilateral", '
        "can name the data source ONE Imputed Multilateral Sectors",
    ]


def test_the_fixed_prompt_text_names_no_declared_dataset():
    fixed = _MENTION_READING_SYSTEM_PROMPT.casefold()
    declared = [
        name for dataset in _INDEX.datasets.values() for name in (dataset.label, *dataset.synonyms)
    ]

    assert declared
    assert not [name for name in declared if name.casefold() in fixed]
    assert dataset_mentions(_MENTION_READING_SYSTEM_PROMPT, _INDEX) == ()


# ---------------------------------------------------------------------------
# What a reading does
# ---------------------------------------------------------------------------

_E = "education aid"
_IM = "imputed multilateral"
_IM_E = "imputed multilateral education aid"
_IMPUTED_LABEL = "ONE Imputed Multilateral Sectors"


def _apply(
    variables: list[str],
    relations: list[tuple[str, int | None]],
    *,
    dropped: tuple[str, ...] = (),
    question: str = "",
    fixed: bool = False,
):
    """Apply one relation to each gated mention, in gate order."""
    gated = gate_mentions(question, variables, dropped, _INDEX)
    assert len(gated) == len(relations), [g.mention.text for g in gated]
    readings = [
        ReadMention(g, relation, variable)
        for g, (relation, variable) in zip(gated, relations, strict=True)
    ]
    applied = apply_readings(variables, readings, index=_INDEX, fixed_variables=fixed)
    return (
        list(applied.variables),
        {
            applied.variables[position]: mention.provenance
            for position, mention in applied.bound.items()
        },
        [read.mention.text for read in applied.reported],
        applied,
    )


def test_only_binds_the_mention_on_the_variable_that_holds_it():
    variables, bound, reported, applied = _apply([_E, _IM_E], [("only", 1)])

    assert (variables, bound, reported) == ([_E, _IM_E], {_IM_E: IMPUTED_PROVENANCE}, [])
    assert applied.phrases == {1: _IM}
    assert applied.origin == (0, 1)


def test_a_dataset_named_on_two_variables_binds_on_each():
    variables, bound, reported, _applied = _apply([f"{_IM} aid", _IM_E], [("only", 0), ("only", 1)])

    assert variables == [f"{_IM} aid", _IM_E]
    assert bound == {f"{_IM} aid": IMPUTED_PROVENANCE, _IM_E: IMPUTED_PROVENANCE}
    assert reported == []


def test_only_from_a_dropped_phrase_or_the_question_binds_on_the_variable_read_on():
    for dropped, question in [((f"from {_IM} sources",), ""), ((), f"{_E}, {_IM} only")]:
        variables, bound, reported, _applied = _apply(
            [_E], [("only", 0)], dropped=dropped, question=question
        )

        assert (variables, bound, reported) == ([_E], {_E: IMPUTED_PROVENANCE}, [])


def test_only_read_on_another_variable_gives_up_a_variable_that_is_the_mention_alone():
    variables, bound, reported, _applied = _apply([_E, f"{_IM} aid"], [("only", 0)])

    assert (variables, bound, reported) == ([_E], {_E: IMPUTED_PROVENANCE}, [])


def test_only_read_on_another_variable_than_the_measure_holding_it_is_reported():
    health = f"{_IM} health aid"

    variables, bound, reported, _applied = _apply([_E, health], [("only", 0)])

    assert (variables, bound, reported) == ([_E, health], {}, [_IM])


def test_a_second_dataset_read_only_on_one_variable_is_reported():
    variables, bound, reported, _applied = _apply([f"CRS {_IM_E}"], [("only", 0), ("only", 0)])

    assert bound == {variables[0]: CRS_PROVENANCE}
    assert reported == [_IM]


@pytest.mark.parametrize(
    "variable",
    [f"{_IM_E} from the {_IMPUTED_LABEL}", f"{_IMPUTED_LABEL} {_IM_E}"],
    ids=["value-first", "dataset-first"],
)
def test_two_names_of_one_dataset_on_one_variable_bind_it_once(variable):
    variables, bound, reported, applied = _apply([variable], [("only", 0), ("only", 0)])

    assert (variables, bound, reported) == ([variable], {variable: IMPUTED_PROVENANCE}, [])
    read_back = whole_mention(applied.phrases[0], _INDEX)
    assert read_back is not None
    assert read_back.provenance == IMPUTED_PROVENANCE


def test_also_takes_the_measure_from_a_variable_naming_the_dataset_another_way():
    variable = f"{_IMPUTED_LABEL} {_E}"

    variables, bound, reported, _applied = _apply(
        [variable], [("only", 0), ("also", 0)], dropped=(f"including {_IM}",)
    )

    assert (variables, bound, reported) == ([variable, _E], {variable: IMPUTED_PROVENANCE}, [])


@pytest.mark.parametrize(
    ("variables", "dropped", "relations", "result"),
    [
        ([_E], (f"including {_IM}",), [("also", 0)], [_E, _IM_E]),
        ([_E], (f"including {_IM} ODA",), [("also", 0)], [_E, _IM_E]),
        (
            ["education spending"],
            (f"including {_IM} aid",),
            [("also", 0)],
            ["education spending", f"{_IM} education spending"],
        ),
        ([_E, f"{_IM} aid"], (), [("also", 0)], [_E, _IM_E]),
        ([_E, f"{_IM} ODA"], (), [("also", 0)], [_E, _IM_E]),
        ([_E, _IM_E], (), [("also", 0)], [_E, _IM_E]),
        ([_E, _IM_E], (), [("also", 1)], [_E, _IM_E]),
        (
            [_E, "Imputed Multilateral Education Aid"],
            (),
            [("also", 0)],
            [_E, "Imputed Multilateral Education Aid"],
        ),
        ([_IM_E], (), [("also", 0)], [_IM_E, _E]),
        ([_IM_E], (f"including {_IM}",), [("only", 0), ("also", 0)], [_IM_E, _E]),
    ],
    ids=[
        "dropped-phrase",
        "dropped-phrase-other-measure-word",
        "measure-lacking-the-mentions-word",
        "name-alone",
        "name-alone-other-measure-word",
        "own-variable-read-on-the-measure",
        "own-variable-read-on-itself",
        "case-variant",
        "collapsed-into-one-variable",
        "variable-and-dropped-phrase",
    ],
)
def test_also_gives_one_default_part_and_one_part_of_the_dataset_for_the_same_measure(
    variables, dropped, relations, result
):
    final, bound, reported, _applied = _apply(variables, relations, dropped=dropped)

    assert final == result
    assert list(bound.values()) == [IMPUTED_PROVENANCE]
    assert next(iter(bound)).casefold().startswith(_IM)
    assert reported == []


@pytest.mark.parametrize(
    ("variables", "relations", "bound", "reported"),
    [
        (["health aid", _IM_E], [("also", 0)], [], [_IM]),
        (["health aid", _IM_E, "humanitarian aid"], [("also", 2)], [], [_IM]),
        ([_E, f"{_IM} flows"], [("also", 0)], [], [_IM]),
        ([_E, f"{_IM} student aid"], [("also", 0)], [], [_IM]),
        ([f"{_IM} aid", _IM_E], [("also", 0), ("also", 0)], [f"{_IM} aid"], [_IM]),
    ],
    ids=[
        "another-measure",
        "three-variables",
        "undeclared-measure-word",
        "another-word",
        "target-is-the-mention",
    ],
)
def test_also_read_away_from_a_variable_naming_another_measure_is_reported(
    variables, relations, bound, reported
):
    final, final_bound, final_reported, applied = _apply(variables, relations)

    assert final == variables
    assert final_bound == dict.fromkeys(bound, IMPUTED_PROVENANCE)
    assert final_reported == reported
    assert applied.operations_of == {}


@pytest.mark.parametrize("own", [_IM_E, f"{_E}, {_IM}", "Imputed Multilateral Education Aid"])
def test_also_read_away_from_a_variable_naming_the_same_measure_binds_it_there(own):
    variables, bound, reported, applied = _apply([_E, own], [("also", 0)])

    assert variables == [_E, own]
    assert (bound, reported) == ({own: IMPUTED_PROVENANCE}, [])
    assert applied.operations_of == {1: 0}


def test_also_of_a_mention_alone_variable_read_on_a_variable_naming_no_measure_is_reported():
    variables, bound, reported, _applied = _apply(
        [f"{_IM} aid", f"{_IM} ODA"], [("unrelated", None), ("also", 0)]
    )

    assert variables == [f"{_IM} aid", f"{_IM} ODA"]
    assert (bound, reported) == ({}, [f"{_IM} aid", f"{_IM} ODA"])


def test_also_read_on_the_variable_holding_the_mention_keeps_that_variables_text():
    health = f"{_IM} health aid"

    variables, bound, reported, _applied = _apply([_E, health], [("also", 1)])

    assert variables == [_E, health, "health aid"]
    assert (bound, reported) == ({health: IMPUTED_PROVENANCE}, [])


def test_also_says_which_variables_operations_the_added_variable_carries():
    _variables, _bound, _reported, added = _apply([_E], [("also", 0)], dropped=(f"plus {_IM}",))
    _variables, _bound, _reported, collapsed = _apply([_IM_E], [("also", 0)])

    assert (added.origin, added.operations_of) == ((0, None), {1: 0})
    assert (collapsed.origin, collapsed.operations_of) == ((0, None), {1: 0})


def test_also_of_the_default_dataset_takes_effect_and_changes_nothing():
    variables, bound, reported, _applied = _apply([_E], [("also", 0)], dropped=("including CRS",))

    assert (variables, bound, reported) == ([_E], {}, [])


def test_excluded_takes_effect_for_a_dataset_the_default_answer_never_holds():
    for variables, dropped in [([_E], (f"excluding {_IM}",)), ([f"{_E} excluding {_IM}"], ())]:
        final, bound, reported, _applied = _apply(variables, [("excluded", None)], dropped=dropped)

        assert (final, bound, reported) == (variables, {}, [])


def test_excluded_removes_a_variable_that_is_the_mention_alone_beside_another_variable():
    for relation in [("excluded", None), ("excluded", 0), ("excluded", 1)]:
        variables, bound, reported, applied = _apply([_E, f"{_IM} aid"], [relation])

        assert (variables, bound, reported) == ([_E], {}, [])
        assert applied.origin == (0,)


@pytest.mark.parametrize(
    ("variable", "measure"), [(f"{_IM} aid", ["aid"]), (f"{_IM} ODA", ["ODA"]), (_IM, [])]
)
def test_excluded_leaves_the_requests_only_variable_its_measure_without_the_mention(
    variable, measure
):
    variables, bound, reported, applied = _apply([variable], [("excluded", None)])

    assert (variables, bound, reported) == (measure, {}, [])
    assert applied.origin == tuple(0 for _ in measure)


@pytest.mark.parametrize(
    "exclusion",
    [f"{_IM} excluded", f"excluding {_IM}", f"not from the {_IMPUTED_LABEL}"],
)
def test_excluded_read_on_another_variable_removes_the_variable_it_stands_in(exclusion):
    variables, bound, reported, _applied = _apply([_E, exclusion], [("excluded", 0)])

    assert (variables, bound, reported) == ([_E], {}, [])


@pytest.mark.parametrize(
    "exclusion",
    [
        f"{_E} excluding {_IM}",
        f"{_IM} excluded, audited",
        f"audited; {_IM} excluded; verified only",
    ],
    ids=["measure", "after", "around"],
)
def test_excluded_read_on_another_variable_keeps_and_reports_a_variable_holding_more(exclusion):
    """A variable naming more than the exclusion keeps its words and reports the mention."""
    variables, bound, reported, _applied = _apply(["health aid", exclusion], [("excluded", 0)])

    assert (variables, bound) == (["health aid", exclusion], {})
    assert reported == [_IM]


def test_excluded_read_on_the_variable_it_stands_in_keeps_that_variable():
    health = f"health aid excluding {_IM}"

    for relation in [("excluded", 1), ("excluded", None)]:
        variables, bound, reported, _applied = _apply([_E, health], [relation])

        assert (variables, bound, reported) == ([_E, health], {}, [])


def test_excluding_a_dataset_that_is_not_opt_in_is_reported():
    variables, bound, reported, _applied = _apply(["non-CRS education aid"], [("excluded", None)])

    assert (variables, bound, reported) == (["non-CRS education aid"], {}, ["CRS"])


@pytest.mark.parametrize(
    "relation", [("only", 1), ("also", 0), ("unrelated", None), ("excluded", None)]
)
def test_a_variable_that_is_a_dataset_name_with_no_measure_is_no_part(relation):
    variables, bound, reported, _applied = _apply(["population", "CRS"], [relation])

    assert (variables, bound) == (["population"], {})
    assert reported == ([] if relation[0] == "also" else ["CRS"])


def test_unrelated_binds_nothing_and_is_reported():
    variables, bound, reported, _applied = _apply([_IM_E], [("unrelated", None)])

    assert (variables, bound, reported) == ([_IM_E], {}, [_IM])


@pytest.mark.parametrize(
    ("variables", "relations", "bound", "reported"),
    [
        ([_IM_E], [("only", 0)], {_IM_E: IMPUTED_PROVENANCE}, []),
        ([_E, _IM_E], [("also", 0)], {_IM_E: IMPUTED_PROVENANCE}, []),
        ([_E, _IM_E], [("also", 1)], {_IM_E: IMPUTED_PROVENANCE}, []),
        ([f"{_IM} aid", "aid"], [("also", 1)], {f"{_IM} aid": IMPUTED_PROVENANCE}, []),
        ([_IM_E], [("also", 0)], {}, [_IM]),
        ([_E, f"{_IM} aid"], [("also", 0)], {}, [f"{_IM} aid"]),
        ([_E, f"{_IM} aid"], [("only", 0)], {}, [f"{_IM} aid"]),
        ([_E, f"{_IM} aid"], [("excluded", None)], {}, [f"{_IM} aid"]),
        ([f"{_IM} aid"], [("excluded", None)], {}, [f"{_IM} aid"]),
        ([_E, f"{_IM} excluded"], [("excluded", 0)], {}, [_IM]),
        ([f"{_E} excluding {_IM}"], [("excluded", None)], {}, []),
        (["non-CRS education aid"], [("excluded", None)], {}, ["CRS"]),
        (["population", "CRS"], [("unrelated", None)], {}, ["CRS"]),
    ],
    ids=[
        "only",
        "also-read-on-the-measure",
        "also-read-on-its-own-part",
        "also-name-before-the-measure",
        "also",
        "also-renames",
        "only-removes",
        "excluded-removes",
        "excluded-renames",
        "excluded-elsewhere",
        "excluded",
        "excluded-default",
        "name-alone",
    ],
)
def test_with_the_variables_held_as_posted_a_reading_binds_only_where_none_changes(
    variables, relations, bound, reported
):
    final, final_bound, final_reported, _applied = _apply(variables, relations, fixed=True)

    assert final == variables
    assert (final_bound, final_reported) == (bound, reported)


def test_a_parsed_part_whose_mention_takes_no_effect_departs_on_the_source_token():
    """The parsed door holds its parts as posted, so `also` on a part that is the measure
    of the mention would rename it and is reported: the part warns and records a
    `measure_not_applied` departure the source token clears."""
    interpretation = asyncio.run(
        interpret_parts(
            [ParsedPart(variable_text=_IM_E)],
            llm=_Model([(0, "also", 0)]),
            deadline=time.monotonic() + 60,
        )
    )

    (warnings,) = interpretation.warnings
    (dropped,) = [w for w in warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert interpretation.departure_facts == [
        {
            warning_key(dropped): DepartureFact(
                code="measure_not_applied", binding_ref="source:", asked=_IM
            )
        }
    ]
