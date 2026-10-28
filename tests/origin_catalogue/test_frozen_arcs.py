"""A table answer is built once per snapshot and set of rows, is read-only, and reads as the
wire answer the graph client returns."""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from dc_graph import IMPORT_ADDED_CONSTRAINT_PROPERTIES
from origin_catalogue import frozen_arcs
from origin_catalogue.frozen_arcs import ArcAnswers, FrozenArcs, served_arcs, serves_arcs
from origin_catalogue.un_mirror import UnArc, wire_arcs
from search.engine.retrieval import facts

_ROWS = (
    UnArc(property="constraintProperties", dcid="recipient"),
    UnArc(property="constraintProperties", dcid="sector"),
    UnArc(property="measurementDenominator", dcid="Count_Person", types=("ProvisionalNode",)),
    UnArc(property="name", value="Gross ODA"),
    UnArc(property="populationType", dcid="Flow"),
    UnArc(property="recipient", dcid="country/KEN", name="Kenya", types=("Country",)),
    UnArc(property="sector", dcid="CRSPurposeCode120"),
)


def test_an_answer_is_built_once_for_its_rows_and_again_for_moved_rows(monkeypatch):
    built: list[tuple[UnArc, ...]] = []
    build = frozen_arcs.served_arcs
    monkeypatch.setattr(frozen_arcs, "served_arcs", lambda rows: built.append(rows) or build(rows))
    answers = ArcAnswers()

    first = answers.answer("sv", _ROWS)
    again = answers.answer("sv", _ROWS)
    # Equal rows from another release are another tuple.
    moved_rows = (*_ROWS,)
    moved = answers.answer("sv", moved_rows)

    assert moved_rows is not _ROWS
    assert again is first
    assert moved == first
    assert moved is not first
    assert len(built) == 2


def test_the_memo_keeps_the_most_recently_read_answers(monkeypatch):
    built: list[tuple[UnArc, ...]] = []
    build = frozen_arcs.served_arcs
    monkeypatch.setattr(frozen_arcs, "served_arcs", lambda rows: built.append(rows) or build(rows))
    answers = ArcAnswers(held=2)

    first_a = answers.answer("a", _ROWS)
    answers.answer("b", _ROWS[:2])
    assert answers.answer("a", _ROWS) is first_a
    answers.answer("c", _ROWS[:3])
    answers.answer("b", _ROWS[:2])

    assert answers.answer("a", _ROWS) is not first_a
    assert len(built) == 5


_IMPORT_ADDED = UnArc(
    property="constraintProperties", dcid=next(iter(IMPORT_ADDED_CONSTRAINT_PROPERTIES))
)


@pytest.mark.parametrize(
    "rows",
    [(), _ROWS, (_IMPORT_ADDED,), (UnArc(property="name", value="Gross ODA"),)],
    ids=["none", "held", "import-added", "name"],
)
def test_serves_arcs_says_whether_the_answer_holds_an_arc(rows):
    assert serves_arcs(rows) is bool(served_arcs(rows))


def test_an_answer_reads_as_the_wire_answer_and_refuses_every_mutation():
    answer = served_arcs(_ROWS)
    node = answer["recipient"]["nodes"][0]

    assert answer == wire_arcs(_ROWS)
    assert json.loads(json.dumps(answer)) == wire_arcs(_ROWS)
    assert node["types"] == ["Country"]
    assert copy.deepcopy(answer) is answer
    assert pickle.loads(pickle.dumps(answer)) == answer
    for mutate in (
        lambda: answer.__setitem__("typeOf", {}),
        lambda: answer.pop("name"),
        lambda: answer["sector"].update(nodes=[]),
        lambda: answer["sector"]["nodes"].append({"dcid": "x"}),
        lambda: node.__setitem__("dcid", "country/USA"),
        lambda: node["types"].append("Place"),
    ):
        with pytest.raises(TypeError):
            mutate()


def test_facts_parse_an_answer_once_and_a_live_answer_on_every_read(monkeypatch):
    """Each phase of a resolve reads a member's constraints and five-tuple again. A table
    answer is parsed on the first read only, and gives what the live answer gives."""
    reads: list[str] = []
    value_for = facts._value_for
    monkeypatch.setattr(
        facts, "_value_for", lambda arcs, prop: reads.append(prop) or value_for(arcs, prop)
    )
    answer = served_arcs(_ROWS)
    live = wire_arcs(_ROWS)

    first = [facts.read_constraints(answer), facts.read_five_tuple(answer)]
    parsed = len(reads)
    for _ in range(3):
        assert [facts.read_constraints(answer), facts.read_five_tuple(answer)] == first
    assert len(reads) == parsed

    assert [facts.read_constraints(live), facts.read_five_tuple(live)] == first
    assert len(reads) == 2 * parsed
    assert first[0] == {"recipient": "country/KEN", "sector": "CRSPurposeCode120"}
    assert first[1].meas_denom_unresolved
    assert isinstance(answer, FrozenArcs)


def test_each_read_of_constraints_returns_a_dict_of_its_own():
    answer = served_arcs(_ROWS)

    mine = facts.read_constraints(answer)
    mine["sector"] = "CRSPurposeCode130"

    assert facts.read_constraints(answer)["sector"] == "CRSPurposeCode120"
