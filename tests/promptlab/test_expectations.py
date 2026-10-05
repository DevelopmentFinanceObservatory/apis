"""Tests for reading a case's expectation per LLM leg."""

from __future__ import annotations

import pytest

from promptlab.expectations import case_entities, case_stat_vars, leg_expectation
from search.eval.cases import Case, validate_case

_SHAPE = {"population_type": "Person", "measured_property": "count"}


def _case(expect: dict) -> Case:
    return validate_case(
        {"id": "c1", "query": "q", "why": "A case for the expectation reader.", "expect": expect},
        families={},
    )


_TWO_PARTS = _case(
    {
        "parts": [
            {
                "definite": {
                    "measure": {"shape": _SHAPE, "stat_vars": ["sv/A"]},
                    "entities": [{"dcid": "country/KEN", "role": "observationAbout"}],
                }
            },
            {
                "candidates": {"measure": {"shape": {"population_type": "Emissions"}}},
                "no_data": {},
            },
        ]
    }
)


def test_a_single_part_case_is_one_leg_with_its_shape_and_stat_vars():
    case = _case({"definite": {"measure": {"shape": _SHAPE, "stat_vars": ["sv/A", "sv/B"]}}})

    expected = leg_expectation(case, 0)

    assert expected is not None
    assert expected.sole_status == "definite"
    assert expected.shapes == (_SHAPE,)
    assert expected.stat_vars == ("sv/A", "sv/B")
    assert leg_expectation(case, 1) is None


def test_parts_are_legs_in_query_order_and_several_statuses_have_no_sole_status():
    first, second = leg_expectation(_TWO_PARTS, 0), leg_expectation(_TWO_PARTS, 1)

    assert first is not None
    assert first.sole_status == "definite"
    assert second is not None
    assert second.statuses == ("candidates", "no_data")
    assert second.sole_status is None
    assert second.shapes == ({"population_type": "Emissions"},)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param({"population_type": {"gl": "read the population type"}}, id="unconfirmed"),
        pytest.param({"population_type": None}, id="asserts-only-absence"),
    ],
)
def test_a_shape_without_a_confirmed_value_is_left_out(shape):
    case = _case({"definite": {"measure": {"shape": shape}}})

    expected = leg_expectation(case, 0)

    assert expected is not None
    assert expected.shapes == ()


def test_a_shape_is_taken_from_definite_and_candidates_only():
    case = _case({"no_data": {"measure": {"shape": _SHAPE}}})

    expected = leg_expectation(case, 0)

    assert expected is not None
    assert expected.shapes == ()


def test_case_entities_and_stat_vars_union_every_leg():
    assert case_entities(_TWO_PARTS) == {"country/KEN"}
    assert case_stat_vars(_TWO_PARTS) == {"sv/A"}


def test_a_catalogue_case_expects_nothing():
    case = _case({"kind": "catalogue"})

    assert leg_expectation(case, 0) is not None
    assert case_entities(case) == set()
    assert case_stat_vars(case) == set()
