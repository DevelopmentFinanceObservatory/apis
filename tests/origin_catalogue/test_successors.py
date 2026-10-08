"""Pins which UN variable stands for a superseded base variable."""

from __future__ import annotations

import pytest

from origin_catalogue.registry import Registry, parse_registry
from origin_catalogue.successors import Successors, derive_successors, unheld_successors
from tests.origin_catalogue.fake_catalogue import un_mirror, un_variable

MORT_SERIES = "undata/who/SERIES-MORT"
ODA_SERIES = "undata/who/SERIES-ODA"
_SEX = {"undataSex": ("Sex", "undata/SEX-F", "Female")}
_AGE = {"undataAge": ("Age", "undata/AGE-Y0", "Under 1")}


def _registry() -> Registry:
    return parse_registry(
        b"""
[[base]]
source = "dc/s/Old"
label = "Old"
exclusions = [
  { provenance = "dc/base/Old", reason = "superseded_by_un_system", successor = "undata/p/WHO", variable_prefix = "old/", code_delimiter = ".", pair_delimiter = "__", value_delimiter = "--" },
]

[[un_system]]
provenance = "undata/p/WHO"
segment = "who"
label = "WHO"
"""
    )


def _derive(*withheld: str, dfo: tuple[str, ...] = ()) -> Successors:
    mirror = un_mirror(
        {
            "undata/who/MORT.SEX--F": un_variable(MORT_SERIES, name="f", constraints=_SEX),
            "undata/who/MORT.AGE--Y0": un_variable(MORT_SERIES, name="a", constraints=_AGE),
            "undata/who/MORT.AGE--Y0__SEX--F": un_variable(
                MORT_SERIES, name="af", constraints=_SEX | _AGE
            ),
        },
        excluded={"undata/who/ODA.DONOR--X": ODA_SERIES},
    )
    return derive_successors(
        _registry(),
        withheld_variables=frozenset(withheld),
        dfo_variables=frozenset(dfo),
        mirror=mirror,
    )


def test_a_variable_with_a_un_variable_of_the_same_id_takes_it() -> None:
    found = _derive("old/MORT.AGE--Y0__SEX--F")

    assert found.rows == {"old/MORT.AGE--Y0__SEX--F": "undata/who/MORT.AGE--Y0__SEX--F"}
    assert (found.twins, found.reordered) == (1, 0)


def test_a_variable_takes_the_un_variable_holding_its_pairs_in_another_order() -> None:
    found = _derive("old/MORT.SEX--F__AGE--Y0")

    assert found.rows == {"old/MORT.SEX--F__AGE--Y0": "undata/who/MORT.AGE--Y0__SEX--F"}
    assert (found.twins, found.reordered) == (0, 1)


@pytest.mark.parametrize(
    "variable",
    [
        # The UN spells the value otherwise.
        "old/MORT.SEX--FEMALE",
        # The UN names the dimension otherwise.
        "old/MORT.GENDER--F",
        # The UN holds no variable of these pairs, though it holds each pair.
        "old/MORT",
        # MORT starts the code, which is another series.
        "old/MORT_ALL.SEX--F",
    ],
)
def test_a_variable_with_no_un_variable_of_equal_pairs_has_no_successor(variable: str) -> None:
    found = _derive(variable)

    assert found.rows == {variable: None}
    assert found.unmatched == 1


def test_a_row_names_no_successor_for_an_excluded_series_and_dfo_keeps_its_own() -> None:
    found = _derive(
        "old/ODA.DONOR--X",
        "old/MORT.SEX--F",
        "other/MORT.SEX--F",
        dfo=("old/MORT.SEX--F",),
    )

    # A variable DFO holds has no row, and one outside the prefix is not superseded.
    assert found.rows == {"old/ODA.DONOR--X": None}
    assert (found.series_excluded, found.held_by_dfo) == (1, 1)


def test_two_un_variables_of_equal_pairs_give_no_successor_and_are_named() -> None:
    first, second = "undata/who/X.A--1__B--2__C--3", "undata/who/X.C--3__B--2__A--1"
    mirror = un_mirror({dcid: un_variable(MORT_SERIES, name=dcid) for dcid in (first, second)})

    found = derive_successors(
        _registry(),
        withheld_variables=frozenset({"old/X.B--2__A--1__C--3", "old/X.A--1__B--2__C--3"}),
        dfo_variables=frozenset(),
        mirror=mirror,
    )

    # A base id equal to one of them is no exception: the declared rule picks neither.
    assert found.rows == {"old/X.A--1__B--2__C--3": None, "old/X.B--2__A--1__C--3": None}
    assert found.ambiguous == {
        "old/X.A--1__B--2__C--3": (first, second),
        "old/X.B--2__A--1__C--3": (first, second),
    }


def test_a_successor_the_un_tables_do_not_admit_is_named() -> None:
    mirror = un_mirror(
        {"undata/who/MORT": un_variable(MORT_SERIES, name="m")},
        excluded={"undata/who/ODA": ODA_SERIES},
    )
    rows = {
        "old/MORT": "undata/who/MORT",
        "old/ODA": "undata/who/ODA",
        "old/X": "undata/who/X",
        "old/Y": None,
    }

    assert unheld_successors(rows, mirror) == ["old/ODA", "old/X"]
