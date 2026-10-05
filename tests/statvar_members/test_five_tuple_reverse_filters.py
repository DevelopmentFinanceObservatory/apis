"""Direct coverage for `five_tuple_reverse_filters` and `five_tuple_arcs_match`
(`statvar_members.enumerate`): the two halves of a five-tuple's own
`populationType`/`measuredProperty`/`statType`/`measurementQualifier`/
`measurementDenominator` match -- the server-side filter dict, and the
client-side check for all five fields, including the ones a filter cannot
always express absence for. Otherwise exercised only indirectly, through
`_reverse_query`/`_matches_five_tuple`."""

from __future__ import annotations

import pytest

from statvar_members.enumerate import five_tuple_arcs_match, five_tuple_reverse_filters
from statvar_members.types import UNRESOLVED_NODE


def test_every_field_given_produces_every_filter():
    filters = five_tuple_reverse_filters(
        population_type="Person",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator="Count_Person",
    )
    assert filters == {
        "populationType": "Person",
        "statType": "measuredValue",
        "measurementQualifier": "Nominal",
        "measurementDenominator": "Count_Person",
    }


def test_absent_fields_are_omitted_not_fabricated():
    filters = five_tuple_reverse_filters(
        population_type="Person",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
    )
    assert filters == {"populationType": "Person"}


def _arcs_match(arcs, **kwargs):
    kwargs.setdefault("population_type", None)
    kwargs.setdefault("measured_property", None)
    kwargs.setdefault("stat_type", None)
    return five_tuple_arcs_match(arcs, **kwargs)


def test_arcs_match_requires_qualifier_absence_when_ask_names_none():
    arcs = {"measurementQualifier": ()}
    assert _arcs_match(arcs, measurement_qualifier=None, measurement_denominator=None)


def test_arcs_match_rejects_a_qualifier_the_ask_does_not_name():
    arcs = {"measurementQualifier": ("Nominal",)}
    assert not _arcs_match(arcs, measurement_qualifier=None, measurement_denominator=None)


def test_arcs_match_requires_denominator_absence_when_ask_names_none():
    arcs = {"measurementDenominator": ()}
    assert _arcs_match(arcs, measurement_qualifier=None, measurement_denominator=None)


def test_arcs_match_rejects_a_denominator_the_ask_does_not_name():
    # This is exactly what a node_subjects reverse-arc filter cannot express
    # server-side: an absent measurement_denominator ask has no dcid to
    # filter on, so a member carrying ANY denominator arc still passes the
    # reverse query and must be excluded here instead.
    arcs = {"measurementDenominator": ("Count_Person",)}
    assert not _arcs_match(arcs, measurement_qualifier=None, measurement_denominator=None)


def test_arcs_match_accepts_the_named_concrete_denominator():
    arcs = {"measurementDenominator": ("Count_Person",)}
    assert _arcs_match(arcs, measurement_qualifier=None, measurement_denominator="Count_Person")


def test_arcs_match_rejects_a_different_concrete_denominator():
    arcs = {"measurementDenominator": ("Count_Person_Female",)}
    assert not _arcs_match(arcs, measurement_qualifier=None, measurement_denominator="Count_Person")


def test_denominator_unresolved_ask_matches_only_the_unresolved_marker():
    assert _arcs_match(
        {"measurementDenominator": (UNRESOLVED_NODE,)},
        measurement_qualifier=None,
        measurement_denominator=None,
        denominator_unresolved=True,
    )
    assert not _arcs_match(
        {"measurementDenominator": ("Count_Person",)},
        measurement_qualifier=None,
        measurement_denominator=None,
        denominator_unresolved=True,
    )
    assert not _arcs_match(
        {"measurementDenominator": ()},
        measurement_qualifier=None,
        measurement_denominator=None,
        denominator_unresolved=True,
    )


def test_arcs_match_requires_population_type_absence_when_ask_names_none():
    arcs = {"populationType": ("Person",)}
    assert not _arcs_match(arcs, measurement_qualifier=None, measurement_denominator=None)


@pytest.mark.parametrize(
    ("arcs", "named_value"),
    [
        pytest.param(
            {"populationType": ("Person",)}, {"population_type": "Person"}, id="population-type"
        ),
        pytest.param(
            {"measuredProperty": ("count",)}, {"measured_property": "count"}, id="measured-property"
        ),
    ],
)
def test_arcs_match_accepts_the_named_five_tuple_value(arcs, named_value):
    assert _arcs_match(
        arcs,
        measurement_qualifier=None,
        measurement_denominator=None,
        **named_value,
    )


def test_arcs_match_rejects_a_different_stat_type():
    arcs = {"statType": ("measuredValue",)}
    assert not _arcs_match(
        arcs,
        stat_type="growthRate",
        measurement_qualifier=None,
        measurement_denominator=None,
    )
