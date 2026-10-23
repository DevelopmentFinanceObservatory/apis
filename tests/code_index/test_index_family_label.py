"""Test `index_family_label` over every family of the fixture index.

An index family's label names only what every one of its rows shares, so the code of a slot
the family spans never enters it, whichever row a sample happens to hold."""

from __future__ import annotations

import re

import pytest

from code_index import IndexFamily
from code_index.labels import index_family_label
from code_index.shared import get_measure_index
from tests.code_index.support import INDEX_FIXTURE, load_index

_INDEX = load_index(INDEX_FIXTURE)
_FAMILIES = [family for families in _INDEX.families_by_five_tuple.values() for family in families]


def _family_id(family: IndexFamily) -> str:
    five_tuple = family.five_tuple
    ratio = "/ratio" if five_tuple.measurement_denominator else ""
    return f"{five_tuple.measured_property}/{five_tuple.population_type}{ratio}/{family.datasets}"


def _label(family: IndexFamily) -> str:
    return index_family_label(
        family,
        index=_INDEX,
        measure_index=get_measure_index(),
        denominator_noun="GNI",
        population_type_label="unlabelled population",
    )


@pytest.mark.parametrize("family", _FAMILIES, ids=_family_id)
def test_a_family_label_names_no_code_its_rows_differ_on(family):
    label = _label(family).lower()
    for prop in family.properties:
        values = {row.constraint_properties.get(prop) for row in family.rows}
        if len(values) == 1:
            continue
        for value in values - {None}:
            row = _INDEX.code_by_dcid(value)
            if row is None:
                continue
            assert not re.search(rf"\b{re.escape(row.label.lower())}\b", label), (
                f"{label!r} names {prop}={value}, which the family's rows differ on"
            )


@pytest.mark.parametrize("family", _FAMILIES, ids=_family_id)
def test_every_family_leads_with_its_population_and_a_declared_measure_label(family):
    """Each family's label names its population type, then the declared partition label
    over the measure axes its rows share, so none falls back to its population type alone."""
    head = _label(family).split(", ")[0].removesuffix(" as a share of GNI")
    assert head.startswith("Unlabelled population ")
    assert head != "Unlabelled population"
