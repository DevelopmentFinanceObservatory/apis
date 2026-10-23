"""Pins where an example may be offered: base places from the origin
catalogue, a role's axis from the coverage index, and the refusals."""

from __future__ import annotations

import pytest

from search.catalogue.examples import Example
from search.catalogue.presence import PresenceRefusal, example_places
from tests.origin_catalogue.support import CENSUS_POVERTY, CRS_VARIABLE, GDP, POPULATION
from tests.search_catalogue.support import origin_snapshot
from tests.series_coverage.support import V1, build_test_index


@pytest.fixture
def sources(tmp_path):
    return origin_snapshot(tmp_path / "origin"), build_test_index()


def _example(stat_var: str, place: str, role=None) -> Example:
    return Example(id=0, template="x {place}", stat_var=stat_var, place=place, role=role)


def test_a_base_variable_is_offered_at_its_countries_only(sources):
    catalogue, coverage = sources
    # GDP's bits cover KEN, USA and nuts/DE11; the NUTS region is not a country.
    assert example_places(_example(GDP, "country/USA"), catalogue=catalogue, coverage=coverage) == {
        "country/KEN",
        "country/USA",
    }


def test_a_coverage_variable_is_offered_on_its_role_axis(sources):
    catalogue, coverage = sources
    to = example_places(
        _example(V1, "country/KEN", "recipient"), catalogue=catalogue, coverage=coverage
    )
    by = example_places(
        _example(V1, "country/FRA", "donor"), catalogue=catalogue, coverage=coverage
    )
    assert to == {"country/KEN", "country/ETH"}
    assert by == {"country/FRA", "country/DEU"}


@pytest.mark.parametrize(
    ("example", "reason"),
    [
        # A DFO variable outside the coverage index: no source knows its places.
        (_example(CRS_VARIABLE, "country/KEN"), "no_presence_source"),
        # A withheld variable is not in the base table.
        (_example(CENSUS_POVERTY, "country/USA"), "no_presence_source"),
        (_example(V1, "country/KEN"), "role_required"),
        (_example(POPULATION, "country/KEN", "recipient"), "role_without_coverage"),
        (_example(V1, "country/USA", "recipient"), "default_place_without_data"),
        (_example(POPULATION, "country/ETH"), "default_place_without_data"),
    ],
)
def test_refusals(sources, example: Example, reason: str):
    catalogue, coverage = sources
    with pytest.raises(PresenceRefusal) as refused:
        example_places(example, catalogue=catalogue, coverage=coverage)
    assert refused.value.reason == reason
