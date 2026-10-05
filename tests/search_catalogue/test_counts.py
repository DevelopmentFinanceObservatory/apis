"""Pins the catalogue counts over the origin catalogue sample build and the
coverage fixture: global figures, per-country figures, and the CRS filter
on the aid block."""

from __future__ import annotations

from dataclasses import replace

import pytest

from origin_catalogue.catalogue import BaseVariable
from search.catalogue.artifact import AidCounts, PlaceCounts, YearRange
from search.catalogue.counts import CRS_PROVENANCE, catalogue_counts
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.origin_catalogue.support import EUROSTAT_GDP, GDP, SCAN_PLACES, WDI
from tests.search_catalogue.support import coverage_fixture, origin_snapshot
from tests.series_coverage.support import FACET_ACTUALS, V1, V5, coverage_table


@pytest.fixture
def counted(tmp_path):
    return catalogue_counts(
        catalogue=origin_snapshot(tmp_path / "origin"),
        coverage=coverage_fixture(tmp_path / "coverage").index,
    )


def test_global_counts(counted):
    counts = counted.counts
    # CRS's OECD source, the World Bank, Eurostat and WHO on the UN.
    assert counts.publishers == 4
    # KEN and USA from base bits; KEN, ETH, ZWE, FRA, DEU from coverage rows.
    assert counts.countries == 6
    # UNICEF is a donor too: the aid block does not filter to countries.
    assert counts.aid == AidCounts(donors=3, recipients=3, years=YearRange(first=1975, last=2023))


def test_places_are_the_counted_countries(counted):
    assert sorted(counted.places) == [
        "country/DEU",
        "country/ETH",
        "country/FRA",
        "country/KEN",
        "country/USA",
        "country/ZWE",
    ]


def test_a_recipient_with_base_data(counted):
    # Two base variables, both held there by the World Bank alone, and four
    # coverage variables (OECD). Eurostat holds GDP at its NUTS region only.
    assert counted.places["country/KEN"] == PlaceCounts(
        publishers=2,
        variables=6,
        aid=AidCounts(donors=2, recipients=0, years=YearRange(first=2010, last=2023)),
    )


def test_a_donor(counted):
    assert counted.places["country/FRA"] == PlaceCounts(
        publishers=1,
        variables=3,
        aid=AidCounts(donors=0, recipients=2, years=YearRange(first=2010, last=2023)),
    )


def test_a_country_with_base_data_only(counted):
    assert counted.places["country/USA"] == PlaceCounts(
        publishers=1, variables=2, aid=AidCounts(donors=0, recipients=0, years=None)
    )


def test_a_publisher_counts_only_where_its_own_provenance_holds_data(tmp_path):
    snapshot = origin_snapshot(tmp_path)
    kenya, usa = (1 << SCAN_PLACES.index(place) for place in ("country/KEN", "country/USA"))
    tables = replace(
        snapshot.tables,
        base_variables={GDP: BaseVariable(provenances=(WDI, EUROSTAT_GDP), places=kenya | usa)},
        provenance_places={WDI: kenya, EUROSTAT_GDP: usa},
    )
    table, horizon = coverage_table([(V1, FACET_ACTUALS, "country/FRA", "country/ETH", [2018])])
    coverage = CoverageIndex(
        table,
        {V1: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        {FACET_ACTUALS: FacetMetadata(provenance_id=CRS_PROVENANCE)},
        horizon=horizon,
    )
    places = catalogue_counts(catalogue=replace(snapshot, tables=tables), coverage=coverage).places
    assert (places["country/KEN"].publishers, places["country/USA"].publishers) == (1, 1)
    assert (places["country/KEN"].variables, places["country/USA"].variables) == (1, 1)


def test_the_aid_block_counts_only_variables_with_a_crs_facet(tmp_path):
    debt_facet = "debt-facet"
    rows = [
        (V1, FACET_ACTUALS, "country/FRA", "country/KEN", [2018, 2019]),
        (V5, debt_facet, "country/CHN", "country/ZMB", [2020, 2031]),
    ]
    table, horizon = coverage_table(rows)
    roles = RoleNames(donor="financingProvider", recipient="financingRecipient")
    coverage = CoverageIndex(
        table,
        {V1: roles, V5: roles},
        {
            FACET_ACTUALS: FacetMetadata(provenance_id=CRS_PROVENANCE),
            debt_facet: FacetMetadata(
                provenance_id="provenance/WorldBankInternationalDebtStatistics"
            ),
        },
        horizon=horizon,
    )
    counted = catalogue_counts(catalogue=origin_snapshot(tmp_path), coverage=coverage)
    assert counted.counts.aid == AidCounts(
        donors=1, recipients=1, years=YearRange(first=2018, last=2019)
    )
    # Debt rows still name countries with data.
    assert {"country/CHN", "country/ZMB"} <= set(counted.places)
    assert counted.places["country/ZMB"].aid == AidCounts(donors=0, recipients=0, years=None)
    assert counted.places["country/ZMB"].variables == 1
