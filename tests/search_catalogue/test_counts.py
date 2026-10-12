"""Pins the catalogue counts over the origin catalogue sample build and the
coverage fixture: global figures, per-country figures, and the declared
datasets' filter on the aid block."""

from __future__ import annotations

from dataclasses import replace

import pytest

from origin_catalogue.catalogue import BaseVariable
from search.catalogue.artifact import AidCounts, PlaceCounts, YearRange
from search.catalogue.counts import catalogue_counts
from series_coverage.index import CoverageIndex
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.origin_catalogue.support import EUROSTAT_GDP, GDP, SCAN_PLACES, WDI
from tests.search_catalogue.support import coverage_fixture, origin_snapshot
from tests.series_coverage.support import (
    FACET_ACTUALS,
    FACET_METADATA,
    V1,
    V2,
    V3,
    V5,
    coverage_table,
)

AID_TABLE = "provenance/AidTable"


@pytest.fixture
def counted(tmp_path):
    return catalogue_counts(
        catalogue=origin_snapshot(tmp_path / "origin"),
        coverage=coverage_fixture(tmp_path / "coverage").index,
        datasets={metadata.provenance_id for metadata in FACET_METADATA.values()},
        totals=(),
    )


def test_global_counts(counted):
    counts = counted.counts
    # Two base and two DFO variables sharing Count_Person, and four coverage
    # variables.
    assert counts.indicators == 7
    # The World Bank's provenance counts as its parent Dataset.
    assert counts.datasets == 4
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
        {FACET_ACTUALS: FacetMetadata(provenance_id=AID_TABLE)},
        horizon=horizon,
    )
    places = catalogue_counts(
        catalogue=replace(snapshot, tables=tables),
        coverage=coverage,
        datasets={AID_TABLE},
        totals=(),
    ).places
    assert (places["country/KEN"].publishers, places["country/USA"].publishers) == (1, 1)
    assert (places["country/KEN"].variables, places["country/USA"].variables) == (1, 1)


def test_the_aid_block_counts_the_rows_of_every_declared_dataset(tmp_path):
    """A one-role table's rows count the role they hold and their years and
    add no entity on the other, and an undeclared provenance's rows stay out
    of the block, on a variable a declared table also publishes (#1161)."""
    totals_table, totals_facet, debt_facet = "provenance/TotalsTable", "totals-facet", "debt-facet"
    receipts_table, receipts_facet = "provenance/ReceiptsTable", "receipts-facet"
    rows = [
        (V1, FACET_ACTUALS, "country/FRA", "country/KEN", [2018, 2019]),
        (V2, totals_facet, "country/FRA", ABSENT_ENTITY, [1961, 2019]),
        (V2, totals_facet, "country/DEU", ABSENT_ENTITY, [1970, 2022]),
        (V3, receipts_facet, ABSENT_ENTITY, "country/ETH", [1965, 1966]),
        (V1, debt_facet, "country/JPN", "country/KEN", [2024]),
        (V5, debt_facet, "country/CHN", "country/ZMB", [2020, 2031]),
    ]
    table, horizon = coverage_table(rows)
    roles = RoleNames(donor="financingProvider", recipient="financingRecipient")
    coverage = CoverageIndex(
        table,
        {
            V1: roles,
            V2: RoleNames(donor="financingProvider", recipient=None),
            V3: RoleNames(donor=None, recipient="financingRecipient"),
            V5: roles,
        },
        {
            FACET_ACTUALS: FacetMetadata(provenance_id=AID_TABLE),
            totals_facet: FacetMetadata(provenance_id=totals_table),
            receipts_facet: FacetMetadata(provenance_id=receipts_table),
            debt_facet: FacetMetadata(provenance_id="provenance/DebtTable"),
        },
        horizon=horizon,
    )
    counted = catalogue_counts(
        catalogue=origin_snapshot(tmp_path),
        coverage=coverage,
        datasets={AID_TABLE, totals_table, receipts_table},
        totals=(),
    )
    assert counted.counts.aid == AidCounts(
        donors=2, recipients=2, years=YearRange(first=1961, last=2022)
    )
    assert counted.places["country/KEN"].aid == AidCounts(
        donors=1, recipients=0, years=YearRange(first=2018, last=2019)
    )
    assert counted.places["country/ETH"].aid == AidCounts(
        donors=0, recipients=0, years=YearRange(first=1965, last=1966)
    )
    assert counted.places["country/FRA"].aid == AidCounts(
        donors=0, recipients=1, years=YearRange(first=1961, last=2019)
    )
    assert counted.places["country/DEU"].aid == AidCounts(
        donors=0, recipients=0, years=YearRange(first=1970, last=2022)
    )
    # Debt rows still name countries with data.
    assert {"country/CHN", "country/ZMB"} <= set(counted.places)
    assert counted.places["country/ZMB"].aid == AidCounts(donors=0, recipients=0, years=None)
    assert counted.places["country/ZMB"].variables == 1


def test_indicators_leave_out_a_withheld_coverage_variable(tmp_path):
    snapshot = origin_snapshot(tmp_path)
    tables = replace(snapshot.tables, withheld_variables=frozenset({V5}))
    table, horizon = coverage_table(
        [
            (V1, FACET_ACTUALS, "country/FRA", "country/KEN", [2018]),
            (V5, FACET_ACTUALS, "country/CHN", "country/ZMB", [2020]),
        ]
    )
    roles = RoleNames(donor="financingProvider", recipient="financingRecipient")
    coverage = CoverageIndex(
        table,
        {V1: roles, V5: roles},
        {FACET_ACTUALS: FacetMetadata(provenance_id=AID_TABLE)},
        horizon=horizon,
    )
    counted = catalogue_counts(
        catalogue=replace(snapshot, tables=tables),
        coverage=coverage,
        datasets={AID_TABLE},
        totals=(),
    )
    # Three base and DFO variables, and V1.
    assert counted.counts.indicators == 4


def test_the_aid_block_leaves_out_the_parties_that_total_others(tmp_path):
    """The G7 as a donor and DAC's Africa total as a recipient count in no aid figure, global
    or at a country, and the rows they hold keep their years."""
    table, horizon = coverage_table(
        [
            (V1, FACET_ACTUALS, "country/FRA", "country/KEN", [2018]),
            (V1, FACET_ACTUALS, "GroupOfSeven", "country/KEN", [2018]),
            (V1, FACET_ACTUALS, "country/FRA", "DACAfrica", [2020]),
        ]
    )
    coverage = CoverageIndex(
        table,
        {V1: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        {FACET_ACTUALS: FacetMetadata(provenance_id=AID_TABLE)},
        horizon=horizon,
    )
    counted = catalogue_counts(
        catalogue=origin_snapshot(tmp_path),
        coverage=coverage,
        datasets={AID_TABLE},
        totals={"GroupOfSeven", "DACAfrica"},
    )
    assert counted.counts.aid == AidCounts(
        donors=1, recipients=1, years=YearRange(first=2018, last=2020)
    )
    assert counted.places["country/KEN"].aid.donors == 1
    assert counted.places["country/FRA"].aid.recipients == 1
