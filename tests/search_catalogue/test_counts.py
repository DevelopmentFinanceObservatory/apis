"""Pins the catalogue counts over the origin catalogue sample build and the
coverage fixture: global figures, per-country figures, and the declared
datasets' filter on the aid block."""

from __future__ import annotations

from dataclasses import replace

import pytest

from code_index.types import PartyRow
from origin_catalogue.catalogue import BaseVariable, UnTables
from search.catalogue.artifact import AidCounts, PlaceCounts, YearRange
from search.catalogue.counts import catalogue_counts
from series_coverage.index import CoverageIndex
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.origin_catalogue.support import (
    CRS,
    EUROSTAT_GDP,
    GDP,
    SCAN_PLACES,
    UN_WHO,
    WDI,
    sample_un_mirror,
)
from tests.search_catalogue.support import origin_snapshot
from tests.series_coverage.support import (
    FACET_ACTUALS,
    FACET_METADATA,
    V1,
    V2,
    V3,
    V5,
    build_test_index,
    coverage_table,
)

AID_TABLE = "provenance/AidTable"


@pytest.fixture
def counted(tmp_path):
    return catalogue_counts(
        catalogue=replace(
            origin_snapshot(tmp_path / "origin"), un=UnTables(mirror=sample_un_mirror())
        ),
        coverage=build_test_index(),
        datasets={metadata.provenance_id for metadata in FACET_METADATA.values()},
        parties={},
    )


def test_provenance_ids_group_by_origin(counted):
    # CRS is the DFO provenance; WDI, Eurostat (base) and WHO (UN) go with Data Commons.
    assert counted.provenances.dfo == [CRS]
    assert counted.provenances.data_commons == sorted([WDI, EUROSTAT_GDP, UN_WHO])


def test_global_counts(counted):
    counts = counted.counts
    # Two base and two DFO variables sharing Count_Person, four coverage
    # variables and two UN variables.
    assert counts.indicators == 9
    # The World Bank's provenance counts as its parent Dataset.
    assert counts.datasets == 4
    # CRS's OECD source, the World Bank, Eurostat and WHO on the UN.
    assert counts.publishers == 4
    # KEN and USA from base bits; KEN, ETH, ZWE, FRA, DEU from coverage rows.
    assert counts.countries == 6
    # UNICEF is a donor too: the aid block does not filter to countries.
    assert counts.aid == AidCounts(donors=3, recipients=3, years=YearRange(first=1975, last=2023))


def test_counts_split_by_origin(counted):
    split = counted.counts.by_origin
    # CRS is the one DFO provenance. WDI under its Dataset, Eurostat and WHO on
    # the UN are the Data Commons rows.
    assert (split.dfo.datasets, split.dfo.publishers) == (1, 1)
    assert (split.data_commons.datasets, split.data_commons.publishers) == (3, 3)
    # The fixture has no source shared across the groups, so datasets and publishers sum to the
    # global figure.
    assert split.dfo.datasets + split.data_commons.datasets == counted.counts.datasets
    assert split.dfo.publishers + split.data_commons.publishers == counted.counts.publishers
    # Two DFO and four coverage variables (all on CRS) against two base and two UN ones.
    # Count_Person is both a base and a DFO variable, so it counts in each group.
    assert (split.dfo.indicators, split.data_commons.indicators) == (6, 4)
    assert split.dfo.indicators + split.data_commons.indicators == counted.counts.indicators + 1


def test_a_coverage_variable_under_a_base_provenance_counts_as_data_commons(tmp_path):
    counted = catalogue_counts(
        catalogue=replace(origin_snapshot(tmp_path), un=UnTables(mirror=sample_un_mirror())),
        coverage=build_test_index(provenance=WDI),
        datasets=set(),
        parties={},
    )
    split = counted.counts.by_origin
    # The DFO group keeps its two DFO variables. The four coverage variables join the two
    # base and two UN ones.
    assert (split.dfo.indicators, split.data_commons.indicators) == (2, 8)
    assert counted.counts.indicators == 9


def test_without_the_un_system_origin_the_un_variables_and_publisher_drop_out(tmp_path):
    enabled = replace(origin_snapshot(tmp_path), un=UnTables(mirror=sample_un_mirror()))
    disabled = replace(enabled, un_system_enabled=False)
    kwargs = {"coverage": build_test_index(), "datasets": set(), "parties": {}}
    on = catalogue_counts(catalogue=enabled, **kwargs).counts
    off = catalogue_counts(catalogue=disabled, **kwargs).counts
    assert off.indicators == on.indicators - 2
    assert off.by_origin.data_commons.indicators == on.by_origin.data_commons.indicators - 2
    # WHO is the UN's publisher.
    assert off.publishers == on.publishers - 1
    assert off.by_origin.data_commons.publishers == on.by_origin.data_commons.publishers - 1
    # A provenance the catalogue does not admit is in neither group.
    off_provenances = catalogue_counts(catalogue=disabled, **kwargs).provenances
    assert UN_WHO not in off_provenances.dfo + off_provenances.data_commons
    assert off_provenances.data_commons == sorted([WDI, EUROSTAT_GDP])


def test_a_publisher_with_a_dfo_and_a_base_provenance_counts_in_both_groups(tmp_path):
    snapshot = origin_snapshot(tmp_path)
    provenances = {
        **snapshot.tables.provenances,
        WDI: replace(
            snapshot.tables.provenances[WDI], source=snapshot.tables.provenances[CRS].source
        ),
    }
    counted = catalogue_counts(
        catalogue=replace(snapshot, tables=replace(snapshot.tables, provenances=provenances)),
        coverage=build_test_index(),
        datasets=set(),
        parties={},
    )
    split = counted.counts.by_origin
    assert (split.dfo.publishers, split.data_commons.publishers) == (1, 3)
    assert counted.counts.publishers == 3


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
        parties={},
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
        parties={},
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
        {FACET_ACTUALS: FacetMetadata(provenance_id=WDI)},
        horizon=horizon,
    )
    counted = catalogue_counts(
        catalogue=replace(snapshot, tables=tables),
        coverage=coverage,
        datasets={WDI},
        parties={},
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
        parties={
            "GroupOfSeven": PartyRow(
                dcid="GroupOfSeven", kind="total", published=((AID_TABLE, "origin"),)
            ),
            "DACAfrica": PartyRow(
                dcid="DACAfrica", kind="total", published=((AID_TABLE, "destination"),)
            ),
        },
    )
    assert counted.counts.aid == AidCounts(
        donors=1, recipients=1, years=YearRange(first=2018, last=2020)
    )
    assert counted.places["country/KEN"].aid.donors == 1
    assert counted.places["country/FRA"].aid.recipients == 1


def test_a_party_that_totals_others_on_one_axis_counts_on_the_other(tmp_path):
    """The IMF trust funds total other donors in the table that publishes them as a donor and
    are a recipient of their own in the one that publishes them as a recipient: the aid block
    leaves them out as a donor and counts them as a recipient."""
    receipts = "provenance/ReceiptsTable"
    receipts_facet = "facet-receipts"
    table, horizon = coverage_table(
        [
            (V1, FACET_ACTUALS, "IMFTrustFunds", "country/KEN", [2018]),
            (V1, receipts_facet, "country/FRA", "IMFTrustFunds", [2019]),
        ]
    )
    coverage = CoverageIndex(
        table,
        {V1: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        {
            FACET_ACTUALS: FacetMetadata(provenance_id=AID_TABLE),
            receipts_facet: FacetMetadata(provenance_id=receipts),
        },
        horizon=horizon,
    )
    trust_funds = PartyRow(
        dcid="IMFTrustFunds",
        kind="party",
        published=((AID_TABLE, "origin"), (receipts, "destination")),
        dataset_kinds=((AID_TABLE, "total"),),
    )

    counted = catalogue_counts(
        catalogue=origin_snapshot(tmp_path),
        coverage=coverage,
        datasets={AID_TABLE, receipts},
        parties={"IMFTrustFunds": trust_funds},
    )

    assert counted.counts.aid == AidCounts(
        donors=1, recipients=2, years=YearRange(first=2018, last=2019)
    )
    assert counted.places["country/FRA"].aid.recipients == 1
    assert counted.places["country/KEN"].aid.donors == 0


def test_the_trust_funds_count_as_a_donor_only_in_the_rows_where_they_total_no_one(tmp_path):
    """The IMF trust funds total other donors in one aid table and give aid of their own in
    another. A row of the first counts them nowhere; a row of the second counts them as a
    donor, and its recipient as a recipient of theirs."""
    own_flows = "provenance/OwnFlowsTable"
    own_facet = "facet-own"
    table, horizon = coverage_table(
        [
            (V1, FACET_ACTUALS, "IMFTrustFunds", "country/KEN", [2018]),
            (V1, own_facet, "IMFTrustFunds", "country/GHA", [2019]),
        ]
    )
    coverage = CoverageIndex(
        table,
        {V1: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        {
            FACET_ACTUALS: FacetMetadata(provenance_id=AID_TABLE),
            own_facet: FacetMetadata(provenance_id=own_flows),
        },
        horizon=horizon,
    )
    trust_funds = PartyRow(
        dcid="IMFTrustFunds",
        kind="party",
        published=((AID_TABLE, "origin"), (own_flows, "origin")),
        dataset_kinds=((AID_TABLE, "total"),),
    )

    counted = catalogue_counts(
        catalogue=origin_snapshot(tmp_path),
        coverage=coverage,
        datasets={AID_TABLE, own_flows},
        parties={"IMFTrustFunds": trust_funds},
    )

    assert counted.counts.aid.donors == 1
    assert counted.places["country/GHA"].aid.donors == 1
    assert counted.places["country/KEN"].aid.donors == 0


def test_the_trust_funds_count_as_a_donor_in_a_table_the_index_does_not_list_them_in(tmp_path):
    """A coverage build newer than the index holds the trust funds in a table the index does
    not list them in. Their kind there is unknown, so they count as a donor, as a party the
    index does not list does, and borrow no total from the table that lists them."""
    new_table = "provenance/NewTable"
    new_facet = "facet-new"
    table, horizon = coverage_table([(V1, new_facet, "IMFTrustFunds", "country/GHA", [2019])])
    coverage = CoverageIndex(
        table,
        {V1: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        {new_facet: FacetMetadata(provenance_id=new_table)},
        horizon=horizon,
    )
    trust_funds = PartyRow(
        dcid="IMFTrustFunds",
        kind="party",
        published=((AID_TABLE, "origin"), (AID_TABLE, "destination")),
        dataset_kinds=((AID_TABLE, "total"),),
    )

    counted = catalogue_counts(
        catalogue=origin_snapshot(tmp_path),
        coverage=coverage,
        datasets={new_table},
        parties={"IMFTrustFunds": trust_funds},
    )

    assert counted.counts.aid.donors == 1


def test_a_facet_without_a_provenance_adds_to_neither_origin_group(tmp_path):
    table, horizon = coverage_table([(V1, FACET_ACTUALS, "country/FRA", "country/ETH", [2018])])
    coverage = CoverageIndex(
        table,
        {V1: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        {FACET_ACTUALS: FacetMetadata(provenance_id=None)},
        horizon=horizon,
    )
    snapshot = origin_snapshot(tmp_path)
    counted = catalogue_counts(catalogue=snapshot, coverage=coverage, datasets=set(), parties={})
    tables = snapshot.tables
    split = counted.counts.by_origin
    assert split.dfo.indicators == len(tables.dfo_variables)
    assert split.data_commons.indicators == len(tables.base_variables)
