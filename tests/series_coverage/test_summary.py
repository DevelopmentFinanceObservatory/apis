"""Pins `CoverageSummary`, the per-variable facet fold `CoverageIndex` builds at load."""

from __future__ import annotations

from code_index.types import PartyRow
from series_coverage.index import CoverageIndex
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from series_coverage.summary import FacetSummary, PartyCounts
from tests.series_coverage.support import (
    DEU,
    ETH,
    FACET_ACTUALS,
    FACET_METADATA,
    FACET_PROJECTED,
    FRA,
    KEN,
    V1,
    V2,
    V3,
    V4,
    V5,
    ZWE,
    build_test_index,
    coverage_table,
)

_FLOW_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")


def test_a_variable_summary_folds_each_facet_of_the_index() -> None:
    index = build_test_index()

    summary = index.facet_summary.variable(V1)

    assert summary is not None
    assert summary.facets == (
        FacetSummary(
            facet_id=FACET_PROJECTED,
            metadata=FACET_METADATA[FACET_PROJECTED],
            first_year=2023,
            last_year=2023,
            observations=1,
            parties={"donor": frozenset({FRA}), "recipient": frozenset({KEN})},
        ),
        FacetSummary(
            facet_id=FACET_ACTUALS,
            metadata=FACET_METADATA[FACET_ACTUALS],
            first_year=2018,
            last_year=2022,
            observations=10,
            parties={"donor": frozenset({FRA, DEU}), "recipient": frozenset({KEN, ETH})},
        ),
    )
    assert (summary.first_year, summary.last_year, summary.observations) == (2018, 2023, 11)
    assert summary.provenances == {"provenance/OECDCreditorReportingSystem"}
    # V4 has role names but no rows; an unknown variable has neither.
    assert index.facet_summary.variable(V4) is None
    assert index.facet_summary.variable("NoSuchVariable") is None


def test_facet_spans_agree_with_the_summary_for_every_variable() -> None:
    """The summary and `facet_spans` with no binding read the same rows."""
    index = build_test_index()
    variables = [V1, V2, V3, V5]

    folded = {
        (variable, facet.facet_id): (facet.observations, facet.first_year, facet.last_year)
        for variable in variables
        if (summary := index.facet_summary.variable(variable)) is not None
        for facet in summary.facets
    }

    spans = index.facet_spans(variables=variables)
    assert folded == {
        key: (span.n_obs, span.first_year, span.last_year) for key, span in spans.items()
    }


def test_party_counts_split_the_union_of_facets_by_code_index_kind() -> None:
    africa_total, africa_residual = "DACAfrica", "DACAfricaUnallocated"
    table, horizon = coverage_table(
        [
            ("V-a", "f-1", FRA, KEN, [2020]),
            ("V-a", "f-1", FRA, africa_total, [2020]),
            ("V-a", "f-2", DEU, KEN, [2021]),
            ("V-a", "f-2", DEU, africa_residual, [2021]),
            ("V-a", "f-2", DEU, ZWE, [2021]),
        ]
    )
    index = CoverageIndex(table, {"V-a": _FLOW_ROLES}, horizon=horizon)
    parties = {
        FRA: PartyRow(dcid=FRA, kind="party"),
        DEU: PartyRow(dcid=DEU, kind="party"),
        KEN: PartyRow(dcid=KEN, kind="party"),
        africa_total: PartyRow(dcid=africa_total, kind="total", totals="africa"),
        africa_residual: PartyRow(dcid=africa_residual, kind="residual"),
    }

    summary = index.facet_summary.variable("V-a")

    assert summary is not None
    # KEN sits in both facets and counts once; ZWE is not listed in the code index.
    assert summary.party_counts(parties=parties) == {
        "donor": PartyCounts(parties=2),
        "recipient": PartyCounts(parties=1, totals=1, residual=1, unclassified=1),
    }
    assert summary.party_counts(parties=parties, facet_ids={"f-1"}) == {
        "donor": PartyCounts(parties=1),
        "recipient": PartyCounts(parties=1, totals=1),
    }
    assert summary.party_counts(parties=parties, facet_ids=set()) == {}


def test_a_one_role_variable_has_no_parties_in_the_role_it_lacks() -> None:
    table, horizon = coverage_table(
        [
            ("V-recipient", "f-1", ABSENT_ENTITY, KEN, [2020]),
            ("V-recipient", "f-1", ABSENT_ENTITY, ETH, [2021]),
            ("V-both", "f-1", FRA, KEN, [2020]),
        ]
    )
    index = CoverageIndex(
        table,
        {
            "V-recipient": RoleNames(donor=None, recipient="financingRecipient"),
            "V-both": _FLOW_ROLES,
        },
        horizon=horizon,
    )

    summary = index.facet_summary.variable("V-recipient")

    assert summary is not None
    assert [facet.parties for facet in summary.facets] == [{"recipient": frozenset({KEN, ETH})}]
    assert summary.party_counts(parties={}) == {"recipient": PartyCounts(unclassified=2)}


def test_rows_out_of_order_fold_into_one_entry_per_facet() -> None:
    """Rows interleaving variables and facets fold as the index groups them, and a facet
    the build gives no metadata reads as all `None`."""
    table, horizon = coverage_table(
        [
            ("V-b", "f-2", FRA, KEN, [2019]),
            ("V-a", "f-1", FRA, KEN, [2020, 2021]),
            ("V-b", "f-1", DEU, ETH, [2015]),
            ("V-a", "f-1", DEU, KEN, [2010]),
            ("V-b", "f-2", DEU, KEN, [2022]),
        ]
    )
    index = CoverageIndex(
        table,
        {"V-a": _FLOW_ROLES, "V-b": _FLOW_ROLES},
        {"f-1": FacetMetadata(provenance_id="provenance/A", unit="CurrencyUSD")},
        horizon=horizon,
    )

    a = index.facet_summary.variable("V-a")
    b = index.facet_summary.variable("V-b")

    assert a is not None
    assert b is not None
    assert [(f.facet_id, f.first_year, f.last_year, f.observations) for f in a.facets] == [
        ("f-1", 2010, 2021, 3)
    ]
    assert [(f.facet_id, f.first_year, f.last_year, f.observations) for f in b.facets] == [
        ("f-1", 2015, 2015, 1),
        ("f-2", 2019, 2022, 2),
    ]
    assert b.facets[1].metadata == FacetMetadata()
    assert b.facets[1].parties == {"donor": frozenset({FRA, DEU}), "recipient": frozenset({KEN})}
