"""Checks that stale-provenance rate rows cannot determine converted values."""

from __future__ import annotations

import pytest

from materialise.rate_tables import RateTable, RateTableCache, load_table

from ._rate_table_support import RateTableServer, rate_csv, rate_row
from .conftest import FakeGraphReader

_STAT_VAR = "Deflator_EconomicProduction_ConversionRecipeIMF"
_CURRENT_PROVENANCE = "provenance/ONEConversionTablesIMF"
_STALE_PROVENANCE = "provenance/ONEConversionTablesIMFStale"


def _rate_row(
    *,
    entity: str,
    year: str,
    value: str,
    method: str = "NotApplicable",
    facet_id: str,
    provenance: str = _CURRENT_PROVENANCE,
) -> str:
    return rate_row(
        stat_var=_STAT_VAR,
        entity=entity,
        year=year,
        value=value,
        provenance=provenance,
        method=method,
        facet_id=facet_id,
    )


def _graph() -> FakeGraphReader:
    return FakeGraphReader(arcs={_CURRENT_PROVENANCE: {"lastDataRefreshDate": ("2026-09-01",)}})


def _load(csv_body: str) -> RateTable:
    return load_table(
        kind="deflator",
        stat_var=_STAT_VAR,
        provenance_id=_CURRENT_PROVENANCE,
        sdmx=RateTableServer({_STAT_VAR: csv_body}).client(),
        graph=_graph(),
        cache=RateTableCache(),
    )


def test_a_lone_stale_row_is_dropped_as_a_coverage_gap_not_an_admitted_value():
    # A lone stale row must be filtered out as a coverage gap.
    csv = rate_csv(
        _rate_row(
            entity="country/FRA",
            year="2020",
            value="999.0",
            facet_id="facet-stale",
            provenance=_STALE_PROVENANCE,
        ),
    )
    result = _load(csv)

    assert result.table.num_rows == 0


def test_a_stale_actual_never_outranks_a_current_projection():
    # The current-provenance projection must outrank the stale actual value 999.
    csv = rate_csv(
        _rate_row(
            entity="country/FRA",
            year="2020",
            value="999.0",
            facet_id="facet-stale",
            provenance=_STALE_PROVENANCE,
        ),
        _rate_row(
            entity="country/FRA",
            year="2020",
            value="108.0",
            facet_id="facet-current",
            method="IMFProjection",
        ),
    )
    result = _load(csv)

    rows = result.table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["entity"] == "country/FRA"
    assert rows[0]["value"] == pytest.approx(108.0)
    assert rows[0]["facet"] == "projected"
