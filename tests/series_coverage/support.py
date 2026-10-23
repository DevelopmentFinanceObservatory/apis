"""Builds coverage fixtures with country and multilateral donors, two facets, and role groups."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from series_coverage.index import CoverageIndex
from series_coverage.loader import checked_index
from series_coverage.schema import (
    DONOR_COLUMN,
    FACET_COLUMN,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)

BUILD_EPOCH_YEAR: Final[int] = 1960
"""The epoch `coverage_table` sizes its horizon from, equal to search's default
`scope_min_year`, the earliest year a search window reaches."""

BUILD_YEAR: Final[int] = 2026
"""The build year of every index `checked_test_index` builds."""

# Two facet ids, matching coverage-prototype.md's "only two distinct
# facet_id values exist in the entire CRS family".
FACET_ACTUALS = "17650800455313554206"
FACET_PROJECTED = "10983524200687618543"

V1 = "FixtureStatVar_Amount_FinancialFlow_Gross_Disbursement_Nominal_Health_StandardGrant"
V2 = "FixtureStatVar_Amount_FinancialFlow_Gross_Disbursement_RealValue_Health_StandardGrant"
V3 = "FixtureStatVar_Amount_FinancialFlow_Net_Disbursement_Nominal_Education_StandardGrant"
# V4 carries no rows at all. A candidate list including it exercises
# filter_candidates' "omitted, not returned empty" contract.
V4 = "FixtureStatVar_Amount_FinancialFlow_Gross_Disbursement_Nominal_WaterSupply_StandardGrant"
# V5 declares a different role-name pair, exercising estimate()'s
# per-role-name-group chunk packing.
V5 = "FixtureStatVar_Amount_Debt_Outstanding_Nominal_ExternalDebt"

FRA = "country/FRA"
DEU = "country/DEU"
UNICEF = "UnitedNationsChildrensFund"
KEN = "country/KEN"
ETH = "country/ETH"
ZWE = "country/ZWE"

ROLE_NAMES: dict[str, RoleNames] = {
    V1: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    V2: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    V3: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    V5: RoleNames(donor="creditor", recipient="debtor"),
}

# Real-shaped, live-matching metadata for the two facet ids above
# (TimeSeries carries no scaling_factor column at all).
FACET_METADATA: dict[str, FacetMetadata] = {
    FACET_ACTUALS: FacetMetadata(
        provenance_id="provenance/OECDCreditorReportingSystem",
        unit="CurrencyUSD",
        observation_period="P1Y",
    ),
    FACET_PROJECTED: FacetMetadata(
        provenance_id="provenance/OECDCreditorReportingSystem",
        unit="CurrencyUSD",
        observation_period="P1Y",
    ),
}

# (variable, facet_id, donor, recipient, years)
_ROWS: list[tuple[str, str, str, str, list[int]]] = [
    (V1, FACET_ACTUALS, FRA, KEN, [2018, 2019, 2020, 2021, 2022]),
    (V1, FACET_ACTUALS, DEU, KEN, [2020, 2021, 2022]),
    (V1, FACET_ACTUALS, FRA, ETH, [2019, 2020]),
    (V1, FACET_PROJECTED, FRA, KEN, [2023]),
    (V2, FACET_ACTUALS, FRA, KEN, [2018, 2019, 2020, 2021, 2022]),
    (V2, FACET_ACTUALS, UNICEF, ZWE, [2005]),
    (V3, FACET_ACTUALS, DEU, ETH, [1975, 1976, 1990]),
    (V3, FACET_ACTUALS, DEU, KEN, [2022]),
    (V5, FACET_ACTUALS, FRA, KEN, [2010, 2011, 2012]),
]


def coverage_table(
    rows: Sequence[tuple[str, str, str, str, Sequence[int]]],
) -> tuple[pa.Table, YearHorizon]:
    """The on-disk coverage table for `(variable, facet_id, donor,
    recipient, years)` rows, and the horizon its year-mask words encode:
    from `BUILD_EPOCH_YEAR` to the latest year any row names."""
    years_seen = [year for *_key, years in rows for year in years]
    horizon = YearHorizon.covering(
        epoch_year=BUILD_EPOCH_YEAR, last_year=max(years_seen, default=BUILD_EPOCH_YEAR)
    )
    masks = [horizon.masks_from_years(years) for *_key, years in rows]
    table = pa.table(
        {
            "variable": [row[0] for row in rows],
            "facet_id": [row[1] for row in rows],
            "donor": [row[2] for row in rows],
            "recipient": [row[3] for row in rows],
            "n_obs": pa.array([len(row[4]) for row in rows], type=pa.uint32()),
            "first_year": pa.array([min(row[4]) for row in rows], type=pa.uint16()),
            "last_year": pa.array([max(row[4]) for row in rows], type=pa.uint16()),
            **{
                name: pa.array([words[word] for words in masks], type=pa.uint64())
                for word, name in enumerate(horizon.mask_columns)
            },
        },
        schema=horizon.schema,
    )
    return table, horizon


def with_repeated_dictionaries(table: pa.Table, *, chunks: int) -> pa.Table:
    """`table` with each key column dictionary-encoded in `chunks` chunks
    whose dictionaries hold every value twice, at codes `i` and `i + n`, in
    ascending order for even chunks and descending for odd ones. Rows
    alternate between a value's two codes. Arrow's `unify_dictionaries`
    collapses the repeats across two or more chunks and keeps them in one."""
    bounds = [table.num_rows * part // chunks for part in range(chunks + 1)]
    columns = []
    for name in table.column_names:
        if name not in ("variable", "facet_id", "donor", "recipient"):
            columns.append(table[name])
            continue
        values = table[name].to_pylist()
        distinct = sorted(set(values))
        encoded = []
        for part, (start, stop) in enumerate(itertools.pairwise(bounds)):
            order = distinct if part % 2 == 0 else distinct[::-1]
            code = {value: i for i, value in enumerate(order)}
            indices = [code[values[row]] + len(order) * (row % 2) for row in range(start, stop)]
            encoded.append(
                pa.DictionaryArray.from_arrays(
                    pa.array(indices, type=pa.int32()), pa.array(order + order, type=pa.string())
                )
            )
        columns.append(pa.chunked_array(encoded, type=pa.dictionary(pa.int32(), pa.string())))
    schema = pa.schema(
        field.with_type(column.type) for field, column in zip(table.schema, columns, strict=True)
    )
    return pa.Table.from_arrays(columns, schema=schema)


def checked_test_index(
    table: pa.Table,
    role_names: Mapping[str, RoleNames],
    facet_metadata: Mapping[str, FacetMetadata],
    *,
    horizon: YearHorizon,
) -> CoverageIndex:
    """`checked_index` over `table` with its series key columns dictionary-encoded, as the
    release's coverage rows hold them, built in `BUILD_YEAR`."""
    keys = (VARIABLE_COLUMN, FACET_COLUMN, DONOR_COLUMN, RECIPIENT_COLUMN)
    schema = pa.schema(
        field.with_type(pa.dictionary(pa.int32(), pa.string())) if field.name in keys else field
        for field in table.schema
    )
    return checked_index(
        table.cast(schema),
        role_names,
        facet_metadata,
        horizon=horizon,
        build_year=BUILD_YEAR,
        label="the test coverage table",
    )


def build_test_index(
    *, provenance: str | None = None, variables: Collection[str] | None = None
) -> CoverageIndex:
    """The `CoverageIndex` of this module's rows, `ROLE_NAMES` and `FACET_METADATA`.
    `provenance`, when set, replaces every facet's provenance. `variables`, when set, keeps
    only the rows of those variables."""
    rows = _ROWS if variables is None else [row for row in _ROWS if row[0] in variables]
    table, horizon = coverage_table(rows)
    facet_metadata = (
        FACET_METADATA
        if provenance is None
        else {
            facet_id: dataclasses.replace(metadata, provenance_id=provenance)
            for facet_id, metadata in FACET_METADATA.items()
        }
    )
    return checked_test_index(table, ROLE_NAMES, facet_metadata, horizon=horizon)


COVERAGE_SLICE_BUILD: Final[Path] = (
    Path(__file__).parents[1]
    / "fixtures"
    / "coverage_slice"
    / "format-3"
    / "versions"
    / "20260927T142252Z-slice"
)
"""The recorded coverage slice: the series of the variables the suite's fixtures and corpora
name, as `coverage.parquet`, `role_names.json`, `facets.json` and a `manifest.json` holding
its horizon and build time."""


@functools.cache
def slice_index() -> CoverageIndex:
    """The `CoverageIndex` of the recorded coverage slice, which was built in `BUILD_YEAR`."""
    manifest = json.loads((COVERAGE_SLICE_BUILD / "manifest.json").read_text())
    horizon = YearHorizon(epoch_year=manifest["epoch_year"], mask_words=manifest["mask_words"])
    roles = {
        variable: RoleNames(**names)
        for variable, names in json.loads(
            (COVERAGE_SLICE_BUILD / "role_names.json").read_text()
        ).items()
    }
    facets = {
        facet_id: FacetMetadata(
            provenance_id=fields.get("provenance_id"),
            measurement_method=fields.get("measurement_method"),
            unit=fields.get("unit"),
            scaling_factor=fields.get("scaling_factor"),
            observation_period=fields.get("observation_period"),
        )
        for facet_id, fields in json.loads(
            (COVERAGE_SLICE_BUILD / "facets.json").read_text()
        ).items()
    }
    table = pq.read_table(COVERAGE_SLICE_BUILD / "coverage.parquet")
    return checked_test_index(table.cast(horizon.schema), roles, facets, horizon=horizon)
