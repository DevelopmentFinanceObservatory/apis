"""Builds coverage fixtures with country and multilateral donors, two facets, and role groups."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from code_index.query import CodeIndex
from code_index.types import IndexStamp, StatVarRow
from series_coverage import shared as coverage_shared
from series_coverage.io import (
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.loader import load_coverage
from series_coverage.schema import (
    BUILD_EPOCH_YEAR,
    FORMAT_ROOT,
    POINTER_FILE,
    VERSIONS_PREFIX,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)

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

# Real-shaped, live-matching metadata for the two facet ids above (see
# series_coverage.job's own module docstring: TimeSeries carries no
# scaling_factor column at all).
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
    from `BUILD_EPOCH_YEAR` to the latest year any row names, as
    `series_coverage.job` sizes it."""
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


def build_dir(base: Path, build_id: str) -> Path:
    """`base`'s build directory for `build_id` under this code's format
    root: `<base>/format-<N>/versions/<build_id>/`."""
    return base / FORMAT_ROOT / VERSIONS_PREFIX / build_id


def pointer_path(base: Path) -> Path:
    """`base`'s `current.json` under this code's format root."""
    return base / FORMAT_ROOT / POINTER_FILE


def build_test_artifact(
    base: Path,
    *,
    stamp: str = "test-coverage-stamp",
    build_id: str | None = None,
    provenance: str | None = None,
) -> str:
    """Writes one build (`manifest.json`, `coverage.parquet`,
    `role_names.json`, `facets.json`) into `build_dir(base, build_id)` and
    points `pointer_path(base)` at it, the layout a published artifact
    uses. `build_id` defaults to `<stamp>-build`. `provenance`, when set,
    replaces every facet's provenance. Returns the `file://` URI a
    `COVERAGE_ARTIFACT_URI` should point at, `base` itself.
    """
    build_id = build_id or f"{stamp}-build"
    prefix = build_dir(base, build_id)
    prefix.mkdir(parents=True, exist_ok=True)

    table, horizon = coverage_table(_ROWS)
    write_coverage_table(table, prefix / "coverage.parquet", horizon=horizon)
    write_role_names(ROLE_NAMES, prefix / "role_names.json")
    facet_metadata = (
        FACET_METADATA
        if provenance is None
        else {
            facet_id: dataclasses.replace(metadata, provenance_id=provenance)
            for facet_id, metadata in FACET_METADATA.items()
        }
    )
    write_facet_metadata(facet_metadata, prefix / "facets.json")
    manifest = CoverageManifest(
        stamp=stamp,
        build_id=build_id,
        built_at="2026-09-24T00:00:00Z",
        source="tests.series_coverage.support.build_test_artifact",
        row_counts={"rows": len(_ROWS)},
        files={
            "coverage": "coverage.parquet",
            "role_names": "role_names.json",
            "facets": "facets.json",
        },
        horizon=horizon,
    )
    write_manifest(manifest, prefix / "manifest.json")
    write_current(CoveragePointer(stamp=stamp, build_id=build_id), pointer_path(base))
    return f"file://{base}"


def build_single_mask_artifact(base: Path, *, stamp: str = "test-coverage-stamp") -> str:
    """`build_test_artifact`'s rows in the single-mask layout, labelled
    with this code's own `format_version`: one `year_mask` column in place
    of `year_mask_0`, and a manifest with no `epoch_year`/`mask_words`.
    Returns the artifact's `file://` URI."""
    uri = build_test_artifact(base, stamp=stamp)
    prefix = build_dir(base, f"{stamp}-build")
    table = pq.read_table(prefix / "coverage.parquet")
    pq.write_table(
        table.rename_columns(
            ["year_mask" if name == "year_mask_0" else name for name in table.column_names]
        ),
        prefix / "coverage.parquet",
    )
    manifest = json.loads((prefix / "manifest.json").read_text(encoding="utf-8"))
    del manifest["epoch_year"], manifest["mask_words"]
    (prefix / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return uri


__all__ = [
    "DEU",
    "ETH",
    "FACET_ACTUALS",
    "FACET_METADATA",
    "FACET_PROJECTED",
    "FRA",
    "KEN",
    "ROLE_NAMES",
    "UNICEF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "ZWE",
    "build_dir",
    "build_single_mask_artifact",
    "build_test_artifact",
    "coverage_table",
    "pointer_path",
]


def code_index_for(
    provenances: Mapping[str, tuple[str, ...]], *, content_hash: str = "test-index"
) -> CodeIndex:
    """A code index with one StatVar row per variable of `provenances`, carrying that
    variable's provenances, and no declared dataset. `content_hash` names its compatibility
    pointer."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fake.example",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash=content_hash,
        ),
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=dcid,
                population_type="FinancialFlow",
                measured_property="amount",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={},
                provenances=row_provenances,
            )
            for dcid, row_provenances in sorted(provenances.items())
        ],
    )


def index_for(uri: str, *, content_hash: str = "test-index") -> CodeIndex:
    """A code index whose rows carry exactly the provenances of the build `current.json`
    names under `uri`, so that build is compatible with it."""
    return code_index_for(
        load_coverage(uri).index.variable_provenances(), content_hash=content_hash
    )


@contextmanager
def session_coverage_service() -> Iterator[None]:
    """Installs, inside the block, the process-wide coverage service over
    `COVERAGE_ARTIFACT_URI`, checked against an index built from that build's own provenances.
    The suite's recorded slice covers a subset of the packaged index's StatVars, so the
    packaged index would reject it."""
    uri = os.environ["COVERAGE_ARTIFACT_URI"]
    service = coverage_shared.CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=0
    )
    with coverage_shared._singleton.poisoned_for_test(service):
        yield
