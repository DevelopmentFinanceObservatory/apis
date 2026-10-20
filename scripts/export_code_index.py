"""Regenerate `code_index/data/index.json.gz` from the live graph.

Four CRS code systems, `dac2aAidType`, `dac1AidType` and `dac2bAidType` use `<-typeOf`
enumeration. `crsPurpose` has 289 codes. Each declared dataset's StatVars are enumerated through
a system every StatVar of that dataset declares, and the union is checked against complete
non-spine reads. This graph repeats reverse-arc pages, so truncated reads split by code axes and
observed `populationType`, to depth 6. A second read of each exported five-tuple refuses the export when the graph holds a
StatVar the index would decide that is not an exported row.

A dataset whose StatVars declare no code system, World Bank IDS, cannot be enumerated that way.
Its StatVars come from the coverage build into a separate `listed_statvars` block, which no
family, measure axis or shape reads; Search and the mint elect members from it in place of a
graph walk (`code_index.members.listed_group`). An unsplit read of each five-tuple a listed row
has refuses the export when the graph holds a StatVar there that is neither an exported nor a
listed row.

Each listed StatVar's `sourceIndicatorCode` arcs give the `source_codes` block: the StatVars each
publisher's series code names under the provenance the arc carries, so search can answer a code
such as `DT.TDS.DEGG.CD` with its variables. A listed StatVar with no code refuses the export, and
so does a code whose arc provenance the coverage build does not give its StatVar, since the
finished document must load (`code_index.loader._check_source_codes`).

The `specializationOf` arc of each constraint value a listed StatVar holds gives the
`value_parents` block, so the sentence renderer knows a general government debtor
(`IDSDebtorTypeGG`) is a public-sector one (`IDSDebtorTypePS`) without a graph read. A value with
two parents refuses the export.

Pass 1 reads each StatVar's type, shape, and declared constraints. Pass 2 reads their values.
Export requires held, resolved targets with `typeOf` and one value per single-value field. Nodes
typed only `ProvisionalNode` and non-StatisticalVariables raise. Constraint and observation
properties are list-valued. Export validates rows before the mint door trusts them.

Authored config supplies system declarations, defaults, measure axes and labels, the IDS and WEO
measure nouns, synonyms, dataset declarations, the shape slot defaults, the constraint values a
dataset states on only some of its series, the measure class patterns, the variable aliases
recall expands, and the words naming a stat type. Graph reads supply codes, StatVars, and measure
values on those rows. Every observed measure partition needs one authored label.

The published series coverage build supplies each StatVar's provenances: `coverage.parquet` maps a
variable to its facet ids and `facets.json` maps a facet id to its provenance. The stamp records
the build id and the number of StatVars the build holds no series for. A declared dataset with no
StatVar row aborts the export.

The same build names every donor and recipient of a declared or listed dataset. Each party's graph
types give it a kind (party, total or residual) under the authored `PARTY_TYPE_KINDS`, except a
party the authored `PARTY_KIND_OVERRIDES` names. A party the authored `PRIVATE_PROVIDERS` lists by
OECD donor code (`dacCode`) has the sector `private`, and every other party `official`. Its graph
name is its label, and its row lists the
datasets that publish it and the flow axes they publish it on. The authored `TOTAL_GROUPS` names the group each total holds, whether
the total equals the sum of the group's members plus the residuals it counts beside them, and those
residuals. A party with an undeclared type, or whose types
give no single kind, aborts the export, as does a `TOTAL_GROUPS` or `PARTY_KIND_OVERRIDES` row
naming a party no declared or listed dataset publishes (a `TOTAL_GROUPS` row also needs that party to be a
total, and each residual it names to be a published residual).

The authored `WHOLE_POPULATION_TOTALS` marks the totals that hold a whole population on their one
axis in the datasets it names (`PartyRow.whole`, with the classes it covers and the years its
composition check holds), gives each the sector those classes share, and `WHOLE_TOTAL_SETS` the unions of such totals read as one population. The export aborts when a
`PRIVATE_PROVIDERS` code is carried by another party than its listed one, when a published donor
carries an unlisted code from 1601 to 1699, and when a whole-population row names a party that is
no published total, is published on another axis or is marked in a dataset that does not publish
it; the finished document must also load, which refuses two whole totals that cover one class on
one dataset and axis, and a whole total that covers different sectors in different datasets.

Requires `SEARCH_GRAPH_BASE` and `COVERAGE_ARTIFACT_URI`, with read access to the coverage bucket
for a `gs://` URI. Sorted JSON is stable. `content_hash` covers data blocks and excludes the stamp.

Usage:
    COVERAGE_ARTIFACT_URI=gs://<coverage bucket>/ uv run --env-file .env \\
        python scripts/export_code_index.py \\
        --out packages/dfo-data-tools/src/code_index/data/index.json.gz
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from code_index.declarations import (
    CODE_SYSTEMS,
    DATASETS,
    LISTED_DATASETS,
    MEASURE_AXES,
    MEASURE_NOUNS,
    MEASURE_PARTITION_LABELS,
    PARTLY_STATED_VALUES,
    PARTY_KIND_OVERRIDES,
    PARTY_TYPE_KINDS,
    PRIVATE_PROVIDERS,
    TOTAL_GROUPS,
    WHOLE_POPULATION_TOTALS,
    WHOLE_TOTAL_SETS,
    CodeSystemConfig,
    KindOverride,
    TotalGroup,
    WholePopulationTotal,
)
from code_index.export import (
    ExportReader,
    ExportRefused,
    PartyPublication,
    content_hash,
    export_graph_blocks,
    export_parties,
    index_bytes,
    listed_dcids,
    stamp_counts,
    verify_complete_systems,
)
from code_index.loader import IndexFormatError, parse_index, parse_measure_axes
from code_index.types import (
    DatasetDeclaration,
    MeasureAxis,
    MeasurePartitionLabel,
    PartlyStatedValue,
    PartyKind,
    WholeTotalSet,
)
from dc_graph import GraphClient

if TYPE_CHECKING:
    from series_coverage.loader import LoadedCoverage

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = (
    _REPO_ROOT / "packages" / "dfo-data-tools" / "src" / "code_index" / "data" / "index.json.gz"
)
# Twenty-dcid `/v2/node` requests avoid mixer paging loops. GraphClient reads chunks concurrently.
_ARC_CHUNK_SIZE = 20


@dataclass(frozen=True)
class CoverageJoin:
    """What the export takes from one series coverage build: its build id, each covered
    variable's sorted provenances, and the donors and recipients of the declared and listed
    datasets."""

    build_id: str
    provenances: Mapping[str, tuple[str, ...]]
    parties: Mapping[str, PartyPublication] = field(default_factory=dict)


def load_coverage_join(base_uri: str, *, datasets: Collection[str]) -> CoverageJoin:
    """`coverage_join` of the published coverage build under `base_uri`.

    `series_coverage` needs the `coverage` extra, which the graph-only callers of this module
    run without, so the import stays inside this function.
    """
    from series_coverage.loader import load_coverage

    return coverage_join(load_coverage(base_uri), datasets=datasets)


def coverage_join(loaded: LoadedCoverage, *, datasets: Collection[str]) -> CoverageJoin:
    """The variable provenances of the coverage build `loaded` and the parties on the facets
    of the `datasets` provenances, with the years each publishes them."""
    years = loaded.index.dataset_party_years(datasets)
    return CoverageJoin(
        build_id=loaded.manifest.build_id,
        provenances=loaded.index.variable_provenances(),
        parties={
            party: PartyPublication(
                datasets=frozenset(years[party]), axes=published_on, years=years[party]
            )
            for party, published_on in loaded.index.dataset_parties(datasets).items()
        },
    )


def _graph_base_url() -> str:
    base = os.environ.get("SEARCH_GRAPH_BASE", "").strip()
    if not base:
        raise SystemExit(
            "SEARCH_GRAPH_BASE must be set -- this script always talks to the live graph"
        )
    return base


def _coverage_artifact_uri() -> str:
    uri = os.environ.get("COVERAGE_ARTIFACT_URI", "").strip()
    if not uri:
        raise SystemExit(
            "COVERAGE_ARTIFACT_URI must be set -- the export joins each StatVar's provenances "
            "from the published coverage build"
        )
    return uri


def export_reader(base_url: str, token: str | None) -> GraphClient:
    """Create the export's GraphClient with its configured chunk size.

    `base_url` is the `/core/api` root.
    """
    return GraphClient(base_url=base_url, token=token, node_arc_chunk_size=_ARC_CHUNK_SIZE)


def _join_provenances(statvars: list[dict], coverage: CoverageJoin) -> int:
    """Set each row's sorted `provenances` from `coverage` and return how many rows it does not
    cover. An uncovered row carries `[]`."""
    uncovered = 0
    for row in statvars:
        provenances = sorted(coverage.provenances.get(row["dcid"], ()))
        row["provenances"] = provenances
        if not provenances:
            uncovered += 1
    return uncovered


def export_index(
    reader: ExportReader,
    *,
    graph_endpoint: str,
    now: datetime,
    coverage: CoverageJoin,
    code_systems: Sequence[CodeSystemConfig] = CODE_SYSTEMS,
    measure_axes: Sequence[MeasureAxis] = MEASURE_AXES,
    measure_partition_labels: Sequence[MeasurePartitionLabel] = MEASURE_PARTITION_LABELS,
    measure_nouns: Sequence[MeasurePartitionLabel] = MEASURE_NOUNS,
    datasets: Sequence[DatasetDeclaration] = DATASETS,
    party_type_kinds: Mapping[str, PartyKind | None] = PARTY_TYPE_KINDS,
    partly_stated_values: Sequence[PartlyStatedValue] = PARTLY_STATED_VALUES,
    party_kind_overrides: Mapping[str, KindOverride] = PARTY_KIND_OVERRIDES,
    total_groups: Mapping[str, TotalGroup] = TOTAL_GROUPS,
    private_providers: Mapping[int, str] = PRIVATE_PROVIDERS,
    whole_totals: Mapping[str, WholePopulationTotal] = WHOLE_POPULATION_TOTALS,
    whole_total_sets: Sequence[WholeTotalSet] = WHOLE_TOTAL_SETS,
    listed_datasets: Collection[str] = (),
) -> dict:
    """Build the index document: the graph blocks, with the StatVars `coverage` assigns to
    `listed_datasets` as listed rows (`listed_dcids`), each row's provenances from `coverage`,
    every party of `coverage` with its kind and group (`export_parties`), and the stamp.
    `main` lists `LISTED_DATASETS`.

    Tests can pass smaller config tables for fixture graphs. Empty `measure_axes` yields an empty
    `measure_values` list. The finished document must load: a document `parse_index` refuses,
    such as one whose declared dataset has no row or whose datasets share a rank, raises
    `ExportRefused`.
    """
    blocks = export_graph_blocks(
        reader,
        code_systems=code_systems,
        measure_axes=measure_axes,
        measure_partition_labels=measure_partition_labels,
        measure_nouns=measure_nouns,
        datasets=datasets,
        party_type_kinds=party_type_kinds,
        partly_stated_values=partly_stated_values,
        whole_total_sets=whole_total_sets,
        listed=listed_dcids(coverage.provenances, listed_datasets),
    )
    statvars_without_coverage = _join_provenances(blocks["statvars"], coverage)
    _join_provenances(blocks["listed_statvars"], coverage)
    verify_complete_systems(blocks["statvars"], code_systems)
    blocks["parties"] = export_parties(
        reader,
        coverage.parties,
        type_kinds=party_type_kinds,
        kind_overrides=party_kind_overrides,
        total_groups=total_groups,
        private_providers=private_providers,
        whole_totals=whole_totals,
    )

    stamp = {
        "graph_endpoint": graph_endpoint,
        "exported_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **stamp_counts(blocks),
        "content_hash": content_hash(blocks),
        "coverage_build": coverage.build_id,
        "statvars_without_coverage": statvars_without_coverage,
    }
    document = {"stamp": stamp, **blocks}
    try:
        parse_index(document)
        parse_measure_axes(document)
    except IndexFormatError as exc:
        raise ExportRefused(f"the export does not load as a code index: {exc}") from exc
    return document


def _write_index(path: Path, document: dict) -> None:
    """Write `document` to `path` as `index_bytes`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(index_bytes(document))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    base = _graph_base_url()
    coverage = load_coverage_join(
        _coverage_artifact_uri(),
        datasets={dataset.provenance for dataset in DATASETS} | set(LISTED_DATASETS),
    )
    reader = export_reader(f"{base}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
    try:
        document = export_index(
            reader,
            graph_endpoint=base,
            now=datetime.now(UTC),
            coverage=coverage,
            listed_datasets=LISTED_DATASETS,
        )
    except ExportRefused as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        reader.close()
    _write_index(Path(args.out), document)
    stamp = document["stamp"]
    print(
        f"wrote {sum(stamp['code_counts'].values())} code rows, "
        f"{len(document['statvars'])} StatVar rows, "
        f"{len(document['listed_statvars'])} listed StatVar rows "
        f"({stamp['statvars_without_coverage']} not in coverage build "
        f"{stamp['coverage_build']}), {len(document['source_codes'])} source code rows, and "
        f"{len(document['measure_values'])} measure values to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
