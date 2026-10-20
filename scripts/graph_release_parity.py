"""Diff the views a graph release serves with the artifacts they replace, built from one import.

The switch gate for `GRAPH_RELEASE_COVERAGE` on an environment. The script opens the release the
pointer under `--release-uri` names and builds its serving views with `graph_release.views`, the
functions a serving process opens it with. It then compares them with what the old path serves
for the same import:

- the release's `CoverageIndex` against the published coverage build under `--coverage-uri`:
  every row (`graph_release.shadow.row_diff`), each variable's role names and provenances, each
  facet's metadata, and the horizon and build year;
- the release's `code_index_blocks` document against a fresh `export_index` (the function
  `scripts/export_code_index.py` runs) over the live mixer at `--graph-base`, joined with that
  same coverage build, every block row by row.

It refuses (exit 2) before comparing anything when the coverage build and the release come from
different ingestion workflow executions (`import_mismatch`, or `coverage_ingestion_unrecorded`
for a build that records none), when this checkout's declarations hash differs from the
release's (`declarations_moved`), since a fresh export would then be built from other
declarations, or when the DC service serves another revision than the release read, before or
after the export (`serving_revision_moved`), since the mixer would then answer from another
import. An export that refuses ends the run too (`export_refused`).

Each difference is either explained by a rule of `RULES`, which names why the two paths differ
there by construction, or unexplained. The JSON report gives, per category, how many items were
compared, how many differ, how many each rule explains and the first `--examples` differences of
each kind. The exit status is 0 when no difference is unexplained, 1 when one is.

Reads only: the release and coverage buckets, the mixer, and the Cloud Run Admin API's
`run.services.get` on the DC service.

Usage:
    uv run python scripts/graph_release_parity.py \\
        --release-uri gs://dfo-dev-graph-release-dev-fin-observatory/ \\
        --coverage-uri gs://dfo-dev-coverage-dev-fin-observatory/ \\
        --graph-base https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app \\
        --datacommons-service projects/dev-fin-observatory/locations/northamerica-northeast1/services/dfo-dev-dc-datacommons-service \\
        --out parity-dev.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pyarrow as pa

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from artifact_publication.uri import join_uri  # noqa: E402
from code_index.declarations import DATASETS, LISTED_DATASETS  # noqa: E402
from code_index.export import ExportRefused, content_hash  # noqa: E402
from graph_release.coverage import coverage_rows, role_map  # noqa: E402
from graph_release.declarations import DECLARATIONS  # noqa: E402
from graph_release.format import LoadedRelease, Manifest, open_release  # noqa: E402
from graph_release.job import builder_code_hash  # noqa: E402
from graph_release.serving_revision import CloudRunAdmin  # noqa: E402
from graph_release.shadow import INDEX_BLOCK_KEYS, PINNED_INDEX_VARIABLES, row_diff  # noqa: E402
from graph_release.views import (  # noqa: E402
    CODE_INDEX_PART,
    DECLARATIONS_PART,
    ViewRefused,
    declared_role_axes,
    read_document,
    release_code_indexes,
    release_coverage_index,
)
from scripts.export_code_index import coverage_join, export_index, export_reader  # noqa: E402
from series_coverage.index import CoverageIndex  # noqa: E402
from series_coverage.loader import (  # noqa: E402
    LoadedCoverage,
    format_root,
    load_coverage,
    read_published_pointer,
    read_table,
)
from series_coverage.schema import VERSIONS_PREFIX, CoverageManifest  # noqa: E402

RULES: Final[Mapping[str, str]] = {
    "pinned_index_variable": (
        "a row naming a variable of graph_release.shadow.PINNED_INDEX_VARIABLES, which the "
        "release's listed_statvars and source_codes hold and the committed index lacks"
    ),
}
"""Each rule that explains a difference, and why the two paths differ there by construction."""

_ROW_KEYS: Final[Mapping[str, tuple[str, ...]]] = {**INDEX_BLOCK_KEYS, "parties": ("dcid",)}
"""The fields keying a code index block's rows. A list block not named here keys each row by
its canonical JSON, so a changed row reads as one row on each side."""

_RELEASE_PARTS: Final[frozenset[str]] = frozenset(
    {"series", "edges", "facets", DECLARATIONS_PART, CODE_INDEX_PART}
)
_DEFAULT_EXAMPLES: Final[int] = 10


class Refused(Exception):
    """A run that cannot compare the release with the old path; `code` names why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class Category:
    """The differences found in one category, the first `limit` of each kind kept."""

    limit: int
    compared: int = 0
    differences: int = 0
    explained: Counter[str] = field(default_factory=Counter)
    examples: list[object] = field(default_factory=list)
    explained_examples: list[object] = field(default_factory=list)

    def add(self, difference: object, *, rule: str | None = None) -> None:
        self.differences += 1
        if rule is None:
            if len(self.examples) < self.limit:
                self.examples.append(difference)
            return
        self.explained[rule] += 1
        if len(self.explained_examples) < self.limit:
            self.explained_examples.append({"rule": rule, "difference": difference})

    @property
    def unexplained(self) -> int:
        return self.differences - self.explained.total()

    def to_json(self) -> dict[str, object]:
        return {
            "compared": self.compared,
            "differences": self.differences,
            "unexplained": self.unexplained,
            "explained": dict(sorted(self.explained.items())),
            "examples": self.examples,
            "explained_examples": self.explained_examples,
        }


# ---------------------------------------------------------------- preconditions


def check_same_import(release: Manifest, coverage: CoverageManifest) -> None:
    """Raises `Refused` unless the coverage build records the release's ingestion execution."""
    if coverage.ingestion is None:
        raise Refused(
            "coverage_ingestion_unrecorded",
            f"coverage build {coverage.build_id} records no ingestion execution",
        )
    if coverage.ingestion.workflow_execution_id != release.import_execution_id:
        raise Refused(
            "import_mismatch",
            f"coverage build {coverage.build_id} was built from execution "
            f"{coverage.ingestion.workflow_execution_id}, release {release.release_id} from "
            f"{release.import_execution_id}",
        )


def check_declarations(release: Manifest) -> None:
    """Raises `Refused` unless this checkout's declarations hash is the release's."""
    found = DECLARATIONS.declarations_hash()
    if found != release.declarations_hash:
        raise Refused(
            "declarations_moved",
            f"this checkout's declarations hash {found} is not the release's "
            f"{release.declarations_hash}",
        )


def check_serving_revision(release: Manifest, serving: Callable[[], str]) -> None:
    """Raises `Refused` unless `serving()` names the revision the release read."""
    token = serving()
    if token != release.dc_serving_revision:
        raise Refused(
            "serving_revision_moved",
            f"the DC service serves {token}, the release read {release.dc_serving_revision}",
        )


# ---------------------------------------------------------------- coverage


def coverage_parity(
    release: LoadedRelease,
    view: CoverageIndex,
    published: LoadedCoverage,
    published_rows: pa.Table,
    *,
    examples: int,
) -> dict[str, Category]:
    """The release's coverage view `view` against the published build: its rows, role names,
    variable provenances, facet metadata, horizon and build year. `published_rows` is the
    build's table as `series_coverage.loader.read_table` reads it."""
    roles = role_map(release.tables["edges"], declared_role_axes(release.blobs[DECLARATIONS_PART]))
    rows = Category(examples, compared=max(view.num_rows, published.index.num_rows))
    count, keys = row_diff(
        coverage_rows(release.tables["series"], roles), published_rows, examples=examples
    )
    rows.differences = count
    rows.examples = list(keys)

    theirs = published.index
    found = {"coverage.rows": rows}
    for name, ours_map, theirs_map in (
        ("coverage.role_names", view.role_names, theirs.role_names),
        (
            "coverage.variable_provenances",
            view.variable_provenances(),
            theirs.variable_provenances(),
        ),
        ("coverage.facet_metadata", view.facet_metadata, theirs.facet_metadata),
    ):
        found[name] = _mapping_diff(ours_map, theirs_map, examples=examples)

    index = Category(examples, compared=2)
    for name, ours_value, theirs_value in (
        ("horizon", asdict(view.horizon), asdict(theirs.horizon)),
        ("build_year", view.build_year, theirs.build_year),
    ):
        if ours_value != theirs_value:
            index.add({"field": name, "release": ours_value, "published": theirs_value})
    found["coverage.index"] = index
    return found


def _mapping_diff(
    ours: Mapping[str, object], theirs: Mapping[str, object], *, examples: int
) -> Category:
    keys = sorted(ours.keys() | theirs.keys())
    category = Category(examples, compared=len(keys))
    for key in keys:
        mine, other = ours.get(key), theirs.get(key)
        if mine != other:
            category.add({"key": key, "release": _plain(mine), "published": _plain(other)})
    return category


def _plain(value: object) -> object:
    """`value` as JSON: a dataclass as its fields, a tuple as a list."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return list(value) if isinstance(value, tuple) else value


# ---------------------------------------------------------------- code index


def code_index_parity(
    release: Mapping[str, object], export: Mapping[str, object], *, examples: int
) -> dict[str, Category]:
    """The blocks of the release's code index document against a fresh export's, row by row.
    The `stamp` is no block: its counts and `content_hash` follow from the blocks, and its
    endpoint, export time and coverage build name each side's source."""
    found: dict[str, Category] = {}
    for block in sorted((release.keys() | export.keys()) - {"stamp"}):
        ours, theirs = _rows(block, release.get(block)), _rows(block, export.get(block))
        keys = sorted(ours.keys() | theirs.keys())
        category = Category(examples, compared=len(keys))
        for key in keys:
            mine, other = ours.get(key), theirs.get(key)
            if mine == other:
                continue
            if mine is None:
                difference: dict[str, object] = {"key": key, "side": "export_only"}
            elif other is None:
                difference = {"key": key, "side": "release_only"}
            else:
                difference = {"key": key, "side": "both", "fields": _moved_fields(mine, other)}
            category.add(difference, rule=_explain(mine, other))
        found[f"code_index.{block}"] = category
    return found


def _rows(block: str, value: object) -> dict[str, object]:
    """`value`'s rows by key: a list block's by `_ROW_KEYS` or canonical JSON, a mapping
    block's by its keys, any other value as one row. An absent block holds none."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): row for key, row in value.items()}
    if not isinstance(value, list):
        return {"": value}
    key = _ROW_KEYS.get(block)
    return {
        " ".join(str(row.get(name)) for name in key)
        if key and isinstance(row, Mapping)
        else json.dumps(row, sort_keys=True): row
        for row in value
    }


def _moved_fields(ours: object, theirs: object) -> list[str] | None:
    if not isinstance(ours, Mapping) or not isinstance(theirs, Mapping):
        return None
    return sorted(
        str(name) for name in ours.keys() | theirs.keys() if ours.get(name) != theirs.get(name)
    )


def _explain(ours: object, theirs: object) -> str | None:
    """The rule of `RULES` that explains why `ours` and `theirs` differ, or `None`."""
    text = json.dumps([ours, theirs], sort_keys=True)
    if any(variable in text for variable in PINNED_INDEX_VARIABLES):
        return "pinned_index_variable"
    return None


# ---------------------------------------------------------------- report


def parity_report(
    release: LoadedRelease,
    published: LoadedCoverage,
    published_rows: pa.Table,
    export: Mapping[str, object],
    *,
    examples: int = _DEFAULT_EXAMPLES,
) -> dict[str, object]:
    """The report on `release` against the coverage build `published` and the fresh code index
    `export`. The release's serving views that do not build count in category `views`."""
    views = Category(examples, compared=2)
    categories: dict[str, Category] = {"views": views}
    try:
        coverage = release_coverage_index(release)
    except ViewRefused as exc:
        views.add({"view": "coverage", "refused": str(exc)})
    else:
        categories |= coverage_parity(
            release, coverage, published, published_rows, examples=examples
        )
    try:
        release_code_indexes(release)
    except ViewRefused as exc:
        views.add({"view": "code_index", "refused": str(exc)})

    document = read_document(release.blobs[CODE_INDEX_PART])
    categories |= code_index_parity(document, export, examples=examples)
    unexplained = sum(category.unexplained for category in categories.values())
    return {
        "outcome": "differences" if unexplained else "parity",
        "unexplained": unexplained,
        "release": _release_summary(release.manifest),
        "coverage_build": _coverage_summary(published.manifest),
        "code_index": {
            "release_content_hash": content_hash(_blocks(document)),
            "export_content_hash": content_hash(_blocks(export)),
            "export_stamp": export.get("stamp"),
        },
        "rules": dict(RULES),
        "categories": {name: category.to_json() for name, category in sorted(categories.items())},
    }


def _blocks(document: Mapping[str, object]) -> dict:
    return {name: value for name, value in document.items() if name != "stamp"}


def _release_summary(manifest: Manifest) -> dict[str, object]:
    return {
        "release_id": manifest.release_id,
        "import_execution_id": manifest.import_execution_id,
        "read_timestamp": manifest.read_timestamp,
        "dc_serving_revision": manifest.dc_serving_revision,
        "declarations_hash": manifest.declarations_hash,
        "builder_code_hash": manifest.builder_code_hash,
        "checkout_builder_code_hash": builder_code_hash(),
    }


def _coverage_summary(manifest: CoverageManifest) -> dict[str, object]:
    return {
        "build_id": manifest.build_id,
        "stamp": manifest.stamp,
        "built_at": manifest.built_at,
        "workflow_execution_id": (
            manifest.ingestion.workflow_execution_id if manifest.ingestion else None
        ),
    }


# ---------------------------------------------------------------- main


def _run(args: argparse.Namespace, local_dir: Path) -> dict[str, object]:
    release = open_release(args.release_uri, local_dir=local_dir, parts=_RELEASE_PARTS)
    manifest = release.manifest
    pointer = read_published_pointer(args.coverage_uri)
    if pointer is None:
        raise Refused("coverage_unreadable", f"no coverage pointer under {args.coverage_uri}")
    published = load_coverage(args.coverage_uri, pointer=pointer)
    check_same_import(manifest, published.manifest)
    check_declarations(manifest)
    admin = CloudRunAdmin(service=args.datacommons_service)

    def serving() -> str:
        return admin.serving().token

    try:
        check_serving_revision(manifest, serving)
        reader = export_reader(f"{args.graph_base}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
        try:
            export = export_index(
                reader,
                graph_endpoint=args.graph_base,
                now=datetime.now(UTC),
                coverage=coverage_join(
                    published,
                    datasets={dataset.provenance for dataset in DATASETS} | set(LISTED_DATASETS),
                ),
                listed_datasets=LISTED_DATASETS,
            )
        except ExportRefused as exc:
            raise Refused("export_refused", str(exc)) from exc
        finally:
            reader.close()
        check_serving_revision(manifest, serving)
    finally:
        admin.close()
    prefix = join_uri(format_root(args.coverage_uri), VERSIONS_PREFIX, pointer.build_id)
    rows = read_table(prefix, published.manifest)
    return parity_report(release, published, rows, export, examples=args.examples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release-uri", required=True, help="the environment's GRAPH_RELEASE_URI")
    parser.add_argument(
        "--coverage-uri", required=True, help="the environment's COVERAGE_ARTIFACT_URI"
    )
    parser.add_argument(
        "--graph-base",
        default=os.environ.get("SEARCH_GRAPH_BASE"),
        help="the mixer root, without /core/api (default SEARCH_GRAPH_BASE)",
    )
    parser.add_argument(
        "--datacommons-service",
        default=os.environ.get("DATACOMMONS_SERVICE_NAME"),
        help="the DC service as projects/<p>/locations/<r>/services/<s> "
        "(default DATACOMMONS_SERVICE_NAME)",
    )
    parser.add_argument("--local-dir", type=Path, help="where to keep the release's parts")
    parser.add_argument("--examples", type=int, default=_DEFAULT_EXAMPLES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.graph_base or not args.datacommons_service:
        parser.error("--graph-base and --datacommons-service are required")

    try:
        with tempfile.TemporaryDirectory(prefix="graph-release-parity-") as scratch:
            report = _run(args, args.local_dir or Path(scratch))
    except Refused as exc:
        report = {"outcome": "refused", "refused": {"code": exc.code, "detail": exc.detail}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": report["outcome"], "unexplained": report.get("unexplained")}))
    sys.exit({"parity": 0, "differences": 1, "refused": 2}[str(report["outcome"])])


if __name__ == "__main__":
    main()
