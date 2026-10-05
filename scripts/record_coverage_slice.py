"""Record a filtered slice of a published coverage artifact.

The real artifact keeps fixture expectations independent of fixture-derived counts, letting
offline tests detect drift. The slice includes role-bearing StatVars referenced in
``tests/fixtures/*.json`` and ``corpus/*.yaml`` (cases and families). A whole-token match against
source ``role_names`` selects a variable and all its donor/recipient rows. Synthetic
``FixtureStatVar_`` package fixtures are excluded because no published artifact carries them.

The output keeps the published layout under ``format-<N>/`` with ``current.json`` and
``versions/<build_id>/`` files, including the source build id. The layout matches
``load_coverage`` and satisfies its completeness check with metadata for every facet id in the
sliced rows. Facet metadata is selected by facet id because facets can span variables.
The manifest records the source URI, stamp, and build id.

Usage:
    uv run python scripts/record_coverage_slice.py \\
        --source file:///path/to/real/coverage/artifact \\
        --out tests/fixtures/coverage_slice
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import cast

from series_coverage.io import (
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.loader import load_coverage
from series_coverage.schema import (
    FORMAT_ROOT,
    POINTER_FILE,
    VERSIONS_PREFIX,
    CoverageManifest,
    CoveragePointer,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "tests" / "fixtures" / "coverage_slice"

# Scan root fixtures and corpus files. Package fixtures use synthetic StatVar dcids.
_SCAN_GLOBS = (
    "tests/fixtures/*.json",
    "corpus/*.yaml",
    "corpus/families/*.yaml",
)

# Match bare dcids in JSON/YAML values and compound fixture keys.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _referenced_variables(known_variables: set[str]) -> set[str]:
    """Return known dcids that appear as whole tokens in scanned files."""
    found: set[str] = set()
    for pattern in _SCAN_GLOBS:
        for path in sorted(_REPO_ROOT.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            for token in _TOKEN_RE.findall(text):
                if token in known_variables:
                    found.add(token)
    return found


def record(*, source_uri: str, out: Path) -> None:
    loaded = load_coverage(source_uri)
    table = loaded.index._table
    known_variables = set(loaded.index.role_names.keys())
    keep_by_reference = _referenced_variables(known_variables)

    import pyarrow as pa
    import pyarrow.compute as pc

    if not keep_by_reference:
        print(
            "record_coverage_slice: no fixture or corpus file references a known role-bearing "
            "StatVar dcid; refusing to write an empty slice",
            file=sys.stderr,
        )
        sys.exit(1)
    mask = pc.is_in(
        table["variable"], value_set=pa.array(sorted(keep_by_reference), type=pa.string())
    )

    # CoverageIndex dictionary-encodes keys. The on-disk schema uses strings.
    horizon = loaded.manifest.horizon
    sliced = table.filter(mask).cast(horizon.schema)
    if sliced.num_rows == 0:
        print(
            "record_coverage_slice: the fixture/corpus match produced no rows; "
            "refusing to write an empty slice",
            file=sys.stderr,
        )
        sys.exit(1)
    keep = set(cast("list[str]", sliced["variable"].to_pylist()))
    role_names = {variable: loaded.index.role_names[variable] for variable in keep}
    sliced_facet_ids = set(cast("list[str]", sliced["facet_id"].to_pylist()))
    facet_metadata = {
        facet_id: loaded.index.facet_metadata[facet_id] for facet_id in sorted(sliced_facet_ids)
    }

    root = out / FORMAT_ROOT
    prefix = root / VERSIONS_PREFIX / loaded.manifest.build_id
    prefix.mkdir(parents=True, exist_ok=True)
    write_coverage_table(sliced, prefix / "coverage.parquet", horizon=horizon)
    write_role_names(role_names, prefix / "role_names.json")
    write_facet_metadata(facet_metadata, prefix / "facets.json")
    manifest = CoverageManifest(
        stamp=loaded.manifest.stamp,
        build_id=loaded.manifest.build_id,
        built_at=loaded.manifest.built_at,
        source=(
            f"{source_uri} (stamp {loaded.manifest.stamp}, build {loaded.manifest.build_id}), "
            "filtered to fixture/corpus-referenced variables by "
            "scripts/record_coverage_slice.py"
        ),
        row_counts={"rows": sliced.num_rows, "variables": len(keep)},
        files={
            "coverage": "coverage.parquet",
            "role_names": "role_names.json",
            "facets": "facets.json",
        },
        horizon=horizon,
    )
    write_manifest(manifest, prefix / "manifest.json")
    write_current(
        CoveragePointer(stamp=loaded.manifest.stamp, build_id=loaded.manifest.build_id),
        root / POINTER_FILE,
    )

    coverage_bytes = (prefix / "coverage.parquet").stat().st_size
    role_names_bytes = (prefix / "role_names.json").stat().st_size
    facets_bytes = (prefix / "facets.json").stat().st_size
    manifest_bytes = (prefix / "manifest.json").stat().st_size
    total_bytes = coverage_bytes + role_names_bytes + facets_bytes + manifest_bytes
    print(f"variables kept: {len(keep)} ({sorted(keep)})")
    print(f"rows kept: {sliced.num_rows}")
    print(f"facet_ids kept: {len(facet_metadata)} ({sorted(facet_metadata)})")
    print(
        f"wrote {out} -- coverage.parquet {coverage_bytes:,}B, role_names.json {role_names_bytes:,}B, "
        f"facets.json {facets_bytes:,}B, manifest.json {manifest_bytes:,}B, total {total_bytes:,}B"
    )
    assert sliced.schema.equals(horizon.schema)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", required=True, help="base URI of the real published coverage artifact"
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT, help="output directory (default: %(default)s)"
    )
    args = parser.parse_args()
    record(source_uri=args.source, out=args.out)


if __name__ == "__main__":
    main()
