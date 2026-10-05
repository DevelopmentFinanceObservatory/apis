"""`python -m series_coverage.verify_index` checks the packaged code index's provenances
against the published coverage build.

The code index export joins each StatVar's provenances from a coverage build. This check loads
the build under `COVERAGE_ARTIFACT_URI` and compares, for every StatVar row of the index, the
build's provenances for that variable with `StatVarRow.provenances`
(`series_coverage.compatibility.index_coverage_differences`). It also requires every declared
dataset to hold a row in the build. Any difference prints up to twenty variables and returns a
non-zero status. Both build ids are printed: the one the index was exported from and the one
read.

`--record` also points the index's compatibility pointer at the build when it passes, so a
service running this index can load that build after a later, incompatible publish. Deploys
pass it; a hand check of a candidate export does not.

Pass an artifact URI as an argument, or set `COVERAGE_ARTIFACT_URI`. `--index` names an index
export other than the packaged one.

Usage:
    python -m series_coverage.verify_index                       # reads COVERAGE_ARTIFACT_URI
    python -m series_coverage.verify_index file:///tmp/coverage   # checks an explicit URI
    python -m series_coverage.verify_index --index index.json.gz  # checks a candidate export
    python -m series_coverage.verify_index --record               # checks and records the build
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from code_index.loader import default_index_path, load_index
from series_coverage.compatibility import (
    describe_differences,
    index_coverage_differences,
    record_if_compatible,
)
from series_coverage.loader import load_coverage

_ARTIFACT_URI_ENV_VAR = "COVERAGE_ARTIFACT_URI"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m series_coverage.verify_index")
    parser.add_argument(
        "artifact_uri", nargs="?", help=f"reads {_ARTIFACT_URI_ENV_VAR} when omitted"
    )
    parser.add_argument("--index", help="a code index export; the packaged one when omitted")
    parser.add_argument(
        "--record",
        action="store_true",
        help="point the index's compatibility pointer at the build when it passes",
    )
    args = parser.parse_args(argv)

    artifact_uri = args.artifact_uri or os.environ.get(_ARTIFACT_URI_ENV_VAR, "").strip()
    if not artifact_uri:
        print(
            f"series_coverage.verify_index: no artifact_uri given and {_ARTIFACT_URI_ENV_VAR} "
            "is not set -- nothing to check the code index against",
            file=sys.stderr,
        )
        return 1
    try:
        loaded = load_coverage(artifact_uri)
        index = load_index(args.index or default_index_path())
    except Exception as exc:
        print(f"series_coverage.verify_index: could not load its inputs: {exc}", file=sys.stderr)
        return 1

    if args.record and not index.stamp.content_hash:
        print(
            "series_coverage.verify_index: --record needs an index whose stamp carries a "
            "content_hash to name its compatibility pointer by",
            file=sys.stderr,
        )
        return 1

    builds = (
        f"index exported from coverage build {index.stamp.coverage_build!r}, "
        f"read build {loaded.manifest.build_id!r} at {artifact_uri!r}"
    )
    differences = (
        record_if_compatible(artifact_uri, index, loaded)
        if args.record
        else index_coverage_differences(index, loaded.index.variable_provenances())
    )
    if not differences:
        recorded = (
            f"; recorded as compatible with index {index.stamp.content_hash!r}"
            if args.record
            else ""
        )
        print(
            f"series_coverage.verify_index: {len(index.statvars)} StatVar rows carry the "
            f"build's provenances ({builds}){recorded}"
        )
        return 0

    print(f"series_coverage.verify_index: the code index differs ({builds})", file=sys.stderr)
    for line in describe_differences(differences):
        print(f"  {line}", file=sys.stderr)
    print(
        "  re-export with scripts/export_code_index.py against the current build",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
