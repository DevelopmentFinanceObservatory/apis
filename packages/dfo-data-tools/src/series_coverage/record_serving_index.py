"""`python -m series_coverage.record_serving_index` records the packaged code index as the one
every service runs.

The deploy runs it as its final step, after the datasets worker and the apis service both moved
to the image, as an execution of the coverage job on that image. It writes
`<format root>/serving/code_index.json` under `COVERAGE_ARTIFACT_URI`
(`series_coverage.compatibility.record_serving_index`). The coverage job's prune expires the
compatibility pointer of another index only when that pointer was last written before this
record.

Usage:
    python -m series_coverage.record_serving_index   # reads COVERAGE_ARTIFACT_URI
"""

from __future__ import annotations

import os
import sys

from code_index.loader import default_index_path, load_index
from series_coverage.compatibility import record_serving_index

_ARTIFACT_URI_ENV_VAR = "COVERAGE_ARTIFACT_URI"


def main() -> int:
    artifact_uri = os.environ.get(_ARTIFACT_URI_ENV_VAR, "").strip()
    if not artifact_uri:
        print(
            f"series_coverage.record_serving_index: {_ARTIFACT_URI_ENV_VAR} is not set",
            file=sys.stderr,
        )
        return 1
    index = load_index(default_index_path())
    try:
        record_serving_index(artifact_uri, index)
    except ValueError as exc:
        print(f"series_coverage.record_serving_index: {exc}", file=sys.stderr)
        return 1
    print(
        "series_coverage.record_serving_index: recorded code index "
        f"{index.stamp.content_hash!r} as serving at {artifact_uri!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
