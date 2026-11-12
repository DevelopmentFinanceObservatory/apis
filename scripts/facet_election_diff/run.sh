#!/usr/bin/env bash
# Run one facet election diff script against this tree, niced, from the staged data directory.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
export FACET_DIFF_DATA=${FACET_DIFF_DATA:-$HERE/data}
export PYTHONPATH=$TREE:$HERE
cd "$FACET_DIFF_DATA"
exec nice -n 10 uv run --project "$TREE" --no-sync python "$HERE/$1" "${@:2}"
