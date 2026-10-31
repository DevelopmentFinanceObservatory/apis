#!/usr/bin/env bash
# For each file moved from dfo-ai into packages/dfo-analysis-engine, diff the file against
# its source at the dfo-ai commit it was copied from, and print only the changed lines
# that are not part of an import statement and are not blank. A pure move prints only the
# summary.
#
# Usage: DFO_AI_REPO=/path/to/dfo-ai scripts/diff-moved-engine.sh
#
# DFO_AI_REPO is a dfo-ai checkout that holds the commit below. A table row with a line
# range compares that range of the source: the file was split and only that part moved.
set -euo pipefail

SOURCE_REV=eefa15be
: "${DFO_AI_REPO:?set DFO_AI_REPO to a dfo-ai checkout holding $SOURCE_REV}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="$here/packages/dfo-analysis-engine"

git -C "$DFO_AI_REPO" cat-file -e "$SOURCE_REV^{commit}"

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

cat > "$scratch/compare.py" <<'PY'
import ast
import difflib
import sys
from pathlib import Path

source_path, dest_path, label = sys.argv[1:4]


def significant(path: str) -> list[str]:
    """The lines of `path` outside import statements, with blank lines dropped."""
    text = Path(path).read_text()
    skipped: set[int] = set()
    if label.endswith(".py"):
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import | ast.ImportFrom):
                skipped.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return [
        line
        for number, line in enumerate(text.splitlines(), 1)
        if number not in skipped and line.strip()
    ]


diff = difflib.unified_diff(significant(source_path), significant(dest_path), n=0, lineterm="")
changed = [line for line in diff if line[:1] in "+-" and not line.startswith(("+++", "---"))]
print(f"@@\t{len(changed)}\t{label}")
for line in changed:
    print(f"    {line}")
PY

# Columns: source path, destination under the package, source line range ("-" is the
# whole file).
while read -r src dest range; do
  git -C "$DFO_AI_REPO" show "$SOURCE_REV:$src" > "$scratch/source"
  if [ "$range" != "-" ]; then
    sed -n "${range}p" "$scratch/source" > "$scratch/slice"
    mv "$scratch/slice" "$scratch/source"
  fi
  python3 -I "$scratch/compare.py" "$scratch/source" "$package/$dest" "$dest"
done > "$scratch/report" <<'TABLE'
services/analysis-mcp/src/analysis_mcp/engine/__init__.py src/dfo_analysis_engine/engine/__init__.py -
services/analysis-mcp/src/analysis_mcp/engine/aggregate.py src/dfo_analysis_engine/engine/aggregate.py -
services/analysis-mcp/src/analysis_mcp/engine/columnar.py src/dfo_analysis_engine/engine/columnar.py -
services/analysis-mcp/src/analysis_mcp/engine/common.py src/dfo_analysis_engine/engine/common.py -
services/analysis-mcp/src/analysis_mcp/engine/compare.py src/dfo_analysis_engine/engine/compare.py -
services/analysis-mcp/src/analysis_mcp/engine/filter.py src/dfo_analysis_engine/engine/filter.py -
services/analysis-mcp/src/analysis_mcp/engine/growth.py src/dfo_analysis_engine/engine/growth.py -
services/analysis-mcp/src/analysis_mcp/engine/join.py src/dfo_analysis_engine/engine/join.py -
services/analysis-mcp/src/analysis_mcp/engine/quotient_of.py src/dfo_analysis_engine/engine/quotient_of.py -
services/analysis-mcp/src/analysis_mcp/engine/rank.py src/dfo_analysis_engine/engine/rank.py -
services/analysis-mcp/src/analysis_mcp/engine/share_of.py src/dfo_analysis_engine/engine/share_of.py -
services/analysis-mcp/src/analysis_mcp/engine/sum_of.py src/dfo_analysis_engine/engine/sum_of.py -
services/analysis-mcp/src/analysis_mcp/engine/two_sided.py src/dfo_analysis_engine/engine/two_sided.py -
services/analysis-mcp/src/analysis_mcp/op_spec.py src/dfo_analysis_engine/op_spec.py -
services/analysis-mcp/src/analysis_mcp/identity.py src/dfo_analysis_engine/identity.py -
services/analysis-mcp/src/analysis_mcp/execution_identity.py src/dfo_analysis_engine/execution_identity.py -
services/analysis-mcp/src/analysis_mcp/units.py src/dfo_analysis_engine/units.py -
services/analysis-mcp/src/analysis_mcp/artifact.py src/dfo_analysis_engine/artifact.py -
services/analysis-mcp/src/analysis_mcp/steps.py src/dfo_analysis_engine/steps.py -
services/analysis-mcp/src/analysis_mcp/result.py src/dfo_analysis_engine/result.py -
packages/common/src/common/rolling.py src/dfo_analysis_engine/rolling.py -
packages/common/src/common/time_grain.py src/dfo_analysis_engine/time_grain.py -
packages/common/src/common/place_type.py src/dfo_analysis_engine/place_type.py -
packages/common/src/common/unit_refs.py src/dfo_analysis_engine/unit_refs.py -
packages/common/src/common/price_basis.py src/dfo_analysis_engine/price_basis.py -
packages/common/src/common/hashed_fields.py src/dfo_analysis_engine/hashed_fields.py -
packages/common/src/common/materialisation/frame_ref.py src/dfo_analysis_engine/frame_ref.py -
packages/common/src/common/recipe.py src/dfo_analysis_engine/recipe.py 1,347
services/analysis-mcp/src/analysis_mcp/fixtures.py tests/frame_fixtures.py 1,436
services/analysis-mcp/tests/test_engine_aggregate.py tests/test_engine_aggregate.py -
services/analysis-mcp/tests/test_engine_aggregate_rank_arrow.py tests/test_engine_aggregate_rank_arrow.py -
services/analysis-mcp/tests/test_engine_columnar.py tests/test_engine_columnar.py -
services/analysis-mcp/tests/test_engine_common.py tests/test_engine_common.py -
services/analysis-mcp/tests/test_engine_compare.py tests/test_engine_compare.py -
services/analysis-mcp/tests/test_engine_filter.py tests/test_engine_filter.py -
services/analysis-mcp/tests/test_engine_filter_arrow.py tests/test_engine_filter_arrow.py -
services/analysis-mcp/tests/test_engine_growth.py tests/test_engine_growth.py -
services/analysis-mcp/tests/test_engine_join.py tests/test_engine_join.py -
services/analysis-mcp/tests/test_engine_join_arrow.py tests/test_engine_join_arrow.py -
services/analysis-mcp/tests/test_engine_quotient_of.py tests/test_engine_quotient_of.py -
services/analysis-mcp/tests/test_engine_rank.py tests/test_engine_rank.py -
services/analysis-mcp/tests/test_engine_share_of.py tests/test_engine_share_of.py -
services/analysis-mcp/tests/test_engine_share_of_columnar.py tests/test_engine_share_of_columnar.py -
services/analysis-mcp/tests/test_engine_sum_of.py tests/test_engine_sum_of.py -
services/analysis-mcp/tests/test_execution_identity.py tests/test_execution_identity.py -
services/analysis-mcp/tests/test_identity.py tests/test_identity.py -
services/analysis-mcp/tests/test_op_spec.py tests/test_op_spec.py -
services/analysis-mcp/tests/test_result.py tests/test_result.py -
services/analysis-mcp/tests/differential/cases.py tests/differential/cases.py -
services/analysis-mcp/tests/differential/harness.py tests/differential/harness.py -
services/analysis-mcp/tests/differential/test_engine_differential.py tests/differential/test_engine_differential.py -
services/analysis-mcp/tests/differential/engine_cases.jsonl tests/differential/engine_cases.jsonl -
services/analysis-mcp/tests/fixtures/aggregate_golden.json tests/fixtures/aggregate_golden.json -
services/analysis-mcp/tests/fixtures/compare_golden.json tests/fixtures/compare_golden.json -
services/analysis-mcp/tests/fixtures/filter_golden.json tests/fixtures/filter_golden.json -
services/analysis-mcp/tests/fixtures/growth_golden.json tests/fixtures/growth_golden.json -
services/analysis-mcp/tests/fixtures/join_golden.json tests/fixtures/join_golden.json -
services/analysis-mcp/tests/fixtures/rank_golden.json tests/fixtures/rank_golden.json -
services/analysis-mcp/tests/fixtures/share_of_golden.json tests/fixtures/share_of_golden.json -
services/analysis-mcp/tests/fixtures/op_spec.json tests/fixtures/op_spec.json -
services/analysis-mcp/tests/fixtures/trace.json tests/fixtures/trace.json -
services/analysis-mcp/tests/fixtures/vectors/op_spec_hash_vectors.json tests/fixtures/vectors/op_spec_hash_vectors.json -
services/analysis-mcp/tests/fixtures/vectors/op_spec_rolling_hash_vectors.json tests/fixtures/vectors/op_spec_rolling_hash_vectors.json -
services/analysis-mcp/tests/fixtures/vectors/toolkit_content_version_vectors.json tests/fixtures/vectors/toolkit_content_version_vectors.json -
packages/common/tests/test_rolling.py tests/test_rolling.py -
packages/common/tests/test_place_type.py tests/test_place_type.py -
packages/common/tests/test_time_grain.py tests/test_time_grain.py -
packages/common/tests/test_hashed_fields.py tests/test_hashed_fields.py -
packages/common/tests/test_price_basis.py tests/test_price_basis.py -
packages/common/tests/test_materialisation_frame_ref.py tests/test_frame_ref.py -
packages/common/tests/test_recipe.py tests/test_recipe.py 384,$
TABLE

awk -F'\t' '/^@@\t/ { if ($2 > 0) print "== " $3; next } { print }' "$scratch/report"
echo
echo "non-import changed lines per file (files with none are omitted):"
awk -F'\t' '/^@@\t/ && $2 > 0 { printf "%6d  %s\n", $2, $3; total += $2 } END { printf "%6d  total\n", total }' "$scratch/report"
echo "files compared: $(grep -c '^@@' "$scratch/report")"
