"""Each role-bearing `dataset_resubmit` case has a complete StatVar family
in the offline slice, or none of it.

`graph_confirm_resolve` confirms family members against coverage only, so a missing sibling reads
as no data and drops out of `resolved.stat_vars`. Re-record the slice per `deploy/coverage.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from search.eval.cases import load_cases
from series_coverage.loader import load_coverage

_SLICE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "coverage_slice"


def _family_cases() -> list[tuple[str, tuple[str, ...]]]:
    """(id, stat_vars) for every `dataset_resubmit` case whose definite answer names its
    StatVars, one row per measure form, sorted by id."""
    rows = []
    for case in load_cases():
        block = case.expect.for_status("definite")
        if case.door != "dataset_resubmit" or block is None or block.measure is None:
            continue
        rows.extend(
            (case.id, tuple(form.stat_vars)) for form in block.measure.forms if form.stat_vars
        )
    return sorted(rows)


@pytest.fixture(scope="module")
def role_names() -> dict:
    return load_coverage(f"file://{_SLICE_DIR}").index.role_names


@pytest.mark.parametrize(
    "case_id,expected_stat_vars",
    [pytest.param(case_id, svs, id=case_id) for case_id, svs in _family_cases()],
)
def test_coverage_slice_confirms_a_whole_family_or_none_of_it(
    case_id, expected_stat_vars, role_names
):
    present = [sv for sv in expected_stat_vars if sv in role_names]
    if not present:
        return
    missing = [sv for sv in expected_stat_vars if sv not in role_names]
    assert not missing, (
        f"{case_id}: coverage_slice confirms {len(present)}/{len(expected_stat_vars)} "
        f"of this family's stat_vars and lacks {missing!r}; re-record it with "
        "scripts/record_coverage_slice.py (deploy/coverage.md)"
    )
