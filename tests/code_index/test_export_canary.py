"""A live check compares a deterministic sample of exported StatVars with the committed index.
The sample takes 12 rows per measure partition and 3 per code system, seeded by the content hash.
`export_statvars` checks sampled rows; its explicit-property reads avoid the roughly 26-node `->*` cap.
Set `SEARCH_GRAPH_BASE` and, behind a WAF, `SEARCH_GRAPH_TOKEN` to run this test:
    uv run --env-file .env pytest -m live tests/code_index/test_export_canary.py
"""

from __future__ import annotations

import os
import random
from collections import defaultdict

import pytest

from code_index.loader import default_index_path, load_index, load_measure_axes
from code_index.types import StatVarRow, label_for_pairs
from scripts.export_code_index import (
    CODE_SYSTEMS,
    _measure_axis_value,
    export_reader,
    export_statvars,
)

_GRAPH_BASE = os.environ.get("SEARCH_GRAPH_BASE")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _GRAPH_BASE, reason="set SEARCH_GRAPH_BASE to run the export canary"),
]

_PER_MEASURE_PARTITION = 12
_PER_CODE_SYSTEM = 3


def _row_dict(row: StatVarRow) -> dict:
    """Return the export dict shape for direct comparison of fresh and committed rows."""
    return {
        "dcid": row.dcid,
        "population_type": row.population_type,
        "measured_property": row.measured_property,
        "stat_type": row.stat_type,
        "measurement_qualifier": row.measurement_qualifier,
        "measurement_denominator": row.measurement_denominator,
        "observation_properties": sorted(row.observation_properties),
        "constraint_properties": dict(sorted(row.constraint_properties.items())),
    }


def _select_sample(committed_rows: dict, axes, partition_labels, seed: str) -> tuple[str, ...]:
    """Return a deterministic sample with up to 12 rows per measure partition and 3 per system.

    Rows without a partition label can still enter a code-system group. The seed makes failures
    reproducible against the same committed index.
    """
    measure_groups: dict[str, list[str]] = defaultdict(list)
    code_system_groups: dict[str, list[str]] = defaultdict(list)
    code_system_names = tuple(config.code_system for config in CODE_SYSTEMS)

    for dcid in sorted(committed_rows):
        row = committed_rows[dcid]
        pairs = [(axis.axis, _measure_axis_value(row, axis)) for axis in axes]
        match = label_for_pairs(pairs, partition_labels)
        if match is not None:
            measure_groups[match.label].append(dcid)
        for code_system in code_system_names:
            if code_system in row["constraint_properties"]:
                code_system_groups[code_system].append(dcid)

    rng = random.Random(seed)
    sampled: set[str] = set()
    for key in sorted(measure_groups):
        pool = list(measure_groups[key])
        rng.shuffle(pool)
        sampled.update(pool[:_PER_MEASURE_PARTITION])
    for key in sorted(code_system_groups):
        pool = list(code_system_groups[key])
        rng.shuffle(pool)
        sampled.update(pool[:_PER_CODE_SYSTEM])
    return tuple(sorted(sampled))


def test_sampled_statvars_match_committed_index():
    index_path = default_index_path()
    if not index_path.is_file():
        pytest.fail(
            f"{index_path} does not exist -- the code index export has not been run yet. "
            "This test fails rather than skips: a live sampling gate that goes green over a "
            "missing export is fail-open."
        )
    committed = load_index(index_path)
    committed_measures = load_measure_axes(index_path)
    committed_rows = {row.dcid: _row_dict(row) for row in committed.statvars}

    sample = _select_sample(
        committed_rows,
        committed_measures.axes,
        committed_measures.partition_labels,
        seed=committed.stamp.content_hash,
    )
    assert sample, "the stratified sample selected zero dcids -- nothing was checked"

    reader = export_reader(f"{_GRAPH_BASE}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
    try:
        fresh_rows = export_statvars(reader, sample)
    finally:
        reader.close()

    fresh_by_dcid = {row["dcid"]: row for row in fresh_rows}
    assert set(fresh_by_dcid) == set(sample)
    for dcid in sample:
        assert fresh_by_dcid[dcid] == committed_rows[dcid], dcid
