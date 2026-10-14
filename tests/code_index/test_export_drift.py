"""Compare a fresh export of the live graph with the committed code index.
It compares every block the graph and the authored config determine: declarations, datasets, codes,
StatVars and listed StatVars without their provenances, the listed StatVars' series codes and
value parents, measure axes, values, and partition labels, and party types, plus the stamp's
counts and content hash. The fresh export lists the StatVars the committed index lists.
The run time and graph endpoint vary, so the check excludes them. Each StatVar's provenances, the
parties, the coverage build id, and the count of StatVars without coverage come from the coverage
build, which `python -m series_coverage.verify_index` checks at deploy.
It fails if the committed export is missing and skips without `SEARCH_GRAPH_BASE`.
Set `SEARCH_GRAPH_BASE` and, behind a WAF, `SEARCH_GRAPH_TOKEN` to run this test:
    uv run --env-file .env pytest -m live tests/code_index/test_export_drift.py
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

from code_index.loader import default_index_path, load_index
from scripts.export_code_index import (
    content_hash,
    export_graph_blocks,
    export_reader,
    stamp_counts,
)

_GRAPH_BASE = os.environ.get("SEARCH_GRAPH_BASE")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _GRAPH_BASE, reason="set SEARCH_GRAPH_BASE to run the export drift test"
    ),
]

_GRAPH_ROW_FIELDS_EXCLUDED = ("provenances",)


def _without_provenances(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _GRAPH_ROW_FIELDS_EXCLUDED}


def test_export_matches_committed_index():
    index_path = default_index_path()
    if not index_path.is_file():
        pytest.fail(
            f"{index_path} does not exist -- the code index export has not been run yet. "
            "This test fails rather than skips: a live drift gate that goes green over a "
            "missing export is fail-open."
        )
    # The committed export must load before its blocks are compared.
    load_index(index_path)
    committed = json.loads(gzip.decompress(index_path.read_bytes()))

    reader = export_reader(f"{_GRAPH_BASE}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
    try:
        fresh = export_graph_blocks(
            reader, listed=[row["dcid"] for row in committed["listed_statvars"]]
        )
    finally:
        reader.close()

    assert set(fresh) == set(committed) - {"stamp", "parties"}
    for block in (
        "declarations",
        "datasets",
        "codes",
        "source_codes",
        "value_parents",
        "party_types",
        "shape_slot_defaults",
        "partly_stated_values",
        "measure_classes",
        "variable_aliases",
        "stat_type_words",
    ):
        assert fresh[block] == committed[block], block
    for block in ("measure_axes", "measure_values", "measure_partition_labels", "measure_nouns"):
        assert fresh[block] == committed[block], block
    for block in ("statvars", "listed_statvars"):
        assert fresh[block] == [_without_provenances(row) for row in committed[block]], block

    counts = stamp_counts(fresh)
    assert counts["code_counts"] == committed["stamp"]["code_counts"]
    assert counts["statvar_shape_counts"] == committed["stamp"]["statvar_shape_counts"]

    # The stamp's hash covers each row's provenances and the parties, so the committed ones
    # complete the fresh blocks before hashing.
    committed_provenances = {
        row["dcid"]: row["provenances"]
        for row in (*committed["statvars"], *committed["listed_statvars"])
    }
    for row in (*fresh["statvars"], *fresh["listed_statvars"]):
        row["provenances"] = committed_provenances[row["dcid"]]
    fresh["parties"] = committed["parties"]
    assert content_hash(fresh) == committed["stamp"]["content_hash"]
