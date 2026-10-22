"""A live check that some StatVar of each `IMPLIED_VALUES` measure states its value in the graph.

The index holds no row of a catalogue family such as life expectancy, so neither the loader nor
the release builder can check these declarations. Set `SEARCH_GRAPH_BASE` and, behind a WAF,
`SEARCH_GRAPH_TOKEN` to run it; `CODE_INDEX_LIVE_REQUIRED` makes collection fail when the base
is unset:
    uv run --env-file .env pytest -m live tests/code_index/test_implied_values_live.py
"""

from __future__ import annotations

import os

import pytest

from code_index.declarations import IMPLIED_VALUES
from dc_graph import GraphClient
from tests.code_index.support import unstated_implied_values

_GRAPH_BASE = os.environ.get("SEARCH_GRAPH_BASE")

if not _GRAPH_BASE and os.environ.get("CODE_INDEX_LIVE_REQUIRED"):
    raise RuntimeError(
        "CODE_INDEX_LIVE_REQUIRED is set but SEARCH_GRAPH_BASE is empty -- this suite would "
        "skip every test and report green over zero live calls"
    )

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _GRAPH_BASE, reason="set SEARCH_GRAPH_BASE to run the check"),
]


def test_some_statvar_of_each_implied_measure_states_its_value():
    graph = GraphClient(
        base_url=f"{_GRAPH_BASE}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN")
    )
    try:
        unstated = unstated_implied_values(graph, IMPLIED_VALUES)
    finally:
        graph.close()

    assert unstated == [], "delete each unstated declaration from IMPLIED_VALUES"
