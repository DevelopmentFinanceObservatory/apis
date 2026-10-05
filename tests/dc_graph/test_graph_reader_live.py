"""Live `child_places_batch` smoke test. Set `SEARCH_GRAPH_BASE` and, if needed, `SEARCH_GRAPH_TOKEN`."""

from __future__ import annotations

import os

import pytest

from dc_graph import GraphClient

_GRAPH_BASE = os.environ.get("SEARCH_GRAPH_BASE")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _GRAPH_BASE, reason="set SEARCH_GRAPH_BASE to run the child_places_batch live test"
    ),
]


def test_child_places_batch_two_costly_parents_each_return_members():
    # Their combined containment closure times out against the dev mixer; query one parent per request.
    reader = GraphClient(
        base_url=f"{_GRAPH_BASE}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN")
    )
    try:
        result = reader.child_places_batch(["africa", "SubSaharanAfrica"], child_type="Country")
    finally:
        reader.close()

    assert set(result) == {"africa", "SubSaharanAfrica"}
    for parent, members in result.items():
        assert members, f"{parent} returned no members"
        assert len(members) == len(set(members)), f"{parent} returned duplicate members"
