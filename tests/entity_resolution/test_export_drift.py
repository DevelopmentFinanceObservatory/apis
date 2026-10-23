"""Live tests compare graph exports with committed entity, region and place-type data.

Set SEARCH_GRAPH_BASE and, behind a WAF, SEARCH_GRAPH_TOKEN. The entities export reads the
declared totals from the code index of a published graph release: set LIVE_GRAPH_RELEASE_URI
to the release the graph's environment serves (`tests/conftest.py` points GRAPH_RELEASE_URI at
the session release). Run with:
`uv run --env-file .env pytest -m live tests/entity_resolution/test_export_drift.py`
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dc_graph import GraphClient
from entity_resolution import pack
from entity_resolution.types import PLACE_TYPES
from graph_release.service import ReleaseService
from scripts.export_graph_entities import export_entities, export_place_types, export_regions

_GRAPH_BASE = os.environ.get("SEARCH_GRAPH_BASE")
_LIVE_RELEASE_URI = os.environ.get("LIVE_GRAPH_RELEASE_URI")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _GRAPH_BASE, reason="set SEARCH_GRAPH_BASE to run the export drift test"
    ),
]


@pytest.mark.skipif(
    not _LIVE_RELEASE_URI, reason="set LIVE_GRAPH_RELEASE_URI to the release the graph serves"
)
def test_entities_export_matches_committed_records(tmp_path: Path):
    release = ReleaseService(uri=_LIVE_RELEASE_URI, local_dir=tmp_path, recheck_seconds=0)
    reader = GraphClient(
        base_url=f"{_GRAPH_BASE}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN")
    )
    try:
        fresh = export_entities(reader, index=release.snapshot().views.code_index)
    finally:
        reader.close()
        release.close()
    committed = json.loads(pack.records_path().read_text())
    assert fresh == committed


def test_regions_export_matches_committed_ids():
    fresh = export_regions(graph_base_url=_GRAPH_BASE, token=os.environ.get("SEARCH_GRAPH_TOKEN"))
    committed = json.loads(pack.region_ids_path().read_text())
    assert fresh == committed


def test_place_types_export_matches_committed_types():
    reader = GraphClient(
        base_url=f"{_GRAPH_BASE}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN")
    )
    try:
        fresh = export_place_types(reader)
    finally:
        reader.close()
    assert fresh == sorted(PLACE_TYPES)
